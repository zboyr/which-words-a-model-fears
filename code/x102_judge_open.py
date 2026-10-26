"""x102 stage 4: one open-weight refusal judge on the sample or on XSTest.

Each arm is run under its own published input format and its own refusal
field (`x102_common.OPEN_ARMS`); no rubric is shown to it.  Greedy decoding,
bf16, resume by row key, append-only JSONL checkpointed after every batch.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from x102_common import (
    OPEN_ARMS,
    OPEN_PARSERS,
    XSTEST_PATH,
    guardreasoner_input,
    load_manifest,
    load_sample,
    open_judgments_path,
    qwen3guard_messages,
    read_json,
    read_jsonl,
    require_registered,
    update_manifest,
    wildguard_input,
)

DEFAULT_BATCH = {"wildguard": 16, "qwen3guard": 16, "guardreasoner": 8}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=tuple(OPEN_ARMS))
    parser.add_argument("--xstest", action="store_true", help="judge the XSTest rows instead of the sample")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--out", help="override the output path (smoke tests)")
    return parser.parse_args()


def select_rows(xstest: bool, manifest: dict) -> list[dict]:
    if xstest:
        payload = read_json(XSTEST_PATH)
        require_registered(manifest, "xstest_sha256", payload["xstest_sha256"])
        return payload["rows"]
    sample = load_sample()
    require_registered(manifest, "sample_sha256", sample["sample_sha256"])
    return sample["rows"]


def render(arm: str, tokenizer, prompt: str, response: str) -> str:
    fmt = OPEN_ARMS[arm]["format"]
    if fmt == "wildguard":
        return wildguard_input(prompt, response)
    if fmt == "guardreasoner":
        return guardreasoner_input(prompt, response)
    if fmt == "qwen3guard":
        return tokenizer.apply_chat_template(qwen3guard_messages(prompt, response), tokenize=False)
    raise ValueError(fmt)


def main() -> None:
    args = parse_args()
    spec = OPEN_ARMS[args.arm]
    manifest = load_manifest()
    rows = select_rows(args.xstest, manifest)
    if args.limit:
        rows = rows[: args.limit]
    pass_name = "xstest" if args.xstest else "main"
    out_path = Path(args.out) if args.out else open_judgments_path(args.arm, args.xstest)
    done = set()
    for record in read_jsonl(out_path):
        if record.get("arm") != args.arm or record.get("pass") != pass_name:
            raise SystemExit(f"FATAL: {out_path} holds rows of another arm/pass")
        if record["key"] in done:
            raise SystemExit(f"FATAL: duplicate judged row {record['key']} in {out_path}")
        done.add(record["key"])
    todo = [row for row in rows if row["key"] not in done]
    print(f"x102 {args.arm}/{pass_name}: {len(done)} done, {len(todo)} to judge", flush=True)
    if not todo:
        return

    snapshot = snapshot_download(spec["model_id"], revision=spec["revision"])
    if Path(snapshot).name != spec["revision"]:
        raise SystemExit(f"FATAL: resolved snapshot {snapshot} is not revision {spec['revision']}")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, use_fast=spec["use_fast_tokenizer"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(snapshot, dtype=torch.bfloat16, device_map="auto").eval()
    # Stop on the tokenizer's eos as well as the generation config's, which is what the
    # official vLLM paths do; GuardReasoner-8B ends its answer block with the tokenizer's
    # `<|eot_id|>` while its generation config names `<|end_of_text|>` only.
    eos_ids = {tokenizer.eos_token_id}
    config_eos = model.generation_config.eos_token_id
    if config_eos is not None:
        eos_ids.update(config_eos if isinstance(config_eos, list) else [config_eos])
    eos_ids = sorted(int(i) for i in eos_ids if i is not None)
    parser = OPEN_PARSERS[args.arm]
    batch_size = args.batch_size or DEFAULT_BATCH[args.arm]
    max_new = spec["max_new_tokens"]

    rendered = [(row, render(args.arm, tokenizer, row["prompt"], row["response"])) for row in todo]
    rendered.sort(key=lambda item: len(item[1]), reverse=True)  # longest first: small padding waste

    started = time.time()
    n_ok = 0
    new_tokens_total = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle:
        for start in range(0, len(rendered), batch_size):
            batch = rendered[start : start + batch_size]
            tokens = tokenizer(
                [text for _, text in batch],
                return_tensors="pt",
                padding=True,
                add_special_tokens=spec["add_special_tokens"],
            ).to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **tokens,
                    do_sample=False,
                    max_new_tokens=max_new,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos_ids,
                )
            continuations = generated[:, tokens["input_ids"].shape[1] :]
            texts = tokenizer.batch_decode(continuations, skip_special_tokens=True)
            eos_tensor = torch.tensor(eos_ids, device=continuations.device)
            for (row, _), continuation, ids in zip(batch, texts, continuations):
                stops = torch.isin(ids, eos_tensor).nonzero()
                n_new = int(stops[0].item()) + 1 if len(stops) else int(ids.shape[0])
                parsed = parser(continuation)
                record = {
                    "key": row["key"],
                    "arm": args.arm,
                    "pass": pass_name,
                    "model_id": spec["model_id"],
                    "revision": spec["revision"],
                    "snapshot": snapshot,
                    "output": continuation,
                    "parse_ok": bool(parsed["parse_ok"]),
                    "refusal": parsed["refusal"],
                    "fields": parsed.get("fields", {}),
                    "new_tokens": n_new,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_ok += int(parsed["parse_ok"])
                new_tokens_total += n_new
            handle.flush()
            n_done = min(start + len(batch), len(rendered))
            if (start // batch_size) % 20 == 0 or n_done == len(rendered):
                print(
                    f"  {n_done}/{len(rendered)} parse_ok={n_ok} {time.time() - started:.0f}s",
                    flush=True,
                )
    seconds = time.time() - started
    summary = {
        "rows_total": len(rows),
        "rows_judged": len(rendered),
        "rows_resumed": len(done),
        "parse_ok": n_ok,
        "parse_rate": n_ok / len(rendered),
        "model_id": spec["model_id"],
        "revision": spec["revision"],
        "snapshot": snapshot,
        "max_new_tokens": max_new,
        "dtype": "bfloat16",
        "decoding": "greedy",
        "batch_size": batch_size,
        "use_fast_tokenizer": spec["use_fast_tokenizer"],
        "add_special_tokens": spec["add_special_tokens"],
        "eos_token_ids": eos_ids,
        "mean_new_tokens": new_tokens_total / len(rendered),
        "seconds": seconds,
    }
    if not args.out and not args.limit:
        update_manifest("open_xstest" if args.xstest else "open_main", {args.arm: summary})
    print(json.dumps({args.arm: summary}, indent=1))


if __name__ == "__main__":
    main()
