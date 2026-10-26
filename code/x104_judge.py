"""x104 stage 2: GuardReasoner-8B judges one shard of the x104 rows.

Exactly x102's open-weight arm: the model card's instruction as raw text with
BOS, bf16, greedy, 1,024 new tokens, stopping on `<|eot_id|>` and
`<|end_of_text|>`, `Completion: refusal/compliance` parsed from the trace.
Resumed by row key, append-only JSONL checkpointed after every batch.

A row whose greedy trace reaches the cap without a parsable answer block is
re-decoded with sampling (temperature 0.6, seeds 1, 2, 3, in that order,
stopping at the first parsable trace); a row still unparsed after seed 3 is
recorded as unparsed.  Greedy records go to the shard file, re-decode records
to the shard's resample file; `x104_analyze` merges them.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from x104_common import (
    DEFAULT_BATCH,
    JUDGE,
    RESAMPLE_SEEDS,
    RESAMPLE_TEMPERATURE,
    atomic_json,
    guardreasoner_input,
    load_rows,
    parse_guardreasoner,
    read_shard_jsonl,
    resample_path,
    shard_path,
    shard_rows,
    summary_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--n-shards", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--limit", type=int, default=0, help="smoke test: first N rows of the shard")
    parser.add_argument("--out", help="smoke test: override the shard output path")
    return parser.parse_args()


def load_judge():
    snapshot = snapshot_download(JUDGE["model_id"], revision=JUDGE["revision"])
    if Path(snapshot).name != JUDGE["revision"]:
        raise SystemExit(f"FATAL: resolved snapshot {snapshot} is not revision {JUDGE['revision']}")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, use_fast=JUDGE["use_fast_tokenizer"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(snapshot, dtype=torch.bfloat16, device_map="auto").eval()
    eos_ids = {tokenizer.eos_token_id}
    config_eos = model.generation_config.eos_token_id
    if config_eos is not None:
        eos_ids.update(config_eos if isinstance(config_eos, list) else [config_eos])
    eos_ids = sorted(int(i) for i in eos_ids if i is not None)
    return snapshot, tokenizer, model, eos_ids


def generate(tokenizer, model, eos_ids, texts: list[str], *, seed: int | None) -> list[tuple[str, int]]:
    """(continuation, new tokens) per text; greedy when seed is None, else temperature sampling."""
    tokens = tokenizer(
        texts, return_tensors="pt", padding=True, add_special_tokens=JUDGE["add_special_tokens"]
    ).to(model.device)
    sampling = (
        {"do_sample": False}
        if seed is None
        else {"do_sample": True, "temperature": RESAMPLE_TEMPERATURE, "top_p": 1.0, "top_k": 0}
    )
    if seed is not None:
        torch.manual_seed(seed)
    with torch.inference_mode():
        generated = model.generate(
            **tokens,
            max_new_tokens=JUDGE["max_new_tokens"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_ids,
            **sampling,
        )
    continuations = generated[:, tokens["input_ids"].shape[1] :]
    decoded = tokenizer.batch_decode(continuations, skip_special_tokens=True)
    eos_tensor = torch.tensor(eos_ids, device=continuations.device)
    out = []
    for text, ids in zip(decoded, continuations):
        stops = torch.isin(ids, eos_tensor).nonzero()
        n_new = int(stops[0].item()) + 1 if len(stops) else int(ids.shape[0])
        out.append((text, n_new))
    return out


def record_for(row: dict, snapshot: str, continuation: str, n_new: int, decoding: str) -> dict:
    parsed = parse_guardreasoner(continuation)
    return {
        "key": row["key"],
        "tag": row["tag"],
        "experiment": row["experiment"],
        "condition": row["condition"],
        "id": row["id"],
        "benchmark": row["benchmark"],
        "model_id": JUDGE["model_id"],
        "revision": JUDGE["revision"],
        "snapshot": snapshot,
        "decoding": decoding,
        "output": continuation,
        "parse_ok": bool(parsed["parse_ok"]),
        "refusal": parsed["refusal"],
        "fields": parsed.get("fields", {}),
        "new_tokens": n_new,
    }


def greedy_pass(rows, out_path: Path, snapshot, tokenizer, model, eos_ids, batch_size: int) -> dict:
    done = {}
    for record in read_shard_jsonl(out_path):
        if record["key"] in done:
            raise SystemExit(f"FATAL: duplicate judged row {record['key']} in {out_path}")
        done[record["key"]] = record
    todo = [row for row in rows if row["key"] not in done]
    print(f"x104 greedy: {len(done)} done, {len(todo)} to judge", flush=True)
    rendered = [(row, guardreasoner_input(row["prompt"], row["response"])) for row in todo]
    rendered.sort(key=lambda item: len(item[1]), reverse=True)  # longest first: small padding waste
    started = time.time()
    n_ok = 0
    new_tokens_total = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle:
        for start in range(0, len(rendered), batch_size):
            batch = rendered[start : start + batch_size]
            outputs = generate(tokenizer, model, eos_ids, [text for _, text in batch], seed=None)
            for (row, _), (continuation, n_new) in zip(batch, outputs):
                record = record_for(row, snapshot, continuation, n_new, "greedy")
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                done[row["key"]] = record
                n_ok += int(record["parse_ok"])
                new_tokens_total += n_new
            handle.flush()
            n_done = min(start + len(batch), len(rendered))
            if (start // batch_size) % 5 == 0 or n_done == len(rendered):
                elapsed = time.time() - started
                print(
                    f"  {n_done}/{len(rendered)} parse_ok={n_ok} {elapsed:.0f}s "
                    f"({n_done / max(elapsed, 1) * 60:.0f} rows/min)",
                    flush=True,
                )
    return {
        "done": done,
        "rows_judged": len(rendered),
        "parse_ok": n_ok,
        "mean_new_tokens": (new_tokens_total / len(rendered)) if rendered else None,
        "seconds": time.time() - started,
    }


def resample_pass(rows_by_key, unparsed_keys, out_path: Path, snapshot, tokenizer, model, eos_ids, batch_size: int) -> dict:
    """Seeds 1, 2, 3 in order on every still-unparsed row; one final record per key."""
    done = {}
    for record in read_shard_jsonl(out_path):
        if record["key"] in done:
            raise SystemExit(f"FATAL: duplicate resampled row {record['key']} in {out_path}")
        done[record["key"]] = record
    pending = [rows_by_key[k] for k in sorted(unparsed_keys) if k not in done]
    print(f"x104 resample: {len(unparsed_keys)} greedy-unparsed, {len(done)} done, {len(pending)} to re-decode", flush=True)
    started = time.time()
    attempts: dict[str, list[dict]] = {row["key"]: [] for row in pending}
    final: dict[str, dict] = {}
    remaining = list(pending)
    for seed in RESAMPLE_SEEDS:
        if not remaining:
            break
        still = []
        for start in range(0, len(remaining), batch_size):
            batch = remaining[start : start + batch_size]
            texts = [guardreasoner_input(row["prompt"], row["response"]) for row in batch]
            outputs = generate(tokenizer, model, eos_ids, texts, seed=seed)
            for row, (continuation, n_new) in zip(batch, outputs):
                record = record_for(
                    row, snapshot, continuation, n_new, f"sample_t{RESAMPLE_TEMPERATURE}_seed{seed}"
                )
                attempts[row["key"]].append(
                    {"seed": seed, "parse_ok": record["parse_ok"], "new_tokens": n_new, "output": continuation}
                )
                if record["parse_ok"]:
                    final[row["key"]] = record
                else:
                    still.append(row)
        remaining = still
        print(f"  seed {seed}: {len(final)} parsed, {len(remaining)} still unparsed", flush=True)
    for row in remaining:
        final[row["key"]] = record_for(row, snapshot, "", 0, "unparsed")
        final[row["key"]]["output"] = None
        final[row["key"]]["new_tokens"] = None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle:
        for row in pending:
            record = dict(final[row["key"]])
            record["attempts"] = [
                {k: v for k, v in a.items() if k != "output"} for a in attempts[row["key"]]
            ]
            record["attempt_outputs"] = [a["output"] for a in attempts[row["key"]]]
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            done[row["key"]] = record
    n_unparsed = sum(1 for r in done.values() if not r["parse_ok"])
    return {
        "rows_resampled": len(pending),
        "resampled_parsed": sum(1 for r in final.values() if r["parse_ok"]),
        "unparsed": n_unparsed,
        "seconds": time.time() - started,
    }


def judge_shard(all_rows, shard: int, n_shards: int, args, judge, lock) -> bool:
    """Judge one shard to completion (greedy pass, resample pass, summary)."""
    snapshot, tokenizer, model, eos_ids = judge
    rows = shard_rows(all_rows, shard, n_shards)
    if args.limit:
        rows = rows[: args.limit]
    out_path = Path(args.out) if args.out else shard_path(shard, n_shards)
    res_path = out_path.with_name(out_path.name.replace("judgments", "resample")) if args.out else resample_path(shard, n_shards)
    print(f"x104 shard {shard}/{n_shards}: {len(rows)} rows -> {out_path}", flush=True)
    greedy = greedy_pass(rows, out_path, snapshot, tokenizer, model, eos_ids, args.batch_size)
    done = greedy.pop("done")
    unparsed = {k for k, r in done.items() if not r["parse_ok"]}
    resample = resample_pass(
        {row["key"]: row for row in rows}, unparsed, res_path, snapshot, tokenizer, model, eos_ids, args.batch_size
    )
    summary = {
        "shard": shard,
        "n_shards": n_shards,
        "rows": len(rows),
        "greedy": greedy,
        "resample": resample | {"temperature": RESAMPLE_TEMPERATURE, "seeds": list(RESAMPLE_SEEDS)},
        "model_id": JUDGE["model_id"],
        "revision": JUDGE["revision"],
        "snapshot": snapshot,
        "max_new_tokens": JUDGE["max_new_tokens"],
        "dtype": "bfloat16",
        "batch_size": args.batch_size,
        "add_special_tokens": JUDGE["add_special_tokens"],
        "eos_token_ids": eos_ids,
    }
    if not args.out and not args.limit:
        atomic_json(summary_path(shard, n_shards), summary)
    print(json.dumps(summary, indent=1), flush=True)
    return True


def try_lock(shard: int, n_shards: int):
    """One writer per shard: the exclusive lock on the shard's lock file, or None."""
    path = shard_path(shard, n_shards).with_suffix(".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def main() -> None:
    args = parse_args()
    all_rows = load_rows()
    judge = None
    # Work stealing: start with the requested shard, then take any shard that is
    # neither finished nor locked by another job, so every allocation does work.
    order = [args.shard] + [s for s in range(args.n_shards) if s != args.shard]
    if args.out or args.limit:
        order = [args.shard]
    for shard in order:
        if not (args.out or args.limit) and summary_path(shard, args.n_shards).exists():
            continue
        lock = try_lock(shard, args.n_shards)
        if lock is None:
            print(f"x104: shard {shard} is being judged by another job", flush=True)
            continue
        try:
            if judge is None:
                judge = load_judge()
            judge_shard(all_rows, shard, args.n_shards, args, judge, lock)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
    print("x104: no shard left to judge", flush=True)


if __name__ == "__main__":
    main()
