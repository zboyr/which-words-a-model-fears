"""x102 stage 5: the incumbent judge (x87 scorer, unchanged) on XSTest's 2,250 completions.

Same unedited Qwen3.5-4B snapshot, same rubric, same parser, same retry
conversation, same decoding as `x87_score.score_orbench`; only the rows
differ.  A second parse failure records the row as `unparsed` instead of
stopping the run.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from x102_common import (
    ORBENCH_JUDGE_PROMPT,
    RETRY_INSTRUCTION,
    RUN,
    XSTEST_PATH,
    load_manifest,
    parse_orbench_label,
    read_json,
    read_jsonl,
    require_registered,
    update_manifest,
)
from x87_common import MODEL_ID, format_user_chat, load_causal_lm

INCUMBENT_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
DEFAULT_SNAPSHOT = (
    "<workdir>/hf_cache/hub/models--Qwen--Qwen3.5-4B/snapshots/" + INCUMBENT_REVISION
)
OUT_PATH = RUN / "x102_xstest_incumbent.jsonl"
MAX_NEW_TOKENS = 512
RETRY_NEW_TOKENS = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    parser.add_argument("--judge-batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", help="override the output path (smoke tests)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if Path(args.snapshot).name != INCUMBENT_REVISION:
        raise SystemExit(f"FATAL: {args.snapshot} is not the incumbent snapshot {INCUMBENT_REVISION}")
    manifest = load_manifest()
    payload = read_json(XSTEST_PATH)
    require_registered(manifest, "xstest_sha256", payload["xstest_sha256"])
    rows = payload["rows"]
    if args.limit:
        rows = rows[: args.limit]
    out_path = Path(args.out) if args.out else OUT_PATH
    done = set()
    for record in read_jsonl(out_path):
        if record["key"] in done:
            raise SystemExit(f"FATAL: duplicate judged row {record['key']} in {out_path}")
        done.add(record["key"])
    pending = [row for row in rows if row["key"] not in done]
    print(f"x102 incumbent/xstest: {len(done)} done, {len(pending)} to judge", flush=True)
    if not pending:
        return

    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    judge = load_causal_lm(args.snapshot)  # always unedited base Qwen
    judge.eval()

    def judge_generate(chats: list[str], max_new_tokens: int) -> list[str]:
        tokens = tokenizer(
            chats, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(judge.device)
        with torch.inference_mode():
            generated = judge.generate(
                **tokens,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        continuations = generated[:, tokens["input_ids"].shape[1] :]
        return tokenizer.batch_decode(continuations, skip_special_tokens=True)

    def retry_chat(rubric_prompt: str, first_reply: str) -> str:
        return tokenizer.apply_chat_template(
            [
                {"role": "user", "content": rubric_prompt},
                {"role": "assistant", "content": first_reply},
                {"role": "user", "content": RETRY_INSTRUCTION},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    started = time.time()
    parse_mix: dict[str, int] = {}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle:
        for start in range(0, len(pending), args.judge_batch_size):
            batch = pending[start : start + args.judge_batch_size]
            rubrics = [
                ORBENCH_JUDGE_PROMPT.format(prompt=row["prompt"], response=row["response"])
                for row in batch
            ]
            texts = judge_generate([format_user_chat(tokenizer, r) for r in rubrics], MAX_NEW_TOKENS)
            records = {}
            unlabeled = []
            for source, rubric, judge_text in zip(batch, rubrics, texts):
                try:
                    label, parse_mode = parse_orbench_label(judge_text)
                except ValueError:
                    unlabeled.append((source, rubric, judge_text))
                    continue
                records[source["key"]] = {
                    "key": source["key"],
                    "id": source["id"],
                    "model": source["model"],
                    "benchmark": "xstest",
                    "label": label,
                    "label_parse": parse_mode,
                    "refusal": int(label != "direct_answer"),
                    "judge_response": judge_text.strip(),
                }
            if unlabeled:
                retries = judge_generate(
                    [retry_chat(rubric, text) for _, rubric, text in unlabeled], RETRY_NEW_TOKENS
                )
                for (source, _, first_text), retry_text in zip(unlabeled, retries):
                    try:
                        label, parse_mode = parse_orbench_label(retry_text)
                        parse_mode = f"{parse_mode}_retry"
                        refusal: int | None = int(label != "direct_answer")
                    except ValueError:
                        label, parse_mode, refusal = None, "unparsed", None
                    records[source["key"]] = {
                        "key": source["key"],
                        "id": source["id"],
                        "model": source["model"],
                        "benchmark": "xstest",
                        "label": label,
                        "label_parse": parse_mode,
                        "refusal": refusal,
                        "judge_response": first_text.strip(),
                        "judge_response_retry": retry_text.strip(),
                    }
            for source in batch:
                record = records[source["key"]]
                parse_mix[record["label_parse"]] = parse_mix.get(record["label_parse"], 0) + 1
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            n_done = min(start + len(batch), len(pending))
            if (start // args.judge_batch_size) % 25 == 0 or n_done == len(pending):
                print(f"  {n_done}/{len(pending)} {time.time() - started:.0f}s {parse_mix}", flush=True)
    summary = {
        "judge_model_id": MODEL_ID,
        "snapshot": args.snapshot,
        "revision": INCUMBENT_REVISION,
        "rows_judged": len(pending),
        "rows_resumed": len(done),
        "parse_mix": parse_mix,
        "unparsed": parse_mix.get("unparsed", 0),
        "max_new_tokens": MAX_NEW_TOKENS,
        "retry_new_tokens": RETRY_NEW_TOKENS,
        "batch_size": args.judge_batch_size,
        "decoding": "greedy, enable_thinking=False",
        "seconds": time.time() - started,
    }
    if not args.out and not args.limit:
        update_manifest("incumbent_xstest", summary)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
