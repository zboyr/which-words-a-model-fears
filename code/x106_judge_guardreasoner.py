"""x106 Stage 3b (GPU, sharded): GuardReasoner-8B judges every sampled response of one (model, arm)
under x104's contract (x102's open-weight arm: the model card's instruction as raw text with BOS,
bf16, greedy, 1,024 new tokens; `Completion: refusal/compliance` parsed from the trace; a trace
that reaches the cap unparsed is re-decoded at temperature 0.6 with seeds 1, 2, 3). One row per
(prompt, response index), keyed `x106|tag|arm|id|k`; append-only JSONL, resumed by key.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from x102_common import guardreasoner_input, parse_guardreasoner
from x104_common import JUDGE, RESAMPLE_SEEDS
from x104_judge import generate, load_judge
from x106_common import ALL_TAGS, N_RESPONSES, all_arm_names, generations_path, guardreasoner_path, read_json

DEFAULT_BATCH = 48


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, choices=ALL_TAGS)
    parser.add_argument("--arm", required=True, choices=all_arm_names())
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out")
    return parser.parse_args()


def response_rows(tag: str, arm: str) -> list[dict]:
    generations = read_json(generations_path(tag, arm))
    if generations.get("experiment") != "x106" or generations.get("arm") != arm or generations.get("model_tag") != tag:
        raise SystemExit("FATAL: generation tag/arm/experiment mismatch")
    rows = []
    for row in generations["rows"]:
        if len(row["llm_responses"]) != N_RESPONSES:
            raise SystemExit(f"FATAL: generation row {row['id']} lacks {N_RESPONSES} responses")
        for k, response in enumerate(row["llm_responses"]):
            rows.append(
                {
                    "key": f"x106|{tag}|{arm}|{row['id']}|{k}",
                    "tag": tag,
                    "arm": arm,
                    "id": row["id"],
                    "k": k,
                    "text_condition": row["text_condition"],
                    "prompt": row["prompt"],
                    "prompt_sha256": row["prompt_sha256"],
                    "response_sha256": row["response_sha256"][k],
                    "arms_sha256": row["arms_sha256"],
                    "input": guardreasoner_input(row["prompt"], response),
                }
            )
    return sorted(rows, key=lambda r: r["key"])


def read_records(path: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as handle:
        lines = [line for line in handle if line.strip()]
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                continue
            raise SystemExit(f"FATAL: malformed line {index + 1} in {path}")
        key = record["key"]
        if key not in done or record["parse_ok"] or not done[key]["parse_ok"]:
            done[key] = record
    return done


def record_for(row: dict, snapshot: str, continuation: str, n_new: int, decoding: str) -> dict:
    parsed = parse_guardreasoner(continuation)
    return {
        "key": row["key"],
        "tag": row["tag"],
        "arm": row["arm"],
        "id": row["id"],
        "k": row["k"],
        "text_condition": row["text_condition"],
        "prompt_sha256": row["prompt_sha256"],
        "response_sha256": row["response_sha256"],
        "arms_sha256": row["arms_sha256"],
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


def run_pass(rows: list[dict], out_path: Path, snapshot, tokenizer, model, eos_ids, batch_size: int, seed: int | None, label: str) -> int:
    n_unparsed = 0
    rows = sorted(rows, key=lambda r: len(r["input"]))
    with out_path.open("a", encoding="utf-8") as handle:
        start = 0
        size = batch_size
        while start < len(rows):
            batch = rows[start : start + size]
            try:
                outputs = generate(tokenizer, model, eos_ids, [r["input"] for r in batch], seed=seed)
            except torch.OutOfMemoryError:
                # the longest inputs sit at the end of the length-sorted rows: shrink the batch and retry
                torch.cuda.empty_cache()
                if size == 1:
                    raise
                size = max(1, size // 2)
                print(f"x106 guardreasoner {label}: out of memory at {start}, batch size -> {size}", flush=True)
                continue
            for row, (continuation, n_new) in zip(batch, outputs):
                record = record_for(row, snapshot, continuation, n_new, label)
                n_unparsed += int(not record["parse_ok"])
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            start += len(batch)
            print(f"x106 guardreasoner {label}: {start}/{len(rows)} rows", flush=True)
    return n_unparsed


def main() -> None:
    args = parse_args()
    rows = response_rows(args.tag, args.arm)[args.shard :: args.n_shards]
    if args.limit:
        rows = rows[: args.limit]
    out_path = Path(args.out) if args.out else guardreasoner_path(args.tag, args.arm, args.shard, args.n_shards)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = read_records(out_path)
    todo = [r for r in rows if r["key"] not in done]
    print(f"x106 guardreasoner {args.tag}/{args.arm} shard {args.shard}/{args.n_shards}: {len(rows)} rows, {len(done)} done, {len(todo)} to judge", flush=True)
    if not todo and all(done[r["key"]]["parse_ok"] or done[r["key"]]["decoding"].endswith(f"seed{RESAMPLE_SEEDS[-1]}") for r in rows):
        print("nothing to do")
        return
    t0 = time.time()
    snapshot, tokenizer, model, eos_ids = load_judge()
    if todo:
        run_pass(todo, out_path, snapshot, tokenizer, model, eos_ids, args.batch_size, None, "greedy")
    done = read_records(out_path)
    for seed in RESAMPLE_SEEDS:
        label = f"resample_seed{seed}"
        unparsed = [r for r in rows if not done[r["key"]]["parse_ok"] and not done[r["key"]]["decoding"].endswith(f"seed{seed}") and (done[r["key"]]["decoding"] == "greedy" or int(done[r["key"]]["decoding"][-1]) < seed)]
        if not unparsed:
            continue
        print(f"x106 guardreasoner: re-decoding {len(unparsed)} unparsed rows with seed {seed}", flush=True)
        run_pass(unparsed, out_path, snapshot, tokenizer, model, eos_ids, args.batch_size, seed, label)
        done = read_records(out_path)
    n_ok = sum(1 for r in rows if done[r["key"]]["parse_ok"])
    print(f"x106 guardreasoner {args.tag}/{args.arm} shard {args.shard}: {n_ok}/{len(rows)} parsed in {(time.time() - t0) / 60:.1f} min -> {out_path}")


if __name__ == "__main__":
    main()
