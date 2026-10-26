"""x108 Stage 3b (GPU, sharded): GuardReasoner-8B judges every sampled response of one arm under
x104's contract, through x106's pass (greedy, then up to three re-decodes of unparsed rows; the
batch halves on an out-of-memory). Both the `Completion` field (refusal) and the `Response`
field (response harmfulness) are parsed and stored. One row per (prompt, response index), keyed
`x108|tag|arm|id|k`; append-only JSONL, resumed by key.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from x102_common import guardreasoner_input
from x104_common import RESAMPLE_SEEDS
from x104_judge import load_judge
from x106_judge_guardreasoner import read_records, run_pass
from x108_common import ALL_TAGS, ARMS, N_RESPONSES, TAG, generations_path, guardreasoner_path, read_json

DEFAULT_BATCH = 48


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=TAG, choices=ALL_TAGS)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out")
    return parser.parse_args()


def response_rows(tag: str, arm: str) -> list[dict]:
    generations = read_json(generations_path(tag, arm))
    if generations.get("experiment") != "x108" or generations.get("arm") != arm or generations.get("model_tag") != tag:
        raise SystemExit("FATAL: generation model/arm/experiment mismatch")
    rows = []
    for row in generations["rows"]:
        if len(row["llm_responses"]) != N_RESPONSES:
            raise SystemExit(f"FATAL: generation row {row['id']} lacks {N_RESPONSES} responses")
        for k, response in enumerate(row["llm_responses"]):
            rows.append(
                {
                    "key": f"x108|{arm}|{row['id']}|{k}" if tag == TAG else f"x108|{tag}|{arm}|{row['id']}|{k}",
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


def main() -> None:
    args = parse_args()
    rows = response_rows(args.tag, args.arm)[args.shard :: args.n_shards]
    if args.limit:
        rows = rows[: args.limit]
    out_path = Path(args.out) if args.out else guardreasoner_path(args.tag, args.arm, args.shard, args.n_shards)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = read_records(out_path)
    todo = [r for r in rows if r["key"] not in done]
    print(f"x108 guardreasoner {args.tag} {args.arm} shard {args.shard}/{args.n_shards}: {len(rows)} rows, {len(done)} done, {len(todo)} to judge", flush=True)
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
        print(f"x108 guardreasoner: re-decoding {len(unparsed)} unparsed rows with seed {seed}", flush=True)
        run_pass(unparsed, out_path, snapshot, tokenizer, model, eos_ids, args.batch_size, seed, label)
        done = read_records(out_path)
    n_ok = sum(1 for r in rows if done[r["key"]]["parse_ok"])
    print(f"x108 guardreasoner {args.tag} {args.arm} shard {args.shard}: {n_ok}/{len(rows)} parsed in {(time.time() - t0) / 60:.1f} min -> {out_path}")


if __name__ == "__main__":
    main()
