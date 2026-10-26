"""x106 Stage 3a (local): the exact x64 refusal2 judge on one (model, arm); x101 / x105's judge,
unchanged. Rows are keyed by the cohort's string prompt id; the judge's integer id field
carries the unit's position in the cohort."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from x81_judge_refusal import CostMeter, judge_row, load_api_key
from x106_common import ALL_TAGS, N_RESPONSES, all_arm_names, generations_path, load_cohort, read_json, refusal_path, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, choices=ALL_TAGS)
    parser.add_argument("--arm", required=True, choices=all_arm_names())
    parser.add_argument("--generations")
    parser.add_argument("--out")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cap", type=float, default=40.0)
    parser.add_argument("--threads", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generation_path = Path(args.generations) if args.generations else generations_path(args.tag, args.arm)
    out_path = Path(args.out) if args.out else refusal_path(args.tag, args.arm)
    if not generation_path.exists():
        raise SystemExit(f"FATAL: generation file not found: {generation_path}")
    generations = read_json(generation_path)
    if generations.get("experiment") != "x106" or generations.get("arm") != args.arm or generations.get("model_tag") != args.tag:
        raise SystemExit("FATAL: generation tag/arm/experiment mismatch")
    cohort = load_cohort()
    if generations["cohort_sha256"] != cohort["cohort_sha256"]:
        raise SystemExit("FATAL: generation file belongs to another cohort")
    index_of = {u["id"]: k for k, u in enumerate(cohort["units"])}
    rows = list(generations["rows"])
    if args.limit:
        rows = rows[: args.limit]
    by_id = {str(row["id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise SystemExit("FATAL: duplicate generation IDs")
    for row in rows:
        if row.get("condition") != args.arm or row.get("arms_sha256") != generations["cohort_sha256"] or len(row.get("llm_responses", [])) != N_RESPONSES or len(row.get("response_sha256", [])) != N_RESPONSES:
            raise SystemExit(f"FATAL: invalid generation row {row.get('id')}")

    generation_sha = sha256_file(generation_path)
    done = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("failed"):
                    continue
                prompt_id = str(record["prompt_id"])
                source = by_id.get(prompt_id)
                if source is None:
                    continue
                if (
                    record.get("condition") != args.arm
                    or record.get("prompt_sha256") != source["prompt_sha256"]
                    or record.get("response_sha256") != source["response_sha256"]
                    or record.get("arms_sha256") != generations["cohort_sha256"]
                    or len(record.get("refusal", [])) != N_RESPONSES
                    or not all(isinstance(value, bool) for value in record["refusal"])
                ):
                    raise SystemExit(f"FATAL: invalid successful judge resume row {prompt_id}")
                if prompt_id in done:
                    raise SystemExit(f"FATAL: duplicate successful judge row {prompt_id}")
                done.add(prompt_id)

    todo = [row for row in rows if str(row["id"]) not in done]
    if not todo:
        print(f"x106 refusal {args.tag}/{args.arm}: nothing new ({len(done)} done)")
        return
    api_key = load_api_key()
    meter = CostMeter(args.cap)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_failed = 0

    def judge(row):
        record = judge_row(api_key, {**row, "id": index_of[str(row["id"])]}, meter, generation_sha)
        if record is not None:
            record["prompt_id"] = str(row["id"])
        return record

    with out_path.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(args.threads) as pool:
        futures = {pool.submit(judge, row): row for row in todo}
        for future in as_completed(futures):
            record = future.result()
            if record is None:
                continue
            record["model_tag"] = args.tag
            record["arm"] = args.arm
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if record.get("failed"):
                n_failed += 1
            else:
                n_ok += 1
    usd, prompt_tokens, completion_tokens = meter.snapshot()
    print(f"x106 refusal {args.tag}/{args.arm}: +{n_ok} ok, {n_failed} failed; ${usd:.2f}, prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}")
    if meter.aborted:
        raise SystemExit("COST CAP HIT")


if __name__ == "__main__":
    main()
