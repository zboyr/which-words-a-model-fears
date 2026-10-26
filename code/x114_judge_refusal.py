"""x114 Stage 3 (local): the exact x64 refusal2 judge on one (tag, condition-or-arm).

x101's / x105's judge, unchanged: `x81_judge_refusal.judge_row`, the same validation of
resume rows, the same cost meter. `--condition` takes an x114 base condition (O/S/M/E) or
one of the eight x105-style dosed arm names.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import x81_judge_refusal
from x105_common import ARM_NAMES
from x114_common import CONDITIONS, N_RESPONSES, TAGS, generations_path, read_json, refusal_path, sha256_file
from x81_judge_refusal import CostMeter, judge_row, load_api_key


def pin_provider(name: str) -> None:
    """Route every call to one named OpenRouter upstream, no fallback. The model, prompt,
    sampling parameters, seeds and validation are untouched; only the serving endpoint is
    fixed, because the default route returned short label arrays on long rows during x114."""
    original = x81_judge_refusal.call

    def call(session, key, model, prompt, seed, reasoning, max_tokens):
        post = session.post

        def pinned_post(url, json=None, **kwargs):
            json = dict(json or {})
            json["provider"] = {"order": [name], "allow_fallbacks": False}
            return post(url, json=json, **kwargs)

        session.post = pinned_post
        try:
            return original(session, key, model, prompt, seed, reasoning, max_tokens)
        finally:
            session.post = post

    x81_judge_refusal.call = call


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, choices=TAGS)
    parser.add_argument("--condition", required=True, choices=CONDITIONS + ARM_NAMES)
    parser.add_argument("--generations")
    parser.add_argument("--out")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cap", type=float, default=40.0)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--provider", help="pin one OpenRouter upstream provider (recorded as judge_provider on every record)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generation_path = Path(args.generations) if args.generations else generations_path(args.tag, args.condition)
    out_path = Path(args.out) if args.out else refusal_path(args.tag, args.condition)
    if not generation_path.exists():
        raise SystemExit(f"FATAL: generation file not found: {generation_path}")
    generations = read_json(generation_path)
    if (
        generations.get("experiment") != "x114"
        or generations.get("condition") != args.condition
        or generations.get("model_tag") != args.tag
    ):
        raise SystemExit("FATAL: generation tag/condition/experiment mismatch")
    rows = list(generations["rows"])
    if args.limit:
        rows = rows[: args.limit]
    by_id = {int(row["id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise SystemExit("FATAL: duplicate generation IDs")
    for row in rows:
        if (
            row.get("condition") != args.condition
            or row.get("arms_sha256") != generations["cohort_sha256"]
            or len(row.get("llm_responses", [])) != N_RESPONSES
            or len(row.get("response_sha256", [])) != N_RESPONSES
        ):
            raise SystemExit(f"FATAL: invalid generation row {row.get('id')}")

    generation_sha = sha256_file(generation_path)
    done = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("failed"):
                    continue
                prompt_id = int(record["id"])
                source = by_id.get(prompt_id)
                if source is None:
                    continue
                if (
                    record.get("condition") != args.condition
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

    todo = [row for row in rows if int(row["id"]) not in done]
    if not todo:
        print(f"x114 refusal {args.tag}/{args.condition}: nothing new ({len(done)} done)")
        return
    if args.provider:
        pin_provider(args.provider)
    api_key = load_api_key()
    meter = CostMeter(args.cap)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_failed = 0
    with out_path.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(args.threads) as pool:
        futures = {pool.submit(judge_row, api_key, row, meter, generation_sha): row for row in todo}
        for future in as_completed(futures):
            record = future.result()
            if record is None:
                continue
            record["model_tag"] = args.tag
            record["judge_provider"] = args.provider or "openrouter-default"
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if record.get("failed"):
                n_failed += 1
            else:
                n_ok += 1
    usd, prompt_tokens, completion_tokens = meter.snapshot()
    print(
        f"x114 refusal {args.tag}/{args.condition}: +{n_ok} ok, {n_failed} failed; "
        f"${usd:.2f}, prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}"
    )
    if meter.aborted:
        raise SystemExit("COST CAP HIT")


if __name__ == "__main__":
    main()
