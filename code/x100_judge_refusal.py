"""x100 Stage 4: exact x83/x64 refusal2 judge over all generation chunks (or the pilot).

refusal2 is the only label construct in x100. Resumable: rows with a
successful record are skipped; failed rows are retried on the next call.
"""

from __future__ import annotations

import argparse
import glob
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from x81_judge_refusal import (
    ATTEMPT_SEEDS,
    EXTRA_TOKENS,
    JUDGE_MODEL,
    METHOD,
    REASONING,
    CostMeter,
    call,
    extract_json_object,
    judge_row,
    load_api_key,
    repair_labels,
)
from x100_common import GEN_DIR, N_RESPONSES, PILOT, RUN, read_json, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--out")
    parser.add_argument("--cap", type=float, default=20.0)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument(
        "--per-response-fallback",
        action="store_true",
        help=(
            "judge each still-unjudged condition one response per call (same model, refusal2 prompt, "
            "seeds); author-approved fallback for rows whose 10-per-call label arrays stay malformed"
        ),
    )
    return parser.parse_args()


def judge_row_per_response(api_key: str, row: dict, meter: CostMeter, generation_sha: str) -> dict | None:
    """refusal2 on one response per call, then assemble the ten labels in response order.

    Same judge model, reasoning, rubric/prompt builder (called with a one-element
    response list) and attempt seeds as the 10-per-call contract. A length-1
    array, the x81 repairs, or a bare bool under the `refusal` key is accepted
    (x86 per-response precedent). Any unjudgeable response fails the row.
    """
    session = requests.Session()
    labels, seeds, reasons = [], [], []
    for index, response in enumerate(row["llm_responses"]):
        value = None
        prompt = METHOD["prompt"]([response], row["prompt"])
        for attempt_seed in ATTEMPT_SEEDS:
            if meter.aborted:
                return None
            try:
                payload = call(session, api_key, JUDGE_MODEL, prompt, attempt_seed, REASONING, METHOD["max_tokens"] + EXTRA_TOKENS)
            except RuntimeError as error:
                reasons.append(f"r{index} transport: {error}"[:100])
                continue
            meter.add(payload.get("usage", {}))
            try:
                content = payload["choices"][0]["message"].get("content")
                parsed = extract_json_object(content or "")
            except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
                reasons.append(f"r{index} parse: {error}"[:100])
                continue
            record = METHOD["validate"](parsed, 1) or repair_labels(parsed, content, 1)[0]
            if record is None and isinstance(parsed, dict) and isinstance(parsed.get("refusal"), bool):
                record = {"refusal": [parsed["refusal"]]}
            if record is not None and len(record["refusal"]) == 1 and isinstance(record["refusal"][0], bool):
                value = record["refusal"][0]
                seeds.append(attempt_seed)
                break
            reasons.append(f"r{index} validate: {json.dumps(parsed)[:80]}")
        if value is None:
            return {
                "id": int(row["id"]), "condition": row["condition"], "prompt_sha256": row["prompt_sha256"],
                "response_sha256": row["response_sha256"], "arms_sha256": row["arms_sha256"],
                "generation_sha256": generation_sha, "judge_model": JUDGE_MODEL,
                "judge_mode": "per_response_fallback", "failed": True, "fail_reasons": reasons,
            }
        labels.append(value)
    return {
        "refusal": labels, "id": int(row["id"]), "condition": row["condition"],
        "prompt_sha256": row["prompt_sha256"], "response_sha256": row["response_sha256"],
        "arms_sha256": row["arms_sha256"], "generation_sha256": generation_sha,
        "judge_model": JUDGE_MODEL, "judge_mode": "per_response_fallback", "attempt_seeds_per_response": seeds,
    }


def load_rows(paths: list[Path]) -> tuple[dict[int, dict], dict[int, str], str]:
    rows: dict[int, dict] = {}
    source_sha: dict[int, str] = {}
    cohort_sha = None
    for path in paths:
        payload = read_json(path)
        if payload.get("experiment") != "x100":
            raise SystemExit(f"FATAL: not an x100 generation file: {path}")
        if cohort_sha is None:
            cohort_sha = payload["cohort_sha256"]
        elif payload["cohort_sha256"] != cohort_sha:
            raise SystemExit("FATAL: generation files from different cohorts")
        file_sha = sha256_file(path)
        for row in payload["rows"]:
            bid = int(row["id"])
            if bid in rows:
                raise SystemExit(f"FATAL: duplicate generation id {bid}")
            if (
                row.get("arms_sha256") != cohort_sha
                or len(row.get("llm_responses", [])) != N_RESPONSES
                or len(row.get("response_sha256", [])) != N_RESPONSES
            ):
                raise SystemExit(f"FATAL: invalid generation row {bid}")
            rows[bid] = row
            source_sha[bid] = file_sha
    if cohort_sha is None:
        raise SystemExit("FATAL: no generation files")
    return rows, source_sha, cohort_sha


def main() -> None:
    args = parse_args()
    if args.pilot:
        paths = [PILOT / "x100_pilot_generations.json"]
        out_path = Path(args.out) if args.out else PILOT / "x100_pilot_refusal2.jsonl"
    else:
        paths = [Path(p) for p in sorted(glob.glob(str(GEN_DIR / "x100_generations_chunk_*.json")))]
        out_path = Path(args.out) if args.out else RUN / "x100_refusal2.jsonl"
    rows, source_sha, cohort_sha = load_rows(paths)

    done = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("failed"):
                    continue
                bid = int(record["id"])
                source = rows.get(bid)
                if source is None:
                    raise SystemExit(f"FATAL: judge row {bid} has no generation")
                if (
                    record.get("cell_id") != source["cell_id"]
                    or record.get("prompt_sha256") != source["prompt_sha256"]
                    or record.get("response_sha256") != source["response_sha256"]
                    or record.get("arms_sha256") != cohort_sha
                    or len(record.get("refusal", [])) != N_RESPONSES
                    or not all(isinstance(v, bool) for v in record["refusal"])
                ):
                    raise SystemExit(f"FATAL: invalid successful judge resume row {bid}")
                if bid in done:
                    raise SystemExit(f"FATAL: duplicate successful judge row {bid}")
                done.add(bid)

    todo = [rows[bid] for bid in sorted(rows) if bid not in done]
    print(f"x100 refusal2: {len(rows)} generation rows, {len(done)} already judged, {len(todo)} to judge")
    if not todo:
        return
    api_key = load_api_key()
    meter = CostMeter(args.cap)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_failed = 0
    with out_path.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(args.threads) as pool:
        judge = judge_row_per_response if args.per_response_fallback else judge_row
        futures = {pool.submit(judge, api_key, row, meter, source_sha[int(row["id"])]): row for row in todo}
        for future in as_completed(futures):
            source = futures[future]
            record = future.result()
            if record is None:
                continue
            record.update(
                {
                    "label_construct": "refusal2",
                    "cell_id": source["cell_id"],
                    "population": source["population"],
                    "prompt_key": source["prompt_key"],
                    "word_index": source["word_index"],
                }
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if record.get("failed"):
                n_failed += 1
            else:
                n_ok += 1
    usd, prompt_tokens, completion_tokens = meter.snapshot()
    print(
        f"x100 refusal2: +{n_ok} ok, {n_failed} failed; ${usd:.4f}, "
        f"prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}"
    )
    if meter.aborted:
        raise SystemExit("COST CAP HIT")


if __name__ == "__main__":
    main()
