"""x102 stage 1: freeze the 300-prompt OR-Bench sample over the 35 model-conditions.

Every judgment file's recorded generation sha256 is re-verified against the
generation file next to it, the row ids are checked to coincide, and the
builder fails closed on any drift.  The same 300 OR-Bench ids are used for
every model and condition.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from x102_common import (
    API_ARMS,
    API_PRICES_READ,
    BENCHMARKS,
    MANIFEST_PATH,
    N_HARD,
    N_PILOT,
    N_REPEAT,
    N_SAMPLE_ROWS,
    N_TOXIC,
    OPEN_ARMS,
    ORBENCH_PROMPT_SHA256,
    ORBENCH_UPSTREAM_COMMIT,
    ORBENCH_UPSTREAM_FILE,
    ORBENCH_UPSTREAM_FILE_SHA256,
    ORBENCH_UPSTREAM_REPO,
    PILOT_SEED,
    REPEAT_SEED,
    SAMPLE_PATH,
    SAMPLE_SEED,
    SOURCES,
    atomic_json,
    draw_sample_ids,
    draw_subset,
    file_record,
    keys_sha256,
    read_json,
    row_key,
    sha256_file,
    source_paths,
)


def load_source(tag: str, experiment: str, condition: str) -> tuple[dict, dict, dict[str, dict], dict[str, dict]]:
    judgment_path, generation_path = source_paths(tag, experiment, condition)
    if not judgment_path.exists() or not generation_path.exists():
        raise SystemExit(f"FATAL: missing source for {tag}/{condition}: {judgment_path} / {generation_path}")
    judgment = read_json(judgment_path)
    generation = read_json(generation_path)
    if judgment.get("condition") != condition or generation.get("condition") != condition:
        raise SystemExit(f"FATAL: condition mismatch in {judgment_path}")
    if experiment == "x88" and (judgment.get("model_tag") != tag or generation.get("model_tag") != tag):
        raise SystemExit(f"FATAL: model tag mismatch in {judgment_path}")
    if judgment.get("judge_model_id") != "Qwen/Qwen3.5-4B":
        raise SystemExit(f"FATAL: {judgment_path} was not judged by the incumbent")
    recorded = judgment.get("generation", {}).get("sha256")
    actual = sha256_file(generation_path)
    if recorded != actual:
        raise SystemExit(f"FATAL: {judgment_path} records generation sha {recorded} but the file has {actual}")
    judged = {row["id"]: row for row in judgment["rows"]}
    generated = {
        row["id"]: row for row in generation["rows"] if row["benchmark"] in BENCHMARKS
    }
    if set(judged) != set(generated):
        raise SystemExit(f"FATAL: judgment/generation ids differ for {tag}/{condition}")
    if len(judged) != 1974:
        raise SystemExit(f"FATAL: {tag}/{condition} has {len(judged)} OR-Bench rows, expected 1974")
    for row_id, row in judged.items():
        if row["label"] not in ("direct_answer", "direct_refusal", "indirect_refusal"):
            raise SystemExit(f"FATAL: bad incumbent label in {tag}/{condition}: {row}")
        if row["refusal"] != int(row["label"] != "direct_answer"):
            raise SystemExit(f"FATAL: incumbent refusal bit inconsistent at {row_id}")
        if row["benchmark"] != generated[row_id]["benchmark"]:
            raise SystemExit(f"FATAL: benchmark mismatch at {row_id}")
    return judgment, generation, judged, generated


def main() -> None:
    sources = []
    per_source: dict[tuple[str, str], tuple[dict[str, dict], dict[str, dict]]] = {}
    reference_ids: dict[str, set[str]] | None = None
    for tag, experiment, condition, role in SOURCES:
        judgment_path, generation_path = source_paths(tag, experiment, condition)
        _, _, judged, generated = load_source(tag, experiment, condition)
        ids_by_benchmark = {
            benchmark: {i for i, row in judged.items() if row["benchmark"] == benchmark}
            for benchmark in BENCHMARKS
        }
        if reference_ids is None:
            reference_ids = ids_by_benchmark
        elif ids_by_benchmark != reference_ids:
            raise SystemExit(f"FATAL: {tag}/{condition} does not share the OR-Bench id set")
        per_source[(tag, condition)] = (judged, generated)
        sources.append(
            {
                "tag": tag,
                "experiment": experiment,
                "condition": condition,
                "role": role,
                "judgment": file_record(judgment_path),
                "generation": file_record(generation_path),
            }
        )
        print(f"verified {tag}/{condition} ({experiment})", flush=True)
    assert reference_ids is not None
    if len(reference_ids["orbench_hard"]) != 1319 or len(reference_ids["orbench_toxic"]) != 655:
        raise SystemExit("FATAL: unexpected OR-Bench id counts")

    hard_ids, toxic_ids = draw_sample_ids(reference_ids["orbench_hard"], reference_ids["orbench_toxic"])
    sampled = [(i, "orbench_hard") for i in hard_ids] + [(i, "orbench_toxic") for i in toxic_ids]
    rows = []
    for source in sources:
        tag, condition = source["tag"], source["condition"]
        judged, generated = per_source[(tag, condition)]
        for orbench_id, benchmark in sampled:
            j = judged[orbench_id]
            g = generated[orbench_id]
            if g["prompt_sha256"] != g["prompt_sha256"]:
                raise SystemExit("unreachable")
            rows.append(
                {
                    "key": row_key(tag, condition, orbench_id),
                    "tag": tag,
                    "experiment": source["experiment"],
                    "condition": condition,
                    "role": source["role"],
                    "id": orbench_id,
                    "benchmark": benchmark,
                    "prompt": g["prompt"],
                    "response": g["response"],
                    "prompt_sha256": g["prompt_sha256"],
                    "response_sha256": g["response_sha256"],
                    "incumbent_label": j["label"],
                    "incumbent_parse": j["label_parse"],
                    "incumbent_refusal": int(j["refusal"]),
                }
            )
    if len(rows) != N_SAMPLE_ROWS:
        raise SystemExit(f"FATAL: built {len(rows)} rows, expected {N_SAMPLE_ROWS}")
    keys = [row["key"] for row in rows]
    if len(set(keys)) != len(keys):
        raise SystemExit("FATAL: duplicate row keys")
    sample_sha = keys_sha256(keys)
    pilot_keys = draw_subset(keys, N_PILOT, PILOT_SEED)
    repeat_keys = draw_subset(keys, N_REPEAT, REPEAT_SEED)
    sample = {
        "schema_version": 1,
        "experiment": "x102",
        "built": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "sample_seed": SAMPLE_SEED,
        "n_hard": N_HARD,
        "n_toxic": N_TOXIC,
        "hard_ids": hard_ids,
        "toxic_ids": toxic_ids,
        "sources": sources,
        "subsets": {
            "pilot": {"seed": PILOT_SEED, "keys": pilot_keys},
            "repeat": {"seed": REPEAT_SEED, "keys": repeat_keys},
        },
        "sample_sha256": sample_sha,
        "rows": rows,
    }
    atomic_json(SAMPLE_PATH, sample)
    manifest = read_json(MANIFEST_PATH) if MANIFEST_PATH.exists() else {}
    manifest.update(
        {
            "schema_version": 1,
            "experiment": "x102",
            "sample_sha256": sample_sha,
            "sample": file_record(SAMPLE_PATH),
            "sources": [
                {k: v for k, v in s.items() if k not in ("judgment", "generation")}
                | {"judgment_sha256": s["judgment"]["sha256"], "generation_sha256": s["generation"]["sha256"]}
                for s in sources
            ],
            "benchmark_prompt": {
                "repo": ORBENCH_UPSTREAM_REPO,
                "commit": ORBENCH_UPSTREAM_COMMIT,
                "file": ORBENCH_UPSTREAM_FILE,
                "file_sha256": ORBENCH_UPSTREAM_FILE_SHA256,
                "prompt_sha256": ORBENCH_PROMPT_SHA256,
            },
            "api_arms_planned": {arm: dict(spec) for arm, spec in API_ARMS.items()},
            "api_prices_read": API_PRICES_READ,
            "open_arms_planned": {arm: dict(spec) for arm, spec in OPEN_ARMS.items()},
        }
    )
    atomic_json(MANIFEST_PATH, manifest)
    n_in = sum(len(r["prompt"]) + len(r["response"]) for r in rows)
    print(
        f"x102 sample: {len(rows)} rows, sha256 {sample_sha[:12]}, "
        f"{n_in / len(rows):.0f} chars/row, pilot {len(pilot_keys)}, repeat {len(repeat_keys)}"
    )
    print(f"wrote {SAMPLE_PATH} ({Path(SAMPLE_PATH).stat().st_size / 1e6:.1f} MB) and {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
