"""x104 stage 1: collect every OR-Bench response of x87/x88/x89 into one row file.

For each of the 49 model-conditions the judgment file's recorded generation
sha256 is re-verified against the generation file next to it, the row ids of
the two files are checked to coincide, and the builder fails closed on any
drift.  Writes `data/runs/x104/x104_rows.jsonl` (96,726 rows) and
`x104_manifest.json` (source records, row-file sha256).
"""

from __future__ import annotations

import datetime as dt

from x104_common import (
    BENCHMARKS,
    JUDGE,
    MANIFEST_PATH,
    N_ROWS,
    RESAMPLE_SEEDS,
    RESAMPLE_TEMPERATURE,
    ROWS_PATH,
    ROWS_PER_SOURCE,
    SOURCES,
    atomic_json,
    file_record,
    keys_sha256,
    read_json,
    row_key,
    sha256_file,
    source_paths,
    write_jsonl,
)


def load_source(tag: str, experiment: str, condition: str) -> tuple[dict[str, dict], dict[str, dict]]:
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
    generated = {row["id"]: row for row in generation["rows"] if row["benchmark"] in BENCHMARKS}
    if set(judged) != set(generated):
        raise SystemExit(f"FATAL: judgment/generation ids differ for {tag}/{condition}")
    if len(judged) != ROWS_PER_SOURCE:
        raise SystemExit(f"FATAL: {tag}/{condition} has {len(judged)} OR-Bench rows, expected {ROWS_PER_SOURCE}")
    for row_id, row in judged.items():
        if row["refusal"] != int(row["label"] != "direct_answer"):
            raise SystemExit(f"FATAL: incumbent refusal bit inconsistent at {row_id}")
        if row["benchmark"] != generated[row_id]["benchmark"]:
            raise SystemExit(f"FATAL: benchmark mismatch at {row_id}")
    return judged, generated


def main() -> None:
    sources = []
    rows = []
    reference_ids: dict[str, set[str]] | None = None
    for tag, experiment, condition in SOURCES:
        judgment_path, generation_path = source_paths(tag, experiment, condition)
        judged, generated = load_source(tag, experiment, condition)
        ids_by_benchmark = {
            benchmark: {i for i, row in judged.items() if row["benchmark"] == benchmark}
            for benchmark in BENCHMARKS
        }
        if reference_ids is None:
            reference_ids = ids_by_benchmark
        elif ids_by_benchmark != reference_ids:
            raise SystemExit(f"FATAL: {tag}/{condition} does not share the OR-Bench id set")
        sources.append(
            {
                "tag": tag,
                "experiment": experiment,
                "condition": condition,
                "judgment": file_record(judgment_path),
                "generation": file_record(generation_path),
            }
        )
        for orbench_id in sorted(judged):
            j = judged[orbench_id]
            g = generated[orbench_id]
            rows.append(
                {
                    "key": row_key(tag, condition, orbench_id),
                    "tag": tag,
                    "experiment": experiment,
                    "condition": condition,
                    "id": orbench_id,
                    "benchmark": j["benchmark"],
                    "prompt": g["prompt"],
                    "response": g["response"],
                    "prompt_sha256": g["prompt_sha256"],
                    "response_sha256": g["response_sha256"],
                    "incumbent_label": j["label"],
                    "incumbent_refusal": int(j["refusal"]),
                }
            )
        print(f"verified {tag}/{condition} ({experiment})", flush=True)
    assert reference_ids is not None
    if len(reference_ids["orbench_hard"]) != 1319 or len(reference_ids["orbench_toxic"]) != 655:
        raise SystemExit("FATAL: unexpected OR-Bench id counts")
    if len(rows) != N_ROWS:
        raise SystemExit(f"FATAL: built {len(rows)} rows, expected {N_ROWS}")
    keys = [row["key"] for row in rows]
    if len(set(keys)) != len(keys):
        raise SystemExit("FATAL: duplicate row keys")
    rows.sort(key=lambda r: r["key"])
    write_jsonl(ROWS_PATH, rows)
    manifest = {
        "schema_version": 1,
        "experiment": "x104",
        "built": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "n_rows": len(rows),
        "n_sources": len(sources),
        "rows": file_record(ROWS_PATH),
        "rows_keys_sha256": keys_sha256(keys),
        "sources": sources,
        "judge": {
            "model_id": JUDGE["model_id"],
            "revision": JUDGE["revision"],
            "max_new_tokens": JUDGE["max_new_tokens"],
            "format": JUDGE["format"],
            "resample": {"temperature": RESAMPLE_TEMPERATURE, "seeds": list(RESAMPLE_SEEDS)},
        },
    }
    atomic_json(MANIFEST_PATH, manifest)
    n_chars = sum(len(r["prompt"]) + len(r["response"]) for r in rows)
    print(
        f"x104 rows: {len(rows)} from {len(sources)} sources, keys sha256 "
        f"{manifest['rows_keys_sha256'][:12]}, {n_chars / len(rows):.0f} chars/row"
    )
    print(f"wrote {ROWS_PATH} ({ROWS_PATH.stat().st_size / 1e6:.1f} MB) and {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
