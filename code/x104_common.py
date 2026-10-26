"""Shared contracts for x104: GuardReasoner-8B re-judges every OR-Bench response of x87/x88/x89.

The judge is x102's GuardReasoner arm (`x102_common.OPEN_ARMS["guardreasoner"]`,
`guardreasoner_input`, `parse_guardreasoner`), imported, never restated.  x104
runs it over all 96,726 OR-Bench responses the three experiments generated and
rebuilds their result tables on its `Completion: refusal/compliance` field.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from x102_common import (  # noqa: E402,F401
    BENCHMARKS,
    OPEN_ARMS,
    QWEN_TAG,
    TRANSFER_CONDITIONS,
    TRANSFER_TAGS,
    atomic_json,
    file_record,
    guardreasoner_input,
    keys_sha256,
    parse_guardreasoner,
    read_json,
    read_jsonl,
    row_key,
    sha256_file,
    source_paths,
)
from x87_common import ALL_CONDITIONS as X87_ALL_CONDITIONS  # noqa: E402

ROOT = HERE.parent
DATA = ROOT / "data"
RUN = DATA / "runs" / "x104"
RESULTS = DATA / "results"
ROWS_PATH = RUN / "x104_rows.jsonl"
MANIFEST_PATH = RUN / "x104_manifest.json"
JUDGMENTS_PATH = RUN / "x104_judgments.jsonl"
RESULT_PATHS = {
    "x87": RESULTS / "x104_external_safety_utility.json",
    "x88": RESULTS / "x104_cross_model_deep_dose.json",
    "x89": RESULTS / "x104_positive_dose_extension.json",
}

# The 49 model-conditions: x87 (14), x89 (7), x88 (7 transfer models x 4).
X89_CONDITIONS = ("x86_p050", "x86_p100", "x86_p150", "x75_m1", "x75_m2", "x75_p1", "x75_p2")
X87_CONDITIONS = tuple(c for c in X87_ALL_CONDITIONS if c not in X89_CONDITIONS)
SOURCES: tuple[tuple[str, str, str], ...] = (
    tuple((QWEN_TAG, "x87", c) for c in X87_CONDITIONS)
    + tuple((QWEN_TAG, "x89", c) for c in X89_CONDITIONS)
    + tuple((tag, "x88", c) for tag in TRANSFER_TAGS for c in TRANSFER_CONDITIONS)
)
ROWS_PER_SOURCE = 1974
N_ROWS = len(SOURCES) * ROWS_PER_SOURCE  # 96,726

JUDGE = OPEN_ARMS["guardreasoner"]
DEFAULT_BATCH = 48  # x102's GuardReasoner run setting (BATCH=48 on one B200)
RESAMPLE_TEMPERATURE = 0.6
RESAMPLE_SEEDS = (1, 2, 3)


def shard_path(shard: int, n_shards: int) -> Path:
    return RUN / f"x104_judgments_shard{shard}of{n_shards}.jsonl"


def resample_path(shard: int, n_shards: int) -> Path:
    return RUN / f"x104_resample_shard{shard}of{n_shards}.jsonl"


def summary_path(shard: int, n_shards: int) -> Path:
    return RUN / f"x104_summary_shard{shard}of{n_shards}.json"


def shard_rows(rows: list[dict[str, Any]], shard: int, n_shards: int) -> list[dict[str, Any]]:
    """Deterministic shard: rows sorted by key, every n-th starting at `shard`."""
    if not 0 <= shard < n_shards:
        raise ValueError(f"shard {shard} outside 0..{n_shards - 1}")
    ordered = sorted(rows, key=lambda r: r["key"])
    return ordered[shard::n_shards]


def load_rows(path: Path = ROWS_PATH, manifest_path: Path = MANIFEST_PATH) -> list[dict[str, Any]]:
    """The frozen row file, checked against the sha256 the manifest registered."""
    manifest = read_json(manifest_path)
    actual = sha256_file(path)
    if manifest.get("rows", {}).get("sha256") != actual:
        raise SystemExit(f"FATAL: {path} ({actual[:12]}) is not the row file the manifest registered")
    rows = read_jsonl(path)
    if len(rows) != N_ROWS or keys_sha256(r["key"] for r in rows) != manifest["rows_keys_sha256"]:
        raise SystemExit(f"FATAL: {path} does not carry the registered {N_ROWS} row keys")
    return rows


def load_judgments(path: Path = JUDGMENTS_PATH) -> dict[str, dict[str, Any]]:
    """The merged judgments, one record per row key."""
    records = read_jsonl(path)
    judged = {}
    for record in records:
        if record["key"] in judged:
            raise SystemExit(f"FATAL: duplicate judged row {record['key']} in {path}")
        judged[record["key"]] = record
    return judged


def read_shard_jsonl(path: Path) -> list[dict[str, Any]]:
    """A shard file as written by the judge; a final line cut short by a job's time
    limit is dropped (that row is judged again), any other malformed line is fatal."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        lines = [line for line in handle if line.strip()]
    records = []
    for index, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                print(f"x104: dropping the truncated final line of {path}", flush=True)
                continue
            raise SystemExit(f"FATAL: malformed line {index + 1} in {path}")
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)
