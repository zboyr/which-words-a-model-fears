"""x116: does it matter that x113 substitutes the synonym its table places lowest?

Plan: knowledge/plans/x116-random-synonym-control.md (deleted once the write-up exists). One new
arm on x113's units: R_rand, the x113 nominee replaced in place by a candidate drawn uniformly at
random from the drafter's accepted candidates of the same unit, excluding the table-lowest one
x113 used (S_T). O and S_T are x113's rows, reused. Rendering, sampling, the read site, the
judge, the splice and the bootstrap are x113's, imported, not restated.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x106_common import stable_seed as x106_stable_seed  # noqa: E402
from x113_common import (  # noqa: E402,F401
    ALL_TAGS,
    DATA,
    FRESH_TAGS,
    IMPORTED_TAG,
    LABELS,
    N_RESPONSES,
    STRATA,
    atomic_json,
    file_record,
    read_json,
    sha256_text,
)
from x113_common import paths as x113_paths  # noqa: E402

RUN = DATA / "runs" / "x116"
SEED = 116
ARM = "R_rand"
TEXT_LABEL = "R_rand"  # x106 stable_seed text label of the new arm (S_T uses "S", C_T "C")
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 11_600


def paths(tag: str) -> dict[str, Path]:
    if tag not in ALL_TAGS:
        raise ValueError(f"unknown tag {tag!r}")
    d = RUN / tag
    return {
        "dir": d,
        "texts": d / "texts.json",
        "gen": d / "generations",
        "genlocks": d / "genlocks",
        "judge": d / "judge",
        "refusal2": d / "judge" / "refusal2.jsonl",
        "cost_log": d / "judge" / "cost_log.jsonl",
        "projections": d / "projections.json",
    }


def draw_rng_seed(tag: str, unit_id: str) -> int:
    """Per-(model, unit) seed of the random candidate draw: independent of unit order and of the
    other models."""
    return int.from_bytes(hashlib.sha256(f"x116|{SEED}|{tag}|{unit_id}".encode()).digest()[:8], "big")


def stable_seed(tag: str, prompt_id: str, retry: bool = False) -> int:
    return x106_stable_seed(tag, prompt_id, TEXT_LABEL, retry=retry)


def texts_sha256(units: list[dict[str, Any]]) -> str:
    return sha256_text("\n".join(f"{u['id']}\t{u.get('sha256') or '-'}" for u in units))


def load_texts(tag: str) -> dict[str, Any]:
    obj = read_json(paths(tag)["texts"])
    if obj.get("experiment") != "x116" or obj.get("model_tag") != tag or obj.get("status") != "frozen":
        raise SystemExit(f"FATAL: x116 texts for {tag} are missing or not frozen")
    if obj.get("texts_sha256") != texts_sha256(obj["units"]):
        raise SystemExit(f"FATAL: {tag}: x116 texts.json fails its sha256 check")
    for u in obj["units"]:
        if u["text"] is not None and sha256_text(u["text"]) != u["sha256"]:
            raise SystemExit(f"FATAL: {tag}: x116 text {u['id']} fails its sha256")
    return obj


def cells(tag: str, texts: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"cell_id": f"{u['id']}|{ARM}", "prompt_id": u["id"], "stratum": u["stratum"], "arm": ARM, "prompt": u["text"], "prompt_sha256": u["sha256"], "seed": stable_seed(tag, u["id"])}
        for u in texts["units"]
        if u["text"] is not None
    ]


def cells_sha256(cell_list: list[dict[str, Any]]) -> str:
    return sha256_text("\n".join(f"{c['cell_id']}\t{c['prompt_sha256']}" for c in cell_list))
