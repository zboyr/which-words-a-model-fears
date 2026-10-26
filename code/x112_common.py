"""x112: the vocabulary table built from eight over-refused prompts on the seven x106 transfer
models, and x111's held-out dose-response check on each.

Per model: the x109 candidate list (71,583 words, x109 / x110 / x111 order) appended to the eight
x100 BP prompts that best represent x111's BP table, read at the model's x106 hook block on its
x106 direction at the two boundary tokens, in the model's x106 Hard-1K sd unit (the model's own
table); 32 held-out prompts (16 x106 Hard-1K benign, 16 harmful from x101's x83 cohort and x108's
OR-Bench Toxic O rows) whose fresh twenty-sample refusal2 sits near 0.5; 40 words stratified on
the model's own table x the Qwen3.5-4B BP table (x111); cells O and E_w, twenty samples each.
The append rule, read site, direction, rendering and sampling contract are x106's / x109's /
x111's, imported, not restated.

Top-up. The 13 harmful held-out prompts that sit in the x85 crossed cohort (x85_exclusions.json)
are dropped from the analysis; on the four models concerned (Gemma 2, Qwen3.5-0.8B 1, Qwen3.5-9B
4, Qwen3.5-27B 6) the harmful stratum is topped back up to 16 prompts outside that cohort by the
same selection rule (x112_select.py --topup-request / --topup-prompts): <tag>/topup_request.json,
<tag>/prompts_topup.json, generations/generations_pool_topup.json, generations_orep_topup.json,
generations_topup_shard*.json, key_words_scan_topup.json. The frozen prompts.json is not changed.
"""

from __future__ import annotations

import hashlib
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x109_pilot_common import append_end, atomic_json, mentions, read_json, sha256_text  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RUN = DATA / "runs" / "x112"
SCAN_PROMPTS = RUN / "scan_prompts.json"
CANDIDATES = RUN / "candidates.json"
COMPARISON = RUN / "table_comparison.json"
RESULTS = RUN / "results.json"
RESULTS_MD = RUN / "results.md"
X85_EXCLUSIONS = RUN / "x85_exclusions.json"
X106_RUN = DATA / "runs" / "x106"
X109_SCAN_JSON = DATA / "runs" / "x109" / "scan.json"
X109_WORDFREQ = DATA / "runs" / "x109" / "bench" / "wordfreq_en_top50000.json"
X111_RUN = DATA / "runs" / "x111"
X111_TABLES_JSON = X111_RUN / "tables.json"
X111_TABLES_NPZ = X111_RUN / "tables.npz"
X111_REPRESENTATIVE = X111_RUN / "representative_prompts.json"
X100_COHORT = DATA / "runs" / "x100" / "x100_cohort.json"

TAGS = ("gemma", "llama", "phi", "qwen08b", "qwen2b", "qwen9b", "qwen27b")
LABELS = {
    "gemma": "Gemma-4-E2B",
    "llama": "Llama-3.2-3B",
    "phi": "Phi-4-mini",
    "qwen08b": "Qwen3.5-0.8B",
    "qwen2b": "Qwen3.5-2B",
    "qwen9b": "Qwen3.5-9B",
    "qwen27b": "Qwen3.5-27B",
}
SCAN_KEYS = ("b00393", "b00520", "b00490", "b00127", "b00711", "b00124", "b00335", "b01314")
SEED = 112
N_SAMPLES = 20
N_HALF = 10
POPULATIONS = ("benign", "harmful")


def tag_dir(tag: str) -> Path:
    if tag not in TAGS:
        raise ValueError(f"unknown x112 tag {tag}")
    return RUN / tag


def paths(tag: str) -> dict[str, Path]:
    d = tag_dir(tag)
    return {
        "dir": d,
        "scan_parts": d / "scan_parts",
        "scan_json": d / "scan.json",
        "scan_npz": d / "scan_shift.npz",
        "table_json": d / "table.json",
        "table_npz": d / "table.npz",
        "pool": d / "pool_candidates.json",
        "pool_request": d / "pool_request.json",
        "prompts": d / "prompts.json",
        "topup_request": d / "topup_request.json",
        "prompts_topup": d / "prompts_topup.json",
        "words": d / "words.json",
        "key_parts": d / "key_parts",
        "key_json": d / "key_words_scan.json",
        "key_npz": d / "key_words_shift.npz",
        "key_parts_topup": d / "key_parts_topup",
        "key_json_topup": d / "key_words_scan_topup.json",
        "key_npz_topup": d / "key_words_shift_topup.npz",
        "gen": d / "generations",
        "judge": d / "judge",
    }


@lru_cache(maxsize=None)
def sd_unit(tag: str) -> float:
    """Population sd (ddof 0) of the model's x106 Stage 0a projections over the Hard-1K prompts."""
    s = read_json(X106_RUN / tag / "x106_stage0_projections.json")
    v = np.asarray([r["proj"] for r in s["rows"] if r.get("source") == "orbench_hard"], float)
    if len(v) != 1319:
        raise SystemExit(f"FATAL: {tag}: {len(v)} Hard-1K Stage 0a rows")
    return float(v.std())


def stable_seed(tag: str, prompt_id: str, condition: str, retry: bool = False) -> int:
    label = f"{condition}:retry4096" if retry else condition
    payload = f"x112|{SEED}|{tag}|{prompt_id}|{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def conditions(words: dict) -> list[dict[str, Any]]:
    """O, then E_w for every chosen word in words.json order."""
    out = [{"condition": "O", "word": None, "kind": "O"}]
    for record in words["chosen"]:
        out.append({"condition": f"E_{record['word']}", "word": record["word"], "kind": "E"})
    return out


def cell(prompt: dict, cond: dict) -> dict[str, Any]:
    text = prompt["text"] if cond["kind"] == "O" else append_end(prompt["text"], cond["word"])
    return {
        "cell_id": f"{prompt['id']}|{cond['condition']}",
        "prompt_id": prompt["id"],
        "population": prompt["population"],
        "condition": cond["condition"],
        "kind": cond["kind"],
        "word": cond["word"],
        "prompt": text,
        "prompt_sha256": sha256_text(text),
    }


def cells(prompts: dict, words: dict) -> list[dict[str, Any]]:
    return [cell(p, c) for p in prompts["prompts"] for c in conditions(words)]


def x85_prompt_ids() -> set[str]:
    """The 96 x85 crossed-cohort prompts as held-out ids (x83:<n>), from the frozen exclusion file."""
    return {f"x83:{i}" for i in read_json(X85_EXCLUSIONS)["x85_prompt_ids_x83"]}


def pool_cells(request: dict) -> list[dict[str, Any]]:
    """The O cells of the held-out candidates requested so far (pool stage; also the top-up pool)."""
    o = {"condition": "O", "word": None, "kind": "O"}
    return [cell(p, o) for p in request["prompts"]]


def cells_sha256(cell_list: list[dict[str, Any]]) -> str:
    return sha256_text("\n".join(f"{c['cell_id']}\t{c['prompt_sha256']}" for c in cell_list))
