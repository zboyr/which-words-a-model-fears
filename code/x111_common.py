"""x111: words sampled across the whole vocabulary table, appended to held-out prompts
Qwen3.5-4B refuses about half the time.

Plan: knowledge/plans/x111-vocabulary-table-dose-response.md. Three fixed per-word tables over
the x109 candidate list (benign: x109's 40 prompts; BP / HP: x110's scanned BP / HP prompts),
40 words stratified on the joint benign / BP percentile, 32 benign (x106 Hard-1K) + 32 harmful
(x83 cohort) test prompts with historical refusal2 near 0.5, outside every table's prompt set.
Cells: every prompt x (O, E_w); twenty samples per cell from one seeded sampling call, judged as
two halves of ten. The
append rule, read site, direction, rendering and sampling contract are x109's / x110's,
imported, not restated.

Top-up of the harmful stratum (after the run): nine of the 32 harmful prompts are x85 crossed-cohort
prompts and are excluded from the analysis (x85_exclusions.json). x111_select.py --topup-request /
--topup-prompts continues the harmful selection rule past the 32 drawn, with the x85 cohort and the
prompts of x110 and x112 also dropped, and adds nine prompts: topup_request.json, prompts_topup.json,
generations/generations_topup_shard{k}of{n}.json, key_words_scan_topup.json + key_words_shift_topup.npz.
prompts.json is frozen and never rewritten. Seeds are the same function of (prompt id, condition).
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x109_scan_common import (  # noqa: E402,F401
    X106_HARD1K_SD,
    append_end,
    atomic_json,
    mentions,
    read_json,
    sha256_text,
)

ROOT = Path(__file__).resolve().parents[1]
# Run namespace: `x111` by default; `X111_RUN=x117` (env, or `--run x117` on the stage scripts, which
# set the variable before this module is imported) points every per-run path and every seed at x117,
# which replays the same pipeline on x117's own prompts. The tables stay x111's fixed design.
RUN_NAME = os.environ.get("X111_RUN", "x111")
DESIGN_RUN = ROOT / "data" / "runs" / "x111"
RUN = ROOT / "data" / "runs" / RUN_NAME
PROMPTS = RUN / "prompts.json"
WORDS = RUN / "words.json"
TABLES_JSON = DESIGN_RUN / "tables.json"
TABLES_NPZ = DESIGN_RUN / "tables.npz"
KEY_JSON = RUN / "key_words_scan.json"
KEY_NPZ = RUN / "key_words_shift.npz"
KEY_PARTS = RUN / "key_parts"
GEN_DIR = RUN / "generations"
JUDGE_DIR = RUN / "judge"
REPRESENTATIVE = RUN / "representative_prompts.json"
X85_EXCLUSIONS = RUN / "x85_exclusions.json"
TOPUP_REQUEST = RUN / "topup_request.json"
PROMPTS_TOPUP = RUN / "prompts_topup.json"
KEY_JSON_TOPUP = RUN / "key_words_scan_topup.json"
KEY_NPZ_TOPUP = RUN / "key_words_shift_topup.npz"
KEY_PARTS_TOPUP = RUN / "key_parts_topup"
X112_RUN = ROOT / "data" / "runs" / "x112"
RESULTS = RUN / "results.json"
RESULTS_MD = RUN / "results.md"
X109_RUN = ROOT / "data" / "runs" / "x109"
X109_SCAN_JSON = X109_RUN / "scan.json"
X109_SCAN_NPZ = X109_RUN / "scan_shift.npz"
X109_PROMPTS = X109_RUN / "prompts.json"
X109_WORDFREQ = X109_RUN / "bench" / "wordfreq_en_top50000.json"
X110_RUN = ROOT / "data" / "runs" / "x110"
X110_SCAN_JSON = X110_RUN / "scan.json"
X110_SCAN_NPZ = X110_RUN / "scan_shift.npz"
X110_PROMPTS = X110_RUN / "prompts.json"
X100_COHORT = ROOT / "data" / "runs" / "x100" / "x100_cohort.json"
X106_QWEN = ROOT / "data" / "runs" / "x106" / "qwen"
X106_REFUSAL_O = X106_QWEN / "x106_refusal_O_base.jsonl"
X106_GENERATIONS_O = X106_QWEN / "x106_generations_O_base.json"
X83_RUN = ROOT / "data" / "runs" / "x83"
X83_REFUSAL_O = X83_RUN / "x83_refusal_O.jsonl"
X83_COHORT = X83_RUN / "x83_cohort.json"

TAG = "qwen"
SEED = 111
N_SAMPLES = 20
N_HALF = 10
POPULATIONS = ("benign", "harmful")
TABLES = ("benign", "BP", "HP")
ANALYSIS_TABLES = ("benign", "BP", "HP", "mixed")


def stable_seed(prompt_id: str, condition: str, retry: bool = False) -> int:
    label = f"{condition}:retry4096" if retry else condition
    payload = f"{RUN_NAME}|{SEED}|{TAG}|{prompt_id}|{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def conditions(words: dict) -> list[dict[str, Any]]:
    """O, then E_w for every chosen word in words.json order."""
    out = [{"condition": "O", "word": None, "kind": "O"}]
    for record in words["chosen"]:
        out.append({"condition": f"E_{record['word']}", "word": record["word"], "kind": "E"})
    return out


def cells(prompts: dict, words: dict) -> list[dict[str, Any]]:
    out = []
    for prompt in prompts["prompts"]:
        for cond in conditions(words):
            text = prompt["text"] if cond["kind"] == "O" else append_end(prompt["text"], cond["word"])
            out.append(
                {
                    "cell_id": f"{prompt['id']}|{cond['condition']}",
                    "prompt_id": prompt["id"],
                    "population": prompt["population"],
                    "condition": cond["condition"],
                    "kind": cond["kind"],
                    "word": cond["word"],
                    "prompt": text,
                    "prompt_sha256": sha256_text(text),
                }
            )
    return out


def cells_sha256(cell_list: list[dict[str, Any]]) -> str:
    return sha256_text("\n".join(f"{c['cell_id']}\t{c['prompt_sha256']}" for c in cell_list))
