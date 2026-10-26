"""x110: the x109 vocabulary table's most negative words appended to x100's prompts.

Plan: knowledge/plans/x110-table-courage-words.md. The prompts are x100's two populations
(HP: 96 harmful prompts in three historical refusal2 strata; BP: 96 OR-Bench Hard-1K prompts
the x87 base refused) exactly as x100_cohort.json records them. The words are the ten most
negative distinct words of the x109 scan table (mean shift over its 40 benign prompts), each
with a neutral twin by x109's rule. The append rule, read site, direction, rendering and
sampling contract are x109's, imported, not restated.

Top-up. The 15 HP prompts that sit in the x85 crossed cohort (x85_exclusions.json: 11 in r0.2-0.5,
4 in r0.6-0.9) are dropped from the analysis; the HP strata are refilled to 32 each with prompts
outside that cohort by x100's HP rule continued (x110_select.py --topup-request / --topup-prompts):
topup_request.json, prompts_topup.json, generations/generations_topup_shard*.json,
key_words_scan_topup.json. The frozen prompts.json is not changed.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x109_scan_common import (  # noqa: E402,F401
    TWIN_CANDIDATES,
    X106_HARD1K_SD,
    append_end,
    atomic_json,
    mentions,
    read_json,
    sha256_text,
)

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data" / "runs" / "x110"
PROMPTS = RUN / "prompts.json"
WORDS = RUN / "words.json"
SCAN_JSON = RUN / "scan.json"
SCAN_NPZ = RUN / "scan_shift.npz"
SCAN_PARTS = RUN / "scan_parts"
KEY_JSON = RUN / "key_words_scan.json"
KEY_NPZ = RUN / "key_words_shift.npz"
KEY_PARTS = RUN / "key_parts"
GEN_DIR = RUN / "generations"
JUDGE_DIR = RUN / "judge"
RESULTS = RUN / "results.json"  # primary: the x85-cohort prompts excluded (EXCLUSIONS)
RESULTS_MD = RUN / "results.md"
RESULTS_FULL = RUN / "results_full.json"  # every prompt, --full
RESULTS_FULL_MD = RUN / "results_full.md"
EXCLUSIONS = RUN / "x85_exclusions.json"  # frozen: the 15 HP prompts that are x85 prompts
RESULTS_PRE_TOPUP = RUN / "results_pre_topup.json"  # the exclusion without the top-up, --no-topup
RESULTS_PRE_TOPUP_MD = RUN / "results_pre_topup.md"
TOPUP_REQUEST = RUN / "topup_request.json"
PROMPTS_TOPUP = RUN / "prompts_topup.json"
KEY_JSON_TOPUP = RUN / "key_words_scan_topup.json"
KEY_NPZ_TOPUP = RUN / "key_words_shift_topup.npz"
KEY_PARTS_TOPUP = RUN / "key_parts_topup"
X109_RUN = ROOT / "data" / "runs" / "x109"
X109_SCAN_JSON = X109_RUN / "scan.json"
X109_SCAN_NPZ = X109_RUN / "scan_shift.npz"
X109_WORDFREQ = X109_RUN / "bench" / "wordfreq_en_top50000.json"
X100_RUN = ROOT / "data" / "runs" / "x100"
X100_COHORT = X100_RUN / "x100_cohort.json"
X100_REFUSAL = X100_RUN / "x100_refusal2.jsonl"

TAG = "qwen"
SEED = 110
N_TOP = 10
POPULATIONS = ("HP", "BP")


def stable_seed(prompt_id: str, condition: str, retry: bool = False) -> int:
    label = f"{condition}:retry4096" if retry else condition
    payload = f"x110|{SEED}|{TAG}|{prompt_id}|{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def conditions(words: dict) -> list[dict[str, Any]]:
    """O, then E_w and N_w for every chosen word in words.json order."""
    out = [{"condition": "O", "word": None, "kind": "O"}]
    for record in words["chosen"]:
        out.append({"condition": f"E_{record['word']}", "word": record["word"], "kind": "E"})
        out.append({"condition": f"N_{record['twin']}", "word": record["twin"], "kind": "N"})
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
                    "stratum": prompt.get("stratum"),
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
