"""Vocabulary scan (unnumbered): shared paths, constants and helpers.

Forty source-clean benign prompts (set A = the benign insertion pilot's 20, set B = 20 new
ones drawn by the same rules), every lowercase English candidate word appended at the end and
read on the Qwen3.5-4B boundary direction; the ten words that move the projection most on
set A, and their neutral twins, are then generated and judged on all 40 prompts.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x109_pilot_common import (  # noqa: E402,F401
    TWIN_CANDIDATES as PILOT_TWIN_CANDIDATES,
    X106_HARD1K_SD,
    append_end,
    atomic_json,
    mentions,
    read_json,
    sha256_text,
)

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data" / "runs" / "x109"
PILOT = ROOT / "data" / "runs" / "x109" / "pilot"
PROMPTS = RUN / "prompts.json"
SCAN_JSON = RUN / "scan.json"
SCAN_NPZ = RUN / "scan_shift.npz"
SCAN_PARTS = RUN / "scan_parts"
WORDS = RUN / "words.json"
GEN_DIR = RUN / "generations"
JUDGE_DIR = RUN / "judge"
RESULTS = RUN / "results.json"
RESULTS_MD = RUN / "results.md"
WORDFREQ = RUN / "bench" / "wordfreq_en_top50000.json"

TAG = "qwen"
SEED = 20261009
N_TOP = 10

# Inert concrete nouns for the neutral twins of words outside x85's 24 pairs, in fixed order:
# the pilot's list, then short nouns for 3- and 4-letter words.
TWIN_CANDIDATES = tuple(dict.fromkeys(
    PILOT_TWIN_CANDIDATES
    + ("cup", "hat", "pen", "mug", "jar", "rug", "owl", "fig", "oar", "bun",
       "bowl", "fork", "sock", "boat", "drum", "leaf", "pond", "nest", "harp", "bell",
       "spoon", "towel", "shelf", "acorn", "melon", "flute", "cloud", "daisy",
       "kettle", "carpet", "saucer", "tomato", "pencil", "basket", "cookie", "muffin")
))


def stable_seed(prompt_id: str, condition: str, retry: bool = False) -> int:
    label = f"{condition}:retry4096" if retry else condition
    payload = f"vocab_scan|{SEED}|{TAG}|{prompt_id}|{label}".encode("utf-8")
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
                    "set": prompt["set"],
                    "shape": prompt["shape"],
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
