"""Benign insertion pilot (unnumbered): shared paths, constants and helpers.

Twenty source-clean benign prompts (Dolly passage+task, No Robots instruction-only prose),
one harm-lexicon word appended at the end (x83 position E) or its neutral twin, Qwen3.5-4B
only. The read site, direction, rendering and sampling contract are x106's.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data" / "runs" / "x109" / "pilot"
PROMPTS = RUN / "prompts.json"
PRE_WORDS = RUN / "words_pre.json"
WORDS = RUN / "words.json"
PROJECTIONS = RUN / "projections.json"
GEN_DIR = RUN / "generations"
JUDGE_DIR = RUN / "judge"
RESULTS = RUN / "results.json"
RESULTS_MD = RUN / "results.md"

TAG = "qwen"
SEED = 20261008
N_PER_SHAPE = 10
X106_HARD1K_SD = 8.645083779734692  # population sd of the x106 Stage 0a Hard-1K projections, Qwen3.5-4B

SHAPE_A_CATEGORIES = ("closed_qa", "information_extraction", "summarization")
SHAPE_B_CATEGORIES = ("Generation", "Brainstorm", "Rewrite", "Summarize", "Coding")

# Inert concrete nouns for the neutral twins of words outside x85's 24 pairs, in fixed order.
# None is in the x83 lexicon, none is an x85 neutral, none is harm-adjacent.
TWIN_CANDIDATES = (
    "lamp", "vase", "kite", "sofa", "pear", "desk",
    "tulip", "apron", "piano", "maple", "lemon", "cabin", "chair", "plate",
    "pillow", "ladder", "bucket", "jacket", "carrot", "almond", "button", "napkin", "violin", "candle",
    "teapot", "pebble", "walnut", "ribbon",
    "trumpet", "cabbage", "curtain", "sweater", "popcorn", "feather", "pancake", "teacup", "biscuit",
    "lemonade", "sailboat", "doorknob", "bookcase", "mandolin", "pinecone", "notebook", "umbrella", "sandwich",
    "pineapple", "blueberry", "porcelain", "telescope", "cardboard", "dandelion", "sunflower", "honeycomb",
    "watercolor", "strawberry", "watermelon", "chandelier", "toothbrush", "tablecloth", "windowsill", "embroidery", "terracotta",
    "grasshopper", "marshmallow", "thermometer", "wheelbarrow", "candlestick", "hummingbird", "cauliflower", "gingerbread",
    "refrigerator", "kaleidoscope", "harpsichord", "chrysanthemum", "constellation",
)

WORD_TOKEN = re.compile(r"[A-Za-z]+")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def append_end(text: str, word: str) -> str:
    """x83_common.insert_word(text, word, "E"): one space and the word after the text."""
    return f"{text} {word}"


def stable_seed(prompt_id: str, condition: str, retry: bool = False) -> int:
    label = f"{condition}:retry4096" if retry else condition
    payload = f"benign_insert_pilot|{SEED}|{TAG}|{prompt_id}|{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def conditions(words: dict) -> list[dict[str, str]]:
    """O, then E_w and N_w for every union word in words.json order."""
    out = [{"condition": "O", "word": None, "kind": "O"}]
    for record in words["union"]:
        out.append({"condition": f"E_{record['word']}", "word": record["word"], "kind": "E"})
        out.append({"condition": f"N_{record['twin']}", "word": record["twin"], "kind": "N"})
    return out


def cells(prompts: dict, words: dict) -> list[dict[str, Any]]:
    """Every (prompt, condition) generation cell with its text and sha256, in fixed order."""
    out = []
    for prompt in prompts["prompts"]:
        for cond in conditions(words):
            text = prompt["text"] if cond["kind"] == "O" else append_end(prompt["text"], cond["word"])
            out.append(
                {
                    "cell_id": f"{prompt['id']}|{cond['condition']}",
                    "prompt_id": prompt["id"],
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


def mentions(response: str, word: str) -> bool:
    return word.lower() in {w.lower() for w in WORD_TOKEN.findall(response)}
