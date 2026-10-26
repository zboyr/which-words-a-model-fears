"""Scout source-guaranteed benign instruction datasets for a clean insertion cohort.

For each candidate dataset: build the user-turn text, count words, flag x83
lexicon hits and a stricter harm-adjacent stem list, keep the task category,
and report how many prompts survive each filter and the cohort length band
(x83 cohort: median 127 words, q1/q3 106/155). Survivors are written to
data/pools/x109_clean_benign_candidates.json for the next step (base generation).
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x83_common import lexicon_words  # noqa: E402

from datasets import load_dataset  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "pools" / "x109_clean_benign_candidates.json"

LEX = {w for w, _ in lexicon_words()}
# Stricter harm-adjacent stems outside the lexicon (prefix match on word).
ADJ = (
    "weapon drug crim violen death dead die dying attack poison steal stole war "
    "abus victim threat injur wound blood knife sword fight army military police "
    "prison jail arrest crime criminal danger risk illegal sex drunk alcohol "
    "cigarette tobacco gamble casino disease cancer virus infect pandemic "
    "terror hostage slave racis sexis nazi hitler genocid"
).split()

BAND_TIGHT = (106, 155)   # cohort q1/q3
BAND_LOOSE = (90, 170)

WORD = re.compile(r"[A-Za-z']+")


def words(text: str) -> list[str]:
    return [w.lower().strip("'") for w in WORD.findall(text)]


def flags(text: str) -> tuple[list[str], list[str]]:
    ws = words(text)
    lex = sorted({w for w in ws if w in LEX})
    adj = sorted({w for w in ws if any(w.startswith(s) for s in ADJ)})
    return lex, adj


def is_english(text: str) -> bool:
    if not text:
        return False
    ascii_ratio = sum(c.isascii() for c in text) / len(text)
    return ascii_ratio > 0.97


def dolly():
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    for i, r in enumerate(ds):
        text = r["instruction"].strip()
        if r.get("context"):
            text = text + "\n\n" + r["context"].strip()
        yield f"dolly:{i}", text, r["category"]


def no_robots():
    ds = load_dataset("HuggingFaceH4/no_robots", split="train")
    for i, r in enumerate(ds):
        yield f"no_robots:{i}", r["prompt"].strip(), r["category"]


def alpaca_cleaned():
    ds = load_dataset("yahma/alpaca-cleaned", split="train")
    for i, r in enumerate(ds):
        text = r["instruction"].strip()
        if r.get("input"):
            text = text + "\n\n" + r["input"].strip()
        yield f"alpaca_cleaned:{i}", text, "with_input" if r.get("input") else "no_input"


def lima():
    ds = load_dataset("GAIR/lima", split="train")
    for i, r in enumerate(ds):
        conv = r["conversations"]
        if not conv:
            continue
        yield f"lima:{i}", conv[0].strip(), r.get("source", "")


def openorca(limit=60000):
    ds = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
    for i, r in enumerate(ds):
        if i >= limit:
            break
        yield f"openorca:{r['id']}", r["question"].strip(), r["id"].split(".")[0]


def gsm8k():
    ds = load_dataset("openai/gsm8k", "main", split="train")
    for i, r in enumerate(ds):
        yield f"gsm8k:{i}", r["question"].strip(), "math"


def ifeval():
    ds = load_dataset("google/IFEval", split="train")
    for i, r in enumerate(ds):
        yield f"ifeval:{i}", r["prompt"].strip(), "ifeval"


SOURCES = {
    "dolly": dolly,
    "no_robots": no_robots,
    "alpaca_cleaned": alpaca_cleaned,
    "lima": lima,
    "openorca_60k": openorca,
    "gsm8k": gsm8k,
    "ifeval": ifeval,
}


def main():
    survivors = []
    summary = {}
    for name, fn in SOURCES.items():
        rows = []
        try:
            for key, text, cat in fn():
                rows.append((key, text, cat))
        except Exception as e:  # gated / network
            summary[name] = {"error": repr(e)[:200]}
            print(name, "ERROR", repr(e)[:200], flush=True)
            continue
        n = len(rows)
        lens = [len(words(t)) for _, t, _ in rows]
        stats = {
            "n": n,
            "words_median": statistics.median(lens) if lens else None,
            "words_q1": sorted(lens)[n // 4] if n else None,
            "words_q3": sorted(lens)[3 * n // 4] if n else None,
        }
        tight = loose = tight_nolex = tight_clean = loose_clean = 0
        cats_tight_clean = Counter()
        for (key, text, cat), L in zip(rows, lens):
            if not is_english(text):
                continue
            lex, adj = flags(text)
            in_tight = BAND_TIGHT[0] <= L <= BAND_TIGHT[1]
            in_loose = BAND_LOOSE[0] <= L <= BAND_LOOSE[1]
            tight += in_tight
            loose += in_loose
            if in_tight and not lex:
                tight_nolex += 1
            if in_tight and not lex and not adj:
                tight_clean += 1
                cats_tight_clean[cat] += 1
            if in_loose and not lex and not adj:
                loose_clean += 1
                survivors.append(
                    {"id": key, "source": name, "category": cat, "words": L,
                     "in_tight_band": in_tight, "text": text}
                )
        stats.update(
            {
                "in_band_106_155": tight,
                "in_band_90_170": loose,
                "tight_no_lexicon": tight_nolex,
                "tight_no_lexicon_no_adjacent": tight_clean,
                "loose_no_lexicon_no_adjacent": loose_clean,
                "tight_clean_by_category": dict(cats_tight_clean.most_common()),
            }
        )
        summary[name] = stats
        print(name, json.dumps(stats), flush=True)
    OUT.write_text(json.dumps({"summary": summary, "bands": {"tight": BAND_TIGHT, "loose": BAND_LOOSE},
                               "survivors": survivors}, indent=1, ensure_ascii=False))
    print("wrote", OUT, len(survivors), "survivors")


if __name__ == "__main__":
    main()
