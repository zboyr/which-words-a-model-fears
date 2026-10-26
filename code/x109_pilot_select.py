"""Benign insertion pilot, selection (local).

`--prepare`: draws the 20 prompts (10 Dolly passage+task re-rendered as context then
instruction, 10 No Robots instruction-only prose, in the 106-155 word band, seeded random
draw) and the word classes (a) x103 cross-model projection shift, (b) x85 Qwen3.5-4B
behaviour, (c) frequency in the harmful source pool. Writes prompts.json and words_pre.json.

`--finalize`: reads projections.json (the 20 x 99 matrix), adds class (d), forms the union,
assigns neutral twins, and writes words.json.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x109_pilot_common import (  # noqa: E402
    N_PER_SHAPE,
    PRE_WORDS,
    PROJECTIONS,
    PROMPTS,
    ROOT,
    SEED,
    SHAPE_A_CATEGORIES,
    SHAPE_B_CATEGORIES,
    TWIN_CANDIDATES,
    WORDS,
    atomic_json,
    read_json,
    sha256_text,
)
from x83_common import lexicon_words  # noqa: E402
from x85_common import WORD_PAIRS  # noqa: E402

CANDIDATES = ROOT / "data" / "pools" / "x109_clean_benign_candidates.json"
HARM_POOL = ROOT / "data" / "pools" / "cp_15k_clustered_v2.json"
X85_RESULTS = ROOT / "data" / "results" / "x85_crossed_keyword_levels.json"
X103_TAGS = ("qwen", "qwen08b", "qwen2b", "qwen9b", "qwen27b", "llama", "gemma", "phi")
TOP_K = 5
WORD = re.compile(r"[A-Za-z']+")


def n_words(text: str) -> int:
    return len(WORD.findall(text))


LEX_FORMS = {w + suffix for w, _ in lexicon_words() for suffix in ("", "s", "es", "d", "ed", "ing", "er", "ers")}


def no_inflected_lexicon(text: str) -> bool:
    """No x83 lexicon word in any simple inflection (the candidate filter matched bare forms only)."""
    return not any(t.lower() in LEX_FORMS for t in re.findall(r"[A-Za-z]+", text))


def select_prompts() -> dict:
    from datasets import load_dataset

    survivors = [s for s in read_json(CANDIDATES)["survivors"] if no_inflected_lexicon(s["text"])]
    pool_a = sorted(
        (s for s in survivors if s["source"] == "dolly" and s["category"] in SHAPE_A_CATEGORIES and s["in_tight_band"]),
        key=lambda s: int(s["id"].split(":")[1]),
    )
    pool_b = sorted(
        (s for s in survivors if s["source"] == "no_robots" and s["category"] in SHAPE_B_CATEGORIES and s["in_tight_band"]),
        key=lambda s: int(s["id"].split(":")[1]),
    )
    rng = np.random.default_rng(SEED)
    pick_a = [pool_a[k] for k in sorted(rng.choice(len(pool_a), N_PER_SHAPE, replace=False))]
    pick_b = [pool_b[k] for k in sorted(rng.choice(len(pool_b), N_PER_SHAPE, replace=False))]
    dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
    rows = []
    for s in pick_a:
        index = int(s["id"].split(":")[1])
        r = dolly[index]
        stored = r["instruction"].strip() + "\n\n" + r["context"].strip()
        if stored != s["text"]:
            raise SystemExit(f"FATAL: {s['id']} does not reproduce the candidate text")
        text = r["context"].strip() + "\n\n" + r["instruction"].strip()
        rows.append({"id": s["id"], "source": "databricks/databricks-dolly-15k", "row_index": index, "category": s["category"], "shape": "A", "render": "context + blank line + instruction", "text": text})
    for s in pick_b:
        index = int(s["id"].split(":")[1])
        rows.append({"id": s["id"], "source": "HuggingFaceH4/no_robots", "row_index": index, "category": s["category"], "shape": "B", "render": "prompt verbatim", "text": s["text"]})
    for r in rows:
        r["n_words"] = n_words(r["text"])
        r["text_sha256"] = sha256_text(r["text"])
    return {
        "pilot": "x109/pilot",
        "selection": {
            "candidates": str(CANDIDATES.relative_to(ROOT)),
            "seed": SEED,
            "shape_A": {"source": "dolly", "categories": list(SHAPE_A_CATEGORIES), "pool_size": len(pool_a), "rule": "in_tight_band (106-155 words), no x83 lexicon word, no harm-adjacent stem, no x83 lexicon word with a suffix s/es/d/ed/ing/er/ers; numpy default_rng(seed).choice without replacement over the id-sorted pool"},
            "shape_B": {"source": "no_robots", "categories": list(SHAPE_B_CATEGORIES), "pool_size": len(pool_b), "rule": "same rule, drawn after shape A from the same generator"},
        },
        "prompts_sha256": sha256_text("\n".join(f"{r['id']}\t{r['text_sha256']}" for r in rows)),
        "prompts": rows,
    }


def class_a() -> tuple[list[str], dict]:
    """Mean within-model z of the x103 direction-projection shift, averaged over the eight models."""
    per_model = {}
    for tag in X103_TAGS:
        rows = read_json(ROOT / "data" / "runs" / "x103" / tag / "x103_projections.json")["rows"]
        proj = np.asarray([r["proj"] for r in rows], dtype=float)
        z = (proj - proj.mean()) / proj.std()
        by_word: dict[str, list[float]] = {}
        for r, value in zip(rows, z):
            by_word.setdefault(r["word"], []).append(float(value))
        per_model[tag] = {w: float(np.mean(v)) for w, v in by_word.items()}
    words = sorted(per_model[X103_TAGS[0]])
    score = {w: float(np.mean([per_model[t][w] for t in X103_TAGS])) for w in words}
    ranked = sorted(words, key=lambda w: -score[w])
    n_cells = Counter(r["word"] for r in read_json(ROOT / "data" / "runs" / "x103" / "qwen" / "x103_projections.json")["rows"])
    return ranked[:TOP_K], {
        "rule": "x103 per-cell proj (sum over the two boundary tokens of the inserted-minus-original state on the model's own direction), z-scored within each model over all 390 cells (S, M and E positions), averaged per word within model, then averaged over the eight models",
        "n_words_covered": len(words),
        "ranking": [{"word": w, "mean_z": score[w], "n_cells_per_model": n_cells[w], "per_model": {t: per_model[t][w] for t in X103_TAGS}} for w in ranked],
    }


def class_b() -> tuple[list[str], dict]:
    rows = read_json(X85_RESULTS)["words"]
    ranked = sorted(rows, key=lambda r: -r["delta_H_minus_N"])
    return [r["harm"] for r in ranked[:TOP_K]], {
        "rule": "x85 Qwen3.5-4B behaviour, delta_H_minus_N (refusal2), top 5",
        "ranking": [{"word": r["harm"], "neutral": r["neutral"], "delta_H_minus_N": r["delta_H_minus_N"]} for r in ranked],
    }


def class_c() -> tuple[list[str], dict]:
    pool = read_json(HARM_POOL)
    lex = [w for w, _ in lexicon_words()]
    patterns = {w: re.compile(rf"\b{w}\b", re.IGNORECASE) for w in lex}
    counts = Counter()
    for row in pool:
        text = row["prompt"]
        for w, pattern in patterns.items():
            if pattern.search(text):
                counts[w] += 1
    ranked = sorted(lex, key=lambda w: (-counts[w], lex.index(w)))
    return ranked[:TOP_K], {
        "rule": "x83 lexicon words ranked by the number of prompts in data/pools/cp_15k_clustered_v2.json containing them (whole word, case-insensitive), top 5",
        "n_pool_prompts": len(pool),
        "ranking": [{"word": w, "n_prompts": counts[w]} for w in ranked],
    }


def prepare() -> None:
    prompts = select_prompts()
    atomic_json(PROMPTS, prompts)
    for r in prompts["prompts"]:
        print(f"  {r['shape']} {r['id']:16s} {r['category']:24s} {r['n_words']:4d} words")
    a, a_info = class_a()
    b, b_info = class_b()
    c, c_info = class_c()
    atomic_json(PRE_WORDS, {"pilot": "x109/pilot", "classes": {"a": a, "b": b, "c": c}, "a": a_info, "b": b_info, "c": c_info})
    print("a", a, "\nb", b, "\nc", c)


def finalize() -> None:
    pre = read_json(PRE_WORDS)
    proj = read_json(PROJECTIONS)
    prompts = read_json(PROMPTS)
    if proj["prompts_sha256"] != prompts["prompts_sha256"]:
        raise SystemExit("FATAL: projections belong to another prompt set")
    lex = [w for w, _ in lexicon_words()]
    mean_shift = {w: proj["word_summary"][w]["mean_shift"] for w in lex}
    d = sorted(lex, key=lambda w: -mean_shift[w])[:TOP_K]
    classes = {**pre["classes"], "d": d}
    union: list[str] = []
    for key in ("a", "b", "c", "d"):
        for w in classes[key]:
            if w not in union:
                union.append(w)
    x85_twin = {harm: neutral for _, harm, neutral in WORD_PAIRS}
    lexset = set(lex)
    used = set()
    projected = set(proj["word_summary"])
    records = []
    for w in union:
        if w in x85_twin:
            twin, rule = x85_twin[w], "x85 WORD_PAIRS neutral"
        else:
            twin = None
            for tolerance in (0, 1):
                for cand in TWIN_CANDIDATES:
                    if cand in used or cand in lexset or cand in x85_twin.values():
                        continue
                    if abs(len(cand) - len(w)) <= tolerance:
                        twin = cand
                        break
                if twin:
                    break
            rule = "first unused TWIN_CANDIDATES noun with the same character length (else +-1)"
        if twin is None or twin not in projected:
            raise SystemExit(f"FATAL: no projected twin for {w}")
        used.add(twin)
        records.append(
            {
                "word": w,
                "classes": [k for k in ("a", "b", "c", "d") if w in classes[k]],
                "twin": twin,
                "twin_rule": rule,
                "len_word": len(w),
                "len_twin": len(twin),
                "word_mean_shift": proj["word_summary"][w]["mean_shift"],
                "twin_mean_shift": proj["word_summary"][twin]["mean_shift"],
            }
        )
    out = {
        "pilot": "x109/pilot",
        "classes": classes,
        "class_rules": {
            "a": pre["a"]["rule"],
            "b": pre["b"]["rule"],
            "c": pre["c"]["rule"],
            "d": "mean over the 20 prompts of proj(E_w) - proj(O) on Qwen3.5-4B (x106 reader, sum of the two boundary tokens), all 99 lexicon words, top 5",
        },
        "union": records,
        "n_union": len(records),
        "class_d_ranking": [{"word": w, "mean_shift": mean_shift[w]} for w in sorted(lex, key=lambda w: -mean_shift[w])],
    }
    atomic_json(WORDS, out)
    for r in records:
        print(f"  {r['word']:14s} {','.join(r['classes']):8s} -> {r['twin']:14s} ({r['twin_rule'][:12]}) shift {r['word_mean_shift']:+.2f} / {r['twin_mean_shift']:+.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.finalize:
        finalize()
    else:
        raise SystemExit("--prepare or --finalize")


if __name__ == "__main__":
    main()
