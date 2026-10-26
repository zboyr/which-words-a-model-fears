"""Vocabulary scan, selection (local).

`--prompts`: set A = the benign insertion pilot's 20 prompts verbatim; set B = 20 new prompts
drawn by the pilot's rules (10 Dolly passage+task from closed_qa / information_extraction /
summarization re-rendered as context then instruction, 10 No Robots instruction-only prose,
tight band, suffix-form lexicon filter) with seed 20261009 over the same pools minus the
pilot's 20 ids. Writes prompts.json.

`--words`: reads scan.json + scan_shift.npz; ranks candidates by mean set-A shift; picks the
top 10 distinct words by the fixed rules; assigns neutral twins; writes words.json.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x109_pilot_select import n_words, no_inflected_lexicon  # noqa: E402
from x109_pilot_common import N_PER_SHAPE, SHAPE_A_CATEGORIES, SHAPE_B_CATEGORIES  # noqa: E402
from x109_scan_common import (  # noqa: E402
    N_TOP,
    PILOT,
    PROMPTS,
    ROOT,
    SCAN_JSON,
    SCAN_NPZ,
    SEED,
    TWIN_CANDIDATES,
    WORDFREQ,
    WORDS,
    X106_HARD1K_SD,
    atomic_json,
    read_json,
    sha256_text,
)
from x83_common import lexicon_words  # noqa: E402
from x85_common import WORD_PAIRS  # noqa: E402

CANDIDATES = ROOT / "data" / "pools" / "x109_clean_benign_candidates.json"
TOP_REPORT = 50

# Proper nouns and spelling variants among the top of the set-A ranking, judged by the agent
# after reading the ranking (each also listed in the drop log with this rule).
PROPER_NOUNS_JUDGED: set[str] = {"tiananmen", "hitler", "pornhub"}
SPELLING_VARIANTS_JUDGED: dict[str, str] = {}


def build_prompts() -> None:
    from datasets import load_dataset

    pilot = read_json(PILOT / "prompts.json")
    pilot_ids = {p["id"] for p in pilot["prompts"]}
    survivors = [s for s in read_json(CANDIDATES)["survivors"] if no_inflected_lexicon(s["text"]) and s["id"] not in pilot_ids]
    pool_a = sorted((s for s in survivors if s["source"] == "dolly" and s["category"] in SHAPE_A_CATEGORIES and s["in_tight_band"]), key=lambda s: int(s["id"].split(":")[1]))
    pool_b = sorted((s for s in survivors if s["source"] == "no_robots" and s["category"] in SHAPE_B_CATEGORIES and s["in_tight_band"]), key=lambda s: int(s["id"].split(":")[1]))
    rng = np.random.default_rng(SEED)
    pick_a = [pool_a[k] for k in sorted(rng.choice(len(pool_a), N_PER_SHAPE, replace=False))]
    pick_b = [pool_b[k] for k in sorted(rng.choice(len(pool_b), N_PER_SHAPE, replace=False))]
    dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
    rows = [{**p, "set": "A"} for p in pilot["prompts"]]
    new = []
    for s in pick_a:
        index = int(s["id"].split(":")[1])
        r = dolly[index]
        if r["instruction"].strip() + "\n\n" + r["context"].strip() != s["text"]:
            raise SystemExit(f"FATAL: {s['id']} does not reproduce the candidate text")
        text = r["context"].strip() + "\n\n" + r["instruction"].strip()
        new.append({"id": s["id"], "source": "databricks/databricks-dolly-15k", "row_index": index, "category": s["category"], "shape": "A", "render": "context + blank line + instruction", "text": text})
    for s in pick_b:
        index = int(s["id"].split(":")[1])
        new.append({"id": s["id"], "source": "HuggingFaceH4/no_robots", "row_index": index, "category": s["category"], "shape": "B", "render": "prompt verbatim", "text": s["text"]})
    for r in new:
        r["n_words"] = n_words(r["text"])
        r["text_sha256"] = sha256_text(r["text"])
        r["set"] = "B"
    rows += new
    out = {
        "run": "x109",
        "sets": {
            "A": {"rule": "the benign insertion pilot's 20 prompts verbatim (data/runs/x109/pilot/prompts.json)", "pilot_prompts_sha256": pilot["prompts_sha256"]},
            "B": {
                "candidates": str(CANDIDATES.relative_to(ROOT)),
                "seed": SEED,
                "excluded": sorted(pilot_ids),
                "shape_A": {"source": "dolly", "categories": list(SHAPE_A_CATEGORIES), "pool_size": len(pool_a), "rule": "in_tight_band (106-155 words), no x83 lexicon word, no harm-adjacent stem, no x83 lexicon word with a suffix s/es/d/ed/ing/er/ers, not a set-A id; numpy default_rng(seed).choice without replacement over the id-sorted pool"},
                "shape_B": {"source": "no_robots", "categories": list(SHAPE_B_CATEGORIES), "pool_size": len(pool_b), "rule": "same rule, drawn after shape A from the same generator"},
            },
        },
        "prompts_sha256": sha256_text("\n".join(f"{r['id']}\t{r['text_sha256']}" for r in rows)),
        "prompts": rows,
    }
    atomic_json(PROMPTS, out)
    for r in new:
        print(f"  B {r['shape']} {r['id']:16s} {r['category']:24s} {r['n_words']:4d} words")


def stem(word: str) -> str:
    from nltk.stem import PorterStemmer

    return PorterStemmer().stem(word)


SUFFIXES = ("s", "es", "ed", "d", "ing", "er", "ers", "ic", "ia", "ism", "ist", "ists", "y", "ly", "ity", "ness", "ation", "ations", "al", "ally", "ian", "ians", "ous")


def common_prefix(a: str, b: str) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def share_stem(a: str, b: str) -> bool:
    """Same Porter stem, one word = the other plus a listed suffix, or a common prefix of at least
    max(5, shorter length - 3) characters; compared after the British 'ae' -> 'e' respelling."""
    a, b = a.replace("ae", "e"), b.replace("ae", "e")
    if stem(a) == stem(b):
        return True
    if common_prefix(a, b) >= max(5, min(len(a), len(b)) - 3):
        return True
    short, long_ = sorted((a, b), key=len)
    for s in SUFFIXES:
        if long_ == short + s:
            return True
        if short.endswith("e") and long_ == short[:-1] + s:
            return True
        if short.endswith("y") and long_ == short[:-1] + "i" + s:
            return True
    return False


def zipf_of(word: str) -> float | None:
    try:
        import wordfreq

        return float(wordfreq.zipf_frequency(word, "en"))
    except ImportError:
        return None


def build_words() -> None:
    scan = read_json(SCAN_JSON)
    prompts = read_json(PROMPTS)
    if scan["prompts_sha256"] != prompts["prompts_sha256"]:
        raise SystemExit("FATAL: scan belongs to another prompt set")
    z = np.load(SCAN_NPZ)
    shift = z["shift"].astype(np.float64).sum(axis=2)  # [P, N]
    cands = scan["candidates"]["words"]
    n_tok = scan["candidates"]["n_tokens"]
    sets = np.asarray([p["set"] for p in prompts["prompts"]])
    mA, mB, mAll = shift[sets == "A"].mean(0), shift[sets == "B"].mean(0), shift.mean(0)
    orderA = np.argsort(-mA, kind="stable")
    rankA = np.empty(len(cands), int)
    rankA[orderA] = np.arange(1, len(cands) + 1)
    lex = [w for w, _ in lexicon_words()]
    lexset = set(lex)
    wf = set(read_json(WORDFREQ)["words"])
    index = {w: i for i, w in enumerate(cands)}

    def row(i: int) -> dict:
        w = cands[i]
        return {"rank_A": int(rankA[i]), "word": w, "mean_shift_A": float(mA[i]), "sd_units_A": float(mA[i] / X106_HARD1K_SD), "mean_shift_B": float(mB[i]), "mean_shift_all": float(mAll[i]), "n_tokens": int(n_tok[i]), "in_x83_lexicon": w in lexset, "in_wordfreq_top50k": w in wf, "zipf": zipf_of(w)}

    top50 = [row(i) for i in orderA[:TOP_REPORT]]
    chosen: list[str] = []
    drops = []
    for i in orderA:
        w = cands[i]
        if len(chosen) == N_TOP:
            break
        if w not in wf and w not in lexset:
            drops.append({**row(i), "rule": "non-word: not in the wordfreq top-50k list and not in the x83 lexicon"})
            continue
        if w in PROPER_NOUNS_JUDGED:
            drops.append({**row(i), "rule": "proper noun (manual judgment on the ranking)"})
            continue
        if w in SPELLING_VARIANTS_JUDGED and SPELLING_VARIANTS_JUDGED[w] in chosen:
            drops.append({**row(i), "rule": f"spelling variant of higher-ranked chosen word '{SPELLING_VARIANTS_JUDGED[w]}'"})
            continue
        hit = next((c for c in chosen if share_stem(c, w)), None)
        if hit:
            drops.append({**row(i), "rule": f"inflection/derivative/respelling sharing a stem with higher-ranked chosen word '{hit}'"})
            continue
        chosen.append(w)

    # Neutral twins.
    x85_twin = {harm: neutral for _, harm, neutral in WORD_PAIRS}
    x85_neutrals = set(x85_twin.values())
    nouns = [n for n in TWIN_CANDIDATES if n in index and n not in lexset and n not in x85_neutrals]
    noun_shift = {n: float(mA[index[n]]) for n in nouns}
    median = float(np.median(list(noun_shift.values())))
    used: set[str] = set()
    records = []
    for w in chosen:
        i = index[w]
        if w in x85_twin:
            twin, rule = x85_twin[w], "x85 WORD_PAIRS neutral"
        else:
            twin = None
            for tol in (0, 1):
                for n in nouns:
                    if n in used or abs(len(n) - len(w)) > tol or abs(noun_shift[n] - median) > 1.0:
                        continue
                    twin = n
                    break
                if twin:
                    break
            rule = "first unused fixed-list noun with the same character length (else +-1) and mean set-A shift within +-1.0 of the median fixed-list noun shift"
        if twin is None or twin not in index:
            raise SystemExit(f"FATAL: no twin for {w}")
        used.add(twin)
        records.append({**row(i), "twin": twin, "twin_rule": rule, "twin_mean_shift_A": float(mA[index[twin]]), "twin_mean_shift_B": float(mB[index[twin]]), "twin_mean_shift_all": float(mAll[index[twin]]), "len_word": len(w), "len_twin": len(twin)})

    def topk(m: np.ndarray, k=10) -> list[dict]:
        return [row(i) for i in np.argsort(-m, kind="stable")[:k]]

    lex_rank = sorted(({"word": w, "rank_A": int(rankA[index[w]]), "mean_shift_A": float(mA[index[w]])} for w in lex if w in index), key=lambda r: r["rank_A"])
    out = {
        "run": "x109",
        "selection_rule": "rank all candidates by mean over the 20 set-A prompts of the shift summed over the two boundary tokens; walk down the ranking and keep the first 10 words, dropping (1) non-words: not in the wordfreq top-50k and not in the x83 lexicon, (2) proper nouns, (3) spelling variants of a chosen word, (4) inflections/derivatives/respellings sharing a stem with a higher-ranked chosen word (after 'ae' -> 'e': same Porter stem, one word = the other + a listed suffix, or a common prefix of at least max(5, shorter length - 3) characters)",
        "stem_suffixes": list(SUFFIXES),
        "chosen": records,
        "drop_log": drops,
        "top50_set_A": top50,
        "top10_set_B": topk(mB),
        "top10_all": topk(mAll),
        "named_ranks": {w: row(index[w]) for w in ("pedophile", "racist") if w in index},
        "lexicon_ranks_A": lex_rank,
        "twin_noun_median_shift_A": median,
        "twin_noun_shifts_A": noun_shift,
        "n_candidates": len(cands),
    }
    atomic_json(WORDS, out)
    print("top 50 (set A):")
    for r in top50:
        print(f"  {r['rank_A']:3d} {r['word']:18s} {r['mean_shift_A']:+7.2f} {r['sd_units_A']:+5.2f}sd tok={r['n_tokens']} lex={int(r['in_x83_lexicon'])} wf={int(r['in_wordfreq_top50k'])} zipf={r['zipf']}")
    print("drops:")
    for d in drops:
        print(f"  {d['rank_A']:4d} {d['word']:18s} {d['rule']}")
    print("chosen:")
    for r in records:
        print(f"  {r['rank_A']:4d} {r['word']:16s} {r['mean_shift_A']:+.2f} -> {r['twin']} ({r['twin_mean_shift_A']:+.2f})")
    print("named", out["named_ranks"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", action="store_true")
    parser.add_argument("--words", action="store_true")
    args = parser.parse_args()
    if args.prompts:
        build_prompts()
    elif args.words:
        build_words()
    else:
        raise SystemExit("--prompts or --words")


if __name__ == "__main__":
    main()
