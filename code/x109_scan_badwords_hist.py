"""x109 follow-up: where the CMU "bad words" list (Luis von Ahn, cs.cmu.edu/~biglou/resources/bad-words.txt)
sits in the x109 vocabulary-scan distribution of mean boundary shift on Qwen3.5-4B.

Reads data/runs/x109/scan_shift.npz (40 prompts x 71,583 candidates x 2 boundary tokens) and
data/runs/x109/scan.json (candidate words), matches the CMU list against the candidate words after
lowercasing and stripping, and draws the x109 sensitivity_distribution figure with the CMU words
overlaid instead of (and next to) the x83 lexicon.

Usage: python3 code/x109_scan_badwords_hist.py <path to bad-words.txt>
Writes data/runs/x109/badwords_distribution.png and badwords_distribution.json.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x109_pilot_common import X106_HARD1K_SD  # noqa: E402
from x83_common import lexicon_words  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data" / "runs" / "x109"
OUT_PNG = RUN / "badwords_distribution.png"
OUT_JSON = RUN / "badwords_distribution.json"

CANDIDATE = re.compile(r"^[a-z]{3,}$")


def main() -> None:
    src = Path(sys.argv[1])
    raw = [ln.strip().lower() for ln in src.read_text(encoding="utf-8").splitlines()]
    raw = [w for w in raw if w]
    cmu = sorted(set(raw))
    cmu_well_formed = [w for w in cmu if CANDIDATE.match(w)]
    cmu_malformed = [w for w in cmu if not CANDIDATE.match(w)]

    scan = json.load(open(RUN / "scan.json"))
    words = scan["candidates"]["words"]
    n_tok = np.asarray(scan["candidates"]["n_tokens"])
    shift = np.load(RUN / "scan_shift.npz")["shift"]  # [prompt, cand, token]
    mean_sd = shift.sum(axis=2).mean(axis=0) / X106_HARD1K_SD  # mean over 40 prompts, sd units
    idx = {w: i for i, w in enumerate(words)}
    order = np.argsort(-mean_sd)
    rank = np.empty(len(words), dtype=int)
    rank[order] = np.arange(1, len(words) + 1)
    pct = 100.0 * (len(words) - rank) / (len(words) - 1)  # percentile of the word's shift, 100 = top

    matched = [w for w in cmu_well_formed if w in idx]
    unmatched = [w for w in cmu_well_formed if w not in idx]
    lex = [w for w, _ in lexicon_words()]
    lex_in = [w for w in lex if w in idx]
    cmu_ix = np.array([idx[w] for w in matched])
    lex_ix = np.array([idx[w] for w in lex_in])
    cmu_sd = mean_sd[cmu_ix]
    lex_sd = mean_sd[lex_ix]
    overlap = sorted(set(matched) & set(lex_in))

    def summary(v: np.ndarray, ix: np.ndarray) -> dict:
        q = np.percentile(v, [1, 10, 25, 50, 75, 90, 99])
        return {
            "n": int(len(v)),
            "mean": float(v.mean()),
            "sd": float(v.std()),
            "p1": float(q[0]), "p10": float(q[1]), "p25": float(q[2]), "median": float(q[3]),
            "p75": float(q[4]), "p90": float(q[5]), "p99": float(q[6]),
            "min": float(v.min()), "max": float(v.max()),
            "frac_below_zero": float((v < 0).mean()),
            "n_above_1sd": int((v > 1).sum()),
            "n_above_2sd": int((v > 2).sum()),
            "n_in_top_1pct": int((pct[ix] >= 99).sum()),
            "n_in_top_10pct": int((pct[ix] >= 90).sum()),
            "median_percentile": float(np.median(pct[ix])),
        }

    all_sd = mean_sd
    all_summary = {
        "n": int(len(all_sd)), "mean": float(all_sd.mean()), "median": float(np.median(all_sd)),
        "p90": float(np.percentile(all_sd, 90)), "p99": float(np.percentile(all_sd, 99)),
    }
    cmu_summary = summary(cmu_sd, cmu_ix)
    lex_summary = summary(lex_sd, lex_ix)

    # Mann-Whitney U, CMU words against the rest of the vocabulary (one-sided, CMU higher)
    from scipy.stats import mannwhitneyu
    rest = np.ones(len(words), dtype=bool)
    rest[cmu_ix] = False
    u = mannwhitneyu(cmu_sd, mean_sd[rest], alternative="greater")
    auc = float(u.statistic / (len(cmu_sd) * rest.sum()))

    def table(ix: np.ndarray) -> list[dict]:
        return [
            {"word": words[i], "mean_shift_sd": float(mean_sd[i]), "rank": int(rank[i]),
             "percentile": float(pct[i]), "n_tokens": int(n_tok[i]), "in_x83_lexicon": words[i] in set(lex_in)}
            for i in ix
        ]

    cmu_rows = sorted(table(cmu_ix), key=lambda r: -r["mean_shift_sd"])
    # how many of the scan's global top-k are CMU words
    cmu_set = set(matched)
    top_hits = {k: int(sum(words[i] in cmu_set for i in order[:k])) for k in (10, 50, 100, 500, 1000)}

    # ---- figure, in the style of sensitivity_distribution.png ----
    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 5.5), dpi=100, gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    bins = np.linspace(-1.2, 2.6, 77)
    ax.hist(all_sd, bins=bins, color="#9aa7a8", label=f"all {len(words):,} words")
    ax.hist(cmu_sd, bins=bins, color="#d62728", alpha=0.95, label=f"CMU bad-words list ({len(cmu_sd):,} matched of {len(cmu):,})")
    ax.hist(lex_sd, bins=bins, histtype="step", color="#1f3b8a", linewidth=1.6, label=f"x83 lexicon ({len(lex_sd)})")
    ax.set_yscale("log")
    ax.set_xlabel("mean boundary shift over 40 benign prompts (sd units)")
    ax.set_ylabel("words (log)")
    ax.legend(loc="upper right", frameon=False)
    # annotate the extremes of the CMU list and a few familiar words
    annot = []
    annot += [r["word"] for r in cmu_rows[:3]]
    annot += [r["word"] for r in cmu_rows[-2:]]
    for w in ("fuck", "shit", "abortion", "gun", "bomb", "hack", "suicide"):
        if w in cmu_set and w not in annot:
            annot.append(w)
    # place the labels left to right, nudging apart any two closer than 0.07 sd, at alternating heights
    xs = sorted((float(mean_sd[idx[w]]), w) for w in annot)
    placed = []
    for x, w in xs:
        xl = x
        if placed and xl - placed[-1][0] < 0.07:
            xl = placed[-1][0] + 0.07
        placed.append((xl, x, w))
    for k, (xl, x, w) in enumerate(placed):
        y = 1.3 if k % 2 == 0 else 2.6
        ax.annotate(w, xy=(x, 1.0), xytext=(xl, y), rotation=90, ha="center", va="bottom", fontsize=8, color="#2b0000",
                    arrowprops={"arrowstyle": "-", "color": "#2b0000", "linewidth": 0.5} if abs(xl - x) > 1e-6 else None)

    ax = axes[1]
    ranks_all = np.arange(1, len(words) + 1)
    ax.plot(ranks_all, mean_sd[order], color="#4b4f9e", linewidth=2, label="all words")
    ax.scatter(rank[cmu_ix], cmu_sd, s=9, color="#d62728", alpha=0.6, zorder=3, label="CMU bad-words list")
    ax.set_xscale("log")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("rank (log)")
    ax.set_ylabel("mean shift (sd units)")
    ax.grid(axis="x", alpha=0.4)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)

    # ---- second figure: percentile-of-vocabulary histogram of the CMU words ----
    fig2, ax2 = plt.subplots(figsize=(7.5, 4.2), dpi=100)
    ax2.hist(pct[cmu_ix], bins=np.linspace(0, 100, 21), color="#d62728", alpha=0.9, label="CMU bad-words list")
    ax2.hist(pct[lex_ix], bins=np.linspace(0, 100, 21), histtype="step", color="#1f3b8a", linewidth=1.6, label="x83 lexicon")
    ax2.axvline(50, color="black", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("percentile of the mean shift in the 71,583-word scan (100 = largest)")
    ax2.set_ylabel("words")
    ax2.legend(loc="upper left", frameon=False)
    fig2.tight_layout()
    fig2.savefig(RUN / "badwords_percentile.png", dpi=150)

    out = {
        "source": "https://www.cs.cmu.edu/~biglou/resources/bad-words.txt",
        "n_lines": len(raw), "n_unique": len(cmu),
        "n_well_formed": len(cmu_well_formed), "n_malformed": len(cmu_malformed),
        "malformed_examples": cmu_malformed[:30],
        "n_matched": len(matched), "n_unmatched_well_formed": len(unmatched),
        "unmatched_examples": unmatched[:40],
        "overlap_with_x83_lexicon": overlap,
        "unit": "mean over 40 prompts of the boundary shift summed over the two boundary tokens, divided by the x106 Hard-1K sd (8.645)",
        "all_words": all_summary,
        "cmu": cmu_summary,
        "x83_lexicon": lex_summary,
        "mannwhitney_cmu_vs_rest": {"U": float(u.statistic), "p_one_sided": float(u.pvalue), "auc": auc},
        "cmu_in_global_top_k": top_hits,
        "cmu_top30": cmu_rows[:30],
        "cmu_bottom30": cmu_rows[-30:],
        "cmu_all": cmu_rows,
        "figures": [str(OUT_PNG.relative_to(ROOT)), str((RUN / "badwords_percentile.png").relative_to(ROOT))],
    }
    OUT_JSON.write_text(json.dumps(out, indent=1, ensure_ascii=False))

    print(f"CMU list: {len(raw)} lines, {len(cmu)} unique, {len(cmu_well_formed)} well-formed [a-z]{{3,}}, {len(matched)} matched in the scan vocabulary")
    print(f"  malformed (hyphen/digit/space/apostrophe/short): {len(cmu_malformed)}  e.g. {cmu_malformed[:12]}")
    print(f"  well-formed but not in the scan vocabulary: {len(unmatched)}  e.g. {unmatched[:20]}")
    print(f"  overlap with x83 lexicon: {len(overlap)} {overlap}")
    print("all words:", {k: round(v, 3) if isinstance(v, float) else v for k, v in all_summary.items()})
    print("CMU      :", {k: round(v, 3) if isinstance(v, float) else v for k, v in cmu_summary.items()})
    print("x83 lex  :", {k: round(v, 3) if isinstance(v, float) else v for k, v in lex_summary.items()})
    print(f"Mann-Whitney CMU > rest: AUC={auc:.3f} p={u.pvalue:.2e}")
    print("CMU words among global top-k:", top_hits)
    print("CMU top 30:")
    for r in cmu_rows[:30]:
        print(f"  {r['rank']:6d} {r['word']:16s} {r['mean_shift_sd']:+.2f}sd  pct={r['percentile']:5.1f} tok={r['n_tokens']}{'  [x83]' if r['in_x83_lexicon'] else ''}")
    print("CMU bottom 15:")
    for r in cmu_rows[-15:]:
        print(f"  {r['rank']:6d} {r['word']:16s} {r['mean_shift_sd']:+.2f}sd  pct={r['percentile']:5.1f} tok={r['n_tokens']}")


if __name__ == "__main__":
    main()
