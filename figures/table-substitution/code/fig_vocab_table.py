"""Figures for the vocabulary-table sections of the v1 draft.

Writes fig/<name>/<name>.pdf for:
  vocab-distribution.pdf  the Qwen3.5-4B table: mean boundary shift of 71,583 appended words
  table-dose-response.pdf held-out dose response per model + own-vs-Qwen table rho
  table-substitution.pdf  in-place substitution chosen by each model's own table
Reads only result files under the project's data/ tree (provenance: x109, x111, x112, x113 runs).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

DRAFT = Path(__file__).resolve().parents[3]
ROOT = DRAFT.parent
RUNS = ROOT / "data" / "runs"
FIG = DRAFT / "fig"

BLUE, ORANGE, INK, MUTED = "#0072B2", "#E69F00", "#333333", "#888888"
GREEN, RED = "#009E73", "#D55E00"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.6, "axes.edgecolor": INK,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "grid.color": "#DDDDDD", "grid.linewidth": 0.5, "pdf.fonttype": 42,
})
SD_QWEN4B = 8.645  # Qwen3.5-4B Hard-1K projection sd (unit of the table)
LABELS = {"qwen": "Qwen3.5-4B", "qwen08b": "Qwen3.5-0.8B", "qwen2b": "Qwen3.5-2B",
          "qwen9b": "Qwen3.5-9B", "qwen27b": "Qwen3.5-27B", "llama": "Llama-3.2-3B",
          "gemma": "Gemma-4-E2B", "phi": "Phi-4-mini"}
ORDER = ["qwen08b", "qwen2b", "qwen", "qwen9b", "qwen27b", "llama", "gemma", "phi"]
QWEN_SHADES = {"qwen08b": "#9ECAE1", "qwen2b": "#6BAED6", "qwen": "#3182BD",
               "qwen9b": "#08519C", "qwen27b": "#08306B"}
OTHER = {"llama": ORANGE, "gemma": GREEN, "phi": RED}


def vocab_distribution() -> None:
    scan = json.load(open(RUNS / "x109" / "scan.json"))
    words = scan["candidates"]["words"]
    shift = np.load(RUNS / "x109" / "scan_shift.npz")["shift"]
    mean_sd = shift.sum(axis=2).mean(axis=0) / SD_QWEN4B
    idx = {w: i for i, w in enumerate(words)}
    bad = json.load(open(RUNS / "x109" / "badwords_distribution.json"))
    bad_ix = np.array([idx[r["word"]] for r in bad["cmu_all"] if r["word"] in idx])
    lex_ix = np.array([idx[r["word"]] for r in bad["cmu_all"] if r.get("in_x83_lexicon")] or [], dtype=int)
    import sys
    sys.path.insert(0, str(ROOT / "code"))
    from x83_common import lexicon_words  # the 99-word insertion lexicon
    lex_ix = np.array([idx[w] for w, _ in lexicon_words() if w in idx])

    fig, ax = plt.subplots(figsize=(3.15, 2.05))
    bins = np.linspace(-1.2, 2.6, 77)
    ax.hist(mean_sd, bins=bins, color="#C9D3D6", label=f"all {len(words):,} words")
    ax.hist(mean_sd[bad_ix], bins=bins, color=ORANGE, alpha=0.9,
            label=f"public profanity list ({len(bad_ix)})")
    ax.hist(mean_sd[lex_ix], bins=bins, histtype="step", color=BLUE, lw=1.1,
            label=f"insertion lexicon ({len(lex_ix)})")
    ax.set_yscale("log")
    ax.set_ylim(0.6, 3e5)
    ax.set_xlabel("mean boundary shift of the appended word (sd)")
    ax.set_ylabel("words")
    ax.legend(loc="upper right", frameon=False, handlelength=1.2, borderaxespad=0.1, fontsize=6)
    marks = [("poetic", 0.0, 2.2), ("whimsical", 0.0, 2.2), ("hack", -0.13, 1500), ("gun", 0.0, 1500),
             ("bomb", 0.13, 1500), ("racist", 0.0, 2.2), ("pedophilia", 0.0, 2.2)]
    for w, dx, y in marks:
        x = float(mean_sd[idx[w]])
        ax.annotate(w, xy=(x, 0.9 if y < 100 else 300), xytext=(x + dx, y), rotation=90, ha="center", va="bottom",
                    fontsize=6, color=INK, arrowprops={"arrowstyle": "-", "lw": 0.4, "color": MUTED})
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "vocab-distribution" / "vocab-distribution.pdf")
    plt.close(fig)


def table_dose_response() -> None:
    x111 = json.load(open(RUNS / "x111" / "results.json"))
    x112 = json.load(open(RUNS / "x112" / "results.json"))
    curves = {"qwen": [b["delta_EO"] for b in x111["dose_response"]["pooled"]["bins"]]}
    bins = [b["bin"] for b in x111["dose_response"]["pooled"]["bins"]]
    rho = {}
    for m in x112["models"]:
        t = m["model_tag"]
        curves[t] = [b["delta_EO"] for b in m["dose_response"]["pooled"]["bins"]]
        rp = m["rho"]["pooled"]
        rho[t] = (rp["table_own"]["all40"]["rho"], rp["table_qwen_BP"]["all40"]["rho"])
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.25), gridspec_kw={"width_ratios": [1.55, 1]})
    ax = axes[0]
    ax.axhline(0, color=MUTED, lw=0.6)
    xs = np.arange(len(bins))
    for t in ORDER:
        c = QWEN_SHADES.get(t) or OTHER[t]
        ls = "-" if t in QWEN_SHADES else "--"
        ax.plot(xs, curves[t], color=c, ls=ls, lw=1.0, marker="o", ms=2.4, label=LABELS[t])
    ax.set_xticks(xs)
    ax.set_xticklabels(bins, rotation=0, fontsize=6.3)
    ax.set_xlabel("percentile bin of the appended word in the model's own table")
    ax.set_ylabel(r"refusal change $r(\mathrm{E})-r(\mathrm{O})$")
    ax.grid(axis="y")
    ax.set_ylim(-0.38, 0.78)
    ax.legend(ncol=2, loc="upper left", frameon=False, handlelength=1.8, columnspacing=0.8,
              labelspacing=0.2, borderaxespad=0.1)
    ax.set_title("held-out prompts, three words per bin", pad=14)
    ax = axes[1]
    tr = [t for t in ORDER if t in rho]
    ys = np.arange(len(tr))[::-1]
    for y, t in zip(ys, tr):
        own, qw = rho[t]
        ax.plot([qw, own], [y, y], color=MUTED, lw=0.8, zorder=1)
        ax.scatter([own], [y], color=BLUE, s=14, zorder=3)
        ax.scatter([qw], [y], facecolor="white", edgecolor=INK, s=14, zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([LABELS[t] for t in tr])
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel(r"Spearman $\rho$ with the refusal change")
    ax.grid(axis="x")
    ax.legend(handles=[Line2D([], [], color=BLUE, marker="o", lw=0, ms=4, label="own table"),
                       Line2D([], [], color=INK, marker="o", markerfacecolor="white", lw=0, ms=4,
                              label="Qwen3.5-4B table")],
              loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, handlelength=1.0, borderaxespad=0.1, columnspacing=0.6)
    ax.set_title("which table ranks the words", pad=14)
    fig.tight_layout(pad=0.3, w_pad=1.0)
    fig.savefig(FIG / "table-dose-response" / "table-dose-response.pdf")
    plt.close(fig)


def table_substitution() -> None:
    x113 = json.load(open(RUNS / "x113" / "results.json"))["models"]
    order = [t for t in ORDER if t in x113]
    xs = np.arange(len(order))

    def est(t, key):
        e = x113[t]["populations"]["hard1k"]["estimands"][key]
        v = e["estimate"]
        return v, v - e["ci95"][0], e["ci95"][1] - v

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.2), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    ax.axhline(0, color=MUTED, lw=0.6)
    for key, shift, color, marker, filled, lab in (
            ("delta_S", -0.22, BLUE, "o", True, "most-feared word to most courageous synonym"),
            ("delta_C", 0.0, MUTED, "s", False, "control: inert word to synonym"),
            ("x106_delta_S", 0.22, GREEN, "^", False, "word found by the deletion pass")):
        v = [est(t, key) for t in order]
        ax.errorbar(xs + shift, [a for a, _, _ in v], yerr=[[b for _, b, _ in v], [c for _, _, c in v]],
                    fmt=marker, color=color, ms=3.2, elinewidth=0.7, capsize=1.2,
                    markerfacecolor=color if filled else "white", markeredgewidth=0.8, label=lab)
    ax.set_ylabel(r"refusal removed, $r(\mathrm{O})-r(\cdot)$")
    ax.set_title("benign Hard-1K prompts the models refuse", pad=2)
    ax.set_ylim(-0.02, 0.48)
    ax.legend(loc="upper left", frameon=False, handlelength=1.0, labelspacing=0.2, borderaxespad=0.1)
    ax = axes[1]
    ax.axhline(0, color=MUTED, lw=0.6)
    s = [x113[t]["populations"]["hard1k"]["representation"]["proj_S_T_minus_O_mean_sd"] for t in order]
    c = [x113[t]["populations"]["hard1k"]["representation"]["proj_C_T_minus_O_mean_sd"] for t in order]
    ax.plot(xs - 0.12, s, "o", color=BLUE, ms=3.2, label="substitution")
    ax.plot(xs + 0.12, c, "s", color=MUTED, ms=3.2, markerfacecolor="white", label="control")
    ax.set_ylabel("projection change (own sd)")
    ax.set_title("shift along the model's own direction", pad=2)
    ax.legend(loc="center right", frameon=False, handlelength=1.0, borderaxespad=0.1)
    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels([LABELS[t] for t in order], rotation=50, ha="right")
        ax.grid(axis="y")
    fig.tight_layout(pad=0.3, w_pad=1.0)
    fig.savefig(FIG / "table-substitution" / "table-substitution.pdf")
    plt.close(fig)


if __name__ == "__main__":
    vocab_distribution()
    table_dose_response()
    table_substitution()
    print("wrote", [p.name for p in sorted(FIG.glob("*/*.pdf"))])
