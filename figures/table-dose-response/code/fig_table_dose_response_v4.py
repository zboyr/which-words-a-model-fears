"""Held-out insertion dose-response of the vocabulary table, v4 (figure*, full width).

Replaces the eight overlaid curves of v1 with one small panel per model
(2 x 4, models in size order, Qwen3.5 family first). Each panel draws the
model's own concordant-bin curve with its 95% band, the three words of each
bin as dots, and the other seven curves as grey ghosts for comparison.
Qwen3.5-4B, whose over-refused table is the reference table of the right
column, is drawn in orange. The right column keeps the paired own-table vs
reference-table Spearman rho per model, in the same order.

Every number is read from (provenance only; nothing below renders an ID):
  data/runs/x111/results.json   Qwen3.5-4B (prompts outside the fitting cohort)
  data/runs/x112/results.json   the seven other models (same exclusion, top-up)
Pass --full to read the *_full.json siblings (whole cohort) instead.
Output: fig/table-dose-response/table-dose-response_v4.pdf and .png
(override the stem with FIG_OUT=/path/stem).
Run with any python3 that has matplotlib (tested with 3.10).
"""

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
RUNS = REPO / "data" / "runs"
FULL = "--full" in sys.argv
SUFFIX = "_full" if FULL else ""
OUT = Path(os.environ.get("FIG_OUT", HERE.parent / ("table-dose-response_v4" + SUFFIX)))

BLUE, ORANGE, INK, MUTED = "#0072B2", "#E69F00", "#333333", "#888888"
PALE_BLUE, PALE_ORANGE, GHOST = "#D6E6F2", "#F8E3B5", "#D9D9D9"

plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "axes.labelcolor": INK, "text.color": INK,
    "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": "#E8E8E8", "grid.linewidth": 0.4, "axes.axisbelow": True,
    "pdf.fonttype": 42,
})

REF = "qwen"  # the model whose over-refused table is the reference table
ORDER = ["qwen08b", "qwen2b", "qwen", "qwen9b", "qwen27b", "llama", "gemma", "phi"]


def load():
    x111 = json.load(open(RUNS / "x111" / f"results{SUFFIX}.json"))
    x112 = json.load(open(RUNS / "x112" / f"results{SUFFIX}.json"))
    out = {}

    def entry(label, dr, per_word, rho_own, rho_ref, diff):
        bins = dr["bins"]
        names = [b["bin"] for b in bins]
        words = {n: [w["pooled"]["delta_EO"] for w in per_word
                     if w["kind"] == "concordant" and w["bin_name"] == n] for n in names}
        return {
            "label": label, "bins": names,
            "mean": np.array([b["delta_EO"] for b in bins]),
            "lo": np.array([b["delta_EO_ci95"][0] for b in bins]),
            "hi": np.array([b["delta_EO_ci95"][1] for b in bins]),
            "words": words,
            "in_order": dr["adjacent_pairs_in_order"], "pairs": dr["adjacent_pairs_total"],
            "rho_own": rho_own, "rho_ref": rho_ref, "diff": diff,
        }

    rp = x111["rho"]["pooled"]
    out[REF] = entry("Qwen3.5-4B", x111["dose_response"]["pooled"], x111["per_word"],
                     rp["table_BP"]["all40"]["rho"], None, None)
    for m in x112["models"]:
        rp = m["rho"]["pooled"]
        out[m["model_tag"]] = entry(
            m["label"], m["dose_response"]["pooled"], m["per_word"],
            rp["table_own"]["all40"]["rho"], rp["table_qwen_BP"]["all40"]["rho"],
            rp["own_minus_qwen_BP_all40"])
    return out


def edge_labels(names):
    """'[0,1)', '[1,5)', ... -> the inner percentile edges '1', '5', ..., '99'."""
    return [n.strip("[]()").split(",")[1] for n in names[:-1]]


def main():
    D = load()
    models = [t for t in ORDER if t in D]
    nb = len(D[models[0]]["bins"])
    xs = np.arange(nb)
    allv = np.concatenate([np.r_[D[t]["lo"], D[t]["hi"], sum(D[t]["words"].values(), [])]
                           for t in models])
    ylo, yhi = np.floor(allv.min() * 10) / 10 - 0.02, np.ceil(allv.max() * 10) / 10 + 0.02

    fig = plt.figure(figsize=(6.3, 2.75))
    L, R4, TOP, BOT = 0.062, 0.712, 0.905, 0.15  # left grid; right column placed explicitly
    gs = GridSpec(2, 4, figure=fig, left=L, right=R4, top=TOP, bottom=BOT, wspace=0.12, hspace=0.42)
    rng = np.random.default_rng(0)
    jit = {n: rng.uniform(-0.17, 0.17, 3) for n in D[models[0]]["bins"]}
    axes = []
    for k, t in enumerate(models):
        r, c = divmod(k, 4)
        ax = fig.add_subplot(gs[r, c], sharey=axes[0] if axes else None)
        axes.append(ax)
        d = D[t]
        col, pale = (ORANGE, PALE_ORANGE) if t == REF else (BLUE, PALE_BLUE)
        ax.axhline(0, color=MUTED, lw=0.5, zorder=1)
        for u in models:
            if u != t:
                ax.plot(xs, D[u]["mean"], color=GHOST, lw=0.6, zorder=1.5)
        ax.fill_between(xs, d["lo"], d["hi"], color=pale, lw=0, zorder=2)
        for i, n in enumerate(d["bins"]):
            w = d["words"][n]
            ax.scatter(i + jit[n][:len(w)], w, s=3.5, color=col, alpha=0.45, lw=0, zorder=3)
        ax.plot(xs, d["mean"], color=col, lw=1.3, zorder=4, solid_capstyle="round")
        ax.scatter(xs, d["mean"], s=7, color=col, zorder=5, lw=0)
        ax.set_title(d["label"], loc="left", pad=2, fontsize=7, color=INK)
        ax.text(0.98, 0.04, f"{d['in_order']}/{d['pairs']} in order", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=5.8, color=MUTED)
        ax.set_xlim(-0.45, nb - 0.55)
        ax.set_ylim(ylo, yhi)
        ax.set_xticks(xs[:-1] + 0.5)
        ax.set_xticklabels(edge_labels(d["bins"]) if r == 1 else [], fontsize=5.4)
        ax.tick_params(axis="x", length=1.5, pad=1.2)
        ax.set_yticks(np.arange(np.ceil(ylo * 5) / 5, yhi, 0.2))
        if c != 0:
            ax.tick_params(axis="y", labelleft=False)
    fig.supylabel(r"refusal change $r(\mathrm{E})-r(\mathrm{O})$", x=0.008, y=0.52, fontsize=7)
    fig.text((L + R4) / 2, 0.035,
             "percentile of the appended word in the model's own table (bin edges)",
             ha="center", va="bottom", fontsize=7)

    # right column: which table ranks the 40 words
    RL = 0.842  # leaves room for the model names on the left of the column
    ax = fig.add_axes([RL, BOT, 0.985 - RL, TOP - BOT - 0.075])
    ax.grid(axis="x", color="#E8E8E8", lw=0.4)
    ax.grid(axis="y", visible=False)
    ys = np.arange(len(models))[::-1]
    n_win = n_cmp = 0
    for y, t in zip(ys, models):
        d = D[t]
        own_col = ORANGE if t == REF else BLUE
        if d["rho_ref"] is not None:
            n_cmp += 1
            win = d["rho_own"] > d["rho_ref"]
            n_win += win
            ax.plot([d["rho_ref"], d["rho_own"]], [y, y], color=BLUE if win else MUTED,
                    lw=1.6 if win else 0.9, alpha=0.55 if win else 0.8, zorder=1,
                    solid_capstyle="butt")
            ax.scatter([d["rho_ref"]], [y], s=16, facecolor="white", edgecolor=INK, lw=0.7, zorder=3)
        else:
            ax.text(d["rho_own"] - 0.025, y, "reference\ntable", ha="right", va="center",
                    fontsize=5.5, color=ORANGE, linespacing=0.9)
        ax.scatter([d["rho_own"]], [y], s=18, color=own_col, zorder=4, lw=0)
    ax.set_yticks(ys)
    ax.set_yticklabels([D[t]["label"] for t in models], fontsize=6)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.6, len(models) - 0.4)
    lo = min(min(D[t]["rho_own"] for t in models),
             min(D[t]["rho_ref"] for t in models if D[t]["rho_ref"] is not None))
    ax.set_xlim(np.floor(lo * 10) / 10 - 0.02, 1.0)
    ax.set_xticks(np.arange(np.ceil(lo * 10) / 10, 1.001, 0.1))
    ax.set_xticklabels([f"{v:.1f}" for v in ax.get_xticks()])
    ax.set_xlabel(r"Spearman $\rho$, 40 words", labelpad=2)
    x0 = RL - 0.125  # align the column header with the model names
    fig.text(x0, TOP + 0.005, f"own table ranks better on {n_win} of {n_cmp}",
             ha="left", va="bottom", fontsize=7)
    fig.legend(handles=[Line2D([], [], color=BLUE, marker="o", lw=0, ms=3.6, label="own table"),
                        Line2D([], [], color=INK, marker="o", markerfacecolor="white", lw=0,
                               ms=3.6, mew=0.7, label="Qwen3.5-4B table")],
               loc="upper left", bbox_to_anchor=(x0 - 0.008, TOP - 0.005), ncol=2, frameon=False,
               fontsize=6, handletextpad=0.15, columnspacing=0.8, borderaxespad=0.0)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"), dpi=220)
    print("wrote", OUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
