"""Table-guided in-place substitution on the benign Hard-1K prompts (v4).

Writes fig/table-substitution/table-substitution_v4.pdf (+ _v4.png preview).

Layout
  top strip  one real prompt: the word every model's table nominates, the
             substitutes the models' tables chose, and the per-model refusal
             rates before / after / under the control.
  (a)        per model, the refusal rate before (open circle) and after the
             substitution (filled, blue arrow) and after the inert-word control
             (grey); right columns: refusal removed with its 95% two-stage
             paired bootstrap interval, the control, the deletion-pass word on
             the same prompts, and the per-prompt Spearman rho between
             projection change and refusal change.
  (b)        per prompt, refusal removed against the change of the boundary
             projection on the model's own direction (model's Hard-1K sd),
             quintile bins of the projection change, one line per model; the
             pale bands span the eight models' mean shift under the
             substitution and under the control; grey squares are the control
             means.

Reads only data/runs/x113/results.json and data/runs/x113/<tag>/texts.json
(provenance: x113, primary population = units outside the direction-fitting
cohort, per x85_exclusions.json; the `hard1k` population = core + Llama add-on).
Run from anywhere:  python3 fig_table_substitution_v4.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

DRAFT = Path(__file__).resolve().parents[3]
ROOT = DRAFT.parent
RUN = ROOT / "data" / "runs" / "x113"
OUT = DRAFT / "fig" / "table-substitution"

BLUE, ORANGE, INK, MUTED = "#0072B2", "#E69F00", "#333333", "#888888"
PALE_BLUE, PALE_GREY = "#EAF3F9", "#F4F4F4"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.6, "axes.edgecolor": MUTED,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK,
    "grid.color": "#E4E4E4", "grid.linewidth": 0.5, "pdf.fonttype": 42, "ps.fonttype": 42,
})
LABELS = {"qwen": "Qwen3.5-4B", "qwen08b": "Qwen3.5-0.8B", "qwen2b": "Qwen3.5-2B",
          "qwen9b": "Qwen3.5-9B", "qwen27b": "Qwen3.5-27B", "llama": "Llama-3.2-3B",
          "gemma": "Gemma-4-E2B", "phi": "Phi-4-mini"}
ORDER = ["qwen08b", "qwen2b", "qwen", "qwen9b", "qwen27b", "llama", "gemma", "phi"]
POP = "hard1k"
EXAMPLE_ID = "orbench_hard:01130:6f17d59259e3"  # illustration; text and rates read from data
EXAMPLE_MODEL = "qwen"
N_BINS = 5


def f3(v: float, sign: bool = True) -> str:
    s = f"{v:+.2f}" if sign else f"{v:.2f}"
    return s.replace("0.", ".", 1)


def load():
    res = json.load(open(RUN / "results.json"))["models"]
    order = [t for t in ORDER if t in res]
    rows = {}
    for t in order:
        p = res[t]["populations"][POP]
        e, rep = p["estimands"], p["representation"]
        dS, dC = e["delta_S"], e["delta_C"]
        rows[t] = dict(
            # r(O) on the units that have a substitute, so that O -> S equals delta_S
            rO_S=e["r_S_T"]["estimate"] + dS["estimate"], rS=e["r_S_T"]["estimate"],
            rO_C=e["r_C_T"]["estimate"] + dC["estimate"], rC=e["r_C_T"]["estimate"],
            dS=dS["estimate"], dS_ci=dS["ci95"], dC=dC["estimate"], dC_ci=dC["ci95"],
            dX=e["x106_delta_S"]["estimate"],
            rho=rep["spearman_proj_change_vs_refusal_change_S_T"]["rho"],
            projS=rep["proj_S_T_minus_O_mean_sd"], projC=rep["proj_C_T_minus_O_mean_sd"])
        pp = [r for r in res[t]["per_prompt"]
              if not r["x85_excluded"] and r["stratum"] in ("core", "llama_extra")
              and r["r_S_T"] is not None and r["proj_S_T_minus_O_sd"] is not None]
        x = np.array([r["proj_S_T_minus_O_sd"] for r in pp])
        y = np.array([r["r_O"] - r["r_S_T"] for r in pp])
        q = np.quantile(x, np.linspace(0, 1, N_BINS + 1)[1:-1])
        k = np.digitize(x, q)
        rows[t]["bins"] = [(x[k == b].mean(), y[k == b].mean()) for b in range(N_BINS)]
        rows[t]["per_prompt"] = {r["id"]: r for r in res[t]["per_prompt"]}
    units = {u["id"]: u for u in json.load(open(RUN / EXAMPLE_MODEL / "texts.json"))["units"]}
    return order, rows, units[EXAMPLE_ID]


def rich_line(ax, x, y, parts, fontsize, renderer, **kw):
    """Draw [(text, color, weight)] left to right starting at axes coords (x, y)."""
    for text, color, weight in parts:
        t = ax.text(x, y, text, color=color, fontsize=fontsize, fontweight=weight,
                    transform=ax.transAxes, va="baseline", ha="left", **kw)
        bb = t.get_window_extent(renderer=renderer).transformed(ax.transAxes.inverted())
        x = bb.x1
    return x


def example_strip(ax, fig, order, rows, unit):
    ax.set_axis_off()
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    prompt, span = unit["prompt"], unit["nominee"]["span"]
    word = unit["nominee"]["norm"]
    pp = [rows[t]["per_prompt"][EXAMPLE_ID] for t in order]
    subs = []
    for d in pp:
        if d["substitute"] not in subs:
            subs.append(d["substitute"])
    before, after = prompt[:span[0]], prompt[span[1]:]
    tail = prompt[span[0]:span[1]][len(word):]  # punctuation glued to the word
    ctl = (unit["control"]["norm"], unit["control"]["synonym"])
    rO, rS, rC = ([d[k] for d in pp] for k in ("r_O", "r_S_T", "r_C_T"))

    def rng(v):
        lo, hi = min(v), max(v)
        return f"{lo:.1f}" if lo == hi else f"{lo:.1f}\u2013{hi:.1f}"
    fs = 7.0
    rich_line(ax, 0.0, 0.70, [("Example  ", INK, "bold"), ("\u201c" + before, INK, "normal"),
                               (word, ORANGE, "bold"), (tail + after + "\u201d", INK, "normal")], fs, r)
    who = (f"all {len(order)} tables nominate " if {d["nominee"] for d in pp} == {word}
           else "most feared word: ")
    rich_line(ax, 0.0, 0.36, [(who, MUTED, "normal"), (word, ORANGE, "bold"), (" \u2192 ", INK, "normal"),
                               (" / ".join(subs), BLUE, "bold")], fs, r)
    rich_line(ax, 0.0, 0.02, [("refusal rate ", MUTED, "normal"), (rng(rO), INK, "bold"),
                               (" \u2192 ", INK, "normal"), (rng(rS), BLUE, "bold"),
                               ("      control, inert word ", MUTED, "normal"), (ctl[0], INK, "normal"),
                               (" \u2192 ", INK, "normal"), (ctl[1], INK, "normal"), (": ", MUTED, "normal"),
                               (rng(rC), MUTED, "bold"),
                               (f"      (ranges over the {len(order)} models)", MUTED, "normal")], fs, r)


def panel_slopes(ax, axt, order, rows):
    ys = np.arange(len(order))[::-1].astype(float)
    dyS, dyC = 0.14, -0.20
    for y, t in zip(ys, order):
        d = rows[t]
        ax.add_patch(FancyArrowPatch((d["rO_S"], y + dyS), (d["rS"], y + dyS), arrowstyle="-|>",
                                     mutation_scale=6.5, color=BLUE, lw=1.3, shrinkA=2.6, shrinkB=2.2, zorder=2))
        ax.plot([d["rO_C"], d["rC"]], [y + dyC] * 2, color=MUTED, lw=0.8, zorder=2)
        ax.plot(d["rC"], y + dyC, "s", ms=2.6, mfc="white", mec=MUTED, mew=0.8, zorder=3)
        ax.plot(d["rO_S"], y + dyS, "o", ms=4.0, mfc="white", mec=INK, mew=0.9, zorder=3)
        ax.plot(d["rS"], y + dyS, "o", ms=3.6, color=BLUE, zorder=4)
    ax.set_yticks(ys)
    ax.set_yticklabels([LABELS[t] for t in order])
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.65, len(order) - 0.35)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", ".25", ".5", ".75", "1"])
    ax.grid(axis="x")
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("refusal rate on the prompts")
    # key
    ax.plot([], [], "o", ms=4.0, mfc="white", mec=INK, mew=0.9, label="original")
    ax.plot([], [], "o", ms=3.6, color=BLUE, label="substituted")
    ax.plot([], [], "s", ms=2.6, mfc="white", mec=MUTED, mew=0.8, label="control")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.03, 0.985), ncol=3, frameon=False, handlelength=0.8,
              handletextpad=0.3, columnspacing=0.8, borderaxespad=0.0, fontsize=6.3)

    axt.set_axis_off()
    axt.set_xlim(0, 1)
    axt.set_ylim(ax.get_ylim())
    cols = [(0.0, "removed [CI]", "left"), (0.70, "control", "right"),
            (0.87, "deletion\npass", "right"), (1.0, r"$\rho$", "right")]
    for x, h, ha in cols:
        axt.text(x, len(order) - 0.42, h, ha=ha, va="bottom", fontsize=6.2, color=MUTED, linespacing=0.95)
    for y, t in zip(ys, order):
        d = rows[t]
        axt.text(0.0, y, f"{f3(d['dS'])}", ha="left", va="center", fontsize=6.8, color=BLUE, fontweight="bold")
        axt.text(0.17, y, f"[{f3(d['dS_ci'][0], False)}, {f3(d['dS_ci'][1], False)}]", ha="left",
                 va="center", fontsize=6.3, color=BLUE)
        axt.text(0.70, y, f3(d["dC"]), ha="right", va="center", fontsize=6.5, color=MUTED)
        axt.text(0.87, y, f3(d["dX"]), ha="right", va="center", fontsize=6.5, color=INK)
        axt.text(1.0, y, f3(d["rho"], False), ha="right", va="center", fontsize=6.5, color=INK)


def panel_mechanism(ax, order, rows):
    pS = [rows[t]["projS"] for t in order]
    pC = [rows[t]["projC"] for t in order]
    ax.axhline(0, color=MUTED, lw=0.6, zorder=1)
    ax.axvline(0, color=MUTED, lw=0.6, zorder=1)
    ax.axvspan(min(pS), max(pS), color=PALE_BLUE, lw=0, zorder=0)
    ax.axvspan(min(pC), max(pC), color=PALE_GREY, lw=0, zorder=0)
    low = min(order, key=lambda t: rows[t]["rO_S"])
    xmin = min(rows[t]["bins"][0][0] for t in order)
    for t in order:
        bx, by = zip(*rows[t]["bins"])
        is_low = t == low
        ax.plot(bx, by, marker="o", color=BLUE, lw=0.9, ms=2.3, alpha=0.85,
                ls="-" if not is_low else (0, (2.5, 1.2)), zorder=3)
    lb = rows[low]["bins"][0]
    ax.annotate(f"{LABELS[low]}\n(refuses {rows[low]['rO_S']:.2f} at base)", xy=lb, xytext=(np.floor(xmin * 2) / 2, 0.03),
                fontsize=6.2, color=INK, ha="left", va="top", linespacing=1.0,
                arrowprops={"arrowstyle": "-", "lw": 0.4, "color": MUTED})
    ax.plot(pC, [rows[t]["dC"] for t in order], "s", ms=2.6, mfc="white", mec=MUTED, mew=0.8, zorder=4)
    ax.set_xlabel("projection change (own sd)")
    ax.set_ylabel("refusal removed")
    ax.set_xlim(np.floor(xmin * 2) / 2 - 0.1, 1.0)
    ax.set_ylim(-0.17, 0.95)
    ax.grid(axis="y")
    ytop = 0.93
    ax.text(min(pS) - 0.05, ytop, "mean shift\nsubstituted", ha="right", va="top", fontsize=6.2,
            color=BLUE, linespacing=1.0)
    ax.text(max(pC) + 0.06, ytop, "control", ha="left", va="top", fontsize=6.2,
            color=MUTED, linespacing=1.0)


def main() -> None:
    order, rows, unit = load()
    fig = plt.figure(figsize=(6.5, 3.0))
    axe = fig.add_axes([0.012, 0.855, 0.98, 0.14])
    axa = fig.add_axes([0.118, 0.125, 0.29, 0.62])
    axt = fig.add_axes([0.418, 0.125, 0.245, 0.62])
    axb = fig.add_axes([0.735, 0.125, 0.255, 0.62])
    panel_slopes(axa, axt, order, rows)
    panel_mechanism(axb, order, rows)
    example_strip(axe, fig, order, rows, unit)
    fig.text(0.012, 0.80, "a", fontsize=9, fontweight="bold", va="center")
    fig.text(0.672, 0.80, "b", fontsize=9, fontweight="bold", va="center")
    fig.text(0.04, 0.80, "refusal on the benign prompts the models refuse", fontsize=7.5, va="center")
    fig.text(0.70, 0.80, "per prompt: refusal vs. projection", fontsize=7.5, va="center")
    fig.savefig(OUT / "table-substitution_v4.pdf")
    fig.savefig(OUT / "table-substitution_v4.png", dpi=220)
    plt.close(fig)
    print("wrote", OUT / "table-substitution_v4.pdf")


if __name__ == "__main__":
    main()
