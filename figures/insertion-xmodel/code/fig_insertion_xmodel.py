"""Cross-model keyword-insertion figure for the appendix: per insertion
position, the paired refusal increase over the regenerated original for
each of the eight transfer models, ungated (all shared prompts) and gated
(prompts whose inserted word the model names with P > 0.90), with 95%
bootstrap intervals. Gated points resting on fewer than MIN_GATED
admitted prompts are not drawn (their counts are in the appendix table).
Qwen sizes are connected by lines; off-family models
stand alone. Reads data/results/x101_cross_model_insertion.json.
Output: fig/insertion-xmodel/insertion-xmodel.pdf.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
RESULTS = REPO / "data" / "results" / "x101_cross_model_insertion.json"
OUT = HERE.parent / "insertion-xmodel.pdf"

BLUE = "#0072B2"
ORANGE = "#E69F00"
INK = "#333333"
MUTED = "#888888"

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.edgecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.4,
        "axes.axisbelow": True,
    }
)

ORDER = ["qwen08b", "qwen2b", "qwen", "qwen9b", "qwen27b", "llama", "gemma", "phi"]
LABEL = {
    "qwen08b": "Q-0.8B",
    "qwen2b": "Q-2B",
    "qwen": "Q-4B",
    "qwen9b": "Q-9B",
    "qwen27b": "Q-27B",
    "llama": "Llama",
    "gemma": "Gemma",
    "phi": "Phi",
}
QWEN = ORDER[:5]
POSITIONS = [("S", "start"), ("M", "middle"), ("E", "end")]
OFFSET = 0.17
MIN_GATED = 10


def rows(data, key):
    return {(r["model_tag"], r["position"]): r for r in data["cross_model"][key]}


def series(ax, table, pos, xs_shift, color, filled, marker, min_n=1):
    pts = []
    for i, tag in enumerate(ORDER):
        r = table.get((tag, pos))
        n = r.get("n_admitted", r.get("n")) if r else 0
        if not r or not n or n < min_n or r.get("delta") is None:
            pts.append(None)
            continue
        x = i + xs_shift
        lo, hi = r["ci95"]
        d = r["delta"]
        ax.errorbar(
            x, d, yerr=[[d - lo], [hi - d]], fmt=marker, ms=3.6, color=color,
            mfc=color if filled else "white", mew=0.9, elinewidth=0.8, capsize=1.6, zorder=3,
        )
        pts.append((x, d))
    qwen_pts = [p for tag, p in zip(ORDER, pts) if tag in QWEN and p is not None]
    if len(qwen_pts) > 1:
        ax.plot(*zip(*qwen_pts), color=color, lw=0.8, zorder=2)


def main():
    data = json.loads(RESULTS.read_text())
    ungated = rows(data, "ungated_all_130")
    gated = rows(data, "gated_admitted")
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.15), sharey=True)
    for ax, (pos, name) in zip(axes, POSITIONS):
        series(ax, ungated, pos, -OFFSET, ORANGE, False, "s")
        series(ax, gated, pos, OFFSET, BLUE, True, "o", MIN_GATED)
        ax.axhline(0, color=MUTED, lw=0.6, zorder=1)
        ax.axvline(4.5, color="#CCCCCC", lw=0.6, ls=":", zorder=1)
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels([LABEL[t] for t in ORDER], rotation=55, ha="right")
        ax.set_xlim(-0.6, len(ORDER) - 0.4)
        ax.set_ylim(-0.08, 0.9)
        ax.set_title(f"{name} insertion")
    axes[0].set_ylabel("refusal increase over original")
    handles = [
        Line2D([], [], color=ORANGE, marker="s", mfc="white", ms=3.6, lw=0.8, label="ungated (shared prompts)"),
        Line2D([], [], color=BLUE, marker="o", ms=3.6, lw=0.8, label=f"gated (models with $\\geq${MIN_GATED} admitted prompts)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, handlelength=1.6,
               bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(w_pad=0.6)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
