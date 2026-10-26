"""Eight-model direction figure for Section 6: per model, (a) the rank
correlation between an inserted word's shift along the model's own
boundary direction and its refusal rise, against the 2.5-97.5% band of
1,000 random directions; (b) refusal removed on the end-inserted text by
writing the direction at -1.5, against norm-matched and KL-matched random
directions; (c) refusal created on the original text at +1.5, against a
norm-matched random direction. Reads
data/results/x103_direction_insertion_projection.json and
data/results/x105_cross_model_direction_causality.json.
Output: fig/direction-xmodel/direction-xmodel.pdf.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
X103 = REPO / "data" / "results" / "x103_direction_insertion_projection.json"
X105 = REPO / "data" / "results" / "x105_cross_model_direction_causality.json"
OUT = HERE.parent / "direction-xmodel.pdf"

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.edgecolor": "#333333",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "text.color": "#333333",
        "axes.labelcolor": "#333333",
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.5,
    }
)

BLUE = "#0072B2"
ORANGE = "#E69F00"
INK = "#333333"
MUTED = "#888888"
BAND = "#E4E4E4"

MODELS = [
    ("qwen08b", "0.8B"),
    ("qwen2b", "2B"),
    ("qwen", "4B"),
    ("qwen9b", "9B"),
    ("qwen27b", "27B"),
    ("llama", "Llama"),
    ("gemma", "Gemma"),
    ("phi", "Phi"),
]
FAMILIES = [range(0, 5), range(5, 6), range(6, 7), range(7, 8)]

x103 = json.loads(X103.read_text())["models"]
x105 = json.loads(X105.read_text())["models"]
xs = list(range(len(MODELS)))


def est(block):
    e, (lo, hi) = block["estimate"], block["ci95"]
    return e, e - lo, hi - e


def series(ax, vals, color, marker, filled, dx=0.0, connect=True, z=4):
    if connect:
        for fam in FAMILIES:
            if len(fam) > 1:
                ax.plot([x + dx for x in fam], [vals[x][0] for x in fam],
                        color=color, lw=1.0, zorder=z - 1)
    ax.errorbar(
        [x + dx for x in xs],
        [v[0] for v in vals],
        yerr=[[v[1] for v in vals], [v[2] for v in vals]],
        fmt=marker, color=color, ms=2.8, elinewidth=0.6, capsize=1.0, lw=0,
        markerfacecolor=color if filled else "white", markeredgewidth=0.8,
        zorder=z,
    )


fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.9),
                         gridspec_kw={"width_ratios": [1, 1, 1]})

# (a) projection tracks the refusal rise
ax = axes[0]
rho = []
part = []
for tag, _ in MODELS:
    m = x103[tag]
    e = m["primary"]["rho"]
    lo, hi = m["primary"]["ci95_prompt_bootstrap"]
    rho.append((e, e - lo, hi - e))
    p = m["controls"]["partial_shift"]
    pe = p["rho"]
    plo, phi = p["ci95_prompt_bootstrap"]
    part.append((pe, pe - plo, phi - pe))
    lo_b, hi_b = m["controls"]["random_directions"]["band_2_5_97_5"]
    ax.bar(xs[len(rho) - 1], hi_b - lo_b, bottom=lo_b, width=0.7,
           color=BAND, zorder=1, linewidth=0)
ax.axhline(0, color=MUTED, lw=0.6, zorder=2)
series(ax, rho, BLUE, "o", True, connect=True)
series(ax, part, ORANGE, "s", False, dx=0.22, connect=False)
ax.set_title("projection tracks the rise", pad=2)
ax.set_ylabel(r"Spearman $\rho$")
ax.legend(
    handles=[
        Line2D([], [], color=BLUE, marker="o", ms=3, lw=1.0, label="own direction"),
        Line2D([], [], color=ORANGE, marker="s", ms=3, lw=0,
               markerfacecolor="white", label="shift norm held fixed"),
        Patch(facecolor=BAND, label="1,000 random directions"),
    ],
    loc="lower left", frameon=False, handlelength=1.3, labelspacing=0.2,
    borderaxespad=0.1,
)

# (b) remove the rise on the end-inserted text, dose -1.5
ax = axes[1]
ax.axhline(0, color=MUTED, lw=0.6, zorder=1)
own = [est(x105[t]["estimands"]["suppress_E_full"]) for t, _ in MODELS]
rnd = [est(x105[t]["estimands"]["random_suppress_E_full"]) for t, _ in MODELS]
rkl = [est(x105[t]["estimands"]["random_kl_suppress_E_full"]) for t, _ in MODELS]
series(ax, own, BLUE, "o", True)
series(ax, rnd, MUTED, "s", False, dx=-0.2, connect=False)
series(ax, rkl, MUTED, "^", False, dx=0.2, connect=False)
ax.set_title("inserted text, dose $-1.5$", pad=2)
ax.set_ylabel("refusal removed")
ax.legend(
    handles=[
        Line2D([], [], color=BLUE, marker="o", ms=3, lw=1.0, label="own direction"),
        Line2D([], [], color=MUTED, marker="s", ms=3, lw=0,
               markerfacecolor="white", label="random, norm-matched"),
        Line2D([], [], color=MUTED, marker="^", ms=3, lw=0,
               markerfacecolor="white", label="random, KL-matched"),
    ],
    loc="upper right", frameon=False, handlelength=1.3, labelspacing=0.2,
    borderaxespad=0.1,
)

# (c) create refusal on the original text, dose +1.5
ax = axes[2]
ax.axhline(0, color=MUTED, lw=0.6, zorder=1)
own = [est(x105[t]["estimands"]["induce_O_full"]) for t, _ in MODELS]
rnd = [est(x105[t]["estimands"]["random_induce_O_full"]) for t, _ in MODELS]
series(ax, own, ORANGE, "o", True)
series(ax, rnd, MUTED, "s", False, dx=-0.2, connect=False)
ax.set_title("original text, dose $+1.5$", pad=2)
ax.set_ylabel("refusal created")
ax.legend(
    handles=[
        Line2D([], [], color=ORANGE, marker="o", ms=3, lw=1.0, label="own direction"),
        Line2D([], [], color=MUTED, marker="s", ms=3, lw=0,
               markerfacecolor="white", label="random, norm-matched"),
    ],
    loc="upper right", frameon=False, handlelength=1.3, labelspacing=0.2,
    borderaxespad=0.1,
)

lo = min(axes[1].get_ylim()[0], axes[2].get_ylim()[0])
for ax in axes[1:]:
    ax.set_ylim(lo, 0.56)  # headroom for the legend
axes[0].set_ylim(-0.95, axes[0].get_ylim()[1])
for ax in axes:
    ax.set_xticks(xs)
    ax.set_xticklabels([lbl for _, lbl in MODELS], rotation=55, ha="right")
    ax.grid(axis="y", zorder=0)
    ax.set_xlim(-0.6, 7.6)

fig.tight_layout(pad=0.4, w_pad=0.8)
fig.savefig(OUT)
print("wrote", OUT)
