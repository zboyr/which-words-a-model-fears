"""Eight-model direction figure, v4 (Section 5, label fig:direction-xmodel).

One row per model, read across two panels that share the rows.

(a) Link: Spearman rho between an inserted word's shift along the model's
    own boundary direction and its refusal rise, after position fixed
    effects (blue, 95% prompt bootstrap), drawn over the 2.5-97.5% band of
    the same statistic for 1,000 random directions (grey).
(b) Write: explicit-refusal rates on two lanes per model. The pale orange
    span is the inserted word's rise, from the original text O to the
    end-inserted text E. Upper lane: the model's own direction written at
    -1.5 on E (blue arrow from E); lower lane: written at +1.5 on O (blue
    arrow from O). Light-blue bar at the arrow head: 95% two-stage bootstrap
    interval of the effect. Grey: random directions on the same text, of the
    same norm (open circle) and, on E, of the same benign KL (diamond).

Inputs (provenance only; nothing below renders an experiment ID):
  (a) data/results/x103_direction_insertion_projection.json  -- canonical
      primary: units outside the direction's fitting cohort (x85-free).
      Set LINK_COHORT = "x101only" to read the 64-unit sibling that shares
      panel (b)'s units exactly.
  (b) data/results/x105_cross_model_direction_causality.json -- the 64 units
      outside the fitting cohort.
Output: fig/direction-xmodel/direction-xmodel_v4.pdf (+ _v4.png preview).
Run from anywhere with a python that has matplotlib (system python3 works).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
RES = REPO / "data" / "results"
LINK_COHORT = "primary"  # "primary" (canonical file) or "x101only"
LINK = RES / ("x103_direction_insertion_projection.json"
              if LINK_COHORT == "primary"
              else "x103_direction_insertion_projection_x101only.json")
WRITE = RES / "x105_cross_model_direction_causality.json"
import os
OUT = Path(os.environ.get("FIG_OUT", HERE.parent / "direction-xmodel_v4.pdf"))

BLUE = "#0072B2"
ORANGE = "#E69F00"
INK = "#333333"
MUTED = "#888888"
PALE_BLUE = "#EAF3F9"
PALE_GREY = "#F4F4F4"
BAND = "#E2E2E2"
RISE = "#FBE7BE"  # pale orange: the word's rise
CI_BLUE = "#B9D5EA"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.edgecolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "axes.grid": True,
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.4,
    "axes.axisbelow": True,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "pdf.fonttype": 42,
})

MODELS = [  # family order, Qwen3.5 by size
    ("qwen08b", "Qwen3.5-0.8B"),
    ("qwen2b", "Qwen3.5-2B"),
    ("qwen", "Qwen3.5-4B"),
    ("qwen9b", "Qwen3.5-9B"),
    ("qwen27b", "Qwen3.5-27B"),
    ("llama", "Llama-3.2-3B"),
    ("gemma", "Gemma-4-E2B"),
    ("phi", "Phi-4-mini"),
]
FAMILY_GAP = 0.45  # extra space between the Qwen3.5 block and the rest
LANE = 0.21        # half-distance between the E lane and the O lane
NULL_DY = 0.13     # vertical offset of the random-direction markers

link = json.loads(LINK.read_text())
write = json.loads(WRITE.read_text())
lm, wm = link["models"], write["models"]
n_link = lm["qwen"]["counts"]["n_prompts"]
n_write = wm["qwen"]["n_prompts"]
assert all(lm[t]["counts"]["n_prompts"] == n_link for t, _ in MODELS)
assert all(wm[t]["n_prompts"] == n_write for t, _ in MODELS)

# row centres, top to bottom
ys = []
y = 0.0
for i, (tag, _) in enumerate(MODELS):
    if tag == "llama":
        y -= FAMILY_GAP
    ys.append(y)
    y -= 1.0

fig = plt.figure(figsize=(6.3, 2.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 2.2], wspace=0.06,
                      left=0.135, right=0.985, top=0.855, bottom=0.135)
axL = fig.add_subplot(gs[0])
axR = fig.add_subplot(gs[1], sharey=axL)

# ---------------------------------------------------------------- (a) link
for (tag, _), yc in zip(MODELS, ys):
    m = lm[tag]
    lo_b, hi_b = m["controls"]["random_directions"]["band_2_5_97_5"]
    axL.add_patch(plt.Rectangle((lo_b, yc - 0.27), hi_b - lo_b, 0.54,
                                facecolor=BAND, edgecolor="none", zorder=1))
    r = m["primary"]["rho"]
    lo, hi = m["primary"]["ci95_prompt_bootstrap"]
    axL.plot([lo, hi], [yc, yc], color=BLUE, lw=1.1, solid_capstyle="butt",
             zorder=3)
    axL.plot(r, yc, "o", ms=4.2, color=BLUE, mec="white", mew=0.6, zorder=4)
axL.axvline(0, color=MUTED, lw=0.6, zorder=2)
axL.set_xlim(-0.66, 0.9)
axL.set_xticks([-0.5, 0, 0.5])
axL.set_xticklabels(["$-0.5$", "0", "$0.5$"])
axL.set_xlabel(r"Spearman $\rho$, shift along direction vs. refusal rise",
               labelpad=2)
axL.grid(axis="y", visible=False)
axL.tick_params(axis="y", length=0, pad=3)
axL.set_yticks(ys)
axL.set_yticklabels([lbl for _, lbl in MODELS])
axL.set_title("(a) projection predicts the rise",
              loc="left", pad=16, fontsize=8)
axL.text(0.0, 1.015, f"{n_link} prompts; grey: 1,000 random directions",
         transform=axL.transAxes, fontsize=6.5, color=MUTED, ha="left",
         va="bottom")

# --------------------------------------------------------------- (b) write
def ci_bar(ax, start, sign, block, yc):
    """Light bar for the 95% interval of an effect applied from `start`."""
    lo, hi = block["ci95"]
    a, b = start + sign * lo, start + sign * hi
    ax.plot([a, b], [yc, yc], color=CI_BLUE, lw=2.4,
            solid_capstyle="butt", zorder=2)


def arrow(ax, x0, x1, yc):
    ax.add_patch(FancyArrowPatch((x0, yc), (x1, yc),
                                 arrowstyle="-|>,head_length=3.2,head_width=1.7",
                                 color=BLUE, lw=1.15, shrinkA=0, shrinkB=0,
                                 zorder=4))


def null_pt(ax, start, sign, block, yc, marker, filled):
    x = start + sign * block["estimate"]
    lo, hi = block["ci95"]
    ax.plot([start + sign * lo, start + sign * hi], [yc, yc], color=MUTED,
            lw=0.6, zorder=3)
    ax.plot(x, yc, marker, ms=3.0, color=MUTED,
            mfc=MUTED if filled else "white", mew=0.7, zorder=4)


for (tag, _), yc in zip(MODELS, ys):
    m = wm[tag]
    est = m["estimands"]
    rO, rE = m["arm_means"]["O_base"], m["arm_means"]["E_base"]
    yE, yO = yc + LANE, yc - LANE
    # the word's rise, spanning both lanes
    axR.add_patch(plt.Rectangle((rO, yc - 0.40), rE - rO, 0.80,
                                facecolor=RISE, edgecolor="none", zorder=0))
    for x in (rO, rE):
        axR.plot([x, x], [yc - 0.40, yc + 0.40], color=ORANGE, lw=0.7,
                 zorder=1)
    # E lane: subtract the direction
    sup = est["suppress_E_full"]
    ci_bar(axR, rE, -1, sup, yE)
    arrow(axR, rE, rE - sup["estimate"], yE)
    axR.plot(rE, yE, "o", ms=3.4, color=INK, zorder=5)
    null_pt(axR, rE, -1, est["random_suppress_E_full"], yE + NULL_DY * 0.9,
            "o", False)
    null_pt(axR, rE, -1, est["random_kl_suppress_E_full"], yE - NULL_DY * 0.9,
            "D", True)
    # O lane: add the direction
    ind = est["induce_O_full"]
    ci_bar(axR, rO, +1, ind, yO)
    arrow(axR, rO, rO + ind["estimate"], yO)
    axR.plot(rO, yO, "o", ms=3.4, color="white", mec=INK, mew=0.9, zorder=5)
    null_pt(axR, rO, +1, est["random_induce_O_full"], yO - NULL_DY * 0.9,
            "o", False)

axR.set_xlim(0, 1.08)
axR.set_xticks([0, 0.2, 0.4, 0.6])
axR.set_xticklabels(["0", "0.2", "0.4", "0.6"])
axR.spines["bottom"].set_bounds(0, 0.7)
axR.set_xlabel("explicit refusal rate", labelpad=2, x=0.35)
axR.grid(axis="y", visible=False)
axR.tick_params(axis="y", left=False, labelleft=False)
axR.set_title("(b) writing the direction removes the rise from E and creates it on O",
              loc="left", pad=16, fontsize=8)
axR.text(0.0, 1.015, f"{n_write} prompts; bars: 95% intervals", transform=axR.transAxes,
         fontsize=6.5, color=MUTED, ha="left", va="bottom")
axL.set_ylim(ys[-1] - 0.62, ys[0] + 0.55)

# legend, right panel, upper right (empty region beyond 0.7)
handles = [
    Patch(facecolor=RISE, edgecolor=ORANGE, lw=0.6,
          label="the word's rise"),
    Line2D([], [], ls="", marker="o", ms=3.4, color="white", mec=INK,
           mew=0.9, label="O, original text"),
    Line2D([], [], ls="", marker="o", ms=3.4, color=INK,
           label="E, word at the end"),
    Line2D([], [], color=BLUE, lw=1.15, marker=">", ms=3.5,
           label="own direction,\n$-1.5$ on E, $+1.5$ on O"),
    Line2D([], [], color=MUTED, lw=0.6, marker="o", ms=3.0, mfc="white",
           label="random, same norm"),
    Line2D([], [], color=MUTED, lw=0.6, marker="D", ms=2.8,
           label="random, same benign KL"),
]
axR.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 0.0),
           frameon=False, handlelength=1.5, labelspacing=0.45,
           borderaxespad=0.1, handletextpad=0.5)

fig.savefig(OUT)
fig.savefig(OUT.with_suffix(".png"), dpi=220)
print("wrote", OUT)
