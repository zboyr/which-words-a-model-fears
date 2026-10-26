"""Dose response of the Qwen3.5-4B boundary write, v4 (single column).

One panel per endpoint instead of four lines in one panel: the over-refusal
endpoint gets its own tall panel on its own scale, and the guards plus the
capability checks sit in a 2x2 grid below on one shared, four-times-finer
scale, each with its registered non-inferiority margin drawn on the adverse
side only. Orange marks the registered plan (dose -0.5 and the margins);
the sign-flipped positive doses are drawn hollow and muted.

Reads (provenance only; nothing below renders an experiment ID):
  data/results/x104_external_safety_utility.json   negative doses, gates
  data/results/x104_positive_dose_extension.json   positive doses, GSM8K +
  data/results/x87_gsm8k_posthoc.json              GSM8K negative doses
  data/results/x86_boundary_intervention.json      a* (registered dose)
Output: fig/dose-write/dose-write_v4.pdf and .png.
Run with any python3 that has matplotlib (tested with 3.10).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
RES = REPO / "data" / "results"
import os
OUT = Path(os.environ.get("FIG_OUT", HERE.parent / "dose-write_v4.pdf"))

BLUE = "#0072B2"
ORANGE = "#E69F00"
INK = "#333333"
MUTED = "#888888"
PALE_BLUE = "#EAF3F9"
PALE = "#F4F4F4"

plt.rcParams.update(
    {
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#E4E4E4",
        "grid.linewidth": 0.4,
        "axes.axisbelow": True,
        "pdf.fonttype": 42,
    }
)

# ---------------------------------------------------------------- data
ext = json.loads((RES / "x104_external_safety_utility.json").read_text())
pos = json.loads((RES / "x104_positive_dose_extension.json").read_text())
gsm_neg = json.loads((RES / "x87_gsm8k_posthoc.json").read_text())
a_star = json.loads((RES / "x86_boundary_intervention.json").read_text())["config"]["a_star"]

comp = dict(ext["comparisons"])
comp.update(pos["comparisons"])
margins = ext["margins"]
base_rates = pos["base_rates"]

# boundary-write conditions -> signed dose (the rank-one conditions in the
# same files belong to another intervention and are not drawn)
DOSES = {
    "x86_p150": 1.5, "x86_p100": 1.0, "x86_p050": 0.5,
    "x86_full": -a_star, "x86_m075": -0.75, "x86_m100": -1.0,
    "x86_m125": -1.25, "x86_m150": -1.5, "x86_m200": -2.0, "x86_m300": -3.0,
}
REGISTERED = "x86_full"
assert abs(DOSES[REGISTERED] + 0.5) < 1e-9


def gsm_entry(cond):
    src = pos["comparisons"][cond]["gsm8k"] if cond in pos["comparisons"] \
        else gsm_neg["comparisons"][cond]
    return src["strict-match"]


def series(key):
    """(dose, delta, lo, hi) sorted by dose; base point (0, 0) included."""
    pts = [(0.0, 0.0, 0.0, 0.0)]
    for cond, d in DOSES.items():
        e = gsm_entry(cond) if key == "gsm8k" else comp[cond][key]
        lo, hi = e["two_sided_ci"]
        pts.append((d, e["mean_delta"], lo, hi))
    return sorted(pts)


# endpoint: (key, title, adverse sign for the margin or None, base rate)
GUARDS = [
    ("orbench_toxic_refusal", "Toxic refusal", -1,
     base_rates["orbench_toxic_refusal"], "keep refusing"),
    ("strongreject_score", "StrongREJECT", +1,
     base_rates["strongreject_score"], "stay harmless"),
    ("ifeval_prompt_strict", "IFEval strict", -1,
     base_rates["ifeval_prompt_strict"], "follow instructions"),
    ("gsm8k", "GSM8K", None,
     base_rates["gsm8k"]["strict-match"], "do arithmetic"),
]

# ---------------------------------------------------------------- layout
W = 3.03  # ACL \columnwidth in inches
fig = plt.figure(figsize=(W, 3.0))
gs = GridSpec(2, 1, figure=fig, height_ratios=[1.45, 2],
              left=0.145, right=0.975, top=0.955, bottom=0.105,
              hspace=0.36)
sub = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[1], hspace=0.42,
                              wspace=0.12)
ax_h = fig.add_subplot(gs[0])
ax_g = [fig.add_subplot(sub[i, j]) for i in range(2) for j in range(2)]

MINUS = "−"
XT = [1.5, 0.5, 0, -0.5, -1, -1.5, -2, -3]
XL = ["+1.5", "+.5", "0", MINUS + ".5", MINUS + "1", MINUS + "1.5",
      MINUS + "2", MINUS + "3"]
XT_S = [1, 0, -1, -2, -3]
XL_S = ["+1", "0", MINUS + "1", MINUS + "2", MINUS + "3"]
XLIM = (1.75, -3.2)  # inverted: removing fear reads left to right


def fmt(v, nd=3):
    s = f"{v:+.{nd}f}".replace("0.", ".", 1)
    return s.replace("-", MINUS)


def draw(ax, pts, color, ms=3.0, lw=1.1):
    ctrl = [p for p in pts if p[0] >= 0]
    write = [p for p in pts if p[0] <= 0]
    for grp, solid in ((ctrl, False), (write, True)):
        x = [p[0] for p in grp]
        y = [p[1] for p in grp]
        c = color if solid else MUTED
        ax.plot(x, y, color=c, lw=lw if solid else 0.8,
                ls="-" if solid else (0, (2, 1.2)), zorder=3)
        for d, m, lo, hi in grp:
            if d == 0:
                continue
            ax.plot([d, d], [lo, hi], color=c, lw=0.7, alpha=0.85,
                    solid_capstyle="butt", zorder=2)
            ax.plot(d, m, "o", ms=ms, zorder=4,
                    mfc=c if solid else "white", mec=c, mew=0.7)
    # base point (zero by construction)
    ax.plot(0, 0, "o", ms=ms * 0.85, mfc="white", mec=INK, mew=0.7, zorder=5)
    # registered dose: orange ring
    reg = [p for p in pts if abs(p[0] + a_star) < 1e-9][0]
    ax.plot(reg[0], reg[1], "o", ms=ms + 2.6, mfc="none", mec=ORANGE,
            mew=1.1, zorder=6)


def frame(ax, xt, xl):
    ax.set_xlim(*XLIM)
    ax.axhline(0, color=MUTED, lw=0.5, zorder=1)
    ax.set_xticks(xt)
    ax.set_xticklabels(xl)


# --- top: the over-refusal endpoint on its own scale
hard = series("orbench_hard_refusal")
frame(ax_h, XT, XL)
ax_h.axvspan(XLIM[0], 0, color=PALE, lw=0, zorder=0)
ax_h.axvline(-a_star, color=ORANGE, lw=0.6, ls=(0, (1.5, 1.5)), zorder=1)
# registered superiority threshold: refusal must fall by at least 0.02
ax_h.plot([0, XLIM[1]], [-0.02, -0.02], color=ORANGE, lw=0.6, zorder=1)
draw(ax_h, hard, BLUE, ms=3.2, lw=1.3)
ax_h.set_ylim(-0.25, 0.09)
ax_h.set_yticks([-0.2, -0.1, 0])
ax_h.set_yticklabels([MINUS + ".2", MINUS + ".1", "0"])
ax_h.set_title(
    f"Hard-1K over-refusal (base {base_rates['orbench_hard_refusal']:.3f})",
    loc="left", pad=3, fontweight="bold", color=BLUE)
ax_h.text(1.62, -0.238, "added\n(sign-flipped)", fontsize=6, color=MUTED,
          ha="left", va="bottom", linespacing=0.95)
ax_h.text(-a_star - 0.08, -0.238, "registered", fontsize=6, color=ORANGE,
          ha="right", va="bottom")
end = hard[0]  # deepest dose (sorted ascending)
ax_h.annotate(fmt(end[1]), (end[0], end[1]), textcoords="offset points",
              xytext=(-6, 0), fontsize=6, color=BLUE, ha="right",
              va="center")

# --- bottom: guards and capability on one shared, finer scale
YG = 0.072
for ax, (key, title, adverse, br, gloss) in zip(ax_g, GUARDS):
    frame(ax, XT_S, XL_S)
    ax.set_ylim(-YG, YG)
    ax.set_yticks([-0.05, 0, 0.05])
    if adverse is not None:
        m = margins[key]
        lo, hi = (m, YG) if adverse > 0 else (-YG, -m)
        ax.axhspan(lo, hi, color=ORANGE, alpha=0.14, lw=0, zorder=0.5)
        ax.axhline(adverse * m, color=ORANGE, lw=0.6, zorder=1)
    draw(ax, series(key), INK, ms=2.4, lw=0.95)
    ax.set_title(f"{title} ({br:.3f})", loc="left", pad=2.5, fontsize=6.6)
    ax.tick_params(axis="x", labelsize=6)
for ax in ax_g[1::2]:
    ax.set_yticklabels([])
for ax in ax_g[0::2]:
    ax.set_yticklabels([MINUS + ".05", "0", "+.05"])
for ax in ax_g[:2]:
    ax.set_xticklabels([])
ax_g[2].text(1.6, -0.047, "beyond margin", fontsize=5.8, color="#9A6A00",
             ha="left", va="center")
fig.text(0.012, 0.53, "change vs. base", rotation=90, ha="left",
         va="center", fontsize=7)
fig.text(0.57, 0.008, "signed dose $a$ (multiples of the mean boundary shift)",
         ha="center", va="bottom", fontsize=7)

fig.savefig(OUT)
fig.savefig(OUT.with_suffix(".png"), dpi=220)
print("wrote", OUT)
