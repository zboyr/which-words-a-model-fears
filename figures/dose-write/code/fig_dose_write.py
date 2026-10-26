"""Dose-response figure for the boundary write alone (v1 paper, appendix on
the dose unit and deep doses): delta vs base on four external endpoints as
the dose of the Qwen3.5-4B boundary write moves from +1.5 to -3. Reads
data/results/x104_external_safety_utility.json and
data/results/x104_positive_dose_extension.json (positive doses).
Output: fig/dose-write/dose-write.pdf (single column).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
RESULTS = REPO / "data" / "results" / "x104_external_safety_utility.json"
RESULTS_POS = REPO / "data" / "results" / "x104_positive_dose_extension.json"
OUT = HERE.parent / "dose-write.pdf"

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
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

ENDPOINTS = [
    ("orbench_hard_refusal", "Hard-1K refusal", BLUE, "o"),
    ("orbench_toxic_refusal", "Toxic refusal", ORANGE, "s"),
    ("strongreject_score", "StrongREJECT", MUTED, "^"),
    ("ifeval_prompt_strict", "IFEval strict", INK, "D"),
]

# condition name -> signed dose; None = base (delta 0 by construction)
WRITE_DOSES = [
    ("x86_p150", 1.5),
    ("x86_p100", 1.0),
    ("x86_p050", 0.5),
    (None, 0.0),
    ("x86_full", -0.5),
    ("x86_m075", -0.75),
    ("x86_m100", -1.0),
    ("x86_m125", -1.25),
    ("x86_m150", -1.5),
    ("x86_m200", -2.0),
    ("x86_m300", -3.0),
]
REGISTERED = "x86_full"

comp = json.loads(RESULTS.read_text())["comparisons"]
comp.update(json.loads(RESULTS_POS.read_text())["comparisons"])

fig, ax = plt.subplots(figsize=(3.05, 2.3))

ax.axhspan(-0.02, 0.02, color=ORANGE, alpha=0.10, lw=0, zorder=0)
ax.axhline(0, color=MUTED, lw=0.6)
for key, label, color, marker in ENDPOINTS:
    xs, ys, lo, hi = [], [], [], []
    for cond, dose in WRITE_DOSES:
        xs.append(dose)
        if cond is None:
            ys.append(0.0)
            lo.append(0.0)
            hi.append(0.0)
        else:
            e = comp[cond][key]
            ys.append(e["mean_delta"])
            lo.append(e["mean_delta"] - e["two_sided_ci"][0])
            hi.append(e["two_sided_ci"][1] - e["mean_delta"])
    ax.errorbar(
        xs,
        ys,
        yerr=[lo, hi],
        color=color,
        marker=marker,
        ms=3.5,
        lw=1.2,
        elinewidth=0.7,
        capsize=1.5,
        label=label,
        markerfacecolor="white" if marker != "o" else color,
        markeredgewidth=0.8,
        zorder=3,
    )
# registered dose: filled marker overlay
reg_dose = [d for c, d in WRITE_DOSES if c == REGISTERED][0]
for key, label, color, marker in ENDPOINTS:
    ax.plot(
        reg_dose,
        comp[REGISTERED][key]["mean_delta"],
        marker=marker,
        ms=4.5,
        color=color,
        zorder=4,
    )
ax.plot(0, 0, marker="o", ms=3.5, color=MUTED, zorder=5)
ax.annotate(
    "base", (0, 0), textcoords="offset points", xytext=(0, 5),
    ha="center", fontsize=6, color=MUTED,
)
ax.invert_xaxis()
ax.set_xlabel("signed dose $a$")
ax.set_ylabel("change vs. base")
ax.set_xticks([1.5, 1.0, 0.5, 0, -0.5, -1.0, -1.5, -2.0, -3.0])
ax.set_xticklabels(["+1.5", "+1", "+.5", "0", "−.5", "−1", "−1.5", "−2", "−3"])
ax.grid(axis="y", zorder=0)
ax.text(-3.0, -0.019, "±0.02 margin", fontsize=6, color=MUTED, ha="right", va="bottom")
ax.legend(loc="lower left", frameon=False, handlelength=1.6)

fig.tight_layout(pad=0.4)
fig.savefig(OUT)
print("wrote", OUT)
