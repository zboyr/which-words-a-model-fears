"""Insertion-effect figure, v4: the paired before/after transition of every
prompt that passes the detection gate, one panel per insertion position.

Each panel is an 11 x 11 grid of per-prompt refusal counts (k of 10
samples): control (unmodified prompt, regenerated in the same jobs) on the
horizontal axis, the same prompt with one inserted harm word on the
vertical axis. A bubble's area is the number of prompts making that
transition; bubbles above the diagonal rose, on it stayed, below it fell.
The top and right margins are the two marginal distributions (what the
earlier histogram version showed); the arrow from the diagonal to the
mean point is the paired mean increase, whose 95% bootstrap interval is
printed with the rose/same/fell shares.

Provenance (internal ids appear only here, never in the render):
  per-prompt refusals  data/runs/x83/x83_refusal_{O,S,M,E}.jsonl
  detection gate       data/runs/x83/x83_detect.json
  estimates and CIs    data/results/x83_keyword_insertion.json
                       (primary_admitted_delta_vs_O)

Run from anywhere with a matplotlib >= 3.7 Python:
    python3 fig_insertion_dist_v4.py [OUT_DIR]
Default output: ../insertion-dist_v4.pdf and ../insertion-dist_v4.png.
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
RUNS = REPO / "data" / "runs" / "x83"
RESULT = REPO / "data" / "results" / "x83_keyword_insertion.json"
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

BLUE = "#0072B2"
ORANGE = "#E69F00"
INK = "#333333"
MUTED = "#888888"
PALEBLUE = "#EAF3F9"
PALEGRAY = "#F4F4F4"
GRID = "#DDDDDD"

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "pdf.fonttype": 42,
})

# ---------------------------------------------------------------- data
def load_counts(cond):
    """Per-prompt refusal count k (of n samples) for one condition."""
    out, n_samp = {}, set()
    with open(RUNS / f"x83_refusal_{cond}.jsonl") as f:
        for line in f:
            row = json.loads(line)
            out[row["id"]] = int(sum(bool(v) for v in row["refusal"]))
            n_samp.add(len(row["refusal"]))
    assert len(n_samp) == 1, "mixed sample counts"
    return out, n_samp.pop()


counts, N_SAMP = {}, None
for c in "OSME":
    counts[c], n = load_counts(c)
    assert N_SAMP in (None, n)
    N_SAMP = n

detect = json.load(open(RUNS / "x83_detect.json"))["rows"]
N_COHORT = len(detect)
result = json.load(open(RESULT))
primary = result["primary_admitted_delta_vs_O"]

POSITIONS = [("S", "Start"), ("M", "Middle"), ("E", "End")]
K = N_SAMP + 1  # number of possible rates, 0..N_SAMP

panels = []
for pos, name in POSITIONS:
    ids = sorted(r["id"] for r in detect if r["positions"][pos]["admitted"])
    ko = np.array([counts["O"][i] for i in ids])
    kt = np.array([counts[pos][i] for i in ids])
    mat = np.zeros((K, K), int)  # mat[control, inserted]
    np.add.at(mat, (ko, kt), 1)
    est = primary[pos]
    assert est["n"] == len(ids)
    # the arrow is drawn from the raw per-prompt means; check they agree
    # with the result file's estimate before trusting either
    assert abs((kt.mean() - ko.mean()) / N_SAMP - est["estimate"]) < 1e-9
    panels.append(dict(
        pos=pos, name=name, n=len(ids), ko=ko, kt=kt, mat=mat,
        est=est["estimate"], ci=est["ci95"],
        rose=np.mean(kt > ko), same=np.mean(kt == ko), fell=np.mean(kt < ko),
    ))

CMAX = max(p["mat"].max() for p in panels)
MARG_MAX = max(max(np.bincount(p["ko"], minlength=K).max(),
                   np.bincount(p["kt"], minlength=K).max()) / p["n"]
               for p in panels)

# -------------------------------------------------------------- layout
W, H = 6.30, 2.39            # ACL \textwidth = 455.24 pt = 6.30 in
LEFT, BOTTOM = 0.43, 0.36    # room for y label and x label
S = 1.385                    # main square side (in)
MG = 0.035                   # gap main -> marginal
MW = 0.30                    # marginal thickness
GAP = 0.30                   # gap between position blocks
TOP_BAND = 0.30              # title + gate strip

fig = plt.figure(figsize=(W, H))
CELL_PT = S / K * 72
SIZE_K = 0.92 * CELL_PT ** 2  # scatter area (pt^2) of the largest cell


def area(c):
    return SIZE_K * c / CMAX


def ax_at(x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


lo, hi = -0.6, N_SAMP + 0.6
for j, p in enumerate(panels):
    x0 = LEFT + j * (S + MG + MW + GAP)
    ax = ax_at(x0, BOTTOM, S, S)
    axt = ax_at(x0, BOTTOM + S + MG, S, MW)
    axr = ax_at(x0 + S + MG, BOTTOM, MW, S)

    # rose / fell half-planes and the no-change diagonal
    ax.add_patch(Polygon([(lo, lo), (lo, hi), (hi, hi)], closed=True,
                         facecolor=PALEBLUE, edgecolor="none", zorder=0))
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.6, ls=(0, (3, 2)),
            zorder=1)

    gx, gy = np.meshgrid(np.arange(K), np.arange(K), indexing="ij")
    m = p["mat"]
    nz = m > 0
    up, eq, dn = nz & (gy > gx), nz & (gy == gx), nz & (gy < gx)
    ax.scatter(gx[up], gy[up], s=area(m[up]), color=BLUE, alpha=0.9,
               linewidths=0, zorder=3)
    ax.scatter(gx[eq], gy[eq], s=area(m[eq]), color=MUTED, alpha=0.9,
               linewidths=0, zorder=3)
    ax.scatter(gx[dn], gy[dn], s=area(m[dn]), facecolor="white",
               edgecolor=INK, linewidths=0.6, zorder=3)

    # the average prompt: diagonal point -> mean point = paired mean increase
    mo, mt = p["ko"].mean(), p["kt"].mean()
    ax.annotate("", xy=(mo, mt), xytext=(mo, mo), zorder=6,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.1,
                                mutation_scale=7, shrinkA=0, shrinkB=0))
    ax.plot([mo], [mo], marker="o", ms=3.2, mfc="white", mec=INK, mew=0.9,
            zorder=7)

    # readout in the empty bottom-right corner of the fell half
    xr = N_SAMP + 0.4
    t1 = ax.text(xr, 0.6, f"[{p['ci'][0]:.2f}, {p['ci'][1]:.2f}]",
                 ha="right", va="bottom", fontsize=6.5, color=INK)
    ax.annotate(f"$+{p['est']:.2f}$", xy=(0, 0), xycoords=t1,
                xytext=(-2.5, -0.6), textcoords="offset points",
                ha="right", va="bottom", fontsize=9, color=BLUE)
    ax.text(xr, -0.45, f"{p['rose']:.0%} rose, {p['fell']:.0%} fell",
            ha="right", va="bottom", fontsize=6.5, color=MUTED)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ticks = [0, N_SAMP // 2, N_SAMP]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_yticklabels([str(t) for t in ticks] if j == 0 else [])
    ax.spines["left"].set_bounds(0, N_SAMP)
    ax.spines["bottom"].set_bounds(0, N_SAMP)
    ax.tick_params(length=2.5, pad=1.5)

    # marginals: control on top (gray), inserted on the right (blue)
    ho = np.bincount(p["ko"], minlength=K) / p["n"]
    ht = np.bincount(p["kt"], minlength=K) / p["n"]
    axt.bar(np.arange(K), ho, width=0.78, color=PALEGRAY, edgecolor=MUTED,
            linewidth=0.6)
    axr.barh(np.arange(K), ht, height=0.78, color=BLUE, alpha=0.85,
             linewidth=0)
    axt.set_xlim(lo, hi)
    axt.set_ylim(0, MARG_MAX * 1.05)
    axr.set_ylim(lo, hi)
    axr.set_xlim(0, MARG_MAX * 1.05)
    for a in (axt, axr):
        a.set_xticks([])
        a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
    axt.spines["bottom"].set_visible(True)
    axt.spines["bottom"].set_bounds(0, N_SAMP)
    axt.spines["bottom"].set_color(GRID)
    axr.spines["left"].set_visible(True)
    axr.spines["left"].set_bounds(0, N_SAMP)
    axr.spines["left"].set_color(GRID)

    # title and detection-gate strip
    ytitle = BOTTOM + S + MG + MW + 0.07
    fig.text((x0) / W, ytitle / H, p["name"], fontsize=8.5,
             fontweight="bold", ha="left", va="bottom", color=INK)
    fig.text((x0 + S + MG + MW) / W, ytitle / H,
             f"{p['n']} of {N_COHORT} pass gate", fontsize=7, ha="right",
             va="bottom", color=INK)
    sx0, sw = x0, S + MG + MW
    ys = (ytitle - 0.045) / H
    fig.lines.append(plt.Line2D([sx0 / W, (sx0 + sw) / W], [ys, ys],
                                transform=fig.transFigure, color=GRID,
                                lw=2.2, solid_capstyle="butt"))
    fig.lines.append(plt.Line2D(
        [sx0 / W, (sx0 + sw * p["n"] / N_COHORT) / W], [ys, ys],
        transform=fig.transFigure, color=ORANGE, lw=2.2,
        solid_capstyle="butt"))

    if j == 0:
        ax.set_ylabel(f"+1 harm word (refusals of {N_SAMP})", labelpad=2)
    ax.set_xlabel(f"control (refusals of {N_SAMP})", labelpad=1.5)

fig.savefig(OUT_DIR / "insertion-dist_v4.pdf")
fig.savefig(OUT_DIR / "insertion-dist_v4.png", dpi=220)
print("wrote", OUT_DIR / "insertion-dist_v4.pdf")
for p in panels:
    print(p["pos"], p["n"], f"{p['est']:+.3f}", p["ci"],
          f"rose {p['rose']:.2f} same {p['same']:.2f} fell {p['fell']:.2f}")
