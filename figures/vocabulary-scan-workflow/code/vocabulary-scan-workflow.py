"""Direction-and-vocabulary workflow figure, one page-wide composite.

Row A, the crossed design and the direction:
  the 96 x 24 crossed grid (fit cells, the evaluated unseen x unseen
  block, one highlighted cell); that cell's two forward passes, read at
  the tbg position, and the difference vector D_pw = h_H - h_N; the fan
  of all cell differences and their common direction mu, the refusal
  direction.
Row B, the direction as a ruler:
  a candidate word appended to an over-refused Hard-1K prompt, read at
  the tbg position, its shift projected on the unit direction; and the
  real Qwen3.5-4B table, the distribution of the 71,583 word shifts over
  the 48 over-refused (BP) prompts of x110's scan (the table x111-x113
  use; colours and labels as the former vocab-distribution figure).

Style follows the insertion-workflow figure: Arial, light-blue token
strips with dots, teal "Target LLM" robots, blue / orange / red / teal
accents, bold section headers.

Real inputs (never typed in):
  data/runs/x109/scan.json (candidate list, sd unit), badwords_distribution.json
  data/runs/x110/scan.json, scan_shift.npz (the BP rows: the Qwen3.5-4B table)
  fig/crossed-grid/code/crossed_grid_v4_cells.json (example cell, x85)
  code/x83_common.lexicon_words (the 99-word insertion lexicon)
The vector geometry in the schematic panels is illustrative, not a
projection of real activations.

Writes ../vocabulary-scan-workflow.{pdf,svg,png} and the audit files.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE.parent
DRAFT = HERE.parents[2]
ROOT = HERE.parents[3]
RUNS = ROOT / "data" / "runs" / "x109"
sys.path.insert(0, str(ROOT / "code"))
def require_matplotlib_panel_alignment(*args, **kwargs):  # layout audit helper, not shipped
    return None

from x83_common import lexicon_words  # noqa: E402

# ------------------------------------------------------------------ data ---
scan = json.load(open(RUNS / "scan.json"))
words = scan["candidates"]["words"]
SD = scan["sd_unit"]["value"]
X110 = ROOT / "data" / "runs" / "x110"
x110 = json.load(open(X110 / "scan.json"))
assert x110["candidates"]["words_sha256"] == scan["candidates"]["words_sha256"]
pop = dict(zip(x110["prompt_ids"], x110["prompt_populations"]))
bp_rows = np.array([pop[i] == "BP" for i in x110["prompt_ids_scanned"]])
shift = np.load(X110 / "scan_shift.npz")["shift"][bp_rows]  # [BP prompt, word, token]
N_PROMPTS, N_WORDS = shift.shape[0], len(words)
assert (N_PROMPTS, N_WORDS) == (48, 71583)
table = shift.astype(np.float64).sum(axis=2).mean(axis=0) / SD
_x111 = np.load(ROOT / "data" / "runs" / "x111" / "tables.npz")["BP"]
assert np.abs(table - _x111).max() < 1e-5                  # the x111-x113 BP table
idx = {w: i for i, w in enumerate(words)}
bad = json.load(open(RUNS / "badwords_distribution.json"))
bad_ix = np.array([idx[r["word"]] for r in bad["cmu_all"] if r["word"] in idx])
lex_ix = np.array([idx[w] for w, _ in lexicon_words() if w in idx])

cells = json.load(open(DRAFT / "fig" / "crossed-grid" / "code" / "crossed_grid_v4_cells.json"))
ex = cells["example"]
EX_HARM, EX_NEUT = ex["harm"], ex["neutral"]
EX_RH, EX_RN = ex["behavior"]["r_H"], ex["behavior"]["r_N"]
POS_FRAC = cells["all_cells_positive_fraction"]
N_P, N_W = len(cells["prompt_folds"]), len(cells["word_folds"])
assert (N_P, N_W) == (96, 24)

# --------------------------------------------------------------- styling ---
INK, MUTED, GRID = "#243449", "#647487", "#C9D2DB"
BLUE, ORANGE, RED, TEAL = "#2675BF", "#D9902F", "#C85155", "#258A87"
LIGHT, TOKFILL, TOKEDGE, DOT = "#9DBBD8", "#E6EFF8", "#90B8DD", "#7FA7D1"
LLMFILL, PALEGRAY, PALEBLUE = "#E1F2F0", "#EEF1F4", "#DCE9F5"
BAD_RED, LEX_NAVY = "#D62728", "#1F3B8A"            # vocab-distribution colours
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6, "text.color": INK, "pdf.fonttype": 42, "svg.fonttype": "none",
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold", "axes.linewidth": 0.6, "axes.edgecolor": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.labelsize": 6, "legend.fontsize": 5.2,
})

FIG_W, FIG_H = 6.5, 4.55
ROW_L, ROW_W, ROW_H = 0.2, 6.1, 1.88
ROW_B = {"A": 2.45, "B": 0.38}
XL = 100.0
YL = XL * ROW_H / ROW_W                                      # 30.8 units

fig = plt.figure(figsize=(FIG_W, FIG_H))
rows = {}
for key, b in ROW_B.items():
    ax = fig.add_axes([ROW_L / FIG_W, b / FIG_H, ROW_W / FIG_W, ROW_H / FIG_H])
    ax.set_xlim(0, XL)
    ax.set_ylim(0, YL)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    rows[key] = ax


def txt(ax, x, y, s, size=6, color=INK, ha="left", va="center", **kw):
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, **kw)


def arrow(ax, a, b, color, lw=0.8, ms=5, ls="-", z=4, alpha=1.0, style="-|>"):
    p = FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=ms, lw=lw, color=color, ls=ls,
                        zorder=z, alpha=alpha, shrinkA=0, shrinkB=0)
    ax.add_patch(p)
    return p


def rbox(ax, x, y, w, h, fc="white", ec=GRID, lw=0.6, z=2, r=0.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z))


def robot(ax, x, yc, w=5.2, h=4.2):
    """A Target-LLM robot in the insertion-workflow style."""
    y = yc - h / 2
    rbox(ax, x, y, w, h, fc=LLMFILL, ec=TEAL, lw=0.7, r=0.6, z=3)
    ax.plot([x + w / 2, x + w / 2], [y + h, y + h + 0.7], color=TEAL, lw=0.6, zorder=3)
    ax.add_patch(Circle((x + w / 2, y + h + 0.85), 0.28, fc=TEAL, ec="none", zorder=4))
    for dx in (-0.9, 0.9):
        ax.add_patch(Circle((x + w / 2 + dx, y + h - 1.0), 0.3, fc=BLUE, ec="none", zorder=4))
    txt(ax, x + w / 2, y + 1.15, "Target\nLLM", 5, TEAL, ha="center", linespacing=1.2, weight="bold")


def strip(ax, x, yc, parts, h=2.3):
    """Token strip: parts are ("dots", n) | ("word", text, fc, ec, color) | ("tbg",).
    Returns (x_end, tbg box extent)."""
    cur = x + 0.6
    segs, tbg_ext = [], None
    for part in parts:
        if part[0] == "dots":
            n = part[1]
            segs.append(("dots", cur, n))
            cur += 1.15 * n + 0.3
        elif part[0] == "word":
            _, s, fc, ec, col = part
            w = 0.64 * len(s) + 1.3                            # ~2.8 pt per bold char at 5.2 pt
            segs.append(("word", cur, w, s, fc, ec, col))
            cur += w + 0.5
        else:
            segs.append(("tbg", cur))
            tbg_ext = (cur, cur + 10.6)
            cur += 10.6 + 0.3
    end = cur + 0.3
    rbox(ax, x, yc - h / 2, end - x, h, fc=TOKFILL, ec=TOKEDGE, lw=0.6, r=0.6, z=2)
    for seg in segs:
        if seg[0] == "dots":
            for k in range(seg[2]):
                ax.add_patch(Circle((seg[1] + 0.55 + 1.15 * k, yc), 0.32, fc=DOT, ec="none", zorder=3))
        elif seg[0] == "word":
            _, sx, w, s, fc, ec, col = seg
            rbox(ax, sx, yc - 0.85, w, 1.7, fc=fc, ec=ec, lw=0.6, r=0.35, z=3)
            txt(ax, sx + w / 2, yc, s, 5.2, col, ha="center", weight="bold")
        else:
            sx = seg[1]
            rbox(ax, sx, yc - 0.85, 6.6, 1.7, fc=PALEBLUE, ec=BLUE, lw=0.6, r=0.35, z=3)
            txt(ax, sx + 3.3, yc, "</think>", 5, INK, ha="center", family="monospace")
            rbox(ax, sx + 6.9, yc - 0.85, 3.7, 1.7, fc=PALEBLUE, ec=BLUE, lw=0.6, r=0.35, z=3)
            txt(ax, sx + 8.75, yc, r"\n\n", 5, INK, ha="center", family="monospace")
    return end, tbg_ext


def tbg_bracket(ax, ext, y):
    x0, x1 = ext
    ax.plot([x0, x0, x1, x1], [y, y + 0.45, y + 0.45, y], color=BLUE, lw=0.5, zorder=3)
    txt(ax, (x0 + x1) / 2, y + 0.75, "tbg position", 5.2, BLUE, ha="center", va="bottom")


def header(ax, letter, title):
    txt(ax, 0, YL - 0.2, f"{letter}  {title}", 7.2, INK, va="top", weight="bold")
    ax.plot([0, XL], [YL - 2.3, YL - 2.3], color=GRID, lw=0.6)


# ============================================================ row A ======
A = rows["A"]
header(A, "A", "Fit the direction on the crossed design")

# ---- the crossed grid ----
txt(A, 4.4, YL - 2.9, f"{N_W} harm / neutral pairs", 5.2, MUTED, va="top")
GX, GY, CS = 4.4, 24.0, 2.4                                   # top-left corner, cell size
for r in range(6):
    for c in range(6):
        fc = PALEBLUE if (r >= 4 and c >= 4) else PALEGRAY
        A.add_patch(Rectangle((GX + c * CS, GY - (r + 1) * CS), CS, CS, fc=fc, ec="white", lw=0.5, zorder=2))
A.add_patch(Rectangle((GX, GY - 6 * CS), 6 * CS, 6 * CS, fc="none", ec=MUTED, lw=0.5, zorder=3))
A.add_patch(Rectangle((GX + 4 * CS, GY - 6 * CS), 2 * CS, 2 * CS, fc="none", ec=BLUE, lw=0.9, zorder=4))
A.add_patch(Rectangle((GX + CS, GY - 2 * CS), CS, CS, fc="#FBEEDC", ec=ORANGE, lw=0.9, zorder=4))
for r in (0, 1, 2, 4, 5):
    if r != 1:                                   # row 2 carries the dashed link to the worked cell
        txt(A, GX + 3.5 * CS, GY - (r + 0.5) * CS, "···", 5.5, MUTED, ha="center")
    txt(A, GX + (r + 0.5) * CS, GY - 3.5 * CS, "···", 5.5, MUTED, ha="center", rotation=90)
txt(A, GX + 3.5 * CS, GY - 3.5 * CS, "···", 5.5, MUTED, ha="center", rotation=-45)
for c, s in ((0, "$w_1$"), (1, "$w_2$"), (2, "$w_3$"), (5, "$w_{24}$")):
    txt(A, GX + (c + 0.5) * CS, GY + 0.35, s, 7.2, INK, ha="center", va="bottom")
for r, s in ((0, "$p_1$"), (1, "$p_2$"), (2, "$p_3$"), (5, "$p_{96}$")):
    txt(A, GX - 0.4, GY - (r + 0.5) * CS, s, 7.2, INK, ha="right")
txt(A, 0.5, GY - 3 * CS, f"{N_P} prompts", 5.2, MUTED, ha="center", rotation=90)
ly = GY - 6 * CS - 1.1
A.add_patch(Rectangle((GX, ly - 1.1), 1.3, 1.1, fc=PALEGRAY, ec=MUTED, lw=0.4, zorder=3))
txt(A, GX + 1.8, ly - 0.55, "fit: μ = mean of D over train folds", 5.2, INK)
A.add_patch(Rectangle((GX, ly - 3.2), 1.3, 1.1, fc=PALEBLUE, ec=BLUE, lw=0.6, zorder=3))
txt(A, GX + 1.8, ly - 2.65, "evaluate: unseen prompts × unseen words", 5.2, INK)
txt(A, GX, ly - 4.9, f"{N_P * N_W:,} cells, one vector D each", 5.2, MUTED)

# dashed link from the highlighted cell to the worked cell
cell_c = (GX + 2 * CS, GY - 1.5 * CS)
assert abs(cell_c[1] - 20.4) < 1e-9
A.plot([cell_c[0], 20.4], [cell_c[1], cell_c[1]], ls=(0, (2, 1.5)), color=ORANGE, lw=0.6, zorder=5)

# ---- one cell: two forward passes ----
txt(A, 23.2, YL - 2.9, "one cell (p, w): two forward passes, read at the tbg position", 5.2, MUTED, va="top")
YH, YN = 20.4, 14.9
for lab, yc, word, fc, ec, col, rr in (("H", YH, EX_HARM, "#F6DADB", RED, RED, EX_RH),
                                        ("N", YN, EX_NEUT, "#DCEFEE", TEAL, TEAL, EX_RN)):
    txt(A, 21.4, yc, lab, 6.5, col, weight="bold")
    end, ext = strip(A, 23.2, yc, [("dots", 5), ("word", word, fc, ec, col), ("tbg",)])
    if lab == "H":
        tbg_bracket(A, ext, yc + 1.35)
    txt(A, ext[0] - 0.6, yc - 1.55, f"refusal rate {rr:.1f}", 5, MUTED, ha="right", va="top")
    rx = end + 1.4
    arrow(A, (end, yc), (rx, yc), GRID, 0.6, 4.5)
    robot(A, rx, yc)
    arrow(A, (rx + 5.2, yc), (rx + 6.6, yc), GRID, 0.6, 4.5)
    txt(A, rx + 6.9, yc, r"$h_{\mathrm{%s}}$" % lab, 7.2, col)
# vector sketch of the difference
O1 = np.array([27.0, 2.6])
hN = O1 + np.array([9.0, 3.0])
hH = O1 + np.array([10.5, 8.6])
arrow(A, O1, hN, TEAL, 1.0, 6)
arrow(A, O1, hH, RED, 1.0, 6)
arrow(A, hN, hH, ORANGE, 1.5, 7, z=5)
A.plot(*O1, "o", ms=2, color=INK, zorder=6)
txt(A, hN[0] + 0.5, hN[1] - 0.2, r"$h_{\mathrm{N}}$", 7.2, TEAL, va="top")
txt(A, hH[0] + 0.5, hH[1] + 0.1, r"$h_{\mathrm{H}}$", 7.2, RED)
txt(A, 37.5, 8.2, r"$D_{pw}=h_{\mathrm{H}}-h_{\mathrm{N}}$", 7.2, ORANGE)
txt(A, 50.5, 11.6, "the prompt and the act of\nappending cancel:\none vector per cell", 5.2, MUTED, va="top",
    linespacing=1.15)
txt(A, 41.0, 3.2, r"$D_{pw}=\mu+A_p+B_w+I_{pw}$", 7.2, INK)
arrow(A, (63.3, (YH + YN) / 2), (65.3, (YH + YN) / 2), MUTED, 0.8, 6)

# ---- the common direction ----
txt(A, 66.5, YL - 2.9, f"the {N_P * N_W:,} cells point the same way", 5.2, MUTED, va="top")
O2 = np.array([68.0, 2.8])
# schematic only: a fixed-seed fan of illustrative arrows, not data
rng = np.random.default_rng(7)
angles = np.concatenate([rng.normal(56, 7, 11), [27, 80]])
lengths = np.concatenate([rng.uniform(10.5, 14.5, 11), [7.0, 7.5]])
vecs = np.stack([lengths * np.cos(np.radians(angles)), lengths * np.sin(np.radians(angles))], 1)
for v in vecs:
    arrow(A, O2, O2 + v, LIGHT, 0.6, 4.5, z=3)
mu = vecs.mean(0)
arrow(A, O2, O2 + mu, ORANGE, 1.9, 8, z=6)
A.plot(*O2, "o", ms=2, color=INK, zorder=6)
lab = O2 + mu + 3.4 * mu / np.linalg.norm(mu)
txt(A, lab[0], lab[1], r"$\mu$", 8, ORANGE, ha="center", weight="bold")
txt(A, 100, 24.6, r"$D_{pw}$: one arrow per cell", 7.2, BLUE, ha="right", va="top")
txt(A, 100, 22.2, "μ holds 0.30 of ‖D‖², all word\neffects 0.15; per-word means\ncos 0.82–0.96 with μ (21/24)", 5.2, INK,
    ha="right", va="top", linespacing=1.15)
txt(A, 100, 15.4, f"μ = mean of D over train folds;\nD · μ̂ > 0 on {POS_FRAC:.0%} of the\nunseen prompt × word cells",
    5.2, ORANGE, ha="right", va="top", linespacing=1.15)
txt(A, 100, 8.6, "μ̂ = μ / ‖μ‖ is the refusal direction:\nwriting ±μ moves refusal both\nways, monotonically with dose",
    5.2, ORANGE, ha="right", va="top", linespacing=1.15, weight="bold")

# ============================================================ row B ======
B = rows["B"]
header(B, "B", "Use the direction as a ruler over the whole vocabulary")
txt(B, 0, YL - 2.9, f"append each of {N_WORDS:,} words to {N_PROMPTS} over-refused Hard-1K prompts the model refuses",
    5.2, MUTED, va="top")
for yc, parts, lab in ((YH, [("dots", 6), ("word", "w", "#E8ECF0", INK, INK), ("tbg",)], "h(q + w)"),
                       (YN, [("dots", 6), ("tbg",)], "h(q)")):
    end, ext = strip(B, 0.6, yc, parts)
    if yc == YH:
        tbg_bracket(B, ext, yc + 1.35)
    rx = end + 1.4
    arrow(B, (end, yc), (rx, yc), GRID, 0.6, 4.5)
    robot(B, rx, yc)
    arrow(B, (rx + 5.2, yc), (rx + 6.6, yc), GRID, 0.6, 4.5)
    txt(B, rx + 6.9, yc, lab, 6.2, INK)
txt(B, 1.2, YH - 1.55, "over-refused prompt q", 5, MUTED, va="top")
txt(B, 1.2, YN - 1.55, "the same prompt, no word", 5, MUTED, va="top")
txt(B, 35.0, (YH + YN) / 2, "Δh(w) = h(q+w) − h(q)", 6.2, INK)
# ruler sketch
O3 = np.array([8.0, 3.0])
u = np.array([np.cos(np.radians(24)), np.sin(np.radians(24))])
arrow(B, O3 - 4.5 * u, O3 + 16.0 * u, ORANGE, 1.3, 7, z=5)
tipu = O3 + 16.0 * u
txt(B, tipu[0] - 0.2, tipu[1] + 1.1, r"$\hat\mu$", 8, ORANGE, ha="right")
B.plot(*O3, "o", ms=2, color=INK, zorder=6)
for name, v, col, sgn in (("illegal", np.array([5.0, 8.0]), RED, 1), ("poetic", np.array([-5.5, 2.0]), BLUE, -1)):
    tip = O3 + v
    arrow(B, O3, tip, col, 1.0, 6, z=6)
    proj = O3 + np.dot(v, u) * u
    B.plot([tip[0], proj[0]], [tip[1], proj[1]], ls=(0, (2, 1.5)), color=col, lw=0.6, zorder=5)
    B.plot(*proj, "o", ms=2.6, color=col, zorder=7)
    if sgn > 0:
        txt(B, tip[0] + 0.6, tip[1] + 0.1, f"Δh({name})", 5.4, col)
        txt(B, proj[0] + 0.9, proj[1] - 0.7, "+", 7, col, weight="bold")
    else:
        txt(B, tip[0] - 0.3, tip[1] + 0.7, f"Δh({name})", 5.4, col, va="bottom")
        txt(B, proj[0] - 1.2, proj[1], "−", 7, col, ha="center", weight="bold")
txt(B, 29.5, 12.6, r"$s(w)=\overline{\langle\Delta h(w),\hat\mu\rangle}\,/\,\mathrm{sd}$", 6.5, INK, va="top")
txt(B, 29.5, 9.6, f"signed component along μ̂, mean over\nthe {N_PROMPTS} prompts, in Hard-1K sd units;\ntwo disjoint"
                  " 24-prompt halves give\nthe word order at ρ = 0.95", 5.2, MUTED, va="top", linespacing=1.15)
txt(B, 29.5, 3.2, "7 other models: own direction, own\ntable, ρ = 0.36–0.76 with Qwen3.5-4B", 5.2, MUTED, va="top",
    linespacing=1.15)
arrow(B, (51.8, (YH + YN) / 2), (53.8, (YH + YN) / 2), MUTED, 0.8, 6)
txt(B, 56.0, YL - 2.9, "the Qwen3.5-4B table: mean tbg shift of each word", 5.2, MUTED, va="top")

# ---- the real table: the vocab-distribution panel, redrawn in place ----
hx = ROW_L + 0.59 * ROW_W + 0.28
hist = fig.add_axes([hx / FIG_W, (ROW_B["B"] + 0.33) / FIG_H, (ROW_L + ROW_W - hx) / FIG_W, 1.22 / FIG_H])
bins = np.linspace(-4.0, 1.2, 105)
hist.hist(table, bins=bins, color="#9AA7A8", label=f"all {N_WORDS:,} words")
hist.hist(table[bad_ix], bins=bins, color=BAD_RED, alpha=0.95,
          label=f"CMU bad-words list ({len(bad_ix)} matched of {bad['n_unique']:,})")
lex_cnt, _ = np.histogram(table[lex_ix], bins=bins)
hist.stairs(np.maximum(lex_cnt, 0.6), bins, baseline=None, color=LEX_NAVY, lw=1.0,
            label=f"insertion lexicon ({len(lex_ix)})")   # empty bins sit on the axis floor (ylim 0.6)
hist.set_yscale("log")
hist.set_ylim(0.6, 3e5)
hist.set_yticks([1, 10, 100, 1e3, 1e4, 1e5], ["1", "10", "100", "1k", "10k", "100k"])
hist.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
hist.set_xlabel("mean tbg shift of the appended word (sd)", labelpad=2.5)
hist.set_ylabel("words", labelpad=1.5)
hist.spines[["top", "right"]].set_visible(False)
hist.tick_params(length=2, pad=1.5)
hist.legend(loc="upper right", frameon=False, handlelength=1.2, borderaxespad=0.1, fontsize=5.2, labelspacing=0.25)
for w, dx, y in (("poetic", 0.0, 2.2), ("whimsical", 0.0, 2.2), ("bomb", -0.17, 1500), ("gun", 0.0, 1500),
                 ("hack", 0.0, 1500)):
    x = float(table[idx[w]])
    hist.annotate(w, xy=(x, 0.9 if y < 100 else 300), xytext=(x + dx, y), rotation=90, ha="center", va="bottom",
                  fontsize=5.4, color=INK, arrowprops={"arrowstyle": "-", "lw": 0.4, "color": MUTED})
BAD_MARKS = ("hook", "quickie", "shit", "fuck", "abortion", "suicide", "porn", "masturbate", "niggers")
LABEL_GAP, LABEL_XMAX = 0.16, 1.12                  # min label spacing (sd); keep labels inside the axis
marks = sorted((float(table[idx[w]]), w) for w in BAD_MARKS)
xls = []
for x, _ in marks:                                  # forward pass: push right to the minimum gap
    xls.append(max(x, xls[-1] + LABEL_GAP) if xls else x)
for k in range(len(xls) - 1, -1, -1):               # backward pass: pull left of the axis edge / next label
    xls[k] = min(xls[k], LABEL_XMAX if k == len(xls) - 1 else xls[k + 1] - LABEL_GAP)
placed = [(xl, x, w) for xl, (x, w) in zip(xls, marks)]
for k, (xl, x, w) in enumerate(placed):
    hist.annotate(w, xy=(x, 0.9), xytext=(xl, 1.15 if k % 2 == 0 else 3.2), rotation=90, ha="center", va="bottom",
                  fontsize=5.2, color="#7A0000",
                  path_effects=[patheffects.withStroke(linewidth=1.6, foreground="white")],
                  arrowprops={"arrowstyle": "-", "lw": 0.4, "color": "#7A0000"} if abs(xl - x) > 1e-6 else None)

fig.text(ROW_L / FIG_W, 0.13 / FIG_H,
         "Vector panels are 2-D schematics; arrows are illustrative. The histogram is the real table: the shift of each "
         "word along μ̂, summed over the two tbg tokens, mean over the 48 prompts.", fontsize=5.2, color=MUTED,
         va="center")

# ---------------------------------------------------------------- export ---
OUT = FIG / "vocabulary-scan-workflow"
fig.canvas.draw()
require_matplotlib_panel_alignment(
    fig, json_out=f"{OUT}.alignment.json", tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
    axes=[rows["A"], rows["B"], hist], panel_ids=["A", "B", "hist"], column_groups=[["A", "B"]],
    exemptions=[{"panels": ["hist"], "checks": ["all"],
                 "reason": "the histogram is a quantitative inset inside row B's schematic canvas, not a grid panel"}])
fig.savefig(f"{OUT}.pdf")
fig.savefig(f"{OUT}.svg")
fig.savefig(f"{OUT}.png", dpi=300)
print("wrote", OUT.with_suffix(".pdf"))
