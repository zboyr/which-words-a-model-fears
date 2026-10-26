"""Section 4 cross-fitting figure pair, v4 (one script, saved under both
fig/projection/code/ and fig/heatmap/code/).

projection_v4 (left minipage, 0.37 textwidth)
    Every held-out behaviour cell of the crossed design: its projection on
    the cross-fitted shared direction (horizontal) against its refusal
    change r_H - r_N (vertical), both residualized on prompt + word fixed
    effects. Faint dots are cells; the dark line and band are binned means
    (eight equal-count bins of the projection) with a 95% two-way
    (prompt x word) cluster-bootstrap interval. Cells whose *raw*
    projection is negative -- the few that point against the direction --
    are drawn as open rings, so the sign claim and the association claim
    sit in one picture. Rho, n and the positive fraction are read from the
    frozen result JSON; the per-cell scatter is recomputed from the feature
    shards exactly as fig56.py does and its rho is asserted to match.

heatmap_v4 (right minipage, 0.59 textwidth)
    Replaces the position x layer heatmaps by the quantity the prose talks
    about: the exact two-way decomposition of the harm-minus-neutral shift
    D = mu + A_p + B_w + I_pw, as shares of its squared norm over layers,
    at the word's own token (a) and at the boundary token </think> (b);
    then (c) the residualized rank correlation of the held-out projection
    with the refusal change at the word token and at both boundary tokens.
    Blue is the shared component / the direction throughout, orange the
    word-specific component.

Provenance (internal ids appear only here, never in the render):
  decomposition, rho by layer   data/results/x85_crossed_keyword_latent_allstates.json
  primary rho, sign fraction    data/results/x85_crossed_keyword_latent.json
  per-cell scatter              data/runs/x85/x85_cohort.json,
                                data/runs/x85/features/x85_hidden_shard_*.pt,
                                data/runs/x85/x85_refusal_{O,H,N}.jsonl
  helpers                       code/x85_analyze.py, code/x85_common.py

Run with a Python that has numpy, scipy, matplotlib >= 3.7 and torch
(the repo's helpers load the shards with torch):
    python3 direction_panels_v4.py [OUT_FIG_DIR]
OUT_FIG_DIR defaults to NAACL2027_ARR_Oct2026/fig; outputs
projection/projection_v4.{pdf,png} and heatmap/heatmap_v4.{pdf,png}.
"""

import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
FIG = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[1]
sys.path.insert(0, str(REPO / "code"))

from x85_analyze import (  # noqa: E402
    behavior_arrays,
    cell_index_grids,
    load_selected_features,
    unit,
)
from x85_common import PRIMARY_STATES, read_json, two_way_residual  # noqa: E402

BLUE = "#0072B2"
ORANGE = "#E69F00"
INK = "#333333"
MUTED = "#888888"
PALEBLUE = "#EAF3F9"
PALEGRAY = "#F4F4F4"
GRID = "#E3E3E3"
PROMPT_BLUE = "#9CC4E4"   # lighter tint of BLUE for the prompt component
INTER_GRAY = "#DCDCDC"    # interaction component

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
    "axes.grid": False,
    "pdf.fonttype": 42,
})

POSITION = "tbg_minus1"          # the boundary token quoted in the text
TEXTWIDTH_IN = 455.24 / 72.27    # ACL two-column \textwidth
HEIGHT_IN = 1.95                 # both panels share one height
N_BINS = 8
N_BOOT = 2000


# ------------------------------------------------------------------ data
def held_out_cells():
    """Per behaviour cell: raw and residualized projection, Delta refusal."""
    manifest = read_json(REPO / "data/runs/x85/x85_cohort.json")
    paths = [Path(p) for p in sorted(glob.glob(
        str(REPO / "data/runs/x85/features/x85_hidden_shard_*.pt")))]
    features, _ = load_selected_features(paths, manifest, list(PRIMARY_STATES), [POSITION])
    harm_grid, neutral_grid = cell_index_grids(manifest)
    prompt_folds = np.asarray([r["prompt_fold"] for r in manifest["prompts"]], dtype=int)
    word_folds = np.asarray([r["word_fold"] for r in manifest["words"]], dtype=int)
    behavior = behavior_arrays(manifest, REPO / "data/runs/x85")
    if behavior is None:
        raise SystemExit("FATAL: behavior artifacts incomplete")
    pidx, widx, y = behavior["prompt_index"], behavior["word_index"], behavior["delta"]

    scores = []
    for s in range(len(PRIMARY_STATES)):
        deltas = features[harm_grid, 0, s] - features[neutral_grid, 0, s]
        units = {
            (pf, wf): unit(deltas[np.ix_(prompt_folds != pf, word_folds != wf)].mean(axis=(0, 1)))
            for pf in range(4) for wf in range(4)
        }
        scores.append(np.asarray(
            [deltas[p, w] @ units[(prompt_folds[p], word_folds[w])] for p, w in zip(pidx, widx)]))
    x_raw = np.mean(scores, axis=0)
    return dict(
        x_raw=x_raw,
        x=two_way_residual(x_raw, pidx, widx),
        y=two_way_residual(y, pidx, widx),
        pidx=np.asarray(pidx), widx=np.asarray(widx),
    )


def binned_means(c, rng):
    """Equal-count bins of x; mean y per bin with a two-way cluster bootstrap."""
    edges = np.quantile(c["x"], np.linspace(0, 1, N_BINS + 1))
    b = np.clip(np.searchsorted(edges, c["x"], side="right") - 1, 0, N_BINS - 1)
    mx = np.array([c["x"][b == k].mean() for k in range(N_BINS)])
    my = np.array([c["y"][b == k].mean() for k in range(N_BINS)])
    up, uw = np.unique(c["pidx"]), np.unique(c["widx"])
    boot = np.full((N_BOOT, N_BINS), np.nan)
    for r in range(N_BOOT):
        wp = np.bincount(rng.choice(len(up), len(up)), minlength=len(up))
        ww = np.bincount(rng.choice(len(uw), len(uw)), minlength=len(uw))
        wt = wp[np.searchsorted(up, c["pidx"])] * ww[np.searchsorted(uw, c["widx"])]
        for k in range(N_BINS):
            m = b == k
            if wt[m].sum() > 0:
                boot[r, k] = np.average(c["y"][m], weights=wt[m])
    lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    return mx, my, lo, hi


# ------------------------------------------------------------ projection
def draw_projection():
    frozen = read_json(REPO / "data/results/x85_crossed_keyword_latent.json")
    rho_frozen = frozen["behavior_association_gate"][POSITION]["association"]["rho"]
    n_frozen = frozen["behavior_association_gate"][POSITION]["association"]["n"]
    pos_frac = frozen["primary_representation_gate"][POSITION]["positive_cell_fraction"]

    c = held_out_cells()
    rho = stats.spearmanr(c["x"], c["y"]).statistic
    if abs(rho - rho_frozen) > 5e-3 or len(c["y"]) != n_frozen:
        raise SystemExit(f"FATAL: recomputed rho {rho:.4f} / n {len(c['y'])} "
                         f"!= frozen {rho_frozen:.4f} / {n_frozen}")
    neg = c["x_raw"] < 0
    mx, my, lo, hi = binned_means(c, np.random.default_rng(56))

    width = 0.37 * TEXTWIDTH_IN
    fig = plt.figure(figsize=(width, HEIGHT_IN))
    ax = fig.add_axes([0.215, 0.20, 0.765, 0.70])
    ax.axhline(0, lw=0.5, color=MUTED, zorder=0)
    ax.axvline(0, lw=0.5, color=MUTED, zorder=0)
    ax.scatter(c["x"][~neg], c["y"][~neg], s=4, color=BLUE, alpha=0.28,
               linewidths=0, zorder=1, rasterized=True)
    ax.scatter(c["x"][neg], c["y"][neg], s=7, facecolors="white",
               edgecolors=INK, linewidths=0.55, zorder=2)
    ax.fill_between(mx, lo, hi, color=BLUE, alpha=0.22, linewidth=0, zorder=3)
    ax.plot(mx, my, "-o", color=BLUE, lw=1.4, ms=2.8, mec="white", mew=0.5, zorder=4)

    ax.set_xlabel(r"projection on held-out $\hat\mu$", labelpad=1.5)
    ax.set_ylabel(r"$\Delta$ refusal $r_H - r_N$", labelpad=1.5)
    ax.text(0.03, 0.98, rf"$\rho = {rho_frozen:.2f}$", transform=ax.transAxes,
            ha="left", va="top", fontsize=8, color=INK)
    ax.text(0.03, 0.865, f"{n_frozen} held-out cells,\nboth axes residualized",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.2, color=MUTED,
            linespacing=1.0, path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
    # key for the open rings, bottom right
    ax.scatter([0.625], [0.105], s=7, facecolors="white", edgecolors=INK,
               linewidths=0.55, transform=ax.transAxes, clip_on=False)
    ax.text(0.66, 0.105, f"raw projection\nbelow 0 ({1 - pos_frac:.0%})",
            transform=ax.transAxes, ha="left", va="center", fontsize=6.2,
            color=INK, linespacing=1.0)
    ax.set_yticks([-0.5, 0, 0.5])
    ax.set_yticklabels(["\u22120.5", "0", "0.5"])
    ax.tick_params(pad=1.5)
    out = FIG / "projection"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "projection_v4.pdf", dpi=600)
    fig.savefig(out / "projection_v4.png", dpi=300)
    plt.close(fig)
    print(f"projection: rho={rho:.4f} (frozen {rho_frozen:.4f}), n={len(c['y'])}, "
          f"raw<0 on behaviour cells {neg.mean():.3f}, frozen positive fraction {pos_frac:.3f}")


# -------------------------------------------------------------- layers
def draw_layers():
    js = read_json(REPO / "data/results/x85_crossed_keyword_latent_allstates.json")
    rep, beh = js["representation"], js["behavior"]["per_position_state"]
    states = sorted(int(s) for s in rep[POSITION])
    L = np.array(states)

    def shares(pos):
        f = [rep[pos][str(s)]["energy"]["fractions"] for s in states]
        return {k: np.array([r[k] for r in f]) for k in ("global", "prompt", "interaction", "word")}

    def rho(pos):
        return np.array([beh[pos][str(s)]["two_way_residualized"]["rho"] for s in states])

    band = js["config"]["primary_states_available"]
    sh_word, sh_bnd = shares("instruction_last"), shares(POSITION)
    cross = next(s for s, g, w in zip(states, sh_bnd["global"], sh_bnd["word"]) if g > w)

    width = 0.59 * TEXTWIDTH_IN
    fig = plt.figure(figsize=(width, HEIGHT_IN))
    bottom, h = 0.20, 0.62
    lefts, w = [0.085, 0.39, 0.722], 0.25
    axa, axb, axc = (fig.add_axes([l, bottom, w, h]) for l in lefts)

    order = [("global", BLUE, "shared $\\mu$"), ("prompt", PROMPT_BLUE, "prompt"),
             ("interaction", INTER_GRAY, "interaction"), ("word", ORANGE, "word")]
    for ax, sh, title in ((axa, sh_word, "word token"),
                          (axb, sh_bnd, "boundary token")):
        base = np.zeros_like(L, dtype=float)
        for key, col, _ in order:
            ax.fill_between(L, base, base + sh[key], color=col, lw=0, step=None)
            base = base + sh[key]
        ax.set_xlim(L[0], L[-1])
        ax.set_ylim(0, 1)
        ax.set_title(title, loc="left", fontsize=7.5, pad=3, color=INK)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
    axa.set_ylabel(r"share of $\|D\|^2$", labelpad=1.5)
    axa.set_yticks([0, 0.5, 1])
    axa.set_yticklabels(["0", ".5", "1"])
    axb.set_yticks([0, 0.5, 1])
    axb.set_yticklabels([])

    # in-area labels: word token panel
    def lab(ax, x, y, text, col="white", **kw):
        ax.text(x, y, text, ha="center", va="center", fontsize=6.5, color=col, **kw)

    i = states.index(24)
    lab(axa, 24, sh_word["global"][i] / 2, "shared", col="white")
    j = states.index(26)
    top_w = 1 - sh_word["word"][j] / 2
    lab(axa, 26, top_w, "word", col=INK)
    k = states.index(24)
    lab(axb, 24, sh_bnd["global"][k] / 2, "shared", col="white")
    lab(axb, 24, sh_bnd["global"][k] + sh_bnd["prompt"][k] / 2, "prompt", col=INK)
    m = states.index(26)
    lab(axb, 25.5, sh_bnd["global"][m] + sh_bnd["prompt"][m] + sh_bnd["interaction"][m] / 2 - 0.02,
        "interaction", col=INK)
    lab(axb, 5, 1 - sh_bnd["word"][states.index(5)] / 2, "word", col=INK)

    # where shared overtakes word at the boundary
    axb.axvline(cross, color="white", lw=0.8, ls=(0, (2, 1.5)))
    axb.text(cross + 1.0, 0.985, f"shared > word\nfrom layer {cross}", ha="left",
             va="top", fontsize=6.3, color=INK, linespacing=1.0)

    # (c) refusal association
    axc.axvspan(band[0] - 0.5, band[-1] + 0.5, color=PALEBLUE, lw=0, zorder=0)
    axc.text(band[0] - 0.9, 0.02, "read\nhere", ha="right", va="bottom",
             fontsize=5.8, color=BLUE, linespacing=0.95)
    axc.axhline(0, lw=0.5, color=MUTED, zorder=1)
    r_b, r_b2, r_w = rho(POSITION), rho("tbg"), rho("instruction_last")
    axc.plot(L, r_b2, color=BLUE, lw=0.8, ls=(0, (2.5, 1.5)), zorder=2)
    axc.plot(L, r_b, color=BLUE, lw=1.5, zorder=3)
    axc.plot(L, r_w, color=ORANGE, lw=1.5, zorder=3)
    axc.set_xlim(L[0], L[-1])
    axc.set_ylim(-0.05, 0.75)
    axc.set_yticks([0, 0.2, 0.4, 0.6])
    axc.set_yticklabels(["0", ".2", ".4", ".6"])
    axc.set_title(r"$\rho$ with $\Delta$ refusal", loc="left", fontsize=7.5, pad=3, color=INK)
    axc.text(23, 0.715, "boundary", color=BLUE, fontsize=6.5, ha="center", va="center")
    axc.text(24, r_w[states.index(24)] + 0.10, "word token", color="#B07800",
             fontsize=6.5, ha="center", va="bottom")
    axc.yaxis.grid(True, color=GRID, lw=0.4)
    axc.set_axisbelow(True)

    for ax in (axa, axb, axc):
        ax.set_xticks([1, 8, 16, 24, 32])
        ax.tick_params(pad=1.5)
    fig.text((lefts[0] + lefts[2] + w) / 2, 0.035, "layer", ha="center", va="bottom",
             fontsize=8, color=INK)
    out = FIG / "heatmap"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "heatmap_v4.pdf")
    fig.savefig(out / "heatmap_v4.png", dpi=300)
    plt.close(fig)
    print(f"layers: shared > word at the boundary from layer {cross}; "
          f"word-token word share {sh_word['word'].min():.2f}-{sh_word['word'].max():.2f} "
          f"(band {[round(sh_word['word'][states.index(s)], 2) for s in band]})")


if __name__ == "__main__":
    draw_projection()
    draw_layers()
