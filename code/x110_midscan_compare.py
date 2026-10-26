"""x110 side analysis: the vocabulary table on ten mid-refusal prompts against the benign table.

Ten of the 132 x100 prompts covered by the x110 vocabulary scan, chosen as the ones whose
x100 `refusal2` rate r(O) is nearest 0.5 (tie-break: historical rate nearest 0.5, BP prompts
counted as 0.5; then position in prompt_ids). Their per-word mean shift (sum over the two
boundary tokens, x106 Hard-1K sd units) is the "mid-10 table"; it is compared with the x109
benign table (40 prompts), x110's own HP (84) and BP (48) tables, and the HP table with the
chosen HP prompts removed, with the statistics of x109_scan_wordtable_checks and
x110_analyze. Local, numpy/scipy/matplotlib only; writes data/runs/x110/midscan/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x110_common import (  # noqa: E402
    PROMPTS,
    RESULTS,
    SCAN_JSON,
    SCAN_NPZ,
    WORDS,
    X106_HARD1K_SD,
    X109_SCAN_JSON,
    X109_SCAN_NPZ,
    X109_WORDFREQ,
    atomic_json,
    read_json,
)
from x83_common import lexicon_words  # noqa: E402

OUT = SCAN_NPZ.parent / "midscan"
N_MID = 10
TOP_K = 1000
NAMED_LEXICON = ("shoot", "meth", "hack", "gun", "bomb")
SD = X106_HARD1K_SD


def spearman(a, b) -> float:
    return float(stats.spearmanr(a, b).statistic)


def pearson(a, b) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def rank_from_top(v: np.ndarray) -> np.ndarray:
    """1 = largest value."""
    order = np.argsort(-v, kind="stable")
    r = np.empty(len(v), int)
    r[order] = np.arange(1, len(v) + 1)
    return r


def distribution(v_raw: np.ndarray) -> dict:
    v = v_raw / SD
    q = np.quantile(v, [0.01, 0.10, 0.50, 0.90, 0.99, 0.999])
    return {
        "p1": float(q[0]),
        "p10": float(q[1]),
        "median": float(q[2]),
        "p90": float(q[3]),
        "p99": float(q[4]),
        "p99_9": float(q[5]),
        "max": float(v.max()),
        "min": float(v.min()),
        "mean": float(v.mean()),
        "sd": float(v.std()),
        "skew": float(stats.skew(v)),
        "excess_kurtosis": float(stats.kurtosis(v)),
        "frac_below_zero": float((v < 0).mean()),
        "n_above_1sd": int((v > 1).sum()),
        "n_above_2sd": int((v > 2).sum()),
    }


def compare(ref: np.ndarray, here: np.ndarray, words: list[str]) -> dict:
    """`here` against a reference table; both raw per-word means over the same word list."""
    r_top, h_top = np.argsort(-ref, kind="stable"), np.argsort(-here, kind="stable")
    ref_top, ref_bot = r_top[:TOP_K], r_top[::-1][:TOP_K]
    return {
        "spearman_all_words": spearman(ref, here),
        "pearson_all_words": pearson(ref, here),
        "top100_overlap": int(len(set(r_top[:100].tolist()) & set(h_top[:100].tolist()))),
        "bottom100_overlap": int(len(set(r_top[::-1][:100].tolist()) & set(h_top[::-1][:100].tolist()))),
        "top1000_overlap": int(len(set(r_top[:TOP_K].tolist()) & set(h_top[:TOP_K].tolist()))),
        "bottom1000_overlap": int(len(set(r_top[::-1][:TOP_K].tolist()) & set(h_top[::-1][:TOP_K].tolist()))),
        "spearman_within_ref_top1000": spearman(ref[ref_top], here[ref_top]),
        "spearman_within_ref_bottom1000": spearman(ref[ref_bot], here[ref_bot]),
        "pearson_within_ref_top1000": pearson(ref[ref_top], here[ref_top]),
        "pearson_within_ref_bottom1000": pearson(ref[ref_bot], here[ref_bot]),
        "ols_slope_here_on_ref": float(np.polyfit(ref, here, 1)[0]),
        "mean_ref_sd": float(ref.mean() / SD),
        "mean_here_sd": float(here.mean() / SD),
    }


def zscore_rows(m: np.ndarray) -> np.ndarray:
    return (m - m.mean(1, keepdims=True)) / m.std(1, keepdims=True)


def decompose(m: np.ndarray) -> dict:
    """x109_scan_wordtable_checks.decompose: two-way additive SS shares plus the rank-1 share."""
    P, W = m.shape
    g = m.mean()
    rows, cols = m.mean(1), m.mean(0)
    ss_total = float(((m - g) ** 2).sum())
    ss_prompt = float(W * ((rows - g) ** 2).sum())
    ss_word = float(P * ((cols - g) ** 2).sum())
    ss_resid = ss_total - ss_prompt - ss_word
    s = np.linalg.svd(m, compute_uv=False)
    ss_unc = float((m ** 2).sum())
    return {
        "shape": [int(P), int(W)],
        "share_prompt_main": ss_prompt / ss_total,
        "share_word_main": ss_word / ss_total,
        "share_residual_interaction": ss_resid / ss_total,
        "rank1_share_of_uncentred_ss": float(s[0] ** 2 / ss_unc),
        "additive_share_of_uncentred_ss": float(1 - ss_resid / ss_unc),
    }


def word_groups(mean_raw: np.ndarray, words: list[str], cidx: dict, lex: list[str], style: list[str], twins: list[str]) -> dict:
    N = len(words)
    rt = rank_from_top(mean_raw)
    pct = 100.0 * (N - rt) / (N - 1)  # x109_scan_badwords_hist: 100 = top
    li = np.asarray([cidx[w] for w in lex])
    out = {
        "x83_lexicon": {
            "n": int(len(lex)),
            "mean_sd": float(mean_raw[li].mean() / SD),
            "median_sd": float(np.median(mean_raw[li]) / SD),
            "n_in_top_1pct": int((pct[li] >= 99).sum()),
            "n_in_top_10pct": int((pct[li] >= 90).sum()),
            "median_percentile": float(np.median(pct[li])),
            "named": {w: {"sd": float(mean_raw[cidx[w]] / SD), "percentile": float(pct[cidx[w]]), "rank_from_top": int(rt[cidx[w]])} for w in NAMED_LEXICON},
            "top5": [{"word": lex[k], "sd": float(mean_raw[li[k]] / SD), "percentile": float(pct[li[k]])} for k in np.argsort(-mean_raw[li])[:5]],
            "bottom5": [{"word": lex[k], "sd": float(mean_raw[li[k]] / SD), "percentile": float(pct[li[k]])} for k in np.argsort(mean_raw[li])[:5]],
        },
        "x110_style_words": [
            {"word": w, "sd": float(mean_raw[cidx[w]] / SD), "rank_from_bottom": int(N + 1 - rt[cidx[w]]), "twin": t, "twin_sd": float(mean_raw[cidx[t]] / SD), "twin_rank_from_bottom": int(N + 1 - rt[cidx[t]])}
            for w, t in zip(style, twins)
        ],
    }
    out["x110_style_words_mean_sd"] = float(np.mean([r["sd"] for r in out["x110_style_words"]]))
    out["x110_twins_mean_sd"] = float(np.mean([r["twin_sd"] for r in out["x110_style_words"]]))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prompts = {p["id"]: p for p in read_json(PROMPTS)["prompts"]}
    scan = read_json(SCAN_JSON)
    x109 = read_json(X109_SCAN_JSON)
    res = read_json(RESULTS)
    words_obj = read_json(WORDS)
    style = [c["word"] for c in words_obj["chosen"]]
    twins = [c["twin"] for c in words_obj["chosen"]]
    words = scan["candidates"]["words"]
    if words != x109["candidates"]["words"] or scan["candidates"]["words_sha256"] != x109["candidates"]["words_sha256"]:
        raise SystemExit("FATAL: x110 and x109 candidate word lists differ")
    N = len(words)
    cidx = {w: j for j, w in enumerate(words)}
    lex = [w for w, _ in lexicon_words()]
    wf = set(read_json(X109_WORDFREQ)["words"])
    n_tok = np.asarray(scan["candidates"]["n_tokens"], int)

    # Tables (raw units; divide by SD for display).
    benign_m = np.load(X109_SCAN_NPZ)["shift"].astype(np.float64).sum(axis=2)  # [40, N]
    benign = benign_m.mean(0)
    S = np.load(SCAN_NPZ)["shift"].astype(np.float64).sum(axis=2)  # [132, N]
    o_proj = np.asarray(scan["o_proj_cached"], float)
    scanned_ids = scan["prompt_ids_scanned"]
    pops = np.asarray([prompts[i]["population"] for i in scanned_ids])

    # Prompt selection.
    rm = res["rate_matrix"]
    rO = dict(zip(rm["prompt_ids"], rm["r_O_x100"]))
    pos = {pid: k for k, pid in enumerate(rm["prompt_ids"])}

    def key(pid: str):
        h = prompts[pid].get("historical_refusal_rate")
        h = 0.5 if h is None else float(h)
        return (abs(rO[pid] - 0.5), abs(h - 0.5), pos[pid])

    ranked = sorted(scanned_ids, key=key)
    chosen_ids = ranked[:N_MID]
    sel = np.asarray([scanned_ids.index(i) for i in chosen_ids])
    chosen = [
        {
            "prompt_id": pid,
            "population": prompts[pid]["population"],
            "stratum": prompts[pid].get("stratum"),
            "r_O_x100": float(rO[pid]),
            "historical_refusal_rate": prompts[pid].get("historical_refusal_rate"),
            "o_proj_sd": float(o_proj[scanned_ids.index(pid)] / SD),
            "n_words": prompts[pid]["n_words"],
            "text_head": prompts[pid]["text"][:80],
        }
        for pid in chosen_ids
    ]
    next_in_line = [{"prompt_id": pid, "r_O_x100": float(rO[pid]), "historical_refusal_rate": prompts[pid].get("historical_refusal_rate")} for pid in ranked[N_MID : N_MID + 5]]

    mid_m = S[sel]  # [10, N]
    mid = mid_m.mean(0)
    hp_mask, bp_mask = pops == "HP", pops == "BP"
    hp_ex = hp_mask.copy()
    hp_ex[sel] = False
    bp_ex = bp_mask.copy()
    bp_ex[sel] = False
    tables = {
        "benign": benign,
        "HP": S[hp_mask].mean(0),
        "BP": S[bp_mask].mean(0),
        "HP_minus_mid": S[hp_ex].mean(0),
        "BP_minus_mid": S[bp_ex].mean(0),
    }
    n_tables = {"benign": int(benign_m.shape[0]), "HP": int(hp_mask.sum()), "BP": int(bp_mask.sum()), "HP_minus_mid": int(hp_ex.sum()), "BP_minus_mid": int(bp_ex.sum())}

    # 1. Distribution.
    dist = {"mid10": distribution(mid)} | {k: distribution(v) for k, v in tables.items()}
    dist["mid10_by_token_count"] = {str(t): float(mid[n_tok == t].mean() / SD) for t in sorted(set(n_tok.tolist()))}

    # 2/3. Comparisons.
    comparisons = {k: compare(v, mid, words) for k, v in tables.items()}
    comparisons["HP_vs_benign"] = compare(benign, tables["HP"], words)
    comparisons["HP_minus_mid_vs_benign"] = compare(benign, tables["HP_minus_mid"], words)
    comparisons["BP_vs_benign"] = compare(benign, tables["BP"], words)
    comparisons["HP_minus_mid_vs_BP_minus_mid"] = compare(tables["BP_minus_mid"], tables["HP_minus_mid"], words)

    order = np.argsort(-mid, kind="stable")

    def row(j: int) -> dict:
        return {
            "word": words[j],
            "mid10_sd": float(mid[j] / SD),
            "benign_sd": float(benign[j] / SD),
            "HP_sd": float(tables["HP"][j] / SD),
            "HP_minus_mid_sd": float(tables["HP_minus_mid"][j] / SD),
            "BP_sd": float(tables["BP"][j] / SD),
            "benign_rank_from_top": int(rank_from_top(benign)[j]),
            "n_tokens": int(n_tok[j]),
            "in_x83_lexicon": words[j] in set(lex),
            "in_wordfreq_top50k": words[j] in wf,
            "n_of_10_prompts_above_zero": int((mid_m[:, j] > 0).sum()),
        }

    top20 = [row(j) for j in order[:20]]
    bottom20 = [row(j) for j in order[::-1][:20]]
    # Benign-table extremes read in the mid-10 table.
    border = np.argsort(-benign, kind="stable")
    benign_top20_here = [row(j) for j in border[:20]]
    benign_bottom20_here = [row(j) for j in border[::-1][:20]]

    # 4. Word groups.
    groups = {"mid10": word_groups(mid, words, cidx, lex, style, twins), "benign": word_groups(benign, words, cidx, lex, style, twins), "HP_minus_mid": word_groups(tables["HP_minus_mid"], words, cidx, lex, style, twins), "BP": word_groups(tables["BP"], words, cidx, lex, style, twins)}

    # 5. Stability inside the ten.
    mid_top = order[:TOP_K]
    benign_top = border[:TOP_K]
    z = zscore_rows(mid_m)

    def halves(a_idx, b_idx, m) -> dict:
        a, b = m[a_idx].mean(0), m[b_idx].mean(0)
        return {"all_words": {"spearman": spearman(a, b), "pearson": pearson(a, b)}, "mid10_top1000": {"spearman": spearman(a[mid_top], b[mid_top]), "pearson": pearson(a[mid_top], b[mid_top])}, "benign_top1000": {"spearman": spearman(a[benign_top], b[benign_top]), "pearson": pearson(a[benign_top], b[benign_top])}, "top100_overlap": int(len(set(np.argsort(-a)[:100].tolist()) & set(np.argsort(-b)[:100].tolist())))}

    first, second = np.arange(0, 5), np.arange(5, 10)
    even, odd = np.arange(0, 10, 2), np.arange(1, 10, 2)
    rng = np.random.default_rng(0)
    rand_sp = []
    for _ in range(20):
        perm = rng.permutation(10)
        rand_sp.append(spearman(mid_m[perm[:5]].mean(0), mid_m[perm[5:]].mean(0)))
    stability = {
        "split_first5_vs_last5": {"raw": halves(first, second, mid_m), "per_prompt_z": halves(first, second, z)},
        "split_even_vs_odd": {"raw": halves(even, odd, mid_m), "per_prompt_z": halves(even, odd, z)},
        "random_5_5_splits_spearman_all_words": {"n_splits": 20, "mean": float(np.mean(rand_sp)), "min": float(np.min(rand_sp)), "max": float(np.max(rand_sp))},
        "per_prompt_vs_benign_mean": [
            {"prompt_id": pid, "population": prompts[pid]["population"], "spearman_all_words": spearman(mid_m[k], benign), "pearson_all_words": pearson(mid_m[k], benign), "spearman_benign_top1000": spearman(mid_m[k][benign_top], benign[benign_top]), "spearman_vs_mid10_mean_leave_one_out": spearman(mid_m[k], np.delete(mid_m, k, axis=0).mean(0)), "spearman_vs_HP_minus_mid": spearman(mid_m[k], tables["HP_minus_mid"]), "spearman_vs_BP_minus_mid": spearman(mid_m[k], tables["BP_minus_mid"]), "mean_shift_sd": float(mid_m[k].mean() / SD), "sd_shift_sd": float(mid_m[k].std() / SD)}
            for k, pid in enumerate(chosen_ids)
        ],
        "variance_decomposition": {"raw": {"all": decompose(mid_m), "mid10_top1000": decompose(mid_m[:, mid_top])}, "per_prompt_z": {"all": decompose(z), "mid10_top1000": decompose(z[:, mid_top])}},
        "benign40_for_reference": {"variance_decomposition_raw_all": decompose(benign_m), "variance_decomposition_z_all": decompose(zscore_rows(benign_m))},
    }
    # Per-prompt correlation with the benign mean for every scanned prompt, so the ten can be placed among the 132.
    all_pp = np.asarray([spearman(S[k], benign) for k in range(S.shape[0])])
    stability["all_132_prompts_spearman_vs_benign"] = {"HP": {"mean": float(all_pp[hp_mask].mean()), "min": float(all_pp[hp_mask].min()), "max": float(all_pp[hp_mask].max())}, "BP": {"mean": float(all_pp[bp_mask].mean()), "min": float(all_pp[bp_mask].min()), "max": float(all_pp[bp_mask].max())}, "mid10": {"mean": float(all_pp[sel].mean()), "min": float(all_pp[sel].min()), "max": float(all_pp[sel].max())}, "spearman_with_r_O_x100_over_132": spearman(all_pp, [rO[i] for i in scanned_ids]), "spearman_with_o_proj_over_132": spearman(all_pp, o_proj)}

    # 6. Per-prompt view.
    per_prompt = []
    for k, pid in enumerate(chosen_ids):
        v = mid_m[k]
        o = np.argsort(-v, kind="stable")
        per_prompt.append(
            {
                "prompt_id": pid,
                "population": prompts[pid]["population"],
                "r_O_x100": float(rO[pid]),
                "o_proj_sd": float(o_proj[sel[k]] / SD),
                "top5": [{"word": words[j], "sd": float(v[j] / SD)} for j in o[:5]],
                "bottom5": [{"word": words[j], "sd": float(v[j] / SD)} for j in o[::-1][:5]],
                "style_words_sd": {w: float(v[cidx[w]] / SD) for w in style},
                "twins_sd": {t: float(v[cidx[t]] / SD) for t in twins},
                "style_words_mean_sd": float(np.mean([v[cidx[w]] for w in style]) / SD),
                "lexicon_mean_sd": float(np.mean([v[cidx[w]] for w in lex]) / SD),
            }
        )

    # Figures.
    xb, ym = benign / SD, mid / SD
    li = [cidx[w] for w in lex]
    si = [cidx[w] for w in style]
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    hb = ax.hexbin(xb, ym, gridsize=70, bins="log", cmap="Blues", mincnt=1, linewidths=0.2)
    lim = [min(xb.min(), ym.min()) - 0.1, max(xb.max(), ym.max()) + 0.1]
    ax.plot(lim, lim, color="#888888", lw=1, ls="--", zorder=2)
    ax.scatter(xb[li], ym[li], s=22, facecolor="none", edgecolor="#d1495b", lw=1.1, zorder=4, label="x83 lexicon (99)")
    ax.scatter(xb[si], ym[si], s=34, marker="^", facecolor="#e9a03b", edgecolor="#5a3a00", lw=0.6, zorder=5, label="x110 ten style words")
    for w in NAMED_LEXICON + ("pedophilia", "poetic", "playfully"):
        j = cidx[w]
        ax.annotate(w, (xb[j], ym[j]), xytext=(4, 3), textcoords="offset points", fontsize=7.5, color="#333333")
    ax.axhline(0, color="#bbbbbb", lw=0.6, zorder=1)
    ax.axvline(0, color="#bbbbbb", lw=0.6, zorder=1)
    ax.set_xlabel("benign table: mean shift over 40 x109 prompts (sd)")
    ax.set_ylabel(f"mid-10 table: mean shift over the {N_MID} mid-refusal prompts (sd)")
    ax.set_title(f"71,583 words; ρ = {comparisons['benign']['spearman_all_words']:+.2f}, r = {comparisons['benign']['pearson_all_words']:+.2f}", fontsize=10)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    cb = fig.colorbar(hb, ax=ax, shrink=0.8)
    cb.set_label("words per cell (log)")
    ax.grid(color="#eeeeee", lw=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "mid10_vs_benign.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    ylim = (ym.min() - 0.15, ym.max() + 0.15)
    for ax, (name, lab) in zip(axes, (("HP_minus_mid", f"HP table without the chosen prompts ({n_tables['HP_minus_mid']})"), ("BP_minus_mid", f"BP table without the chosen prompts ({n_tables['BP_minus_mid']})"))):
        xr = tables[name] / SD
        ax.hexbin(xr, ym, gridsize=70, bins="log", cmap="Blues", mincnt=1, linewidths=0.2)
        ax.plot(ylim, ylim, color="#888888", lw=1, ls="--", zorder=2)
        ax.set_ylim(*ylim)
        ax.scatter(xr[li], ym[li], s=22, facecolor="none", edgecolor="#d1495b", lw=1.1, zorder=4, label="x83 lexicon (99)")
        ax.scatter(xr[si], ym[si], s=34, marker="^", facecolor="#e9a03b", edgecolor="#5a3a00", lw=0.6, zorder=5, label="x110 ten style words")
        ax.axhline(0, color="#bbbbbb", lw=0.6, zorder=1)
        ax.axvline(0, color="#bbbbbb", lw=0.6, zorder=1)
        c = comparisons[name]
        ax.set_xlabel(f"{lab} (sd)")
        ax.set_title(f"ρ = {c['spearman_all_words']:+.2f}, r = {c['pearson_all_words']:+.2f}", fontsize=10)
        ax.grid(color="#eeeeee", lw=0.5)
    for ax in axes:
        ax.set_ylabel("mid-10 table (sd)")
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "mid10_vs_hp_bp.png", dpi=150)
    plt.close(fig)

    out = {
        "run": "x110 midscan compare",
        "selection_rule": "among the prompts in x110 scan.json prompt_ids_scanned, rank by |r_O_x100 - 0.5| ascending, then |historical_refusal_rate - 0.5| ascending (BP prompts have no historical rate and count as 0.5), then position in rate_matrix.prompt_ids; take the first ten",
        "statistic": "per-word mean over the prompts of the shift summed over the two boundary tokens; sd unit = x106 Hard-1K sd on Qwen3.5-4B",
        "sd_unit": SD,
        "n_words": N,
        "word_lists_identical_x109_x110": True,
        "n_prompts": n_tables | {"mid10": N_MID, "scanned": int(S.shape[0])},
        "chosen_prompts": chosen,
        "next_in_line": next_in_line,
        "distribution_sd": dist,
        "comparisons_mid10_vs": {k: comparisons[k] for k in tables},
        "comparisons_other": {k: comparisons[k] for k in comparisons if k not in tables},
        "mid10_top20": top20,
        "mid10_bottom20": bottom20,
        "benign_top20_in_mid10": benign_top20_here,
        "benign_bottom20_in_mid10": benign_bottom20_here,
        "word_groups": groups,
        "stability": stability,
        "per_prompt": per_prompt,
        "figures": [str((OUT / "mid10_vs_benign.png").relative_to(OUT.parents[2])), str((OUT / "mid10_vs_hp_bp.png").relative_to(OUT.parents[2]))],
    }
    atomic_json(OUT / "results.json", out)
    write_md(out)
    print(f"wrote {OUT}")


def ordinal(x: float) -> str:
    n = int(round(x))
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def write_md(r: dict) -> None:
    L = ["# x110 side analysis: the vocabulary table on ten mid-refusal prompts (Qwen3.5-4B)\n"]
    L.append(f"Ten of the {r['n_prompts']['scanned']} x100 prompts covered by the x110 vocabulary scan, chosen by the rule: {r['selection_rule']}. Per-word statistic: {r['statistic']} ({r['sd_unit']:.3f}). The x110 candidate list is string-identical to the x109 list ({r['n_words']:,} words). Reference tables: the x109 benign table (40 prompts), x110's HP table ({r['n_prompts']['HP']} scanned HP prompts), BP table ({r['n_prompts']['BP']}), and the same two with the chosen prompts removed (HP {r['n_prompts']['HP_minus_mid']}, BP {r['n_prompts']['BP_minus_mid']}).\n")

    L.append("## The ten prompts\n")
    L.append("| # | id | population | stratum | r(O) x100 | historical | O proj (sd) | words | text |\n|---|---|---|---|---|---|---|---|---|")
    for k, c in enumerate(r["chosen_prompts"], 1):
        h = "–" if c["historical_refusal_rate"] is None else f"{c['historical_refusal_rate']:.1f}"
        L.append(f"| {k} | {c['prompt_id']} | {c['population']} | {c['stratum'] or '–'} | {c['r_O_x100']:.1f} | {h} | {c['o_proj_sd']:+.2f} | {c['n_words']} | {c['text_head'].replace('|', '/')}… |")
    L.append("\nNext in line (not chosen): " + ", ".join(f"{n['prompt_id']} (r(O) {n['r_O_x100']:.1f})" for n in r["next_in_line"]) + ".\n")

    L.append("## Distribution of the per-word mean (sd units)\n")
    L.append("| table | prompts | p1 | p10 | median | p90 | p99 | p99.9 | max | mean | skew | excess kurtosis | below zero | > 1 sd | > 2 sd |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name in ("mid10", "benign", "HP", "HP_minus_mid", "BP", "BP_minus_mid"):
        d = r["distribution_sd"][name]
        L.append(f"| {name} | {r['n_prompts'][name]} | {d['p1']:+.2f} | {d['p10']:+.2f} | {d['median']:+.2f} | {d['p90']:+.2f} | {d['p99']:+.2f} | {d['p99_9']:+.2f} | {d['max']:+.2f} | {d['mean']:+.2f} | {d['skew']:.1f} | {d['excess_kurtosis']:.0f} | {d['frac_below_zero']:.3f} | {d['n_above_1sd']} | {d['n_above_2sd']} |")
    tc = r["distribution_sd"]["mid10_by_token_count"]
    L.append("\nMid-10 mean by token count: " + ", ".join(f"{t} token{'s' if t != '1' else ''} {v:+.2f}" for t, v in tc.items()) + ".\n")

    L.append("## The mid-10 table against the other tables\n")
    L.append("Spearman and Pearson over all words; top-/bottom-100 overlap of the rankings; Spearman restricted to the reference table's own top 1,000 and bottom 1,000 words.\n")
    L.append("| reference | prompts | ρ all | r all | top-100 overlap | bottom-100 overlap | top-1000 overlap | bottom-1000 overlap | ρ within ref top 1,000 | ρ within ref bottom 1,000 | mean ref | mean mid-10 |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name in ("benign", "HP", "HP_minus_mid", "BP", "BP_minus_mid"):
        c = r["comparisons_mid10_vs"][name]
        L.append(f"| {name} | {r['n_prompts'][name]} | {c['spearman_all_words']:+.3f} | {c['pearson_all_words']:+.3f} | {c['top100_overlap']} | {c['bottom100_overlap']} | {c['top1000_overlap']} | {c['bottom1000_overlap']} | {c['spearman_within_ref_top1000']:+.3f} | {c['spearman_within_ref_bottom1000']:+.3f} | {c['mean_ref_sd']:+.2f} | {c['mean_here_sd']:+.2f} |")
    L.append("\nThe other tables against each other, same statistics (row = table, reference = column):\n")
    L.append("| table | reference | ρ all | r all | top-100 overlap | bottom-100 overlap | ρ within ref top 1,000 | ρ within ref bottom 1,000 |\n|---|---|---|---|---|---|---|---|")
    for name, (a, b) in (("HP_vs_benign", ("HP", "benign")), ("HP_minus_mid_vs_benign", ("HP_minus_mid", "benign")), ("BP_vs_benign", ("BP", "benign")), ("HP_minus_mid_vs_BP_minus_mid", ("HP_minus_mid", "BP_minus_mid"))):
        c = r["comparisons_other"][name]
        L.append(f"| {a} | {b} | {c['spearman_all_words']:+.3f} | {c['pearson_all_words']:+.3f} | {c['top100_overlap']} | {c['bottom100_overlap']} | {c['spearman_within_ref_top1000']:+.3f} | {c['spearman_within_ref_bottom1000']:+.3f} |")
    L.append(f"\nFigures: `{r['figures'][0]}` (benign vs mid-10, hexbin, x83 lexicon and the x110 ten style words marked), `{r['figures'][1]}` (HP-minus-mid and BP-minus-mid vs mid-10).\n")

    def wt(rows: list[dict], title: str) -> None:
        L.append(f"### {title}\n")
        L.append("| word | mid-10 | benign | benign rank | HP | HP−mid | BP | tokens | lexicon | wordfreq-50k | prompts > 0 |\n|---|---|---|---|---|---|---|---|---|---|---|")
        for w in rows:
            L.append(f"| {w['word']} | {w['mid10_sd']:+.2f} | {w['benign_sd']:+.2f} | {w['benign_rank_from_top']:,} | {w['HP_sd']:+.2f} | {w['HP_minus_mid_sd']:+.2f} | {w['BP_sd']:+.2f} | {w['n_tokens']} | {'yes' if w['in_x83_lexicon'] else ''} | {'yes' if w['in_wordfreq_top50k'] else ''} | {w['n_of_10_prompts_above_zero']} |")
        L.append("")

    wt(r["mid10_top20"], "Top 20 of the mid-10 table (raw ranking, no drop rules)")
    wt(r["mid10_bottom20"], "Bottom 20 of the mid-10 table")
    wt(r["benign_top20_in_mid10"], "The benign table's top 20, read in the mid-10 table")
    wt(r["benign_bottom20_in_mid10"], "The benign table's bottom 20, read in the mid-10 table")

    L.append("## Word groups\n")
    L.append("x83 lexicon (99 words): mean shift, count in the table's top 1 % and top 10 %, and the percentile (100 = largest) of five weapon/drug/cyber words.\n")
    L.append("| table | lexicon mean (sd) | median percentile | in top 1 % | in top 10 % | shoot | meth | hack | gun | bomb |\n|---|---|---|---|---|---|---|---|---|---|")
    for name in ("mid10", "benign", "HP_minus_mid", "BP"):
        g = r["word_groups"][name]["x83_lexicon"]
        nm = g["named"]
        L.append(f"| {name} | {g['mean_sd']:+.2f} | {g['median_percentile']:.0f} | {g['n_in_top_1pct']} | {g['n_in_top_10pct']} | " + " | ".join(f"{ordinal(nm[w]['percentile'])} ({nm[w]['sd']:+.2f})" for w in NAMED_LEXICON) + " |")
    g = r["word_groups"]["mid10"]["x83_lexicon"]
    L.append("\nMid-10 lexicon top 5: " + ", ".join(f"{t['word']} ({t['sd']:+.2f}, {ordinal(t['percentile'])})" for t in g["top5"]) + "; bottom 5: " + ", ".join(f"{t['word']} ({t['sd']:+.2f}, {ordinal(t['percentile'])})" for t in g["bottom5"]) + ".\n")
    L.append("The x110 ten style words and their twins, with the rank from the bottom of each table:\n")
    L.append("| word | mid-10 | rank from bottom | benign | rank | HP−mid | rank | BP | rank | twin | mid-10 twin | twin rank |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    gm, gb, gh, gp = (r["word_groups"][k]["x110_style_words"] for k in ("mid10", "benign", "HP_minus_mid", "BP"))
    for a, b, c, d in zip(gm, gb, gh, gp):
        L.append(f"| {a['word']} | {a['sd']:+.2f} | {a['rank_from_bottom']:,} | {b['sd']:+.2f} | {b['rank_from_bottom']:,} | {c['sd']:+.2f} | {c['rank_from_bottom']:,} | {d['sd']:+.2f} | {d['rank_from_bottom']:,} | {a['twin']} | {a['twin_sd']:+.2f} | {a['twin_rank_from_bottom']:,} |")
    L.append(f"\nMean over the ten style words: mid-10 {r['word_groups']['mid10']['x110_style_words_mean_sd']:+.2f}, benign {r['word_groups']['benign']['x110_style_words_mean_sd']:+.2f}, HP−mid {r['word_groups']['HP_minus_mid']['x110_style_words_mean_sd']:+.2f}, BP {r['word_groups']['BP']['x110_style_words_mean_sd']:+.2f}; twins mid-10 {r['word_groups']['mid10']['x110_twins_mean_sd']:+.2f}, benign {r['word_groups']['benign']['x110_twins_mean_sd']:+.2f}.\n")

    L.append("## Stability inside the ten\n")
    st = r["stability"]
    L.append("Split halves of the ten prompts (per-word means of each half); 'top 1,000' = the mid-10 table's own top 1,000, 'benign top 1,000' = the benign table's.\n")
    L.append("| split | scale | ρ all words | r all words | ρ top 1,000 | ρ benign top 1,000 | top-100 overlap |\n|---|---|---|---|---|---|---|")
    for name, lab in (("split_first5_vs_last5", "prompts 1–5 vs 6–10"), ("split_even_vs_odd", "odd vs even")):
        for sc in ("raw", "per_prompt_z"):
            h = st[name][sc]
            L.append(f"| {lab} | {sc} | {h['all_words']['spearman']:+.3f} | {h['all_words']['pearson']:+.3f} | {h['mid10_top1000']['spearman']:+.3f} | {h['benign_top1000']['spearman']:+.3f} | {h['top100_overlap']} |")
    rs = st["random_5_5_splits_spearman_all_words"]
    L.append(f"\nRandom 5/5 splits ({rs['n_splits']}): ρ over all words mean {rs['mean']:+.3f} [{rs['min']:+.3f}, {rs['max']:+.3f}] (x109's 10/10 halves within a 20-prompt set: +0.875 and +0.864).\n")
    L.append("Each prompt's own word vector against the table means:\n")
    L.append("| prompt | population | ρ vs benign (all) | r vs benign | ρ vs benign (benign top 1,000) | ρ vs the other nine | ρ vs HP−mid | ρ vs BP−mid | mean shift (sd) | sd of shift (sd) |\n|---|---|---|---|---|---|---|---|---|---|")
    for p in st["per_prompt_vs_benign_mean"]:
        L.append(f"| {p['prompt_id']} | {p['population']} | {p['spearman_all_words']:+.3f} | {p['pearson_all_words']:+.3f} | {p['spearman_benign_top1000']:+.3f} | {p['spearman_vs_mid10_mean_leave_one_out']:+.3f} | {p['spearman_vs_HP_minus_mid']:+.3f} | {p['spearman_vs_BP_minus_mid']:+.3f} | {p['mean_shift_sd']:+.2f} | {p['sd_shift_sd']:.2f} |")
    a = st["all_132_prompts_spearman_vs_benign"]
    L.append(f"\nOver all 132 scanned prompts, per-prompt ρ with the benign mean: HP mean {a['HP']['mean']:+.3f} [{a['HP']['min']:+.3f}, {a['HP']['max']:+.3f}], BP mean {a['BP']['mean']:+.3f} [{a['BP']['min']:+.3f}, {a['BP']['max']:+.3f}], the chosen ten {a['mid10']['mean']:+.3f} [{a['mid10']['min']:+.3f}, {a['mid10']['max']:+.3f}]; ρ of this per-prompt correlation with r(O) x100 over the 132: {a['spearman_with_r_O_x100_over_132']:+.3f}, with the O projection {a['spearman_with_o_proj_over_132']:+.3f}.\n")
    L.append("Two-way decomposition of the 10 × 71,583 matrix (x109 wordtable_checks statistic):\n")
    L.append("| scale | words | prompt main | word main | residual | rank-1 share (uncentred) | additive share (uncentred) |\n|---|---|---|---|---|---|---|")
    for sc in ("raw", "per_prompt_z"):
        for ws in ("all", "mid10_top1000"):
            d = st["variance_decomposition"][sc][ws]
            L.append(f"| {sc} | {ws} | {d['share_prompt_main']:.3f} | {d['share_word_main']:.3f} | {d['share_residual_interaction']:.3f} | {d['rank1_share_of_uncentred_ss']:.3f} | {d['additive_share_of_uncentred_ss']:.3f} |")
    b = st["benign40_for_reference"]
    L.append(f"| raw (benign 40, reference) | all | {b['variance_decomposition_raw_all']['share_prompt_main']:.3f} | {b['variance_decomposition_raw_all']['share_word_main']:.3f} | {b['variance_decomposition_raw_all']['share_residual_interaction']:.3f} | {b['variance_decomposition_raw_all']['rank1_share_of_uncentred_ss']:.3f} | {b['variance_decomposition_raw_all']['additive_share_of_uncentred_ss']:.3f} |")
    L.append(f"| per_prompt_z (benign 40, reference) | all | {b['variance_decomposition_z_all']['share_prompt_main']:.3f} | {b['variance_decomposition_z_all']['share_word_main']:.3f} | {b['variance_decomposition_z_all']['share_residual_interaction']:.3f} | {b['variance_decomposition_z_all']['rank1_share_of_uncentred_ss']:.3f} | {b['variance_decomposition_z_all']['additive_share_of_uncentred_ss']:.3f} |")

    L.append("\n## Per prompt\n")
    L.append("| prompt | pop | r(O) | O proj (sd) | top 5 (sd) | bottom 5 (sd) | style-word mean | lexicon mean |\n|---|---|---|---|---|---|---|---|")
    for p in r["per_prompt"]:
        t5 = ", ".join(f"{t['word']} {t['sd']:+.1f}" for t in p["top5"])
        b5 = ", ".join(f"{t['word']} {t['sd']:+.1f}" for t in p["bottom5"])
        L.append(f"| {p['prompt_id']} | {p['population']} | {p['r_O_x100']:.1f} | {p['o_proj_sd']:+.2f} | {t5} | {b5} | {p['style_words_mean_sd']:+.2f} | {p['lexicon_mean_sd']:+.2f} |")
    style = list(r["per_prompt"][0]["style_words_sd"])
    L.append("\nShift of the x110 ten style words on each prompt (sd):\n")
    L.append("| prompt | " + " | ".join(style) + " |\n|---|" + "---|" * len(style))
    for p in r["per_prompt"]:
        L.append(f"| {p['prompt_id']} | " + " | ".join(f"{p['style_words_sd'][w]:+.2f}" for w in style) + " |")
    (OUT / "results.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
