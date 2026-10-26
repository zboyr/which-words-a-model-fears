"""Vocabulary scan, word-table checks (local, numpy/scipy only).

Three checks on the scan matrix (40 prompts x 71,583 words, shift on the Qwen3.5-4B boundary
direction) and the generated cells: (1) stability of the per-word mean shift across prompt
sets and split halves; (2) two-way variance decomposition of the matrix into prompt, word and
residual, raw and after per-prompt z-scoring; (3) dose-response of refusal2 r(E) against the
scanned shift over every generated word (pilot 15 + scan 10 on set A, scan 10 on set B), a
cell-level logistic fit, the shift range with no generated words, and the binomial noise
ceiling of the cell-level correlation. Writes wordtable_checks/results.json and results.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x109_scan_common import (  # noqa: E402
    JUDGE_DIR,
    PILOT,
    PROMPTS,
    RESULTS,
    SCAN_JSON,
    SCAN_NPZ,
    WORDS,
    X106_HARD1K_SD,
    atomic_json,
    read_json,
)

OUT = SCAN_NPZ.parent / "wordtable_checks"
SPLIT_SEEDS = list(range(10))
HALF_SEEDS = list(range(20))
TOP_K = (1000, 100)


def load_refusal(path: Path) -> dict[str, np.ndarray]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("failed") or "refusal" not in rec:
            continue
        out[rec["cell_id"]] = np.asarray(rec["refusal"], dtype=float)
    return out


def corr(x: np.ndarray, y: np.ndarray) -> dict:
    return {"spearman": float(stats.spearmanr(x, y).statistic), "pearson": float(stats.pearsonr(x, y).statistic), "n": int(len(x))}


def slope_ratio(a: np.ndarray, b: np.ndarray) -> dict:
    """OLS slope of b on a (with intercept), through-origin slope, and the ratio of means."""
    A = np.column_stack([a, np.ones_like(a)])
    coef = np.linalg.lstsq(A, b, rcond=None)[0]
    return {
        "ols_slope_b_on_a": float(coef[0]),
        "ols_intercept": float(coef[1]),
        "through_origin_slope": float((a * b).sum() / (a * a).sum()),
        "ratio_of_means": float(b.mean() / a.mean()) if abs(a.mean()) > 1e-3 else None,
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
    }


def zscore_rows(m: np.ndarray) -> np.ndarray:
    return (m - m.mean(1, keepdims=True)) / m.std(1, keepdims=True)


def decompose(m: np.ndarray) -> dict:
    """Two-way additive decomposition of SS about the grand mean, plus the rank-1 share."""
    P, W = m.shape
    g = m.mean()
    rows, cols = m.mean(1), m.mean(0)
    ss_total = float(((m - g) ** 2).sum())
    ss_prompt = float(W * ((rows - g) ** 2).sum())
    ss_word = float(P * ((cols - g) ** 2).sum())
    ss_resid = ss_total - ss_prompt - ss_word
    # Multiplicative alternative: best rank-1 fit of the uncentred matrix (prompt scale x word value).
    s = np.linalg.svd(m, compute_uv=False)
    ss_uncentred = float((m ** 2).sum())
    return {
        "shape": [int(P), int(W)],
        "ss_total_about_grand_mean": ss_total,
        "share_prompt_main": ss_prompt / ss_total,
        "share_word_main": ss_word / ss_total,
        "share_residual_interaction": ss_resid / ss_total,
        "rank1_share_of_uncentred_ss": float(s[0] ** 2 / ss_uncentred),
        "rank2_share_of_uncentred_ss": float((s[0] ** 2 + s[1] ** 2) / ss_uncentred),
        "additive_share_of_uncentred_ss": float(1 - ss_resid / ss_uncentred),
    }


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_logistic(x: np.ndarray, k: np.ndarray, n: np.ndarray, ceiling: bool = False) -> dict:
    """Binomial logistic fit of k/n on x. ceiling=True adds a free upper asymptote."""
    p0 = (k.sum() + 0.5) / (n.sum() + 1)

    def nll(theta):
        b0, b1 = theta[0], theta[1]
        p = sigmoid(b0 + b1 * x)
        if ceiling:
            p = sigmoid(theta[2]) * p
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -(k * np.log(p) + (n - k) * np.log(1 - p)).sum()

    x0 = [np.log(p0 / (1 - p0)), 0.0] + ([2.0] if ceiling else [])
    res = optimize.minimize(nll, x0, method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 20000})
    res = optimize.minimize(nll, res.x, method="BFGS")
    b0, b1 = float(res.x[0]), float(res.x[1])
    null = -(k * np.log(p0) + (n - k) * np.log(1 - p0)).sum()
    out = {
        "intercept": b0,
        "slope_per_raw_unit": b1,
        "slope_per_sd_unit": b1 * X106_HARD1K_SD,
        "midpoint_raw": -b0 / b1 if b1 != 0 else None,
        "midpoint_sd_units": (-b0 / b1) / X106_HARD1K_SD if b1 != 0 else None,
        "nll": float(res.fun),
        "null_nll": float(null),
        "deviance_explained": float(1 - res.fun / null),
        "n_cells": int(len(x)),
        "n_samples": int(n.sum()),
        "x_min": float(x.min()),
        "x_max": float(x.max()),
    }
    if ceiling:
        c = float(sigmoid(res.x[2]))
        out["asymptote"] = c
        out["note"] = "midpoint is the shift where the fitted rate reaches half the asymptote"
    else:
        out["note"] = "midpoint is the shift where the fitted rate is 0.5"
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prompts = read_json(PROMPTS)["prompts"]
    scan = read_json(SCAN_JSON)
    words_obj = read_json(WORDS)
    res = read_json(RESULTS)
    pilot_res = read_json(PILOT / "results.json")
    npz = np.load(SCAN_NPZ)
    shift = npz["shift"].astype(np.float64).sum(axis=2)  # [P, W]
    o_proj = npz["o_cached"].astype(np.float64).sum(axis=1)  # [P]
    cand = scan["candidates"]["words"]
    cidx = {w: j for j, w in enumerate(cand)}
    pids = [p["id"] for p in prompts]
    if scan["prompt_ids"] != pids:
        raise SystemExit("FATAL: scan prompt order differs")
    sets = np.asarray([p["set"] for p in prompts])
    shapes = np.asarray([p["shape"] for p in prompts])
    mA, mB = sets == "A", sets == "B"
    P, W = shift.shape
    z = zscore_rows(shift)
    mean_all = shift.mean(0)
    order_all = np.argsort(-mean_all)
    subsets = {"all": np.arange(W)} | {f"top{k}_by_all40_mean": order_all[:k] for k in TOP_K}

    # ---------------- Check 1: word-table stability ----------------
    def ab_block(m: np.ndarray, label: str) -> dict:
        a, b = m[mA].mean(0), m[mB].mean(0)
        out = {}
        for name, idx in subsets.items():
            out[name] = {"corr_A_vs_B": corr(a[idx], b[idx]), "B_vs_A": slope_ratio(a[idx], b[idx])}
        return out

    def split_half(m: np.ndarray, mask: np.ndarray, seeds: list[int]) -> dict:
        rows = np.where(mask)[0]
        out = {name: {"spearman": [], "pearson": []} for name in subsets}
        for seed in seeds:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(rows)
            h1, h2 = perm[: len(rows) // 2], perm[len(rows) // 2 :]
            a, b = m[h1].mean(0), m[h2].mean(0)
            for name, idx in subsets.items():
                c = corr(a[idx], b[idx])
                out[name]["spearman"].append(c["spearman"])
                out[name]["pearson"].append(c["pearson"])
        summ = {}
        for name, d in out.items():
            sp, pe = np.asarray(d["spearman"]), np.asarray(d["pearson"])
            summ[name] = {
                "n_prompts_per_half": int(len(rows) // 2),
                "n_splits": len(seeds),
                "spearman_mean": float(sp.mean()),
                "spearman_min": float(sp.min()),
                "spearman_max": float(sp.max()),
                "pearson_mean": float(pe.mean()),
                "pearson_min": float(pe.min()),
                "pearson_max": float(pe.max()),
                "pearson_spearman_brown_to_full_set": float(2 * pe.mean() / (1 + pe.mean())),
            }
        return summ

    check1 = {
        "raw": {"set_A_vs_set_B": ab_block(shift, "raw"), "split_half_within_A": split_half(shift, mA, SPLIT_SEEDS), "split_half_within_B": split_half(shift, mB, SPLIT_SEEDS)},
        "per_prompt_z": {"set_A_vs_set_B": ab_block(z, "z"), "split_half_within_A": split_half(z, mA, SPLIT_SEEDS), "split_half_within_B": split_half(z, mB, SPLIT_SEEDS)},
        "top_sets": {f"top{k}": [cand[j] for j in order_all[:k]][: min(k, 100)] for k in TOP_K},
        "rank_overlap": {},
    }
    a_rank, b_rank = np.argsort(-shift[mA].mean(0)), np.argsort(-shift[mB].mean(0))
    for k in (10, 50, 100, 1000):
        sa, sb = set(a_rank[:k].tolist()), set(b_rank[:k].tolist())
        check1["rank_overlap"][f"top{k}"] = {"n_common": len(sa & sb), "jaccard": len(sa & sb) / len(sa | sb)}

    # ---------------- Check 2: variance decomposition ----------------
    check2 = {"raw": {}, "per_prompt_z": {}}
    for name, idx in subsets.items():
        check2["raw"][name] = decompose(shift[:, idx])
        check2["per_prompt_z"][name] = decompose(z[:, idx])
    pm, ps = shift.mean(1), shift.std(1)
    set_b, shape_b = (sets == "B").astype(float), (shapes == "B").astype(float)
    pf = {
        "per_prompt": [
            {"prompt_id": pids[i], "set": str(sets[i]), "shape": str(shapes[i]), "mean_shift_over_words": float(pm[i]), "sd_shift_over_words": float(ps[i]), "o_proj_cached": float(o_proj[i]), "n_prefix_tokens": int(scan["prefix_tokens"][i]), "top10_scan_word_mean_shift": float(shift[i, [cidx[c["word"]] for c in words_obj["chosen"]]].mean())}
            for i in range(P)
        ],
        "spearman_prompt_mean_vs_set_B": float(stats.spearmanr(pm, set_b).statistic),
        "spearman_prompt_sd_vs_set_B": float(stats.spearmanr(ps, set_b).statistic),
        "spearman_prompt_mean_vs_shape_B": float(stats.spearmanr(pm, shape_b).statistic),
        "spearman_prompt_sd_vs_shape_B": float(stats.spearmanr(ps, shape_b).statistic),
        "spearman_prompt_mean_vs_prompt_sd": float(stats.spearmanr(pm, ps).statistic),
        "spearman_prompt_sd_vs_o_proj": float(stats.spearmanr(ps, o_proj).statistic),
        "spearman_prompt_sd_vs_prefix_tokens": float(stats.spearmanr(ps, scan["prefix_tokens"]).statistic),
        "group_means": {},
        "mannwhitney_sd_A_vs_B_p": float(stats.mannwhitneyu(ps[mA], ps[mB]).pvalue),
        "mannwhitney_sd_shapeA_vs_shapeB_p": float(stats.mannwhitneyu(ps[shapes == "A"], ps[shapes == "B"]).pvalue),
    }
    for gname, gm in (("set_A", mA), ("set_B", mB), ("shape_A", shapes == "A"), ("shape_B", shapes == "B")):
        pf["group_means"][gname] = {"prompt_mean": float(pm[gm].mean()), "prompt_sd": float(ps[gm].mean()), "prompt_sd_min": float(ps[gm].min()), "prompt_sd_max": float(ps[gm].max())}
    # Within-set ranking of prompt sd: is the set difference a shift of the whole distribution?
    pf["sd_quantiles"] = {"set_A": np.quantile(ps[mA], [0, 0.25, 0.5, 0.75, 1]).tolist(), "set_B": np.quantile(ps[mB], [0, 0.25, 0.5, 0.75, 1]).tolist()}
    check2["prompt_factors"] = pf

    # ---------------- Check 3: dose-response ----------------
    ref_scan = load_refusal(JUDGE_DIR / "refusal2.jsonl")
    ref_pilot = load_refusal(PILOT / "judge" / "refusal2.jsonl")
    chosen = [c["word"] for c in words_obj["chosen"]]
    pilot_pw = {r["word"]: r for r in pilot_res["per_word"]}
    scan_pw = {r["word"]: r for r in res["per_word"]}
    pidx = {p: i for i, p in enumerate(pids)}

    # Cell tables: one row per (prompt, word) with generation data.
    def cell_rows(word_list: list[str], source: str, ref: dict, mask: np.ndarray) -> list[dict]:
        rows = []
        for w in word_list:
            for i in np.where(mask)[0]:
                rec = ref.get(f"{pids[i]}|E_{w}")
                if rec is None:
                    continue
                rows.append({"prompt_id": pids[i], "prompt_index": int(i), "word": w, "source": source, "shift": float(shift[i, cidx[w]]), "k": int(rec.sum()), "n": int(len(rec)), "samples": rec})
        return rows

    pilot_words = [w for w in pilot_pw if w not in chosen]
    cells_A_scan = cell_rows(chosen, "x109", ref_scan, mA)
    cells_A_pilot = cell_rows(pilot_words, "pilot", ref_pilot, mA)
    cells_B_scan = cell_rows(chosen, "x109", ref_scan, mB)
    cell_sets = {"A_scan10": cells_A_scan, "A_union25": cells_A_scan + cells_A_pilot, "A_pilot15": cells_A_pilot, "B_scan10": cells_B_scan, "all_scan10": cells_A_scan + cells_B_scan, "all_union": cells_A_scan + cells_A_pilot + cells_B_scan}

    # Per-word table.
    per_word = []
    for w in pilot_words:
        r = pilot_pw[w]
        per_word.append({"word": w, "source": "pilot", "set": "A", "twin": r["twin"], "mean_shift": float(shift[mA, cidx[w]].mean()), "mean_shift_sd_units": float(shift[mA, cidx[w]].mean() / X106_HARD1K_SD), "delta_EN": r["delta_EN"], "delta_EN_ci95": r["delta_EN_ci95"], "r_E": r["r_E"], "r_N": r["r_N"], "n_prompts": 20})
    for w in chosen:
        for s, m in (("A", mA), ("B", mB)):
            r = scan_pw[w][s]
            per_word.append({"word": w, "source": "x109", "set": s, "twin": scan_pw[w]["twin"], "mean_shift": float(shift[m, cidx[w]].mean()), "mean_shift_sd_units": float(shift[m, cidx[w]].mean() / X106_HARD1K_SD), "delta_EN": r["delta_EN"], "delta_EN_ci95": r["delta_EN_ci95"], "r_E": r["r_E"], "r_N": r["r_N"], "n_prompts": r["n_prompts"]})
    per_word.sort(key=lambda d: (d["set"], -d["mean_shift"]))

    # Per-word correlations of shift with Delta_EN, by table.
    def pw_corr(rows):
        x = np.asarray([r["mean_shift"] for r in rows])
        return {"vs_delta_EN": corr(x, np.asarray([r["delta_EN"] for r in rows])), "vs_r_E": corr(x, np.asarray([r["r_E"] for r in rows]))}

    per_word_corr = {"A_union25": pw_corr([r for r in per_word if r["set"] == "A"]), "A_scan10": pw_corr([r for r in per_word if r["set"] == "A" and r["source"] == "vocab_scan"]), "A_pilot15": pw_corr([r for r in per_word if r["set"] == "A" and r["source"] == "pilot"]), "B_scan10": pw_corr([r for r in per_word if r["set"] == "B"])}

    # Cell-level fits and correlations.
    cell_level = {}
    for name, rows in cell_sets.items():
        x = np.asarray([r["shift"] for r in rows])
        k = np.asarray([r["k"] for r in rows], float)
        n = np.asarray([r["n"] for r in rows], float)
        rate = k / n
        cell_level[name] = {
            "n_cells": len(rows),
            "spearman_shift_vs_r_E": float(stats.spearmanr(x, rate).statistic),
            "pearson_shift_vs_r_E": float(stats.pearsonr(x, rate).statistic),
            "logistic_2param": fit_logistic(x, k, n),
            "logistic_3param_asymptote": fit_logistic(x, k, n, ceiling=True),
        }
    # Fitted curve on a grid for the main sets (raw shift 0..30).
    grid = np.linspace(0, 30, 61)
    curves = {}
    for name in ("A_union25", "B_scan10", "all_union"):
        f2, f3 = cell_level[name]["logistic_2param"], cell_level[name]["logistic_3param_asymptote"]
        curves[name] = {"shift_raw": grid.tolist(), "rate_2param": sigmoid(f2["intercept"] + f2["slope_per_raw_unit"] * grid).tolist(), "rate_3param": (f3["asymptote"] * sigmoid(f3["intercept"] + f3["slope_per_raw_unit"] * grid)).tolist()}

    # Word-table vs cell shift vs prompt factor as predictors of cell-level r(E).
    # Word table = the word's mean scanned shift over the OTHER prompt set (out-of-sample lookup);
    # also the same-set mean (in-sample lookup), the prompt's own mean over all words, and the sum.
    predictors = {}
    for name in ("A_scan10", "A_union25", "B_scan10", "all_scan10"):
        rows = cell_sets[name]
        k = np.asarray([r["k"] for r in rows], float)
        n = np.asarray([r["n"] for r in rows], float)
        rate = k / n
        pi = np.asarray([r["prompt_index"] for r in rows])
        wj = np.asarray([cidx[r["word"]] for r in rows])
        other = np.where(mA[pi], 1, 0)  # 1 -> prompt is in A, so other set is B
        cell_shift = shift[pi, wj]
        word_other = np.where(other == 1, shift[mB].mean(0)[wj], shift[mA].mean(0)[wj])
        word_same = np.where(other == 1, shift[mA].mean(0)[wj], shift[mB].mean(0)[wj])
        prompt_mean = pm[pi]
        prompt_sd = ps[pi]
        cell_z = z[pi, wj]
        additive = word_other + (prompt_mean - prompt_mean.mean())
        multiplicative = word_other / word_other.mean() * prompt_sd  # prompt scale x word value
        feats = {"cell_shift": cell_shift, "cell_shift_per_prompt_z": cell_z, "word_mean_other_set": word_other, "word_mean_same_set": word_same, "prompt_mean_over_all_words": prompt_mean, "prompt_sd_over_all_words": prompt_sd, "word_other_plus_prompt_mean": additive, "word_other_times_prompt_sd": multiplicative}
        predictors[name] = {"n_cells": len(rows)}
        for fname, fx in feats.items():
            fit = fit_logistic(fx, k, n)
            predictors[name][fname] = {"spearman": float(stats.spearmanr(fx, rate).statistic) if fx.std() > 0 else None, "pearson": float(stats.pearsonr(fx, rate).statistic) if fx.std() > 0 else None, "logistic_deviance_explained": fit["deviance_explained"], "logistic_nll": fit["nll"]}
        # Two-predictor logistic: word table (other set) + cell residual (cell shift - word mean other set).
        resid = cell_shift - word_other

        def nll2(theta):
            p = np.clip(sigmoid(theta[0] + theta[1] * word_other + theta[2] * resid), 1e-9, 1 - 1e-9)
            return -(k * np.log(p) + (n - k) * np.log(1 - p)).sum()

        p0 = (k.sum() + 0.5) / (n.sum() + 1)
        r2 = optimize.minimize(nll2, [np.log(p0 / (1 - p0)), 0.0, 0.0], method="BFGS")
        null = -(k * np.log(p0) + (n - k) * np.log(1 - p0)).sum()
        predictors[name]["word_other_plus_cell_residual"] = {"coef_word_table": float(r2.x[1]), "coef_cell_residual": float(r2.x[2]), "logistic_deviance_explained": float(1 - r2.fun / null), "logistic_nll": float(r2.fun), "lr_test_residual_adds_over_word_table": {"chi2": float(2 * (predictors[name]["word_mean_other_set"]["logistic_nll"] - r2.fun)), "df": 1, "p": float(stats.chi2.sf(2 * (predictors[name]["word_mean_other_set"]["logistic_nll"] - r2.fun), 1))}}
        # Within-word (across prompts) and within-prompt (across words) Spearman of cell shift vs r(E).
        ww, wp = [], []
        for w in sorted(set(r["word"] for r in rows)):
            sel = np.asarray([r["word"] == w for r in rows])
            if sel.sum() >= 5 and rate[sel].std() > 0 and cell_shift[sel].std() > 0:
                ww.append({"word": w, "n": int(sel.sum()), "spearman": float(stats.spearmanr(cell_shift[sel], rate[sel]).statistic)})
        for i in sorted(set(pi.tolist())):
            sel = pi == i
            if sel.sum() >= 5 and rate[sel].std() > 0:
                wp.append({"prompt_id": pids[i], "n": int(sel.sum()), "spearman": float(stats.spearmanr(cell_shift[sel], rate[sel]).statistic)})
        predictors[name]["within_word_across_prompts"] = {"per_word": ww, "mean_spearman": float(np.mean([d["spearman"] for d in ww])) if ww else None, "n_words": len(ww)}
        predictors[name]["within_prompt_across_words"] = {"per_prompt": wp, "mean_spearman": float(np.mean([d["spearman"] for d in wp])) if wp else None, "n_prompts_with_variation": len(wp)}

    # Shift range with no generated words.
    gen_A = sorted(set((r["word"], r["mean_shift"]) for r in per_word if r["set"] == "A"), key=lambda t: t[1])
    gen_B = sorted(set((r["word"], r["mean_shift"]) for r in per_word if r["set"] == "B"), key=lambda t: t[1])
    twins_A = sorted((c["twin"], float(shift[mA, cidx[c["twin"]]].mean())) for c in words_obj["chosen"])
    twins_A += sorted((r["twin"], float(shift[mA, cidx[r["twin"]]].mean())) for r in pilot_pw.values())
    twins_B = sorted((c["twin"], float(shift[mB, cidx[c["twin"]]].mean())) for c in words_obj["chosen"])

    def gaps(points):
        out = []
        for (w0, s0), (w1, s1) in zip(points[:-1], points[1:]):
            out.append({"from_word": w0, "from_shift": s0, "to_word": w1, "to_shift": s1, "gap_raw": s1 - s0, "gap_sd_units": (s1 - s0) / X106_HARD1K_SD})
        return sorted(out, key=lambda d: -d["gap_raw"])

    bins = np.arange(0, 3.25, 0.25)
    mean_A_sd, mean_B_sd = shift[mA].mean(0) / X106_HARD1K_SD, shift[mB].mean(0) / X106_HARD1K_SD
    coverage = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        coverage.append({"bin_sd_units": [float(lo), float(hi)], "n_vocab_words_set_A_mean": int(((mean_A_sd >= lo) & (mean_A_sd < hi)).sum()), "n_generated_E_words_set_A": int(sum(lo <= s / X106_HARD1K_SD < hi for _, s in gen_A)), "n_vocab_words_set_B_mean": int(((mean_B_sd >= lo) & (mean_B_sd < hi)).sum()), "n_generated_E_words_set_B": int(sum(lo <= s / X106_HARD1K_SD < hi for _, s in gen_B))})
    coverage.append({"bin_sd_units": [3.0, None], "n_vocab_words_set_A_mean": int((mean_A_sd >= 3).sum()), "n_generated_E_words_set_A": int(sum(s / X106_HARD1K_SD >= 3 for _, s in gen_A)), "n_vocab_words_set_B_mean": int((mean_B_sd >= 3).sum()), "n_generated_E_words_set_B": int(sum(s / X106_HARD1K_SD >= 3 for _, s in gen_B))})
    gap_info = {
        "set_A_generated_E_words_sorted": [{"word": w, "mean_shift": s, "sd_units": s / X106_HARD1K_SD} for w, s in gen_A],
        "set_A_largest_gaps_between_generated_E_words": gaps(gen_A)[:3],
        "set_A_twin_N_words_shift_range": [twins_A[0][1], twins_A[-1][1]],
        "set_B_generated_E_words_sorted": [{"word": w, "mean_shift": s, "sd_units": s / X106_HARD1K_SD} for w, s in gen_B],
        "set_B_largest_gaps_between_generated_E_words": gaps(gen_B)[:3],
        "set_B_twin_N_words_shift_range": [twins_B[0][1], twins_B[-1][1]],
        "set_B_gap_between_twins_and_lowest_E_word": {"from": twins_B[-1], "to": gen_B[0], "gap_raw": gen_B[0][1] - twins_B[-1][1]},
        "coverage_by_sd_bin": coverage,
    }
    # Vocabulary words inside the set-A gap (candidates for a follow-up), top of the gap first.
    g = gaps(gen_A)[0]
    in_gap = [j for j in order_all if g["from_shift"] < shift[mA, j].mean() < g["to_shift"]]
    gap_info["set_A_gap_vocab"] = {"n_words": len(in_gap), "examples_top_by_all40_mean": [cand[j] for j in in_gap[:40]]}

    # Noise ceiling: split-half of the 10 samples per cell.
    noise = {}
    for name in ("A_scan10", "A_union25", "B_scan10", "all_scan10", "all_union"):
        rows = cell_sets[name]
        S = np.stack([r["samples"] for r in rows if len(r["samples"]) == 10])
        x = np.asarray([r["shift"] for r in rows if len(r["samples"]) == 10])
        rate = S.mean(1)
        sp, pe, sp_x1, pe_x1 = [], [], [], []
        for seed in HALF_SEEDS:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(10)
            h1, h2 = S[:, perm[:5]].mean(1), S[:, perm[5:]].mean(1)
            sp.append(stats.spearmanr(h1, h2).statistic)
            pe.append(stats.pearsonr(h1, h2).statistic)
            sp_x1.append(stats.spearmanr(x, h1).statistic)
            pe_x1.append(stats.pearsonr(x, h1).statistic)
        r5_p, r5_s = float(np.mean(pe)), float(np.mean(sp))
        r10_p, r10_s = 2 * r5_p / (1 + r5_p), 2 * r5_s / (1 + r5_s)
        # Variance-based reliability: var(rate) = var(true p) + E[p(1-p)/10]; unbiased per-cell noise p(1-p)/9.
        noise_var = float(np.mean(rate * (1 - rate) / 9))
        total_var = float(rate.var(ddof=1))
        rel_var = 1 - noise_var / total_var
        obs_p, obs_s = float(stats.pearsonr(x, rate).statistic), float(stats.spearmanr(x, rate).statistic)
        noise[name] = {
            "n_cells": int(len(S)),
            "half_half_pearson_5_samples_mean": r5_p,
            "half_half_spearman_5_samples_mean": r5_s,
            "reliability_10_samples_spearman_brown_pearson": float(r10_p),
            "reliability_10_samples_spearman_brown_spearman": float(r10_s),
            "reliability_10_samples_variance_based": float(rel_var),
            "ceiling_pearson_for_noiseless_predictor": float(np.sqrt(r10_p)),
            "ceiling_pearson_variance_based": float(np.sqrt(rel_var)),
            "observed_pearson_shift_vs_r_E_10_samples": obs_p,
            "observed_spearman_shift_vs_r_E_10_samples": obs_s,
            "observed_pearson_shift_vs_r_E_5_samples_mean": float(np.mean(pe_x1)),
            "observed_spearman_shift_vs_r_E_5_samples_mean": float(np.mean(sp_x1)),
            "disattenuated_pearson": float(obs_p / np.sqrt(r10_p)),
            "disattenuated_spearman": float(obs_s / np.sqrt(r10_s)),
            "disattenuated_pearson_variance_based": float(obs_p / np.sqrt(rel_var)),
            "shift_vs_half_over_sqrt_half_vs_half_pearson": float(np.mean(pe_x1) / np.sqrt(r5_p)),
            "shift_vs_half_over_sqrt_half_vs_half_spearman": float(np.mean(sp_x1) / np.sqrt(r5_s)),
            "n_half_splits": len(HALF_SEEDS),
        }

    check3 = {"per_word": per_word, "per_word_corr": per_word_corr, "cell_level": cell_level, "curves": curves, "predictors": predictors, "shift_coverage": gap_info, "noise_ceiling": noise}

    results = {
        "run": "x109/wordtable_checks",
        "construct": "refusal2 (an explicit refusal statement is present), x64 judge, DeepSeek V4 Flash",
        "shift": "sum over the two boundary tokens of proj(text + ' ' + w) - proj(text), Qwen3.5-4B boundary direction (scan.json 'projection'); sd unit = x106 Hard-1K sd " + f"{X106_HARD1K_SD:.4f}",
        "n_prompts": int(P),
        "n_words": int(W),
        "check1_word_table_stability": check1,
        "check2_variance_decomposition": check2,
        "check3_dose_response": check3,
        "seeds": {"split_half_prompts": SPLIT_SEEDS, "split_half_samples": HALF_SEEDS},
    }
    atomic_json(OUT / "results.json", results)
    write_md(results)
    print((OUT / "results.md").read_text())


def f(x, d=3):
    return "nan" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:+.{d}f}"


def write_md(r: dict) -> None:
    L = ["# Vocabulary scan: word-table checks (Qwen3.5-4B)\n"]
    L.append(f"Scan matrix: {r['n_prompts']} prompts x {r['n_words']:,} words, shift = {r['shift']}. Refusal is `refusal2` (x64 judge), 10 samples per cell. Δ_EN CIs are the prompt-level paired bootstraps from the pilot's and the scan's results.json.\n")
    c1 = r["check1_word_table_stability"]
    L.append("## 1. Word-table stability\n")
    L.append("Per-word mean shift over the set-A prompts against the set-B prompts; split halves are random 10/10 prompt halves within a set (10 seeds, mean [min, max]). Rows restrict the word list to the top 1,000 / top 100 words by all-40 mean.\n")
    L.append("| scale | words | ρ A vs B | r A vs B | split-half A ρ | split-half A r | split-half B ρ | split-half B r |\n|---|---|---|---|---|---|---|---|")
    for scale in ("raw", "per_prompt_z"):
        for name in ("all", "top1000_by_all40_mean", "top100_by_all40_mean"):
            ab = c1[scale]["set_A_vs_set_B"][name]["corr_A_vs_B"]
            ha, hb = c1[scale]["split_half_within_A"][name], c1[scale]["split_half_within_B"][name]
            L.append(f"| {scale} | {name} | {ab['spearman']:+.3f} | {ab['pearson']:+.3f} | {ha['spearman_mean']:+.3f} [{ha['spearman_min']:+.3f}, {ha['spearman_max']:+.3f}] | {ha['pearson_mean']:+.3f} [{ha['pearson_min']:+.3f}, {ha['pearson_max']:+.3f}] | {hb['spearman_mean']:+.3f} [{hb['spearman_min']:+.3f}, {hb['spearman_max']:+.3f}] | {hb['pearson_mean']:+.3f} [{hb['pearson_min']:+.3f}, {hb['pearson_max']:+.3f}] |")
    L.append("\nSet-B mean against set-A mean (b on a):\n")
    L.append("| scale | words | OLS slope | intercept | through-origin slope | mean A | mean B | ratio B/A |\n|---|---|---|---|---|---|---|---|")
    for scale in ("raw", "per_prompt_z"):
        for name in ("all", "top1000_by_all40_mean", "top100_by_all40_mean"):
            s = c1[scale]["set_A_vs_set_B"][name]["B_vs_A"]
            L.append(f"| {scale} | {name} | {s['ols_slope_b_on_a']:+.3f} | {s['ols_intercept']:+.3f} | {s['through_origin_slope']:+.3f} | {s['mean_a']:+.3f} | {s['mean_b']:+.3f} | {f(s['ratio_of_means'])} |")
    L.append("\nTop-k overlap between the set-A and set-B rankings: " + "; ".join(f"top {k[3:]}: {v['n_common']} common (Jaccard {v['jaccard']:.2f})" for k, v in c1["rank_overlap"].items()) + ".\n")
    c2 = r["check2_variance_decomposition"]
    L.append("## 2. Variance decomposition\n")
    L.append("Two-way additive decomposition of the sum of squares about the grand mean into prompt main effect, word main effect and residual (prompt x word interaction); rank-1 share = share of the uncentred sum of squares carried by the best prompt-scale x word-value product.\n")
    L.append("| scale | words | prompt main | word main | residual | rank-1 share (uncentred) | additive share (uncentred) |\n|---|---|---|---|---|---|---|")
    for scale in ("raw", "per_prompt_z"):
        for name in ("all", "top1000_by_all40_mean", "top100_by_all40_mean"):
            d = c2[scale][name]
            L.append(f"| {scale} | {name} | {d['share_prompt_main']:.3f} | {d['share_word_main']:.3f} | {d['share_residual_interaction']:.3f} | {d['rank1_share_of_uncentred_ss']:.3f} | {d['additive_share_of_uncentred_ss']:.3f} |")
    pf = c2["prompt_factors"]
    L.append("\nPer-prompt factors (mean and sd of the shift over all words) against the prompt's set and shape (Spearman with the 0/1 indicator for B):\n")
    L.append(f"- prompt mean: vs set B {pf['spearman_prompt_mean_vs_set_B']:+.3f}, vs shape B (No Robots) {pf['spearman_prompt_mean_vs_shape_B']:+.3f}; prompt sd: vs set B {pf['spearman_prompt_sd_vs_set_B']:+.3f}, vs shape B {pf['spearman_prompt_sd_vs_shape_B']:+.3f}; prompt mean vs prompt sd {pf['spearman_prompt_mean_vs_prompt_sd']:+.3f}; prompt sd vs O projection {pf['spearman_prompt_sd_vs_o_proj']:+.3f}; prompt sd vs prefix length {pf['spearman_prompt_sd_vs_prefix_tokens']:+.3f}.")
    L.append(f"- Mann-Whitney p for prompt sd, set A vs B: {pf['mannwhitney_sd_A_vs_B_p']:.4f}; shape A vs B: {pf['mannwhitney_sd_shapeA_vs_shapeB_p']:.4f}.")
    L.append("- sd quantiles (min, q1, median, q3, max): set A " + ", ".join(f"{v:.2f}" for v in pf["sd_quantiles"]["set_A"]) + "; set B " + ", ".join(f"{v:.2f}" for v in pf["sd_quantiles"]["set_B"]) + ".\n")
    L.append("| group | mean of prompt means | mean of prompt sds | sd min | sd max |\n|---|---|---|---|---|")
    for g, v in pf["group_means"].items():
        L.append(f"| {g} | {v['prompt_mean']:+.3f} | {v['prompt_sd']:.3f} | {v['prompt_sd_min']:.3f} | {v['prompt_sd_max']:.3f} |")
    L.append("\n| prompt | set | shape | prefix tokens | O proj | mean shift | sd shift | top-10 word mean shift |\n|---|---|---|---|---|---|---|---|")
    for p in pf["per_prompt"]:
        L.append(f"| {p['prompt_id']} | {p['set']} | {p['shape']} | {p['n_prefix_tokens']} | {p['o_proj_cached']:+.2f} | {p['mean_shift_over_words']:+.3f} | {p['sd_shift_over_words']:.3f} | {p['top10_scan_word_mean_shift']:+.2f} |")
    c3 = r["check3_dose_response"]
    L.append("\n## 3. Dose-response\n")
    L.append("Every word with generation data, sorted by mean scanned shift within its prompt set. Set A: the pilot's 15 words (pilot Δ_EN) and the scan's 10 (scan Δ_EN); set B: the scan's 10.\n")
    L.append("| set | word | source | twin | mean shift | sd units | r(E) | r(N) | Δ_EN [95% CI] |\n|---|---|---|---|---|---|---|---|---|")
    for w in c3["per_word"]:
        L.append(f"| {w['set']} | {w['word']} | {w['source']} | {w['twin']} | {w['mean_shift']:+.2f} | {w['mean_shift_sd_units']:+.2f} | {w['r_E']:.3f} | {w['r_N']:.3f} | {f(w['delta_EN'])} [{f(w['delta_EN_ci95'][0])}, {f(w['delta_EN_ci95'][1])}] |")
    L.append("\nPer-word correlations of mean shift with Δ_EN and r(E):\n")
    L.append("| table | n words | ρ vs Δ_EN | r vs Δ_EN | ρ vs r(E) | r vs r(E) |\n|---|---|---|---|---|---|")
    for k, v in c3["per_word_corr"].items():
        L.append(f"| {k} | {v['vs_delta_EN']['n']} | {v['vs_delta_EN']['spearman']:+.3f} | {v['vs_delta_EN']['pearson']:+.3f} | {v['vs_r_E']['spearman']:+.3f} | {v['vs_r_E']['pearson']:+.3f} |")
    L.append("\nCell-level (prompt x word) logistic fits of r(E) on the per-cell shift (binomial likelihood, 10 samples per cell). 2-param: rate = σ(b0 + b1·shift); 3-param: rate = c·σ(b0 + b1·shift) with free asymptote c.\n")
    L.append("| cells | n | ρ shift vs r(E) | r | 2p midpoint raw (sd) | 2p slope /sd unit | 2p deviance expl. | 3p asymptote | 3p midpoint raw (sd) | 3p slope /sd unit | 3p deviance expl. | shift range |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k, v in c3["cell_level"].items():
        a, b = v["logistic_2param"], v["logistic_3param_asymptote"]
        L.append(f"| {k} | {v['n_cells']} | {v['spearman_shift_vs_r_E']:+.3f} | {v['pearson_shift_vs_r_E']:+.3f} | {a['midpoint_raw']:+.2f} ({a['midpoint_sd_units']:+.2f}) | {a['slope_per_sd_unit']:+.3f} | {a['deviance_explained']:.3f} | {b['asymptote']:.3f} | {b['midpoint_raw']:+.2f} ({b['midpoint_sd_units']:+.2f}) | {b['slope_per_sd_unit']:+.3f} | {b['deviance_explained']:.3f} | [{a['x_min']:+.1f}, {a['x_max']:+.1f}] |")
    L.append("\nWord table against prompt-level projection as cell-level predictors of r(E). `word_mean_other_set` is the word's mean scanned shift over the prompts of the other set (a lookup with no access to the cell's own prompt); `cell_shift` is the cell's own projection shift; `word_other_plus_cell_residual` is a two-predictor logistic fit with the likelihood-ratio test of the residual term. In the pooled `all_scan10` table the lookup crosses sets (set-A cells read the set-B means, set-B cells the set-A means), so it carries the between-set scale difference with the opposite sign; the within-set tables are the word-table comparison.\n")
    for name, pv in c3["predictors"].items():
        L.append(f"\n{name} (n = {pv['n_cells']} cells):\n")
        L.append("| predictor | ρ vs r(E) | r vs r(E) | logistic deviance explained |\n|---|---|---|---|")
        for fname in ("cell_shift", "cell_shift_per_prompt_z", "word_mean_other_set", "word_mean_same_set", "prompt_mean_over_all_words", "prompt_sd_over_all_words", "word_other_plus_prompt_mean", "word_other_times_prompt_sd"):
            v = pv[fname]
            L.append(f"| {fname} | {f(v['spearman'])} | {f(v['pearson'])} | {v['logistic_deviance_explained']:.3f} |")
        t = pv["word_other_plus_cell_residual"]
        L.append(f"| word_other_plus_cell_residual | | | {t['logistic_deviance_explained']:.3f} (LR χ² = {t['lr_test_residual_adds_over_word_table']['chi2']:.1f}, p = {t['lr_test_residual_adds_over_word_table']['p']:.2e}) |")
        ww, wp = pv["within_word_across_prompts"], pv["within_prompt_across_words"]
        L.append(f"\nWithin-word (across prompts) Spearman of cell shift vs r(E): mean {f(ww['mean_spearman'])} over {ww['n_words']} words (" + ", ".join(f"{d['word']} {d['spearman']:+.2f}" for d in ww["per_word"]) + f"). Within-prompt (across words): mean {f(wp['mean_spearman'])} over {wp['n_prompts_with_variation']} prompts with r(E) variation.")
    sc = c3["shift_coverage"]
    L.append("\n### Shift range with no generated words\n")
    ga = sc["set_A_largest_gaps_between_generated_E_words"][0]
    gb = sc["set_B_largest_gaps_between_generated_E_words"][0]
    gbt = sc["set_B_gap_between_twins_and_lowest_E_word"]
    L.append(f"- Set A (25 generated E words): the largest gap between consecutive generated words runs from {ga['from_word']} ({ga['from_shift']:+.2f}, {ga['from_shift'] / X106_HARD1K_SD:+.2f} sd) to {ga['to_word']} ({ga['to_shift']:+.2f}, {ga['to_shift'] / X106_HARD1K_SD:+.2f} sd), width {ga['gap_raw']:.2f} raw = {ga['gap_sd_units']:.2f} sd; {sc['set_A_gap_vocab']['n_words']} vocabulary words have a set-A mean inside it (top by all-40 mean: {', '.join(sc['set_A_gap_vocab']['examples_top_by_all40_mean'][:25])}). Twin (N) words span {sc['set_A_twin_N_words_shift_range'][0]:+.2f} to {sc['set_A_twin_N_words_shift_range'][1]:+.2f}.")
    L.append(f"- Set B (10 generated E words): E words span {sc['set_B_generated_E_words_sorted'][0]['mean_shift']:+.2f} to {sc['set_B_generated_E_words_sorted'][-1]['mean_shift']:+.2f} ({sc['set_B_generated_E_words_sorted'][0]['sd_units']:+.2f} to {sc['set_B_generated_E_words_sorted'][-1]['sd_units']:+.2f} sd); twins span {sc['set_B_twin_N_words_shift_range'][0]:+.2f} to {sc['set_B_twin_N_words_shift_range'][1]:+.2f}; nothing was generated between {gbt['from'][0]} ({gbt['from'][1]:+.2f}) and {gbt['to'][0]} ({gbt['to'][1]:+.2f}), a gap of {gbt['gap_raw']:.2f} raw = {gbt['gap_raw'] / X106_HARD1K_SD:.2f} sd. Largest gap among the E words: {gb['from_word']} to {gb['to_word']}, {gb['gap_raw']:.2f} raw.\n")
    L.append("| sd bin | vocab words (set-A mean) | generated E words, set A | vocab words (set-B mean) | generated E words, set B |\n|---|---|---|---|---|")
    for b in sc["coverage_by_sd_bin"]:
        lo, hi = b["bin_sd_units"]
        L.append(f"| [{lo:.2f}, {'∞' if hi is None else f'{hi:.2f}'}) | {b['n_vocab_words_set_A_mean']:,} | {b['n_generated_E_words_set_A']} | {b['n_vocab_words_set_B_mean']:,} | {b['n_generated_E_words_set_B']} |")
    L.append("\n### Noise ceiling of the cell-level correlation\n")
    L.append("The 10 samples per cell are split into two random halves of 5 (20 splits); half-half correlation across cells, Spearman-Brown to 10 samples, the ceiling for a noiseless predictor (sqrt of the reliability), and the observed shift–r(E) correlation disattenuated by it. Variance-based reliability: 1 − mean(p̂(1−p̂)/9) / var(r̂).\n")
    L.append("| cells | n | half-half r (5) | half-half ρ (5) | reliability r (10, S-B) | reliability (variance) | ceiling r | observed r | observed ρ | disattenuated r | disattenuated ρ | r(shift, half) / sqrt r(half, half) |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k, v in c3["noise_ceiling"].items():
        L.append(f"| {k} | {v['n_cells']} | {v['half_half_pearson_5_samples_mean']:.3f} | {v['half_half_spearman_5_samples_mean']:.3f} | {v['reliability_10_samples_spearman_brown_pearson']:.3f} | {v['reliability_10_samples_variance_based']:.3f} | {v['ceiling_pearson_for_noiseless_predictor']:.3f} | {v['observed_pearson_shift_vs_r_E_10_samples']:+.3f} | {v['observed_spearman_shift_vs_r_E_10_samples']:+.3f} | {v['disattenuated_pearson']:+.3f} | {v['disattenuated_spearman']:+.3f} | {v['shift_vs_half_over_sqrt_half_vs_half_pearson']:.3f} |")
    (OUT / "results.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
