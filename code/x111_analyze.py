"""x111 analysis (local, numpy / scipy).

refusal2 (x64 judge) per (prompt, condition) over twenty samples; per population (benign, harmful
test prompts) and pooled: r(O); per word r(E) and Delta_EO with a prompt-level paired bootstrap;
the dose-response curve over the eight concordant bins; for each fixed table (benign, BP, HP) and
for the in-context key-word shift: Spearman rho with the per-word Delta_EO (all 40 words, the 24
concordant words, the middle four bins), bootstrap CI and permutation p, partial rho with token
count and log wordfreq rank held fixed, and the per-cell rho against the split-half noise ceiling;
the discordant-word comparison of the benign and BP tables; the hijack readout per bin.

By default the prompts listed in data/runs/x111/x85_exclusions.json (the nine harmful test prompts
that belong to the x85 cohort the Qwen3.5-4B boundary direction was fitted on) are left out, and the
analysis of the remaining 55 prompts is written to results.json, results.md, dose_response.png and
tables_vs_effect.png. The harmful stratum is topped back up to 32 by nine prompts outside the x85
cohort (prompts_topup.json; x111_select.py --topup-request / --topup-prompts): by default the
analysis runs on the 55 retained prompts plus the nine top-up prompts (32 benign + 32 harmful), whose
cells, judgments and key-word scan (key_words_scan_topup.json) are read beside the originals, and
results.json carries `cohort` and `topup` blocks. `--no-topup` analyses the 55 retained prompts alone
into results_pre_topup.{json,md}, dose_response_pre_topup.png and tables_vs_effect_pre_topup.png.
`--full` runs the same analysis on the original 64 prompts, without the top-up, and writes
results_full.json, results_full.md, dose_response_full.png and tables_vs_effect_full.png. Seeds,
bootstrap, permutation and Holm procedures are the same in all three.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x111_common import (  # noqa: E402
    JUDGE_DIR,
    KEY_JSON,
    KEY_JSON_TOPUP,
    KEY_NPZ,
    KEY_NPZ_TOPUP,
    N_HALF,
    N_SAMPLES,
    POPULATIONS,
    PROMPTS,
    PROMPTS_TOPUP,
    RESULTS,
    RESULTS_MD,
    RUN,
    ANALYSIS_TABLES as TABLES,
    TABLES_JSON,
    TABLES_NPZ,
    WORDS,
    X106_HARD1K_SD,
    atomic_json,
    cells,
    read_json,
)
from x111_judge import load_generations  # noqa: E402
from x111_select import BIN_NAMES  # noqa: E402

N_BOOT = 10_000
N_PERM = 10_000
BOOT_SEED = 111_000
HALF_SEEDS = list(range(20))
MIDDLE_BINS = (2, 3, 4, 5)
SETS = ("benign", "harmful", "pooled")
EXCLUSIONS = RUN / "x85_exclusions.json"


def load_units(path: Path) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if not rec.get("failed"):
                out[rec["cell_id"]] = rec
    return out


def ci(x) -> list[float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(v) for v in np.quantile(x, (0.025, 0.975))] if len(x) else [float("nan")] * 2


def boot_p(samples: np.ndarray) -> float:
    b = len(samples)
    low = (np.sum(samples <= 0) + 1) / (b + 1)
    high = (np.sum(samples >= 0) + 1) / (b + 1)
    return float(min(1.0, 2 * min(low, high)))


def holm(p: list[float]) -> list[float]:
    order = np.argsort(p)
    out = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        out[i] = running
    return out.tolist()


def rho(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return float("nan")
    return float(stats.spearmanr(x[ok], y[ok]).statistic)


def rho_rows(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Spearman of a fixed vector x with every row of Y (rows = bootstrap draws)."""
    rx = stats.rankdata(x)
    rx = (rx - rx.mean()) / rx.std()
    rY = stats.rankdata(Y, axis=1)
    rY = (rY - rY.mean(1, keepdims=True)) / rY.std(1, keepdims=True)
    return (rY * rx).mean(1)


def perm_p(x, y, n=N_PERM, seed=1) -> float:
    rng = np.random.default_rng(seed)
    obs = abs(rho(x, y))
    y = np.asarray(y, float)
    P = np.stack([rng.permutation(y) for _ in range(n)])
    return float((1 + np.sum(np.abs(rho_rows(np.asarray(x, float), P)) >= obs - 1e-12)) / (n + 1))


def partial_rho(x, y, covs: list) -> float:
    """Spearman partial correlation: Pearson of the rank residuals after regressing out the covariate ranks."""
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    C = np.column_stack([np.ones(len(rx))] + [stats.rankdata(c) for c in covs])
    res = lambda v: v - C @ np.linalg.lstsq(C, v, rcond=None)[0]  # noqa: E731
    return float(np.corrcoef(res(rx), res(ry))[0, 1])


def main(exclude: set[str], suffix: str, topup: bool = False, cohort: str = "") -> None:
    prompts_obj = read_json(PROMPTS)
    all_prompts = prompts_obj["prompts"]
    all_pids = [p["id"] for p in all_prompts]
    for pid in exclude:
        if pid not in all_pids:
            raise SystemExit(f"FATAL: excluded id {pid} is not an x111 prompt id")
    keep_idx = [i for i, p in enumerate(all_prompts) if p["id"] not in exclude]
    prompts = [all_prompts[i] for i in keep_idx]
    topup_obj = read_json(PROMPTS_TOPUP) if topup else None
    if topup_obj:
        if set(all_pids) & {p["id"] for p in topup_obj["prompts"]} or set(exclude) & {p["id"] for p in topup_obj["prompts"]}:
            raise SystemExit("FATAL: a top-up prompt repeats an original prompt")
        prompts = prompts + list(topup_obj["prompts"])
    prompts_obj = {**prompts_obj, "prompts": prompts}
    words_obj = read_json(WORDS)
    chosen = words_obj["chosen"]
    cw = [c["word"] for c in chosen]
    tz = np.load(TABLES_NPZ)
    tmeta = read_json(TABLES_JSON)
    gens, _, _ = load_generations()
    if topup_obj:
        gens = gens + load_generations(topup=True)[0]
    gen = {g["cell_id"]: g for g in gens}
    ref = {u: rec for u, rec in load_units(JUDGE_DIR / "refusal2.jsonl").items() if u.rsplit("#h", 1)[0] in gen}
    hij_reason = load_units(JUDGE_DIR / "hijack.jsonl")
    hij_fast = load_units(JUDGE_DIR / "hijack_fast.jsonl")
    hij = hij_fast if hij_fast else hij_reason
    both = [u for u in hij_fast if u in hij_reason and hij_fast[u]["response_sha256"] == hij_reason[u]["response_sha256"]]
    hijack_agreement = None
    if both:
        a = np.asarray([v for u in both for v in hij_fast[u]["hijack_llm"]], bool)
        b = np.asarray([v for u in both for v in hij_reason[u]["hijack_llm"]], bool)
        hijack_agreement = {"n_units": len(both), "n_responses": int(len(a)), "agreement": float((a == b).mean()), "rate_no_reasoning": float(a.mean()), "rate_reasoning": float(b.mean()), "cohen_kappa": float(((a == b).mean() - (a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))) / (1 - (a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))))}
    grid = cells(prompts_obj, words_obj)
    for uid, rec in ref.items():
        cid, h = uid.rsplit("#h", 1)
        if cid in gen and rec["response_sha256"] != gen[cid]["response_sha256"][int(h) * N_HALF : (int(h) + 1) * N_HALF]:
            raise SystemExit(f"FATAL: refusal2 record {uid} does not match its responses")

    def labels(store: dict, cid: str, key: str):
        parts = [store.get(f"{cid}#h{h}") for h in range(N_SAMPLES // N_HALF)]
        if any(p is None for p in parts):
            return None
        return np.asarray(sum((list(p[key]) for p in parts), []), float)

    missing_gen = [c["cell_id"] for c in grid if c["cell_id"] not in gen]
    missing_ref = [c["cell_id"] for c in grid if c["cell_id"] in gen and labels(ref, c["cell_id"], "refusal") is None]
    pids = [p["id"] for p in prompts]
    pops = np.asarray([p["population"] for p in prompts])
    P, W = len(pids), len(cw)
    LO = np.full((P, N_SAMPLES), np.nan)
    LE = np.full((W, P, N_SAMPLES), np.nan)
    for j, pid in enumerate(pids):
        v = labels(ref, f"{pid}|O", "refusal")
        if v is not None:
            LO[j] = v
        for k, w in enumerate(cw):
            v = labels(ref, f"{pid}|E_{w}", "refusal")
            if v is not None:
                LE[k, j] = v
    rO = LO.mean(1)
    rE = LE.mean(2)  # [W, P]
    d = rE - rO[None]
    masks = {"benign": pops == "benign", "harmful": pops == "harmful", "pooled": np.ones(P, bool)}
    hist = np.asarray([p["historical_refusal2"] for p in prompts])

    kind = np.asarray([c["kind"] for c in chosen])
    bins = np.asarray([c["bin"] if c["bin"] is not None else -1 for c in chosen])
    conc = kind == "concordant"
    mid = np.isin(bins, MIDDLE_BINS)
    tab = {t: np.asarray([float(tz[t][c["index"]]) for c in chosen]) for t in TABLES}
    n_tok = np.asarray([c["n_tokens"] for c in chosen], float)
    log_wf = np.log(np.asarray([c["wordfreq_rank"] if c["wordfreq_rank"] else 50_001 for c in chosen], float))
    for c in chosen:
        i = c["index"]
        for t in ("benign", "BP", "HP"):
            if abs(float(tz[t][i]) - c[f"{t}_sd"]) > 1e-5:
                raise SystemExit(f"FATAL: words.json table value of {c['word']} differs from tables.npz")

    # Key-word shift (in context): [P, W] in sd units.
    key = None
    if KEY_JSON.exists() and KEY_NPZ.exists():
        kj = read_json(KEY_JSON)
        if kj["prompt_ids"] != all_pids or kj["candidates"]["words"] != cw:
            raise SystemExit("FATAL: key-word scan does not match prompts / words order")
        if kj["prompt_indices_scanned"] == list(range(len(all_pids))):
            key = (np.load(KEY_NPZ)["shift"].astype(np.float64).sum(axis=2) / X106_HARD1K_SD)[keep_idx]
        key_checks = {"o_cached_vs_full_max_abs_per_token": kj["o_cached_vs_full_max_abs_per_token"], "scan_minutes": float(sum(kj["seconds_per_prompt"]) / 60), "prefix_tokens_max": int(max(kj["prefix_tokens"])), "n_prompts_scanned": kj["n_prompts_scanned"]}
        if topup_obj:
            kt = read_json(KEY_JSON_TOPUP) if KEY_JSON_TOPUP.exists() and KEY_NPZ_TOPUP.exists() else None
            t_ids = [p["id"] for p in topup_obj["prompts"]]
            if kt is None or kt["prompt_indices_scanned"] != list(range(len(t_ids))):
                print("WARNING: the top-up prompts have no complete key-word scan; key_in_context omitted", file=sys.stderr)
                key = None
            else:
                if kt["prompt_ids"] != t_ids or kt["candidates"]["words"] != cw:
                    raise SystemExit("FATAL: top-up key-word scan does not match prompts_topup / words order")
                if key is not None:
                    key = np.concatenate([key, np.load(KEY_NPZ_TOPUP)["shift"].astype(np.float64).sum(axis=2) / X106_HARD1K_SD], axis=0)
                key_checks["topup"] = {"o_cached_vs_full_max_abs_per_token": kt["o_cached_vs_full_max_abs_per_token"], "scan_minutes": float(sum(kt["seconds_per_prompt"]) / 60), "prefix_tokens_max": int(max(kt["prefix_tokens"])), "n_prompts_scanned": kt["n_prompts_scanned"]}
    else:
        key_checks = None

    rng = np.random.default_rng(BOOT_SEED)
    boot_idx = {}
    for s, m in masks.items():
        sub = np.where(m)[0]
        if s == "pooled":
            # stratified: resample within each population
            idx = np.concatenate([np.where(masks[q])[0][rng.integers(0, masks[q].sum(), size=(N_BOOT, masks[q].sum()))] for q in POPULATIONS], axis=1)
        else:
            idx = sub[rng.integers(0, len(sub), size=(N_BOOT, len(sub)))]
        boot_idx[s] = idx

    def bmean(M: np.ndarray, s: str) -> np.ndarray:
        """M [..., P] -> bootstrap means [N_BOOT, ...]."""
        return np.moveaxis(np.nanmean(M[..., boot_idx[s]], axis=-1), -1, 0)

    r_O = {s: {"run": float(np.nanmean(rO[m])), "historical": float(hist[m].mean()), "spearman_run_vs_historical": rho(hist[m], rO[m]), "n_prompts": int(m.sum())} for s, m in masks.items()}

    # Per word.
    per_word = []
    dEO_boot = {s: bmean(d, s) for s in SETS}  # [B, W]
    for k, c in enumerate(chosen):
        rec = {kk: c[kk] for kk in ("word", "stratum", "kind", "bin_name", "quadrant", "benign_sd", "BP_sd", "HP_sd", "pct_benign", "pct_BP", "pct_HP", "n_tokens", "wordfreq_rank")}
        rec["mixed_sd"] = float(tab["mixed"][k])
        rec["pct_mixed"] = float(tz["pct_mixed"][c["index"]])
        for s, m in masks.items():
            b = dEO_boot[s][:, k]
            rec[s] = {
                "r_E": float(np.nanmean(rE[k][m])),
                "delta_EO": float(np.nanmean(d[k][m])),
                "delta_EO_ci95": ci(b),
                "delta_EO_p": boot_p(b),
                "n_prompts_E_below_O": int(np.nansum(d[k][m] < 0)),
                "n_prompts_E_above_O": int(np.nansum(d[k][m] > 0)),
            }
            if key is not None:
                rec[s]["key_shift_sd"] = float(key[m, k].mean())
        per_word.append(rec)
    for s in SETS:
        for pw, v in zip(per_word, holm([pw[s]["delta_EO_p"] for pw in per_word])):
            pw[s]["delta_EO_p_holm40"] = v

    # Dose-response over the eight concordant bins.
    dose = {}
    for s, m in masks.items():
        rows = []
        bin_boot = []
        for b in range(len(BIN_NAMES)):
            sel = conc & (bins == b)
            bb = dEO_boot[s][:, sel].mean(1)
            bin_boot.append(bb)
            rows.append({"bin": BIN_NAMES[b], "words": [cw[k] for k in np.where(sel)[0]], "benign_sd_mean": float(tab["benign"][sel].mean()), "BP_sd_mean": float(tab["BP"][sel].mean()), "HP_sd_mean": float(tab["HP"][sel].mean()), "mixed_sd_mean": float(tab["mixed"][sel].mean()), "r_E": float(np.nanmean(rE[sel][:, m])), "delta_EO": float(np.nanmean(d[sel][:, m])), "delta_EO_ci95": ci(bb), "delta_EO_p": boot_p(bb)})
        est = np.asarray([r["delta_EO"] for r in rows])
        BB = np.stack(bin_boot, 1)  # [B, 8]
        dose[s] = {
            "bins": rows,
            "adjacent_pairs_in_order": int(np.sum(np.diff(est) > 0)),
            "adjacent_pairs_total": len(BIN_NAMES) - 1,
            "fraction_adjacent_in_order": float(np.mean(np.diff(est) > 0)),
            "fraction_adjacent_in_order_boot_mean": float(np.mean(np.diff(BB, axis=1) > 0)),
            "spearman_bin_index_vs_delta": rho(np.arange(len(BIN_NAMES)), est),
            "top_minus_bottom_bin": float(est[-1] - est[0]),
            "top_minus_bottom_bin_ci95": ci(BB[:, -1] - BB[:, 0]),
            "middle_four_bins_range": float(est[list(MIDDLE_BINS)].max() - est[list(MIDDLE_BINS)].min()),
        }

    # Per-word rho: table / key shift vs Delta_EO.
    predictors = {f"table_{t}": tab[t] for t in TABLES}
    if key is not None:
        predictors["key_in_context"] = None  # per set, mean over the set's prompts
    subsets = {"all40": np.ones(W, bool), "concordant24": conc, "middle4bins": conc & mid}
    rho_table = {}
    for s, m in masks.items():
        dEO = np.nanmean(d[:, m], axis=1)
        rho_table[s] = {}
        for name in predictors:
            x = key[m].mean(0) if name == "key_in_context" else predictors[name]
            entry = {}
            for sub, sm in subsets.items():
                xb = x[sm]
                Yb = dEO_boot[s][:, sm]
                okb = np.all(np.isfinite(Yb), axis=1)
                boot = rho_rows(xb, Yb[okb])
                entry[sub] = {"n_words": int(sm.sum()), "rho": rho(xb, dEO[sm]), "ci95_prompt_bootstrap": ci(boot), "perm_p": perm_p(xb, dEO[sm])}
            entry["partial_rho_all40_given_ntokens_logfreq"] = partial_rho(x, dEO, [n_tok, log_wf])
            # per cell
            if name == "key_in_context":
                xc = key[m].T  # [W, P_s]
            else:
                xc = np.repeat(x[:, None], m.sum(), axis=1)
            dc = d[:, m]
            ok = np.isfinite(dc)
            entry["per_cell"] = {"n_cells": int(ok.sum()), "rho": rho(xc[ok], dc[ok])}
            rho_table[s][name] = entry

    # Split-half noise ceiling of the per-cell r(E) - r(O).
    ceiling = {}
    for s, m in masks.items():
        LEs, LOs = LE[:, m], LO[m]
        ok = np.isfinite(LEs).all(2) & np.isfinite(LOs).all(1)[None]
        hh_s, hh_p = [], []
        for seed in HALF_SEEDS:
            perm = np.random.default_rng(seed).permutation(N_SAMPLES)
            a, b = perm[: N_SAMPLES // 2], perm[N_SAMPLES // 2 :]
            d1 = LEs[..., a].mean(-1) - LOs[None, :, a].mean(-1)
            d2 = LEs[..., b].mean(-1) - LOs[None, :, b].mean(-1)
            hh_s.append(stats.spearmanr(d1[ok], d2[ok]).statistic)
            hh_p.append(stats.pearsonr(d1[ok], d2[ok]).statistic)
        r10s, r10p = float(np.mean(hh_s)), float(np.mean(hh_p))
        rel_s, rel_p = 2 * r10s / (1 + r10s), 2 * r10p / (1 + r10p)
        ceiling[s] = {"n_cells": int(ok.sum()), "half_half_spearman_10_samples": r10s, "half_half_pearson_10_samples": r10p, "reliability_20_samples_spearman_brown_spearman": rel_s, "reliability_20_samples_spearman_brown_pearson": rel_p, "ceiling_spearman": float(np.sqrt(max(rel_s, 0))), "ceiling_pearson": float(np.sqrt(max(rel_p, 0))), "n_half_splits": len(HALF_SEEDS)}

    # Discordant words: which table's position predicts Delta_EO (linear fit on the 24 concordant words).
    discordant = {}
    disc_idx = np.where(kind == "discordant")[0]
    for s, m in masks.items():
        dEO = np.nanmean(d[:, m], axis=1)
        fits = {}
        for t in TABLES:
            slope, intercept = np.polyfit(tab[t][conc], dEO[conc], 1)
            fits[t] = {"slope": float(slope), "intercept": float(intercept), "pred": slope * tab[t] + intercept}
        rows = []
        counts = {"closer_benign": 0, "closer_BP": 0}
        sign_counts = {t: 0 for t in TABLES}
        for k in disc_idx:
            preds = {t: float(fits[t]["pred"][k]) for t in TABLES}
            err = {t: abs(preds[t] - dEO[k]) for t in TABLES}
            closer = "benign" if err["benign"] < err["BP"] else "BP"
            counts[f"closer_{closer}"] += 1
            sgn = {t: bool(np.sign(preds[t] - dEO[conc].mean()) == np.sign(dEO[k] - dEO[conc].mean())) for t in TABLES}
            for t in TABLES:
                sign_counts[t] += sgn[t]
            rows.append({"word": cw[k], "quadrant": chosen[k]["quadrant"], "pct_benign": chosen[k]["pct_benign"], "pct_BP": chosen[k]["pct_BP"], "pct_HP": chosen[k]["pct_HP"], "pct_mixed": float(tz["pct_mixed"][chosen[k]["index"]]), "delta_EO": float(dEO[k]), "delta_EO_ci95": per_word[k][s]["delta_EO_ci95"], "pred": preds, "abs_err": err, "closer_of_benign_BP": closer, "best_of_four": min(err, key=err.get), "side_match": sgn})
        quad = {}
        for qname in sorted({chosen[k]["quadrant"] for k in disc_idx}):
            qs = [k for k in disc_idx if chosen[k]["quadrant"] == qname]
            qb = dEO_boot[s][:, qs].mean(1)
            quad[qname] = {"words": [cw[k] for k in qs], "delta_EO": float(dEO[qs].mean()), "delta_EO_ci95": ci(qb)}
        discordant[s] = {
            "rule": "per table, a least-squares line of Delta_EO on the table shift over the 24 concordant words; each discordant word's Delta_EO is compared with each table's prediction (absolute error); side match = prediction and observation on the same side of the concordant-word mean Delta_EO",
            "fits": {t: {"slope": fits[t]["slope"], "intercept": fits[t]["intercept"]} for t in TABLES},
            "words": rows,
            "count_closer": counts,
            "count_best_of_four": {t: sum(1 for r in rows if r["best_of_four"] == t) for t in TABLES},
            "count_side_match": sign_counts,
            "sum_abs_err": {t: float(sum(r["abs_err"][t] for r in rows)) for t in TABLES},
            "quadrant_means": quad,
        }

    # Hijack per bin.
    hij_rows = []
    hij_cov = {"n_E_units": sum(1 for c in grid if c["kind"] == "E") * (N_SAMPLES // N_HALF), "n_units_judged": 0}
    HJ = np.full((W, P, N_SAMPLES), np.nan)
    for k, w in enumerate(cw):
        for j, pid in enumerate(pids):
            v = labels(hij, f"{pid}|E_{w}", "hijack_llm")
            if v is not None:
                HJ[k, j] = v
    hij_cov["n_units_judged"] = sum(1 for u in hij if "|E_" in u and u.split("|", 1)[0] in set(pids))
    hij_cov["n_cells_complete"] = int(np.isfinite(HJ).all(2).sum())
    groups = [(BIN_NAMES[b], conc & (bins == b)) for b in range(len(BIN_NAMES))] + [(f"discordant {q}", np.asarray([c["quadrant"] == q for c in chosen])) for q in sorted({c["quadrant"] for c in chosen if c["quadrant"]})] + [("all 40", np.ones(W, bool))]
    for gname, sel in groups:
        row = {"group": gname}
        for s, m in masks.items():
            h = HJ[sel][:, m].ravel()
            r = LE[sel][:, m].ravel()
            ok = np.isfinite(h) & np.isfinite(r)
            h, r = h[ok].astype(bool), r[ok]
            row[s] = {"n_responses": int(ok.sum()), "hijack_rate": float(h.mean()) if len(h) else None, "refusal2_given_hijack": float(r[h].mean()) if h.any() else None, "refusal2_given_no_hijack": float(r[~h].mean()) if (~h).any() else None}
        hij_rows.append(row)
    hij_word = []
    for k, w in enumerate(cw):
        rec = {"word": w}
        for s, m in masks.items():
            h = HJ[k][m].ravel()
            rec[s] = float(np.nanmean(h)) if np.isfinite(h).any() else None
        hij_word.append(rec)
    if np.isfinite(HJ).any():
        for s, m in masks.items():
            hw = np.asarray([hw[s] for hw in hij_word], float)
            dEO = np.nanmean(d[:, m], axis=1)
            rho_table[s]["hijack_rate_vs_delta_EO"] = rho(hw, dEO)
            rho_table[s]["hijack_rate_vs_table_benign"] = rho(hw, tab["benign"])

    n_pop = {q: int(masks[q].sum()) for q in POPULATIONS}
    per_pop = f"each population's {n_pop['benign']}" if len(set(n_pop.values())) == 1 else "each population (" + ", ".join(f"{q} {n}" for q, n in n_pop.items()) + ")"
    table_choice = {"rule": f"Spearman rho between the table's per-word shift and the per-word Delta_EO over the 40 words; Delta_EO averaged over all {P} test prompts (pooled), and separately over {per_pop}; CI from the prompt-level bootstrap (pooled: resampled within each population)"}
    for t in TABLES:
        e = rho_table["pooled"][f"table_{t}"]["all40"]
        table_choice[t] = {"rho": e["rho"], "ci": e["ci95_prompt_bootstrap"], "perm_p": e["perm_p"], "rho_benign_prompts": rho_table["benign"][f"table_{t}"]["all40"]["rho"], "ci_benign_prompts": rho_table["benign"][f"table_{t}"]["all40"]["ci95_prompt_bootstrap"], "rho_harmful_prompts": rho_table["harmful"][f"table_{t}"]["all40"]["rho"], "ci_harmful_prompts": rho_table["harmful"][f"table_{t}"]["all40"]["ci95_prompt_bootstrap"]}

    topup_block = None
    if topup_obj:
        tu = topup_obj["topup"]
        topup_block = {
            "file": str(PROMPTS_TOPUP.relative_to(RUN.parents[2])),
            "rule": topup_obj["rule"],
            "n_added": len(topup_obj["prompts"]),
            "added": [{k: p[k] for k in ("id", "population", "historical_refusal2", "rank", "x83_word", "x83_category")} for p in topup_obj["prompts"]],
            **{k: tu[k] for k in ("n_chosen_original", "n_retained_after_x85", "n_needed", "rate_counts_retained", "rate_counts_added", "pool")},
            "prompts_sha256": topup_obj["prompts_sha256"],
            "n_prompts_by_population": n_pop,
        }
    results = {
        "run": "x111",
        **({"cohort": cohort} if cohort else {}),
        **({"prompt_filter": {"file": str(EXCLUSIONS.relative_to(RUN.parents[2])), "excluded_ids": sorted(exclude), "n_excluded": len(exclude), "n_kept": len(keep_idx), "n_kept_by_population": {q: sum(1 for i in keep_idx if all_prompts[i]["population"] == q) for q in POPULATIONS}}} if exclude else {}),
        **({"topup": topup_block} if topup_block else {}),
        "table_choice": table_choice,
        "construct": "refusal2 (an explicit refusal statement is present), x64 judge, DeepSeek V4 Flash; twenty samples per cell judged as two units of ten",
        "n_prompts": P,
        "n_words": W,
        "missing_generations": missing_gen,
        "missing_refusal2": missing_ref,
        "n_refusal2_units": len(ref),
        "n_refusal2_per_response_fallback": sum(1 for v in ref.values() if v.get("judge_mode") == "per_response_fallback"),
        "n_cells_retry4096": sum(1 for g in gens if g["used_retry"]),
        "r_O": r_O,
        "per_word": per_word,
        "dose_response": dose,
        "rho": rho_table,
        "noise_ceiling_per_cell": ceiling,
        "discordant": discordant,
        "hijack_by_group": hij_rows,
        "hijack_per_word": hij_word,
        "hijack_coverage": hij_cov | {"label_source": "hijack_fast.jsonl (reasoning disabled)" if hij_fast else "hijack.jsonl (reasoning)"},
        "hijack_agreement_fast_vs_reasoning": hijack_agreement,
        "key_scan_checks": key_checks,
        "table_correlations_eligible": tmeta["table_correlations"],
        "rate_matrix": {"words": cw, "prompt_ids": pids, "populations": pops.tolist(), "r_E": rE.tolist(), "r_O": rO.tolist(), "historical_r_O": hist.tolist(), "key_shift_sd": key.tolist() if key is not None else None},
        "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED, "unit": "prompts resampled with replacement within each population (pooled: within each population, then combined), paired across conditions; words fixed; two-sided bootstrap p", "permutation": f"{N_PERM} permutations of Delta_EO over words, two-sided on |rho|"},
    }
    out_json = RESULTS.with_name(f"results{suffix}.json")
    out_md = RESULTS_MD.with_name(f"results{suffix}.md")
    atomic_json(out_json, results)
    write_md(results, out_md)
    figures(results, tab, kind, suffix)
    print(out_md.read_text())


def f(x, d=3):
    return "nan" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:+.{d}f}"


def u(x, d=3):
    return "nan" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def cis(c, d=3):
    return f"[{f(c[0], d)}, {f(c[1], d)}]"


def write_md(r: dict, path: Path) -> None:
    nb, nh = r["r_O"]["benign"]["n_prompts"], r["r_O"]["harmful"]["n_prompts"]
    L = ["# x111: words sampled across the vocabulary table on held-out prompts (Qwen3.5-4B)\n"]
    if r.get("prompt_filter"):
        pf = r["prompt_filter"]
        L.append(f"Prompts excluded ({pf['file']}): {', '.join(pf['excluded_ids'])} ({pf['n_excluded']} x85-cohort prompts).\n")
    if r.get("topup"):
        tu = r["topup"]
        L.append(f"Harmful stratum topped back up by {tu['n_added']} prompts outside the x85 cohort ({tu['file']}): " + ", ".join(f"{a['id']} {a['historical_refusal2']:.1f}" for a in tu["added"]) + ".\n")
    if r.get("cohort"):
        L.append(f"Cohort: {r['cohort']}.\n")
    L.append(f"{nb} benign (x106 Hard-1K) + {nh} harmful (x83 cohort) test prompts with historical `refusal2` near 0.5; 40 words (24 concordant over eight bins of the joint benign / BP table percentile, 16 discordant) appended at the end; twenty samples per cell; `refusal2` (x64 judge). CIs: prompt-level paired bootstrap, {r['bootstrap']['n']} draws. Missing generations: {len(r['missing_generations'])}; missing refusal2 cells: {len(r['missing_refusal2'])}; refusal2 units judged one response per call: {r['n_refusal2_per_response_fallback']}; cells regenerated at 4,096 tokens: {r['n_cells_retry4096']}.\n")
    L.append("## Table choice (ρ(table shift, Δ_EO) over the 40 words)\n")
    L.append(f"| table | pooled {r['n_prompts']} prompts ρ [95% CI] | perm p | benign prompts ρ [95% CI] | harmful prompts ρ [95% CI] |\n|---|---|---|---|---|")
    for t in TABLES:
        v = r["table_choice"][t]
        L.append(f"| {t} | {f(v['rho'], 2)} {cis(v['ci'], 2)} | {v['perm_p']:.4f} | {f(v['rho_benign_prompts'], 2)} {cis(v['ci_benign_prompts'], 2)} | {f(v['rho_harmful_prompts'], 2)} {cis(v['ci_harmful_prompts'], 2)} |")
    L.append("\n## r(O)\n")
    L.append("| prompts | n | r(O) this run | historical (ten samples) | Spearman run vs historical |\n|---|---|---|---|---|")
    for s, v in r["r_O"].items():
        L.append(f"| {s} | {v['n_prompts']} | {v['run']:.3f} | {v['historical']:.3f} | {f(v['spearman_run_vs_historical'], 2)} |")
    L.append("\n## Dose-response over the eight concordant bins (Δ_EO, mean over the bin's three words)\n")
    L.append("| bin | words | benign sd | BP sd | HP sd | mixed sd | " + " | ".join(f"{s} Δ_EO [95% CI]" for s in SETS) + " |\n|---|---|---|---|---|---|" + "---|" * len(SETS))
    for b in range(len(BIN_NAMES)):
        row = r["dose_response"]["benign"]["bins"][b]
        L.append(f"| {row['bin']} | {', '.join(row['words'])} | {row['benign_sd_mean']:+.2f} | {row['BP_sd_mean']:+.2f} | {row['HP_sd_mean']:+.2f} | {row['mixed_sd_mean']:+.2f} | " + " | ".join(f"{f(r['dose_response'][s]['bins'][b]['delta_EO'])} {cis(r['dose_response'][s]['bins'][b]['delta_EO_ci95'])}" for s in SETS) + " |")
    for s in SETS:
        v = r["dose_response"][s]
        L.append(f"\n- {s}: adjacent bin pairs in order {v['adjacent_pairs_in_order']}/{v['adjacent_pairs_total']} (bootstrap mean fraction {v['fraction_adjacent_in_order_boot_mean']:.2f}); Spearman bin index vs Δ_EO {f(v['spearman_bin_index_vs_delta'], 2)}; top minus bottom bin {f(v['top_minus_bottom_bin'])} {cis(v['top_minus_bottom_bin_ci95'])}; range over the middle four bins {v['middle_four_bins_range']:.3f}.")
    L.append("\n## Spearman ρ(predictor, per-word Δ_EO)\n")
    L.append("| prompts | predictor | ρ all 40 [95% CI] | perm p | ρ concordant 24 [95% CI] | perm p | ρ middle 4 bins (12) [95% CI] | partial ρ (tokens, log freq) | per-cell ρ | per-cell ceiling |\n|---|---|---|---|---|---|---|---|---|---|")
    for s in SETS:
        for name, e in r["rho"][s].items():
            if not isinstance(e, dict):
                continue
            a, c, mm = e["all40"], e["concordant24"], e["middle4bins"]
            L.append(f"| {s} | {name} | {f(a['rho'], 2)} {cis(a['ci95_prompt_bootstrap'], 2)} | {a['perm_p']:.4f} | {f(c['rho'], 2)} {cis(c['ci95_prompt_bootstrap'], 2)} | {c['perm_p']:.4f} | {f(mm['rho'], 2)} {cis(mm['ci95_prompt_bootstrap'], 2)} | {f(e['partial_rho_all40_given_ntokens_logfreq'], 2)} | {f(e['per_cell']['rho'], 2)} (n = {e['per_cell']['n_cells']}) | {r['noise_ceiling_per_cell'][s]['ceiling_spearman']:.2f} |")
    L.append("\nPer-cell ceiling: twenty samples split into random halves of ten (20 splits), half-half Spearman of r(E) − r(O) across cells, Spearman-Brown to twenty samples, square root.\n")
    for s in SETS:
        c = r["noise_ceiling_per_cell"][s]
        L.append(f"- {s}: {c['n_cells']} cells, half-half ρ {c['half_half_spearman_10_samples']:.3f}, reliability (20) {c['reliability_20_samples_spearman_brown_spearman']:.3f}, ceiling ρ {c['ceiling_spearman']:.3f} (Pearson {c['ceiling_pearson']:.3f}).")
    L.append("\n## Discordant words\n")
    for s in SETS:
        v = r["discordant"][s]
        L.append(f"\n### {s}\n")
        L.append(f"Fits on the 24 concordant words: " + "; ".join(f"{t}: Δ_EO = {v['fits'][t]['slope']:+.3f} × shift {v['fits'][t]['intercept']:+.3f}" for t in TABLES) + ".")
        L.append(f"Closer of benign / BP: benign {v['count_closer']['closer_benign']}, BP {v['count_closer']['closer_BP']}; best of four: " + ", ".join(f"{t} {n}" for t, n in v["count_best_of_four"].items()) + "; side match: " + ", ".join(f"{t} {n}/16" for t, n in v["count_side_match"].items()) + "; summed |error|: " + ", ".join(f"{t} {e:.3f}" for t, e in v["sum_abs_err"].items()) + ".\n")
        L.append("| word | quadrant | pct benign | pct BP | pct HP | pct mixed | Δ_EO [95% CI] | pred benign | pred BP | pred HP | pred mixed | closer (benign/BP) | best of four |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for w in v["words"]:
            L.append(f"| {w['word']} | {w['quadrant']} | {w['pct_benign']:.1f} | {w['pct_BP']:.1f} | {w['pct_HP']:.1f} | {w['pct_mixed']:.1f} | {f(w['delta_EO'])} {cis(w['delta_EO_ci95'])} | {f(w['pred']['benign'])} | {f(w['pred']['BP'])} | {f(w['pred']['HP'])} | {f(w['pred']['mixed'])} | {w['closer_of_benign_BP']} | {w['best_of_four']} |")
        L.append("\n| quadrant | words | Δ_EO [95% CI] |\n|---|---|---|")
        for q, qv in v["quadrant_means"].items():
            L.append(f"| {q} | {', '.join(qv['words'])} | {f(qv['delta_EO'])} {cis(qv['delta_EO_ci95'])} |")
    L.append("\n## Per word\n")
    L.append("| word | stratum | benign sd | BP sd | HP sd | mixed sd | tok | " + " | ".join(f"{s} Δ_EO [95% CI]" for s in POPULATIONS) + " | " + " | ".join(f"{s} key sd" for s in POPULATIONS) + " |\n|---|---|---|---|---|---|---|" + "---|" * (2 * len(POPULATIONS)))
    for w in r["per_word"]:
        L.append(f"| {w['word']} | {w['stratum']} | {w['benign_sd']:+.2f} | {w['BP_sd']:+.2f} | {w['HP_sd']:+.2f} | {w['mixed_sd']:+.2f} | {w['n_tokens']} | " + " | ".join(f"{f(w[s]['delta_EO'])} {cis(w[s]['delta_EO_ci95'])}" for s in POPULATIONS) + " | " + " | ".join(f(w[s].get("key_shift_sd"), 2) for s in POPULATIONS) + " |")
    L.append("\n## Hijack\n")
    hc = r["hijack_coverage"]
    L.append(f"Hijack labels ({hc['label_source']}): {hc['n_units_judged']} of {hc['n_E_units']} E units; {hc['n_cells_complete']} cells complete.\n")
    ag = r.get("hijack_agreement_fast_vs_reasoning")
    if ag:
        L.append(f"Reasoning-disabled vs reasoning pass on the {ag['n_units']} units both judged ({ag['n_responses']} responses): agreement {ag['agreement']:.3f}, kappa {ag['cohen_kappa']:.3f}, hijack rate {ag['rate_no_reasoning']:.3f} vs {ag['rate_reasoning']:.3f}.\n")
    L.append("| group | " + " | ".join(f"{s} hijack | {s} refusal2 given hijack / not" for s in POPULATIONS) + " |\n|---|" + "---|---|" * len(POPULATIONS))
    for g in r["hijack_by_group"]:
        L.append(f"| {g['group']} | " + " | ".join(f"{u(g[s]['hijack_rate'])} | {u(g[s]['refusal2_given_hijack'])} / {u(g[s]['refusal2_given_no_hijack'])}" for s in POPULATIONS) + " |")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def figures(r: dict, tab: dict, kind: np.ndarray, suffix: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"benign": "#2a6fdb", "harmful": "#d1495b"}
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=160)
    x = np.arange(len(BIN_NAMES))
    for off, s in ((-0.09, "benign"), (0.09, "harmful")):
        rows = r["dose_response"][s]["bins"]
        y = np.asarray([b["delta_EO"] for b in rows])
        lo = y - np.asarray([b["delta_EO_ci95"][0] for b in rows])
        hi = np.asarray([b["delta_EO_ci95"][1] for b in rows]) - y
        ax.errorbar(x + off, y, yerr=[lo, hi], marker="o", ms=5, lw=1.6, capsize=3, color=colors[s], label=f"{s} test prompts (n = {r['r_O'][s]['n_prompts']})")
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xticks(x, BIN_NAMES)
    ax.set_xlabel("percentile bin in both the benign and the BP table (concordant words, 3 per bin)")
    ax.set_ylabel("Δ refusal2, E − O")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(RUN / f"dose_response{suffix}.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.4), dpi=150, sharey="row")
    words = r["rate_matrix"]["words"]
    for i, s in enumerate(POPULATIONS):
        dEO = np.asarray([w[s]["delta_EO"] for w in r["per_word"]])
        for j, t in enumerate(TABLES):
            ax = axes[i, j]
            c = kind == "concordant"
            ax.scatter(tab[t][c], dEO[c], s=22, color=colors[s], alpha=0.85, label="concordant")
            ax.scatter(tab[t][~c], dEO[~c], s=34, marker="D", facecolors="none", edgecolors="#333", label="discordant")
            e = r["rho"][s][f"table_{t}"]["all40"]
            ax.set_title(f"{s} prompts vs {t} table: ρ = {e['rho']:+.2f}", fontsize=9)
            ax.axhline(0, color="#aaa", lw=0.7)
            ax.spines[["top", "right"]].set_visible(False)
            if i == 1:
                ax.set_xlabel(f"{t} table shift (sd)")
            if j == 0:
                ax.set_ylabel("Δ refusal2, E − O")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(RUN / f"tables_vs_effect{suffix}.png")
    plt.close(fig)
    _ = words


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="no prompt exclusion: all 64 prompts, outputs with the _full suffix")
    ap.add_argument("--no-topup", action="store_true", help="the x85 exclusion without the top-up prompts (55 prompts), outputs with the _pre_topup suffix")
    a = ap.parse_args()
    if a.full and a.no_topup:
        raise SystemExit("--full and --no-topup exclude each other")
    ex = set(read_json(EXCLUSIONS)["excluded_x111_prompt_ids"])
    if a.full:
        main(set(), "_full", cohort="the original 64 prompts (32 benign + 32 harmful), unfiltered, without the top-up (--full)")
    elif a.no_topup:
        main(ex, "_pre_topup", cohort="the original prompts outside the x85 crossed cohort, before the top-up: 32 benign + 23 harmful (--no-topup)")
    else:
        main(ex, "", topup=True, cohort="the original prompts outside the x85 crossed cohort plus the nine harmful top-up prompts: 32 benign + 32 harmful")
