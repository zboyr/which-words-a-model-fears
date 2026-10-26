"""x112 analysis (local, numpy / scipy).

Per model (refusal2, x64 judge, twenty samples per cell) over its 32 held-out prompts, per
population and pooled: r(O); per word r(E) and Delta_EO with a prompt-level paired bootstrap;
the dose-response curve over the eight concordant bins of the model's own table; Spearman rho
of the per-word Delta_EO with the model's own table, the Qwen3.5-4B BP table (x111) and the
in-context key-word shift (all 40 words, the 24 concordant words, the middle four bins), bootstrap
CI and permutation p, partial rho with token count and log wordfreq rank held fixed, the per-cell
rho against the split-half noise ceiling; the discordant-word verdict; hijack per bin.
Across models: per-bin Delta_EO averaged over the seven models, rho over all (model, word)
points on the percentile scale, and the between-model table agreement (table_comparison.json).
Writes results.json, results.md, dose_response.png, tables_vs_effect.png.

Exclusion. Every model's direction was fitted on the x85 crossed cohort (96 x83 prompts, 24 harm
words, 24 neutral twins). By default the held-out prompts and chosen words listed in
x85_exclusions.json (the grid rows that sit in that cohort) are dropped before any statistic:
Gemma 2 harmful prompts, Qwen3.5-0.8B 1, Qwen3.5-9B 4, Qwen3.5-27B 6, and the word "rape" on
Llama (Holm then runs over its 39 words, key delta_EO_p_holm39). The bootstrap, permutation and
split-half seeds are those of the whole grid; only the prompt set is smaller. `--full` analyses
the whole grid instead and writes results_full.json, results_full.md, dose_response_full.png and
tables_vs_effect_full.png. A model record carries `excluded` only when something was dropped.

Top-up. On the four models that lost harmful prompts to the exclusion, the harmful stratum was
topped back up to 16 prompts outside the x85 cohort (prompts_topup.json; x112_select.py
--topup-request / --topup-prompts, x112_common). The default cohort is the retained original
prompts plus the top-up prompts (16 benign + 16 harmful per model); their E, O, Orep cells and
key-word scan (key_words_scan_topup.json) are read beside the originals, and results.json
carries a `topup` block (the added ids, their fresh rates, the pool-stage reuse). `--no-topup`
analyses the retained original prompts alone (the pre-top-up cohort) into the _pre_topup
siblings; `--full` is the original 32 prompts per model, unfiltered, without the top-up.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x111_analyze import boot_p, ci, holm, partial_rho, perm_p, rho, rho_rows  # noqa: E402
from x111_select import BIN_NAMES  # noqa: E402
from x112_common import (  # noqa: E402
    COMPARISON,
    LABELS,
    N_HALF,
    N_SAMPLES,
    POPULATIONS,
    RESULTS,
    RESULTS_MD,
    RUN,
    TAGS,
    X111_TABLES_NPZ,
    atomic_json,
    cells,
    paths,
    read_json,
    sd_unit,
)
from x112_judge import load_generations  # noqa: E402

EXCLUSIONS = RUN / "x85_exclusions.json"
N_BOOT = 10_000
N_PERM = 10_000
BOOT_SEED = 112_000
HALF_SEEDS = list(range(20))
MIDDLE_BINS = (2, 3, 4, 5)
SETS = ("benign", "harmful", "pooled")
TABLES = ("own", "qwen_BP")
TABLE_LABEL = {"own": "own table", "qwen_BP": "Qwen3.5-4B BP table"}


def load_units(path: Path) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if not rec.get("failed"):
                out[rec["cell_id"]] = rec
    return out


def analyze_model(tag: str, mi: int, drop_prompts: tuple[str, ...] = (), drop_words: tuple[str, ...] = (), topup: bool = True) -> dict:
    P_ = paths(tag)
    prompts_obj = read_json(P_["prompts"])
    prompts_orig = list(prompts_obj["prompts"])
    topup_obj = read_json(P_["prompts_topup"]) if topup and P_["prompts_topup"].exists() else None
    prompts_all = prompts_orig + (list(topup_obj["prompts"]) if topup_obj else [])
    if len({p["id"] for p in prompts_all}) != len(prompts_all):
        raise SystemExit(f"FATAL: {tag} top-up prompt repeats an original prompt")
    words_obj = read_json(P_["words"])
    chosen_all = words_obj["chosen"]
    pids_all = [p["id"] for p in prompts_all]
    cw_all = [c["word"] for c in chosen_all]
    keep_p = [j for j, p in enumerate(prompts_all) if p["id"] not in set(drop_prompts)]
    keep_w = [k for k, c in enumerate(chosen_all) if c["word"] not in set(drop_words)]
    prompts = [prompts_all[j] for j in keep_p]
    chosen = [chosen_all[k] for k in keep_w]
    excluded = {"prompt_ids": [p for p in pids_all if p in set(drop_prompts)], "words": [w for w in cw_all if w in set(drop_words)]}
    cw = [c["word"] for c in chosen]
    tz = np.load(P_["table_npz"])
    qz = np.load(X111_TABLES_NPZ)
    sd = sd_unit(tag)
    gens, _ = load_generations(tag)
    gen = {g["cell_id"]: g for g in gens}
    ref = load_units(P_["judge"] / "refusal2.jsonl")
    hij = load_units(P_["judge"] / "hijack_fast.jsonl")
    grid = cells({"prompts": prompts}, {"chosen": chosen})
    for uid, rec in ref.items():
        cid, h = uid.rsplit("#h", 1)
        if cid in gen and rec["response_sha256"] != gen[cid]["response_sha256"][int(h) * N_HALF : (int(h) + 1) * N_HALF]:
            raise SystemExit(f"FATAL: {tag} refusal2 record {uid} does not match its responses")
    for c in grid:
        if c["cell_id"] in gen and gen[c["cell_id"]]["prompt_sha256"] != c["prompt_sha256"]:
            raise SystemExit(f"FATAL: {tag} generation {c['cell_id']} has another prompt")

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
    LR = np.full((P, N_SAMPLES), np.nan)
    for j, pid in enumerate(pids):
        v = labels(ref, f"{pid}|Orep", "refusal")
        if v is not None:
            LR[j] = v
    rO = LO.mean(1)
    rR = LR.mean(1)
    rE = LE.mean(2)
    d = rE - rO[None]
    masks = {"benign": pops == "benign", "harmful": pops == "harmful", "pooled": np.ones(P, bool)}
    hist = np.asarray([p["historical_refusal2"] for p in prompts])
    kind = np.asarray([c["kind"] for c in chosen])
    bins = np.asarray([c["bin"] if c["bin"] is not None else -1 for c in chosen])
    conc = kind == "concordant"
    mid = np.isin(bins, MIDDLE_BINS)
    tab = {"own": np.asarray([float(tz["own"][c["index"]]) for c in chosen]), "qwen_BP": np.asarray([float(qz["BP"][c["index"]]) for c in chosen])}
    pct = {"own": np.asarray([c["pct_own"] for c in chosen]), "qwen_BP": np.asarray([c["pct_qwen_BP"] for c in chosen])}
    for c in chosen:
        if abs(float(tz["own"][c["index"]]) - c["own_sd"]) > 1e-5 or abs(float(qz["BP"][c["index"]]) - c["qwen_BP_sd"]) > 1e-5:
            raise SystemExit(f"FATAL: {tag} words.json table value of {c['word']} differs from the tables")
    n_tok = np.asarray([c["n_tokens"] for c in chosen], float)
    log_wf = np.log(np.asarray([c["wordfreq_rank"] if c["wordfreq_rank"] else 50_001 for c in chosen], float))

    key, key_checks = None, None
    if P_["key_json"].exists() and P_["key_npz"].exists():
        kj = read_json(P_["key_json"])
        if kj["prompt_ids"] != [p["id"] for p in prompts_orig] or kj["candidates"]["words"] != cw_all:
            raise SystemExit(f"FATAL: {tag} key-word scan does not match prompts / words order")
        shift = np.load(P_["key_npz"])["shift"].astype(np.float64).sum(axis=2) / sd  # [P_orig, W_all]
        o_sd, o_max, secs, modes, complete = list(kj["o_proj_cached_sd"]), [kj["o_cached_vs_full_max_abs_per_token_sd"]], list(kj["seconds_per_prompt"]), {str(kj["mode"])}, True
        if topup_obj:
            if P_["key_json_topup"].exists() and P_["key_npz_topup"].exists():
                kt = read_json(P_["key_json_topup"])
                if kt["prompt_ids"] != [p["id"] for p in topup_obj["prompts"]] or kt["candidates"]["words"] != cw_all:
                    raise SystemExit(f"FATAL: {tag} top-up key-word scan does not match prompts_topup / words order")
                shift = np.concatenate([shift, np.load(P_["key_npz_topup"])["shift"].astype(np.float64).sum(axis=2) / sd], axis=0)
                o_sd += list(kt["o_proj_cached_sd"])
                o_max.append(kt["o_cached_vs_full_max_abs_per_token_sd"])
                secs += list(kt["seconds_per_prompt"])
                modes.add(str(kt["mode"]))
            else:
                print(f"WARNING: {tag}: the top-up prompts have no key-word scan yet; key_in_context omitted", file=sys.stderr)
                complete = False
        if complete:
            key = shift[np.ix_(keep_p, keep_w)]  # [P, W]
            key_checks = {"mode": sorted(modes)[0] if len(modes) == 1 else sorted(modes), "o_cached_vs_full_max_abs_per_token_sd": max(o_max), "scan_minutes": float(sum(secs) / 60), "o_proj_sd_mean": {s: float(np.mean(np.asarray(o_sd)[keep_p][m])) for s, m in masks.items()}}

    rng = np.random.default_rng([BOOT_SEED, mi])
    boot_idx = {}
    for s, m in masks.items():
        if s == "pooled":
            boot_idx[s] = np.concatenate([np.where(masks[q])[0][rng.integers(0, masks[q].sum(), size=(N_BOOT, masks[q].sum()))] for q in POPULATIONS], axis=1)
        else:
            sub = np.where(m)[0]
            boot_idx[s] = sub[rng.integers(0, len(sub), size=(N_BOOT, len(sub)))]

    def bmean(M: np.ndarray, s: str) -> np.ndarray:
        return np.moveaxis(np.nanmean(M[..., boot_idx[s]], axis=-1), -1, 0)

    r_O = {}
    for s, m in masks.items():
        off = bmean(rO - rR, s) if np.isfinite(rR).all() else np.full(N_BOOT, np.nan)
        r_O[s] = {"run": float(np.nanmean(rO[m])), "historical": float(hist[m].mean()), "spearman_run_vs_historical": rho(hist[m], rO[m]), "n_prompts": int(m.sum()), "replicate": float(np.nanmean(rR[m])) if np.isfinite(rR[m]).any() else None, "n_replicate": int(np.isfinite(rR[m]).sum()), "run_minus_replicate": float(np.nanmean((rO - rR)[m])) if np.isfinite(rR[m]).any() else None, "run_minus_replicate_ci95": ci(off), "spearman_run_vs_replicate": rho(rO[m], rR[m])}
    dEO_boot = {s: bmean(d, s) for s in SETS}
    per_word = []
    for k, c in enumerate(chosen):
        rec = {kk: c[kk] for kk in ("word", "stratum", "kind", "bin_name", "quadrant", "own_sd", "qwen_BP_sd", "pct_own", "pct_qwen_BP", "n_tokens", "wordfreq_rank")}
        for s, m in masks.items():
            b = dEO_boot[s][:, k]
            rec[s] = {"r_E": float(np.nanmean(rE[k][m])), "delta_EO": float(np.nanmean(d[k][m])), "delta_EO_ci95": ci(b), "delta_EO_p": boot_p(b), "n_prompts_E_below_O": int(np.nansum(d[k][m] < 0)), "n_prompts_E_above_O": int(np.nansum(d[k][m] > 0))}
            if key is not None:
                rec[s]["key_shift_sd"] = float(key[m, k].mean())
        per_word.append(rec)
    for s in SETS:
        for pw, v in zip(per_word, holm([pw[s]["delta_EO_p"] for pw in per_word])):
            pw[s][f"delta_EO_p_holm{W}"] = v  # Holm over the retained words

    dose, bin_boot_all = {}, {}
    for s, m in masks.items():
        rows, bin_boot = [], []
        for b in range(len(BIN_NAMES)):
            sel = conc & (bins == b)
            bb = dEO_boot[s][:, sel].mean(1)
            bin_boot.append(bb)
            rows.append({"bin": BIN_NAMES[b], "words": [cw[k] for k in np.where(sel)[0]], "own_sd_mean": float(tab["own"][sel].mean()), "qwen_BP_sd_mean": float(tab["qwen_BP"][sel].mean()), "r_E": float(np.nanmean(rE[sel][:, m])), "delta_EO": float(np.nanmean(d[sel][:, m])), "delta_EO_ci95": ci(bb), "delta_EO_p": boot_p(bb), "delta_E_minus_Orep": float(np.nanmean((rE - rR[None])[sel][:, m])) if np.isfinite(rR[m]).all() else None})
        est = np.asarray([r["delta_EO"] for r in rows])
        BB = np.stack(bin_boot, 1)
        bin_boot_all[s] = BB
        dose[s] = {"bins": rows, "adjacent_pairs_in_order": int(np.sum(np.diff(est) > 0)), "adjacent_pairs_total": len(BIN_NAMES) - 1, "fraction_adjacent_in_order_boot_mean": float(np.mean(np.diff(BB, axis=1) > 0)), "spearman_bin_index_vs_delta": rho(np.arange(len(BIN_NAMES)), est), "top_minus_bottom_bin": float(est[-1] - est[0]), "top_minus_bottom_bin_ci95": ci(BB[:, -1] - BB[:, 0]), "middle_four_bins_range": float(est[list(MIDDLE_BINS)].max() - est[list(MIDDLE_BINS)].min())}

    predictors = {f"table_{t}": tab[t] for t in TABLES}
    if key is not None:
        predictors["key_in_context"] = None
    subsets = {"all40": np.ones(W, bool), "concordant24": conc, "middle4bins": conc & mid}
    rho_table = {}
    for s, m in masks.items():
        dEO = np.nanmean(d[:, m], axis=1)
        rho_table[s] = {}
        for name in predictors:
            x = key[m].mean(0) if name == "key_in_context" else predictors[name]
            entry = {}
            for sub, sm in subsets.items():
                Yb = dEO_boot[s][:, sm]
                okb = np.all(np.isfinite(Yb), axis=1)
                entry[sub] = {"n_words": int(sm.sum()), "rho": rho(x[sm], dEO[sm]), "ci95_prompt_bootstrap": ci(rho_rows(x[sm], Yb[okb])), "perm_p": perm_p(x[sm], dEO[sm])}
            entry["partial_rho_all40_given_ntokens_logfreq"] = partial_rho(x, dEO, [n_tok, log_wf])
            xc = key[m].T if name == "key_in_context" else np.repeat(x[:, None], m.sum(), axis=1)
            dc = d[:, m]
            ok = np.isfinite(dc)
            entry["per_cell"] = {"n_cells": int(ok.sum()), "rho": rho(xc[ok], dc[ok])}
            rho_table[s][name] = entry
        # Difference own minus Qwen rho over the 40 words, bootstrap.
        Yb = dEO_boot[s]
        okb = np.all(np.isfinite(Yb), axis=1)
        diff = rho_rows(tab["own"], Yb[okb]) - rho_rows(tab["qwen_BP"], Yb[okb])
        rho_table[s]["own_minus_qwen_BP_all40"] = {"diff": rho_table[s]["table_own"]["all40"]["rho"] - rho_table[s]["table_qwen_BP"]["all40"]["rho"], "ci95_prompt_bootstrap": ci(diff), "p": boot_p(diff)}

    ceiling = {}
    for s, m in masks.items():
        LEs, LOs = LE[:, m], LO[m]
        ok = np.isfinite(LEs).all(2) & np.isfinite(LOs).all(1)[None]
        hh_s = []
        for seed in HALF_SEEDS:
            perm = np.random.default_rng(seed).permutation(N_SAMPLES)
            a, b = perm[: N_SAMPLES // 2], perm[N_SAMPLES // 2 :]
            d1 = LEs[..., a].mean(-1) - LOs[None, :, a].mean(-1)
            d2 = LEs[..., b].mean(-1) - LOs[None, :, b].mean(-1)
            hh_s.append(stats.spearmanr(d1[ok], d2[ok]).statistic)
        r10 = float(np.mean(hh_s))
        rel = 2 * r10 / (1 + r10)
        ceiling[s] = {"n_cells": int(ok.sum()), "half_half_spearman_10_samples": r10, "reliability_20_samples": rel, "ceiling_spearman": float(np.sqrt(max(rel, 0)))}

    discordant = {}
    disc_idx = np.where(kind == "discordant")[0]
    for s, m in masks.items():
        dEO = np.nanmean(d[:, m], axis=1)
        fits = {}
        for t in TABLES:
            slope, intercept = np.polyfit(tab[t][conc], dEO[conc], 1)
            fits[t] = {"slope": float(slope), "intercept": float(intercept), "pred": slope * tab[t] + intercept}
        rows = []
        counts = {"own": 0, "qwen_BP": 0}
        side = {t: 0 for t in TABLES}
        for k in disc_idx:
            preds = {t: float(fits[t]["pred"][k]) for t in TABLES}
            err = {t: abs(preds[t] - dEO[k]) for t in TABLES}
            closer = "own" if err["own"] < err["qwen_BP"] else "qwen_BP"
            counts[closer] += 1
            sgn = {t: bool(np.sign(preds[t] - dEO[conc].mean()) == np.sign(dEO[k] - dEO[conc].mean())) for t in TABLES}
            for t in TABLES:
                side[t] += sgn[t]
            rows.append({"word": cw[k], "quadrant": chosen[k]["quadrant"], "pct_own": chosen[k]["pct_own"], "pct_qwen_BP": chosen[k]["pct_qwen_BP"], "delta_EO": float(dEO[k]), "delta_EO_ci95": per_word[k][s]["delta_EO_ci95"], "pred": preds, "abs_err": err, "closer": closer, "side_match": sgn})
        quad = {}
        for qname in sorted({chosen[k]["quadrant"] for k in disc_idx}):
            qs = [k for k in disc_idx if chosen[k]["quadrant"] == qname]
            qb = dEO_boot[s][:, qs].mean(1)
            quad[qname] = {"words": [cw[k] for k in qs], "delta_EO": float(dEO[qs].mean()), "delta_EO_ci95": ci(qb)}
        discordant[s] = {"fits": {t: {"slope": fits[t]["slope"], "intercept": fits[t]["intercept"]} for t in TABLES}, "words": rows, "count_closer": counts, "count_side_match": side, "sum_abs_err": {t: float(sum(r["abs_err"][t] for r in rows)) for t in TABLES}, "quadrant_means": quad}

    HJ = np.full((W, P, N_SAMPLES), np.nan)
    for k, w in enumerate(cw):
        for j, pid in enumerate(pids):
            v = labels(hij, f"{pid}|E_{w}", "hijack_llm")
            if v is not None:
                HJ[k, j] = v
    hij_rows = []
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
    hij_word = [float(np.nanmean(HJ[k])) if np.isfinite(HJ[k]).any() else None for k in range(W)]
    if np.isfinite(HJ).any():
        rho_table["pooled"]["hijack_rate_vs_delta_EO"] = rho(np.asarray(hij_word, float), np.nanmean(d, axis=1))

    gen_tok = {kd: float(np.mean([np.mean(g["gen_tokens"]) for g in gens if g["cell_id"] in {c["cell_id"] for c in grid if c["kind"] == kd}])) if gens else None for kd in ("O", "E")}
    excl = {"excluded": excluded | {"n_prompts_by_population": {q: int((pops == q).sum()) for q in POPULATIONS}}} if excluded["prompt_ids"] or excluded["words"] else {}
    tu = {}
    if topup_obj:
        tu = {"topup": {"n_added": len(topup_obj["prompts"]), "added": [{k: p[k] for k in ("id", "population", "historical_refusal2", "fresh_refusal2", "rank", "topup_source", "historical_source")} for p in topup_obj["prompts"]], **{k: topup_obj["topup"][k] for k in ("n_chosen_original", "n_retained_after_x85", "n_needed", "overdraw", "reused_from_pool_stage", "pool_stage_judged_out_of_band", "topup_pool_ids")}, "pools": topup_obj["pools"], "n_prompts_by_population": {q: int((pops == q).sum()) for q in POPULATIONS}}}
    return {
        "model_tag": tag,
        "label": LABELS[tag],
        "sd_unit": sd,
        "n_prompts": P,
        "n_words": W,
        **excl,
        **tu,
        "missing_generations": missing_gen,
        "missing_refusal2": missing_ref,
        "n_refusal2_units_grid": sum(1 for c in grid for h in range(2) if f"{c['cell_id']}#h{h}" in ref),
        "n_refusal2_per_response_fallback": sum(1 for v in ref.values() if v.get("judge_mode") == "per_response_fallback"),
        "n_hijack_units": sum(1 for c in grid if c["kind"] == "E" for h in range(2) if f"{c['cell_id']}#h{h}" in hij),
        "n_cells_retry4096": sum(1 for c in grid if c["cell_id"] in gen and gen[c["cell_id"]]["used_retry"]),
        "n_responses_truncated": int(sum(sum(gen[c["cell_id"]]["truncated"]) for c in grid if c["cell_id"] in gen)),
        "n_responses_empty": int(sum(sum(not t.strip() for t in gen[c["cell_id"]]["llm_responses"]) for c in grid if c["cell_id"] in gen)),
        "mean_gen_tokens": gen_tok,
        "pools": prompts_obj["pools"],
        "r_O": r_O,
        "per_word": per_word,
        "dose_response": dose,
        "rho": rho_table,
        "noise_ceiling_per_cell": ceiling,
        "discordant": discordant,
        "hijack_by_group": hij_rows,
        "hijack_per_word": dict(zip(cw, hij_word)),
        "key_scan_checks": key_checks,
        "rate_matrix": {"words": cw, "prompt_ids": pids, "populations": pops.tolist(), "r_E": rE.tolist(), "r_O": rO.tolist(), "historical_r_O": hist.tolist(), "key_shift_sd": key.tolist() if key is not None else None},
        "_pct": pct,
        "_dEO": {s: np.nanmean(d[:, m], axis=1) for s, m in masks.items()},
        "_bin_boot": bin_boot_all,
        "_tab": tab,
        "_kind": kind,
    }


def pooled_over_models(models: list[dict]) -> dict:
    out = {"n_models": len(models)}
    for s in SETS:
        est = np.asarray([[m["dose_response"][s]["bins"][b]["delta_EO"] for b in range(len(BIN_NAMES))] for m in models])
        BB = np.mean([m["_bin_boot"][s] for m in models], axis=0)  # independent per model, averaged per draw
        mean = est.mean(0)
        out.setdefault("dose_response", {})[s] = {"bins": [{"bin": BIN_NAMES[b], "delta_EO_mean_over_models": float(mean[b]), "ci95": ci(BB[:, b]), "per_model": {m["model_tag"]: float(est[i, b]) for i, m in enumerate(models)}} for b in range(len(BIN_NAMES))], "adjacent_pairs_in_order": int(np.sum(np.diff(mean) > 0)), "top_minus_bottom": float(mean[-1] - mean[0]), "top_minus_bottom_ci95": ci(BB[:, -1] - BB[:, 0]), "models_with_7_of_7": int(sum(m["dose_response"][s]["adjacent_pairs_in_order"] == 7 for m in models)), "adjacent_pairs_in_order_per_model": {m["model_tag"]: m["dose_response"][s]["adjacent_pairs_in_order"] for m in models}}
        for t in TABLES:
            x = np.concatenate([m["_pct"][t] for m in models])
            y = np.concatenate([m["_dEO"][s] for m in models])
            per = [m["rho"][s][f"table_{t}"]["all40"]["rho"] for m in models]
            out.setdefault("rho_percentile_all_points", {}).setdefault(s, {})[t] = {"n_points": int(len(x)), "rho": rho(x, y), "mean_per_model_rho": float(np.mean(per)), "min_per_model_rho": float(np.min(per)), "max_per_model_rho": float(np.max(per))}
        out["rho_percentile_all_points"][s]["n_models_own_higher"] = int(sum(m["rho"][s]["table_own"]["all40"]["rho"] > m["rho"][s]["table_qwen_BP"]["all40"]["rho"] for m in models))
        out.setdefault("discordant_totals", {})[s] = {"closer_own": int(sum(m["discordant"][s]["count_closer"]["own"] for m in models)), "closer_qwen_BP": int(sum(m["discordant"][s]["count_closer"]["qwen_BP"] for m in models)), "side_match_own": int(sum(m["discordant"][s]["count_side_match"]["own"] for m in models)), "side_match_qwen_BP": int(sum(m["discordant"][s]["count_side_match"]["qwen_BP"] for m in models)), "n_words": int(sum(len(m["discordant"][s]["words"]) for m in models)), "quadrant_mean_over_models": {q: float(np.mean([m["discordant"][s]["quadrant_means"][q]["delta_EO"] for m in models])) for q in models[0]["discordant"][s]["quadrant_means"]}}
    return out


def f(x, d=3):
    return "nan" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:+.{d}f}"


def u(x, d=3):
    return "nan" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def cis(c, d=3):
    return f"[{f(c[0], d)}, {f(c[1], d)}]"


def write_md(r: dict, md_path: Path) -> None:
    L = ["# x112: the eight-prompt vocabulary table on seven models, held-out dose-response\n"]
    L.append("Per model: 16 benign (x106 Hard-1K) + 16 harmful (x101 x83 cohort / x108 OR-Bench Toxic) held-out prompts with fresh twenty-sample `refusal2` near 0.5; 40 words (24 concordant over eight bins of the joint own / Qwen3.5-4B BP percentile, 16 discordant) appended at the end; twenty samples per cell; `refusal2` (x64 judge). CIs: prompt-level paired bootstrap, 10,000 draws.\n")
    if r.get("exclusions"):
        L.append("Grid rows in the x85 crossed cohort (the direction's fitting cohort) are dropped (`x85_exclusions.json`): " + "; ".join(f"{m['label']} {len(m['excluded']['prompt_ids'])} prompt(s)" + (f", word(s) {', '.join(m['excluded']['words'])}" if m["excluded"]["words"] else "") + f" → {m['n_prompts']} prompts × {m['n_words']} words" for m in r["models"] if m.get("excluded")) + ". The other models keep 32 × 40.\n")
        if any(m.get("topup") for m in r["models"]):
            L.append("The harmful stratum is topped back up to 16 prompts outside the x85 cohort (`prompts_topup.json`; the count above includes them): " + "; ".join(f"{m['label']} +{m['topup']['n_added']} (" + ", ".join(f"{a['id']} {a['fresh_refusal2']:.2f}" for a in m["topup"]["added"]) + ")" for m in r["models"] if m.get("topup")) + ".\n")
        else:
            L.append("Pre-top-up cohort (`--no-topup`): the retained original prompts alone.\n")
    else:
        L.append("Whole grid, no exclusion (`--full`): 32 prompts × 40 words per model.\n")
    cmp_ = r.get("table_comparison")
    if cmp_:
        L.append("## Between-model table agreement (eligible words)\n")
        names = list(cmp_["spearman"])
        L.append("| table | " + " | ".join(names) + " |\n|---|" + "---|" * len(names))
        for a in names:
            L.append(f"| {a} | " + " | ".join(f"{cmp_['spearman'][a][b]:+.2f}" for b in names) + " |")
        L.append(f"\nTop-100 / bottom-100 overlap with the Qwen3.5-4B BP table (chance {cmp_['chance_overlap']['top100']:.2f}): " + "; ".join(f"{a} {cmp_['overlap_top100'][a]['qwen4b_BP']} / {cmp_['overlap_bottom100'][a]['qwen4b_BP']}" for a in names) + ".\n")
    L.append("## Summary per model\n")
    L.append("| model | r(O) benign / harmful | adjacent bins in order (pooled) | top − bottom bin [95% CI] | ρ own table [95% CI] | ρ Qwen BP table [95% CI] | own − Qwen [95% CI] | ρ in-context | discordant closer own / Qwen | per-cell ρ own / ceiling |\n|---|---|---|---|---|---|---|---|---|---|")
    for m in r["models"]:
        ro, rq = m["rho"]["pooled"]["table_own"]["all40"], m["rho"]["pooled"]["table_qwen_BP"]["all40"]
        rk = m["rho"]["pooled"].get("key_in_context", {}).get("all40", {}).get("rho")
        dd = m["rho"]["pooled"]["own_minus_qwen_BP_all40"]
        dc = m["discordant"]["pooled"]["count_closer"]
        L.append(f"| {m['label']} | {m['r_O']['benign']['run']:.3f} / {m['r_O']['harmful']['run']:.3f} | {m['dose_response']['pooled']['adjacent_pairs_in_order']}/7 | {f(m['dose_response']['pooled']['top_minus_bottom_bin'])} {cis(m['dose_response']['pooled']['top_minus_bottom_bin_ci95'])} | {f(ro['rho'], 2)} {cis(ro['ci95_prompt_bootstrap'], 2)} | {f(rq['rho'], 2)} {cis(rq['ci95_prompt_bootstrap'], 2)} | {f(dd['diff'], 2)} {cis(dd['ci95_prompt_bootstrap'], 2)} | {f(rk, 2)} | {dc['own']} / {dc['qwen_BP']} | {f(m['rho']['pooled']['table_own']['per_cell']['rho'], 2)} / {m['noise_ceiling_per_cell']['pooled']['ceiling_spearman']:.2f} |")
    po = r["pooled_over_models"]
    L.append("\n## Dose-response averaged over the seven models (pooled prompts)\n")
    L.append("| bin | mean Δ_EO [95% CI] | " + " | ".join(LABELS[t] for t in TAGS if any(m["model_tag"] == t for m in r["models"])) + " |\n|---|---|" + "---|" * len(r["models"]))
    for b in po["dose_response"]["pooled"]["bins"]:
        L.append(f"| {b['bin']} | {f(b['delta_EO_mean_over_models'])} {cis(b['ci95'])} | " + " | ".join(f(v) for v in b["per_model"].values()) + " |")
    for s in SETS:
        v = po["dose_response"][s]
        L.append(f"\n- {s}: mean curve adjacent pairs in order {v['adjacent_pairs_in_order']}/7; top − bottom {f(v['top_minus_bottom'])} {cis(v['top_minus_bottom_ci95'])}; models at 7/7: {v['models_with_7_of_7']}; per model {v['adjacent_pairs_in_order_per_model']}.")
        for t in TABLES:
            e = po["rho_percentile_all_points"][s][t]
            L.append(f"  - {TABLE_LABEL[t]}: ρ over {e['n_points']} (model, word) points on the percentile scale {f(e['rho'], 2)}; per-model ρ mean {f(e['mean_per_model_rho'], 2)} (range {f(e['min_per_model_rho'], 2)} to {f(e['max_per_model_rho'], 2)}).")
        L.append(f"  - models where the own table ranks higher: {po['rho_percentile_all_points'][s]['n_models_own_higher']}/7; discordant words closer to own / Qwen: {po['discordant_totals'][s]['closer_own']} / {po['discordant_totals'][s]['closer_qwen_BP']}; side match own / Qwen: {po['discordant_totals'][s]['side_match_own']} / {po['discordant_totals'][s]['side_match_qwen_BP']} of {po['discordant_totals'][s]['n_words']}.")
    for m in r["models"]:
        L.append(f"\n## {m['label']} ({m['model_tag']})\n")
        L.append(f"sd unit {m['sd_unit']:.4f}. Missing generations {len(m['missing_generations'])}, missing refusal2 cells {len(m['missing_refusal2'])}, per-response fallback units {m['n_refusal2_per_response_fallback']}, cells at 4,096 {m['n_cells_retry4096']}, truncated responses {m['n_responses_truncated']}, empty responses {m['n_responses_empty']}; hijack units {m['n_hijack_units']}. Mean generated tokens O {u(m['mean_gen_tokens']['O'], 0)} / E {u(m['mean_gen_tokens']['E'], 0)}.\n")
        L.append("| prompts | n | r(O) (selection cell) | r(O) replicate | O − replicate [95% CI] | Spearman O vs replicate | historical | Spearman O vs historical |\n|---|---|---|---|---|---|---|---|")
        for s, v in m["r_O"].items():
            L.append(f"| {s} | {v['n_prompts']} | {v['run']:.3f} | {u(v['replicate'])} | {f(v['run_minus_replicate'])} {cis(v['run_minus_replicate_ci95'])} | {f(v['spearman_run_vs_replicate'], 2)} | {v['historical']:.3f} | {f(v['spearman_run_vs_historical'], 2)} |")
        L.append("\n| bin | words | own sd | Qwen BP sd | " + " | ".join(f"{s} Δ_EO [95% CI]" for s in SETS) + " |\n|---|---|---|---|" + "---|" * len(SETS))
        for b in range(len(BIN_NAMES)):
            row = m["dose_response"]["pooled"]["bins"][b]
            L.append(f"| {row['bin']} | {', '.join(row['words'])} | {row['own_sd_mean']:+.2f} | {row['qwen_BP_sd_mean']:+.2f} | " + " | ".join(f"{f(m['dose_response'][s]['bins'][b]['delta_EO'])} {cis(m['dose_response'][s]['bins'][b]['delta_EO_ci95'])}" for s in SETS) + " |")
        for s in SETS:
            v = m["dose_response"][s]
            L.append(f"\n- {s}: adjacent in order {v['adjacent_pairs_in_order']}/7 (bootstrap mean fraction {v['fraction_adjacent_in_order_boot_mean']:.2f}); top − bottom {f(v['top_minus_bottom_bin'])} {cis(v['top_minus_bottom_bin_ci95'])}; middle-four range {v['middle_four_bins_range']:.3f}.")
        L.append("\n| prompts | predictor | ρ all 40 [95% CI] | perm p | ρ concordant 24 [95% CI] | ρ middle 4 bins [95% CI] | partial ρ | per-cell ρ | ceiling |\n|---|---|---|---|---|---|---|---|---|")
        for s in SETS:
            for name, e in m["rho"][s].items():
                if not isinstance(e, dict) or "all40" not in e:
                    continue
                a, c, mm = e["all40"], e["concordant24"], e["middle4bins"]
                L.append(f"| {s} | {name} | {f(a['rho'], 2)} {cis(a['ci95_prompt_bootstrap'], 2)} | {a['perm_p']:.4f} | {f(c['rho'], 2)} {cis(c['ci95_prompt_bootstrap'], 2)} | {f(mm['rho'], 2)} {cis(mm['ci95_prompt_bootstrap'], 2)} | {f(e['partial_rho_all40_given_ntokens_logfreq'], 2)} | {f(e['per_cell']['rho'], 2)} | {m['noise_ceiling_per_cell'][s]['ceiling_spearman']:.2f} |")
            dd = m["rho"][s]["own_minus_qwen_BP_all40"]
            L.append(f"| {s} | own − Qwen BP | {f(dd['diff'], 2)} {cis(dd['ci95_prompt_bootstrap'], 2)} | p {dd['p']:.4f} | | | | | |")
        for s in SETS:
            v = m["discordant"][s]
            L.append(f"\nDiscordant ({s}): closer own {v['count_closer']['own']} / Qwen {v['count_closer']['qwen_BP']}; side match own {v['count_side_match']['own']}/16, Qwen {v['count_side_match']['qwen_BP']}/16; summed |error| own {v['sum_abs_err']['own']:.3f}, Qwen {v['sum_abs_err']['qwen_BP']:.3f}.")
            L.append("\n| quadrant | words | Δ_EO [95% CI] |\n|---|---|---|")
            for q, qv in v["quadrant_means"].items():
                L.append(f"| {q} | {', '.join(qv['words'])} | {f(qv['delta_EO'])} {cis(qv['delta_EO_ci95'])} |")
        L.append("\n| word | stratum | own sd (pct) | Qwen BP sd (pct) | tok | benign Δ_EO [95% CI] | harmful Δ_EO [95% CI] | pooled Δ_EO | key sd (pooled) | hijack |\n|---|---|---|---|---|---|---|---|---|---|")
        for w in m["per_word"]:
            L.append(f"| {w['word']} | {w['stratum']} | {w['own_sd']:+.2f} ({w['pct_own']:.1f}) | {w['qwen_BP_sd']:+.2f} ({w['pct_qwen_BP']:.1f}) | {w['n_tokens']} | {f(w['benign']['delta_EO'])} {cis(w['benign']['delta_EO_ci95'])} | {f(w['harmful']['delta_EO'])} {cis(w['harmful']['delta_EO_ci95'])} | {f(w['pooled']['delta_EO'])} | {f(w['pooled'].get('key_shift_sd'), 2)} | {u(m['hijack_per_word'].get(w['word']))} |")
        L.append("\n| group | benign hijack | benign refusal2 hijack / not | harmful hijack | harmful refusal2 hijack / not |\n|---|---|---|---|---|")
        for g in m["hijack_by_group"]:
            L.append(f"| {g['group']} | " + " | ".join(f"{u(g[s]['hijack_rate'])} | {u(g[s]['refusal2_given_hijack'])} / {u(g[s]['refusal2_given_no_hijack'])}" for s in POPULATIONS) + " |")
    md_path.write_text("\n".join(L) + "\n", encoding="utf-8")


def figures(models: list[dict], suffix: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"benign": "#2a6fdb", "harmful": "#d1495b", "pooled": "#333333"}
    n = len(models)
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.6), dpi=150, sharey=True)
    x = np.arange(len(BIN_NAMES))
    for i, ax in enumerate(axes.ravel()):
        if i >= n:
            ax.axis("off")
            continue
        m = models[i]
        for off, s in ((-0.1, "benign"), (0.1, "harmful")):
            rows = m["dose_response"][s]["bins"]
            y = np.asarray([b["delta_EO"] for b in rows])
            lo = y - np.asarray([b["delta_EO_ci95"][0] for b in rows])
            hi = np.asarray([b["delta_EO_ci95"][1] for b in rows]) - y
            ax.errorbar(x + off, y, yerr=[lo, hi], marker="o", ms=4, lw=1.4, capsize=2, color=colors[s], label=f"{s} prompts (n = {m['r_O'][s]['n_prompts']})")
        ax.axhline(0, color="#999", lw=0.7)
        ax.set_xticks(x, BIN_NAMES, rotation=45, fontsize=7)
        ax.set_title(m["label"], fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        if i % 4 == 0:
            ax.set_ylabel("Δ refusal2, E − O")
    for i, m in enumerate(models):
        axes.ravel()[i].legend(frameon=False, fontsize=7)
    fig.supxlabel("percentile bin in both the model's own table and the Qwen3.5-4B BP table (concordant words, 3 per bin)", fontsize=9)
    fig.tight_layout()
    fig.savefig(RUN / f"dose_response{suffix}.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.6), dpi=140, sharey="col")
    axes = np.asarray(axes).reshape(2, n)
    for j, m in enumerate(models):
        y = m["_dEO"]["pooled"]
        c = m["_kind"] == "concordant"
        for i, t in enumerate(TABLES):
            ax = axes[i, j]
            xv = m["_tab"][t]
            ax.scatter(xv[c], y[c], s=16, color="#2a6fdb", alpha=0.85, label="concordant")
            ax.scatter(xv[~c], y[~c], s=26, marker="D", facecolors="none", edgecolors="#333", label="discordant")
            e = m["rho"]["pooled"][f"table_{t}"]["all40"]
            ax.set_title(f"{m['label']}\n{TABLE_LABEL[t]}: ρ = {e['rho']:+.2f}", fontsize=8)
            ax.axhline(0, color="#aaa", lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=7)
            ax.set_xlabel("shift (sd)", fontsize=7)
            if j == 0:
                ax.set_ylabel("Δ refusal2, E − O (pooled prompts)", fontsize=8)
    axes[0, 0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(RUN / f"tables_vs_effect{suffix}.png")
    plt.close(fig)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tags", nargs="*", help="model tags (default: every model with prompts.json and words.json)")
    parser.add_argument("--full", action="store_true", help="the original 32 prompts per model, no x85 exclusion, no top-up; outputs carry the _full suffix")
    parser.add_argument("--no-topup", action="store_true", help="the x85 exclusion without the top-up prompts (the pre-top-up cohort); outputs carry the _pre_topup suffix")
    args = parser.parse_args()
    if args.full and args.no_topup:
        raise SystemExit("--full and --no-topup exclude each other")
    tags = [t for t in TAGS if paths(t)["prompts"].exists() and paths(t)["words"].exists()]
    if args.tags:
        tags = [t for t in args.tags if t in tags]
    suffix = "_full" if args.full else ("_pre_topup" if args.no_topup else "")
    exclusions = None if args.full else read_json(EXCLUSIONS)
    topup = not (args.full or args.no_topup)
    models = []
    for t in tags:
        ex = exclusions["models"][t] if exclusions else {"prompt_ids": [], "words": []}
        models.append(analyze_model(t, TAGS.index(t), tuple(ex["prompt_ids"]), tuple(ex["words"]), topup=topup))
    pooled = pooled_over_models(models)
    clean = [{k: v for k, v in m.items() if not k.startswith("_")} for m in models]
    results = {
        "run": "x112",
        "construct": "refusal2 (an explicit refusal statement is present), x64 judge, DeepSeek V4 Flash; twenty samples per cell judged as two units of ten",
        "cohort": "the original 32 prompts per model, unfiltered (--full)" if args.full else ("the original prompts outside the x85 crossed cohort, before the top-up (--no-topup)" if args.no_topup else "the original prompts outside the x85 crossed cohort plus the harmful top-up prompts (16 benign + 16 harmful per model)"),
        **({"exclusions": {"file": str(EXCLUSIONS.relative_to(RUN.parent.parent.parent)), "rule": exclusions["rule"], "x85_cohort": exclusions["x85_cohort"], "per_model": {t: exclusions["models"][t] for t in tags}}} if exclusions else {}),
        **({"topup": {"rule": "x112_select.py --topup-request / --topup-prompts: the harmful stratum of a model that lost prompts to the exclusion is refilled to 16 from pool_candidates.json (x85 ids excluded, same band and |rate - 0.5| order), reusing pool-stage O cells judged in band first and running a fresh O cell on an over-draw of max(2 x shortfall, 4) for the rest; the added prompts' O cell is the one that selected them", "per_model": {m["model_tag"]: m["topup"] for m in clean if m.get("topup")}}} if topup else {}),
        "models": clean,
        "pooled_over_models": pooled,
        "table_comparison": read_json(COMPARISON) if COMPARISON.exists() else None,
        "bootstrap": {"n": N_BOOT, "seed": f"numpy default_rng([{BOOT_SEED}, model index])", "unit": "prompts resampled with replacement within each population (pooled: within each population, then combined), paired across conditions; words fixed; two-sided bootstrap p; the across-model curve averages the seven models' independent bootstrap draws", "permutation": f"{N_PERM} permutations of Delta_EO over words, two-sided on |rho|"},
    }
    results_path = RESULTS.with_name(f"results{suffix}.json")
    md_path = RESULTS_MD.with_name(f"results{suffix}.md")
    atomic_json(results_path, results)
    write_md(results, md_path)
    figures(models, suffix)
    print(md_path.read_text()[:6000])


if __name__ == "__main__":
    main()
