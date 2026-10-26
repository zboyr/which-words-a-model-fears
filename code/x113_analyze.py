"""x113 Stage 6 (local): per model, does the model's own vocabulary table nominate the carrier word
of a natural over-refusal, and does replacing it in place by the table's lowest-shift substitute
remove the refusal?

Per model and population (core, llama_extra, hard1k = core + llama_extra, xstest, all), prompt-
paired on the prompts carrying both arms' labels, one 10,000-replicate two-stage bootstrap
(prompts within each stratum, then the ten binary responses within each arm): r(O), r(S_T),
r(C_T), delta_S = r(O) - r(S_T) and delta_C = r(O) - r(C_T) with Holm within the pair, S_T - C_T;
against x106 on the same prompts, x106's r(O) - r(S) and the paired r(S_T) - r(S), both split by
whether the table nominee is the x106 unit's top word; nomination agreement; Spearman between
the nominee's table shift (and the shift drop) and the per-prompt delta_S, with delta_C as the
comparison; the projection change of S_T and C_T on the model's direction and its Spearman
with the refusal change; the uptake rate and refusal2 given uptake; reach. Across models, Holm
over the seven transfer models within each contrast (Qwen3.5-4B the reference row), delta_S
against r(O), and nominee agreement with Qwen3.5-4B against the x112 table correlation.
Exclusion. Every model's direction was fitted on the x85 crossed cohort (96 prompts, 24 harm
words, 24 neutral twins) and the vocabulary tables are read on that direction, so those 48 words
are in-sample. By default the units listed in x85_exclusions.json — nominee an x85 word, prompt
containing an x85 harm word, or an x85 word in either arm's substitute — are dropped before any
statistic (45-48 of 397 per model), and the dropped units are reported on their own as the
`x85_excluded` population. `--full` analyses every unit instead and writes results_full.json,
results_full.md, substitution_full.{pdf,png} and nomination_full.{pdf,png}; the seeds, bootstrap
and Holm families are the same, only the unit set differs.

Writes results.json, results.md and two figures.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x106_common import holm  # noqa: E402
from x113_common import (  # noqa: E402
    ALL_TAGS,
    exclusions_path,
    ALPHA,
    ARMS,
    BOOTSTRAP_SEED,
    FRESH_TAGS,
    HARD_STRATA,
    IMPORTED_TAG,
    LABELS,
    N_BOOTSTRAP,
    N_RESPONSES,
    PROMPTS,
    RESULTS,
    RESULTS_MD,
    RUN,
    STRATA,
    TABLES_JSON,
    X112_COMPARISON,
    atomic_json,
    cells,
    cells_sha256,
    file_record,
    load_prompts,
    load_texts,
    normalize_word,
    paths,
    read_json,
    x106_refusal_rows,
)

POPULATIONS = {"core": ("core",), "llama_extra": ("llama_extra",), "hard1k": ("core", "llama_extra"), "xstest": ("xstest",), "all": ("core", "llama_extra", "xstest")}
FIGURE_ORDER = (IMPORTED_TAG, "qwen08b", "qwen2b", "qwen9b", "qwen27b", "llama", "gemma", "phi")
N_RHO_BOOT = 2_000
HEADLINE_POP = "hard1k"
EXCLUDED_POP = "x85_excluded"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="*", default=list(ALL_TAGS))
    parser.add_argument("--n-boot", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--full", action="store_true", help="every unit, no x85 exclusion; outputs carry the _full suffix")
    return parser.parse_args()


# --- small statistics ---------------------------------------------------------------------


def interval(samples: np.ndarray) -> list[float | None]:
    finite = np.asarray(samples)[np.isfinite(samples)]
    if not len(finite):
        return [None, None]
    return [float(v) for v in np.quantile(finite, (ALPHA / 2, 1 - ALPHA / 2))]


def one_sided_p(samples: np.ndarray) -> float | None:
    finite = np.asarray(samples)[np.isfinite(samples)]
    if not len(finite):
        return None
    return float((1 + np.sum(finite <= 0)) / (len(finite) + 1))


def two_sided_p(samples: np.ndarray) -> float | None:
    finite = np.asarray(samples)[np.isfinite(samples)]
    if not len(finite):
        return None
    low = (np.sum(finite <= 0) + 1) / (len(finite) + 1)
    high = (np.sum(finite >= 0) + 1) / (len(finite) + 1)
    return float(min(1.0, 2 * min(low, high)))


def summarize(point: float, samples: np.ndarray, n: int) -> dict:
    return {"estimate": (float(point) if np.isfinite(point) else None), "ci95": interval(samples), "one_sided_p": one_sided_p(samples), "two_sided_p": two_sided_p(samples), "n": int(n)}


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.ptp(x[mask]) == 0 or np.ptp(y[mask]) == 0:
        return float("nan")
    return float(stats.spearmanr(x[mask], y[mask]).statistic)


def spearman_record(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_boot: int = N_RHO_BOOT) -> dict:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return {"n": int(len(x)), "rho": None, "ci95": [None, None]}
    reps = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        reps[b] = spearman(x[idx], y[idx])
    return {"n": int(len(x)), "rho": spearman(x, y), "ci95": interval(reps)}


def nanmean_rows(a: np.ndarray) -> np.ndarray:
    """nanmean along the last axis, nan where a row has no finite value."""
    with np.errstate(invalid="ignore"):
        counts = np.isfinite(a).sum(axis=-1)
        sums = np.nansum(a, axis=-1)
        return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


# --- loading -------------------------------------------------------------------------------


def load_jsonl_cells(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("failed"):
            continue
        out[rec["cell_id"]] = rec
    return out


def load_generation_rows(tag: str, grid_sha: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for path in sorted(glob.glob(str(paths(tag)["gen"] / "generations_shard*of*.json"))):
        obj = read_json(path)
        if obj.get("experiment") != "x113" or obj.get("model_tag") != tag or obj.get("cells_sha256") != grid_sha:
            raise SystemExit(f"FATAL: {path} belongs to another tag or grid")
        for r in obj["rows"]:
            rows[r["cell_id"]] = r
    return rows


def rates_from_records(records: dict[str, dict], key: str, cell_ids: list[str | None], response_sha: dict[str, list[str]]) -> np.ndarray:
    out = np.full(len(cell_ids), np.nan)
    for k, cid in enumerate(cell_ids):
        if cid is None:
            continue
        rec = records.get(cid)
        if rec is None:
            continue
        if cid in response_sha and rec.get("response_sha256") != response_sha[cid]:
            raise SystemExit(f"FATAL: judge record {cid} is not this cell's responses")
        values = rec[key]
        if len(values) != N_RESPONSES or not all(isinstance(v, bool) for v in values):
            raise SystemExit(f"FATAL: invalid {key} record {cid}")
        out[k] = float(np.mean(values))
    return out


# --- per model -----------------------------------------------------------------------------


def estimands(r: dict[str, np.ndarray], agree: np.ndarray) -> dict[str, np.ndarray | float]:
    """Every estimand from per-prompt rate arrays (NaN where an arm lacks the prompt); works on
    [prompts] vectors and on [replicates, prompts] matrices."""
    o, s, c, s106, c106 = r["O"], r["S_T"], r["C_T"], r["S_x106"], r["C_x106"]
    both_s = np.isfinite(s) & np.isfinite(s106)
    ag = agree if o.ndim == 1 else agree
    out = {
        "r_O": nanmean_rows(o),
        "r_S_T": nanmean_rows(s),
        "r_C_T": nanmean_rows(c),
        "delta_S": nanmean_rows(o - s),
        "delta_C": nanmean_rows(o - c),
        "S_T_minus_C_T": nanmean_rows(c - s),
        "x106_delta_S": nanmean_rows(o - s106),
        "x106_delta_C": nanmean_rows(o - c106),
        "r_S_T_minus_r_S_x106": nanmean_rows(s - s106),
        "delta_S_on_shared": nanmean_rows(np.where(both_s, o - s, np.nan)),
        "x106_delta_S_on_shared": nanmean_rows(np.where(both_s, o - s106, np.nan)),
        "delta_S_agree": nanmean_rows(np.where(ag, o - s, np.nan)),
        "delta_S_disagree": nanmean_rows(np.where(~ag, o - s, np.nan)),
        "x106_delta_S_agree": nanmean_rows(np.where(ag, o - s106, np.nan)),
        "x106_delta_S_disagree": nanmean_rows(np.where(~ag, o - s106, np.nan)),
        "delta_C_agree": nanmean_rows(np.where(ag, o - c, np.nan)),
        "delta_C_disagree": nanmean_rows(np.where(~ag, o - c, np.nan)),
        "r_O_agree": nanmean_rows(np.where(ag, o, np.nan)),
        "r_O_disagree": nanmean_rows(np.where(~ag, o, np.nan)),
    }
    return out


def bootstrap(r: dict[str, np.ndarray], agree: np.ndarray, strata: np.ndarray, n_boot: int, seed: int, chunk: int = 500) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    point = {k: float(v) for k, v in estimands(r, agree).items()}
    rng = np.random.default_rng(seed)
    n = len(agree)
    groups = [np.where(strata == s)[0] for s in np.unique(strata)]
    samples = {k: np.empty(n_boot) for k in point}
    for start in range(0, n_boot, chunk):
        stop = min(start + chunk, n_boot)
        # resample prompts within each stratum, keeping the stratum sizes
        index = np.concatenate([g[rng.integers(0, len(g), size=(stop - start, len(g)))] for g in groups], axis=1)
        drawn = {}
        for arm, values in r.items():
            picked = values[index]
            redraw = rng.binomial(N_RESPONSES, np.nan_to_num(picked, nan=0.0)) / N_RESPONSES
            drawn[arm] = np.where(np.isfinite(picked), redraw, np.nan)
        rep = estimands(drawn, agree[index])
        for k in point:
            samples[k][start:stop] = rep[k]
    return point, samples


def model_report(tag: str, prompts: dict, n_boot: int, seed: int, qwen_nominees: dict[str, str] | None, excluded_ids: frozenset[str] = frozenset()) -> dict:
    texts = load_texts(tag)
    grid = cells(tag, texts)
    grid_sha = cells_sha256(grid)
    units = texts["units"]
    ids = [u["id"] for u in units]
    strata = np.asarray([u["stratum"] for u in units])
    gen_rows = load_generation_rows(tag, grid_sha)
    refusal = load_jsonl_cells(paths(tag)["refusal2"])
    uptake = load_jsonl_cells(paths(tag)["uptake"])
    response_sha = {cid: row["response_sha256"] for cid, row in gen_rows.items()}
    cell_of = {arm: [f"{i}|{arm}" if u["texts"].get(arm) is not None else None for i, u in zip(ids, units)] for arm in ARMS}
    for arm in ARMS:
        for cid in cell_of[arm]:
            if cid is not None and cid in refusal and cid not in gen_rows:
                raise SystemExit(f"FATAL: {tag}: refusal record {cid} without its generation row")
    x106_o = x106_refusal_rows(tag, "O_base")
    x106_s = x106_refusal_rows(tag, "S_base")
    x106_c = x106_refusal_rows(tag, "C_base")
    prompt_units = {u["id"]: u for u in prompts["units"]}

    def x106_rate(rows: dict[str, dict], prompt_id: str, sha: str | None) -> float:
        rec = rows.get(prompt_id)
        if rec is None or sha is None:
            return float("nan")
        if rec.get("prompt_sha256") != sha:
            raise SystemExit(f"FATAL: {tag}: x106 record {prompt_id} is not the expected text")
        return float(np.mean(rec["refusal"]))

    r = {
        "O": np.asarray([x106_rate(x106_o, i, prompt_units[i]["prompt_sha256"]) for i in ids]),
        "S_T": rates_from_records(refusal, "refusal", cell_of["S_T"], response_sha),
        "C_T": rates_from_records(refusal, "refusal", cell_of["C_T"], response_sha),
        "S_x106": np.asarray([x106_rate(x106_s, i, prompt_units[i]["x106"]["S_sha256"]) for i in ids]),
        "C_x106": np.asarray([x106_rate(x106_c, i, prompt_units[i]["x106"]["C_sha256"]) for i in ids]),
    }
    if not np.isfinite(r["O"]).all():
        raise SystemExit(f"FATAL: {tag}: an O_base refusal2 row is missing")
    up = {arm: rates_from_records(uptake, "uptake", cell_of[arm], response_sha) for arm in ARMS}
    agree = np.asarray([bool((u["x106_agreement"] or {}).get("nominee_is_top_word")) for u in units])
    nominee_shift = np.asarray([(u["nominee"] or {}).get("shift", np.nan) if u["nominee"] else np.nan for u in units], dtype=float)
    substitute_shift = np.asarray([((u["substitute"] or {}).get("shift") if u["substitute"] and u["substitute"].get("shift") is not None else np.nan) for u in units], dtype=float)
    control_shift = np.asarray([((u["control"] or {}).get("shift") if u["control"] else np.nan) for u in units], dtype=float)
    drop = nominee_shift - substitute_shift
    proj_obj = read_json(paths(tag)["projections"]) if paths(tag)["projections"].exists() else None
    proj = {}
    if proj_obj is not None:
        if proj_obj.get("texts_sha256") != texts["texts_sha256"]:
            raise SystemExit(f"FATAL: {tag}: projections belong to other texts")
        by_id = {row["id"]: row for row in proj_obj["rows"]}
        for text in ("O",) + ARMS:
            proj[text] = np.asarray([by_id[i]["proj_sd"].get(text, np.nan) if i in by_id else np.nan for i in ids], dtype=float)

    report: dict[str, Any] = {
        "model_id": LABELS[tag],
        "n_units": len(ids),
        "texts": {"n_S_T": int(sum(1 for u in units if u["texts"]["S_T"] is not None)), "n_C_T": int(sum(1 for u in units if u["texts"]["C_T"] is not None)), "n_same_as_x106": texts["n_same_as_x106"], "n_hedge": texts["n_hedge"], "n_no_substitute_note": texts["n_no_substitute_note"], "n_without_S_T": texts["n_without_substitute"]},
        "n_labelled": {"S_T": int(np.isfinite(r["S_T"]).sum()), "C_T": int(np.isfinite(r["C_T"]).sum()), "uptake_S_T": int(np.isfinite(up["S_T"]).sum()), "uptake_C_T": int(np.isfinite(up["C_T"]).sum())},
        "n_generated": {arm: int(sum(1 for cid in cell_of[arm] if cid in gen_rows)) for arm in ARMS},
        "n_copied_from_x106": {arm: int(sum(1 for cid in cell_of[arm] if cid in gen_rows and gen_rows[cid].get("copied_from_x106"))) for arm in ARMS},
        "populations": {},
    }
    nominees_norm = {u["id"]: (u["nominee"] or {}).get("norm") for u in units}
    kept = np.asarray([u["id"] not in excluded_ids for u in units])
    report["x85_exclusion"] = {
        "applied": bool(excluded_ids),
        "n_excluded": int((~kept).sum()),
        "n_retained": int(kept.sum()),
        "n_retained_by_stratum": {s: int((kept & (strata == s)).sum()) for s in STRATA},
    }
    population_masks: list[tuple[str, np.ndarray]] = [(name, np.isin(strata, members) & kept) for name, members in POPULATIONS.items()]
    if excluded_ids:
        population_masks.append((EXCLUDED_POP, ~kept))
    rng = np.random.default_rng(seed)
    for pop_index, (name, mask) in enumerate(population_masks):
        if mask.sum() < 2:
            continue
        sub = {k: v[mask] for k, v in r.items()}
        point, samples = bootstrap(sub, agree[mask], strata[mask], n_boot, seed + 10 * pop_index)
        n_both = {"S": int((np.isfinite(sub["O"]) & np.isfinite(sub["S_T"])).sum()), "C": int((np.isfinite(sub["O"]) & np.isfinite(sub["C_T"])).sum()), "SC": int((np.isfinite(sub["S_T"]) & np.isfinite(sub["C_T"])).sum()), "S_x106": int(np.isfinite(sub["S_x106"]).sum()), "S_shared": int((np.isfinite(sub["S_T"]) & np.isfinite(sub["S_x106"])).sum()), "agree": int((agree[mask] & np.isfinite(sub["S_T"])).sum()), "disagree": int((~agree[mask] & np.isfinite(sub["S_T"])).sum())}
        est = {k: summarize(point[k], samples[k], n_both["S"] if "S" in k and "C" not in k else n_both["C"] if "C" in k else int(mask.sum())) for k in point}
        pair = holm([est["delta_S"]["one_sided_p"] or 1.0, est["delta_C"]["one_sided_p"] or 1.0])
        est["delta_S"]["holm_p_within_pair"], est["delta_C"]["holm_p_within_pair"] = float(pair[0]), float(pair[1])
        eff_s = sub["O"] - sub["S_T"]
        eff_c = sub["O"] - sub["C_T"]
        with_s = np.isfinite(sub["S_T"])
        pop: dict[str, Any] = {
            "n": int(mask.sum()),
            "n_paired": n_both,
            "estimands": est,
            "reach": {
                "share_without_S_T": float(np.mean(~with_s)),
                "share_S_T_refusal_below_half_among_with_S_T": float(np.mean(sub["S_T"][with_s] < 0.5)) if with_s.any() else None,
                "share_S_T_refusal_below_half_of_population": float(np.sum(sub["S_T"][with_s] < 0.5) / mask.sum()),
                "share_S_T_does_not_lower_among_with_S_T": float(np.mean(eff_s[with_s] <= 0)) if with_s.any() else None,
                "share_O_refusal_at_or_above_half": float(np.mean(sub["O"] >= 0.5)),
            },
            "nomination_predicts_effect": {
                "spearman_nominee_shift_vs_delta_S": spearman_record(nominee_shift[mask], eff_s, rng),
                "spearman_shift_drop_vs_delta_S": spearman_record(drop[mask], eff_s, rng),
                "spearman_nominee_shift_vs_delta_C": spearman_record(nominee_shift[mask], eff_c, rng),
                "spearman_shift_drop_vs_delta_C": spearman_record(drop[mask], eff_c, rng),
                "spearman_control_shift_vs_delta_C": spearman_record(control_shift[mask], eff_c, rng),
                "mean_nominee_shift_sd": float(np.nanmean(nominee_shift[mask])),
                "mean_substitute_shift_sd": float(np.nanmean(substitute_shift[mask])),
                "mean_shift_drop_sd": float(np.nanmean(drop[mask])),
                "mean_control_shift_sd": float(np.nanmean(control_shift[mask])),
            },
        }
        ag_rows = [u["x106_agreement"] for u, m in zip(units, mask) if m and u["x106_agreement"]]
        pop["nomination_agreement"] = {
            "n": len(ag_rows),
            "nominee_is_top_word": float(np.mean([a["nominee_is_top_word"] for a in ag_rows])) if ag_rows else None,
            "nominee_in_unit": float(np.mean([a["nominee_in_unit"] for a in ag_rows])) if ag_rows else None,
            "top_word_rank_median": float(np.nanmedian([a["top_word_rank"] for a in ag_rows if a["top_word_rank"]])) if any(a["top_word_rank"] for a in ag_rows) else None,
            "top_word_in_top3": float(np.mean([(a["top_word_rank"] or 99) <= 3 for a in ag_rows])) if ag_rows else None,
            "agreement_with_qwen4b_nominee": (float(np.mean([nominees_norm[i] == qwen_nominees.get(i) for i, m in zip(ids, mask) if m])) if qwen_nominees and tag != IMPORTED_TAG else None),
        }
        if proj:
            d_s = proj["S_T"][mask] - proj["O"][mask]
            d_c = proj["C_T"][mask] - proj["O"][mask]
            pop["representation"] = {
                "proj_O_mean_sd": float(np.nanmean(proj["O"][mask])),
                "proj_S_T_minus_O_mean_sd": float(np.nanmean(d_s)) if np.isfinite(d_s).any() else None,
                "proj_S_T_minus_O_ci95": interval(np.asarray([np.nanmean(d_s[rng.integers(0, len(d_s), len(d_s))]) for _ in range(N_RHO_BOOT)])) if np.isfinite(d_s).any() else [None, None],
                "proj_C_T_minus_O_mean_sd": float(np.nanmean(d_c)) if np.isfinite(d_c).any() else None,
                "proj_C_T_minus_O_ci95": interval(np.asarray([np.nanmean(d_c[rng.integers(0, len(d_c), len(d_c))]) for _ in range(N_RHO_BOOT)])) if np.isfinite(d_c).any() else [None, None],
                "share_S_T_projection_below_O": float(np.nanmean(d_s < 0)) if np.isfinite(d_s).any() else None,
                "spearman_proj_change_vs_refusal_change_S_T": spearman_record(d_s, -eff_s, rng),
                "spearman_proj_change_vs_refusal_change_C_T": spearman_record(d_c, -eff_c, rng),
                "spearman_proj_change_S_T_vs_shift_drop": spearman_record(d_s, -drop[mask], rng),
            }
        pop["uptake"] = {}
        for arm in ARMS:
            cids = [cid for cid, m in zip(cell_of[arm], mask) if m and cid is not None and cid in uptake and cid in refusal]
            up_flags = np.asarray([v for cid in cids for v in uptake[cid]["uptake"]], dtype=bool)
            refs = np.asarray([v for cid in cids for v in refusal[cid]["refusal"]], dtype=float)
            pop["uptake"][arm] = {
                "n_cells": len(cids),
                "n_responses": int(len(up_flags)),
                "uptake_rate": float(up_flags.mean()) if len(up_flags) else None,
                "refusal2_given_uptake": float(refs[up_flags].mean()) if up_flags.any() else None,
                "refusal2_given_no_uptake": float(refs[~up_flags].mean()) if (~up_flags).any() else None,
                "per_cell_uptake_mean": float(np.nanmean(up[arm][mask])) if np.isfinite(up[arm][mask]).any() else None,
            }
        report["populations"][name] = pop

    # nomination summary and agreement
    nom = read_json(paths(tag)["nominations"])["summary"]
    hard = np.isin(strata, HARD_STRATA)
    nominees = {u["id"]: (u["nominee"] or {}).get("norm") for u in units}
    report["nomination"] = {
        "summary": nom,
        "agreement_with_qwen4b_nominee_hard1k": (float(np.mean([nominees[i] == qwen_nominees.get(i) for i, h in zip(ids, hard) if h])) if qwen_nominees and tag != IMPORTED_TAG else None),
        "agreement_with_qwen4b_nominee_all": (float(np.mean([nominees[i] == qwen_nominees.get(i) for i in ids])) if qwen_nominees and tag != IMPORTED_TAG else None),
        "most_chosen_substitutes_hard1k": [{"substitute": s, "n": n} for s, n in Counter((u["substitute"] or {}).get("substitute") for u, h in zip(units, hard) if h and u["substitute"]).most_common(15)],
        "n_candidates_per_unit_mean": float(np.mean([len(u["candidates"]) for u in units if u["candidates"]])) if any(u["candidates"] for u in units) else None,
        "n_accepted_per_unit_mean": float(np.mean([sum(1 for c in u["candidates"] if c["accepted"]) for u in units if u["candidates"]])) if any(u["candidates"] for u in units) else None,
        "share_chosen_has_table_shift": float(np.mean([u["substitute"].get("shift") is not None for u in units if u["substitute"]])) if any(u["substitute"] for u in units) else None,
        "chosen_is_drafter_first": float(np.mean([u["candidates"][0].get("chosen", False) for u in units if u["candidates"] and u["substitute"]])) if any(u["substitute"] for u in units) else None,
    }
    # output integrity
    texts_all = [t for row in gen_rows.values() for t in row["llm_responses"]]
    report["integrity"] = {
        "n_rows": len(gen_rows),
        "n_responses": len(texts_all),
        "retry_rows": int(sum(bool(row.get("used_retry")) for row in gen_rows.values())),
        "truncation_rate": float(np.mean([bool(v) for row in gen_rows.values() for v in row["truncated"]])) if gen_rows else None,
        "median_char_length": float(np.median([len(t) for t in texts_all])) if texts_all else None,
        "mean_gen_tokens": float(np.mean([v for row in gen_rows.values() for v in row["gen_tokens"]])) if gen_rows else None,
    }
    report["per_prompt"] = [
        {"id": i, "stratum": str(strata[k]), "x85_excluded": bool(not kept[k]), "r_O": float(r["O"][k]), "r_S_T": (float(r["S_T"][k]) if np.isfinite(r["S_T"][k]) else None), "r_C_T": (float(r["C_T"][k]) if np.isfinite(r["C_T"][k]) else None), "r_S_x106": (float(r["S_x106"][k]) if np.isfinite(r["S_x106"][k]) else None), "nominee": nominees[i], "nominee_is_x106_top": bool(agree[k]), "nominee_shift": (float(nominee_shift[k]) if np.isfinite(nominee_shift[k]) else None), "substitute": (units[k]["substitute"] or {}).get("substitute"), "substitute_shift": (float(substitute_shift[k]) if np.isfinite(substitute_shift[k]) else None), "control": (units[k]["control"] or {}).get("norm"), "control_synonym": (units[k]["control"] or {}).get("synonym"), "uptake_S_T": (float(up["S_T"][k]) if np.isfinite(up["S_T"][k]) else None), "uptake_C_T": (float(up["C_T"][k]) if np.isfinite(up["C_T"][k]) else None), **({"proj_S_T_minus_O_sd": (float(proj["S_T"][k] - proj["O"][k]) if np.isfinite(proj["S_T"][k]) else None), "proj_C_T_minus_O_sd": (float(proj["C_T"][k] - proj["O"][k]) if np.isfinite(proj["C_T"][k]) else None)} if proj else {})}
        for k, i in enumerate(ids)
    ]
    return report


# --- across models -------------------------------------------------------------------------


def cross_model(per_model: dict) -> dict:
    out: dict[str, Any] = {}
    comparison = read_json(X112_COMPARISON) if X112_COMPARISON.exists() else None
    pops = list(dict.fromkeys(p for t in per_model for p in per_model[t]["populations"]))
    for pop in pops:
        table = {}
        tags = [t for t in per_model if pop in per_model[t]["populations"]]
        transfer = [t for t in FRESH_TAGS if t in tags]
        for key in ("delta_S", "delta_C", "S_T_minus_C_T", "x106_delta_S", "r_S_T_minus_r_S_x106"):
            entries = {t: per_model[t]["populations"][pop]["estimands"][key] for t in tags}
            p_values = [entries[t]["one_sided_p"] if entries[t]["one_sided_p"] is not None else 1.0 for t in transfer]
            adjusted = dict(zip(transfer, holm(p_values))) if p_values else {}
            table[key] = {t: {**entries[t], "holm_p_over_transfer_models": adjusted.get(t), "in_family": t in adjusted} for t in tags}
        out[pop] = {"table": table, "transfer_models_in_family": transfer, "reference_model": IMPORTED_TAG}
    for pop in (HEADLINE_POP,):
        hard = [t for t in per_model if pop in per_model[t]["populations"]]
        r_o = [per_model[t]["populations"][pop]["estimands"]["r_O"]["estimate"] for t in hard]
        d_s = [per_model[t]["populations"][pop]["estimands"]["delta_S"]["estimate"] for t in hard]
        out[f"delta_S_vs_r_O_{pop}"] = {"tags": hard, "r_O": r_o, "delta_S": d_s, "spearman": spearman(np.asarray(r_o), np.asarray(d_s))}
        agree = {t: per_model[t]["populations"][pop]["nomination_agreement"]["agreement_with_qwen4b_nominee"] for t in hard if t != IMPORTED_TAG}
        corr = {t: comparison["spearman"]["qwen4b_BP"].get(t) for t in agree} if comparison else {}
        pairs = [(agree[t], corr[t]) for t in agree if corr.get(t) is not None and agree[t] is not None]
        out[f"nominee_agreement_with_qwen4b_vs_table_correlation_{pop}"] = {"agreement": agree, "x112_table_spearman_with_qwen4b_BP": corr, "spearman_over_models": spearman(np.asarray([p[1] for p in pairs]), np.asarray([p[0] for p in pairs])) if len(pairs) >= 3 else None}
        out[f"x106_top_word_agreement_{pop}"] = {t: per_model[t]["populations"][pop]["nomination_agreement"] for t in hard}
    out["x85_exclusion"] = {t: per_model[t]["x85_exclusion"] for t in per_model}
    return out


# --- figures -------------------------------------------------------------------------------


def draw_figures(per_model: dict, suffix: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 6.5, "ytick.labelsize": 7, "legend.fontsize": 6.0, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "axes.edgecolor": "#333333", "text.color": "#333333", "axes.labelcolor": "#333333", "grid.color": "#DDDDDD", "grid.linewidth": 0.5})
    blue, grey, green, orange, muted = "#0072B2", "#666666", "#009E73", "#E69F00", "#888888"
    order = [t for t in FIGURE_ORDER if t in per_model and "hard1k" in per_model[t]["populations"]]
    if not order:
        return
    xs = list(range(len(order)))

    def series(ax, pop, key, color, marker, shift, filled, with_ci=True):
        pts, lo, hi = [], [], []
        for t in order:
            e = per_model[t]["populations"][pop]["estimands"][key]
            if e["estimate"] is None:
                pts.append(np.nan)
                lo.append(0)
                hi.append(0)
                continue
            pts.append(e["estimate"])
            lo.append(e["estimate"] - (e["ci95"][0] if e["ci95"][0] is not None else e["estimate"]))
            hi.append((e["ci95"][1] if e["ci95"][1] is not None else e["estimate"]) - e["estimate"])
        ax.errorbar([x + shift for x in xs], pts, yerr=[lo, hi] if with_ci else None, fmt=marker, color=color, ms=3.0, elinewidth=0.7, capsize=1.2, lw=0, markerfacecolor=color if filled else "white", markeredgewidth=0.8, zorder=4)

    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.3))
    ax = axes[0]
    ax.axhline(0, color=muted, lw=0.6, zorder=1)
    series(ax, HEADLINE_POP, "delta_S", blue, "o", -0.22, True)
    series(ax, HEADLINE_POP, "delta_C", grey, "s", 0.0, False)
    series(ax, HEADLINE_POP, "x106_delta_S", green, "^", 0.22, False)
    ax.set_title("in-place substitution, Hard-1K", pad=2)
    ax.set_ylabel("refusal2 removed, r(O) − r(·)")
    ax.legend(handles=[Line2D([], [], color=blue, marker="o", ms=3, markerfacecolor=blue, lw=0, label="table nominee (S_T)"), Line2D([], [], color=grey, marker="s", ms=3, markerfacecolor="white", lw=0, label="table-nearest-zero word (C_T)"), Line2D([], [], color=green, marker="^", ms=3, markerfacecolor="white", lw=0, label="x106 direction nominee (S)")], loc="upper left", frameon=False, handlelength=1.2, labelspacing=0.2, borderaxespad=0.1)
    ax = axes[1]
    ax.axhline(0, color=muted, lw=0.6, zorder=1)
    series(ax, HEADLINE_POP, "delta_S_agree", blue, "o", -0.12, True)
    series(ax, HEADLINE_POP, "delta_S_disagree", blue, "D", 0.12, False)
    ax.set_title("S_T by nominee = x106 top word", pad=2)
    ax.set_ylabel("refusal2 removed by S_T")
    ax.legend(handles=[Line2D([], [], color=blue, marker="o", ms=3, markerfacecolor=blue, lw=0, label="same word as x106"), Line2D([], [], color=blue, marker="D", ms=3, markerfacecolor="white", lw=0, label="different word")], loc="upper left", frameon=False, handlelength=1.2, labelspacing=0.2, borderaxespad=0.1)
    ax = axes[2]
    ax.axhline(0, color=muted, lw=0.6, zorder=1)
    for shift, key, color, marker, filled in ((-0.12, "proj_S_T_minus_O_mean_sd", blue, "o", True), (0.12, "proj_C_T_minus_O_mean_sd", grey, "s", False)):
        vals = []
        for t in order:
            rep = per_model[t]["populations"][HEADLINE_POP].get("representation") or {}
            vals.append(rep.get(key) if rep.get(key) is not None else np.nan)
        ax.plot([x + shift for x in xs], vals, marker=marker, color=color, ms=3.0, lw=0, markerfacecolor=color if filled else "white", markeredgewidth=0.8, zorder=4)
    ax.set_title("boundary projection change", pad=2)
    ax.set_ylabel("proj(·) − proj(O), own sd")
    ax.legend(handles=[Line2D([], [], color=blue, marker="o", ms=3, markerfacecolor=blue, lw=0, label="S_T"), Line2D([], [], color=grey, marker="s", ms=3, markerfacecolor="white", lw=0, label="C_T")], loc="lower left", frameon=False, handlelength=1.2, labelspacing=0.2, borderaxespad=0.1)
    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels([LABELS[t] for t in order], rotation=55, ha="right")
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.grid(axis="y", zorder=0)
    fig.tight_layout(pad=0.4, w_pad=0.8)
    fig.savefig(RUN / f"substitution{suffix}.pdf")
    fig.savefig(RUN / f"substitution{suffix}.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.3))
    ax = axes[0]
    width = 0.27
    a_top = [per_model[t]["nomination"]["summary"]["agreement_hard1k"]["nominee_is_top_word"] for t in order]
    a_unit = [per_model[t]["nomination"]["summary"]["agreement_hard1k"]["nominee_in_unit"] for t in order]
    a_top3 = [per_model[t]["nomination"]["summary"]["agreement_hard1k"]["top_word_in_top3"] for t in order]
    ax.bar([x - width for x in xs], a_top, width, color=blue, label="nominee = x106 top word")
    ax.bar(xs, a_unit, width, color="#56B4E9", label="nominee inside x106 unit")
    ax.bar([x + width for x in xs], a_top3, width, color=grey, label="x106 top word in table top 3")
    ax.set_ylim(0, 1)
    ax.set_title("nomination agreement with x106, Hard-1K", pad=2)
    ax.set_ylabel("share of prompts")
    ax.legend(loc="upper right", frameon=False, handlelength=1.0, labelspacing=0.2, borderaxespad=0.1)
    ax = axes[1]
    ax.axhline(0, color=muted, lw=0.6, zorder=1)
    for shift, pop, color, marker, filled in ((-0.22, "core", blue, "o", True), (0.0, "llama_extra", orange, "D", False), (0.22, "xstest", green, "s", False)):
        series(ax, pop, "delta_S", color, marker, shift, filled)
    ax.set_title("S_T by population", pad=2)
    ax.set_ylabel("refusal2 removed by S_T")
    ax.legend(handles=[Line2D([], [], color=blue, marker="o", ms=3, markerfacecolor=blue, lw=0, label="core"), Line2D([], [], color=orange, marker="D", ms=3, markerfacecolor="white", lw=0, label="Llama add-on"), Line2D([], [], color=green, marker="s", ms=3, markerfacecolor="white", lw=0, label="XSTest")], loc="upper left", frameon=False, handlelength=1.2, labelspacing=0.2, borderaxespad=0.1)
    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels([LABELS[t] for t in order], rotation=55, ha="right")
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.grid(axis="y", zorder=0)
    fig.tight_layout(pad=0.4, w_pad=0.8)
    fig.savefig(RUN / f"nomination{suffix}.pdf")
    fig.savefig(RUN / f"nomination{suffix}.png", dpi=200)
    plt.close(fig)


# --- markdown ------------------------------------------------------------------------------


def f(x, d=3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:+.{d}f}"


def g(x, d=3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:.{d}f}"


def ci(e: dict) -> str:
    if e is None or e.get("estimate") is None:
        return "–"
    lo, hi = e["ci95"]
    return f"{f(e['estimate'])} [{f(lo)}, {f(hi)}]"


def write_md(out: dict, md_path: Path) -> None:
    pm = out["models"]
    order = [t for t in FIGURE_ORDER if t in pm]
    L = ["# x113: table nomination and table-lowest-shift substitution on the eight models\n"]
    L.append(f"Label construct: `refusal2` (x64 judge) everywhere. O rows are x106's O_base; S_T and C_T are judged here. Bootstrap: {out['n_bootstrap']} two-stage replicates (prompts within stratum, then the ten binary responses), 95 % intervals; Holm within the (Δ_S, Δ_C) pair and over the seven transfer models per contrast.\n")
    if out.get("exclusions"):
        e = out["exclusions"]
        L.append("Units touching the x85 crossed cohort — the cohort the directions, and so the tables, were fitted on — are dropped (`x85_exclusions.json`): " + "; ".join(f"{LABELS[t]} {pm[t]['x85_exclusion']['n_excluded']} → {pm[t]['x85_exclusion']['n_retained']}" for t in order) + ". The dropped units are reported below as the `x85_excluded` population. Whole-population results: `results_full.md`.\n")
    else:
        L.append("Every unit, no x85 exclusion (`--full`): 397 per model. The primary results are in `results.md`.\n")
    titles = {"hard1k": "Hard-1K (core + Llama add-on)", "core": "core", "llama_extra": "Llama add-on", "xstest": "XSTest", "all": "all strata", EXCLUDED_POP: "the x85 units dropped from the primary analysis"}
    for pop in ("hard1k", "core", "llama_extra", "xstest", "all", EXCLUDED_POP):
        if not any(pop in pm[t]["populations"] for t in order):
            continue
        L.append(f"\n## {titles[pop]}\n")
        L.append("| model | n S_T / C_T | r(O) | r(S_T) | r(C_T) | Δ_S [95% CI] | Holm pair | Holm transfer | Δ_C [95% CI] | S_T − C_T [95% CI] | x106 Δ_S same prompts | r(S_T) − r(S) paired | below 0.5 under S_T | no S_T |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for t in order:
            p = pm[t]["populations"].get(pop)
            if not p:
                continue
            e = p["estimands"]
            hp = out["cross_model"].get(pop, {}).get("table", {}).get("delta_S", {}).get(t, {}).get("holm_p_over_transfer_models")
            L.append(f"| {LABELS[t]} | {p['n_paired']['S']} / {p['n_paired']['C']} | {g(e['r_O']['estimate'])} | {g(e['r_S_T']['estimate'])} | {g(e['r_C_T']['estimate'])} | {ci(e['delta_S'])} | {g(e['delta_S']['holm_p_within_pair'], 4)} | {g(hp, 4) if hp is not None else ('ref' if t == IMPORTED_TAG else '–')} | {ci(e['delta_C'])} | {ci(e['S_T_minus_C_T'])} | {ci(e['x106_delta_S'])} (n {p['n_paired']['S_x106']}) | {ci(e['r_S_T_minus_r_S_x106'])} (n {p['n_paired']['S_shared']}) | {g(p['reach']['share_S_T_refusal_below_half_among_with_S_T'])} | {g(p['reach']['share_without_S_T'])} |")
    L.append("\n## Split by whether the table nominee is x106's top word (Hard-1K)\n")
    L.append("| model | n same / different | r(O) same / different | Δ_S same [95% CI] | Δ_S different [95% CI] | x106 Δ_S same | x106 Δ_S different | Δ_C same | Δ_C different |\n|---|---|---|---|---|---|---|---|---|")
    for t in order:
        p = pm[t]["populations"][HEADLINE_POP]
        e = p["estimands"]
        L.append(f"| {LABELS[t]} | {p['n_paired']['agree']} / {p['n_paired']['disagree']} | {g(e['r_O_agree']['estimate'])} / {g(e['r_O_disagree']['estimate'])} | {ci(e['delta_S_agree'])} | {ci(e['delta_S_disagree'])} | {ci(e['x106_delta_S_agree'])} | {ci(e['x106_delta_S_disagree'])} | {ci(e['delta_C_agree'])} | {ci(e['delta_C_disagree'])} |")
    L.append("\n## Nomination (Hard-1K)\n")
    L.append("| model | share of content words in table | nominee z median | nominee pct median | nominee = x106 top word | nominee in x106 unit | x106 top word rank median | x106 top word in top 3 | agreement with Qwen3.5-4B nominee | x112 table ρ with Qwen BP | most nominated |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for t in order:
        n = pm[t]["nomination"]
        s_ = n["summary"]
        a = pm[t]["populations"][HEADLINE_POP]["nomination_agreement"]
        corr = out["cross_model"][f"nominee_agreement_with_qwen4b_vs_table_correlation_{HEADLINE_POP}"]["x112_table_spearman_with_qwen4b_BP"].get(t)
        L.append(f"| {LABELS[t]} | {g(s_['share_looked_up_mean'])} | {g(s_['nominee_z_median'], 2)} | {g(s_['nominee_pct_median'], 1)} | {g(a['nominee_is_top_word'])} | {g(a['nominee_in_unit'])} | {g(a['top_word_rank_median'], 0)} | {g(a['top_word_in_top3'])} | {g(a['agreement_with_qwen4b_nominee'])} | {g(corr, 2) if corr is not None else '–'} | {', '.join(w['word'] + ' (' + str(w['n']) + ')' for w in s_['most_nominated_hard1k'][:6])} |")
    L.append("\n## Substitutes, shift and effect (Hard-1K)\n")
    L.append("| model | candidates per unit | accepted per unit | chosen has table shift | chosen = drafter's first | nominee shift (sd) | substitute shift (sd) | drop (sd) | control shift (sd) | ρ(nominee shift, Δ_S) | ρ(drop, Δ_S) | ρ(nominee shift, Δ_C) | ρ(drop, Δ_C) | most chosen substitutes |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in order:
        n = pm[t]["nomination"]
        q = pm[t]["populations"][HEADLINE_POP]["nomination_predicts_effect"]
        L.append(f"| {LABELS[t]} | {g(n['n_candidates_per_unit_mean'], 1)} | {g(n['n_accepted_per_unit_mean'], 1)} | {g(n['share_chosen_has_table_shift'])} | {g(n['chosen_is_drafter_first'])} | {f(q['mean_nominee_shift_sd'], 2)} | {f(q['mean_substitute_shift_sd'], 2)} | {f(q['mean_shift_drop_sd'], 2)} | {f(q['mean_control_shift_sd'], 2)} | {f(q['spearman_nominee_shift_vs_delta_S']['rho'], 2)} | {f(q['spearman_shift_drop_vs_delta_S']['rho'], 2)} | {f(q['spearman_nominee_shift_vs_delta_C']['rho'], 2)} | {f(q['spearman_shift_drop_vs_delta_C']['rho'], 2)} | {', '.join(w['substitute'] + ' (' + str(w['n']) + ')' for w in n['most_chosen_substitutes_hard1k'][:6])} |")
    L.append("\n## Representation (Hard-1K; own direction, own sd unit)\n")
    L.append("| model | proj(O) | proj(S_T) − proj(O) [95% CI] | proj(C_T) − proj(O) [95% CI] | share S_T below O | ρ(proj change, refusal change) S_T | ρ C_T | ρ(proj change S_T, shift drop) |\n|---|---|---|---|---|---|---|---|")
    for t in order:
        rep = pm[t]["populations"][HEADLINE_POP].get("representation")
        if not rep:
            L.append(f"| {LABELS[t]} | – | – | – | – | – | – | – |")
            continue
        L.append(f"| {LABELS[t]} | {f(rep['proj_O_mean_sd'], 2)} | {f(rep['proj_S_T_minus_O_mean_sd'], 2)} [{f(rep['proj_S_T_minus_O_ci95'][0], 2)}, {f(rep['proj_S_T_minus_O_ci95'][1], 2)}] | {f(rep['proj_C_T_minus_O_mean_sd'], 2)} [{f(rep['proj_C_T_minus_O_ci95'][0], 2)}, {f(rep['proj_C_T_minus_O_ci95'][1], 2)}] | {g(rep['share_S_T_projection_below_O'])} | {f(rep['spearman_proj_change_vs_refusal_change_S_T']['rho'], 2)} | {f(rep['spearman_proj_change_vs_refusal_change_C_T']['rho'], 2)} | {f(rep['spearman_proj_change_S_T_vs_shift_drop']['rho'], 2)} |")
    L.append("\n## Uptake (Hard-1K; reasoning-disabled label, response level)\n")
    L.append("| model | S_T cells | S_T uptake | refusal2 given uptake / not | C_T cells | C_T uptake | refusal2 given uptake / not |\n|---|---|---|---|---|---|---|")
    for t in order:
        u = pm[t]["populations"][HEADLINE_POP]["uptake"]
        a_, c_ = u["S_T"], u["C_T"]
        L.append(f"| {LABELS[t]} | {a_['n_cells']} | {g(a_['uptake_rate'])} | {g(a_['refusal2_given_uptake'])} / {g(a_['refusal2_given_no_uptake'])} | {c_['n_cells']} | {g(c_['uptake_rate'])} | {g(c_['refusal2_given_uptake'])} / {g(c_['refusal2_given_no_uptake'])} |")
    L.append("\n## Output integrity and reuse (all generated cells)\n")
    L.append("| model | S_T texts | C_T texts | same as x106 S / C | hedge | no substitute (drafter) | rows generated | copied from x106 | retry rows | truncation | median chars |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for t in order:
        m = pm[t]
        L.append(f"| {LABELS[t]} | {m['texts']['n_S_T']} | {m['texts']['n_C_T']} | {m['texts']['n_same_as_x106']['S_T']} / {m['texts']['n_same_as_x106']['C_T']} | {m['texts']['n_hedge']} | {m['texts']['n_no_substitute_note']} | {m['integrity']['n_rows']} | {m['n_copied_from_x106']['S_T'] + m['n_copied_from_x106']['C_T']} | {m['integrity']['retry_rows']} | {g(m['integrity']['truncation_rate'], 4)} | {g(m['integrity']['median_char_length'], 0)} |")
    x = out["cross_model"]
    L.append(f"\nAcross models (Hard-1K): Spearman between r(O) and Δ_S over the eight models {f(x[f'delta_S_vs_r_O_{HEADLINE_POP}']['spearman'], 2)}; between the seven transfer models' nominee agreement with Qwen3.5-4B and their x112 table correlation with the Qwen3.5-4B BP table {f(x[f'nominee_agreement_with_qwen4b_vs_table_correlation_{HEADLINE_POP}']['spearman_over_models'], 2)}.")
    md_path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    prompts = load_prompts()
    suffix = "_full" if args.full else ""
    exclusions = None if args.full else read_json(exclusions_path())
    qwen_texts = load_texts(IMPORTED_TAG) if paths(IMPORTED_TAG)["texts"].exists() else None
    qwen_nominees = {u["id"]: (u["nominee"] or {}).get("norm") for u in qwen_texts["units"]} if qwen_texts else None
    per_model: dict[str, dict] = {}
    for tag in args.tags:
        if not paths(tag)["texts"].exists():
            print(f"x113 analyze {tag}: no texts yet, skipped")
            continue
        dropped = frozenset()
        if exclusions is not None:
            entry = exclusions["models"][tag]
            if entry["texts_sha256"] != load_texts(tag)["texts_sha256"]:
                raise SystemExit(f"FATAL: {tag}: x85_exclusions.json was built from other texts; rerun x113_exclusions.py")
            dropped = frozenset(e["id"] for e in entry["excluded"])
        print(f"x113 analyze {tag}{' (full)' if args.full else f', {len(dropped)} x85 units dropped'}", flush=True)
        per_model[tag] = model_report(tag, prompts, args.n_boot, args.seed + 100 * ALL_TAGS.index(tag), qwen_nominees, dropped)
    out = {
        "experiment": "x113",
        "stage": 6,
        "prompts": file_record(PROMPTS),
        "tables": file_record(TABLES_JSON),
        "n_bootstrap": args.n_boot,
        "bootstrap": "two-stage paired: prompts within each stratum, then the ten binary responses within each arm; one replicate carries every estimand",
        "label_construct": "x64 refusal2 (an explicit refusal statement is present); O from x106's O_base rows, S_T and C_T judged here with the same contract",
        "populations": {k: list(v) for k, v in POPULATIONS.items()},
        **({"exclusions": {"file": str(exclusions_path().relative_to(RUN.parents[2])), "rule": exclusions["rule"], "x85_cohort": exclusions["x85_cohort"], "per_model": {t: {k: v for k, v in exclusions["models"][t].items() if k != "excluded"} for t in per_model}}} if exclusions else {"exclusions": None}),
        "models": per_model,
        "cross_model": cross_model(per_model),
    }
    results_path = RESULTS.with_name(f"results{suffix}.json")
    md_path = RESULTS_MD.with_name(f"results{suffix}.md")
    atomic_json(results_path, out)
    draw_figures(per_model, suffix)
    write_md(out, md_path)
    for t, m in per_model.items():
        e = m["populations"][HEADLINE_POP]["estimands"]
        a = m["populations"][HEADLINE_POP]["nomination_agreement"]
        print(f"  {t:8s} n {m['populations'][HEADLINE_POP]['n']:3d} r(O) {e['r_O']['estimate']:.3f} Δ_S {f(e['delta_S']['estimate'])} {[round(v, 3) for v in e['delta_S']['ci95']]} Δ_C {f(e['delta_C']['estimate'])} x106 Δ_S {f(e['x106_delta_S']['estimate'])} | agree top {g(a['nominee_is_top_word'])}")
    print(f"x113 results -> {results_path}")


if __name__ == "__main__":
    main()
