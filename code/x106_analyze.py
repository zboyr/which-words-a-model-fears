"""x106 Stage 4: per model, does the boundary direction nominate the carrier word of a natural
over-refusal, and does replacing that word in place remove it?

Stage 0 part (every model): the projection's AUC for the x104 base refusal (all Hard-1K prompts
and the request-unharmful stratum), a random-direction null, the cross-validated in-domain
refused-minus-answered direction and its cosine to the consensus direction, the flip and
induction AUCs under the written x104 conditions, the nomination summary, the lexicon
enrichment, and the top-1 agreement with Qwen3.5-4B's nomination on the cohort prompts.

Behaviour part (models with generations and labels): per-prompt `refusal2` rates per arm, the
three substitution contrasts r(O) - r(S|C|R) under one two-stage paired bootstrap (prompts, then
the ten binary responses) with Holm over the three, per projection stratum and for XSTest
separately; the Spearman between the Stage 0 top delta and the per-prompt substitution effect;
the representation change proj(S) - proj(O) against proj(C) - proj(O); the boundary write's
suppression on O against on S; reach. The same contrasts under GuardReasoner-8B's label are
the secondary table. Holm runs over the seven transfer models within each contrast; Qwen3.5-4B
is the reference row outside the family, as in x103 / x105.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from scipy import stats

from x86_analyze import printable_fraction, repeated_fourgram_dominated
from x106_common import (
    ALL_TAGS,
    ALPHA,
    BOOTSTRAP_SEED,
    BOUNDARY_POSITIONS,
    FRESH_TAGS,
    IMPORTED_TAG,
    MODELS,
    N_BOOTSTRAP,
    N_RESPONSES,
    QWEN4B_MODEL_ID,
    STRATA_EDGES,
    TEXT_CONDITIONS,
    arms_for,
    atomic_json,
    cohort_path,
    dose_label,
    doses_for,
    file_record,
    generations_path,
    guardreasoner_path,
    holm,
    induction_conditions,
    inputs_path,
    is_xstest,
    lexicon_words,
    load_cohort,
    load_mu,
    nominations_path,
    projection_of,
    projections_path,
    read_json,
    refusal_path,
    results_path,
    states_path,
    strata_of,
    unit_directions,
    units_with_text,
    written_conditions,
    x104_labels,
)

N_RANDOM_DIRECTIONS = 1_000
N_AUC_BOOTSTRAP = 2_000
N_FOLDS = 5
FIGURE_ORDER = (
    (IMPORTED_TAG, "Qwen3.5-4B"),
    ("qwen08b", "Qwen3.5-0.8B"),
    ("qwen2b", "Qwen3.5-2B"),
    ("qwen9b", "Qwen3.5-9B"),
    ("qwen27b", "Qwen3.5-27B"),
    ("llama", "Llama-3.2-3B"),
    ("gemma", "Gemma-4-E2B"),
    ("phi", "Phi-4-mini"),
)
CONTRASTS = ("substitute_S", "control_C", "removed_R")
JUDGES = ("refusal2", "guardreasoner")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="*", default=list(ALL_TAGS))
    parser.add_argument("--stage0-only", action="store_true")
    parser.add_argument("--n-boot", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--out", default=str(results_path()))
    parser.add_argument("--figure", default=str(results_path().with_name("x106_natural_lexical_carrier.pdf")))
    return parser.parse_args()


# --- small statistics ---------------------------------------------------------------------


def auc(score: np.ndarray, y: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    if y.min() == y.max():
        return float("nan")
    return float(stats.mannwhitneyu(score[y == 1], score[y == 0]).statistic / (np.sum(y == 1) * np.sum(y == 0)))


def auc_record(score: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_boot: int = N_AUC_BOOTSTRAP) -> dict:
    score, y = np.asarray(score, dtype=float), np.asarray(y, dtype=int)
    n = len(y)
    reps = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        reps[b] = auc(score[idx], y[idx])
    return {"n": int(n), "n_positive": int(y.sum()), "auc": auc(score, y), "ci95": [float(v) for v in np.nanquantile(reps, [ALPHA / 2, 1 - ALPHA / 2])]}


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.ptp(x[mask]) == 0 or np.ptp(y[mask]) == 0:
        return float("nan")
    return float(stats.spearmanr(x[mask], y[mask]).statistic)


def spearman_record(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_boot: int) -> dict:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    reps = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        reps[b] = spearman(x[idx], y[idx])
    finite = reps[np.isfinite(reps)]
    return {
        "n": int(len(x)),
        "rho": spearman(x, y),
        "ci95": [float(v) for v in np.quantile(finite, [ALPHA / 2, 1 - ALPHA / 2])] if len(finite) else [None, None],
        "p_rho_le_0": float((1 + np.sum(finite <= 0)) / (len(finite) + 1)) if len(finite) else None,
    }


def interval(samples: np.ndarray) -> list[float]:
    finite = np.asarray(samples)[np.isfinite(samples)]
    if not len(finite):
        return [float("nan"), float("nan")]
    return [float(v) for v in np.quantile(finite, (ALPHA / 2, 1 - ALPHA / 2))]


def one_sided_p(samples: np.ndarray) -> float:
    finite = np.asarray(samples)[np.isfinite(samples)]
    if not len(finite):
        return float("nan")
    return float((1 + np.sum(finite <= 0)) / (len(finite) + 1))


def summarize(point: Mapping[str, float], samples: Mapping[str, np.ndarray]) -> dict:
    return {
        name: {"estimate": (float(value) if np.isfinite(value) else None), "ci95": interval(samples[name]), "one_sided_p": one_sided_p(samples[name])}
        for name, value in point.items()
    }


# --- Stage 0 -------------------------------------------------------------------------------


def stage0_report(tag: str, cohort: dict | None, qwen_nominations: dict | None, rng: np.random.Generator) -> dict:
    proj_obj = read_json(projections_path(tag))
    nom_obj = read_json(nominations_path(tag, "O"))
    if proj_obj.get("model_tag") != tag or nom_obj.get("model_tag") != tag or nom_obj.get("cohort_only"):
        raise SystemExit(f"FATAL: {tag}: Stage 0 artifacts are missing or restricted")
    labels = x104_labels(tag)
    base = labels["base"]
    with np.load(states_path(tag)) as artifact:
        states = artifact["states"].astype(np.float32)
        state_ids = [str(v) for v in artifact["ids"]]
    mu = load_mu(tag)
    units, _ = unit_directions(mu)
    proj_by_id = {r["id"]: r["proj"] for r in proj_obj["rows"]}
    if state_ids != [r["id"] for r in proj_obj["rows"]]:
        raise SystemExit(f"FATAL: {tag}: states and projections are not aligned")
    hard_idx = [k for k, i in enumerate(state_ids) if not is_xstest(i) and i in base]
    ids = [state_ids[k] for k in hard_idx]
    h = states[hard_idx].astype(np.float64)
    proj = np.asarray([proj_by_id[i] for i in ids])
    y = np.asarray([int(base[i]["refusal"]) for i in ids])
    unharmful = np.asarray([base[i]["fields"].get("request") == "unharmful" for i in ids])

    out: dict[str, Any] = {
        "n_hard1k_labelled": len(ids),
        "base_refusal_rate": float(y.mean()),
        "request_unharmful_rate": float(unharmful.mean()),
        "projection": {"mean_refused": float(proj[y == 1].mean()), "mean_answered": float(proj[y == 0].mean()), "sd": float(proj.std())},
        "auc_base_refusal": auc_record(proj, y, rng),
        "auc_base_refusal_request_unharmful": auc_record(proj[unharmful], y[unharmful], rng),
        "auc_base_refusal_by_token": {p: auc_record(np.einsum("nd,d->n", h[:, t], units[t]), y, rng) for t, p in enumerate(BOUNDARY_POSITIONS)},
    }
    g = rng.standard_normal((N_RANDOM_DIRECTIONS, 2, mu.shape[1])).astype(np.float32)
    g /= np.linalg.norm(g, axis=2, keepdims=True)
    rand_auc = np.asarray([auc(v, y) for v in np.einsum("ntd,rtd->rn", h.astype(np.float32), g)])
    observed = out["auc_base_refusal"]["auc"]
    out["random_direction_null"] = {
        "n_random": N_RANDOM_DIRECTIONS,
        "mean_auc": float(np.nanmean(rand_auc)),
        "q95_abs_dev_from_half": float(np.nanquantile(np.abs(rand_auc - 0.5), 0.95)),
        "max_auc": float(np.nanmax(rand_auc)),
        "p_random_ge_observed": float((np.sum(rand_auc >= observed) + 1) / (N_RANDOM_DIRECTIONS + 1)),
    }
    folds = rng.permutation(len(ids)) % N_FOLDS
    cv_score = np.zeros(len(ids))
    cosines = []
    for f in range(N_FOLDS):
        tr, te = folds != f, folds == f
        d = h[tr][y[tr] == 1].mean(0) - h[tr][y[tr] == 0].mean(0)
        cosines.append({p: float(np.dot(d[t], units[t]) / np.linalg.norm(d[t])) for t, p in enumerate(BOUNDARY_POSITIONS)})
        cv_score[te] = np.einsum("ntd,td->n", h[te], d / np.linalg.norm(d, axis=1, keepdims=True))
    d_all = h[y == 1].mean(0) - h[y == 0].mean(0)
    out["in_domain_refused_minus_answered_direction"] = {
        "cv_auc": auc_record(cv_score, y, rng),
        "cosine_to_consensus_by_fold": cosines,
        "cosine_to_consensus_all_data": {p: float(np.dot(d_all[t], units[t]) / np.linalg.norm(d_all[t])) for t, p in enumerate(BOUNDARY_POSITIONS)},
        "random_unit_cosine_q99_abs": float(np.quantile(np.abs(g[:, 1, :] @ units[1].astype(np.float32)), 0.99)),
    }
    out["flip_prediction_among_base_refused"] = {}
    for condition in written_conditions(tag):
        if condition not in labels:
            continue
        ed = labels[condition]
        mask = np.asarray([(y[k] == 1) and (i in ed) for k, i in enumerate(ids)])
        flipped = np.asarray([1 - int(ed[i]["refusal"]) if i in ed else 0 for i in ids])[mask]
        out["flip_prediction_among_base_refused"][condition] = {**auc_record(-proj[mask], flipped, rng), "n_flipped": int(flipped.sum()), "statistic": "AUC of the negative projection for flipping to an answer"}
    out["induction_prediction_among_base_answered"] = {}
    for condition in induction_conditions(tag):
        if condition not in labels:
            continue
        ed = labels[condition]
        mask = np.asarray([(y[k] == 0) and (i in ed) for k, i in enumerate(ids)])
        induced = np.asarray([int(ed[i]["refusal"]) if i in ed else 0 for i in ids])[mask]
        out["induction_prediction_among_base_answered"][condition] = {**auc_record(proj[mask], induced, rng), "n_induced": int(induced.sum()), "statistic": "AUC of the projection for induced refusal"}

    lexicon = lexicon_words()
    rows = {r["id"]: r for r in nom_obj["rows"]}
    population = [i for i in ids if y[ids.index(i)] == 1 and unharmful[ids.index(i)]]

    def nomination_summary(sub_ids: list[str]) -> dict:
        sub = [rows[i] for i in sub_ids if i in rows]
        z = np.asarray([r["top_z"] for r in sub], dtype=float)
        share = np.asarray([r["top_share_of_excess"] for r in sub if r["top_share_of_excess"] is not None], dtype=float)
        content = [w for r in sub for w, k in zip(r["words"], range(len(r["words"]))) if k in {c["index"] for c in r["content_word_z"]} or k in r["unit_indices"]]
        from collections import Counter

        from x106_common import normalize_word

        base_rate = float(np.mean([normalize_word(w) in lexicon for w in content])) if content else float("nan")
        counter = Counter(r["unit_norm"] for r in sub)
        return {
            "n": len(sub),
            "top_z_median": float(np.nanmedian(z)) if len(z) else None,
            "top_z_quartiles": [float(v) for v in np.nanquantile(z, [0.25, 0.75])] if len(z) else None,
            "frac_top_z_ge_2": float(np.nanmean(z >= 2)) if len(z) else None,
            "top_share_of_excess_median": float(np.median(share)) if len(share) else None,
            "frac_pair_wins": float(np.mean([r["pair_wins"] for r in sub])) if sub else None,
            "frac_top_in_lexicon": float(np.mean([r["top_in_lexicon"] for r in sub])) if sub else None,
            "frac_unit_in_lexicon": float(np.mean([r["unit_in_lexicon"] for r in sub])) if sub else None,
            "content_word_lexicon_base_rate": base_rate,
            "most_nominated_units": [{"unit": u, "n": c, "in_lexicon": all(w in lexicon for w in u.split())} for u, c in counter.most_common(25)],
        }

    out["nomination_all_hard1k"] = nomination_summary(ids)
    out["nomination_population"] = nomination_summary(population)
    out["nomination_xstest"] = nomination_summary([i for i in state_ids if is_xstest(i)])
    out["population_n"] = len(population)
    out["population_strata"] = {s: int(n) for s, n in zip(*np.unique(strata_of([proj_by_id[i] for i in population]), return_counts=True))}
    td = np.asarray([rows[i]["top_delta"] for i in ids])
    out["top_delta_predicts_base_refusal"] = auc_record(td, y, rng)
    if qwen_nominations is not None and tag != IMPORTED_TAG:
        q = {r["id"]: r for r in qwen_nominations["rows"]}
        shared = [u["id"] for u in cohort["units"]] if cohort else population
        shared = [i for i in shared if i in q and i in rows]
        out["agreement_with_qwen_nomination"] = {
            "n": len(shared),
            "top_word": float(np.mean([rows[i]["top_norm"] == q[i]["top_norm"] for i in shared])) if shared else None,
            "unit": float(np.mean([rows[i]["unit_norm"] == q[i]["unit_norm"] for i in shared])) if shared else None,
            "unit_overlaps": float(np.mean([bool(set(rows[i]["unit_norm"].split()) & set(q[i]["unit_norm"].split())) for i in shared])) if shared else None,
            "qwen_unit_in_this_models_top3": float(np.mean([q[i]["top_norm"] in {normalize_top(rows[i], k) for k in top_k_indices(rows[i], 3)} for i in shared])) if shared else None,
        }
    out["artifacts"] = {"projections": file_record(projections_path(tag)), "nominations": file_record(nominations_path(tag, "O")), "states": file_record(states_path(tag)), "batched_vs_single_audits": nom_obj.get("batched_vs_single_audits")}
    return out


def top_k_indices(row: dict, k: int) -> list[int]:
    from x106_common import is_content_word

    d = np.asarray(row["delta_unigram"], dtype=float)
    content = np.asarray([is_content_word(w) for w in row["words"]])
    order = np.argsort(-np.where(content, d, -np.inf))
    return [int(i) for i in order[:k] if content[i]]


def normalize_top(row: dict, index: int) -> str:
    from x106_common import normalize_word

    return normalize_word(row["words"][index])


# --- behaviour -----------------------------------------------------------------------------


def load_generation_arm(tag: str, arm: str, cohort: dict) -> tuple[dict[str, dict], dict]:
    spec = next(s for s in arms_for(tag) if s["arm"] == arm)
    path = generations_path(tag, arm)
    obj = read_json(path)
    if obj.get("experiment") != "x106" or obj.get("arm") != arm or obj.get("model_tag") != tag or obj.get("cohort_sha256") != cohort["cohort_sha256"] or obj.get("text_condition") != spec["text"]:
        raise SystemExit(f"FATAL: {tag}/{arm}: generation lineage is not valid")
    if spec["dose"] != 0.0 and obj.get("site_audit", {}).get("status") != "pass":
        raise SystemExit(f"FATAL: {tag}/{arm}: write-site audit did not pass")
    units = {u["id"]: u for u in units_with_text(cohort, spec["text"])}
    rows = {}
    for row in obj["rows"]:
        prompt_id = str(row["id"])
        if prompt_id not in units:
            continue
        if row.get("condition") != arm or row.get("prompt_sha256") != units[prompt_id]["prompt_sha256"][spec["text"]] or row.get("arms_sha256") != cohort["cohort_sha256"] or len(row.get("llm_responses", [])) != N_RESPONSES or abs(float(row["signed_dose"]) - float(obj["signed_dose"])) > 1e-12:
            raise SystemExit(f"FATAL: invalid {tag}/{arm} generation row {prompt_id}")
        if prompt_id in rows:
            raise SystemExit(f"FATAL: duplicate {tag}/{arm} generation row {prompt_id}")
        rows[prompt_id] = row
    meta = {"generations": file_record(path), "signed_dose": float(obj["signed_dose"]), "text_condition": spec["text"], "n_rows": len(rows), "n_units_with_text": len(units)}
    return rows, meta


def load_refusal2(tag: str, arm: str, rows: dict[str, dict], cohort_sha: str) -> dict[str, np.ndarray]:
    path = refusal_path(tag, arm)
    labels: dict[str, np.ndarray] = {}
    if not path.exists():
        return labels
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("failed"):
                continue
            prompt_id = str(record["prompt_id"])
            source = rows.get(prompt_id)
            if source is None:
                continue
            values = record.get("refusal", [])
            if record.get("condition") != arm or record.get("response_sha256") != source["response_sha256"] or record.get("arms_sha256") != cohort_sha or len(values) != N_RESPONSES or not all(isinstance(v, bool) for v in values):
                raise SystemExit(f"FATAL: invalid {tag}/{arm} refusal record {prompt_id}")
            if prompt_id in labels:
                raise SystemExit(f"FATAL: duplicate successful {tag}/{arm} refusal record {prompt_id}")
            labels[prompt_id] = np.asarray(values, dtype=float)
    return labels


def load_guardreasoner(tag: str, arm: str, rows: dict[str, dict]) -> dict[str, np.ndarray]:
    pattern = str(guardreasoner_path(tag, arm, 0, 1)).replace("shard0of1", "shard*of*")
    paths = sorted(glob.glob(pattern))
    merged = guardreasoner_path(tag, arm)
    if merged.exists():
        paths.append(str(merged))
    best: dict[str, dict] = {}
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = record["key"]
                if key not in best or (record["parse_ok"] and not best[key]["parse_ok"]):
                    best[key] = record
    labels: dict[str, np.ndarray] = {}
    for prompt_id, row in rows.items():
        values = np.full(N_RESPONSES, np.nan)
        for k in range(N_RESPONSES):
            record = best.get(f"x106|{tag}|{arm}|{prompt_id}|{k}")
            if record is None:
                continue
            if record.get("response_sha256") != row["response_sha256"][k]:
                raise SystemExit(f"FATAL: {tag}/{arm}: GuardReasoner record {prompt_id}/{k} is not this response")
            if record["parse_ok"]:
                values[k] = float(record["refusal"])
        if np.isfinite(values).any():
            labels[prompt_id] = values
    return labels


def rates_from(labels: dict[str, np.ndarray], ids: list[str]) -> np.ndarray:
    out = np.full(len(ids), np.nan)
    for k, prompt_id in enumerate(ids):
        values = labels.get(prompt_id)
        if values is not None and np.isfinite(values).any():
            out[k] = float(np.nanmean(values))
    return out


def paired_mean(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(a[mask] - b[mask])) if mask.any() else float("nan")


def behaviour_estimands(rates: Mapping[str, np.ndarray], doses: tuple[float, ...], hard: np.ndarray) -> dict[str, float]:
    """Every behavioural estimand from per-prompt rate arrays (NaN where an arm lacks the prompt)."""
    o, s, c, r = (rates[f"{t}_base"] for t in TEXT_CONDITIONS)
    out = {
        "substitute_S": paired_mean(o[hard], s[hard]),
        "control_C": paired_mean(o[hard], c[hard]),
        "removed_R": paired_mean(o[hard], r[hard]),
        "S_minus_C": paired_mean(o[hard] - s[hard], o[hard] - c[hard]),
        "xstest_substitute_S": paired_mean(o[~hard], s[~hard]),
        "xstest_control_C": paired_mean(o[~hard], c[~hard]),
        "xstest_removed_R": paired_mean(o[~hard], r[~hard]),
    }
    for dose in doses:
        label = dose_label(dose)
        o_w, s_w = rates[f"O_{label}"], rates[f"S_{label}"]
        out[f"suppress_O_{label}"] = paired_mean(o[hard], o_w[hard])
        out[f"suppress_S_{label}"] = paired_mean(s[hard], s_w[hard])
        out[f"suppress_O_minus_S_{label}"] = paired_mean(o[hard] - o_w[hard], s[hard] - s_w[hard])
        out[f"residual_S_written_{label}"] = float(np.nanmean(s_w[hard]))
        out[f"xstest_suppress_O_{label}"] = paired_mean(o[~hard], o_w[~hard])
    if len(doses) == 2:
        a, b = (dose_label(d) for d in sorted(doses, key=abs))
        out["suppress_O_dose_order"] = out[f"suppress_O_{b}"] - out[f"suppress_O_{a}"]
    return out


def bootstrap_behaviour(labels: Mapping[str, dict[str, np.ndarray]], ids: list[str], doses: tuple[float, ...], hard: np.ndarray, n_boot: int, seed: int, chunk: int = 250) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    rates = {arm: rates_from(values, ids) for arm, values in labels.items()}
    point = behaviour_estimands(rates, doses, hard)
    rng = np.random.default_rng(seed)
    n = len(ids)
    names = list(point)
    samples = {name: np.empty(n_boot) for name in names}
    arms = list(rates)
    for start in range(0, n_boot, chunk):
        stop = min(start + chunk, n_boot)
        index = rng.integers(0, n, size=(stop - start, n))
        drawn = {}
        for arm in arms:
            picked = rates[arm][index]
            redraw = rng.binomial(N_RESPONSES, np.nan_to_num(picked, nan=0.0)) / N_RESPONSES
            drawn[arm] = np.where(np.isfinite(picked), redraw, np.nan)
        hard_drawn = hard[index]
        for offset in range(stop - start):
            replicate = behaviour_estimands({arm: drawn[arm][offset] for arm in arms}, doses, hard_drawn[offset])
            for name in names:
                samples[name][start + offset] = replicate[name]
    return point, samples


def integrity(rows_by_arm: dict[str, dict[str, dict]]) -> dict:
    report = {}
    for arm, rows in rows_by_arm.items():
        texts = [t for row in rows.values() for t in row["llm_responses"]]
        truncated = [bool(v) for row in rows.values() for v in row["truncated"]]
        report[arm] = {
            "n_rows": len(rows),
            "n_responses": len(texts),
            "retry_rows": sum(bool(row.get("used_retry")) for row in rows.values()),
            "truncation_rate": float(np.mean(truncated)) if truncated else None,
            "median_char_length": float(np.median([len(t) for t in texts])) if texts else None,
            "median_printable_fraction": float(np.median([printable_fraction(t) for t in texts])) if texts else None,
            "repeated_fourgram_dominated_fraction": float(np.mean([repeated_fourgram_dominated(t) for t in texts])) if texts else None,
        }
    return report


def text_projections(tag: str, cohort: dict) -> dict[str, dict[str, float]]:
    """{text: {id: proj}} from the Stage 0 pass on O and the Stage 2 passes on S, C, R where they exist."""
    out: dict[str, dict[str, float]] = {}
    for text in TEXT_CONDITIONS:
        path = nominations_path(tag, text)
        if not path.exists():
            continue
        obj = read_json(path)
        if obj.get("model_tag") != tag or obj.get("text") != text:
            raise SystemExit(f"FATAL: {tag}: {path} belongs to another tag/text")
        out[text] = {r["id"]: float(r["proj_full"]) for r in obj["rows"]}
    return out


def behaviour_report(tag: str, cohort: dict, stage0: dict, n_boot: int, seed: int) -> dict | None:
    arms = arms_for(tag)
    available = [spec["arm"] for spec in arms if generations_path(tag, spec["arm"]).exists()]
    if len(available) != len(arms):
        return {"status": "incomplete", "generated_arms": available, "missing_arms": [s["arm"] for s in arms if s["arm"] not in available]}
    units = list(cohort["units"])
    ids = [u["id"] for u in units]
    hard = np.asarray([not is_xstest(i) for i in ids])
    rows_by_arm: dict[str, dict[str, dict]] = {}
    arm_meta = {}
    labels_by_judge: dict[str, dict[str, dict[str, np.ndarray]]] = {j: {} for j in JUDGES}
    for spec in arms:
        rows, meta = load_generation_arm(tag, spec["arm"], cohort)
        rows_by_arm[spec["arm"]] = rows
        labels_by_judge["refusal2"][spec["arm"]] = load_refusal2(tag, spec["arm"], rows, cohort["cohort_sha256"])
        labels_by_judge["guardreasoner"][spec["arm"]] = load_guardreasoner(tag, spec["arm"], rows)
        meta["n_refusal2"] = len(labels_by_judge["refusal2"][spec["arm"]])
        meta["n_guardreasoner"] = len(labels_by_judge["guardreasoner"][spec["arm"]])
        arm_meta[spec["arm"]] = meta
    doses = doses_for(tag)
    proj = text_projections(tag, cohort)
    delta_top = np.asarray([u["top_delta"] for u in units])
    strata = np.asarray([u["stratum"] for u in units])
    report: dict[str, Any] = {"status": "complete", "arms": arm_meta, "integrity": integrity(rows_by_arm), "n_units": len(ids), "n_hard1k": int(hard.sum()), "n_xstest": int((~hard).sum()), "judges": {}}
    for judge in JUDGES:
        labels = labels_by_judge[judge]
        if any(len(labels[arm]) == 0 for arm in labels):
            report["judges"][judge] = {"status": "missing labels", "arms_without_labels": [arm for arm in labels if len(labels[arm]) == 0]}
            continue
        rng = np.random.default_rng(seed + (0 if judge == "refusal2" else 1))
        point, samples = bootstrap_behaviour(labels, ids, doses, hard, n_boot, seed + (0 if judge == "refusal2" else 1))
        estimands = summarize(point, samples)
        family = [estimands[c]["one_sided_p"] for c in CONTRASTS]
        adjusted = holm(family)
        for c, p in zip(CONTRASTS, adjusted):
            estimands[c]["holm_p_within_contrast_family"] = float(p)
        rates = {arm: rates_from(values, ids) for arm, values in labels.items()}
        o, s, c_ = rates["O_base"], rates["S_base"], rates["C_base"]
        per_stratum = {}
        for stratum in ("low", "mid", "high"):
            mask = hard & (strata == stratum)
            per_stratum[stratum] = {"n": int(mask.sum()), "substitute_S": paired_mean(o[mask], s[mask]), "control_C": paired_mean(o[mask], c_[mask]), "removed_R": paired_mean(o[mask], rates["R_base"][mask]), "r_O": float(np.nanmean(o[mask])) if mask.any() else None}
        effect_s = o - s
        effect_c = o - c_
        nomination_predicts = {
            "spearman_top_delta_vs_substitution_effect": spearman_record(delta_top[hard], effect_s[hard], rng, n_boot),
            "spearman_top_delta_vs_control_effect": spearman_record(delta_top[hard], effect_c[hard], rng, n_boot),
        }
        representation = None
        if "S" in proj and "O" in proj:
            answered_mean = float(read_json(nominations_path(tag, "O"))["answered_mean_proj"])
            d_s = np.asarray([proj["S"].get(i, np.nan) - proj["O"].get(i, np.nan) for i in ids])
            d_c = np.asarray([proj.get("C", {}).get(i, np.nan) - proj["O"].get(i, np.nan) for i in ids])
            p_s = np.asarray([proj["S"].get(i, np.nan) for i in ids])
            representation = {
                "answered_mean_proj": answered_mean,
                "proj_S_minus_O_mean": float(np.nanmean(d_s[hard])),
                "proj_S_minus_O_ci95": interval(np.asarray([np.nanmean(d_s[hard][rng.integers(0, hard.sum(), hard.sum())]) for _ in range(min(n_boot, 2000))])),
                "proj_C_minus_O_mean": float(np.nanmean(d_c[hard])) if np.isfinite(d_c[hard]).any() else None,
                "share_S_below_answered_mean": float(np.nanmean(p_s[hard] < answered_mean)) if np.isfinite(p_s[hard]).any() else None,
                "share_O_below_answered_mean": float(np.mean(np.asarray([proj["O"][i] for i in ids])[hard] < answered_mean)),
                "spearman_proj_change_vs_refusal_change_S": spearman_record(d_s[hard], -effect_s[hard], rng, n_boot),
                "spearman_proj_change_vs_refusal_change_C": spearman_record(d_c[hard], -effect_c[hard], rng, n_boot) if np.isfinite(d_c[hard]).any() else None,
            }
        same_axis = {}
        for dose in doses:
            label = dose_label(dose)
            suppress_o = o - rates[f"O_{label}"]
            same_axis[label] = {"spearman_substitution_effect_vs_suppress_O": spearman_record(effect_s[hard], suppress_o[hard], rng, n_boot)}
        with_s = hard & np.isfinite(s)
        reach = {
            "n_population": int(hard.sum()),
            "share_without_substitute": float(np.mean(~np.isfinite(s[hard]))),
            "share_S_refusal_below_half": float(np.mean(with_s & (s < 0.5)) / max(hard.sum(), 1) * hard.sum() / hard.sum()) if hard.any() else None,
            "share_S_refusal_below_half_among_with_S": float(np.mean(s[with_s] < 0.5)) if with_s.any() else None,
            "share_S_does_not_lower_refusal_among_with_S": float(np.mean(effect_s[with_s] <= 0)) if with_s.any() else None,
            "mean_r_O": float(np.nanmean(o[hard])),
            "mean_r_S": float(np.nanmean(s[hard])),
            "mean_r_C": float(np.nanmean(c_[hard])),
        }
        reach["share_S_refusal_below_half"] = float(np.sum(with_s & (s < 0.5)) / hard.sum())
        report["judges"][judge] = {
            "status": "complete",
            "n_labelled_per_arm": {arm: int(np.isfinite(rates[arm]).sum()) for arm in rates},
            "estimands": estimands,
            "contrast_family": list(CONTRASTS),
            "per_stratum": per_stratum,
            "nomination_predicts_effect": nomination_predicts,
            "representation": representation,
            "same_axis_as_write": same_axis,
            "reach": reach,
            "dose_order_O": (point.get("suppress_O_dose_order") > 0) if "suppress_O_dose_order" in point else None,
        }
    return report


def cross_model(per_model: dict) -> dict:
    out = {}
    for judge in JUDGES:
        rows = {tag: per_model[tag]["behaviour"]["judges"][judge] for tag in per_model if per_model[tag].get("behaviour", {}).get("status") == "complete" and per_model[tag]["behaviour"]["judges"].get(judge, {}).get("status") == "complete"}
        if not rows:
            continue
        table = {}
        transfer = [tag for tag in FRESH_TAGS if tag in rows]
        keys = list(CONTRASTS) + ["S_minus_C", "suppress_O_m150", "suppress_S_m150", "suppress_O_minus_S_m150"]
        for key in keys:
            entries = {tag: rows[tag]["estimands"].get(key) for tag in rows}
            p_values = [entries[tag]["one_sided_p"] for tag in transfer if entries.get(tag)]
            adjusted = dict(zip([tag for tag in transfer if entries.get(tag)], holm(p_values))) if p_values else {}
            table[key] = {tag: {**(entries[tag] or {}), "holm_p_over_transfer_models": adjusted.get(tag), "in_family": tag in adjusted} for tag in rows}
        out[judge] = {"table": table, "transfer_models_in_family": transfer, "reference_model": IMPORTED_TAG}
    agreement = {tag: per_model[tag]["stage0"].get("agreement_with_qwen_nomination") for tag in per_model if tag != IMPORTED_TAG}
    out["nomination_agreement_with_qwen"] = agreement
    return out


def draw_figure(per_model: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 6.5, "ytick.labelsize": 7, "legend.fontsize": 6.0, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "axes.edgecolor": "#333333", "text.color": "#333333", "axes.labelcolor": "#333333", "grid.color": "#DDDDDD", "grid.linewidth": 0.5})
    blue, orange, grey, green, muted = "#0072B2", "#E69F00", "#666666", "#009E73", "#888888"
    order = [(tag, label) for tag, label in FIGURE_ORDER if per_model.get(tag, {}).get("behaviour", {}).get("judges", {}).get("refusal2", {}).get("status") == "complete"]
    if not order:
        return
    xs = list(range(len(order)))
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.3))

    def series(ax, key, color, marker, shift, filled):
        points, lower, upper = [], [], []
        for tag, _ in order:
            entry = per_model[tag]["behaviour"]["judges"]["refusal2"]["estimands"].get(key)
            if entry is None or entry["estimate"] is None:
                points.append(np.nan)
                lower.append(0)
                upper.append(0)
                continue
            points.append(entry["estimate"])
            lower.append(entry["estimate"] - entry["ci95"][0])
            upper.append(entry["ci95"][1] - entry["estimate"])
        ax.errorbar([x + shift for x in xs], points, yerr=[lower, upper], fmt=marker, color=color, ms=3.0, elinewidth=0.7, capsize=1.2, lw=0, markerfacecolor=color if filled else "white", markeredgewidth=0.8, zorder=4)

    ax = axes[0]
    ax.axhline(0, color=muted, lw=0.6, zorder=1)
    series(ax, "substitute_S", blue, "o", -0.2, True)
    series(ax, "control_C", grey, "s", 0.0, False)
    series(ax, "removed_R", green, "^", 0.2, False)
    ax.set_title("in-place substitution, dose 0", pad=2)
    ax.set_ylabel("refusal2 removed, r(O) − r(·)")
    ax.legend(handles=[Line2D([], [], color=blue, marker="o", ms=3, markerfacecolor=blue, lw=0, label="nominated word (S)"), Line2D([], [], color=grey, marker="s", ms=3, markerfacecolor="white", lw=0, label="matched inert word (C)"), Line2D([], [], color=green, marker="^", ms=3, markerfacecolor="white", lw=0, label="hedge removed (R)")], loc="upper left", frameon=False, handlelength=1.2, labelspacing=0.2, borderaxespad=0.1)

    ax = axes[1]
    ax.axhline(0, color=muted, lw=0.6, zorder=1)
    series(ax, "suppress_O_m150", orange, "o", -0.12, True)
    series(ax, "suppress_S_m150", orange, "D", 0.12, False)
    ax.set_title("boundary write, dose −1.5", pad=2)
    ax.set_ylabel("refusal2 removed by the write")
    ax.legend(handles=[Line2D([], [], color=orange, marker="o", ms=3, markerfacecolor=orange, lw=0, label="on O"), Line2D([], [], color=orange, marker="D", ms=3, markerfacecolor="white", lw=0, label="on S")], loc="upper left", frameon=False, handlelength=1.2, labelspacing=0.2, borderaxespad=0.1)

    ax = axes[2]
    ax.axhline(0, color=muted, lw=0.6, zorder=1)
    for shift, key, color, marker, filled in ((-0.12, "proj_S_minus_O_mean", blue, "o", True), (0.12, "proj_C_minus_O_mean", grey, "s", False)):
        values = []
        for tag, _ in order:
            rep = per_model[tag]["behaviour"]["judges"]["refusal2"].get("representation") or {}
            values.append(rep.get(key) if rep.get(key) is not None else np.nan)
        ax.plot([x + shift for x in xs], values, marker=marker, color=color, ms=3.0, lw=0, markerfacecolor=color if filled else "white", markeredgewidth=0.8, zorder=4)
    ax.set_title("boundary projection change", pad=2)
    ax.set_ylabel("proj(·) − proj(O), own direction")
    ax.legend(handles=[Line2D([], [], color=blue, marker="o", ms=3, markerfacecolor=blue, lw=0, label="S"), Line2D([], [], color=grey, marker="s", ms=3, markerfacecolor="white", lw=0, label="C")], loc="lower left", frameon=False, handlelength=1.2, labelspacing=0.2, borderaxespad=0.1)

    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels([label for _, label in order], rotation=55, ha="right")
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.grid(axis="y", zorder=0)
    fig.tight_layout(pad=0.4, w_pad=0.8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    inputs = read_json(inputs_path())
    cohort = load_cohort() if cohort_path().exists() else None
    qwen_nominations = read_json(nominations_path(IMPORTED_TAG, "O")) if nominations_path(IMPORTED_TAG, "O").exists() else None
    per_model: dict[str, dict] = {}
    for tag in args.tags:
        rng = np.random.default_rng(args.seed + ALL_TAGS.index(tag))
        print(f"x106 analyze {tag}: stage 0", flush=True)
        entry = {"model_id": inputs["models"][tag]["model_id"], "hook_block": inputs["models"][tag]["hook_block"], "doses": list(doses_for(tag)), "stage0": stage0_report(tag, cohort, qwen_nominations, rng)}
        if not args.stage0_only and cohort is not None:
            print(f"x106 analyze {tag}: behaviour", flush=True)
            entry["behaviour"] = behaviour_report(tag, cohort, entry["stage0"], args.n_boot, args.seed + 100 * ALL_TAGS.index(tag))
        per_model[tag] = entry
    out = {
        "experiment": "x106",
        "stage": 4,
        "inputs": file_record(inputs_path()),
        "cohort": file_record(cohort_path()) if cohort is not None else None,
        "cohort_sha256": cohort["cohort_sha256"] if cohort is not None else None,
        "n_bootstrap": args.n_boot,
        "bootstrap": "two-stage paired: prompts, then the ten binary responses within each arm; one replicate carries every estimand",
        "label_constructs": {"refusal2": "x64 refusal2 (an explicit refusal statement is present), x101 / x105's judge", "guardreasoner": "GuardReasoner-8B `Completion: refusal` under x104's contract"},
        "strata_edges": list(STRATA_EDGES),
        "models": per_model,
        "cross_model": cross_model(per_model) if not args.stage0_only else None,
    }
    atomic_json(args.out, out)
    if not args.stage0_only:
        draw_figure(per_model, Path(args.figure))
    for tag, entry in per_model.items():
        s0 = entry["stage0"]
        line = f"  {tag:8s} AUC {s0['auc_base_refusal']['auc']:.3f} (unharmful {s0['auc_base_refusal_request_unharmful']['auc']:.3f}); z med {s0['nomination_population']['top_z_median']:.2f}; pop {s0['population_n']}"
        if entry.get("behaviour", {}).get("status") == "complete" and entry["behaviour"]["judges"].get("refusal2", {}).get("status") == "complete":
            e = entry["behaviour"]["judges"]["refusal2"]["estimands"]
            line += f"; S {e['substitute_S']['estimate']:+.3f} C {e['control_C']['estimate']:+.3f} R {e['removed_R']['estimate'] if e['removed_R']['estimate'] is not None else float('nan'):+.3f}"
        print(line)
    print(f"x106 results -> {args.out}")


if __name__ == "__main__":
    main()
