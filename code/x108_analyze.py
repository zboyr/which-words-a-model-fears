"""x108 Stage 4: on the harmful prompts the models refuse, what the boundary direction nominates,
how far x106's substitution rule reaches, and what the substitution does where it applies, on
Qwen3.5-4B and the seven transfer models that read its texts.

Stage 0 part, per model: the Toxic and XSTest-contrast projections on the model's own direction
against x106's Hard-1K refused and answered distributions on the same model; the nomination
summaries (within-prompt z, share of excess, pair rate, x83-lexicon membership, the most
nominated units) next to x106's Hard-1K population and XSTest safe summaries on the same model,
computed by one function from both nomination files; top-1 agreement of each transfer model's
nomination with Qwen3.5-4B's.

Reach part (shared by every model, the texts are one cohort): the share of units with an S
text, split by why not (the drafter found no meaning-preserving harm-free substitute, the drafts
failed the checks, the provider refused), for the Toxic and contrast units and for x106's
Hard-1K and XSTest units.

Behaviour part (dose 0 only), per model: per-prompt rates per arm under three label constructs —
`refusal2` (x64), GuardReasoner-8B `Completion: refusal`, GuardReasoner-8B `Response: harmful` —
the contrasts r(O) - r(S), r(O) - r(R) and h(S) - h(O), h(R) - h(O) under one two-stage paired
bootstrap (prompts, then the ten binary responses), Holm within the contrasts present, Toxic and
XSTest contrast separately; the representation change proj(S) - proj(O) on the model's own
direction in sd units of its x106 Hard-1K projection, the share of S texts below its x106
answered-prompt mean, the Spearman between projection change and refusal change;
nomination-predicts-effect; reach under `refusal2`. Across models, Holm over the seven transfer
models within each contrast and judge, Qwen3.5-4B the reference row outside the family.

Benign next to harmful, per model: one table with x106's Hard-1K and XSTest safe rows of the
same model (its behavioural estimates taken from the x106 result file) above the x108 Toxic and
XSTest contrast rows. One figure over the eight models.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from x106_analyze import auc_record, integrity, interval, normalize_top, paired_mean, rates_from, spearman_record, summarize, top_k_indices
from x106_common import N_BOOTSTRAP, holm, is_content_word, lexicon_words, normalize_word
from x106_common import load_cohort as load_x106_cohort
from x108_common import (
    ALL_TAGS,
    ARMS,
    BOOTSTRAP_SEED,
    FRESH_TAGS,
    IMPORTED_TAG,
    N_RESPONSES,
    TAG,
    TEXT_CONDITIONS,
    arm_spec,
    atomic_json,
    cohort_path,
    file_record,
    generations_path,
    guardreasoner_path,
    hard1k_reference,
    is_toxic,
    load_cohort,
    no_substitute_reason,
    nominations_path,
    population_ids,
    projections_path,
    read_json,
    refusal_path,
    results_path,
    strata_of,
    units_with_text,
    x104_toxic_labels,
    x106_cohort,
    x106_inputs,
    x106_nominations,
    x106_results,
)

JUDGES = ("refusal2", "guardreasoner_refusal", "guardreasoner_response_harmful")
REFUSAL_JUDGES = ("refusal2", "guardreasoner_refusal")
CONTRASTS = ("substitute_S", "removed_R")
CROSS_MODEL_KEYS = ("substitute_S", "removed_R", "contrast_substitute_S")
ROWS = ("x106_hard1k", "x106_xstest_safe", "x108_toxic", "x108_xstest_contrast")
ROW_LABELS = {"x106_hard1k": "Hard-1K (x106)", "x106_xstest_safe": "XSTest safe (x106)", "x108_toxic": "Toxic (x108)", "x108_xstest_contrast": "XSTest contrast (x108)"}
REASONS = ("no_substitute", "failed_checks", "drafter_refused")
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
CONSTRUCTS = {
    "refusal2": "x64 refusal2 (an explicit refusal statement is present), x101 / x106's judge",
    "guardreasoner_refusal": "GuardReasoner-8B `Completion: refusal` under x104's contract",
    "guardreasoner_response_harmful": "GuardReasoner-8B `Response: harmful` under x104's contract",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="*", default=list(ALL_TAGS))
    parser.add_argument("--stage0-only", action="store_true")
    parser.add_argument("--n-boot", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--out", default=str(results_path()))
    parser.add_argument("--figure", default=str(results_path().with_suffix(".pdf")))
    parser.add_argument("--redraw", action="store_true", help="only redraw the figure from the result file")
    return parser.parse_args()


# --- Stage 0 ------------------------------------------------------------------------------


def nomination_summary(rows: list[dict], lexicon: set[str]) -> dict:
    z = np.asarray([r["top_z"] for r in rows], dtype=float)
    share = np.asarray([r["top_share_of_excess"] for r in rows if r["top_share_of_excess"] is not None], dtype=float)
    content = [normalize_word(w) for r in rows for k, w in enumerate(r["words"]) if is_content_word(w)]
    counter = Counter(r["unit_norm"] for r in rows)
    return {
        "n": len(rows),
        "top_z_median": float(np.nanmedian(z)) if len(z) else None,
        "top_z_quartiles": [float(v) for v in np.nanquantile(z, [0.25, 0.75])] if len(z) else None,
        "frac_top_z_ge_2": float(np.nanmean(z >= 2)) if len(z) else None,
        "top_share_of_excess_median": float(np.median(share)) if len(share) else None,
        "top_share_of_excess_quartiles": [float(v) for v in np.quantile(share, [0.25, 0.75])] if len(share) else None,
        "frac_pair_wins": float(np.mean([r["pair_wins"] for r in rows])) if rows else None,
        "frac_top_in_lexicon": float(np.mean([r["top_in_lexicon"] for r in rows])) if rows else None,
        "frac_unit_in_lexicon": float(np.mean([r["unit_in_lexicon"] for r in rows])) if rows else None,
        "content_word_lexicon_base_rate": float(np.mean([w in lexicon for w in content])) if content else None,
        "mean_top_delta": float(np.mean([r["top_delta"] for r in rows])) if rows else None,
        "most_nominated_units": [{"unit": u, "n": c, "in_lexicon": all(w in lexicon for w in u.split())} for u, c in counter.most_common(25)],
    }


def projection_summary(values: np.ndarray, reference: dict) -> dict:
    sd = reference["sd"]
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "sd": float(values.std()),
        "quantiles_10_50_90": [float(v) for v in np.quantile(values, [0.1, 0.5, 0.9])],
        "mean_minus_hard1k_refused_mean_in_sd": float((values.mean() - reference["refused_mean"]) / sd),
        "mean_minus_hard1k_answered_mean_in_sd": float((values.mean() - reference["answered_mean"]) / sd),
        "share_above_hard1k_refused_mean": float(np.mean(values > reference["refused_mean"])),
        "share_below_hard1k_answered_mean": float(np.mean(values < reference["answered_mean"])),
    }


def agreement(rows: Mapping[str, dict], qwen_rows: Mapping[str, dict], ids: list[str]) -> dict:
    shared = [i for i in ids if i in qwen_rows and i in rows]
    return {
        "n": len(shared),
        "top_word": float(np.mean([rows[i]["top_norm"] == qwen_rows[i]["top_norm"] for i in shared])) if shared else None,
        "unit": float(np.mean([rows[i]["unit_norm"] == qwen_rows[i]["unit_norm"] for i in shared])) if shared else None,
        "unit_overlaps": float(np.mean([bool(set(rows[i]["unit_norm"].split()) & set(qwen_rows[i]["unit_norm"].split())) for i in shared])) if shared else None,
        "qwen_unit_in_this_models_top3": float(np.mean([qwen_rows[i]["top_norm"] in {normalize_top(rows[i], k) for k in top_k_indices(rows[i], 3)} for i in shared])) if shared else None,
    }


def stage0_report(tag: str, cohort: dict | None, qwen_nominations: dict | None, rng: np.random.Generator) -> dict:
    proj_obj = read_json(projections_path(tag))
    nom_obj = read_json(nominations_path(tag, "O"))
    if proj_obj.get("experiment") != "x108" or nom_obj.get("experiment") != "x108" or proj_obj.get("model_tag") != tag or nom_obj.get("model_tag") != tag or nom_obj.get("cohort_only"):
        raise SystemExit(f"FATAL: {tag}: x108 Stage 0 artifacts are missing or restricted")
    reference = hard1k_reference(tag)
    labels = x104_toxic_labels(tag)
    population = set(population_ids(labels))
    proj = {r["id"]: float(r["proj"]) for r in proj_obj["rows"]}
    toxic_all = np.asarray([proj[i] for i in proj if is_toxic(i)])
    toxic_pop = np.asarray([proj[i] for i in proj if i in population])
    contrast = np.asarray([proj[i] for i in proj if not is_toxic(i)])
    refused = np.asarray(reference["refused"])
    answered = np.asarray(reference["answered"])
    lexicon = lexicon_words()
    rows = {r["id"]: r for r in nom_obj["rows"]}
    x106_rows = {r["id"]: r for r in read_json(x106_nominations(tag, "O"))["rows"]}
    x106_units = load_x106_cohort()["units"]
    x106_hard_ids = [u["id"] for u in x106_units if u["source"] == "orbench_hard"]
    x106_xstest_ids = [u["id"] for u in x106_units if u["source"] == "xstest"]
    x106_proj = {r["id"]: float(r["proj"]) for r in read_json(reference["projections"]["path"])["rows"]}
    cohort_toxic_ids = [u["id"] for u in cohort["units"] if u["source"] == "orbench_toxic"] if cohort else sorted(population)
    out: dict[str, Any] = {
        "n_prompts": len(proj),
        "population": proj_obj["population"],
        "x106_hard1k_reference": {k: reference[k] for k in ("answered_mean", "refused_mean", "sd", "n_hard1k", "n_refused", "n_answered")},
        "projection": {
            "x108_toxic_all": projection_summary(toxic_all, reference),
            "x108_toxic_population": projection_summary(toxic_pop, reference),
            "x108_toxic_cohort": projection_summary(np.asarray([proj[i] for i in cohort_toxic_ids]), reference),
            "x108_xstest_contrast": projection_summary(contrast, reference),
            "x106_hard1k_refused": projection_summary(refused, reference),
            "x106_hard1k_answered": projection_summary(answered, reference),
            "x106_hard1k_population": projection_summary(np.asarray([x106_proj[i] for i in x106_hard_ids]), reference),
            "x106_xstest_safe": projection_summary(np.asarray([x106_proj[i] for i in x106_xstest_ids]), reference),
            "auc_toxic_population_vs_hard1k_answered": auc_record(np.concatenate([toxic_pop, answered]), np.concatenate([np.ones(len(toxic_pop)), np.zeros(len(answered))]), rng),
            "auc_toxic_population_vs_hard1k_refused": auc_record(np.concatenate([toxic_pop, refused]), np.concatenate([np.ones(len(toxic_pop)), np.zeros(len(refused))]), rng),
            "auc_contrast_vs_x106_xstest_safe": auc_record(np.concatenate([contrast, [x106_proj[i] for i in x106_xstest_ids]]), np.concatenate([np.ones(len(contrast)), np.zeros(len(x106_xstest_ids))]), rng),
        },
        "population_strata": {s: int(n) for s, n in zip(*np.unique(strata_of(toxic_pop), return_counts=True))},
        "x104_toxic_base_labels": dict(Counter(f"refusal={int(r['refusal'])}, request={r['fields'].get('request')}, response={r['fields'].get('response')}" for r in labels.values())),
        "nomination": {
            "x108_toxic_population": nomination_summary([rows[i] for i in sorted(population)], lexicon),
            "x108_toxic_cohort": nomination_summary([rows[i] for i in cohort_toxic_ids], lexicon),
            "x108_toxic_all": nomination_summary([rows[i] for i in rows if is_toxic(i)], lexicon),
            "x108_xstest_contrast": nomination_summary([rows[i] for i in rows if not is_toxic(i)], lexicon),
            "x106_hard1k_population": nomination_summary([x106_rows[i] for i in x106_hard_ids], lexicon),
            "x106_xstest_safe": nomination_summary([x106_rows[i] for i in x106_xstest_ids], lexicon),
        },
        "nomination_by_toxic_category": {},
        "artifacts": {"projections": file_record(projections_path(tag)), "nominations": file_record(nominations_path(tag, "O")), "batched_vs_single_audits": nom_obj.get("batched_vs_single_audits")},
    }
    categories = {r["id"]: r["category"] for r in proj_obj["rows"]}
    for category in sorted({categories[i] for i in population}):
        ids = [i for i in sorted(population) if categories[i] == category]
        summary = nomination_summary([rows[i] for i in ids], lexicon)
        out["nomination_by_toxic_category"][category] = {k: summary[k] for k in ("n", "top_z_median", "frac_unit_in_lexicon", "frac_pair_wins")} | {"most_nominated_units": summary["most_nominated_units"][:5]}
    if qwen_nominations is not None and tag != IMPORTED_TAG:
        q = {r["id"]: r for r in qwen_nominations["rows"]}
        out["agreement_with_qwen_nomination"] = {
            "toxic_cohort": agreement(rows, q, cohort_toxic_ids),
            "toxic_own_population": agreement(rows, q, sorted(population)),
            "xstest_contrast": agreement(rows, q, [i for i in rows if not is_toxic(i)]),
        }
    return out


# --- reach ---------------------------------------------------------------------------------


def reach_summary(units: list[dict]) -> dict:
    n = len(units)
    without = [u for u in units if u["prompts"].get("S") is None]
    reasons = Counter(no_substitute_reason(u) for u in without)
    substituted = [u for u in units if u["prompts"].get("S") is not None]
    return {
        "n": n,
        "n_with_S": len(substituted),
        "share_with_S": float(len(substituted) / n) if n else None,
        "n_without_S": len(without),
        "without_S_by_reason": {reason: int(reasons.get(reason, 0)) for reason in REASONS},
        "share_without_S_by_reason": {reason: float(reasons.get(reason, 0) / n) if n else None for reason in REASONS},
        "n_hedge": sum(1 for u in units if u.get("unit_kind") == "hedge"),
        "n_with_R": sum(1 for u in units if u["prompts"].get("R") is not None),
        "share_unit_in_lexicon": float(np.mean([bool(u["unit_in_lexicon"]) for u in units])) if n else None,
        "share_unit_in_lexicon_among_without_S": float(np.mean([bool(u["unit_in_lexicon"]) for u in without])) if without else None,
        "share_unit_in_lexicon_among_with_S": float(np.mean([bool(u["unit_in_lexicon"]) for u in substituted])) if substituted else None,
        "n_draft_attempts_mean": float(np.mean([u.get("n_draft_attempts", 0) for u in units])) if n else None,
        "most_frequent_units_without_S": [{"unit": u, "n": c} for u, c in Counter(normalize_word(x["unit"]) if " " not in x["unit"] else " ".join(normalize_word(w) for w in x["unit"].split()) for x in without).most_common(20)],
        "most_frequent_substitutions": [{"unit": u, "substitute": s, "n": c} for (u, s), c in Counter((x["substitute"]["unit"].lower(), x["substitute"]["substitute"].lower()) for x in substituted if x.get("substitute")).most_common(20)],
        "no_substitute_notes_sample": [{"id": u["id"], "unit": u["unit"], "note": u["no_substitute_note"]} for u in without if u.get("no_substitute_note")][:15],
    }


def reach_report(cohort: dict) -> dict:
    x106 = load_x106_cohort()
    return {
        "x108_toxic": reach_summary([u for u in cohort["units"] if u["source"] == "orbench_toxic"]),
        "x108_xstest_contrast": reach_summary([u for u in cohort["units"] if u["source"] == "xstest"]),
        "x106_hard1k": reach_summary([u for u in x106["units"] if u["source"] == "orbench_hard"]),
        "x106_xstest_safe": reach_summary([u for u in x106["units"] if u["source"] == "xstest"]),
        "x106_cohort": file_record(x106_cohort()),
    }


# --- behaviour -----------------------------------------------------------------------------


def load_generation_arm(tag: str, arm: str, cohort: dict) -> tuple[dict[str, dict], dict]:
    spec = arm_spec(arm)
    path = generations_path(tag, arm)
    obj = read_json(path)
    if obj.get("experiment") != "x108" or obj.get("arm") != arm or obj.get("model_tag") != tag or obj.get("cohort_sha256") != cohort["cohort_sha256"] or obj.get("text_condition") != spec["text"] or float(obj.get("signed_dose", 1.0)) != 0.0:
        raise SystemExit(f"FATAL: {tag} {arm}: generation lineage is not valid")
    units = {u["id"]: u for u in units_with_text(cohort, spec["text"])}
    rows = {}
    for row in obj["rows"]:
        prompt_id = str(row["id"])
        if prompt_id not in units:
            continue
        if row.get("condition") != arm or row.get("prompt_sha256") != units[prompt_id]["prompt_sha256"][spec["text"]] or row.get("arms_sha256") != cohort["cohort_sha256"] or len(row.get("llm_responses", [])) != N_RESPONSES:
            raise SystemExit(f"FATAL: invalid {tag} {arm} generation row {prompt_id}")
        if prompt_id in rows:
            raise SystemExit(f"FATAL: duplicate {tag} {arm} generation row {prompt_id}")
        rows[prompt_id] = row
    if len(rows) != len(units):
        raise SystemExit(f"FATAL: {tag} {arm}: {len(rows)} generated rows for {len(units)} units")
    return rows, {"generations": file_record(path), "text_condition": spec["text"], "n_rows": len(rows), "n_units_with_text": len(units)}


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
                raise SystemExit(f"FATAL: invalid {tag} {arm} refusal record {prompt_id}")
            if prompt_id in labels:
                raise SystemExit(f"FATAL: duplicate successful {tag} {arm} refusal record {prompt_id}")
            labels[prompt_id] = np.asarray(values, dtype=float)
    return labels


def guardreasoner_key(tag: str, arm: str, prompt_id: str, k: int) -> str:
    """The record key of x108_judge_guardreasoner.py: Qwen3.5-4B's rows carry no tag."""
    return f"x108|{arm}|{prompt_id}|{k}" if tag == TAG else f"x108|{tag}|{arm}|{prompt_id}|{k}"


def load_guardreasoner(tag: str, arm: str, rows: dict[str, dict]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    """Per prompt the ten `Completion: refusal` labels and the ten `Response: harmful` labels."""
    pattern = str(guardreasoner_path(tag, arm, 0, 1)).replace("shard0of1", "shard*of*")
    paths = sorted(glob.glob(pattern))
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
    refusal: dict[str, np.ndarray] = {}
    harmful: dict[str, np.ndarray] = {}
    n_parsed = n_response_field = 0
    for prompt_id, row in rows.items():
        r_values = np.full(N_RESPONSES, np.nan)
        h_values = np.full(N_RESPONSES, np.nan)
        for k in range(N_RESPONSES):
            record = best.get(guardreasoner_key(tag, arm, prompt_id, k))
            if record is None:
                continue
            if record.get("response_sha256") != row["response_sha256"][k]:
                raise SystemExit(f"FATAL: {tag} {arm}: GuardReasoner record {prompt_id}/{k} is not this response")
            if record["parse_ok"]:
                n_parsed += 1
                r_values[k] = float(record["refusal"])
                response = (record.get("fields") or {}).get("response")
                if response in ("harmful", "unharmful"):
                    n_response_field += 1
                    h_values[k] = float(response == "harmful")
        if np.isfinite(r_values).any():
            refusal[prompt_id] = r_values
        if np.isfinite(h_values).any():
            harmful[prompt_id] = h_values
    return refusal, harmful, {"shards": [file_record(p) for p in paths], "n_parsed": n_parsed, "n_with_response_field": n_response_field, "n_responses": N_RESPONSES * len(rows)}


def estimands(rates: Mapping[str, np.ndarray], toxic: np.ndarray, kind: str) -> dict[str, float]:
    """Refusal constructs: refusal removed, r(O) - r(S) and r(O) - r(R). Harm construct: harm
    added, h(S) - h(O) and h(R) - h(O). Toxic and XSTest contrast separately."""
    o, s, r = (rates[f"{t}_base"] for t in TEXT_CONDITIONS)
    sign = 1.0 if kind == "refusal" else -1.0
    out = {}
    for name, mask in (("", toxic), ("contrast_", ~toxic)):
        out[f"{name}substitute_S"] = sign * paired_mean(o[mask], s[mask])
        out[f"{name}removed_R"] = sign * paired_mean(o[mask], r[mask])
        out[f"{name}mean_O"] = float(np.nanmean(o[mask])) if np.isfinite(o[mask]).any() else float("nan")
        out[f"{name}mean_S"] = float(np.nanmean(s[mask])) if np.isfinite(s[mask]).any() else float("nan")
        out[f"{name}mean_R"] = float(np.nanmean(r[mask])) if np.isfinite(r[mask]).any() else float("nan")
        out[f"{name}mean_O_where_S"] = float(np.nanmean(o[mask & np.isfinite(s)])) if (mask & np.isfinite(s)).any() else float("nan")
    return out


def bootstrap(labels: Mapping[str, dict[str, np.ndarray]], ids: list[str], toxic: np.ndarray, kind: str, n_boot: int, seed: int, chunk: int = 250) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    rates = {arm: rates_from(values, ids) for arm, values in labels.items()}
    point = estimands(rates, toxic, kind)
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
        toxic_drawn = toxic[index]
        for offset in range(stop - start):
            replicate = estimands({arm: drawn[arm][offset] for arm in arms}, toxic_drawn[offset], kind)
            for name in names:
                samples[name][start + offset] = replicate[name]
    return point, samples


def text_projections(tag: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for text in TEXT_CONDITIONS:
        path = nominations_path(tag, text)
        if not path.exists():
            continue
        obj = read_json(path)
        if obj.get("experiment") != "x108" or obj.get("text") != text or obj.get("model_tag") != tag:
            raise SystemExit(f"FATAL: {path} belongs to another experiment/model/text")
        out[text] = {r["id"]: float(r["proj_full"]) for r in obj["rows"]}
    return out


def representation(tag: str, ids: list[str], toxic: np.ndarray, effect_s: np.ndarray, effect_r: np.ndarray, reference: dict, rng: np.random.Generator, n_boot: int) -> dict | None:
    proj = text_projections(tag)
    if "S" not in proj or "O" not in proj:
        return None
    sd = reference["sd"]
    answered_mean = reference["answered_mean"]
    p_o = np.asarray([proj["O"].get(i, np.nan) for i in ids])
    p_s = np.asarray([proj["S"].get(i, np.nan) for i in ids])
    p_r = np.asarray([proj.get("R", {}).get(i, np.nan) for i in ids])
    out = {"answered_mean_proj": answered_mean, "hard1k_sd": sd, "unit": f"sd of x106's {tag} Hard-1K projection on its own direction"}
    for name, mask in (("toxic", toxic), ("contrast", ~toxic)):
        d_s = p_s[mask] - p_o[mask]
        d_r = p_r[mask] - p_o[mask]
        finite = np.isfinite(d_s)
        boot = np.asarray([np.mean(d_s[finite][rng.integers(0, finite.sum(), finite.sum())]) for _ in range(min(n_boot, 2000))]) if finite.any() else np.asarray([np.nan])
        out[name] = {
            "n_with_S": int(finite.sum()),
            "proj_O_mean": float(np.nanmean(p_o[mask])),
            "proj_S_mean": float(np.nanmean(p_s[mask])) if finite.any() else None,
            "proj_S_minus_O_mean": float(np.nanmean(d_s)) if finite.any() else None,
            "proj_S_minus_O_mean_in_sd": float(np.nanmean(d_s) / sd) if finite.any() else None,
            "proj_S_minus_O_ci95_in_sd": [v / sd for v in interval(boot)],
            "proj_R_minus_O_mean_in_sd": float(np.nanmean(d_r) / sd) if np.isfinite(d_r).any() else None,
            "share_S_below_answered_mean": float(np.nanmean(p_s[mask][finite] < answered_mean)) if finite.any() else None,
            "share_O_below_answered_mean": float(np.mean(p_o[mask] < answered_mean)),
            "share_S_below_hard1k_refused_mean": float(np.nanmean(p_s[mask][finite] < reference["refused_mean"])) if finite.any() else None,
            "spearman_proj_change_vs_refusal_change_S": spearman_record(d_s, -effect_s[mask], rng, n_boot),
            "spearman_proj_change_vs_refusal_change_R": spearman_record(d_r, -effect_r[mask], rng, n_boot) if np.isfinite(d_r).sum() >= 3 else None,
        }
    return out


def behaviour_report(tag: str, cohort: dict, n_boot: int, seed: int) -> dict:
    available = [arm for arm in ARMS if generations_path(tag, arm).exists()]
    if len(available) != len(ARMS):
        return {"status": "incomplete", "generated_arms": available, "missing_arms": [a for a in ARMS if a not in available]}
    units = list(cohort["units"])
    ids = [u["id"] for u in units]
    toxic = np.asarray([is_toxic(i) for i in ids])
    rows_by_arm: dict[str, dict[str, dict]] = {}
    arm_meta = {}
    labels: dict[str, dict[str, dict[str, np.ndarray]]] = {j: {} for j in JUDGES}
    for arm in ARMS:
        rows, meta = load_generation_arm(tag, arm, cohort)
        rows_by_arm[arm] = rows
        labels["refusal2"][arm] = load_refusal2(tag, arm, rows, cohort["cohort_sha256"])
        gr_refusal, gr_harm, gr_meta = load_guardreasoner(tag, arm, rows)
        labels["guardreasoner_refusal"][arm] = gr_refusal
        labels["guardreasoner_response_harmful"][arm] = gr_harm
        meta["n_refusal2"] = len(labels["refusal2"][arm])
        meta["guardreasoner"] = gr_meta
        arm_meta[arm] = meta
    reference = hard1k_reference(tag)
    delta_top = np.asarray([u["top_delta"] for u in units])
    strata = np.asarray([u["stratum"] for u in units])
    report: dict[str, Any] = {"status": "complete", "arms": arm_meta, "integrity": integrity(rows_by_arm), "n_units": len(ids), "n_toxic": int(toxic.sum()), "n_contrast": int((~toxic).sum()), "judges": {}}
    for j, judge in enumerate(JUDGES):
        if any(len(labels[judge][arm]) == 0 for arm in ARMS):
            report["judges"][judge] = {"status": "missing labels", "arms_without_labels": [arm for arm in ARMS if len(labels[judge][arm]) == 0]}
            continue
        kind = "refusal" if judge in REFUSAL_JUDGES else "harm"
        rng = np.random.default_rng(seed + j)
        point, samples = bootstrap(labels[judge], ids, toxic, kind, n_boot, seed + j)
        summary = summarize(point, samples)
        for prefix in ("", "contrast_"):
            family = [f"{prefix}{c}" for c in CONTRASTS if summary[f"{prefix}{c}"]["estimate"] is not None]
            for name, p in zip(family, holm([summary[name]["one_sided_p"] for name in family])):
                summary[name]["holm_p_within_contrast_family"] = float(p)
        rates = {arm: rates_from(values, ids) for arm, values in labels[judge].items()}
        o, s, r = rates["O_base"], rates["S_base"], rates["R_base"]
        effect_s = (o - s) if kind == "refusal" else (s - o)
        effect_r = (o - r) if kind == "refusal" else (r - o)
        entry: dict[str, Any] = {
            "status": "complete",
            "construct": {"refusal2": "x64 refusal2: an explicit refusal statement is present", "guardreasoner_refusal": "GuardReasoner-8B `Completion: refusal` under x104's contract", "guardreasoner_response_harmful": "GuardReasoner-8B `Response: harmful` under x104's contract"}[judge],
            "direction": "refusal removed, r(O) - r(text)" if kind == "refusal" else "harm added, h(text) - h(O)",
            "n_labelled_per_arm": {arm: int(np.isfinite(rates[arm]).sum()) for arm in rates},
            "estimands": summary,
            "contrast_family": list(CONTRASTS),
            "per_stratum": {
                stratum: {"n": int((toxic & (strata == stratum)).sum()), "substitute_S": float(np.nanmean(effect_s[toxic & (strata == stratum)])) if np.isfinite(effect_s[toxic & (strata == stratum)]).any() else None, "mean_O": float(np.nanmean(o[toxic & (strata == stratum)])) if (toxic & (strata == stratum)).any() else None}
                for stratum in ("low", "mid", "high")
            },
            "nomination_predicts_effect": {"spearman_qwen_top_delta_vs_substitution_effect": spearman_record(delta_top[toxic], effect_s[toxic], rng, n_boot)},
        }
        if kind == "refusal":
            entry["representation"] = representation(tag, ids, toxic, effect_s, effect_r, reference, rng, n_boot)
            with_s = toxic & np.isfinite(s)
            entry["reach"] = {
                "n_population": int(toxic.sum()),
                "share_without_substitute": float(np.mean(~np.isfinite(s[toxic]))),
                "share_S_refusal_below_half": float(np.sum(with_s & (s < 0.5)) / toxic.sum()),
                "share_S_refusal_below_half_among_with_S": float(np.mean(s[with_s] < 0.5)) if with_s.any() else None,
                "share_S_does_not_lower_refusal_among_with_S": float(np.mean(effect_s[with_s] <= 0)) if with_s.any() else None,
                "share_O_refusal_below_half": float(np.mean(o[toxic] < 0.5)),
            }
        else:
            with_s = toxic & np.isfinite(s)
            entry["harm_profile"] = {
                "share_S_any_harmful_response_among_with_S": float(np.mean(s[with_s] > 0)) if with_s.any() else None,
                "share_O_any_harmful_response": float(np.mean(o[toxic] > 0)),
                "share_S_majority_harmful_among_with_S": float(np.mean(s[with_s] >= 0.5)) if with_s.any() else None,
            }
        report["judges"][judge] = entry
    # the joint reading per prompt: refusal removed and harm added on the same S texts
    if all(report["judges"].get(j, {}).get("status") == "complete" for j in ("refusal2", "guardreasoner_response_harmful")):
        r2 = {arm: rates_from(v, ids) for arm, v in labels["refusal2"].items()}
        hm = {arm: rates_from(v, ids) for arm, v in labels["guardreasoner_response_harmful"].items()}
        d_ref = r2["O_base"] - r2["S_base"]
        d_harm = hm["S_base"] - hm["O_base"]
        mask = toxic & np.isfinite(d_ref) & np.isfinite(d_harm)
        rng = np.random.default_rng(seed + 10)
        report["refusal_removed_vs_harm_added"] = {
            "n": int(mask.sum()),
            "spearman": spearman_record(d_ref[mask], d_harm[mask], rng, n_boot),
            "share_refusal_falls_and_harm_rises": float(np.mean((d_ref[mask] > 0) & (d_harm[mask] > 0))),
            "share_refusal_falls_and_harm_flat": float(np.mean((d_ref[mask] > 0) & (d_harm[mask] <= 0))),
            "share_refusal_flat": float(np.mean(d_ref[mask] <= 0)),
        }
    return report


# --- benign next to harmful ---------------------------------------------------------------


def comparison_table(tag: str, stage0: dict, reach: dict, behaviour: dict | None) -> dict:
    x106 = read_json(x106_results())
    model = x106["models"][tag]
    x106_r2 = model["behaviour"]["judges"]["refusal2"]
    x106_gr = model["behaviour"]["judges"]["guardreasoner"]
    sd = stage0["x106_hard1k_reference"]["sd"]
    x106_nom = read_json(x106_nominations(tag, "O"))
    x106_s = read_json(x106_nominations(tag, "S")) if x106_nominations(tag, "S").exists() else None
    x106_proj_o = {r["id"]: float(r["proj_full"]) for r in x106_nom["rows"]}
    x106_proj_s = {r["id"]: float(r["proj_full"]) for r in x106_s["rows"]} if x106_s else {}
    x106_units = load_x106_cohort()["units"]

    def x106_proj_change(source: str) -> float | None:
        d = [x106_proj_s[u["id"]] - x106_proj_o[u["id"]] for u in x106_units if u["source"] == source and u["id"] in x106_proj_s]
        return float(np.mean(d) / sd) if d else None

    def pick(entry: dict | None) -> dict | None:
        return {"estimate": entry["estimate"], "ci95": entry["ci95"]} if entry else None

    rows = {
        "x106_hard1k": {
            "label": ROW_LABELS["x106_hard1k"],
            "population": "Hard-1K prompts Qwen3.5-4B refused at base, request unharmful (x106)",
            "n": reach["x106_hard1k"]["n"],
            "share_with_S": reach["x106_hard1k"]["share_with_S"],
            "share_without_S_by_reason": reach["x106_hard1k"]["share_without_S_by_reason"],
            "r_O_refusal2": x106_r2["reach"]["mean_r_O"],
            "r_S_refusal2": x106_r2["reach"]["mean_r_S"],
            "refusal2_substitute_S": pick(x106_r2["estimands"]["substitute_S"]),
            "guardreasoner_refusal_substitute_S": pick(x106_gr["estimands"]["substitute_S"]),
            "refusal2_removed_R": pick(x106_r2["estimands"]["removed_R"]),
            "proj_S_minus_O_in_sd": x106_proj_change("orbench_hard"),
            "proj_O_mean": stage0["projection"]["x106_hard1k_population"]["mean"],
            "share_unit_in_lexicon": stage0["nomination"]["x106_hard1k_population"]["frac_unit_in_lexicon"],
            "top_z_median": stage0["nomination"]["x106_hard1k_population"]["top_z_median"],
            "frac_pair_wins": stage0["nomination"]["x106_hard1k_population"]["frac_pair_wins"],
            "source": "x106 result file and artifacts",
        },
        "x106_xstest_safe": {
            "label": ROW_LABELS["x106_xstest_safe"],
            "population": "100 safe XSTest prompts of the four lexical categories (x106)",
            "n": reach["x106_xstest_safe"]["n"],
            "share_with_S": reach["x106_xstest_safe"]["share_with_S"],
            "share_without_S_by_reason": reach["x106_xstest_safe"]["share_without_S_by_reason"],
            "r_O_refusal2": None,
            "r_S_refusal2": None,
            "refusal2_substitute_S": pick(x106_r2["estimands"]["xstest_substitute_S"]),
            "guardreasoner_refusal_substitute_S": pick(x106_gr["estimands"]["xstest_substitute_S"]),
            "refusal2_removed_R": pick(x106_r2["estimands"].get("xstest_removed_R")),
            "proj_S_minus_O_in_sd": x106_proj_change("xstest"),
            "proj_O_mean": stage0["projection"]["x106_xstest_safe"]["mean"],
            "share_unit_in_lexicon": stage0["nomination"]["x106_xstest_safe"]["frac_unit_in_lexicon"],
            "top_z_median": stage0["nomination"]["x106_xstest_safe"]["top_z_median"],
            "frac_pair_wins": stage0["nomination"]["x106_xstest_safe"]["frac_pair_wins"],
            "source": "x106 result file and artifacts",
        },
    }
    for key, prefix, reach_key, proj_key, nom_key in (("x108_toxic", "", "x108_toxic", "x108_toxic_cohort", "x108_toxic_cohort"), ("x108_xstest_contrast", "contrast_", "x108_xstest_contrast", "x108_xstest_contrast", "x108_xstest_contrast")):
        r2 = (behaviour or {}).get("judges", {}).get("refusal2", {})
        gr = (behaviour or {}).get("judges", {}).get("guardreasoner_refusal", {})
        hm = (behaviour or {}).get("judges", {}).get("guardreasoner_response_harmful", {})
        rep = (r2.get("representation") or {}).get("toxic" if prefix == "" else "contrast") or {}
        rows[key] = {
            "label": ROW_LABELS[key],
            "population": "Toxic prompts Qwen3.5-4B refused at base, request harmful (the x108 cohort)" if prefix == "" else "100 XSTest contrast prompts of the four lexical categories",
            "n": reach[reach_key]["n"],
            "share_with_S": reach[reach_key]["share_with_S"],
            "share_without_S_by_reason": reach[reach_key]["share_without_S_by_reason"],
            "r_O_refusal2": r2.get("estimands", {}).get(f"{prefix}mean_O", {}).get("estimate"),
            "r_S_refusal2": r2.get("estimands", {}).get(f"{prefix}mean_S", {}).get("estimate"),
            "refusal2_substitute_S": pick(r2.get("estimands", {}).get(f"{prefix}substitute_S")),
            "guardreasoner_refusal_substitute_S": pick(gr.get("estimands", {}).get(f"{prefix}substitute_S")),
            "guardreasoner_response_harmful_substitute_S": pick(hm.get("estimands", {}).get(f"{prefix}substitute_S")),
            "h_O_guardreasoner": hm.get("estimands", {}).get(f"{prefix}mean_O", {}).get("estimate"),
            "h_S_guardreasoner": hm.get("estimands", {}).get(f"{prefix}mean_S", {}).get("estimate"),
            "refusal2_removed_R": pick(r2.get("estimands", {}).get(f"{prefix}removed_R")),
            "proj_S_minus_O_in_sd": rep.get("proj_S_minus_O_mean_in_sd"),
            "proj_O_mean": stage0["projection"][proj_key]["mean"],
            "share_unit_in_lexicon": stage0["nomination"][nom_key]["frac_unit_in_lexicon"],
            "top_z_median": stage0["nomination"][nom_key]["top_z_median"],
            "frac_pair_wins": stage0["nomination"][nom_key]["frac_pair_wins"],
            "source": "this experiment",
        }
    return {"rows": rows, "order": list(ROWS), "x106_results": file_record(x106_results())}


# --- across models -------------------------------------------------------------------------


def cross_model(per_model: dict) -> dict:
    """Holm over the seven transfer models within each contrast and judge; Qwen3.5-4B, whose
    nomination the texts come from, is the reference row outside the family."""
    out: dict[str, Any] = {}
    for judge in JUDGES:
        rows = {tag: per_model[tag]["behaviour"]["judges"][judge] for tag in per_model if per_model[tag].get("behaviour", {}).get("status") == "complete" and per_model[tag]["behaviour"]["judges"].get(judge, {}).get("status") == "complete"}
        if not rows:
            continue
        table = {}
        transfer = [tag for tag in FRESH_TAGS if tag in rows]
        for key in CROSS_MODEL_KEYS:
            entries = {tag: rows[tag]["estimands"].get(key) for tag in rows}
            in_family = [tag for tag in transfer if entries.get(tag) and entries[tag].get("estimate") is not None]
            adjusted = dict(zip(in_family, holm([entries[tag]["one_sided_p"] for tag in in_family]))) if in_family else {}
            table[key] = {tag: {**(entries[tag] or {}), "holm_p_over_transfer_models": adjusted.get(tag), "in_family": tag in adjusted} for tag in rows}
        out[judge] = {"table": table, "transfer_models_in_family": transfer, "reference_model": IMPORTED_TAG}
    out["representation"] = {tag: {name: ((per_model[tag].get("behaviour", {}).get("judges", {}).get("refusal2", {}).get("representation") or {}).get(name) or {}).get("proj_S_minus_O_mean_in_sd") for name in ("toxic", "contrast")} for tag in per_model}
    out["nomination_agreement_with_qwen"] = {tag: per_model[tag]["stage0"].get("agreement_with_qwen_nomination") for tag in per_model if tag != IMPORTED_TAG}
    return out


# --- figure --------------------------------------------------------------------------------


def draw_figure(per_model: dict, reach: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 6.5, "ytick.labelsize": 7, "legend.fontsize": 6.0, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "axes.edgecolor": "#333333", "text.color": "#333333", "axes.labelcolor": "#333333", "grid.color": "#DDDDDD", "grid.linewidth": 0.5})
    blue, orange, grey, red, muted = "#0072B2", "#E69F00", "#666666", "#D55E00", "#888888"
    order = [(tag, label) for tag, label in FIGURE_ORDER if per_model.get(tag, {}).get("behaviour", {}).get("judges", {}).get("refusal2", {}).get("status") == "complete"]
    if not order:
        return
    xs = list(range(len(order)))
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.7), gridspec_kw={"width_ratios": [1.6, 1.1, 0.9]})

    def entry_of(tag: str, judge: str, key: str) -> dict | None:
        judge_entry = per_model[tag]["behaviour"]["judges"].get(judge, {})
        if judge_entry.get("status") != "complete":
            return None
        entry = judge_entry["estimands"].get(key)
        return entry if entry and entry["estimate"] is not None else None

    # (a) the substitution's effect per model under the three constructs, Toxic; the XSTest contrast under refusal2
    ax = axes[0]
    ax.axhline(0, color=muted, lw=0.6, zorder=1)
    series = (
        ("refusal2", "substitute_S", blue, "o", -0.27, True, "refusal2 removed, r(O) − r(S), Toxic"),
        ("guardreasoner_refusal", "substitute_S", blue, "D", -0.09, False, "GuardReasoner refusal removed, Toxic"),
        ("guardreasoner_response_harmful", "substitute_S", red, "^", 0.09, True, "GuardReasoner harm added, h(S) − h(O), Toxic"),
        ("refusal2", "contrast_substitute_S", grey, "o", 0.27, False, "refusal2 removed, XSTest contrast"),
    )
    for judge, key, color, marker, shift, filled, _ in series:
        for x, (tag, _) in zip(xs, order):
            entry = entry_of(tag, judge, key)
            if entry is None:
                continue
            ax.errorbar([x + shift], [entry["estimate"]], yerr=[[entry["estimate"] - entry["ci95"][0]], [entry["ci95"][1] - entry["estimate"]]], fmt=marker, color=color, ms=3.0, elinewidth=0.7, capsize=1.2, markerfacecolor=color if filled else "white", markeredgewidth=0.8, zorder=4)
    ax.set_title("in-place substitution, dose 0", pad=2)
    ax.set_ylabel("rate change, S against O")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.6 * (hi - lo))
    ax.legend(handles=[Line2D([], [], color=c, marker=m, ms=3, markerfacecolor=c if f else "white", lw=0, label=l) for _, _, c, m, _, f, l in series], loc="upper left", frameon=False, handlelength=1.2, labelspacing=0.15, borderaxespad=0.1)
    ax.set_xticks(xs)
    ax.set_xticklabels([label for _, label in order], rotation=55, ha="right")
    ax.set_xlim(-0.6, len(order) - 0.4)

    # (b) the projection change of S on each model's own direction
    ax = axes[1]
    ax.axhline(0, color=muted, lw=0.6, zorder=1)
    for shift, name, color, marker, filled, label in ((-0.15, "toxic", blue, "o", True, "Toxic (x108)"), (0.15, "contrast", grey, "o", False, "XSTest contrast (x108)")):
        values, lower, upper = [], [], []
        for tag, _ in order:
            rep = (per_model[tag]["behaviour"]["judges"]["refusal2"].get("representation") or {}).get(name) or {}
            v = rep.get("proj_S_minus_O_mean_in_sd")
            ci = rep.get("proj_S_minus_O_ci95_in_sd") or [np.nan, np.nan]
            values.append(v if v is not None else np.nan)
            lower.append((v - ci[0]) if v is not None and np.isfinite(ci[0]) else 0)
            upper.append((ci[1] - v) if v is not None and np.isfinite(ci[1]) else 0)
        ax.errorbar([x + shift for x in xs], values, yerr=[lower, upper], fmt=marker, color=color, ms=3.0, elinewidth=0.7, capsize=1.2, lw=0, markerfacecolor=color if filled else "white", markeredgewidth=0.8, zorder=4, label=label)
    x106_values = []
    for tag, _ in order:
        row = (per_model[tag].get("benign_next_to_harmful") or {}).get("rows", {}).get("x106_hard1k") or {}
        x106_values.append(row.get("proj_S_minus_O_in_sd") if row.get("proj_S_minus_O_in_sd") is not None else np.nan)
    ax.plot(xs, x106_values, marker="s", color=orange, ms=3.0, lw=0, markerfacecolor="white", markeredgewidth=0.8, zorder=4, label="Hard-1K (x106)")
    ax.set_title("projection change of S, own direction", pad=2)
    ax.set_ylabel("proj(S) − proj(O), Hard-1K sd")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.35 * (hi - lo), hi)
    ax.legend(loc="lower left", frameon=False, handlelength=1.2, labelspacing=0.2, borderaxespad=0.1)
    ax.set_xticks(xs)
    ax.set_xticklabels([label for _, label in order], rotation=55, ha="right")
    ax.set_xlim(-0.6, len(order) - 0.4)

    # (c) reach of the rule, once: the texts are one cohort
    ax = axes[2]
    rx = list(range(len(ROWS)))
    bottoms = np.zeros(len(ROWS))
    layers = (("share_with_S", blue, "substitute found (S)"), ("no_substitute", grey, "no harm-free substitute"), ("failed_checks", orange, "drafts failed the checks"), ("drafter_refused", red, "provider refused"))
    for layer, color, label in layers:
        values = np.asarray([reach[key]["share_with_S"] if layer == "share_with_S" else reach[key]["share_without_S_by_reason"][layer] for key in ROWS], dtype=float)
        ax.bar(rx, values, bottom=bottoms, color=color, width=0.6, edgecolor="white", linewidth=0.4, zorder=3)
        bottoms += values
    ax.set_ylim(0, 1.42)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title("reach of the rule", pad=2)
    ax.set_ylabel("share of nominated units")
    ax.legend(handles=[Patch(color=c, label=l) for _, c, l in layers], loc="upper left", frameon=False, handlelength=1.0, labelspacing=0.2, borderaxespad=0.1)
    short = {"x106_hard1k": "Hard-1K\n(x106)", "x106_xstest_safe": "XSTest\nsafe\n(x106)", "x108_toxic": "Toxic\n(x108)", "x108_xstest_contrast": "XSTest\ncontrast\n(x108)"}
    ax.set_xticks(rx)
    ax.set_xticklabels([short[k] for k in ROWS], rotation=0, fontsize=5.5)
    ax.set_xlim(-0.6, len(ROWS) - 0.4)

    for ax in axes:
        ax.grid(axis="y", zorder=0)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.30, top=0.91, wspace=0.42)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.redraw:
        previous = read_json(args.out)
        draw_figure(previous["models"], previous["reach"], Path(args.figure))
        print(f"x108 figure -> {args.figure}")
        return
    inputs = read_json(x106_inputs())
    cohort = load_cohort() if cohort_path().exists() else None
    reach = reach_report(cohort) if cohort is not None else None
    qwen_nominations = read_json(nominations_path(IMPORTED_TAG, "O")) if nominations_path(IMPORTED_TAG, "O").exists() else None
    per_model: dict[str, dict] = {}
    for tag in args.tags:
        if not (projections_path(tag).exists() and nominations_path(tag, "O").exists()):
            print(f"x108 analyze {tag}: no Stage 0 artifacts yet, skipped", flush=True)
            continue
        rng = np.random.default_rng(args.seed + ALL_TAGS.index(tag))
        print(f"x108 analyze {tag}: stage 0", flush=True)
        entry: dict[str, Any] = {"model_id": inputs["models"][tag]["model_id"], "hook_block": inputs["models"][tag]["hook_block"], "stage0": stage0_report(tag, cohort, qwen_nominations, rng)}
        behaviour = None
        if cohort is not None and not args.stage0_only:
            print(f"x108 analyze {tag}: behaviour", flush=True)
            behaviour = behaviour_report(tag, cohort, args.n_boot, args.seed + 100 * (ALL_TAGS.index(tag) + 1))
            entry["behaviour"] = behaviour
        if reach is not None:
            entry["benign_next_to_harmful"] = comparison_table(tag, entry["stage0"], reach, behaviour if behaviour and behaviour.get("status") == "complete" else None)
        per_model[tag] = entry
    out = {
        "experiment": "x108",
        "stage": 4,
        "texts_from": IMPORTED_TAG,
        "inputs": file_record(x106_inputs()),
        "cohort": file_record(cohort_path()) if cohort is not None else None,
        "cohort_sha256": cohort["cohort_sha256"] if cohort is not None else None,
        "n_bootstrap": args.n_boot,
        "bootstrap": "two-stage paired: prompts, then the ten binary responses within each arm; one replicate carries every estimand of one label construct; seed per model",
        "label_constructs": CONSTRUCTS,
        "reach": reach,
        "models": per_model,
        "cross_model": cross_model(per_model) if not args.stage0_only else None,
    }
    atomic_json(args.out, out)
    if not args.stage0_only and reach is not None:
        draw_figure(per_model, reach, Path(args.figure))
    if reach:
        print(f"  reach: toxic S {reach['x108_toxic']['share_with_S']:.2f} {reach['x108_toxic']['without_S_by_reason']}; contrast S {reach['x108_xstest_contrast']['share_with_S']:.2f}; hard1k S {reach['x106_hard1k']['share_with_S']:.2f}")
    for tag, entry in per_model.items():
        s0 = entry["stage0"]
        line = f"  {tag:8s} pop {s0['population']['n_population']}; proj toxic pop {s0['projection']['x108_toxic_population']['mean']:.1f} vs hard1k refused {s0['projection']['x106_hard1k_refused']['mean']:.1f} / answered {s0['projection']['x106_hard1k_answered']['mean']:.1f}; z med {s0['nomination']['x108_toxic_population']['top_z_median']:.2f}"
        agree = (s0.get("agreement_with_qwen_nomination") or {}).get("toxic_cohort")
        if agree:
            line += f"; top-word agreement with 4B {agree['top_word']:.2f}"
        behaviour = entry.get("behaviour") or {}
        if behaviour.get("status") == "complete":
            for judge in JUDGES:
                e = behaviour["judges"].get(judge, {}).get("estimands")
                if e:
                    line += f"\n           {judge}: S {e['substitute_S']['estimate']:+.3f} {[round(v, 3) for v in e['substitute_S']['ci95']]} R {e['removed_R']['estimate'] if e['removed_R']['estimate'] is not None else float('nan'):+.3f}; contrast S {e['contrast_substitute_S']['estimate']:+.3f}"
            rep = (behaviour["judges"].get("refusal2", {}).get("representation") or {}).get("toxic") or {}
            if rep.get("proj_S_minus_O_mean_in_sd") is not None:
                line += f"\n           proj(S) − proj(O) {rep['proj_S_minus_O_mean_in_sd']:+.3f} sd, Spearman with refusal change {rep['spearman_proj_change_vs_refusal_change_S']['rho']:+.3f}"
        else:
            line += f"; behaviour {behaviour.get('status', 'not run')}"
        print(line)
    print(f"x108 results -> {args.out}")


if __name__ == "__main__":
    main()
