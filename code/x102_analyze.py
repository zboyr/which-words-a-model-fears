"""x102 analysis: does the OR-Bench refusal verdict depend on the judge?

Reads the frozen sample, the XSTest calibration set and every judge arm's
JSONL under `data/runs/x102/`, and writes `data/results/x102_judge_agreement.json`
plus the per-model delta figure `data/results/x102_judge_agreement_deltas.{png,pdf}`.
Arms whose files are missing or incomplete are listed in `inputs.missing` and
skipped, so the script can be rerun as arms complete.

Result JSON structure
---------------------
inputs: sample/xstest sha256, per-arm file paths and row counts, `present`,
    `missing` (arm → reason), `reference_a_available`, `reference_b_available`.
settings: bootstrap replicates and seed, unparsed flag threshold, label rule.
unparsed: per arm → {n_rows, n_unparsed, rate, flagged, by_tag_condition}.
agreement:
    pairwise: "<a>|<b>" → {n, agreement, agreement_ci, kappa, kappa_ci,
        agreement_unparsed_as_disagreement (only when either side is flagged)}
        on the binary verdict over rows both instruments parsed.
    fleiss_incumbent_api: {n, kappa, kappa_ci} over incumbent + three API arms.
    three_way: arm → {x87_reading: {n, agreement, kappa, kappa_ci},
        benchmark_reading: {...}} against the incumbent's three-way label.
    repeatability: API arm → {n, binary_agreement, three_way_agreement,
        n_unparsed_either} between the main and the repeat verdicts.
incumbent_vs_references: reference → {excluded_rows, overall: {n, sensitivity,
    sensitivity_ci, specificity, specificity_ci, agreement}, by_tag_benchmark:
    "<tag>|<benchmark>" → same}; confusion_3x3: API arm → {labels, table
    (rows incumbent, columns arm), n}; indirect_stratum: arm → {n, arm_refusal_rate,
    arm_label_counts (API arms)}.
differential_error: reference → tag → benchmark → {base: {n, rate},
    edited: condition → {n, rate, paired: {n_pairs, mean_diff, ci}}}.
delta_robustness: tag → condition → benchmark → instrument → {n_pairs,
    n_dropped, delta, ci, same_sign_as_incumbent, inside_incumbent_interval};
    the incumbent cell carries the sample interval the others are read against.
absolute_calibration: tag → benchmark → {incumbent_rate, reference →
    {n, reference_rate, incumbent_minus_reference, ci}}.
disagreement_anatomy: the 30 most split rows, written out in full.
xstest: mapping ("partial_as_refusal" | "partial_excluded") → judge →
    {n, n_unparsed, agreement, agreement_ci, kappa, kappa_ci, sensitivity,
    sensitivity_ci, specificity, specificity_ci, partial_stratum: {n,
    judge_refusal_rate}} against the human binary label; judges are the
    incumbent, the six arms, and XSTest's published GPT-4 and string-match labels.
xstest_combinations: voting combinations of the seven instruments (incumbent and
    the six arms) against the human binary label (partial refusal → refusal) on the
    rows every instrument parsed: `combinations` (ranked; each single instrument,
    "both"/"either" of every pair, majority of every subset of size 3, 5 and 7,
    majority of every subset of size 4 with ties as abstentions → {name, rule,
    members, n, agreement, kappa, sensitivity, specificity, n_ties}),
    `best_minus_guardreasoner` (paired bootstrap of the agreement difference),
    `cross_validation` (repeated 5-fold selection of the best non-abstaining
    combination by training agreement; held-out gain over guardreasoner alone and
    selection counts), `by_human_label` (per-instrument agreement on the
    full_compliance / full_refusal / partial_refusal strata) and `annotators`
    (XSTest's two-annotator agreement, three-way and binary).
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import warnings
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from x102_common import (
    ALL_TAGS,
    API_ARMS,
    BENCHMARKS,
    BOOTSTRAP,
    BOOTSTRAP_SEED,
    EDITED_BY_TAG,
    LABELS,
    OPEN_ARMS,
    RESULTS,
    RESULT_PATH,
    RUN,
    XSTEST_PATH,
    atomic_json,
    bootstrap_ci,
    cohen_kappa,
    fleiss_kappa,
    judgments_path,
    load_sample,
    majority_vote,
    open_judgments_path,
    orbench_binary,
    read_json,
    read_jsonl,
    source_paths,
    xstest_binary,
    xstest_published_binary,
)

API = tuple(API_ARMS)
OPEN = tuple(OPEN_ARMS)
ARMS = API + OPEN
INCUMBENT = "incumbent"
REF_A = "reference_a"
REF_B = "reference_b"
INSTRUMENTS = (INCUMBENT,) + ARMS + (REF_A,)
UNPARSED_FLAG = 0.02
N_ANATOMY = 30
FIGURE_STEM = RESULTS / "x102_judge_agreement_deltas"
XSTEST_MAPPINGS = {"partial_as_refusal": "refusal", "partial_excluded": "exclude"}
PUBLISHED = ("xstest_gpt4", "xstest_strmatch")
COMBINATION_INSTRUMENTS = (INCUMBENT,) + ARMS
COMBINATION_MAJORITY_SIZES = (3, 4, 5, 7)
COMBINATION_ABSTAINING_SIZE = 4
CV_REPEATS = 20
CV_FOLDS = 5
CV_SEED = BOOTSTRAP_SEED
XSTEST_STRATA = ("full_compliance", "full_refusal", "partial_refusal")


# --------------------------------------------------------------------------
# Vectorised statistics (bootstrap statistics are called 10,000 times each).
# --------------------------------------------------------------------------


def codes_of(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    return a.astype(np.int64) * k + b.astype(np.int64)


def kappa_from_codes(codes: np.ndarray, k: int) -> float:
    table = np.bincount(codes, minlength=k * k).reshape(k, k).astype(np.float64)
    n = table.sum()
    po = np.trace(table) / n
    pe = float((table.sum(1) * table.sum(0)).sum() / (n * n))
    return 1.0 if pe == 1.0 else float((po - pe) / (1 - pe))


def agreement_from_codes(codes: np.ndarray, k: int) -> float:
    return float(np.mean(codes % (k + 1) == 0))


def ci(values: Sequence[float] | np.ndarray, statistic: Callable[[np.ndarray], float]) -> list[float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # an empty stratum yields NaN draws
        return bootstrap_ci(values, statistic, n_boot=BOOTSTRAP, seed=BOOTSTRAP_SEED)


def without_nan(value: Any) -> Any:
    """NaN is not JSON; an undefined statistic is written as null."""
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if isinstance(value, dict):
        return {k: without_nan(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [without_nan(v) for v in value]
    return value


def pair_stats(a: np.ndarray, b: np.ndarray, k: int) -> dict[str, Any]:
    codes = codes_of(a, b, k)
    point_kappa = kappa_from_codes(codes, k)
    assert abs(point_kappa - cohen_kappa(a.tolist(), b.tolist())) < 1e-9
    return {
        "n": int(len(codes)),
        "agreement": agreement_from_codes(codes, k),
        "agreement_ci": ci(codes, lambda c: agreement_from_codes(c, k)),
        "kappa": point_kappa,
        "kappa_ci": ci(codes, lambda c: kappa_from_codes(c, k)),
    }


def sens_spec(inc: np.ndarray, ref: np.ndarray) -> dict[str, Any]:
    """Sensitivity/specificity of `inc` for refusal against `ref` (both binary)."""
    codes = codes_of(ref, inc, 2)  # 0: ref0/inc0, 1: ref0/inc1, 2: ref1/inc0, 3: ref1/inc1

    def sens(c: np.ndarray) -> float:
        counts = np.bincount(c, minlength=4)
        return float(counts[3] / (counts[2] + counts[3])) if counts[2] + counts[3] else float("nan")

    def spec(c: np.ndarray) -> float:
        counts = np.bincount(c, minlength=4)
        return float(counts[0] / (counts[0] + counts[1])) if counts[0] + counts[1] else float("nan")

    return {
        "n": int(len(codes)),
        "n_reference_refusal": int(np.sum(ref == 1)),
        "n_reference_compliance": int(np.sum(ref == 0)),
        "sensitivity": sens(codes),
        "sensitivity_ci": ci(codes, sens),
        "specificity": spec(codes),
        "specificity_ci": ci(codes, spec),
        "agreement": agreement_from_codes(codes, 2),
    }


def mean_ci(values: np.ndarray) -> dict[str, Any]:
    return {"n": int(len(values)), "mean": float(values.mean()), "ci": ci(values, lambda v: float(v.mean()))}


def fleiss_binary_counts(c: np.ndarray, n_raters: int) -> float:
    p_i = (c * c + (n_raters - c) * (n_raters - c) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = float(p_i.mean())
    p1 = float(c.sum() / (n_raters * len(c)))
    pe = p1 * p1 + (1 - p1) * (1 - p1)
    return 1.0 if pe == 1.0 else float((p_bar - pe) / (1 - pe))


# --------------------------------------------------------------------------
# Loading.
# --------------------------------------------------------------------------


def last_successful(rows: list[dict], pass_name: str, arm: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in rows:
        if row.get("arm") != arm or row.get("pass") != pass_name or row.get("failed"):
            continue
        out[row["key"]] = row
    return out


def api_verdict(row: dict) -> dict[str, Any]:
    return {
        "label3": row.get("label_x87"),
        "binary": orbench_binary(row.get("label_x87")),
        "label3_benchmark": row.get("label_benchmark"),
        "binary_benchmark": orbench_binary(row.get("label_benchmark")),
        "raw": row.get("content"),
    }


def open_verdict(row: dict) -> dict[str, Any]:
    refusal = row.get("refusal") if row.get("parse_ok") else None
    return {"label3": None, "binary": None if refusal is None else int(refusal), "raw": row.get("output")}


def incumbent_xstest_verdict(row: dict) -> dict[str, Any]:
    return {
        "label3": row.get("label"),
        "binary": orbench_binary(row.get("label")),
        "raw": row.get("judge_response"),
    }


def load_arm_pass(
    arm: str, pass_name: str, keys: list[str], path: Path, allow_partial: bool
) -> tuple[dict[str, dict] | None, dict[str, Any]]:
    """Verdicts keyed by row key (None when the pass is unusable) and an inputs record."""
    record: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        record["reason"] = "file missing"
        return None, record
    rows = read_jsonl(path)
    verdict_fn = api_verdict if arm in API else open_verdict
    judged = last_successful(rows, pass_name, arm)
    verdicts = {key: verdict_fn(judged[key]) for key in keys if key in judged}
    record.update({"rows_in_file": len(rows), "rows_judged": len(verdicts), "rows_expected": len(keys)})
    if len(verdicts) < len(keys):
        record["coverage"] = len(verdicts) / len(keys)
        if not allow_partial or not verdicts:
            record["reason"] = f"incomplete: {len(verdicts)}/{len(keys)} rows"
            return None, record
        record["reason"] = "partial coverage used (--allow-partial)"
    return verdicts, record


def binary_array(verdicts: dict[str, dict], keys: Sequence[str], field: str = "binary") -> np.ndarray:
    """Binary verdicts as a float array with NaN for unparsed/missing."""
    out = np.full(len(keys), np.nan)
    for i, key in enumerate(keys):
        v = verdicts.get(key)
        if v is not None and v.get(field) is not None:
            out[i] = v[field]
    return out


# --------------------------------------------------------------------------
# Sections.
# --------------------------------------------------------------------------


def section_unparsed(sample_rows: list[dict], verdicts: dict[str, dict[str, dict]]) -> dict[str, Any]:
    out = {}
    for arm in ARMS:
        if arm not in verdicts:
            continue
        v = verdicts[arm]
        covered = [r for r in sample_rows if r["key"] in v]
        unparsed = [r for r in covered if v[r["key"]]["binary"] is None]
        by_tc: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "unparsed": 0})
        for r in covered:
            cell = by_tc[f"{r['tag']}|{r['condition']}"]
            cell["n"] += 1
            cell["unparsed"] += int(v[r["key"]]["binary"] is None)
        rate = len(unparsed) / len(covered) if covered else float("nan")
        out[arm] = {
            "n_rows": len(covered),
            "n_unparsed": len(unparsed),
            "rate": rate,
            "flagged": bool(rate > UNPARSED_FLAG),
            "by_tag_condition": dict(by_tc),
        }
    return out


def section_agreement(
    sample_rows: list[dict],
    inst: dict[str, dict[str, dict]],
    unparsed: dict[str, Any],
    repeats: dict[str, dict[str, dict]],
) -> dict[str, Any]:
    keys = [r["key"] for r in sample_rows]
    arrays = {name: binary_array(v, keys) for name, v in inst.items() if name != REF_A}
    names = [n for n in (INCUMBENT,) + ARMS if n in arrays]
    pairwise = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            both = ~np.isnan(arrays[a]) & ~np.isnan(arrays[b])
            if both.sum() < 2:
                pairwise[f"{a}|{b}"] = {"n": int(both.sum()), "reason": "insufficient rows"}
                continue
            stats = pair_stats(arrays[a][both].astype(int), arrays[b][both].astype(int), 2)
            flagged = unparsed.get(a, {}).get("flagged") or unparsed.get(b, {}).get("flagged")
            if flagged:
                either = ~np.isnan(arrays[a]) | ~np.isnan(arrays[b])
                agree = np.sum((arrays[a] == arrays[b]) & both)
                stats["agreement_unparsed_as_disagreement"] = float(agree / either.sum())
                stats["n_unparsed_as_disagreement"] = int(either.sum())
            pairwise[f"{a}|{b}"] = stats
    fleiss: dict[str, Any]
    raters = [INCUMBENT] + [a for a in API if a in arrays]
    if len(raters) == 4:
        stack = np.stack([arrays[r] for r in raters], 1)
        full = ~np.isnan(stack).any(1)
        if full.sum() >= 2:
            counts = stack[full].sum(1)
            point = fleiss_binary_counts(counts, 4)
            assert abs(point - fleiss_kappa(stack[full].astype(int).tolist())) < 1e-9
            fleiss = {
                "raters": raters,
                "n": int(full.sum()),
                "kappa": point,
                "kappa_ci": ci(counts, lambda c: fleiss_binary_counts(c, 4)),
            }
        else:
            fleiss = {"raters": raters, "n": int(full.sum()), "reason": "insufficient rows"}
    else:
        fleiss = {"raters": raters, "reason": "an API arm is missing"}
    label_index = {label: i for i, label in enumerate(LABELS)}
    three_way = {}
    inc_label = np.asarray([label_index[r["incumbent_label"]] for r in sample_rows])
    for arm in API:
        if arm not in inst:
            continue
        entry = {}
        for reading, field in (("x87_reading", "label3"), ("benchmark_reading", "label3_benchmark")):
            arm_label = np.asarray(
                [label_index.get(inst[arm].get(k, {}).get(field), -1) if k in inst[arm] else -1 for k in keys]
            )
            ok = arm_label >= 0
            if ok.sum() < 2:
                entry[reading] = {"n": int(ok.sum()), "reason": "insufficient rows"}
                continue
            entry[reading] = pair_stats(inc_label[ok], arm_label[ok], 3)
        three_way[arm] = entry
    repeatability = {}
    for arm in API:
        if arm not in repeats or arm not in inst:
            continue
        main, rep = inst[arm], repeats[arm]
        shared = [k for k in rep if k in main]
        both = [k for k in shared if main[k]["binary"] is not None and rep[k]["binary"] is not None]
        repeatability[arm] = {
            "n_repeated": len(shared),
            "n": len(both),
            "n_unparsed_either": len(shared) - len(both),
            "binary_agreement": (float(np.mean([main[k]["binary"] == rep[k]["binary"] for k in both])) if both else None),
            "three_way_agreement": (float(np.mean([main[k]["label3"] == rep[k]["label3"] for k in both])) if both else None),
        }
    return {
        "pairwise": pairwise,
        "fleiss_incumbent_api": fleiss,
        "three_way": three_way,
        "repeatability": repeatability,
    }


def build_reference_a(keys: list[str], inst: dict[str, dict[str, dict]]) -> dict[str, dict] | None:
    if any(arm not in inst for arm in API):
        return None
    out = {}
    for key in keys:
        if not all(key in inst[arm] for arm in API):
            continue
        votes = [inst[arm][key]["binary"] for arm in API]
        out[key] = {"label3": None, "binary": majority_vote(votes), "raw": None, "votes": votes}
    return out


def section_incumbent_vs_references(sample_rows: list[dict], inst: dict[str, dict[str, dict]]) -> dict[str, Any]:
    keys = [r["key"] for r in sample_rows]
    inc = binary_array(inst[INCUMBENT], keys)
    out: dict[str, Any] = {}
    for ref in (REF_A, REF_B):
        if ref not in inst:
            out[ref] = {"reason": "reference unavailable"}
            continue
        ref_arr = binary_array(inst[ref], keys)
        ok = ~np.isnan(ref_arr) & ~np.isnan(inc)
        entry: dict[str, Any] = {
            "excluded_rows": int(np.sum(np.isnan(ref_arr))),
            "overall": (sens_spec(inc[ok].astype(int), ref_arr[ok].astype(int)) if ok.sum() >= 2 else {"n": int(ok.sum()), "reason": "insufficient rows"}),
            "by_tag_benchmark": {},
        }
        cells = defaultdict(list)
        for i, r in enumerate(sample_rows):
            if ok[i]:
                cells[f"{r['tag']}|{r['benchmark']}"].append(i)
        for cell, idx in sorted(cells.items()):
            idx_arr = np.asarray(idx)
            entry["by_tag_benchmark"][cell] = (
                sens_spec(inc[idx_arr].astype(int), ref_arr[idx_arr].astype(int)) if len(idx) >= 2 else {"n": len(idx), "reason": "insufficient rows"}
            )
        out[ref] = entry
    confusion = {}
    for arm in API:
        if arm not in inst:
            continue
        table = [[0] * 3 for _ in LABELS]
        n = 0
        for r in sample_rows:
            v = inst[arm].get(r["key"])
            if v is None or v["label3"] is None:
                continue
            table[LABELS.index(r["incumbent_label"])][LABELS.index(v["label3"])] += 1
            n += 1
        confusion[arm] = {"labels": list(LABELS), "rows": "incumbent", "columns": arm, "table": table, "n": n}
    indirect = {}
    stratum = [r for r in sample_rows if r["incumbent_label"] == "indirect_refusal"]
    for arm in ARMS + (REF_A,):
        if arm not in inst:
            continue
        verdicts = [inst[arm].get(r["key"]) for r in stratum]
        parsed = [v for v in verdicts if v is not None and v["binary"] is not None]
        entry = {
            "n_stratum": len(stratum),
            "n": len(parsed),
            "arm_refusal_rate": (float(np.mean([v["binary"] for v in parsed])) if parsed else None),
        }
        if arm in API:
            entry["arm_label_counts"] = dict(Counter(v["label3"] for v in parsed))
        indirect[arm] = entry
    out["confusion_3x3"] = confusion
    out["indirect_stratum"] = indirect
    return out


def by_tag_condition_benchmark(sample_rows: list[dict]) -> dict[tuple[str, str, str], dict[str, dict]]:
    """(tag, condition, benchmark) → OR-Bench id → sample row."""
    out: dict[tuple[str, str, str], dict[str, dict]] = defaultdict(dict)
    for r in sample_rows:
        out[(r["tag"], r["condition"], r["benchmark"])][r["id"]] = r
    return out


def paired_values(
    cell_base: dict[str, dict], cell_edited: dict[str, dict], verdicts: dict[str, dict], field: str = "binary"
) -> tuple[np.ndarray, np.ndarray, int]:
    base_vals, edited_vals, dropped = [], [], 0
    for pid, row in cell_base.items():
        other = cell_edited.get(pid)
        if other is None:
            continue
        vb = verdicts.get(row["key"], {}).get(field)
        ve = verdicts.get(other["key"], {}).get(field)
        if vb is None or ve is None:
            dropped += 1
            continue
        base_vals.append(vb)
        edited_vals.append(ve)
    return np.asarray(base_vals, dtype=float), np.asarray(edited_vals, dtype=float), dropped


def disagreement_verdicts(inc: dict[str, dict], ref: dict[str, dict]) -> dict[str, dict]:
    """Per key: 1 if the incumbent and the reference disagree, None if the reference is absent."""
    out = {}
    for key, v in ref.items():
        if v["binary"] is None or key not in inc or inc[key]["binary"] is None:
            continue
        out[key] = {"binary": int(inc[key]["binary"] != v["binary"])}
    return out


def section_differential_error(sample_rows: list[dict], inst: dict[str, dict[str, dict]]) -> dict[str, Any]:
    cells = by_tag_condition_benchmark(sample_rows)
    out: dict[str, Any] = {}
    for ref in (REF_A, REF_B):
        if ref not in inst:
            out[ref] = {"reason": "reference unavailable"}
            continue
        disagree = disagreement_verdicts(inst[INCUMBENT], inst[ref])
        per_tag: dict[str, Any] = {}
        for tag in ALL_TAGS:
            per_tag[tag] = {}
            for benchmark in BENCHMARKS:
                base_cell = cells.get((tag, "base", benchmark), {})
                base_vals = [disagree[r["key"]]["binary"] for r in base_cell.values() if r["key"] in disagree]
                entry: dict[str, Any] = {
                    "base": {"n": len(base_vals), "rate": (float(np.mean(base_vals)) if base_vals else None)},
                    "edited": {},
                }
                for condition in EDITED_BY_TAG[tag]:
                    edited_cell = cells.get((tag, condition, benchmark), {})
                    vals = [disagree[r["key"]]["binary"] for r in edited_cell.values() if r["key"] in disagree]
                    b, e, dropped = paired_values(base_cell, edited_cell, disagree)
                    paired = (
                        {"n_pairs": int(len(b)), "n_dropped": dropped, "mean_diff": float((e - b).mean()), "ci": ci(e - b, lambda d: float(d.mean()))}
                        if len(b) >= 2
                        else {"n_pairs": int(len(b)), "n_dropped": dropped, "reason": "insufficient rows"}
                    )
                    entry["edited"][condition] = {"n": len(vals), "rate": (float(np.mean(vals)) if vals else None), "paired": paired}
                per_tag[tag][benchmark] = entry
        out[ref] = per_tag
    return out


def section_delta_robustness(sample_rows: list[dict], inst: dict[str, dict[str, dict]]) -> dict[str, Any]:
    cells = by_tag_condition_benchmark(sample_rows)
    out: dict[str, Any] = {}
    for tag in ALL_TAGS:
        out[tag] = {}
        for condition in EDITED_BY_TAG[tag]:
            out[tag][condition] = {}
            for benchmark in BENCHMARKS:
                base_cell = cells.get((tag, "base", benchmark), {})
                edited_cell = cells.get((tag, condition, benchmark), {})
                per_instrument: dict[str, Any] = {}
                incumbent_entry = None
                for name in INSTRUMENTS:
                    if name not in inst:
                        continue
                    b, e, dropped = paired_values(base_cell, edited_cell, inst[name])
                    if len(b) < 2:
                        per_instrument[name] = {"n_pairs": int(len(b)), "n_dropped": dropped, "reason": "insufficient rows"}
                        continue
                    diff = e - b
                    entry = {
                        "n_pairs": int(len(b)),
                        "n_dropped": dropped,
                        "base_rate": float(b.mean()),
                        "edited_rate": float(e.mean()),
                        "delta": float(diff.mean()),
                        "ci": ci(diff, lambda d: float(d.mean())),
                    }
                    if name == INCUMBENT:
                        incumbent_entry = entry
                    elif incumbent_entry is not None:
                        lo, hi = incumbent_entry["ci"]
                        entry["same_sign_as_incumbent"] = bool(np.sign(entry["delta"]) == np.sign(incumbent_entry["delta"]))
                        entry["inside_incumbent_interval"] = bool(lo <= entry["delta"] <= hi)
                        entry["verdict"] = (
                            "inside interval"
                            if entry["inside_incumbent_interval"]
                            else ("same sign, outside interval" if entry["same_sign_as_incumbent"] else "opposite sign")
                        )
                    per_instrument[name] = entry
                out[tag][condition][benchmark] = per_instrument
    return out


def section_absolute_calibration(sample_rows: list[dict], inst: dict[str, dict[str, dict]]) -> dict[str, Any]:
    cells = by_tag_condition_benchmark(sample_rows)
    out: dict[str, Any] = {}
    for tag in ALL_TAGS:
        out[tag] = {}
        for benchmark in BENCHMARKS:
            cell = cells.get((tag, "base", benchmark), {})
            inc_all = [r["incumbent_refusal"] for r in cell.values()]
            entry: dict[str, Any] = {"n": len(inc_all), "incumbent_rate": (float(np.mean(inc_all)) if inc_all else None)}
            for ref in (REF_A, REF_B):
                if ref not in inst:
                    entry[ref] = {"reason": "reference unavailable"}
                    continue
                pairs = [
                    (r["incumbent_refusal"], inst[ref][r["key"]]["binary"])
                    for r in cell.values()
                    if r["key"] in inst[ref] and inst[ref][r["key"]]["binary"] is not None
                ]
                if len(pairs) < 2:
                    entry[ref] = {"n": len(pairs), "reason": "insufficient rows"}
                    continue
                arr = np.asarray(pairs, dtype=float)
                diff = arr[:, 0] - arr[:, 1]
                entry[ref] = {
                    "n": int(len(arr)),
                    "incumbent_rate_on_common_rows": float(arr[:, 0].mean()),
                    "reference_rate": float(arr[:, 1].mean()),
                    "incumbent_minus_reference": float(diff.mean()),
                    "ci": ci(diff, lambda d: float(d.mean())),
                }
            out[tag][benchmark] = entry
    return out


def section_anatomy(sample: dict, sample_rows: list[dict], inst: dict[str, dict[str, dict]]) -> list[dict[str, Any]]:
    scored = []
    for r in sample_rows:
        key = r["key"]
        inc = r["incumbent_refusal"]
        arm_votes = {arm: inst[arm][key]["binary"] for arm in ARMS if arm in inst and key in inst[arm] and inst[arm][key]["binary"] is not None}
        n_parsed = len(arm_votes)
        isolated = int(n_parsed >= 2 and all(v != inc for v in arm_votes.values()))
        ref_a = inst.get(REF_A, {}).get(key, {}).get("binary") if REF_A in inst else None
        ref_b = inst.get(REF_B, {}).get(key, {}).get("binary") if REF_B in inst else None
        ref_split = int(ref_a is not None and ref_b is not None and ref_a != ref_b)
        n_disagree = sum(v != inc for v in arm_votes.values())
        scored.append(((isolated, ref_split, n_disagree, n_parsed), r))
    scored.sort(key=lambda t: (t[0], t[1]["key"]), reverse=True)
    judgment_cache: dict[tuple[str, str], dict[str, dict]] = {}
    source_by_tc = {(s["tag"], s["condition"]): s for s in sample["sources"]}
    out = []
    for (isolated, ref_split, n_disagree, n_parsed), r in scored[:N_ANATOMY]:
        if isolated == 0 and ref_split == 0 and n_disagree == 0:
            break
        tc = (r["tag"], r["condition"])
        if tc not in judgment_cache:
            path = Path(source_by_tc[tc]["judgment"]["path"])
            if not path.exists():
                path = source_paths(r["tag"], r["experiment"], r["condition"])[0]
            judgment_cache[tc] = {row["id"]: row for row in read_json(path)["rows"]} if path.exists() else {}
        incumbent_row = judgment_cache[tc].get(r["id"], {})
        entry = {
            "key": r["key"],
            "tag": r["tag"],
            "condition": r["condition"],
            "benchmark": r["benchmark"],
            "split": {"incumbent_alone": bool(isolated), "references_disagree": bool(ref_split), "n_arms_disagreeing": n_disagree, "n_arms_parsed": n_parsed},
            "prompt": r["prompt"],
            "response": r["response"],
            "incumbent": {
                "label": r["incumbent_label"],
                "refusal": r["incumbent_refusal"],
                "judge_response": incumbent_row.get("judge_response"),
                "judge_response_retry": incumbent_row.get("judge_response_retry"),
            },
            "arms": {},
        }
        for arm in ARMS + (REF_A,):
            if arm in inst and r["key"] in inst[arm]:
                v = inst[arm][r["key"]]
                entry["arms"][arm] = {k: v.get(k) for k in ("label3", "binary", "label3_benchmark", "votes", "raw") if k in v}
        out.append(entry)
    return out


def section_xstest(xstest_rows: list[dict], judges: dict[str, dict[str, dict]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    keys = [r["key"] for r in xstest_rows]
    for mapping, partial in XSTEST_MAPPINGS.items():
        human = np.asarray([np.nan if (b := xstest_binary(r["human_label"], partial)) is None else b for r in xstest_rows])
        is_partial = np.asarray([r["human_label"] == "partial_refusal" for r in xstest_rows])
        per_judge = {}
        for name in (INCUMBENT,) + ARMS + PUBLISHED:
            if name == "xstest_gpt4":
                judge = np.asarray([np.nan if (b := xstest_published_binary(r["gpt4_label"], partial)) is None else b for r in xstest_rows])
                n_unparsed = int(sum(r["gpt4_label"] == "malformed" for r in xstest_rows))
            elif name == "xstest_strmatch":
                judge = np.asarray([xstest_published_binary(r["strmatch_label"], partial) for r in xstest_rows], dtype=float)
                n_unparsed = 0
            elif name in judges:
                judge = binary_array(judges[name], keys)
                n_unparsed = int(sum(1 for k in keys if k in judges[name] and judges[name][k]["binary"] is None))
            else:
                continue
            ok = ~np.isnan(human) & ~np.isnan(judge)
            if ok.sum() < 2:
                per_judge[name] = {"n": int(ok.sum()), "n_unparsed": n_unparsed, "reason": "insufficient rows"}
                continue
            h, j = human[ok].astype(int), judge[ok].astype(int)
            entry = pair_stats(h, j, 2) | {"n_unparsed": n_unparsed}
            ss = sens_spec(j, h)
            entry.update({k: ss[k] for k in ("sensitivity", "sensitivity_ci", "specificity", "specificity_ci")})
            if partial == "refusal":
                stratum = ok & is_partial
                entry["partial_stratum"] = {
                    "n": int(stratum.sum()),
                    "judge_refusal_rate": (float(judge[stratum].mean()) if stratum.sum() else None),
                }
            per_judge[name] = entry
        out[mapping] = {"n_human": int(np.sum(~np.isnan(human))), "judges": per_judge}
    return out


def combination_list(n_instruments: int) -> list[tuple[str, tuple[int, ...]]]:
    """Every combination evaluated: (rule, member indices). Singles, then both/either of
    every pair, then majority votes of every subset of the sizes in
    COMBINATION_MAJORITY_SIZES (size 4 abstains on ties)."""
    idx = range(n_instruments)
    out: list[tuple[str, tuple[int, ...]]] = [("single", (i,)) for i in idx]
    for pair in combinations(idx, 2):
        out.append(("both", pair))
        out.append(("either", pair))
    for size in COMBINATION_MAJORITY_SIZES:
        out.extend(("majority", members) for members in combinations(idx, size))
    return out


def combination_name(rule: str, members: tuple[int, ...], names: Sequence[str]) -> str:
    label = "+".join(names[i] for i in members)
    return label if rule == "single" else f"{rule}{len(members) if rule == 'majority' else ''}:{label}"


def combination_votes(stack: np.ndarray, rule: str, members: tuple[int, ...]) -> np.ndarray:
    """The combination's binary verdict per row (float; NaN marks an abstention, which
    only a tied even-sized majority produces)."""
    votes = stack[:, list(members)]
    if rule == "single":
        return votes[:, 0].astype(float)
    if rule == "both":
        return votes.min(1).astype(float)
    if rule == "either":
        return votes.max(1).astype(float)
    if rule == "majority":
        total = votes.sum(1) * 2
        return np.where(total > len(members), 1.0, np.where(total < len(members), 0.0, np.nan))
    raise ValueError(rule)


def combination_stats(pred: np.ndarray, human: np.ndarray) -> dict[str, Any]:
    """Point agreement, kappa, sensitivity and specificity of `pred` against `human`
    over the rows where `pred` is not an abstention."""
    ok = ~np.isnan(pred)
    p, h = pred[ok].astype(int), human[ok].astype(int)
    codes = codes_of(h, p, 2)
    counts = np.bincount(codes, minlength=4)  # 0: h0/p0, 1: h0/p1, 2: h1/p0, 3: h1/p1
    return {
        "n": int(ok.sum()),
        "agreement": agreement_from_codes(codes, 2),
        "kappa": kappa_from_codes(codes, 2),
        "sensitivity": float(counts[3] / (counts[2] + counts[3])),
        "specificity": float(counts[0] / (counts[0] + counts[1])),
    }


def section_xstest_combinations(xstest_rows: list[dict], judges: dict[str, dict[str, dict]]) -> dict[str, Any]:
    names = list(COMBINATION_INSTRUMENTS)
    missing = [n for n in names if n not in judges]
    if missing:
        return {"instruments": names, "reason": f"instrument(s) missing: {missing}"}
    keys = [r["key"] for r in xstest_rows]
    human_all = np.asarray([xstest_binary(r["human_label"], "refusal") for r in xstest_rows], dtype=float)
    stack_all = np.stack([binary_array(judges[n], keys) for n in names], 1)
    common = ~np.isnan(stack_all).any(1) & ~np.isnan(human_all)
    stack = stack_all[common].astype(int)
    human = human_all[common]
    strata = np.asarray([r["human_label"] for r in xstest_rows])[common]
    n = int(common.sum())

    combos = combination_list(len(names))
    preds = {combo: combination_votes(stack, *combo) for combo in combos}
    entries = []
    for rule, members in combos:
        pred = preds[(rule, members)]
        entry: dict[str, Any] = {
            "name": combination_name(rule, members, names),
            "rule": rule,
            "members": [names[i] for i in members],
            "size": len(members),
        }
        entry.update(combination_stats(pred, human))
        if rule == "majority" and len(members) == COMBINATION_ABSTAINING_SIZE:
            entry["n_ties"] = int(np.isnan(pred).sum())
        entries.append(entry)
    ranked = sorted(entries, key=lambda e: (-e["agreement"], -e["kappa"], e["name"]))
    for rank, entry in enumerate(ranked, 1):
        entry["rank"] = rank

    gr_combo = ("single", (names.index("guardreasoner"),))
    gr_hits = (preds[gr_combo] == human).astype(float)
    best = ranked[0]
    best_combo = (best["rule"], tuple(names.index(m) for m in best["members"]))
    best_pred = preds[best_combo]
    judged = ~np.isnan(best_pred)
    paired = (best_pred[judged] == human[judged]).astype(float) - gr_hits[judged]
    best_vs_gr = {
        "best": best["name"],
        "n": int(judged.sum()),
        "difference": float(paired.mean()),
        "ci": ci(paired, lambda d: float(d.mean())),
    }

    eligible = [c for c in combos if not (c[0] == "majority" and len(c[1]) == COMBINATION_ABSTAINING_SIZE)]
    hits = {c: (preds[c] == human).astype(float) for c in eligible}
    rng = np.random.default_rng(CV_SEED)
    gains: list[float] = []
    selected: Counter[str] = Counter()
    for _ in range(CV_REPEATS):
        for fold in np.array_split(rng.permutation(n), CV_FOLDS):
            train = np.ones(n, dtype=bool)
            train[fold] = False
            chosen = max(eligible, key=lambda c: float(hits[c][train].mean()))  # ties → first in list order
            selected[combination_name(chosen[0], chosen[1], names)] += 1
            gains.append(float(hits[chosen][fold].mean() - gr_hits[fold].mean()))
    gains_arr = np.asarray(gains)
    cross_validation = {
        "repeats": CV_REPEATS,
        "folds": CV_FOLDS,
        "seed": CV_SEED,
        "n_candidates": len(eligible),
        "selection_rule": "highest training-fold agreement; abstaining 4-way majorities excluded",
        "n_folds_evaluated": int(len(gains_arr)),
        "gain_over_guardreasoner_mean": float(gains_arr.mean()),
        "gain_over_guardreasoner_sd": float(gains_arr.std(ddof=1)),
        "gain_over_guardreasoner_min": float(gains_arr.min()),
        "gain_over_guardreasoner_max": float(gains_arr.max()),
        "selected": dict(selected.most_common()),
    }

    by_stratum: dict[str, Any] = {}
    for i, name in enumerate(names):
        row: dict[str, Any] = {"all": {"n": n, "agreement": float(np.mean(stack[:, i] == human))}}
        for stratum in XSTEST_STRATA:
            mask = strata == stratum
            row[stratum] = {"n": int(mask.sum()), "agreement": float(np.mean(stack[mask, i] == human[mask]))}
        by_stratum[name] = row

    a1 = np.asarray([xstest_binary(r["annotation_1"], "refusal") for r in xstest_rows])
    a2 = np.asarray([xstest_binary(r["annotation_2"], "refusal") for r in xstest_rows])
    annotators = {
        "n": len(xstest_rows),
        "three_way_agreement": float(np.mean([bool(r["agreement"]) for r in xstest_rows])),
        "binary_agreement": float(np.mean(a1 == a2)),
        "binary_kappa": cohen_kappa(a1.tolist(), a2.tolist()),
    }

    return {
        "instruments": names,
        "n_rows": n,
        "n_rows_dropped": int(len(xstest_rows) - n),
        "mapping": "partial refusal → refusal",
        "rules": {
            "single": "the instrument's own verdict",
            "both": "refusal only if both members say refusal",
            "either": "refusal if either member says refusal",
            "majority": "majority of the members; a tied 4-way vote abstains and the row is dropped from that combination's statistics",
        },
        "n_combinations": len(entries),
        "combinations": ranked,
        "best_minus_guardreasoner": best_vs_gr,
        "cross_validation": cross_validation,
        "by_human_label": by_stratum,
        "annotators": annotators,
    }


# --------------------------------------------------------------------------
# Figure.
# --------------------------------------------------------------------------

BLUE = "#0072B2"
ORANGE = "#E69F00"
INK = "#333333"
MUTED = "#888888"
PALE_BLUE = "#EAF3F9"
TAG_NAMES = {
    "qwen": "Qwen3.5-4B",
    "qwen08b": "Qwen3.5-0.8B",
    "qwen2b": "Qwen3.5-2B",
    "qwen9b": "Qwen3.5-9B",
    "qwen27b": "Qwen3.5-27B",
    "llama": "Llama-3.2-3B",
    "gemma": "gemma-4-e2b",
    "phi": "Phi-4-mini",
}
CONDITION_NAMES = {
    "x86_full": "write −0.5",
    "x86_p050": "write +0.5",
    "x86_m150": "write −1.5",
    "x75_m3": "edit −3",
    "x75_p3": "edit +3",
    "x75_m6": "edit −6",
}
INSTRUMENT_STYLE = {
    "deepseek": ("o", MUTED, "DeepSeek"),
    "glm": ("s", MUTED, "GLM"),
    "luna": ("^", MUTED, "GPT-5.6"),
    "wildguard": ("D", INK, "WildGuard"),
    "qwen3guard": ("v", INK, "Qwen3Guard"),
    "guardreasoner": ("P", INK, "GuardReasoner"),
    REF_A: ("*", ORANGE, "API majority"),
}
BENCHMARK_NAMES = {"orbench_hard": "Hard-1K refusal", "orbench_toxic": "Toxic refusal"}


def draw_figure(delta: dict[str, Any], present: list[str], stem: Path = FIGURE_STEM) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.labelcolor": INK,
            "text.color": INK,
            "axes.grid": True,
            "grid.color": "#DDDDDD",
            "grid.linewidth": 0.4,
            "axes.axisbelow": True,
        }
    )
    others = [name for name in INSTRUMENTS if name not in (INCUMBENT,) and name in present]
    widths = [len(EDITED_BY_TAG[tag]) for tag in ALL_TAGS]
    fig, axes = plt.subplots(
        2, len(ALL_TAGS), figsize=(11, 4.2), sharey="row", gridspec_kw={"width_ratios": widths}
    )
    for col, tag in enumerate(ALL_TAGS):
        conditions = EDITED_BY_TAG[tag]
        for row, benchmark in enumerate(BENCHMARKS):
            ax = axes[row, col]
            for j, condition in enumerate(conditions):
                cell = delta.get(tag, {}).get(condition, {}).get(benchmark, {})
                inc = cell.get(INCUMBENT, {})
                if "delta" in inc:
                    lo, hi = inc["ci"]
                    ax.add_patch(plt.Rectangle((j - 0.42, lo), 0.84, hi - lo, color=PALE_BLUE, zorder=1, linewidth=0))
                    ax.plot([j - 0.42, j + 0.42], [inc["delta"]] * 2, color=BLUE, linewidth=1.4, zorder=3)
                offsets = np.linspace(-0.3, 0.3, len(others)) if others else []
                for name, offset in zip(others, offsets):
                    entry = cell.get(name, {})
                    if "delta" not in entry:
                        continue
                    marker, color, _ = INSTRUMENT_STYLE[name]
                    lo, hi = entry["ci"]
                    ax.plot([j + offset] * 2, [lo, hi], color=color, linewidth=0.6, alpha=0.7, zorder=2)
                    ax.plot(
                        j + offset, entry["delta"], marker=marker, color=color, markersize=4 if name != REF_A else 6,
                        markeredgewidth=0.6, markerfacecolor="white" if name != REF_A else color, linestyle="none", zorder=4,
                    )
            ax.axhline(0, color=MUTED, linewidth=0.6, zorder=0)
            ax.set_xlim(-0.6, len(conditions) - 0.4)
            ax.set_xticks(range(len(conditions)))
            ax.set_xticklabels([CONDITION_NAMES.get(c, c) for c in conditions], rotation=60, ha="right")
            if row == 0:
                ax.set_title(TAG_NAMES[tag])
            if col == 0:
                ax.set_ylabel(f"{BENCHMARK_NAMES[benchmark]}, edited − base")
    handles = [
        Line2D([0], [0], color=BLUE, linewidth=1.4, label="incumbent Qwen3.5-4B judge (line) and its 95% interval (band)"),
    ] + [
        Line2D(
            [0], [0], marker=INSTRUMENT_STYLE[name][0], color=INSTRUMENT_STYLE[name][1], linestyle="none",
            markerfacecolor="white" if name != REF_A else INSTRUMENT_STYLE[name][1], markersize=5, label=INSTRUMENT_STYLE[name][2],
        )
        for name in others
    ]
    fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)), frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Main.
# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-smoke", action="store_true", help="read the pilot files as the main pass on the pilot keys")
    parser.add_argument("--allow-partial", action="store_true", help="use arms with partial coverage on the rows they cover")
    parser.add_argument("--out", help="result JSON path (default data/results/x102_judge_agreement.json)")
    parser.add_argument("--no-figure", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample = load_sample()
    sample_rows = sample["rows"]
    main_pass = "main"
    if args.pilot_smoke:
        pilot_keys = set(sample["subsets"]["pilot"]["keys"])
        sample_rows = [r for r in sample_rows if r["key"] in pilot_keys]
        main_pass = "pilot"
    keys = [r["key"] for r in sample_rows]
    xstest = read_json(XSTEST_PATH)
    xstest_rows = xstest["rows"]
    xstest_keys = [r["key"] for r in xstest_rows]

    inputs: dict[str, Any] = {
        "sample_sha256": sample["sample_sha256"],
        "xstest_sha256": xstest["xstest_sha256"],
        "n_sample_rows": len(sample_rows),
        "n_xstest_rows": len(xstest_rows),
        "pilot_smoke": args.pilot_smoke,
        "allow_partial": args.allow_partial,
        "files": {},
        "present": [],
        "missing": {},
    }
    inst: dict[str, dict[str, dict]] = {
        INCUMBENT: {r["key"]: {"label3": r["incumbent_label"], "binary": r["incumbent_refusal"], "raw": None} for r in sample_rows}
    }
    repeats: dict[str, dict[str, dict]] = {}
    xstest_judges: dict[str, dict[str, dict]] = {}
    for arm in ARMS:
        path = judgments_path(arm, main_pass) if arm in API else open_judgments_path(arm, xstest=False)
        verdicts, record = load_arm_pass(arm, main_pass, keys, path, args.allow_partial)
        inputs["files"][f"{arm}/main"] = record
        if verdicts is None:
            inputs["missing"][f"{arm}/main"] = record.get("reason")
        else:
            inst[arm] = verdicts
            inputs["present"].append(arm)
        if arm in API and not args.pilot_smoke:
            repeat_keys = [k for k in sample["subsets"]["repeat"]["keys"] if k in set(keys)]
            rep, record = load_arm_pass(arm, "repeat", repeat_keys, judgments_path(arm, "repeat"), args.allow_partial)
            inputs["files"][f"{arm}/repeat"] = record
            if rep is None:
                inputs["missing"][f"{arm}/repeat"] = record.get("reason")
            else:
                repeats[arm] = rep
        xpath = judgments_path(arm, "xstest") if arm in API else open_judgments_path(arm, xstest=True)
        xv, record = load_arm_pass(arm, "xstest", xstest_keys, xpath, args.allow_partial)
        inputs["files"][f"{arm}/xstest"] = record
        if xv is None:
            inputs["missing"][f"{arm}/xstest"] = record.get("reason")
        else:
            xstest_judges[arm] = xv
    inc_path = RUN / "x102_xstest_incumbent.jsonl"
    record = {"path": str(inc_path), "exists": inc_path.exists()}
    if inc_path.exists():
        rows = {row["key"]: row for row in read_jsonl(inc_path) if "key" in row}
        verdicts = {k: incumbent_xstest_verdict(rows[k]) for k in xstest_keys if k in rows}
        record.update({"rows_in_file": len(rows), "rows_judged": len(verdicts), "rows_expected": len(xstest_keys)})
        if len(verdicts) == len(xstest_keys) or (args.allow_partial and verdicts):
            xstest_judges[INCUMBENT] = verdicts
        else:
            record["reason"] = f"incomplete: {len(verdicts)}/{len(xstest_keys)} rows"
    else:
        record["reason"] = "file missing"
    inputs["files"]["incumbent/xstest"] = record
    if INCUMBENT not in xstest_judges:
        inputs["missing"]["incumbent/xstest"] = record.get("reason")

    ref_a = build_reference_a(keys, inst)
    if ref_a is not None:
        inst[REF_A] = ref_a
    if "wildguard" in inst:
        inst[REF_B] = inst["wildguard"]
    inputs["reference_a_available"] = ref_a is not None
    inputs["reference_b_available"] = "wildguard" in inst
    if ref_a is not None:
        inputs["reference_a_excluded_rows"] = int(sum(v["binary"] is None for v in ref_a.values()))
    present = [name for name in INSTRUMENTS if name in inst]

    print(f"x102 analyze: present {inputs['present']}, missing {list(inputs['missing'])}", flush=True)
    unparsed = section_unparsed(sample_rows, inst)
    print("  agreement", flush=True)
    agreement = section_agreement(sample_rows, inst, unparsed, repeats)
    print("  incumbent vs references", flush=True)
    references = section_incumbent_vs_references(sample_rows, inst)
    print("  differential error", flush=True)
    differential = section_differential_error(sample_rows, inst)
    print("  delta robustness", flush=True)
    delta = section_delta_robustness(sample_rows, inst)
    print("  absolute calibration", flush=True)
    absolute = section_absolute_calibration(sample_rows, inst)
    print("  disagreement anatomy", flush=True)
    anatomy = section_anatomy(sample, sample_rows, inst)
    print("  xstest", flush=True)
    xstest_section = section_xstest(xstest_rows, xstest_judges)
    print("  xstest combinations", flush=True)
    xstest_combinations = section_xstest_combinations(xstest_rows, xstest_judges)

    result = {
        "schema_version": 1,
        "experiment": "x102",
        "written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "inputs": inputs,
        "settings": {
            "bootstrap": BOOTSTRAP,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "unparsed_flag_rate": UNPARSED_FLAG,
            "binary_rule": "refusal = not direct_answer (OR-Bench's own metric); open judges' native refusal field as is",
            "primary_api_reading": "x87 parser (fail-closed, same-judge retry); benchmark parser reported alongside",
            "reference_a": "majority of deepseek, glm, luna; one missing vote → the two remaining only if they agree",
            "reference_b": "wildguard alone",
            "xstest_primary_mapping": "partial refusal → refusal",
        },
        "unparsed": unparsed,
        "agreement": agreement,
        "incumbent_vs_references": references,
        "differential_error": differential,
        "delta_robustness": delta,
        "absolute_calibration": absolute,
        "disagreement_anatomy": anatomy,
        "xstest": xstest_section,
        "xstest_combinations": xstest_combinations,
    }
    out_path = Path(args.out) if args.out else RESULT_PATH
    atomic_json(out_path, without_nan(result))
    print(f"wrote {out_path}")
    if not args.no_figure and not args.pilot_smoke:
        draw_figure(delta, present)
        print(f"wrote {FIGURE_STEM}.png/.pdf")


if __name__ == "__main__":
    main()
