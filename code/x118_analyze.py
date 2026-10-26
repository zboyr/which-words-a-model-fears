"""x118 Stage 3: what the independent judge accepted, and the refusal effects on those pairs.

Reads `data/runs/x118/pairs.jsonl` and `judge/meaning.jsonl` (+ `meaning_retest.jsonl`),
then recomputes the x106, x108 and x113 `refusal2` substitution contrasts on the pairs the
judge accepts, next to the all-pairs numbers.

    python code/x118_analyze.py [--n-boot 10000]

A second judge's file is analysed the same way into its own results files, with an
agreement section against the first judge and the effects on the pairs both accept:

    python code/x118_analyze.py --judge-file data/runs/x118/judge/meaning_haiku.jsonl \
        --retest-file data/runs/x118/judge/meaning_haiku_retest.jsonl \
        --compare-with data/runs/x118/judge/meaning.jsonl --out-stem results_haiku
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ROOT = HERE.parent
DATA = ROOT / "data"
RUN = DATA / "runs" / "x118"
PAIRS = RUN / "pairs.jsonl"
MEANING = RUN / "judge" / "meaning.jsonl"
MEANING_RETEST = RUN / "judge" / "meaning_retest.jsonl"
COST_LOG = RUN / "judge" / "cost_log.jsonl"

X106_RUN = DATA / "runs" / "x106"
X108_RUN = DATA / "runs" / "x108"
X113_RUN = DATA / "runs" / "x113"

TAGS = ("qwen", "qwen08b", "qwen2b", "qwen9b", "qwen27b", "llama", "gemma", "phi")
LABELS = {
    "qwen": "Qwen3.5-4B",
    "qwen08b": "Qwen3.5-0.8B",
    "qwen2b": "Qwen3.5-2B",
    "qwen9b": "Qwen3.5-9B",
    "qwen27b": "Qwen3.5-27B",
    "llama": "Llama-3.2-3B",
    "gemma": "Gemma-4-E2B",
    "phi": "Phi-4-mini",
}
N_RESPONSES = 10
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 11_800
RELATIONS = ("same", "narrower", "broader", "different")


# --- small statistics ----------------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float | None]:
    if n == 0:
        return [None, None]
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


def share_record(k: int, n: int) -> dict:
    return {"n": n, "accepted": k, "share": (k / n if n else None), "wilson95": wilson(k, n)}


def interval(samples: np.ndarray) -> list[float | None]:
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return [None, None]
    return [float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))]


def cohens_kappa(a: list[bool], b: list[bool]) -> float | None:
    if not a:
        return None
    n = len(a)
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    pa = sum(a) / n
    pb = sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected >= 1.0:
        return None
    return (observed - expected) / (1 - expected)


# --- inputs --------------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_labels(pairs: list[dict], judged: list[dict]) -> tuple[dict[tuple[str, str], dict], dict]:
    """{(original sha, edited sha): judge record} plus a judging summary."""
    by_pair_id = {}
    failed: set[str] = set()
    for record in judged:
        if record.get("failed"):
            failed.add(record["pair_id"])
            continue
        by_pair_id[record["pair_id"]] = record
    failed -= set(by_pair_id)
    by_sha: dict[tuple[str, str], dict] = {}
    missing = 0
    for row in pairs:
        record = by_pair_id.get(row["pair_id"])
        if record is None:
            missing += 1
            continue
        by_sha[(row["original_sha256"], row["edited_sha256"])] = record
    summary = {
        "n_pairs": len(pairs),
        "n_judged": len(by_pair_id),
        "n_failed": len(failed),
        "n_unjudged": missing,
        "judge_model": judged[0]["judge_model"] if judged else None,
        "parameters": next((r["parameters"] for r in judged if not r.get("failed")), None),
        "usage": {
            "prompt_tokens": sum(r.get("usage", {}).get("prompt_tokens", 0) for r in judged if not r.get("failed")),
            "completion_tokens": sum(
                r.get("usage", {}).get("completion_tokens", 0) for r in judged if not r.get("failed")
            ),
            "reasoning_tokens": sum(
                r.get("usage", {}).get("reasoning_tokens", 0) for r in judged if not r.get("failed")
            ),
            "cost_usd": sum(r.get("usage", {}).get("cost", 0.0) for r in judged if not r.get("failed")),
        },
    }
    return by_sha, summary


# --- acceptance tables ---------------------------------------------------------------------


def acceptance_tables(pairs: list[dict], by_sha: dict[tuple[str, str], dict]) -> dict:
    """Shares over owner rows: every (source, model, arm, unit) use of a pair counts once."""
    buckets: dict[str, dict[str, list[int]]] = {}

    def add(table: str, key: str, accepted: bool) -> None:
        cell = buckets.setdefault(table, {}).setdefault(key, [0, 0])
        cell[0] += 1
        cell[1] += int(accepted)

    relations: dict[str, Counter] = {}
    for row in pairs:
        record = by_sha.get((row["original_sha256"], row["edited_sha256"]))
        if record is None:
            continue
        accepted = bool(record["same_request"])
        relation = record["relation"]
        for owner in row["owners"]:
            source, arm = owner["source"], owner["arm"]
            add("by_source_arm", f"{source}/{arm}", accepted)
            relations.setdefault(f"{source}/{arm}", Counter())[relation] += 1
            if owner["unit_kind"]:
                add("by_unit_kind", f"{source}/{arm}/{owner['unit_kind']}", accepted)
            if owner["stratum"]:
                add("by_stratum", f"{source}/{arm}/{owner['stratum']}", accepted)
            if source == "x113" and owner["model"]:
                add("by_model_x113", f"{owner['model']}/{arm}", accepted)
    out = {
        table: {key: share_record(k, n) for key, (n, k) in sorted(cells.items())}
        for table, cells in buckets.items()
    }
    out["relations_by_source_arm"] = {
        key: {relation: counter.get(relation, 0) for relation in RELATIONS} for key, counter in sorted(relations.items())
    }
    distinct = Counter()
    distinct_accepted = Counter()
    for row in pairs:
        record = by_sha.get((row["original_sha256"], row["edited_sha256"]))
        if record is None:
            continue
        distinct[row["arm"]] += 1
        distinct_accepted[row["arm"]] += int(bool(record["same_request"]))
    out["by_arm_distinct_pairs"] = {
        arm: share_record(distinct_accepted[arm], distinct[arm]) for arm in sorted(distinct)
    }
    return out


def human_agreement(pairs: list[dict], by_sha: dict[tuple[str, str], dict]) -> dict:
    cells = Counter()
    rows = []
    for row in pairs:
        if row["readthrough"] is None:
            continue
        record = by_sha.get((row["original_sha256"], row["edited_sha256"]))
        if record is None:
            continue
        human = row["readthrough"]
        judge = "accept" if record["same_request"] else "reject"
        cells[(human, judge)] += 1
        rows.append({"pair_id": row["pair_id"], "arm": row["arm"], "human": human, "judge": judge, "relation": record["relation"]})
    n = sum(cells.values())
    n_ok = cells[("ok", "accept")] + cells[("ok", "reject")]
    n_bad = cells[("bad", "accept")] + cells[("bad", "reject")]
    return {
        "n_covered": n,
        "human_ok": n_ok,
        "human_bad": n_bad,
        "confusion": {f"human_{h}/judge_{j}": cells[(h, j)] for h in ("ok", "bad") for j in ("accept", "reject")},
        "judge_accepts_of_human_ok": share_record(cells[("ok", "accept")], n_ok),
        "judge_accepts_of_human_bad": share_record(cells[("bad", "accept")], n_bad),
        "rejected_pairs_the_human_passed": [r for r in rows if r["human"] == "ok" and r["judge"] == "reject"],
    }


def test_retest(main_rows: list[dict], retest_rows: list[dict]) -> dict:
    first = {r["pair_id"]: r for r in main_rows if not r.get("failed")}
    second = {r["pair_id"]: r for r in retest_rows if not r.get("failed")}
    shared = sorted(set(first) & set(second))
    a = [bool(first[p]["same_request"]) for p in shared]
    b = [bool(second[p]["same_request"]) for p in shared]
    same_relation = sum(1 for p in shared if first[p]["relation"] == second[p]["relation"])
    return {
        "n": len(shared),
        "seed_first": next((first[p]["parameters"]["seed"] for p in shared), None),
        "seed_second": next((second[p]["parameters"]["seed"] for p in shared), None),
        "same_request_agreement": (sum(1 for x, y in zip(a, b) if x == y) / len(shared)) if shared else None,
        "same_request_kappa": cohens_kappa(a, b),
        "relation_agreement": (same_relation / len(shared)) if shared else None,
        "accept_share_first": (sum(a) / len(a)) if a else None,
        "accept_share_second": (sum(b) / len(b)) if b else None,
    }


def judge_agreement(
    pairs: list[dict], by_sha: dict[tuple[str, str], dict], other: dict[tuple[str, str], dict]
) -> dict:
    """This judge against the other one: `same_request` agreement and kappa over distinct pairs
    (per arm) and over owner rows (per source and arm), the relation cross-table, and the share
    of owner rows both judges accept."""

    def agreement(a: list[bool], b: list[bool]) -> dict:
        n = len(a)
        return {
            "n": n,
            "agreement": (sum(1 for x, y in zip(a, b) if x == y) / n) if n else None,
            "kappa": cohens_kappa(a, b),
            "accept_this": (sum(a) / n) if n else None,
            "accept_other": (sum(b) / n) if n else None,
            "accept_both": (sum(1 for x, y in zip(a, b) if x and y) / n) if n else None,
            "confusion": {
                "both_accept": sum(1 for x, y in zip(a, b) if x and y),
                "this_only": sum(1 for x, y in zip(a, b) if x and not y),
                "other_only": sum(1 for x, y in zip(a, b) if y and not x),
                "both_reject": sum(1 for x, y in zip(a, b) if not x and not y),
            },
        }

    distinct: dict[str, tuple[list[bool], list[bool]]] = {}
    owner: dict[str, tuple[list[bool], list[bool]]] = {}
    both_accepted: dict[str, list[int]] = {}
    cross = Counter()
    for row in pairs:
        key = (row["original_sha256"], row["edited_sha256"])
        this, that = by_sha.get(key), other.get(key)
        if this is None or that is None:
            continue
        a, b = bool(this["same_request"]), bool(that["same_request"])
        cross[(this["relation"], that["relation"])] += 1
        for bucket in ("all", row["arm"]):
            distinct.setdefault(bucket, ([], []))
            distinct[bucket][0].append(a)
            distinct[bucket][1].append(b)
        for owner_row in row["owners"]:
            bucket = f"{owner_row['source']}/{owner_row['arm']}"
            owner.setdefault(bucket, ([], []))
            owner[bucket][0].append(a)
            owner[bucket][1].append(b)
            cell = both_accepted.setdefault(bucket, [0, 0])
            cell[0] += 1
            cell[1] += int(a and b)
    return {
        "distinct_pairs": {bucket: agreement(a, b) for bucket, (a, b) in sorted(distinct.items())},
        "owner_rows_by_source_arm": {bucket: agreement(a, b) for bucket, (a, b) in sorted(owner.items())},
        "both_accepted_by_source_arm": {
            bucket: share_record(k, n) for bucket, (n, k) in sorted(both_accepted.items())
        },
        "relation_cross_table": {
            this: {that: cross.get((this, that), 0) for that in RELATIONS} for this in RELATIONS
        },
    }


# --- refusal2 effects ----------------------------------------------------------------------


def refusal_rows(path: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("failed"):
                continue
            values = record.get("refusal", [])
            if len(values) != N_RESPONSES or not all(isinstance(v, bool) for v in values):
                raise SystemExit(f"FATAL: invalid refusal record in {path}")
            key = str(record.get("prompt_id") or record.get("cell_id"))
            out[key] = np.asarray(values, dtype=float)
    return out


def x113_refusal(tag: str) -> dict[str, dict[str, np.ndarray]]:
    """{arm: {unit id: labels}} for S_T and C_T from x113's judge file."""
    arms: dict[str, dict[str, np.ndarray]] = {"S_T": {}, "C_T": {}}
    path = X113_RUN / tag / "judge" / "refusal2.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("failed"):
                continue
            arm = record["condition"]
            if arm not in arms:
                continue
            values = record["refusal"]
            if len(values) != N_RESPONSES or not all(isinstance(v, bool) for v in values):
                raise SystemExit(f"FATAL: invalid x113 refusal record {record.get('cell_id')}")
            arms[arm][str(record["prompt_id"])] = np.asarray(values, dtype=float)
    return arms


def rates(labels: dict[str, np.ndarray], ids: list[str]) -> np.ndarray:
    out = np.full(len(ids), np.nan)
    for k, unit_id in enumerate(ids):
        values = labels.get(unit_id)
        if values is not None:
            out[k] = float(np.mean(values))
    return out


def paired_delta(o: np.ndarray, e: np.ndarray, mask: np.ndarray) -> float:
    use = mask & np.isfinite(o) & np.isfinite(e)
    return float(np.mean(o[use] - e[use])) if use.any() else float("nan")


def bootstrap_delta(
    o: np.ndarray, e: np.ndarray, mask: np.ndarray, strata: np.ndarray, n_boot: int, seed: int
) -> tuple[float, np.ndarray]:
    point = paired_delta(o, e, mask)
    rng = np.random.default_rng(seed)
    groups = [np.where(strata == s)[0] for s in np.unique(strata)]
    samples = np.empty(n_boot)
    for start in range(0, n_boot, 500):
        stop = min(start + 500, n_boot)
        index = np.concatenate(
            [g[rng.integers(0, len(g), size=(stop - start, len(g)))] for g in groups if len(g)], axis=1
        )
        drawn = {}
        for name, values in (("o", o), ("e", e)):
            picked = values[index]
            redraw = rng.binomial(N_RESPONSES, np.nan_to_num(picked, nan=0.0)) / N_RESPONSES
            drawn[name] = np.where(np.isfinite(picked), redraw, np.nan)
        mask_drawn = mask[index]
        use = mask_drawn & np.isfinite(drawn["o"]) & np.isfinite(drawn["e"])
        diff = np.where(use, drawn["o"] - drawn["e"], np.nan)
        with np.errstate(invalid="ignore"):
            samples[start:stop] = np.nanmean(diff, axis=1)
    return point, samples


UNLABELLED: Counter = Counter()


def accepted_masks(
    has: np.ndarray, keys: list[tuple[str, str] | None], judges: dict[str, dict], where: str
) -> dict[str, np.ndarray]:
    """{"all_pairs": has, "accepted": ..., "accepted_<other>": ..., "accepted_both": ...} —
    one mask per judge, the pairs that judge reads as the same request, and their conjunction.
    A pair a judge returned no label for is not in that judge's accepted set; `UNLABELLED` counts
    those, and `main` compares the total against `--allow-unlabelled`."""
    masks = {"all_pairs": has}
    for name, by_sha in judges.items():
        accepted = np.zeros(len(keys), dtype=bool)
        for k, key in enumerate(keys):
            if key is None:
                continue
            record = by_sha.get(key)
            if record is None:
                UNLABELLED[f"{name}:{where}"] += 1
                continue
            accepted[k] = bool(record["same_request"])
        masks[name] = has & accepted
    if len(judges) > 1:
        both = has.copy()
        for name in judges:
            both &= masks[name]
        masks["accepted_both"] = both
    return masks


def effect_record(
    o: np.ndarray,
    e: np.ndarray,
    masks: dict[str, np.ndarray],
    strata: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict:
    out = {}
    for name, mask in masks.items():
        point, samples = bootstrap_delta(o, e, mask, strata, n_boot, seed)
        use = mask & np.isfinite(o) & np.isfinite(e)
        out[name] = {
            "n": int(use.sum()),
            "r_O": float(np.mean(o[use])) if use.any() else None,
            "r_edited": float(np.mean(e[use])) if use.any() else None,
            "delta": point,
            "ci95": interval(samples),
        }
    out["accepted_minus_all"] = (
        out["accepted"]["delta"] - out["all_pairs"]["delta"]
        if np.isfinite(out["accepted"]["delta"]) and np.isfinite(out["all_pairs"]["delta"])
        else None
    )
    return out


def x106_effects(judges: dict[str, dict], n_boot: int) -> dict:
    cohort = read_json(X106_RUN / "x106_cohort.json")
    units = [u for u in cohort["units"] if u["source"] != "xstest"]
    ids = [u["id"] for u in units]
    strata = np.asarray([u["stratum"] for u in units])
    masks = {}
    for arm in ("S", "C", "R"):
        has = np.asarray([bool(u["prompts"].get(arm)) for u in units])
        keys = [
            (u["prompt_sha256"]["O"], u["prompt_sha256"][arm]) if u["prompts"].get(arm) else None for u in units
        ]
        masks[arm] = accepted_masks(has, keys, judges, f"x106 {arm}")
    out = {}
    for tag in TAGS:
        o = rates(refusal_rows(X106_RUN / tag / "x106_refusal_O_base.jsonl"), ids)
        per_arm = {}
        for arm in ("S", "C", "R"):
            e = rates(refusal_rows(X106_RUN / tag / f"x106_refusal_{arm}_base.jsonl"), ids)
            per_arm[arm] = effect_record(o, e, masks[arm], strata, n_boot, BOOTSTRAP_SEED)
        out[tag] = {"label": LABELS[tag], "population": "x106 Hard-1K", "arms": per_arm}
    return out


def x108_effects(judges: dict[str, dict], n_boot: int) -> dict:
    cohort = read_json(X108_RUN / "x108_cohort.json")
    units = [u for u in cohort["units"] if u["source"] == "orbench_toxic"]
    ids = [u["id"] for u in units]
    strata = np.asarray([u["stratum"] for u in units])
    masks = {}
    for arm in ("S", "R"):
        has = np.asarray([bool(u["prompts"].get(arm)) for u in units])
        keys = [
            (u["prompt_sha256"]["O"], u["prompt_sha256"][arm]) if u["prompts"].get(arm) else None for u in units
        ]
        masks[arm] = accepted_masks(has, keys, judges, f"x108 {arm}")
    out = {}
    for tag in TAGS:
        o = rates(refusal_rows(X108_RUN / tag / "x108_refusal_O_base.jsonl"), ids)
        per_arm = {}
        for arm in ("S", "R"):
            e = rates(refusal_rows(X108_RUN / tag / f"x108_refusal_{arm}_base.jsonl"), ids)
            per_arm[arm] = effect_record(o, e, masks[arm], strata, n_boot, BOOTSTRAP_SEED)
        out[tag] = {"label": LABELS[tag], "population": "x108 OR-Bench Toxic", "arms": per_arm}
    return out


def x113_effects(judges: dict[str, dict], n_boot: int) -> dict:
    exclusions = read_json(X113_RUN / "x85_exclusions.json")
    out = {}
    for tag in TAGS:
        texts = read_json(X113_RUN / tag / "texts.json")
        excluded = {row["id"] for row in exclusions["models"][tag]["excluded"]}
        units = [
            u for u in texts["units"] if u["stratum"] in ("core", "llama_extra") and u["id"] not in excluded
        ]
        ids = [u["id"] for u in units]
        strata = np.asarray([u["stratum"] for u in units])
        o = rates(refusal_rows(X106_RUN / tag / "x106_refusal_O_base.jsonl"), ids)
        labels = x113_refusal(tag)
        per_arm = {}
        for arm in ("S_T", "C_T"):
            e = rates(labels[arm], ids)
            has = np.asarray([bool(u["texts"].get(arm)) for u in units])
            keys = [(u["prompt_sha256"], u["sha256"][arm]) if u["texts"].get(arm) else None for u in units]
            masks = accepted_masks(has, keys, judges, f"x113 {tag} {arm}")
            per_arm[arm] = effect_record(o, e, masks, strata, n_boot, BOOTSTRAP_SEED)
        out[tag] = {"label": LABELS[tag], "population": "x113 Hard-1K, x85 units excluded", "arms": per_arm}
    return out


# --- output --------------------------------------------------------------------------------


def f(x, digits=3, sign=False) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"


def ci(values, digits=3, sign=False) -> str:
    low, high = values
    if low is None or high is None:
        return "—"
    return f"[{f(low, digits, sign)}, {f(high, digits, sign)}]"


def share_cell(record: dict) -> str:
    if record["share"] is None:
        return "—"
    low, high = record["wilson95"]
    return f"{record['share']:.3f} [{low:.3f}, {high:.3f}] ({record['accepted']}/{record['n']})"


def effect_cell(record: dict) -> str:
    return f"{f(record['delta'], 3, True)} {ci(record['ci95'], 3, True)} (n {record['n']})"


def write_md(out: dict, path: Path) -> None:
    lines = [f"# x118: an independent judge of the substituted prompts — `{out['judge']['judge_model']}`", ""]
    judge = out["judge"]
    params = judge["parameters"] or {}
    lines += [
        f"Judge `{judge['judge_model']}`, reasoning effort `{params.get('reasoning_effort')}`, "
        f"temperature {params.get('temperature') if params.get('temperature') is not None else 'n/a'}, "
        f"seed {params.get('seed') if params.get('seed') is not None else 'n/a'}, "
        f"`{params.get('response_format')}` output. "
        f"{judge['n_judged']} of {judge['n_pairs']} distinct pairs judged, {judge['n_failed']} failed, "
        f"${judge['usage']['cost_usd']:.4f}.",
        "",
        "## Accepted as the same request, per source and arm (owner rows)",
        "",
        "| source / arm | share accepted [Wilson 95%] |",
        "|---|---|",
    ]
    for key, record in out["acceptance"]["by_source_arm"].items():
        lines.append(f"| {key} | {share_cell(record)} |")
    lines += ["", "## Relation, per source and arm (owner rows)", "", "| source / arm | same | narrower | broader | different |", "|---|---|---|---|---|"]
    for key, counts in out["acceptance"]["relations_by_source_arm"].items():
        lines.append(f"| {key} | {counts['same']} | {counts['narrower']} | {counts['broader']} | {counts['different']} |")
    lines += ["", "## Per model, the table-guided substitution texts", "", "| model / arm | share accepted [Wilson 95%] |", "|---|---|"]
    for key, record in out["acceptance"].get("by_model_x113", {}).items():
        lines.append(f"| {key} | {share_cell(record)} |")
    lines += ["", "## By unit kind and stratum", "", "| bucket | share accepted [Wilson 95%] |", "|---|---|"]
    for table in ("by_unit_kind", "by_stratum"):
        for key, record in out["acceptance"].get(table, {}).items():
            lines.append(f"| {key} | {share_cell(record)} |")
    human = out["human_agreement"]
    lines += [
        "",
        "## Agreement with the human read-through",
        "",
        f"{human['n_covered']} read-through pairs ({human['human_ok']} the human passed, "
        f"{human['human_bad']} the human marked `BAD:`).",
        "",
        "| | judge accepts | judge rejects |",
        "|---|---|---|",
        f"| human passed | {human['confusion']['human_ok/judge_accept']} | {human['confusion']['human_ok/judge_reject']} |",
        f"| human marked BAD | {human['confusion']['human_bad/judge_accept']} | {human['confusion']['human_bad/judge_reject']} |",
    ]
    retest = out["test_retest"]
    lines += [
        "",
        "## Test–retest",
        "",
        f"{retest['n']} pairs re-judged at seed {retest['seed_second']} against seed {retest['seed_first']}: "
        f"`same_request` agreement {f(retest['same_request_agreement'])}, "
        f"Cohen's kappa {f(retest['same_request_kappa'])}, "
        f"relation agreement {f(retest['relation_agreement'])}, "
        f"accept share {f(retest['accept_share_first'])} then {f(retest['accept_share_second'])}.",
    ]
    agreement = out.get("judge_agreement")
    if agreement:
        other = out["other_judge"]["judge_model"]
        lines += [
            "",
            f"## Agreement with `{other}`",
            "",
            "| bucket | n | agreement | kappa | this accepts | other accepts | both accept |",
            "|---|---|---|---|---|---|---|",
        ]
        for scope in ("distinct_pairs", "owner_rows_by_source_arm"):
            for key, record in agreement[scope].items():
                lines.append(
                    f"| {scope}: {key} | {record['n']} | {f(record['agreement'])} | {f(record['kappa'])} | "
                    f"{f(record['accept_this'])} | {f(record['accept_other'])} | {f(record['accept_both'])} |"
                )
        lines += [
            "",
            f"Relation cross-table over distinct pairs (rows: this judge, columns: `{other}`)",
            "",
            "| this \\ other | " + " | ".join(RELATIONS) + " |",
            "|---" * (1 + len(RELATIONS)) + "|",
        ]
        for this in RELATIONS:
            lines.append(f"| {this} | " + " | ".join(str(agreement["relation_cross_table"][this][that]) for that in RELATIONS) + " |")
    this_judge = out["judge"]["judge_model"]
    other_judge = (out.get("other_judge") or {}).get("judge_model")
    titles = {
        "all_pairs": "all pairs",
        "accepted": f"accepted by {this_judge}",
        "accepted_other": f"accepted by {other_judge}",
        "accepted_both": "accepted by both",
    }
    mask_names = [
        m
        for m in next(iter(out["effects"]["x106"].values()))["arms"]["S"]
        if m != "accepted_minus_all"
    ]
    for block, title, arms in (
        ("x106", "x106 Hard-1K (deletion-pass nomination), `refusal2`", ("S", "C", "R")),
        ("x113", "x113 Hard-1K (table nomination, x85 units excluded), `refusal2`", ("S_T", "C_T")),
        ("x108", "x108 OR-Bench Toxic, `refusal2`", ("S", "R")),
    ):
        header = " | ".join(f"{a}: {titles.get(m, m)}" for a in arms for m in mask_names)
        lines += ["", f"## {title}", "", f"| model | {header} |", "|---" * (1 + len(arms) * len(mask_names)) + "|"]
        for tag in TAGS:
            record = out["effects"][block][tag]
            cells = [effect_cell(record["arms"][arm][m]) for arm in arms for m in mask_names]
            lines.append(f"| {record['label']} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--judge-file", default=str(MEANING), help="the judge file to analyse")
    parser.add_argument("--retest-file", default=str(MEANING_RETEST), help="its test-retest file")
    parser.add_argument("--compare-with", help="another judge's file: agreement and effects on the pairs both accept")
    parser.add_argument("--out-stem", default="results", help="results.<json|md> in data/runs/x118")
    parser.add_argument(
        "--allow-unlabelled",
        type=int,
        default=0,
        help="pairs the judge returned no label for (its route refused the call); they are not accepted",
    )
    args = parser.parse_args()

    pairs = read_jsonl(PAIRS)
    judged = read_jsonl(Path(args.judge_file))
    retest = read_jsonl(Path(args.retest_file))
    by_sha, judge_summary = load_labels(pairs, judged)
    if judge_summary["n_unjudged"] > args.allow_unlabelled:
        raise SystemExit(f"FATAL: {judge_summary['n_unjudged']} pairs are still unjudged")
    judges = {"accepted": by_sha}
    other_summary = agreement = None
    if args.compare_with:
        other_rows = read_jsonl(Path(args.compare_with))
        other_sha, other_summary = load_labels(pairs, other_rows)
        if other_summary["n_unjudged"] > args.allow_unlabelled:
            raise SystemExit(f"FATAL: {other_summary['n_unjudged']} pairs are unjudged in {args.compare_with}")
        judges["accepted_other"] = other_sha
        agreement = judge_agreement(pairs, by_sha, other_sha)

    model = judge_summary["judge_model"]
    cost_log = [row for row in read_jsonl(COST_LOG) if row.get("judge_model") == model]
    out = {
        "experiment": "x118",
        "stage": 3,
        "judge": judge_summary,
        "other_judge": other_summary,
        "cost_log": cost_log,
        "acceptance": acceptance_tables(pairs, by_sha),
        "human_agreement": human_agreement(pairs, by_sha),
        "test_retest": test_retest(judged, retest),
        "judge_agreement": agreement,
        "effects": {
            "x106": x106_effects(judges, args.n_boot),
            "x113": x113_effects(judges, args.n_boot),
            "x108": x108_effects(judges, args.n_boot),
        },
        "bootstrap": {"n_replicates": args.n_boot, "seed": BOOTSTRAP_SEED, "within_stratum": True},
        "unlabelled_in_effect_masks": dict(UNLABELLED),
    }
    results_json = RUN / f"{args.out_stem}.json"
    results_md = RUN / f"{args.out_stem}.md"
    results_json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    write_md(out, results_md)
    print(f"wrote {results_json} and {results_md}")
    for key, record in out["acceptance"]["by_source_arm"].items():
        print(f"  {key}: {share_cell(record)}")


if __name__ == "__main__":
    main()
