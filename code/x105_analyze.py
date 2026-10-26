"""x105 Stage 4: per model, is the boundary write bidirectional, dose-ordered and direction-specific?

Per model, on the prompts with successful `refusal2` labels in every arm: the suppression the
negative dose produces on the inserted (E) texts, the induction the positive dose produces on
the original (O) texts, both at two magnitudes, the oriented dose slope, and the same
suppression under the model's norm-matched and KL-matched random directions. One two-stage
paired bootstrap (prompts, then the ten binary responses within each arm) carries every
estimand, so the target-minus-control differences are intervals on the same prompt sample.
Holm runs over the seven transfer models within each primary test; Qwen3.5-4B is the pipeline
reference row, outside the family, as in x103.

Three cohort modes, same seeds, bootstrap and Holm in each (`x114_cohorts`). `extended`
(the canonical result files) pools the 64 x101 units whose prompt and inserted word are both
outside the x85 cohort the directions were fit on, as frozen in
`data/runs/x105/x105_x85_exclusions.json`, with the x114 units, whose arms x114 ran under
this same design and which the rule leaves intact; `x101_leak_free` runs those 64 x101 units
alone into the `_x101only` siblings; `full` runs all 130 x101 units into the `_full`
siblings. The default runs all three (`--cohort all`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from x83_analyze import load_generations as load_base_generations, load_judgments as load_base_judgments
from x86_analyze import printable_fraction, repeated_fourgram_dominated
from x114_cohorts import MODES, arm_loader, cohort_record, result_path_for, uses_x114, x101_mode
import x114_common
from x105_common import (
    ALL_TAGS,
    ARM_NAMES,
    ARMS,
    BASE_ARMS,
    BOOTSTRAP_SEED,
    CONTROL_DRAWS,
    DOSE_MAGNITUDES,
    FRESH_TAGS,
    HALF_MAGNITUDE,
    IMPORTED_TAG,
    MAIN_MAGNITUDE,
    MODELS,
    N_BOOTSTRAP,
    N_RESPONSES,
    RANDOM_ARMS_E,
    RANDOM_ARM_E_KL,
    RANDOM_ARM_O,
    RESULTS,
    TEXT_CONDITIONS,
    TRUE_ARMS,
    atomic_json,
    base_cohort_sha,
    base_generations_path,
    base_refusal_path,
    bootstrap_estimands,
    estimands,
    file_record,
    generations_path,
    holm,
    inputs_path,
    interval,
    kl_calibration_path,
    load_cohort,
    load_exclusions,
    monotone_flags,
    one_sided_p,
    rate_matrix,
    read_json,
    refusal_path,
    retained_ids,
    summarize,
    verdict,
)

PRIMARY_TESTS = ("suppress_E_full", "induce_O_full", "true_minus_random_E")
X101_RESULT = RESULTS / "x101_cross_model_insertion.json"
FIGURE_ORDER = (
    ("qwen08b", "0.8B"),
    ("qwen2b", "2B"),
    ("qwen", "4B"),
    ("qwen9b", "9B"),
    ("qwen27b", "27B"),
    ("llama", "Llama"),
    ("gemma", "Gemma"),
    ("phi", "Phi"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", default=str(inputs_path()))
    parser.add_argument("--out", default=str(RESULTS / "x105_cross_model_direction_causality.json"))
    parser.add_argument("--figure", default=str(RESULTS / "x105_cross_model_direction_causality.pdf"))
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--tags", nargs="*", default=None, help="debug: analyse a subset of models")
    parser.add_argument(
        "--cohort", choices=("all", *MODES), default="all",
        help=(
            "extended (x101 leak-free + x114) writes the canonical files, x101_leak_free the "
            "`_x101only` siblings, full the `_full` siblings; all (default) writes all three"
        ),
    )
    return parser.parse_args()


def load_arm(tag: str, arm: str, units: dict[int, dict], cohort_sha: str) -> tuple[dict[int, dict], dict[int, np.ndarray], dict]:
    """One x105 arm's generations and successful refusal2 labels, validated row by row."""
    spec = next(s for s in ARMS if s["arm"] == arm)
    text_condition = spec["text"]
    generation_path, judge_path = generations_path(tag, arm), refusal_path(tag, arm)
    obj = read_json(generation_path)
    if (
        obj.get("experiment") != "x105"
        or obj.get("arm") != arm
        or obj.get("model_tag") != tag
        or obj.get("cohort_sha256") != cohort_sha
        or obj.get("text_condition") != text_condition
        or obj.get("site_audit", {}).get("status") != "pass"
    ):
        raise SystemExit(f"FATAL: {tag}/{arm} generation lineage or site audit is not valid")
    rows = {}
    for row in obj["rows"]:
        prompt_id = int(row["id"])
        if prompt_id not in units:
            continue
        responses = row.get("llm_responses", [])
        if (
            row.get("condition") != arm
            or row.get("prompt_sha256") != units[prompt_id]["prompt_sha256"][text_condition]
            or row.get("arms_sha256") != cohort_sha
            or len(responses) != N_RESPONSES
            or abs(float(row["signed_dose"]) - float(obj["signed_dose"])) > 1e-12
        ):
            raise SystemExit(f"FATAL: invalid {tag}/{arm} generation row {prompt_id}")
        if prompt_id in rows:
            raise SystemExit(f"FATAL: duplicate {tag}/{arm} generation row {prompt_id}")
        rows[prompt_id] = row
    labels: dict[int, np.ndarray] = {}
    with judge_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("failed"):
                continue
            prompt_id = int(record["id"])
            source = rows.get(prompt_id)
            if source is None:
                continue
            values = record.get("refusal", [])
            if (
                record.get("condition") != arm
                or record.get("response_sha256") != source["response_sha256"]
                or record.get("arms_sha256") != cohort_sha
                or len(values) != N_RESPONSES
                or not all(isinstance(value, bool) for value in values)
            ):
                raise SystemExit(f"FATAL: invalid {tag}/{arm} refusal record {prompt_id}")
            if prompt_id in labels:
                raise SystemExit(f"FATAL: duplicate successful {tag}/{arm} refusal record {prompt_id}")
            labels[prompt_id] = np.asarray(values, dtype=np.int8)
    meta = {
        "generations": file_record(generation_path),
        "refusal": file_record(judge_path),
        "signed_dose": float(obj["signed_dose"]),
        "family": obj["family"],
        "control_draw": obj.get("control_draw"),
        "text_condition": text_condition,
        "direction": obj["direction"].get("kind"),
        "site": obj["site"],
        "n_rows": len(rows),
        "n_labelled": len(labels),
    }
    return rows, labels, meta


def load_base_arm(tag: str, condition: str, units: dict[int, dict], cohort: dict) -> tuple[dict[int, dict], dict[int, np.ndarray], dict]:
    """The dose-0 arm: x101's (x83's for Qwen3.5-4B) own generations and judgments, unchanged."""
    sha = base_cohort_sha(tag, cohort)
    generation_path, judge_path = base_generations_path(tag, condition), base_refusal_path(tag, condition)
    rows = load_base_generations(generation_path, condition, units, sha)
    labels = load_base_judgments(judge_path, condition, units, rows, sha)
    return rows, labels, {
        "generations": file_record(generation_path),
        "refusal": file_record(judge_path),
        "signed_dose": 0.0,
        "family": "base",
        "control_draw": None,
        "text_condition": condition,
        "direction": None,
        "site": None,
        "n_rows": len(rows),
        "n_labelled": len(labels),
    }


def integrity(rows_by_arm: dict[str, dict[int, dict]], ids: list[int]) -> dict:
    report = {}
    for arm, rows in rows_by_arm.items():
        texts = [text for prompt_id in ids for text in rows[prompt_id]["llm_responses"]]
        truncated = [bool(v) for prompt_id in ids for v in rows[prompt_id]["truncated"]]
        report[arm] = {
            "n_responses": len(texts),
            "retry_rows": sum(bool(rows[prompt_id].get("used_retry")) for prompt_id in ids),
            "truncation_rate": float(np.mean(truncated)),
            "median_char_length": float(np.median([len(text) for text in texts])),
            "median_printable_fraction": float(np.median([printable_fraction(text) for text in texts])),
            "repeated_fourgram_dominated_fraction": float(np.mean([repeated_fourgram_dominated(text) for text in texts])),
        }
    return report


def load_tag_arms(tag: str, cohort: dict) -> tuple[dict, dict, dict]:
    """Every arm of one model, validated row by row; loaded once and analysed per cohort mode."""
    units = {int(unit["id"]): unit for unit in cohort["units"]}
    rows_by_arm: dict[str, dict[int, dict]] = {}
    labels_by_arm: dict[str, dict[int, np.ndarray]] = {}
    arm_meta: dict[str, dict] = {}
    for condition in TEXT_CONDITIONS:
        arm = BASE_ARMS[condition]
        rows_by_arm[arm], labels_by_arm[arm], arm_meta[arm] = load_base_arm(tag, condition, units, cohort)
    for arm in ARM_NAMES:
        rows_by_arm[arm], labels_by_arm[arm], arm_meta[arm] = load_arm(tag, arm, units, cohort["cohort_sha256"])
    return rows_by_arm, labels_by_arm, arm_meta


def source_ids(tag: str, cohort: dict, labels_by_arm: dict, keep_ids: list[int] | None) -> tuple[list[int], list[int]]:
    """The prompts of one cohort labelled in every arm, restricted to `keep_ids` when given."""
    units = {int(unit["id"]): unit for unit in cohort["units"]}
    ids = sorted(set(units) & set.intersection(*(set(labels_by_arm[arm]) for arm in labels_by_arm)))
    dropped = sorted(set(units) - set(ids))
    if len(ids) < len(units) - 3:
        raise SystemExit(f"FATAL: {tag}: only {len(ids)} of {len(units)} prompts are labelled in every arm")
    if keep_ids is not None:
        keep = set(keep_ids)
        ids = [i for i in ids if i in keep]
    return ids, dropped


load_x114_arm = arm_loader("x114", x114_common.generations_path, x114_common.refusal_path)


def load_x114_base_arm(tag: str, condition: str, units: dict[int, dict], cohort: dict):
    """The x114 dose-0 arm: its own base generations and judgments, read with x83's validators."""
    sha = cohort["cohort_sha256"]
    generation_path = x114_common.generations_path(tag, condition)
    judge_path = x114_common.refusal_path(tag, condition)
    rows = load_base_generations(generation_path, condition, units, sha)
    labels = load_base_judgments(judge_path, condition, units, rows, sha)
    return rows, labels, {
        "generations": file_record(generation_path),
        "refusal": file_record(judge_path),
        "signed_dose": 0.0,
        "family": "base",
        "control_draw": None,
        "text_condition": condition,
        "direction": None,
        "site": None,
        "n_rows": len(rows),
        "n_labelled": len(labels),
    }


def load_x114_arms(tag: str, cohort: dict) -> tuple[dict, dict, dict]:
    """Every arm of one model on the x114 cohort: its two base arms and the eight dosed arms."""
    units = {int(unit["id"]): unit for unit in cohort["units"]}
    rows_by_arm: dict[str, dict[int, dict]] = {}
    labels_by_arm: dict[str, dict[int, np.ndarray]] = {}
    arm_meta: dict[str, dict] = {}
    for condition in TEXT_CONDITIONS:
        arm = BASE_ARMS[condition]
        rows_by_arm[arm], labels_by_arm[arm], arm_meta[arm] = load_x114_base_arm(tag, condition, units, cohort)
    for arm in ARM_NAMES:
        rows_by_arm[arm], labels_by_arm[arm], arm_meta[arm] = load_x114_arm(tag, arm, units, cohort["cohort_sha256"])
    return rows_by_arm, labels_by_arm, arm_meta


def analyze_tag(
    tag: str,
    sources: list[dict],
    inputs: dict,
    n_boot: int,
    seed: int,
) -> dict:
    """One model's report over one or more cohorts pooled on their prompts.

    Each source is {"source": name, "cohort": cohort, "arms": (rows, labels, meta),
    "keep_ids": ids or None}; the per-prompt refusal2 rates of every arm are stacked over the
    pooled prompt set, so one bootstrap carries every estimand on one prompt sample."""
    pooled_ids: list[int] = []
    pooled_words: set[str] = set()
    labels_parts: dict[str, list[np.ndarray]] = {}
    rows_parts: dict[str, dict[int, dict]] = {}
    per_source: dict[str, dict] = {}
    arm_meta: dict[str, dict] = {}
    arm_meta_by_source: dict[str, dict] = {}
    for entry in sources:
        cohort, (rows_by_arm, labels_by_arm, meta) = entry["cohort"], entry["arms"]
        units = {int(unit["id"]): unit for unit in cohort["units"]}
        ids, dropped = source_ids(tag, cohort, labels_by_arm, entry.get("keep_ids"))
        if labels_parts and sorted(labels_parts) != sorted(labels_by_arm):
            raise SystemExit(f"FATAL: {tag}: the pooled cohorts do not carry the same arms")
        for arm, values in labels_by_arm.items():
            labels_parts.setdefault(arm, []).append(np.stack([values[i] for i in ids]))
            rows_parts.setdefault(arm, {}).update({i: rows_by_arm[arm][i] for i in ids})
        pooled_ids.extend(ids)
        pooled_words |= {str(units[i]["word"]) for i in ids}
        per_source[entry["source"]] = {
            "n_prompts": len(ids),
            "n_distinct_words": len({str(units[i]["word"]) for i in ids}),
            "prompt_ids": ids,
            "dropped_prompts": dropped,
            "cohort_sha256": cohort["cohort_sha256"],
        }
        arm_meta_by_source[entry["source"]] = meta
        if not arm_meta:
            arm_meta = meta
    ids = pooled_ids
    rows_by_arm = rows_parts
    labels = {arm: np.concatenate(parts) for arm, parts in labels_parts.items()}
    rates = rate_matrix(labels)
    point = estimands(rates)
    samples = bootstrap_estimands(labels, n_bootstrap=n_boot, seed=seed)
    report_all = summarize(point, samples)

    calibration = None
    calibration_path = kl_calibration_path(tag)
    if calibration_path.exists():
        obj = read_json(calibration_path)
        calibration = {
            "file": file_record(calibration_path),
            "status": obj.get("status"),
            "true_main_dose_mean_kl": obj.get("true_main_dose_mean_kl"),
            "true_dose_kl_monotone": obj.get("true_dose_kl_monotone"),
            "random_match": {
                side: {
                    "matched_signed_dose": record.get("matched_signed_dose"),
                    "matched_mean_kl": record.get("matched_mean_kl"),
                    "target_kl": record.get("target_kl"),
                    "multiple_of_true_dose": abs(float(record["matched_signed_dose"])) / MAIN_MAGNITUDE,
                    "status": record.get("status"),
                }
                for side, record in obj.get("random_kl_match", {}).items()
            },
        }

    report = {
        "model_tag": tag,
        "model_id": MODELS[tag]["model_id"] if tag != IMPORTED_TAG else inputs["models"][tag]["model_id"],
        "role": inputs["models"][tag]["role"],
        "n_prompts": len(ids),
        "n_distinct_words": len(pooled_words),
        "prompt_ids": ids,
        "dropped_prompts": [i for entry in per_source.values() for i in entry["dropped_prompts"]],
        "per_source": per_source,
        "arms_by_source": arm_meta_by_source if len(arm_meta_by_source) > 1 else None,
        "hook_block": inputs["models"][tag]["hook_block"],
        "direction_norms": inputs["models"][tag]["direction_norms"],
        "control_diagnostics": inputs["models"][tag]["control_directions"]["diagnostics"],
        "benign_kl": calibration,
        "arm_means": {arm: float(labels[arm].mean()) for arm in labels},
        "arm_doses": {arm: arm_meta[arm]["signed_dose"] for arm in arm_meta},
        "arms": arm_meta,
        "primary": {name: report_all[name] for name in PRIMARY_TESTS},
        "estimands": report_all,
        "monotonicity": monotone_flags(point, samples),
        "generation_integrity": integrity(rows_by_arm, ids),
    }
    report["verdict"] = verdict(report)
    return report


def cross_model(per_model: dict) -> dict:
    tested = [tag for tag in FRESH_TAGS if tag in per_model]
    table = {}
    for name in PRIMARY_TESTS:
        raw = [per_model[tag]["primary"][name]["one_sided_p"] for tag in tested]
        adjusted = holm(raw)
        table[name] = [
            {
                "model_tag": tag,
                "estimate": per_model[tag]["primary"][name]["estimate"],
                "ci95": per_model[tag]["primary"][name]["ci95"],
                "one_sided_p": p,
                "holm_p_over_transfer_models": adj,
            }
            for tag, p, adj in zip(tested, raw, adjusted)
        ]
    return {
        "holm_family": "the seven x88 transfer models, one family per primary test",
        "reference_outside_family": IMPORTED_TAG,
        "tests": table,
        "verdicts": {tag: per_model[tag]["verdict"] for tag in per_model},
    }


def x101_cross_check(per_model: dict) -> dict | None:
    """The base arms reproduce x101's own end-position rise, since they are x101's rows."""
    if not X101_RESULT.exists():
        return None
    x101 = read_json(X101_RESULT)
    rows = {row["model_tag"]: row for row in x101["cross_model"]["ungated_all_130"] if row["position"] == "E"}
    out = {}
    for tag, report in per_model.items():
        if tag not in rows:
            continue
        out[tag] = {
            "x105_base_insertion_rise": report["estimands"]["insertion_rise_E_minus_O"]["estimate"],
            "x101_ungated_E_delta": rows[tag]["delta"],
            "x101_n": rows[tag]["n"],
            "x105_n": report["n_prompts"],
            "difference": report["estimands"]["insertion_rise_E_minus_O"]["estimate"] - rows[tag]["delta"],
        }
    return {"source": file_record(X101_RESULT), "per_model": out}


def draw_figure(per_model: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 6.5,
            "ytick.labelsize": 7, "legend.fontsize": 6.0, "axes.spines.top": False,
            "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
            "ytick.major.width": 0.6, "axes.edgecolor": "#333333", "xtick.color": "#333333",
            "ytick.color": "#333333", "text.color": "#333333", "axes.labelcolor": "#333333",
            "grid.color": "#DDDDDD", "grid.linewidth": 0.5,
        }
    )
    blue, orange, grey, muted = "#0072B2", "#E69F00", "#666666", "#888888"
    order = [(tag, label) for tag, label in FIGURE_ORDER if tag in per_model]
    xs = list(range(len(order)))
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.3))

    def series(ax, key, color, marker, shift, filled, node="estimands"):
        points, lower, upper = [], [], []
        for tag, _ in order:
            entry = per_model[tag][node][key]
            ci = entry["ci95"]
            points.append(entry["estimate"])
            lower.append(entry["estimate"] - ci[0])
            upper.append(ci[1] - entry["estimate"])
        ax.errorbar(
            [x + shift for x in xs], points, yerr=[lower, upper], fmt=marker, color=color, ms=3.0,
            elinewidth=0.7, capsize=1.2, lw=0, markerfacecolor=color if filled else "white",
            markeredgewidth=0.8, zorder=4,
        )

    ax = axes[0]
    ax.axhline(0, color=muted, lw=0.6, zorder=1)
    series(ax, "suppress_E_full", blue, "o", -0.16, True)
    series(ax, "random_suppress_E_full", grey, "s", 0.05, False)
    series(ax, "random_kl_suppress_E_full", grey, "^", 0.24, False)
    ax.set_title(f"inserted text, dose {-MAIN_MAGNITUDE:+g}", pad=2)
    ax.set_ylabel("refusal2 removed")
    ax.legend(
        handles=[
            Line2D([], [], color=blue, marker="o", ms=3, markerfacecolor=blue, lw=0, label="own direction"),
            Line2D([], [], color=grey, marker="s", ms=3, markerfacecolor="white", lw=0, label="random, norm-matched"),
            Line2D([], [], color=grey, marker="^", ms=3, markerfacecolor="white", lw=0, label="random, KL-matched"),
        ],
        loc="upper left", frameon=False, handlelength=1.2, labelspacing=0.2, borderaxespad=0.1,
    )

    ax = axes[1]
    ax.axhline(0, color=muted, lw=0.6, zorder=1)
    series(ax, "induce_O_full", orange, "o", -0.10, True)
    series(ax, "random_induce_O_full", grey, "s", 0.10, False)
    ax.set_title(f"original text, dose {MAIN_MAGNITUDE:+g}", pad=2)
    ax.set_ylabel("refusal2 induced")
    ax.legend(
        handles=[
            Line2D([], [], color=orange, marker="o", ms=3, markerfacecolor=orange, lw=0, label="own direction"),
            Line2D([], [], color=grey, marker="s", ms=3, markerfacecolor="white", lw=0, label="random, norm-matched"),
        ],
        loc="upper left", frameon=False, handlelength=1.2, labelspacing=0.2, borderaxespad=0.1,
    )

    for ax in axes[:2]:
        ax.set_xticks(xs)
        ax.set_xticklabels([label for _, label in order], rotation=55, ha="right")
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.grid(axis="y", zorder=0)

    ax = axes[2]
    ax.axhline(0, color=muted, lw=0.6, zorder=1)
    magnitudes = [0.0, HALF_MAGNITUDE, MAIN_MAGNITUDE]
    cmap = plt.get_cmap("viridis")
    for index, (tag, label) in enumerate(order):
        values = [
            0.0,
            per_model[tag]["estimands"]["bidirectional_half"]["estimate"],
            per_model[tag]["estimands"]["bidirectional_full"]["estimate"],
        ]
        ax.plot(
            magnitudes, values, marker="o", ms=2.4, lw=0.9,
            color=cmap(index / max(len(order) - 1, 1)), label=label, zorder=3,
        )
    ax.set_title("dose response (bidirectional)", pad=2)
    ax.set_xlabel("dose (own boundary shifts)")
    ax.set_ylabel("mean of the two effects")
    ax.set_xticks(magnitudes)
    ax.grid(axis="y", zorder=0)
    ax.legend(loc="upper left", frameon=False, ncol=2, handlelength=1.1, labelspacing=0.2, columnspacing=0.9, borderaxespad=0.1)

    fig.tight_layout(pad=0.4, w_pad=0.8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def run_mode(
    mode: str,
    args: argparse.Namespace,
    cohort: dict,
    inputs: dict,
    exclusions: dict,
    tags: list[str],
    arms: dict,
    x114: dict | None = None,
    x114_exclusions: dict | None = None,
) -> None:
    out_path, figure = result_path_for(args.out, mode), result_path_for(args.figure, mode)
    per_model = {}
    for index, tag in enumerate(tags):
        keep = retained_ids([int(u["id"]) for u in cohort["units"]], x101_mode(mode), exclusions)
        sources = [{"source": "x101", "cohort": cohort, "arms": arms[tag], "keep_ids": keep}]
        if uses_x114(mode):
            sources.append({"source": "x114", "cohort": x114["cohort"], "arms": x114["arms"][tag], "keep_ids": None})
        per_model[tag] = analyze_tag(tag, sources, inputs, args.bootstrap, args.seed + 100 * index)
        report = per_model[tag]
        line = " ".join(
            f"{name}={report['primary'][name]['estimate']:+.3f}"
            f"[{report['primary'][name]['ci95'][0]:+.3f},{report['primary'][name]['ci95'][1]:+.3f}]"
            for name in PRIMARY_TESTS
        )
        print(f"x105 {mode:9s} {tag:8s} n={report['n_prompts']} {line} slope="
              f"{report['estimands']['oriented_dose_slope']['estimate']:+.3f} -> {report['verdict']}", flush=True)
    if len(per_model) > 1:
        draw_figure(per_model, figure)
    n_x101 = max(r["per_source"]["x101"]["n_prompts"] for r in per_model.values())
    n_x114 = max(r["per_source"].get("x114", {"n_prompts": 0})["n_prompts"] for r in per_model.values())
    output = {
        "experiment": "x105",
        "status": "complete",
        "pre_registered": False,
        "scope": (
            "the frozen x101 and x114 insertion cohorts, original (O) and end-inserted (E) texts; each "
            "model's own x88 consensus boundary direction written at the x88 hook block, doses "
            f"±{HALF_MAGNITUDE} and ±{MAIN_MAGNITUDE} of the model's own mean boundary shift; "
            "controls: two norm-matched orthogonal random directions and one KL-matched random "
            "direction; refusal2 only"
            + {
                "extended": (
                    "; the x101 units whose prompt and word are both outside the x85 cohort the directions "
                    "were fit on, pooled with the x114 units, which are outside it by construction"
                ),
                "x101_leak_free": "; the x101 units outside the x85 cohort only",
                "full": "; all 130 x101 units, no x114 units",
            }[mode]
        ),
        "cohort": cohort_record(mode, exclusions, x114_exclusions if uses_x114(mode) else None, n_x101, n_x114),
        "config": {
            "n_bootstrap": args.bootstrap,
            "bootstrap_seed": args.seed,
            "bootstrap_unit": "prompt, then the ten binary responses within each arm (x83's two-stage scheme)",
            "dose_magnitudes": list(DOSE_MAGNITUDES),
            "control_draws": list(CONTROL_DRAWS),
            "primary_tests": list(PRIMARY_TESTS),
            "holm_family": "the seven transfer models, per primary test",
            "verdict_rule": (
                "holds = all three primary 95% intervals exclude zero on the predicted side; "
                "partial = some; absent = none. A reading of the intervals, not a decision rule"
            ),
        },
        "inputs": {
            "frozen_inputs": file_record(args.inputs),
            "cohort": file_record(inputs["cohort"]["path"]) if Path(inputs["cohort"]["path"]).exists() else inputs["cohort"],
            "x114": {
                "cohort": file_record(x114_common.cohort_path()),
                "frozen_inputs": file_record(x114_common.inputs_path()),
            }
            if uses_x114(mode)
            else None,
        },
        "tags": tags,
        "transfer_models": [tag for tag in FRESH_TAGS if tag in per_model],
        "reference_model": IMPORTED_TAG,
        "arms": [dict(spec) for spec in ARMS],
        "base_arms": {c: BASE_ARMS[c] for c in TEXT_CONDITIONS},
        "cross_model": cross_model(per_model),
        # The base arms are x101's own rows, so the cross-check is a check only on x101's cohort.
        "x101_base_cross_check": x101_cross_check(per_model) if mode == "full" else None,
        "models": per_model,
        "figure": {"pdf": str(figure), "png": str(figure.with_suffix(".png"))} if len(per_model) > 1 else None,
    }
    atomic_json(out_path, output)
    print(f"x105 {mode}: result -> {out_path}" + (f"; figure -> {figure}" if len(per_model) > 1 else ""))


def main() -> None:
    args = parse_args()
    inputs = read_json(args.inputs)
    if inputs.get("experiment") != "x105" or inputs.get("status") != "frozen":
        raise SystemExit("FATAL: invalid x105 inputs")
    cohort = load_cohort()
    if cohort["cohort_sha256"] != inputs["cohort_sha256"]:
        raise SystemExit("FATAL: the cohort differs from the frozen inputs")
    exclusions = load_exclusions(cohort)
    tags = [tag for tag in ALL_TAGS if args.tags is None or tag in args.tags]
    arms = {tag: load_tag_arms(tag, cohort) for tag in tags}
    modes = MODES if args.cohort == "all" else (args.cohort,)
    x114 = x114_exclusions = None
    if any(uses_x114(mode) for mode in modes):
        x114_cohort = x114_common.load_cohort()
        x114_exclusions = x114_common.load_exclusions(x114_cohort)
        if x114_exclusions["counts"]["n_units_excluded"]:
            raise SystemExit("FATAL: the x85 exclusion rule drops x114 units; the top-up is not x85-free")
        x114 = {"cohort": x114_cohort, "arms": {tag: load_x114_arms(tag, x114_cohort) for tag in tags}}
    for mode in modes:
        run_mode(mode, args, cohort, inputs, exclusions, tags, arms, x114, x114_exclusions)


if __name__ == "__main__":
    main()
