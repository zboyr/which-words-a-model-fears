"""x101 Stage 4: x83's analysis per model, then one cross-model table and figure.

Per model (x83_analyze functions imported unchanged): primary delta_pos =
mean_i(r_pos,i - r_O,i) on that model's admitted prompts, two-stage paired
bootstrap, Holm over the three positions; secondary: ungated (all 130),
common-admitted, non-admitted, floor prompts, per-category, moderators,
run-era diagnostic against that model's historical labels. Qwen3.5-4B rows are
the x83 artifacts restricted to the shared cohort. Descriptive: no decision rule.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from x83_analyze import contrast, finite_correlation, holm, load_generations, load_judgments
from x101_common import (
    ALL_TAGS,
    CONDITIONS,
    DETECT_THRESHOLD,
    FRESH_TAGS,
    IMPORTED_TAG,
    MODELS,
    POSITIONS,
    QWEN4B_MODEL_ID,
    RESULTS,
    X83_RUN,
    atomic_json,
    cohort_path,
    detect_path,
    file_record,
    generations_path,
    labels_path,
    load_cohort,
    load_tag_historical_refusals,
    manifest_path,
    read_json,
    refusal_path,
)

N_BOOTSTRAP = 10_000
SEED = 42
MIN_CATEGORY_N = 20
MAX_DROPPED_PROMPTS = 2  # judge-failure tolerance per model; more than this aborts the analysis
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
    parser.add_argument("--out", default=str(RESULTS / "x101_cross_model_insertion.json"))
    parser.add_argument("--figure", default=str(RESULTS / "x101_cross_model_insertion.pdf"))
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def tag_paths(tag: str, cohort: dict) -> dict:
    if tag == IMPORTED_TAG:
        return {
            "cohort_sha256": cohort["x83_cohort_sha256"],
            "detect": X83_RUN / "x83_detect.json",
            "generations": {c: X83_RUN / f"x83_generations_{c}.json" for c in CONDITIONS},
            "refusal": {c: X83_RUN / f"x83_refusal_{c}.jsonl" for c in CONDITIONS},
            "model_id": QWEN4B_MODEL_ID,
        }
    return {
        "cohort_sha256": cohort["cohort_sha256"],
        "detect": detect_path(tag),
        "generations": {c: generations_path(tag, c) for c in CONDITIONS},
        "refusal": {c: refusal_path(tag, c) for c in CONDITIONS},
        "model_id": MODELS[tag]["model_id"],
    }


def unjudged_prompts(path: Path, condition: str, units: dict[int, dict]) -> list[int]:
    """Cohort prompts with no successful refusal2 record in one judgment file."""
    successful = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not row.get("failed") and row.get("condition") == condition and int(row["id"]) in units:
                successful.add(int(row["id"]))
    return sorted(set(units) - successful)


def analyze_tag(tag: str, cohort: dict, units: dict[int, dict], ids: list[int], n_boot: int, seed: int) -> dict:
    paths = tag_paths(tag, cohort)
    cohort_sha = paths["cohort_sha256"]
    detect = read_json(paths["detect"])
    if detect.get("cohort_sha256") != cohort_sha:
        raise SystemExit(f"FATAL: {tag} detection manifest belongs to another cohort")
    detect_rows = {int(r["id"]): r for r in detect["rows"]}
    if not set(ids) <= set(detect_rows):
        raise SystemExit(f"FATAL: {tag} detection rows do not cover the cohort")

    # A prompt whose judge call failed on every attempt in any condition is dropped from
    # this model in all four conditions, so every contrast stays paired on one prompt set.
    excluded = {c: unjudged_prompts(paths["refusal"][c], c, units) for c in CONDITIONS}
    dropped = sorted(set().union(*excluded.values()))
    if len(dropped) > MAX_DROPPED_PROMPTS:
        raise SystemExit(f"FATAL: {tag} lacks labels for {len(dropped)} prompts: {dropped[:10]}")
    units = {i: u for i, u in units.items() if i not in dropped}
    ids = [i for i in ids if i not in dropped]

    generations = {c: load_generations(paths["generations"][c], c, units, cohort_sha) for c in CONDITIONS}
    labels = {c: load_judgments(paths["refusal"][c], c, units, generations[c], cohort_sha) for c in CONDITIONS}
    arrays = {c: np.stack([labels[c][i] for i in ids]) for c in CONDITIONS}
    historical = load_tag_historical_refusals(tag, ids)
    hist = np.stack([np.asarray(historical[i], dtype=np.int8) for i in ids])
    p_detect = {p: np.asarray([detect_rows[i]["positions"][p]["p_detect"] for i in ids], dtype=float) for p in POSITIONS}
    admitted = {p: p_detect[p] > DETECT_THRESHOLD for p in POSITIONS}
    common = np.logical_and.reduce([admitted[p] for p in POSITIONS])
    categories = np.asarray([units[i]["category"] for i in ids])
    words = np.asarray([units[i]["word"] for i in ids])
    r_o = arrays["O"].mean(axis=1)

    primary = {}
    for k, p in enumerate(POSITIONS):
        mask = admitted[p]
        primary[p] = contrast(arrays["O"][mask], arrays[p][mask], n_boot, seed + 10 + k)
    present = [p for p in POSITIONS if primary[p]["n"] > 0]
    for p, adj in zip(present, holm([primary[p]["bootstrap_p_two_sided"] for p in present])):
        primary[p]["holm_adjusted_p"] = adj

    ungated = {p: contrast(arrays["O"], arrays[p], n_boot, seed + 40 + k) for k, p in enumerate(POSITIONS)}
    for p, adj in zip(POSITIONS, holm([ungated[p]["bootstrap_p_two_sided"] for p in POSITIONS])):
        ungated[p]["holm_adjusted_p"] = adj
    common_reports = {p: contrast(arrays["O"][common], arrays[p][common], n_boot, seed + 20 + k) for k, p in enumerate(POSITIONS)}
    pairwise = {}
    if common.sum() > 0:
        for k, (a, b) in enumerate((("S", "M"), ("S", "E"), ("M", "E"))):
            pairwise[f"{b}_minus_{a}"] = contrast(arrays[a][common], arrays[b][common], n_boot, seed + 30 + k)
    non_admitted = {
        p: contrast(arrays["O"][~admitted[p]], arrays[p][~admitted[p]], n_boot, seed + 50 + k) for k, p in enumerate(POSITIONS)
    }
    run_era = contrast(hist, arrays["O"], n_boot, seed + 60)

    per_category = {}
    for k, p in enumerate(POSITIONS):
        reports = []
        for j, category in enumerate(sorted(set(categories))):
            mask = admitted[p] & (categories == category)
            if mask.sum() < MIN_CATEGORY_N:
                continue
            reports.append({"category": category, **contrast(arrays["O"][mask], arrays[p][mask], n_boot, seed + 100 + 10 * k + j)})
        for r, adj in zip(reports, holm([r["bootstrap_p_two_sided"] for r in reports]) if reports else []):
            r["holm_adjusted_p"] = adj
        per_category[p] = reports

    floor = r_o == 0
    floor_prompts = {"n_floor": int(floor.sum())}
    for p in POSITIONS:
        r_p = arrays[p].mean(axis=1)[floor]
        floor_prompts[p] = {
            "pushed_to_half_or_more": int((r_p >= 0.5).sum()),
            "mean_r_pos": float(r_p.mean()) if floor.any() else None,
        }

    per_word = {}
    for p in POSITIONS:
        groups = defaultdict(list)
        for idx in np.flatnonzero(admitted[p]):
            groups[str(words[idx])].append(float(arrays[p][idx].mean() - r_o[idx]))
        per_word[p] = {w: {"n": len(v), "mean_delta": float(np.mean(v))} for w, v in sorted(groups.items())}

    moderators = {}
    for p in POSITIONS:
        delta = arrays[p].mean(axis=1) - r_o
        moderators[p] = {
            "p_detect_vs_delta_all_prompts": finite_correlation(p_detect[p], delta),
            "historical_rO_vs_delta_admitted": finite_correlation(hist.mean(axis=1)[admitted[p]], delta[admitted[p]]),
        }

    integrity = {
        c: {
            "rows": len(rows),
            "retry_rows": sum(bool(r.get("used_retry")) for r in rows.values()),
            "truncated_after_retry": int(sum(sum(bool(v) for v in r.get("truncated", [])) for r in rows.values())),
            "mean_gen_tokens": float(np.mean([np.mean(r["gen_tokens"]) for r in rows.values()])),
        }
        for c, rows in generations.items()
    }
    return {
        "model_tag": tag,
        "model_id": paths["model_id"],
        "model_source": detect.get("model_source"),
        "imported_from_x83": tag == IMPORTED_TAG,
        "gate_scoring_offset": detect.get("config", {}).get("scoring_offset", 0),
        "inputs": {
            "detect": file_record(paths["detect"]),
            "historical_labels": file_record(labels_path(tag)),
            **{f"generation_{c}": file_record(paths["generations"][c]) for c in CONDITIONS},
            **{f"refusal_{c}": file_record(paths["refusal"][c]) for c in CONDITIONS},
        },
        "counts": {
            "cohort": len(ids),
            "dropped_unjudged_prompts": dropped,
            "dropped_in_condition": {c: v for c, v in excluded.items() if v},
            "admitted": {p: int(admitted[p].sum()) for p in POSITIONS},
            "admitted_all_three": int(common.sum()),
            "p_detect_quantiles": {p: [float(v) for v in np.quantile(p_detect[p], (0.1, 0.25, 0.5, 0.75, 0.9))] for p in POSITIONS},
            "greedy_match_rate": {p: float(np.mean([detect_rows[i]["positions"][p]["greedy_matches"] for i in ids])) for p in POSITIONS},
            "words_admitted": {p: int(len(set(words[admitted[p]]))) for p in POSITIONS},
        },
        "condition_means_all_prompts": {c: float(arrays[c].mean()) for c in CONDITIONS},
        "historical_rO_mean": float(hist.mean()),
        "generation_integrity": integrity,
        "primary_admitted_delta_vs_O": primary,
        "secondary": {
            "ungated_all_prompts_delta_vs_O": ungated,
            "common_cohort_delta_vs_O": common_reports,
            "common_cohort_position_differences": pairwise,
            "non_admitted_delta_vs_O": non_admitted,
            "run_era_diagnostic_O_new_minus_historical": run_era,
            "floor_prompts_regenerated_O_zero": floor_prompts,
            "per_category_admitted": per_category,
            "per_word_admitted_descriptive": per_word,
            "moderators": moderators,
        },
    }


def cross_model_table(per_model: dict) -> dict:
    gated_rows, ungated_rows = [], []
    for tag in ALL_TAGS:
        report = per_model[tag]
        for p in POSITIONS:
            g = report["primary_admitted_delta_vs_O"][p]
            gated_rows.append(
                {
                    "model_tag": tag,
                    "position": p,
                    "n_admitted": g["n"],
                    "mean_rO": g.get("mean_base"),
                    "mean_r_pos": g.get("mean_treated"),
                    "delta": g.get("estimate"),
                    "ci95": g.get("ci95"),
                    "p_two_sided": g.get("bootstrap_p_two_sided"),
                    "holm_within_model_p": g.get("holm_adjusted_p"),
                }
            )
            u = report["secondary"]["ungated_all_prompts_delta_vs_O"][p]
            ungated_rows.append(
                {
                    "model_tag": tag,
                    "position": p,
                    "n": u["n"],
                    "mean_rO": u["mean_base"],
                    "mean_r_pos": u["mean_treated"],
                    "delta": u["estimate"],
                    "ci95": u["ci95"],
                    "p_two_sided": u["bootstrap_p_two_sided"],
                    "holm_within_model_p": u["holm_adjusted_p"],
                }
            )
    tested = [row for row in gated_rows if row["n_admitted"] > 0]
    for row, adj in zip(tested, holm([row["p_two_sided"] for row in tested])):
        row["holm_all_tests_p"] = adj
    for row, adj in zip(ungated_rows, holm([row["p_two_sided"] for row in ungated_rows])):
        row["holm_all_tests_p"] = adj
    return {
        "gated_admitted": gated_rows,
        "ungated_all_130": ungated_rows,
        "n_gated_tests_in_family": len(tested),
    }


def draw_figure(per_model: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 6.5,
            "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.spines.top": False,
            "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
            "ytick.major.width": 0.6, "axes.edgecolor": "#333333", "xtick.color": "#333333",
            "ytick.color": "#333333", "text.color": "#333333", "axes.labelcolor": "#333333",
            "grid.color": "#DDDDDD", "grid.linewidth": 0.5,
        }
    )
    blue, orange, muted = "#0072B2", "#E69F00", "#888888"
    families = [range(0, 5), range(5, 6), range(6, 7), range(7, 8)]
    xs = list(range(len(FIGURE_ORDER)))
    titles = {"S": "start", "M": "middle", "E": "end"}
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.1), sharey=True)
    for ax, p in zip(axes, POSITIONS):
        ax.axhline(0, color=muted, lw=0.6, zorder=1)
        series = (
            ("primary_admitted_delta_vs_O", blue, "o", True, 0.0),
            ("ungated", orange, "s", False, 0.18),
        )
        for key, color, marker, filled, shift in series:
            vals = []
            for tag, _ in FIGURE_ORDER:
                r = per_model[tag]
                e = r["primary_admitted_delta_vs_O"][p] if key != "ungated" else r["secondary"]["ungated_all_prompts_delta_vs_O"][p]
                if e.get("n", 0) == 0:
                    vals.append(None)
                else:
                    vals.append((e["estimate"], e["estimate"] - e["ci95"][0], e["ci95"][1] - e["estimate"]))
            for fam in families:
                fx = [x for x in fam if vals[x] is not None]
                ax.plot([x + shift for x in fx], [vals[x][0] for x in fx], color=color, lw=1.0 if len(fx) > 1 else 0, zorder=3)
            px = [x + shift for x in xs if vals[x] is not None]
            pv = [v for v in vals if v is not None]
            ax.errorbar(
                px, [v[0] for v in pv], yerr=[[v[1] for v in pv], [v[2] for v in pv]], fmt=marker, color=color,
                ms=2.8, elinewidth=0.6, capsize=1.0, lw=0, markerfacecolor=color if filled else "white",
                markeredgewidth=0.8, zorder=4,
            )
        for x, (tag, _) in zip(xs, FIGURE_ORDER):
            n = per_model[tag]["primary_admitted_delta_vs_O"][p]["n"]
            ax.text(x, 0.02, f"{n}", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=5, color=blue)
        ax.set_title(f"{titles[p]} insertion", pad=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([label for _, label in FIGURE_ORDER], rotation=55, ha="right")
        ax.grid(axis="y", zorder=0)
        ax.set_xlim(-0.6, 7.8)
    axes[0].set_ylabel("refusal2 change vs. O")
    axes[0].legend(
        handles=[
            Line2D([], [], color=blue, marker="o", ms=3, markerfacecolor=blue, lw=1.0, label="gated (n admitted shown)"),
            Line2D([], [], color=orange, marker="s", ms=3, markerfacecolor="white", lw=1.0, label="ungated (130 prompts)"),
        ],
        loc="upper left", frameon=False, handlelength=1.3, labelspacing=0.25, borderaxespad=0.1,
    )
    fig.tight_layout(pad=0.4, w_pad=0.6)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cohort = load_cohort()
    units = {int(u["id"]): u for u in cohort["units"]}
    ids = sorted(units)
    per_model = {}
    for t, tag in enumerate(ALL_TAGS):
        per_model[tag] = analyze_tag(tag, cohort, units, ids, args.bootstrap, args.seed + 1000 * t)
        for p in POSITIONS:
            r = per_model[tag]["primary_admitted_delta_vs_O"][p]
            u = per_model[tag]["secondary"]["ungated_all_prompts_delta_vs_O"][p]
            gated = (
                f"n={r['n']} delta={r['estimate']:+.3f} [{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}] holm={r['holm_adjusted_p']:.4f}"
                if r["n"] else "n=0"
            )
            print(f"x101 {tag:8s} {p}: gated {gated}; ungated {u['estimate']:+.3f} [{u['ci95'][0]:+.3f},{u['ci95'][1]:+.3f}]")
    table = cross_model_table(per_model)
    figure = Path(args.figure)
    draw_figure(per_model, figure)
    output = {
        "experiment": "x101",
        "status": "complete",
        "pre_registered": False,
        "scope": (
            "130 x83 prompts with historical refusal2 <= 0.8 in all eight transfer models; one inserted "
            "harm word at S/M/E vs contemporaneously regenerated O; refusal2 only"
        ),
        "config": {
            "n_bootstrap": args.bootstrap,
            "seed": args.seed,
            "seed_rule": "x83 offsets within a model, plus 1000 x tag index",
            "detect_threshold": DETECT_THRESHOLD,
            "holm_within_model_family": list(POSITIONS),
            "holm_all_tests_family": "every model x position with n_admitted > 0 (gated) / all 24 (ungated)",
            "bootstrap_unit": "prompt then ten binary responses independently within condition",
            "min_category_n": MIN_CATEGORY_N,
        },
        "inputs": {
            "cohort": file_record(cohort_path()),
            "manifest": file_record(manifest_path()) if manifest_path().exists() else None,
        },
        "tags": list(ALL_TAGS),
        "fresh_tags": list(FRESH_TAGS),
        "imported_tag": IMPORTED_TAG,
        "cross_model": table,
        "models": per_model,
        "figure": {"pdf": str(figure), "png": str(figure.with_suffix(".png"))},
    }
    atomic_json(args.out, output)
    print(f"result -> {args.out}; figure -> {figure}")


if __name__ == "__main__":
    main()
