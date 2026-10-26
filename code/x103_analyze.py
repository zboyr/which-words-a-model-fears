"""x103 Stage 2: per model, does the projection of the insertion-induced boundary shift onto
the model's own boundary direction track the x101 refusal2 increase across prompts?

Over the (prompt, position) cells of one model: primary Spearman rho between proj and
delta refusal2 after position fixed effects, permutation p (delta refusal2 permuted
across prompts within position), prompt-level bootstrap CI, Holm over the seven fresh
models. Controls: partial Spearman given the shift norm, a random-direction null for the
same statistic, prompt + position fixed effects, per-position raw Spearman, and the
cells whose prompt and word both lie outside the x85 cohort the directions were fit on.
Qwen3.5-4B is the reference row (x87/x86 consensus direction, x83 rows), outside Holm.

Three cohort modes, same seeds, permutations, bootstrap and Holm in each
(`x114_cohorts`). `extended` (the canonical result files) pools the 64 x101 units whose
prompt and inserted word are both outside the x85 cohort the directions were fit on, as
frozen in `data/runs/x105/x105_x85_exclusions.json`, with the x114 units, which the same
rule leaves intact; `x101_leak_free` runs those 64 x101 units alone into the `_x101only`
siblings; `full` runs all 130 x101 units into the `_full` siblings. The default runs all
three (`--cohort all`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import stats

from x103_common import (
    ALL_TAGS,
    ALPHA,
    BOUNDARY_POSITIONS,
    FRESH_TAGS,
    IMPORTED_TAG,
    N_BOOTSTRAP,
    N_PERMUTATIONS,
    N_RANDOM_DIRECTIONS,
    POSITIONS,
    RESULTS,
    X85_COHORT,
    atomic_json,
    deltas_path,
    file_record,
    holm,
    inputs_path,
    leakage_free_mask,
    leakage_sets,
    load_cohort,
    load_refusal_arrays,
    partial_spearman,
    permutation_p,
    primary_rho,
    projections_onto,
    projections_path,
    prompt_bootstrap,
    random_unit_directions,
    read_json,
    residualize,
    sha256_file,
    spearman,
    stable_seed,
)
from x105_common import load_exclusions, retained_ids
from x114_cohorts import MODES, cohort_record, pool_cells, result_path_for, uses_x114, x101_mode
import x114_common

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
    parser.add_argument("--out", default=str(RESULTS / "x103_direction_insertion_projection.json"))
    parser.add_argument("--figure", default=str(RESULTS / "x103_direction_insertion_projection.pdf"))
    parser.add_argument("--random", type=int, default=N_RANDOM_DIRECTIONS)
    parser.add_argument("--permutations", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument(
        "--cohort", choices=("all", *MODES), default="all",
        help=(
            "extended (x101 leak-free + x114) writes the canonical files, x101_leak_free the "
            "`_x101only` siblings, full the `_full` siblings; all (default) writes all three"
        ),
    )
    return parser.parse_args()


def restrict_cells(cells: dict, keep_ids: list[int]) -> dict:
    """The cells of the retained prompts, with the prompt index re-based for the bootstrap."""
    keep = set(keep_ids)
    mask = np.asarray([i in keep for i in cells["cell_ids"]], dtype=bool)
    ids = [i for i in cells["ids"] if i in keep]
    new_index = {old: new for new, old in enumerate(i for i, prompt_id in enumerate(cells["ids"]) if prompt_id in keep)}
    out = {k: v[mask] for k, v in cells.items() if k in ("proj", "shift", "dr", "pos_idx", "deltas")}
    out["prompt_idx"] = np.asarray([new_index[int(old)] for old in cells["prompt_idx"][mask]])
    out["cell_ids"] = [i for i, m in zip(cells["cell_ids"], mask) if m]
    out["cell_words"] = [w for w, m in zip(cells["cell_words"], mask) if m]
    out["proj_token"] = {p: v[mask] for p, v in cells["proj_token"].items()}
    out["ids"], out["dropped"] = ids, cells["dropped"]
    out["r_o"] = cells["r_o"][[prompt_id in keep for prompt_id in cells["ids"]]]
    file = dict(cells["file"])
    file["rows"] = [r for r in cells["file"]["rows"] if int(r["id"]) in keep]
    out["file"] = file
    out["source"] = cells.get("source", "x101")
    return out


def load_cells(
    tag: str,
    cohort: dict,
    *,
    source: str = "x101",
    experiment: str = "x103",
    projections=projections_path,
    deltas_file=deltas_path,
    refusals=load_refusal_arrays,
) -> dict:
    """Align one run's extracted cells with the per-prompt refusal2 rows of the same run.

    `source="x101"` reads the x103 projections against the x101 behaviour; `source="x114"`
    reads the x114 projections against the x114 behaviour. The layouts are identical, so
    `x114_cohorts.pool_cells` can concatenate them."""
    proj_file = read_json(projections(tag))
    if proj_file.get("experiment") != experiment or proj_file.get("cohort_sha256") != cohort["cohort_sha256"]:
        raise SystemExit(f"FATAL: {tag}: {source} projections belong to another cohort")
    if proj_file["site_audit"]["status"] != "pass":
        raise SystemExit(f"FATAL: {tag}: {source} site audit did not pass")
    if sha256_file(deltas_file(tag)) != proj_file["deltas"]["sha256"]:
        raise SystemExit(f"FATAL: {tag}: {source} delta artifact differs from the one the projections recorded")
    with np.load(deltas_file(tag)) as artifact:
        deltas = artifact["deltas"].astype(np.float32)  # [n, 3, 2, hidden]
        delta_ids = [int(v) for v in artifact["ids"]]
        if list(artifact["positions"]) != list(POSITIONS) or list(artifact["boundary_positions"]) != list(BOUNDARY_POSITIONS):
            raise SystemExit(f"FATAL: {tag}: delta axes are not in the frozen order")
    rows = {(int(r["id"]), r["position"]): r for r in proj_file["rows"]}
    ids, arrays, dropped = refusals(tag, cohort)
    units = {int(u["id"]): u for u in cohort["units"]}
    delta_index = {i: k for k, i in enumerate(delta_ids)}
    r_o = arrays["O"].mean(axis=1)
    cells: dict[str, list] = {k: [] for k in ("proj", "shift", "dr", "prompt_idx", "pos_idx", "cell_ids", "cell_words", "deltas")}
    proj_token = {p: [] for p in BOUNDARY_POSITIONS}
    for i, prompt_id in enumerate(ids):
        for k, position in enumerate(POSITIONS):
            row = rows[(prompt_id, position)]
            cells["proj"].append(row["proj"])
            cells["shift"].append(row["shift"])
            for p in BOUNDARY_POSITIONS:
                proj_token[p].append(row["proj_by_token"][p])
            cells["dr"].append(float(arrays[position][i].mean() - r_o[i]))
            cells["prompt_idx"].append(i)
            cells["pos_idx"].append(k)
            cells["cell_ids"].append(prompt_id)
            cells["cell_words"].append(units[prompt_id]["word"])
            cells["deltas"].append(deltas[delta_index[prompt_id], k])
    out = {k: np.asarray(v) for k, v in cells.items() if k not in ("cell_ids", "cell_words", "deltas")}
    out["deltas"] = np.stack(cells["deltas"])  # [cells, 2, hidden]
    out["cell_ids"], out["cell_words"] = cells["cell_ids"], cells["cell_words"]
    out["proj_token"] = {p: np.asarray(v) for p, v in proj_token.items()}
    out.update({"file": proj_file, "ids": ids, "dropped": dropped, "r_o": r_o, "source": source})
    return out


def load_x114_cells(tag: str) -> dict:
    """The x114 top-up cells: its own projections, deltas, cohort and refusal2 rows."""
    return load_cells(
        tag,
        x114_common.load_cohort(),
        source="x114",
        experiment="x114",
        projections=x114_common.projections_path,
        deltas_file=x114_common.deltas_path,
        refusals=x114_common.load_refusal_arrays,
    )


def analyze_tag(
    tag: str,
    cells: dict,
    x85_prompts: set,
    x85_words: set,
    n_random: int,
    n_perm: int,
    n_boot: int,
    leakage_control: bool = True,
) -> dict:
    """One model's report. `leakage_control=False` on a run whose cells are already the
    leakage-free subset, where the subset control would repeat the primary statistic."""
    proj, shift, dr = cells["proj"], cells["shift"], cells["dr"]
    prompt_idx, pos_idx = cells["prompt_idx"], cells["pos_idx"]
    xr, yr, zr = residualize(proj, pos_idx), residualize(dr, pos_idx), residualize(shift, pos_idx)

    perm = permutation_p(proj, dr, pos_idx, n_perm, stable_seed(tag, "perm"))
    boot = prompt_bootstrap(proj, dr, shift, prompt_idx, pos_idx, n_boot, stable_seed(tag, "boot"))
    partial = partial_spearman(xr, yr, zr)
    two_way = spearman(residualize(proj, prompt_idx, pos_idx), residualize(dr, prompt_idx, pos_idx))

    dim = cells["deltas"].shape[-1]
    random_dirs = random_unit_directions(dim, n_random, stable_seed(tag, "random"))
    random_proj = projections_onto(cells["deltas"], random_dirs)  # [cells, n_random]
    random_rho = np.asarray([primary_rho(random_proj[:, k], dr, pos_idx) for k in range(n_random)])
    percentile = float(100.0 * np.mean(random_rho < perm["rho"]))

    per_position = {}
    for k, position in enumerate(POSITIONS):
        mask = pos_idx == k
        res = stats.spearmanr(proj[mask], dr[mask])
        per_position[position] = {
            "n": int(mask.sum()),
            "rho": float(res.statistic),
            "p": float(res.pvalue),
            "mean_proj": float(proj[mask].mean()),
            "mean_shift": float(shift[mask].mean()),
            "mean_delta_refusal": float(dr[mask].mean()),
            "rho_shift_vs_delta_refusal": float(stats.spearmanr(shift[mask], dr[mask]).statistic),
        }

    free = leakage_free_mask(cells["cell_ids"], cells["cell_words"], x85_prompts, x85_words)
    leakage_free_subset = None
    if leakage_control:
        free_perm = permutation_p(proj[free], dr[free], pos_idx[free], n_perm, stable_seed(tag, "perm_free"))
        free_prompts = np.unique(prompt_idx[free], return_inverse=True)[1]
        free_boot = prompt_bootstrap(proj[free], dr[free], shift[free], free_prompts, pos_idx[free], n_boot, stable_seed(tag, "boot_free"))
        leakage_free_subset = {
            "n_cells": int(free.sum()),
            "rho": free_perm["rho"],
            "ci95_prompt_bootstrap": free_boot["primary_ci95"],
            "permutation_p_two_sided": free_perm["p_two_sided"],
            "partial_shift_rho_ci95": free_boot["partial_shift_ci95"],
        }
    prompt_in_x85 = np.asarray([i in x85_prompts for i in cells["cell_ids"]])
    word_in_x85 = np.asarray([w in x85_words for w in cells["cell_words"]])

    token_rho = {p: spearman(residualize(cells["proj_token"][p], pos_idx), yr) for p in BOUNDARY_POSITIONS}
    audit = cells["file"]["site_audit"]
    return {
        "model_tag": tag,
        "model_id": cells["file"]["model_id"],
        "model_source": cells["file"]["model_source"],
        "site": cells["file"]["site"],
        "directions": cells["file"]["directions"],
        "site_audit": {
            "status": audit["status"],
            "reader_equals_output_hidden_states": all(t["reader_equals_output_hidden_states"] for t in audit["texts"]),
            "write_shift_within_bf16_ulp": all(t["edit_minus_base_equals_dose_mu_within_bf16_ulp"] for t in audit["texts"]),
            "min_cosine_of_write_shift_to_mu": min(min(t["edit_minus_base_cosine_to_mu"]) for t in audit["texts"]),
        },
        "rendering": cells["file"]["rendering"],
        "counts": {
            "n_prompts": len(cells["ids"]),
            "n_cells": int(len(proj)),
            "dropped_unjudged_prompts": cells["dropped"],
            "n_cells_prompt_in_x85": int(prompt_in_x85.sum()),
            "n_cells_word_in_x85": int(word_in_x85.sum()),
            "n_cells_leakage_free": int(free.sum()),
        },
        "descriptive": {
            "mean_r_O": float(cells["r_o"].mean()),
            "per_position": per_position,
            "rho_proj_vs_shift_position_fe": spearman(xr, zr),
            "rho_shift_vs_delta_refusal_position_fe": spearman(zr, yr),
            "raw_rho_all_cells": spearman(proj, dr),
            "rho_by_boundary_token_position_fe": token_rho,
            "mean_cosine_by_token": {
                p: float(np.mean([r["cosine_by_token"][p] for r in cells["file"]["rows"]])) for p in BOUNDARY_POSITIONS
            },
        },
        "primary": {
            "statistic": "Spearman rho of proj vs delta refusal2, both residualised on position fixed effects",
            "rho": perm["rho"],
            "ci95_prompt_bootstrap": boot["primary_ci95"],
            "permutation_p_two_sided": perm["p_two_sided"],
            "n_permutations": perm["n_permutations"],
            "n_bootstrap": boot["n_bootstrap"],
        },
        "controls": {
            "partial_shift": {"rho": partial, "ci95_prompt_bootstrap": boot["partial_shift_ci95"]},
            "random_directions": {
                "n": int(n_random),
                "percentile_of_real_rho": percentile,
                "band_2_5_97_5": [float(np.quantile(random_rho, 0.025)), float(np.quantile(random_rho, 0.975))],
                "mean": float(random_rho.mean()),
                "sd": float(random_rho.std()),
                "max": float(random_rho.max()),
                "p_one_sided": float((np.sum(random_rho >= perm["rho"]) + 1) / (n_random + 1)),
            },
            "prompt_and_position_fe": {"rho": two_way, "ci95_prompt_bootstrap": boot["two_way_fe_ci95"]},
            "per_position_raw": per_position,
            "leakage_free_subset": leakage_free_subset,
        },
        "verdict": {
            "rho_positive": perm["rho"] > 0,
            "significant_unadjusted": bool(perm["p_two_sided"] < ALPHA),
            "above_random_directions_97_5": bool(percentile >= 97.5),
            "above_shift_norm_control": bool(boot["partial_shift_ci95"][0] > 0 and partial > 0),
        },
    }


def finish_verdicts(per_model: dict) -> None:
    """Holm over the seven fresh models; the reference row keeps its unadjusted test."""
    fresh = [t for t in FRESH_TAGS if t in per_model]
    for tag, p in zip(fresh, holm([per_model[t]["primary"]["permutation_p_two_sided"] for t in fresh])):
        per_model[tag]["primary"]["holm_p_over_seven_models"] = float(p)
        per_model[tag]["verdict"]["significant_holm"] = bool(p < ALPHA)
    if IMPORTED_TAG in per_model:
        ref = per_model[IMPORTED_TAG]
        ref["primary"]["holm_p_over_seven_models"] = None
        ref["verdict"]["significant_holm"] = ref["verdict"]["significant_unadjusted"]
    for report in per_model.values():
        v = report["verdict"]
        v["direction_specific"] = bool(
            v["rho_positive"] and v["significant_holm"] and v["above_random_directions_97_5"] and v["above_shift_norm_control"]
        )


def cross_model_table(per_model: dict) -> list[dict]:
    rows = []
    for tag in ALL_TAGS:
        if tag not in per_model:
            continue
        r = per_model[tag]
        c = r["controls"]
        rows.append(
            {
                "model_tag": tag,
                "n_cells": r["counts"]["n_cells"],
                "rho": r["primary"]["rho"],
                "ci95": r["primary"]["ci95_prompt_bootstrap"],
                "permutation_p": r["primary"]["permutation_p_two_sided"],
                "holm_p": r["primary"]["holm_p_over_seven_models"],
                "partial_shift_rho": c["partial_shift"]["rho"],
                "partial_shift_ci95": c["partial_shift"]["ci95_prompt_bootstrap"],
                "random_percentile": c["random_directions"]["percentile_of_real_rho"],
                "random_band": c["random_directions"]["band_2_5_97_5"],
                "prompt_position_fe_rho": c["prompt_and_position_fe"]["rho"],
                "leakage_free_n": c["leakage_free_subset"]["n_cells"] if c["leakage_free_subset"] else None,
                "leakage_free_rho": c["leakage_free_subset"]["rho"] if c["leakage_free_subset"] else None,
                "leakage_free_p": c["leakage_free_subset"]["permutation_p_two_sided"] if c["leakage_free_subset"] else None,
                "direction_specific": r["verdict"]["direction_specific"],
            }
        )
    return rows


def draw_figure(per_model: dict, path: Path) -> None:
    """Per-model primary rho with its bootstrap interval, over the random-direction band;
    the shift-controlled partial rho beside it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update(
        {
            "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
            "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.spines.top": False,
            "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
            "ytick.major.width": 0.6, "axes.edgecolor": "#333333", "xtick.color": "#333333",
            "ytick.color": "#333333", "text.color": "#333333", "axes.labelcolor": "#333333",
            "grid.color": "#DDDDDD", "grid.linewidth": 0.5,
        }
    )
    blue, orange, muted, band = "#0072B2", "#E69F00", "#888888", "#DCDCDC"
    order = [(tag, label) for tag, label in FIGURE_ORDER if tag in per_model]
    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    ax.axhline(0, color=muted, lw=0.6, zorder=1)
    for x, (tag, _) in enumerate(order):
        lo, hi = per_model[tag]["controls"]["random_directions"]["band_2_5_97_5"]
        ax.add_patch(plt.Rectangle((x - 0.42, lo), 0.84, hi - lo, facecolor=band, edgecolor="none", zorder=2))
    for key, color, marker, shift, filled in (
        ("primary", blue, "o", -0.10, True),
        ("partial", orange, "s", 0.10, False),
    ):
        xs, points, lower, upper = [], [], [], []
        for x, (tag, _) in enumerate(order):
            report = per_model[tag]
            node = report["primary"] if key == "primary" else report["controls"]["partial_shift"]
            ci = node["ci95_prompt_bootstrap"]
            xs.append(x + shift)
            points.append(node["rho"])
            lower.append(node["rho"] - ci[0])
            upper.append(ci[1] - node["rho"])
        ax.errorbar(
            xs, points, yerr=[lower, upper], fmt=marker, color=color, ms=3.0, elinewidth=0.7,
            capsize=1.2, lw=0, markerfacecolor=color if filled else "white", markeredgewidth=0.8, zorder=4,
        )
    for x, (tag, _) in enumerate(order):
        if tag == IMPORTED_TAG:
            ax.axvspan(x - 0.5, x + 0.5, color="#F5F5F5", zorder=0)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([label for _, label in order])
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylabel("Spearman rho (position FE)")
    ax.grid(axis="y", zorder=0)
    ax.legend(
        handles=[
            Line2D([], [], color=blue, marker="o", ms=3, markerfacecolor=blue, lw=0, label="projection vs refusal2 rise"),
            Line2D([], [], color=orange, marker="s", ms=3, markerfacecolor="white", lw=0, label="controlling for shift size"),
            Patch(facecolor=band, label="random directions, 2.5-97.5%"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False, handlelength=1.3,
        labelspacing=0.25, borderaxespad=0.0, ncol=3, columnspacing=1.2,
    )
    ax.text(
        [i for i, (tag, _) in enumerate(order) if tag == IMPORTED_TAG][0], 0.985, "home model",
        transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6, color=muted,
    )
    fig.tight_layout(pad=0.4)
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
    all_cells: dict,
    x114: dict | None = None,
    x114_exclusions: dict | None = None,
) -> None:
    x85_prompts, x85_words = leakage_sets(read_json(X85_COHORT))
    out_path, figure = result_path_for(args.out, mode), result_path_for(args.figure, mode)
    per_model = {}
    for tag in ALL_TAGS:
        parts = [restrict_cells(all_cells[tag], retained_ids(all_cells[tag]["ids"], x101_mode(mode), exclusions))]
        if uses_x114(mode):
            parts.append(x114["cells"][tag])
        cells = pool_cells(parts)
        per_model[tag] = analyze_tag(
            tag, cells, x85_prompts, x85_words, args.random, args.permutations, args.bootstrap,
            leakage_control=(mode == "full"),
        )
        per_model[tag]["counts"]["n_prompts_by_source"] = cells.get(
            "n_by_source", {parts[0]["source"]: len(parts[0]["ids"])}
        )
        if uses_x114(mode):
            x114_file = cells["file_by_source"]["x114"]
            per_model[tag]["x114"] = {
                "projections": file_record(x114_common.projections_path(tag)),
                "site": x114_file["site"],
                "site_audit_status": x114_file["site_audit"]["status"],
                "rendering": x114_file["rendering"],
                "directions_sha256": x114_file["directions"]["sha256"],
                "n_prompts": len(parts[-1]["ids"]),
                "dropped_unjudged_prompts": parts[-1]["dropped"],
            }
        r = per_model[tag]
        free = r["controls"]["leakage_free_subset"]
        print(
            f"x103 {mode:9s} {tag:8s} n={r['counts']['n_cells']:4d} rho={r['primary']['rho']:+.3f} "
            f"[{r['primary']['ci95_prompt_bootstrap'][0]:+.3f},{r['primary']['ci95_prompt_bootstrap'][1]:+.3f}] "
            f"perm_p={r['primary']['permutation_p_two_sided']:.4f} "
            f"partial={r['controls']['partial_shift']['rho']:+.3f} "
            f"rand_pct={r['controls']['random_directions']['percentile_of_real_rho']:.1f} "
            f"twoway={r['controls']['prompt_and_position_fe']['rho']:+.3f}"
            + (f" free={free['rho']:+.3f}" if free else ""),
            flush=True,
        )
    finish_verdicts(per_model)
    draw_figure(per_model, figure)
    n_x101 = max(r["counts"]["n_prompts_by_source"].get("x101", 0) for r in per_model.values())
    n_x114 = max(r["counts"]["n_prompts_by_source"].get("x114", 0) for r in per_model.values())
    output = {
        "experiment": "x103",
        "status": "complete",
        "pre_registered": False,
        "scope": (
            "the x101 and x114 insertion cohorts in eight models; forward passes only on their own texts at "
            "the x88 boundary-write site; projection of the insertion-induced boundary shift onto each model's "
            "own consensus boundary direction against that model's refusal2 rise"
            + {
                "extended": (
                    "; the x101 units whose prompt and word are both outside the x85 cohort the directions were "
                    "fit on, pooled with the x114 units, which are outside it by construction"
                ),
                "x101_leak_free": "; the x101 units outside the x85 cohort only",
                "full": "; all 130 x101 units, no x114 units",
            }[mode]
        ),
        "cohort": cohort_record(mode, exclusions, x114_exclusions if uses_x114(mode) else None, n_x101, n_x114),
        "config": {
            "primary": "Spearman rho after position fixed effects on both variables",
            "n_permutations": args.permutations,
            "permutation_unit": "delta refusal2 permuted across prompts within position",
            "n_bootstrap": args.bootstrap,
            "bootstrap_unit": "prompts, with all three positions of a drawn prompt",
            "n_random_directions": args.random,
            "random_direction_rule": "Gaussian unit vector per boundary position, matched dimension",
            "holm_family": list(FRESH_TAGS),
            "reference_row": IMPORTED_TAG,
            "alpha": ALPHA,
            "seed_rule": "sha256 of x103|103|<tag>|<purpose>",
        },
        "inputs": {
            "frozen_inputs": file_record(inputs_path()),
            "cohort": inputs["cohort"],
            "x101_manifest": inputs["x101_manifest"],
            "x88_manifest": inputs["x88_manifest"],
            "projections": {tag: file_record(projections_path(tag)) for tag in ALL_TAGS},
            "x114": {
                "cohort": file_record(x114_common.cohort_path()),
                "frozen_inputs": file_record(x114_common.inputs_path()),
                "projections": {tag: file_record(x114_common.projections_path(tag)) for tag in ALL_TAGS},
            }
            if uses_x114(mode)
            else None,
        },
        "leakage": inputs["leakage"],
        "cross_model": cross_model_table(per_model),
        "models": per_model,
        "figure": {"pdf": str(figure), "png": str(figure.with_suffix(".png"))},
    }
    atomic_json(out_path, output)
    print(f"x103 {mode}: result -> {out_path}; figure -> {figure}")


def main() -> None:
    args = parse_args()
    cohort = load_cohort()
    inputs = read_json(inputs_path())
    if inputs.get("cohort_sha256") != cohort["cohort_sha256"]:
        raise SystemExit("FATAL: frozen inputs belong to another cohort")
    exclusions = load_exclusions(cohort)
    all_cells = {}
    for tag in ALL_TAGS:
        if not projections_path(tag).exists():
            raise SystemExit(f"FATAL: {tag} has no x103 projections")
        all_cells[tag] = load_cells(tag, cohort)
    modes = MODES if args.cohort == "all" else (args.cohort,)
    x114 = x114_exclusions = None
    if any(uses_x114(mode) for mode in modes):
        x114_cohort = x114_common.load_cohort()
        x114_exclusions = x114_common.load_exclusions(x114_cohort)
        if x114_exclusions["counts"]["n_units_excluded"]:
            raise SystemExit("FATAL: the x85 exclusion rule drops x114 units; the top-up is not x85-free")
        for tag in ALL_TAGS:
            if not x114_common.projections_path(tag).exists():
                raise SystemExit(f"FATAL: {tag} has no x114 projections")
        x114 = {"cohort": x114_cohort, "cells": {tag: load_x114_cells(tag) for tag in ALL_TAGS}}
    for mode in modes:
        run_mode(mode, args, cohort, inputs, exclusions, all_cells, x114, x114_exclusions)


if __name__ == "__main__":
    main()
