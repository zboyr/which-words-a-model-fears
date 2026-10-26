"""x114 Stage 3b (local): the x101 insertion effect on the x114 units, per model.

x101's own estimator and bootstrap on the new cohort: `delta_pos = mean_i(r_pos,i - r_O,i)`
for S / M / E against the contemporaneously regenerated original, a 10,000-replicate
two-stage paired bootstrap (prompts, then the ten binary responses), Holm within model over
the three positions and across the 24 model x position tests, plus the detection-gate
admission counts, the admitted-prompt deltas, and the run-era diagnostic against each
model's historical labels. This is the behavioural check that the top-up units behave like
x101's; the direction results on them are x103's and x105's.
"""

from __future__ import annotations

import argparse

import numpy as np

from x83_analyze import contrast, holm
from x114_common import (
    CONDITIONS,
    DETECT_THRESHOLD,
    POSITIONS,
    RESULTS,
    TAGS,
    atomic_json,
    cohort_path,
    detect_path,
    file_record,
    generations_path,
    labels_path,
    load_cohort,
    load_manifest,
    load_refusal_arrays,
    load_tag_historical_refusals,
    manifest_path,
    read_json,
    refusal_path,
)

N_BOOTSTRAP = 10_000
SEED = 114


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(RESULTS / "x114_insertion_topup.json"))
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def analyze_tag(tag: str, cohort: dict, n_boot: int, seed: int) -> dict:
    ids, arrays, dropped = load_refusal_arrays(tag, cohort)
    units = {int(u["id"]): u for u in cohort["units"]}
    historical = load_tag_historical_refusals(tag, ids)
    hist = np.stack([np.asarray(historical[i], dtype=np.int8) for i in ids])
    detect = read_json(detect_path(tag))
    if detect.get("cohort_sha256") != cohort["cohort_sha256"]:
        raise SystemExit(f"FATAL: {tag}: the detection file belongs to another cohort")
    rows = {int(r["id"]): r for r in detect["rows"]}
    p_detect = {p: np.asarray([rows[i]["positions"][p]["p_detect"] for i in ids], dtype=float) for p in POSITIONS}
    admitted = {p: p_detect[p] > DETECT_THRESHOLD for p in POSITIONS}
    r_o = arrays["O"].mean(axis=1)

    ungated = {p: contrast(arrays["O"], arrays[p], n_boot, seed + 10 + k) for k, p in enumerate(POSITIONS)}
    for p, adjusted in zip(POSITIONS, holm([ungated[p]["bootstrap_p_two_sided"] for p in POSITIONS])):
        ungated[p]["holm_adjusted_p_within_model"] = adjusted
    gated = {
        p: contrast(arrays["O"][admitted[p]], arrays[p][admitted[p]], n_boot, seed + 20 + k)
        for k, p in enumerate(POSITIONS)
    }
    generations = {c: read_json(generations_path(tag, c)) for c in CONDITIONS}
    integrity = {
        c: {
            "rows": len(obj["rows"]),
            "retry_rows": sum(bool(r.get("used_retry")) for r in obj["rows"]),
            "truncated_after_retry": int(sum(sum(bool(v) for v in r["truncated"]) for r in obj["rows"])),
            "mean_gen_tokens": float(np.mean([np.mean(r["gen_tokens"]) for r in obj["rows"]])),
        }
        for c, obj in generations.items()
    }
    return {
        "model_tag": tag,
        "model_id": detect["model_id"],
        "model_source": detect["model_source"],
        "inputs": {
            "detect": file_record(detect_path(tag)),
            "historical_labels": file_record(labels_path(tag)),
            **{f"generation_{c}": file_record(generations_path(tag, c)) for c in CONDITIONS},
            **{f"refusal_{c}": file_record(refusal_path(tag, c)) for c in CONDITIONS},
        },
        "counts": {
            "cohort": len(ids),
            "dropped_unjudged_prompts": dropped,
            "admitted": {p: int(admitted[p].sum()) for p in POSITIONS},
            "admitted_all_three": int(np.logical_and.reduce([admitted[p] for p in POSITIONS]).sum()),
            "median_p_detect": {p: float(np.median(p_detect[p])) for p in POSITIONS},
            "greedy_match_rate": {
                p: float(np.mean([rows[i]["positions"][p]["greedy_matches"] for i in ids])) for p in POSITIONS
            },
            "n_distinct_words": len({units[i]["word"] for i in ids}),
            "floor_prompts_regenerated_O_zero": int((r_o == 0).sum()),
        },
        "condition_means": {c: float(arrays[c].mean()) for c in CONDITIONS},
        "historical_rO_mean": float(hist.mean()),
        "run_era_diagnostic_O_new_minus_historical": contrast(hist, arrays["O"], n_boot, seed + 30),
        "ungated_delta_vs_O": ungated,
        "gated_delta_vs_O": gated,
        "generation_integrity": integrity,
    }


def main() -> None:
    args = parse_args()
    cohort = load_cohort()
    manifest = load_manifest()
    if manifest.get("cohort_sha256") != cohort["cohort_sha256"]:
        raise SystemExit("FATAL: the x114 manifest is not this cohort's")
    per_model = {tag: analyze_tag(tag, cohort, args.bootstrap, args.seed + 100 * i) for i, tag in enumerate(TAGS)}
    flat = [(tag, p) for tag in TAGS for p in POSITIONS]
    adjusted = holm([per_model[tag]["ungated_delta_vs_O"][p]["bootstrap_p_two_sided"] for tag, p in flat])
    for (tag, p), value in zip(flat, adjusted):
        per_model[tag]["ungated_delta_vs_O"][p]["holm_adjusted_p_over_24_tests"] = value
    table = [
        {
            "model_tag": tag,
            "position": p,
            "n": per_model[tag]["counts"]["cohort"],
            "mean_r_O": per_model[tag]["condition_means"]["O"],
            "mean_r_pos": per_model[tag]["condition_means"][p],
            "delta": per_model[tag]["ungated_delta_vs_O"][p]["estimate"],
            "ci95": per_model[tag]["ungated_delta_vs_O"][p]["ci95"],
            "holm_p_24": per_model[tag]["ungated_delta_vs_O"][p]["holm_adjusted_p_over_24_tests"],
            "n_admitted": per_model[tag]["counts"]["admitted"][p],
            "gated_delta": per_model[tag]["gated_delta_vs_O"][p].get("estimate"),
        }
        for tag, p in flat
    ]
    output = {
        "experiment": "x114",
        "stage": "3b",
        "status": "complete",
        "pre_registered": False,
        "scope": (
            "the 72 x114 units in all eight models: one inserted harm word at the start, middle or end "
            "against the contemporaneously regenerated original, refusal2 only"
        ),
        "cohort": {
            "file": file_record(cohort_path()),
            "cohort_sha256": cohort["cohort_sha256"],
            "n_units": len(cohort["units"]),
            "n_words": cohort["counts"]["words_used"],
            "rule": cohort["design"]["cohort_rule"],
        },
        "manifest": file_record(manifest_path()),
        "config": {
            "estimator": "delta_pos = mean_i(r_pos,i - r_O,i), refusal2",
            "n_bootstrap": args.bootstrap,
            "bootstrap_unit": "prompts, then the ten binary responses within each prompt-condition (x83's scheme)",
            "bootstrap_seed": args.seed,
            "holm": "within model over the three positions, and over all 24 model x position tests",
            "detection_gate": f"P_detect > {DETECT_THRESHOLD} (x83's statistic), reported as the gated table",
        },
        "cross_model": table,
        "models": per_model,
    }
    atomic_json(args.out, output)
    for tag in TAGS:
        line = " ".join(
            f"{p}={per_model[tag]['ungated_delta_vs_O'][p]['estimate']:+.3f}"
            f"[{per_model[tag]['ungated_delta_vs_O'][p]['ci95'][0]:+.3f},"
            f"{per_model[tag]['ungated_delta_vs_O'][p]['ci95'][1]:+.3f}]"
            for p in POSITIONS
        )
        print(f"x114 {tag:8s} n={per_model[tag]['counts']['cohort']} O={per_model[tag]['condition_means']['O']:.3f} {line}")
    print(f"x114 insertion top-up: result -> {args.out}")


if __name__ == "__main__":
    main()
