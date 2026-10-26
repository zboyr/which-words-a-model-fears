"""x105 Stage 0 (local): freeze the inputs and build the per-model control directions.

Reads only frozen local artifacts — the x101 cohort and manifest, the x88 manifest, each
model's direction artifact (x86's for Qwen3.5-4B), and the rendered-chat sha256 of the O and
E generation rows x101 (x83) judged — and writes:

  data/runs/x105/x105_inputs.json                     the frozen per-model inputs and arm table
  data/runs/x105/<tag>/x105_control_directions.npz    two norm-matched orthogonal random pairs

Nothing here touches a model or a GPU; the control directions are deterministic in
(tag, draw, position), so the GPU jobs load them from the file and re-check its sha256.
"""

from __future__ import annotations

import argparse

import numpy as np

from x105_common import (
    ALL_TAGS,
    ARMS,
    BASE_ARMS,
    BOUNDARY_POSITIONS,
    CONTROL_DRAWS,
    CONTROL_SEED,
    DOSE_MAGNITUDES,
    IMPORTED_TAG,
    MAIN_MAGNITUDE,
    MODELS,
    QWEN4B_HOOK_BLOCK,
    QWEN4B_MODEL_ID,
    TEXT_CONDITIONS,
    X86_MANIFEST,
    X88_RUN,
    atomic_json,
    base_generations_path,
    base_refusal_path,
    benign_calibration_prompts,
    cohort_path,
    consensus_x86_vectors,
    control_arrays,
    control_diagnostics,
    control_direction_pair,
    control_directions_path,
    date_candidates,
    directions_path,
    expected_renderings,
    file_record,
    hook_block_for,
    inputs_path,
    load_cohort,
    manifest_path,
    read_json,
    sha256_file,
    unit_directions,
)
from x87_common import X86_EXPECTED_BOUNDARY_IDS

MAX_CONTROL_COSINE = 1e-6  # orthogonality of the random control to the true direction
MAX_NORM_RATIO_ERROR = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(inputs_path()))
    return parser.parse_args()


def model_record(tag: str, cohort: dict, x101_manifest: dict, x88_manifest: dict) -> dict:
    entry = x101_manifest["models"][tag]
    if tag == IMPORTED_TAG:
        if entry["model_id"] != QWEN4B_MODEL_ID:
            raise SystemExit("FATAL: the imported row is not Qwen3.5-4B")
        record = {
            "model_id": QWEN4B_MODEL_ID,
            "snapshot": entry["snapshot"],
            "revision": entry.get("revision"),
            "n_blocks": None,  # checked at run time; the site is x86/x87/x89's block 29
            "hook_block": int(QWEN4B_HOOK_BLOCK),
            "boundary_ids": list(X86_EXPECTED_BOUNDARY_IDS),
            "directions_kind": "x87 consensus of the four x86 fold directions (Qwen3.5-4B)",
            "base_source": "x83 generations and judgments restricted to the x101 cohort",
            "date_candidates": [],
            "role": "pipeline reference (the model x85/x86/x87 established the mechanism on)",
        }
    else:
        x88_entry = x88_manifest["models"][tag]
        if x88_entry["snapshot"] != entry["snapshot"] or x88_entry["model_id"] != MODELS[tag]["model_id"]:
            raise SystemExit(f"FATAL: {tag}: the x101 and x88 snapshots differ")
        if hook_block_for(int(x88_entry["n_blocks"])) != int(x88_entry["hook_block"]):
            raise SystemExit(f"FATAL: {tag}: the x88 hook block is not the frozen relative-depth rule")
        record = {
            "model_id": MODELS[tag]["model_id"],
            "snapshot": entry["snapshot"],
            "revision": entry.get("revision"),
            "n_blocks": int(x88_entry["n_blocks"]),
            "hook_block": int(x88_entry["hook_block"]),
            "boundary_ids": [int(v) for v in x88_entry["boundary_ids"]],
            "directions_kind": "x88 consensus (mean of four fold directions) at the x88 hook block",
            "base_source": "x101 generations and judgments",
            "date_candidates": date_candidates(entry["chat_template"]["scaffold"]),
            "role": "transfer model",
        }
    record["state_index"] = record["hook_block"] + 1
    record["directions"] = file_record(directions_path(tag))

    minus1, tbg = consensus_x86_vectors(directions_path(tag))
    mu = np.stack([minus1.numpy(), tbg.numpy()]).astype(np.float64)
    _, mu_norms = unit_directions(mu)
    record["direction_norms"] = {p: float(v) for p, v in zip(BOUNDARY_POSITIONS, mu_norms)}
    record["hidden_dim"] = int(mu.shape[1])

    arrays = control_arrays(mu, tag)
    out_path = control_directions_path(tag)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        **arrays,
        mu=mu.astype(np.float32),
        draws=np.asarray(CONTROL_DRAWS),
        boundary_positions=np.asarray(BOUNDARY_POSITIONS),
    )
    diagnostics = {}
    for draw in CONTROL_DRAWS:
        pair = control_direction_pair(mu, tag, draw)
        checks = control_diagnostics(mu, pair)
        if checks["max_abs_cosine_to_true"] > MAX_CONTROL_COSINE or checks["max_norm_ratio_error"] > MAX_NORM_RATIO_ERROR:
            raise SystemExit(f"FATAL: {tag} draw {draw} is not an orthogonal norm-matched control: {checks}")
        diagnostics[f"rand{draw}"] = checks
    record["control_directions"] = {
        **file_record(out_path),
        "construction": (
            "x86_common.orthogonal_random_null per boundary position (Gaussian, projected off the "
            f"true direction, rescaled to its norm), seed namespace x105|{tag}|<draw>, base {CONTROL_SEED}"
        ),
        "draws": list(CONTROL_DRAWS),
        "diagnostics": diagnostics,
    }

    renderings = expected_renderings(tag, cohort)
    record["rendered_chat_sha256"] = {c: renderings[c] for c in TEXT_CONDITIONS}
    record["base_arms"] = {
        c: {
            "arm": BASE_ARMS[c],
            "generations": file_record(base_generations_path(tag, c)),
            "refusal": file_record(base_refusal_path(tag, c)),
        }
        for c in TEXT_CONDITIONS
    }
    return record


def main() -> None:
    args = parse_args()
    cohort = load_cohort()
    x101_manifest = read_json(manifest_path())
    x88_manifest = read_json(X88_RUN / "x88_manifest.json")
    if x101_manifest.get("cohort_sha256") != cohort["cohort_sha256"]:
        raise SystemExit("FATAL: the x101 manifest is not this cohort's")
    if x88_manifest.get("status") != "frozen":
        raise SystemExit("FATAL: the x88 manifest is not frozen")
    benign = benign_calibration_prompts()

    models = {tag: model_record(tag, cohort, x101_manifest, x88_manifest) for tag in ALL_TAGS}
    out = {
        "experiment": "x105",
        "stage": 0,
        "status": "frozen",
        "pre_registered": False,
        "question": (
            "In the seven x88 transfer models, does writing the model's own consensus boundary "
            "direction remove the insertion-induced refusal2 rise on the inserted (E) texts at a "
            "negative dose and create one on the original (O) texts at a positive dose, "
            "dose-ordered and above norm-matched and KL-matched random directions?"
        ),
        "cohort_sha256": cohort["cohort_sha256"],
        "cohort": file_record(cohort_path()),
        "x101_manifest": file_record(manifest_path()),
        "x88_manifest": file_record(X88_RUN / "x88_manifest.json"),
        "x86_manifest": file_record(X86_MANIFEST),
        "text_conditions": list(TEXT_CONDITIONS),
        "dose_units": "multiples of the model's own mean boundary shift (the raw consensus vector)",
        "dose_magnitudes": list(DOSE_MAGNITUDES),
        "main_magnitude": MAIN_MAGNITUDE,
        "arms": [dict(spec) for spec in ARMS],
        "base_arms": {c: BASE_ARMS[c] for c in TEXT_CONDITIONS},
        "benign_kl_calibration": {
            "source": "the 256 frozen x86 benign calibration prompts (model-independent texts)",
            "n_prompts": len(benign),
            "first_row": benign[0]["full_row"],
        },
        "models": models,
    }
    atomic_json(args.out, out)
    print(f"x105 inputs -> {args.out}: {len(models)} models, {len(ARMS)} generated arms per model")
    for tag, record in models.items():
        norms = ", ".join(f"{p} {record['direction_norms'][p]:.2f}" for p in BOUNDARY_POSITIONS)
        print(f"  {tag:8s} block {record['hook_block']:2d} dim {record['hidden_dim']:5d} norms {norms}")


if __name__ == "__main__":
    main()
