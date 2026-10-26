"""x114 Stage 4 (local, CPU): freeze the inputs the boundary-state extraction and the dosed
arms read, after the base generations exist.

Per model: the snapshot, the x88 hook block and boundary ids (x86/x87's block 29 for
Qwen3.5-4B), the direction artifact and its sha256, the direction norms, the
`rendered_chat_sha256` of all four x114 generation rows per prompt (so every later forward
pass can be checked byte-identical to the text the responses came from), the Llama date
candidates recovered from the scaffold the generation job recorded, and the x105 artifacts
the dosed arms reuse unchanged: that model's norm-matched orthogonal control directions
(deterministic in (tag, draw, position)) and its benign-KL calibration, which is measured on
the 256 frozen x86 benign prompts and therefore does not depend on this cohort.
"""

from __future__ import annotations

import argparse

import numpy as np

from x105_common import (
    ARMS,
    BASE_ARMS,
    BOUNDARY_POSITIONS,
    CONTROL_DRAWS,
    DOSE_MAGNITUDES,
    MAIN_MAGNITUDE,
    TEXT_CONDITIONS,
    X86_MANIFEST,
    benign_calibration_prompts,
    consensus_x86_vectors,
    control_diagnostics,
    control_direction_pair,
    control_directions_path,
    directions_path,
    kl_calibration_path,
    unit_directions,
)
from x103_common import QWEN4B_HOOK_BLOCK, date_candidates, hook_block_for
from x114_common import (
    CONDITIONS,
    IMPORTED_TAG,
    MODELS,
    QWEN4B_MODEL_ID,
    TAGS,
    X88_RUN,
    atomic_json,
    cohort_path,
    expected_renderings,
    file_record,
    generations_path,
    inputs_path,
    load_cohort,
    load_manifest,
    manifest_path,
    read_json,
    refusal_path,
    sha256_file,
)
from x87_common import X86_EXPECTED_BOUNDARY_IDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(inputs_path()))
    return parser.parse_args()


def model_record(tag: str, cohort: dict, entry: dict, x88_manifest: dict) -> dict:
    if tag == IMPORTED_TAG:
        if entry["model_id"] != QWEN4B_MODEL_ID:
            raise SystemExit("FATAL: the Qwen3.5-4B row is not Qwen3.5-4B")
        record = {
            "model_id": QWEN4B_MODEL_ID,
            "snapshot": entry["snapshot"],
            "revision": entry.get("revision"),
            "n_blocks": None,  # checked at run time; the site is x86/x87's block 29
            "hook_block": int(QWEN4B_HOOK_BLOCK),
            "boundary_ids": list(X86_EXPECTED_BOUNDARY_IDS),
            "directions_kind": "x87 consensus of the four x86 fold directions (Qwen3.5-4B)",
            "role": "pipeline reference (the model x85/x86/x87 established the mechanism on)",
        }
    else:
        x88_entry = x88_manifest["models"][tag]
        if x88_entry["snapshot"] != entry["snapshot"] or x88_entry["model_id"] != MODELS[tag]["model_id"]:
            raise SystemExit(f"FATAL: {tag}: the x114 and x88 snapshots differ")
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
            "role": "transfer model",
        }
    record["state_index"] = record["hook_block"] + 1
    record["directions"] = file_record(directions_path(tag))

    minus1, tbg = consensus_x86_vectors(directions_path(tag))
    mu = np.stack([minus1.numpy(), tbg.numpy()]).astype(np.float64)
    _, mu_norms = unit_directions(mu)
    record["direction_norms"] = {p: float(v) for p, v in zip(BOUNDARY_POSITIONS, mu_norms)}
    record["hidden_dim"] = int(mu.shape[1])

    control_path = control_directions_path(tag)
    diagnostics = {}
    for draw in CONTROL_DRAWS:
        diagnostics[f"rand{draw}"] = control_diagnostics(mu, control_direction_pair(mu, tag, draw))
    record["control_directions"] = {
        **file_record(control_path),
        "source": "x105's own per-model artifact, reused unchanged (deterministic in (tag, draw, position))",
        "draws": list(CONTROL_DRAWS),
        "diagnostics": diagnostics,
    }
    calibration = read_json(kl_calibration_path(tag))
    if calibration.get("model_tag") != tag or calibration.get("status") != "pass":
        raise SystemExit(f"FATAL: {tag}: x105's benign-KL calibration is missing or did not pass")
    record["kl_calibration"] = {
        **file_record(kl_calibration_path(tag)),
        "source": "x105's own calibration, reused unchanged (measured on the 256 frozen x86 benign prompts, cohort-independent)",
        "true_main_dose_mean_kl": calibration["true_main_dose_mean_kl"],
        "random_match": {
            side: {k: record_side.get(k) for k in ("matched_signed_dose", "matched_mean_kl", "target_kl", "status")}
            for side, record_side in calibration["random_kl_match"].items()
        },
    }

    record["rendered_chat_sha256"] = expected_renderings(tag, cohort)
    scaffold = read_json(generations_path(tag, "O"))["chat_template"]["scaffold"]
    record["date_candidates"] = date_candidates(scaffold)
    record["base_arms"] = {
        c: {
            "arm": BASE_ARMS[c],
            "generations": file_record(generations_path(tag, c)),
            # The dosed arms need only the base texts and seeds, so the inputs can be frozen
            # (and the dosed and extraction jobs can start) while the base arms are being judged.
            "refusal": file_record(refusal_path(tag, c)) if refusal_path(tag, c).exists() else None,
        }
        for c in TEXT_CONDITIONS
    }
    return record


def main() -> None:
    args = parse_args()
    cohort = load_cohort()
    manifest = load_manifest()
    x88_manifest = read_json(X88_RUN / "x88_manifest.json")
    if manifest.get("cohort_sha256") != cohort["cohort_sha256"]:
        raise SystemExit("FATAL: the x114 manifest is not this cohort's")
    if x88_manifest.get("status") != "frozen":
        raise SystemExit("FATAL: the x88 manifest is not frozen")
    benign = benign_calibration_prompts()
    models = {tag: model_record(tag, cohort, manifest["models"][tag], x88_manifest) for tag in TAGS}
    out = {
        "experiment": "x114",
        "stage": 4,
        "status": "frozen",
        "pre_registered": False,
        "cohort_sha256": cohort["cohort_sha256"],
        "cohort": file_record(cohort_path()),
        "x114_manifest": file_record(manifest_path()),
        "x88_manifest": file_record(X88_RUN / "x88_manifest.json"),
        "x86_manifest": file_record(X86_MANIFEST),
        "conditions": list(CONDITIONS),
        "text_conditions": list(TEXT_CONDITIONS),
        "dose_units": "multiples of the model's own mean boundary shift (the raw consensus vector), the x86/x88/x105 unit",
        "dose_magnitudes": list(DOSE_MAGNITUDES),
        "main_magnitude": MAIN_MAGNITUDE,
        "arms": [dict(spec) for spec in ARMS],
        "base_arms": {c: BASE_ARMS[c] for c in TEXT_CONDITIONS},
        "benign_kl_calibration": {
            "source": "x105's per-model calibration on the 256 frozen x86 benign calibration prompts",
            "n_prompts": len(benign),
        },
        "models": models,
    }
    atomic_json(args.out, out)
    print(f"x114 inputs -> {args.out}: {len(models)} models, {len(ARMS)} dosed arms per model")
    for tag, record in models.items():
        norms = ", ".join(f"{p} {record['direction_norms'][p]:.2f}" for p in BOUNDARY_POSITIONS)
        print(f"  {tag:8s} block {record['hook_block']:2d} dim {record['hidden_dim']:5d} norms {norms}")


if __name__ == "__main__":
    main()
