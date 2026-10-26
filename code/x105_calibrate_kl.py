"""x105 Stage 1 (GPU, one model per job): benign KL of the write, and the KL-matched random dose.

On the 256 frozen x86 benign calibration prompts (model-independent texts, rendered with this
model's own template), measure `KL(base || written)` of the next-token distribution at the
generation boundary for the true consensus direction at each dose of the grid, then bisect the
dose of the model's random control draw 1 until its benign KL equals the true direction's at
the main magnitude. That matched dose is what the `*_randkl_*` arm writes, so the control
perturbs the model's output distribution exactly as hard as the real direction does — x86's
KL-matched null, transferred.

Nothing is gated on the KL value: unlike x86's `a*`, the x88 transfer dose is fixed by design,
and the number is recorded for comparison across models.
"""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from x105_common import (
    ALL_TAGS,
    BOUNDARY_POSITIONS,
    CONTROL_DRAWS,
    DOSE_MAGNITUDES,
    IMPORTED_TAG,
    KL_CAP,
    KL_SLACK,
    MAIN_MAGNITUDE,
    ExternalBoundaryHook,
    assert_boundary,
    atomic_json,
    benign_calibration_prompts,
    consensus_x86_vectors,
    control_diagnostics,
    control_directions_path,
    directions_path,
    file_record,
    format_user_chat,
    hook_block_for,
    hook_site,
    inputs_path,
    kl_calibration_path,
    load_control_pair,
    load_tag_model,
    load_tag_tokenizer,
    n_blocks_of,
    read_json,
    sha256_file,
    unit_directions,
)

# x86_calibrate_kl's matching contract, with its bisection ceiling kept and its iteration
# budget trimmed (a 5 % relative tolerance converges well inside twelve steps).
ABS_MATCH_TOL = 5e-5
REL_MATCH_TOL = 0.05
NULL_MAX_MULTIPLIER = 8.0
MAX_BISECTIONS = 12
MATCH_DRAW = CONTROL_DRAWS[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, choices=ALL_TAGS)
    parser.add_argument("--inputs", default=str(inputs_path()))
    parser.add_argument("--n-prompts", type=int, default=0, help="debug: only the first N benign prompts")
    parser.add_argument("--out")
    return parser.parse_args()


class Runner:
    def __init__(self, tag: str, entry: dict, prompts: list[dict]):
        self.tag = tag
        self.tokenizer = load_tag_tokenizer(entry["snapshot"], tag)
        self.model = load_tag_model(entry["snapshot"], tag)
        self.model.eval()
        n_blocks = n_blocks_of(self.model)
        self.block = int(entry["hook_block"])
        if entry["n_blocks"] is not None and n_blocks != int(entry["n_blocks"]):
            raise SystemExit(f"FATAL: {tag}: model has {n_blocks} blocks, the inputs record {entry['n_blocks']}")
        if tag != IMPORTED_TAG and hook_block_for(n_blocks) != self.block:
            raise SystemExit(f"FATAL: {tag}: block {self.block} is not the x88 rule on {n_blocks} blocks")
        self.site = hook_site(self.model, self.block)
        self.examples = []
        boundary_ids = [int(v) for v in entry["boundary_ids"]]
        for row in prompts:
            chat = format_user_chat(self.tokenizer, row["prompt"])
            assert_boundary(self.tokenizer, chat, boundary_ids, tag)
            encoded = self.tokenizer(chat, return_tensors="pt", add_special_tokens=False).to(self.model.device)
            with torch.inference_mode():
                logits = self.model(**encoded, use_cache=False).logits[0, -1].float()
            self.examples.append({"tokens": encoded, "base_logp": F.log_softmax(logits, dim=-1).cpu()})

    def measure(self, family: str, vectors: np.ndarray, signed_dose: float) -> dict:
        minus1 = torch.from_numpy(np.asarray(vectors[0], dtype=np.float32))
        tbg = torch.from_numpy(np.asarray(vectors[1], dtype=np.float32))
        values = []
        hook = ExternalBoundaryHook(signed_dose, minus1, tbg)
        handle = self.site.register_forward_hook(hook)
        try:
            for example in self.examples:
                with torch.inference_mode():
                    logits = self.model(**example["tokens"], use_cache=False).logits[0, -1].float()
                edited = F.log_softmax(logits, dim=-1).cpu()
                base = example["base_logp"]
                kl = float((base.exp() * (base - edited)).sum())
                if not np.isfinite(kl):
                    raise SystemExit("FATAL: non-finite benign KL")
                values.append(kl)
        finally:
            handle.remove()
        if hook.prefill_calls != len(self.examples) or hook.decode_calls:
            raise SystemExit(f"FATAL: hook fired {hook.prefill_calls}/{hook.decode_calls} on {len(self.examples)} prefills")
        return {
            "family": family,
            "signed_dose": float(signed_dose),
            "magnitude": abs(float(signed_dose)),
            "n_prompts": len(values),
            "mean_kl": float(np.mean(values)),
            "median_kl": float(np.median(values)),
            "max_kl": float(np.max(values)),
        }


def monotone(records: list[dict], slack: float = KL_SLACK) -> bool:
    ordered = sorted(records, key=lambda row: row["magnitude"])
    values = [row["mean_kl"] for row in ordered]
    return all(right + slack >= left for left, right in zip(values, values[1:]))


def match_random(runner: Runner, vectors: np.ndarray, sign: float, target_kl: float) -> dict:
    tolerance = max(ABS_MATCH_TOL, REL_MATCH_TOL * target_kl)
    high = runner.measure("random", vectors, sign * NULL_MAX_MULTIPLIER)
    evaluations = [high]
    if high["mean_kl"] + tolerance < target_kl:
        return {"status": "unbracketed", "target_kl": target_kl, "tolerance": tolerance, "evaluations": evaluations}
    low_value, high_value, selected = 0.0, NULL_MAX_MULTIPLIER, None
    for _ in range(MAX_BISECTIONS):
        midpoint = (low_value + high_value) / 2.0
        result = runner.measure("random", vectors, sign * midpoint)
        evaluations.append(result)
        if abs(result["mean_kl"] - target_kl) <= tolerance:
            selected = result
            break
        if result["mean_kl"] < target_kl:
            low_value = midpoint
        else:
            high_value = midpoint
    if selected is None:
        selected = min(evaluations, key=lambda row: abs(row["mean_kl"] - target_kl))
    status = "pass" if abs(selected["mean_kl"] - target_kl) <= tolerance else "tolerance_miss"
    return {
        "status": status,
        "target_kl": target_kl,
        "tolerance": tolerance,
        "matched_signed_dose": selected["signed_dose"],
        "matched_mean_kl": selected["mean_kl"],
        "n_evaluations": len(evaluations),
        "evaluations": sorted(evaluations, key=lambda row: row["signed_dose"]),
    }


def main() -> None:
    args = parse_args()
    inputs = read_json(args.inputs)
    if inputs.get("experiment") != "x105" or inputs.get("status") != "frozen":
        raise SystemExit("FATAL: invalid x105 inputs")
    tag = args.tag
    entry = inputs["models"][tag]
    out_path = Path(args.out) if args.out else kl_calibration_path(tag)

    dir_path = directions_path(tag)
    if sha256_file(dir_path) != entry["directions"]["sha256"]:
        raise SystemExit(f"FATAL: {tag}: direction artifact differs from the frozen inputs")
    control_path = control_directions_path(tag)
    if sha256_file(control_path) != entry["control_directions"]["sha256"]:
        raise SystemExit(f"FATAL: {tag}: control directions differ from the frozen inputs")
    minus1, tbg = consensus_x86_vectors(dir_path)
    mu = np.stack([minus1.numpy(), tbg.numpy()]).astype(np.float64)
    control = load_control_pair(control_path, MATCH_DRAW)
    _, mu_norms = unit_directions(mu)

    prompts = benign_calibration_prompts()
    if args.n_prompts:
        prompts = prompts[: args.n_prompts]
    runner = Runner(tag, entry, prompts)

    true_records = {"minus": [], "plus": []}
    for magnitude in DOSE_MAGNITUDES:
        true_records["minus"].append(runner.measure("true", mu, -float(magnitude)))
        true_records["plus"].append(runner.measure("true", mu, +float(magnitude)))
        print(f"x105 kl {tag}: true -{magnitude} {true_records['minus'][-1]['mean_kl']:.4f} / "
              f"+{magnitude} {true_records['plus'][-1]['mean_kl']:.4f}", flush=True)

    targets = {
        side: next(row["mean_kl"] for row in records if row["magnitude"] == MAIN_MAGNITUDE)
        for side, records in true_records.items()
    }
    matches = {
        side: match_random(runner, control, -1.0 if side == "minus" else +1.0, targets[side])
        for side in ("minus", "plus")
    }
    for side, record in matches.items():
        print(f"x105 kl {tag}: random {side} match {record['status']} dose "
              f"{record.get('matched_signed_dose')} kl {record.get('matched_mean_kl')}", flush=True)

    out = {
        "experiment": "x105",
        "stage": 1,
        "model_tag": tag,
        "model_id": entry["model_id"],
        "model_source": entry["snapshot"],
        "inputs": file_record(args.inputs),
        "site": {"hook_block": runner.block, "state_index": runner.block + 1, "boundary_ids": entry["boundary_ids"]},
        "directions": {**entry["directions"], "norms": {p: float(v) for p, v in zip(BOUNDARY_POSITIONS, mu_norms)}},
        "control_draw": MATCH_DRAW,
        "control_diagnostics": control_diagnostics(mu, control),
        "n_benign_prompts": len(prompts),
        "config": {
            "measure": "KL(base || written) of the next-token distribution at the final prefill position",
            "dose_magnitudes": list(DOSE_MAGNITUDES),
            "abs_match_tolerance": ABS_MATCH_TOL,
            "relative_match_tolerance": REL_MATCH_TOL,
            "null_max_multiplier": NULL_MAX_MULTIPLIER,
            "max_bisections": MAX_BISECTIONS,
            "x86_kl_cap_for_reference": KL_CAP,
        },
        "true_dose_kl": true_records,
        "true_dose_kl_monotone": {side: monotone(records) for side, records in true_records.items()},
        "true_main_dose_mean_kl": targets,
        "random_kl_match": matches,
        "status": "pass" if all(record["status"] == "pass" for record in matches.values()) else "match_failed",
        "environment": {
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
    }
    atomic_json(out_path, out)
    print(f"x105 kl {tag}: status={out['status']} -> {out_path}")
    if out["status"] != "pass":
        raise SystemExit("x105 KL matching failed")


if __name__ == "__main__":
    main()
