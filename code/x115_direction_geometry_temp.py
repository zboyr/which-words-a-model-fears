"""Compare saved Path A and Path B residual-space directions."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PATHS = {
    "probe": "data/runs/x87/x75_qwen_probes/x75_refusal_weighted_true.pt",
    "adapter": "data/runs/x87/x75_qwen_refusal_weighted_true.pt",
    "global": "data/results/x85_crossed_keyword_directions.npz",
    "folds": "data/runs/x86/x86_directions.npz",
}


def metrics(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return {
        "cosine": cosine,
        "axis_angle_degrees": float(np.degrees(np.arccos(np.clip(abs(cosine), 0, 1)))),
        "squared_cosine": cosine**2,
    }


def main():
    probe = torch.load(ROOT / PATHS["probe"], map_location="cpu", weights_only=True)
    adapter = torch.load(ROOT / PATHS["adapter"], map_location="cpu", weights_only=True)
    q = probe["q_raw"].double().numpy()
    reconstructed = (probe["w_standardized"].double() / probe["std"].double()).numpy()
    result = {
        "experiment": "x115",
        "model": adapter["model_id"],
        "path_a_state": probe["state_index"],
        "path_b_deployed_state": 30,
        "dimension": len(q),
        "probe_adapter": metrics(q, adapter["q_unit"].double().numpy()),
        "probe_reconstruction": metrics(q, reconstructed),
        "inputs": {
            k: {"path": p, "sha256": hashlib.sha256((ROOT / p).read_bytes()).hexdigest()}
            for k, p in PATHS.items()
        },
        "global_comparisons": {},
        "deployed_comparisons": {},
    }
    with np.load(ROOT / PATHS["global"]) as vectors:
        for pos in ("tbg_minus1", "tbg"):
            for state in (29, 30, 31):
                key = f"global_{pos}_s{state}"
                result["global_comparisons"][key] = metrics(q, vectors[key])
    with np.load(ROOT / PATHS["folds"]) as vectors:
        for pos in ("tbg_minus1", "tbg"):
            folds = np.stack([vectors[f"true_f{f}_{pos}"] for f in range(4)])
            result["deployed_comparisons"][pos] = {
                "consensus": metrics(q, folds.mean(axis=0)),
                "folds": [metrics(q, v) for v in folds],
            }
    output = ROOT / "data/results/x115_path_a_b_direction_geometry.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "inputs"}, indent=2))


if __name__ == "__main__":
    main()
