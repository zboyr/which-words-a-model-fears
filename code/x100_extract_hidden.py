"""x100 Stage 2: dense-grid residual states 29-31 at tbg_minus1 and tbg (fp16 shards)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from x100_common import (
    FEATURE_POSITIONS,
    FEATURE_STATES,
    FEATURES,
    MODEL_ID,
    N_FEATURE_SHARDS,
    cohort_path,
    format_user_chat,
    load_causal_lm,
    locate_positions,
    read_json,
    sha256_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default=str(cohort_path()))
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=N_FEATURE_SHARDS)
    parser.add_argument("--out")
    return parser.parse_args()


def default_out(shard: int, num_shards: int) -> Path:
    return FEATURES / f"x100_hidden_shard_{shard:03d}-of-{num_shards:03d}.pt"


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard < args.num_shards:
        raise SystemExit("FATAL: shard must satisfy 0 <= shard < num_shards")
    manifest = read_json(Path(args.cohort))
    if manifest.get("experiment") != "x100" or manifest.get("model_id") != MODEL_ID or manifest.get("status") != "frozen":
        raise SystemExit("FATAL: invalid x100 cohort")
    cells = [c for c in manifest["cells"] if int(c["cell_index"]) % args.num_shards == args.shard]
    if not cells:
        raise SystemExit("FATAL: empty extraction shard")
    out_path = Path(args.out) if args.out else default_out(args.shard, args.num_shards)
    expected_ids = [c["cell_id"] for c in cells]
    if out_path.exists():
        prior = torch.load(out_path, map_location="cpu", weights_only=False)
        if prior.get("experiment") == "x100" and prior.get("cohort_sha256") == manifest["cohort_sha256"] and prior.get("cell_ids") == expected_ids:
            print(f"x100 hidden shard already complete: {out_path}")
            return
        raise SystemExit("FATAL: existing shard output does not match requested cells/cohort")

    model_source = manifest["model_snapshot"]
    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
    model = load_causal_lm(model_source)
    states = list(FEATURE_STATES)
    positions = list(FEATURE_POSITIONS)
    rows = []
    records = []
    n_states = hidden_dim = None
    for index, cell in enumerate(cells, start=1):
        text = cell["prompt"]
        if sha256_text(text) != cell["prompt_sha256"]:
            raise SystemExit(f"FATAL: prompt hash mismatch for {cell['cell_id']}")
        rendered = format_user_chat(tokenizer, text)
        if sha256_text(rendered) != cell["rendered_chat_sha256"]:
            raise SystemExit(f"FATAL: rendered chat mismatch for {cell['cell_id']}")
        span = None if cell["inserted_span"] is None else tuple(cell["inserted_span"])
        input_ids, located = locate_positions(tokenizer, rendered, text, span)
        if located != cell["position_indices"] or len(input_ids) != cell["n_input_tokens"]:
            raise SystemExit(f"FATAL: token-position audit mismatch for {cell['cell_id']}")
        inputs = torch.tensor([input_ids], dtype=torch.long, device=model.device)
        with torch.inference_mode():
            outputs = model(input_ids=inputs, output_hidden_states=True, return_dict=True)
        if n_states is None:
            n_states = len(outputs.hidden_states)
            hidden_dim = int(outputs.hidden_states[0].shape[-1])
            if max(states) >= n_states:
                raise SystemExit("FATAL: requested state beyond hidden-state count")
        row = torch.empty((len(positions), len(states), hidden_dim), dtype=torch.float16)
        for si, state in enumerate(states):
            hidden = outputs.hidden_states[state]
            for pi, name in enumerate(positions):
                row[pi, si] = hidden[0, located[name]].float().cpu().to(torch.float16)
        rows.append(row)
        records.append({"cell_index": int(cell["cell_index"]), "cell_id": cell["cell_id"], "prompt_sha256": cell["prompt_sha256"], "position_indices": located})
        if index % 50 == 0 or index == len(cells):
            print(f"x100 hidden shard {args.shard}: {index}/{len(cells)}", flush=True)

    artifact = {
        "experiment": "x100",
        "stage": 2,
        "model_id": MODEL_ID,
        "model_source": model_source,
        "cohort_sha256": manifest["cohort_sha256"],
        "shard": args.shard,
        "num_shards": args.num_shards,
        "position_names": positions,
        "state_indices": states,
        "num_states_total": int(n_states),
        "hidden_dim": int(hidden_dim),
        "cell_ids": expected_ids,
        "cells": records,
        "states": torch.stack(rows),  # [cell, position, state, hidden]
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(artifact, temp)
    os.replace(temp, out_path)
    print(f"x100 hidden shard {args.shard}: {len(cells)} cells -> {out_path}")


if __name__ == "__main__":
    main()
