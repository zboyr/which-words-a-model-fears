"""x116 Stage 2 (GPU, the cluster, one (model, shard) per job): ten responses per R_rand cell.

x113_generate.py with the R_rand cells: x106 rendering (`render_chat` with the date x106 Stage 0a
recovered), x101 sampling through x81's `generate_row` (T = 1.0, top-p 0.9, top-k 50, ten
sequences, 2,048 tokens, one 4,096 retry), x106's `stable_seed` under the text label "R_rand".
Resumable by cell from the shard file.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x81_generate import MAX_NEW, RETRY_NEW, TEMPERATURE, TOP_K, TOP_P, eos_id_set, generate_row  # noqa: E402
from x106_common import assert_boundary, load_tag_model, load_tag_tokenizer, n_blocks_of  # noqa: E402
from x113_common import render_chat, x106_inputs_path, x106_projections_path  # noqa: E402
from x116_common import ALL_TAGS, N_RESPONSES, atomic_json, cells, cells_sha256, file_record, load_texts, paths, read_json, sha256_text, stable_seed  # noqa: E402


def shard_path(tag: str, shard: int, n_shards: int) -> Path:
    return paths(tag)["gen"] / f"generations_shard{shard}of{n_shards}.json"


def valid_row(row: dict, cell: dict) -> bool:
    return (
        row.get("cell_id") == cell["cell_id"]
        and row.get("prompt_sha256") == cell["prompt_sha256"]
        and row.get("seed") == cell["seed"]
        and len(row.get("llm_responses", [])) == N_RESPONSES
        and all(sha256_text(t) == s for t, s in zip(row["llm_responses"], row["response_sha256"]))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, choices=ALL_TAGS)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    texts = load_texts(args.tag)
    grid = cells(args.tag, texts)
    grid_sha = cells_sha256(grid)
    mine = grid[args.shard :: args.n_shards]
    if args.limit:
        mine = mine[: args.limit]
    inputs = read_json(x106_inputs_path())
    entry = inputs["models"][args.tag]
    stage0a = read_json(x106_projections_path(args.tag))
    if stage0a.get("model_tag") != args.tag:
        raise SystemExit("FATAL: x106 Stage 0a projections are missing for this tag")
    date = stage0a["rendering"]["date"]
    out_path = shard_path(args.tag, args.shard, args.n_shards)
    done: dict[str, dict] = {}
    if out_path.exists():
        prior = read_json(out_path)
        if prior.get("model_tag") != args.tag or prior.get("shard") != args.shard or prior.get("n_shards") != args.n_shards or prior.get("cells_sha256") != grid_sha:
            raise SystemExit("FATAL: resume output belongs to another tag/shard/grid")
        by_id = {c["cell_id"]: c for c in mine}
        for row in prior.get("rows", []):
            cell = by_id.get(row.get("cell_id"))
            if cell is None or not valid_row(row, cell):
                raise SystemExit(f"FATAL: invalid resume row {row.get('cell_id')}")
            done[cell["cell_id"]] = row
    todo = [c for c in mine if c["cell_id"] not in done]
    print(f"x116 generate {args.tag} shard {args.shard}/{args.n_shards}: {len(mine)} cells, {len(done)} done, {len(todo)} to generate", flush=True)
    if not todo:
        return
    tokenizer = load_tag_tokenizer(entry["snapshot"], args.tag)
    model = load_tag_model(entry["snapshot"], args.tag)
    model.eval()
    n_blocks = n_blocks_of(model)
    if entry["n_blocks"] is not None and n_blocks != int(entry["n_blocks"]):
        raise SystemExit(f"FATAL: {args.tag}: model has {n_blocks} blocks, the inputs record {entry['n_blocks']}")
    boundary_ids = [int(v) for v in entry["boundary_ids"]]

    def checkpoint() -> None:
        atomic_json(
            out_path,
            {
                "experiment": "x116",
                "stage": 2,
                "model_tag": args.tag,
                "model_id": entry["model_id"],
                "model_source": entry["snapshot"],
                "shard": args.shard,
                "n_shards": args.n_shards,
                "cells_sha256": grid_sha,
                "texts_sha256": texts["texts_sha256"],
                "texts": file_record(paths(args.tag)["texts"]),
                "inputs": file_record(x106_inputs_path()),
                "rendering": {"renderer": "x87_common.format_user_chat via x106_common.render_chat", "date": date},
                "config": {"temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K, "num_return_sequences": N_RESPONSES, "max_new_tokens": MAX_NEW, "retry_max_new_tokens": RETRY_NEW, "thinking": False, "seed_policy": "x106 stable_seed(tag, prompt id, 'R_rand')"},
                "n_cells": len(mine),
                "environment": {"torch": torch.__version__, "transformers": importlib.metadata.version("transformers"), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
                "rows": [done[c["cell_id"]] for c in mine if c["cell_id"] in done],
            },
        )

    checkpoint()
    eos_ids = eos_id_set(tokenizer, model)
    for index, cell in enumerate(todo, start=1):
        chat = render_chat(tokenizer, cell["prompt"], date)
        assert_boundary(tokenizer, chat, boundary_ids, args.tag)
        responses, lengths, truncated = generate_row(model, tokenizer, eos_ids, chat, MAX_NEW, cell["seed"])
        used_retry = any(truncated)
        retry_seed = None
        if used_retry:
            retry_seed = stable_seed(args.tag, cell["prompt_id"], retry=True)
            responses, lengths, truncated = generate_row(model, tokenizer, eos_ids, chat, RETRY_NEW, retry_seed)
        if len(responses) != N_RESPONSES or any(not value.strip() for value in responses):
            raise SystemExit(f"FATAL: incomplete/empty generation row {cell['cell_id']}")
        done[cell["cell_id"]] = {
            "cell_id": cell["cell_id"],
            "id": cell["prompt_id"],
            "stratum": cell["stratum"],
            "condition": cell["arm"],
            "arm": cell["arm"],
            "prompt": cell["prompt"],
            "prompt_sha256": cell["prompt_sha256"],
            "rendered_chat_sha256": sha256_text(chat),
            "arms_sha256": grid_sha,
            "seed": cell["seed"],
            "used_retry": used_retry,
            "retry_seed": retry_seed,
            "llm_responses": responses,
            "response_sha256": [sha256_text(value) for value in responses],
            "gen_tokens": lengths,
            "truncated": truncated,
        }
        checkpoint()
        print(f"x116 generate {args.tag} shard {args.shard}: {index}/{len(todo)} new", flush=True)
    print(f"x116 generation {args.tag} shard {args.shard}: {len(done)} rows -> {out_path}")


if __name__ == "__main__":
    main()
