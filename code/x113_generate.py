"""x113 Stage 3 (GPU, the cluster, one (model, shard) per job): ten responses per S_T / C_T cell.

The cells are every (unit, arm) with a text, split round-robin into resumable shards. Rendering
is x106's (`x106_common.render_chat` with the date x106 Stage 0a recovered for the model),
sampling is x101's contract through x81's `generate_row` (T = 1.0, top-p 0.9, top-k 50, ten
sequences, 2,048 tokens, one 4,096 retry), the seed is x106's `stable_seed` on (tag, prompt,
x106 text label), so a text identical to x106's S / C text of the same prompt has x106's seed and
its x106 generation row is copied instead of regenerated. Resumable by cell from the shard file.
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
from x113_common import (  # noqa: E402
    ALL_TAGS,
    N_RESPONSES,
    X106_ARM_OF_ARM,
    atomic_json,
    cells,
    cells_sha256,
    file_record,
    load_texts,
    paths,
    read_json,
    render_chat,
    sha256_text,
    stable_seed,
    x106_generations_path,
    x106_inputs_path,
    x106_projections_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, choices=ALL_TAGS)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def shard_path(tag: str, shard: int, n_shards: int) -> Path:
    return paths(tag)["gen"] / f"generations_shard{shard}of{n_shards}.json"


def valid_row(row: dict, cell: dict) -> bool:
    """A prior row is reusable when it is this cell's text, sampled with this cell's seed. The
    grid hash is not part of the test: a text rebuilt after the run changes the grid hash, and
    the rows of the cells whose text did not change are kept."""
    return (
        row.get("cell_id") == cell["cell_id"]
        and row.get("prompt_sha256") == cell["prompt_sha256"]
        and row.get("seed") == cell["seed"]
        and len(row.get("llm_responses", [])) == N_RESPONSES
        and len(row.get("response_sha256", [])) == N_RESPONSES
        and all(sha256_text(t) == s for t, s in zip(row["llm_responses"], row["response_sha256"]))
    )


def main() -> None:
    args = parse_args()
    texts = load_texts(args.tag)
    grid = cells(args.tag, texts)
    grid_sha = cells_sha256(grid)
    mine = grid[args.shard :: args.n_shards]
    if args.limit:
        mine = mine[: args.limit]
    inputs = read_json(x106_inputs_path())
    if inputs.get("experiment") != "x106" or inputs.get("status") != "frozen":
        raise SystemExit("FATAL: invalid x106 inputs")
    entry = inputs["models"][args.tag]
    stage0a = read_json(x106_projections_path(args.tag))
    if stage0a.get("model_tag") != args.tag:
        raise SystemExit("FATAL: x106 Stage 0a projections are missing for this tag")
    date = stage0a["rendering"]["date"]
    out_path = shard_path(args.tag, args.shard, args.n_shards)

    done: dict[str, dict] = {}
    if out_path.exists():
        prior = read_json(out_path)
        if prior.get("model_tag") != args.tag or prior.get("shard") != args.shard or prior.get("n_shards") != args.n_shards:
            raise SystemExit("FATAL: resume output belongs to another tag/shard")
        by_id = {c["cell_id"]: c for c in mine}
        n_stale = 0
        for row in prior.get("rows", []):
            cell = by_id.get(row.get("cell_id"))
            if cell is None:
                continue
            if not valid_row(row, cell):
                n_stale += 1  # the cell's text was rebuilt after this row was sampled
                continue
            done[cell["cell_id"]] = dict(row, arms_sha256=grid_sha)
        if prior.get("cells_sha256") != grid_sha:
            print(f"x113 generate {args.tag} shard {args.shard}: grid rebuilt, {n_stale} row(s) dropped, {len(done)} kept", flush=True)
        elif n_stale:
            raise SystemExit(f"FATAL: {n_stale} invalid resume row(s) under an unchanged grid")

    # byte-identical x106 texts: copy the x106 row (same model, same seed) instead of regenerating
    n_copied = sum(1 for r in done.values() if r.get("copied_from_x106"))
    x106_rows: dict[str, dict[str, dict]] = {}
    for cell in mine:
        if cell["cell_id"] in done or not cell["same_as_x106"]:
            continue
        arm106 = X106_ARM_OF_ARM[cell["arm"]]
        if arm106 not in x106_rows:
            obj = read_json(x106_generations_path(args.tag, arm106))
            if obj.get("experiment") != "x106" or obj.get("arm") != arm106 or obj.get("model_tag") != args.tag:
                raise SystemExit(f"FATAL: {args.tag}: x106 {arm106} generation file is not valid")
            x106_rows[arm106] = {str(r["id"]): r for r in obj["rows"]}
        src = x106_rows[arm106].get(cell["prompt_id"])
        if src is None:
            continue
        if src["prompt_sha256"] != cell["prompt_sha256"] or int(src["seed"]) != cell["seed"] or len(src["llm_responses"]) != N_RESPONSES:
            raise SystemExit(f"FATAL: {args.tag}: x106 row {cell['cell_id']} does not match the x113 text or seed")
        done[cell["cell_id"]] = {
            "cell_id": cell["cell_id"],
            "id": cell["prompt_id"],
            "stratum": cell["stratum"],
            "condition": cell["arm"],
            "arm": cell["arm"],
            "text_condition": cell["arm"],
            "prompt": cell["prompt"],
            "prompt_sha256": cell["prompt_sha256"],
            "rendered_chat_sha256": src["rendered_chat_sha256"],
            "arms_sha256": grid_sha,
            "signed_dose": 0.0,
            "seed": cell["seed"],
            "used_retry": src["used_retry"],
            "retry_seed": src["retry_seed"],
            "llm_responses": src["llm_responses"],
            "response_sha256": src["response_sha256"],
            "gen_tokens": src["gen_tokens"],
            "truncated": src["truncated"],
            "copied_from_x106": arm106,
        }
        n_copied += 1
    todo = [c for c in mine if c["cell_id"] not in done]
    print(f"x113 generate {args.tag} shard {args.shard}/{args.n_shards}: {len(mine)} cells, {len(done)} done ({n_copied} copied from x106), {len(todo)} to generate", flush=True)

    tokenizer = model = None
    n_blocks = None
    if todo:
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
                "experiment": "x113",
                "stage": 3,
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
                "config": {"temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K, "num_return_sequences": N_RESPONSES, "max_new_tokens": MAX_NEW, "retry_max_new_tokens": RETRY_NEW, "thinking": False, "seed_policy": "x106 per-(tag, prompt, x106 text label) seed: S_T uses x106's S seed, C_T x106's C seed"},
                "reuse": "a text byte-identical to x106's S / C text of the same prompt copies x106's generation row (copied_from_x106)",
                "n_cells": len(mine),
                "n_copied_from_x106": n_copied,
                "environment": {"torch": torch.__version__, "transformers": importlib.metadata.version("transformers"), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
                "rows": [done[c["cell_id"]] for c in mine if c["cell_id"] in done],
            },
        )

    checkpoint()
    if not todo:
        print(f"x113 generate {args.tag} shard {args.shard}: nothing to generate -> {out_path}")
        return
    eos_ids = eos_id_set(tokenizer, model)
    for index, cell in enumerate(todo, start=1):
        chat = render_chat(tokenizer, cell["prompt"], date)
        assert_boundary(tokenizer, chat, boundary_ids, args.tag)
        seed = cell["seed"]
        responses, lengths, truncated = generate_row(model, tokenizer, eos_ids, chat, MAX_NEW, seed)
        used_retry = any(truncated)
        retry_seed = None
        if used_retry:
            retry_seed = stable_seed(args.tag, cell["prompt_id"], cell["arm"], retry=True)
            responses, lengths, truncated = generate_row(model, tokenizer, eos_ids, chat, RETRY_NEW, retry_seed)
        if len(responses) != N_RESPONSES or any(not value.strip() for value in responses):
            raise SystemExit(f"FATAL: incomplete/empty generation row {cell['cell_id']}")
        done[cell["cell_id"]] = {
            "cell_id": cell["cell_id"],
            "id": cell["prompt_id"],
            "stratum": cell["stratum"],
            "condition": cell["arm"],
            "arm": cell["arm"],
            "text_condition": cell["arm"],
            "prompt": cell["prompt"],
            "prompt_sha256": cell["prompt_sha256"],
            "rendered_chat_sha256": sha256_text(chat),
            "arms_sha256": grid_sha,
            "signed_dose": 0.0,
            "seed": seed,
            "used_retry": used_retry,
            "retry_seed": retry_seed,
            "llm_responses": responses,
            "response_sha256": [sha256_text(value) for value in responses],
            "gen_tokens": lengths,
            "truncated": truncated,
            "copied_from_x106": None,
        }
        checkpoint()
        print(f"x113 generate {args.tag} shard {args.shard}: {index}/{len(todo)} new", flush=True)
    print(f"x113 generation {args.tag} shard {args.shard}: {len(done)} rows -> {out_path}")


if __name__ == "__main__":
    main()
