"""Vocabulary scan, generation (GPU, Qwen3.5-4B, one shard per job).

Every (prompt, condition) cell of the full cross (O, E_w, N_w) gets ten sampled responses
under the x106 / x101 sampling contract (x81_generate: temperature, top-p, top-k, 2,048 new
tokens with a 4,096-token retry when any response truncates), at dose 0, with a stable seed
per (prompt id, condition). Rendering is x106's. Resumable by cell id from the shard file.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x109_scan_common import (  # noqa: E402
    GEN_DIR,
    PROMPTS,
    TAG,
    WORDS,
    atomic_json,
    cells,
    cells_sha256,
    read_json,
    sha256_text,
    stable_seed,
)
from x81_generate import MAX_NEW, RETRY_NEW, TEMPERATURE, TOP_K, TOP_P, eos_id_set, generate_row  # noqa: E402
from x106_common import N_RESPONSES, assert_boundary, inputs_path, load_tag_model, load_tag_tokenizer, projections_path, render_chat  # noqa: E402


def shard_path(shard: int, n_shards: int) -> Path:
    return GEN_DIR / f"generations_shard{shard}of{n_shards}.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    args = parser.parse_args()
    all_cells = cells(read_json(PROMPTS), read_json(WORDS))
    grid_sha = cells_sha256(all_cells)
    mine = all_cells[args.shard :: args.n_shards]
    out_path = shard_path(args.shard, args.n_shards)
    done: dict[str, dict] = {}
    if out_path.exists():
        prior = read_json(out_path)
        if prior.get("cells_sha256") != grid_sha:
            raise SystemExit("FATAL: resume file belongs to another cell grid")
        done = {r["cell_id"]: r for r in prior["rows"]}
    entry = read_json(inputs_path())["models"][TAG]
    date = read_json(projections_path(TAG))["rendering"]["date"]
    tokenizer = load_tag_tokenizer(entry["snapshot"], TAG)
    model = load_tag_model(entry["snapshot"], TAG)
    model.eval()
    eos_ids = eos_id_set(tokenizer, model)
    boundary_ids = [int(v) for v in entry["boundary_ids"]]

    def checkpoint() -> None:
        atomic_json(
            out_path,
            {
                "run": "x109",
                "model_tag": TAG,
                "model_id": entry["model_id"],
                "model_source": entry["snapshot"],
                "shard": args.shard,
                "n_shards": args.n_shards,
                "cells_sha256": grid_sha,
                "n_cells_total": len(all_cells),
                "rendering": {"renderer": "x106_common.render_chat", "date": date},
                "config": {"temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K, "num_return_sequences": N_RESPONSES, "max_new_tokens": MAX_NEW, "retry_max_new_tokens": RETRY_NEW, "thinking": False, "seed_policy": "sha256(vocab_scan|20261009|qwen|prompt id|condition)"},
                "environment": {"torch": torch.__version__, "transformers": importlib.metadata.version("transformers"), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
                "rows": [done[c["cell_id"]] for c in mine if c["cell_id"] in done],
            },
        )

    todo = [c for c in mine if c["cell_id"] not in done]
    print(f"shard {args.shard}/{args.n_shards}: {len(mine)} cells, {len(todo)} to do", flush=True)
    for index, cell in enumerate(todo, start=1):
        chat = render_chat(tokenizer, cell["prompt"], date)
        assert_boundary(tokenizer, chat, boundary_ids, TAG)
        seed = stable_seed(cell["prompt_id"], cell["condition"])
        texts, lengths, truncated = generate_row(model, tokenizer, eos_ids, chat, MAX_NEW, seed)
        used_retry = any(truncated)
        retry_seed = None
        if used_retry:
            retry_seed = stable_seed(cell["prompt_id"], cell["condition"], retry=True)
            texts, lengths, truncated = generate_row(model, tokenizer, eos_ids, chat, RETRY_NEW, retry_seed)
        if len(texts) != N_RESPONSES or any(not t.strip() for t in texts):
            raise SystemExit(f"FATAL: incomplete/empty generation {cell['cell_id']}")
        done[cell["cell_id"]] = {
            **cell,
            "rendered_chat_sha256": sha256_text(chat),
            "seed": seed,
            "used_retry": used_retry,
            "retry_seed": retry_seed,
            "llm_responses": texts,
            "response_sha256": [sha256_text(t) for t in texts],
            "gen_tokens": lengths,
            "truncated": truncated,
        }
        checkpoint()
        print(f"shard {args.shard}: {index}/{len(todo)} {cell['cell_id']} max_tokens {max(lengths)}", flush=True)
    checkpoint()
    print(f"shard {args.shard}: {len(done)} cells -> {out_path}")


if __name__ == "__main__":
    main()
