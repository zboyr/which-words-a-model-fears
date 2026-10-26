"""x111 generation (GPU, Qwen3.5-4B, one shard per job).

Every (prompt, condition) cell of the cross (O, E_w for the 40 words) over the 64 test prompts gets
twenty sampled responses from one seeded sampling call under the x81 / x85 sampling contract
(temperature, top-p, top-k, 2,048 new tokens with a 4,096-token retry of the whole cell when any
response truncates), x111 stable seeds, x106 rendering. Benign O renderings are asserted equal to
x106's recorded rendered-chat sha256. Resumable by cell id.
`--run x117`: the same grid and contract on x117's 64 prompts (data/runs/x117/prompts.json and
words.json, generations under data/runs/x117/generations/), seeds in the x117 namespace; the flag is
read before x111_common is imported, so every per-run path and seed follows it.
`--topup`: the same grid (O + 40 E_w) over the nine harmful top-up prompts (prompts_topup.json) ->
generations/generations_topup_shard{k}of{n}.json; seeds are the same function of (prompt id,
condition), so no top-up cell shares a seed with an original one.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
if "--run" in sys.argv:  # the run namespace must be set before x111_common computes its paths
    os.environ["X111_RUN"] = sys.argv[sys.argv.index("--run") + 1]
from x111_common import GEN_DIR, N_SAMPLES, PROMPTS, PROMPTS_TOPUP, RUN_NAME, TAG, WORDS, atomic_json, cells, cells_sha256, read_json, sha256_text, stable_seed  # noqa: E402
from x81_generate import MAX_NEW, RETRY_NEW, TEMPERATURE, TOP_K, TOP_P, decode_sequences, eos_id_set  # noqa: E402
from x106_common import assert_boundary, inputs_path, load_tag_model, load_tag_tokenizer, projections_path, render_chat  # noqa: E402


def shard_path(shard: int, n_shards: int, topup: bool = False) -> Path:
    return GEN_DIR / (f"generations_topup_shard{shard}of{n_shards}.json" if topup else f"generations_shard{shard}of{n_shards}.json")


def generate_cell(model, tokenizer, eos_ids, chat: str, max_new: int, seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    inputs = tokenizer(chat, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_new, do_sample=True, temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K, num_return_sequences=N_SAMPLES, pad_token_id=tokenizer.pad_token_id)
    return decode_sequences(generated[:, inputs["input_ids"].shape[1] :], tokenizer, eos_ids, max_new)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--topup", action="store_true", help="the harmful top-up prompts (prompts_topup.json)")
    parser.add_argument("--run", default="x111", help="run namespace: x111 (default) or x117; read before the imports, so it switches every per-run path and seed")
    args = parser.parse_args()
    prompts = read_json(PROMPTS_TOPUP if args.topup else PROMPTS)
    all_cells = cells(prompts, read_json(WORDS))
    grid_sha = cells_sha256(all_cells)
    mine = all_cells[args.shard :: args.n_shards]
    out_path = shard_path(args.shard, args.n_shards, args.topup)
    done: dict[str, dict] = {}
    if out_path.exists():
        prior = read_json(out_path)
        if prior.get("cells_sha256") != grid_sha:
            raise SystemExit("FATAL: resume file belongs to another cell grid")
        done = {r["cell_id"]: r for r in prior["rows"]}
    entry = read_json(inputs_path())["models"][TAG]
    if entry["snapshot"] != prompts["model_snapshot"]:
        raise SystemExit("FATAL: x106 inputs snapshot differs from the x106 O_base generations snapshot")
    date = read_json(projections_path(TAG))["rendering"]["date"]
    tokenizer = load_tag_tokenizer(entry["snapshot"], TAG)
    for p in prompts["prompts"]:
        if p["population"] == "benign" and sha256_text(render_chat(tokenizer, p["text"], date)) != p["x106_rendered_chat_sha256"]:
            raise SystemExit(f"FATAL: rendering of {p['id']} differs from x106's")
    print("rendering matches x106 on all benign O cells", flush=True)
    model = load_tag_model(entry["snapshot"], TAG)
    model.eval()
    eos_ids = eos_id_set(tokenizer, model)
    boundary_ids = [int(v) for v in entry["boundary_ids"]]

    def checkpoint() -> None:
        atomic_json(
            out_path,
            {
                "run": RUN_NAME,
                "stage": "topup" if args.topup else "grid",
                "model_tag": TAG,
                "model_id": entry["model_id"],
                "model_source": entry["snapshot"],
                "shard": args.shard,
                "n_shards": args.n_shards,
                "cells_sha256": grid_sha,
                "n_cells_total": len(all_cells),
                "rendering": {"renderer": "x106_common.render_chat", "date": date, "benign_o_cells_match_x106_rendered_chat_sha256": True},
                "config": {"temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K, "num_return_sequences": N_SAMPLES, "max_new_tokens": MAX_NEW, "retry_max_new_tokens": RETRY_NEW, "thinking": False, "seed_policy": f"sha256({RUN_NAME}|111|qwen|prompt id|condition)"},
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
        texts, lengths, truncated = generate_cell(model, tokenizer, eos_ids, chat, MAX_NEW, seed)
        used_retry = any(truncated)
        retry_seed = None
        if used_retry:
            retry_seed = stable_seed(cell["prompt_id"], cell["condition"], retry=True)
            texts, lengths, truncated = generate_cell(model, tokenizer, eos_ids, chat, RETRY_NEW, retry_seed)
        if len(texts) != N_SAMPLES or any(not t.strip() for t in texts):
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
