"""x112 generation (GPU, one model per job).

`--stage pool`: the O cell of every requested held-out candidate (pool_request.json) ->
generations/generations_pool.json. These are the grid's O cells.
`--stage orep`: a second O cell (condition label `Orep`, its own seed) of each of the 32 chosen
held-out prompts -> generations/generations_orep.json, the O replicate independent of selection.
`--stage grid --shard k --n-shards n`: the E_w cells (40 words x 32 held-out prompts) of shard k ->
generations/generations_shard{k}of{n}.json.
Every cell gets twenty sampled responses from one seeded sampling call under the x81 / x85
sampling contract (temperature, top-p, top-k, 2,048 new tokens with a 4,096-token retry of the
whole cell when any response truncates), x112 stable seeds, x106 rendering (the model's Stage 0a
date). Benign O renderings are asserted equal to x106's recorded rendered-chat sha256; harmful O
renderings are compared with the source run's and the match is recorded. Resumable by cell id.
Top-up stages (the harmful prompts added outside the x85 cohort, x112_common): `pooltopup`
(topup_request.json -> generations_pool_topup.json), `oreptopup` (prompts_topup.json ->
generations_orep_topup.json), `gridtopup --shard k --n-shards n` (the E_w cells of the top-up
prompts -> generations_topup_shard{k}of{n}.json). Seeds are the same function of (tag, prompt id,
condition), so a top-up prompt's cells cannot share a seed with any earlier cell.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x112_common import N_SAMPLES, atomic_json, cell, cells, cells_sha256, paths, pool_cells, read_json, sha256_text, stable_seed  # noqa: E402
from x81_generate import MAX_NEW, RETRY_NEW, TEMPERATURE, TOP_K, TOP_P, decode_sequences, eos_id_set  # noqa: E402
from x106_common import assert_boundary, inputs_path, load_tag_model, load_tag_tokenizer, projections_path, render_chat  # noqa: E402


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
    parser.add_argument("--tag", required=True)
    parser.add_argument("--stage", choices=("pool", "orep", "grid", "pooltopup", "oreptopup", "gridtopup"), required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    args = parser.parse_args()
    tag = args.tag
    P_ = paths(tag)
    hist_sha = {}
    topup = args.stage.endswith("topup")
    prompts_path = P_["prompts_topup"] if topup else P_["prompts"]
    if args.stage in ("pool", "pooltopup"):
        req = read_json(P_["topup_request"] if topup else P_["pool_request"])
        mine = pool_cells(req)
        hist_sha = {r["id"]: (r["population"], r["historical_rendered_chat_sha256"]) for r in req["prompts"]}
        out_path = P_["gen"] / ("generations_pool_topup.json" if topup else "generations_pool.json")
        grid_sha = None
    elif args.stage in ("orep", "oreptopup"):
        orep = {"condition": "Orep", "word": None, "kind": "O"}
        mine = [cell(p, orep) for p in read_json(prompts_path)["prompts"]]
        out_path = P_["gen"] / ("generations_orep_topup.json" if topup else "generations_orep.json")
        grid_sha = None
    else:
        all_cells = [c for c in cells(read_json(prompts_path), read_json(P_["words"])) if c["kind"] == "E"]
        grid_sha = cells_sha256(all_cells)
        mine = all_cells[args.shard :: args.n_shards]
        out_path = P_["gen"] / (f"generations_topup_shard{args.shard}of{args.n_shards}.json" if topup else f"generations_shard{args.shard}of{args.n_shards}.json")
    done: dict[str, dict] = {}
    if out_path.exists():
        prior = read_json(out_path)
        if grid_sha is not None and prior.get("cells_sha256") != grid_sha:
            raise SystemExit("FATAL: resume file belongs to another cell grid")
        want = {c["cell_id"]: c["prompt_sha256"] for c in mine}
        for r in prior["rows"]:
            if want.get(r["cell_id"]) == r["prompt_sha256"]:
                done[r["cell_id"]] = r
    todo = [c for c in mine if c["cell_id"] not in done]
    print(f"{tag} {args.stage} shard {args.shard}/{args.n_shards}: {len(mine)} cells, {len(todo)} to do", flush=True)
    if not todo:
        return
    entry = read_json(inputs_path())["models"][tag]
    date = read_json(projections_path(tag))["rendering"]["date"]
    tokenizer = load_tag_tokenizer(entry["snapshot"], tag)
    rendering_match = {}
    for c in mine:
        if c["prompt_id"] in hist_sha:
            pop, sha = hist_sha[c["prompt_id"]]
            ok = sha256_text(render_chat(tokenizer, c["prompt"], date)) == sha
            if pop == "benign" and not ok:
                raise SystemExit(f"FATAL: rendering of {c['prompt_id']} differs from x106's")
            rendering_match[c["prompt_id"]] = ok
    model = load_tag_model(entry["snapshot"], tag)
    model.eval()
    eos_ids = eos_id_set(tokenizer, model)
    boundary_ids = [int(v) for v in entry["boundary_ids"]]

    def checkpoint() -> None:
        atomic_json(
            out_path,
            {
                "run": "x112",
                "stage": args.stage,
                "model_tag": tag,
                "model_id": entry["model_id"],
                "model_source": entry["snapshot"],
                "shard": args.shard,
                "n_shards": args.n_shards,
                "cells_sha256": grid_sha,
                "rendering": {"renderer": "x106_common.render_chat", "date": date, "o_matches_source_rendered_chat_sha256": rendering_match or None},
                "config": {"temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K, "num_return_sequences": N_SAMPLES, "max_new_tokens": MAX_NEW, "retry_max_new_tokens": RETRY_NEW, "thinking": False, "seed_policy": f"sha256(x112|112|{tag}|prompt id|condition)"},
                "environment": {"torch": torch.__version__, "transformers": importlib.metadata.version("transformers"), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
                "rows": [done[c["cell_id"]] for c in mine if c["cell_id"] in done],
            },
        )

    for index, c in enumerate(todo, start=1):
        chat = render_chat(tokenizer, c["prompt"], date)
        assert_boundary(tokenizer, chat, boundary_ids, tag)
        seed = stable_seed(tag, c["prompt_id"], c["condition"])
        texts, lengths, truncated = generate_cell(model, tokenizer, eos_ids, chat, MAX_NEW, seed)
        used_retry = any(truncated)
        retry_seed = None
        if used_retry:
            retry_seed = stable_seed(tag, c["prompt_id"], c["condition"], retry=True)
            texts, lengths, truncated = generate_cell(model, tokenizer, eos_ids, chat, RETRY_NEW, retry_seed)
        if len(texts) != N_SAMPLES or any(not t.strip() for t in texts):
            # An empty response is kept as generated; the judge sees it as is.
            print(f"note: {c['cell_id']} has {sum(not t.strip() for t in texts)} empty responses", flush=True)
        done[c["cell_id"]] = {**c, "rendered_chat_sha256": sha256_text(chat), "seed": seed, "used_retry": used_retry, "retry_seed": retry_seed, "llm_responses": texts, "response_sha256": [sha256_text(t) for t in texts], "gen_tokens": lengths, "truncated": truncated}
        checkpoint()
        print(f"{tag} {args.stage} {args.shard}: {index}/{len(todo)} {c['cell_id']} max_tokens {max(lengths)}", flush=True)
    checkpoint()
    print(f"{tag} {args.stage} {args.shard}: {len(done)} cells -> {out_path}")


if __name__ == "__main__":
    main()
