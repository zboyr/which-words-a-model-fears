"""x114 Stage 5 (GPU, one model per job): x103's boundary-state read on the x114 texts.

288 forward passes per model (72 prompts x O/S/M/E), the block output at the two trailing
prefill positions read through a forward hook on the decoder block the x88 boundary write
hooks, with x103's own in-job site audit (the read tensor is `output_hidden_states[block+1]`
at those positions and moves by exactly `dose * mu` when the write hook is registered ahead
of the reader). Per-(prompt, position) projections, shift norms and fp16 boundary deltas,
in x103's own record layout, so `x103_analyze` can pool the two cohorts.
"""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path

import numpy as np
import torch

from x103_extract import audit_site
from x103_common import (
    BOUNDARY_POSITIONS,
    N_AUDIT_UNITS,
    assert_boundary,
    consensus_x86_vectors,
    directions_path,
    hook_site,
    load_tag_model,
    n_blocks_of,
    projection_record,
    render_like_x101,
    unit_directions,
    BoundaryReader,
    hook_block_for,
)
from x105_common import load_tag_tokenizer
from x114_common import (
    CONDITIONS,
    IMPORTED_TAG,
    POSITIONS,
    TAGS,
    atomic_json,
    deltas_path,
    file_record,
    inputs_path,
    load_cohort,
    projections_path,
    read_json,
    sha256_file,
    sha256_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, choices=TAGS)
    parser.add_argument("--inputs", default=str(inputs_path()))
    parser.add_argument("--limit", type=int, default=0, help="debug: only the first N cohort units")
    return parser.parse_args()


@torch.inference_mode()
def forward(model, encoded, **kwargs):
    return model(**encoded, use_cache=False, **kwargs)


def main() -> None:
    args = parse_args()
    inputs = read_json(args.inputs)
    if inputs.get("experiment") != "x114" or inputs.get("status") != "frozen":
        raise SystemExit("FATAL: invalid x114 inputs")
    tag = args.tag
    entry = inputs["models"][tag]
    cohort = load_cohort()
    if cohort["cohort_sha256"] != inputs["cohort_sha256"]:
        raise SystemExit("FATAL: the cohort differs from the frozen inputs")
    units = list(cohort["units"])
    if args.limit:
        units = units[: args.limit]

    dir_path = Path(entry["directions"]["path"])
    if not dir_path.exists():
        dir_path = directions_path(tag)
    if sha256_file(dir_path) != entry["directions"]["sha256"]:
        raise SystemExit(f"FATAL: {tag}: direction artifact sha256 differs from the frozen inputs")
    minus1, tbg = consensus_x86_vectors(dir_path)
    mu = np.stack([minus1.numpy(), tbg.numpy()]).astype(np.float64)
    _, mu_norms = unit_directions(mu)

    tokenizer = load_tag_tokenizer(entry["snapshot"], tag)
    model = load_tag_model(entry["snapshot"], tag)
    model.eval()
    n_blocks = n_blocks_of(model)
    block = int(entry["hook_block"])
    if entry["n_blocks"] is not None and n_blocks != int(entry["n_blocks"]):
        raise SystemExit(f"FATAL: {tag}: model has {n_blocks} blocks, the inputs record {entry['n_blocks']}")
    if tag != IMPORTED_TAG and hook_block_for(n_blocks) != block:
        raise SystemExit(f"FATAL: {tag}: block {block} is not the x88 rule on {n_blocks} blocks")
    hidden_dim = int(model.config.get_text_config().hidden_size)
    if hidden_dim != mu.shape[1]:
        raise SystemExit(f"FATAL: {tag}: direction dim {mu.shape[1]} != hidden {hidden_dim}")
    site = hook_site(model, block)
    boundary_ids = [int(v) for v in entry["boundary_ids"]]
    expected = entry["rendered_chat_sha256"]
    dates = list(entry.get("date_candidates") or [])

    audit_chats = []
    for unit in units[:N_AUDIT_UNITS]:
        for condition in ("O", "E"):
            chat, _ = render_like_x101(tokenizer, unit["prompts"][condition], expected[condition][str(unit["id"])], dates)
            assert_boundary(tokenizer, chat, boundary_ids, tag)
            audit_chats.append(chat)
    audit = audit_site(model, tokenizer, site, block, minus1, tbg, audit_chats)
    print(f"x114 extract {tag}: site audit {audit['status']}", flush=True)
    if audit["status"] != "pass":
        atomic_json(projections_path(tag).with_name("x114_site_audit_FAILED.json"), audit)
        raise SystemExit(f"FATAL: {tag}: hook-site audit failed")

    reader = BoundaryReader()
    handle = site.register_forward_hook(reader)
    states = {c: np.empty((len(units), 2, hidden_dim), dtype=np.float32) for c in CONDITIONS}
    row_meta = []
    dates_used: dict[str, int] = {}
    for index, unit in enumerate(units):
        prompt_id = int(unit["id"])
        meta = {"id": prompt_id, "word": unit["word"], "category": unit["category"], "n_input_tokens": {}}
        for condition in CONDITIONS:
            prompt = unit["prompts"][condition]
            if sha256_text(prompt) != unit["prompt_sha256"][condition]:
                raise SystemExit(f"FATAL: text hash mismatch for {prompt_id}/{condition}")
            chat, date_used = render_like_x101(tokenizer, prompt, expected[condition][str(prompt_id)], dates)
            if date_used is not None:
                dates_used[date_used] = dates_used.get(date_used, 0) + 1
            assert_boundary(tokenizer, chat, boundary_ids, tag)
            encoded = tokenizer(chat, return_tensors="pt", add_special_tokens=False).to(model.device)
            calls = reader.calls
            forward(model, encoded)
            if reader.calls != calls + 1 or reader.last.shape != (1, 2, hidden_dim):
                raise SystemExit(f"FATAL: reader fired {reader.calls - calls} times for {prompt_id}/{condition}")
            states[condition][index] = reader.last[0].numpy()
            meta["n_input_tokens"][condition] = int(encoded["input_ids"].shape[1])
        row_meta.append(meta)
        if index % 10 == 0 or index == len(units) - 1:
            print(f"x114 extract {tag}: {index + 1}/{len(units)} prompts", flush=True)
    handle.remove()

    deltas = np.stack([states[p] - states["O"] for p in POSITIONS], axis=1)  # [n, 3, 2, hidden]
    rows = []
    for index, meta in enumerate(row_meta):
        o_proj = np.einsum("td,td->t", states["O"][index].astype(np.float64), mu / mu_norms[:, None])
        for k, position in enumerate(POSITIONS):
            record = projection_record(deltas[index, k], mu)
            rows.append(
                {
                    "id": meta["id"],
                    "word": meta["word"],
                    "category": meta["category"],
                    "position": position,
                    **record,
                    "o_proj_by_token": {p: float(v) for p, v in zip(BOUNDARY_POSITIONS, o_proj)},
                    "o_norm_by_token": {
                        p: float(v) for p, v in zip(BOUNDARY_POSITIONS, np.linalg.norm(states["O"][index], axis=1))
                    },
                    "n_input_tokens": {"O": meta["n_input_tokens"]["O"], position: meta["n_input_tokens"][position]},
                }
            )

    out_deltas = deltas_path(tag)
    out_deltas.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_deltas,
        deltas=deltas.astype(np.float16),
        ids=np.asarray([m["id"] for m in row_meta], dtype=np.int64),
        positions=np.asarray(POSITIONS),
        boundary_positions=np.asarray(BOUNDARY_POSITIONS),
        mu=mu.astype(np.float32),
    )
    fp16_max_rel_error = float(
        np.max(np.abs(deltas.astype(np.float16).astype(np.float32) - deltas)) / (np.abs(deltas).max() + 1e-12)
    )
    atomic_json(
        projections_path(tag),
        {
            "experiment": "x114",
            "stage": 5,
            "model_tag": tag,
            "model_id": entry["model_id"],
            "model_source": entry["snapshot"],
            "cohort_sha256": cohort["cohort_sha256"],
            "inputs": file_record(args.inputs),
            "n_units": len(units),
            "site": {
                "hook_block": block,
                "state_index": block + 1,
                "n_blocks": n_blocks,
                "relative_depth_rule_block": hook_block_for(n_blocks),
                "hidden_dim": hidden_dim,
                "boundary_ids": boundary_ids,
                "read": "forward hook on the decoder block the x88 boundary write hooks; block output at the last two prefill positions",
                "batch_size": 1,
            },
            "directions": {
                **entry["directions"],
                "kind": entry["directions_kind"],
                "norms": {p: float(v) for p, v in zip(BOUNDARY_POSITIONS, mu_norms)},
            },
            "rendering": {
                "renderer": "x87_common.format_user_chat, verified against every x114 generation row's rendered_chat_sha256",
                "date_strings_recovered": dates_used,
            },
            "site_audit": audit,
            "deltas": {**file_record(out_deltas), "dtype": "float16", "fp16_max_relative_error": fp16_max_rel_error},
            "environment": {
                "torch": torch.__version__,
                "transformers": importlib.metadata.version("transformers"),
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            },
            "rows": rows,
        },
    )
    print(f"x114 extract {tag}: {len(rows)} cells -> {projections_path(tag)}; deltas -> {out_deltas}")


if __name__ == "__main__":
    main()
