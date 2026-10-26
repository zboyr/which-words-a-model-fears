"""x106 Stage 0a: freeze the inputs (local, `--prepare`), then read every prompt's boundary
state on one model (GPU, `--tag`).

`--prepare` writes data/runs/x106/x106_inputs.json from local artifacts: per model the x87 /
x88 snapshot, the x88 hook block and boundary ids, the direction artifact and its sha256, the
base generation file whose Hard-1K rows fix the prompt texts and their rendered-chat sha256,
the x104 label file, and the arm table.

`--tag T` renders the 1,319 Hard-1K prompts exactly as that model's base generation rows
record them (sha256 checked per row; Llama's stamped run date recovered from the first row)
and the 100 XSTest prompts the same way, audits the read site against the x88 write hook,
and reads the block output at the two trailing prefill positions, one prompt per forward.
Writes the fp16 states and the per-prompt projection on the model's consensus direction.
"""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path

import numpy as np
import torch

from x103_extract import audit_site
from x106_common import (
    ALL_TAGS,
    BOUNDARY_POSITIONS,
    IMPORTED_TAG,
    MODELS,
    QWEN4B_HOOK_BLOCK,
    QWEN4B_MODEL_ID,
    X102_XSTEST,
    XSTEST_TYPES,
    X104_LABELS,
    X86_EXPECTED_BOUNDARY_IDS,
    X88_MANIFEST,
    BoundaryReader,
    all_prompts,
    arms_for,
    assert_boundary,
    atomic_json,
    base_generations_path,
    directions_path,
    doses_for,
    file_record,
    hard1k_rows,
    hook_block_for,
    hook_site,
    inputs_path,
    is_xstest,
    load_mu,
    load_tag_model,
    load_tag_tokenizer,
    n_blocks_of,
    projection_of,
    projections_path,
    read_json,
    recover_date,
    render_chat,
    sha256_file,
    sha256_text,
    states_path,
    unit_directions,
    xstest_prompts,
)

N_AUDIT_PROMPTS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--tag", choices=ALL_TAGS)
    parser.add_argument("--inputs", default=str(inputs_path()))
    parser.add_argument("--limit", type=int, default=0, help="debug: only the first N prompts")
    return parser.parse_args()


def prepare() -> None:
    x88_manifest = read_json(X88_MANIFEST)
    if x88_manifest.get("status") != "frozen":
        raise SystemExit("FATAL: the x88 manifest is not frozen")
    models = {}
    for tag in ALL_TAGS:
        rows = hard1k_rows(tag)
        base = read_json(base_generations_path(tag))
        if tag == IMPORTED_TAG:
            if base["model_id"] != QWEN4B_MODEL_ID:
                raise SystemExit("FATAL: the x87 base file is not Qwen3.5-4B's")
            record = {
                "model_id": QWEN4B_MODEL_ID,
                "snapshot": base["model_source"],
                "revision": Path(base["model_source"]).name,
                "n_blocks": None,
                "hook_block": int(QWEN4B_HOOK_BLOCK),
                "boundary_ids": list(X86_EXPECTED_BOUNDARY_IDS),
                "directions_kind": "x87 consensus of the four x86 fold directions (Qwen3.5-4B)",
                "base_source": "x87 base generations (greedy), judged by GuardReasoner-8B in x104",
                "role": "pipeline reference",
            }
        else:
            entry = x88_manifest["models"][tag]
            if entry["model_id"] != MODELS[tag]["model_id"] or entry["snapshot"] != base["model_source"]:
                raise SystemExit(f"FATAL: {tag}: the x88 manifest and base file disagree on the snapshot")
            if hook_block_for(int(entry["n_blocks"])) != int(entry["hook_block"]):
                raise SystemExit(f"FATAL: {tag}: the x88 hook block is not the frozen relative-depth rule")
            record = {
                "model_id": MODELS[tag]["model_id"],
                "snapshot": entry["snapshot"],
                "revision": entry.get("revision"),
                "n_blocks": int(entry["n_blocks"]),
                "hook_block": int(entry["hook_block"]),
                "boundary_ids": [int(v) for v in entry["boundary_ids"]],
                "directions_kind": "x88 consensus (mean of four fold directions) at the x88 hook block",
                "base_source": "x88 base generations (greedy), judged by GuardReasoner-8B in x104",
                "role": "transfer model",
            }
        record["state_index"] = record["hook_block"] + 1
        record["directions"] = file_record(directions_path(tag))
        mu = load_mu(tag)
        _, norms = unit_directions(mu)
        record["direction_norms"] = {p: float(v) for p, v in zip(BOUNDARY_POSITIONS, norms)}
        record["hidden_dim"] = int(mu.shape[1])
        record["base_generations"] = file_record(base_generations_path(tag))
        record["n_hard1k"] = len(rows)
        record["hard1k_ids_sha256"] = sha256_text("\n".join(r["id"] for r in rows))
        record["hard1k_prompts_sha256"] = sha256_text("\n".join(r["prompt_sha256"] for r in rows))
        record["doses"] = list(doses_for(tag))
        record["arms"] = arms_for(tag)
        models[tag] = record
    shared = {m["hard1k_prompts_sha256"] for m in models.values()}
    if len(shared) != 1:
        raise SystemExit("FATAL: the eight base files do not carry one Hard-1K prompt set")
    xstest = xstest_prompts()
    out = {
        "experiment": "x106",
        "stage": 0,
        "status": "frozen",
        "question": (
            "On the Hard-1K prompts a model refuses although GuardReasoner reads the request as "
            "unharmful, does the insertion-fitted boundary direction nominate the word that carries "
            "the refusal, and does replacing that word in place remove it?"
        ),
        "hard1k_prompts_sha256": shared.pop(),
        "xstest": {**file_record(X102_XSTEST), "types": list(XSTEST_TYPES), "n": len(xstest), "ids_sha256": sha256_text("\n".join(r["id"] for r in xstest))},
        "x104_labels": file_record(X104_LABELS),
        "x88_manifest": file_record(X88_MANIFEST),
        "projection": "sum over the two trailing prefill tokens of <h, mu/|mu|>, h the block output at the x88 write site",
        "dose_units": "multiples of the model's own mean boundary shift (the raw consensus vector)",
        "models": models,
    }
    atomic_json(inputs_path(), out)
    print(f"x106 inputs -> {inputs_path()}: {len(models)} models")
    for tag, record in models.items():
        print(f"  {tag:8s} block {record['hook_block']:2d} dim {record['hidden_dim']:5d} doses {record['doses']}")


@torch.inference_mode()
def forward(model, encoded, **kwargs):
    return model(**encoded, use_cache=False, **kwargs)


def project(tag: str, inputs_file: Path, limit: int) -> None:
    inputs = read_json(inputs_file)
    if inputs.get("experiment") != "x106" or inputs.get("status") != "frozen":
        raise SystemExit("FATAL: invalid x106 inputs")
    entry = inputs["models"][tag]
    prompts = all_prompts(tag)
    if sha256_text("\n".join(r["prompt_sha256"] for r in prompts if not is_xstest(r["id"]))) != entry["hard1k_prompts_sha256"]:
        raise SystemExit(f"FATAL: {tag}: the Hard-1K prompts differ from the frozen inputs")
    if limit:
        prompts = prompts[:limit]
    dir_path = directions_path(tag)
    if sha256_file(dir_path) != entry["directions"]["sha256"]:
        raise SystemExit(f"FATAL: {tag}: direction artifact differs from the frozen inputs")
    mu = load_mu(tag, dir_path)
    minus1 = torch.from_numpy(mu[0].astype(np.float32))
    tbg = torch.from_numpy(mu[1].astype(np.float32))
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

    first = next(r for r in prompts if not is_xstest(r["id"]))
    date = recover_date(tokenizer, first["prompt"], first["rendered_chat_sha256"])
    print(f"x106 {tag}: renderer date {date!r}", flush=True)

    def render(row: dict) -> str:
        chat = render_chat(tokenizer, row["prompt"], date)
        if row["rendered_chat_sha256"] is not None and sha256_text(chat) != row["rendered_chat_sha256"]:
            raise SystemExit(f"FATAL: {tag}: rendered chat does not reproduce the base row's sha256 at {row['id']}")
        assert_boundary(tokenizer, chat, boundary_ids, tag)
        return chat

    audit = audit_site(model, tokenizer, site, block, minus1, tbg, [render(r) for r in prompts[:N_AUDIT_PROMPTS]])
    print(f"x106 {tag}: site audit {audit['status']}", flush=True)
    if audit["status"] != "pass":
        atomic_json(projections_path(tag).with_name("x106_site_audit_FAILED.json"), audit)
        raise SystemExit(f"FATAL: {tag}: hook-site audit failed")

    reader = BoundaryReader()
    handle = site.register_forward_hook(reader)
    states = np.empty((len(prompts), 2, hidden_dim), dtype=np.float32)
    n_tokens = []
    for index, row in enumerate(prompts):
        chat = render(row)
        encoded = tokenizer(chat, return_tensors="pt", add_special_tokens=False).to(model.device)
        calls = reader.calls
        forward(model, encoded)
        if reader.calls != calls + 1 or reader.last.shape != (1, 2, hidden_dim):
            raise SystemExit(f"FATAL: reader fired {reader.calls - calls} times for {row['id']}")
        states[index] = reader.last[0].numpy()
        n_tokens.append(int(encoded["input_ids"].shape[1]))
        if index % 100 == 0 or index == len(prompts) - 1:
            print(f"x106 {tag}: {index + 1}/{len(prompts)} prompts", flush=True)
    handle.remove()

    proj = projection_of(states, mu)
    per_token = np.einsum("ntd,td->nt", states.astype(np.float64), mu / mu_norms[:, None])
    norms = np.linalg.norm(states.astype(np.float64), axis=2)
    rows = []
    for index, row in enumerate(prompts):
        rows.append(
            {
                "id": row["id"],
                "source": row["source"],
                "category": row["category"],
                "prompt_sha256": row["prompt_sha256"],
                "n_input_tokens": n_tokens[index],
                "proj": float(proj[index]),
                "proj_by_token": {p: float(v) for p, v in zip(BOUNDARY_POSITIONS, per_token[index])},
                "norm_by_token": {p: float(v) for p, v in zip(BOUNDARY_POSITIONS, norms[index])},
            }
        )
    out_states = states_path(tag)
    out_states.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_states,
        states=states.astype(np.float16),
        ids=np.asarray([r["id"] for r in prompts]),
        boundary_positions=np.asarray(BOUNDARY_POSITIONS),
        mu=mu.astype(np.float32),
    )
    fp16_err = float(np.max(np.abs(states.astype(np.float16).astype(np.float32) - states)) / (np.abs(states).max() + 1e-12))
    out = {
        "experiment": "x106",
        "stage": "0a",
        "model_tag": tag,
        "model_id": entry["model_id"],
        "model_source": entry["snapshot"],
        "inputs": file_record(inputs_file),
        "n_prompts": len(prompts),
        "site": {
            "hook_block": block,
            "state_index": block + 1,
            "n_blocks": n_blocks,
            "hidden_dim": hidden_dim,
            "boundary_ids": boundary_ids,
            "read": "forward hook on the decoder block the x88 boundary write hooks; block output at the last two prefill positions",
            "batch_size": 1,
        },
        "directions": {**entry["directions"], "kind": entry["directions_kind"], "norms": {p: float(v) for p, v in zip(BOUNDARY_POSITIONS, mu_norms)}},
        "rendering": {"renderer": "x87_common.format_user_chat, verified against every Hard-1K base row's rendered_chat_sha256", "date": date},
        "site_audit": audit,
        "states": {**file_record(out_states), "dtype": "float16", "fp16_max_relative_error": fp16_err},
        "environment": {
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
        "rows": rows,
    }
    atomic_json(projections_path(tag), out)
    print(f"x106 {tag}: {len(rows)} projections -> {projections_path(tag)}; states -> {out_states}")


def main() -> None:
    args = parse_args()
    if args.prepare:
        prepare()
        return
    if not args.tag:
        raise SystemExit("--tag or --prepare is required")
    project(args.tag, Path(args.inputs), args.limit)


if __name__ == "__main__":
    main()
