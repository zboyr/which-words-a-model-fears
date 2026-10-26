"""x106 Stage 2 (GPU, one (model, arm) per job): sample ten responses per cohort text, at dose 0
or under the boundary write.

The text is the cohort's O / S / C / R string, rendered with the model's renderer (the date
Stage 0a recovered); for O on Hard-1K the rendering is checked against the base generation
row's recorded sha256, so the dosed O arm sees the byte-identical prompt the x104 base label
came from. The write is x105's: the consensus direction through x88's prefill-only
`ExternalBoundaryHook` at the x88 hook block, audited against `output_hidden_states[block + 1]`
before the first row. Sampling is x101's contract; the seed is per (tag, prompt, text) in the
x106 namespace, shared by every dose of that text. Resumable by prompt id from the output file.
"""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path

import numpy as np
import torch

from x81_generate import MAX_NEW, RETRY_NEW, TEMPERATURE, TOP_K, TOP_P, eos_id_set, generate_row, valid_completed
from x106_common import (
    ALL_TAGS,
    IMPORTED_TAG,
    N_RESPONSES,
    ExternalBoundaryHook,
    all_arm_names,
    arm_spec,
    assert_boundary,
    atomic_json,
    audit_write_site,
    directions_path,
    file_record,
    generations_path,
    hard1k_rows,
    hook_block_for,
    hook_site,
    inputs_path,
    is_xstest,
    load_cohort,
    load_mu,
    load_tag_model,
    load_tag_tokenizer,
    n_blocks_of,
    projections_path,
    read_json,
    render_chat,
    sha256_file,
    sha256_text,
    stable_seed,
    units_with_text,
)

N_AUDIT_TEXTS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, choices=ALL_TAGS)
    parser.add_argument("--arm", required=True, choices=all_arm_names())
    parser.add_argument("--inputs", default=str(inputs_path()))
    parser.add_argument("--out")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = arm_spec(args.tag, args.arm)
    text, dose = spec["text"], float(spec["dose"])
    inputs = read_json(args.inputs)
    if inputs.get("experiment") != "x106" or inputs.get("status") != "frozen":
        raise SystemExit("FATAL: invalid x106 inputs")
    entry = inputs["models"][args.tag]
    cohort = load_cohort()
    cohort_sha = cohort["cohort_sha256"]
    units = units_with_text(cohort, text)
    if args.limit:
        units = units[: args.limit]
    expected = {u["id"]: u for u in units}
    out_path = Path(args.out) if args.out else generations_path(args.tag, args.arm)
    stage0a = read_json(projections_path(args.tag))
    if stage0a.get("model_tag") != args.tag:
        raise SystemExit("FATAL: Stage 0a projections are missing for this tag")
    date = stage0a["rendering"]["date"]
    base_sha = {r["id"]: r["rendered_chat_sha256"] for r in hard1k_rows(args.tag)}

    dir_path = directions_path(args.tag)
    if sha256_file(dir_path) != entry["directions"]["sha256"]:
        raise SystemExit(f"FATAL: {args.tag}: direction artifact differs from the frozen inputs")
    mu = load_mu(args.tag, dir_path)

    done: dict[str, dict] = {}
    if out_path.exists():
        prior = read_json(out_path)
        if prior.get("arm") != args.arm or prior.get("model_tag") != args.tag or prior.get("cohort_sha256") != cohort_sha or abs(float(prior.get("signed_dose", 0.0)) - dose) > 1e-12:
            raise SystemExit("FATAL: resume output belongs to another tag/arm/cohort/dose")
        for row in prior.get("rows", []):
            prompt_id = str(row["id"])
            if prompt_id not in expected:
                continue
            if not valid_completed(row, expected[prompt_id]["prompt_sha256"][text], cohort_sha):
                raise SystemExit(f"FATAL: invalid completed resume row {prompt_id}")
            if prompt_id in done:
                raise SystemExit(f"FATAL: duplicate completed resume row {prompt_id}")
            done[prompt_id] = row

    tokenizer = load_tag_tokenizer(entry["snapshot"], args.tag)
    model = load_tag_model(entry["snapshot"], args.tag)
    model.eval()
    n_blocks = n_blocks_of(model)
    block = int(entry["hook_block"])
    if entry["n_blocks"] is not None and n_blocks != int(entry["n_blocks"]):
        raise SystemExit(f"FATAL: {args.tag}: model has {n_blocks} blocks, the inputs record {entry['n_blocks']}")
    if args.tag != IMPORTED_TAG and hook_block_for(n_blocks) != block:
        raise SystemExit(f"FATAL: {args.tag}: block {block} is not the x88 rule on {n_blocks} blocks")
    hidden_dim = int(model.config.get_text_config().hidden_size)
    if hidden_dim != mu.shape[1]:
        raise SystemExit(f"FATAL: {args.tag}: direction dim {mu.shape[1]} != hidden {hidden_dim}")
    site = hook_site(model, block)
    boundary_ids = [int(v) for v in entry["boundary_ids"]]
    eos_ids = eos_id_set(tokenizer, model)

    def render(unit: dict) -> str:
        prompt = unit["prompts"][text]
        if sha256_text(prompt) != unit["prompt_sha256"][text]:
            raise SystemExit(f"FATAL: prompt hash mismatch for {unit['id']}/{text}")
        chat = render_chat(tokenizer, prompt, date)
        if text == "O" and not is_xstest(unit["id"]) and sha256_text(chat) != base_sha[unit["id"]]:
            raise SystemExit(f"FATAL: {args.tag}: O rendering does not reproduce the base row's sha256 at {unit['id']}")
        assert_boundary(tokenizer, chat, boundary_ids, args.tag)
        return chat

    audit = None
    hook = None
    handle = None
    if dose != 0.0:
        audit = audit_write_site(model, tokenizer, site, block, mu, dose, [render(u) for u in units[:N_AUDIT_TEXTS]])
        print(f"x106 {args.tag}/{args.arm}: write-site audit {audit['status']}", flush=True)
        if audit["status"] != "pass":
            atomic_json(out_path.with_name(f"x106_site_audit_FAILED_{args.tag}_{args.arm}.json"), audit)
            raise SystemExit(f"FATAL: {args.tag}/{args.arm}: write-site audit failed")
        hook = ExternalBoundaryHook(dose, torch.from_numpy(mu[0].astype(np.float32)), torch.from_numpy(mu[1].astype(np.float32)))
        handle = site.register_forward_hook(hook)

    def checkpoint() -> None:
        atomic_json(
            out_path,
            {
                "experiment": "x106",
                "stage": 2,
                "model_tag": args.tag,
                "model_id": entry["model_id"],
                "model_source": entry["snapshot"],
                "arm": args.arm,
                "text_condition": text,
                "condition": args.arm,
                "signed_dose": dose,
                "dose_units": inputs["dose_units"],
                "cohort_sha256": cohort_sha,
                "arms_sha256": cohort_sha,
                "inputs": file_record(args.inputs),
                "direction": {"kind": entry["directions_kind"], "artifact": entry["directions"]} if dose != 0.0 else None,
                "site": {"hook_block": block, "state_index": block + 1, "n_blocks": n_blocks, "hidden_dim": hidden_dim, "boundary_ids": boundary_ids, "write": "x88 ExternalBoundaryHook: prefill only, the two trailing prefill positions" if dose != 0.0 else None},
                "site_audit": audit,
                "rendering": {"renderer": "x87_common.format_user_chat", "date": date, "o_checked_against_base_rows": text == "O"},
                "config": {
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "top_k": TOP_K,
                    "num_return_sequences": N_RESPONSES,
                    "max_new_tokens": MAX_NEW,
                    "retry_max_new_tokens": RETRY_NEW,
                    "thinking": False,
                    "seed_policy": "x106 per-(tag, prompt, text) seed, shared by every dose of that text",
                },
                "environment": {"torch": torch.__version__, "transformers": importlib.metadata.version("transformers"), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
                "rows": [done[k] for k in sorted(done)],
            },
        )

    todo = [u for u in units if u["id"] not in done]
    try:
        for index, unit in enumerate(todo, start=1):
            prompt_id = unit["id"]
            chat = render(unit)
            seed = stable_seed(args.tag, prompt_id, text)
            if hook is not None:
                hook.prefill_calls = hook.decode_calls = 0
            texts, lengths, truncated = generate_row(model, tokenizer, eos_ids, chat, MAX_NEW, seed)
            counters = {"prefill_calls": hook.prefill_calls, "decode_calls": hook.decode_calls} if hook is not None else None
            used_retry = any(truncated)
            retry_seed = None
            retry_counters = None
            if used_retry:
                retry_seed = stable_seed(args.tag, prompt_id, text, retry=True)
                if hook is not None:
                    hook.prefill_calls = hook.decode_calls = 0
                texts, lengths, truncated = generate_row(model, tokenizer, eos_ids, chat, RETRY_NEW, retry_seed)
                retry_counters = {"prefill_calls": hook.prefill_calls, "decode_calls": hook.decode_calls} if hook is not None else None
            for name, record in (("first", counters), ("retry", retry_counters)):
                if record is not None and record["prefill_calls"] != 1:
                    raise SystemExit(f"FATAL: {name} write fired on {record['prefill_calls']} prefills for {prompt_id}")
            if len(texts) != N_RESPONSES or any(not value.strip() for value in texts):
                raise SystemExit(f"FATAL: incomplete/empty generation row {prompt_id}")
            done[prompt_id] = {
                "id": prompt_id,
                "condition": args.arm,
                "arm": args.arm,
                "text_condition": text,
                "unit": unit["unit"],
                "prompt": unit["prompts"][text],
                "prompt_sha256": unit["prompt_sha256"][text],
                "rendered_chat_sha256": sha256_text(chat),
                "arms_sha256": cohort_sha,
                "signed_dose": dose,
                "seed": seed,
                "used_retry": used_retry,
                "retry_seed": retry_seed,
                "hook_counters": counters,
                "retry_hook_counters": retry_counters,
                "llm_responses": texts,
                "response_sha256": [sha256_text(value) for value in texts],
                "gen_tokens": lengths,
                "truncated": truncated,
            }
            checkpoint()
            print(f"x106 generate {args.tag}/{args.arm}: {index}/{len(todo)} new", flush=True)
    finally:
        if handle is not None:
            handle.remove()
    checkpoint()
    print(f"x106 generation {args.tag}/{args.arm}: {len(done)} rows at dose {dose:+.2f} -> {out_path}")


if __name__ == "__main__":
    main()
