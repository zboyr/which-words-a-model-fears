"""x105 Stage 2 (GPU, one (model, arm) per job): generate under the boundary write.

One arm is one (text condition, direction, signed dose). The text is the x101 cohort's O or E
string, rendered so that its sha256 equals the one the base generation row recorded — the
prompt entering the model is byte-identical to the prompt the dose-0 responses came from. The
write is x88's prefill-only `ExternalBoundaryHook` on the x88 hook block, audited against
`output_hidden_states[block + 1]` before the first row is generated. Sampling is x101's
contract and x101's own per-(tag, prompt, condition) seed, so each dosed arm shares its base
arm's random-number stream. Resumable by prompt id from the output file.
"""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path

import numpy as np
import torch

from x81_generate import MAX_NEW, RETRY_NEW, TEMPERATURE, TOP_K, TOP_P, eos_id_set, generate_row, valid_completed
from x105_common import (
    ALL_TAGS,
    ARM_NAMES,
    IMPORTED_TAG,
    N_RESPONSES,
    audit_write_site,
    arm_spec,
    atomic_json,
    base_cohort_sha,
    consensus_x86_vectors,
    control_directions_path,
    directions_path,
    file_record,
    generation_seed,
    generations_path,
    hook_block_for,
    hook_site,
    inputs_path,
    kl_calibration_path,
    load_cohort,
    load_tag_model,
    load_tag_tokenizer,
    n_blocks_of,
    read_json,
    render_like_x101,
    resolve_arm_vectors,
    sha256_file,
    sha256_text,
    ExternalBoundaryHook,
    assert_boundary,
)

N_AUDIT_TEXTS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, choices=ALL_TAGS)
    parser.add_argument("--arm", required=True, choices=ARM_NAMES)
    parser.add_argument("--inputs", default=str(inputs_path()))
    parser.add_argument("--out")
    parser.add_argument("--limit", type=int, default=0, help="debug: only the first N cohort units")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = arm_spec(args.arm)
    text_condition = spec["text"]
    inputs = read_json(args.inputs)
    if inputs.get("experiment") != "x105" or inputs.get("status") != "frozen":
        raise SystemExit("FATAL: invalid x105 inputs")
    entry = inputs["models"][args.tag]
    cohort = load_cohort()
    if cohort["cohort_sha256"] != inputs["cohort_sha256"]:
        raise SystemExit("FATAL: the cohort differs from the frozen inputs")
    cohort_sha = cohort["cohort_sha256"]
    base_sha = base_cohort_sha(args.tag, cohort)
    units = list(cohort["units"])
    if args.limit:
        units = units[: args.limit]
    expected_units = {int(unit["id"]): unit for unit in units}
    out_path = Path(args.out) if args.out else generations_path(args.tag, args.arm)

    dir_path = directions_path(args.tag)
    if sha256_file(dir_path) != entry["directions"]["sha256"]:
        raise SystemExit(f"FATAL: {args.tag}: direction artifact differs from the frozen inputs")
    control_path = control_directions_path(args.tag)
    if sha256_file(control_path) != entry["control_directions"]["sha256"]:
        raise SystemExit(f"FATAL: {args.tag}: control directions differ from the frozen inputs")
    minus1, tbg = consensus_x86_vectors(dir_path)
    mu = np.stack([minus1.numpy(), tbg.numpy()]).astype(np.float64)
    calibration = None
    calibration_record = None
    if spec["family"] == "random_kl":
        calibration_path = kl_calibration_path(args.tag)
        calibration = read_json(calibration_path)
        if calibration.get("model_tag") != args.tag or calibration.get("status") != "pass":
            raise SystemExit(f"FATAL: {args.tag}: benign-KL calibration is missing or did not pass")
        calibration_record = file_record(calibration_path)
    vectors, dose, direction_meta = resolve_arm_vectors(args.arm, mu, control_path, calibration)

    done: dict[int, dict] = {}
    if out_path.exists():
        prior = read_json(out_path)
        if (
            prior.get("arm") != args.arm
            or prior.get("model_tag") != args.tag
            or prior.get("cohort_sha256") != cohort_sha
            or abs(float(prior.get("signed_dose", 0.0)) - dose) > 1e-12
        ):
            raise SystemExit("FATAL: resume output belongs to another tag/arm/cohort/dose")
        for row in prior.get("rows", []):
            prompt_id = int(row["id"])
            if prompt_id not in expected_units:
                continue
            if not valid_completed(row, expected_units[prompt_id]["prompt_sha256"][text_condition], cohort_sha):
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
    if hidden_dim != mu.shape[1] or vectors.shape[1] != hidden_dim:
        raise SystemExit(f"FATAL: {args.tag}: direction dim {vectors.shape[1]} != hidden {hidden_dim}")
    site = hook_site(model, block)
    boundary_ids = [int(v) for v in entry["boundary_ids"]]
    expected_chat = entry["rendered_chat_sha256"][text_condition]
    dates = list(entry.get("date_candidates") or [])
    eos_ids = eos_id_set(tokenizer, model)

    def render(unit: dict) -> str:
        prompt = unit["prompts"][text_condition]
        if sha256_text(prompt) != unit["prompt_sha256"][text_condition]:
            raise SystemExit(f"FATAL: prompt hash mismatch for {unit['id']}/{text_condition}")
        chat, _ = render_like_x101(tokenizer, prompt, expected_chat[str(int(unit["id"]))], dates)
        assert_boundary(tokenizer, chat, boundary_ids, args.tag)
        return chat

    audit = audit_write_site(model, tokenizer, site, block, vectors, dose, [render(u) for u in units[:N_AUDIT_TEXTS]])
    print(f"x105 {args.tag}/{args.arm}: write-site audit {audit['status']}", flush=True)
    if audit["status"] != "pass":
        atomic_json(out_path.with_name(f"x105_site_audit_FAILED_{args.tag}_{args.arm}.json"), audit)
        raise SystemExit(f"FATAL: {args.tag}/{args.arm}: write-site audit failed")

    hook = ExternalBoundaryHook(
        dose,
        torch.from_numpy(np.asarray(vectors[0], dtype=np.float32)),
        torch.from_numpy(np.asarray(vectors[1], dtype=np.float32)),
    )
    handle = site.register_forward_hook(hook)

    def checkpoint() -> None:
        atomic_json(
            out_path,
            {
                "experiment": "x105",
                "stage": 2,
                "model_tag": args.tag,
                "model_id": entry["model_id"],
                "model_source": entry["snapshot"],
                "arm": args.arm,
                "text_condition": text_condition,
                "condition": args.arm,  # the field the shared x81/x83 judge and validators read
                "family": spec["family"],
                "control_draw": spec["draw"],
                "signed_dose": float(dose),
                "dose_units": inputs["dose_units"],
                "cohort_sha256": cohort_sha,
                "arms_sha256": cohort_sha,
                "base_arms_sha256": base_sha,
                "inputs": file_record(args.inputs),
                "kl_calibration": calibration_record,
                "direction": {**direction_meta, "artifact": entry["directions"]},
                "site": {
                    "hook_block": block,
                    "state_index": block + 1,
                    "n_blocks": n_blocks,
                    "hidden_dim": hidden_dim,
                    "boundary_ids": boundary_ids,
                    "write": "x88 ExternalBoundaryHook: prefill only, the two trailing prefill positions",
                },
                "site_audit": audit,
                "config": {
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "top_k": TOP_K,
                    "num_return_sequences": N_RESPONSES,
                    "max_new_tokens": MAX_NEW,
                    "retry_max_new_tokens": RETRY_NEW,
                    "thinking": False,
                    "seed_policy": (
                        "the base arm's own x101 (x83 for Qwen3.5-4B) per-(tag, prompt, text condition) "
                        "seed, shared by every dose of that text: common random numbers"
                    ),
                },
                "environment": {
                    "torch": torch.__version__,
                    "transformers": importlib.metadata.version("transformers"),
                    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                },
                "rows": [done[prompt_id] for prompt_id in sorted(done)],
            },
        )

    todo = [unit for unit in units if int(unit["id"]) not in done]
    try:
        for index, unit in enumerate(todo, start=1):
            prompt_id = int(unit["id"])
            chat = render(unit)
            seed = generation_seed(args.tag, prompt_id, text_condition)
            hook.prefill_calls = hook.decode_calls = 0
            texts, lengths, truncated = generate_row(model, tokenizer, eos_ids, chat, MAX_NEW, seed)
            counters = {"prefill_calls": hook.prefill_calls, "decode_calls": hook.decode_calls}
            used_retry = any(truncated)
            retry_seed = None
            retry_counters = None
            if used_retry:
                retry_seed = generation_seed(args.tag, prompt_id, text_condition, retry=True)
                hook.prefill_calls = hook.decode_calls = 0
                texts, lengths, truncated = generate_row(model, tokenizer, eos_ids, chat, RETRY_NEW, retry_seed)
                retry_counters = {"prefill_calls": hook.prefill_calls, "decode_calls": hook.decode_calls}
            for name, record in (("first", counters), ("retry", retry_counters)):
                if record is not None and record["prefill_calls"] != 1:
                    raise SystemExit(f"FATAL: {name} write fired on {record['prefill_calls']} prefills for {prompt_id}")
            if len(texts) != N_RESPONSES or any(not value.strip() for value in texts):
                raise SystemExit(f"FATAL: incomplete/empty generation row {prompt_id}")
            done[prompt_id] = {
                "id": prompt_id,
                "condition": args.arm,
                "arm": args.arm,
                "text_condition": text_condition,
                "word": unit["word"],
                "prompt": unit["prompts"][text_condition],
                "prompt_sha256": unit["prompt_sha256"][text_condition],
                "rendered_chat_sha256": sha256_text(chat),
                "arms_sha256": cohort_sha,
                "signed_dose": float(dose),
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
            print(f"x105 generate {args.tag}/{args.arm}: {index}/{len(todo)} new", flush=True)
    finally:
        handle.remove()
    checkpoint()
    print(f"x105 generation {args.tag}/{args.arm}: {len(done)} rows at dose {dose:+.3f} -> {out_path}")


if __name__ == "__main__":
    main()
