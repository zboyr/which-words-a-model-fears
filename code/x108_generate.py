"""x108 Stage 2 (GPU, one (model, arm) per job): sample ten responses per cohort text on one
model at dose 0, no boundary write.

The text is the cohort's O / S / R string, rendered with x106's renderer (the date Stage 0a
recovered); for O on Toxic the rendering is checked against the model's x87 / x88 base row's
recorded sha256. Sampling is x101's contract (T = 1.0, top-p 0.9, top-k 50, 10 sequences, 2,048
tokens, one 4,096 retry); the seed is per (model, prompt, text) in the x108 namespace.
Resumable by prompt id.
"""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path

import torch

from x81_generate import MAX_NEW, RETRY_NEW, TEMPERATURE, TOP_K, TOP_P, eos_id_set, generate_row, valid_completed
from x106_common import assert_boundary, render_chat
from x108_common import ALL_TAGS, ARMS, N_RESPONSES, TAG, arm_spec, atomic_json, file_record, generations_path, is_toxic, load_cohort, load_model, model_entry, projections_path, read_json, sha256_text, stable_seed, toxic_rows, units_with_text, x106_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=TAG, choices=ALL_TAGS)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--out")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tag = args.tag
    spec = arm_spec(args.arm)
    text = spec["text"]
    entry = model_entry(tag)
    cohort = load_cohort()
    cohort_sha = cohort["cohort_sha256"]
    units = units_with_text(cohort, text)
    if args.limit:
        units = units[: args.limit]
    expected = {u["id"]: u for u in units}
    out_path = Path(args.out) if args.out else generations_path(tag, args.arm)
    stage0a = read_json(projections_path(tag))
    if stage0a.get("experiment") != "x108" or stage0a.get("model_tag") != tag:
        raise SystemExit("FATAL: Stage 0a projections are missing")
    date = stage0a["rendering"]["date"]
    base_sha = {r["id"]: r["rendered_chat_sha256"] for r in toxic_rows(tag)}

    done: dict[str, dict] = {}
    if out_path.exists():
        prior = read_json(out_path)
        if prior.get("experiment") != "x108" or prior.get("arm") != args.arm or prior.get("model_tag") != tag or prior.get("cohort_sha256") != cohort_sha:
            raise SystemExit("FATAL: resume output belongs to another model/arm/cohort")
        for row in prior.get("rows", []):
            prompt_id = str(row["id"])
            if prompt_id not in expected:
                continue
            if not valid_completed(row, expected[prompt_id]["prompt_sha256"][text], cohort_sha):
                raise SystemExit(f"FATAL: invalid completed resume row {prompt_id}")
            if prompt_id in done:
                raise SystemExit(f"FATAL: duplicate completed resume row {prompt_id}")
            done[prompt_id] = row

    tokenizer, model, _site, _mu, site_record = load_model(tag, entry)
    boundary_ids = site_record["boundary_ids"]
    eos_ids = eos_id_set(tokenizer, model)

    def render(unit: dict) -> str:
        prompt = unit["prompts"][text]
        if sha256_text(prompt) != unit["prompt_sha256"][text]:
            raise SystemExit(f"FATAL: prompt hash mismatch for {unit['id']}/{text}")
        chat = render_chat(tokenizer, prompt, date)
        if text == "O" and is_toxic(unit["id"]) and sha256_text(chat) != base_sha[unit["id"]]:
            raise SystemExit(f"FATAL: O rendering does not reproduce the base row's sha256 at {unit['id']}")
        assert_boundary(tokenizer, chat, boundary_ids, tag)
        return chat

    def checkpoint() -> None:
        atomic_json(
            out_path,
            {
                "experiment": "x108",
                "stage": 2,
                "model_tag": tag,
                "model_id": entry["model_id"],
                "model_source": entry["snapshot"],
                "arm": args.arm,
                "text_condition": text,
                "condition": args.arm,
                "signed_dose": 0.0,
                "cohort_sha256": cohort_sha,
                "arms_sha256": cohort_sha,
                "inputs": file_record(x106_inputs()),
                "site": site_record,
                "rendering": {"renderer": "x87_common.format_user_chat", "date": date, "o_checked_against_base_rows": text == "O"},
                "config": {"temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K, "num_return_sequences": N_RESPONSES, "max_new_tokens": MAX_NEW, "retry_max_new_tokens": RETRY_NEW, "thinking": False, "seed_policy": "x108 per-(model, prompt, text) seed"},
                "environment": {"torch": torch.__version__, "transformers": importlib.metadata.version("transformers"), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
                "rows": [done[k] for k in sorted(done)],
            },
        )

    todo = [u for u in units if u["id"] not in done]
    for index, unit in enumerate(todo, start=1):
        prompt_id = unit["id"]
        chat = render(unit)
        seed = stable_seed(prompt_id, text, tag=tag)
        texts, lengths, truncated = generate_row(model, tokenizer, eos_ids, chat, MAX_NEW, seed)
        used_retry = any(truncated)
        retry_seed = None
        if used_retry:
            retry_seed = stable_seed(prompt_id, text, retry=True, tag=tag)
            texts, lengths, truncated = generate_row(model, tokenizer, eos_ids, chat, RETRY_NEW, retry_seed)
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
            "signed_dose": 0.0,
            "seed": seed,
            "used_retry": used_retry,
            "retry_seed": retry_seed,
            "llm_responses": texts,
            "response_sha256": [sha256_text(value) for value in texts],
            "gen_tokens": lengths,
            "truncated": truncated,
        }
        checkpoint()
        print(f"x108 generate {tag} {args.arm}: {index}/{len(todo)} new", flush=True)
    checkpoint()
    print(f"x108 generation {tag} {args.arm}: {len(done)} rows -> {out_path}")


if __name__ == "__main__":
    main()
