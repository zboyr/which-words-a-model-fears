"""x114 Stage 1-2 (GPU, one (tag, stage) per job): the x101 base pipeline on the x114 units.

`--stage gate` scores x83's admission statistic per (prompt, position) with x101_detect's
imported functions, at the scoring position the x114 manifest carries from x101's preflight.
`--stage O|S|M|E` generates that condition's ten responses per prompt with x101's sampling
contract (x81_generate's engine), seeds in the x114 namespace, resumable by prompt id.
"""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path

import torch

from x81_generate import MAX_NEW, RETRY_NEW, TEMPERATURE, TOP_K, TOP_P, eos_id_set, generate_row, valid_completed
from x83_detect import GREEDY_TOKENS, first_token_top10, greedy_answer, variant_log_probs
from x103_common import load_tag_model
from x105_common import load_tag_tokenizer
from x114_common import (
    AUDIT_PROMPT,
    BASE_SEED,
    CONDITIONS,
    DETECT_INSTRUCTION,
    DETECT_THRESHOLD,
    N_RESPONSES,
    POSITIONS,
    TAGS,
    atomic_json,
    cohort_path,
    detect_path,
    file_record,
    format_user_chat,
    generations_path,
    load_cohort,
    load_manifest,
    manifest_path,
    read_json,
    scoring_prefix_ids,
    sha256_text,
    stable_seed,
    surface_variants,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, choices=TAGS)
    parser.add_argument("--stage", required=True, choices=("gate",) + CONDITIONS)
    parser.add_argument("--out")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def environment() -> dict:
    return {
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def scaffold_record(tokenizer) -> dict:
    rendered = format_user_chat(tokenizer, AUDIT_PROMPT)
    if rendered.count(AUDIT_PROMPT) != 1:
        raise SystemExit("FATAL: the rendered chat does not contain the audit prompt exactly once")
    scaffold = rendered.replace(AUDIT_PROMPT, "{PROMPT}")
    return {"scaffold": scaffold, "scaffold_sha256": sha256_text(scaffold)}


def run_gate(tag: str, cohort: dict, entry: dict, units: list[dict], out_path: Path) -> None:
    fixed = [int(v) for v in entry["gate"]["fixed_pre_answer_ids"]]
    tokenizer = load_tag_tokenizer(entry["snapshot"], tag)
    model = load_tag_model(entry["snapshot"], tag)
    model.eval()
    rows = []
    for index, unit in enumerate(units, start=1):
        variants = surface_variants(unit["word"])
        record = {"id": int(unit["id"]), "word": unit["word"], "category": unit["category"], "positions": {}}
        for position in POSITIONS:
            text = unit["prompts"][position]
            if sha256_text(text) != unit["prompt_sha256"][position]:
                raise SystemExit(f"FATAL: text hash mismatch {unit['id']}/{position}")
            chat = format_user_chat(tokenizer, DETECT_INSTRUCTION.format(text=text))
            chat_ids = tokenizer(chat, add_special_tokens=False)["input_ids"]
            prefix_ids = scoring_prefix_ids(chat_ids, fixed)
            variant_rows = variant_log_probs(model, tokenizer, prefix_ids, variants)
            p_detect = float(sum(v["prob"] for v in variant_rows))
            top10, entropy = first_token_top10(model, tokenizer, prefix_ids)
            greedy = greedy_answer(model, tokenizer, prefix_ids)
            record["positions"][position] = {
                "prompt_sha256": unit["prompt_sha256"][position],
                "rendered_chat_sha256": sha256_text(chat),
                "scoring_position": len(prefix_ids),
                "p_detect": p_detect,
                "admitted": p_detect > DETECT_THRESHOLD,
                "variants": variant_rows,
                "first_token_top10": top10,
                "first_token_entropy_nats": entropy,
                "greedy_answer": greedy,
                "greedy_matches": greedy.strip().strip(".\"'").casefold() == unit["word"],
            }
        rows.append(record)
        if index % 10 == 0 or index == len(units):
            print(f"x114 gate {tag}: {index}/{len(units)}", flush=True)
    admitted = {p: sum(r["positions"][p]["admitted"] for r in rows) for p in POSITIONS}
    atomic_json(
        out_path,
        {
            "experiment": "x114",
            "stage": 1,
            "model_tag": tag,
            "model_id": entry["model_id"],
            "model_source": entry["snapshot"],
            "cohort_sha256": cohort["cohort_sha256"],
            "input": file_record(cohort_path()),
            "manifest": file_record(manifest_path()),
            "config": {
                "threshold": DETECT_THRESHOLD,
                "statistic": "sum of teacher-forced sequence probabilities over surface variants",
                "variants": "word, Word, WORD, and the same three with a leading space",
                "greedy_tokens": GREEDY_TOKENS,
                "thinking": False,
                "fixed_pre_answer_ids": fixed,
                "scoring_offset": len(fixed),
            },
            "chat_template": scaffold_record(tokenizer),
            "counts": {
                "prompts": len(rows),
                "admitted": admitted,
                "admitted_all_three": sum(all(r["positions"][p]["admitted"] for p in POSITIONS) for r in rows),
            },
            "environment": environment(),
            "rows": rows,
        },
    )
    print(f"x114 gate {tag}: admitted {admitted} -> {out_path}")


def run_generation(tag: str, cohort: dict, entry: dict, units: list[dict], condition: str, out_path: Path) -> None:
    cohort_sha = cohort["cohort_sha256"]
    expected = {int(unit["id"]): unit for unit in units}
    done: dict[int, dict] = {}
    if out_path.exists():
        prior = read_json(out_path)
        if (
            prior.get("condition") != condition
            or prior.get("model_tag") != tag
            or prior.get("cohort_sha256") != cohort_sha
        ):
            raise SystemExit("FATAL: resume output belongs to another tag/condition/cohort")
        for row in prior.get("rows", []):
            prompt_id = int(row["id"])
            if prompt_id not in expected:
                continue
            if not valid_completed(row, expected[prompt_id]["prompt_sha256"][condition], cohort_sha):
                raise SystemExit(f"FATAL: invalid completed resume row {prompt_id}")
            if prompt_id in done:
                raise SystemExit(f"FATAL: duplicate completed resume row {prompt_id}")
            done[prompt_id] = row

    tokenizer = load_tag_tokenizer(entry["snapshot"], tag)
    model = load_tag_model(entry["snapshot"], tag)
    model.eval()
    eos_ids = eos_id_set(tokenizer, model)
    template = scaffold_record(tokenizer)

    def checkpoint() -> None:
        atomic_json(
            out_path,
            {
                "experiment": "x114",
                "stage": 2,
                "model_tag": tag,
                "model_id": entry["model_id"],
                "model_source": entry["snapshot"],
                "condition": condition,
                "cohort_sha256": cohort_sha,
                "arms_sha256": cohort_sha,  # the field name the shared x81/x83 judge and validators read
                "input": file_record(cohort_path()),
                "manifest": file_record(manifest_path()),
                "chat_template": template,
                "config": {
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "top_k": TOP_K,
                    "num_return_sequences": N_RESPONSES,
                    "max_new_tokens": MAX_NEW,
                    "retry_max_new_tokens": RETRY_NEW,
                    "thinking": False,
                    "seed_policy": f"sha256-derived per-(tag,prompt,condition), x114 namespace, base {BASE_SEED}",
                },
                "environment": environment(),
                "rows": [done[prompt_id] for prompt_id in sorted(done)],
            },
        )

    todo = [unit for unit in units if int(unit["id"]) not in done]
    for index, unit in enumerate(todo, start=1):
        prompt_id = int(unit["id"])
        prompt = unit["prompts"][condition]
        if sha256_text(prompt) != unit["prompt_sha256"][condition]:
            raise SystemExit(f"FATAL: text hash mismatch for {prompt_id}")
        chat = format_user_chat(tokenizer, prompt)
        seed = stable_seed(tag, prompt_id, condition)
        texts, lengths, truncated = generate_row(model, tokenizer, eos_ids, chat, MAX_NEW, seed)
        used_retry = any(truncated)
        retry_seed = None
        if used_retry:
            retry_seed = stable_seed(tag, prompt_id, f"{condition}:retry4096")
            texts, lengths, truncated = generate_row(model, tokenizer, eos_ids, chat, RETRY_NEW, retry_seed)
        if len(texts) != N_RESPONSES or any(not value.strip() for value in texts):
            raise SystemExit(f"FATAL: incomplete/empty generation row {prompt_id}")
        done[prompt_id] = {
            "id": prompt_id,
            "condition": condition,
            "word": unit["word"],
            "prompt": prompt,
            "prompt_sha256": unit["prompt_sha256"][condition],
            "rendered_chat_sha256": sha256_text(chat),
            "arms_sha256": cohort_sha,
            "seed": seed,
            "used_retry": used_retry,
            "retry_seed": retry_seed,
            "llm_responses": texts,
            "response_sha256": [sha256_text(value) for value in texts],
            "gen_tokens": lengths,
            "truncated": truncated,
        }
        checkpoint()
        print(f"x114 generate {tag}/{condition}: {index}/{len(todo)} new", flush=True)
    checkpoint()
    print(f"x114 generation {tag}/{condition}: {len(done)} rows -> {out_path}")


def main() -> None:
    args = parse_args()
    cohort = load_cohort()
    entry = load_manifest(args.tag)
    units = list(cohort["units"])
    if args.limit:
        units = units[: args.limit]
    if args.stage == "gate":
        run_gate(args.tag, cohort, entry, units, Path(args.out) if args.out else detect_path(args.tag))
        return
    out_path = Path(args.out) if args.out else generations_path(args.tag, args.stage)
    run_generation(args.tag, cohort, entry, units, args.stage, out_path)


if __name__ == "__main__":
    main()
