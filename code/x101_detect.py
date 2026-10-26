"""x101 Stage 1: x83's detection gate, run on one transfer model with its own chat template.

The statistic, template, surface variants, threshold and recorded fields are
x83_detect's (its functions are imported). The one x101 addition is the gate
scoring position: the teacher-forcing prefix is the rendered chat followed by
the fixed pre-answer tokens the model's preflight located (empty for most
templates), so the inserted word is always scored at the first answer token.
"""

from __future__ import annotations

import argparse

from x83_detect import GREEDY_TOKENS, first_token_top10, greedy_answer, variant_log_probs
from x101_common import (
    DETECT_INSTRUCTION,
    DETECT_THRESHOLD,
    FRESH_TAGS,
    MODELS,
    POSITIONS,
    atomic_json,
    cohort_path,
    detect_path,
    file_record,
    format_user_chat,
    load_cohort,
    load_model,
    load_preflight,
    load_tokenizer,
    preflight_path,
    scoring_prefix_ids,
    sha256_text,
    surface_variants,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, choices=FRESH_TAGS)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = args.out or detect_path(args.tag)
    cohort = load_cohort()
    preflight = load_preflight(args.tag)
    if preflight["cohort_sha256"] != cohort["cohort_sha256"]:
        raise SystemExit("FATAL: preflight belongs to another cohort")
    fixed = [int(v) for v in preflight["gate"]["fixed_pre_answer_ids"]]
    model_source = preflight["snapshot"]
    units = list(cohort["units"])
    if args.limit:
        units = units[: args.limit]

    tokenizer = load_tokenizer(model_source, args.tag)
    model = load_model(model_source, args.tag)
    model.eval()

    rows = []
    for index, unit in enumerate(units, start=1):
        prompt_id = int(unit["id"])
        word = unit["word"]
        variants = surface_variants(word)
        record = {"id": prompt_id, "word": word, "category": unit["category"], "positions": {}}
        for position in POSITIONS:
            text = unit["prompts"][position]
            if sha256_text(text) != unit["prompt_sha256"][position]:
                raise SystemExit(f"FATAL: prompt hash mismatch {prompt_id}/{position}")
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
                "greedy_matches": greedy.strip().strip(".\"'").casefold() == word,
            }
        rows.append(record)
        if index % 10 == 0 or index == len(units):
            print(f"x101 detect {args.tag}: {index}/{len(units)}", flush=True)

    admitted = {p: sum(r["positions"][p]["admitted"] for r in rows) for p in POSITIONS}
    common = sum(all(r["positions"][p]["admitted"] for p in POSITIONS) for r in rows)
    output = {
        "experiment": "x101",
        "stage": 1,
        "model_tag": args.tag,
        "model_id": MODELS[args.tag]["model_id"],
        "model_source": model_source,
        "cohort_sha256": cohort["cohort_sha256"],
        "input": file_record(cohort_path()),
        "preflight": file_record(preflight_path(args.tag)),
        "config": {
            "threshold": DETECT_THRESHOLD,
            "statistic": "sum of teacher-forced sequence probabilities over surface variants",
            "variants": "word, Word, WORD, and the same three with a leading space",
            "greedy_tokens": GREEDY_TOKENS,
            "thinking": False,
            "fixed_pre_answer_ids": fixed,
            "scoring_offset": len(fixed),
        },
        "counts": {"prompts": len(rows), "admitted": admitted, "admitted_all_three": common},
        "rows": rows,
    }
    atomic_json(out_path, output)
    print(f"x101 detect {args.tag}: admitted {admitted}, common {common} -> {out_path}")


if __name__ == "__main__":
    main()
