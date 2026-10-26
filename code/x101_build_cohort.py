"""x101 Stage 0: freeze the shared 130-prompt cohort inherited from x83.

CPU only. Starts from the 321 x83 prompts and keeps those whose historical
refusal2 rate is <= 0.8 in every one of the eight transfer models' v4 label
files. Words and the O/S/M/E texts are x83's, unchanged. Reads historical
labels only; no x101 output exists yet. Fails unless the count is 130.
"""

from __future__ import annotations

import argparse
import platform
from collections import Counter

import numpy as np

from x101_common import (
    ALL_TAGS,
    CONDITIONS,
    DETECT_INSTRUCTION,
    DETECT_THRESHOLD,
    EXPECTED_COHORT,
    MAX_HISTORICAL_RATE,
    MODELS,
    QWEN4B_MODEL_ID,
    atomic_json,
    cohort_path,
    cohort_sha256,
    file_record,
    historical_rate,
    inherit_unit,
    labels_path,
    lexicon_sha256,
    load_tag_historical_refusals,
    read_json,
    select_cohort_ids,
    sha256_text,
    x83_cohort_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(cohort_path()))
    parser.add_argument("--expected", type=int, default=EXPECTED_COHORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x83 = read_json(x83_cohort_path())
    if x83.get("experiment") != "x83" or x83.get("lexicon_sha256") != lexicon_sha256():
        raise SystemExit("FATAL: x83 cohort missing or its lexicon differs from code/x83_common.py")
    if x83.get("detect_instruction") != DETECT_INSTRUCTION:
        raise SystemExit("FATAL: x83 detection template differs from code/x83_common.py")
    x83_units = list(x83["units"])
    if cohort_sha256(x83_units) != x83["cohort_sha256"]:
        raise SystemExit("FATAL: x83 cohort fails its recorded sha256")
    ids = [int(unit["id"]) for unit in x83_units]

    historical = {tag: load_tag_historical_refusals(tag, ids) for tag in ALL_TAGS}
    kept = select_cohort_ids(x83_units, historical)
    per_tag_eligible = {
        tag: sum(historical_rate(historical[tag][i]) <= MAX_HISTORICAL_RATE for i in ids) for tag in ALL_TAGS
    }
    if len(kept) != args.expected:
        raise SystemExit(
            f"FATAL: cohort has {len(kept)} prompts, plan requires {args.expected}; "
            f"per-tag eligible {per_tag_eligible}"
        )

    by_id = {int(unit["id"]): unit for unit in x83_units}
    units = []
    for prompt_id in kept:
        unit = inherit_unit(by_id[prompt_id])
        unit["historical_refusal_rate"] = {
            tag: historical_rate(historical[tag][prompt_id]) for tag in ALL_TAGS
        }
        units.append(unit)
    cohort_sha = cohort_sha256(units)
    word_counts = Counter(unit["word"] for unit in units)
    category_counts = Counter(unit["category"] for unit in units)

    output = {
        "experiment": "x101",
        "stage": 0,
        "status": "frozen",
        "plan": "knowledge/plans/x101-cross-model-keyword-insertion.md",
        "pre_registered": False,
        "models": {
            "imported": {"qwen": {"model_id": QWEN4B_MODEL_ID, "source": "x83 artifacts restricted to this cohort"}},
            "fresh": {tag: MODELS[tag]["model_id"] for tag in MODELS},
        },
        "design": {
            "question": (
                "On one shared set of harmful prompts, does inserting one harm-topic word at the "
                "start, middle, or end raise refusal2 relative to the contemporaneously regenerated "
                "original, in each of the eight transfer models?"
            ),
            "population": "x83 cohort (x66 seed-42 evaluation split, Qwen3.5-4B historical r_O <= 0.8)",
            "cohort_rule": (
                f"historical refusal2 rate <= {MAX_HISTORICAL_RATE} in every tag's "
                "data/labels/{tag}_labels_v4.json; selection reads historical labels only"
            ),
            "expected_cohort": args.expected,
            "words_and_texts": "inherited from x83 unchanged (same word per prompt, same S/M/E insertion texts)",
            "conditions": list(CONDITIONS),
            "control": "O regenerated contemporaneously per model",
            "detection_gate": (
                f"x83 template and statistic per model; admitted if P_detect > {DETECT_THRESHOLD}; "
                "scored at the first answer position after any fixed pre-answer tokens the template emits"
            ),
            "generation": "x83 contract (T=1.0, top-p 0.9, top-k 50, 10 sequences, 2048 tokens, one 4096 retry), seeds in the x101 namespace",
            "judge": "exact x64 refusal2 contract as x83 ran it (deepseek/deepseek-v4-flash-0731, reasoning low)",
            "analysis": "x83 analysis per model (admitted delta, two-stage paired bootstrap, Holm over three positions); descriptive",
        },
        "lexicon_sha256": lexicon_sha256(),
        "detect_instruction": DETECT_INSTRUCTION,
        "detect_instruction_sha256": sha256_text(DETECT_INSTRUCTION),
        "inputs": {
            "x83_cohort": file_record(x83_cohort_path()),
            "labels": {tag: file_record(labels_path(tag)) for tag in ALL_TAGS},
        },
        "x83_cohort_sha256": x83["cohort_sha256"],
        "counts": {
            "x83_prompts": len(ids),
            "eligible_per_tag": per_tag_eligible,
            "cohort": len(units),
            "words_used": len(word_counts),
            "word_counts": dict(sorted(word_counts.items())),
            "category_counts": dict(sorted(category_counts.items())),
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "cohort_sha256": cohort_sha,
        "units": units,
    }
    atomic_json(args.out, output)
    print(
        f"x101 cohort: {len(units)} prompts, {len(word_counts)} words, "
        f"{len(category_counts)} categories, sha {cohort_sha[:8]} -> {args.out}"
    )


if __name__ == "__main__":
    main()
