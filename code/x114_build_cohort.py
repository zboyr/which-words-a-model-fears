"""x114 Stage 0 (local, CPU): freeze the 72-unit x85-free insertion cohort, the per-model
manifest and the x85 exclusion check.

Prompt pool: the 13,527 campaign prompts the eight per-model v4 label files cover, kept
where the historical refusal2 rate is <= 0.8 in every model (x101's rule) and the id is
neither an x83 cohort prompt (321) nor an x85 cohort prompt (96; a subset of x83's). The
x83 prompts all came from the x66 seed-42 test split, so these lie outside that split.
Words: the x83 lexicon minus x85's 24 harm words and 24 neutral twins. One
`numpy.random.default_rng(114)` draws 72 prompts without replacement and then one word per
prompt, uniformly. O/S/M/E texts follow x83's `insert_word`. Reads historical labels only.

The manifest carries, per model, x101's preflight constants (snapshot, gate scoring offset,
scaffold sha256) so no preflight is rerun; Qwen3.5-4B takes x83's snapshot and offset.
"""

from __future__ import annotations

import argparse
import platform
from collections import Counter

import numpy as np

from x114_common import (
    BASE_SEED,
    CONDITIONS,
    DETECT_INSTRUCTION,
    DETECT_THRESHOLD,
    IMPORTED_TAG,
    MAX_HISTORICAL_RATE,
    MODELS,
    N_UNITS,
    PROMPTS_PATH,
    QWEN4B_MODEL_ID,
    TAGS,
    TEST_INDICES_PATH,
    X83_RUN,
    X85_COHORT,
    atomic_json,
    build_exclusions,
    build_unit,
    cohort_path,
    cohort_sha256,
    draw_units,
    eligible_prompt_ids,
    exclusions_path,
    file_record,
    historical_rate,
    labels_path,
    lexicon_sha256,
    load_prompt_rows,
    load_tag_historical_refusals,
    manifest_path,
    read_json,
    sha256_text,
    split_prompt_ids,
    word_pool,
    x101_manifest_path,
    x83_cohort_path,
    x83_prompt_ids,
    x85_prompt_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-units", type=int, default=N_UNITS)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    return parser.parse_args()


def build_manifest(cohort: dict) -> dict:
    x101 = read_json(x101_manifest_path())
    x83_detect = read_json(X83_RUN / "x83_detect.json")
    models = {}
    for tag in TAGS:
        entry = x101["models"][tag]
        if tag == IMPORTED_TAG:
            if entry["model_id"] != QWEN4B_MODEL_ID or entry["snapshot"] != x83_detect["model_source"]:
                raise SystemExit("FATAL: the x101 manifest's Qwen3.5-4B row is not x83's snapshot")
            models[tag] = {
                "model_id": QWEN4B_MODEL_ID,
                "snapshot": entry["snapshot"],
                "revision": None,
                "trust_remote_code": True,
                "loader": "lib.edit_common.load_causal_lm (x85 / x86 / x87 / x103 / x105 loader for Qwen3.5-4B)",
                "gate": {
                    "fixed_pre_answer_ids": [],
                    "scoring_offset": 0,
                    "source": "x83_detect: scored at the first generated token of the rendered detection chat",
                },
                "chat_template": {
                    "renderer": "x87_common.format_user_chat (user turn only, add_generation_prompt, enable_thinking=False)",
                    "scaffold_sha256_x101": None,
                },
                "source": "x83 snapshot and gate; generated fresh here (x101 imported x83's rows instead)",
            }
            continue
        if entry["model_id"] != MODELS[tag]["model_id"]:
            raise SystemExit(f"FATAL: {tag}: the x101 manifest model id is not the registry's")
        models[tag] = {
            "model_id": entry["model_id"],
            "snapshot": entry["snapshot"],
            "revision": entry.get("revision"),
            "trust_remote_code": bool(entry["trust_remote_code"]),
            "loader": "x88_common.load_model / x105_common.load_tag_tokenizer (x101's loaders)",
            "gate": {
                "fixed_pre_answer_ids": [int(v) for v in entry["gate"]["fixed_pre_answer_ids"]],
                "scoring_offset": int(entry["gate"]["scoring_offset"]),
                "source": "x101 preflight (longest common greedy prefix over distinct-word probes)",
            },
            "chat_template": {
                "renderer": entry["chat_template"]["renderer"],
                "scaffold_sha256_x101": entry["chat_template"]["scaffold_sha256"],
                "thinking_block": entry["chat_template"]["thinking_block"],
            },
            "source": "x101 preflight constants reused; generated fresh here",
        }
    return {
        "experiment": "x114",
        "status": "frozen",
        "cohort_sha256": cohort["cohort_sha256"],
        "cohort": file_record(cohort_path()),
        "x101_manifest": file_record(x101_manifest_path()),
        "x83_detect": file_record(X83_RUN / "x83_detect.json"),
        "tags": list(TAGS),
        "models": models,
        "notes": [
            "No preflight is rerun: x101's per-model preflight located no fixed pre-answer tokens in any "
            "template, so the gate scoring offset is 0 everywhere, and the frozen x88 snapshots x101 / x103 / "
            "x105 ran on are still in the shared cache.",
            "Every generation file records the scaffold its job rendered and its sha256; the Llama template "
            "stamps the run date, which x103's date recovery handles downstream.",
        ],
    }


def main() -> None:
    args = parse_args()
    x83 = read_json(x83_cohort_path())
    if x83.get("lexicon_sha256") != lexicon_sha256():
        raise SystemExit("FATAL: the x83 lexicon differs from code/x83_common.py")
    x85 = read_json(X85_COHORT)
    prompts = load_prompt_rows()
    historical = {tag: load_tag_historical_refusals(tag, prompts) for tag in TAGS}
    excluded_x83, excluded_x85 = x83_prompt_ids(), x85_prompt_ids()
    if not excluded_x85 <= excluded_x83:
        raise SystemExit("FATAL: the x85 prompts are not a subset of the x83 cohort")
    eligible = eligible_prompt_ids(historical, excluded_x83 | excluded_x85)
    test_split = set(split_prompt_ids(TEST_INDICES_PATH))
    per_tag_eligible = {
        tag: sum(historical_rate(historical[tag][i]) <= MAX_HISTORICAL_RATE for i in prompts) for tag in TAGS
    }
    pool = word_pool()

    drawn = draw_units(eligible, pool, args.n_units, args.seed)
    units = []
    for prompt_id, word, category in drawn:
        unit = build_unit(prompt_id, str(prompts[prompt_id]["prompt"]), word, category)
        unit["historical_refusal_rate"] = {tag: historical_rate(historical[tag][prompt_id]) for tag in TAGS}
        units.append(unit)
    cohort_sha = cohort_sha256(units)
    word_counts = Counter(u["word"] for u in units)
    category_counts = Counter(u["category"] for u in units)

    output = {
        "experiment": "x114",
        "stage": 0,
        "status": "frozen",
        "pre_registered": False,
        "design": {
            "question": (
                "On 72 further harmful prompts that are neither x83 nor x85 prompts, with words outside x85's "
                "48, does one inserted harm word raise refusal2 in each of the eight models (x101), does the "
                "shift along each model's own boundary direction predict that rise (x103), and does writing "
                "the direction remove and create it (x105)?"
            ),
            "population": (
                "the 13,527 campaign prompts covered by the eight per-model v4 label files, outside the x83 "
                "cohort (321) and the x85 cohort (96); all x83 prompts came from the x66 seed-42 test split, so "
                "every eligible prompt lies outside that split (the split matters only to the x66 probes)"
            ),
            "cohort_rule": (
                f"historical refusal2 rate <= {MAX_HISTORICAL_RATE} in every tag's data/labels/{{tag}}_labels_v4.json "
                "(x101's rule); selection reads historical labels only"
            ),
            "word_pool": "x83 lexicon (99) minus x85's 24 harm words and 24 neutral twins (the twins are not lexicon words)",
            "sampling": (
                f"numpy.random.default_rng({args.seed}): {args.n_units} prompt ids without replacement (rng.choice on the "
                "sorted eligible ids), then one word per prompt in ascending id order (rng.integers over the pool "
                "in frozen lexicon order); no stratification"
            ),
            "positions": {
                "S": "word + ' ' + prompt",
                "M": "' ' + word inserted before the whitespace character nearest the character midpoint",
                "E": "prompt + ' ' + word",
            },
            "conditions": list(CONDITIONS),
            "control": "O regenerated contemporaneously per model",
            "detection_gate": f"x83 template and statistic per model; admitted if P_detect > {DETECT_THRESHOLD}",
            "generation": "x101 contract (T=1.0, top-p 0.9, top-k 50, 10 sequences, 2048 tokens, one 4096 retry), seeds in the x114 namespace, base 114",
            "judge": "exact x64 refusal2 contract as x101 ran it (deepseek/deepseek-v4-flash-0731, reasoning low)",
            "downstream": "x103 extraction and projection; the eight x105 arms at x105's doses with x105's control directions and KL calibration",
        },
        "lexicon_sha256": lexicon_sha256(),
        "detect_instruction": DETECT_INSTRUCTION,
        "detect_instruction_sha256": sha256_text(DETECT_INSTRUCTION),
        "inputs": {
            "prompts": file_record(PROMPTS_PATH),
            "labels": {tag: file_record(labels_path(tag)) for tag in TAGS},
            "x83_cohort": file_record(x83_cohort_path()),
            "x85_cohort": file_record(X85_COHORT),
            "x66_test_indices": file_record(TEST_INDICES_PATH),
        },
        "x83_cohort_sha256": x83["cohort_sha256"],
        "x85_cohort_sha256": x85["cohort_sha256"],
        "seeds": {"rng": args.seed, "generation_base": BASE_SEED},
        "counts": {
            "campaign_prompts": len(prompts),
            "eligible_per_tag": per_tag_eligible,
            "excluded_x83_prompts": len(excluded_x83),
            "excluded_x85_prompts": len(excluded_x85),
            "eligible": len(eligible),
            "eligible_in_x66_seed42_test_split": len(set(eligible) & test_split),
            "word_pool": len(pool),
            "cohort": len(units),
            "words_used": len(word_counts),
            "word_counts": dict(sorted(word_counts.items())),
            "category_counts": dict(sorted(category_counts.items())),
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "cohort_sha256": cohort_sha,
        "units": units,
    }
    atomic_json(cohort_path(), output)
    print(
        f"x114 cohort: {len(units)} prompts from {len(eligible)} eligible, {len(word_counts)} of {len(pool)} pool words, "
        f"{len(category_counts)} categories, sha {cohort_sha[:8]} -> {cohort_path()}"
    )
    manifest = build_manifest(output)
    atomic_json(manifest_path(), manifest)
    print(f"x114 manifest: {len(manifest['models'])} models -> {manifest_path()}")
    exclusions = build_exclusions(output)
    atomic_json(exclusions_path(), exclusions)
    counts = exclusions["counts"]
    print(f"x114 exclusions: {counts['n_units_excluded']} of {counts['n_units']} units excluded -> {exclusions_path()}")


if __name__ == "__main__":
    main()
