"""x100 Stage 0: freeze words, neutral controls, both populations, cells, and graphs.

CPU only (tokenizer). Reads historical refusal2 labels (HP) and x87 OR-Bench
base labels (BP) solely for population selection. No x100 outcome exists yet.
"""

from __future__ import annotations

import argparse
import platform
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from x83_common import TEST_INDICES_PATH, LABELS_PATH, load_historical_refusals, load_split_rows
from x100_common import (
    BASE_SEED,
    CANDIDATE_CLASSES,
    FEATURE_POSITIONS,
    FEATURE_STATES,
    HP_MIN_RATE,
    MODEL_ID,
    N_FEATURE_SHARDS,
    N_GEN_CHUNKS,
    N_PROMPTS,
    N_RESPONSES,
    NEUTRAL_POOL,
    POPULATIONS,
    PROMPT_DEGREE,
    SEEDS,
    WORD_CLASSES,
    WORD_DEGREE,
    X83_COHORT,
    X85_COHORT,
    X86_DIRECTIONS,
    X87_GENERATIONS,
    X87_ORBENCH,
    append_word,
    atomic_json,
    behavior_graph,
    cohort_path,
    file_record,
    format_user_chat,
    hp_stratum,
    locate_positions,
    prompt_key,
    read_json,
    resolve_neutral_controls,
    select_bp,
    select_hp,
    sha256_file,
    sha256_text,
    word_table_sha256,
)

PILOT_SEED = 1000  # pilot prompt/edge choice only; pilot outputs never enter analysis
PILOT_PROMPTS = 8
PILOT_PAIRS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-snapshot", help="default: snapshot recorded by x83")
    parser.add_argument("--out", default=str(cohort_path()))
    return parser.parse_args()


def tokenizer_record(snapshot: str) -> dict:
    root = Path(snapshot)
    files = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja"):
        path = root / name
        if path.exists():
            files[name] = sha256_file(path)
    return {"files_sha256": files, "sha256": sha256_text("\n".join(f"{k}\t{v}" for k, v in sorted(files.items())))}


def hp_units() -> list[dict]:
    rows = load_split_rows(TEST_INDICES_PATH)
    ids = [int(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("FATAL: duplicate evaluation prompt IDs")
    historical = load_historical_refusals(ids)
    units = []
    for row in rows:
        prompt_id = int(row["id"])
        rate = sum(historical[prompt_id]) / N_RESPONSES
        if rate >= HP_MIN_RATE - 1e-9:
            units.append({"id": prompt_id, "prompt": str(row["prompt"]), "historical_refusal_rate": rate})
    return units


def bp_candidates() -> list[dict]:
    labels = read_json(X87_ORBENCH)
    generations = read_json(X87_GENERATIONS)
    if labels.get("experiment") != "x87" or labels.get("condition") != "base":
        raise SystemExit("FATAL: unexpected x87 OR-Bench label file")
    if generations.get("experiment") != "x87" or generations.get("condition") != "base":
        raise SystemExit("FATAL: unexpected x87 generation file")
    text = {row["id"]: row for row in generations["rows"]}
    out = []
    for row in labels["rows"]:
        if row.get("benchmark") != "orbench_hard" or int(row.get("refusal", 0)) != 1:
            continue
        source = text[row["id"]]
        if sha256_text(source["prompt"]) != source["prompt_sha256"]:
            raise SystemExit(f"FATAL: x87 prompt hash mismatch {row['id']}")
        out.append(
            {
                "id": row["id"],
                "source_index": int(source["source_index"]),
                "prompt": source["prompt"],
                "category": source.get("category"),
                "x87_label": row.get("label"),
            }
        )
    return out


def template_audit(tokenizer) -> dict:
    sentinel = format_user_chat(tokenizer, "alpha bomb")
    ids, positions = locate_positions(tokenizer, sentinel, "alpha bomb", (6, 10))
    if ids[-2] != int(tokenizer.convert_tokens_to_ids("</think>")) or tokenizer.decode([ids[-1]]) != "\n\n":
        raise SystemExit("FATAL: x100 requires TBG-1=</think> and TBG=double-newline")
    audit = {
        "sentinel_rendered_sha256": sha256_text(sentinel),
        "tail_ids": ids[-8:],
        "tail_tokens": tokenizer.convert_ids_to_tokens(ids[-8:]),
        "tail_decoded": [tokenizer.decode([value]) for value in ids[-8:]],
        "positions": positions,
    }
    x85 = read_json(X85_COHORT)["template_audit"]
    for key in ("sentinel_rendered_sha256", "tail_ids", "positions"):
        if audit[key] != x85[key]:
            raise SystemExit(f"FATAL: template audit {key} differs from x85: {audit[key]} vs {x85[key]}")
    return audit


def make_cell(tokenizer, population, prompt_row, condition, word_row, cell_prefix=""):
    base = prompt_row["prompt"]
    key = prompt_row["key"]
    if condition == "O":
        text, span, inserted = base, None, None
        cell_id = f"{cell_prefix}{population}:O:{key}"
    else:
        inserted = word_row["word"] if condition == "W" else word_row["neutral"]
        text, span = append_word(base, inserted)
        cell_id = f"{cell_prefix}{population}:{condition}:{key}:w{word_row['word_index']:02d}"
    if span is not None and text[span[0] : span[1]] != inserted:
        raise SystemExit(f"FATAL: append span mismatch {cell_id}")
    rendered = format_user_chat(tokenizer, text)
    input_ids, positions = locate_positions(tokenizer, rendered, text, span)
    return {
        "cell_id": cell_id,
        "population": population,
        "condition": condition,
        "prompt_key": key,
        "prompt_index": prompt_row["prompt_index"],
        "word_index": None if word_row is None else word_row["word_index"],
        "class": None if word_row is None else word_row["class"],
        "word": None if word_row is None else word_row["word"],
        "neutral": None if word_row is None else word_row["neutral"],
        "inserted_word": inserted,
        "inserted_span": None if span is None else list(span),
        "prompt": text,
        "prompt_sha256": sha256_text(text),
        "rendered_chat_sha256": sha256_text(rendered),
        "n_input_tokens": len(input_ids),
        "position_indices": positions,
    }


def main() -> None:
    args = parse_args()
    x83 = read_json(X83_COHORT)
    if x83.get("experiment") != "x83" or x83.get("model_id") != MODEL_ID:
        raise SystemExit("FATAL: invalid x83 source cohort")
    model_source = args.model_snapshot or x83["model_snapshot"]
    if not model_source.rstrip("/").endswith(x83["model_snapshot"].rstrip("/").split("/")[-1]):
        raise SystemExit("FATAL: model snapshot differs from the x83 snapshot")
    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
    audit = template_audit(tokenizer)

    try:
        words = resolve_neutral_controls(tokenizer)
    except ValueError as error:
        raise SystemExit(f"FATAL: neutral-control resolution failed: {error}")

    hp = select_hp(hp_units(), SEEDS["hp_selection"])
    bp = select_bp(bp_candidates(), SEEDS["bp_selection"])
    populations = {}
    for population, selected in (("HP", hp), ("BP", bp)):
        prompts = []
        for prompt_index, unit in enumerate(selected):
            row = {
                "prompt_index": prompt_index,
                "id": unit["id"],
                "prompt": unit["prompt"],
                "prompt_sha256": sha256_text(unit["prompt"]),
            }
            if population == "HP":
                row["historical_refusal_rate"] = float(unit["historical_refusal_rate"])
                row["stratum"] = hp_stratum(row["historical_refusal_rate"])
            else:
                row["source_index"] = unit["source_index"]
                row["category"] = unit["category"]
                row["x87_label"] = unit["x87_label"]
            row["key"] = prompt_key(population, row)
            prompts.append(row)
        if len({row["key"] for row in prompts}) != N_PROMPTS:
            raise SystemExit(f"FATAL: {population} prompt keys are not unique")
        seed = SEEDS["hp_graph"] if population == "HP" else SEEDS["bp_graph"]
        edges, draws = behavior_graph(N_PROMPTS, words, seed)
        populations[population] = {"prompts": prompts, "edges": edges, "graph_draws": draws, "graph_seed": seed}

    cells = []
    behavior_id = 0
    for population in POPULATIONS:
        block = populations[population]
        edge_set = set(block["edges"])
        for prompt_row in block["prompts"]:
            p = prompt_row["prompt_index"]
            plan = [("O", None, True)]
            for word_row in words:
                in_behavior = (p, word_row["word_index"]) in edge_set
                plan.extend([("W", word_row, in_behavior), ("N", word_row, in_behavior)])
            for condition, word_row, in_behavior in plan:
                cell = make_cell(tokenizer, population, prompt_row, condition, word_row)
                cell["cell_index"] = len(cells)
                cell["behavior"] = bool(in_behavior)
                cell["behavior_id"] = behavior_id if in_behavior else None
                if in_behavior:
                    behavior_id += 1
                cells.append(cell)

    dense = Counter((c["population"], c["condition"]) for c in cells)
    beh = Counter((c["population"], c["condition"]) for c in cells if c["behavior"])
    for population in POPULATIONS:
        if (dense[(population, "O")], dense[(population, "W")], dense[(population, "N")]) != (96, 2688, 2688):
            raise SystemExit(f"FATAL: dense counts wrong for {population}: {dense}")
        if (beh[(population, "O")], beh[(population, "W")], beh[(population, "N")]) != (96, 672, 672):
            raise SystemExit(f"FATAL: behavior counts wrong for {population}: {beh}")
    if len(cells) != 10944 or behavior_id != 2880:
        raise SystemExit(f"FATAL: totals wrong: {len(cells)} dense, {behavior_id} behavior")
    if len({c["cell_id"] for c in cells}) != len(cells):
        raise SystemExit("FATAL: duplicate cell IDs")

    # Pilot: 8 HP prompts x (O + 2 W/N pairs from their own behavior edges); pilot-namespaced IDs
    rng = np.random.default_rng(PILOT_SEED)
    hp_block = populations["HP"]
    pilot_prompt_indices = sorted(int(v) for v in rng.choice(N_PROMPTS, size=PILOT_PROMPTS, replace=False))
    pilot_cells = []
    for p in pilot_prompt_indices:
        prompt_row = hp_block["prompts"][p]
        own = sorted(w for q, w in hp_block["edges"] if q == p)
        chosen = sorted(int(v) for v in rng.choice(own, size=PILOT_PAIRS, replace=False))
        pilot_cells.append(make_cell(tokenizer, "HP", prompt_row, "O", None, "pilot|"))
        for w in chosen:
            for condition in ("W", "N"):
                pilot_cells.append(make_cell(tokenizer, "HP", prompt_row, condition, words[w], "pilot|"))
    for index, cell in enumerate(pilot_cells):
        cell["pilot_id"] = index

    cohort_sha = sha256_text(
        "\n".join(
            f"{c['cell_index']}\t{c['cell_id']}\t{c['prompt_sha256']}\t{int(c['behavior'])}" for c in cells
        )
    )
    graph_sha = {
        population: sha256_text(
            "\n".join(f"{populations[population]['prompts'][p]['key']}\t{w}" for p, w in populations[population]["edges"])
        )
        for population in POPULATIONS
    }
    prompts_sha = {
        population: sha256_text(
            "\n".join(f"{row['key']}\t{row['prompt_sha256']}" for row in populations[population]["prompts"])
        )
        for population in POPULATIONS
    }
    word_sha = sha256_text(
        "\n".join(f"{w['word_index']}\t{w['class']}\t{w['word']}\t{w['neutral']}\t{w['leading_space_word_ids']}\t{w['leading_space_neutral_ids']}" for w in words)
    )
    output = {
        "experiment": "x100",
        "stage": 0,
        "status": "frozen",
        "model_id": MODEL_ID,
        "model_snapshot": model_source,
        "design": {
            "plan": "knowledge/plans/x100-lexical-courage-words.md (commit c41b40a)",
            "position": "append-only E position: prompt + ' ' + word (x85 append_word)",
            "classes": {name: {"words": list(ws), "expectation": exp} for name, ws, exp in WORD_CLASSES},
            "candidate_family": list(CANDIDATE_CLASSES),
            "neutral_pool": list(NEUTRAL_POOL),
            "hp": "x66 seed-42 evaluation split, historical refusal2 r_O >= 0.2; strata [0.2,0.5], [0.6,0.9], [1.0] x 32, rng(100), sorted by id",
            "bp": "OR-Bench Hard-1K prompts with x87 base refusal == 1 (OR-Bench rubric, selection only); 96 via rng(101), sorted by id",
            "behavior_graph": (
                "per population 7-regular on prompts, 24-regular on words; seeded rejection sampler "
                "(HP rng(102), BP rng(103)) with class-wise stub-permutation proposal; rejects repeated "
                "edges, two same-class words on one prompt, and degree violations"
            ),
            "feature_states": list(FEATURE_STATES),
            "feature_positions": list(FEATURE_POSITIONS),
            "feature_shards": N_FEATURE_SHARDS,
            "generation_chunks": N_GEN_CHUNKS,
            "generation": "x83/x85 contract: 10 samples, T=1.0, top-p 0.9, top-k 50, 2048 tokens plus one 4096 retry; seed x85 stable_seed form in the x100 namespace",
            "judge": "x83/x64 refusal2 contract (deepseek/deepseek-v4-flash-0731, reasoning low, seeds 42/1042/2042); refusal2 only",
            "pilot": f"{PILOT_PROMPTS} HP prompts x (O + {PILOT_PAIRS} W/N pairs), rng({PILOT_SEED}), 'pilot|' cell namespace and seeds; excluded from every analysis",
        },
        "template_audit": audit,
        "tokenizer": tokenizer_record(model_source),
        "inputs": {
            "x83_cohort": file_record(X83_COHORT),
            "x85_cohort": file_record(X85_COHORT),
            "labels": file_record(LABELS_PATH),
            "test_indices": file_record(TEST_INDICES_PATH),
            "x87_orbench_base": file_record(X87_ORBENCH),
            "x87_generations_base": file_record(X87_GENERATIONS),
            "x86_directions": file_record(X86_DIRECTIONS),
        },
        "seeds": {"base": BASE_SEED, **SEEDS, "pilot": PILOT_SEED},
        "hashes": {
            "word_table_sha256": word_table_sha256(),
            "word_pairs_sha256": word_sha,
            "prompts_sha256": prompts_sha,
            "graph_sha256": graph_sha,
            "cohort_sha256": cohort_sha,
        },
        "counts": {
            "dense_cells": len(cells),
            "behavior_cells": behavior_id,
            "dense_by_population_condition": {f"{k[0]}:{k[1]}": v for k, v in sorted(dense.items())},
            "behavior_by_population_condition": {f"{k[0]}:{k[1]}": v for k, v in sorted(beh.items())},
            "prompt_degree": PROMPT_DEGREE,
            "word_degree": WORD_DEGREE,
            "graph_draws": {p: populations[p]["graph_draws"] for p in POPULATIONS},
            "hp_strata": dict(Counter(row["stratum"] for row in populations["HP"]["prompts"])),
            "pilot_cells": len(pilot_cells),
            "responses_per_behavior_cell": N_RESPONSES,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "tokenizer_class": tokenizer.__class__.__name__,
        },
        "cohort_sha256": cohort_sha,
        "words": words,
        "populations": {
            p: {"prompts": populations[p]["prompts"], "edges": [list(e) for e in populations[p]["edges"]]}
            for p in POPULATIONS
        },
        "cells": cells,
        "pilot_cells": pilot_cells,
    }
    out_path = Path(args.out)
    atomic_json(out_path, output)
    print(f"x100 cohort: {len(cells)} dense / {behavior_id} behavior / {len(pilot_cells)} pilot cells, sha {cohort_sha[:8]} -> {out_path}")
    for w in words:
        print(f"  {w['class']:<14} {w['word']:<14} {w['neutral']:<12} n_sub={w['n_subtokens']}")


if __name__ == "__main__":
    main()
