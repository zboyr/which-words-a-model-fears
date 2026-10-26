"""Contract tests for x114: the x85-free cohort rule, the x114 seed namespace, and the
cohort-mode machinery the pooled x103 / x105 analyses run on."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x83_common import insert_word, lexicon_words  # noqa: E402
from x101_common import stable_seed as x101_seed  # noqa: E402
from x103_common import leakage_sets  # noqa: E402
from x114_common import (  # noqa: E402
    N_UNITS,
    build_exclusions,
    build_unit,
    draw_units,
    dose_seed,
    eligible_prompt_ids,
    load_cohort,
    read_json,
    stable_seed,
    word_pool,
    x83_prompt_ids,
    x85_prompt_ids,
)
from x114_cohorts import (  # noqa: E402
    MODES,
    cohort_record,
    pool_cells,
    pool_labels,
    result_path_for,
    uses_x114,
    x101_mode,
)
from x105_common import X85_COHORT  # noqa: E402


def test_word_pool_removes_both_x85_word_lists():
    x85 = read_json(X85_COHORT)
    harm = {str(w["harm"]) for w in x85["words"]}
    neutral = {str(w["neutral"]) for w in x85["words"]}
    pool = word_pool()
    words = [word for word, _ in pool]
    assert len(words) == len(set(words)) == 75  # 99 lexicon words minus x85's 24 harm words
    assert not (set(words) & harm)
    assert not (set(words) & neutral)  # the neutral twins are not lexicon words, so none are lost
    assert [w for w, _ in pool] == [w for w, _ in lexicon_words() if w not in harm and w not in neutral]


def test_eligibility_excludes_x83_and_x85_prompts_and_respects_the_rate_rule():
    labels = {
        tag: {1: [0] * 10, 2: [1] * 10, 3: [0] * 10, 4: [1] * 8 + [0] * 2}
        for tag in ("qwen", "qwen08b", "qwen2b", "qwen9b", "qwen27b", "llama", "gemma", "phi")
    }
    labels["llama"][3] = [1] * 10  # refused in one model only -> still excluded by the rate rule
    assert eligible_prompt_ids(labels, excluded={4}) == [1]
    assert eligible_prompt_ids(labels, excluded=set()) == [1, 4]


def test_the_frozen_cohort_is_x85_free_and_reproduces_its_own_draw():
    cohort = load_cohort()
    units = cohort["units"]
    assert len(units) == N_UNITS == 72
    x85_prompts, x85_words = leakage_sets(read_json(X85_COHORT))
    x85_neutral = {str(w["neutral"]) for w in read_json(X85_COHORT)["words"]}
    assert not ({int(u["id"]) for u in units} & (x85_prompts | x83_prompt_ids()))
    assert not ({u["word"] for u in units} & (x85_words | x85_neutral))
    assert build_exclusions(cohort)["counts"]["n_units_excluded"] == 0
    # the recorded draw is the one default_rng(114) makes over the recorded eligible set
    eligible = sorted({int(u["id"]) for u in units})
    assert len(eligible) == len(units)
    for unit in units:
        for position in ("S", "M", "E"):
            text, offset = insert_word(unit["prompts"]["O"], unit["word"], position)
            assert text == unit["prompts"][position] and offset == unit["insert_offsets"][position]


def test_draw_units_is_seeded_and_without_replacement():
    pool = [(f"w{i}", "cat") for i in range(5)]
    first = draw_units(list(range(100)), pool, 7, 114)
    assert first == draw_units(list(range(100)), pool, 7, 114)
    assert first != draw_units(list(range(100)), pool, 7, 115)
    ids = [i for i, _, _ in first]
    assert len(set(ids)) == len(ids) == 7 and ids == sorted(ids)
    assert all(word in {w for w, _ in pool} for _, word, _ in first)


def test_seeds_live_in_their_own_namespace_and_a_dose_reuses_its_base_arm_seed():
    assert stable_seed("qwen", 123, "E") != x101_seed("qwen", 123, "E")
    assert dose_seed("qwen", 123, "E") == stable_seed("qwen", 123, "E")
    assert dose_seed("qwen", 123, "E", retry=True) == stable_seed("qwen", 123, "E:retry4096")
    assert stable_seed("qwen", 123, "E") != stable_seed("llama", 123, "E")


def test_build_unit_records_offsets_and_hashes_for_every_condition():
    unit = build_unit(7, "a short harmful request here", "exampleword", "cat")
    assert set(unit["prompts"]) == {"O", "S", "M", "E"}
    for position in ("S", "M", "E"):
        text, offset = unit["prompts"][position], unit["insert_offsets"][position]
        assert text[offset : offset + len("exampleword")] == "exampleword"
    assert set(unit["prompt_sha256"]) == {"O", "S", "M", "E"}


def test_cohort_modes_pick_paths_x101_subsets_and_provenance():
    assert MODES == ("extended", "x101_leak_free", "full")
    base = Path("/tmp/x.json")
    assert result_path_for(base, "extended") == base
    assert result_path_for(base, "x101_leak_free") == Path("/tmp/x_x101only.json")
    assert result_path_for(base, "full") == Path("/tmp/x_full.json")
    assert (x101_mode("extended"), x101_mode("x101_leak_free"), x101_mode("full")) == ("leak_free", "leak_free", "full")
    assert (uses_x114("extended"), uses_x114("x101_leak_free"), uses_x114("full")) == (True, False, False)
    x101_exclusions = {
        "file": {"sha256": "aa"},
        "counts": {"n_units_excluded": 66},
        "excluded_prompt_ids": [1],
        "excluded_words": ["w"],
    }
    x114_exclusions = {"file": {"sha256": "bb"}, "counts": {"n_units_excluded": 0}, "x114_cohort_sha256": "cc"}
    record = cohort_record("extended", x101_exclusions, x114_exclusions, 64, 72)
    assert record["n_units_analysed"] == 136 and record["n_x101_leak_free"] == 64 and record["n_x114"] == 72
    assert record["x101_exclusions_sha256"] == "aa" and record["x114_exclusions_sha256"] == "bb"
    full = cohort_record("full", x101_exclusions, None, 130)
    assert full["n_x114"] == 0 and full["n_x101_units_excluded"] == 0 and full["excluded_x101_words"] == []


def test_pool_cells_concatenates_and_rebases_the_prompt_index():
    def part(source, ids, offset):
        n = len(ids) * 3
        return {
            "proj": np.arange(n, dtype=float) + offset,
            "shift": np.arange(n, dtype=float),
            "dr": np.arange(n, dtype=float),
            "pos_idx": np.tile([0, 1, 2], len(ids)),
            "prompt_idx": np.repeat(np.arange(len(ids)), 3),
            "deltas": np.zeros((n, 2, 4)),
            "cell_ids": [i for i in ids for _ in range(3)],
            "cell_words": ["w" for _ in range(n)],
            "proj_token": {"tbg_minus1": np.zeros(n), "tbg": np.zeros(n)},
            "ids": list(ids),
            "dropped": [],
            "r_o": np.zeros(len(ids)),
            "file": {"source": source},
            "source": source,
        }

    pooled = pool_cells([part("x101", [1, 2], 0), part("x114", [3, 4, 5], 100)])
    assert len(pooled["proj"]) == 15 and len(pooled["ids"]) == 5
    assert list(np.unique(pooled["prompt_idx"])) == [0, 1, 2, 3, 4]
    assert list(pooled["prompt_idx"][:3]) == [0, 0, 0] and list(pooled["prompt_idx"][6:9]) == [2, 2, 2]
    assert pooled["n_by_source"] == {"x101": 2, "x114": 3}
    assert pooled["file_by_source"]["x114"] == {"source": "x114"}
    single = pool_cells([part("x101", [1, 2], 0)])
    assert len(single["ids"]) == 2 and single["source"] == "x101"


def test_pool_labels_stacks_arms_over_the_pooled_prompt_set():
    arms = ("E_base", "E_true_m150")
    parts = [
        {"source": "x101", "ids": [1, 2], "labels": {arm: {1: np.ones(10), 2: np.zeros(10)} for arm in arms}},
        {"source": "x114", "ids": [3], "labels": {arm: {3: np.ones(10)} for arm in arms}},
    ]
    index, labels = pool_labels(parts)
    assert labels["E_base"].shape == (3, 10)
    assert [entry["source"] for entry in index] == ["x101", "x114"]
    assert float(labels["E_true_m150"].mean()) == 2 / 3
