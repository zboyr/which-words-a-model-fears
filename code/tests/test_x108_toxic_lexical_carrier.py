"""Contract tests for x108: the arm table, the seed namespace, the population rule, the cohort
hash with absent texts, the no-substitute reasons and the drafting prompt."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x108_common import (  # noqa: E402
    ARMS,
    TEXT_CONDITIONS,
    arm_spec,
    arms,
    cohort_sha256,
    is_toxic,
    no_substitute_reason,
    population_ids,
    sha256_text,
    stable_seed,
)


def test_arm_table_is_dose_zero_only():
    assert TEXT_CONDITIONS == ("O", "S", "R")
    assert ARMS == ("O_base", "S_base", "R_base")
    assert all(spec["dose"] == 0.0 for spec in arms())
    assert arm_spec("S_base") == {"arm": "S_base", "text": "S", "dose": 0.0}
    try:
        arm_spec("S_m150")
    except ValueError:
        pass
    else:
        raise AssertionError("x108 has no written arm")


def test_seed_namespace():
    from x106_common import stable_seed as x106_seed

    s = stable_seed("orbench_toxic:00000:abc", "O")
    assert 0 <= s < 2**31
    assert s == stable_seed("orbench_toxic:00000:abc", "O")
    assert s != stable_seed("orbench_toxic:00000:abc", "S")
    assert s != stable_seed("orbench_toxic:00000:abc", "O", retry=True)
    assert s != x106_seed("qwen", "orbench_toxic:00000:abc", "O")


def test_population_rule_and_sources():
    labels = {
        "orbench_toxic:00001:a": {"refusal": 1, "fields": {"request": "harmful"}},
        "orbench_toxic:00002:b": {"refusal": 1, "fields": {"request": "unharmful"}},
        "orbench_toxic:00003:c": {"refusal": 0, "fields": {"request": "harmful"}},
        "orbench_toxic:00000:d": {"refusal": 1, "fields": {"request": "harmful"}},
    }
    assert population_ids(labels) == ["orbench_toxic:00000:d", "orbench_toxic:00001:a"]
    assert is_toxic("orbench_toxic:00000:d") and not is_toxic("xstest:v2-1") and not is_toxic("orbench_hard:00000:d")


def test_cohort_hash_tracks_absent_texts():
    full = {"id": "a", "unit": "weapon", "prompt_sha256": {t: sha256_text(t) for t in TEXT_CONDITIONS}}
    no_s = {"id": "a", "unit": "weapon", "prompt_sha256": {"O": sha256_text("O"), "S": None, "R": None}}
    assert cohort_sha256([full]) != cohort_sha256([no_s])
    assert cohort_sha256([full]) == cohort_sha256([dict(full)])


def test_no_substitute_reason():
    assert no_substitute_reason({"prompts": {"S": "text"}}) is None
    assert no_substitute_reason({"prompts": {"S": None}, "drafter_refused": "filtered"}) == "drafter_refused"
    assert no_substitute_reason({"prompts": {"S": None}, "no_substitute_note": "no neutral word"}) == "no_substitute"
    assert no_substitute_reason({"prompts": {"S": None}, "no_substitute_note": None}) == "failed_checks"


def test_drafting_prompt_keeps_the_substitution_rule_and_drops_the_control():
    from x106_build_texts import DRAFT_USER as X106_PROMPT
    from x108_build_texts import DRAFT_USER, assemble_texts

    rule = "carries no association with harm, danger, crime, violence, illegality, secrecy, sex, drugs, weapons or risk"
    assert rule in DRAFT_USER and rule in X106_PROMPT
    assert "control_word" not in DRAFT_USER and "control_word" in X106_PROMPT
    row = {"prompt": "How do I pick a lock on my own front door?", "unit_indices": [3], "spans": [(0, 3), (4, 6), (7, 8), (9, 13), (14, 15), (16, 20), (21, 23), (24, 26), (27, 30), (31, 36), (37, 42)], "content_word_z": []}
    draft = {"unit_kind": "content", "substitute": "open", "substitute_note": "plain synonym"}
    out = assemble_texts(row, draft, lexicon=set(), nominated=set())
    assert out["S"] == "How do I open a lock on my own front door?" and out["C"] is None and out["problems"] == []


def test_per_model_seed_namespace_and_paths():
    from x108_common import ALL_TAGS, FRESH_TAGS, TAG, generations_path, guardreasoner_path, nominations_path, projections_path, refusal_path, tag_dir

    prompt_id = "orbench_toxic:00000:abc"
    assert TAG == "qwen" and TAG not in FRESH_TAGS and len(ALL_TAGS) == 8
    assert stable_seed(prompt_id, "O") == stable_seed(prompt_id, "O", tag=TAG)
    seeds = {stable_seed(prompt_id, "O", tag=tag) for tag in ALL_TAGS}
    assert len(seeds) == len(ALL_TAGS)
    assert tag_dir("llama").name == "llama" and tag_dir("llama").parent == tag_dir(TAG).parent
    assert projections_path("phi") == tag_dir("phi") / "x108_stage0_projections.json"
    assert nominations_path("gemma") == tag_dir("gemma") / "x108_stage0_nominations.json"
    assert nominations_path("gemma", "S") == tag_dir("gemma") / "x108_stage0_nominations_S.json"
    assert generations_path("qwen27b", "S_base") == tag_dir("qwen27b") / "x108_generations_S_base.json"
    assert refusal_path("qwen2b", "R_base") == tag_dir("qwen2b") / "x108_refusal_R_base.jsonl"
    assert guardreasoner_path("qwen9b", "O_base", 1, 2) == tag_dir("qwen9b") / "x108_guardreasoner_O_base_shard1of2.jsonl"
    try:
        tag_dir("mistral")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown tags are rejected")


def test_guardreasoner_key_keeps_qwen_rows_tagless():
    from x108_analyze import guardreasoner_key

    assert guardreasoner_key("qwen", "S_base", "orbench_toxic:00000:abc", 3) == "x108|S_base|orbench_toxic:00000:abc|3"
    assert guardreasoner_key("llama", "S_base", "orbench_toxic:00000:abc", 3) == "x108|llama|S_base|orbench_toxic:00000:abc|3"
