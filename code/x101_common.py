"""x101: one inserted harm keyword, eight transfer models (cross-model x83).

Plan: knowledge/plans/x101-cross-model-keyword-insertion.md (descriptive, not
pre-registered). Everything the plan calls "x83 verbatim" is imported from
x83_common / x83_detect / x83_analyze / x81_generate rather than restated; the
model registry (ids, Phi native-impl flag) and the per-model chat rendering are
imported from x88_common. This module adds only what x101 changes: the shared
130-prompt cohort rule, per-tag paths and seeds, and the gate scoring offset
for templates that emit fixed tokens before the answer.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from x83_common import (  # noqa: F401  (re-exported for the stage scripts)
    CONDITIONS,
    DATA,
    DETECT_INSTRUCTION,
    DETECT_THRESHOLD,
    LEXICON,
    MAX_HISTORICAL_RATE,
    N_RESPONSES,
    POSITIONS,
    RESULTS,
    RUN as X83_RUN,
    atomic_json,
    cohort_path as x83_cohort_path,
    file_record,
    insert_word,
    lexicon_sha256,
    read_json,
    sha256_file,
    sha256_text,
    surface_variants,
)
from x88_common import (  # noqa: F401  (re-exported for the stage scripts)
    IMPORTED_TAG,
    MODELS,
    RUN as X88_RUN,
    TAGS as FRESH_TAGS,
    format_user_chat,
    load_model,
    load_tokenizer,
)

RUN = DATA / "runs" / "x101"
LABELS = DATA / "labels"
BASE_SEED = 101
EXPECTED_COHORT = 130
ALL_TAGS = (IMPORTED_TAG,) + FRESH_TAGS  # qwen (Qwen3.5-4B, imported from x83) first
QWEN4B_MODEL_ID = "Qwen/Qwen3.5-4B"
GREEDY_TOKENS = 8  # x83_detect.GREEDY_TOKENS; repeated here so the pure helpers need no torch
N_GATE_PROBES = 4  # distinct-word detection prompts used to find fixed pre-answer tokens

# Two template-audit prompts rendered per model before anything is generated.
AUDIT_PROMPTS = (
    "x101 chat-template audit",
    "A second, longer audit prompt to show the scaffold around the user turn is a template constant.",
)


def labels_path(tag: str) -> Path:
    if tag not in ALL_TAGS:
        raise ValueError(f"unknown x101 tag {tag!r}")
    return LABELS / f"{tag}_labels_v4.json"


def cohort_path() -> Path:
    return RUN / "x101_cohort.json"


def manifest_path() -> Path:
    return RUN / "x101_manifest.json"


def tag_dir(tag: str) -> Path:
    if tag not in ALL_TAGS:
        raise ValueError(f"unknown x101 tag {tag!r}")
    return RUN / tag


def preflight_path(tag: str) -> Path:
    return tag_dir(tag) / "x101_preflight.json"


def detect_path(tag: str) -> Path:
    return tag_dir(tag) / "x101_detect.json"


def generations_path(tag: str, condition: str) -> Path:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    return tag_dir(tag) / f"x101_generations_{condition}.json"


def refusal_path(tag: str, condition: str) -> Path:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    return tag_dir(tag) / f"x101_refusal_{condition}.jsonl"


def stable_seed(tag: str, prompt_id: int, condition: str, base: int = BASE_SEED) -> int:
    """Per-(tag, prompt, condition) sampling seed in the x101 namespace."""
    payload = f"x101|{base}|{tag}|{prompt_id}|{condition}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def load_tag_historical_refusals(tag: str, prompt_ids: Iterable[int]) -> dict[int, list[int]]:
    """Historical refusal2 arrays of one campaign model for the given prompts."""
    rows = read_json(labels_path(tag))["rows"]
    wanted = sorted({int(value) for value in prompt_ids})
    result: dict[int, list[int]] = {}
    for prompt_id in wanted:
        row = rows.get(str(prompt_id))
        if row is None:
            raise ValueError(f"{tag}: no historical labels for prompt {prompt_id}")
        values = row["refusal"]
        if len(values) != N_RESPONSES or any(value not in (0, 1, False, True) for value in values):
            raise ValueError(f"{tag}: invalid historical refusal array for {prompt_id}")
        result[prompt_id] = [int(value) for value in values]
    return result


def historical_rate(values: Sequence[int]) -> float:
    return sum(int(value) for value in values) / N_RESPONSES


def select_cohort_ids(
    x83_units: Sequence[Mapping[str, Any]],
    historical_by_tag: Mapping[str, Mapping[int, Sequence[int]]],
    max_rate: float = MAX_HISTORICAL_RATE,
) -> list[int]:
    """x83 prompts whose historical refusal2 rate is <= max_rate in every tag (sorted ids).

    Reads historical labels only; no x101 output can enter this selection.
    """
    if set(historical_by_tag) != set(ALL_TAGS):
        raise ValueError("cohort selection needs historical labels for all eight tags")
    kept = []
    for unit in x83_units:
        prompt_id = int(unit["id"])
        if all(historical_rate(historical_by_tag[tag][prompt_id]) <= max_rate for tag in ALL_TAGS):
            kept.append(prompt_id)
    return sorted(kept)


def inherit_unit(x83_unit: Mapping[str, Any]) -> dict[str, Any]:
    """Copy one x83 unit (word, category, exact O/S/M/E texts) after re-deriving its texts."""
    word = str(x83_unit["word"])
    prompts = dict(x83_unit["prompts"])
    offsets = dict(x83_unit["insert_offsets"])
    for position in POSITIONS:
        text, offset = insert_word(prompts["O"], word, position)
        if text != prompts[position] or offset != offsets[position]:
            raise ValueError(f"x83 unit {x83_unit['id']}: {position} text is not x83's insertion rule")
    for condition in CONDITIONS:
        if sha256_text(prompts[condition]) != x83_unit["prompt_sha256"][condition]:
            raise ValueError(f"x83 unit {x83_unit['id']}: {condition} sha256 mismatch")
    return {
        "id": int(x83_unit["id"]),
        "word": word,
        "category": str(x83_unit["category"]),
        "prompts": prompts,
        "insert_offsets": offsets,
        "prompt_sha256": dict(x83_unit["prompt_sha256"]),
        "detection_prompts_sha256": {
            position: sha256_text(DETECT_INSTRUCTION.format(text=prompts[position]))
            for position in POSITIONS
        },
    }


def cohort_sha256(units: Sequence[Mapping[str, Any]]) -> str:
    """The x83 cohort-hash rule (id, word, four prompt hashes per line)."""
    return sha256_text(
        "\n".join(
            f"{unit['id']}\t{unit['word']}\t" + "\t".join(unit["prompt_sha256"][c] for c in CONDITIONS)
            for unit in units
        )
    )


def fixed_pre_answer_ids(greedy_token_lists: Sequence[Sequence[int]]) -> list[int]:
    """Longest common token prefix of greedy detection answers to distinct-word probes.

    The probes name different words, so any shared prefix is template machinery
    the model emits before every answer (e.g. an empty thinking block), not the
    answer itself. Empty for a template whose first generated token is the answer.
    """
    if len(greedy_token_lists) < 2:
        raise ValueError("need at least two probes to separate fixed tokens from the answer")
    prefix: list[int] = []
    for position, token_id in enumerate(greedy_token_lists[0]):
        if all(len(other) > position and other[position] == token_id for other in greedy_token_lists[1:]):
            prefix.append(int(token_id))
        else:
            break
    if prefix and len(prefix) == min(len(seq) for seq in greedy_token_lists):
        raise ValueError("probes are identical over their whole length; cannot locate the answer position")
    return prefix


def template_like(decoded: str) -> bool:
    """A fixed prefix may only be scaffolding: special tokens (<...>) and whitespace, no words."""
    stripped = re.sub(r"<[^<>\s]*>", "", decoded)
    return not any(character.isalnum() for character in stripped)


def scoring_prefix_ids(chat_ids: Sequence[int], fixed_ids: Sequence[int]) -> list[int]:
    """Teacher-forcing prefix for the gate: rendered chat, then the fixed pre-answer tokens."""
    return [int(value) for value in chat_ids] + [int(value) for value in fixed_ids]


def gate_scoring_position(chat_ids: Sequence[int], fixed_ids: Sequence[int]) -> int:
    """Index of the first scored (answer) token in the teacher-forced sequence."""
    return len(scoring_prefix_ids(chat_ids, fixed_ids))


def load_cohort() -> dict[str, Any]:
    cohort = read_json(cohort_path())
    if cohort.get("experiment") != "x101" or cohort.get("status") != "frozen":
        raise SystemExit("FATAL: x101 cohort is missing or not frozen")
    units = list(cohort["units"])
    if len(units) != EXPECTED_COHORT or cohort.get("cohort_sha256") != cohort_sha256(units):
        raise SystemExit("FATAL: x101 cohort fails its size or sha256 check")
    return cohort


def load_preflight(tag: str) -> dict[str, Any]:
    if tag not in FRESH_TAGS:
        raise ValueError(f"{tag} is not a fresh x101 tag")
    preflight = read_json(preflight_path(tag))
    if preflight.get("experiment") != "x101" or preflight.get("model_tag") != tag:
        raise SystemExit(f"FATAL: invalid x101 preflight for {tag}")
    if preflight.get("status") != "pass":
        raise SystemExit(f"FATAL: {tag} preflight did not pass: {preflight.get('status')}")
    return preflight
