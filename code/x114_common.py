"""x114: an x85-free top-up of the cross-model insertion cohort, then x103 and x105 on it.

The per-model boundary directions (x87 consensus for Qwen3.5-4B, x88 consensus for the
other seven) were fit on the x85 crossed cohort, and the x101 insertion cohort shares 47 of
its 96 prompts and 17 of its 24 harm words, which left x103 and x105 with 64 units outside
that cohort. x114 builds 72 further insertion units on campaign prompts that are neither
x83 nor x85 prompts, with words drawn from the x83 lexicon minus x85's 24 harm words and
24 neutral twins, runs x101's exact generation / detection / judge contract on them in all
eight models (Qwen3.5-4B included, run fresh), then x103's boundary-state extraction and
x105's eight dosed arms. The x103 and x105 analyses take "x101's 64 leak-free units plus
the x114 units" as their primary cohort.

Everything x101 / x103 / x105 fixed is imported, not restated: the insertion rule, the
detection gate, the sampling engine and judge, the model registry and chat rendering, the
x88 hook site and direction artifacts, the x105 arm table, control directions and KL
calibration. This module adds the x114 cohort rule, paths and seeds.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from x83_common import (  # noqa: F401  (re-exported for the stage scripts)
    CONDITIONS,
    DATA,
    DETECT_INSTRUCTION,
    DETECT_THRESHOLD,
    LABELS_PATH,
    LEXICON,
    MAX_HISTORICAL_RATE,
    N_RESPONSES,
    POSITIONS,
    RESULTS,
    RUN as X83_RUN,
    TEST_INDICES_PATH,
    atomic_json,
    cohort_path as x83_cohort_path,
    file_record,
    insert_word,
    lexicon_sha256,
    lexicon_words,
    read_json,
    sha256_file,
    sha256_text,
    surface_variants,
)
from x81_common import PROMPTS_PATH, load_prompt_rows, split_prompt_ids
from x101_common import (  # noqa: F401  (re-exported for the stage scripts)
    ALL_TAGS,
    FRESH_TAGS,
    IMPORTED_TAG,
    MODELS,
    QWEN4B_MODEL_ID,
    X88_RUN,
    cohort_path as x101_cohort_path,
    cohort_sha256,
    format_user_chat,
    historical_rate,
    labels_path,
    load_tag_historical_refusals,
    manifest_path as x101_manifest_path,
    scoring_prefix_ids,
)
from x103_common import X85_COHORT, leakage_sets

RUN = DATA / "runs" / "x114"
BASE_SEED = 114
N_UNITS = 72
TAGS = ALL_TAGS  # qwen (Qwen3.5-4B) first; every tag runs fresh here, nothing is imported
AUDIT_PROMPT = "x114 chat-template audit"  # rendered once per generation job to record the scaffold


# --- paths ---------------------------------------------------------------------------


def cohort_path() -> Path:
    return RUN / "x114_cohort.json"


def manifest_path() -> Path:
    return RUN / "x114_manifest.json"


def exclusions_path() -> Path:
    return RUN / "x114_x85_exclusions.json"


def inputs_path() -> Path:
    return RUN / "x114_inputs.json"


def tag_dir(tag: str) -> Path:
    if tag not in TAGS:
        raise ValueError(f"unknown x114 tag {tag!r}")
    return RUN / tag


def detect_path(tag: str) -> Path:
    return tag_dir(tag) / "x114_detect.json"


def generations_path(tag: str, condition: str) -> Path:
    """O/S/M/E base arms and the eight x105 dosed arms share one naming rule."""
    label = arm_label(condition)
    return tag_dir(tag) / f"x114_generations_{label}.json"


def refusal_path(tag: str, condition: str) -> Path:
    label = arm_label(condition)
    return tag_dir(tag) / f"x114_refusal_{label}.jsonl"


def projections_path(tag: str) -> Path:
    return tag_dir(tag) / "x114_projections.json"


def deltas_path(tag: str) -> Path:
    return tag_dir(tag) / "x114_boundary_deltas.npz"


def arm_label(condition: str) -> str:
    from x105_common import ARM_NAMES

    if condition in CONDITIONS or condition in ARM_NAMES:
        return condition
    raise ValueError(f"unknown x114 condition / arm {condition!r}")


def stable_seed(tag: str, prompt_id: int, condition: str, base: int = BASE_SEED) -> int:
    """Per-(tag, prompt, condition) sampling seed in the x114 namespace (x101's rule, base 114)."""
    payload = f"x114|{base}|{tag}|{prompt_id}|{condition}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def dose_seed(tag: str, prompt_id: int, text_condition: str, retry: bool = False) -> int:
    """A dosed arm reuses its base arm's seed (x105's common-random-number contract)."""
    label = f"{text_condition}:retry4096" if retry else text_condition
    return stable_seed(tag, int(prompt_id), label)


# --- the cohort rule --------------------------------------------------------------------


def word_pool() -> list[tuple[str, str]]:
    """The x83 lexicon (frozen order) minus x85's 24 harm words and 24 neutral twins."""
    x85 = read_json(X85_COHORT)
    harm = {str(w["harm"]) for w in x85["words"]}
    neutral = {str(w["neutral"]) for w in x85["words"]}
    return [(word, category) for word, category in lexicon_words() if word not in harm and word not in neutral]


def x83_prompt_ids() -> set[int]:
    x83 = read_json(x83_cohort_path())
    if x83.get("experiment") != "x83":
        raise ValueError("not the x83 cohort")
    return {int(u["id"]) for u in x83["units"]}


def x85_prompt_ids() -> set[int]:
    return leakage_sets(read_json(X85_COHORT))[0]


def eligible_prompt_ids(
    historical_by_tag: Mapping[str, Mapping[int, Sequence[int]]],
    excluded: Iterable[int],
    max_rate: float = MAX_HISTORICAL_RATE,
) -> list[int]:
    """Campaign prompts with historical refusal2 rate <= max_rate in every tag, minus the
    excluded ids (x83's 321 and x85's 96 prompts). Reads historical labels only."""
    if set(historical_by_tag) != set(TAGS):
        raise ValueError("eligibility needs historical labels for all eight tags")
    ids = set.intersection(*(set(historical_by_tag[tag]) for tag in TAGS)) - {int(v) for v in excluded}
    return sorted(i for i in ids if all(historical_rate(historical_by_tag[tag][i]) <= max_rate for tag in TAGS))


def draw_units(eligible: Sequence[int], pool: Sequence[tuple[str, str]], n_units: int, seed: int) -> list[tuple[int, str, str]]:
    """One rng: n_units prompt ids without replacement, then one uniform word per prompt
    in ascending id order. Returns (prompt id, word, category) triples."""
    rng = np.random.default_rng(seed)
    chosen = sorted(int(v) for v in rng.choice(np.asarray(sorted(eligible), dtype=np.int64), size=n_units, replace=False))
    out = []
    for prompt_id in chosen:
        word, category = pool[int(rng.integers(0, len(pool)))]
        out.append((prompt_id, word, category))
    return out


def build_unit(prompt_id: int, prompt: str, word: str, category: str) -> dict[str, Any]:
    """x83's unit layout: O/S/M/E strings from `insert_word`, offsets and sha256 per text."""
    prompts = {"O": prompt}
    offsets: dict[str, int | None] = {"O": None}
    for position in POSITIONS:
        text, offset = insert_word(prompt, word, position)
        if text[offset : offset + len(word)] != word:
            raise ValueError(f"insertion offset mismatch for {prompt_id}/{position}")
        prompts[position] = text
        offsets[position] = offset
    return {
        "id": int(prompt_id),
        "word": word,
        "category": category,
        "prompts": prompts,
        "insert_offsets": offsets,
        "prompt_sha256": {key: sha256_text(value) for key, value in prompts.items()},
        "detection_prompts_sha256": {
            position: sha256_text(DETECT_INSTRUCTION.format(text=prompts[position])) for position in POSITIONS
        },
    }


# --- frozen artifacts -----------------------------------------------------------------------


def load_cohort() -> dict[str, Any]:
    cohort = read_json(cohort_path())
    if cohort.get("experiment") != "x114" or cohort.get("status") != "frozen":
        raise SystemExit("FATAL: x114 cohort is missing or not frozen")
    units = list(cohort["units"])
    if len(units) != int(cohort["counts"]["cohort"]) or cohort.get("cohort_sha256") != cohort_sha256(units):
        raise SystemExit("FATAL: x114 cohort fails its size or sha256 check")
    return cohort


def load_manifest(tag: str | None = None) -> dict[str, Any]:
    manifest = read_json(manifest_path())
    if manifest.get("experiment") != "x114" or manifest.get("status") != "frozen":
        raise SystemExit("FATAL: x114 manifest is missing or not frozen")
    if tag is None:
        return manifest
    if tag not in manifest["models"]:
        raise SystemExit(f"FATAL: x114 manifest has no entry for {tag}")
    return manifest["models"][tag]


def build_exclusions(cohort: Mapping[str, Any]) -> dict[str, Any]:
    """The x105 exclusion rule applied to the x114 cohort; by construction it excludes nothing."""
    from x105_common import EXCLUSION_RULE

    x85 = read_json(X85_COHORT)
    x85_prompts, x85_words = leakage_sets(x85)
    units = [(int(u["id"]), str(u["word"])) for u in cohort["units"]]
    prompt_hits = sorted(i for i, _ in units if i in x85_prompts)
    word_hits = sorted({w for _, w in units if w in x85_words})
    retained = sorted(i for i, w in units if i not in x85_prompts and w not in x85_words)
    return {
        "experiment": "x114",
        "stage": "exclusions",
        "status": "frozen",
        "rule": EXCLUSION_RULE.replace("x101 unit", "x114 unit"),
        "derivation": "x103_common.leakage_sets(x85 cohort), intersected with the x114 cohort's prompt ids and words",
        "x85_cohort": file_record(X85_COHORT),
        "x85_cohort_sha256": x85["cohort_sha256"],
        "x114_cohort": file_record(cohort_path()),
        "x114_cohort_sha256": cohort["cohort_sha256"],
        "x85_prompt_ids": sorted(x85_prompts),
        "x85_harm_words": sorted(x85_words),
        "excluded_prompt_ids": prompt_hits,
        "excluded_words": word_hits,
        "retained_unit_ids": retained,
        "counts": {
            "n_units": len(units),
            "n_units_prompt_in_x85": len(prompt_hits),
            "n_units_word_in_x85": sum(w in x85_words for _, w in units),
            "n_units_excluded": len(units) - len(retained),
            "n_units_retained": len(retained),
            "n_distinct_words_retained": len({w for i, w in units if i in set(retained)}),
        },
    }


def load_exclusions(cohort: Mapping[str, Any], path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path is not None else exclusions_path()
    frozen = read_json(path)
    if frozen.get("experiment") != "x114" or frozen.get("stage") != "exclusions" or frozen.get("status") != "frozen":
        raise SystemExit(f"FATAL: {path} is not the frozen x114 exclusion artifact")
    if frozen.get("x114_cohort_sha256") != cohort["cohort_sha256"]:
        raise SystemExit("FATAL: the x114 exclusion artifact belongs to another cohort")
    live = build_exclusions(cohort)
    for key in ("x85_cohort_sha256", "excluded_prompt_ids", "excluded_words", "retained_unit_ids", "counts"):
        if frozen.get(key) != live[key]:
            raise SystemExit(f"FATAL: the x114 exclusion artifact's {key} differs from the live derivation")
    frozen["file"] = file_record(path)
    return frozen


# --- behaviour loaders (x101's readers on the x114 files) ------------------------------------


def unjudged_prompts(path: Path, condition: str, units: Mapping[int, Any]) -> list[int]:
    successful = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not row.get("failed") and row.get("condition") == condition and int(row["id"]) in units:
                successful.add(int(row["id"]))
    return sorted(set(units) - successful)


def load_refusal_arrays(tag: str, cohort: Mapping[str, Any], max_dropped: int = 3) -> tuple[list[int], dict[str, np.ndarray], list[int]]:
    """Per-prompt refusal2 arrays [n, 10] per condition on the x114 units, exactly as x101's
    analysis read its own files; a prompt unjudged in any condition is dropped in all four."""
    from x83_analyze import load_generations, load_judgments

    units = {int(u["id"]): u for u in cohort["units"]}
    excluded = {c: unjudged_prompts(refusal_path(tag, c), c, units) for c in CONDITIONS}
    dropped = sorted(set().union(*excluded.values()))
    if len(dropped) > max_dropped:
        raise SystemExit(f"FATAL: {tag} lacks x114 labels for {len(dropped)} prompts: {dropped[:10]}")
    units = {i: u for i, u in units.items() if i not in dropped}
    ids = sorted(units)
    sha = cohort["cohort_sha256"]
    generations = {c: load_generations(generations_path(tag, c), c, units, sha) for c in CONDITIONS}
    labels = {c: load_judgments(refusal_path(tag, c), c, units, generations[c], sha) for c in CONDITIONS}
    return ids, {c: np.stack([labels[c][i] for i in ids]) for c in CONDITIONS}, dropped


def expected_renderings(tag: str, cohort: Mapping[str, Any], conditions: Sequence[str] = CONDITIONS) -> dict[str, dict[str, str]]:
    """{condition: {prompt id: rendered_chat_sha256}} from the x114 generation rows."""
    units = {int(u["id"]): u for u in cohort["units"]}
    out: dict[str, dict[str, str]] = {}
    for condition in conditions:
        obj = read_json(generations_path(tag, condition))
        if obj.get("experiment") != "x114" or obj.get("condition") != condition or obj.get("model_tag") != tag:
            raise ValueError(f"{tag}/{condition}: not the x114 generation file")
        rows = {int(r["id"]): r for r in obj["rows"]}
        out[condition] = {}
        for prompt_id, unit in units.items():
            row = rows[prompt_id]
            if row["prompt_sha256"] != unit["prompt_sha256"][condition]:
                raise ValueError(f"{tag}/{condition}: generation row {prompt_id} is not the cohort text")
            out[condition][str(prompt_id)] = row["rendered_chat_sha256"]
    return out
