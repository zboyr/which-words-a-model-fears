"""The cohort modes the x103 and x105 analyses run in, once x114 topped the cohort up.

Three modes, the same seeds, permutations, bootstrap and Holm scheme in each:

  `extended`        the canonical result files: the 64 x101 units whose prompt and inserted
                    word are both outside the x85 cohort the directions were fit on, plus
                    every x114 unit (built from prompts and words outside x85 by
                    construction, so the same exclusion rule drops none of them)
  `x101_leak_free`  the `_x101only` siblings: the 64 x101 units alone, as reported before x114
  `full`            the `_full` siblings: all 130 x101 units, exclusion off

This module owns the mode table, the result-path rule, the provenance block the result files
record, and the loaders that pool one model's x101 and x114 cells (x103) or arms (x105).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from x105_common import EXCLUSION_RULE, retained_ids
from x114_common import file_record, read_json

MODES = ("extended", "x101_leak_free", "full")
DEFAULT_MODE = "extended"
SUFFIX = {"extended": "", "x101_leak_free": "_x101only", "full": "_full"}
MODE_RULE = {
    "extended": (
        "the x101 units outside the x85 cohort the directions were fit on, plus the x114 units "
        "(x85-free by construction)"
    ),
    "x101_leak_free": EXCLUSION_RULE,
    "full": "all x101 cohort units; no exclusion, no x114 units",
}


def result_path_for(base: str | Path, mode: str) -> Path:
    base = Path(base)
    if mode not in SUFFIX:
        raise ValueError(f"unknown cohort mode {mode!r}; expected one of {MODES}")
    suffix = SUFFIX[mode]
    return base if not suffix else base.with_name(f"{base.stem}{suffix}{base.suffix}")


def x101_mode(mode: str) -> str:
    """Which x101 units a mode keeps, in x105_common's vocabulary."""
    return "full" if mode == "full" else "leak_free"


def uses_x114(mode: str) -> bool:
    return mode == "extended"


def cohort_record(
    mode: str,
    x101_exclusions: Mapping[str, Any],
    x114_exclusions: Mapping[str, Any] | None,
    n_x101: int,
    n_x114: int = 0,
) -> dict[str, Any]:
    """The provenance block every result file carries: both cohorts and their exclusion artifacts."""
    record = {
        "mode": mode,
        "rule": MODE_RULE[mode],
        "n_units_analysed": int(n_x101 + n_x114),
        "n_x101_leak_free": int(n_x101) if mode != "full" else 0,
        "n_x101_full": int(n_x101) if mode == "full" else 0,
        "n_x114": int(n_x114),
        "x101_exclusions": x101_exclusions["file"],
        "x101_exclusions_sha256": x101_exclusions["file"]["sha256"],
        "n_x101_units_excluded": x101_exclusions["counts"]["n_units_excluded"] if mode != "full" else 0,
        "excluded_x101_prompt_ids": x101_exclusions["excluded_prompt_ids"] if mode != "full" else [],
        "excluded_x101_words": x101_exclusions["excluded_words"] if mode != "full" else [],
    }
    if x114_exclusions is not None:
        record["x114_exclusions"] = x114_exclusions["file"]
        record["x114_exclusions_sha256"] = x114_exclusions["file"]["sha256"]
        record["n_x114_units_excluded_by_the_same_rule"] = x114_exclusions["counts"]["n_units_excluded"]
        record["x114_cohort_sha256"] = x114_exclusions["x114_cohort_sha256"]
    if not uses_x114(mode):
        record["n_x114"] = 0
    return record


# --- x103: pooling two models' cell sets --------------------------------------------------


def pool_cells(parts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Concatenate cell dictionaries (the layout `x103_analyze.load_cells` returns), re-basing
    the prompt index so the prompt-level bootstrap draws over the pooled prompt set."""
    parts = [part for part in parts if len(part["ids"])]
    if not parts:
        raise ValueError("no cells to pool")
    if len(parts) == 1:
        return dict(parts[0])
    out: dict[str, Any] = {}
    for key in ("proj", "shift", "dr", "pos_idx"):
        out[key] = np.concatenate([np.asarray(part[key]) for part in parts])
    out["deltas"] = np.concatenate([np.asarray(part["deltas"]) for part in parts])
    offsets, offset = [], 0
    for part in parts:
        offsets.append(offset)
        offset += len(part["ids"])
    out["prompt_idx"] = np.concatenate(
        [np.asarray(part["prompt_idx"]) + start for part, start in zip(parts, offsets)]
    )
    out["cell_ids"] = [i for part in parts for i in part["cell_ids"]]
    out["cell_words"] = [w for part in parts for w in part["cell_words"]]
    out["proj_token"] = {
        position: np.concatenate([np.asarray(part["proj_token"][position]) for part in parts])
        for position in parts[0]["proj_token"]
    }
    out["ids"] = [i for part in parts for i in part["ids"]]
    out["dropped"] = [i for part in parts for i in part["dropped"]]
    out["r_o"] = np.concatenate([np.asarray(part["r_o"]) for part in parts])
    out["file"] = parts[0]["file"]
    out["sources"] = [part.get("source", "x101") for part in parts]
    out["n_by_source"] = {part.get("source", "x101"): len(part["ids"]) for part in parts}
    out["file_by_source"] = {part.get("source", "x101"): part["file"] for part in parts}
    return out


# --- x105: pooling two cohorts' arms ------------------------------------------------------


def pool_labels(parts: Sequence[Mapping[str, Any]]) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Stack per-arm (prompts, 10) label matrices over the pooled prompt set.

    Each part is {"source": str, "ids": [...], "labels": {arm: {id: array}}}; every part must
    carry the same arms, and a prompt enters only if it is labelled in every arm of its part.
    """
    parts = [part for part in parts if len(part["ids"])]
    if not parts:
        raise ValueError("no arms to pool")
    arms = sorted(parts[0]["labels"])
    for part in parts:
        if sorted(part["labels"]) != arms:
            raise ValueError("the pooled cohorts do not carry the same arms")
    labels = {
        arm: np.concatenate([np.stack([part["labels"][arm][i] for i in part["ids"]]) for part in parts])
        for arm in arms
    }
    index = [{"source": part["source"], "ids": list(part["ids"])} for part in parts]
    return index, labels


def arm_loader(expected_experiment: str, generations_path, refusal_path):
    """A row-by-row validating loader for one dosed arm of one experiment's file pair."""
    import json

    from x105_common import ARMS, N_RESPONSES

    def load(tag: str, arm: str, units: Mapping[int, Mapping[str, Any]], cohort_sha: str):
        spec = next(s for s in ARMS if s["arm"] == arm)
        text_condition = spec["text"]
        generation_file, judge_file = generations_path(tag, arm), refusal_path(tag, arm)
        obj = read_json(generation_file)
        if (
            obj.get("experiment") != expected_experiment
            or obj.get("arm") != arm
            or obj.get("model_tag") != tag
            or obj.get("cohort_sha256") != cohort_sha
            or obj.get("text_condition") != text_condition
            or obj.get("site_audit", {}).get("status") != "pass"
        ):
            raise SystemExit(f"FATAL: {tag}/{arm} generation lineage or site audit is not valid")
        rows = {}
        for row in obj["rows"]:
            prompt_id = int(row["id"])
            if prompt_id not in units:
                continue
            if (
                row.get("condition") != arm
                or row.get("prompt_sha256") != units[prompt_id]["prompt_sha256"][text_condition]
                or row.get("arms_sha256") != cohort_sha
                or len(row.get("llm_responses", [])) != N_RESPONSES
                or abs(float(row["signed_dose"]) - float(obj["signed_dose"])) > 1e-12
            ):
                raise SystemExit(f"FATAL: invalid {tag}/{arm} generation row {prompt_id}")
            if prompt_id in rows:
                raise SystemExit(f"FATAL: duplicate {tag}/{arm} generation row {prompt_id}")
            rows[prompt_id] = row
        labels: dict[int, np.ndarray] = {}
        with judge_file.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("failed"):
                    continue
                prompt_id = int(record["id"])
                source = rows.get(prompt_id)
                if source is None:
                    continue
                values = record.get("refusal", [])
                if (
                    record.get("condition") != arm
                    or record.get("response_sha256") != source["response_sha256"]
                    or record.get("arms_sha256") != cohort_sha
                    or len(values) != N_RESPONSES
                    or not all(isinstance(value, bool) for value in values)
                ):
                    raise SystemExit(f"FATAL: invalid {tag}/{arm} refusal record {prompt_id}")
                if prompt_id in labels:
                    raise SystemExit(f"FATAL: duplicate successful {tag}/{arm} refusal record {prompt_id}")
                labels[prompt_id] = np.asarray(values, dtype=np.int8)
        meta = {
            "generations": file_record(generation_file),
            "refusal": file_record(judge_file),
            "signed_dose": float(obj["signed_dose"]),
            "family": obj["family"],
            "control_draw": obj.get("control_draw"),
            "text_condition": text_condition,
            "direction": obj["direction"].get("kind"),
            "site": obj["site"],
            "n_rows": len(rows),
            "n_labelled": len(labels),
        }
        return rows, labels, meta

    return load
