"""x113 exclusions: freeze the x85-overlap exclusion set the x113 analysis runs without.

Every model's boundary direction was fitted on the x85 crossed cohort (96 prompts crossed with
24 harm words and 24 neutral twins), and the vocabulary tables x113 nominates with are read on
that direction, so those 48 words are in-sample for the table. None of the 397 x113 prompts is
an x85 prompt and no scan prompt of the x111 BP or x112 tables is one either; what the tables
carry in-sample is the words. A unit is excluded when its table nominee is one of the 48 words,
when its prompt contains an x85 harm word anywhere, or when the substitute or the control word
of either arm is an x85 word. The exclusion is per model, because the nominee is.

Writes data/runs/x113/x85_exclusions.json, which x113_analyze.py loads and re-verifies; the
analysis drops these units by default and `--full` keeps them, writing the `_full` siblings.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x85_common import WORD_PAIRS  # noqa: E402
from x113_common import (  # noqa: E402
    ALL_TAGS,
    DATA,
    PROMPTS,
    LABELS,
    STRATA,
    atomic_json,
    exclusions_path,
    file_record,
    load_prompts,
    load_texts,
    normalize_word,
    read_json,
    sha256_text,
)

X85_COHORT = DATA / "runs" / "x85" / "x85_cohort.json"
X85_HARM = frozenset(harm for _, harm, _ in WORD_PAIRS)
X85_NEUTRAL = frozenset(neutral for _, _, neutral in WORD_PAIRS)
X85_WORDS = X85_HARM | X85_NEUTRAL
REASONS = ("nominee_x85_harm", "nominee_x85_neutral", "prompt_has_x85_harm", "substitute_x85", "control_x85")


def unit_reasons(unit: dict) -> list[str]:
    """Why this unit leaves the primary analysis; empty when it stays."""
    nominee = (unit.get("nominee") or {}).get("norm")
    prompt_words = {normalize_word(w) for w in unit["prompt"].split()}
    substitute_words = {normalize_word(w) for w in ((unit.get("substitute") or {}).get("substitute") or "").split()}
    control = unit.get("control") or {}
    control_words = {normalize_word(w) for w in (f"{control.get('norm') or ''} {control.get('synonym') or ''}").split()}
    out = []
    if nominee in X85_HARM:
        out.append("nominee_x85_harm")
    if nominee in X85_NEUTRAL:
        out.append("nominee_x85_neutral")
    if prompt_words & X85_HARM:
        out.append("prompt_has_x85_harm")
    if substitute_words & X85_WORDS:
        out.append("substitute_x85")
    if control_words & X85_WORDS:
        out.append("control_x85")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(exclusions_path()))
    args = parser.parse_args()
    cohort = read_json(X85_COHORT)
    prompts = load_prompts()
    x85_prompt_sha = {str(r["prompt_sha256"]) for r in cohort["prompts"]}
    overlap = [u["id"] for u in prompts["units"] if u["prompt_sha256"] in x85_prompt_sha]
    models = {}
    for tag in ALL_TAGS:
        texts = load_texts(tag)
        excluded = []
        for unit in texts["units"]:
            reasons = unit_reasons(unit)
            if reasons:
                excluded.append({"id": unit["id"], "stratum": unit["stratum"], "nominee": (unit.get("nominee") or {}).get("norm"), "reasons": reasons})
        ids = {e["id"] for e in excluded}
        retained = [u for u in texts["units"] if u["id"] not in ids]
        models[tag] = {
            "label": LABELS[tag],
            "n_excluded": len(excluded),
            "n_retained": len(retained),
            "n_excluded_by_reason": {r: sum(1 for e in excluded if r in e["reasons"]) for r in REASONS},
            "n_excluded_by_stratum": {s: sum(1 for e in excluded if e["stratum"] == s) for s in STRATA},
            "n_retained_by_stratum": {s: sum(1 for u in retained if u["stratum"] == s) for s in STRATA},
            "n_retained_with_S_T": sum(1 for u in retained if u["texts"]["S_T"] is not None),
            "n_retained_with_C_T": sum(1 for u in retained if u["texts"]["C_T"] is not None),
            "texts_sha256": texts["texts_sha256"],
            "excluded": excluded,
        }
        print(f"  {tag:8s} excluded {len(excluded):3d} of {len(texts['units'])}, retained {len(retained):3d} {models[tag]['n_excluded_by_reason']}")
    record = {
        "run": "x113",
        "artifact": "x85_exclusions",
        "purpose": (
            "Units of the x113 population whose table nomination or whose text touches the x85 crossed cohort, "
            "the cohort every model's boundary direction was fitted on and the vocabulary tables are read through. "
            "code/x113_analyze.py drops them by default; --full analyses every unit into the _full siblings."
        ),
        "rule": (
            "A unit is excluded when its table nominee is one of the 24 x85 harm words or 24 neutral twins, when its prompt "
            "contains an x85 harm word anywhere (whitespace words, edge punctuation stripped, lower-cased), or when the S_T "
            "substitute or the C_T control word or synonym contains an x85 word. The exclusion is per model: the nominee is."
        ),
        "x85_cohort": {**file_record(X85_COHORT), "cohort_sha256": cohort["cohort_sha256"], "n_prompts": len(cohort["prompts"]), "n_harm_words": len(X85_HARM), "n_neutral_words": len(X85_NEUTRAL)},
        "x85_prompt_sha256": sorted(x85_prompt_sha),
        "x85_prompt_overlap_with_x113": {"n": len(overlap), "ids": overlap, "note": "the x113 population shares no prompt with the x85 cohort; the x111 BP table's 48 scan prompts and x112's eight hold none either"},
        "x85_harm_words": sorted(X85_HARM),
        "x85_neutral_words": sorted(X85_NEUTRAL),
        "x113_prompts": {**file_record(PROMPTS), "units_sha256": prompts["units_sha256"], "n_units": prompts["n_units"]},
        "reasons": list(REASONS),
        "models": models,
    }
    atomic_json(args.out, record)
    print(f"x113 exclusions -> {args.out}: {min(m['n_retained'] for m in models.values())}-{max(m['n_retained'] for m in models.values())} units retained per model")


if __name__ == "__main__":
    main()
