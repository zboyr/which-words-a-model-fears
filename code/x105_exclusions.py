"""x105 exclusions: freeze the x85-overlap exclusion set the x103 and x105 analyses run without.

The per-model boundary directions (x87 consensus for Qwen3.5-4B, x88 consensus for the
other seven) were fit on the x85 crossed cohort. Of the 130 x101 units, 47 use an x85
prompt and 17 of the 75 inserted words are x85 harm words; the primary x103 and x105
results are reported on the 64 units that share neither. This stage derives those sets
with `x103_common.leakage_sets`, pins the x85 cohort's sha256 beside them and writes
`data/runs/x105/x105_x85_exclusions.json`, which both analyses load and re-verify.
"""

from __future__ import annotations

import argparse

from x105_common import atomic_json, build_exclusions, exclusions_path, load_cohort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(exclusions_path()))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = build_exclusions(load_cohort())
    atomic_json(args.out, record)
    counts = record["counts"]
    print(
        f"x105 exclusions -> {args.out}: {counts['n_units_prompt_in_x85']} units on x85 prompts, "
        f"{counts['n_units_word_in_x85']} on x85 harm words ({counts['n_words_in_x85']} words), "
        f"{counts['n_units_retained']} of {counts['n_units']} units retained"
    )


if __name__ == "__main__":
    main()
