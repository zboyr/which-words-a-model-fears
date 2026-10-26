"""x113 Stage 0 (local): freeze the population from x106's cohort and its O_base refusal2 rows.

core: the Hard-1K cohort prompts whose x106 O_base refusal2 (ten samples) is at or above 0.5 in
at least five of the eight models. llama_extra: the remaining Hard-1K cohort prompts whose
Llama-3.2-3B O_base refusal2 is above zero, in descending order of that rate, at most 100.
xstest: x106's 100 safe XSTest prompts. Writes data/runs/x113/prompts.json with the rules, the
ids, the stratum and every model's r(O). `--expect` compares the core / llama_extra ids with an
externally computed list.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x113_common import (  # noqa: E402
    ALL_TAGS,
    CORE_MIN_MODELS,
    CORE_RATE,
    LLAMA_EXTRA_MAX,
    PROMPTS,
    STRATA,
    atomic_json,
    cohort_path,
    file_record,
    is_xstest,
    load_cohort,
    read_json,
    units_sha256,
    x106_refusal_path,
    x106_refusal_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect", help="JSON with core_ids and llama_extra_ids to compare against")
    args = parser.parse_args()
    cohort = load_cohort()
    units = {u["id"]: u for u in cohort["units"]}
    rates: dict[str, dict[str, float]] = {}
    for tag in ALL_TAGS:
        rows = x106_refusal_rows(tag, "O_base")
        r: dict[str, float] = {}
        for prompt_id, rec in rows.items():
            u = units.get(prompt_id)
            if u is None:
                continue
            if rec.get("prompt_sha256") != u["prompt_sha256"]["O"] or rec.get("arms_sha256") != cohort["cohort_sha256"]:
                raise SystemExit(f"FATAL: {tag}: x106 O_base refusal row {prompt_id} is not the cohort's O text")
            r[prompt_id] = float(np.mean(rec["refusal"]))
        missing = [i for i in units if i not in r]
        if missing:
            raise SystemExit(f"FATAL: {tag}: {len(missing)} cohort prompts lack an O_base refusal2 row")
        rates[tag] = r
    hard = [i for i in units if not is_xstest(i)]
    xs = [i for i in units if is_xstest(i)]
    core = [i for i in hard if sum(rates[t][i] >= CORE_RATE for t in ALL_TAGS) >= CORE_MIN_MODELS]
    rest = [i for i in hard if i not in set(core) and rates["llama"][i] > 0]
    llama_extra = sorted(rest, key=lambda i: (-rates["llama"][i], i))[:LLAMA_EXTRA_MAX]
    if args.expect:
        exp = read_json(args.expect)
        same_core = sorted(exp["core_ids"]) == sorted(core)
        same_extra = sorted(exp["llama_extra_ids"]) == sorted(llama_extra)
        print(f"expect: core {len(exp['core_ids'])} vs {len(core)} same={same_core}; llama_extra {len(exp['llama_extra_ids'])} vs {len(llama_extra)} same={same_extra}")
        if not (same_core and same_extra):
            raise SystemExit("FATAL: the recomputed population differs from the expected ids")
    out_units = []
    for stratum, ids in (("core", core), ("llama_extra", llama_extra), ("xstest", xs)):
        for i in ids:
            u = units[i]
            out_units.append(
                {
                    "id": i,
                    "source": u["source"],
                    "category": u["category"],
                    "stratum": stratum,
                    "prompt": u["prompts"]["O"],
                    "prompt_sha256": u["prompt_sha256"]["O"],
                    "r_O_x106": {t: rates[t][i] for t in ALL_TAGS},
                    "n_models_refusing_at_half": int(sum(rates[t][i] >= CORE_RATE for t in ALL_TAGS)),
                    "x106": {
                        "unit": u["unit"],
                        "unit_indices": u["unit_indices"],
                        "unit_kind": u["unit_kind"],
                        "top_word": u["top_word"],
                        "pair_wins": u["pair_wins"],
                        "unit_in_lexicon": u["unit_in_lexicon"],
                        "proj_qwen": u["proj_qwen"],
                        "stratum_qwen": u["stratum"],
                        "S_sha256": u["prompt_sha256"]["S"],
                        "C_sha256": u["prompt_sha256"]["C"],
                        "substitute": u["substitute"],
                        "control": u["control"],
                        "hedge": u["hedge"],
                    },
                }
            )
    by_stratum = {s: [u for u in out_units if u["stratum"] == s] for s in STRATA}
    out = {
        "experiment": "x113",
        "stage": 0,
        "status": "frozen",
        "cohort": file_record(cohort_path()),
        "cohort_sha256": cohort["cohort_sha256"],
        "refusal_rows": {t: file_record(x106_refusal_path(t, "O_base")) for t in ALL_TAGS},
        "label_construct": "x64 refusal2 (an explicit refusal statement is present), x106's O_base rows, ten samples per prompt",
        "rules": {
            "core": f"Hard-1K cohort prompts with x106 O_base refusal2 >= {CORE_RATE} in >= {CORE_MIN_MODELS} of the 8 models",
            "llama_extra": f"the remaining Hard-1K cohort prompts with Llama-3.2-3B O_base refusal2 > 0, in descending order of that rate (ties by id), at most {LLAMA_EXTRA_MAX}",
            "xstest": "x106's 100 safe XSTest prompts of the homonyms, figurative language, safe contexts and safe targets categories",
        },
        "n_units": len(out_units),
        "n_by_stratum": {s: len(v) for s, v in by_stratum.items()},
        "n_hard1k_cohort": len(hard),
        "n_llama_extra_available": len(rest),
        "r_O_by_model": {s: {t: float(np.mean([u["r_O_x106"][t] for u in v])) for t in ALL_TAGS} for s, v in by_stratum.items() if v},
        "r_O_by_model_hard1k": {t: float(np.mean([u["r_O_x106"][t] for u in by_stratum["core"] + by_stratum["llama_extra"]])) for t in ALL_TAGS},
        "units": out_units,
    }
    out["units_sha256"] = units_sha256(out_units)
    atomic_json(PROMPTS, out)
    print(f"x113 prompts -> {PROMPTS}: {out['n_by_stratum']}; llama_extra available {len(rest)}")
    for s, v in out["r_O_by_model"].items():
        print(f"  {s:12s} " + " ".join(f"{t} {r:.3f}" for t, r in v.items()))


if __name__ == "__main__":
    main()
