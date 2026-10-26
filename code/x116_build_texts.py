"""x116 Stage 1 (local, no API): per model, the R_rand text of every x113 unit with an S_T text.

The candidates are x113's, read from data/runs/x113/<tag>/texts.json (the drafter is not called
again). The pool is the accepted candidates of the unit (x113's structural checks, the same hedge
or content span) minus the one x113 chose (the table-lowest, S_T), minus any candidate whose
spliced text is byte-identical to S_T, minus any candidate carrying one of the 48 x85 words (so
that x113's x85 exclusion remains the complete unit filter). One candidate is drawn uniformly
from the pool with numpy seeded by sha256("x116|116|<tag>|<unit id>"); the text is x113's
`splice` of the candidate over the S_T span (hedge-separator fix included). Every candidate's
table shift is x113's record; the shift gap is shift(R_rand) - shift(S_T) where both exist.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x113_build_texts import splice  # noqa: E402
from x113_common import load_texts as load_x113_texts  # noqa: E402
from x113_exclusions import X85_WORDS  # noqa: E402
from x116_common import ALL_TAGS, SEED, atomic_json, draw_rng_seed, file_record, paths, sha256_text, texts_sha256, x113_paths  # noqa: E402
from x113_common import normalize_word  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="*", default=list(ALL_TAGS))
    args = parser.parse_args()
    for tag in args.tags:
        src = load_x113_texts(tag)
        units = []
        n_pool = []
        reasons = {"no_S_T": 0, "pool_empty": 0}
        dropped = {"identical_to_S_T": 0, "x85_word": 0}
        for u in src["units"]:
            rec = {"id": u["id"], "stratum": u["stratum"], "prompt_sha256": u["prompt_sha256"], "nominee": (u.get("nominee") or {}).get("norm"), "unit_kind": u.get("unit_kind"), "text": None, "sha256": None, "substitute": None, "S_T_substitute": None, "pool": [], "n_accepted": 0, "no_arm_reason": None}
            st = u["texts"]["S_T"]
            if st is None:
                rec["no_arm_reason"] = "no S_T"
                reasons["no_S_T"] += 1
                units.append(rec)
                continue
            start, end = u["substitute"]["span"]
            accepted = [c for c in u["candidates"] if c["accepted"]]
            chosen = [c for c in accepted if c.get("chosen")]
            if len(chosen) != 1 or splice(u["prompt"], start, end, chosen[0]["text"]) != st:
                raise SystemExit(f"FATAL: {tag}/{u['id']}: x113's S_T is not the splice of its chosen candidate")
            chosen = chosen[0]
            pool = []
            for c in accepted:
                if c is chosen:
                    continue
                text = splice(u["prompt"], start, end, c["text"])
                if text == st:
                    dropped["identical_to_S_T"] += 1
                    continue
                if {normalize_word(w) for w in c["text"].split()} & X85_WORDS:
                    dropped["x85_word"] += 1
                    continue
                pool.append({"text": c["text"], "shift": c["shift"], "shift_word": c["shift_word"], "assembled": text})
            rec["n_accepted"] = len(accepted)
            rec["S_T_substitute"] = {"text": chosen["text"], "shift": chosen["shift"], "shift_word": chosen["shift_word"], "span": [start, end], "kind": u["substitute"]["kind"]}
            rec["pool"] = [{k: p[k] for k in ("text", "shift", "shift_word")} for p in pool]
            n_pool.append(len(pool))
            if not pool:
                rec["no_arm_reason"] = "only the table-lowest candidate is admissible"
                reasons["pool_empty"] += 1
                units.append(rec)
                continue
            rng_seed = draw_rng_seed(tag, u["id"])
            k = int(np.random.default_rng(rng_seed).integers(len(pool)))
            pick = pool[k]
            rec.update(
                {
                    "text": pick["assembled"],
                    "sha256": sha256_text(pick["assembled"]),
                    "draw": {"rng_seed": rng_seed, "index_in_pool": k, "pool_size": len(pool)},
                    "substitute": {"text": pick["text"], "shift": pick["shift"], "shift_word": pick["shift_word"], "span": [start, end]},
                    "shift_gap": (pick["shift"] - chosen["shift"]) if (pick["shift"] is not None and chosen["shift"] is not None) else None,
                    "pick_rank_in_accepted_by_shift": (1 + sum(1 for c in accepted if c["shift"] is not None and pick["shift"] is not None and c["shift"] < pick["shift"])) if pick["shift"] is not None else None,
                }
            )
            units.append(rec)
        out = {
            "experiment": "x116",
            "stage": 1,
            "status": "frozen",
            "model_tag": tag,
            "x113_texts": file_record(x113_paths(tag)["texts"]),
            "x113_texts_sha256": src["texts_sha256"],
            "rule": "uniform draw from x113's accepted candidates of the unit minus the chosen (table-lowest) one, minus candidates whose text equals S_T, minus candidates carrying an x85 word; x113 splice over the S_T span",
            "draw_seed": f"numpy default_rng(int(sha256('x116|{SEED}|<tag>|<unit id>')[:8 bytes]))",
            "n_units": len(units),
            "n_R_rand": sum(1 for u in units if u["text"] is not None),
            "n_without": reasons,
            "n_pool_dropped": dropped,
            "pool_size_mean": float(np.mean(n_pool)) if n_pool else None,
            "units": units,
        }
        out["texts_sha256"] = texts_sha256(units)
        atomic_json(paths(tag)["texts"], out)
        gaps = [u["shift_gap"] for u in units if u.get("shift_gap") is not None]
        print(f"x116 texts {tag}: {out['n_R_rand']} R_rand of {len(units)}; without {reasons}; dropped {dropped}; pool mean {out['pool_size_mean']:.2f}; shift gap mean {np.mean(gaps):+.3f} (n {len(gaps)})")


if __name__ == "__main__":
    main()
