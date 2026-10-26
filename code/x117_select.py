"""x117 prompt selection (local, tokenizer-free): x111's prompt rule continued onto 64 prompts that
none of the earlier experiments used, so the four Qwen3.5-4B vocabulary tables of x111 can be ranked
on prompts outside every table's prompt set and outside the set that chose the table.

Pools and rule are x111's, imported: `x111_select.load_pools()` (x106 Hard-1K O_base for benign,
x83 cohort O for harmful; each row already flagged when it is an x100 HP / BP prompt, one of x110's
192 or one of x109's 40) and `x111_select.topup_exclusions()` (the x85 crossed cohort's 96 prompts,
x111's 64, x110's 192 and every x112 prompt, by id and text sha256). Added here: x111's nine
harmful top-up prompts (`data/runs/x111/prompts_topup.json`).

Draw, with x111's seed-111 tie keys over the whole pool (so a prompt's key does not depend on which
other prompts are dropped) and nearest-0.5-first order:
- benign: the first 32 of the surviving prompts with historical `refusal2` in [0.3, 0.7];
- harmful: every surviving prompt in [0.3, 0.7], then, in the same order, prompts at 0.2 / 0.8
  until the stratum holds 32.
The historical rate selects, as in x111; no fresh O cell. Writes `data/runs/x117/prompts.json` in
x111's `prompts.json` schema (plus `rank_in_draw` and `band` per row, and a `selection` block
with the exclusion sources, the per-rate counts and the harmful relaxation) and copies x111's
`words.json` to `data/runs/x117/words.json`, which the x117 stages read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import x111_select  # noqa: E402
from x111_common import ROOT, SEED, WORDS as X111_WORDS, atomic_json, read_json, sha256_text  # noqa: E402
from x111_select import POPULATION_INDEX, POPULATIONS_ORDER, N_PER_POP, load_pools, topup_exclusions  # noqa: E402

RUN = ROOT / "data" / "runs" / "x117"
PROMPTS = RUN / "prompts.json"
WORDS = RUN / "words.json"
X111_PROMPTS_TOPUP = ROOT / "data" / "runs" / "x111" / "prompts_topup.json"
RELAXED = (0.2, 0.8)
BAND_PRIMARY = (0.3, 0.4, 0.5, 0.6, 0.7)
ALL_RATES = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def counts_of(rates) -> dict[str, int]:
    rates = list(rates)
    return {f"{r:.1f}": sum(1 for v in rates if abs(v - r) < 1e-9) for r in ALL_RATES}


def _sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exclusions() -> tuple[dict, dict]:
    """x111's top-up exclusion sources plus x111's nine harmful top-up prompts."""
    sources, prov = topup_exclusions()
    topup = read_json(X111_PROMPTS_TOPUP)["prompts"]
    sources["x111_prompts_topup"] = {"ids": {p["id"] for p in topup}, "sha": {p["text_sha256"] for p in topup}}
    prov["x111_prompts_topup"] = {"file": str(X111_PROMPTS_TOPUP.relative_to(ROOT)), "file_sha256": _sha_file(X111_PROMPTS_TOPUP), "n_prompts": len(topup)}
    return sources, prov


def tie_keys(pool: list[dict], population: str) -> np.ndarray:
    """x111's tie keys: one uniform per pool position, drawn over the whole pool."""
    return np.random.default_rng([SEED, POPULATION_INDEX[population]]).random(len(pool))


def draw(pool: list[dict], population: str, sources: dict, bands: tuple[tuple[float, ...], ...]) -> tuple[list[dict], dict]:
    """The rule on one pool: band by band, the surviving prompts whose rate is one of the band's
    rates, sorted by |rate - 0.5| with x111's tie keys; take prompts until the stratum holds
    N_PER_POP. A later band is used only for the places the earlier bands left."""
    tie = tie_keys(pool, population)

    def hits(p: dict) -> list[str]:
        return [name for name, s in sources.items() if p["id"] in s["ids"] or p["text_sha256"] in s["sha"]]

    def key(k: int) -> tuple[float, float]:
        return (round(abs(pool[k]["historical_refusal2"] - 0.5), 6), tie[k])

    band_summary, picked = [], []
    for rates in bands:
        in_band = [k for k, p in enumerate(pool) if any(abs(p["historical_refusal2"] - r) < 1e-9 for r in rates)]
        eligible, dropped = [], []
        for k in in_band:
            h = (["x100_or_x109_or_x110 (x111_select.load_pools)"] if pool[k]["excluded"] else []) + hits(pool[k])
            (dropped if h else eligible).append((k, h) if h else k)
        order = sorted(eligible, key=key)
        pick = order[: N_PER_POP - len(picked)]
        picked += pick
        band_summary.append(
            {
                "rates": list(rates),
                "n_in_band": len(in_band),
                "n_dropped": len(dropped),
                "n_eligible": len(eligible),
                "n_taken": len(pick),
                "eligible_rate_counts": counts_of(pool[k]["historical_refusal2"] for k in eligible),
                "taken_rate_counts": counts_of(pool[k]["historical_refusal2"] for k in pick),
                "dropped_by_source": {name: sum(1 for _, h in dropped if name in h) for name in ["x100_or_x109_or_x110 (x111_select.load_pools)", *sources]},
            }
        )
        if len(picked) == N_PER_POP:
            break
    rank = {k: i for i, k in enumerate(sorted(picked, key=key))}
    rows = []
    for k in picked:
        p = {kk: v for kk, v in pool[k].items() if kk != "excluded"}
        band = next(r for r in bands if any(abs(p["historical_refusal2"] - v) < 1e-9 for v in r))
        rows.append({**p, "population": population, "n_words": len(p["text"].split()), "rank_in_draw": rank[k], "band": list(band), "tie_key": float(tie[k])})
    summary = {"n_pool": len(pool), "n_chosen": len(rows), "bands": band_summary, "chosen_rate_counts": counts_of(r["historical_refusal2"] for r in rows)}
    return rows, summary


def build_prompts(bands_harmful: tuple[tuple[float, ...], ...]) -> None:
    benign_pool, harmful_pool, gen = load_pools()
    sources, prov = exclusions()
    frozen = read_json(x111_select.PROMPTS)
    frozen_ids = {q["id"] for q in frozen["prompts"]}
    for population, pool in (("benign", benign_pool), ("harmful", harmful_pool)):
        for p in pool:
            if p["id"] in frozen_ids and not p["excluded"] and not any(p["id"] in s["ids"] or p["text_sha256"] in s["sha"] for s in sources.values()):
                raise SystemExit(f"FATAL: x111 {population} prompt {p['id']} survives the x117 exclusions")
    rows, pool_summary = [], {}
    for population, pool, bands in (("benign", benign_pool, (BAND_PRIMARY,)), ("harmful", harmful_pool, bands_harmful)):
        got, summary = draw(pool, population, sources, bands)
        if len(got) != N_PER_POP:
            raise SystemExit(f"FATAL: {population} filled {len(got)}/{N_PER_POP}")
        pool_summary[population] = summary
        rows += got
    out = {
        "run": "x117",
        "rule": "x111's prompt rule continued onto prompts no earlier experiment used: rate = historical refusal2 over ten O samples (x106 Qwen3.5-4B O_base for benign, x83 O for harmful; x64 judge); dropped by id and text sha256: x100 HP / BP (= x110's 192), x109's 40, the x85 crossed cohort's 96, x111's 64 and its nine harmful top-up prompts, and every x112 prompt; sorted by |rate - 0.5|, ties by x111's numpy default_rng([111, population index]) uniform key over the whole pool; benign keeps rate in [0.3, 0.7] and takes the first 32; harmful takes every surviving prompt in [0.3, 0.7] and then, in the same order, prompts at 0.2 / 0.8 until the stratum holds 32. The historical rate selects; no fresh O cell.",
        "sources": frozen["sources"],
        "model_id": gen["model_id"],
        "model_snapshot": gen["model_source"],
        "design": {
            "words": str(WORDS.relative_to(ROOT)),
            "words_source": str(X111_WORDS.relative_to(ROOT)),
            "words_sha256_file": _sha_file(X111_WORDS),
            "tables": "data/runs/x111/tables.json + tables.npz (benign, BP, HP, mixed), unchanged",
            "seed_namespace": "x117",
        },
        "selection": {
            "seed": SEED,
            "tie_key_rng": "numpy default_rng([111, population index]).random(n_pool), indexed by pool position (x111's keys, unchanged)",
            "band_primary": list(BAND_PRIMARY),
            "band_relaxed_harmful": list(RELAXED),
            "relaxation": "the harmful stratum exhausts [0.3, 0.7] before 32 prompts are reached, so the remaining places go to prompts at 0.2 / 0.8 in the same |rate - 0.5| and tie-key order",
            "exclusion_sources": prov,
            "exclusion_sources_in_load_pools": "x100 HP / BP (x110's 192) and x109's 40, flagged by x111_select.load_pools",
            "x111_prompts_all_excluded": True,
            "pools": pool_summary,
        },
        "n_prompts": len(rows),
        "prompts_sha256": sha256_text("\n".join(f"{r['id']}\t{r['text_sha256']}" for r in rows)),
        "prompts": rows,
    }
    RUN.mkdir(parents=True, exist_ok=True)
    atomic_json(PROMPTS, out)
    WORDS.write_bytes(X111_WORDS.read_bytes())
    print(json.dumps(pool_summary, indent=1))
    for population in POPULATIONS_ORDER:
        sub = [r for r in rows if r["population"] == population]
        print(population, len(sub), "mean historical", round(float(np.mean([r["historical_refusal2"] for r in sub])), 4), "mean words", round(float(np.mean([r["n_words"] for r in sub])), 1))
    print("->", PROMPTS, out["prompts_sha256"][:12], "words ->", WORDS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts", action="store_true")
    args = parser.parse_args()
    if not args.prompts:
        raise SystemExit("--prompts")
    build_prompts((BAND_PRIMARY, RELAXED))


if __name__ == "__main__":
    main()
