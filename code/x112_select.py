"""x112 selection (local, tokenizer-free).

`--scan-inputs`: the eight scan prompts (x100 BP by key, x111 representative_prompts.json table BP
top 8) and the candidate list (x109 scan.json, identical order and sha256) -> scan_prompts.json,
candidates.json.
`--pool --tag T`: the two held-out candidate pools of model T, ordered by the distance of the
historical ten-sample refusal2 from 0.5 inside the band -> <tag>/pool_candidates.json.
`--request --tag T --n N`: the first N candidates of each pool get a fresh O cell ->
<tag>/pool_request.json (only ever extended).
`--prompts --tag T`: from the judged fresh O cells, per pool the 16 whose fresh rate lies in the
band nearest 0.5 -> <tag>/prompts.json.
`--words --tag T`: 40 words, seed 112, stratified on the model's own table x the Qwen3.5-4B BP
table (x111's strata and drop rules) -> <tag>/words.json.
`--compare`: between-model table agreement over the eligible words -> table_comparison.json.
No held-out prompt is scanned before its model's words and prompts are fixed.

Top-up of the harmful stratum (after the run). The x85 crossed cohort, on which every model's
direction was fitted, was not excluded by `--pool`; x85_exclusions.json drops its 13 harmful
held-out prompts from the analysis. `--topup-request --tag T` lists, for a model short of 16
harmful prompts outside that cohort, the candidates of pool_candidates.json that are neither
chosen nor x85 prompts, inside the pool's band: those whose pool-stage O cell was judged in band
are reused first; for the shortfall left, the nearest-to-0.5 unjudged candidates get a fresh O
cell (over-draw max(2 x shortfall, 4), bounded by the candidates left) -> <tag>/topup_request.json.
`--topup-prompts --tag T`: from the reused and the newly judged O cells, the in-band ones nearest
0.5 (reused first, then the new ones; ties by candidate rank) fill the stratum back to 16 ->
<tag>/prompts_topup.json, the same row structure as prompts.json plus provenance. prompts.json
itself is frozen and never rewritten.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x109_scan_select import PROPER_NOUNS_JUDGED, share_stem, zipf_of  # noqa: E402
from x111_select import BIN_NAMES, N_CONCORDANT, N_DISCORDANT, PROPER_NOUNS_X111, bin_of  # noqa: E402
from x112_common import (  # noqa: E402
    CANDIDATES,
    COMPARISON,
    LABELS,
    N_HALF,
    N_SAMPLES,
    ROOT,
    RUN,
    SCAN_KEYS,
    SCAN_PROMPTS,
    SEED,
    TAGS,
    X100_COHORT,
    X106_RUN,
    X109_SCAN_JSON,
    X109_WORDFREQ,
    X111_REPRESENTATIVE,
    X111_TABLES_JSON,
    X111_TABLES_NPZ,
    X85_EXCLUSIONS,
    atomic_json,
    paths,
    pool_cells,
    read_json,
    sha256_text,
    x85_prompt_ids,
)
from x83_common import lexicon_words  # noqa: E402

N_PER_POP = 16
N_FIRST = 24
TOPUP_OVERDRAW_FACTOR = 2
TOPUP_MIN_OVERDRAW = 4
BAND = (0.3, 0.7)
BAND_WIDE = (0.2, 0.8)
POP_INDEX = {"benign": 0, "harmful": 1}
# Discordant quadrants: (name, rule) on the own-table (o) and Qwen3.5-4B BP (q) percentiles.
QUADRANTS = (
    ("Qwen_low_own_not", "Qwen BP <= p5 and own >= p25", lambda o, q: (q <= 5) & (o >= 25)),
    ("own_low_Qwen_not", "own <= p5 and Qwen BP >= p25", lambda o, q: (o <= 5) & (q >= 25)),
    ("Qwen_high_own_not", "Qwen BP >= p95 and own <= p75", lambda o, q: (q >= 95) & (o <= 75)),
    ("own_high_Qwen_not", "own >= p95 and Qwen BP <= p75", lambda o, q: (o >= 95) & (q <= 75)),
)
# Proper nouns met on the seeded walks (manual judgment on the drawn words), on top of x109's / x111's.
PROPER_NOUNS_X112: set[str] = {
    "esperanto", "swedish", "thani", "sainte", "palme", "bahamas", "targaryen", "newsweek", "lovecraft", "freud", "sauron",
    "germanic", "everglades", "adnan", "hezbollah", "ewen", "kimi", "wang", "chapo",
    "verona", "yuki", "nabokov", "isaiah", "uzbek", "pag",
    "sophocles", "zhou", "aussie", "hendrik", "goebbels", "mdc",
    "wittgenstein", "aquinas", "virginians", "kamara", "nxt", "streep", "aqsa", "dickens", "kant", "lego", "kilda",
    "socratic", "stevenson", "merseyside", "wentworth", "florentine", "brando", "felipe", "balinese", "eisenhower",
    "deutsch", "prussian", "geeta", "albanian", "ont", "norbert", "kayla", "malagasy",
    "tamil", "pamela", "afrika", "kent", "cortana", "dachau", "chopra", "mildred", "ronald", "yukon", "mogadishu", "darfur",
    "jelena", "mukherjee", "lopez", "xiang", "lyft", "twain", "crossfit", "hooters", "mcveigh", "namibia", "englishmen", "missoula", "devon",
}
# Pools whose every [0.3, 0.7] candidate was run and judged and that still held fewer than 16
# fresh rates in band: widened to [0.2, 0.8] (the band order keeps the [0.3, 0.7] candidates first).
WIDENED_AFTER_EXHAUSTION = {("llama", "benign")}
X101_RUN = ROOT / "data" / "runs" / "x101"
X108_RUN = ROOT / "data" / "runs" / "x108"
X109_PROMPTS = ROOT / "data" / "runs" / "x109" / "prompts.json"


def scan_inputs() -> None:
    rep = read_json(X111_REPRESENTATIVE)["tables"]["BP"]["top8"]
    if tuple(r["id"] for r in rep) != SCAN_KEYS:
        raise SystemExit("FATAL: x111 BP top 8 differs from the plan's eight")
    bp = {p["key"]: p for p in read_json(X100_COHORT)["populations"]["BP"]["prompts"]}
    rows = []
    for k, r in zip(SCAN_KEYS, rep):
        p = bp[k]
        if sha256_text(p["prompt"]) != p["prompt_sha256"]:
            raise SystemExit(f"FATAL: x100 BP {k} fails its sha256")
        rows.append({"id": k, "x100_id": p["id"], "text": p["prompt"], "text_sha256": p["prompt_sha256"], "x111_spearman_with_BP_table": r["spearman_with_table"]})
    atomic_json(SCAN_PROMPTS, {"run": "x112", "rule": "x111 representative_prompts.json, table BP, top 8 by the Spearman of the prompt's own per-word shift with the BP table on Qwen3.5-4B; texts from x100_cohort.json populations.BP by key", "prompts_sha256": sha256_text("\n".join(f"{r['id']}\t{r['text_sha256']}" for r in rows)), "prompts": rows})
    x109 = read_json(X109_SCAN_JSON)
    words = x109["candidates"]["words"]
    if sha256_text("\n".join(words)) != x109["candidates"]["words_sha256"] or x109["candidates"]["words_sha256"] != read_json(X111_TABLES_JSON)["candidates"]["words_sha256"]:
        raise SystemExit("FATAL: candidate list differs from x111's")
    atomic_json(CANDIDATES, {"run": "x112", "source": str(X109_SCAN_JSON.relative_to(ROOT)), "n": len(words), "words_sha256": x109["candidates"]["words_sha256"], "words": words})
    print(len(rows), "scan prompts;", len(words), "candidates")


# ---- held-out pools -------------------------------------------------------------------------


def _rows_by_sha(path: Path) -> dict:
    return {r["prompt_sha256"]: r for r in read_json(path)["rows"]}


def _refusal_rows(path: Path) -> list[dict]:
    out, seen = [], set()
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("failed"):
            continue
        if rec["prompt_sha256"] in seen:
            continue  # x101 qwen9b carries one duplicate line
        seen.add(rec["prompt_sha256"])
        out.append(rec)
    return out


def exclusions() -> tuple[set[str], set]:
    cohort = read_json(X100_COHORT)
    ex_sha, ex_ids = set(), set()
    for pop in ("HP", "BP"):
        for p in cohort["populations"][pop]["prompts"]:
            ex_sha.add(p["prompt_sha256"])
            ex_ids.add(str(p["id"]))
    for p in read_json(X109_PROMPTS)["prompts"]:
        ex_sha.add(p["text_sha256"])
    for p in read_json(SCAN_PROMPTS)["prompts"]:
        ex_sha.add(p["text_sha256"])
    return ex_sha, ex_ids


def build_pool(tag: str) -> None:
    ex_sha, ex_ids = exclusions()
    pools: dict[str, list[dict]] = {"benign": [], "harmful": []}
    # Benign: x106 Hard-1K O_base (XSTest rows dropped).
    g = _rows_by_sha(X106_RUN / tag / "x106_generations_O_base.json")
    for rec in _refusal_rows(X106_RUN / tag / "x106_refusal_O_base.jsonl"):
        row = g[rec["prompt_sha256"]]
        if not str(row["id"]).startswith("orbench_hard:"):
            continue
        if sha256_text(row["prompt"]) != rec["prompt_sha256"]:
            raise SystemExit(f"FATAL: x106 {tag} {row['id']} fails its sha256")
        pools["benign"].append({"id": str(row["id"]), "source": "x106 O_base (OR-Bench Hard-1K)", "text": row["prompt"], "text_sha256": rec["prompt_sha256"], "historical_refusal2": float(np.mean(rec["refusal"])), "historical_n": len(rec["refusal"]), "historical_rendered_chat_sha256": row["rendered_chat_sha256"], "excluded": str(row["id"]) in ex_ids or rec["prompt_sha256"] in ex_sha})
    # Harmful: x101 O (x83 cohort) and x108 O_base (OR-Bench Toxic); merged by text sha256.
    merged: dict[str, dict] = {}
    for src, gen_path, ref_path, prefix in (
        ("x101 O (x83 cohort)", X101_RUN / tag / "x101_generations_O.json", X101_RUN / tag / "x101_refusal_O.jsonl", "x83:"),
        ("x108 O_base (OR-Bench Toxic)", X108_RUN / tag / "x108_generations_O_base.json", X108_RUN / tag / "x108_refusal_O_base.jsonl", ""),
    ):
        g = _rows_by_sha(gen_path)
        for rec in _refusal_rows(ref_path):
            row = g[rec["prompt_sha256"]]
            if sha256_text(row["prompt"]) != rec["prompt_sha256"]:
                raise SystemExit(f"FATAL: {src} {tag} {row['id']} fails its sha256")
            sha = rec["prompt_sha256"]
            if sha in merged:
                m = merged[sha]
                m["labels"] += list(rec["refusal"])
                m["source"] += f" + {src}"
                continue
            merged[sha] = {"id": f"{prefix}{row['id']}", "source": src, "text": row["prompt"], "text_sha256": sha, "labels": list(rec["refusal"]), "historical_rendered_chat_sha256": row["rendered_chat_sha256"], "excluded": str(row["id"]) in ex_ids or sha in ex_sha}
    for m in merged.values():
        labels = m.pop("labels")
        pools["harmful"].append({**m, "historical_refusal2": float(np.mean(labels)), "historical_n": len(labels)})
    out = {"run": "x112", "model_tag": tag, "rule": "rate = historical refusal2 (x64 judge, ten O samples per source; a prompt in both harmful sources pools its labels); drop x100 HP / BP, x109's 40 and the eight scan prompts by id and text sha256; band [0.3, 0.7], or [0.2, 0.8] for a pool with fewer than 16 eligible prompts in [0.3, 0.7]; order by |rate - 0.5|, ties by a numpy default_rng([112, model index, population index]) uniform key", "pools": {}}
    for pop, pool in pools.items():
        rng = np.random.default_rng([SEED, TAGS.index(tag), POP_INDEX[pop]])
        tie = rng.random(len(pool))
        n_band = sum(1 for p in pool if not p["excluded"] and BAND[0] - 1e-9 <= p["historical_refusal2"] <= BAND[1] + 1e-9)
        band = BAND if n_band >= N_PER_POP and (tag, pop) not in WIDENED_AFTER_EXHAUSTION else BAND_WIDE
        eligible = [k for k, p in enumerate(pool) if not p["excluded"] and band[0] - 1e-9 <= p["historical_refusal2"] <= band[1] + 1e-9]
        order = sorted(eligible, key=lambda k: (round(abs(pool[k]["historical_refusal2"] - 0.5), 6), tie[k]))
        out["pools"][pop] = {
            "n_pool": len(pool),
            "n_excluded": sum(p["excluded"] for p in pool),
            "n_in_0.3_0.7_eligible": n_band,
            "band": list(band),
            "widened": band == BAND_WIDE,
            "widened_after_exhaustion": (tag, pop) in WIDENED_AFTER_EXHAUSTION,
            "n_candidates": len(order),
            "candidates": [{kk: v for kk, v in pool[k].items() if kk != "excluded"} | {"population": pop, "rank": r} for r, k in enumerate(order)],
        }
        print(f"{tag} {pop}: pool {len(pool)}, excluded {out['pools'][pop]['n_excluded']}, in [0.3,0.7] {n_band}, band {band}, candidates {len(order)}")
    atomic_json(paths(tag)["pool"], out)


def request(tag: str, n: int, pops: tuple[str, ...] = ("benign", "harmful")) -> None:
    pool = read_json(paths(tag)["pool"])
    p = paths(tag)["pool_request"]
    prior = read_json(p) if p.exists() else {"prompts": [], "n_per_pool": {}}
    have = {r["id"] for r in prior["prompts"]}
    rows = list(prior["prompts"])
    n_per = dict(prior.get("n_per_pool", {}))
    for pop in pops:
        cands = pool["pools"][pop]["candidates"]
        k = min(n, len(cands))
        for c in cands[:k]:
            if c["id"] not in have:
                rows.append({"id": c["id"], "population": pop, "text": c["text"], "text_sha256": c["text_sha256"], "historical_refusal2": c["historical_refusal2"], "historical_source": c["source"], "historical_rendered_chat_sha256": c["historical_rendered_chat_sha256"], "rank": c["rank"]})
                have.add(c["id"])
        n_per[pop] = max(k, n_per.get(pop, 0))
    atomic_json(p, {"run": "x112", "model_tag": tag, "rule": "the first n candidates of each pool in pool_candidates.json order; only ever extended", "n_per_pool": n_per, "prompts": rows})
    print(tag, n_per, len(rows), "O cells requested")


def load_unit_labels(tag: str) -> dict:
    out = {}
    path = paths(tag)["judge"] / "refusal2.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if not rec.get("failed"):
                out[rec["cell_id"]] = rec
    return out


def build_prompts(tag: str) -> None:
    pool = read_json(paths(tag)["pool"])
    req = read_json(paths(tag)["pool_request"])
    units = load_unit_labels(tag)
    rows, summary = [], {}
    for pop in ("benign", "harmful"):
        band = pool["pools"][pop]["band"]
        judged = []
        for c in [r for r in req["prompts"] if r["population"] == pop]:
            parts = [units.get(f"{c['id']}|O#h{h}") for h in range(N_SAMPLES // N_HALF)]
            if any(x is None for x in parts):
                continue
            labels = sum((list(x["refusal"]) for x in parts), [])
            judged.append({**c, "fresh_refusal2": float(np.mean(labels))})
        inb = [c for c in judged if band[0] - 1e-9 <= c["fresh_refusal2"] <= band[1] + 1e-9]
        inb.sort(key=lambda c: (round(abs(c["fresh_refusal2"] - 0.5), 6), c["rank"]))
        pick = inb[:N_PER_POP]
        summary[pop] = {"band": band, "widened": pool["pools"][pop]["widened"], "widened_after_exhaustion": pool["pools"][pop].get("widened_after_exhaustion", False), "n_requested": sum(1 for r in req["prompts"] if r["population"] == pop), "n_judged": len(judged), "n_fresh_in_band": len(inb), "n_chosen": len(pick), "short": len(pick) < N_PER_POP, "fresh_rates_chosen": sorted(c["fresh_refusal2"] for c in pick), "fresh_vs_historical_spearman_judged": float(np.corrcoef(np.argsort(np.argsort([c["fresh_refusal2"] for c in judged])), np.argsort(np.argsort([c["historical_refusal2"] for c in judged])))[0, 1]) if len(judged) > 2 else None}
        print(tag, pop, {k: v for k, v in summary[pop].items() if k != "fresh_rates_chosen"})
        for c in pick:
            rows.append({**c, "n_words": len(c["text"].split())})
    if any(s["short"] for s in summary.values()):
        print(f"{tag}: SHORT -> extend the request")
        return
    out = {"run": "x112", "model_tag": tag, "rule": "per pool, the requested candidates' fresh O cells (twenty samples, refusal2); keep the fresh rate inside the pool's band, sort by |fresh - 0.5|, ties by candidate rank; first 16 per pool. These O cells are the grid's O cells.", "pools": summary, "n_prompts": len(rows), "prompts_sha256": sha256_text("\n".join(f"{r['id']}\t{r['text_sha256']}" for r in rows)), "prompts": rows}
    atomic_json(paths(tag)["prompts"], out)
    print(tag, "->", paths(tag)["prompts"])


# ---- top-up of the harmful stratum outside the x85 cohort -------------------------------------


def _fresh_rate(units: dict, pid: str) -> float | None:
    parts = [units.get(f"{pid}|O#h{h}") for h in range(N_SAMPLES // N_HALF)]
    if any(x is None for x in parts):
        return None
    return float(np.mean(sum((list(x["refusal"]) for x in parts), [])))


def _in_band(rate: float, band) -> bool:
    return band[0] - 1e-9 <= rate <= band[1] + 1e-9


def topup_request(tag: str, pop: str = "harmful") -> None:
    pool = read_json(paths(tag)["pool"])["pools"][pop]
    band = pool["band"]
    chosen = {p["id"] for p in read_json(paths(tag)["prompts"])["prompts"]}
    x85 = x85_prompt_ids()
    retained = [p for p in read_json(paths(tag)["prompts"])["prompts"] if p["population"] == pop and p["id"] not in x85]
    need = N_PER_POP - len(retained)
    req = {r["id"]: r for r in read_json(paths(tag)["pool_request"])["prompts"] if r["population"] == pop}
    units = load_unit_labels(tag)
    spares = [c for c in pool["candidates"] if c["id"] not in chosen and c["id"] not in x85 and _in_band(c["historical_refusal2"], band)]
    reused, judged_out, unjudged = [], [], []
    for c in spares:
        fresh = _fresh_rate(units, c["id"]) if c["id"] in req else None
        if fresh is None:
            unjudged.append(c)
        elif _in_band(fresh, band):
            reused.append({**c, "fresh_refusal2": fresh})
        else:
            judged_out.append({"id": c["id"], "rank": c["rank"], "fresh_refusal2": fresh})
    reused.sort(key=lambda c: (round(abs(c["fresh_refusal2"] - 0.5), 6), c["rank"]))
    shortfall = max(need - len(reused), 0)
    overdraw = min(len(unjudged), max(TOPUP_OVERDRAW_FACTOR * shortfall, TOPUP_MIN_OVERDRAW)) if shortfall > 0 else 0
    rows = [{"id": c["id"], "population": pop, "text": c["text"], "text_sha256": c["text_sha256"], "historical_refusal2": c["historical_refusal2"], "historical_source": c["source"], "historical_rendered_chat_sha256": c["historical_rendered_chat_sha256"], "rank": c["rank"]} for c in unjudged[:overdraw]]
    out = {
        "run": "x112",
        "stage": "topup",
        "model_tag": tag,
        "population": pop,
        "rule": "candidates of pool_candidates.json (ordered by |historical rate - 0.5|) that are neither chosen in prompts.json nor x85 crossed-cohort prompts, historical rate inside the pool's band; a candidate whose pool-stage O cell (generations_pool.json, refusal2) is in band is reused first; for the shortfall, the first unjudged candidates get a fresh O cell, over-draw max(2 x shortfall, 4) bounded by the candidates left",
        "x85_exclusions": {"file": str(X85_EXCLUSIONS.relative_to(ROOT)), "cohort_sha256": read_json(X85_EXCLUSIONS)["x85_cohort"]["cohort_sha256"]},
        "band": band,
        "n_chosen_original": sum(1 for p in read_json(paths(tag)["prompts"])["prompts"] if p["population"] == pop),
        "n_retained_after_x85": len(retained),
        "n_needed": need,
        "n_spares_in_band": len(spares),
        "reused_from_pool_stage": [{"id": c["id"], "rank": c["rank"], "historical_refusal2": c["historical_refusal2"], "fresh_refusal2": c["fresh_refusal2"]} for c in reused],
        "pool_stage_judged_out_of_band": judged_out,
        "shortfall_after_reuse": shortfall,
        "n_unjudged_spares": len(unjudged),
        "overdraw": overdraw,
        "prompts": rows,
    }
    atomic_json(paths(tag)["topup_request"], out)
    print(f"{tag} {pop}: retained {len(retained)}, need {need}, spares {len(spares)}, reused {len(reused)} ({[round(c['fresh_refusal2'], 2) for c in reused]}), judged out of band {len(judged_out)}, shortfall {shortfall}, over-draw {overdraw} of {len(unjudged)} unjudged -> {paths(tag)['topup_request']}")


def topup_prompts(tag: str) -> None:
    req = read_json(paths(tag)["topup_request"])
    pop, band, need = req["population"], req["band"], req["n_needed"]
    cands = {c["id"]: c for c in read_json(paths(tag)["pool"])["pools"][pop]["candidates"]}
    units = load_unit_labels(tag)
    ranked = []
    for c in req["reused_from_pool_stage"]:
        fresh = _fresh_rate(units, c["id"])
        if fresh is None or abs(fresh - c["fresh_refusal2"]) > 1e-9:
            raise SystemExit(f"FATAL: {tag} reused cell {c['id']} no longer judges to {c['fresh_refusal2']}")
        ranked.append((0, round(abs(fresh - 0.5), 6), c["rank"], c["id"], fresh, "pool stage (generations_pool.json)"))
    n_judged_new = 0
    for r in req["prompts"]:
        fresh = _fresh_rate(units, r["id"])
        if fresh is None:
            continue
        n_judged_new += 1
        if _in_band(fresh, band):
            ranked.append((1, round(abs(fresh - 0.5), 6), r["rank"], r["id"], fresh, "top-up pool (generations_pool_topup.json)"))
    ranked.sort()
    pick = ranked[:need]
    rows = []
    for _, _, rank, pid, fresh, src in pick:
        c = cands[pid]
        rows.append({"id": pid, "population": pop, "text": c["text"], "text_sha256": c["text_sha256"], "historical_refusal2": c["historical_refusal2"], "historical_source": c["source"], "historical_rendered_chat_sha256": c["historical_rendered_chat_sha256"], "rank": rank, "fresh_refusal2": fresh, "n_words": len(c["text"].split()), "topup_source": src})
    summary = {"band": band, "n_requested": len(req["prompts"]), "n_judged": n_judged_new, "n_fresh_in_band": sum(1 for r in ranked if r[0] == 1), "n_reused_in_band": sum(1 for r in ranked if r[0] == 0), "n_chosen": len(rows), "short": len(rows) < need, "fresh_rates_chosen": sorted(r["fresh_refusal2"] for r in rows), "fresh_rates_new_out_of_band": sorted(_fresh_rate(units, r["id"]) for r in req["prompts"] if _fresh_rate(units, r["id"]) is not None and not _in_band(_fresh_rate(units, r["id"]), band))}
    print(tag, pop, {k: v for k, v in summary.items()})
    if summary["short"]:
        print(f"{tag}: SHORT ({len(rows)} of {need}) -> extend topup_request.json or report the shortfall")
    out = {
        "run": "x112",
        "stage": "topup",
        "model_tag": tag,
        "rule": "prompts added to the harmful stratum after the x85 exclusion: in-band fresh O cells (twenty samples, refusal2), the pool-stage ones first, then the top-up pool ones, each sorted by |fresh - 0.5| with ties by candidate rank; as many as the stratum lacks. Their O cell is the one that selected them, as in prompts.json.",
        "topup": {"x85_exclusions": req["x85_exclusions"], "n_chosen_original": req["n_chosen_original"], "n_retained_after_x85": req["n_retained_after_x85"], "n_needed": need, "overdraw": req["overdraw"], "reused_from_pool_stage": req["reused_from_pool_stage"], "pool_stage_judged_out_of_band": req["pool_stage_judged_out_of_band"], "topup_pool_ids": [r["id"] for r in req["prompts"]]},
        "pools": {pop: summary},
        "n_prompts": len(rows),
        "prompts_sha256": sha256_text("\n".join(f"{r['id']}\t{r['text_sha256']}" for r in rows)),
        "prompts": rows,
    }
    atomic_json(paths(tag)["prompts_topup"], out)
    print(tag, "->", paths(tag)["prompts_topup"], [(r["id"], r["fresh_refusal2"], r["topup_source"][:10]) for r in rows])


# ---- words ---------------------------------------------------------------------------------


def build_words(tag: str) -> None:
    tz = np.load(paths(tag)["table_npz"])
    tmeta = read_json(paths(tag)["table_json"])
    q = np.load(X111_TABLES_NPZ)
    words = read_json(CANDIDATES)["words"]
    if sha256_text("\n".join(words)) != tmeta["candidates"]["words_sha256"]:
        raise SystemExit("FATAL: candidate list differs from the table")
    eligible = q["eligible"]
    if not np.array_equal(eligible, tz["eligible"]):
        raise SystemExit("FATAL: eligibility differs from x111")
    po, pq = tz["pct_own"], q["pct_BP"]
    bo = np.where(eligible, bin_of(np.nan_to_num(po)), -1)
    bq = np.where(eligible, bin_of(np.nan_to_num(pq)), -1)
    lexset = {w for w, _ in lexicon_words()}
    wf_rank = {w: i + 1 for i, w in enumerate(read_json(X109_WORDFREQ)["words"])}
    proper = PROPER_NOUNS_JUDGED | PROPER_NOUNS_X111 | PROPER_NOUNS_X112
    strata = []
    for k, name in enumerate(BIN_NAMES):
        strata.append({"stratum": f"concordant {name}", "kind": "concordant", "bin": k, "bin_name": name, "rule": f"own and Qwen BP percentiles both in {name}", "mask": eligible & (bo == k) & (bq == k), "n": N_CONCORDANT})
    for name, rule, fn in QUADRANTS:
        strata.append({"stratum": f"discordant {name}", "kind": "discordant", "bin": None, "bin_name": None, "quadrant": name, "rule": rule, "mask": eligible & fn(np.nan_to_num(po), np.nan_to_num(pq)), "n": N_DISCORDANT})

    def row(i: int) -> dict:
        w = words[i]
        return {"word": w, "index": int(i), "own_sd": float(tz["own"][i]), "qwen_BP_sd": float(q["BP"][i]), "qwen_benign_sd": float(q["benign"][i]), "pct_own": float(po[i]), "pct_qwen_BP": float(pq[i]), "pct_qwen_benign": float(q["pct_benign"][i]), "bin_own": BIN_NAMES[bo[i]], "bin_qwen_BP": BIN_NAMES[bq[i]], "n_tokens": int(tz["n_tokens"][i]), "wordfreq_rank": wf_rank.get(w), "zipf": zipf_of(w), "in_x83_lexicon": w in lexset}

    chosen, draw_log = [], []
    for si, s in enumerate(strata):
        idx = np.where(s["mask"])[0]
        rng = np.random.default_rng([SEED, si])
        order = idx[rng.permutation(len(idx))]
        got = 0
        log = {"stratum": s["stratum"], "rule": s["rule"], "n_eligible_in_stratum": int(len(idx)), "n_wanted": s["n"], "walk": []}
        for i in order:
            if got == s["n"]:
                break
            w = words[i]
            if w in proper:
                log["walk"].append({"word": w, "action": "dropped: proper noun (manual judgment)"})
                continue
            hit = next((c["word"] for c in chosen if share_stem(c["word"], w)), None)
            if hit:
                log["walk"].append({"word": w, "action": f"dropped: shares a stem with chosen '{hit}'"})
                continue
            log["walk"].append({"word": w, "action": "chosen"})
            chosen.append({**row(i), "stratum": s["stratum"], "kind": s["kind"], "bin": s["bin"], "bin_name": s["bin_name"], "quadrant": s.get("quadrant")})
            got += 1
        log["n_chosen"] = got
        draw_log.append(log)
        if got < s["n"]:
            print(f"WARNING: {tag} stratum {s['stratum']} filled {got}/{s['n']}")
    out = {
        "run": "x112",
        "model_tag": tag,
        "tables": {"own": str(paths(tag)["table_npz"].relative_to(ROOT)), "qwen_BP": str(X111_TABLES_NPZ.relative_to(ROOT))},
        "seed": SEED,
        "bins": list(BIN_NAMES),
        "selection_rule": "x111's: per stratum, the eligible words in a numpy default_rng([112, stratum index]) permutation; keep the first n that are not judged proper nouns and share no stem (x109_scan_select.share_stem) with any word already chosen; strata in the listed order (8 concordant bins, then 4 discordant quadrants); own table = this model's eight-prompt table, Qwen table = x111's BP table",
        "proper_nouns_judged": sorted(proper),
        "strata": [{k: v for k, v in s.items() if k != "mask"} | {"n_eligible": int(s["mask"].sum())} for s in strata],
        "chosen": chosen,
        "draw_log": draw_log,
        "chosen_sha256": sha256_text("\n".join(c["word"] for c in chosen)),
    }
    atomic_json(paths(tag)["words"], out)
    for s in out["strata"]:
        print(f"{tag} {s['stratum']:36s} eligible {s['n_eligible']}")
    for log in draw_log:
        print(log["stratum"], [x["word"] + ("" if x["action"] == "chosen" else " [" + x["action"] + "]") for x in log["walk"]])


# ---- between-model comparison -----------------------------------------------------------------


def compare() -> None:
    q = np.load(X111_TABLES_NPZ)
    elig = q["eligible"]
    words = np.asarray(read_json(CANDIDATES)["words"])

    def rank(v):
        return np.argsort(np.argsort(v)).astype(float)

    tables = {"qwen4b_BP": q["BP"].astype(np.float64)[elig], "qwen4b_benign": q["benign"].astype(np.float64)[elig]}
    for t in TAGS:
        p = paths(t)["table_npz"]
        if p.exists():
            tables[t] = np.load(p)["own"].astype(np.float64)[elig]
    names = list(tables)
    R = {n: rank(v) for n, v in tables.items()}
    n = int(elig.sum())
    out = {"run": "x112", "n_eligible": n, "spearman": {}, "overlap_top100": {}, "overlap_bottom100": {}, "overlap_top1pct": {}, "overlap_bottom1pct": {}}
    k1 = int(round(0.01 * n))
    for a in names:
        out["spearman"][a] = {}
        out["overlap_top100"][a] = {}
        out["overlap_bottom100"][a] = {}
        out["overlap_top1pct"][a] = {}
        out["overlap_bottom1pct"][a] = {}
        for b in names:
            out["spearman"][a][b] = float(np.corrcoef(R[a], R[b])[0, 1])
            for key, kk, top in (("overlap_top100", 100, True), ("overlap_bottom100", 100, False), ("overlap_top1pct", k1, True), ("overlap_bottom1pct", k1, False)):
                sa = set(np.argsort(tables[a])[-kk:] if top else np.argsort(tables[a])[:kk])
                sb = set(np.argsort(tables[b])[-kk:] if top else np.argsort(tables[b])[:kk])
                out[key][a][b] = len(sa & sb)
    out["k_1pct"] = k1
    # Chance overlap of two random k-subsets of n: k^2 / n.
    out["chance_overlap"] = {"top100": 100 * 100 / n, "1pct": k1 * k1 / n}
    out["labels"] = {"qwen4b_BP": "Qwen3.5-4B BP table (x111, 48 prompts)", "qwen4b_benign": "Qwen3.5-4B benign table (x109, 40 prompts)"} | {t: LABELS[t] + " (x112, 8 BP prompts)" for t in TAGS}
    out["mean_own_vs_qwen4b_BP"] = float(np.mean([out["spearman"][t]["qwen4b_BP"] for t in TAGS if t in tables]))
    eq = words[elig]
    out["n_words_shared_in_all_bottom1pct"] = int(len(set.intersection(*[set(eq[np.argsort(tables[t])[:k1]]) for t in names if t != "qwen4b_benign"]))) if len(names) > 2 else None
    out["n_words_shared_in_all_top1pct"] = int(len(set.intersection(*[set(eq[np.argsort(tables[t])[-k1:]]) for t in names if t != "qwen4b_benign"]))) if len(names) > 2 else None
    atomic_json(COMPARISON, out)
    print("Spearman vs qwen4b_BP:", {t: round(out["spearman"][t]["qwen4b_BP"], 3) for t in names})
    print("top100 overlap vs qwen4b_BP:", {t: out["overlap_top100"][t]["qwen4b_BP"] for t in names})
    print("bottom100 overlap vs qwen4b_BP:", {t: out["overlap_bottom100"][t]["qwen4b_BP"] for t in names})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-inputs", action="store_true")
    parser.add_argument("--pool", action="store_true")
    parser.add_argument("--request", action="store_true")
    parser.add_argument("--n", type=int, default=N_FIRST)
    parser.add_argument("--pop", action="append", choices=("benign", "harmful"))
    parser.add_argument("--prompts", action="store_true")
    parser.add_argument("--words", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--topup-request", action="store_true")
    parser.add_argument("--topup-prompts", action="store_true")
    parser.add_argument("--tag", action="append", choices=TAGS)
    args = parser.parse_args()
    tags = args.tag or list(TAGS)
    if args.scan_inputs:
        scan_inputs()
    for t in tags:
        if args.pool:
            build_pool(t)
        if args.request:
            request(t, args.n, tuple(args.pop or ("benign", "harmful")))
        if args.prompts:
            build_prompts(t)
        if args.words:
            build_words(t)
        if args.topup_request:
            topup_request(t)
        if args.topup_prompts:
            topup_prompts(t)
    if args.compare:
        compare()


if __name__ == "__main__":
    main()
