"""x111 selection (local, tokenizer-free). Tables, then words, then test prompts.

`--tables`: the three per-word tables over the x109 candidate list (71,583 words), each the mean
over its prompt set of the boundary shift summed over the two boundary tokens, in x106 Hard-1K sd
units: benign (x109 scan_shift.npz, 40 prompts), BP and HP (x110 scan_shift.npz, the 48 / 84
scanned BP / HP prompts), plus "mixed" (all 172 scanned prompts, equal weight; analysis only). Each word gets a percentile rank per table over the eligible words
(x109's drop rules 1: in the wordfreq top-50k or the x83 lexicon). Writes tables.json + tables.npz.

`--words`: 40 words, seed 111, stratified on the joint benign / BP percentile: 3 concordant words
from each of eight bins and 4 from each of four discordant quadrants; within a stratum the
eligible words are walked in a seeded random order and the first ones kept that are not judged
proper nouns and share no stem with an already chosen word (x109_scan_select.share_stem, applied
across the whole set). Writes words.json with the strata, per-word table positions and the draw log.

`--representative` (for x112): per table (benign, BP, HP, and mixed = all 172 scanned prompts), every
prompt of the table's set ranked by the Spearman rho between its own per-word shift and the table.
Writes representative_prompts.json.

`--prompts`: 32 benign (x106 Hard-1K, Qwen3.5-4B O_base refusal2 over ten samples) and 32 harmful
(x83 cohort, O refusal2 over ten samples) prompts with that rate in [0.3, 0.7], nearest 0.5 first,
ties broken by a seed-111 random key; none in x100's HP / BP (= x110's 192), none among x109's 40.
No test prompt is scanned before words and prompts are fixed. Writes prompts.json.

Top-up of the harmful stratum (after the run). The rule above did not drop the x85 crossed cohort's
96 x83 prompts, on which the x87 consensus direction was fitted, and nine of the 32 harmful prompts
drawn are in it (x85_exclusions.json; the analysis leaves them out). `--topup-request` continues the
same harmful rule (same pool, band, |rate - 0.5| order and seed-111 tie keys) past the 32 drawn and
additionally drops, by x83 id and text sha256, the x85 cohort's 96 prompts, x111's own 64, x110's 192
and every x112 prompt (prompts.json and prompts_topup.json of every model); it checks that the rule
still draws the frozen 32 and writes the ordered spares and the shortfall to topup_request.json.
`--topup-prompts` takes the first spares in that order, as many as the stratum lacks, ->
prompts_topup.json (prompts.json's row structure plus the spare's rank). The historical rate is used
directly, as for the original 64; no fresh O cell selects a prompt. prompts.json is never rewritten.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x109_scan_select import PROPER_NOUNS_JUDGED, share_stem, zipf_of  # noqa: E402
from x111_common import (  # noqa: E402
    PROMPTS,
    PROMPTS_TOPUP,
    REPRESENTATIVE,
    ROOT,
    SEED,
    TABLES_JSON,
    TABLES_NPZ,
    TOPUP_REQUEST,
    WORDS,
    X100_COHORT,
    X106_GENERATIONS_O,
    X106_HARD1K_SD,
    X106_REFUSAL_O,
    X109_PROMPTS,
    X109_SCAN_JSON,
    X109_SCAN_NPZ,
    X109_WORDFREQ,
    X110_PROMPTS,
    X110_SCAN_JSON,
    X110_SCAN_NPZ,
    X83_COHORT,
    X83_REFUSAL_O,
    X85_EXCLUSIONS,
    X112_RUN,
    atomic_json,
    read_json,
    sha256_text,
)
from x83_common import lexicon_words  # noqa: E402

N_PER_POP = 32
RATE_LO, RATE_HI = 0.3, 0.7
BIN_EDGES = (0, 1, 5, 25, 50, 75, 95, 99, 100)
BIN_NAMES = ("[0,1)", "[1,5)", "[5,25)", "[25,50)", "[50,75)", "[75,95)", "[95,99)", "[99,100]")
N_CONCORDANT = 3
N_DISCORDANT = 4
# Discordant quadrants: (name, rule) on the benign (b) and BP (p) percentiles.
QUADRANTS = (
    ("BP_low_benign_not", "BP <= p5 and benign >= p25", lambda b, p: (p <= 5) & (b >= 25)),
    ("benign_low_BP_not", "benign <= p5 and BP >= p25", lambda b, p: (b <= 5) & (p >= 25)),
    ("BP_high_benign_not", "BP >= p95 and benign <= p75", lambda b, p: (p >= 95) & (b <= 75)),
    ("benign_high_BP_not", "benign >= p95 and BP <= p75", lambda b, p: (b >= 95) & (p <= 75)),
)
# Proper nouns met on the seeded walk (manual judgment on the drawn words), on top of x109's.
PROPER_NOUNS_X111: set[str] = {"halle", "anton", "mcqueen", "iran", "huffpost", "yugoslavia"}


def bin_of(pct: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(np.asarray(BIN_EDGES[1:-1]), pct, side="right"), 0, len(BIN_NAMES) - 1)


def build_tables() -> None:
    x109 = read_json(X109_SCAN_JSON)
    x110 = read_json(X110_SCAN_JSON)
    if x109["candidates"]["words_sha256"] != x110["candidates"]["words_sha256"]:
        raise SystemExit("FATAL: x109 and x110 candidate lists differ")
    words = x109["candidates"]["words"]
    if words != x110["candidates"]["words"] or sha256_text("\n".join(words)) != x109["candidates"]["words_sha256"]:
        raise SystemExit("FATAL: candidate lists differ in order or fail their sha256")
    s109 = np.load(X109_SCAN_NPZ)["shift"].astype(np.float64).sum(axis=2)  # [40, N]
    s110 = np.load(X110_SCAN_NPZ)["shift"].astype(np.float64).sum(axis=2)  # [132, N]
    scanned = list(x110["prompt_indices_scanned"])
    if s110.shape[0] != len(scanned) or s109.shape[0] != len(x109["prompt_ids"]):
        raise SystemExit("FATAL: scan shapes do not match their json")
    pops = np.asarray([x110["prompt_populations"][i] for i in scanned])
    tables = {
        "benign": s109.mean(0) / X106_HARD1K_SD,
        "BP": s110[pops == "BP"].mean(0) / X106_HARD1K_SD,
        "HP": s110[pops == "HP"].mean(0) / X106_HARD1K_SD,
        "mixed": np.concatenate([s109, s110]).mean(0) / X106_HARD1K_SD,
    }
    lexset = {w for w, _ in lexicon_words()}
    wf = read_json(X109_WORDFREQ)["words"]
    wf_rank = {w: i + 1 for i, w in enumerate(wf)}
    eligible = np.asarray([w in wf_rank or w in lexset for w in words])
    pct = {}
    for name, t in tables.items():
        p = np.full(len(words), np.nan)
        idx = np.where(eligible)[0]
        order = idx[np.argsort(t[idx], kind="stable")]
        p[order] = 100.0 * np.arange(len(order)) / len(order)
        pct[name] = p
    np.savez_compressed(
        TABLES_NPZ,
        benign=tables["benign"].astype(np.float32),
        BP=tables["BP"].astype(np.float32),
        HP=tables["HP"].astype(np.float32),
        mixed=tables["mixed"].astype(np.float32),
        pct_benign=pct["benign"],
        pct_BP=pct["BP"],
        pct_HP=pct["HP"],
        pct_mixed=pct["mixed"],
        eligible=eligible,
        n_tokens=np.asarray(x109["candidates"]["n_tokens"], np.int16),
    )
    elig = eligible
    corr = {}
    for a, b in (("benign", "BP"), ("benign", "HP"), ("BP", "HP"), ("mixed", "benign"), ("mixed", "BP"), ("mixed", "HP")):
        ra, rb = np.argsort(np.argsort(tables[a][elig])), np.argsort(np.argsort(tables[b][elig]))
        corr[f"{a}_vs_{b}"] = {"spearman_eligible": float(np.corrcoef(ra, rb)[0, 1]), "pearson_eligible": float(np.corrcoef(tables[a][elig], tables[b][elig])[0, 1])}
    out = {
        "run": "x111",
        "candidates": {"source": str(X109_SCAN_JSON.relative_to(ROOT)), "n": len(words), "words_sha256": x109["candidates"]["words_sha256"], "same_list_in_x110": True},
        "statistic": "per word: mean over the table's prompts of the shift summed over the two boundary tokens (tbg_minus1, tbg), divided by the x106 Hard-1K sd (8.645); percentile = 100 * ascending rank / n over the eligible words (0 = most negative)",
        "eligible_rule": "in the wordfreq top-50k list or the x83 lexicon (x109 drop rule 1)",
        "n_eligible": int(eligible.sum()),
        "tables": {
            "benign": {"scan": str(X109_SCAN_NPZ.relative_to(ROOT)), "n_prompts": int(s109.shape[0]), "prompt_ids": x109["prompt_ids"], "prompts_sha256": x109["prompts_sha256"]},
            "BP": {"scan": str(X110_SCAN_NPZ.relative_to(ROOT)), "n_prompts": int((pops == "BP").sum()), "prompt_ids": [x110["prompt_ids"][i] for i in scanned if x110["prompt_populations"][i] == "BP"], "prompts_sha256": x110["prompts_sha256"]},
            "HP": {"scan": str(X110_SCAN_NPZ.relative_to(ROOT)), "n_prompts": int((pops == "HP").sum()), "prompt_ids": [x110["prompt_ids"][i] for i in scanned if x110["prompt_populations"][i] == "HP"], "prompts_sha256": x110["prompts_sha256"], "use": "analysis only"},
            "mixed": {"scan": "both", "n_prompts": int(s109.shape[0] + s110.shape[0]), "rule": "mean over all 172 scanned prompts (x109's 40 + x110's 48 BP + 84 HP), equal weight per prompt", "use": "analysis only"},
        },
        "distribution_sd_eligible": {k: {q: float(np.quantile(t[elig], v)) for q, v in (("p1", 0.01), ("p5", 0.05), ("p25", 0.25), ("median", 0.5), ("p75", 0.75), ("p95", 0.95), ("p99", 0.99))} for k, t in tables.items()},
        "table_correlations": corr,
        "npz_arrays": "benign, BP, HP, mixed (sd units, candidate order); pct_benign, pct_BP, pct_HP, pct_mixed (nan for ineligible); eligible; n_tokens",
    }
    atomic_json(TABLES_JSON, out)
    print(json.dumps({k: v for k, v in out.items() if k in ("n_eligible", "distribution_sd_eligible", "table_correlations")}, indent=1))


def build_words() -> None:
    meta = read_json(TABLES_JSON)
    z = np.load(TABLES_NPZ)
    words = read_json(X109_SCAN_JSON)["candidates"]["words"]
    if sha256_text("\n".join(words)) != meta["candidates"]["words_sha256"]:
        raise SystemExit("FATAL: candidate list differs from tables.json")
    eligible = z["eligible"]
    pb, pp, ph = z["pct_benign"], z["pct_BP"], z["pct_HP"]
    bb = np.where(eligible, bin_of(np.nan_to_num(pb)), -1)
    bp = np.where(eligible, bin_of(np.nan_to_num(pp)), -1)
    lexset = {w for w, _ in lexicon_words()}
    wf = read_json(X109_WORDFREQ)["words"]
    wf_rank = {w: i + 1 for i, w in enumerate(wf)}
    proper = PROPER_NOUNS_JUDGED | PROPER_NOUNS_X111

    strata = []
    for k, name in enumerate(BIN_NAMES):
        strata.append({"stratum": f"concordant {name}", "kind": "concordant", "bin": k, "bin_name": name, "rule": f"benign and BP percentiles both in {name}", "mask": eligible & (bb == k) & (bp == k), "n": N_CONCORDANT})
    for name, rule, fn in QUADRANTS:
        strata.append({"stratum": f"discordant {name}", "kind": "discordant", "bin": None, "bin_name": None, "quadrant": name, "rule": rule, "mask": eligible & fn(np.nan_to_num(pb), np.nan_to_num(pp)), "n": N_DISCORDANT})

    def row(i: int) -> dict:
        w = words[i]
        return {
            "word": w,
            "index": int(i),
            "benign_sd": float(z["benign"][i]),
            "BP_sd": float(z["BP"][i]),
            "HP_sd": float(z["HP"][i]),
            "pct_benign": float(pb[i]),
            "pct_BP": float(pp[i]),
            "pct_HP": float(ph[i]),
            "bin_benign": BIN_NAMES[bb[i]],
            "bin_BP": BIN_NAMES[bp[i]],
            "n_tokens": int(z["n_tokens"][i]),
            "wordfreq_rank": wf_rank.get(w),
            "zipf": zipf_of(w),
            "in_x83_lexicon": w in lexset,
        }

    chosen: list[dict] = []
    draw_log = []
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
            print(f"WARNING: stratum {s['stratum']} filled {got}/{s['n']}")
    out = {
        "run": "x111",
        "tables": str(TABLES_JSON.relative_to(ROOT)),
        "words_sha256_candidates": meta["candidates"]["words_sha256"],
        "seed": SEED,
        "bins": list(BIN_NAMES),
        "selection_rule": "per stratum, the eligible words (wordfreq top-50k or x83 lexicon) in a numpy default_rng([111, stratum index]) permutation; keep the first n that are not judged proper nouns and share no stem (x109_scan_select.share_stem) with any word already chosen in an earlier stratum or earlier in this one; strata in the listed order (8 concordant bins, then 4 discordant quadrants)",
        "proper_nouns_judged": sorted(proper),
        "strata": [{k: v for k, v in s.items() if k != "mask"} | {"n_eligible": int(s["mask"].sum())} for s in strata],
        "chosen": chosen,
        "draw_log": draw_log,
        "chosen_sha256": sha256_text("\n".join(c["word"] for c in chosen)),
    }
    atomic_json(WORDS, out)
    for s in out["strata"]:
        print(f"{s['stratum']:36s} eligible {s['n_eligible']}")
    for c in chosen:
        print(f"  {c['stratum']:36s} {c['word']:18s} benign {c['benign_sd']:+.2f} ({c['pct_benign']:5.1f}) BP {c['BP_sd']:+.2f} ({c['pct_BP']:5.1f}) HP {c['HP_sd']:+.2f} tok {c['n_tokens']} wf {c['wordfreq_rank']}")
    for log in draw_log:
        drops = [x for x in log["walk"] if x["action"] != "chosen"]
        if drops:
            print(log["stratum"], drops)


def load_pools() -> tuple[list[dict], list[dict], dict]:
    """The benign and harmful candidate pools of the prompt rule, each row flagged `excluded` when it
    is an x100 HP / BP prompt, one of x110's 192 or one of x109's 40; also the x106 O_base file."""
    cohort = read_json(X100_COHORT)
    ex_sha, ex_ids = set(), set()
    for pop in ("HP", "BP"):
        for p in cohort["populations"][pop]["prompts"]:
            ex_sha.add(p["prompt_sha256"])
            ex_ids.add((pop, p["id"]))
    for p in read_json(X110_PROMPTS)["prompts"]:
        ex_sha.add(p["text_sha256"])
    for p in read_json(X109_PROMPTS)["prompts"]:
        ex_sha.add(p["text_sha256"])
    x100_hp_ids = {i for pop, i in ex_ids if pop == "HP"}
    x100_bp_ids = {i for pop, i in ex_ids if pop == "BP"}

    # Benign pool: x106 Hard-1K O_base generations (Qwen3.5-4B), refusal2 rows matched by sha256.
    gen = read_json(X106_GENERATIONS_O)
    rows_by_sha = {r["prompt_sha256"]: r for r in gen["rows"]}
    benign_pool = []
    for line in X106_REFUSAL_O.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        g = rows_by_sha[rec["prompt_sha256"]]
        if sha256_text(g["prompt"]) != rec["prompt_sha256"]:
            raise SystemExit(f"FATAL: x106 row {g['id']} fails its sha256")
        benign_pool.append({"id": g["id"], "source_id": g["id"], "text": g["prompt"], "text_sha256": rec["prompt_sha256"], "historical_refusal2": float(np.mean(rec["refusal"])), "historical_n": len(rec["refusal"]), "unit": g.get("unit"), "x106_rendered_chat_sha256": g["rendered_chat_sha256"], "excluded": g["id"] in x100_bp_ids or rec["prompt_sha256"] in ex_sha})
    # Harmful pool: x83 cohort O prompts with x83's O refusal2 rows.
    units = {u["id"]: u for u in read_json(X83_COHORT)["units"]}
    harmful_pool = []
    for line in X83_REFUSAL_O.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        u = units[rec["id"]]
        text = u["prompts"]["O"]
        if sha256_text(text) != u["prompt_sha256"]["O"] or rec["prompt_sha256"] != u["prompt_sha256"]["O"]:
            raise SystemExit(f"FATAL: x83 unit {rec['id']} fails its sha256")
        harmful_pool.append({"id": f"x83:{rec['id']}", "source_id": rec["id"], "text": text, "text_sha256": u["prompt_sha256"]["O"], "historical_refusal2": float(np.mean(rec["refusal"])), "historical_n": len(rec["refusal"]), "x83_word": u["word"], "x83_category": u["category"], "excluded": rec["id"] in x100_hp_ids or u["prompt_sha256"]["O"] in ex_sha})
    return benign_pool, harmful_pool, gen


def rule_order(pool: list[dict], population: str, drop=lambda p: False) -> tuple[list[int], list[int], list[int], np.ndarray]:
    """The prompt rule on one pool: in-band indices, eligible indices (not `excluded`, not `drop`),
    and the eligible indices sorted by |rate - 0.5| with the seed-111 tie keys (drawn over the whole
    pool, so an extra drop does not change any other prompt's key)."""
    rng = np.random.default_rng([SEED, POPULATION_INDEX[population]])
    tie = rng.random(len(pool))
    in_band = [k for k, p in enumerate(pool) if RATE_LO - 1e-9 <= p["historical_refusal2"] <= RATE_HI + 1e-9]
    eligible = [k for k in in_band if not pool[k]["excluded"] and not drop(pool[k])]
    order = sorted(eligible, key=lambda k: (round(abs(pool[k]["historical_refusal2"] - 0.5), 6), tie[k]))
    return in_band, eligible, order, tie


def rate_counts(rates) -> dict[str, int]:
    rates = list(rates)
    return {f"{r:.1f}": sum(1 for v in rates if abs(v - r) < 1e-9) for r in (0.3, 0.4, 0.5, 0.6, 0.7)}


def build_prompts() -> None:
    benign_pool, harmful_pool, gen = load_pools()
    rows, pool_summary = [], {}
    for population, pool in (("benign", benign_pool), ("harmful", harmful_pool)):
        in_band, eligible, order, _ = rule_order(pool, population)
        pick = order[:N_PER_POP]
        pool_summary[population] = {
            "n_pool": len(pool),
            "n_excluded_total": sum(p["excluded"] for p in pool),
            "n_in_band": len(in_band),
            "n_in_band_excluded": len(in_band) - len(eligible),
            "n_eligible": len(eligible),
            "eligible_rate_counts": {f"{r:.1f}": sum(1 for k in eligible if abs(pool[k]["historical_refusal2"] - r) < 1e-9) for r in (0.3, 0.4, 0.5, 0.6, 0.7)},
            "chosen_rate_counts": {f"{r:.1f}": sum(1 for k in pick if abs(pool[k]["historical_refusal2"] - r) < 1e-9) for r in (0.3, 0.4, 0.5, 0.6, 0.7)},
        }
        for k in pick:
            p = {kk: v for kk, v in pool[k].items() if kk != "excluded"}
            rows.append({**p, "population": population, "n_words": len(p["text"].split())})
    out = {
        "run": "x111",
        "rule": "rate = historical refusal2 over ten O samples (x106 O_base for benign, x83 O for harmful; x64 judge); keep rate in [0.3, 0.7]; drop x100 HP / BP (= x110's 192) and x109's 40 prompts by id and text sha256; sort by |rate - 0.5|, ties by a numpy default_rng([111, population index]) uniform key; first 32 per population",
        "sources": {"benign": {"refusal2": str(X106_REFUSAL_O.relative_to(ROOT)), "text": str(X106_GENERATIONS_O.relative_to(ROOT))}, "harmful": {"refusal2": str(X83_REFUSAL_O.relative_to(ROOT)), "text": str(X83_COHORT.relative_to(ROOT))}},
        "model_id": gen["model_id"],
        "model_snapshot": gen["model_source"],
        "pools": pool_summary,
        "n_prompts": len(rows),
        "prompts_sha256": sha256_text("\n".join(f"{r['id']}\t{r['text_sha256']}" for r in rows)),
        "prompts": rows,
    }
    atomic_json(PROMPTS, out)
    print(json.dumps(pool_summary, indent=1))
    for population in POPULATIONS_ORDER:
        sub = [r for r in rows if r["population"] == population]
        print(population, len(sub), "mean historical", np.mean([r["historical_refusal2"] for r in sub]), "mean words", np.mean([r["n_words"] for r in sub]))


def _sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def topup_exclusions() -> tuple[dict, dict]:
    """Prompts the top-up may not draw, by id and by text sha256, per source; plus provenance."""
    ex = read_json(X85_EXCLUSIONS)
    x85_path = ROOT / ex["x85_cohort_file"]
    if _sha_file(x85_path) != ex["x85_cohort_file_sha256"]:
        raise SystemExit("FATAL: the x85 cohort file differs from the one x85_exclusions.json froze")
    x85 = read_json(x85_path)
    if x85["cohort_sha256"] != ex["x85_cohort_sha256"] or len(x85["prompts"]) != ex["x85_n_prompts"]:
        raise SystemExit("FATAL: the x85 cohort does not match x85_exclusions.json")
    sources: dict[str, dict[str, set]] = {
        "x85_cohort": {"ids": {f"x83:{p['id']}" for p in x85["prompts"]}, "sha": {sha256_text(p["prompt"]) for p in x85["prompts"]}},
        "x111_prompts": {"ids": {p["id"] for p in read_json(PROMPTS)["prompts"]}, "sha": {p["text_sha256"] for p in read_json(PROMPTS)["prompts"]}},
        "x110_prompts": {"ids": {p["id"] for p in read_json(X110_PROMPTS)["prompts"]}, "sha": {p["text_sha256"] for p in read_json(X110_PROMPTS)["prompts"]}},
    }
    x112_files = sorted(glob.glob(str(X112_RUN / "*" / "prompts.json")) + glob.glob(str(X112_RUN / "*" / "prompts_topup.json")))
    x112 = [p for f in x112_files for p in read_json(f)["prompts"]]
    sources["x112_prompts"] = {"ids": {p["id"] for p in x112}, "sha": {p["text_sha256"] for p in x112}}
    prov = {
        "x85_cohort": {"file": ex["x85_cohort_file"], "file_sha256": ex["x85_cohort_file_sha256"], "cohort_sha256": ex["x85_cohort_sha256"], "n_prompts": len(x85["prompts"])},
        "x111_prompts": {"file": str(PROMPTS.relative_to(ROOT)), "prompts_sha256": read_json(PROMPTS)["prompts_sha256"], "n_prompts": len(sources["x111_prompts"]["ids"])},
        "x110_prompts": {"file": str(X110_PROMPTS.relative_to(ROOT)), "file_sha256": _sha_file(X110_PROMPTS), "n_prompts": len(read_json(X110_PROMPTS)["prompts"])},
        "x112_prompts": {"files": [{"file": str(Path(f).relative_to(ROOT)), "file_sha256": _sha_file(f)} for f in x112_files], "n_rows": len(x112), "n_distinct_ids": len(sources["x112_prompts"]["ids"])},
    }
    return sources, prov


def topup_request() -> None:
    _, harmful_pool, _ = load_pools()
    frozen = read_json(PROMPTS)
    frozen_h = [p["id"] for p in frozen["prompts"] if p["population"] == "harmful"]
    in_band, eligible, order, tie = rule_order(harmful_pool, "harmful")
    if [harmful_pool[k]["id"] for k in order[:N_PER_POP]] != frozen_h:
        raise SystemExit("FATAL: the harmful rule no longer draws the frozen 32 prompts of prompts.json")
    x85_ids = set(read_json(X85_EXCLUSIONS)["excluded_x111_prompt_ids"])
    retained = [p for p in frozen["prompts"] if p["population"] == "harmful" and p["id"] not in x85_ids]
    need = N_PER_POP - len(retained)
    sources, prov = topup_exclusions()

    def hits(p: dict) -> list[str]:
        return [name for name, s in sources.items() if p["id"] in s["ids"] or p["text_sha256"] in s["sha"]]

    rank_of = {k: i for i, k in enumerate(order)}
    spares, dropped = [], []
    for k in order[N_PER_POP:]:
        p = harmful_pool[k]
        h = hits(p)
        row = {"id": p["id"], "rank_in_rule_order": rank_of[k], "historical_refusal2": p["historical_refusal2"], "tie_key": float(tie[k]), "text_sha256": p["text_sha256"]}
        (dropped if h else spares).append(row | ({"dropped_as": h} if h else {}))
    out = {
        "run": "x111",
        "stage": "topup",
        "population": "harmful",
        "rule": "x111's harmful prompt rule continued past the 32 drawn: x83 cohort O prompts with historical refusal2 (x83 O, ten samples, x64 judge) in [0.3, 0.7]; x100 HP / BP (= x110's 192) and x109's 40 dropped by id and text sha256; sorted by |rate - 0.5|, ties by the numpy default_rng([111, 1]) uniform key over the whole pool; additionally dropped by x83 id and text sha256: the x85 crossed cohort's 96 prompts, x111's 64 prompts, x110's 192 prompts and every x112 prompt (prompts.json and prompts_topup.json of every model). The first spares in that order fill the stratum back to 32. The historical rate selects, as for the original prompts; no fresh O cell.",
        "seed": SEED,
        "tie_key_rng": "numpy default_rng([111, 1]).random(n_pool), indexed by pool position (x83 O refusal file order)",
        "sources": {"refusal2": str(X83_REFUSAL_O.relative_to(ROOT)), "refusal2_sha256": _sha_file(X83_REFUSAL_O), "text": str(X83_COHORT.relative_to(ROOT)), "text_sha256_file": _sha_file(X83_COHORT)},
        "x85_exclusions": {"file": str(X85_EXCLUSIONS.relative_to(ROOT)), "excluded_x111_prompt_ids": sorted(x85_ids)},
        "exclusion_sources": prov,
        "n_pool": len(harmful_pool),
        "n_in_band": len(in_band),
        "n_eligible_original_rule": len(eligible),
        "n_chosen_original": len(frozen_h),
        "frozen_32_reproduced": True,
        "n_retained_after_x85": len(retained),
        "n_needed": need,
        "rate_counts_chosen_original": rate_counts(p["historical_refusal2"] for p in frozen["prompts"] if p["population"] == "harmful"),
        "rate_counts_retained": rate_counts(p["historical_refusal2"] for p in retained),
        "n_beyond_32": len(order) - N_PER_POP,
        "n_dropped_beyond_32": len(dropped),
        "dropped_beyond_32": dropped,
        "n_spares": len(spares),
        "rate_counts_spares": rate_counts(r["historical_refusal2"] for r in spares),
        "spares": spares,
    }
    atomic_json(TOPUP_REQUEST, out)
    print(f"retained {len(retained)} ({out['rate_counts_retained']}), need {need}; beyond 32: {out['n_beyond_32']}, dropped {len(dropped)}, spares {len(spares)} {out['rate_counts_spares']} -> {TOPUP_REQUEST}")
    for d in dropped:
        print("  dropped", d["id"], d["historical_refusal2"], d["dropped_as"])


def topup_prompts() -> None:
    req = read_json(TOPUP_REQUEST)
    _, harmful_pool, _ = load_pools()
    by_id = {p["id"]: p for p in harmful_pool}
    need = req["n_needed"]
    pick = req["spares"][:need]
    if len(pick) < need:
        raise SystemExit(f"FATAL: {len(pick)} spares for {need} places")
    rows = []
    for s_ in pick:
        p = by_id[s_["id"]]
        if p["text_sha256"] != s_["text_sha256"]:
            raise SystemExit(f"FATAL: {s_['id']} text differs from topup_request.json")
        row = {k: v for k, v in p.items() if k != "excluded"}
        rows.append({**row, "population": "harmful", "n_words": len(row["text"].split()), "rank": s_["rank_in_rule_order"]})
    frozen = read_json(PROMPTS)
    out = {
        "run": "x111",
        "stage": "topup",
        "rule": "the first n_needed spares of topup_request.json in the rule's order (x111's harmful rule continued past the 32 drawn, with the x85 cohort, x111's, x110's and x112's prompts dropped); rows have prompts.json's structure plus `rank` (position in the rule's order)",
        "model_id": frozen["model_id"],
        "model_snapshot": frozen["model_snapshot"],
        "topup": {
            "request": str(TOPUP_REQUEST.relative_to(ROOT)),
            "request_sha256": _sha_file(TOPUP_REQUEST),
            "n_chosen_original": req["n_chosen_original"],
            "n_retained_after_x85": req["n_retained_after_x85"],
            "n_needed": need,
            "rate_counts_retained": req["rate_counts_retained"],
            "rate_counts_added": rate_counts(r["historical_refusal2"] for r in rows),
            "pool": {"n_pool": req["n_pool"], "n_in_band": req["n_in_band"], "n_eligible_original_rule": req["n_eligible_original_rule"], "n_dropped_beyond_32": req["n_dropped_beyond_32"], "n_spares": req["n_spares"], "rate_counts_spares": req["rate_counts_spares"]},
        },
        "n_prompts": len(rows),
        "prompts_sha256": sha256_text("\n".join(f"{r['id']}\t{r['text_sha256']}" for r in rows)),
        "prompts": rows,
    }
    atomic_json(PROMPTS_TOPUP, out)
    for r in rows:
        print(r["id"], r["historical_refusal2"], r["rank"], r["x83_word"], r["x83_category"], r["n_words"])
    print("->", PROMPTS_TOPUP, out["topup"]["rate_counts_added"])


def build_representative() -> None:
    """Per table, every prompt of its prompt set ranked by the Spearman rho between the prompt's own
    per-word shift (eligible words, both boundary tokens summed) and the table's per-word mean."""
    from scipy import stats

    z = np.load(TABLES_NPZ)
    elig = z["eligible"]
    x109 = read_json(X109_SCAN_JSON)
    x110 = read_json(X110_SCAN_JSON)
    p109 = {p["id"]: p for p in read_json(X109_PROMPTS)["prompts"]}
    p110 = {p["id"]: p for p in read_json(X110_PROMPTS)["prompts"]}
    s109 = np.load(X109_SCAN_NPZ)["shift"].astype(np.float64).sum(axis=2)[:, elig]
    s110 = np.load(X110_SCAN_NPZ)["shift"].astype(np.float64).sum(axis=2)[:, elig]
    rows = []
    for k, pid in enumerate(x109["prompt_ids"]):
        p = p109[pid]
        rows.append({"id": pid, "population": "benign", "source_set": f"x109 {x109['prompt_sets'][k]}", "text": p["text"], "shift": s109[k]})
    for k, pi in enumerate(x110["prompt_indices_scanned"]):
        pid = x110["prompt_ids"][pi]
        pop = x110["prompt_populations"][pi]
        rows.append({"id": pid, "population": pop, "source_set": f"x110 {pop}", "x100_id": p110[pid]["x100_id"], "text": p110[pid]["text"], "shift": s110[k]})
    members = {"benign": lambda r: r["source_set"].startswith("x109"), "BP": lambda r: r["source_set"] == "x110 BP", "HP": lambda r: r["source_set"] == "x110 HP", "mixed": lambda r: True}
    out = {"run": "x111", "rule": "per table, each prompt of the table's prompt set: Spearman rho between the prompt's per-word shift (eligible candidate words, shift summed over the two boundary tokens) and the table's per-word mean (which includes that prompt); sorted descending", "n_eligible_words": int(elig.sum()), "top_n": 8, "tables": {}}
    for t, fn in members.items():
        tv = z[t].astype(np.float64)[elig]
        ranked = []
        for r in rows:
            if fn(r):
                ranked.append({kk: v for kk, v in r.items() if kk != "shift"} | {"spearman_with_table": float(stats.spearmanr(r["shift"], tv).statistic)})
        ranked.sort(key=lambda r: -r["spearman_with_table"])
        out["tables"][t] = {"n_prompts": len(ranked), "top8": [{"id": r["id"], "population": r["population"], "source_set": r["source_set"], "spearman_with_table": r["spearman_with_table"]} for r in ranked[:8]], "prompts": ranked}
        print(t, len(ranked), [(r["id"], round(r["spearman_with_table"], 3)) for r in ranked[:8]])
    atomic_json(REPRESENTATIVE, out)


POPULATIONS_ORDER = ("benign", "harmful")
POPULATION_INDEX = {"benign": 0, "harmful": 1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", action="store_true")
    parser.add_argument("--words", action="store_true")
    parser.add_argument("--prompts", action="store_true")
    parser.add_argument("--representative", action="store_true")
    parser.add_argument("--topup-request", action="store_true")
    parser.add_argument("--topup-prompts", action="store_true")
    args = parser.parse_args()
    if args.tables:
        build_tables()
    if args.words:
        build_words()
    if args.prompts:
        build_prompts()
    if args.representative:
        build_representative()
    if args.topup_request:
        topup_request()
    if args.topup_prompts:
        topup_prompts()
    if not (args.tables or args.words or args.prompts or args.representative or args.topup_request or args.topup_prompts):
        raise SystemExit("--tables, --words, --prompts, --representative, --topup-request and/or --topup-prompts")


if __name__ == "__main__":
    main()
