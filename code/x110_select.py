"""x110 selection (local, tokenizer-free).

`--prompts`: x100's HP and BP populations verbatim from x100_cohort.json, with the x100
rendered-chat sha256 of each O cell so the GPU stage can assert the same rendering. Writes
prompts.json.

`--words`: reads the x109 scan (scan.json + scan_shift.npz, 40 benign prompts x 71,583
candidates), ranks candidates by mean shift over all 40 prompts ascending, keeps the first ten
distinct words under x109's drop rules, assigns neutral twins by x109's rule (table shift
over all 40 prompts in place of its set-A shift), and writes words.json.

Top-up of the HP strata (after the run). 15 HP prompts are x85 crossed-cohort prompts and are
dropped from the analysis (x85_exclusions.json: 11 in r0.2-0.5, 4 in r0.6-0.9). x100's HP rule
draws 32 prompts per historical-refusal2 stratum of the x66 seed-42 evaluation split with one
numpy default_rng(100), stratum by stratum (x100_common.select_hp). `--topup-request` replays that
draw (asserting it reproduces x100's 96), then continues the same generator: per stratum with a
shortfall, in x100's stratum order, it draws the shortfall with the same call over the stratum's
id-sorted candidates that are neither x100 prompts nor in the exclusion set (the x85 cohort, x100 /
x110, x109, x111 and x112 prompts, by id and text sha256) -> topup_request.json. The rule has no
fresh-rate check (the stratum is the historical rate), so there is no over-draw.
`--topup-prompts` freezes the drawn prompts in prompts.json's row structure plus provenance ->
prompts_topup.json. prompts.json itself is frozen and never rewritten.
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x109_scan_select import PROPER_NOUNS_JUDGED, SUFFIXES, share_stem, zipf_of  # noqa: E402
from x110_common import (  # noqa: E402
    EXCLUSIONS,
    N_TOP,
    POPULATIONS,
    PROMPTS,
    PROMPTS_TOPUP,
    ROOT,
    TOPUP_REQUEST,
    TWIN_CANDIDATES,
    WORDS,
    X100_COHORT,
    X106_HARD1K_SD,
    X109_SCAN_JSON,
    X109_SCAN_NPZ,
    X109_WORDFREQ,
    atomic_json,
    read_json,
    sha256_text,
)
from x83_common import lexicon_words  # noqa: E402
from x85_common import WORD_PAIRS  # noqa: E402
from x100_common import HP_STRATA, N_PROMPTS, SEEDS, hp_stratum, prompt_key, select_hp, sha256_file  # noqa: E402

BOTTOM_REPORT = 50
# x100's 28 words, for the table positions reported next to the chosen ten.
X100_WORDS = {
    "containment": ("sandbox", "simulation", "testnet", "localhost"),
    "authorization": ("authorized", "licensed", "permitted", "sanctioned"),
    "defensive": ("defensive", "mitigation", "detection", "prevention"),
    "educational": ("educational", "academic", "historical", "theoretical"),
    "fictional": ("hypothetical", "fictional", "screenplay", "roleplay"),
    "professional": ("researcher", "clinician", "toxicologist", "journalist"),
    "harm_control": ("bomb", "illegal", "malware", "cyanide"),
}


def build_prompts() -> None:
    cohort = read_json(X100_COHORT)
    if cohort.get("experiment") != "x100" or cohort.get("status") != "frozen":
        raise SystemExit("FATAL: x100 cohort missing or not frozen")
    o_cells = {(c["population"], c["prompt_key"]): c for c in cohort["cells"] if c["condition"] == "O"}
    rows = []
    for population in POPULATIONS:
        for p in cohort["populations"][population]["prompts"]:
            if sha256_text(p["prompt"]) != p["prompt_sha256"]:
                raise SystemExit(f"FATAL: x100 prompt {p['key']} fails its sha256")
            o = o_cells[(population, p["key"])]
            if o["prompt_sha256"] != p["prompt_sha256"]:
                raise SystemExit(f"FATAL: x100 O cell {p['key']} text differs from its population row")
            rows.append(
                {
                    "id": p["key"],
                    "population": population,
                    "x100_id": p["id"],
                    "x100_prompt_index": p["prompt_index"],
                    "stratum": p.get("stratum"),
                    "historical_refusal_rate": p.get("historical_refusal_rate"),
                    "category": p.get("category"),
                    "x87_label": p.get("x87_label"),
                    "text": p["prompt"],
                    "text_sha256": p["prompt_sha256"],
                    "x100_rendered_chat_sha256": o["rendered_chat_sha256"],
                    "n_words": len(p["prompt"].split()),
                }
            )
    out = {
        "run": "x110",
        "source": {"path": str(X100_COHORT.relative_to(ROOT)), "cohort_sha256": cohort["cohort_sha256"], "rule": "x100's HP (96 harmful prompts, 32 per historical refusal2 stratum) and BP (96 OR-Bench Hard-1K prompts the x87 base refused), verbatim, in x100 population order"},
        "model_id": cohort["model_id"],
        "model_snapshot": cohort["model_snapshot"],
        "n_prompts": len(rows),
        "prompts_sha256": sha256_text("\n".join(f"{r['id']}\t{r['text_sha256']}" for r in rows)),
        "prompts": rows,
    }
    atomic_json(PROMPTS, out)
    for population in POPULATIONS:
        sub = [r for r in rows if r["population"] == population]
        print(f"{population}: {len(sub)} prompts, {np.mean([r['n_words'] for r in sub]):.0f} words on average")


def build_words() -> None:
    scan = read_json(X109_SCAN_JSON)
    z = np.load(X109_SCAN_NPZ)
    shift = z["shift"].astype(np.float64).sum(axis=2)  # [40, N]
    cands = scan["candidates"]["words"]
    n_tok = scan["candidates"]["n_tokens"]
    sets = np.asarray(scan["prompt_sets"])
    mAll, mA, mB = shift.mean(0), shift[sets == "A"].mean(0), shift[sets == "B"].mean(0)
    neg_frac = (shift < 0).mean(0)
    order = np.argsort(mAll, kind="stable")  # ascending: most negative first
    rank_bottom = np.empty(len(cands), int)
    rank_bottom[order] = np.arange(1, len(cands) + 1)
    rank_top = len(cands) + 1 - rank_bottom
    lexset = {w for w, _ in lexicon_words()}
    wf = set(read_json(X109_WORDFREQ)["words"])
    index = {w: i for i, w in enumerate(cands)}

    def row(i: int) -> dict:
        w = cands[i]
        return {
            "rank_from_bottom": int(rank_bottom[i]),
            "rank_from_top": int(rank_top[i]),
            "word": w,
            "mean_shift_all": float(mAll[i]),
            "sd_units_all": float(mAll[i] / X106_HARD1K_SD),
            "mean_shift_A": float(mA[i]),
            "mean_shift_B": float(mB[i]),
            "frac_prompts_negative": float(neg_frac[i]),
            "n_tokens": int(n_tok[i]),
            "in_x83_lexicon": w in lexset,
            "in_wordfreq_top50k": w in wf,
            "zipf": zipf_of(w),
        }

    bottom50 = [row(i) for i in order[:BOTTOM_REPORT]]
    chosen: list[str] = []
    drops = []
    for i in order:
        w = cands[i]
        if len(chosen) == N_TOP:
            break
        if w not in wf and w not in lexset:
            drops.append({**row(i), "rule": "non-word: not in the wordfreq top-50k list and not in the x83 lexicon"})
            continue
        if w in PROPER_NOUNS_JUDGED:
            drops.append({**row(i), "rule": "proper noun (manual judgment on the ranking)"})
            continue
        hit = next((c for c in chosen if share_stem(c, w)), None)
        if hit:
            drops.append({**row(i), "rule": f"inflection/derivative/respelling sharing a stem with a lower-ranked chosen word '{hit}'"})
            continue
        chosen.append(w)

    x85_twin = {harm: neutral for _, harm, neutral in WORD_PAIRS}
    x85_neutrals = set(x85_twin.values())
    nouns = [n for n in TWIN_CANDIDATES if n in index and n not in lexset and n not in x85_neutrals]
    noun_shift = {n: float(mAll[index[n]]) for n in nouns}
    median = float(np.median(list(noun_shift.values())))
    used: set[str] = set()
    records = []
    for w in chosen:
        i = index[w]
        if w in x85_twin:
            twin, rule = x85_twin[w], "x85 WORD_PAIRS neutral"
        else:
            twin = None
            for tol in (0, 1):
                for n in nouns:
                    if n in used or abs(len(n) - len(w)) > tol or abs(noun_shift[n] - median) > 1.0:
                        continue
                    twin = n
                    break
                if twin:
                    break
            rule = "first unused fixed-list noun with the same character length (else +-1) and mean table shift within +-1.0 of the median fixed-list noun shift"
        if twin is None:
            raise SystemExit(f"FATAL: no twin for {w}")
        used.add(twin)
        records.append({**row(i), "twin": twin, "twin_rule": rule, **{f"twin_{k}": v for k, v in row(index[twin]).items() if k in ("mean_shift_all", "sd_units_all", "n_tokens", "rank_from_bottom")}, "len_word": len(w), "len_twin": len(twin)})

    x100_rows = {cls: [row(index[w]) if w in index else {"word": w, "rank_from_bottom": None} for w in ws] for cls, ws in X100_WORDS.items()}
    out = {
        "run": "x110",
        "table": {"scan_json": str(X109_SCAN_JSON.relative_to(ROOT)), "scan_npz": str(X109_SCAN_NPZ.relative_to(ROOT)), "prompts_sha256": scan["prompts_sha256"], "words_sha256": scan["candidates"]["words_sha256"], "n_candidates": len(cands), "statistic": "mean over the 40 x109 prompts (sets A and B) of the shift summed over the two boundary tokens; ascending"},
        "selection_rule": "rank all candidates by mean shift over the 40 x109 prompts, ascending; walk up from the most negative word and keep the first 10, dropping (1) non-words: not in the wordfreq top-50k and not in the x83 lexicon, (2) proper nouns, (3) inflections/derivatives/respellings sharing a stem with a chosen word (x109_scan_select.share_stem)",
        "stem_suffixes": list(SUFFIXES),
        "chosen": records,
        "drop_log": drops,
        "bottom50": bottom50,
        "x100_words": x100_rows,
        "twin_noun_median_shift_all": median,
        "twin_noun_shifts_all": noun_shift,
        "sd_unit": X106_HARD1K_SD,
    }
    atomic_json(WORDS, out)
    print("bottom 50 of the table:")
    for r in bottom50:
        print(f"  {r['rank_from_bottom']:3d} {r['word']:16s} {r['sd_units_all']:+.2f}sd A {r['mean_shift_A'] / X106_HARD1K_SD:+.2f} B {r['mean_shift_B'] / X106_HARD1K_SD:+.2f} neg {r['frac_prompts_negative']:.2f} tok={r['n_tokens']} wf={int(r['in_wordfreq_top50k'])} zipf={r['zipf']}")
    print("drops:")
    for d in drops:
        print(f"  {d['rank_from_bottom']:4d} {d['word']:16s} {d['rule']}")
    print("chosen:")
    for r in records:
        print(f"  {r['rank_from_bottom']:4d} {r['word']:16s} {r['sd_units_all']:+.2f}sd tok={r['n_tokens']} -> {r['twin']} ({r['twin_sd_units_all']:+.2f}sd, tok={r['twin_n_tokens']})")
    print("x100 words, rank from bottom:", {cls: [(r['word'], r['rank_from_bottom']) for r in rows] for cls, rows in x100_rows.items()})


# ---- top-up of the HP strata outside the x85 cohort ----------------------------------------

X85_COHORT = ROOT / "data" / "runs" / "x85" / "x85_cohort.json"
N_PER_STRATUM = N_PROMPTS // len(HP_STRATA)


def x66_id(value) -> int | None:
    """The x66 evaluation-split id behind an x85 id ('84'), an x110 / x100 key ('p84') or an
    x111 / x112 harmful id ('x83:84'); None for any other id (OR-Bench, XSTest, dolly, ...)."""
    m = re.fullmatch(r"(?:p|x83:)?(\d+)", str(value))
    return int(m.group(1)) if m else None


def exclusion_sources() -> list[dict]:
    """Every prompt set the top-up must not reuse: (path, ids, text sha256s)."""
    specs = [(X85_COHORT, "prompt_sha256", "x85 crossed cohort (the x87 direction's fitting cohort)"), (PROMPTS, "text_sha256", "x100 HP and BP = x110 prompts.json")]
    specs.append((ROOT / "data" / "runs" / "x109" / "prompts.json", "text_sha256", "x109 prompts"))
    for name in ("prompts.json", "prompts_topup.json"):
        path = ROOT / "data" / "runs" / "x111" / name
        if path.exists():
            specs.append((path, "text_sha256", f"x111 {name}"))
    specs.append((ROOT / "data" / "runs" / "x112" / "scan_prompts.json", "text_sha256", "x112 scan prompts"))
    for path in sorted(glob.glob(str(ROOT / "data" / "runs" / "x112" / "*" / "prompts*.json"))):
        specs.append((Path(path), "text_sha256", f"x112 {Path(path).parent.name} {Path(path).name}"))
    out = []
    for path, sha_key, label in specs:
        rows = read_json(path)["prompts"]
        out.append({"source": label, "path": str(path.relative_to(ROOT)), "file_sha256": sha256_file(path), "n_prompts": len(rows), "ids": {x66_id(r["id"]) for r in rows} - {None}, "shas": {r[sha_key] for r in rows}})
    return out


def topup_request() -> None:
    import x100_build_cohort  # tokenizer-free path: hp_units reads the x66 split and the x63 labels

    cohort = read_json(X100_COHORT)
    prompts = read_json(PROMPTS)
    excl = read_json(EXCLUSIONS)
    if excl["x110_prompts"]["prompts_sha256"] != prompts["prompts_sha256"]:
        raise SystemExit("FATAL: x85_exclusions.json was frozen against a different prompts.json")
    units = x100_build_cohort.hp_units()
    x100_hp = cohort["populations"]["HP"]["prompts"]
    if sorted(int(u["id"]) for u in select_hp(units, SEEDS["hp_selection"])) != sorted(int(p["id"]) for p in x100_hp):
        raise SystemExit("FATAL: replaying x100's HP rule does not reproduce x100's 96 prompts")
    x100_ids = {int(p["id"]) for p in x100_hp}
    retained = [p for p in prompts["prompts"] if p["population"] == "HP" and p["id"] not in set(excl["prompt_ids"])]
    sources = exclusion_sources()
    ex_ids = set().union(*(s["ids"] for s in sources))
    ex_sha = set().union(*(s["shas"] for s in sources))
    grouped: dict[str, list[dict]] = {name: [] for name, _, _ in HP_STRATA}
    for u in units:
        name = hp_stratum(float(u["historical_refusal_rate"]))
        if name is not None:
            grouped[name].append(u)
    rng = np.random.default_rng(SEEDS["hp_selection"])
    for name, _, _ in HP_STRATA:  # x100's three draws, replayed on the same generator
        cands = sorted(grouped[name], key=lambda r: int(r["id"]))
        drawn = {int(cands[int(i)]["id"]) for i in rng.choice(len(cands), size=N_PROMPTS // 3, replace=False)}
        if drawn != {int(p["id"]) for p in x100_hp if p["stratum"] == name}:
            raise SystemExit(f"FATAL: replayed draw of stratum {name} differs from x100's")
    strata, rows = {}, []
    for name, _, _ in HP_STRATA:  # the continuation, same generator, same stratum order
        cands = sorted(grouped[name], key=lambda r: int(r["id"]))
        n_retained = sum(1 for p in retained if p["stratum"] == name)
        need = N_PER_STRATUM - n_retained
        rest = [u for u in cands if int(u["id"]) not in x100_ids]
        eligible = [u for u in rest if int(u["id"]) not in ex_ids and sha256_text(u["prompt"]) not in ex_sha]
        blocked = {s["source"]: sum(1 for u in rest if int(u["id"]) in s["ids"] or sha256_text(u["prompt"]) in s["shas"]) for s in sources}
        picked = []
        if need > 0:
            if need > len(eligible):
                raise SystemExit(f"FATAL: stratum {name} has {len(eligible)} eligible prompts for a shortfall of {need}")
            picked = [eligible[int(i)] for i in rng.choice(len(eligible), size=need, replace=False)]
        strata[name] = {"n_candidates": len(cands), "n_x100": len(cands) - len(rest), "n_retained_after_x85": n_retained, "n_needed": need, "n_not_x100": len(rest), "n_blocked_by_source": {k: v for k, v in blocked.items() if v}, "n_eligible": len(eligible), "drawn_ids": [int(u["id"]) for u in picked]}
        for order, u in enumerate(picked):
            rows.append({"id": prompt_key("HP", u), "population": "HP", "x66_id": int(u["id"]), "stratum": name, "historical_refusal_rate": float(u["historical_refusal_rate"]), "text": u["prompt"], "text_sha256": sha256_text(u["prompt"]), "draw_order": order})
        print(f"{name}: {len(cands)} candidates, x100 {len(cands) - len(rest)}, retained {n_retained}, need {need}, eligible {len(eligible)} (blocked {blocked}) -> {[r['id'] for r in rows if r['stratum'] == name]}")
    out = {
        "run": "x110",
        "stage": "topup",
        "population": "HP",
        "rule": "x100's HP rule (x100_common.select_hp): the x66 seed-42 evaluation split, historical refusal2 r_O >= 0.2, strata r0.2-0.5 / r0.6-0.9 / r1.0, 32 per stratum from one numpy default_rng(100), choice(n, 32, replace=False) over the stratum's id-sorted candidates, stratum by stratum. Replayed and asserted equal to x100's 96, then the same generator continues: per stratum with a shortfall (32 minus the prompts left after the x85 exclusion), in the same stratum order, choice(n_eligible, shortfall, replace=False) over the stratum's id-sorted candidates that are neither x100 prompts nor in any exclusion source (by x66 id and by text sha256); prompts are kept in draw order. numpy's choice without replacement uses Floyd's algorithm at these sizes, so a larger first draw would not have extended x100's 32; the continued generator is the rule's next draw. The rule has no fresh-rate check (the stratum is the historical refusal2 rate), so there is no over-draw.",
        "seed": {"generator": "numpy.random.default_rng", "seed": SEEDS["hp_selection"], "numpy": np.__version__, "replayed_x100_draws": [name for name, _, _ in HP_STRATA]},
        "x100_cohort": {"path": str(X100_COHORT.relative_to(ROOT)), "cohort_sha256": cohort["cohort_sha256"]},
        "x85_exclusions": {"file": str(EXCLUSIONS.relative_to(ROOT)), "file_sha256": sha256_file(EXCLUSIONS), "x85_cohort_sha256": excl["x85_cohort"]["cohort_sha256"], "prompt_ids": excl["prompt_ids"], "excluded_by_stratum": excl["excluded_by_stratum"]},
        "exclusion_sources": [{k: v for k, v in s.items() if k not in ("ids", "shas")} for s in sources],
        "n_hp_units": len(units),
        "n_retained_after_x85": len(retained),
        "n_needed": N_PROMPTS - len(retained),
        "overdraw": 0,
        "strata": strata,
        "prompts": rows,
    }
    atomic_json(TOPUP_REQUEST, out)
    print("->", TOPUP_REQUEST)


def topup_prompts() -> None:
    import x100_build_cohort

    req = read_json(TOPUP_REQUEST)
    units = {int(u["id"]): u for u in x100_build_cohort.hp_units()}
    have = {p["id"] for p in read_json(PROMPTS)["prompts"]}
    rows = []
    for r in req["prompts"]:
        u = units[r["x66_id"]]
        if sha256_text(u["prompt"]) != r["text_sha256"] or hp_stratum(float(u["historical_refusal_rate"])) != r["stratum"] or r["id"] in have:
            raise SystemExit(f"FATAL: top-up prompt {r['id']} no longer matches its source or collides with prompts.json")
        rows.append({**{k: r[k] for k in ("id", "population", "x66_id", "stratum", "historical_refusal_rate", "text", "text_sha256", "draw_order")}, "n_words": len(r["text"].split()), "topup_source": "x100 HP rule continued (topup_request.json)"})
    if len(rows) != req["n_needed"]:
        raise SystemExit(f"FATAL: {len(rows)} top-up prompts for a shortfall of {req['n_needed']}")
    out = {
        "run": "x110",
        "stage": "topup",
        "rule": "HP prompts added after the x85 exclusion, one stratum at a time, by x100's HP rule continued (topup_request.json); the same O, E_w and N_w cells, seeds, rendering and judges as prompts.json. These rows carry no x100 rendered-chat sha256 (they were never x100 prompts).",
        "topup": {"request": str(TOPUP_REQUEST.relative_to(ROOT)), "request_sha256": sha256_file(TOPUP_REQUEST), "x85_exclusions": req["x85_exclusions"], "n_retained_after_x85": req["n_retained_after_x85"], "n_needed": req["n_needed"], "overdraw": req["overdraw"], "strata": {k: {kk: v[kk] for kk in ("n_retained_after_x85", "n_needed", "n_eligible", "drawn_ids")} for k, v in req["strata"].items()}},
        "n_prompts": len(rows),
        "prompts_sha256": sha256_text("\n".join(f"{r['id']}\t{r['text_sha256']}" for r in rows)),
        "prompts": rows,
    }
    atomic_json(PROMPTS_TOPUP, out)
    print("->", PROMPTS_TOPUP, [(r["id"], r["stratum"], r["historical_refusal_rate"]) for r in rows])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", action="store_true")
    parser.add_argument("--words", action="store_true")
    parser.add_argument("--topup-request", action="store_true")
    parser.add_argument("--topup-prompts", action="store_true")
    args = parser.parse_args()
    if args.prompts:
        build_prompts()
    if args.words:
        build_words()
    if args.topup_request:
        topup_request()
    if args.topup_prompts:
        topup_prompts()
    if not (args.prompts or args.words or args.topup_request or args.topup_prompts):
        raise SystemExit("--prompts, --words, --topup-request and/or --topup-prompts")


if __name__ == "__main__":
    main()
