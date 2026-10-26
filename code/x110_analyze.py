"""x110 analysis (local, numpy only).

refusal2 (x64 judge) per (prompt, condition); per word, pooled over the ten words, per
population (HP, BP) and per HP stratum: r(O), r(E), r(N), Delta_EN, Delta_EO, Delta_NO with a
prompt-level paired bootstrap; x100's per-prompt r(O) and class contrasts on the same prompts
for comparison; the direction link from the x110 key-word scan (shift of every chosen word and twin on
the prompts it was generated on) and the vocabulary ranking on the prompts the vocabulary scan covers against the x109 benign
table); the hijack readout.

Three prompt sets, same estimators, same bootstrap seed and draw order:
  default      the primary analysis: the prompts listed in x85_exclusions.json (the 15 HP prompts
               that are x85 prompts, the cohort the x87 direction was fit on) are left out of every
               set and the HP strata are refilled by the 15 top-up prompts of prompts_topup.json
               (x100's HP rule continued, x110_select.py --topup-request / --topup-prompts): HP 96
               (32 / 32 / 32 by stratum), BP 96, all 192; the excluded prompts are reported on their
               own as `HP|excluded` and the added ones as `HP|topup`; results.json carries `cohort`
               and `topup` blocks; writes results.json and results.md.
  --no-topup   the exclusion without the top-up (HP 81, BP 96, all 177, HP strata within the 81);
               writes results_pre_topup.json and results_pre_topup.md.
  --full       the original 192 prompts, unfiltered, without the top-up; writes results_full.json
               and results_full.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x109_pilot_analyze import ci, perm_p, spearman  # noqa: E402
from x110_common import (  # noqa: E402
    EXCLUSIONS,
    JUDGE_DIR,
    KEY_JSON,
    KEY_JSON_TOPUP,
    KEY_NPZ,
    KEY_NPZ_TOPUP,
    PROMPTS,
    PROMPTS_TOPUP,
    RESULTS,
    RESULTS_FULL,
    RESULTS_FULL_MD,
    RESULTS_MD,
    RESULTS_PRE_TOPUP,
    RESULTS_PRE_TOPUP_MD,
    SCAN_JSON,
    SCAN_NPZ,
    WORDS,
    X100_COHORT,
    X100_REFUSAL,
    X106_HARD1K_SD,
    X109_SCAN_JSON,
    X109_SCAN_NPZ,
    atomic_json,
    cells,
    mentions,
    read_json,
)
from x110_judge import load_generations  # noqa: E402

N_BOOT = 10_000
BOOT_SEED = 110_000
X100_RESULT = Path(__file__).resolve().parents[1] / "data" / "results" / "x100_lexical_courage.json"
STRATA = ("r0.2-0.5", "r0.6-0.9", "r1.0")


def load_jsonl(path: Path) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if not rec.get("failed"):
                out[rec["cell_id"]] = rec
    return out


def boot_p(samples: np.ndarray) -> float:
    b = len(samples)
    low = (np.sum(samples <= 0) + 1) / (b + 1)
    high = (np.sum(samples >= 0) + 1) / (b + 1)
    return float(min(1.0, 2 * min(low, high)))


def holm(p: list[float]) -> list[float]:
    order = np.argsort(p)
    out = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        out[i] = running
    return out.tolist()


def x100_rates() -> dict[str, dict]:
    """x100's refusal2 per (population, prompt key, condition, word) from its judged rows."""
    cohort = read_json(X100_COHORT)
    by_id = {int(c["behavior_id"]): c for c in cohort["cells"] if c["behavior"]}
    out: dict[str, dict] = {}
    for line in X100_REFUSAL.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("failed"):
            continue
        c = by_id[int(rec["id"])]
        out[c["cell_id"]] = {"population": c["population"], "key": c["prompt_key"], "condition": c["condition"], "class": c["class"], "word": c["word"], "rate": float(np.mean(rec["refusal"]))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="every original prompt, no top-up; writes results_full.json / results_full.md")
    ap.add_argument("--no-topup", action="store_true", help="the x85 exclusion without the top-up prompts; writes results_pre_topup.json / results_pre_topup.md")
    args = ap.parse_args()
    if args.full and args.no_topup:
        raise SystemExit("--full and --no-topup exclude each other")
    use_topup = not (args.full or args.no_topup)
    prompts_obj = read_json(PROMPTS)
    topup_obj = read_json(PROMPTS_TOPUP) if use_topup else None
    if use_topup and topup_obj["topup"]["x85_exclusions"]["prompt_ids"] != read_json(EXCLUSIONS)["prompt_ids"]:
        raise SystemExit("FATAL: prompts_topup.json was drawn against a different exclusion list")
    prompts = prompts_obj["prompts"] + (topup_obj["prompts"] if topup_obj else [])
    n_orig = len(prompts_obj["prompts"])
    topup_ids = {p["id"] for p in topup_obj["prompts"]} if topup_obj else set()
    exclusions = None
    excluded_ids: set[str] = set()
    if not args.full:
        exclusions = read_json(EXCLUSIONS)
        excluded_ids = set(exclusions["prompt_ids"])
        if exclusions["x110_prompts"]["prompts_sha256"] != prompts_obj["prompts_sha256"]:
            raise SystemExit("FATAL: x85_exclusions.json was frozen against a different prompts.json")
        if len(excluded_ids) != exclusions["n_excluded"] or not excluded_ids <= {p["id"] for p in prompts}:
            raise SystemExit("FATAL: x85_exclusions.json ids do not match prompts.json")
    out_json, out_md = (RESULTS_FULL, RESULTS_FULL_MD) if args.full else ((RESULTS_PRE_TOPUP, RESULTS_PRE_TOPUP_MD) if args.no_topup else (RESULTS, RESULTS_MD))
    words = read_json(WORDS)
    chosen = words["chosen"]
    cw = [c["word"] for c in chosen]
    tw = [c["twin"] for c in chosen]
    gens, _, _ = load_generations()
    if topup_obj:
        gens += load_generations("topup")[0]
    gen = {g["cell_id"]: g for g in gens}
    ref = load_jsonl(JUDGE_DIR / "refusal2.jsonl")
    # Hijack label: the no-reasoning pass (hijack_fast.jsonl) covers every E and N cell and is the
    # label used; the reasoning pass (hijack.jsonl), which covered part of the grid before it was
    # stopped for speed, is kept for an agreement check on the cells both judged.
    hij_fast = load_jsonl(JUDGE_DIR / "hijack_fast.jsonl")
    hij_reason = load_jsonl(JUDGE_DIR / "hijack.jsonl")
    hij = hij_fast if hij_fast else hij_reason
    hijack_agreement = None
    both = [cid for cid in hij_fast if cid in hij_reason]
    if both:
        a = np.asarray([v for cid in both for v in hij_fast[cid]["hijack_llm"]], bool)
        b = np.asarray([v for cid in both for v in hij_reason[cid]["hijack_llm"]], bool)
        po = float((a == b).mean())
        pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
        ra = np.asarray([np.mean(hij_fast[cid]["hijack_llm"]) for cid in both])
        rb = np.asarray([np.mean(hij_reason[cid]["hijack_llm"]) for cid in both])
        pooled_reason = {}
        for cid, h in hij_reason.items():
            g = gen.get(cid)
            if g is None:
                continue
            pooled_reason.setdefault(f"{g['kind']}|{g['population']}", []).extend(h["hijack_llm"])
        hijack_agreement = {
            "reasoning_pass_rates_on_its_cells": {k: {"n_responses": len(v), "hijack_llm_rate": float(np.mean(v))} for k, v in sorted(pooled_reason.items())},
            "n_cells_both": len(both),
            "n_responses_both": int(len(a)),
            "response_agreement": po,
            "cohen_kappa": float((po - pe) / (1 - pe)) if pe < 1 else None,
            "rate_no_reasoning": float(a.mean()),
            "rate_reasoning": float(b.mean()),
            "per_cell_rate_pearson": float(np.corrcoef(ra, rb)[0, 1]),
            "per_cell_rate_spearman": spearman(ra, rb),
            "mean_abs_cell_rate_diff": float(np.abs(ra - rb).mean()),
        }
    grid = cells(prompts_obj, words) + (cells(topup_obj, words) if topup_obj else [])
    missing_gen = [c["cell_id"] for c in grid if c["cell_id"] not in gen]
    missing_ref = [c["cell_id"] for c in grid if c["cell_id"] in gen and c["cell_id"] not in ref]
    for cid, rec in ref.items():
        if cid in gen and rec["response_sha256"] != gen[cid]["response_sha256"]:
            raise SystemExit(f"FATAL: refusal2 record {cid} does not match its responses")

    pids = [p["id"] for p in prompts]
    pops = np.asarray([p["population"] for p in prompts])
    strata = np.asarray([p.get("stratum") or "" for p in prompts])
    inc = np.asarray([p["id"] not in excluded_ids for p in prompts])  # prompts in the analysis
    P, W = len(pids), len(cw)

    def rate(cid: str) -> float:
        rec = ref.get(cid)
        return float(np.mean(rec["refusal"])) if rec else float("nan")

    rO = np.asarray([rate(f"{p}|O") for p in pids])
    rE = np.asarray([[rate(f"{p}|E_{w}") for p in pids] for w in cw])  # [W, P]
    rN = np.asarray([[rate(f"{p}|N_{t}") for p in pids] for t in tw])
    dEN, dEO, dNO = rE - rN, rE - rO[None], rN - rO[None]
    rng = np.random.default_rng(BOOT_SEED)
    masks = {"HP": (pops == "HP") & inc, "BP": (pops == "BP") & inc, "all": inc.copy()}
    for s in STRATA:
        masks[f"HP|{s}"] = (pops == "HP") & (strata == s) & inc
    if excluded_ids:
        masks["HP|excluded"] = ~inc  # the x85 prompts on their own; last, so the draw order of the other sets is unchanged
    if topup_ids:
        masks["HP|topup"] = np.asarray([p["id"] in topup_ids for p in prompts])  # the added prompts on their own, after HP|excluded

    def boot(m: np.ndarray, mask: np.ndarray) -> np.ndarray:
        sub = np.where(mask)[0]
        idx = rng.integers(0, len(sub), size=(N_BOOT, len(sub)))
        return np.nanmean(m[..., sub][..., idx], axis=-1)

    r_O = {k: float(np.nanmean(rO[m])) for k, m in masks.items()}
    r_O["by_prompt"] = dict(zip(pids, rO.tolist()))

    per_word = []
    for k, c in enumerate(chosen):
        rec = {"word": c["word"], "twin": c["twin"], "rank_from_bottom": c["rank_from_bottom"], "n_tokens": c["n_tokens"], "table_sd_units": c["sd_units_all"]}
        for s, m in masks.items():
            bEN, bEO, bNO = boot(dEN[k], m), boot(dEO[k], m), boot(dNO[k], m)
            rec[s] = {
                "n_prompts": int(m.sum()),
                "r_E": float(np.nanmean(rE[k][m])),
                "r_N": float(np.nanmean(rN[k][m])),
                "r_O": float(np.nanmean(rO[m])),
                "delta_EN": float(np.nanmean(dEN[k][m])),
                "delta_EN_ci95": ci(bEN),
                "delta_EN_p": boot_p(bEN),
                "delta_EO": float(np.nanmean(dEO[k][m])),
                "delta_EO_ci95": ci(bEO),
                "delta_EO_p": boot_p(bEO),
                "delta_NO": float(np.nanmean(dNO[k][m])),
                "delta_NO_ci95": ci(bNO),
                "n_prompts_E_below_O": int(np.nansum(dEO[k][m] < 0)),
                "n_prompts_E_above_O": int(np.nansum(dEO[k][m] > 0)),
            }
        per_word.append(rec)
    for s in ("HP", "BP"):
        hp = holm([pw[s]["delta_EN_p"] for pw in per_word])
        for pw, v in zip(per_word, hp):
            pw[s]["delta_EN_p_holm10"] = v

    pooled = {}
    for s, m in masks.items():
        row = {"n_prompts": int(m.sum()), "r_O": float(np.nanmean(rO[m])), "r_E_mean": float(np.nanmean(rE[:, m])), "r_N_mean": float(np.nanmean(rN[:, m]))}
        for label, d in (("delta_EN", dEN), ("delta_EO", dEO), ("delta_NO", dNO)):
            b = np.nanmean(boot(d, m), axis=0)
            row[label] = float(np.nanmean(d[:, m]))
            row[label + "_ci95"] = ci(b)
            row[label + "_p"] = boot_p(b)
        pooled[s] = row

    # x100 on the same prompts.
    x100 = x100_rates()
    x100_O = {(v["population"], v["key"]): v["rate"] for v in x100.values() if v["condition"] == "O"}
    rO_x100 = np.asarray([x100_O.get((p["population"], p["id"]), np.nan) for p in prompts])
    x100_res = read_json(X100_RESULT)
    x100_classes = {}
    for pop, section in (("HP", "primary_HP"), ("BP", "secondary_BP")):
        rep = x100_res[section]["report"]
        x100_classes[pop] = {cls: {"delta_WN": v["delta"], "ci95": v["ci95"], "r_W": v["mean_r_W"], "r_N": v["mean_r_N"]} for cls, v in rep["classes"].items()}
        x100_classes[pop]["r_O"] = rep["mean_r_O"]
    x100_cmp = {
        "r_O_x100_vs_x110": {s: {"x100": float(np.nanmean(rO_x100[m])), "x110": float(np.nanmean(rO[m])), "mean_abs_diff_per_prompt": float(np.nanmean(np.abs(rO_x100[m] - rO[m]))), "spearman": spearman(rO_x100[m][np.isfinite(rO[m]) & np.isfinite(rO_x100[m])], rO[m][np.isfinite(rO[m]) & np.isfinite(rO_x100[m])]), "n_prompts_with_x100_O": int(np.isfinite(rO_x100[m]).sum())} for s, m in masks.items() if "|" not in s},
        "x100_classes": x100_classes,
        "x110_pooled_delta_EN": {s: pooled[s]["delta_EN"] for s in ("HP", "BP")},
        "x110_pooled_delta_EN_ci95": {s: pooled[s]["delta_EN_ci95"] for s in ("HP", "BP")},
    }

    # Direction link: the key-word scan (chosen words and twins on every prompt) gives the
    # per-word and per-cell links; the vocabulary scan (x109 candidate list on the prompts it
    # covers) gives the ranking of the vocabulary on the harmful and benign prompts.
    link = None
    if KEY_JSON.exists() and KEY_NPZ.exists():
        key = read_json(KEY_JSON)
        if key["prompt_ids"] != pids[:n_orig] or key["prompt_indices_scanned"] != list(range(n_orig)):
            raise SystemExit("FATAL: key-word scan does not cover the x110 prompts in order")
        kshift = np.load(KEY_NPZ)["shift"].astype(np.float64).sum(axis=2)  # [P, 20]
        o_proj_key = list(key["o_proj_cached"])
        if topup_obj:
            kt = read_json(KEY_JSON_TOPUP)
            if kt["prompt_ids"] != pids[n_orig:] or kt["prompt_indices_scanned"] != list(range(len(pids) - n_orig)) or kt["candidates"]["words"] != key["candidates"]["words"]:
                raise SystemExit("FATAL: top-up key-word scan does not cover prompts_topup.json in order with the same words")
            kshift = np.concatenate([kshift, np.load(KEY_NPZ_TOPUP)["shift"].astype(np.float64).sum(axis=2)], axis=0)
            o_proj_key += list(kt["o_proj_cached"])
        kidx = {w: j for j, w in enumerate(key["candidates"]["words"])}
        link = {"per_set": {}, "vocabulary": None, "key_scan_checks": {"o_cached_vs_full_max_abs_per_token": key["o_cached_vs_full_max_abs_per_token"], "scan_minutes": float(sum(key["seconds_per_prompt"]) / 60), "prefix_tokens_max": int(max(key["prefix_tokens"]))}}
        if topup_obj:
            link["key_scan_checks_topup"] = {"o_cached_vs_full_max_abs_per_token": kt["o_cached_vs_full_max_abs_per_token"], "scan_minutes": float(sum(kt["seconds_per_prompt"]) / 60), "prefix_tokens_max": int(max(kt["prefix_tokens"]))}
        jw, jt = [kidx[w] for w in cw], [kidx[t] for t in tw]
        for s, m in masks.items():
            ws = kshift[m][:, jw].mean(0)
            ts = kshift[m][:, jt].mean(0)
            en = np.asarray([pw[s]["delta_EN"] for pw in per_word])
            eo = np.asarray([pw[s]["delta_EO"] for pw in per_word])
            cs = kshift[m][:, jw].T.ravel()
            ce = rE[:, m].ravel()
            co = np.repeat(rO[m][None], W, axis=0).ravel()
            ok = np.isfinite(ce)
            link["per_set"][s] = {
                "n_words": W,
                "n_prompts": int(m.sum()),
                "word_mean_shift_sd": dict(zip(cw, (ws / X106_HARD1K_SD).tolist())),
                "twin_mean_shift_sd": dict(zip(tw, (ts / X106_HARD1K_SD).tolist())),
                "word_minus_twin_mean_shift_sd": float((ws - ts).mean() / X106_HARD1K_SD),
                "frac_cells_word_shift_below_zero": float((kshift[m][:, jw] < 0).mean()),
                "per_word_spearman_shift_vs_delta_EN": spearman(ws, en),
                "per_word_spearman_shift_vs_delta_EN_perm_p": perm_p(ws, en),
                "per_word_spearman_shift_vs_delta_EO": spearman(ws, eo),
                "per_cell_spearman_shift_vs_r_E": spearman(cs[ok], ce[ok]),
                "per_cell_spearman_shift_vs_r_E_minus_r_O": spearman(cs[ok], (ce - co)[ok]),
                "n_cells": int(ok.sum()),
                "o_proj_mean_sd": float(np.asarray(o_proj_key)[m].mean() / X106_HARD1K_SD),
            }
    if link is not None and SCAN_JSON.exists() and SCAN_NPZ.exists():
        scan = read_json(SCAN_JSON)
        if scan["prompt_ids"] != pids[:n_orig]:
            raise SystemExit("FATAL: x110 scan prompt order differs")
        covered = np.asarray(scan["prompt_indices_scanned"], int)
        shift = np.load(SCAN_NPZ)["shift"].astype(np.float64).sum(axis=2)  # [P_scanned, N]
        cands = scan["candidates"]["words"]
        cidx = {w: j for j, w in enumerate(cands)}
        x109 = read_json(X109_SCAN_JSON)
        if x109["candidates"]["words_sha256"] != scan["candidates"]["words_sha256"]:
            raise SystemExit("FATAL: x110 and x109 candidate lists differ")
        table = np.load(X109_SCAN_NPZ)["shift"].astype(np.float64).sum(axis=2).mean(0)  # [N]
        kcheck = {w: float(np.abs(shift[:, cidx[w]] - kshift[covered, kidx[w]]).max()) for w in cw + tw}
        link["vocabulary"] = {"n_prompts_scanned": int(len(covered)), "key_vs_vocabulary_scan_max_abs_diff_per_word": kcheck}
        for s in ("HP", "BP", "all"):
            m = masks[s][covered]
            if not m.any():
                continue
            mean_s = shift[m].mean(0)
            order = np.argsort(mean_s, kind="stable")
            rank_bottom = np.empty(len(cands), int)
            rank_bottom[order] = np.arange(1, len(cands) + 1)
            bottom_table = set(np.argsort(table, kind="stable")[:100].tolist())
            link["vocabulary"][s] = {
                "n_prompts": int(m.sum()),
                "spearman_table_vs_here_all_words": spearman(table, mean_s),
                "pearson_table_vs_here_all_words": float(np.corrcoef(table, mean_s)[0, 1]),
                "bottom100_overlap_with_table": int(len(bottom_table & set(order[:100].tolist()))),
                "chosen_rank_from_bottom": {w: int(rank_bottom[cidx[w]]) for w in cw},
                "twin_rank_from_bottom": {t: int(rank_bottom[cidx[t]]) for t in tw},
                "bottom20": [{"word": cands[j], "mean_shift_sd": float(mean_s[j] / X106_HARD1K_SD), "table_sd": float(table[j] / X106_HARD1K_SD)} for j in order[:20]],
                "top10": [{"word": cands[j], "mean_shift_sd": float(mean_s[j] / X106_HARD1K_SD), "table_sd": float(table[j] / X106_HARD1K_SD)} for j in order[::-1][:10]],
                "distribution_sd": {"p1": float(np.quantile(mean_s, 0.01) / X106_HARD1K_SD), "median": float(np.median(mean_s) / X106_HARD1K_SD), "p99": float(np.quantile(mean_s, 0.99) / X106_HARD1K_SD), "mean": float(mean_s.mean() / X106_HARD1K_SD), "frac_below_zero": float((mean_s < 0).mean())},
            }
        link["vocabulary_scan_checks"] = {"o_cached_vs_full_max_abs_per_token": scan["o_cached_vs_full_max_abs_per_token"], "scan_minutes": float(sum(scan["seconds_per_prompt"]) / 60), "prefix_tokens_max": int(max(scan["prefix_tokens"]))}

    # Hijack.
    hijack_words = []
    pool: dict = {}
    for c in chosen:
        rec = {"word": c["word"], "twin": c["twin"]}
        for kind, w in (("E", c["word"]), ("N", c["twin"])):
            for s in ("HP", "BP"):
                llm, st, rf = [], [], []
                for p in prompts:
                    if p["population"] != s or p["id"] in excluded_ids:
                        continue
                    cid = f"{p['id']}|{kind}_{w}"
                    g = gen.get(cid)
                    if g is None:
                        continue
                    s_ = [mentions(t, w) for t in g["llm_responses"]]
                    st.extend(s_)
                    h, r = hij.get(cid), ref.get(cid)
                    if h:
                        llm.extend(h["hijack_llm"])
                    if h and r:
                        bucket = pool.setdefault((kind, s), {"llm": [], "str": [], "ref": []})
                        bucket["llm"].extend(h["hijack_llm"])
                        bucket["str"].extend(s_)
                        bucket["ref"].extend(r["refusal"])
                rec[f"{kind}_{s}_mention_rate"] = float(np.mean(st)) if st else None
                rec[f"{kind}_{s}_hijack_llm_rate"] = float(np.mean(llm)) if llm else None
        hijack_words.append(rec)
    hijack_pool = {}
    for (kind, s), b in sorted(pool.items()):
        llm, st, rf = np.asarray(b["llm"], bool), np.asarray(b["str"], bool), np.asarray(b["ref"], float)
        hijack_pool[f"{kind}|{s}"] = {
            "n_responses": int(len(rf)),
            "mention_rate": float(st.mean()) if len(st) else None,
            "hijack_llm_rate": float(llm.mean()) if len(llm) else None,
            "refusal2_given_hijack_llm": float(rf[llm].mean()) if llm.any() else None,
            "refusal2_given_not_hijack_llm": float(rf[~llm].mean()) if (~llm).any() else None,
            "refusal2_given_mention": float(rf[st].mean()) if st.any() else None,
            "refusal2_given_no_mention": float(rf[~st].mean()) if (~st).any() else None,
        }

    prompt_set = None
    if exclusions is not None:
        prompt_set = {
            "exclusions": str(EXCLUSIONS.relative_to(EXCLUSIONS.parents[3])),
            "rule": exclusions["rule"],
            "x85_cohort_sha256": exclusions["x85_cohort"]["file_sha256"],
            "excluded_prompt_ids": sorted(excluded_ids, key=pids.index),
            "n_excluded": len(excluded_ids),
            "n_included": {k: int(m.sum()) for k, m in masks.items() if k in ("HP", "BP", "all")},
        }
    topup_block = None
    if topup_obj:
        prompt_set["n_topup"] = len(topup_ids)
        prompt_set["topup_prompt_ids"] = [p["id"] for p in topup_obj["prompts"]]
        topup_block = {
            "file": str(PROMPTS_TOPUP.relative_to(PROMPTS_TOPUP.parents[3])),
            "rule": topup_obj["rule"],
            "request": topup_obj["topup"],
            "n_added": len(topup_ids),
            "added": [{k: p[k] for k in ("id", "x66_id", "stratum", "historical_refusal_rate", "n_words", "draw_order")} | {"r_O": r_O["by_prompt"][p["id"]]} for p in topup_obj["prompts"]],
            "n_prompts_by_stratum": {s_: int(masks[f"HP|{s_}"].sum()) for s_ in STRATA},
        }
    cohort = (
        "the original 192 prompts, unfiltered, without the top-up (--full)" if args.full
        else ("the original prompts outside the x85 crossed cohort, before the top-up (--no-topup): HP 81, BP 96" if args.no_topup
              else "the original prompts outside the x85 crossed cohort plus the 15 HP top-up prompts: HP 96 (32 per stratum), BP 96")
    )
    results = {
        "run": "x110",
        "construct": "refusal2 (an explicit refusal statement is present), x64 judge, DeepSeek V4 Flash",
        "cohort": cohort,
        **({"prompt_set": prompt_set} if prompt_set else {}),
        **({"topup": topup_block} if topup_block else {}),
        "n_prompts": int(inc.sum()),
        "n_words": W,
        "missing_generations": missing_gen,
        "missing_refusal2": missing_ref,
        "n_refusal2_per_response_fallback": sum(1 for v in ref.values() if v.get("judge_mode") == "per_response_fallback"),
        "r_O": r_O,
        "per_word": per_word,
        "pooled": pooled,
        "x100_comparison": x100_cmp,
        "direction_link": link,
        "hijack_per_word": hijack_words,
        "hijack_pooled": hijack_pool,
        "hijack_coverage": {"label_source": "hijack_fast.jsonl (reasoning disabled)" if hij_fast else "hijack.jsonl (reasoning)", "n_E_N_cells": sum(1 for c in grid if c["kind"] in ("E", "N")), "n_cells_judged": sum(1 for c in grid if c["kind"] in ("E", "N") and c["cell_id"] in hij), "n_cells_reasoning_pass": sum(1 for c in grid if c["kind"] in ("E", "N") and c["cell_id"] in hij_reason)},
        "hijack_agreement_fast_vs_reasoning": hijack_agreement,
        "chosen": chosen,
        "drop_log": words["drop_log"],
        "rate_matrix": {"words": cw, "twins": tw, "prompt_ids": pids, "populations": pops.tolist(), "strata": strata.tolist(), **({"included": inc.tolist()} if prompt_set else {}), **({"topup": [p in topup_ids for p in pids]} if topup_ids else {}), "r_E": rE.tolist(), "r_N": rN.tolist(), "r_O": rO.tolist(), "r_O_x100": rO_x100.tolist()},
        "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED, "unit": "prompts resampled with replacement within the named prompt set, paired across conditions; two-sided bootstrap p"},
    }
    atomic_json(out_json, results)
    write_md(results, out_md)
    print(out_md.read_text())


def f(x, d=3):
    return "nan" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:+.{d}f}"


def cis(c):
    return f"[{f(c[0])}, {f(c[1])}]"


def write_md(r: dict, out_md: Path) -> None:
    ps = r.get("prompt_set")
    tu = r.get("topup")
    L = ["# x110: the x109 table's most negative words on x100's prompts (Qwen3.5-4B)" + ((" — primary analysis, x85 prompts excluded, HP topped up" if tu else " — primary analysis, x85 prompts excluded") if ps else " — every prompt") + "\n"]
    L.append(f"x100's 96 HP and 96 BP prompts; the ten most negative distinct words of the x109 vocabulary table ({', '.join(r['rate_matrix']['words'])}) appended at the end, each against a neutral twin ({', '.join(r['rate_matrix']['twins'])}); ten samples per cell; `refusal2` (x64 judge). CIs: prompt-level paired bootstrap, {r['bootstrap']['n']} draws. Missing generations: {len(r['missing_generations'])}; missing refusal2: {len(r['missing_refusal2'])}; refusal2 rows judged one response per call (x100 fallback): {r['n_refusal2_per_response_fallback']}.\n")
    if tu:
        L.append(f"Prompt set: the {ps['n_excluded']} HP prompts that are x85 prompts (the cohort the x87 direction was fit on; `{ps['exclusions']}`) are excluded from every set and reported on their own as `HP|excluded`; the HP strata are refilled by {tu['n_added']} prompts drawn by x100's HP rule continued (`{tu['file']}`: " + ", ".join(f"{a['id']} {a['stratum']}" for a in tu["added"]) + f"), reported on their own as `HP|topup`; HP {ps['n_included']['HP']} ({' / '.join(str(v) for v in tu['n_prompts_by_stratum'].values())} by stratum), BP {ps['n_included']['BP']}, all {ps['n_included']['all']}. Before the top-up: `results_pre_topup.json` / `.md` (`--no-topup`); the original 192 prompts unfiltered: `results_full.json` / `.md` (`--full`).\n")
    elif ps:
        L.append(f"Prompt set: the {ps['n_excluded']} HP prompts that are x85 prompts (the cohort the x87 direction was fit on; `{ps['exclusions']}`) are excluded from every set and reported on their own as `HP|excluded`; HP {ps['n_included']['HP']}, BP {ps['n_included']['BP']}, all {ps['n_included']['all']}. The full-cohort analysis is `results_full.json` / `results_full.md` (`--full`).\n")
    L.append("## Pooled over the ten words\n")
    L.append("| prompts | n | r(O) | mean r(E) | mean r(N) | Δ_EN [95% CI] | p | Δ_EO [95% CI] | r(N)−r(O) [95% CI] |\n|---|---|---|---|---|---|---|---|---|")
    for k, c in r["pooled"].items():
        L.append(f"| {k} | {c['n_prompts']} | {c['r_O']:.3f} | {c['r_E_mean']:.3f} | {c['r_N_mean']:.3f} | {f(c['delta_EN'])} {cis(c['delta_EN_ci95'])} | {c['delta_EN_p']:.3f} | {f(c['delta_EO'])} {cis(c['delta_EO_ci95'])} | {f(c['delta_NO'])} {cis(c['delta_NO_ci95'])} |")
    x = r["x100_comparison"]
    L.append("\n## x100 on the same prompts\n")
    for s in ("HP", "BP"):
        o = x["r_O_x100_vs_x110"][s]
        L.append(f"- {s} r(O): x100 {o['x100']:.3f}, x110 {o['x110']:.3f}; per-prompt mean |diff| {o['mean_abs_diff_per_prompt']:.3f}, Spearman {o['spearman']:+.2f}.")
    L.append("\n| population | class (x100) | Δ_WN [95% CI] | | x110 ten words Δ_EN [95% CI] |\n|---|---|---|---|---|")
    for s in ("HP", "BP"):
        for cls, v in x["x100_classes"][s].items():
            if cls == "r_O":
                continue
            L.append(f"| {s} | {cls} | {f(v['delta_WN'])} {cis(v['ci95'])} | | {f(x['x110_pooled_delta_EN'][s])} {cis(x['x110_pooled_delta_EN_ci95'][s])} |")
    for s in ("HP", "BP", "HP|r0.2-0.5", "HP|r0.6-0.9", "HP|r1.0") + (("HP|excluded",) if ps else ()) + (("HP|topup",) if tu else ()):
        L.append(f"\n## Per word, {s}\n")
        L.append("| word | twin | table sd | r(O) | r(E) | r(N) | Δ_EN [95% CI] | p | Holm(10) | Δ_EO [95% CI] | r(N)−r(O) | prompts E<O / E>O |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
        for w in sorted(r["per_word"], key=lambda z: z[s]["delta_EN"]):
            z = w[s]
            L.append(f"| {w['word']} | {w['twin']} | {w['table_sd_units']:+.2f} | {z['r_O']:.3f} | {z['r_E']:.3f} | {z['r_N']:.3f} | {f(z['delta_EN'])} {cis(z['delta_EN_ci95'])} | {z['delta_EN_p']:.3f} | {z.get('delta_EN_p_holm10', float('nan')):.3f} | {f(z['delta_EO'])} {cis(z['delta_EO_ci95'])} | {f(z['delta_NO'])} | {z['n_prompts_E_below_O']} / {z['n_prompts_E_above_O']} |")
    if r["direction_link"]:
        d = r["direction_link"]
        L.append(f"\n## Direction link (key-word scan: the ten words and twins on the {d['per_set']['all']['n_prompts']} prompts of the analysis)\n")
        for s in ("HP", "BP", "all") + (("HP|excluded",) if ps else ()) + (("HP|topup",) if tu else ()):
            e = d["per_set"][s]
            L.append(f"- {s} (n = {e['n_prompts']} prompts): mean shift of the words {np.mean(list(e['word_mean_shift_sd'].values())):+.2f} sd, of the twins {np.mean(list(e['twin_mean_shift_sd'].values())):+.2f} sd, {e['frac_cells_word_shift_below_zero']:.3f} of word cells below zero; per word (n = {e['n_words']}) ρ(mean shift, Δ_EN) = {f(e['per_word_spearman_shift_vs_delta_EN'], 2)} (permutation p = {e['per_word_spearman_shift_vs_delta_EN_perm_p']:.3f}), ρ(shift, Δ_EO) = {f(e['per_word_spearman_shift_vs_delta_EO'], 2)}; per cell (n = {e['n_cells']}) ρ(shift, r(E)) = {f(e['per_cell_spearman_shift_vs_r_E'], 2)}, ρ(shift, r(E) − r(O)) = {f(e['per_cell_spearman_shift_vs_r_E_minus_r_O'], 2)}; mean O projection {e['o_proj_mean_sd']:+.2f} sd.")
        voc = d.get("vocabulary")
        L.append("\n| word | table sd | HP shift sd | BP shift sd | twin | HP twin sd | BP twin sd | HP rank from bottom | BP rank from bottom |\n|---|---|---|---|---|---|---|---|---|")
        for w, t, c in zip(r["rate_matrix"]["words"], r["rate_matrix"]["twins"], r["chosen"]):
            rk = [str(voc[s]["chosen_rank_from_bottom"][w]) if voc and s in voc else "–" for s in ("HP", "BP")]
            L.append(f"| {w} | {c['sd_units_all']:+.2f} | {d['per_set']['HP']['word_mean_shift_sd'][w]:+.2f} | {d['per_set']['BP']['word_mean_shift_sd'][w]:+.2f} | {t} | {d['per_set']['HP']['twin_mean_shift_sd'][t]:+.2f} | {d['per_set']['BP']['twin_mean_shift_sd'][t]:+.2f} | {rk[0]} | {rk[1]} |")
        if voc:
            L.append(f"\n### Vocabulary ranking (x109 candidate list on {voc['n_prompts_scanned']} of the 192 prompts)\n")
            L.append(f"Key-word scan and vocabulary scan agree on the shared cells to {max(voc['key_vs_vocabulary_scan_max_abs_diff_per_word'].values()):.4f} (max abs difference, raw units).")
            for s in ("HP", "BP"):
                if s not in voc:
                    continue
                v = voc[s]
                L.append(f"\n{s} (n = {v['n_prompts']} prompts): Spearman with the benign table over all words {v['spearman_table_vs_here_all_words']:+.3f} (Pearson {v['pearson_table_vs_here_all_words']:+.3f}); bottom-100 overlap {v['bottom100_overlap_with_table']}; distribution in sd: p1 {v['distribution_sd']['p1']:+.2f}, median {v['distribution_sd']['median']:+.2f}, p99 {v['distribution_sd']['p99']:+.2f}, {v['distribution_sd']['frac_below_zero']:.3f} below zero.")
                L.append("Bottom 20 here: " + ", ".join(f"{b['word']} ({b['mean_shift_sd']:+.2f}; table {b['table_sd']:+.2f})" for b in v["bottom20"]) + ".")
                L.append("Top 10 here: " + ", ".join(f"{b['word']} ({b['mean_shift_sd']:+.2f})" for b in v["top10"]) + ".")
    L.append("\n## Hijack readout\n")
    hc = r["hijack_coverage"]
    L.append(f"LLM hijack labels ({hc['label_source']}) cover {hc['n_cells_judged']} of the {hc['n_E_N_cells']} E and N cells; the reasoning pass covered {hc['n_cells_reasoning_pass']}.")
    ag = r.get("hijack_agreement_fast_vs_reasoning")
    if ag:
        L.append("Reasoning pass on its own cells: " + "; ".join(f"{k} {v['hijack_llm_rate']:.3f} ({v['n_responses']} responses)" for k, v in ag["reasoning_pass_rates_on_its_cells"].items()) + ".")
        L.append(f"On the {ag['n_cells_both']} cells both passes judged ({ag['n_responses_both']} responses): response-level agreement {ag['response_agreement']:.3f} (Cohen's kappa {f(ag['cohen_kappa'])}), hijack rate {ag['rate_no_reasoning']:.3f} without reasoning vs {ag['rate_reasoning']:.3f} with; per-cell rates Pearson {ag['per_cell_rate_pearson']:+.3f}, Spearman {f(ag['per_cell_rate_spearman'])}, mean absolute difference {ag['mean_abs_cell_rate_diff']:.3f}.")
    L.append("")
    for k, h in r["hijack_pooled"].items():
        L.append(f"- {k} ({h['n_responses']} responses): hijack (LLM) {f(h['hijack_llm_rate'])}, mention {f(h['mention_rate'])}; refusal2 given hijack {f(h['refusal2_given_hijack_llm'])}, given no hijack {f(h['refusal2_given_not_hijack_llm'])}.")
    L.append("\n| word | E HP hijack | E BP hijack | twin | N HP hijack | N BP hijack |\n|---|---|---|---|---|---|")
    for h in r["hijack_per_word"]:
        L.append(f"| {h['word']} | {f(h['E_HP_hijack_llm_rate'])} | {f(h['E_BP_hijack_llm_rate'])} | {h['twin']} | {f(h['N_HP_hijack_llm_rate'])} | {f(h['N_BP_hijack_llm_rate'])} |")
    out_md.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
