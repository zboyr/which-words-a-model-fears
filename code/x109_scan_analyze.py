"""Vocabulary scan, analysis (local, numpy only).

refusal2 (x64 judge) per (prompt, condition); per word and prompt set, r(E), r(N), Delta_EN and
Delta_EO with a prompt-level paired bootstrap; pooled contrasts per set and shape next to the
benign insertion pilot; the link between the scanned projection shift and refusal2 on the
held-out set B and over the union of the pilot's and the new words on set A; the top-50
ranking; the twin check and the hijack readout. Writes results.json and results.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x109_pilot_analyze import ci, perm_p, spearman  # noqa: E402
from x109_scan_common import (  # noqa: E402
    JUDGE_DIR,
    PILOT,
    PROMPTS,
    RESULTS,
    RESULTS_MD,
    SCAN_JSON,
    SCAN_NPZ,
    WORDS,
    X106_HARD1K_SD,
    atomic_json,
    cells,
    mentions,
    read_json,
)
from x109_scan_judge import load_generations  # noqa: E402

N_BOOT = 10_000
BOOT_SEED = 20_261_009


def load_jsonl(path: Path) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if not rec.get("failed"):
                out[rec["cell_id"]] = rec
    return out


def main() -> None:
    prompts_obj = read_json(PROMPTS)
    prompts = prompts_obj["prompts"]
    words = read_json(WORDS)
    scan = read_json(SCAN_JSON)
    shift_tok = np.load(SCAN_NPZ)["shift"].astype(np.float64)
    shift_all = shift_tok.sum(axis=2)  # [P, N]
    cand_index = {w: j for j, w in enumerate(scan["candidates"]["words"])}
    gens, _, _ = load_generations()
    gen = {g["cell_id"]: g for g in gens}
    ref = load_jsonl(JUDGE_DIR / "refusal2.jsonl")
    hij = load_jsonl(JUDGE_DIR / "hijack.jsonl")
    grid = cells(prompts_obj, words)
    missing_gen = [c["cell_id"] for c in grid if c["cell_id"] not in gen]
    missing_ref = [c["cell_id"] for c in grid if c["cell_id"] in gen and c["cell_id"] not in ref]
    for cid, rec in ref.items():
        if cid in gen and rec["response_sha256"] != gen[cid]["response_sha256"]:
            raise SystemExit(f"FATAL: refusal2 record {cid} does not match its responses")
    if scan["prompt_ids"] != [p["id"] for p in prompts]:
        raise SystemExit("FATAL: scan prompt order differs")

    pids = [p["id"] for p in prompts]
    sets = np.asarray([p["set"] for p in prompts])
    shapes = np.asarray([p["shape"] for p in prompts])
    chosen = words["chosen"]
    cw = [c["word"] for c in chosen]

    def rate(cid: str) -> float:
        rec = ref.get(cid)
        return float(np.mean(rec["refusal"])) if rec else float("nan")

    rO = np.asarray([rate(f"{p}|O") for p in pids])
    rE = np.asarray([[rate(f"{p}|E_{c['word']}") for p in pids] for c in chosen])  # [W, P]
    rN = np.asarray([[rate(f"{p}|N_{c['twin']}") for p in pids] for c in chosen])
    dEN, dEO, dNO = rE - rN, rE - rO[None], rN - rO[None]
    W, P = rE.shape
    rng = np.random.default_rng(BOOT_SEED)
    masks = {"A": sets == "A", "B": sets == "B", "all": np.ones(P, bool)}
    shape_masks = {"shape_A": shapes == "A", "shape_B": shapes == "B"}

    def boot(m: np.ndarray, mask: np.ndarray) -> np.ndarray:
        sub = np.where(mask)[0]
        idx = rng.integers(0, len(sub), size=(N_BOOT, len(sub)))
        return np.nanmean(m[..., sub][..., idx], axis=-1)

    r_O = {}
    for s, m in masks.items():
        r_O[s] = {"all": float(np.nanmean(rO[m]))}
        for sh, sm in shape_masks.items():
            r_O[s][sh] = float(np.nanmean(rO[m & sm]))
    r_O["by_prompt"] = dict(zip(pids, rO.tolist()))

    per_word = []
    for k, c in enumerate(chosen):
        j, jt = cand_index[c["word"]], cand_index[c["twin"]]
        rec = {"word": c["word"], "twin": c["twin"], "rank_A": c["rank_A"], "n_tokens": c["n_tokens"], "in_x83_lexicon": c["in_x83_lexicon"]}
        for s, m in masks.items():
            bEN, bEO, bNO = boot(dEN[k], m), boot(dEO[k], m), boot(dNO[k], m)
            rec[s] = {
                "n_prompts": int(m.sum()),
                "r_E": float(np.nanmean(rE[k][m])),
                "r_N": float(np.nanmean(rN[k][m])),
                "delta_EN": float(np.nanmean(dEN[k][m])),
                "delta_EN_ci95": ci(bEN),
                "delta_EO": float(np.nanmean(dEO[k][m])),
                "delta_EO_ci95": ci(bEO),
                "delta_NO": float(np.nanmean(dNO[k][m])),
                "delta_NO_ci95": ci(bNO),
                "n_prompts_r_E_gt_0": int(np.sum(rE[k][m] > 0)),
                "mean_shift": float(shift_all[m, j].mean()),
                "mean_shift_sd_units": float(shift_all[m, j].mean() / X106_HARD1K_SD),
                "twin_mean_shift": float(shift_all[m, jt].mean()),
            }
        per_word.append(rec)

    pooled = {}
    for s, m in masks.items():
        for sh, sm in [("all", np.ones(P, bool)), *shape_masks.items()]:
            mm = m & sm
            row = {"n_prompts": int(mm.sum()), "r_O": float(np.nanmean(rO[mm])), "r_E_mean": float(np.nanmean(rE[:, mm])), "r_N_mean": float(np.nanmean(rN[:, mm]))}
            for label, d in (("delta_EN", dEN), ("delta_EO", dEO), ("delta_NO", dNO)):
                row[label] = float(np.nanmean(d[:, mm]))
                row[label + "_ci95"] = ci(np.nanmean(boot(d, mm), axis=0))
            pooled[f"{s}|{sh}"] = row

    pilot_res = read_json(PILOT / "results.json")
    pilot_pw = {r["word"]: r for r in pilot_res["per_word"]}
    pilot_cmp = {
        "pilot_pooled_delta_EN_set_A_15_words": pilot_res["pooled"]["all"]["delta_EN"],
        "pilot_pooled_delta_EN_ci95": pilot_res["pooled"]["all"]["delta_EN_ci95"],
        "pilot_pooled_delta_EO": pilot_res["pooled"]["all"]["delta_EO"],
        "pilot_r_O_set_A": pilot_res["r_O"]["all"],
        "pilot_words": {w: {"delta_EN": r["delta_EN"], "delta_EN_ci95": r["delta_EN_ci95"], "r_E": r["r_E"]} for w, r in pilot_pw.items()},
        "new_pooled_delta_EN_set_A_10_words": pooled["A|all"]["delta_EN"],
        "new_pooled_delta_EN_set_A_ci95": pooled["A|all"]["delta_EN_ci95"],
        "new_r_O_set_A": pooled["A|all"]["r_O"],
        "words_in_both": [w for w in cw if w in pilot_pw],
        "words_in_both_new_set_A_delta_EN": {w: per_word[cw.index(w)]["A"]["delta_EN"] for w in cw if w in pilot_pw},
    }

    # Direction link.
    jw = [cand_index[w] for w in cw]
    link = {}
    for s in ("B", "A", "all"):
        m = masks[s]
        ws = shift_all[m][:, jw].mean(0)
        en = np.asarray([pw[s]["delta_EN"] for pw in per_word])
        eo = np.asarray([pw[s]["delta_EO"] for pw in per_word])
        cs = shift_all[m][:, jw].T.ravel()  # [W * Pm]
        ce = rE[:, m].ravel()
        ok = np.isfinite(ce)
        link[s] = {
            "per_word_spearman_shift_vs_delta_EN": spearman(ws, en),
            "per_word_spearman_shift_vs_delta_EN_perm_p": perm_p(ws, en),
            "per_word_spearman_shift_vs_delta_EO": spearman(ws, eo),
            "n_words": W,
            "per_cell_spearman_shift_vs_r_E": spearman(cs[ok], ce[ok]),
            "per_cell_spearman_shift_vs_r_E_perm_p": perm_p(cs[ok], ce[ok], n=2000),
            "n_cells": int(ok.sum()),
        }
    # Union over the pilot's 15 words and the new 10 on set A, all shifts from the scan on set A.
    mA = masks["A"]
    union_rows = []
    for w, r in pilot_pw.items():
        if w in cw:
            continue
        union_rows.append({"word": w, "source": "pilot", "delta_EN_A": r["delta_EN"], "mean_shift_A": float(shift_all[mA, cand_index[w]].mean())})
    for k, w in enumerate(cw):
        union_rows.append({"word": w, "source": "x109" + ("(also pilot)" if w in pilot_pw else ""), "delta_EN_A": per_word[k]["A"]["delta_EN"], "mean_shift_A": float(shift_all[mA, cand_index[w]].mean())})
    us = np.asarray([u["mean_shift_A"] for u in union_rows])
    ue = np.asarray([u["delta_EN_A"] for u in union_rows])
    link["union_set_A"] = {"n_words": len(union_rows), "rule": "pilot's 15 words (pilot Delta_EN) plus the new 10 (this run's set-A Delta_EN; a word in both uses this run's value); shift = scan mean over set A", "per_word_spearman_shift_vs_delta_EN": spearman(us, ue), "perm_p": perm_p(us, ue), "rows": union_rows}

    # Twin check.
    twin = {"per_word": [{"word": pw["word"], "twin": pw["twin"], **{s: {"delta_NO": pw[s]["delta_NO"], "delta_NO_ci95": pw[s]["delta_NO_ci95"]} for s in masks}} for pw in per_word], "pooled": {s: {"delta_NO": pooled[f"{s}|all"]["delta_NO"], "delta_NO_ci95": pooled[f"{s}|all"]["delta_NO_ci95"]} for s in masks}}

    # Hijack.
    hijack_words = []
    pool = {"E": {"llm": [], "str": [], "ref": []}, "N": {"llm": [], "str": [], "ref": []}}
    for c in chosen:
        rec = {"word": c["word"], "twin": c["twin"]}
        for kind, w in (("E", c["word"]), ("N", c["twin"])):
            llm, st = [], []
            for p in pids:
                cid = f"{p}|{kind}_{w}"
                g = gen.get(cid)
                if g is None:
                    continue
                s_ = [mentions(t, w) for t in g["llm_responses"]]
                st.extend(s_)
                h, r = hij.get(cid), ref.get(cid)
                if h:
                    llm.extend(h["hijack_llm"])
                if h and r:
                    pool[kind]["llm"].extend(h["hijack_llm"])
                    pool[kind]["str"].extend(s_)
                    pool[kind]["ref"].extend(r["refusal"])
            rec[f"{kind}_mention_rate"] = float(np.mean(st)) if st else None
            rec[f"{kind}_hijack_llm_rate"] = float(np.mean(llm)) if llm else None
        hijack_words.append(rec)
    hijack_pool = {}
    for kind in ("E", "N"):
        llm = np.asarray(pool[kind]["llm"], bool)
        st = np.asarray(pool[kind]["str"], bool)
        rf = np.asarray(pool[kind]["ref"], float)
        hijack_pool[kind] = {
            "n_responses": int(len(rf)),
            "mention_rate": float(st.mean()) if len(st) else None,
            "hijack_llm_rate": float(llm.mean()) if len(llm) else None,
            "refusal2_given_hijack_llm": float(rf[llm].mean()) if llm.any() else None,
            "refusal2_given_not_hijack_llm": float(rf[~llm].mean()) if (~llm).any() else None,
            "refusal2_given_mention": float(rf[st].mean()) if st.any() else None,
            "refusal2_given_no_mention": float(rf[~st].mean()) if (~st).any() else None,
            "n_refusals": int((rf == 1).sum()),
            "share_of_refusals_with_hijack_llm": float(llm[rf == 1].mean()) if (rf == 1).any() else None,
        }

    top50 = words["top50_set_A"]
    ranking = {
        "n_top50_in_x83_lexicon": int(sum(r["in_x83_lexicon"] for r in top50)),
        "top50_words": [r["word"] for r in top50],
        "top50_lexicon_words": [r["word"] for r in top50 if r["in_x83_lexicon"]],
        "top50_not_in_wordfreq_or_lexicon": [r["word"] for r in top50 if not r["in_wordfreq_top50k"] and not r["in_x83_lexicon"]],
        "named_ranks": words["named_ranks"],
    }

    results = {
        "run": "x109",
        "construct": "refusal2 (an explicit refusal statement is present), x64 judge, DeepSeek V4 Flash",
        "n_prompts": P,
        "n_words": W,
        "missing_generations": missing_gen,
        "missing_refusal2": missing_ref,
        "r_O": r_O,
        "per_word": per_word,
        "pooled": pooled,
        "pilot_comparison": pilot_cmp,
        "direction_link": link,
        "twin_check": twin,
        "hijack_per_word": hijack_words,
        "hijack_pooled": hijack_pool,
        "hijack_coverage": {"n_E_N_cells": sum(1 for c in grid if c["kind"] in ("E", "N")), "n_cells_judged": sum(1 for c in grid if c["kind"] in ("E", "N") and c["cell_id"] in hij), "missing": [c["cell_id"] for c in grid if c["kind"] in ("E", "N") and c["cell_id"] not in hij]},
        "ranking": ranking,
        "chosen": chosen,
        "drop_log": words["drop_log"],
        "scan_checks": {k: scan[k] for k in ("scale_check_vs_x106_stage0a", "vs_pilot_projections", "o_cached_vs_full_max_abs_per_token")} | {"site_audit_status": scan["site_audit"]["status"], "junction": {k: scan["junction"][k] for k in ("n_checked", "n_fail")}, "n_candidates": scan["candidates"]["n"], "extras_added": scan["candidates"]["extras_added"], "scan_seconds": sum(scan["seconds_per_prompt"])},
        "rate_matrix": {"words": cw, "twins": [c["twin"] for c in chosen], "prompt_ids": pids, "r_E": rE.tolist(), "r_N": rN.tolist(), "r_O": rO.tolist()},
        "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED, "unit": "prompts resampled with replacement within the named prompt set, paired across conditions"},
    }
    atomic_json(RESULTS, results)
    write_md(results, words)
    print(RESULTS_MD.read_text())


def f(x, d=3):
    return "nan" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:+.{d}f}"


def cis(c):
    return f"[{f(c[0])}, {f(c[1])}]"


def write_md(r: dict, words: dict) -> None:
    L = ["# Vocabulary scan (Qwen3.5-4B)\n"]
    sc = r["scan_checks"]
    L.append(f"40 source-clean benign prompts: set A = the benign insertion pilot's 20, set B = 20 new held-out prompts drawn by the same rules (seed 20261009). Every one of {sc['n_candidates']:,} lowercase candidate words was appended to every prompt and read on the boundary direction (cached-prefix reader). The ten words with the largest mean set-A shift (after the drop rules) and their neutral twins were generated (10 samples per cell) and judged on all 40 prompts. Refusal is `refusal2` (x64 judge). CIs: prompt-level paired bootstrap, {r['bootstrap']['n']} draws.\n")
    pb, pa = r["pooled"]["B|all"], r["pooled"]["A|all"]
    lb = r["direction_link"]["B"]
    top = max(r["per_word"], key=lambda x: x["B"]["delta_EN"])
    hp = r["hijack_pooled"]["E"]
    L.append("## Findings\n")
    L.append(
        f"The scan ranks {', '.join(w['word'] for w in r['chosen'])} as the ten distinct words that move the Qwen3.5-4B boundary projection most when appended to set A. "
        f"On the held-out set B, r(O) = {pb['r_O']:.3f}; the ten words give a mean r(E) of {pb['r_E_mean']:.3f} against {pb['r_N_mean']:.3f} for their twins, pooled Δ_EN = {f(pb['delta_EN'])} {cis(pb['delta_EN_ci95'])}. "
        f"On set A the same words give Δ_EN = {f(pa['delta_EN'])} {cis(pa['delta_EN_ci95'])}, against the pilot's {f(r['pilot_comparison']['pilot_pooled_delta_EN_set_A_15_words'])} for its 15 words. "
        f"The largest set-B word is {top['word']} (r(E) = {top['B']['r_E']:.3f}, Δ_EN = {f(top['B']['delta_EN'])} {cis(top['B']['delta_EN_ci95'])}). "
        f"On set B the per-word Spearman ρ between the scanned shift and Δ_EN is {f(lb['per_word_spearman_shift_vs_delta_EN'], 2)} (n = 10) and the per-cell ρ between shift and r(E) is {f(lb['per_cell_spearman_shift_vs_r_E'], 2)} (n = {lb['n_cells']}); "
        f"over the union of the pilot's and the new words on set A the per-word ρ is {f(r['direction_link']['union_set_A']['per_word_spearman_shift_vs_delta_EN'], 2)} (n = {r['direction_link']['union_set_A']['n_words']}). "
        f"Among E responses, refusal2 is {hp['refusal2_given_hijack_llm']:.3f} when the hijack judge says the response treats the appended word as part of the request and {hp['refusal2_given_not_hijack_llm']:.3f} otherwise; {hp['share_of_refusals_with_hijack_llm']:.3f} of refused E responses carry the hijack label.\n"
    )
    o = r["r_O"]
    L.append("## r(O)\n")
    L.append("| set | all | shape A (Dolly) | shape B (No Robots) |\n|---|---|---|---|")
    for s in ("A", "B", "all"):
        L.append(f"| {s} | {o[s]['all']:.3f} | {o[s]['shape_A']:.3f} | {o[s]['shape_B']:.3f} |")
    nz = {k: v for k, v in o["by_prompt"].items() if v > 0}
    L.append(f"\nPrompts with r(O) > 0: {', '.join(f'{k} {v:.1f}' for k, v in nz.items()) or 'none'}.")
    L.append("\n## Chosen words\n")
    L.append("| rank (set A) | word | tokens | lexicon | mean shift A | sd units | twin | twin shift A | twin rule |\n|---|---|---|---|---|---|---|---|---|")
    for c in r["chosen"]:
        L.append(f"| {c['rank_A']} | {c['word']} | {c['n_tokens']} | {'yes' if c['in_x83_lexicon'] else 'no'} | {c['mean_shift_A']:+.2f} | {c['sd_units_A']:+.2f} | {c['twin']} | {c['twin_mean_shift_A']:+.2f} | {c['twin_rule'][:22]} |")
    L.append("\nDrop log:\n")
    for d in r["drop_log"]:
        L.append(f"- rank {d['rank_A']} {d['word']}: {d['rule']}")
    for s in ("B", "A", "all"):
        L.append(f"\n## Per word, set {s}{' (held-out)' if s == 'B' else ''}\n")
        L.append("| word | twin | r(E) | r(N) | Δ_EN [95% CI] | Δ_EO [95% CI] | r(N)−r(O) | prompts with r(E)>0 | shift (sd) |\n|---|---|---|---|---|---|---|---|---|")
        for w in sorted(r["per_word"], key=lambda x: -x[s]["delta_EN"]):
            x = w[s]
            L.append(f"| {w['word']} | {w['twin']} | {x['r_E']:.3f} | {x['r_N']:.3f} | {f(x['delta_EN'])} {cis(x['delta_EN_ci95'])} | {f(x['delta_EO'])} {cis(x['delta_EO_ci95'])} | {f(x['delta_NO'])} | {x['n_prompts_r_E_gt_0']}/{x['n_prompts']} | {x['mean_shift_sd_units']:+.2f} |")
    L.append("\n## Pooled over the ten words\n")
    L.append("| prompts | n | r(O) | mean r(E) | mean r(N) | Δ_EN [95% CI] | Δ_EO [95% CI] | r(N)−r(O) [95% CI] |\n|---|---|---|---|---|---|---|---|")
    for k, c in r["pooled"].items():
        L.append(f"| {k} | {c['n_prompts']} | {c['r_O']:.3f} | {c['r_E_mean']:.3f} | {c['r_N_mean']:.3f} | {f(c['delta_EN'])} {cis(c['delta_EN_ci95'])} | {f(c['delta_EO'])} {cis(c['delta_EO_ci95'])} | {f(c['delta_NO'])} {cis(c['delta_NO_ci95'])} |")
    pc = r["pilot_comparison"]
    L.append(f"\nPilot on set A (15 lexicon-union words): pooled Δ_EN {f(pc['pilot_pooled_delta_EN_set_A_15_words'])} {cis(pc['pilot_pooled_delta_EN_ci95'])}, pedophile {f(pc['pilot_words']['pedophile']['delta_EN'])}, racist {f(pc['pilot_words']['racist']['delta_EN'])}. This run on set A (10 scan words): pooled Δ_EN {f(pc['new_pooled_delta_EN_set_A_10_words'])} {cis(pc['new_pooled_delta_EN_set_A_ci95'])}. Words in both runs: {', '.join(f'{w} pilot {f(pc['pilot_words'][w]['delta_EN'])} / here {f(v)}' for w, v in pc['words_in_both_new_set_A_delta_EN'].items()) or 'none'}.\n")
    L.append("## Direction link\n")
    for s in ("B", "A", "all"):
        d = r["direction_link"][s]
        L.append(f"- Set {s}: per word (n = {d['n_words']}) ρ(mean shift, Δ_EN) = {f(d['per_word_spearman_shift_vs_delta_EN'], 2)} (permutation p = {d['per_word_spearman_shift_vs_delta_EN_perm_p']:.3f}), ρ(shift, Δ_EO) = {f(d['per_word_spearman_shift_vs_delta_EO'], 2)}; per cell (n = {d['n_cells']}) ρ(shift, r(E)) = {f(d['per_cell_spearman_shift_vs_r_E'], 2)} (p = {d['per_cell_spearman_shift_vs_r_E_perm_p']:.3f}).")
    u = r["direction_link"]["union_set_A"]
    L.append(f"- Union on set A (pilot 15 + new 10, n = {u['n_words']} distinct words): ρ(mean scanned shift, Δ_EN) = {f(u['per_word_spearman_shift_vs_delta_EN'], 2)} (p = {u['perm_p']:.3f}).\n")
    L.append("## Hijack readout\n")
    hc = r["hijack_coverage"]
    L.append(f"LLM hijack labels cover {hc['n_cells_judged']} of the {hc['n_E_N_cells']} E and N cells; not labelled: {', '.join(hc['missing']) or 'none'}. String labels cover all cells.\n")
    L.append("| word | E mention | E hijack (LLM) | twin | N mention | N hijack (LLM) |\n|---|---|---|---|---|---|")
    for h in r["hijack_per_word"]:
        L.append(f"| {h['word']} | {h['E_mention_rate']:.3f} | {h['E_hijack_llm_rate']:.3f} | {h['twin']} | {h['N_mention_rate']:.3f} | {h['N_hijack_llm_rate']:.3f} |")
    L.append("")
    for kind, h in r["hijack_pooled"].items():
        L.append(f"- {kind} ({h['n_responses']} responses, {h['n_refusals']} refused): hijack (LLM) {h['hijack_llm_rate']:.3f}, mention {h['mention_rate']:.3f}; refusal2 given hijack {h['refusal2_given_hijack_llm']:.3f}, given no hijack {h['refusal2_given_not_hijack_llm']:.3f}; refusal2 given mention {h['refusal2_given_mention']:.3f}, given no mention {h['refusal2_given_no_mention']:.3f}; share of refusals labelled hijack {h['share_of_refusals_with_hijack_llm']:.3f}.")
    rk = r["ranking"]
    L.append("\n## Top 50 of the set-A ranking\n")
    L.append(f"{rk['n_top50_in_x83_lexicon']} of the top 50 are x83 lexicon words ({', '.join(rk['top50_lexicon_words']) or 'none'}). pedophile ranks {rk['named_ranks'].get('pedophile', {}).get('rank_A')}, racist ranks {rk['named_ranks'].get('racist', {}).get('rank_A')}. Candidates in the top 50 that are in neither the wordfreq top-50k nor the lexicon (tokenizer-only strings): {', '.join(rk['top50_not_in_wordfreq_or_lexicon']) or 'none'}.\n")
    L.append("| rank | word | mean shift A | sd units | mean shift B | tokens | lexicon | zipf |\n|---|---|---|---|---|---|---|---|")
    for t in words["top50_set_A"]:
        z = "" if t["zipf"] is None else f"{t['zipf']:.2f}"
        L.append(f"| {t['rank_A']} | {t['word']} | {t['mean_shift_A']:+.2f} | {t['sd_units_A']:+.2f} | {t['mean_shift_B']:+.2f} | {t['n_tokens']} | {'yes' if t['in_x83_lexicon'] else ''} | {z} |")
    L.append("\nTop 10 by set-B mean shift: " + ", ".join(f"{t['word']} ({t['mean_shift_B']:+.1f})" for t in words["top10_set_B"]) + ".")
    L.append("Top 10 by all-40 mean shift: " + ", ".join(f"{t['word']} ({t['mean_shift_all']:+.1f})" for t in words["top10_all"]) + ".\n")
    L.append("## Scan checks\n")
    L.append(f"- Site audit {sc['site_audit_status']}; junction ids check {sc['junction']['n_checked']} renders, {sc['junction']['n_fail']} failures; scale check vs x106 Stage 0a: " + "; ".join(f"{s['id']} {s['proj_here']:.2f} vs {s['proj_x106_stage0a']:.2f}" for s in sc["scale_check_vs_x106_stage0a"]) + ".")
    v = sc["vs_pilot_projections"]
    L.append(f"- Against the pilot's batch-1 full-prefill shifts ({v['n_prompts']} prompts × {v['n_words']} words): per-cell Spearman {v['spearman_per_cell']:.4f}, word-mean Spearman {v['spearman_word_means']:.4f}, max abs diff {v['max_abs_diff_summed_shift']:.3f}, mean abs diff {v['mean_abs_diff_summed_shift']:.3f}.")
    L.append(f"- Scan GPU time {sc['scan_seconds'] / 60:.1f} min over 40 prompts.")
    RESULTS_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
