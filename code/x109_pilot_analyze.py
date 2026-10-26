"""Benign insertion pilot, analysis (local, numpy only).

refusal2 rates per (prompt, condition) from the x64 judge; per-word contrasts with a
prompt-level paired bootstrap; class and pooled means; the link between the boundary
projection shift and refusal2; the hijack readout; the neutral-twin check.
Writes results.json and results.md in the pilot directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x109_pilot_common import (  # noqa: E402
    JUDGE_DIR,
    PROJECTIONS,
    PROMPTS,
    RESULTS,
    RESULTS_MD,
    WORDS,
    X106_HARD1K_SD,
    atomic_json,
    cells,
    mentions,
    read_json,
)
from x109_pilot_judge import load_generations  # noqa: E402

N_BOOT = 10_000
BOOT_SEED = 20_261_008


def ranks(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x), dtype=float)
    for value in np.unique(x):
        tie = x == value
        if tie.sum() > 1:
            r[tie] = r[tie].mean()
    return r


def spearman(x, y) -> float:
    rx, ry = ranks(np.asarray(x)), ranks(np.asarray(y))
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def perm_p(x, y, n=10_000, seed=1) -> float:
    rng = np.random.default_rng(seed)
    obs = abs(spearman(x, y))
    y = np.asarray(y)
    hits = sum(abs(spearman(x, rng.permutation(y))) >= obs for _ in range(n))
    return float((1 + hits) / (n + 1))


def load_jsonl(path: Path, key: str) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if not rec.get("failed"):
            out[rec["cell_id"]] = rec
    return out


def ci(samples: np.ndarray) -> list[float]:
    return [float(v) for v in np.quantile(samples, (0.025, 0.975))]


def main() -> None:
    prompts = read_json(PROMPTS)["prompts"]
    words = read_json(WORDS)
    proj = read_json(PROJECTIONS)
    gens, _, _ = load_generations()
    gen = {g["cell_id"]: g for g in gens}
    ref = load_jsonl(JUDGE_DIR / "refusal2.jsonl", "refusal")
    hij = load_jsonl(JUDGE_DIR / "hijack.jsonl", "hijack_llm")
    grid = cells(read_json(PROMPTS), words)
    missing_gen = [c["cell_id"] for c in grid if c["cell_id"] not in gen]
    missing_ref = [c["cell_id"] for c in grid if c["cell_id"] in gen and c["cell_id"] not in ref]
    for cid, rec in ref.items():
        if cid in gen and rec["response_sha256"] != gen[cid]["response_sha256"]:
            raise SystemExit(f"FATAL: refusal2 record {cid} does not match its responses")
    pids = [p["id"] for p in prompts]
    shapes = np.asarray([p["shape"] for p in prompts])
    union = words["union"]
    uw = [u["word"] for u in union]

    def rate(cid: str) -> float:
        rec = ref.get(cid)
        return float(np.mean(rec["refusal"])) if rec else float("nan")

    rO = np.asarray([rate(f"{p}|O") for p in pids])
    rE = np.asarray([[rate(f"{p}|E_{u['word']}") for p in pids] for u in union])  # [W, P]
    rN = np.asarray([[rate(f"{p}|N_{u['twin']}") for p in pids] for u in union])
    W, P = rE.shape

    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, P, size=(N_BOOT, P))

    def boot_mean(m: np.ndarray) -> np.ndarray:  # m: [..., P] -> [..., N_BOOT]
        return np.nanmean(m[..., idx], axis=-1)

    dEN, dEO, dNO = rE - rN, rE - rO[None], rN - rO[None]
    bEN, bEO, bNO = boot_mean(dEN), boot_mean(dEO), boot_mean(dNO)

    per_word = []
    for k, u in enumerate(union):
        per_word.append(
            {
                "word": u["word"],
                "classes": u["classes"],
                "twin": u["twin"],
                "r_E": float(np.nanmean(rE[k])),
                "r_N": float(np.nanmean(rN[k])),
                "delta_EN": float(np.nanmean(dEN[k])),
                "delta_EN_ci95": ci(bEN[k]),
                "delta_EO": float(np.nanmean(dEO[k])),
                "delta_EO_ci95": ci(bEO[k]),
                "delta_NO": float(np.nanmean(dNO[k])),
                "delta_NO_ci95": ci(bNO[k]),
                "delta_EN_shape_A": float(np.nanmean(dEN[k][shapes == "A"])),
                "delta_EN_shape_B": float(np.nanmean(dEN[k][shapes == "B"])),
                "n_prompts_refusal_E_gt_0": int(np.sum(rE[k] > 0)),
                "mean_shift": u["word_mean_shift"],
                "mean_shift_sd_units": u["word_mean_shift"] / X106_HARD1K_SD,
                "twin_mean_shift": u["twin_mean_shift"],
            }
        )

    per_class = {}
    for key, members in words["classes"].items():
        rows = [uw.index(w) for w in members]
        per_class[key] = {
            "words": members,
            "delta_EN": float(np.nanmean(dEN[rows])),
            "delta_EN_ci95": ci(np.nanmean(bEN[rows], axis=0)),
            "delta_EO": float(np.nanmean(dEO[rows])),
            "delta_EO_ci95": ci(np.nanmean(bEO[rows], axis=0)),
            "r_E": float(np.nanmean(rE[rows])),
        }

    pooled = {"all": {}, "shape_A": {}, "shape_B": {}}
    for name, mask in (("all", np.ones(P, bool)), ("shape_A", shapes == "A"), ("shape_B", shapes == "B")):
        sub_idx = np.where(mask)[0]
        bidx = rng.integers(0, len(sub_idx), size=(N_BOOT, len(sub_idx)))
        for label, m in (("delta_EN", dEN), ("delta_EO", dEO), ("delta_NO", dNO)):
            sub = m[:, sub_idx]
            pooled[name][label] = float(np.nanmean(sub))
            pooled[name][label + "_ci95"] = ci(np.nanmean(sub[:, bidx], axis=(0, 2)))
        pooled[name]["r_O"] = float(np.nanmean(rO[mask]))
        pooled[name]["r_E_mean"] = float(np.nanmean(rE[:, mask]))
        pooled[name]["r_N_mean"] = float(np.nanmean(rN[:, mask]))

    # Direction link.
    pw = proj["words"]
    shift = np.asarray(proj["shift"])  # [P, all words]
    if proj["prompt_ids"] != pids:
        raise SystemExit("FATAL: projection prompt order differs")
    word_shift = np.asarray([shift[:, pw.index(w)].mean() for w in uw])
    word_en = np.asarray([r["delta_EN"] for r in per_word])
    word_eo = np.asarray([r["delta_EO"] for r in per_word])
    cell_shift = np.asarray([[shift[i, pw.index(w)] for i in range(P)] for w in uw])  # [W, P]
    twin_shift = np.asarray([[shift[i, pw.index(u["twin"])] for i in range(P)] for u in union])
    flat_s = np.concatenate([cell_shift.ravel(), twin_shift.ravel()])
    flat_r = np.concatenate([rE.ravel(), rN.ravel()])
    ok = np.isfinite(flat_r)
    eo_cell = (rE - rO[None]).ravel()
    within = []
    for i in range(P):
        if np.nanstd(rE[:, i]) > 0:
            within.append(spearman(cell_shift[:, i], rE[:, i]))
    argmax_rows = []
    for i, pid in enumerate(pids):
        top_union = uw[int(np.argmax(cell_shift[:, i]))]
        best = np.nanmax(rE[:, i])
        top_ref = [uw[k] for k in range(W) if rE[k, i] == best]
        argmax_rows.append(
            {
                "prompt_id": pid,
                "shape": shapes[i],
                "argmax_lexicon_word": proj["argmax_lexicon_word_by_prompt"][pid],
                "argmax_in_union": proj["argmax_lexicon_word_by_prompt"][pid] in uw,
                "argmax_union_word": top_union,
                "highest_refusal_union_words": top_ref,
                "highest_refusal_rate": float(best),
                "r_O": float(rO[i]),
                "argmax_union_is_a_highest_refusal_word": top_union in top_ref,
            }
        )
    link = {
        "per_word_spearman_shift_vs_delta_EN": spearman(word_shift, word_en),
        "per_word_spearman_shift_vs_delta_EN_perm_p": perm_p(word_shift, word_en),
        "per_word_spearman_shift_vs_delta_EO": spearman(word_shift, word_eo),
        "per_word_spearman_shift_vs_delta_EO_perm_p": perm_p(word_shift, word_eo),
        "n_words": W,
        "per_cell_spearman_shift_vs_refusal2_E_cells": spearman(cell_shift.ravel()[np.isfinite(rE.ravel())], rE.ravel()[np.isfinite(rE.ravel())]),
        "per_cell_spearman_shift_vs_refusal2_E_minus_O": spearman(cell_shift.ravel()[np.isfinite(eo_cell)], eo_cell[np.isfinite(eo_cell)]),
        "per_cell_spearman_shift_vs_refusal2_E_and_N_cells": spearman(flat_s[ok], flat_r[ok]),
        "n_cells_E": int(np.isfinite(rE).sum()),
        "within_prompt_spearman_mean": float(np.mean(within)) if within else None,
        "n_prompts_with_refusal_variation": len(within),
        "argmax": argmax_rows,
        "n_argmax_in_union": int(sum(r["argmax_in_union"] for r in argmax_rows)),
        "n_argmax_union_is_highest_refusal": int(sum(r["argmax_union_is_a_highest_refusal_word"] for r in argmax_rows)),
        "n_argmax_union_is_highest_refusal_among_prompts_with_any_E_refusal": int(sum(r["argmax_union_is_a_highest_refusal_word"] for r in argmax_rows if r["highest_refusal_rate"] > 0)),
        "n_prompts_with_any_E_refusal": int(sum(r["highest_refusal_rate"] > 0 for r in argmax_rows)),
    }

    # Hijack.
    hijack_words = []
    pool = {"E": {"llm": [], "str": [], "ref": []}, "N": {"llm": [], "str": [], "ref": []}}
    for u in union:
        rec = {"word": u["word"], "twin": u["twin"]}
        for kind, w in (("E", u["word"]), ("N", u["twin"])):
            llm, st = [], []
            for p in pids:
                cid = f"{p}|{kind}_{w}"
                g = gen.get(cid)
                if g is None:
                    continue
                s = [mentions(t, w) for t in g["llm_responses"]]
                st.extend(s)
                h = hij.get(cid)
                r = ref.get(cid)
                if h:
                    llm.extend(h["hijack_llm"])
                if h and r:
                    pool[kind]["llm"].extend(h["hijack_llm"])
                    pool[kind]["str"].extend(s)
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
            "n_hijack_llm": int(llm.sum()),
            "n_mention": int(st.sum()),
            "share_of_refusals_with_hijack_llm": float(llm[rf == 1].mean()) if (rf == 1).any() else None,
        }

    # Projection summary.
    ws = proj["word_summary"]
    lex = [w for w in pw if ws[w]["set"] == "lexicon"]
    top_proj = sorted(lex, key=lambda w: -ws[w]["mean_shift"])
    o_proj = np.asarray(proj["o_proj"])
    neutral_shifts = [ws[w]["mean_shift"] for w in pw if ws[w]["set"] != "lexicon"]
    proj_summary = {
        "o_proj_mean": float(o_proj.mean()),
        "o_proj_by_prompt": dict(zip(pids, o_proj.tolist())),
        "o_proj_mean_shape_A": float(o_proj[shapes == "A"].mean()),
        "o_proj_mean_shape_B": float(o_proj[shapes == "B"].mean()),
        "lexicon_mean_shift": float(np.mean([ws[w]["mean_shift"] for w in lex])),
        "nonlexicon_mean_shift": float(np.mean(neutral_shifts)),
        "top10_lexicon_by_mean_shift": [{"word": w, "mean_shift": ws[w]["mean_shift"], "sd_units": ws[w]["mean_shift_sd_units"], "tbg_minus1": ws[w]["mean_shift_tbg_minus1"], "tbg": ws[w]["mean_shift_tbg"]} for w in top_proj[:10]],
        "bottom5_lexicon_by_mean_shift": [{"word": w, "mean_shift": ws[w]["mean_shift"]} for w in top_proj[-5:]],
        "argmax_word_counts": {w: list(proj["argmax_lexicon_word_by_prompt"].values()).count(w) for w in sorted(set(proj["argmax_lexicon_word_by_prompt"].values()))},
        "sd_unit": X106_HARD1K_SD,
    }

    results = {
        "pilot": "x109/pilot",
        "construct": "refusal2 (an explicit refusal statement is present), x64 judge, DeepSeek V4 Flash",
        "n_prompts": P,
        "n_words": W,
        "missing_generations": missing_gen,
        "missing_refusal2": missing_ref,
        "r_O": {"all": float(np.nanmean(rO)), "shape_A": float(np.nanmean(rO[shapes == "A"])), "shape_B": float(np.nanmean(rO[shapes == "B"])), "by_prompt": dict(zip(pids, rO.tolist()))},
        "per_word": per_word,
        "per_class": per_class,
        "pooled": pooled,
        "direction_link": link,
        "hijack_per_word": hijack_words,
        "hijack_pooled": hijack_pool,
        "projection_summary": proj_summary,
        "rate_matrix": {"words": uw, "twins": [u["twin"] for u in union], "prompt_ids": pids, "r_E": rE.tolist(), "r_N": rN.tolist(), "r_O": rO.tolist()},
        "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED, "unit": "prompts resampled with replacement, paired across conditions"},
    }
    atomic_json(RESULTS, results)
    write_md(results)
    print(RESULTS_MD.read_text())


def f(x, d=3):
    return "nan" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:+.{d}f}"


def write_md(r: dict) -> None:
    L = []
    L.append("# Benign insertion pilot (Qwen3.5-4B)\n")
    L.append(f"20 source-clean benign prompts (10 Dolly passage+task, 10 No Robots instruction-only prose), one word appended at the end. Refusal is `refusal2` (x64 judge), 10 sampled responses per cell. CIs: prompt-level paired bootstrap, {r['bootstrap']['n']} draws.\n")
    o = r["r_O"]
    pa, d0, hp = r["pooled"]["all"], r["direction_link"], r["hijack_pooled"]["E"]
    top = max(r["per_word"], key=lambda x: x["delta_EN"])
    L.append("## Findings\n")
    L.append(
        f"Qwen3.5-4B refuses none of the original prompts (r(O) = {o['all']:.3f}). Appending one harm word raises refusal2 to a mean of {pa['r_E_mean']:.3f} over the {r['n_words']} words "
        f"(Δ_EN = {pa['delta_EN']:+.3f} [{pa['delta_EN_ci95'][0]:+.3f}, {pa['delta_EN_ci95'][1]:+.3f}]); the neutral twins give {pa['r_N_mean']:.3f}. "
        f"The largest word is {top['word']} (r(E) = {top['r_E']:.3f}, Δ_EN = {top['delta_EN']:+.3f}). "
        f"The per-word projection shift tracks Δ_EN with Spearman ρ = {d0['per_word_spearman_shift_vs_delta_EN']:+.2f} (n = {r['n_words']}) and the per-cell refusal2 rate with ρ = {d0['per_cell_spearman_shift_vs_refusal2_E_cells']:+.2f}. "
        f"Every refused E response is one the hijack judge labels as treating the appended word as part of the request (refusal2 {hp['refusal2_given_hijack_llm']:.3f} when labelled hijack, {hp['refusal2_given_not_hijack_llm']:.3f} otherwise).\n"
    )
    L.append(f"r(O): all {o['all']:.3f}, shape A {o['shape_A']:.3f}, shape B {o['shape_B']:.3f}.\n")
    L.append("## Per word\n")
    L.append("| word | classes | twin | r(E) | r(N) | Δ_EN [95% CI] | Δ_EO [95% CI] | r(N)−r(O) | shift (sd) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for w in sorted(r["per_word"], key=lambda x: -x["delta_EN"]):
        L.append(f"| {w['word']} | {','.join(w['classes'])} | {w['twin']} | {w['r_E']:.3f} | {w['r_N']:.3f} | {f(w['delta_EN'])} [{f(w['delta_EN_ci95'][0])}, {f(w['delta_EN_ci95'][1])}] | {f(w['delta_EO'])} [{f(w['delta_EO_ci95'][0])}, {f(w['delta_EO_ci95'][1])}] | {f(w['delta_NO'])} | {w['mean_shift_sd_units']:+.2f} |")
    L.append("\n## Per class (mean over its 5 words)\n")
    L.append("| class | words | r(E) | Δ_EN [95% CI] | Δ_EO [95% CI] |")
    L.append("|---|---|---|---|---|")
    for k, c in r["per_class"].items():
        L.append(f"| {k} | {', '.join(c['words'])} | {c['r_E']:.3f} | {f(c['delta_EN'])} [{f(c['delta_EN_ci95'][0])}, {f(c['delta_EN_ci95'][1])}] | {f(c['delta_EO'])} [{f(c['delta_EO_ci95'][0])}, {f(c['delta_EO_ci95'][1])}] |")
    L.append("\n## Pooled over the union\n")
    L.append("| prompts | r(O) | mean r(E) | mean r(N) | Δ_EN [95% CI] | Δ_EO [95% CI] | r(N)−r(O) [95% CI] |")
    L.append("|---|---|---|---|---|---|---|")
    for k, c in r["pooled"].items():
        L.append(f"| {k} | {c['r_O']:.3f} | {c['r_E_mean']:.3f} | {c['r_N_mean']:.3f} | {f(c['delta_EN'])} [{f(c['delta_EN_ci95'][0])}, {f(c['delta_EN_ci95'][1])}] | {f(c['delta_EO'])} [{f(c['delta_EO_ci95'][0])}, {f(c['delta_EO_ci95'][1])}] | {f(c['delta_NO'])} [{f(c['delta_NO_ci95'][0])}, {f(c['delta_NO_ci95'][1])}] |")
    d = r["direction_link"]
    L.append("\n## Direction link\n")
    L.append(f"- Per word (n = {d['n_words']}): Spearman ρ(mean projection shift, Δ_EN) = {f(d['per_word_spearman_shift_vs_delta_EN'])} (permutation p = {d['per_word_spearman_shift_vs_delta_EN_perm_p']:.3f}); ρ(shift, Δ_EO) = {f(d['per_word_spearman_shift_vs_delta_EO'])} (p = {d['per_word_spearman_shift_vs_delta_EO_perm_p']:.3f}).")
    L.append(f"- Per cell (prompt × word, n = {d['n_cells_E']} E cells): ρ(shift, refusal2 rate) = {f(d['per_cell_spearman_shift_vs_refusal2_E_cells'])}; ρ(shift, r(E) − r(O)) = {f(d['per_cell_spearman_shift_vs_refusal2_E_minus_O'])}; E and N cells together ρ = {f(d['per_cell_spearman_shift_vs_refusal2_E_and_N_cells'])}.")
    L.append(f"- Within-prompt ρ(shift, r(E)) averaged over the {d['n_prompts_with_refusal_variation']} prompts whose E rates vary: {f(d['within_prompt_spearman_mean'])}.")
    L.append(f"- Argmax: the prompt's argmax lexicon word is in the union for {d['n_argmax_in_union']}/20 prompts; the union word with the largest shift is a highest-refusal union word on {d['n_argmax_union_is_highest_refusal_among_prompts_with_any_E_refusal']}/{d['n_prompts_with_any_E_refusal']} prompts with any E refusal.\n")
    L.append("| prompt | shape | argmax (99 words) | argmax (union) | highest-refusal union word(s) | max r(E) | r(O) |")
    L.append("|---|---|---|---|---|---|---|")
    for a in d["argmax"]:
        hr = ", ".join(a["highest_refusal_union_words"]) if a["highest_refusal_rate"] > 0 else "(none refused)"
        L.append(f"| {a['prompt_id']} | {a['shape']} | {a['argmax_lexicon_word']} | {a['argmax_union_word']} | {hr} | {a['highest_refusal_rate']:.1f} | {a['r_O']:.1f} |")
    L.append("\n## Hijack readout\n")
    L.append("String label: the response contains the appended word (case-insensitive whole word). LLM label: the judge model says the response treats the word as part of the request or topic.\n")
    L.append("| word | E mention | E hijack (LLM) | twin | N mention | N hijack (LLM) |")
    L.append("|---|---|---|---|---|---|")
    for h in r["hijack_per_word"]:
        L.append(f"| {h['word']} | {h['E_mention_rate']:.3f} | {h['E_hijack_llm_rate'] if h['E_hijack_llm_rate'] is None else round(h['E_hijack_llm_rate'], 3)} | {h['twin']} | {h['N_mention_rate']:.3f} | {h['N_hijack_llm_rate'] if h['N_hijack_llm_rate'] is None else round(h['N_hijack_llm_rate'], 3)} |")
    L.append("")
    for kind, h in r["hijack_pooled"].items():
        L.append(f"- {kind} cells ({h['n_responses']} responses): mention {h['mention_rate']}, hijack (LLM) {h['hijack_llm_rate']}; refusal2 | hijack {h['refusal2_given_hijack_llm']} vs | no hijack {h['refusal2_given_not_hijack_llm']}; refusal2 | mention {h['refusal2_given_mention']} vs | no mention {h['refusal2_given_no_mention']}; share of refusals labelled hijack {h['share_of_refusals_with_hijack_llm']}.")
    p = r["projection_summary"]
    L.append("\n## Projection matrix (20 prompts × 99 lexicon words)\n")
    L.append(f"Mean O projection {p['o_proj_mean']:.2f} (shape A {p['o_proj_mean_shape_A']:.2f}, shape B {p['o_proj_mean_shape_B']:.2f}); mean shift over the 99 lexicon words {p['lexicon_mean_shift']:+.2f}, over the non-lexicon nouns {p['nonlexicon_mean_shift']:+.2f} (sd unit {p['sd_unit']:.3f}).\n")
    L.append("| word | mean shift | sd units | tbg_minus1 | tbg |")
    L.append("|---|---|---|---|---|")
    for t in p["top10_lexicon_by_mean_shift"]:
        L.append(f"| {t['word']} | {t['mean_shift']:+.2f} | {t['sd_units']:+.2f} | {t['tbg_minus1']:+.2f} | {t['tbg']:+.2f} |")
    L.append(f"\nArgmax word counts over the 20 prompts: {p['argmax_word_counts']}.")
    RESULTS_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
