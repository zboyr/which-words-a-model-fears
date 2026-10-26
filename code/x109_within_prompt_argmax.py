"""x109 insertion: within a prompt, is the word that raises refusal most over its twin the word
whose shift over its twin is largest?

Cells: scan (10 words x 40 prompts, sets A and B), pilot (15 lexicon words x 20 set-A prompts),
and their union on set A (25 words). Refusal effect per cell = r(E_w) - r(N_w) (10 samples
each); shift effect per cell = shift(w) - shift(twin), both from the scan table (sd units), plus
the raw shift(w). Per prompt: does the shift argmax land on a refusal argmax (ties allowed), the
top-k overlap, and the within-prompt Spearman; against the chance level 1 / W adjusted for ties.
Prompts whose largest refusal effect is <= 0 are reported separately. Local, numpy/scipy only.

Writes data/runs/x109/within_prompt_argmax/results.{json,md}.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
X109 = ROOT / "data" / "runs" / "x109"
OUT = X109 / "within_prompt_argmax"
N_BOOT = 10_000
SEED = 10_913


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_scan_table():
    scan = read(X109 / "scan.json")
    sd = float(scan["sd_unit"]["value"])
    shift = np.load(X109 / "scan_shift.npz")["shift"].astype(np.float64).sum(2) / sd  # [40, N]
    index = {w: j for j, w in enumerate(scan["candidates"]["words"])}
    return shift, index, scan["prompt_ids"], np.asarray(scan["prompt_sets"])


def cell_arrays(rate_matrix: dict, shift: np.ndarray, index: dict, scan_pids: list[str]):
    """Return words, twins, prompt ids, rE, rN [W, P], shift_w, shift_twin [W, P] on the scan table."""
    words, twins, pids = rate_matrix["words"], rate_matrix["twins"], rate_matrix["prompt_ids"]
    prow = [scan_pids.index(p) for p in pids]
    rE = np.asarray(rate_matrix["r_E"], float)
    rN = np.asarray(rate_matrix["r_N"], float)
    sw = np.asarray([[shift[i, index[w]] for i in prow] for w in words])
    st = np.asarray([[shift[i, index[t]] for i in prow] for t in twins])
    return words, twins, pids, rE, rN, sw, st


def within_prompt(words, pids, dR, dS, rng) -> dict:
    """dR, dS: [W, P]. Per prompt argmax agreement of dS with dR (ties in dR allowed), top-k, Spearman."""
    W, P = dR.shape
    agree, chance, top3, rho, rows = [], [], [], [], []
    active = []
    for p in range(P):
        r, s = dR[:, p], dS[:, p]
        if not np.isfinite(r).all():
            continue
        rmax = r.max()
        active.append(rmax > 0)
        tied = np.flatnonzero(r == rmax)
        s_arg = int(np.argmax(s))
        agree.append(s_arg in tied)
        chance.append(len(tied) / W)
        # top-3 overlap: refusal top-3 (ties broken by s? no: by rank with ties sharing) vs shift top-3
        r_top3 = set(np.argsort(-r, kind="stable")[:3])
        s_top3 = set(np.argsort(-s, kind="stable")[:3])
        top3.append(len(r_top3 & s_top3) / 3)
        rho.append(stats.spearmanr(r, s)[0] if np.std(r) > 0 else np.nan)
        rows.append({"prompt": pids[p], "r_max": float(rmax), "refusal_argmax": [words[i] for i in tied],
                     "shift_argmax": words[s_arg], "shift_at_refusal_argmax_rank": int(1 + np.sum(s > s[tied].max())),
                     "agree": bool(s_arg in tied), "spearman": None if np.isnan(rho[-1]) else float(rho[-1])})
    agree, chance, top3, rho, active = map(np.asarray, (agree, chance, top3, rho, active))

    def summ(mask):
        if mask.sum() == 0:
            return None
        a = agree[mask].astype(float)
        idx = rng.integers(0, len(a), size=(N_BOOT, len(a)))
        ci = [float(np.percentile(a[idx].mean(1), 2.5)), float(np.percentile(a[idx].mean(1), 97.5))]
        # exact binomial-ish p under per-prompt chance (Poisson-binomial by simulation)
        sims = (rng.random((N_BOOT, mask.sum())) < chance[mask][None]).mean(1)
        p_val = float(np.mean(sims >= a.mean()))
        rr = rho[mask]
        return {"n_prompts": int(mask.sum()), "agree": float(a.mean()), "agree_ci": ci,
                "chance": float(chance[mask].mean()), "p_vs_chance": p_val,
                "top3_overlap": float(top3[mask].mean()), "top3_chance": 3.0 / W,
                "median_rank_of_shift_at_refusal_argmax": float(np.median([r_["shift_at_refusal_argmax_rank"] for r_, m in zip(rows, mask) if m])),
                "spearman_within_prompt_mean": float(np.nanmean(rr)) if np.isfinite(rr).any() else None,
                "spearman_within_prompt_median": float(np.nanmedian(rr)) if np.isfinite(rr).any() else None,
                "share_spearman_positive": float(np.nanmean(rr > 0)) if np.isfinite(rr).any() else None}

    return {"n_words": W, "all_prompts": summ(np.ones(len(agree), bool)),
            "prompts_with_positive_effect": summ(active), "prompts_without_effect": summ(~active), "rows": rows}


def analyse(name, words, pids, rE, rN, sw, st, rng, set_mask=None):
    if set_mask is not None:
        rE, rN, sw, st, pids = rE[:, set_mask], rN[:, set_mask], sw[:, set_mask], st[:, set_mask], [p for p, m in zip(pids, set_mask) if m]
    dR = rE - rN
    out = {"name": name, "n_words": len(words), "n_prompts": len(pids), "words": words,
           "mean_dR": float(np.nanmean(dR)),
           "shift_minus_twin": within_prompt(words, pids, dR, sw - st, rng),
           "raw_shift": within_prompt(words, pids, dR, sw, rng),
           "rE_vs_shift": within_prompt(words, pids, rE, sw, rng)}
    # pooled per-cell Spearman for reference
    ok = np.isfinite(dR)
    out["pooled_cell_spearman_dR_vs_shift_minus_twin"] = float(stats.spearmanr(dR[ok], (sw - st)[ok])[0])
    out["pooled_cell_spearman_dR_vs_shift"] = float(stats.spearmanr(dR[ok], sw[ok])[0])
    # between-prompt: does the prompt-level refusal argmax word vary? how concentrated?
    argmaxes = [words[int(np.argmax(dR[:, p]))] for p in range(dR.shape[1]) if np.isfinite(dR[:, p]).all() and dR[:, p].max() > 0]
    out["refusal_argmax_word_counts"] = {w: argmaxes.count(w) for w in sorted(set(argmaxes), key=lambda w: -argmaxes.count(w))}
    sargs = [words[int(np.argmax((sw - st)[:, p]))] for p in range(sw.shape[1])]
    out["shift_argmax_word_counts"] = {w: sargs.count(w) for w in sorted(set(sargs), key=lambda w: -sargs.count(w))}
    return out


def md(results: list[dict]) -> str:
    L = ["# x109 insertion: within-prompt argmax of refusal effect against shift effect (Qwen3.5-4B)", "",
         "Per cell (word w, prompt p): refusal effect = r(E_w) − r(N_w), shift effect = shift(w) − shift(twin) from the scan table (sd units), raw shift = shift(w). Per prompt, `agree` = the word with the largest shift effect is among the words with the largest refusal effect (ties allowed); `chance` = mean over prompts of (number of tied refusal argmaxes / W); p = simulated probability of at least the observed agreement under per-prompt chance. Prompts with a positive largest refusal effect are the ones where some insertion worked.", ""]
    for r in results:
        L.append(f"## {r['name']} ({r['n_words']} words × {r['n_prompts']} prompts; mean Δ_EN {r['mean_dR']:+.3f})")
        L.append("")
        L.append("| predictor | prompts | n | agree [95% CI] | chance | p | top-3 overlap (chance) | median rank of shift at refusal argmax | within-prompt Spearman mean / median | share ρ > 0 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for pred, lab in (("shift_minus_twin", "shift(w) − shift(twin) vs r(E) − r(N)"), ("raw_shift", "shift(w) vs r(E) − r(N)"), ("rE_vs_shift", "shift(w) vs r(E)")):
            for key, plab in (("prompts_with_positive_effect", "effect > 0"), ("all_prompts", "all"), ("prompts_without_effect", "no effect")):
                s = r[pred][key]
                if not s:
                    continue
                sm = "–" if s["spearman_within_prompt_mean"] is None else f"{s['spearman_within_prompt_mean']:+.2f} / {s['spearman_within_prompt_median']:+.2f}"
                sp = "–" if s["share_spearman_positive"] is None else f"{s['share_spearman_positive']:.2f}"
                L.append(f"| {lab} | {plab} | {s['n_prompts']} | {s['agree']:.2f} [{s['agree_ci'][0]:.2f}, {s['agree_ci'][1]:.2f}] | {s['chance']:.2f} | {s['p_vs_chance']:.3f} | {s['top3_overlap']:.2f} ({s['top3_chance']:.2f}) | {s['median_rank_of_shift_at_refusal_argmax']:.0f} | {sm} | {sp} |")
        L.append("")
        L.append(f"Pooled per-cell Spearman: Δ_EN vs shift − twin shift {r['pooled_cell_spearman_dR_vs_shift_minus_twin']:+.2f}; Δ_EN vs raw shift {r['pooled_cell_spearman_dR_vs_shift']:+.2f}.")
        L.append("")
        L.append("Refusal-argmax word per prompt (effect > 0): " + ", ".join(f"{w} ({c})" for w, c in r["refusal_argmax_word_counts"].items()))
        L.append("")
        L.append("Shift-argmax word per prompt: " + ", ".join(f"{w} ({c})" for w, c in r["shift_argmax_word_counts"].items()))
        L.append("")
        L.append("Per prompt (shift − twin predictor):")
        L.append("")
        L.append("| prompt | max Δ_EN | refusal argmax | shift argmax | rank of shift at refusal argmax | agree | ρ |")
        L.append("|---|---|---|---|---|---|---|")
        for row in r["shift_minus_twin"]["rows"]:
            rho = "–" if row["spearman"] is None else f"{row['spearman']:+.2f}"
            L.append(f"| {row['prompt']} | {row['r_max']:+.1f} | {', '.join(row['refusal_argmax'])} | {row['shift_argmax']} | {row['shift_at_refusal_argmax_rank']} | {'yes' if row['agree'] else 'no'} | {rho} |")
        L.append("")
    return "\n".join(L)


def main() -> None:
    rng = np.random.default_rng(SEED)
    shift, index, scan_pids, sets = load_scan_table()
    scan_rm = read(X109 / "results.json")["rate_matrix"]
    pilot_rm = read(X109 / "pilot" / "results.json")["rate_matrix"]
    s_words, s_twins, s_pids, s_rE, s_rN, s_sw, s_st = cell_arrays(scan_rm, shift, index, scan_pids)
    p_words, p_twins, p_pids, p_rE, p_rN, p_sw, p_st = cell_arrays(pilot_rm, shift, index, scan_pids)
    s_sets = np.asarray([sets[scan_pids.index(p)] for p in s_pids])
    results = [
        analyse("scan ten, set B (held-out)", s_words, s_pids, s_rE, s_rN, s_sw, s_st, rng, s_sets == "B"),
        analyse("scan ten, set A", s_words, s_pids, s_rE, s_rN, s_sw, s_st, rng, s_sets == "A"),
        analyse("scan ten, both sets", s_words, s_pids, s_rE, s_rN, s_sw, s_st, rng),
        analyse("pilot fifteen lexicon words, set A", p_words, p_pids, p_rE, p_rN, p_sw, p_st, rng),
    ]
    # union on set A: pilot 15 + scan 10 (a word in both uses the scan run's cells)
    assert p_pids == [p for p, m in zip(s_pids, s_sets == "A") if m]
    keep = [i for i, w in enumerate(p_words) if w not in s_words]
    u_words = [p_words[i] for i in keep] + s_words
    u_rE = np.vstack([p_rE[keep], s_rE[:, s_sets == "A"]])
    u_rN = np.vstack([p_rN[keep], s_rN[:, s_sets == "A"]])
    u_sw = np.vstack([p_sw[keep], s_sw[:, s_sets == "A"]])
    u_st = np.vstack([p_st[keep], s_st[:, s_sets == "A"]])
    results.append(analyse("union 25 words, set A", u_words, p_pids, u_rE, u_rN, u_sw, u_st, rng))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    text = md(results)
    (OUT / "results.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
