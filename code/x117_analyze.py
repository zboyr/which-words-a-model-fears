"""x117 analysis (local, numpy / scipy): x111's statistics on any set of x111-contract runs.

`refusal2` (x64 judge, twenty samples per cell judged as two units of ten) per (prompt, condition)
cell; per population (benign, harmful) and pooled: r(O), per-word Delta_EO = r(E) - r(O) with a
prompt-level paired bootstrap, the dose-response curve over the eight concordant bins with its
adjacent-pair ordering and top-minus-bottom gap, and the Spearman rho of each of the four fixed
x111 tables (benign, BP, HP, mixed) and of the in-context key-word shift with the per-word Delta_EO
over all 40 words, over the 24 concordant words and over the middle four bins, with prompt-bootstrap
intervals, word-permutation p, partial rho holding token count and log wordfreq rank fixed, per-cell
rho and the split-half noise ceiling.

`--runs x117` (default) reads data/runs/x117 and writes data/runs/x117/results.{json,md}.
`--runs x111` reads x111's analysed cohort (the 32 benign and 32 harmful prompts of
prompts.json minus x85_exclusions.json plus prompts_topup.json) and reproduces x111's numbers.
`--runs x111,x117` pools the two cohorts (64 benign + 64 harmful prompts) and writes
results_pooled_<runs>.{json,md} under the last run's directory. Words, tables, bins, seeds,
bootstrap (10,000 draws, seed 111,000, resampled within each population), permutation (10,000 word
permutations) and Holm procedure are x111's, imported.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x111_common import (  # noqa: E402
    ANALYSIS_TABLES as TABLES,
    N_HALF,
    N_SAMPLES,
    POPULATIONS,
    ROOT,
    TABLES_JSON,
    TABLES_NPZ,
    X106_HARD1K_SD,
    atomic_json,
    read_json,
)
from x111_analyze import (  # noqa: E402
    BOOT_SEED,
    MIDDLE_BINS,
    N_BOOT,
    N_PERM,
    SETS,
    boot_p,
    ci,
    cis,
    f,
    holm,
    partial_rho,
    perm_p,
    rho,
    rho_rows,
)
from x111_select import BIN_NAMES  # noqa: E402

HALF_SEEDS = list(range(20))
WORDS_FILE = ROOT / "data" / "runs" / "x111" / "words.json"


def run_dir(name: str) -> Path:
    return ROOT / "data" / "runs" / name


def units_of(path: Path) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if not rec.get("failed"):
                out[rec["cell_id"]] = rec
    return out


def load_run(name: str, words: list[str]) -> dict:
    """One run's analysed cohort: prompt rows in analysis order, the refusal2 label matrices, the
    in-context key-word shift and the generation bookkeeping. For x111 the cohort is the analysed
    one (x85 exclusions applied, harmful top-up appended), which is what its results.json reports."""
    d = run_dir(name)
    obj = read_json(d / "prompts.json")
    rows = list(obj["prompts"])
    excluded: list[str] = []
    if (d / "x85_exclusions.json").exists():
        excluded = sorted(read_json(d / "x85_exclusions.json")["excluded_x111_prompt_ids"])
        rows = [p for p in rows if p["id"] not in set(excluded)]
    topup = read_json(d / "prompts_topup.json")["prompts"] if (d / "prompts_topup.json").exists() else []
    rows += list(topup)
    pids = [p["id"] for p in rows]

    gen: dict[str, dict] = {}
    for path in sorted(glob.glob(str(d / "generations" / "generations_*shard*of*.json"))):
        for r in read_json(path)["rows"]:
            gen[r["cell_id"]] = r
    ref = units_of(d / "judge" / "refusal2.jsonl")

    def labels(cid: str):
        parts = [ref.get(f"{cid}#h{h}") for h in range(N_SAMPLES // N_HALF)]
        if any(p is None for p in parts):
            return None
        g = gen.get(cid)
        if g is None:
            return None
        for h, p in enumerate(parts):
            if p["response_sha256"] != g["response_sha256"][h * N_HALF : (h + 1) * N_HALF]:
                raise SystemExit(f"FATAL: {name}: refusal2 record {cid}#h{h} does not match its responses")
        return np.asarray(sum((list(p["refusal"]) for p in parts), []), float)

    P, W = len(pids), len(words)
    LO = np.full((P, N_SAMPLES), np.nan)
    LE = np.full((W, P, N_SAMPLES), np.nan)
    missing_gen, missing_ref = [], []
    for j, pid in enumerate(pids):
        for cid, put in [(f"{pid}|O", lambda v, j=j: LO.__setitem__(j, v))] + [(f"{pid}|E_{w}", lambda v, j=j, k=k: LE.__setitem__((k, j), v)) for k, w in enumerate(words)]:
            if cid not in gen:
                missing_gen.append(cid)
                continue
            v = labels(cid)
            if v is None:
                missing_ref.append(cid)
            else:
                put(v)

    # In-context key-word shift, sd units, by prompt id.
    key_by_pid: dict[str, np.ndarray] = {}
    key_checks = {}
    for tag, js, npz in (("main", d / "key_words_scan.json", d / "key_words_shift.npz"), ("topup", d / "key_words_scan_topup.json", d / "key_words_shift_topup.npz")):
        if not (js.exists() and npz.exists()):
            continue
        kj = read_json(js)
        if kj["candidates"]["words"] != words:
            raise SystemExit(f"FATAL: {name}: {js.name} does not match the word list")
        shift = np.load(npz)["shift"].astype(np.float64).sum(axis=2) / X106_HARD1K_SD
        for row, pi in enumerate(kj["prompt_indices_scanned"]):
            key_by_pid[kj["prompt_ids"][pi]] = shift[row]
        key_checks[tag] = {"n_prompts_scanned": kj["n_prompts_scanned"], "o_cached_vs_full_max_abs_per_token": kj["o_cached_vs_full_max_abs_per_token"]}
    key = np.stack([key_by_pid[p] for p in pids]) if all(p in key_by_pid for p in pids) else None

    return {
        "run": name,
        "prompts": rows,
        "pids": pids,
        "pops": np.asarray([p["population"] for p in rows]),
        "hist": np.asarray([p["historical_refusal2"] for p in rows]),
        "LO": LO,
        "LE": LE,
        "key": key,
        "key_checks": key_checks,
        "n_cells_expected": P * (W + 1),
        "n_cells_generated": sum(1 for pid in pids for c in [f"{pid}|O"] + [f"{pid}|E_{w}" for w in words] if c in gen),
        "n_cells_retry4096": sum(1 for pid in pids for c in [f"{pid}|O"] + [f"{pid}|E_{w}" for w in words] if c in gen and gen[c]["used_retry"]),
        "missing_generations": sorted(set(missing_gen)),
        "missing_refusal2": sorted(set(missing_ref)),
        "n_refusal2_units": sum(1 for u in ref if u.rsplit("#h", 1)[0] in gen),
        "prompts_sha256": obj["prompts_sha256"],
        "excluded_ids": excluded,
        "n_topup": len(topup),
        "n_by_population": {q: int(sum(1 for p in rows if p["population"] == q)) for q in POPULATIONS},
    }


def analyse(runs: list[dict], words_obj: dict, suffix: str, cohort: str, out_dir: Path) -> dict:
    chosen = words_obj["chosen"]
    cw = [c["word"] for c in chosen]
    tz = np.load(TABLES_NPZ)
    tmeta = read_json(TABLES_JSON)
    tab = {t: np.asarray([float(tz[t][c["index"]]) for c in chosen]) for t in TABLES}
    for c in chosen:
        for t in ("benign", "BP", "HP"):
            if abs(float(tz[t][c["index"]]) - c[f"{t}_sd"]) > 1e-5:
                raise SystemExit(f"FATAL: words.json table value of {c['word']} differs from tables.npz")
    kind = np.asarray([c["kind"] for c in chosen])
    bins = np.asarray([c["bin"] if c["bin"] is not None else -1 for c in chosen])
    conc = kind == "concordant"
    mid = np.isin(bins, MIDDLE_BINS)
    n_tok = np.asarray([c["n_tokens"] for c in chosen], float)
    log_wf = np.log(np.asarray([c["wordfreq_rank"] if c["wordfreq_rank"] else 50_001 for c in chosen], float))

    LO = np.concatenate([r["LO"] for r in runs], axis=0)
    LE = np.concatenate([r["LE"] for r in runs], axis=1)
    pops = np.concatenate([r["pops"] for r in runs])
    hist = np.concatenate([r["hist"] for r in runs])
    pids = [p for r in runs for p in r["pids"]]
    run_of = np.asarray([r["run"] for r in runs for _ in r["pids"]])
    key = np.concatenate([r["key"] for r in runs], axis=0) if all(r["key"] is not None for r in runs) else None
    P, W = len(pids), len(cw)

    rO = LO.mean(1)
    rE = LE.mean(2)  # [W, P]
    d = rE - rO[None]
    masks = {"benign": pops == "benign", "harmful": pops == "harmful", "pooled": np.ones(P, bool)}

    rng = np.random.default_rng(BOOT_SEED)
    boot_idx = {}
    for s, m in masks.items():
        sub = np.where(m)[0]
        if s == "pooled":
            idx = np.concatenate([np.where(masks[q])[0][rng.integers(0, masks[q].sum(), size=(N_BOOT, masks[q].sum()))] for q in POPULATIONS], axis=1)
        else:
            idx = sub[rng.integers(0, len(sub), size=(N_BOOT, len(sub)))]
        boot_idx[s] = idx

    def bmean(M: np.ndarray, s: str) -> np.ndarray:
        return np.moveaxis(np.nanmean(M[..., boot_idx[s]], axis=-1), -1, 0)

    r_O = {s: {"run": float(np.nanmean(rO[m])), "historical": float(hist[m].mean()), "spearman_run_vs_historical": rho(hist[m], rO[m]), "n_prompts": int(m.sum())} for s, m in masks.items()}

    per_word = []
    dEO_boot = {s: bmean(d, s) for s in SETS}
    for k, c in enumerate(chosen):
        rec = {kk: c[kk] for kk in ("word", "stratum", "kind", "bin_name", "quadrant", "benign_sd", "BP_sd", "HP_sd", "pct_benign", "pct_BP", "pct_HP", "n_tokens", "wordfreq_rank")}
        rec["mixed_sd"] = float(tab["mixed"][k])
        rec["pct_mixed"] = float(tz["pct_mixed"][c["index"]])
        for s, m in masks.items():
            b = dEO_boot[s][:, k]
            rec[s] = {"r_E": float(np.nanmean(rE[k][m])), "delta_EO": float(np.nanmean(d[k][m])), "delta_EO_ci95": ci(b), "delta_EO_p": boot_p(b)}
            if key is not None:
                rec[s]["key_shift_sd"] = float(key[m, k].mean())
        per_word.append(rec)
    for s in SETS:
        for pw, v in zip(per_word, holm([pw[s]["delta_EO_p"] for pw in per_word])):
            pw[s]["delta_EO_p_holm40"] = v

    dose = {}
    for s, m in masks.items():
        rows, bin_boot = [], []
        for b in range(len(BIN_NAMES)):
            sel = conc & (bins == b)
            bb = dEO_boot[s][:, sel].mean(1)
            bin_boot.append(bb)
            rows.append({"bin": BIN_NAMES[b], "words": [cw[k] for k in np.where(sel)[0]], **{f"{t}_sd_mean": float(tab[t][sel].mean()) for t in TABLES}, "r_E": float(np.nanmean(rE[sel][:, m])), "delta_EO": float(np.nanmean(d[sel][:, m])), "delta_EO_ci95": ci(bb), "delta_EO_p": boot_p(bb)})
        est = np.asarray([r["delta_EO"] for r in rows])
        BB = np.stack(bin_boot, 1)
        dose[s] = {
            "bins": rows,
            "adjacent_pairs_in_order": int(np.sum(np.diff(est) > 0)),
            "adjacent_pairs_total": len(BIN_NAMES) - 1,
            "fraction_adjacent_in_order": float(np.mean(np.diff(est) > 0)),
            "fraction_adjacent_in_order_boot_mean": float(np.mean(np.diff(BB, axis=1) > 0)),
            "spearman_bin_index_vs_delta": rho(np.arange(len(BIN_NAMES)), est),
            "top_minus_bottom_bin": float(est[-1] - est[0]),
            "top_minus_bottom_bin_ci95": ci(BB[:, -1] - BB[:, 0]),
            "middle_four_bins_range": float(est[list(MIDDLE_BINS)].max() - est[list(MIDDLE_BINS)].min()),
        }

    predictors = {f"table_{t}": tab[t] for t in TABLES}
    if key is not None:
        predictors["key_in_context"] = None
    subsets = {"all40": np.ones(W, bool), "concordant24": conc, "middle4bins": conc & mid}
    rho_table = {}
    for s, m in masks.items():
        dEO = np.nanmean(d[:, m], axis=1)
        rho_table[s] = {}
        for name in predictors:
            x = key[m].mean(0) if name == "key_in_context" else predictors[name]
            entry = {}
            for sub, sm in subsets.items():
                xb = x[sm]
                Yb = dEO_boot[s][:, sm]
                okb = np.all(np.isfinite(Yb), axis=1)
                entry[sub] = {"n_words": int(sm.sum()), "rho": rho(xb, dEO[sm]), "ci95_prompt_bootstrap": ci(rho_rows(xb, Yb[okb])), "perm_p": perm_p(xb, dEO[sm])}
            entry["partial_rho_all40_given_ntokens_logfreq"] = partial_rho(x, dEO, [n_tok, log_wf])
            xc = key[m].T if name == "key_in_context" else np.repeat(x[:, None], m.sum(), axis=1)
            dc = d[:, m]
            ok = np.isfinite(dc)
            entry["per_cell"] = {"n_cells": int(ok.sum()), "rho": rho(xc[ok], dc[ok])}
            rho_table[s][name] = entry

    ceiling = {}
    for s, m in masks.items():
        LEs, LOs = LE[:, m], LO[m]
        ok = np.isfinite(LEs).all(2) & np.isfinite(LOs).all(1)[None]
        hh_s = []
        for seed in HALF_SEEDS:
            perm = np.random.default_rng(seed).permutation(N_SAMPLES)
            a, b = perm[: N_SAMPLES // 2], perm[N_SAMPLES // 2 :]
            d1 = LEs[..., a].mean(-1) - LOs[None, :, a].mean(-1)
            d2 = LEs[..., b].mean(-1) - LOs[None, :, b].mean(-1)
            hh_s.append(rho(d1[ok], d2[ok]))
        r10s = float(np.mean(hh_s))
        rel_s = 2 * r10s / (1 + r10s)
        ceiling[s] = {"n_cells": int(ok.sum()), "half_half_spearman_10_samples": r10s, "reliability_20_samples_spearman_brown": rel_s, "ceiling_spearman": float(np.sqrt(max(rel_s, 0))), "n_half_splits": len(HALF_SEEDS)}

    n_pop = {q: int(masks[q].sum()) for q in POPULATIONS}
    table_choice = {"rule": f"Spearman rho between the table's per-word shift and the per-word Delta_EO over the 40 words; Delta_EO averaged over all {P} test prompts (pooled) and separately over each population ({', '.join(f'{q} {n}' for q, n in n_pop.items())}); CI from the prompt-level bootstrap (pooled: resampled within each population), p from 10,000 permutations of Delta_EO over the words"}
    for t in TABLES:
        e = rho_table["pooled"][f"table_{t}"]["all40"]
        table_choice[t] = {
            "rho": e["rho"],
            "ci": e["ci95_prompt_bootstrap"],
            "perm_p": e["perm_p"],
            "rho_concordant24": rho_table["pooled"][f"table_{t}"]["concordant24"]["rho"],
            "rho_benign_prompts": rho_table["benign"][f"table_{t}"]["all40"]["rho"],
            "ci_benign_prompts": rho_table["benign"][f"table_{t}"]["all40"]["ci95_prompt_bootstrap"],
            "rho_harmful_prompts": rho_table["harmful"][f"table_{t}"]["all40"]["rho"],
            "ci_harmful_prompts": rho_table["harmful"][f"table_{t}"]["all40"]["ci95_prompt_bootstrap"],
        }
    table_choice["best_pooled"] = max(TABLES, key=lambda t: table_choice[t]["rho"])
    table_choice["best_benign_prompts"] = max(TABLES, key=lambda t: table_choice[t]["rho_benign_prompts"])
    table_choice["best_harmful_prompts"] = max(TABLES, key=lambda t: table_choice[t]["rho_harmful_prompts"])

    results = {
        "run": "x117",
        "cohort": cohort,
        "runs": [
            {
                "run": r["run"],
                "prompts_sha256": r["prompts_sha256"],
                "n_prompts": len(r["pids"]),
                "n_by_population": r["n_by_population"],
                "n_excluded_x85": len(r["excluded_ids"]),
                "excluded_x85_ids": r["excluded_ids"],
                "n_topup_prompts": r["n_topup"],
                "n_cells_generated": r["n_cells_generated"],
                "n_cells_expected": r["n_cells_expected"],
                "n_cells_retry4096": r["n_cells_retry4096"],
                "n_refusal2_units": r["n_refusal2_units"],
                "missing_generations": r["missing_generations"],
                "missing_refusal2": r["missing_refusal2"],
                "key_scan_checks": r["key_checks"],
            }
            for r in runs
        ],
        "construct": "refusal2 (an explicit refusal statement is present), x64 judge, DeepSeek V4 Flash; twenty samples per cell judged as two units of ten",
        "words": {"file": str(WORDS_FILE.relative_to(ROOT)), "n": W, "chosen_sha256": words_obj["chosen_sha256"], "source_run": "x111"},
        "tables": {"file": str(TABLES_JSON.relative_to(ROOT)), "names": list(TABLES), "note": "x111's fixed tables, unchanged"},
        "n_prompts": P,
        "n_prompts_by_population": n_pop,
        "n_words": W,
        "table_choice": table_choice,
        "r_O": r_O,
        "per_word": per_word,
        "dose_response": dose,
        "rho": rho_table,
        "noise_ceiling_per_cell": ceiling,
        "table_correlations_eligible": tmeta["table_correlations"],
        "rate_matrix": {"words": cw, "prompt_ids": pids, "runs": run_of.tolist(), "populations": pops.tolist(), "r_E": rE.tolist(), "r_O": rO.tolist(), "historical_r_O": hist.tolist(), "key_shift_sd": key.tolist() if key is not None else None},
        "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED, "unit": "prompts resampled with replacement within each population (pooled: within each population, then combined), paired across conditions; words fixed; two-sided bootstrap p", "permutation": f"{N_PERM} permutations of Delta_EO over words, two-sided on |rho|"},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(out_dir / f"results{suffix}.json", results)
    write_md(results, out_dir / f"results{suffix}.md")
    print((out_dir / f"results{suffix}.md").read_text())
    return results


def write_md(r: dict, path: Path) -> None:
    L = [f"# x117: x111's four vocabulary tables against the change in `refusal2` ({r['cohort']})\n"]
    L.append("Runs: " + "; ".join(f"{q['run']} {q['n_prompts']} prompts ({', '.join(f'{k} {v}' for k, v in q['n_by_population'].items())})" for q in r["runs"]) + ".\n")
    L.append(f"{r['n_prompts']} test prompts ({', '.join(f'{k} {v}' for k, v in r['n_prompts_by_population'].items())}); {r['n_words']} words (x111's, 24 concordant over eight bins of the joint benign / BP percentile and 16 discordant) appended at the end; twenty samples per cell; `refusal2` (x64 judge). CIs: prompt-level paired bootstrap, {r['bootstrap']['n']} draws. Missing generations: {sum(len(q['missing_generations']) for q in r['runs'])}; missing refusal2 cells: {sum(len(q['missing_refusal2']) for q in r['runs'])}; cells regenerated at 4,096 tokens: {sum(q['n_cells_retry4096'] for q in r['runs'])}.\n")
    tc = r["table_choice"]
    L.append("## Table choice (ρ(table shift, Δ_EO) over the 40 words)\n")
    L.append(f"| table | pooled {r['n_prompts']} prompts ρ [95% CI] | perm p | ρ concordant 24 | benign prompts ρ [95% CI] | harmful prompts ρ [95% CI] |\n|---|---|---|---|---|---|")
    for t in TABLES:
        v = tc[t]
        L.append(f"| {t} | {f(v['rho'], 2)} {cis(v['ci'], 2)} | {v['perm_p']:.4f} | {f(v['rho_concordant24'], 2)} | {f(v['rho_benign_prompts'], 2)} {cis(v['ci_benign_prompts'], 2)} | {f(v['rho_harmful_prompts'], 2)} {cis(v['ci_harmful_prompts'], 2)} |")
    L.append(f"\nHighest: pooled {tc['best_pooled']}, benign prompts {tc['best_benign_prompts']}, harmful prompts {tc['best_harmful_prompts']}.\n")
    L.append("## r(O)\n")
    L.append("| prompts | n | r(O) this run | historical (ten samples) | Spearman run vs historical |\n|---|---|---|---|---|")
    for s, v in r["r_O"].items():
        L.append(f"| {s} | {v['n_prompts']} | {v['run']:.3f} | {v['historical']:.3f} | {f(v['spearman_run_vs_historical'], 2)} |")
    L.append("\n## Dose-response over the eight concordant bins (Δ_EO, mean over the bin's three words)\n")
    L.append("| bin | words | benign sd | BP sd | HP sd | mixed sd | " + " | ".join(f"{s} Δ_EO [95% CI]" for s in SETS) + " |\n|---|---|---|---|---|---|" + "---|" * len(SETS))
    for b in range(len(BIN_NAMES)):
        row = r["dose_response"]["pooled"]["bins"][b]
        L.append(f"| {row['bin']} | {', '.join(row['words'])} | {row['benign_sd_mean']:+.2f} | {row['BP_sd_mean']:+.2f} | {row['HP_sd_mean']:+.2f} | {row['mixed_sd_mean']:+.2f} | " + " | ".join(f"{f(r['dose_response'][s]['bins'][b]['delta_EO'])} {cis(r['dose_response'][s]['bins'][b]['delta_EO_ci95'])}" for s in SETS) + " |")
    for s in SETS:
        v = r["dose_response"][s]
        L.append(f"\n- {s}: adjacent bin pairs in order {v['adjacent_pairs_in_order']}/{v['adjacent_pairs_total']} (bootstrap mean fraction {v['fraction_adjacent_in_order_boot_mean']:.2f}); Spearman bin index vs Δ_EO {f(v['spearman_bin_index_vs_delta'], 2)}; top minus bottom bin {f(v['top_minus_bottom_bin'])} {cis(v['top_minus_bottom_bin_ci95'])}; range over the middle four bins {v['middle_four_bins_range']:.3f}.")
    L.append("\n## Spearman ρ(predictor, per-word Δ_EO)\n")
    L.append("| prompts | predictor | ρ all 40 [95% CI] | perm p | ρ concordant 24 [95% CI] | perm p | ρ middle 4 bins (12) | partial ρ (tokens, log freq) | per-cell ρ | per-cell ceiling |\n|---|---|---|---|---|---|---|---|---|---|")
    for s in SETS:
        for name, e in r["rho"][s].items():
            a, c, mm = e["all40"], e["concordant24"], e["middle4bins"]
            L.append(f"| {s} | {name} | {f(a['rho'], 2)} {cis(a['ci95_prompt_bootstrap'], 2)} | {a['perm_p']:.4f} | {f(c['rho'], 2)} {cis(c['ci95_prompt_bootstrap'], 2)} | {c['perm_p']:.4f} | {f(mm['rho'], 2)} | {f(e['partial_rho_all40_given_ntokens_logfreq'], 2)} | {f(e['per_cell']['rho'], 2)} (n = {e['per_cell']['n_cells']}) | {r['noise_ceiling_per_cell'][s]['ceiling_spearman']:.2f} |")
    L.append("\n## Per word\n")
    L.append("| word | stratum | benign sd | BP sd | HP sd | mixed sd | tok | " + " | ".join(f"{s} Δ_EO [95% CI] | {s} Holm p" for s in SETS) + " | key sd (pooled) |\n|---|---|---|---|---|---|---|" + "---|---|" * len(SETS) + "---|")
    for w in r["per_word"]:
        L.append(f"| {w['word']} | {w['stratum']} | {w['benign_sd']:+.2f} | {w['BP_sd']:+.2f} | {w['HP_sd']:+.2f} | {w['mixed_sd']:+.2f} | {w['n_tokens']} | " + " | ".join(f"{f(w[s]['delta_EO'])} {cis(w[s]['delta_EO_ci95'])} | {w[s]['delta_EO_p_holm40']:.4f}" for s in SETS) + f" | {f(w['pooled'].get('key_shift_sd'), 2)} |")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="x117", help="comma-separated run names: x117 (default), x111, or x111,x117 for the pooled cohort")
    a = ap.parse_args()
    names = [n.strip() for n in a.runs.split(",") if n.strip()]
    words_obj = read_json(WORDS_FILE)
    cw = [c["word"] for c in words_obj["chosen"]]
    loaded = [load_run(n, cw) for n in names]
    # Every output of this script lives under data/runs/x117, whatever it reads: `results.json` for
    # x117 alone, `results_check_<run>.json` for another run on its own (the reproduction check of
    # that run's numbers, which its own analyze script owns) and `results_pooled_<runs>.json` for a
    # pooled cohort.
    if names == ["x117"]:
        suffix, cohort = "", "x117's 64 test prompts (32 benign + 32 harmful), which chose none of the four tables"
    elif len(names) == 1:
        suffix, cohort = f"_check_{names[0]}", {"x111": "x111's analysed cohort: 32 benign + 32 harmful prompts (x85 exclusions applied, harmful stratum topped up)"}.get(names[0], names[0])
    else:
        suffix, cohort = "_pooled_" + "_".join(names), " ∪ ".join(names) + " pooled: " + ", ".join(f"{r['run']} {len(r['pids'])}" for r in loaded)
    analyse(loaded, words_obj, suffix, cohort, run_dir("x117"))
