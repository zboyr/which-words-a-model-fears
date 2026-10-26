"""x116 Stage 5 (local): does the table-lowest synonym matter, against a random admissible one?

Per model, on the units with both an S_T (x113) and an R_rand (x116) text, x85-free by x113's
frozen exclusion (`--full` keeps every unit and writes the `_full` siblings): r(O) (x106 O_base
rows), r(S_T) (x113 refusal2 rows), r(R_rand); delta_S = r(O) - r(S_T), delta_R = r(O) -
r(R_rand), and the paired r(R_rand) - r(S_T), with x113's two-stage prompt-paired bootstrap
(10,000 replicates; prompts resampled within stratum, then the ten binary responses of each
arm), per population (core, llama_extra, hard1k, xstest, all). Holm over the seven transfer
models per contrast (Qwen3.5-4B the reference row): one-sided for delta_S > 0, delta_R > 0 and
r(R_rand) - r(S_T) > 0, and two-sided for r(R_rand) - r(S_T). Per unit, the table-shift gap
shift(R_rand) - shift(S_T) against r(R_rand) - r(S_T) (Spearman, unit bootstrap), and the
projection change of R_rand against x113's S_T on the same units.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x106_common import holm  # noqa: E402
from x113_analyze import interval, load_generation_rows, load_jsonl_cells, nanmean_rows, one_sided_p, rates_from_records, spearman, spearman_record, two_sided_p  # noqa: E402
from x113_common import EXCLUSIONS, load_prompts, x106_refusal_rows  # noqa: E402
from x113_common import cells as x113_cells, cells_sha256 as x113_cells_sha256, load_texts as load_x113_texts, paths as x113_paths  # noqa: E402
from x116_common import ALL_TAGS, BOOTSTRAP_SEED, FRESH_TAGS, IMPORTED_TAG, LABELS, N_BOOTSTRAP, N_RESPONSES, RUN, atomic_json, cells, cells_sha256, file_record, load_texts, paths, read_json  # noqa: E402

POPULATIONS = {"core": ("core",), "llama_extra": ("llama_extra",), "hard1k": ("core", "llama_extra"), "xstest": ("xstest",), "all": ("core", "llama_extra", "xstest")}
HEADLINE = "hard1k"
ORDER = (IMPORTED_TAG, "qwen08b", "qwen2b", "qwen9b", "qwen27b", "llama", "gemma", "phi")
KEYS = ("r_O", "r_S_T", "r_R_rand", "delta_S", "delta_R", "R_minus_S")


def estimands(r: dict[str, np.ndarray]) -> dict[str, Any]:
    o, s, q = r["O"], r["S_T"], r["R"]
    return {"r_O": nanmean_rows(o), "r_S_T": nanmean_rows(s), "r_R_rand": nanmean_rows(q), "delta_S": nanmean_rows(o - s), "delta_R": nanmean_rows(o - q), "R_minus_S": nanmean_rows(q - s)}


def bootstrap(r: dict[str, np.ndarray], strata: np.ndarray, n_boot: int, seed: int, chunk: int = 500):
    point = {k: float(v) for k, v in estimands(r).items()}
    rng = np.random.default_rng(seed)
    groups = [np.where(strata == s)[0] for s in np.unique(strata)]
    samples = {k: np.empty(n_boot) for k in point}
    for start in range(0, n_boot, chunk):
        stop = min(start + chunk, n_boot)
        index = np.concatenate([g[rng.integers(0, len(g), size=(stop - start, len(g)))] for g in groups], axis=1)
        drawn = {}
        for arm, values in r.items():
            picked = values[index]
            drawn[arm] = np.where(np.isfinite(picked), rng.binomial(N_RESPONSES, np.nan_to_num(picked, nan=0.0)) / N_RESPONSES, np.nan)
        rep = estimands(drawn)
        for k in point:
            samples[k][start:stop] = rep[k]
    return point, samples


def model_report(tag: str, prompts: dict, excluded: frozenset[str], n_boot: int, seed: int) -> dict:
    t113 = load_x113_texts(tag)
    t116 = load_texts(tag)
    u116 = {u["id"]: u for u in t116["units"]}
    grid113 = x113_cells(tag, t113)
    gen113 = load_generation_rows(tag, x113_cells_sha256(grid113)) if any(x113_paths(tag)["gen"].glob("*.json")) else {}
    ref113 = load_jsonl_cells(x113_paths(tag)["refusal2"])
    grid = cells(tag, t116)
    gsha = cells_sha256(grid)
    gen116: dict[str, dict] = {}
    for path in sorted(paths(tag)["gen"].glob("generations_shard*of*.json")):
        obj = read_json(path)
        if obj.get("cells_sha256") != gsha:
            raise SystemExit(f"FATAL: {path} is another grid")
        gen116.update({r["cell_id"]: r for r in obj["rows"]})
    ref116 = load_jsonl_cells(paths(tag)["refusal2"])
    x106_o = x106_refusal_rows(tag, "O_base")
    punits = {u["id"]: u for u in prompts["units"]}
    units = [u for u in t113["units"] if u["texts"]["S_T"] is not None and u116[u["id"]]["text"] is not None]
    ids = [u["id"] for u in units]
    strata = np.asarray([u["stratum"] for u in units])
    o = []
    for i in ids:
        rec = x106_o[i]
        if rec.get("prompt_sha256") != punits[i]["prompt_sha256"]:
            raise SystemExit(f"FATAL: {tag}: x106 O row {i} is not the prompt")
        o.append(float(np.mean(rec["refusal"])))
    r = {
        "O": np.asarray(o),
        "S_T": rates_from_records(ref113, "refusal", [f"{i}|S_T" for i in ids], {k: v["response_sha256"] for k, v in gen113.items()}),
        "R": rates_from_records(ref116, "refusal", [f"{i}|R_rand" for i in ids], {k: v["response_sha256"] for k, v in gen116.items()}),
    }
    gap = np.asarray([u116[i].get("shift_gap") if u116[i].get("shift_gap") is not None else np.nan for i in ids], dtype=float)
    proj113 = {row["id"]: row for row in read_json(x113_paths(tag)["projections"])["rows"]}
    proj116 = {row["id"]: row for row in read_json(paths(tag)["projections"])["rows"]} if paths(tag)["projections"].exists() else {}
    p_o = np.asarray([proj113[i]["proj_sd"]["O"] for i in ids])
    p_s = np.asarray([proj113[i]["proj_sd"].get("S_T", np.nan) for i in ids])
    p_r = np.asarray([proj116[i]["proj_sd"]["R_rand"] - proj116[i]["proj_sd"]["O"] + proj113[i]["proj_sd"]["O"] if i in proj116 else np.nan for i in ids])
    kept = np.asarray([i not in excluded for i in ids])
    t_units = t116["units"]
    report: dict[str, Any] = {
        "model_id": LABELS[tag],
        "arm_counts_all_units": {"n_units": len(t_units), "n_with_S_T": sum(1 for u in t113["units"] if u["texts"]["S_T"] is not None), "n_with_R_rand": t116["n_R_rand"], "n_without_R_rand": t116["n_without"], "pool_size_mean": t116["pool_size_mean"]},
        "arm_counts_x85_free": {"n_units": sum(1 for u in t_units if u["id"] not in excluded), "n_with_S_T": sum(1 for u in t113["units"] if u["id"] not in excluded and u["texts"]["S_T"] is not None), "n_with_R_rand": sum(1 for u in t_units if u["id"] not in excluded and u["text"] is not None), "n_S_T_without_R_rand": sum(1 for u in t_units if u["id"] not in excluded and u["text"] is None and u["no_arm_reason"] != "no S_T")},
        "n_labelled": {"S_T": int(np.isfinite(r["S_T"]).sum()), "R_rand": int(np.isfinite(r["R"]).sum()), "R_rand_generated": len(gen116)},
        "integrity": {"retry_rows": int(sum(bool(v.get("used_retry")) for v in gen116.values())), "truncation_rate": float(np.mean([bool(x) for v in gen116.values() for x in v["truncated"]])) if gen116 else None, "projection_O_vs_x113_max_abs_diff_sd": (read_json(paths(tag)["projections"]).get("o_vs_x113_max_abs_diff_sd") if proj116 else None)},
        "populations": {},
    }
    rng = np.random.default_rng(seed + 7)
    for k, (name, members) in enumerate(POPULATIONS.items()):
        mask = np.isin(strata, members) & kept & np.isfinite(r["S_T"]) & np.isfinite(r["R"])
        if mask.sum() < 2:
            continue
        sub = {a: v[mask] for a, v in r.items()}
        point, samples = bootstrap(sub, strata[mask], n_boot, seed + 10 * k)
        est = {key: {"estimate": point[key], "ci95": interval(samples[key]), "one_sided_p": one_sided_p(samples[key]), "two_sided_p": two_sided_p(samples[key]), "n": int(mask.sum())} for key in KEYS}
        diff = sub["R"] - sub["S_T"]
        pop: dict[str, Any] = {
            "n": int(mask.sum()),
            "estimands": est,
            "share_units_R_rand_above_S_T": float(np.mean(diff > 0)),
            "share_units_R_rand_below_S_T": float(np.mean(diff < 0)),
            "shift_gap": {"mean_sd": float(np.nanmean(gap[mask])), "median_sd": float(np.nanmedian(gap[mask])), "n_defined": int(np.isfinite(gap[mask]).sum()), "spearman_gap_vs_R_minus_S": spearman_record(gap[mask], diff, rng), "spearman_gap_vs_R_minus_S_O_refused": spearman_record(gap[mask][sub["O"] >= 0.5], diff[sub["O"] >= 0.5], rng)},
        }
        if proj116:
            ds, dr = p_s[mask] - p_o[mask], p_r[mask] - p_o[mask]
            boot = lambda d: interval(np.asarray([np.nanmean(d[rng.integers(0, len(d), len(d))]) for _ in range(2000)]))  # noqa: E731
            pop["representation"] = {
                "proj_S_T_minus_O_mean_sd": float(np.nanmean(ds)), "proj_S_T_minus_O_ci95": boot(ds),
                "proj_R_rand_minus_O_mean_sd": float(np.nanmean(dr)), "proj_R_rand_minus_O_ci95": boot(dr),
                "proj_R_rand_minus_S_T_mean_sd": float(np.nanmean(dr - ds)), "proj_R_rand_minus_S_T_ci95": boot(dr - ds),
                "share_R_rand_projection_above_S_T": float(np.nanmean((dr - ds) > 0)),
                "spearman_proj_gap_vs_R_minus_S": spearman_record(dr - ds, diff, rng),
                "spearman_shift_gap_vs_proj_gap": spearman_record(gap[mask], dr - ds, rng),
                "spearman_proj_change_vs_refusal_change_R_rand": spearman_record(dr, sub["R"] - sub["O"], rng),
            }
        report["populations"][name] = pop
    report["per_unit"] = [
        {"id": i, "stratum": str(strata[j]), "x85_excluded": bool(not kept[j]), "nominee": u116[i]["nominee"], "S_T_substitute": u116[i]["S_T_substitute"]["text"], "R_rand_substitute": u116[i]["substitute"]["text"], "shift_gap": (float(gap[j]) if np.isfinite(gap[j]) else None), "r_O": float(r["O"][j]), "r_S_T": (float(r["S_T"][j]) if np.isfinite(r["S_T"][j]) else None), "r_R_rand": (float(r["R"][j]) if np.isfinite(r["R"][j]) else None), "proj_S_T_minus_O_sd": (float(p_s[j] - p_o[j]) if np.isfinite(p_s[j]) else None), "proj_R_rand_minus_O_sd": (float(p_r[j] - p_o[j]) if np.isfinite(p_r[j]) else None)}
        for j, i in enumerate(ids)
    ]
    return report


def cross_model(per_model: dict) -> dict:
    out: dict[str, Any] = {}
    for pop in POPULATIONS:
        tags = [t for t in per_model if pop in per_model[t]["populations"]]
        transfer = [t for t in FRESH_TAGS if t in tags]
        table = {}
        for key in ("delta_S", "delta_R", "R_minus_S"):
            e = {t: per_model[t]["populations"][pop]["estimands"][key] for t in tags}
            one = dict(zip(transfer, holm([e[t]["one_sided_p"] or 1.0 for t in transfer]))) if transfer else {}
            two = dict(zip(transfer, holm([e[t]["two_sided_p"] or 1.0 for t in transfer]))) if transfer else {}
            table[key] = {t: {**e[t], "holm_one_sided_over_transfer": one.get(t), "holm_two_sided_over_transfer": two.get(t)} for t in tags}
        out[pop] = {"table": table, "transfer_models_in_family": transfer, "reference_model": IMPORTED_TAG, "mean_over_models": {k: float(np.mean([per_model[t]["populations"][pop]["estimands"][k]["estimate"] for t in tags])) for k in KEYS} if tags else None}
    return out


def fmt(e: dict) -> str:
    lo, hi = e["ci95"]
    return f"{e['estimate']:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def pfmt(p) -> str:
    return "ref" if p is None else (f"{p:.0e}" if p < 0.01 else f"{p:.3f}")


def write_md(per_model: dict, cm: dict, path: Path, full: bool) -> None:
    lines = [f"# x116 results{' (full population, no x85 exclusion)' if full else ' (x85-free units)'}", ""]
    for pop in ("hard1k", "core", "llama_extra", "xstest", "all"):
        if pop not in cm:
            continue
        tab = cm[pop]["table"]
        lines += [f"## {pop}", "", "| model | n | r(O) | r(S_T) | r(R_rand) | Δ_S [95% CI] | Δ_rand [95% CI] | Holm Δ_rand | r(R_rand) − r(S_T) [95% CI] | Holm one-sided | Holm two-sided | shift gap (sd) | ρ(gap, R−S) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for t in ORDER:
            if t not in tab["delta_S"]:
                continue
            p = per_model[t]["populations"][pop]
            e = p["estimands"]
            rho = p["shift_gap"]["spearman_gap_vs_R_minus_S"]
            rho_s = f"{rho['rho']:+.2f} [{rho['ci95'][0]:+.2f}, {rho['ci95'][1]:+.2f}]" if rho["rho"] is not None and rho["ci95"][0] is not None else "–"
            lines.append(f"| {LABELS[t]} | {p['n']} | {e['r_O']['estimate']:.3f} | {e['r_S_T']['estimate']:.3f} | {e['r_R_rand']['estimate']:.3f} | {fmt(e['delta_S'])} | {fmt(e['delta_R'])} | {pfmt(tab['delta_R'][t]['holm_one_sided_over_transfer'])} | {fmt(e['R_minus_S'])} | {pfmt(tab['R_minus_S'][t]['holm_one_sided_over_transfer'])} | {pfmt(tab['R_minus_S'][t]['holm_two_sided_over_transfer'])} | {p['shift_gap']['mean_sd']:+.2f} | {rho_s} |")
        lines.append("")
        if all("representation" in per_model[t]["populations"].get(pop, {}) for t in per_model if pop in per_model[t]["populations"]):
            lines += ["| model | proj(S_T) − proj(O) | proj(R_rand) − proj(O) | proj(R_rand) − proj(S_T) [95% CI] | ρ(proj gap, R−S) | ρ(shift gap, proj gap) |", "|---|---|---|---|---|---|"]
            for t in ORDER:
                if pop not in per_model.get(t, {}).get("populations", {}):
                    continue
                rep = per_model[t]["populations"][pop]["representation"]
                ci = rep["proj_R_rand_minus_S_T_ci95"]
                lines.append(f"| {LABELS[t]} | {rep['proj_S_T_minus_O_mean_sd']:+.2f} | {rep['proj_R_rand_minus_O_mean_sd']:+.2f} | {rep['proj_R_rand_minus_S_T_mean_sd']:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] | {rep['spearman_proj_gap_vs_R_minus_S']['rho']:+.2f} | {rep['spearman_shift_gap_vs_proj_gap']['rho']:+.2f} |")
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="*", default=list(ALL_TAGS))
    parser.add_argument("--n-boot", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    prompts = load_prompts()
    excl = read_json(EXCLUSIONS)
    per_model = {}
    for tag in args.tags:
        excluded = frozenset() if args.full else frozenset(e["id"] for e in excl["models"][tag]["excluded"])
        per_model[tag] = model_report(tag, prompts, excluded, args.n_boot, args.seed)
        e = per_model[tag]["populations"][HEADLINE]["estimands"]
        print(f"{tag:8s} n={per_model[tag]['populations'][HEADLINE]['n']:3d} dS {fmt(e['delta_S'])} dR {fmt(e['delta_R'])} R-S {fmt(e['R_minus_S'])}", flush=True)
    cm = cross_model(per_model)
    suffix = "_full" if args.full else ""
    out = {"experiment": "x116", "x85_exclusion_applied": not args.full, "exclusions": file_record(EXCLUSIONS), "n_boot": args.n_boot, "seed": args.seed, "per_model": per_model, "cross_model": cm, "texts": {t: file_record(paths(t)["texts"]) for t in args.tags}}
    atomic_json(RUN / f"results{suffix}.json", out)
    write_md(per_model, cm, RUN / f"results{suffix}.md", args.full)
    print(f"-> {RUN / f'results{suffix}.json'}")


if __name__ == "__main__":
    main()
