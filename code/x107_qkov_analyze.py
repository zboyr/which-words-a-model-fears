"""x107 analysis for x107_qkov_extract.py: who reads the appended word on
Qwen3.5-4B, is the harm word attended more than its neutral control (QK), or attended
equally with a different value content (OV), and does either track the x85 refusal change."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

TAIL_SHORT = ["im_end", "nl", "im_start", "assistant", "nl2", "think", "nlnl", "/think", "tbg"]


def two_way_bootstrap(values: np.ndarray, p_idx: np.ndarray, w_idx: np.ndarray, fn, n_rep: int, rng) -> tuple:
    """Cluster bootstrap over prompts x words of a statistic fn(mask) computed on resampled cells."""
    up, uw = np.unique(p_idx), np.unique(w_idx)
    cells = {}
    for i, (p, w) in enumerate(zip(p_idx, w_idx)):
        cells.setdefault((int(p), int(w)), []).append(i)
    out = []
    for _ in range(n_rep):
        ps = rng.choice(up, size=len(up), replace=True)
        ws = rng.choice(uw, size=len(uw), replace=True)
        idx = []
        for p in ps:
            for w in ws:
                idx.extend(cells.get((int(p), int(w)), []))
        idx = np.asarray(idx, dtype=int)
        out.append(fn(values[idx]))
    out = np.asarray(out)
    return float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))


def residualize(y: np.ndarray, p_idx: np.ndarray, w_idx: np.ndarray) -> np.ndarray:
    up = {v: i for i, v in enumerate(np.unique(p_idx))}
    uw = {v: i for i, v in enumerate(np.unique(w_idx))}
    X = np.zeros((len(y), len(up) + len(uw)))
    for i, (p, w) in enumerate(zip(p_idx, w_idx)):
        X[i, up[p]] = 1.0
        X[i, len(up) + uw[w]] = 1.0
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def anova_shares(grid: np.ndarray) -> dict:
    """grid [prompt, word]: variance shares of the prompt effect, word effect and interaction."""
    g = grid - grid.mean()
    pe = g.mean(1, keepdims=True)
    we = g.mean(0, keepdims=True)
    inter = g - pe - we
    tot = (g**2).sum()
    return {
        "prompt": float((pe**2).sum() * grid.shape[1] / tot),
        "word": float((we**2).sum() * grid.shape[0] / tot),
        "interaction": float((inter**2).sum() / tot),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="dir with x107_qkov.npz and x107_qkov_meta.json")
    ap.add_argument("--x85", default="data/runs/x85")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-rep", type=int, default=2000)
    args = ap.parse_args()
    run = Path(args.run)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(run / "x107_qkov.npz")
    meta = json.loads((run / "x107_qkov_meta.json").read_text())
    layers = meta["attention_layers"]
    L, H = len(layers), meta["n_heads"]
    pi, wi = z["prompt_index"].astype(int), z["word_index"].astype(int)
    n = len(pi)
    rng = np.random.default_rng(1072)
    words = [x[1] for x in meta["word_pairs"]]
    neutrals = [x[2] for x in meta["word_pairs"]]
    rep: dict = {"meta": {k: meta[k] for k in ("n_pairs", "limit", "sanity", "elapsed_s", "mu_norms")}}
    lines = [f"# QK-vs-OV pilot on Qwen3.5-4B, x85 grid ({n} pairs)\n", f"Sanity: `{json.dumps(meta['sanity'])}`\n"]

    # 1. boundary projection on the consensus direction
    bH, bN = z["bproj"][:, 0].sum(-1), z["bproj"][:, 1].sum(-1)
    hn = z["hn_proj"].sum(-1)
    ci = two_way_bootstrap(hn, pi, wi, np.mean, args.n_rep, rng)
    rep["boundary"] = {"proj_H_minus_O": float(bH.mean()), "proj_N_minus_O": float(bN.mean()), "proj_H_minus_N": float(hn.mean()), "H_minus_N_ci": ci}
    lines.append(f"## Boundary projection (block 29, consensus direction, two tokens summed)\n\nH − O {bH.mean():.2f}, N − O {bN.mean():.2f}, H − N {hn.mean():.2f} [{ci[0]:.2f}, {ci[1]:.2f}]\n")

    # 2. direct-read share: heads' writes from the word span into the boundary tokens, layers <= 29
    keep = [i for i, l in enumerate(layers) if l <= meta["hook_block"]]
    cH = z["cproj"][:, 0][:, keep].sum(-1)  # [n, L', H] summed over the two tokens
    cN = z["cproj"][:, 1][:, keep].sum(-1)
    dc = cH - cN
    direct_total = dc.sum((1, 2))
    share = direct_total.mean() / hn.mean()
    per_layer = {str(layers[i]): float(dc[:, j].sum(-1).mean()) for j, i in enumerate(keep)}
    rep["direct_share"] = {"direct_H_minus_N": float(direct_total.mean()), "share_of_boundary_H_minus_N": float(share), "per_layer": per_layer,
                           "direct_H_minus_O_like": float(cH.sum((1, 2)).mean()), "direct_N": float(cN.sum((1, 2)).mean())}
    lines.append("## Direct attention reads of the word into the boundary tokens (layers 3–27, all heads)\n")
    lines.append(f"Σ_heads ⟨c_H − c_N, μ⟩ = {direct_total.mean():.2f}, i.e. {share:.0%} of the boundary H − N shift along μ; by layer: " + ", ".join(f"L{k} {v:.2f}" for k, v in per_layer.items()) + "\n")
    lines.append(f"Σ_heads ⟨c_H, μ⟩ = {cH.sum((1, 2)).mean():.2f}, Σ_heads ⟨c_N, μ⟩ = {cN.sum((1, 2)).mean():.2f}; "
                 f"mean Σ_heads ‖c‖ at the boundary tokens H {z['cnorm'][:, 0][:, keep][:, :, :, 7:].sum((1, 2)).mean(0).round(2).tolist()}, N {z['cnorm'][:, 1][:, keep][:, :, :, 7:].sum((1, 2)).mean(0).round(2).tolist()} "
                 f"(boundary shift norms ‖h_H − h_O‖ {z['bshift_norm'][:, 0].mean(0).round(1).tolist()})\n")

    # 2b. layer profile of the boundary H − N shift: token-mixer and MLP writes along mu, per layer
    if "mixer_proj" in z.files:
        mixH, mixN = z["mixer_proj"][:, 0].sum(-1), z["mixer_proj"][:, 1].sum(-1)  # [n, 32]
        mlpH, mlpN = z["mlp_proj"][:, 0].sum(-1), z["mlp_proj"][:, 1].sum(-1)
        dmix, dmlp = (mixH - mixN).mean(0), (mlpH - mlpN).mean(0)
        n_layers = dmix.shape[0]
        upto = meta["hook_block"] + 1
        attn_set = set(layers)
        tot_attn = float(sum(dmix[i] for i in range(upto) if i in attn_set))
        tot_delta = float(sum(dmix[i] for i in range(upto) if i not in attn_set))
        tot_mlp = float(dmlp[:upto].sum())
        recon = tot_attn + tot_delta + tot_mlp
        rep["layer_profile"] = {"attention_mixers": tot_attn, "deltanet_mixers": tot_delta, "mlps": tot_mlp, "sum": recon, "boundary_H_minus_N": float(hn.mean()),
                                "per_layer_mixer": [float(x) for x in dmix], "per_layer_mlp": [float(x) for x in dmlp],
                                "embedding_residual": float(hn.mean() - recon)}
        lines.append("## Where the boundary H − N shift along μ is written (layers 0–29; the sum of the writes reproduces the block-29 difference up to the embedding term)\n")
        lines.append(f"| writer | Σ ⟨write_H − write_N, μ⟩ |\n|---|---|\n| 7 softmax-attention mixers (L3…L27) | {tot_attn:+.2f} |\n| 23 Gated DeltaNet mixers | {tot_delta:+.2f} |\n| 30 MLPs | {tot_mlp:+.2f} |\n| sum | {recon:+.2f} (block-29 difference {hn.mean():+.2f}) |\n")
        prof = [f"L{i}{'*' if i in attn_set else ''} mix {dmix[i]:+.2f} / mlp {dmlp[i]:+.2f}" for i in range(upto)]
        lines.append("Per layer (* = softmax attention): " + "; ".join(prof) + "\n")
        # attention-layer writes: whole module output vs the word-span-only part
        word_part = {str(l): float((z["cproj"][:, 0, i].sum((1, 2)) - z["cproj"][:, 1, i].sum((1, 2))).mean()) for i, l in enumerate(layers) if l <= meta["hook_block"]}
        lines.append("Attention mixers, whole output vs the part read from the word span: " + "; ".join(f"L{l} {dmix[int(l)]:+.2f} vs {word_part[l]:+.2f}" for l in word_part) + "\n")
        rep["layer_profile"]["attention_word_part"] = word_part

    # 2c. inheritance inside the tail: attention from the boundary tokens to the earlier tail tokens
    if "t2t" in z.files:
        t2t = z["t2t"].astype(np.float32)  # [n, 2, L, H, 9, 9]
        lines.append("## Attention inside the tail (base texts, H): from each later tail token to the earlier ones, summed over all 128 heads\n")
        lines.append("| query \\ key | " + " | ".join(TAIL_SHORT) + " |\n|---|" + "---|" * 9)
        inh = t2t[:, 0].sum((1, 2)).mean(0)  # [9, 9]
        for j in range(9):
            lines.append(f"| {TAIL_SHORT[j]} | " + " | ".join(f"{inh[j, k]:.2f}" if k <= j else "" for k in range(9)) + " |")
        rep["tail_inheritance_H_sum_heads"] = inh.tolist()
        # which earlier tail token is the main source for the boundary tokens, and does it differ H vs N
        dinh = (t2t[:, 0] - t2t[:, 1]).sum((1, 2)).mean(0)
        lines.append("\nH − N of the same (sum over heads): boundary tokens → im_end " + f"{dinh[7, 0]:+.2f}, {dinh[8, 0]:+.2f}; → nl {dinh[7, 1]:+.2f}, {dinh[8, 1]:+.2f}; → im_start {dinh[7, 2]:+.2f}, {dinh[8, 2]:+.2f}\n")

    # 3. who reads the word: attention mass per tail token
    aH, aN = z["a"][:, 0], z["a"][:, 1]  # [n, L, H, 9]
    lines.append("## Attention mass to the word's tokens, by tail query token\n")
    lines.append("| tail token | mean over heads H / N | max head (mean over cells) H | that head's N | Δ = H − N for that head |\n|---|---|---|---|---|")
    reader = {}
    for j, name in enumerate(TAIL_SHORT):
        mH = aH[:, :, :, j].mean(0)  # [L, H]
        mN = aN[:, :, :, j].mean(0)
        li, hi = np.unravel_index(np.argmax(mH), mH.shape)
        reader[name] = {"mean_H": float(mH.mean()), "mean_N": float(mN.mean()), "max_head": f"L{layers[li]}.H{hi}", "max_H": float(mH[li, hi]), "max_head_N": float(mN[li, hi])}
        lines.append(f"| {name} | {mH.mean():.4f} / {mN.mean():.4f} | L{layers[li]}.H{hi} {mH[li, hi]:.3f} | {mN[li, hi]:.3f} | {mH[li, hi] - mN[li, hi]:+.3f} |")
    rep["reader"] = reader
    lines.append("")

    # 4. QK test per head: Δa = a_H − a_N, ranked, with the logit split (key side vs query side)
    sH, sN = z["s_own"][:, 0], z["s_own"][:, 1]
    xH, xN = z["s_cross"][:, 0], z["s_cross"][:, 1]  # xH = q_N·k_H ; xN = q_H·k_N
    key_side = xH - sN  # q_N fixed: k_H vs k_N
    query_side = sH - xH  # k_H fixed: q_H vs q_N
    da = aH - aN
    flat = da.mean(0)  # [L, H, 9]
    order = np.argsort(-np.abs(flat).reshape(-1))[:20]
    lines.append("## Heads with the largest |mean Δa| (H − N attention mass to the word), any tail token\n")
    lines.append("| head | tail | a_H | a_N | Δa [95% CI] | cells Δa>0 | Δlogit | key side | query side | cproj_H | cproj_N (boundary only) |\n|---|---|---|---|---|---|---|---|---|---|---|")
    top = []
    for flat_i in order:
        li, hi, j = np.unravel_index(flat_i, flat.shape)
        v = da[:, li, hi, j]
        ci = two_way_bootstrap(v, pi, wi, np.mean, 500, rng)
        cp = ""
        if j >= 7:
            cp = f"{z['cproj'][:, 0, li, hi, j - 7].mean():+.3f} | {z['cproj'][:, 1, li, hi, j - 7].mean():+.3f}"
        else:
            cp = "— | —"
        row = {"head": f"L{layers[li]}.H{hi}", "tail": TAIL_SHORT[j], "a_H": float(aH[:, li, hi, j].mean()), "a_N": float(aN[:, li, hi, j].mean()),
               "delta_a": float(v.mean()), "ci": ci, "frac_pos": float((v > 0).mean()), "delta_logit": float((sH - sN)[:, li, hi, j].mean()),
               "key_side": float(key_side[:, li, hi, j].mean()), "query_side": float(query_side[:, li, hi, j].mean())}
        top.append(row)
        lines.append(f"| {row['head']} | {row['tail']} | {row['a_H']:.3f} | {row['a_N']:.3f} | {row['delta_a']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | {row['frac_pos']:.2f} | {row['delta_logit']:+.2f} | {row['key_side']:+.2f} | {row['query_side']:+.2f} | {cp} |")
    rep["top_heads_delta_a"] = top
    lines.append("")
    # global QK summary: total mass to word from each tail token, H vs N
    tot = {}
    for j, name in enumerate(TAIL_SHORT):
        dH, dN = aH[:, :, :, j].sum((1, 2)), aN[:, :, :, j].sum((1, 2))
        d = dH - dN
        ci = two_way_bootstrap(d, pi, wi, np.mean, 500, rng)
        tot[name] = {"sum_H": float(dH.mean()), "sum_N": float(dN.mean()), "delta": float(d.mean()), "ci": ci, "frac_pos": float((d > 0).mean())}
    rep["total_mass_by_tail"] = tot
    lines.append("### Total mass to the word summed over all 128 heads, by tail token\n\n| tail | Σ a_H | Σ a_N | Δ [CI] | cells Δ>0 |\n|---|---|---|---|---|")
    for name, r in tot.items():
        lines.append(f"| {name} | {r['sum_H']:.3f} | {r['sum_N']:.3f} | {r['delta']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] | {r['frac_pos']:.2f} |")
    lines.append("")

    # 5. OV decomposition at the boundary tokens (symmetric split), layers <= 29
    cpH = z["cproj"][:, 0][:, keep]  # [n, L', H, 2]
    cpN = z["cproj"][:, 1][:, keep]
    aHb = aH[:, keep][:, :, :, 7:]  # [n, L', H, 2]
    aNb = aN[:, keep][:, :, :, 7:]
    with np.errstate(divide="ignore", invalid="ignore"):
        vH = np.where(aHb > 1e-9, cpH / aHb, 0.0)
        vN = np.where(aNb > 1e-9, cpN / aNb, 0.0)
    qk_term = (aHb - aNb) * (vH + vN) / 2
    ov_term = (aHb + aNb) / 2 * (vH - vN)
    total = cpH - cpN
    agg = {k: v.sum((1, 2, 3)) for k, v in {"total": total, "qk": qk_term, "ov": ov_term}.items()}
    rep["ov_decomposition"] = {}
    lines.append("## Boundary-token decomposition of Σ_heads ⟨c_H − c_N, μ⟩ into a QK term (mass differs, content averaged) and an OV term (content differs, mass averaged)\n")
    lines.append("| term | mean [95% CI] |\n|---|---|")
    for k, v in agg.items():
        ci = two_way_bootstrap(v, pi, wi, np.mean, args.n_rep, rng)
        rep["ov_decomposition"][k] = {"mean": float(v.mean()), "ci": ci}
        lines.append(f"| {k} | {v.mean():+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] |")
    # per-head view of the decomposition for the heads with the largest |total|
    per_head_total = total.sum(-1).mean(0)  # [L', H]
    order = np.argsort(-np.abs(per_head_total).reshape(-1))[:12]
    lines.append("\n### Heads with the largest direct write difference into the boundary tokens\n\n| head | ⟨c_H−c_N,μ⟩ | QK term | OV term | a_H (tbg−1, tbg) | a_N | ⟨v̄_H,μ⟩ | ⟨v̄_N,μ⟩ |\n|---|---|---|---|---|---|---|---|")
    ph = []
    for flat_i in order:
        li, hi = np.unravel_index(flat_i, per_head_total.shape)
        r = {"head": f"L{layers[keep[li]]}.H{hi}", "total": float(per_head_total[li, hi]), "qk": float(qk_term[:, li, hi].sum(-1).mean()), "ov": float(ov_term[:, li, hi].sum(-1).mean()),
             "a_H": [float(x) for x in aHb[:, li, hi].mean(0)], "a_N": [float(x) for x in aNb[:, li, hi].mean(0)],
             "vH": [float(x) for x in vH[:, li, hi].mean(0)], "vN": [float(x) for x in vN[:, li, hi].mean(0)]}
        ph.append(r)
        lines.append(f"| {r['head']} | {r['total']:+.3f} | {r['qk']:+.3f} | {r['ov']:+.3f} | {r['a_H'][0]:.3f}, {r['a_H'][1]:.3f} | {r['a_N'][0]:.3f}, {r['a_N'][1]:.3f} | {r['vH'][0]:+.1f}, {r['vH'][1]:+.1f} | {r['vN'][0]:+.1f}, {r['vN'][1]:+.1f} |")
    rep["ov_decomposition"]["per_head"] = ph
    lines.append("")

    # 6. behaviour: x85 refusal edges
    x85 = Path(args.x85)
    rates = {}
    for cond in ("H", "N"):
        for line in (x85 / f"x85_refusal_{cond}.jsonl").read_text().splitlines():
            r = json.loads(line)
            if "refusal" not in r or not r["refusal"]:
                continue
            rates[(cond, int(r["prompt_id"]), int(r["word_index"]))] = float(np.mean(r["refusal"]))
    pid = z["prompt_id"].astype(int)
    edge = np.array([(("H", int(p), int(w)) in rates) and (("N", int(p), int(w)) in rates) for p, w in zip(pid, wi)])
    dr = np.array([rates.get(("H", int(p), int(w)), np.nan) - rates.get(("N", int(p), int(w)), np.nan) for p, w in zip(pid, wi)])
    lines.append(f"## Behaviour on the {int(edge.sum())} x85 behaviour edges in this run (Δrefusal = r_H − r_N, mean {np.nanmean(dr[edge]):+.3f})\n")
    lines.append("Spearman with Δrefusal, raw and after prompt + word fixed effects:\n\n| predictor | ρ raw | ρ residualised |\n|---|---|---|")
    # reader head: the head with max mean a_H at the im_end token and the head with max |Δa|
    preds = {
        "boundary ⟨h_H − h_N, μ⟩ (x85 statistic)": hn,
        "Σ direct ⟨c_H − c_N, μ⟩ (layers ≤ 29)": agg["total"],
        "QK term": agg["qk"],
        "OV term": agg["ov"],
        "Σ_heads Δa, im_end": da[:, :, :, 0].sum((1, 2)),
        "Σ_heads Δa, tbg": da[:, :, :, 8].sum((1, 2)),
        "Σ_heads Δa, all tail tokens": da.sum((1, 2, 3)),
        "Σ_heads a_H, all tail tokens": aH.sum((1, 2, 3)),
        "Σ_heads ‖c_H‖ − ‖c_N‖, all tail": (z["cnorm"][:, 0] - z["cnorm"][:, 1]).sum((1, 2, 3)),
    }
    for row in top[:3]:
        li = layers.index(int(row["head"][1:].split(".")[0]))
        hi = int(row["head"].split(".H")[1])
        j = TAIL_SHORT.index(row["tail"])
        preds[f"Δa {row['head']} {row['tail']}"] = da[:, li, hi, j]
    rep["behaviour"] = {}
    if edge.sum() > 10:
        y = dr[edge]
        yr = residualize(y, pid[edge], wi[edge])
        for name, v in preds.items():
            x = v[edge]
            raw = stats.spearmanr(x, y).correlation
            res = stats.spearmanr(residualize(x, pid[edge], wi[edge]), yr).correlation
            rep["behaviour"][name] = {"rho_raw": float(raw), "rho_resid": float(res)}
            lines.append(f"| {name} | {raw:+.3f} | {res:+.3f} |")
    lines.append("")

    # 7. word level: per-word mean Δa (sum over heads, all tail) vs per-word mean Δrefusal and per-word hn
    lines.append("## Word level (24 words): mean over prompts\n\n| word / neutral | hn proj | Σ Δa all tail | Σ Δa im_end | direct total | Δrefusal (edges) |\n|---|---|---|---|---|---|")
    wl = []
    for w in range(24):
        m = wi == w
        if not m.any():
            continue
        e = m & edge
        r = {"word": words[w], "neutral": neutrals[w], "hn": float(hn[m].mean()), "da_all": float(da[m].sum((1, 2, 3)).mean()), "da_im_end": float(da[m][:, :, :, 0].sum((1, 2)).mean()),
             "direct": float(agg["total"][m].mean()), "dref": float(np.nanmean(dr[e])) if e.any() else float("nan")}
        wl.append(r)
        lines.append(f"| {r['word']} / {r['neutral']} | {r['hn']:+.2f} | {r['da_all']:+.3f} | {r['da_im_end']:+.3f} | {r['direct']:+.2f} | {r['dref']:+.3f} |")
    rep["word_level"] = wl
    if len(wl) >= 8:
        d = np.array([r["dref"] for r in wl])
        ok = ~np.isnan(d)
        for k in ("hn", "da_all", "da_im_end", "direct"):
            v = np.array([r[k] for r in wl])
            rep[f"word_rho_{k}_vs_dref"] = float(stats.spearmanr(v[ok], d[ok]).correlation)
            rep[f"word_rho_{k}_vs_hn"] = float(stats.spearmanr(v, np.array([r['hn'] for r in wl])).correlation)
        lines.append("\nWord-level Spearman vs Δrefusal: " + ", ".join(f"{k} {rep[f'word_rho_{k}_vs_dref']:+.2f}" for k in ("hn", "da_all", "da_im_end", "direct")))
        lines.append("Word-level Spearman vs hn proj: " + ", ".join(f"{k} {rep[f'word_rho_{k}_vs_hn']:+.2f}" for k in ("da_all", "da_im_end", "direct")) + "\n")

    # 8. context-independence: variance shares of a_H on the grid for the top reader heads
    if n == 96 * 24:
        lines.append("## Is the attention to the word set by the word or by the prompt? Variance shares on the 96 × 24 grid\n\n| head, tail | a_H: prompt / word / interaction | a_N: prompt / word / interaction | key side logit: prompt / word / inter |\n|---|---|---|---|")
        rep["variance_shares"] = []
        seen = set()
        for row in top:
            key = (row["head"], row["tail"])
            if key in seen or len(seen) >= 8:
                continue
            seen.add(key)
            li = layers.index(int(row["head"][1:].split(".")[0]))
            hi = int(row["head"].split(".H")[1])
            j = TAIL_SHORT.index(row["tail"])
            gH = np.zeros((96, 24)); gN = np.zeros((96, 24)); gK = np.zeros((96, 24))
            gH[pi, wi] = aH[:, li, hi, j]; gN[pi, wi] = aN[:, li, hi, j]; gK[pi, wi] = key_side[:, li, hi, j]
            sH_, sN_, sK_ = anova_shares(gH), anova_shares(gN), anova_shares(gK)
            rep["variance_shares"].append({"head": row["head"], "tail": row["tail"], "a_H": sH_, "a_N": sN_, "key_side": sK_})
            lines.append(f"| {row['head']}, {row['tail']} | {sH_['prompt']:.2f} / {sH_['word']:.2f} / {sH_['interaction']:.2f} | {sN_['prompt']:.2f} / {sN_['word']:.2f} / {sN_['interaction']:.2f} | {sK_['prompt']:.2f} / {sK_['word']:.2f} / {sK_['interaction']:.2f} |")
        lines.append("")

    (out / "x107_qkov_analysis.json").write_text(json.dumps(rep, indent=2, default=float))
    (out / "x107_qkov_analysis.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
