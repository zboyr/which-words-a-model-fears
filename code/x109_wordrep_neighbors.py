"""x109 word-representation probe, follow-up: are words that sit next to each other in the
vocabulary table (ranked by mean boundary shift) also close in embedding space?

The ridge in x109_wordrep_analyze.py reads the table order from one direction of the input
embedding (out-of-fold Spearman +0.78). That does not imply that table neighbours are embedding
neighbours: they share a projection on w and can be far apart in cosine. Two checks on
L2-normalised features (emb_mean primary; block7 and block29 word-alone states secondary):

1. Table neighbours -> cosine. Words sorted by y; for rank offsets d the mean and median cosine
   between rank i and rank i + d, against random pairs (fixed seed), whole table / true top 1 % /
   bottom 1 %; share of adjacent pairs with the same 5-letter stem and the cosine without them;
   per word, where the table-adjacent cosine falls inside its own cosine distribution.
2. Cosine neighbours -> shift. For each word its k nearest cosine neighbours (k = 1, 5, 20):
   Spearman between y and the neighbours' mean y, all neighbours / excluding same stem /
   excluding stem or common prefix >= 4; share of 1-NN pairs with the same stem; for the top 1 %
   the share whose 1-NN is also top 1 %.

Baseline: the ridge direction itself (lambda 1e3, in-sample on all words): Spearman of y with the
projection, and the share of embedding variance along it. Writes wordrep/neighbors.{json,md}.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x109_scan_common import SCAN_JSON, SCAN_NPZ, WORDS, X106_HARD1K_SD, atomic_json, read_json  # noqa: E402

OUT = SCAN_NPZ.parent / "wordrep"
FEATURES = ["emb_mean", "block7", "block29"]
OFFSETS = [1, 2, 5, 10, 50, 100, 1000]
KS = [1, 5, 20]
K_STORE = 60  # neighbours kept per word, so the filtered lists still reach k = 20
LAM = 1e3
SEED = 20261010
CHUNK = 2048
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))


def stem(w: str) -> str:
    return w[:5]


def common_prefix_len(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def targets() -> tuple[list[str], np.ndarray, np.ndarray]:
    scan = read_json(SCAN_JSON)
    words = scan["candidates"]["words"]
    shift = np.load(SCAN_NPZ)["shift"].astype(np.float64).sum(-1)  # [40, W]
    y = shift.mean(0) / X106_HARD1K_SD
    z = (shift - shift.mean(1, keepdims=True)) / shift.std(1, keepdims=True)
    return words, y, z.mean(0)


def load_unit(name: str) -> np.ndarray:
    X = np.load(OUT / "features.npz")[name].astype(np.float32)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def knn_and_percentiles(Xn: np.ndarray, adj: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-k cosine neighbours (self excluded), their cosines, and for every word the fraction of
    the other words whose cosine is below the cosine to its table-adjacent word adj[i]."""
    X = torch.from_numpy(Xn)
    n = X.shape[0]
    idx = np.zeros((n, k), dtype=np.int64)
    cos = np.zeros((n, k), dtype=np.float32)
    pct = np.zeros(n, dtype=np.float64)
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        S = X[s:e] @ X.T  # [c, n]
        rows = torch.arange(e - s)
        S[rows, torch.arange(s, e)] = -2.0  # drop self
        c_adj = S[rows, torch.from_numpy(adj[s:e])]
        pct[s:e] = ((S < c_adj[:, None]).sum(1).double() / (n - 1)).numpy()
        v, i = torch.topk(S, k, dim=1)
        idx[s:e], cos[s:e] = i.numpy(), v.numpy()
        print(f"  knn {e}/{n}", flush=True)
    return idx, cos, pct


def offset_cosines(Xn: np.ndarray, order: np.ndarray, words: list[str], rng: np.random.Generator) -> list[dict]:
    """Mean/median cosine of rank i vs rank i + d within `order` (indices sorted by y), against
    random pairs drawn within the same index set and within the whole vocabulary."""
    n_all = Xn.shape[0]
    rows = []
    for d in OFFSETS:
        m = len(order) - d
        if m < 10:
            continue
        a, b = order[:-d], order[d:]
        c = (Xn[a] * Xn[b]).sum(1)
        same = np.array([stem(words[i]) == stem(words[j]) for i, j in zip(a, b)])
        ra, rb = rng.choice(order, m), rng.choice(order, m)
        keep = ra != rb
        c_rand_within = (Xn[ra[keep]] * Xn[rb[keep]]).sum(1)
        ga, gb = rng.integers(0, n_all, m), rng.integers(0, n_all, m)
        keep = ga != gb
        c_rand_global = (Xn[ga[keep]] * Xn[gb[keep]]).sum(1)
        rows.append({
            "offset": d, "n_pairs": int(m),
            "mean_cos": float(c.mean()), "median_cos": float(np.median(c)),
            "same_stem_frac": float(same.mean()),
            "mean_cos_no_same_stem": float(c[~same].mean()) if (~same).any() else None,
            "random_within_mean_cos": float(c_rand_within.mean()), "random_within_median_cos": float(np.median(c_rand_within)),
            "random_global_mean_cos": float(c_rand_global.mean()),
            "gap_vs_random_within": float(c.mean() - c_rand_within.mean()),
        })
    return rows


def neighbour_filters(words: list[str], idx: np.ndarray) -> dict[str, np.ndarray]:
    """For each filter, a [n, K_STORE] boolean mask of neighbours that pass it."""
    n = len(words)
    stems = np.array([stem(w) for w in words])
    ok_stem = stems[idx] != stems[:, None]
    ok_prefix = np.ones_like(ok_stem)
    for i in range(n):
        wi = words[i]
        for j in range(idx.shape[1]):
            if common_prefix_len(wi, words[idx[i, j]]) >= 4:
                ok_prefix[i, j] = False
    ok_prefix &= ok_stem
    return {"all": np.ones_like(ok_stem), "no_same_stem": ok_stem, "no_stem_or_prefix4": ok_prefix}


def first_k(idx: np.ndarray, ok: np.ndarray, k: int) -> np.ndarray:
    """Indices of the first k neighbours passing `ok`, -1 where fewer exist."""
    n = idx.shape[0]
    out = -np.ones((n, k), dtype=np.int64)
    for i in range(n):
        sel = idx[i][ok[i]][:k]
        out[i, :len(sel)] = sel
    return out


def check2(words: list[str], y: np.ndarray, y_z: np.ndarray, idx: np.ndarray, ok_masks: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for fname, ok in ok_masks.items():
        for k in KS:
            nb = first_k(idx, ok, k)
            have = (nb >= 0).all(1)
            yn = np.where(nb >= 0, y[np.clip(nb, 0, None)], np.nan)
            yzn = np.where(nb >= 0, y_z[np.clip(nb, 0, None)], np.nan)
            my, myz = np.nanmean(yn, 1), np.nanmean(yzn, 1)
            rows.append({
                "filter": fname, "k": k, "n_words_with_k": int(have.sum()),
                "spearman_y": float(stats.spearmanr(y[have], my[have]).statistic),
                "pearson_y": float(stats.pearsonr(y[have], my[have]).statistic),
                "spearman_y_z": float(stats.spearmanr(y_z[have], myz[have]).statistic),
            })
    return rows


def ridge_direction(Xn: np.ndarray, y: np.ndarray) -> dict:
    """Single ridge on all words (standardised columns, lambda LAM); Spearman of y with the
    projection, and the share of the (L2-normalised) feature variance along the direction."""
    X = torch.from_numpy(Xn).double()
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    yt = torch.from_numpy(y).double()
    A = Z.T @ Z + LAM * torch.eye(Z.shape[1], dtype=Z.dtype)
    w = torch.linalg.solve(A, Z.T @ (yt - yt.mean()))
    pred = (Z @ w).numpy()
    u = w / sd
    u = u / u.norm()
    Xc = X - mu
    cov = (Xc.T @ Xc) / (X.shape[0] - 1)
    share = float(u @ cov @ u / torch.trace(cov))
    evals = torch.linalg.eigvalsh(cov).flip(0)
    pc1_share = float(evals[0] / evals.sum())
    # how many principal directions carry at least as much variance as w
    rank_of_w = int((evals >= u @ cov @ u).sum())
    return {
        "lambda": LAM, "spearman_in_sample": float(stats.spearmanr(pred, y).statistic),
        "pearson_in_sample": float(stats.pearsonr(pred, y).statistic),
        "variance_share_along_w": share, "pc1_variance_share": pc1_share,
        "n_pcs_with_at_least_w_variance": rank_of_w, "dim": int(X.shape[1]),
    }


def main() -> None:
    words, y, y_z = targets()
    n = len(words)
    order = np.argsort(-y)  # table order, most sensitive first
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    # table-adjacent partner: the next word down the table (the previous one for the last word)
    adj = np.empty(n, dtype=np.int64)
    adj[order[:-1]] = order[1:]
    adj[order[-1]] = order[-2]
    n1 = int(round(0.01 * n))
    top1, bot1 = order[:n1], order[-n1:][::-1]  # bot1 sorted from least to more sensitive
    top1_set = set(top1.tolist())
    chosen = [r["word"] for r in read_json(WORDS)["chosen"]] if WORDS.exists() else []

    res = {"n_words": n, "n_top1pct": n1, "seed": SEED, "offsets": OFFSETS, "ks": KS, "k_store": K_STORE,
           "stem_rule": "first 5 characters", "prefix_rule": "common prefix >= 4 characters",
           "ridge_oof_reference": "x109_wordrep_analyze.py: emb_mean ridge out-of-fold Spearman +0.78 on y",
           "features": {}}
    examples = {}
    for name in FEATURES:
        print("feature", name, flush=True)
        rng = np.random.default_rng(SEED)
        Xn = load_unit(name)
        f = {}
        # check 1
        f["offsets_all"] = offset_cosines(Xn, order, words, rng)
        f["offsets_top1pct"] = offset_cosines(Xn, top1, words, rng)
        f["offsets_bottom1pct"] = offset_cosines(Xn, bot1, words, rng)
        # pairwise cosine within the top / bottom 1 % against random pairs
        for lab, sub in (("top1pct", top1), ("bottom1pct", bot1)):
            S = Xn[sub] @ Xn[sub].T
            iu = np.triu_indices(len(sub), 1)
            ra, rb = rng.integers(0, n, len(iu[0])), rng.integers(0, n, len(iu[0]))
            keep = ra != rb
            f[f"pairwise_{lab}"] = {"mean_cos": float(S[iu].mean()), "median_cos": float(np.median(S[iu])),
                                    "random_global_mean_cos": float((Xn[ra[keep]] * Xn[rb[keep]]).sum(1).mean())}
        idx, cos, pct = knn_and_percentiles(Xn, adj, K_STORE)
        c_adj = (Xn * Xn[adj]).sum(1)
        ra = rng.permutation(n)
        c_rand = (Xn * Xn[ra]).sum(1)
        f["per_word"] = {
            "mean_cos_table_adjacent": float(c_adj.mean()), "median_cos_table_adjacent": float(np.median(c_adj)),
            "mean_cos_1nn": float(cos[:, 0].mean()), "median_cos_1nn": float(np.median(cos[:, 0])),
            "mean_cos_random_partner": float(c_rand.mean()),
            "median_percentile_of_table_adjacent": float(np.median(pct)), "mean_percentile_of_table_adjacent": float(pct.mean()),
            "frac_table_adjacent_above_p90": float((pct > 0.9).mean()),
            "frac_table_adjacent_is_1nn": float((adj == idx[:, 0]).mean()),
            "frac_table_adjacent_in_20nn": float((idx[:, :20] == adj[:, None]).any(1).mean()),
            "median_percentile_top1pct": float(np.median(pct[top1])), "median_percentile_bottom1pct": float(np.median(pct[bot1])),
        }
        # check 2
        ok = neighbour_filters(words, idx)
        f["knn_to_shift"] = check2(words, y, y_z, idx, ok)
        nn1 = idx[:, 0]
        same1 = ~ok["no_same_stem"][:, 0]
        nn1_nostem = first_k(idx, ok["no_same_stem"], 1)[:, 0]
        f["nn1"] = {
            "frac_same_stem": float(same1.mean()),
            "spearman_y_nn1_y": float(stats.spearmanr(y, y[nn1]).statistic),
            "spearman_y_nn1_y_no_same_stem": float(stats.spearmanr(y, y[nn1_nostem]).statistic),
            "top1pct_frac_nn1_in_top1pct": float(np.mean([nn1[i] in top1_set for i in top1])),
            "top1pct_frac_nn1_in_top1pct_no_same_stem": float(np.mean([nn1_nostem[i] in top1_set for i in top1])),
            "top1pct_frac_nn1_in_top10pct": float(np.mean([rank[nn1[i]] < 10 * n1 for i in top1])),
            "top1pct_frac_nn1_in_top10pct_no_same_stem": float(np.mean([rank[nn1_nostem[i]] < 10 * n1 for i in top1])),
            "bottom1pct_frac_nn1_in_bottom1pct": float(np.mean([rank[nn1[i]] >= n - n1 for i in bot1])),
            "bottom1pct_frac_nn1_in_bottom1pct_no_same_stem": float(np.mean([rank[nn1_nostem[i]] >= n - n1 for i in bot1])),
        }
        f["ridge_direction"] = ridge_direction(Xn, y)
        res["features"][name] = f
        # examples (emb_mean only, as primary)
        if name == "emb_mean":
            examples["top20_words_5nn"] = [
                {"word": words[i], "y": round(float(y[i]), 3), "rank": int(rank[i]),
                 "neighbours": [[words[j], round(float(cos[i, t]), 3), round(float(y[j]), 3), int(rank[j])] for t, j in enumerate(idx[i, :5])]}
                for i in order[:20]]
            examples["chosen_words_5nn"] = [
                {"word": w, "y": round(float(y[words.index(w)]), 3), "rank": int(rank[words.index(w)]),
                 "neighbours": [[words[j], round(float(cos[words.index(w), t]), 3), round(float(y[j]), 3), int(rank[j])]
                                for t, j in enumerate(idx[words.index(w), :5])]}
                for w in chosen if w in words]
            examples["top_adjacent_pairs"] = [
                {"rank": int(r), "a": words[order[r]], "b": words[order[r + 1]], "y_a": round(float(y[order[r]]), 3),
                 "y_b": round(float(y[order[r + 1]]), 3), "cos": round(float(c_adj[order[r]]), 3),
                 "percentile_in_a": round(float(pct[order[r]]), 3), "same_stem": stem(words[order[r]]) == stem(words[order[r + 1]])}
                for r in range(0, 20)]
            examples["bottom_adjacent_pairs"] = [
                {"rank": int(r), "a": words[order[r]], "b": words[order[r + 1]], "y_a": round(float(y[order[r]]), 3),
                 "y_b": round(float(y[order[r + 1]]), 3), "cos": round(float(c_adj[order[r]]), 3),
                 "percentile_in_a": round(float(pct[order[r]]), 3)}
                for r in range(n - 11, n - 1)]
        del Xn, idx, cos
        print({k: v for k, v in f["per_word"].items()}, flush=True)
        print(f["nn1"], f["ridge_direction"], flush=True)
    res["examples"] = examples
    atomic_json(OUT / "neighbors.json", res)
    write_md(res, words)
    print("wrote", OUT / "neighbors.json", flush=True)


def write_md(res: dict, words: list[str]) -> None:
    L = ["# x109 table neighbours vs embedding neighbours (Qwen3.5-4B)", "",
         f"{res['n_words']:,} words ranked by y (mean boundary shift over the 40 benign prompts, x106 sd units). "
         f"Features L2-normalised. Stem key: {res['stem_rule']}. Top 1 % = {res['n_top1pct']} words. Seed {res['seed']}.", ""]
    for name, f in res["features"].items():
        L += [f"## {name}", "", "### Check 1: table neighbours -> cosine", "",
              "| set | offset d | n pairs | mean cos | median cos | random (within set) mean | random (global) mean | gap vs random within | same-stem frac | mean cos without same stem |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for lab in ("offsets_all", "offsets_top1pct", "offsets_bottom1pct"):
            for r in f[lab]:
                nos = "" if r["mean_cos_no_same_stem"] is None else f"{r['mean_cos_no_same_stem']:+.3f}"
                L.append(f"| {lab[8:]} | {r['offset']} | {r['n_pairs']} | {r['mean_cos']:+.3f} | {r['median_cos']:+.3f} | {r['random_within_mean_cos']:+.3f} | {r['random_global_mean_cos']:+.3f} | {r['gap_vs_random_within']:+.3f} | {r['same_stem_frac']:.3f} | {nos} |")
        pw = f["per_word"]
        L += ["", f"Pairwise cosine inside the top 1 %: mean {f['pairwise_top1pct']['mean_cos']:+.3f} (random global {f['pairwise_top1pct']['random_global_mean_cos']:+.3f}); "
              f"inside the bottom 1 %: mean {f['pairwise_bottom1pct']['mean_cos']:+.3f}.", "",
              f"Per word: cosine to the table-adjacent word mean {pw['mean_cos_table_adjacent']:+.3f} (median {pw['median_cos_table_adjacent']:+.3f}); "
              f"to its 1-NN mean {pw['mean_cos_1nn']:+.3f} (median {pw['median_cos_1nn']:+.3f}); to a random word {pw['mean_cos_random_partner']:+.3f}. "
              f"Percentile of the table-adjacent cosine inside the word's own cosine distribution: median {pw['median_percentile_of_table_adjacent']:.3f}, mean {pw['mean_percentile_of_table_adjacent']:.3f}, "
              f"above p90 for {pw['frac_table_adjacent_above_p90']:.3f} of words; top 1 % median {pw['median_percentile_top1pct']:.3f}, bottom 1 % median {pw['median_percentile_bottom1pct']:.3f}. "
              f"Table-adjacent word is the 1-NN for {pw['frac_table_adjacent_is_1nn']:.4f} of words, among the 20-NN for {pw['frac_table_adjacent_in_20nn']:.4f}.", "",
              "### Check 2: cosine neighbours -> shift", "",
              "| filter | k | n words | Spearman(y, mean neighbour y) | Pearson | Spearman on y_z |", "|---|---|---|---|---|---|"]
        for r in f["knn_to_shift"]:
            L.append(f"| {r['filter']} | {r['k']} | {r['n_words_with_k']} | {r['spearman_y']:+.3f} | {r['pearson_y']:+.3f} | {r['spearman_y_z']:+.3f} |")
        nn = f["nn1"]
        L += ["", f"1-NN shares the stem for {nn['frac_same_stem']:.3f} of words. Spearman(y, y of 1-NN) {nn['spearman_y_nn1_y']:+.3f}, excluding same stem {nn['spearman_y_nn1_y_no_same_stem']:+.3f}. "
              f"Top 1 % words whose 1-NN is also top 1 %: {nn['top1pct_frac_nn1_in_top1pct']:.3f} (no same stem {nn['top1pct_frac_nn1_in_top1pct_no_same_stem']:.3f}); in the top 10 %: {nn['top1pct_frac_nn1_in_top10pct']:.3f} ({nn['top1pct_frac_nn1_in_top10pct_no_same_stem']:.3f}). "
              f"Bottom 1 % whose 1-NN is bottom 1 %: {nn['bottom1pct_frac_nn1_in_bottom1pct']:.3f} ({nn['bottom1pct_frac_nn1_in_bottom1pct_no_same_stem']:.3f}).", ""]
        rd = f["ridge_direction"]
        L += [f"Ridge direction (lambda {rd['lambda']:g}, in-sample on all words): Spearman {rd['spearman_in_sample']:+.3f}, Pearson {rd['pearson_in_sample']:+.3f}; "
              f"share of feature variance along w {rd['variance_share_along_w']:.4f} (PC1 {rd['pc1_variance_share']:.4f}; {rd['n_pcs_with_at_least_w_variance']} of {rd['dim']} principal directions carry at least as much).", ""]
    ex = res["examples"]
    L += ["## Examples (emb_mean)", "", "Top 20 table words, 5 nearest cosine neighbours as word (cos, y, table rank):", ""]
    for e in ex["top20_words_5nn"]:
        L.append(f"- {e['word']} ({e['y']:+.2f}, rank {e['rank']}): " + "; ".join(f"{w} ({c:.2f}, {yy:+.2f}, {r})" for w, c, yy, r in e["neighbours"]))
    L += ["", "Chosen x109 words:", ""]
    for e in ex["chosen_words_5nn"]:
        L.append(f"- {e['word']} ({e['y']:+.2f}, rank {e['rank']}): " + "; ".join(f"{w} ({c:.2f}, {yy:+.2f}, {r})" for w, c, yy, r in e["neighbours"]))
    L += ["", "Adjacent table pairs at the top (cos, percentile of that cosine inside a's distribution):", ""]
    for e in ex["top_adjacent_pairs"]:
        L.append(f"- {e['rank']}: {e['a']} ({e['y_a']:+.2f}) / {e['b']} ({e['y_b']:+.2f}) cos {e['cos']:+.3f}, pct {e['percentile_in_a']:.3f}" + (" same stem" if e["same_stem"] else ""))
    L += ["", "Adjacent pairs at the bottom:", ""]
    for e in ex["bottom_adjacent_pairs"]:
        L.append(f"- {e['rank']}: {e['a']} ({e['y_a']:+.2f}) / {e['b']} ({e['y_b']:+.2f}) cos {e['cos']:+.3f}, pct {e['percentile_in_a']:.3f}")
    (OUT / "neighbors.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
