"""x109 word-representation probe, analysis: can a context-free representation of a word predict
its scanned sensitivity (mean boundary shift over the 40 benign prompts)?

Targets: y = per-word mean shift over the 40 prompts in x106 sd units; y_z = per-word mean of the
per-prompt z-scored shift. Predictors: ridge on each feature set from wordrep/features.npz,
nested CV (5 outer folds grouped by a 5-letter stem key so inflections never straddle a split;
lambda chosen by 4 inner folds); zero-parameter reads (word-alone chat projection, each block's
last-token state on the direction); a lexical baseline (token count, log wordfreq rank, length).
Metrics on out-of-fold predictions: R2, Pearson, Spearman, Spearman within the true top 1,000,
AUC for the true top 1 %, and how many of the predicted top 100 are in the true top 1 %.
Writes wordrep/results.json, results.md and oof.npz.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x109_scan_common import SCAN_JSON, SCAN_NPZ, TAG, WORDFREQ, X106_HARD1K_SD, atomic_json, read_json  # noqa: E402
from x106_common import load_mu, unit_directions  # noqa: E402

OUT = SCAN_NPZ.parent / "wordrep"
LAMBDAS = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
K_OUT, K_IN = 5, 4
dev = "cuda" if torch.cuda.is_available() else "cpu"


def fold_of(word: str, k: int, salt: str) -> int:
    key = (salt + word[:5]).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big") % k


def ridge_fit(X, y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    ym = y.mean()
    A = Z.T @ Z + lam * torch.eye(Z.shape[1], device=Z.device, dtype=Z.dtype)
    w = torch.linalg.solve(A, Z.T @ (y - ym))
    return lambda Xt: ((Xt - mu) / sd) @ w + ym


def nested_ridge(X: np.ndarray, y: np.ndarray, words: list[str]) -> tuple[np.ndarray, list[float]]:
    Xg = torch.from_numpy(X.astype(np.float32)).to(dev).double()
    yg = torch.from_numpy(y).to(dev).double()
    outer = np.array([fold_of(w, K_OUT, "o") for w in words])
    inner = np.array([fold_of(w, K_IN, "i") for w in words])
    oof = np.zeros_like(y)
    chosen = []
    for f in range(K_OUT):
        tr, te = np.where(outer != f)[0], np.where(outer == f)[0]
        scores = []
        for lam in LAMBDAS:
            sse = 0.0
            for g in range(K_IN):
                a, b = tr[inner[tr] != g], tr[inner[tr] == g]
                pred = ridge_fit(Xg[a], yg[a], lam)(Xg[b])
                sse += float(((pred - yg[b]) ** 2).sum())
            scores.append(sse)
        lam = LAMBDAS[int(np.argmin(scores))]
        chosen.append(lam)
        oof[te] = ridge_fit(Xg[tr], yg[tr], lam)(Xg[te]).cpu().numpy()
    return oof, chosen


def metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    top1 = y >= np.quantile(y, 0.99)
    top1000 = np.argsort(-y)[:1000]
    ss = ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    pred_top100 = np.argsort(-pred)[:100]
    return {
        "r2": float(1 - ss),
        "pearson": float(stats.pearsonr(pred, y).statistic),
        "spearman": float(stats.spearmanr(pred, y).statistic),
        "spearman_true_top1000": float(stats.spearmanr(pred[top1000], y[top1000]).statistic),
        "auc_top1pct": float(stats.mannwhitneyu(pred[top1], pred[~top1]).statistic / (top1.sum() * (~top1).sum())),
        "pred_top100_in_true_top1pct": int(top1[pred_top100].sum()),
    }


def main() -> None:
    scan = read_json(SCAN_JSON)
    words = scan["candidates"]["words"]
    ntok = np.array(scan["candidates"]["n_tokens"])
    shift = np.load(SCAN_NPZ)["shift"].astype(np.float64).sum(-1)  # [40, W]
    y = shift.mean(0) / X106_HARD1K_SD
    z = (shift - shift.mean(1, keepdims=True)) / shift.std(1, keepdims=True)
    y_z = z.mean(0)
    F = np.load(OUT / "features.npz")
    rank = {w: i for i, w in enumerate(read_json(WORDFREQ)["words"])}
    lexical = np.column_stack([
        np.log1p([rank.get(w, 50000) for w in words]),
        [len(w) for w in words],
        *[(ntok == k).astype(float) for k in (1, 2, 3)],
    ])
    units, _ = unit_directions(load_mu(TAG))
    res = {"n_words": len(words), "targets": {"y": "mean shift over 40 prompts, sd units", "y_z": "mean per-prompt z"},
           "lambdas": LAMBDAS, "folds": "5 outer x 4 inner, grouped by sha256 of the first 5 letters", "rows": []}
    oof_all = {}

    # zero-parameter reads
    zero = {"chat_proj (word alone as the user message)": F["chat_proj"]}
    for k in F.files:
        if k.startswith("block"):
            h = F[k].astype(np.float64)
            zero[f"{k} last token on mu_tbg"] = h @ units[1]
    for name, p in zero.items():
        for tname, t in (("y", y), ("y_z", y_z)):
            m = metrics(p, t)
            m.update({"predictor": name, "kind": "zero-parameter", "target": tname})
            res["rows"].append(m)
            print(name, tname, {k: round(v, 3) for k, v in m.items() if isinstance(v, float)}, flush=True)

    sets = {"lexical": lexical}
    for k in F.files:
        if k.startswith(("emb", "block", "v")):
            sets[k] = F[k]
    sets["chat_states (boundary, word alone)"] = F["chat_states"].reshape(len(words), -1)
    sets["emb_mean + lexical"] = np.column_stack([F["emb_mean"].astype(np.float32), lexical])
    for name, X in sets.items():
        for tname, t in (("y", y), ("y_z", y_z)):
            oof, lam = nested_ridge(np.asarray(X, dtype=np.float32), t, words)
            m = metrics(oof, t)
            m.update({"predictor": name, "kind": "ridge", "target": tname, "dim": int(X.shape[1]), "lambda_per_fold": lam})
            res["rows"].append(m)
            oof_all[f"{name}|{tname}"] = oof
            print(name, tname, {k: round(v, 3) for k, v in m.items() if isinstance(v, float)}, flush=True)

    # residual lists for the best ridge set on y
    best = max((r for r in res["rows"] if r["kind"] == "ridge" and r["target"] == "y"), key=lambda r: r["spearman"])
    p = oof_all[f"{best['predictor']}|y"]
    resid = y - p
    res["best_ridge_y"] = best["predictor"]
    res["under_predicted"] = [[words[i], round(float(y[i]), 3), round(float(p[i]), 3)] for i in np.argsort(-resid)[:40]]
    res["over_predicted"] = [[words[i], round(float(y[i]), 3), round(float(p[i]), 3)] for i in np.argsort(resid)[:40]]
    res["true_top30_pred"] = [[words[i], round(float(y[i]), 3), round(float(p[i]), 3)] for i in np.argsort(-y)[:30]]
    res["pred_top30"] = [[words[i], round(float(y[i]), 3), round(float(p[i]), 3)] for i in np.argsort(-p)[:30]]
    np.savez(OUT / "oof.npz", words=np.array(words), y=y, y_z=y_z, **{k.replace(" ", "_"): v for k, v in oof_all.items()})
    atomic_json(OUT / "results.json", res)

    lines = ["# x109 word-representation probe (Qwen3.5-4B)", "",
             "Target y: per-word mean boundary shift over the 40 benign prompts (x106 sd units); y_z: mean of per-prompt z-scored shift. 71,583 words. Ridge: nested CV, folds grouped by 5-letter stem key. Out-of-fold metrics.", "",
             "| predictor | kind | target | dim | R² | Pearson | Spearman | Spearman in true top 1000 | AUC top 1 % | pred top 100 in true top 1 % |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        lines.append(f"| {r['predictor']} | {r['kind']} | {r['target']} | {r.get('dim', '')} | {r['r2']:+.3f} | {r['pearson']:+.3f} | {r['spearman']:+.3f} | {r['spearman_true_top1000']:+.3f} | {r['auc_top1pct']:.3f} | {r['pred_top100_in_true_top1pct']} |")
    lines += ["", f"Best ridge set on y (Spearman): {best['predictor']}.", "",
              "Under-predicted (actual − predicted largest): " + ", ".join(f"{w} ({a:+.2f} vs {b:+.2f})" for w, a, b in res["under_predicted"][:25]), "",
              "Over-predicted: " + ", ".join(f"{w} ({a:+.2f} vs {b:+.2f})" for w, a, b in res["over_predicted"][:25]), "",
              "Predicted top 30: " + ", ".join(f"{w} ({a:+.2f})" for w, a, b in res["pred_top30"])]
    (OUT / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
