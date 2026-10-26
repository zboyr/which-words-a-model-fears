"""x107 analysis: is what the model carries about the appended word linearly
separable between harm and neutral words, and does the separation generalise to words it was
never fitted on? Two sources: (1) x85's stored residual-stream states (33 states, positions
word-last / <|im_end|> / tbg-1 / tbg); (2) x107_value_sep_extract.py's value vectors and
attention writes. Classifier: ridge regression to +-1 (closed form) with 4-fold held-out word
folds (train 18 pairs x 96 prompts, test 6 unseen pairs x 96 prompts), features standardised on
the training fold. Reported: held-out accuracy, AUC, and the word-level Spearman between the
classifier score averaged per word and the word's x85 mean refusal change."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    n, d = Xs.shape
    if d > n:  # dual form
        K = Xs @ Xs.T
        alpha = np.linalg.solve(K + lam * np.eye(n), y)
        w = Xs.T @ alpha
    else:
        w = np.linalg.solve(Xs.T @ Xs + lam * np.eye(d), Xs.T @ y)
    return w / sd, -(mu / sd) @ w


def auc(score: np.ndarray, y: np.ndarray) -> float:
    pos, neg = score[y > 0], score[y < 0]
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def heldout_words(X: np.ndarray, y: np.ndarray, word_fold: np.ndarray, lam: float = 10.0) -> dict:
    score = np.zeros(len(y))
    for f in np.unique(word_fold):
        tr, te = word_fold != f, word_fold == f
        w, b = ridge_fit(X[tr], y[tr], lam)
        score[te] = X[te] @ w + b
    return {"acc": float((np.sign(score) == y).mean()), "auc": auc(score, y), "score": score}


def word_rho(score: np.ndarray, word_index: np.ndarray, cond: np.ndarray, dref: dict) -> float:
    """Spearman over the 24 words between (mean score of H cells − mean score of N cells) and Δrefusal."""
    ws = sorted(set(word_index.tolist()))
    xs, ys = [], []
    for w in ws:
        m = word_index == w
        xs.append(score[m & (cond == 0)].mean() - score[m & (cond == 1)].mean())
        ys.append(dref[w])
    return float(stats.spearmanr(xs, ys).correlation)


def load_dref(x85: Path) -> dict:
    rates = {}
    for c in ("H", "N"):
        for line in (x85 / f"x85_refusal_{c}.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r.get("refusal"):
                rates[(c, int(r["prompt_id"]), int(r["word_index"]))] = float(np.mean(r["refusal"]))
    per_word: dict[int, list] = {}
    for (c, p, w), v in rates.items():
        if c == "H" and ("N", p, w) in rates:
            per_word.setdefault(w, []).append(v - rates[("N", p, w)])
    return {w: float(np.mean(v)) for w, v in per_word.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--x85", default="data/runs/x85")
    ap.add_argument("--vsep", default="", help="dir with x107_value_sep.npz (optional)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--states", default="0,4,8,12,16,20,24,28,30,32")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    x85 = Path(args.x85)
    cohort = json.loads((x85 / "x85_cohort.json").read_text())
    word_fold_of = {int(w["word_index"]): int(w["word_fold"]) for w in cohort["words"]}
    words = {int(w["word_index"]): (w["harm"], w["neutral"]) for w in cohort["words"]}
    dref = load_dref(x85)
    rep: dict = {}
    lines = ["# Linear separability of harm vs neutral appended words, held-out word folds (Qwen3.5-4B, x85 grid)\n"]
    rng = np.random.default_rng(1074)

    if not args.vsep or True:
        import torch

        states_wanted = [int(s) for s in args.states.split(",")]
        shards = sorted(glob.glob(str(x85 / "features" / "x85_hidden_shard_*.pt")))
        feats, cond, widx, pidx = {s: {p: [] for p in range(4)} for s in states_wanted}, [], [], []
        for path in shards:
            d = torch.load(path, map_location="cpu", weights_only=False)
            keep = [i for i, c in enumerate(d["cell_ids"]) if c[0] in "HN"]
            st = d["states"][keep].float().numpy()  # [k, 4, 33, D]
            for i in keep:
                cid = d["cell_ids"][i]
                cond.append(0 if cid[0] == "H" else 1)
                widx.append(int(cid.split(":w")[1]))
                pidx.append(int(cid.split(":p")[1].split(":")[0]))
            for s in states_wanted:
                for p in range(4):
                    feats[s][p].append(st[:, p, s, :])
        cond, widx = np.asarray(cond), np.asarray(widx)
        y = np.where(cond == 0, 1.0, -1.0)
        wf = np.asarray([word_fold_of[w] for w in widx])
        pos_names = d["position_names"]
        lines.append(f"## Residual stream (x85 stored states, {len(y)} H+N cells)\n")
        lines.append("| state | " + " | ".join(f"{p}: acc / AUC / word ρ" for p in pos_names) + " |\n|---|" + "---|" * 4)
        rep["residual"] = {}
        for s in states_wanted:
            row = []
            rep["residual"][s] = {}
            for p in range(4):
                X = np.concatenate(feats[s][p], 0)
                r = heldout_words(X, y, wf)
                wr = word_rho(r["score"], widx, cond, dref)
                rep["residual"][s][pos_names[p]] = {"acc": r["acc"], "auc": r["auc"], "word_rho": wr}
                row.append(f"{r['acc']:.3f} / {r['auc']:.3f} / {wr:+.2f}")
            lines.append(f"| {s} | " + " | ".join(row) + " |")
        # label-permutation baseline at one state/position (permute word pairs' labels consistently)
        X = np.concatenate(feats[states_wanted[len(states_wanted) // 2]][0], 0)
        flips = {w: rng.choice([1, -1]) for w in words}
        yp = y * np.asarray([flips[w] for w in widx])
        r = heldout_words(X, yp, wf)
        lines.append(f"\nControl (word-pair labels randomly flipped per pair, state {states_wanted[len(states_wanted) // 2]} word-last): acc {r['acc']:.3f} / AUC {r['auc']:.3f}\n")
        rep["residual_control"] = {"acc": r["acc"], "auc": r["auc"]}

    if args.vsep:
        z = np.load(Path(args.vsep) / "x107_value_sep.npz")
        meta = json.loads((Path(args.vsep) / "x107_value_sep_meta.json").read_text())
        layers = meta["attention_layers"]
        cond, widx = z["cond"].astype(int), z["word_index"].astype(int)
        y = np.where(cond == 0, 1.0, -1.0)
        wf = np.asarray([word_fold_of[w] for w in widx])
        lines.append(f"\n## What softmax attention carries ({len(y)} H+N texts)\n")
        lines.append("| layer | v at word token (4 kv heads): acc / AUC / word ρ | OV write word→<\\|im_end\\|>: acc / AUC / word ρ | OV write word→tbg | attention output at <\\|im_end\\|> (all keys) | mean a(im_end→word) H / N |\n|---|---|---|---|---|---|")
        rep["attention"] = {}
        for li, l in enumerate(layers):
            cells = {}
            for name, X in (("v_word", z["v_word"][:, li].reshape(len(y), -1)), ("c_imend", z["c_imend"][:, li]), ("c_tbg", z["c_tbg"][:, li]), ("attn_imend", z["attn_imend"][:, li])):
                X = X.astype(np.float32)
                r = heldout_words(X, y, wf)
                cells[name] = {"acc": r["acc"], "auc": r["auc"], "word_rho": word_rho(r["score"], widx, cond, dref)}
            aH, aN = z["a_imend"][cond == 0, li].sum(-1).mean(), z["a_imend"][cond == 1, li].sum(-1).mean()
            rep["attention"][l] = cells
            lines.append(f"| L{l} | " + " | ".join(f"{c['acc']:.3f} / {c['auc']:.3f} / {c['word_rho']:+.2f}" for c in cells.values()) + f" | {aH:.2f} / {aN:.2f} |")
        lines.append("")

    (out / "x107_value_sep_analysis.json").write_text(json.dumps(rep, indent=2, default=float))
    (out / "x107_value_sep_analysis.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
