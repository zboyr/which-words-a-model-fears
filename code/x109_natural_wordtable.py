"""Vocabulary scan, natural prompts: is refusal predictable from per-word sensitivity aggregates?

On Qwen3.5-4B's natural prompts (x106 Hard-1K population and safe XSTest, x108 Toxic and
XSTest contrast; refusal2, 10 samples per prompt; plus all 1,319 Hard-1K prompts under the
x104 GuardReasoner-8B base label), each prompt is reduced to per-word displacement
aggregates under two definitions of a word's displacement: (A) the out-of-context scan table
(mean shift of the word appended to the 40 benign prompts, sd units), looked up for every
lowercase alphabetic word of the prompt; (B) the in-context leave-one-word-out delta from
the x106/x108 Stage 0b nominations (proj(full) - proj(without the word), sd units). The
aggregates (mean, max, sum, top-3 mean, counts and indicators above cutoffs, lexicon hit
counts) are scored against the refusal rate (Spearman, AUC with prompt-level bootstrap
intervals), compared with the full-prompt boundary projection, put into binomial logistic
models with likelihood-ratio tests and cross-validated AUC, swept as a single-word trigger
rule, and the prompts where the table fails are listed. Local, numpy/scipy only.

Writes data/runs/x109/natural_wordtable/results.json and results.md.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x106_common import X104_LABELS, is_content_word, lexicon_words, normalize_word, x104_labels  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
X109 = ROOT / "data" / "runs" / "x109"
OUT = X109 / "natural_wordtable"
SCAN_JSON = X109 / "scan.json"
SCAN_NPZ = X109 / "scan_shift.npz"
CMU = X109 / "cmu_bad_words.txt"
X106_QWEN = ROOT / "data" / "runs" / "x106" / "qwen"
X108_QWEN = ROOT / "data" / "runs" / "x108" / "qwen"

TAG = "qwen"
N_SAMPLES = 10
N_BOOT = 10_000
BOOT_SEED = 10_910
CV_FOLDS = 5
CV_REPEATS = 5
CV_SEED = 10_911
RIDGE = 1e-3
CUTOFFS = (0.5, 1.0, 1.5, 2.0)
SWEEP = np.round(np.arange(0.0, 3.0001, 0.1), 2)
ALPHA3 = re.compile(r"[a-z]{3,}")
ALPHA_ALL = re.compile(r"[a-z]+")
FAIL_N = 15


# --- loading ---------------------------------------------------------------------------


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_table() -> dict[str, Any]:
    """Per-word mean shift (sd units) over all 40 scan prompts, set A and set B."""
    scan = read_json(SCAN_JSON)
    sd = float(scan["sd_unit"]["value"])
    words = scan["candidates"]["words"]
    shift = np.load(SCAN_NPZ)["shift"].astype(np.float64).sum(axis=2) / sd  # [40, N] sd units
    sets = np.asarray(scan["prompt_sets"])
    means = {"all": shift.mean(0), "A": shift[sets == "A"].mean(0), "B": shift[sets == "B"].mean(0)}
    index = {w: j for j, w in enumerate(words)}
    return {"words": words, "index": index, "means": means, "sd_unit": sd, "n_words": len(words), "n_prompts": int(shift.shape[0])}


def load_rate_rows(refusal_path: Path, nominations_path: Path, projections_path: Path, cohort_path: Path) -> list[dict]:
    """One row per judged prompt: id, source, category, prompt, refusal counts, proj, nomination fields."""
    nominations = {r["id"]: r for r in read_json(nominations_path)["rows"]}
    projections = {r["id"]: r for r in read_json(projections_path)["rows"]}
    cohort = {u["id"]: u for u in read_json(cohort_path)["units"]}
    rows = []
    for rec in read_jsonl(refusal_path):
        if rec.get("failed"):
            continue
        pid = rec["prompt_id"]
        nom, prj = nominations[pid], projections[pid]
        if nom["prompt_sha256"] != rec["prompt_sha256"] or prj["prompt_sha256"] != rec["prompt_sha256"]:
            raise ValueError(f"prompt sha mismatch at {pid}")
        votes = [bool(v) for v in rec["refusal"]]
        if len(votes) != N_SAMPLES:
            raise ValueError(f"{pid}: {len(votes)} samples")
        rows.append(
            {
                "id": pid,
                "source": prj["source"],
                "category": prj.get("category") or cohort.get(pid, {}).get("category"),
                "prompt": nom["prompt"],
                "k": int(sum(votes)),
                "n": N_SAMPLES,
                "rate": float(np.mean(votes)),
                "proj": float(prj["proj"]),
                "nom": nom,
            }
        )
    return rows


def load_hard1k_binary() -> list[dict]:
    """All 1,319 Hard-1K prompts with the x104 GuardReasoner base refusal label (greedy response)."""
    labels = x104_labels(TAG, X104_LABELS)["base"]
    nominations = {r["id"]: r for r in read_json(X106_QWEN / "x106_stage0_nominations.json")["rows"]}
    rows = []
    for prj in read_json(X106_QWEN / "x106_stage0_projections.json")["rows"]:
        if prj["source"] != "orbench_hard":
            continue
        lab = labels[prj["id"]]
        nom = nominations[prj["id"]]
        rows.append(
            {
                "id": prj["id"],
                "source": prj["source"],
                "category": prj.get("category"),
                "prompt": nom["prompt"],
                "k": int(lab["refusal"]),
                "n": 1,
                "rate": float(lab["refusal"]),
                "proj": float(prj["proj"]),
                "request": lab["fields"].get("request"),
                "nom": nom,
            }
        )
    if len(rows) != 1319:
        raise ValueError(f"{len(rows)} Hard-1K rows")
    return rows


# --- per-prompt features --------------------------------------------------------------


def topk_mean(values: np.ndarray, k: int) -> float:
    if values.size == 0:
        return 0.0
    return float(np.sort(values)[::-1][:k].mean())


def table_features(prompt: str, table: dict, mean_key: str, tokenizer: re.Pattern, prefix: str) -> dict[str, float]:
    """Definition A aggregates of one prompt: the table's mean shift (sd units) per word."""
    words = tokenizer.findall(prompt.lower())
    means = table["means"][mean_key]
    index = table["index"]
    hits = [(w, float(means[index[w]])) for w in words if w in index]
    covered = np.asarray([v for _, v in hits], dtype=np.float64)
    zero_filled = np.zeros(len(words), dtype=np.float64)
    for j, w in enumerate(words):
        if w in index:
            zero_filled[j] = means[index[w]]
    out = {
        f"{prefix}n_words": float(len(words)),
        f"{prefix}n_covered": float(len(hits)),
        f"{prefix}coverage": float(len(hits) / len(words)) if words else 0.0,
        f"{prefix}mean0": float(zero_filled.mean()) if words else 0.0,
        f"{prefix}mean_cov": float(covered.mean()) if covered.size else 0.0,
        f"{prefix}max": float(covered.max()) if covered.size else 0.0,
        f"{prefix}sum": float(covered.sum()),
        f"{prefix}top3": topk_mean(covered, 3),
    }
    for tau in CUTOFFS:
        out[f"{prefix}count_gt{tau:g}"] = float((covered > tau).sum())
        out[f"{prefix}any_gt{tau:g}"] = float((covered > tau).any())
    out[f"{prefix}argmax_word"] = max(hits, key=lambda t: t[1])[0] if hits else None
    return out


def incontext_features(nom: dict, sd_unit: float, prefix: str, content_only: bool) -> dict[str, float]:
    """Definition B aggregates: leave-one-word-out deltas (sd units) of the prompt's words."""
    words = nom["words"]
    delta = np.asarray(nom["delta_unigram"], dtype=np.float64) / sd_unit
    keep = np.asarray([is_content_word(w) for w in words], dtype=bool) if content_only else np.ones(len(words), dtype=bool)
    if not keep.any():
        keep = np.ones(len(words), dtype=bool)
    d = delta[keep]
    kept_words = [w for w, k in zip(words, keep) if k]
    out = {
        f"{prefix}n_words": float(len(d)),
        f"{prefix}mean": float(d.mean()),
        f"{prefix}max": float(d.max()),
        f"{prefix}sum_pos": float(d[d > 0].sum()),
        f"{prefix}sum": float(d.sum()),
        f"{prefix}top3": topk_mean(d, 3),
    }
    for tau in CUTOFFS:
        out[f"{prefix}count_gt{tau:g}"] = float((d > tau).sum())
        out[f"{prefix}any_gt{tau:g}"] = float((d > tau).any())
    out[f"{prefix}argmax_word"] = normalize_word(kept_words[int(np.argmax(d))])
    return out


def lexicon_features(prompt: str, x83: set[str], cmu: set[str]) -> dict[str, float]:
    words = ALPHA_ALL.findall(prompt.lower())
    x83_hits = [w for w in words if w in x83]
    cmu_hits = [w for w in words if w in cmu]
    return {
        "lex_x83_count": float(len(x83_hits)),
        "lex_x83_any": float(bool(x83_hits)),
        "lex_cmu_count": float(len(cmu_hits)),
        "lex_cmu_any": float(bool(cmu_hits)),
        "lex_x83_words": sorted(set(x83_hits)),
        "lex_cmu_words": sorted(set(cmu_hits)),
    }


def featurize(rows: list[dict], table: dict, x83: set[str], cmu: set[str]) -> None:
    for row in rows:
        f: dict[str, Any] = {"proj": row["proj"]}
        f.update(table_features(row["prompt"], table, "all", ALPHA3, "A_"))
        f.update(table_features(row["prompt"], table, "A", ALPHA3, "AsetA_"))
        f.update(table_features(row["prompt"], table, "B", ALPHA3, "AsetB_"))
        f.update(table_features(row["prompt"], table, "all", ALPHA_ALL, "Aall_"))
        f.update(incontext_features(row["nom"], table["sd_unit"], "B_", content_only=True))
        f.update(incontext_features(row["nom"], table["sd_unit"], "Ball_", content_only=False))
        f.update(lexicon_features(row["prompt"], x83, cmu))
        row["f"] = f


# --- statistics ------------------------------------------------------------------------


def auc_from_scores(score: np.ndarray, y: np.ndarray) -> float:
    pos, neg = int(y.sum()), int((~y).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    r = stats.rankdata(score)
    return float((r[y].sum() - pos * (pos + 1) / 2) / (pos * neg))


def auc_matrix(score: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Row-wise AUC of a [B, n] score matrix against a [B, n] boolean label matrix."""
    pos = y.sum(1)
    neg = y.shape[1] - pos
    r = stats.rankdata(score, axis=1)
    s = (r * y).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (s - pos * (pos + 1) / 2) / (pos * neg)


def spearman_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ra = stats.rankdata(a, axis=1)
    rb = stats.rankdata(b, axis=1)
    ra = ra - ra.mean(1, keepdims=True)
    rb = rb - rb.mean(1, keepdims=True)
    num = (ra * rb).sum(1)
    den = np.sqrt((ra**2).sum(1) * (rb**2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def bootstrap(score: np.ndarray, rate: np.ndarray, y_half: np.ndarray, y_any: np.ndarray, rng: np.random.Generator) -> dict:
    n = len(score)
    out = {"spearman": [], "auc_half": [], "auc_any": []}
    chunk = 1000
    for start in range(0, N_BOOT, chunk):
        idx = rng.integers(0, n, size=(min(chunk, N_BOOT - start), n))
        s, r = score[idx], rate[idx]
        out["spearman"].append(spearman_matrix(s, r))
        out["auc_half"].append(auc_matrix(s, y_half[idx]))
        out["auc_any"].append(auc_matrix(s, y_any[idx]))
    res = {}
    for key, parts in out.items():
        v = np.concatenate(parts)
        v = v[np.isfinite(v)]
        res[key] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v.size else [float("nan"), float("nan")]
    return res


def score_feature(score: np.ndarray, rate: np.ndarray, y_half: np.ndarray, y_any: np.ndarray, proj: np.ndarray, rng: np.random.Generator) -> dict:
    const = np.allclose(score, score[0])
    sp = float("nan") if const else float(stats.spearmanr(score, rate).statistic)
    sp_proj = float("nan") if const else float(stats.spearmanr(score, proj).statistic)
    ci = bootstrap(score, rate, y_half, y_any, rng) if not const else {"spearman": [float("nan")] * 2, "auc_half": [float("nan")] * 2, "auc_any": [float("nan")] * 2}
    return {
        "spearman_rate": sp,
        "spearman_rate_ci95": ci["spearman"],
        "auc_half": auc_from_scores(score, y_half),
        "auc_half_ci95": ci["auc_half"],
        "auc_any": auc_from_scores(score, y_any),
        "auc_any_ci95": ci["auc_any"],
        "spearman_proj": sp_proj,
    }


# --- binomial logistic regression ---------------------------------------------------------


def fit_logit(X: np.ndarray, k: np.ndarray, n: np.ndarray) -> dict:
    """Binomial logistic regression with intercept on standardised columns and a small ridge."""
    X = np.asarray(X, dtype=np.float64)
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    p_dim = Z.shape[1]

    def nll(beta):
        eta = Z @ beta
        ll = k * eta - n * np.logaddexp(0.0, eta)
        pen = RIDGE * (beta[1:] ** 2).sum()
        return -ll.sum() + pen

    def grad(beta):
        eta = Z @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        g = -(Z.T @ (k - n * p))
        g[1:] += 2 * RIDGE * beta[1:]
        return g

    res = optimize.minimize(nll, np.zeros(p_dim), jac=grad, method="L-BFGS-B", options={"maxiter": 2000, "gtol": 1e-8})
    beta = res.x
    eta = Z @ beta
    ll = float((k * eta - n * np.logaddexp(0.0, eta)).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        phat = k / n
        ll_sat = float(np.nansum(np.where(k > 0, k * np.log(phat), 0.0) + np.where(n - k > 0, (n - k) * np.log(1 - phat), 0.0)))
    p0 = k.sum() / n.sum()
    ll_null = float((k * np.log(p0) + (n - k) * np.log(1 - p0)).sum()) if 0 < p0 < 1 else 0.0
    return {
        "beta_std": beta.tolist(),
        "mu": mu.tolist(),
        "sd": sd.tolist(),
        "ll": ll,
        "ll_null": ll_null,
        "ll_sat": ll_sat,
        "deviance": 2 * (ll_sat - ll),
        "null_deviance": 2 * (ll_sat - ll_null),
        "deviance_explained": 1 - (ll_sat - ll) / (ll_sat - ll_null) if ll_sat > ll_null else float("nan"),
        "converged": bool(res.success),
    }


def predict_logit(fit: dict, X: np.ndarray) -> np.ndarray:
    Z = np.column_stack([np.ones(len(X)), (np.asarray(X, dtype=np.float64) - np.asarray(fit["mu"])) / np.asarray(fit["sd"])])
    return 1.0 / (1.0 + np.exp(-(Z @ np.asarray(fit["beta_std"]))))


def lr_test(ll_full: float, ll_reduced: float, df: int) -> dict:
    stat = max(2 * (ll_full - ll_reduced), 0.0)
    return {"lr_stat": float(stat), "df": df, "p": float(stats.chi2.sf(stat, df))}


def cv_auc(X: np.ndarray, k: np.ndarray, n: np.ndarray, y: np.ndarray, seed: int) -> dict:
    """Repeated stratified 5-fold out-of-fold AUC of the binomial logistic model against y."""
    N = len(y)
    aucs = []
    for rep in range(CV_REPEATS):
        rng = np.random.default_rng(seed + rep)
        folds = np.empty(N, dtype=int)
        for cls in (True, False):
            idx = np.flatnonzero(y == cls)
            rng.shuffle(idx)
            folds[idx] = np.arange(len(idx)) % CV_FOLDS
        oof = np.empty(N)
        for f in range(CV_FOLDS):
            train, test = folds != f, folds == f
            if test.sum() == 0:
                continue
            fit = fit_logit(X[train], k[train], n[train])
            oof[test] = predict_logit(fit, X[test])
        aucs.append(auc_from_scores(oof, y))
    aucs = [a for a in aucs if np.isfinite(a)]
    return {"cv_auc_mean": float(np.mean(aucs)) if aucs else float("nan"), "cv_auc_per_repeat": aucs, "folds": CV_FOLDS, "repeats": CV_REPEATS}


# --- the evaluation --------------------------------------------------------------------


SCORED_FEATURES = (
    "proj",
    "A_mean0", "A_mean_cov", "A_max", "A_sum", "A_top3",
    "A_count_gt0.5", "A_count_gt1", "A_count_gt1.5", "A_count_gt2",
    "A_any_gt0.5", "A_any_gt1", "A_any_gt1.5", "A_any_gt2",
    "A_n_words", "A_n_covered", "A_coverage",
    "AsetA_mean_cov", "AsetA_max", "AsetA_count_gt1",
    "AsetB_mean_cov", "AsetB_max", "AsetB_count_gt1",
    "Aall_mean0", "Aall_mean_cov", "Aall_max", "Aall_sum", "Aall_count_gt1",
    "B_mean", "B_max", "B_sum_pos", "B_sum", "B_top3",
    "B_count_gt0.5", "B_count_gt1", "B_count_gt1.5", "B_count_gt2",
    "B_any_gt0.5", "B_any_gt1", "B_any_gt1.5", "B_any_gt2",
    "Ball_max", "Ball_sum_pos", "Ball_mean", "B_n_words",
    "lex_x83_count", "lex_x83_any", "lex_cmu_count", "lex_cmu_any",
)
CANDIDATE_SINGLE = tuple(f for f in SCORED_FEATURES if f.startswith(("A_", "B_", "Aall_", "Ball_")) and "n_words" not in f and "coverage" not in f and "n_covered" not in f)
MULTI_A = ("A_mean_cov", "A_max", "A_count_gt1", "A_n_words")
MULTI_B = ("B_mean", "B_max", "B_sum_pos", "B_n_words")
LEX_MULTI = ("lex_x83_count", "lex_cmu_count", "A_n_words")


def evaluate_set(name: str, rows: list[dict], seed_offset: int) -> dict:
    rate = np.asarray([r["rate"] for r in rows])
    k = np.asarray([r["k"] for r in rows], dtype=np.float64)
    n = np.asarray([r["n"] for r in rows], dtype=np.float64)
    proj = np.asarray([r["proj"] for r in rows])
    y_half = rate >= 0.5
    y_any = rate > 0
    rng = np.random.default_rng(BOOT_SEED + seed_offset)
    feats = {f: np.asarray([r["f"][f] for r in rows], dtype=np.float64) for f in SCORED_FEATURES}

    # coverage
    cov = feats["A_coverage"]
    n_words = feats["A_n_words"]
    n_cov = feats["A_n_covered"]
    coverage = {
        "tokens_total": int(n_words.sum()),
        "tokens_covered": int(n_cov.sum()),
        "token_share_covered": float(n_cov.sum() / n_words.sum()),
        "prompt_share_with_a_covered_word": float((n_cov > 0).mean()),
        "mean_prompt_coverage": float(cov.mean()),
        "mean_prompt_coverage_refused_half": float(cov[y_half].mean()) if y_half.any() else float("nan"),
        "mean_prompt_coverage_answered_half": float(cov[~y_half].mean()) if (~y_half).any() else float("nan"),
        "mean_n_words_refused_half": float(n_words[y_half].mean()) if y_half.any() else float("nan"),
        "mean_n_words_answered_half": float(n_words[~y_half].mean()) if (~y_half).any() else float("nan"),
        "all_words_token_share_covered": float(feats["A_n_covered"].sum() / np.asarray([r["f"]["Aall_n_words"] for r in rows]).sum()),
        "A_max_percentiles": {str(q): float(np.percentile(feats["A_max"], q)) for q in (10, 25, 50, 75, 90, 95, 99, 100)},
        "B_max_percentiles": {str(q): float(np.percentile(feats["B_max"], q)) for q in (10, 25, 50, 75, 90, 95, 99, 100)},
        "A_max_share_above": {f"{tau:g}": float((feats["A_max"] > tau).mean()) for tau in (0.25, 0.5, 1.0, 1.5, 2.0)},
        "B_max_share_above": {f"{tau:g}": float((feats["B_max"] > tau).mean()) for tau in (0.25, 0.5, 1.0, 1.5, 2.0)},
    }

    # per-feature scores
    scores = {f: score_feature(feats[f], rate, y_half, y_any, proj, rng) for f in SCORED_FEATURES}

    # logistic models
    def X_of(names):
        return np.column_stack([feats[f] for f in names])

    fit_proj = fit_logit(X_of(["proj"]), k, n)
    singles = {}
    for f in CANDIDATE_SINGLE:
        if np.allclose(feats[f], feats[f][0]):
            continue
        singles[f] = fit_logit(X_of([f]), k, n)
    best = max(singles, key=lambda f: singles[f]["deviance_explained"])
    best_by_auc = max((f for f in singles), key=lambda f: (scores[f]["auc_half"] if np.isfinite(scores[f]["auc_half"]) else -1))
    fit_best = singles[best]
    fit_both = fit_logit(X_of(["proj", best]), k, n)
    fit_multiA = fit_logit(X_of(list(MULTI_A)), k, n)
    fit_multiA_proj = fit_logit(X_of(["proj", *MULTI_A]), k, n)
    fit_multiB = fit_logit(X_of(list(MULTI_B)), k, n)
    fit_multiB_proj = fit_logit(X_of(["proj", *MULTI_B]), k, n)
    fit_lex = fit_logit(X_of(list(LEX_MULTI)), k, n)
    fit_lex_proj = fit_logit(X_of(["proj", *LEX_MULTI]), k, n)
    fit_AB = fit_logit(X_of([*MULTI_A, *MULTI_B]), k, n)

    def cvrow(names):
        return cv_auc(X_of(list(names)), k, n, y_half, CV_SEED + seed_offset)

    models = {
        "proj": {"features": ["proj"], "fit": fit_proj, "cv": cvrow(["proj"])},
        "best_single_aggregate": {"features": [best], "fit": fit_best, "cv": cvrow([best]), "chosen_by": "deviance explained among single-aggregate models", "best_by_auc_half": best_by_auc},
        "proj_plus_best": {"features": ["proj", best], "fit": fit_both, "cv": cvrow(["proj", best])},
        "table_multi_A": {"features": list(MULTI_A), "fit": fit_multiA, "cv": cvrow(MULTI_A)},
        "table_multi_A_plus_proj": {"features": ["proj", *MULTI_A], "fit": fit_multiA_proj, "cv": cvrow(["proj", *MULTI_A])},
        "incontext_multi_B": {"features": list(MULTI_B), "fit": fit_multiB, "cv": cvrow(MULTI_B)},
        "incontext_multi_B_plus_proj": {"features": ["proj", *MULTI_B], "fit": fit_multiB_proj, "cv": cvrow(["proj", *MULTI_B])},
        "lexicon_multi": {"features": list(LEX_MULTI), "fit": fit_lex, "cv": cvrow(LEX_MULTI)},
        "lexicon_multi_plus_proj": {"features": ["proj", *LEX_MULTI], "fit": fit_lex_proj, "cv": cvrow(["proj", *LEX_MULTI])},
        "table_A_plus_incontext_B": {"features": [*MULTI_A, *MULTI_B], "fit": fit_AB, "cv": cvrow([*MULTI_A, *MULTI_B])},
    }
    for m in models.values():
        m["fit"] = {key: m["fit"][key] for key in ("beta_std", "ll", "ll_null", "deviance", "null_deviance", "deviance_explained", "converged")}
    lr = {
        "best_over_proj": lr_test(fit_both["ll"], fit_proj["ll"], 1),
        "proj_over_best": lr_test(fit_both["ll"], fit_best["ll"], 1),
        "multiA_over_proj": lr_test(fit_multiA_proj["ll"], fit_proj["ll"], len(MULTI_A)),
        "proj_over_multiA": lr_test(fit_multiA_proj["ll"], fit_multiA["ll"], 1),
        "multiB_over_proj": lr_test(fit_multiB_proj["ll"], fit_proj["ll"], len(MULTI_B)),
        "proj_over_multiB": lr_test(fit_multiB_proj["ll"], fit_multiB["ll"], 1),
        "lexicon_over_proj": lr_test(fit_lex_proj["ll"], fit_proj["ll"], len(LEX_MULTI)),
        "proj_over_lexicon": lr_test(fit_lex_proj["ll"], fit_lex["ll"], 1),
    }

    # cutoff sweep on the "any word above tau" rule
    sweep = {}
    for key in ("A_max", "B_max", "Ball_max"):
        s = feats[key]
        grid = []
        for tau in SWEEP:
            flag = s > tau
            rec = {"tau": float(tau), "n_flagged": int(flag.sum()), "share_flagged": float(flag.mean())}
            for lab, y in (("half", y_half), ("any", y_any)):
                tp = int((flag & y).sum())
                prec = tp / flag.sum() if flag.sum() else float("nan")
                rec_ = tp / y.sum() if y.sum() else float("nan")
                f1 = 2 * prec * rec_ / (prec + rec_) if np.isfinite(prec) and np.isfinite(rec_) and (prec + rec_) > 0 else float("nan")
                acc = float((flag == y).mean())
                rec.update({f"precision_{lab}": prec, f"recall_{lab}": rec_, f"f1_{lab}": f1, f"accuracy_{lab}": acc})
            grid.append(rec)
        finite = [g for g in grid if np.isfinite(g["f1_half"])]
        best_tau = max(finite, key=lambda g: g["f1_half"]) if finite else None
        sweep[key] = {"grid": grid, "best_f1_half": best_tau, "base_rate_half": float(y_half.mean()), "base_rate_any": float(y_any.mean())}

    return {
        "n_prompts": len(rows),
        "n_refused_half": int(y_half.sum()),
        "n_refused_any": int(y_any.sum()),
        "mean_rate": float(rate.mean()),
        "coverage": coverage,
        "scores": scores,
        "models": models,
        "lr_tests": lr,
        "sweep": sweep,
    }


def failure_lists(rows: list[dict]) -> dict:
    refused_low = [r for r in rows if r["rate"] >= 0.8 and r["f"]["A_max"] < 0.5]
    answered_high = [r for r in rows if r["rate"] == 0 and r["f"]["A_max"] > 1.5]

    def rec(r):
        return {
            "id": r["id"],
            "source": r["source"],
            "rate": r["rate"],
            "proj": round(r["proj"], 2),
            "A_max": round(r["f"]["A_max"], 2),
            "A_argmax_word": r["f"]["A_argmax_word"],
            "B_max": round(r["f"]["B_max"], 2),
            "B_argmax_word": r["f"]["B_argmax_word"],
            "x106_nominated_unit": r["nom"].get("unit"),
            "prompt": r["prompt"],
        }

    return {
        "refused_rate_ge_0.8_table_max_lt_0.5sd": {
            "n": len(refused_low),
            "n_refused_ge_0.8": int(sum(r["rate"] >= 0.8 for r in rows)),
            "rows": [rec(r) for r in sorted(refused_low, key=lambda r: r["f"]["A_max"])[:FAIL_N]],
        },
        "answered_rate_0_table_max_gt_1.5sd": {
            "n": len(answered_high),
            "n_answered_0": int(sum(r["rate"] == 0 for r in rows)),
            "rows": [rec(r) for r in sorted(answered_high, key=lambda r: -r["f"]["A_max"])[:FAIL_N]],
        },
        "answered_rate_0_highest_table_max": {
            "n_answered_0": int(sum(r["rate"] == 0 for r in rows)),
            "rows": [rec(r) for r in sorted((r for r in rows if r["rate"] == 0), key=lambda r: -r["f"]["A_max"])[:FAIL_N]],
        },
    }


# --- report ------------------------------------------------------------------------------


def fmt(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(fmt(v, nd) for v in x) + "]"
    if isinstance(x, float) and not np.isfinite(x):
        return "nan"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


MD_FEATURES = (
    "proj", "A_mean0", "A_mean_cov", "A_max", "A_sum", "A_top3", "A_count_gt1", "A_count_gt2", "A_any_gt1", "A_any_gt2", "A_n_words",
    "Aall_max", "AsetA_max", "AsetB_max",
    "B_mean", "B_max", "B_sum_pos", "B_top3", "B_count_gt1", "B_any_gt1", "B_any_gt2", "Ball_max",
    "lex_x83_count", "lex_x83_any", "lex_cmu_count", "lex_cmu_any",
)


def write_md(results: dict, path: Path) -> None:
    L = []
    L.append("# Natural prompts: per-word sensitivity aggregates against refusal (Qwen3.5-4B)\n")
    L.append(results["description"] + "\n")
    s = results["sanity"]
    L.append("## Sanity: the full-prompt projection on the 1,319 Hard-1K prompts\n")
    L.append(f"AUC of `proj` for x104 GuardReasoner base refusal (n refused {s['n_refused']}, answered {s['n_answered']}): {fmt(s['auc'])} [{fmt(s['auc_ci95'][0])}, {fmt(s['auc_ci95'][1])}]; x106's table says {s['x106_reported']}.\n")
    L.append("## Table coverage\n")
    L.append("| set | n | tokens covered | prompts with a covered word | mean coverage refused / answered | mean words refused / answered |")
    L.append("|---|---|---|---|---|---|")
    for name, res in results["sets"].items():
        c = res["coverage"]
        L.append(f"| {name} | {res['n_prompts']} | {fmt(c['token_share_covered'])} | {fmt(c['prompt_share_with_a_covered_word'])} | {fmt(c['mean_prompt_coverage_refused_half'])} / {fmt(c['mean_prompt_coverage_answered_half'])} | {fmt(c['mean_n_words_refused_half'],1)} / {fmt(c['mean_n_words_answered_half'],1)} |")
    L.append("")
    L.append("Missing words count as 0 shift in `*_mean0`; `*_mean_cov`, `*_max`, `*_sum`, `*_top3` and the counts use covered words only. `A_*`: table mean over all 40 scan prompts, `[a-z]{3,}` tokens; `AsetA_*` / `AsetB_*`: set-A / set-B means; `Aall_*`: `[a-z]+` tokens (1–2 letter words are never in the table). `B_*`: in-context leave-one-out delta over content words; `Ball_*`: over all whitespace words. All displacements in x106 Hard-1K sd units (sd " + f"{results['table']['sd_unit']:.3f}" + ").\n")

    t = results["table"]
    L.append("Scale of the table itself (per-word mean shift over the 40 scan prompts, sd units): percentiles " + ", ".join(f"p{q} {v:.2f}" for q, v in t["word_mean_percentiles"].items()) + "; share of the 71,583 words above " + ", ".join(f"{k} sd: {v:.4f}" for k, v in t["word_share_above"].items()) + ".\n")
    L.append("| set | A_max p50 / p90 / p99 / max | share A_max > 0.25 / 0.5 / 1.0 | B_max p50 / p90 / p99 / max | share B_max > 0.5 / 1.0 / 2.0 |")
    L.append("|---|---|---|---|---|")
    for name, res in results["sets"].items():
        c = res["coverage"]
        ap, bp, asx, bsx = c["A_max_percentiles"], c["B_max_percentiles"], c["A_max_share_above"], c["B_max_share_above"]
        L.append(f"| {name} | {ap['50']:.2f} / {ap['90']:.2f} / {ap['99']:.2f} / {ap['100']:.2f} | {asx['0.25']:.3f} / {asx['0.5']:.3f} / {asx['1']:.3f} | {bp['50']:.2f} / {bp['90']:.2f} / {bp['99']:.2f} / {bp['100']:.2f} | {bsx['0.5']:.3f} / {bsx['1']:.3f} / {bsx['2']:.3f} |")
    L.append("")
    for name, res in results["sets"].items():
        L.append(f"## Set `{name}` — n {res['n_prompts']}, refused (rate ≥ 0.5) {res['n_refused_half']}, refused (rate > 0) {res['n_refused_any']}, mean rate {fmt(res['mean_rate'])}\n")
        L.append("| feature | Spearman vs rate [95%] | AUC rate≥0.5 [95%] | AUC rate>0 [95%] | Spearman vs proj |")
        L.append("|---|---|---|---|---|")
        for f in MD_FEATURES:
            sc = res["scores"][f]
            L.append(f"| {f} | {fmt(sc['spearman_rate'])} {fmt(sc['spearman_rate_ci95'])} | {fmt(sc['auc_half'])} {fmt(sc['auc_half_ci95'])} | {fmt(sc['auc_any'])} {fmt(sc['auc_any_ci95'])} | {fmt(sc['spearman_proj'])} |")
        L.append("")
        L.append("Logistic models (binomial, 10 trials per prompt where the rate is available; standardised features, ridge 1e-3):\n")
        L.append("| model | features | deviance explained | CV AUC (rate≥0.5, 5-fold × 5) |")
        L.append("|---|---|---|---|")
        for mname, m in res["models"].items():
            L.append(f"| {mname} | {', '.join(m['features'])} | {fmt(m['fit']['deviance_explained'])} | {fmt(m['cv']['cv_auc_mean'])} |")
        L.append("")
        lr = res["lr_tests"]
        L.append("Likelihood-ratio tests: " + "; ".join(f"{k}: LR {fmt(v['lr_stat'],1)} (df {v['df']}), p {v['p']:.2e}" for k, v in lr.items()) + ".\n")
        for key in ("A_max", "B_max"):
            sw = res["sweep"][key]
            L.append(f"Cutoff sweep, `{key} > τ` as the refusal rule (base rate rate≥0.5: {fmt(sw['base_rate_half'])}, rate>0: {fmt(sw['base_rate_any'])}):\n")
            L.append("| τ (sd) | flagged share | P / R / F1 (rate≥0.5) | P / R / F1 (rate>0) |")
            L.append("|---|---|---|---|")
            for g in sw["grid"]:
                step = 1 if g["tau"] <= 1.0 + 1e-9 else 5
                if round(g["tau"] * 10) % step != 0:
                    continue
                L.append(f"| {g['tau']:.2f} | {fmt(g['share_flagged'])} | {fmt(g['precision_half'])} / {fmt(g['recall_half'])} / {fmt(g['f1_half'])} | {fmt(g['precision_any'])} / {fmt(g['recall_any'])} / {fmt(g['f1_any'])} |")
            b = sw["best_f1_half"]
            if b:
                L.append(f"\nBest F1 (rate≥0.5) on the 0.1 grid: τ = {b['tau']:.1f}, P {fmt(b['precision_half'])}, R {fmt(b['recall_half'])}, F1 {fmt(b['f1_half'])}, flagged {fmt(b['share_flagged'])}.\n")
    fl = results["failures"]
    L.append("## Where the table fails (x106 + x108 rate sets pooled)\n")
    a = fl["refused_rate_ge_0.8_table_max_lt_0.5sd"]
    L.append(f"Refused prompts (rate ≥ 0.8) whose maximum table shift is below 0.5 sd: {a['n']} of {a['n_refused_ge_0.8']}. The 15 with the lowest maximum:\n")
    L.append("| id | rate | proj | table max (word) | in-context max (word) | x106 unit | prompt |")
    L.append("|---|---|---|---|---|---|---|")
    for r in a["rows"]:
        L.append(f"| {r['id']} | {r['rate']:.1f} | {r['proj']} | {r['A_max']} ({r['A_argmax_word']}) | {r['B_max']} ({r['B_argmax_word']}) | {r['x106_nominated_unit']} | {r['prompt'][:110].replace('|', '/')} |")
    b = fl["answered_rate_0_table_max_gt_1.5sd"]
    L.append(f"\nAnswered prompts (rate = 0) whose maximum table shift is above 1.5 sd: {b['n']} of {b['n_answered_0']}. The 15 with the highest maximum:\n")
    L.append("| id | rate | proj | table max (word) | in-context max (word) | x106 unit | prompt |")
    L.append("|---|---|---|---|---|---|---|")
    for r in b["rows"]:
        L.append(f"| {r['id']} | {r['rate']:.1f} | {r['proj']} | {r['A_max']} ({r['A_argmax_word']}) | {r['B_max']} ({r['B_argmax_word']}) | {r['x106_nominated_unit']} | {r['prompt'][:110].replace('|', '/')} |")
    c = fl["answered_rate_0_highest_table_max"]
    L.append(f"\nThe 15 answered prompts (rate = 0, {c['n_answered_0']} in all) with the highest maximum table shift, whatever its value:\n")
    L.append("| id | rate | proj | table max (word) | in-context max (word) | x106 unit | prompt |")
    L.append("|---|---|---|---|---|---|---|")
    for r in c["rows"]:
        L.append(f"| {r['id']} | {r['rate']:.1f} | {r['proj']} | {r['A_max']} ({r['A_argmax_word']}) | {r['B_max']} ({r['B_argmax_word']}) | {r['x106_nominated_unit']} | {r['prompt'][:110].replace('|', '/')} |")
    L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    table = load_table()
    x83 = lexicon_words()
    cmu = {w.strip().lower() for w in CMU.read_text(encoding="utf-8").splitlines() if w.strip()}

    x106_rows = load_rate_rows(X106_QWEN / "x106_refusal_O_base.jsonl", X106_QWEN / "x106_stage0_nominations.json", X106_QWEN / "x106_stage0_projections.json", ROOT / "data" / "runs" / "x106" / "x106_cohort.json")
    x108_rows = load_rate_rows(X108_QWEN / "x108_refusal_O_base.jsonl", X108_QWEN / "x108_stage0_nominations.json", X108_QWEN / "x108_stage0_projections.json", ROOT / "data" / "runs" / "x108" / "x108_cohort.json")
    hard_rows = load_hard1k_binary()
    for rows in (x106_rows, x108_rows, hard_rows):
        featurize(rows, table, x83, cmu)

    # sanity: the x106 Stage 0a AUC of proj on the 1,319 Hard-1K prompts under the x104 label
    y = np.asarray([r["rate"] for r in hard_rows]) >= 0.5
    proj = np.asarray([r["proj"] for r in hard_rows])
    rng = np.random.default_rng(BOOT_SEED)
    ci = bootstrap(proj, proj, y, y, rng)["auc_half"]
    sanity = {"auc": auc_from_scores(proj, y), "auc_ci95": ci, "n_refused": int(y.sum()), "n_answered": int((~y).sum()), "x106_reported": "0.964 [0.951, 0.975]"}
    print("sanity proj AUC on Hard-1K 1,319:", sanity)

    sets = {
        "x106_hard1k_population": [r for r in x106_rows if r["source"] == "orbench_hard"],
        "x106_xstest_safe": [r for r in x106_rows if r["source"] == "xstest"],
        "x106_all": x106_rows,
        "x108_toxic": [r for r in x108_rows if r["source"] == "orbench_toxic"],
        "x108_xstest_contrast": [r for r in x108_rows if r["source"] == "xstest"],
        "x108_all": x108_rows,
        "pooled_x106_x108": x106_rows + x108_rows,
        "hard1k_1319_x104_label": hard_rows,
    }
    results_sets = {}
    for i, (name, rows) in enumerate(sets.items()):
        print(f"evaluating {name} (n {len(rows)})", flush=True)
        results_sets[name] = evaluate_set(name, rows, i)

    failures = failure_lists(x106_rows + x108_rows)
    per_prompt = [
        {"id": r["id"], "set": "x106" if r in x106_rows else "x108", "source": r["source"], "category": r["category"], "rate": r["rate"], "proj": r["proj"], **{k: v for k, v in r["f"].items() if k != "proj"}}
        for r in x106_rows + x108_rows
    ]
    results = {
        "run": "x109 natural_wordtable",
        "description": (
            "Per-prompt aggregates of per-word displacement on Qwen3.5-4B's natural prompts, scored against refusal2 "
            "(10 samples per prompt, x106 and x108 O_base) and against the x104 GuardReasoner base label on all 1,319 "
            "Hard-1K prompts, next to the full-prompt boundary projection. Definition A looks each lowercase word up in "
            "the 71,583-word scan table (mean shift over the 40 benign scan prompts); definition B uses the in-context "
            "leave-one-word-out delta from Stage 0b. Bootstrap intervals resample prompts (10,000 draws)."
        ),
        "table": {
            "n_words": table["n_words"],
            "n_scan_prompts": table["n_prompts"],
            "sd_unit": table["sd_unit"],
            "tokenizer_primary": ALPHA3.pattern,
            "tokenizer_all": ALPHA_ALL.pattern,
            "missing_words": "0 shift in *_mean0; dropped in every other aggregate",
            "word_mean_percentiles": {str(q): float(np.percentile(table["means"]["all"], q)) for q in (1, 10, 50, 90, 99, 99.9, 100)},
            "word_share_above": {f"{tau:g}": float((table["means"]["all"] > tau).mean()) for tau in CUTOFFS},
            "set_A_vs_B_word_mean_spearman": float(stats.spearmanr(table["means"]["A"], table["means"]["B"]).statistic),
        },
        "lexicons": {"x83_n": len(x83), "cmu_n": len(cmu)},
        "labels": {"rate sets": "refusal2 (DeepSeek V4 Flash judge), 10 samples per prompt; refused = rate >= 0.5 and rate > 0", "hard1k_1319": "x104 GuardReasoner-8B refusal of the greedy base response"},
        "sanity": sanity,
        "sets": results_sets,
        "failures": failures,
        "per_prompt": per_prompt,
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    write_md(results, OUT / "results.md")
    print("wrote", OUT / "results.json", OUT / "results.md")


if __name__ == "__main__":
    main()
