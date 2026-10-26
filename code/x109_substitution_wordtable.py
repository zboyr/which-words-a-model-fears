"""x109 word table against the x106 substitutions on Qwen3.5-4B.

For every x106 cohort unit with a substitute text (S), look the prompt's words up in the x109
scan table (mean shift of the word appended to the 40 benign scan prompts, sd units) and ask:

1. Is the substituted unit the word with the largest table value among the content words of O?
2. Is the drop in table value from the unit to its substitute large, and does it track the
   in-context projection drop proj(S) - proj(O) and the refusal removed r(O) - r(S)?

The control substitution (C: inert word -> synonym) is the comparison. Rewrites are grouped by
how much refusal they removed. Local, numpy/scipy only.

Writes data/runs/x109/substitution_wordtable/results.{json,md}.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x106_common import is_content_word, normalize_word  # noqa: E402
from x109_natural_wordtable import load_table, read_json, read_jsonl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
X106 = ROOT / "data" / "runs" / "x106"
QWEN = X106 / "qwen"
OUT = ROOT / "data" / "runs" / "x109" / "substitution_wordtable"
N_BOOT = 10_000
SEED = 10_912
ALPHA = re.compile(r"[a-z]+")


# --- table lookup with an inflection fallback ---------------------------------------------


def variants(word: str) -> list[str]:
    out = [word]
    for suf in ("s", "es", "d", "ed", "ing", "er", "ers", "ly"):
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            stem = word[: -len(suf)]
            out.append(stem)
            if suf in ("d", "ed", "ing", "er", "ers"):
                out.append(stem + "e")
            if suf in ("ed", "ing", "er") and len(stem) >= 4 and stem[-1] == stem[-2]:
                out.append(stem[:-1])
    if word.endswith("ies") and len(word) > 4:
        out.append(word[:-3] + "y")
    return out


def lookup(word: str, table: dict, mean_key: str = "all") -> tuple[float | None, str | None]:
    """Table value of a word (max over hyphen parts), exact then inflection-stripped."""
    index, means = table["index"], table["means"][mean_key]
    best = None
    for part in ALPHA.findall(normalize_word(word)):
        for cand in variants(part):
            if cand in index:
                val = float(means[index[cand]])
                if best is None or val > best[0]:
                    best = (val, cand)
                break
    return best if best else (None, None)


def content_words(text: str) -> list[str]:
    return [w for w in text.split() if is_content_word(w)]


def phrase_value(phrase: str, table: dict) -> dict:
    """Max and mean table value over the content words of a phrase, with the words found."""
    found = []
    missing = []
    for w in content_words(phrase):
        val, key = lookup(w, table)
        if val is None:
            missing.append(normalize_word(w))
        else:
            found.append((normalize_word(w), key, val))
    vals = [v for _, _, v in found]
    return {
        "max": max(vals) if vals else None,
        "mean": float(np.mean(vals)) if vals else None,
        "argmax": max(found, key=lambda t: t[2])[0] if found else None,
        "found": found,
        "missing": missing,
    }


# --- loading ---------------------------------------------------------------------------


def rates(path: Path) -> dict[str, float]:
    return {str(r["prompt_id"]): float(np.mean(r["refusal"])) for r in read_jsonl(path)}


def build_rows(table: dict) -> list[dict]:
    cohort = read_json(X106 / "x106_cohort.json")
    nom = {t: {r["id"]: r for r in read_json(QWEN / f"x106_stage0_nominations{s}.json")["rows"]}
           for t, s in (("O", ""), ("S", "_S"), ("C", "_C"))}
    r_o, r_s, r_c = (rates(QWEN / f"x106_refusal_{a}_base.jsonl") for a in ("O", "S", "C"))
    sd = table["sd_unit"]
    rows = []
    for u in cohort["units"]:
        if not u["prompts"]["S"]:
            continue
        uid = u["id"]
        o = nom["O"][uid]
        words = o["words"]
        # content words of O with table values and in-context deltas
        per_word = []
        for j, w in enumerate(words):
            if not is_content_word(w):
                continue
            val, key = lookup(w, table)
            per_word.append({"index": j, "word": normalize_word(w), "key": key, "table": val,
                             "delta": float(o["delta_unigram"][j]) / sd})
        covered = [p for p in per_word if p["table"] is not None]
        unit_idx = set(u["unit_indices"])
        unit_words = [p for p in per_word if p["index"] in unit_idx]
        unit_val = max((p["table"] for p in unit_words if p["table"] is not None), default=None)
        # rank of the unit among the covered content words of O (1 = largest table value)
        rank = pct = None
        others = [p["table"] for p in covered if p["index"] not in unit_idx]
        if unit_val is not None:
            rank = 1 + sum(1 for v in others for _ in [0] if v > unit_val)
            pct = 1.0 - (rank - 1) / max(len(covered), 1)
        table_argmax = max(covered, key=lambda p: p["table"]) if covered else None
        second = max(others) if others else None
        sub = phrase_value(u["substitute"]["substitute"], table)
        # the table value of the span that was actually replaced (unit or hedge phrase)
        replaced = phrase_value(u["substitute"]["unit"], table)
        ctrl = u.get("control") or {}
        c_word = phrase_value(ctrl["word"], table) if ctrl else {"max": None}
        c_syn = phrase_value(ctrl["synonym"], table) if ctrl else {"max": None}
        s_text_max = phrase_value(u["prompts"]["S"], table)["max"]
        rows.append({
            "id": uid, "source": u["source"], "stratum": u.get("stratum"), "unit_kind": u["unit_kind"],
            "pair": bool(u["pair_wins"]), "unit": u["unit"], "unit_in_lexicon": bool(u["unit_in_lexicon"]),
            "substitute": u["substitute"]["substitute"], "replaced_span": u["substitute"]["unit"],
            "n_content": len(per_word), "n_covered": len(covered),
            "unit_table": unit_val, "unit_rank": rank, "unit_pct": pct,
            "table_argmax_word": table_argmax["word"] if table_argmax else None,
            "table_argmax_value": table_argmax["table"] if table_argmax else None,
            "table_argmax_is_unit": (table_argmax["index"] in unit_idx) if table_argmax else None,
            "second_table": second,
            "replaced_table_max": replaced["max"], "sub_table_max": sub["max"], "sub_table_mean": sub["mean"],
            "sub_argmax_word": sub["argmax"], "sub_missing": sub["missing"],
            "table_drop": (replaced["max"] - sub["max"]) if replaced["max"] is not None and sub["max"] is not None else None,
            "s_text_max": s_text_max,
            "ctrl_word": ctrl.get("word"), "ctrl_synonym": ctrl.get("synonym"),
            "ctrl_table": c_word["max"], "ctrl_syn_table": c_syn["max"],
            "ctrl_drop": (c_word["max"] - c_syn["max"]) if c_word["max"] is not None and c_syn["max"] is not None else None,
            "proj_O": float(o["proj_full"]) / sd,
            "proj_S": float(nom["S"][uid]["proj_full"]) / sd,
            "proj_C": float(nom["C"][uid]["proj_full"]) / sd if uid in nom["C"] else None,
            "delta_top": float(o["top_delta"]) / sd, "top_z": float(o["top_z"]),
            "r_O": r_o.get(uid), "r_S": r_s.get(uid), "r_C": r_c.get(uid),
        })
    for r in rows:
        r["proj_drop_S"] = r["proj_O"] - r["proj_S"]
        r["proj_drop_C"] = (r["proj_O"] - r["proj_C"]) if r["proj_C"] is not None else None
        r["effect_S"] = (r["r_O"] - r["r_S"]) if r["r_O"] is not None and r["r_S"] is not None else None
        r["effect_C"] = (r["r_O"] - r["r_C"]) if r["r_O"] is not None and r["r_C"] is not None else None
        e = r["effect_S"]
        r["group"] = None if e is None else ("strong" if e >= 0.5 else "moderate" if e >= 0.2 else "weak" if e > 0 else "none")
    return rows


# --- statistics -------------------------------------------------------------------------


def boot_mean(x: np.ndarray, rng: np.random.Generator) -> list[float]:
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    m = x[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def boot_diff(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> list[float]:
    ia = rng.integers(0, len(a), size=(N_BOOT, len(a)))
    ib = rng.integers(0, len(b), size=(N_BOOT, len(b)))
    d = a[ia].mean(1) - b[ib].mean(1)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def spearman(x, y) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return {"rho": None, "p": None, "n": int(ok.sum())}
    rho, p = stats.spearmanr(x[ok], y[ok])
    return {"rho": float(rho), "p": float(p), "n": int(ok.sum())}


def summarize(rows: list[dict], rng: np.random.Generator) -> dict:
    out = {"n": len(rows)}
    arr = lambda k: np.asarray([r[k] for r in rows if r[k] is not None], float)  # noqa: E731
    # Q1: rank of the unit in the table among O's content words
    ranks = arr("unit_rank")
    out["q1"] = {
        "n_with_unit_value": int(len(ranks)),
        "unit_is_table_argmax": float(np.mean([r["table_argmax_is_unit"] for r in rows if r["table_argmax_is_unit"] is not None])),
        "unit_is_table_argmax_ci": boot_mean(np.asarray([r["table_argmax_is_unit"] for r in rows if r["table_argmax_is_unit"] is not None], float), rng),
        "unit_rank_le_2": float(np.mean(ranks <= 2)),
        "unit_rank_le_3": float(np.mean(ranks <= 3)),
        "unit_rank_median": float(np.median(ranks)),
        "unit_pct_mean": float(arr("unit_pct").mean()),
        "chance_argmax": float(np.mean([1.0 / r["n_covered"] for r in rows if r["n_covered"]])),
        "unit_table_mean": float(arr("unit_table").mean()),
        "unit_table_median": float(np.median(arr("unit_table"))),
        "other_words_table_mean": float(np.mean([p for r in rows for p in [r["second_table"]] if p is not None])),
        "unit_minus_second_mean": float(np.mean([r["unit_table"] - r["second_table"] for r in rows if r["unit_table"] is not None and r["second_table"] is not None])),
        "unit_above_1sd": float(np.mean(arr("unit_table") >= 1.0)),
        "unit_above_0_5sd": float(np.mean(arr("unit_table") >= 0.5)),
    }
    # Q2: the table drop unit -> substitute, vs control
    drop = arr("table_drop")
    cdrop = arr("ctrl_drop")
    out["q2"] = {
        "n_drop": int(len(drop)), "n_ctrl_drop": int(len(cdrop)),
        "table_drop_mean": float(drop.mean()), "table_drop_ci": boot_mean(drop, rng),
        "table_drop_median": float(np.median(drop)),
        "table_drop_positive": float(np.mean(drop > 0)),
        "ctrl_drop_mean": float(cdrop.mean()) if len(cdrop) else None,
        "ctrl_drop_ci": boot_mean(cdrop, rng) if len(cdrop) else None,
        "ctrl_drop_positive": float(np.mean(cdrop > 0)) if len(cdrop) else None,
        "sub_table_mean": float(arr("sub_table_max").mean()),
        "sub_table_median": float(np.median(arr("sub_table_max"))),
        "sub_below_second": float(np.mean([r["sub_table_max"] < r["second_table"] for r in rows if r["sub_table_max"] is not None and r["second_table"] is not None])),
        "sub_is_s_text_argmax": float(np.mean([abs(r["sub_table_max"] - r["s_text_max"]) < 1e-9 for r in rows if r["sub_table_max"] is not None and r["s_text_max"] is not None])),
        "proj_drop_S_mean": float(arr("proj_drop_S").mean()),
        "proj_drop_C_mean": float(arr("proj_drop_C").mean()) if len(arr("proj_drop_C")) else None,
        "spearman_table_drop_vs_proj_drop_S": spearman([r["table_drop"] for r in rows], [r["proj_drop_S"] for r in rows]),
        "spearman_table_drop_vs_effect_S": spearman([r["table_drop"] for r in rows], [r["effect_S"] for r in rows]),
        "spearman_unit_table_vs_effect_S": spearman([r["unit_table"] for r in rows], [r["effect_S"] for r in rows]),
        "spearman_unit_table_vs_delta_top": spearman([r["unit_table"] for r in rows], [r["delta_top"] for r in rows]),
        "spearman_sub_table_vs_effect_S": spearman([r["sub_table_max"] for r in rows], [r["effect_S"] for r in rows]),
        "spearman_sub_table_vs_r_S": spearman([r["sub_table_max"] for r in rows], [r["r_S"] for r in rows]),
        "spearman_proj_drop_S_vs_effect_S": spearman([r["proj_drop_S"] for r in rows], [r["effect_S"] for r in rows]),
        "spearman_ctrl_drop_vs_effect_C": spearman([r["ctrl_drop"] for r in rows], [r["effect_C"] for r in rows]),
        "spearman_ctrl_drop_vs_proj_drop_C": spearman([r["ctrl_drop"] for r in rows], [r["proj_drop_C"] for r in rows]),
    }
    return out


def by_group(rows: list[dict], rng: np.random.Generator) -> dict:
    out = {}
    for g in ("strong", "moderate", "weak", "none"):
        sub = [r for r in rows if r["group"] == g]
        if len(sub) >= 5:
            s = summarize(sub, rng)
            s["r_O_mean"] = float(np.mean([r["r_O"] for r in sub]))
            s["r_S_mean"] = float(np.mean([r["r_S"] for r in sub]))
            s["effect_S_mean"] = float(np.mean([r["effect_S"] for r in sub]))
            out[g] = s
    strong = [r for r in rows if r["group"] == "strong"]
    none = [r for r in rows if r["group"] == "none"]
    if len(strong) >= 5 and len(none) >= 5:
        f = lambda rs, k: np.asarray([r[k] for r in rs if r[k] is not None], float)  # noqa: E731
        out["strong_minus_none"] = {
            "unit_is_table_argmax": {"diff": float(f(strong, "table_argmax_is_unit").mean() - f(none, "table_argmax_is_unit").mean()),
                                     "ci": boot_diff(f(strong, "table_argmax_is_unit"), f(none, "table_argmax_is_unit"), rng)},
            "unit_table": {"diff": float(f(strong, "unit_table").mean() - f(none, "unit_table").mean()),
                           "ci": boot_diff(f(strong, "unit_table"), f(none, "unit_table"), rng)},
            "table_drop": {"diff": float(f(strong, "table_drop").mean() - f(none, "table_drop").mean()),
                           "ci": boot_diff(f(strong, "table_drop"), f(none, "table_drop"), rng)},
            "sub_table": {"diff": float(f(strong, "sub_table_max").mean() - f(none, "sub_table_max").mean()),
                          "ci": boot_diff(f(strong, "sub_table_max"), f(none, "sub_table_max"), rng)},
            "proj_drop_S": {"diff": float(f(strong, "proj_drop_S").mean() - f(none, "proj_drop_S").mean()),
                            "ci": boot_diff(f(strong, "proj_drop_S"), f(none, "proj_drop_S"), rng)},
        }
    return out


def disagreements(rows: list[dict], n: int = 25) -> list[dict]:
    """Effective rewrites where the table's argmax is not the unit, largest effect first."""
    sel = [r for r in rows if r["group"] == "strong" and r["table_argmax_is_unit"] is False]
    sel.sort(key=lambda r: -(r["effect_S"] or 0))
    return [{k: r[k] for k in ("id", "unit", "unit_table", "unit_rank", "table_argmax_word", "table_argmax_value",
                               "substitute", "sub_table_max", "table_drop", "proj_drop_S", "effect_S")} for r in sel[:n]]


def failures(rows: list[dict], n: int = 25) -> list[dict]:
    """Rewrites with a large table drop that removed no refusal."""
    sel = [r for r in rows if r["group"] == "none" and r["table_drop"] is not None]
    sel.sort(key=lambda r: -r["table_drop"])
    return [{k: r[k] for k in ("id", "unit", "unit_table", "unit_rank", "substitute", "sub_table_max", "table_drop",
                               "proj_drop_S", "r_O", "r_S")} for r in sel[:n]]


# --- report -----------------------------------------------------------------------------


def f3(x, nd=3):
    if x is None:
        return "–"
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(f3(v, nd) for v in x) + "]"
    return f"{x:+.{nd}f}" if isinstance(x, float) and nd == 2 else f"{x:.{nd}f}"


def sp(d: dict) -> str:
    return "–" if d["rho"] is None else f"{d['rho']:+.2f} (n={d['n']}, p={d['p']:.1e})"


def write_md(res: dict, path: Path) -> None:
    L = ["# x109 word table against the x106 substitutions (Qwen3.5-4B)", ""]
    L.append("Each x106 cohort unit with a substitute text is looked up in the x109 scan table (mean shift of the word appended to the 40 benign scan prompts, sd units; exact match, then inflection stripped). `unit_table` is the table value of the nominated unit (max over its words), `rank` its rank among O's covered content words, `table drop` = table(replaced span) − table(substitute) (max over content words of each), `ctrl drop` the same for the control word → synonym. Groups by `r(O) − r(S)` under refusal2: strong ≥ 0.5, moderate [0.2, 0.5), weak (0, 0.2), none ≤ 0. Bootstrap intervals resample prompts (10,000 draws).")
    L.append("")
    L.append(f"Coverage: {res['coverage']['units_with_table_value']} of {res['coverage']['n_units']} units have a table value for the unit; "
             f"{res['coverage']['substitutes_with_table_value']} have one for the substitute; content-word coverage {res['coverage']['content_word_coverage']:.3f}. "
             f"Missing unit words: {res['coverage']['missing_unit_words']}.")
    L.append("")
    for name, sec in res["sets"].items():
        L.append(f"## {name} (n = {sec['all']['n']})")
        L.append("")
        L.append("### Q1: is the nominated unit the prompt's largest-table word?")
        L.append("")
        L.append("| group | n | r(O) | r(S) | unit is table argmax [95% CI] | chance | rank ≤ 2 | rank ≤ 3 | median rank | unit table (sd) | runner-up table | unit ≥ 1 sd |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for g in ("all", "strong", "moderate", "weak", "none"):
            s = sec.get(g) if g != "all" else sec["all"]
            if not s:
                continue
            q = s["q1"]
            ro = f"{s['r_O_mean']:.2f}" if "r_O_mean" in s else "–"
            rs = f"{s['r_S_mean']:.2f}" if "r_S_mean" in s else "–"
            L.append(f"| {g} | {s['n']} | {ro} | {rs} | {q['unit_is_table_argmax']:.3f} {f3(q['unit_is_table_argmax_ci'])} | {q['chance_argmax']:.3f} | {q['unit_rank_le_2']:.3f} | {q['unit_rank_le_3']:.3f} | {q['unit_rank_median']:.0f} | {q['unit_table_mean']:.2f} (median {q['unit_table_median']:.2f}) | {q['other_words_table_mean']:.2f} | {q['unit_above_1sd']:.3f} |")
        L.append("")
        L.append("### Q2: the table drop from the unit to the substitute")
        L.append("")
        L.append("| group | n | table drop [95% CI] | drop > 0 | substitute table | sub below runner-up | ctrl drop [95% CI] | ctrl drop > 0 | proj drop S (sd) | proj drop C (sd) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for g in ("all", "strong", "moderate", "weak", "none"):
            s = sec.get(g) if g != "all" else sec["all"]
            if not s:
                continue
            q = s["q2"]
            L.append(f"| {g} | {q['n_drop']} | {q['table_drop_mean']:+.2f} {f3(q['table_drop_ci'], 2)} | {q['table_drop_positive']:.3f} | {q['sub_table_mean']:.2f} (median {q['sub_table_median']:.2f}) | {q['sub_below_second']:.3f} | {f3(q['ctrl_drop_mean'], 2)} {f3(q['ctrl_drop_ci'], 2)} | {f3(q['ctrl_drop_positive'])} | {q['proj_drop_S_mean']:+.2f} | {f3(q['proj_drop_C_mean'], 2)} |")
        L.append("")
        q = sec["all"]["q2"]
        L.append("Per-prompt Spearman over the whole set:")
        L.append("")
        L.append(f"- table drop vs in-context projection drop of S: {sp(q['spearman_table_drop_vs_proj_drop_S'])}")
        L.append(f"- table drop vs refusal removed r(O) − r(S): {sp(q['spearman_table_drop_vs_effect_S'])}")
        L.append(f"- unit table value vs r(O) − r(S): {sp(q['spearman_unit_table_vs_effect_S'])}")
        L.append(f"- unit table value vs in-context deletion delta of the unit: {sp(q['spearman_unit_table_vs_delta_top'])}")
        L.append(f"- substitute table value vs r(S): {sp(q['spearman_sub_table_vs_r_S'])}")
        L.append(f"- in-context projection drop of S vs r(O) − r(S) (x106's own link): {sp(q['spearman_proj_drop_S_vs_effect_S'])}")
        L.append(f"- control: ctrl drop vs r(O) − r(C): {sp(q['spearman_ctrl_drop_vs_effect_C'])}; ctrl drop vs projection drop of C: {sp(q['spearman_ctrl_drop_vs_proj_drop_C'])}")
        L.append("")
        if "strong_minus_none" in sec:
            L.append("Strong minus none (mean difference, 95% CI):")
            L.append("")
            for k, v in sec["strong_minus_none"].items():
                L.append(f"- {k}: {v['diff']:+.3f} {f3(v['ci'])}")
            L.append("")
        L.append("Strong rewrites where the table's argmax is not the unit (largest effect first):")
        L.append("")
        L.append("| id | unit | unit table | rank | table argmax (value) | substitute | sub table | table drop | proj drop | effect |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for d in sec["disagreements"]:
            L.append(f"| {d['id'].split(':')[1]} | {d['unit']} | {f3(d['unit_table'], 2)} | {d['unit_rank']} | {d['table_argmax_word']} ({f3(d['table_argmax_value'], 2)}) | {d['substitute']} | {f3(d['sub_table_max'], 2)} | {f3(d['table_drop'], 2)} | {d['proj_drop_S']:+.2f} | {d['effect_S']:+.2f} |")
        L.append("")
        L.append("Rewrites with the largest table drop that removed no refusal:")
        L.append("")
        L.append("| id | unit | unit table | rank | substitute | sub table | table drop | proj drop | r(O) | r(S) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for d in sec["failures"]:
            L.append(f"| {d['id'].split(':')[1]} | {d['unit']} | {f3(d['unit_table'], 2)} | {d['unit_rank']} | {d['substitute']} | {f3(d['sub_table_max'], 2)} | {f3(d['table_drop'], 2)} | {d['proj_drop_S']:+.2f} | {d['r_O']:.1f} | {d['r_S']:.1f} |")
        L.append("")
        L.append("Most frequent table-argmax words that are not the unit (all groups): " + ", ".join(f"{w} ({c})" for w, c in sec["argmax_not_unit_words"]))
        L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    table = load_table()
    rows = build_rows(table)
    rng = np.random.default_rng(SEED)
    all_content = sum(r["n_content"] for r in rows)
    covered = sum(r["n_covered"] for r in rows)
    coverage = {
        "n_units": len(rows),
        "units_with_table_value": int(sum(r["unit_table"] is not None for r in rows)),
        "substitutes_with_table_value": int(sum(r["sub_table_max"] is not None for r in rows)),
        "content_word_coverage": covered / all_content,
        "missing_unit_words": sorted({r["unit"] for r in rows if r["unit_table"] is None}),
        "missing_substitute_words": sorted({w for r in rows for w in r["sub_missing"]}),
    }
    sets = {}
    for name, sel in (("Hard-1K population", [r for r in rows if r["source"] == "orbench_hard"]),
                      ("XSTest safe", [r for r in rows if r["source"] != "orbench_hard"]),
                      ("content units only, Hard-1K", [r for r in rows if r["source"] == "orbench_hard" and r["unit_kind"] == "content"]),
                      ("single-word units only, Hard-1K", [r for r in rows if r["source"] == "orbench_hard" and not r["pair"] and r["unit_kind"] == "content"])):
        if len(sel) < 5:
            continue
        sec = {"all": summarize(sel, rng)}
        sec.update(by_group(sel, rng))
        sec["disagreements"] = disagreements(sel)
        sec["failures"] = failures(sel)
        sec["argmax_not_unit_words"] = Counter(r["table_argmax_word"] for r in sel if r["table_argmax_is_unit"] is False).most_common(20)
        sets[name] = sec
    res = {"coverage": coverage, "sets": sets, "rows": rows,
           "table": {"n_words": table["n_words"], "sd_unit": table["sd_unit"], "mean_key": "all"}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    write_md(res, OUT / "results.md")
    print((OUT / "results.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
