"""x113 Stage 1 (local): the per-model vocabulary tables and the table nomination.

`--tables`: per model one npz under data/runs/x113/tables/ over the x109 candidate list (71,583
words, the list x111 / x112 share): Qwen3.5-4B's is x111's BP table (sd units, x111's
percentiles); each other model's is its x112 scan (`scan_shift.npz`, summed over the two
boundary tokens, mean over the eight scan prompts, divided by the model's x112 sd unit),
checked against x112's own `table.npz`; percentiles over x111's eligible words.

`--tag T` (default every model): per prompt of the population, every whitespace word with edge
punctuation stripped and lower-cased; x106's stop words excluded; looked up in the table exactly,
then through the inflection fallback (x113_common.Table.lookup). The nominee is the content word
with the highest shift in the model's own table. Records the nominee's shift, percentile and
within-prompt z over the looked-up words, the share of words looked up, agreement with the
x106 unit and the table rank of the x106 unit's top word among the prompt's words.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x113_common import (  # noqa: E402
    ALL_TAGS,
    IMPORTED_TAG,
    PROMPTS,
    TABLES_DIR,
    TABLES_JSON,
    X111_TABLES_JSON,
    X111_TABLES_NPZ,
    X112_CANDIDATES,
    X112_RUN,
    Table,
    atomic_json,
    file_record,
    is_content_word,
    lexicon_words,
    load_prompts,
    load_table,
    normalize_word,
    paths,
    read_json,
    sd_unit,
    sha256_text,
    words_with_spans,
)


def build_tables() -> None:
    cand = read_json(X112_CANDIDATES)
    words = [str(w) for w in cand["words"]]
    x111 = read_json(X111_TABLES_JSON)
    words_sha = sha256_text("\n".join(words))
    if words_sha != cand["words_sha256"] or words_sha != x111["candidates"]["words_sha256"]:
        raise SystemExit("FATAL: the x112 candidate list is not x111's")
    with np.load(X111_TABLES_NPZ) as z:
        eligible = np.asarray(z["eligible"], bool)
        qwen_shift = np.asarray(z["BP"], np.float64)
        qwen_pct = np.asarray(z["pct_BP"], np.float64)
    if len(qwen_shift) != len(words):
        raise SystemExit("FATAL: x111 tables are not over the candidate list")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    tables: dict[str, Any] = {}
    for tag in ALL_TAGS:
        unit = sd_unit(tag)
        if tag == IMPORTED_TAG:
            shift, pct = qwen_shift, qwen_pct
            source = {"table": "x111 BP (data/runs/x111/tables.npz, array BP; x110's 48 over-refused Hard-1K prompts)", "x111_tables": file_record(X111_TABLES_NPZ)}
            check = None
        else:
            d = X112_RUN / tag
            scan = read_json(d / "scan.json")
            if scan.get("model_tag") != tag or scan.get("n_prompts_scanned") != 8 or scan["candidates"]["words_sha256"] != words_sha:
                raise SystemExit(f"FATAL: {tag}: the x112 scan is not the eight-prompt scan over the candidate list")
            if abs(float(scan["sd_unit"]["value"]) - unit) > 1e-9:
                raise SystemExit(f"FATAL: {tag}: x112 sd unit {scan['sd_unit']['value']} differs from {unit}")
            with np.load(d / "scan_shift.npz") as z:
                raw = np.asarray(z["shift"], np.float64)
            if raw.shape != (8, len(words), 2):
                raise SystemExit(f"FATAL: {tag}: scan_shift has shape {raw.shape}")
            shift = raw.sum(axis=2).mean(axis=0) / unit
            with np.load(d / "table.npz") as z:
                own = np.asarray(z["own"], np.float64)
                own_pct = np.asarray(z["pct_own"], np.float64)
            check = {"max_abs_diff_vs_x112_table": float(np.max(np.abs(shift - own)))}
            pct = np.full(len(words), np.nan)
            idx = np.where(eligible)[0]
            order = idx[np.argsort(shift[idx], kind="stable")]
            pct[order] = 100.0 * np.arange(len(order)) / len(order)
            check["max_abs_pct_diff_vs_x112"] = float(np.nanmax(np.abs(pct - own_pct)))
            source = {"table": "x112 own table (data/runs/x112/<tag>/scan_shift.npz summed over the two boundary tokens, mean over the eight scan prompts, divided by the x112 sd unit)", "x112_scan": file_record(d / "scan_shift.npz"), "x112_table": file_record(d / "table.npz")}
        np.savez_compressed(paths(tag)["table"], words=np.asarray(words), shift=shift.astype(np.float32), pct=pct, eligible=eligible)
        el = shift[eligible]
        tables[tag] = {
            "sd_unit": unit,
            "source": source,
            "check": check,
            "npz": file_record(paths(tag)["table"]),
            "distribution_sd_eligible": {q: float(np.quantile(el, v)) for q, v in (("p1", 0.01), ("p5", 0.05), ("p25", 0.25), ("median", 0.5), ("p75", 0.75), ("p95", 0.95), ("p99", 0.99))},
        }
        print(f"  {tag:8s} sd unit {unit:8.3f} eligible median {tables[tag]['distribution_sd_eligible']['median']:+.3f} check {check}")
    out = {
        "experiment": "x113",
        "stage": 1,
        "candidates": {"source": str(X112_CANDIDATES.relative_to(X112_RUN.parents[2])), "n": len(words), "words_sha256": words_sha, "same_list_as_x109_x110_x111_x112": True},
        "eligible_rule": x111["eligible_rule"],
        "n_eligible": int(eligible.sum()),
        "statistic": "per word the shift of the boundary state when the word is appended, summed over the two boundary tokens, mean over the table's prompts, in the model's x106 Hard-1K sd unit; percentile = 100 * ascending rank / n over the eligible words (0 = most negative)",
        "npz_arrays": "words, shift (sd units, candidate order), pct (nan for ineligible), eligible",
        "tables": tables,
    }
    atomic_json(TABLES_JSON, out)
    print(f"x113 tables -> {TABLES_JSON}")


def nominate(tag: str) -> None:
    table = load_table(tag)
    prompts = load_prompts()
    lexicon = lexicon_words()
    rows = []
    for u in prompts["units"]:
        text = u["prompt"]
        spans = words_with_spans(text)
        words = [w for w, _, _ in spans]
        entries: list[dict[str, Any]] = []
        for k, (w, s, e) in enumerate(spans):
            content = is_content_word(w)
            entry = table.entry(w) if content else None
            entries.append({"index": k, "word": w, "norm": normalize_word(w), "span": [s, e], "content": content, **({"table_index": entry["table_index"], "table_word": entry["table_word"], "how": entry["how"], "shift": entry["shift"], "pct": entry["pct"]} if entry else {"table_index": None, "table_word": None, "how": None, "shift": None, "pct": None})})
        looked = [e for e in entries if e["shift"] is not None]
        n_content = sum(1 for e in entries if e["content"])
        x106 = u["x106"]
        top_norm = normalize_word(x106["top_word"])
        row: dict[str, Any] = {
            "id": u["id"],
            "stratum": u["stratum"],
            "prompt": text,
            "prompt_sha256": u["prompt_sha256"],
            "words": entries,
            "n_words": len(words),
            "n_content_words": n_content,
            "n_looked_up": len(looked),
            "share_looked_up": (len(looked) / n_content) if n_content else None,
            "content_words_without_entry": [e["norm"] for e in entries if e["content"] and e["shift"] is None],
        }
        if not looked:
            row.update({"nominee": None, "x106_agreement": None})
            rows.append(row)
            continue
        shifts = np.asarray([e["shift"] for e in looked])
        best = int(np.argmax(shifts))
        nominee = looked[best]
        others = np.delete(shifts, best)
        z = float((shifts[best] - others.mean()) / others.std()) if len(others) > 1 and others.std() > 0 else None
        order = sorted(range(len(looked)), key=lambda j: -shifts[j])
        rank_of_index = {looked[j]["index"]: r + 1 for r, j in enumerate(order)}
        top_entry = next((e for e in entries if e["norm"] == top_norm and e["shift"] is not None), None)
        row["nominee"] = {
            "index": nominee["index"],
            "word": nominee["word"],
            "norm": nominee["norm"],
            "span": nominee["span"],
            "table_word": nominee["table_word"],
            "how": nominee["how"],
            "shift": nominee["shift"],
            "pct": nominee["pct"],
            "z": z,
            "in_lexicon": nominee["norm"] in lexicon,
            "margin_over_second": float(shifts[best] - np.max(others)) if len(others) else None,
        }
        row["x106_agreement"] = {
            "top_norm": top_norm,
            "unit_indices": x106["unit_indices"],
            "nominee_is_top_word": nominee["norm"] == top_norm,
            "nominee_in_unit": nominee["index"] in set(x106["unit_indices"]) or nominee["norm"] in {normalize_word(words[i]) for i in x106["unit_indices"] if i < len(words)},
            "top_word_looked_up": top_entry is not None,
            "top_word_rank": rank_of_index.get(top_entry["index"]) if top_entry else None,
            "top_word_shift": top_entry["shift"] if top_entry else None,
            "n_ranked": len(looked),
        }
        rows.append(row)
    nominated = [r for r in rows if r["nominee"]]
    agree = [r for r in nominated if r["stratum"] != "xstest"]

    def frac(key: str, sub: list[dict]) -> float | None:
        return float(np.mean([r["x106_agreement"][key] for r in sub])) if sub else None

    summary = {
        "n_units": len(rows),
        "n_nominated": len(nominated),
        "share_looked_up_mean": float(np.mean([r["share_looked_up"] for r in rows if r["share_looked_up"] is not None])),
        "nominee_z_median": float(np.nanmedian([r["nominee"]["z"] for r in nominated if r["nominee"]["z"] is not None])),
        "nominee_pct_median": float(np.nanmedian([r["nominee"]["pct"] for r in nominated if r["nominee"]["pct"] is not None])),
        "nominee_in_lexicon": float(np.mean([r["nominee"]["in_lexicon"] for r in nominated])),
        "how_counts": {h: int(sum(1 for r in nominated if r["nominee"]["how"].split(":")[0] == h)) for h in ("exact", "inflection", "hyphen")},
        "agreement_hard1k": {"n": len(agree), "nominee_is_top_word": frac("nominee_is_top_word", agree), "nominee_in_unit": frac("nominee_in_unit", agree), "top_word_rank_median": float(np.nanmedian([r["x106_agreement"]["top_word_rank"] for r in agree if r["x106_agreement"]["top_word_rank"]])), "top_word_in_top3": float(np.mean([(r["x106_agreement"]["top_word_rank"] or 99) <= 3 for r in agree]))},
        "agreement_xstest": {"n": len(nominated) - len(agree), "nominee_is_top_word": frac("nominee_is_top_word", [r for r in nominated if r["stratum"] == "xstest"])},
    }
    for s in ("core", "llama_extra"):
        sub = [r for r in agree if r["stratum"] == s]
        summary["agreement_hard1k"][f"nominee_is_top_word_{s}"] = frac("nominee_is_top_word", sub)
    from collections import Counter

    summary["most_nominated_hard1k"] = [{"word": w, "n": n} for w, n in Counter(r["nominee"]["norm"] for r in agree).most_common(20)]
    out = {
        "experiment": "x113",
        "stage": 1,
        "model_tag": tag,
        "table": file_record(paths(tag)["table"]),
        "prompts": file_record(PROMPTS),
        "rule": "content words (x106 stop words excluded), exact lower-cased lookup in the candidate list, then the inflection fallback (possessive 's, plural -s / -es / -ies, -ed, -ing with the final e or doubled consonant restored), then for a hyphenated word its highest-shift part; the nominee is the content word with the highest shift in this model's table; z = (nominee shift - mean of the other looked-up words) / their sd",
        "summary": summary,
        "rows": rows,
    }
    atomic_json(paths(tag)["nominations"], out)
    a = summary["agreement_hard1k"]
    print(f"x113 nominate {tag}: {summary['n_nominated']}/{len(rows)} nominated; looked up {summary['share_looked_up_mean']:.3f}; z med {summary['nominee_z_median']:.2f}; = x106 top {a['nominee_is_top_word']:.3f}, in unit {a['nominee_in_unit']:.3f}, top word rank med {a['top_word_rank_median']:.0f}; how {summary['how_counts']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", action="store_true")
    parser.add_argument("--tags", nargs="*", default=list(ALL_TAGS))
    args = parser.parse_args()
    if args.tables:
        build_tables()
        return
    for tag in args.tags:
        nominate(tag)


if __name__ == "__main__":
    main()
