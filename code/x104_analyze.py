"""x104 stage 3: rebuild the x87, x88 and x89 result tables on GuardReasoner-8B labels.

The three analyses are the x87/x88/x89 analysis scripts themselves, run with
their OR-Bench reader pointed at the x104 labels (StrongREJECT, IFEval and
GSM8K values pass through unchanged) and their pairing tolerant of the rows
GuardReasoner left unparsed, which are dropped together with their paired
rows.  Outputs keep the schemas of the files they replace:

    data/results/x104_external_safety_utility.json   (x87 schema)
    data/results/x104_cross_model_deep_dose.json     (x88 schema)
    data/results/x104_positive_dose_extension.json   (x89 schema)

plus `data/results/x104_judge_comparison.json`, the incumbent-vs-GuardReasoner
deltas side by side.  When shard outputs are present they are merged first
into `x104_judgments.jsonl` (full traces) and `x104_labels.jsonl` (the fields
read here).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from x104_common import (
    BENCHMARKS,
    JUDGE,
    JUDGMENTS_PATH,
    MANIFEST_PATH,
    N_ROWS,
    RESULT_PATHS,
    RESULTS,
    ROWS_PATH,
    RUN,
    SOURCES,
    atomic_json,
    file_record,
    keys_sha256,
    load_rows,
    read_json,
    read_jsonl,
    read_shard_jsonl,
    sha256_file,
    write_jsonl,
)

import x87_analyze
import x88_analyze
import x89_analyze
from x87_common import paired_arrays as _paired_arrays

LABELS_PATH = RUN / "x104_labels.jsonl"
COMPARISON_PATH = RESULTS / "x104_judge_comparison.json"
X102_GUARDREASONER = RUN.parent / "x102" / "x102_judgments_guardreasoner.jsonl"
LABEL_FIELDS = (
    "key", "tag", "experiment", "condition", "id", "benchmark",
    "decoding", "parse_ok", "refusal", "fields", "new_tokens", "attempts",
)
ORBENCH_FILE_RE = re.compile(r"^(x8[789])_orbench_(.+)\.json$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-merge", action="store_true", help="use the existing labels file as is")
    return parser.parse_args()


# --------------------------------------------------------------------------
# Merge: shard files + resample files -> judgments.jsonl + labels.jsonl
# --------------------------------------------------------------------------


def merge_shards() -> None:
    shard_files = sorted(RUN.glob("x104_judgments_shard*of*.jsonl"))
    if not shard_files:
        if not LABELS_PATH.exists():
            raise SystemExit(f"FATAL: no shard files and no {LABELS_PATH}")
        print(f"no shard files; using {LABELS_PATH}")
        return
    rows = {row["key"]: row for row in load_rows()}
    greedy: dict[str, dict] = {}
    n_duplicate = 0
    for path in shard_files:
        for record in read_shard_jsonl(path):
            # A shard carried by jobs on several partitions can judge a row twice
            # (identical settings); the first record in file order is kept.
            if record["key"] in greedy:
                n_duplicate += 1
                continue
            greedy[record["key"]] = record
    resampled: dict[str, dict] = {}
    for path in sorted(RUN.glob("x104_resample_shard*of*.jsonl")):
        for record in read_shard_jsonl(path):
            if record["key"] in resampled:
                continue
            if record["key"] not in greedy or greedy[record["key"]]["parse_ok"]:
                raise SystemExit(f"FATAL: resample record for a row that was not greedy-unparsed: {record['key']}")
            resampled[record["key"]] = record
    missing = set(rows) - set(greedy)
    extra = set(greedy) - set(rows)
    if missing or extra:
        raise SystemExit(f"FATAL: judgments cover {len(greedy)} rows; missing {len(missing)}, extra {len(extra)}")
    unresampled = [k for k, r in greedy.items() if not r["parse_ok"] and k not in resampled]
    if n_duplicate:
        print(f"{n_duplicate} rows judged twice across shard files; first record kept")
    if unresampled:
        raise SystemExit(f"FATAL: {len(unresampled)} greedy-unparsed rows have no resample record")
    merged = []
    for key in sorted(rows):
        record = dict(greedy[key])
        if key in resampled:
            final = resampled[key]
            record = dict(final)
            record["greedy_output"] = greedy[key]["output"]
            record["greedy_new_tokens"] = greedy[key]["new_tokens"]
        else:
            record["attempts"] = []
        record["incumbent_label"] = rows[key]["incumbent_label"]
        record["incumbent_refusal"] = rows[key]["incumbent_refusal"]
        merged.append(record)
    write_jsonl(JUDGMENTS_PATH, merged)
    labels = [
        {k: r.get(k) for k in LABEL_FIELDS} | {"incumbent_label": r["incumbent_label"], "incumbent_refusal": r["incumbent_refusal"]}
        for r in merged
    ]
    write_jsonl(LABELS_PATH, labels)
    n_unparsed = sum(1 for r in merged if not r["parse_ok"])
    print(f"merged {len(merged)} judgments from {len(shard_files)} shards; {len(resampled)} resampled, {n_unparsed} unparsed")


def load_labels() -> list[dict]:
    manifest = read_json(MANIFEST_PATH)
    labels = read_jsonl(LABELS_PATH)
    if len(labels) != N_ROWS or keys_sha256(r["key"] for r in labels) != manifest["rows_keys_sha256"]:
        raise SystemExit(f"FATAL: {LABELS_PATH} does not carry the registered {N_ROWS} row keys")
    for r in labels:
        if r["parse_ok"] and r["refusal"] not in (0, 1):
            raise SystemExit(f"FATAL: parsed row without a refusal bit: {r['key']}")
        if not r["parse_ok"] and r["refusal"] is not None:
            raise SystemExit(f"FATAL: unparsed row with a refusal bit: {r['key']}")
    return labels


# --------------------------------------------------------------------------
# Patches: OR-Bench refusal from x104, pairing tolerant of unparsed rows only
# --------------------------------------------------------------------------


class Patches:
    def __init__(self, labels: list[dict], manifest: dict):
        self.refusal: dict[tuple[str, str, str, str], dict[str, int]] = {}
        self.unparsed_ids: set[str] = set()
        self.unparsed_keys: list[str] = []
        for r in labels:
            cell = (r["experiment"], r["tag"], r["condition"], r["benchmark"])
            if r["parse_ok"]:
                self.refusal.setdefault(cell, {})[r["id"]] = int(r["refusal"])
            else:
                self.refusal.setdefault(cell, {})
                self.unparsed_ids.add(r["id"])
                self.unparsed_keys.append(r["key"])
        self.judgment_sha = {
            (s["experiment"], s["tag"], s["condition"]): s["judgment"]["sha256"] for s in manifest["sources"]
        }
        self.verified: set[Path] = set()

    def source_of(self, path: Path) -> tuple[str, str, str]:
        match = ORBENCH_FILE_RE.match(path.name)
        if match is None:
            raise SystemExit(f"FATAL: not an x87/x88/x89 OR-Bench judgment file: {path}")
        experiment, condition = match.groups()
        tag = path.parent.name if experiment == "x88" else "qwen"
        key = (experiment, tag, condition)
        if key not in self.judgment_sha:
            raise SystemExit(f"FATAL: {path} is not one of the 49 x104 sources")
        if path not in self.verified:
            if sha256_file(path) != self.judgment_sha[key]:
                raise SystemExit(f"FATAL: {path} is not the file x104 built its rows from")
            self.verified.add(path)
        return key

    def indexed(self, path: Path, benchmark: str, field: str) -> dict[str, float]:
        if benchmark not in BENCHMARKS:
            return x87_indexed(path, benchmark, field)
        experiment, tag, condition = self.source_of(Path(path))
        if field != "refusal":
            raise SystemExit(f"FATAL: unexpected OR-Bench field {field!r}")
        return {i: float(v) for i, v in self.refusal[(experiment, tag, condition, benchmark)].items()}

    def paired_arrays(self, base: dict[str, float], edited: dict[str, float]):
        dropped = set(base) ^ set(edited)
        if dropped - self.unparsed_ids:
            raise ValueError("condition item IDs do not match base")
        ids = sorted(set(base) & set(edited))
        if not ids:
            raise ValueError("no paired items")
        return ids, np.asarray([base[i] for i in ids]), np.asarray([edited[i] for i in ids])


x87_indexed = x87_analyze.indexed


def install(patches: Patches) -> None:
    # x87's result file holds the thirteen conditions that existed before x89 appended
    # its seven to EXTENSION_CONDITIONS; x89_analyze carries those seven.
    x87_analyze.EXTENSION_CONDITIONS = tuple(
        c for c in x87_analyze.EXTENSION_CONDITIONS if c not in x89_analyze.X89_CONDITIONS
    )
    x87_analyze.indexed = patches.indexed
    x87_analyze.paired_arrays = patches.paired_arrays
    x88_analyze.indexed = patches.indexed
    x88_analyze.paired_arrays = patches.paired_arrays
    x89_analyze.indexed = patches.indexed
    x89_analyze.paired_arrays = patches.paired_arrays


def run(module, argv: list[str]) -> None:
    saved = sys.argv
    sys.argv = [module.__name__] + argv
    try:
        module.main()
    finally:
        sys.argv = saved


def stamp(path: Path, source_experiment: str, patches: Patches) -> dict:
    result = read_json(path)
    result["experiment"] = "x104"
    result["rebuilt_from"] = source_experiment
    result["refusal_judge"] = {
        "model_id": JUDGE["model_id"],
        "revision": JUDGE["revision"],
        "field": "Completion: refusal/compliance",
        "labels": file_record(LABELS_PATH),
    }
    result["unparsed_rows"] = {"n": len(patches.unparsed_keys), "keys": sorted(patches.unparsed_keys)}
    atomic_json(path, result)
    return result


# --------------------------------------------------------------------------
# Incumbent-vs-GuardReasoner comparison
# --------------------------------------------------------------------------


def delta_row(block: dict) -> dict:
    return {
        "mean_delta": block["mean_delta"],
        "two_sided_ci": block["two_sided_ci"],
        "one_sided_lower": block["one_sided_lower"],
        "one_sided_upper": block["one_sided_upper"],
        "n": block["n"],
    }


def comparison(labels: list[dict], new: dict[str, dict]) -> dict:
    old = {
        "x87": read_json(RESULTS / "x87_external_safety_utility.json"),
        "x88": read_json(RESULTS / "x88_cross_model_deep_dose.json"),
        "x89": read_json(RESULTS / "x89_positive_dose_extension.json"),
    }
    endpoints = ("orbench_hard_refusal", "orbench_toxic_refusal")
    cells = []
    for condition in old["x87"]["comparisons"]:
        o, n = old["x87"]["comparisons"][condition], new["x87"]["comparisons"][condition]
        cells.append(
            {
                "model": "qwen",
                "condition": condition,
                "source": "x87",
                **{e: {"incumbent": delta_row(o[e]), "guardreasoner": delta_row(n[e])} for e in endpoints},
                "gates": {"incumbent": o["gates"], "guardreasoner": n["gates"]},
                "claim_pass": {"incumbent": o["claim_pass"], "guardreasoner": n["claim_pass"]},
            }
        )
    for condition in old["x89"]["comparisons"]:
        o, n = old["x89"]["comparisons"][condition], new["x89"]["comparisons"][condition]
        cells.append(
            {
                "model": "qwen",
                "condition": condition,
                "source": "x89",
                **{e: {"incumbent": delta_row(o[e]), "guardreasoner": delta_row(n[e])} for e in endpoints},
            }
        )
    for tag, block in old["x88"]["models"].items():
        if tag == "qwen":
            continue
        for condition, o in block["comparisons"].items():
            n = new["x88"]["models"][tag]["comparisons"][condition]
            cells.append(
                {
                    "model": tag,
                    "condition": condition,
                    "source": "x88",
                    **{e: {"incumbent": delta_row(o[e]), "guardreasoner": delta_row(n[e])} for e in endpoints},
                }
            )
    base_rates = {
        "qwen": {
            "incumbent": {e: old["x89"]["base_rates"][e] for e in endpoints},
            "guardreasoner": {e: new["x89"]["base_rates"][e] for e in endpoints},
        }
    }
    for tag, block in old["x88"]["models"].items():
        if tag == "qwen":
            continue
        base_rates[tag] = {
            "incumbent": {e: block["base_rates"][e] for e in endpoints},
            "guardreasoner": {e: new["x88"]["models"][tag]["base_rates"][e] for e in endpoints},
        }
    parsed = [r for r in labels if r["parse_ok"]]
    agree = sum(int(r["refusal"] == r["incumbent_refusal"]) for r in parsed)
    by_benchmark = {}
    for benchmark in BENCHMARKS:
        rows = [r for r in parsed if r["benchmark"] == benchmark]
        by_benchmark[benchmark] = {
            "n": len(rows),
            "agreement": sum(int(r["refusal"] == r["incumbent_refusal"]) for r in rows) / len(rows),
            "incumbent_refusal_rate": sum(r["incumbent_refusal"] for r in rows) / len(rows),
            "guardreasoner_refusal_rate": sum(r["refusal"] for r in rows) / len(rows),
        }
    x102_check = None
    if X102_GUARDREASONER.exists():
        x102 = {r["key"]: r for r in read_jsonl(X102_GUARDREASONER)}
        mine = {r["key"]: r for r in labels}
        both = [k for k in x102 if x102[k]["parse_ok"] and mine[k]["parse_ok"]]
        x102_check = {
            "n_sample_rows": len(x102),
            "n_both_parsed": len(both),
            "agreement_with_x102_run": sum(int(x102[k]["refusal"] == mine[k]["refusal"]) for k in both) / len(both),
            "x104_greedy_unparsed_on_sample": sum(1 for k in x102 if mine[k]["decoding"] != "greedy"),
            "x102_unparsed_on_sample": sum(1 for k in x102 if not x102[k]["parse_ok"]),
        }
    decoding = {}
    for r in labels:
        decoding[r["decoding"]] = decoding.get(r["decoding"], 0) + 1
    return {
        "schema_version": 1,
        "experiment": "x104",
        "rows": len(labels),
        "parsed": len(parsed),
        "unparsed": len(labels) - len(parsed),
        "decoding_counts": decoding,
        "mean_new_tokens_greedy": float(np.mean([r["new_tokens"] for r in labels if r["decoding"] == "greedy"])),
        "binary_agreement_with_incumbent": agree / len(parsed),
        "by_benchmark": by_benchmark,
        "x102_sample_check": x102_check,
        "base_rates": base_rates,
        "cells": cells,
    }


def print_table(cmp: dict) -> None:
    print(f"\nrows {cmp['rows']}, parsed {cmp['parsed']}, unparsed {cmp['unparsed']}, "
          f"agreement with incumbent {cmp['binary_agreement_with_incumbent']:.4f}")
    print(f"{'model':8s} {'cond':9s} {'Hard old':>9s} {'Hard new':>9s} {'Tox old':>9s} {'Tox new':>9s}  gates(old->new)")
    for c in cmp["cells"]:
        h_o = c["orbench_hard_refusal"]["incumbent"]["mean_delta"]
        h_n = c["orbench_hard_refusal"]["guardreasoner"]["mean_delta"]
        t_o = c["orbench_toxic_refusal"]["incumbent"]["mean_delta"]
        t_n = c["orbench_toxic_refusal"]["guardreasoner"]["mean_delta"]
        gates = ""
        if "gates" in c:
            changed = [k for k in c["gates"]["incumbent"] if c["gates"]["incumbent"][k] != c["gates"]["guardreasoner"][k]]
            gates = " ".join(f"{k}:{int(c['gates']['incumbent'][k])}->{int(c['gates']['guardreasoner'][k])}" for k in changed)
        print(f"{c['model']:8s} {c['condition']:9s} {h_o:+9.4f} {h_n:+9.4f} {t_o:+9.4f} {t_n:+9.4f}  {gates}")


def main() -> None:
    args = parse_args()
    if not args.skip_merge:
        merge_shards()
    labels = load_labels()
    manifest = read_json(MANIFEST_PATH)
    patches = Patches(labels, manifest)
    install(patches)
    x87_out, x88_out, x89_out = (str(RESULT_PATHS[k]) for k in ("x87", "x88", "x89"))
    run(x87_analyze, ["--out", x87_out])
    run(x88_analyze, ["--x87-results", x87_out, "--out", x88_out])
    run(x89_analyze, ["--x87-results", x87_out, "--out", x89_out])
    if len(patches.verified) != len(SOURCES):
        raise SystemExit(f"FATAL: the analyses read {len(patches.verified)} OR-Bench files, expected {len(SOURCES)}")
    new = {k: stamp(RESULT_PATHS[k], k, patches) for k in ("x87", "x88", "x89")}
    cmp = comparison(labels, new)
    atomic_json(COMPARISON_PATH, cmp)
    print_table(cmp)
    print(f"successful_candidates: {new['x87']['successful_candidates']}")
    print(f"wrote {x87_out}, {x88_out}, {x89_out}, {COMPARISON_PATH}")


if __name__ == "__main__":
    main()
