"""x118 Stage 1: collect every (original, edited) prompt pair of the substitution experiments.

Reads the frozen cohorts of x106 (S / C / R), x108 (S / R) and each model's x113 texts
(S_T / C_T), writes one row per distinct (original sha256, edited sha256) pair to
`data/runs/x118/pairs.jsonl` with the owners that use it, and flags the pairs the human
read-through covers.

    python code/x118_collect_pairs.py [--out data/runs/x118/pairs.jsonl]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ROOT = HERE.parent
DATA = ROOT / "data"
X106_COHORT = DATA / "runs" / "x106" / "x106_cohort.json"
X106_READTHROUGH = DATA / "runs" / "x106" / "x106_readthrough.md"
X108_COHORT = DATA / "runs" / "x108" / "x108_cohort.json"
X108_READTHROUGH = DATA / "runs" / "x108" / "x108_readthrough.md"
X113_RUN = DATA / "runs" / "x113"
OUT = DATA / "runs" / "x118" / "pairs.jsonl"

X113_TAGS = ("qwen", "qwen08b", "qwen2b", "qwen9b", "qwen27b", "llama", "gemma", "phi")
# x106 / x108 text label -> x118 arm
ARM_OF_TEXT = {"S": "S", "C": "C", "R": "R", "S_T": "S", "C_T": "C"}

UNIT_HEAD = re.compile(r"^##\s+(\S+)\s")
TEXT_LINE = re.compile(r"^-\s+(?:\*\*(?P<label>[OSCR])\*\*|(?P<plain>[OSCR]))\s*:\s*(?P<rest>.*)$")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_readthrough(path: Path) -> dict[tuple[str, str], str]:
    """{(unit id, text label): "ok" | "bad"} for every text line the human read.

    A line the human marked with `BAD:` (anywhere on the line) is a rejection; every other
    listed text is `ok`. Lines for a unit without the text (`S absent: ...`) are skipped.
    """
    marks: dict[tuple[str, str], str] = {}
    unit_id = None
    for line in path.read_text(encoding="utf-8").splitlines():
        head = UNIT_HEAD.match(line)
        if head:
            unit_id = head.group(1)
            continue
        if unit_id is None:
            continue
        match = TEXT_LINE.match(line.strip())
        if not match:
            continue
        label = match.group("label") or match.group("plain")
        rest = match.group("rest")
        marks[(unit_id, label)] = "bad" if "BAD:" in rest or "BAD:" in line else "ok"
    return marks


class PairTable:
    """Pairs keyed by (original sha256, edited sha256), with every owner kept."""

    def __init__(self) -> None:
        self.by_key: dict[tuple[str, str], dict] = {}
        self.order: list[tuple[str, str]] = []

    def add(
        self,
        *,
        source: str,
        model: str | None,
        arm: str,
        unit_id: str,
        unit_kind: str | None,
        stratum: str | None,
        pool: str | None,
        text_label: str,
        original: str,
        edited: str,
        nominee: str | None,
        readthrough: str | None,
        same_as_x106: bool | None,
        hedge: bool,
    ) -> None:
        key = (sha256_text(original), sha256_text(edited))
        owner = {
            "source": source,
            "model": model,
            "arm": arm,
            "unit_id": unit_id,
            "text_label": text_label,
            "unit_kind": unit_kind,
            "stratum": stratum,
            "pool": pool,
            "nominee": nominee,
            "readthrough": readthrough,
            "same_as_x106": same_as_x106,
            "hedge": hedge,
        }
        row = self.by_key.get(key)
        if row is None:
            row = {
                "pair_id": f"x118:{key[0][:12]}:{key[1][:12]}",
                "original": original,
                "edited": edited,
                "original_sha256": key[0],
                "edited_sha256": key[1],
                "arm": arm,
                "sources": [source],
                "owners": [owner],
            }
            self.by_key[key] = row
            self.order.append(key)
            return
        if row["arm"] != arm:
            # the same text change used in two arms: keep the first arm, record both owners
            row["arm_conflict"] = sorted({row["arm"], arm})
        if source not in row["sources"]:
            row["sources"].append(source)
        row["owners"].append(owner)

    def rows(self) -> list[dict]:
        out = []
        for key in self.order:
            row = self.by_key[key]
            owners = row["owners"]
            marks = {o["readthrough"] for o in owners if o["readthrough"] is not None}
            row = dict(row)
            row["n_owners"] = len(owners)
            row["models"] = sorted({o["model"] for o in owners if o["model"]})
            row["unit_kinds"] = sorted({o["unit_kind"] for o in owners if o["unit_kind"]})
            row["strata"] = sorted({o["stratum"] for o in owners if o["stratum"]})
            row["pools"] = sorted({o["pool"] for o in owners if o["pool"]})
            row["hedge"] = any(o["hedge"] for o in owners)
            row["readthrough"] = ("bad" if "bad" in marks else "ok") if marks else None
            out.append(row)
        return out


def collect_x106(table: PairTable) -> dict:
    cohort = read_json(X106_COHORT)
    marks = parse_readthrough(X106_READTHROUGH)
    covered = set(cohort["human_readthrough_ids"])
    counts = {"S": 0, "C": 0, "R": 0}
    for unit in cohort["units"]:
        unit_id = unit["id"]
        for label in ("S", "C", "R"):
            edited = unit["prompts"].get(label)
            if not edited:
                continue
            counts[label] += 1
            table.add(
                source="x106",
                model=None,
                arm=ARM_OF_TEXT[label],
                unit_id=unit_id,
                unit_kind=unit.get("unit_kind"),
                stratum=unit.get("stratum"),
                pool=unit.get("source"),
                text_label=label,
                original=unit["prompts"]["O"],
                edited=edited,
                nominee=unit.get("unit"),
                readthrough=marks.get((unit_id, label)) if unit_id in covered else None,
                same_as_x106=None,
                hedge=unit.get("unit_kind") == "hedge",
            )
    return {"cohort_sha256": cohort["cohort_sha256"], "n_units": cohort["n_units"], "pairs": counts}


def collect_x108(table: PairTable) -> dict:
    cohort = read_json(X108_COHORT)
    marks = parse_readthrough(X108_READTHROUGH)
    covered = set(cohort["human_readthrough_ids"])
    counts = {"S": 0, "R": 0}
    for unit in cohort["units"]:
        unit_id = unit["id"]
        for label in ("S", "R"):
            edited = unit["prompts"].get(label)
            if not edited:
                continue
            counts[label] += 1
            table.add(
                source="x108",
                model=None,
                arm=ARM_OF_TEXT[label],
                unit_id=unit_id,
                unit_kind=unit.get("unit_kind"),
                stratum=unit.get("stratum"),
                pool=unit.get("source"),
                text_label=label,
                original=unit["prompts"]["O"],
                edited=edited,
                nominee=unit.get("unit"),
                readthrough=marks.get((unit_id, label)) if unit_id in covered else None,
                same_as_x106=None,
                hedge=unit.get("unit_kind") == "hedge",
            )
    return {"cohort_sha256": cohort["cohort_sha256"], "n_units": cohort["n_units"], "pairs": counts}


def collect_x113(table: PairTable) -> dict:
    per_model = {}
    for tag in X113_TAGS:
        texts = read_json(X113_RUN / tag / "texts.json")
        counts = {"S_T": 0, "C_T": 0}
        for unit in texts["units"]:
            for label in ("S_T", "C_T"):
                edited = unit["texts"].get(label)
                if not edited:
                    continue
                counts[label] += 1
                nominee = (unit.get("nominee") or {}).get("word")
                table.add(
                    source="x113",
                    model=tag,
                    arm=ARM_OF_TEXT[label],
                    unit_id=unit["id"],
                    unit_kind=unit.get("unit_kind"),
                    stratum=unit.get("stratum"),
                    pool=unit.get("source"),
                    text_label=label,
                    original=unit["prompt"],
                    edited=edited,
                    nominee=nominee,
                    readthrough=None,
                    same_as_x106=bool((unit.get("same_as_x106") or {}).get(label)),
                    hedge=unit.get("unit_kind") == "hedge",
                )
        per_model[tag] = {"texts_sha256": texts["texts_sha256"], "n_units": texts["n_units"], "pairs": counts}
    return per_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    table = PairTable()
    inputs = {
        "x106": collect_x106(table),
        "x108": collect_x108(table),
        "x113": collect_x113(table),
    }
    rows = table.rows()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            row = {k: v for k, v in row.items() if k != "owners"} | {"owners": row["owners"]}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_source_arm: dict[str, dict[str, int]] = {}
    for row in rows:
        for source in row["sources"]:
            by_source_arm.setdefault(source, {}).setdefault(row["arm"], 0)
            by_source_arm[source][row["arm"]] += 1
    n_owner_rows = sum(row["n_owners"] for row in rows)
    summary = {
        "experiment": "x118",
        "stage": 1,
        "n_pairs": len(rows),
        "n_owner_rows": n_owner_rows,
        "distinct_pairs_by_source_arm": by_source_arm,
        "pairs_shared_by_two_or_more_owners": sum(1 for row in rows if row["n_owners"] > 1),
        "readthrough_covered": sum(1 for row in rows if row["readthrough"] is not None),
        "readthrough_bad": sum(1 for row in rows if row["readthrough"] == "bad"),
        "inputs": inputs,
        "out": str(out_path),
    }
    print(json.dumps(summary, indent=1))
    (out_path.parent / "pairs_summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
