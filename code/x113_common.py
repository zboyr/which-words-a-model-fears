"""x113: on the Hard-1K prompts most of the eight models refuse, does each model's own vocabulary
table nominate the word that carries the refusal, and does replacing it in place by the table's
lowest-shift synonym remove the refusal?

Plan: knowledge/plans/x113-cross-model-table-substitution.md. x106's substitution pipeline with
the vocabulary table (x111's BP table for Qwen3.5-4B, the x112 table for the seven other models)
in place of the direction's deletion pass. The population is x106's cohort narrowed to the
Hard-1K prompts refused at or above 0.5 in at least five of the eight models (core), the further
Hard-1K cohort prompts Llama-3.2-3B refuses at all (llama_extra), and x106's 100 XSTest prompts.
Arms: O (x106's O_base rows reused), S_T (table nominee replaced) and C_T (table-nearest-zero
inert word replaced). Rendering, sampling, seeds, the read site, the directions and the judge are
x106's, imported, not restated.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x106_common import (  # noqa: E402,F401  (re-exported for the stage scripts)
    ALL_TAGS,
    DATA,
    FRESH_TAGS,
    IMPORTED_TAG,
    MODELS,
    N_RESPONSES,
    STOP_WORDS,
    atomic_json,
    cohort_path,
    file_record,
    generations_path as x106_generations_path,
    inputs_path as x106_inputs_path,
    is_content_word,
    is_xstest,
    lexicon_words,
    load_cohort,
    normalize_word,
    projections_path as x106_projections_path,
    read_json,
    refusal_path as x106_refusal_path,
    render_chat,
    sha256_file,
    sha256_text,
    stable_seed as x106_stable_seed,
    words_with_spans,
)

RUN = DATA / "runs" / "x113"
PROMPTS = RUN / "prompts.json"
TABLES_DIR = RUN / "tables"
TABLES_JSON = RUN / "tables.json"
DRAFTS = RUN / "drafts.jsonl"
RESULTS = RUN / "results.json"
RESULTS_MD = RUN / "results.md"
EXCLUSIONS = RUN / "x85_exclusions.json"
X111_RUN = DATA / "runs" / "x111"
X111_TABLES_JSON = X111_RUN / "tables.json"
X111_TABLES_NPZ = X111_RUN / "tables.npz"
X112_RUN = DATA / "runs" / "x112"
X112_CANDIDATES = X112_RUN / "candidates.json"
X112_COMPARISON = X112_RUN / "table_comparison.json"

LABELS = {
    "qwen": "Qwen3.5-4B",
    "qwen08b": "Qwen3.5-0.8B",
    "qwen2b": "Qwen3.5-2B",
    "qwen9b": "Qwen3.5-9B",
    "qwen27b": "Qwen3.5-27B",
    "llama": "Llama-3.2-3B",
    "gemma": "Gemma-4-E2B",
    "phi": "Phi-4-mini",
}
SEED = 113
STRATA = ("core", "llama_extra", "xstest")
HARD_STRATA = ("core", "llama_extra")
CORE_RATE = 0.5
CORE_MIN_MODELS = 5
LLAMA_EXTRA_MAX = 100
# x113 arm -> the x106 text label that fixes its sampling seed and the x106 arm whose rows a
# byte-identical text reuses.
ARMS = ("S_T", "C_T")
X106_TEXT_OF_ARM = {"S_T": "S", "C_T": "C"}
X106_ARM_OF_ARM = {"S_T": "S_base", "C_T": "C_base"}
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 11_300
ALPHA = 0.05


# --- paths ---------------------------------------------------------------------------------


def exclusions_path() -> Path:
    return EXCLUSIONS


def tag_dir(tag: str) -> Path:
    if tag not in ALL_TAGS:
        raise ValueError(f"unknown x113 tag {tag!r}")
    return RUN / tag


def paths(tag: str) -> dict[str, Path]:
    d = tag_dir(tag)
    return {
        "dir": d,
        "table": TABLES_DIR / f"{tag}.npz",
        "nominations": d / "nominations.json",
        "texts": d / "texts.json",
        "gen": d / "generations",
        "genlocks": d / "genlocks",
        "judge": d / "judge",
        "refusal2": d / "judge" / "refusal2.jsonl",
        "uptake": d / "judge" / "uptake.jsonl",
        "cost_log": d / "judge" / "cost_log.jsonl",
        "projections": d / "projections.json",
    }


def sd_unit(tag: str) -> float:
    """The model's x106 Hard-1K sd unit: x109's constant for Qwen3.5-4B, x112's for the others."""
    if tag == IMPORTED_TAG:
        from x109_scan_common import X106_HARD1K_SD

        return float(X106_HARD1K_SD)
    from x112_common import sd_unit as x112_sd_unit

    return float(x112_sd_unit(tag))


def stable_seed(tag: str, prompt_id: str, arm: str, retry: bool = False) -> int:
    """x106's seed for the x106 text label of the arm, so a text identical to x106's S / C text of
    the same prompt samples with the same seed."""
    return x106_stable_seed(tag, prompt_id, X106_TEXT_OF_ARM[arm], retry=retry)


# --- the vocabulary table -------------------------------------------------------------------


def inflection_forms(norm: str) -> list[str]:
    """Base forms to try when a word has no exact entry: possessive 's, plural -s / -es (-ies -> -y),
    -ed and -ing with the final e or the doubled consonant restored."""
    forms: list[str] = []
    w = norm
    if w.endswith("'s") and len(w) > 3:
        forms.append(w[:-2])
        w = w[:-2]
    if w.endswith("ies") and len(w) > 4:
        forms.append(w[:-3] + "y")
    if w.endswith("es") and len(w) > 3:
        forms.append(w[:-2])
    if w.endswith("s") and len(w) > 2:
        forms.append(w[:-1])
    if w.endswith("ed") and len(w) > 3:
        forms.extend([w[:-2], w[:-1]])
        if len(w) > 4 and w[-3] == w[-4]:
            forms.append(w[:-3])
    if w.endswith("ing") and len(w) > 4:
        forms.extend([w[:-3], w[:-3] + "e"])
        if len(w) > 5 and w[-4] == w[-5]:
            forms.append(w[:-4])
    seen: list[str] = []
    for f in forms:
        if f and f != norm and f not in seen:
            seen.append(f)
    return seen


class Table:
    """One model's per-word shift over the x109 candidate list, in the model's sd unit."""

    def __init__(self, tag: str, words: list[str], shift: np.ndarray, pct: np.ndarray, eligible: np.ndarray, unit: float):
        self.tag = tag
        self.words = words
        self.index = {w: i for i, w in enumerate(words)}
        self.shift = np.asarray(shift, dtype=np.float64)
        self.pct = np.asarray(pct, dtype=np.float64)
        self.eligible = np.asarray(eligible, dtype=bool)
        self.unit = float(unit)

    def lookup(self, word: str) -> tuple[int | None, str | None]:
        """(candidate index, how) for a text word: exact lower-cased entry, an inflection fallback,
        or for a hyphenated word the highest-shift part after stop words are removed."""
        norm = normalize_word(word)
        if not norm:
            return None, None
        if norm in self.index:
            return self.index[norm], "exact"
        for form in inflection_forms(norm):
            if form in self.index:
                return self.index[form], f"inflection:{form}"
        if "-" in norm:
            best = None
            for part in norm.split("-"):
                if not part or part in STOP_WORDS:
                    continue
                idx, how = self.lookup(part)
                if idx is not None and (best is None or self.shift[idx] > self.shift[best[0]]):
                    best = (idx, f"hyphen:{part}" + ("" if how == "exact" else f"/{how}"))
            if best is not None:
                return best
        return None, None

    def entry(self, word: str) -> dict[str, Any] | None:
        idx, how = self.lookup(word)
        if idx is None:
            return None
        return {"table_index": int(idx), "table_word": self.words[idx], "how": how, "shift": float(self.shift[idx]), "pct": (float(self.pct[idx]) if np.isfinite(self.pct[idx]) else None), "eligible": bool(self.eligible[idx])}

    def phrase_entry(self, text: str) -> dict[str, Any] | None:
        """For a word or phrase: the entry of its highest-shift word after stop words are removed;
        None when no word has an entry."""
        best = None
        for raw in text.split():
            if not is_content_word(raw):
                continue
            e = self.entry(raw)
            if e is not None and (best is None or e["shift"] > best["shift"]):
                best = {**e, "word": normalize_word(raw)}
        return best


def load_table(tag: str) -> Table:
    meta = read_json(TABLES_JSON)
    if meta.get("experiment") != "x113" or tag not in meta["tables"]:
        raise SystemExit(f"FATAL: x113 table for {tag} is not built")
    with np.load(paths(tag)["table"]) as z:
        words = [str(w) for w in z["words"]]
        shift, pct, eligible = z["shift"], z["pct"], z["eligible"]
    if sha256_text("\n".join(words)) != meta["candidates"]["words_sha256"]:
        raise SystemExit(f"FATAL: {tag}: table words differ from the candidate list")
    return Table(tag, words, shift, pct, eligible, meta["tables"][tag]["sd_unit"])


# --- population, texts, cells --------------------------------------------------------------


def load_prompts() -> dict[str, Any]:
    obj = read_json(PROMPTS)
    if obj.get("experiment") != "x113" or obj.get("status") != "frozen":
        raise SystemExit("FATAL: x113 prompts.json is missing or not frozen")
    if obj.get("units_sha256") != units_sha256(obj["units"]):
        raise SystemExit("FATAL: x113 prompts.json fails its sha256 check")
    return obj


def units_sha256(units: list[dict[str, Any]]) -> str:
    return sha256_text("\n".join(f"{u['id']}\t{u['stratum']}\t{u['prompt_sha256']}" for u in units))


def texts_sha256(units: list[dict[str, Any]]) -> str:
    """One line per unit: id, nominee, the sha256 of each defined text (the lineage every
    generation and judge row carries as `arms_sha256`)."""
    lines = []
    for u in units:
        nominee = (u.get("nominee") or {}).get("norm") or "-"
        lines.append("\t".join([u["id"], nominee] + [(u["sha256"].get(a) or "-") for a in ARMS]))
    return sha256_text("\n".join(lines))


def load_texts(tag: str) -> dict[str, Any]:
    obj = read_json(paths(tag)["texts"])
    if obj.get("experiment") != "x113" or obj.get("model_tag") != tag or obj.get("status") != "frozen":
        raise SystemExit(f"FATAL: x113 texts for {tag} are missing or not frozen")
    if obj.get("texts_sha256") != texts_sha256(obj["units"]):
        raise SystemExit(f"FATAL: {tag}: texts.json fails its sha256 check")
    for u in obj["units"]:
        for arm in ARMS:
            text = u["texts"].get(arm)
            if text is not None and sha256_text(text) != u["sha256"][arm]:
                raise SystemExit(f"FATAL: {tag}: text {u['id']}/{arm} fails its sha256")
    return obj


def cells(tag: str, texts: dict[str, Any]) -> list[dict[str, Any]]:
    """Every (unit, arm) with a text, in unit order then arm order."""
    out = []
    for u in texts["units"]:
        for arm in ARMS:
            text = u["texts"].get(arm)
            if text is None:
                continue
            out.append(
                {
                    "cell_id": f"{u['id']}|{arm}",
                    "prompt_id": u["id"],
                    "stratum": u["stratum"],
                    "arm": arm,
                    "prompt": text,
                    "prompt_sha256": u["sha256"][arm],
                    "same_as_x106": bool(u["same_as_x106"].get(arm)),
                    "seed": stable_seed(tag, u["id"], arm),
                }
            )
    return out


def cells_sha256(cell_list: list[dict[str, Any]]) -> str:
    return sha256_text("\n".join(f"{c['cell_id']}\t{c['prompt_sha256']}" for c in cell_list))


def x106_refusal_rows(tag: str, arm: str) -> dict[str, dict[str, Any]]:
    """{prompt id: successful x106 refusal2 record} of one x106 arm."""
    import json

    out: dict[str, dict[str, Any]] = {}
    with x106_refusal_path(tag, arm).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("failed"):
                continue
            if len(rec.get("refusal", [])) != N_RESPONSES or not all(isinstance(v, bool) for v in rec["refusal"]):
                raise SystemExit(f"FATAL: invalid x106 refusal record {tag}/{arm}/{rec.get('prompt_id')}")
            out[str(rec["prompt_id"])] = rec
    return out
