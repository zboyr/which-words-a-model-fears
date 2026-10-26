"""x106: in naturally occurring benign prompts, does the boundary direction nominate the word
that carries the over-refusal, and does replacing that word in place remove it?

Plan: knowledge/plans/x106-natural-benign-lexical-carrier.md. The prompts are the 1,319
OR-Bench Hard-1K prompts exactly as the x87 (Qwen3.5-4B) and x88 (seven transfer models) base
generation rows record them, plus the 100 safe XSTest prompts of the four lexical categories
from the x102 copy. Labels at base are x104's GuardReasoner-8B judgments of those rows. The
read site, the write site, the direction per model, the chat rendering and the sampling
contract are x103 / x105's, imported, not restated.

Stage 0 reads every prompt's boundary state and projects it on the model's consensus
direction, then nominates one unit per prompt by leave-one-word-out and leave-one-pair-out
deletion. The texts (O / S / C / R) are built once from the Qwen3.5-4B nomination and shared
by all eight models; generation runs the four texts at dose 0 and the boundary write on O
and S; the judges are the x64 `refusal2` contract and GuardReasoner-8B under x104's contract.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from x105_common import (  # noqa: F401  (re-exported for the stage scripts)
    ALL_TAGS,
    BOUNDARY_POSITIONS,
    DATA,
    FRESH_TAGS,
    IMPORTED_TAG,
    MODELS,
    N_RESPONSES,
    QWEN4B_HOOK_BLOCK,
    QWEN4B_MODEL_ID,
    RESULTS,
    X83_RUN,
    X88_RUN,
    BoundaryReader,
    ExternalBoundaryHook,
    assert_boundary,
    atomic_json,
    audit_write_site,
    bf16_ulp_bound,
    consensus_x86_vectors,
    directions_path,
    file_record,
    format_user_chat,
    holm,
    hook_block_for,
    hook_site,
    load_tag_model,
    load_tag_tokenizer,
    n_blocks_of,
    read_json,
    sha256_file,
    sha256_text,
    unit_directions,
)
from x87_common import X86_EXPECTED_BOUNDARY_IDS

RUN = DATA / "runs" / "x106"
X87_RUN = DATA / "runs" / "x87"
X102_RUN = DATA / "runs" / "x102"
X104_RUN = DATA / "runs" / "x104"
X104_LABELS = X104_RUN / "x104_labels.jsonl"
X102_XSTEST = X102_RUN / "x102_xstest.json"
X88_MANIFEST = X88_RUN / "x88_manifest.json"

SEED = 106
HARD1K = "orbench_hard"
N_HARD1K = 1_319
XSTEST_TYPES = ("homonyms", "figurative_language", "safe_contexts", "safe_targets")
N_XSTEST = 100
XSTEST_PREFIX = "xstest"

# Texts: the original, the nominated unit substituted, a matched inert content word
# substituted, and (hedge prompts only) the nominated phrase removed.
TEXT_CONDITIONS = ("O", "S", "C", "R")
WRITTEN_TEXTS = ("O", "S")
# Negative doses in multiples of the model's own mean boundary shift (the x86/x88 unit).
DOSES: dict[str, tuple[float, ...]] = {IMPORTED_TAG: (-0.5, -1.5)}
TRANSFER_DOSES: tuple[float, ...] = (-1.5,)
# The written x104 conditions whose labels Stage 0 reads for the flip and induction AUCs.
X104_WRITTEN_CONDITIONS = {
    IMPORTED_TAG: ("x86_full", "x86_m075", "x86_m100", "x86_m125", "x86_m150", "x86_m200", "x86_m300"),
    "transfer": ("x86_m150",),
}
X104_INDUCTION_CONDITIONS = {IMPORTED_TAG: ("x86_p050", "x86_p100", "x86_p150"), "transfer": ()}

# Projection strata (sum over the two boundary tokens of <h, mu/|mu|>) on Qwen3.5-4B.
STRATA_EDGES = (15.0, 25.0)
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 10_600
ALPHA = 0.05

# Llama's template stamps the run date into its system header; the x88 base rows were rendered
# on x88's run date, recovered per model from the first row's recorded sha256 over this window.
DATE_FORMAT = "%d %b %Y"
DATE_WINDOW = (_dt.date(2026, 6, 1), _dt.date(2026, 12, 31))

# Standard English stop-word list (NLTK's English list plus the request verbs and generic
# nouns every prompt shares); words on it are never nominated.
STOP_WORDS = frozenset(
    """a about above after again against ain all am an and any are aren as at be because been
before being below between both but by can could couldn d did didn do does doesn doing don
down during each few for from further had hadn has hasn have haven having he her here hers
herself him himself his how i if in into is isn it its itself just ll m ma me mightn more most
mustn my myself needn no nor not now o of off on once only or other our ours ourselves out over
own re s same shan she should shouldn so some such t than that the their theirs them themselves
then there these they this those through to too under until up ve very was wasn we were weren
what when where which while who whom why will with won would wouldn y you your yours yourself
yourselves might must shall may ought need dare also please tell give provide explain describe
write list help want like make get go know think see way ways thing things someone something
anything everything one ones use using used""".split()
)


# --- paths ---------------------------------------------------------------------------


def inputs_path() -> Path:
    return RUN / "x106_inputs.json"


def tag_dir(tag: str) -> Path:
    if tag not in ALL_TAGS:
        raise ValueError(f"unknown x106 tag {tag!r}")
    return RUN / tag


def projections_path(tag: str) -> Path:
    return tag_dir(tag) / "x106_stage0_projections.json"


def states_path(tag: str) -> Path:
    return tag_dir(tag) / "x106_stage0_states.npz"


def nominations_shard_path(tag: str, text: str, shard: int, n_shards: int) -> Path:
    if text not in TEXT_CONDITIONS:
        raise ValueError(f"unknown text condition {text!r}")
    return tag_dir(tag) / f"x106_stage0_nominations_{text}_shard{shard}of{n_shards}.json"


def nominations_path(tag: str, text: str = "O") -> Path:
    if text not in TEXT_CONDITIONS:
        raise ValueError(f"unknown text condition {text!r}")
    return tag_dir(tag) / ("x106_stage0_nominations.json" if text == "O" else f"x106_stage0_nominations_{text}.json")


def cohort_path() -> Path:
    return RUN / "x106_cohort.json"


def drafts_path() -> Path:
    return RUN / "x106_substitute_drafts.jsonl"


def text_projections_path(tag: str) -> Path:
    return tag_dir(tag) / "x106_text_projections.json"


def generations_path(tag: str, arm: str) -> Path:
    arm_spec(tag, arm)
    return tag_dir(tag) / f"x106_generations_{arm}.json"


def refusal_path(tag: str, arm: str) -> Path:
    arm_spec(tag, arm)
    return tag_dir(tag) / f"x106_refusal_{arm}.jsonl"


def guardreasoner_path(tag: str, arm: str, shard: int | None = None, n_shards: int | None = None) -> Path:
    arm_spec(tag, arm)
    if shard is None:
        return tag_dir(tag) / f"x106_guardreasoner_{arm}.jsonl"
    return tag_dir(tag) / f"x106_guardreasoner_{arm}_shard{shard}of{n_shards}.jsonl"


def results_path() -> Path:
    return RESULTS / "x106_natural_lexical_carrier.json"


# --- arms ----------------------------------------------------------------------------


def doses_for(tag: str) -> tuple[float, ...]:
    if tag not in ALL_TAGS:
        raise ValueError(f"unknown x106 tag {tag!r}")
    return DOSES.get(tag, TRANSFER_DOSES)


def dose_label(dose: float) -> str:
    if dose == 0.0:
        return "base"
    return f"{'m' if dose < 0 else 'p'}{int(round(abs(dose) * 100)):03d}"


def arms_for(tag: str) -> list[dict[str, Any]]:
    """Every (text, dose) arm of one model, base arms first, then the written arms by dose."""
    arms = [{"arm": f"{text}_base", "text": text, "dose": 0.0} for text in TEXT_CONDITIONS]
    for dose in doses_for(tag):
        for text in WRITTEN_TEXTS:
            arms.append({"arm": f"{text}_{dose_label(dose)}", "text": text, "dose": float(dose)})
    return arms


def arm_spec(tag: str, arm: str) -> dict[str, Any]:
    for spec in arms_for(tag):
        if spec["arm"] == arm:
            return dict(spec)
    raise ValueError(f"unknown x106 arm {arm!r} for {tag}")


def arm_names(tag: str) -> tuple[str, ...]:
    return tuple(spec["arm"] for spec in arms_for(tag))


def all_arm_names() -> tuple[str, ...]:
    names: list[str] = []
    for tag in ALL_TAGS:
        for name in arm_names(tag):
            if name not in names:
                names.append(name)
    return tuple(names)


def stable_seed(tag: str, prompt_id: str, text: str, retry: bool = False) -> int:
    """Per-(tag, prompt, text) sampling seed; every dose of one text shares it (common random
    numbers across doses)."""
    label = f"{text}:retry4096" if retry else text
    payload = f"x106|{SEED}|{tag}|{prompt_id}|{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


# --- the prompts: Hard-1K rows exactly as x87 / x88 generated them, plus XSTest -----------


def base_generations_path(tag: str) -> Path:
    if tag == IMPORTED_TAG:
        return X87_RUN / "x87_generations_base.json"
    return X88_RUN / tag / "x88_generations_base.json"


def hard1k_rows(tag: str) -> list[dict[str, Any]]:
    """The Hard-1K prompt rows (id, prompt, sha256s, category) of one model's base generation."""
    path = base_generations_path(tag)
    obj = read_json(path)
    expected_experiment = "x87" if tag == IMPORTED_TAG else "x88"
    if obj.get("experiment") != expected_experiment or obj.get("condition") != "base":
        raise ValueError(f"{path} is not the {expected_experiment} base generation file")
    if tag != IMPORTED_TAG and obj.get("model_tag") != tag:
        raise ValueError(f"{path} is not {tag}'s generation file")
    rows = []
    for row in obj["rows"]:
        if row.get("benchmark") != HARD1K:
            continue
        if sha256_text(row["prompt"]) != row["prompt_sha256"]:
            raise ValueError(f"{path}: prompt sha256 mismatch at {row['id']}")
        rows.append(
            {
                "id": str(row["id"]),
                "source": HARD1K,
                "source_index": int(row["source_index"]),
                "category": row.get("category"),
                "prompt": row["prompt"],
                "prompt_sha256": row["prompt_sha256"],
                "rendered_chat_sha256": row["rendered_chat_sha256"],
            }
        )
    if len(rows) != N_HARD1K:
        raise ValueError(f"{path}: {len(rows)} Hard-1K rows, expected {N_HARD1K}")
    return rows


def xstest_prompts(path: Path = X102_XSTEST) -> list[dict[str, Any]]:
    """The 100 safe XSTest prompts of the four lexical categories, one row per prompt id."""
    obj = read_json(path)
    if obj.get("experiment") != "x102":
        raise ValueError(f"{path} is not the x102 XSTest copy")
    seen: dict[str, dict[str, Any]] = {}
    for row in obj["rows"]:
        if row.get("type") not in XSTEST_TYPES:
            continue
        xstest_id = str(row["id"])
        prompt = str(row["prompt"])
        if xstest_id in seen:
            if seen[xstest_id]["prompt"] != prompt:
                raise ValueError(f"XSTest {xstest_id} carries two prompt texts")
            continue
        seen[xstest_id] = {
            "id": f"{XSTEST_PREFIX}:{xstest_id}",
            "source": XSTEST_PREFIX,
            "source_index": xstest_id,
            "category": str(row["type"]),
            "prompt": prompt,
            "prompt_sha256": sha256_text(prompt),
            "rendered_chat_sha256": None,
        }
    rows = sorted(seen.values(), key=lambda r: (XSTEST_TYPES.index(r["category"]), int(r["source_index"].split("-")[-1])))
    if len(rows) != N_XSTEST:
        raise ValueError(f"{len(rows)} XSTest prompts, expected {N_XSTEST}")
    return rows


def all_prompts(tag: str) -> list[dict[str, Any]]:
    return hard1k_rows(tag) + xstest_prompts()


def is_xstest(prompt_id: str) -> bool:
    return str(prompt_id).startswith(XSTEST_PREFIX + ":")


# --- x104 labels (GuardReasoner-8B on the greedy base / written responses) ------------------


def x104_labels(tag: str, path: Path = X104_LABELS) -> dict[str, dict[str, dict[str, Any]]]:
    """{condition: {prompt id: label record}} of one model's parsed Hard-1K judgments."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = __import__("json").loads(line)
            if record.get("tag") != tag or record.get("benchmark") != HARD1K or not record.get("parse_ok"):
                continue
            out.setdefault(record["condition"], {})[str(record["id"])] = record
    if "base" not in out:
        raise ValueError(f"no x104 base labels for {tag}")
    return out


def written_conditions(tag: str) -> tuple[str, ...]:
    return X104_WRITTEN_CONDITIONS[IMPORTED_TAG if tag == IMPORTED_TAG else "transfer"]


def induction_conditions(tag: str) -> tuple[str, ...]:
    return X104_INDUCTION_CONDITIONS[IMPORTED_TAG if tag == IMPORTED_TAG else "transfer"]


def population_ids(labels: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> list[str]:
    """Hard-1K prompts refused at base whose request GuardReasoner reads as unharmful."""
    base = labels["base"]
    return sorted(
        prompt_id
        for prompt_id, record in base.items()
        if int(record["refusal"]) == 1 and record["fields"].get("request") == "unharmful"
    )


# --- rendering ------------------------------------------------------------------------


def render_chat(tokenizer, prompt: str, date: str | None) -> str:
    """x87's renderer; with a date, the same call with that date stamped (Llama)."""
    if date is None:
        return format_user_chat(tokenizer, prompt)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
        date_string=date,
    )


def date_strings(window: tuple[_dt.date, _dt.date] = DATE_WINDOW) -> list[str]:
    start, stop = window
    return [(start + _dt.timedelta(days=k)).strftime(DATE_FORMAT) for k in range((stop - start).days + 1)]


def recover_date(tokenizer, prompt: str, expected_sha: str) -> str | None:
    """None when the plain renderer reproduces the recorded sha256, else the stamped date that does."""
    if sha256_text(render_chat(tokenizer, prompt, None)) == expected_sha:
        return None
    for date in date_strings():
        if sha256_text(render_chat(tokenizer, prompt, date)) == expected_sha:
            return date
    raise RuntimeError("no renderer date reproduces the recorded rendered-chat sha256")


# --- words, deletion variants, nomination ----------------------------------------------

WORD_RE = re.compile(r"\S+")
TRIM_RE = re.compile(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$")


def words_with_spans(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in WORD_RE.finditer(text)]


def normalize_word(word: str) -> str:
    """Lower-cased, punctuation stripped from both ends (kept in the text itself)."""
    return TRIM_RE.sub("", word.lower())


def is_content_word(word: str) -> bool:
    norm = normalize_word(word)
    return bool(norm) and norm not in STOP_WORDS


def delete_span(text: str, start: int, end: int) -> str:
    """The text without [start, end) and one adjacent run of whitespace, other whitespace kept."""
    before, after = text[:start], text[end:]
    if before and before[-1].isspace():
        before = before.rstrip()
    elif after and after[0].isspace():
        after = after.lstrip()
    return before + after


def deletion_variants(text: str) -> dict[str, Any]:
    """Every leave-one-word-out and leave-one-adjacent-pair-out variant of a prompt."""
    spans = words_with_spans(text)
    words = [w for w, _, _ in spans]
    unigrams = [delete_span(text, s, e) for _, s, e in spans]
    bigrams = [delete_span(text, spans[k][1], spans[k + 1][2]) for k in range(len(spans) - 1)]
    return {"words": words, "spans": [(s, e) for _, s, e in spans], "unigram_texts": unigrams, "bigram_texts": bigrams}


def nominate(
    words: Sequence[str],
    delta_unigram: Sequence[float],
    delta_bigram: Sequence[float],
    proj_full: float,
    answered_mean: float,
) -> dict[str, Any]:
    """One nominated unit per prompt from the deletion deltas.

    Ranks content words (stop words excluded) by delta = proj(full) - proj(without the word);
    records the top word's within-prompt z against every other word, its share of the
    prompt's projection above the answered-prompt mean, and whether the top adjacent pair
    (pairs with at least one content word) beats the top word by more than one within-prompt
    sd of the unigram deltas. The unit is the pair when it does, else the word.
    """
    d = np.asarray(delta_unigram, dtype=np.float64)
    b = np.asarray(delta_bigram, dtype=np.float64)
    if len(d) != len(words) or len(b) != max(len(words) - 1, 0):
        raise ValueError("deltas do not match the word list")
    content = np.asarray([is_content_word(w) for w in words], dtype=bool)
    if not content.any():
        content = np.ones(len(words), dtype=bool)
    ranked = np.where(content, d, -np.inf)
    top = int(np.argmax(ranked))
    others = np.delete(d, top)
    z = float((d[top] - others.mean()) / others.std()) if len(others) > 1 and others.std() > 0 else float("nan")
    excess = proj_full - answered_mean
    share = float(d[top] / excess) if excess > 1e-6 else None
    sd = float(d.std()) if len(d) > 1 else float("nan")
    pair_ok = np.asarray([content[k] or content[k + 1] for k in range(len(b))], dtype=bool)
    top_pair = int(np.argmax(np.where(pair_ok, b, -np.inf))) if pair_ok.any() else None
    pair_wins = bool(top_pair is not None and np.isfinite(sd) and b[top_pair] - d[top] > sd)
    unit_indices = [top_pair, top_pair + 1] if pair_wins else [top]
    unit_text = " ".join(words[k] for k in unit_indices)
    inert = [
        {"index": int(k), "word": words[k], "z": float((d[k] - np.delete(d, k).mean()) / np.delete(d, k).std()) if len(d) > 2 and np.delete(d, k).std() > 0 else float("nan")}
        for k in range(len(words))
        if content[k] and k not in unit_indices
    ]
    return {
        "top_index": top,
        "top_word": words[top],
        "top_norm": normalize_word(words[top]),
        "top_delta": float(d[top]),
        "top_z": z,
        "top_share_of_excess": share,
        "unigram_sd": sd,
        "top_pair_index": top_pair,
        "top_pair": " ".join(words[top_pair : top_pair + 2]) if top_pair is not None else None,
        "top_pair_delta": float(b[top_pair]) if top_pair is not None else None,
        "pair_wins": pair_wins,
        "unit_indices": unit_indices,
        "unit": unit_text,
        "unit_norm": " ".join(normalize_word(words[k]) for k in unit_indices),
        "n_words": len(words),
        "n_content_words": int(content.sum()),
        "content_word_z": inert,
    }


def lexicon_words() -> set[str]:
    from x83_common import LEXICON

    return {w for ws in LEXICON.values() for w in ws}


def strata_of(projection: Iterable[float], edges: tuple[float, float] = STRATA_EDGES) -> list[str]:
    low, high = edges
    out = []
    for value in projection:
        out.append("low" if value < low else ("mid" if value <= high else "high"))
    return out


# --- batched boundary reading (right padding; the last two real tokens per row) -----------


class BatchBoundaryReader:
    """Forward hook on the x88 write site that gathers each row's final two real-token outputs
    from a right-padded batch. Set `lengths` (real tokens per row) before each forward."""

    def __init__(self) -> None:
        self.lengths: Sequence[int] | None = None
        self.last = None
        self.calls = 0

    def __call__(self, _module, _inputs, output):
        import torch

        hidden = output[0] if isinstance(output, tuple) else output
        if self.lengths is None or len(self.lengths) != hidden.shape[0]:
            raise RuntimeError("x106 batch reader needs one length per row")
        self.calls += 1
        index = torch.as_tensor([[l - 2, l - 1] for l in self.lengths], device=hidden.device)
        if int(index.min()) < 0:
            raise RuntimeError("x106 reads a prefill of at least two tokens")
        gathered = hidden[torch.arange(hidden.shape[0], device=hidden.device)[:, None], index, :]
        self.last = gathered.detach().float().cpu()
        return output


def projection_of(states: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """[n, 2, hidden] boundary states -> [n] sum over the two tokens of <h, mu/|mu|>."""
    units, _ = unit_directions(mu)
    return np.einsum("ntd,td->n", np.asarray(states, dtype=np.float64), units)


def load_mu(tag: str, path: str | Path | None = None) -> np.ndarray:
    minus1, tbg = consensus_x86_vectors(path or directions_path(tag))
    return np.stack([minus1.numpy(), tbg.numpy()]).astype(np.float64)


# --- cohort ----------------------------------------------------------------------------


def cohort_sha256(units: Sequence[Mapping[str, Any]]) -> str:
    """One line per unit: id, nominated unit, the sha256 of each defined text."""
    lines = []
    for unit in units:
        parts = [str(unit["id"]), str(unit["unit"])]
        for text in TEXT_CONDITIONS:
            parts.append(unit["prompt_sha256"].get(text) or "-")
        lines.append("\t".join(parts))
    return sha256_text("\n".join(lines))


def load_cohort(path: Path | None = None) -> dict[str, Any]:
    cohort = read_json(path or cohort_path())
    if cohort.get("experiment") != "x106" or cohort.get("status") != "frozen":
        raise SystemExit("FATAL: x106 cohort is missing or not frozen")
    units = list(cohort["units"])
    if cohort.get("cohort_sha256") != cohort_sha256(units):
        raise SystemExit("FATAL: x106 cohort fails its sha256 check")
    for unit in units:
        for text, value in unit["prompts"].items():
            if value is not None and sha256_text(value) != unit["prompt_sha256"][text]:
                raise SystemExit(f"FATAL: x106 cohort text {unit['id']}/{text} fails its sha256")
    return cohort


def units_with_text(cohort: Mapping[str, Any], text: str) -> list[dict[str, Any]]:
    if text not in TEXT_CONDITIONS:
        raise ValueError(f"unknown text condition {text!r}")
    return [unit for unit in cohort["units"] if unit["prompts"].get(text) is not None]
