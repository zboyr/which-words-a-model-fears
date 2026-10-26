"""x108: on the harmful prompts the models refuse, does the boundary direction nominate a word,
and does the x106 substitution rule have anything to replace it with?

Plan: knowledge/plans/x108-toxic-lexical-carrier-specificity.md. The prompts are the 655
OR-Bench Toxic prompts exactly as each model's x87 / x88 base generation rows record them, plus
the 100 XSTest contrast prompts of the four lexical categories whose safe counterparts x106 ran.
Labels at base are x104's GuardReasoner-8B judgments of those rows. Qwen3.5-4B nominates and the
texts are built once from its nomination; the seven transfer models read the same texts under
their own x88 direction, as in x106. Everything but the population, the paths and the seeds is
x106's, imported: the reader, the write site, the directions, the chat rendering, the deletion
pass, the nomination, the drafter and its rule, the sampling contract, the judges and the
bootstrap.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from x106_common import (  # noqa: F401  (re-exported for the stage scripts)
    ALL_TAGS,
    DATA,
    FRESH_TAGS,
    IMPORTED_TAG,
    N_RESPONSES,
    RESULTS,
    X102_XSTEST,
    X104_LABELS,
    XSTEST_PREFIX,
    atomic_json,
    directions_path,
    file_record,
    is_xstest,
    load_mu,
    read_json,
    sha256_file,
    sha256_text,
    strata_of,
)
from x106_common import x104_labels as x106_x104_labels
from x106_common import base_generations_path as x106_base_generations_path
from x106_common import cohort_path as x106_cohort_path
from x106_common import inputs_path as x106_inputs_path
from x106_common import nominations_path as x106_nominations_path
from x106_common import projections_path as x106_projections_path
from x106_common import results_path as x106_results_path

RUN = DATA / "runs" / "x108"
# Qwen3.5-4B: the model that nominates and whose nomination the texts are built from.
TAG = IMPORTED_TAG
SEED = 108
BOOTSTRAP_SEED = 10_800
TOXIC = "orbench_toxic"
N_TOXIC = 655
CONTRAST_TYPES = ("contrast_homonyms", "contrast_figurative_language", "contrast_safe_contexts", "contrast_safe_targets")
N_CONTRAST = 100
# Texts: the original, the nominated unit substituted, and (hedge units only) the phrase removed.
TEXT_CONDITIONS = ("O", "S", "R")
ARMS = tuple(f"{text}_base" for text in TEXT_CONDITIONS)
READTHROUGH_SEED = 108


# --- paths -----------------------------------------------------------------------------


def x106_inputs() -> Path:
    return x106_inputs_path()


def x106_projections(tag: str = TAG) -> Path:
    return x106_projections_path(tag)


def x106_nominations(tag: str = TAG, text: str = "O") -> Path:
    return x106_nominations_path(tag, text)


def x106_cohort() -> Path:
    return x106_cohort_path()


def x106_results() -> Path:
    return x106_results_path()


def tag_dir(tag: str) -> Path:
    if tag not in ALL_TAGS:
        raise ValueError(f"unknown x108 tag {tag!r}")
    return RUN / tag


def projections_path(tag: str) -> Path:
    return tag_dir(tag) / "x108_stage0_projections.json"


def states_path(tag: str) -> Path:
    return tag_dir(tag) / "x108_stage0_states.npz"


def nominations_shard_path(tag: str, text: str, shard: int, n_shards: int) -> Path:
    if text not in TEXT_CONDITIONS:
        raise ValueError(f"unknown text condition {text!r}")
    return tag_dir(tag) / f"x108_stage0_nominations_{text}_shard{shard}of{n_shards}.json"


def nominations_path(tag: str, text: str = "O") -> Path:
    if text not in TEXT_CONDITIONS:
        raise ValueError(f"unknown text condition {text!r}")
    return tag_dir(tag) / ("x108_stage0_nominations.json" if text == "O" else f"x108_stage0_nominations_{text}.json")


def cohort_path() -> Path:
    return RUN / "x108_cohort.json"


def drafts_path() -> Path:
    return RUN / "x108_substitute_drafts.jsonl"


def readthrough_path() -> Path:
    return RUN / "x108_readthrough.md"


def generations_path(tag: str, arm: str) -> Path:
    arm_spec(arm)
    return tag_dir(tag) / f"x108_generations_{arm}.json"


def refusal_path(tag: str, arm: str) -> Path:
    arm_spec(arm)
    return tag_dir(tag) / f"x108_refusal_{arm}.jsonl"


def guardreasoner_path(tag: str, arm: str, shard: int | None = None, n_shards: int | None = None) -> Path:
    arm_spec(arm)
    if shard is None:
        return tag_dir(tag) / f"x108_guardreasoner_{arm}.jsonl"
    return tag_dir(tag) / f"x108_guardreasoner_{arm}_shard{shard}of{n_shards}.jsonl"


def results_path() -> Path:
    return RESULTS / "x108_toxic_lexical_carrier.json"


# --- arms and seeds --------------------------------------------------------------------


def arms() -> list[dict[str, Any]]:
    return [{"arm": f"{text}_base", "text": text, "dose": 0.0} for text in TEXT_CONDITIONS]


def arm_spec(arm: str) -> dict[str, Any]:
    for spec in arms():
        if spec["arm"] == arm:
            return dict(spec)
    raise ValueError(f"unknown x108 arm {arm!r}")


def stable_seed(prompt_id: str, text: str, retry: bool = False, tag: str = TAG) -> int:
    """Per-(model, prompt, text) sampling seed in the x108 namespace (each text is its own
    sample, as in x106's base arms)."""
    label = f"{text}:retry4096" if retry else text
    payload = f"x108|{SEED}|{tag}|{prompt_id}|{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


# --- the prompts: Toxic rows exactly as x87 / x88 generated them, plus the XSTest contrast prompts ---


def base_generations_path(tag: str) -> Path:
    return x106_base_generations_path(tag)


def toxic_rows(tag: str) -> list[dict[str, Any]]:
    """The 655 Toxic prompt rows (id, prompt, sha256s, category) of one model's base generation."""
    path = base_generations_path(tag)
    obj = read_json(path)
    expected_experiment = "x87" if tag == IMPORTED_TAG else "x88"
    if obj.get("experiment") != expected_experiment or obj.get("condition") != "base":
        raise ValueError(f"{path} is not the {expected_experiment} base generation file")
    if tag != IMPORTED_TAG and obj.get("model_tag") != tag:
        raise ValueError(f"{path} is not {tag}'s generation file")
    rows = []
    for row in obj["rows"]:
        if row.get("benchmark") != TOXIC:
            continue
        if sha256_text(row["prompt"]) != row["prompt_sha256"]:
            raise ValueError(f"{path}: prompt sha256 mismatch at {row['id']}")
        rows.append(
            {
                "id": str(row["id"]),
                "source": TOXIC,
                "source_index": int(row["source_index"]),
                "category": row.get("category"),
                "prompt": row["prompt"],
                "prompt_sha256": row["prompt_sha256"],
                "rendered_chat_sha256": row["rendered_chat_sha256"],
            }
        )
    if len(rows) != N_TOXIC:
        raise ValueError(f"{path}: {len(rows)} Toxic rows, expected {N_TOXIC}")
    return rows


def contrast_prompts(path: Path = X102_XSTEST) -> list[dict[str, Any]]:
    """The 100 XSTest contrast prompts of the four lexical categories, one row per prompt id."""
    obj = read_json(path)
    if obj.get("experiment") != "x102":
        raise ValueError(f"{path} is not the x102 XSTest copy")
    seen: dict[str, dict[str, Any]] = {}
    for row in obj["rows"]:
        if row.get("type") not in CONTRAST_TYPES:
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
    rows = sorted(seen.values(), key=lambda r: (CONTRAST_TYPES.index(r["category"]), int(r["source_index"].split("-")[-1])))
    if len(rows) != N_CONTRAST:
        raise ValueError(f"{len(rows)} XSTest contrast prompts, expected {N_CONTRAST}")
    return rows


def all_prompts(tag: str) -> list[dict[str, Any]]:
    return toxic_rows(tag) + contrast_prompts()


def is_toxic(prompt_id: str) -> bool:
    return str(prompt_id).startswith(TOXIC + ":")


# --- x104 labels (GuardReasoner-8B on the greedy x87 / x88 Toxic base responses) -----------


def x104_toxic_labels(tag: str, path: Path = X104_LABELS) -> dict[str, dict[str, Any]]:
    """{prompt id: label record} of one model's parsed Toxic base judgments."""
    if tag not in ALL_TAGS:
        raise ValueError(f"unknown x108 tag {tag!r}")
    out: dict[str, dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("tag") != tag or record.get("benchmark") != TOXIC or record.get("condition") != "base" or not record.get("parse_ok"):
                continue
            out[str(record["id"])] = record
    if not out:
        raise ValueError(f"no x104 Toxic base labels for {tag}")
    return out


def population_ids(labels: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Toxic prompts refused at base whose request GuardReasoner reads as harmful."""
    return sorted(
        prompt_id
        for prompt_id, record in labels.items()
        if int(record["refusal"]) == 1 and record["fields"].get("request") == "harmful"
    )


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
    if cohort.get("experiment") != "x108" or cohort.get("status") != "frozen":
        raise SystemExit("FATAL: x108 cohort is missing or not frozen")
    units = list(cohort["units"])
    if cohort.get("cohort_sha256") != cohort_sha256(units):
        raise SystemExit("FATAL: x108 cohort fails its sha256 check")
    for unit in units:
        for text, value in unit["prompts"].items():
            if value is not None and sha256_text(value) != unit["prompt_sha256"][text]:
                raise SystemExit(f"FATAL: x108 cohort text {unit['id']}/{text} fails its sha256")
    return cohort


def units_with_text(cohort: Mapping[str, Any], text: str) -> list[dict[str, Any]]:
    if text not in TEXT_CONDITIONS:
        raise ValueError(f"unknown text condition {text!r}")
    return [unit for unit in cohort["units"] if unit["prompts"].get(text) is not None]


def no_substitute_reason(unit: Mapping[str, Any]) -> str | None:
    """Why a unit has no S text: the drafter's provider refused, the drafter found no
    meaning-preserving harm-free substitute, or every draft failed the structural checks."""
    if unit["prompts"].get("S") is not None:
        return None
    if unit.get("drafter_refused"):
        return "drafter_refused"
    if unit.get("no_substitute_note") is not None:
        return "no_substitute"
    return "failed_checks"


# --- the models, their sites and their directions: x106's frozen entries -------------------


def model_entry(tag: str, inputs_file: Path | None = None) -> dict[str, Any]:
    """One model's record of x106's frozen inputs: snapshot, hook block, boundary ids, direction."""
    if tag not in ALL_TAGS:
        raise ValueError(f"unknown x108 tag {tag!r}")
    inputs = read_json(inputs_file or x106_inputs())
    if inputs.get("experiment") != "x106" or inputs.get("status") != "frozen":
        raise SystemExit("FATAL: the x106 inputs are missing or not frozen")
    return dict(inputs["models"][tag])


def load_model(tag: str, entry: Mapping[str, Any]):
    """Tokenizer, model, hook site, consensus direction [2, hidden] and the site record, with the
    direction artifact checked against the frozen inputs."""
    from x106_common import hook_site, load_tag_model, load_tag_tokenizer, n_blocks_of

    dir_path = directions_path(tag)
    if sha256_file(dir_path) != entry["directions"]["sha256"]:
        raise SystemExit(f"FATAL: {tag} direction artifact differs from the frozen x106 inputs")
    mu = load_mu(tag, dir_path)
    tokenizer = load_tag_tokenizer(entry["snapshot"], tag)
    model = load_tag_model(entry["snapshot"], tag)
    model.eval()
    block = int(entry["hook_block"])
    n_blocks = n_blocks_of(model)
    hidden_dim = int(model.config.get_text_config().hidden_size)
    if hidden_dim != mu.shape[1]:
        raise SystemExit(f"FATAL: direction dim {mu.shape[1]} != hidden {hidden_dim}")
    site = hook_site(model, block)
    record = {"hook_block": block, "state_index": block + 1, "n_blocks": n_blocks, "hidden_dim": hidden_dim, "boundary_ids": [int(v) for v in entry["boundary_ids"]]}
    return tokenizer, model, site, mu, record


# --- x106's Hard-1K reference per model ------------------------------------------------


def hard1k_reference(tag: str) -> dict[str, Any]:
    """x106's Stage 0 projection of one model over the Hard-1K prompts: the answered-prompt mean
    the nomination shares are taken against, the refused-prompt mean, the sd, and the counts."""
    import numpy as np

    projections = read_json(x106_projections(tag))
    nominations = read_json(x106_nominations(tag, "O"))
    if projections.get("model_tag") != tag or nominations.get("model_tag") != tag or nominations.get("cohort_only"):
        raise SystemExit(f"FATAL: x106's {tag} Stage 0 artifacts are missing")
    base = x106_x104_labels(tag)["base"]
    proj = {r["id"]: float(r["proj"]) for r in projections["rows"] if not is_xstest(r["id"])}
    refused = np.asarray([proj[i] for i in proj if i in base and int(base[i]["refusal"]) == 1])
    answered = np.asarray([proj[i] for i in proj if i in base and int(base[i]["refusal"]) == 0])
    answered_mean = float(nominations["answered_mean_proj"])
    if abs(answered_mean - float(answered.mean())) > 1e-6:
        raise SystemExit("FATAL: x106's answered-prompt mean does not reproduce from its projections")
    everything = np.asarray(list(proj.values()))
    return {
        "answered_mean": answered_mean,
        "refused_mean": float(refused.mean()),
        "sd": float(everything.std()),
        "n_hard1k": int(len(everything)),
        "n_refused": int(len(refused)),
        "n_answered": int(len(answered)),
        "refused": [float(v) for v in refused],
        "answered": [float(v) for v in answered],
        "projections": file_record(x106_projections(tag)),
        "nominations": file_record(x106_nominations(tag, "O")),
    }
