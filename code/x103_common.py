"""x103: does each model's own boundary direction predict its insertion-induced refusal rise?

Plan: knowledge/plans/x103-cross-model-direction-insertion-projection.md (exploratory,
not pre-registered). Forward passes only, on the x101 texts rendered exactly as x101
rendered them for generation (verified against the rendered-chat sha256 every x101 /
x83 generation row recorded). The hidden state is read by a forward hook on the very
decoder block the x88 boundary write hooks (`x88_common.setup_condition`, condition
`x86_m150`), so the tensor read is the tensor that write edits, at the same two
trailing prefill positions. Directions, model registry, chat rendering and loaders
are imported from x88 / x87 / x101; the behaviour is x101's `refusal2` (x83's rows
for Qwen3.5-4B), read with the x83 loaders.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from scipy import stats

from x86_common import HOOK_BLOCK as QWEN4B_HOOK_BLOCK  # noqa: F401  (x86/x87 site on Qwen3.5-4B: block 29)
from x87_common import X86_EXPECTED_BOUNDARY_IDS
from x88_common import (  # noqa: F401  (re-exported for the stage scripts)
    ExternalBoundaryHook,
    _decoder_layers,
    assert_boundary,
    consensus_x86_vectors,
    hook_block_for,
    n_blocks_of,
)
from x101_common import (  # noqa: F401  (re-exported for the stage scripts)
    ALL_TAGS,
    CONDITIONS,
    DATA,
    FRESH_TAGS,
    IMPORTED_TAG,
    MODELS,
    N_RESPONSES,
    POSITIONS,
    QWEN4B_MODEL_ID,
    RESULTS,
    X83_RUN,
    X88_RUN,
    atomic_json,
    cohort_path,
    file_record,
    format_user_chat,
    generations_path,
    load_cohort,
    manifest_path,
    read_json,
    sha256_file,
    sha256_text,
)

RUN = DATA / "runs" / "x103"
X85_COHORT = DATA / "runs" / "x85" / "x85_cohort.json"
QWEN4B_DIRECTIONS = DATA / "runs" / "x86" / "x86_directions.npz"

BOUNDARY_POSITIONS = ("tbg_minus1", "tbg")  # the two trailing prefill tokens, in order
SEED = 103
N_RANDOM_DIRECTIONS = 1_000
N_PERMUTATIONS = 10_000
N_BOOTSTRAP = 10_000
ALPHA = 0.05
N_AUDIT_UNITS = 2  # cohort units used for the in-job site-match audit

# Llama's template stamps the run date into the system header; x101 rendered on its
# run dates, so the date is recovered per row from the recorded rendered-chat sha256.
LLAMA_DATE_FORMAT = "%d %b %Y"
DATE_LINE = re.compile(r"Today Date: (\d{2} [A-Z][a-z]{2} \d{4})")
DATE_WINDOW_DAYS = 3


# --- paths ------------------------------------------------------------------------


def inputs_path() -> Path:
    return RUN / "x103_inputs.json"


def tag_dir(tag: str) -> Path:
    if tag not in ALL_TAGS:
        raise ValueError(f"unknown x103 tag {tag!r}")
    return RUN / tag


def projections_path(tag: str) -> Path:
    return tag_dir(tag) / "x103_projections.json"


def deltas_path(tag: str) -> Path:
    return tag_dir(tag) / "x103_boundary_deltas.npz"


def directions_path(tag: str) -> Path:
    if tag == IMPORTED_TAG:
        return QWEN4B_DIRECTIONS
    return X88_RUN / tag / "x88_directions.npz"


def stable_seed(*parts: object) -> int:
    payload = "x103|" + "|".join(str(value) for value in (SEED, *parts))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "big") & 0x7FFFFFFF


# --- model loading (x101's loaders; Qwen3.5-4B as x85/x87 loaded it) ----------------


def load_tag_tokenizer(source: str, tag: str):
    if tag == IMPORTED_TAG:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    from x101_common import load_tokenizer

    return load_tokenizer(source, tag)


def load_tag_model(source: str, tag: str) -> torch.nn.Module:
    if tag == IMPORTED_TAG:
        from lib.edit_common import load_causal_lm

        os.environ["X63_TRUST_REMOTE_CODE"] = "1"
        return load_causal_lm(source)
    from x101_common import load_model

    return load_model(source, tag)


# --- the read site --------------------------------------------------------------------


class BoundaryReader:
    """Forward hook that reads what `ExternalBoundaryHook` writes: the block output at
    the final two prefill positions. Registered on the same module, it sees the same
    tensor (after any hook registered before it has returned its edit)."""

    def __init__(self) -> None:
        self.last: torch.Tensor | None = None
        self.calls = 0

    def __call__(self, _module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[1] < 2:
            raise RuntimeError("x103 reads a prefill of at least two tokens")
        self.calls += 1
        self.last = hidden[:, -2:, :].detach().float().cpu()
        return output


def hook_site(model: torch.nn.Module, block: int) -> torch.nn.Module:
    """The decoder block whose forward output the x88 boundary write edits."""
    return _decoder_layers(model)[int(block)]


def bf16_ulp_bound(reference: torch.Tensor, added: torch.Tensor) -> torch.Tensor:
    """Per-element rounding bound for `bf16(h) + bf16(added)` landing at `reference`: one bf16
    ulp of the result plus one ulp of the addend (the x88 hook casts mu to bf16 before adding)."""
    return (reference.abs() + added.abs()) * 2.0 ** -7 + 1e-6


# --- rendering exactly as x101 did ----------------------------------------------------


def date_candidates(scaffold: str, window_days: int = DATE_WINDOW_DAYS) -> list[str]:
    """Llama date strings around the one x101's manifest scaffold recorded."""
    match = DATE_LINE.search(scaffold)
    if match is None:
        return []
    center = _dt.datetime.strptime(match.group(1), LLAMA_DATE_FORMAT).date()
    return [
        (center + _dt.timedelta(days=offset)).strftime(LLAMA_DATE_FORMAT)
        for offset in range(-window_days, window_days + 1)
    ]


def render_like_x101(tokenizer, prompt: str, expected_sha: str, dates: Sequence[str]) -> tuple[str, str | None]:
    """x101's renderer first; if the template stamps a date, the x101 run date is recovered
    from the recorded sha256. Returns (rendered chat, date string used or None)."""
    chat = format_user_chat(tokenizer, prompt)
    if sha256_text(chat) == expected_sha:
        return chat, None
    for date in dates:
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
            date_string=date,
        )
        if sha256_text(chat) == expected_sha:
            return chat, date
    raise RuntimeError("rendered chat does not reproduce the x101 generation row's sha256")


def expected_renderings(tag: str, cohort: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """{condition: {prompt id: rendered_chat_sha256}} from the generation rows x101 judged."""
    units = {int(u["id"]): u for u in cohort["units"]}
    out: dict[str, dict[str, str]] = {}
    for condition in CONDITIONS:
        path = X83_RUN / f"x83_generations_{condition}.json" if tag == IMPORTED_TAG else generations_path(tag, condition)
        obj = read_json(path)
        if obj.get("condition") != condition:
            raise ValueError(f"{path} is not the {condition} generation file")
        rows = {int(r["id"]): r for r in obj["rows"]}
        out[condition] = {}
        for prompt_id, unit in units.items():
            row = rows[prompt_id]
            if row["prompt_sha256"] != unit["prompt_sha256"][condition]:
                raise ValueError(f"{tag}/{condition}: generation row {prompt_id} is not the cohort text")
            out[condition][str(prompt_id)] = row["rendered_chat_sha256"]
    return out


# --- projection arithmetic -------------------------------------------------------------


def unit_directions(mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-position unit vectors and norms of a [2, hidden] direction pair."""
    mu = np.asarray(mu, dtype=np.float64)
    if mu.ndim != 2 or mu.shape[0] != 2:
        raise ValueError("expected one direction per boundary position")
    norms = np.linalg.norm(mu, axis=1)
    if np.any(norms == 0):
        raise ValueError("zero boundary direction")
    return mu / norms[:, None], norms


def projection_record(delta: np.ndarray, mu: np.ndarray) -> dict[str, Any]:
    """proj = sum_t <delta_t, mu_t>/||mu_t||; shift = ||concat_t delta_t||; per-token parts."""
    delta = np.asarray(delta, dtype=np.float64)
    units, _ = unit_directions(mu)
    if delta.shape != units.shape:
        raise ValueError(f"delta {delta.shape} does not match directions {units.shape}")
    per_token = np.einsum("td,td->t", delta, units)
    token_norms = np.linalg.norm(delta, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosines = np.where(token_norms > 0, per_token / token_norms, 0.0)
    return {
        "proj": float(per_token.sum()),
        "proj_by_token": {p: float(v) for p, v in zip(BOUNDARY_POSITIONS, per_token)},
        "shift": float(np.linalg.norm(delta)),
        "shift_by_token": {p: float(v) for p, v in zip(BOUNDARY_POSITIONS, token_norms)},
        "cosine_by_token": {p: float(v) for p, v in zip(BOUNDARY_POSITIONS, cosines)},
    }


def random_unit_directions(dim: int, n: int, seed: int) -> np.ndarray:
    """[n, 2, dim] Gaussian unit directions, one per boundary position per draw."""
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((n, len(BOUNDARY_POSITIONS), dim))
    return (g / np.linalg.norm(g, axis=2, keepdims=True)).astype(np.float32)


def projections_onto(deltas: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """deltas [cells, 2, dim] x unit directions [n, 2, dim] -> [cells, n] summed over tokens."""
    return np.einsum("ctd,ntd->cn", deltas.astype(np.float32), directions.astype(np.float32)).astype(np.float64)


# --- statistics ------------------------------------------------------------------------


def group_means(x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    sums = np.bincount(groups, weights=x)
    counts = np.bincount(groups)
    return (sums / np.maximum(counts, 1))[groups]


def residualize(x: np.ndarray, *group_sets: np.ndarray, iterations: int = 200, tol: float = 1e-12) -> np.ndarray:
    """Remove additive fixed effects for each grouping (alternating demeaning; exact in one
    pass for a single grouping and for balanced two-way designs)."""
    r = np.asarray(x, dtype=np.float64).copy()
    for _ in range(iterations):
        before = r.copy()
        for groups in group_sets:
            r = r - group_means(r, np.asarray(groups, dtype=int))
        if np.max(np.abs(r - before)) < tol:
            break
    return r


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Spearman of x and y with z partialled out (Pearson partial correlation on ranks)."""
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    if np.ptp(rz) == 0:
        return spearman(x, y)
    design = np.column_stack([np.ones_like(rz), rz])
    ex = rx - design @ np.linalg.lstsq(design, rx, rcond=None)[0]
    ey = ry - design @ np.linalg.lstsq(design, ry, rcond=None)[0]
    if np.ptp(ex) == 0 or np.ptp(ey) == 0:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def primary_rho(proj: np.ndarray, dr: np.ndarray, pos_idx: np.ndarray) -> float:
    """The plan's primary statistic: Spearman after position fixed effects on both sides."""
    return spearman(residualize(proj, pos_idx), residualize(dr, pos_idx))


def permutation_p(proj: np.ndarray, dr: np.ndarray, pos_idx: np.ndarray, n_perm: int, seed: int) -> dict[str, Any]:
    """Two-sided p from permuting dr across prompts within each position block."""
    xr = residualize(proj, pos_idx)
    yr = residualize(dr, pos_idx)  # a within-position permutation of dr permutes yr the same way
    observed = spearman(xr, yr)
    rng = np.random.default_rng(seed)
    blocks = [np.flatnonzero(pos_idx == k) for k in np.unique(pos_idx)]
    null = np.empty(n_perm)
    rx = stats.rankdata(xr)
    for i in range(n_perm):
        yp = yr.copy()
        for block in blocks:
            yp[block] = yr[rng.permutation(block)]
        null[i] = np.corrcoef(rx, stats.rankdata(yp))[0, 1]
    exceed = int(np.sum(np.abs(null) >= abs(observed) - 1e-12))
    return {
        "rho": observed,
        "p_two_sided": (exceed + 1) / (n_perm + 1),
        "n_permutations": int(n_perm),
        "null_sd": float(null.std()),
    }


def prompt_bootstrap(
    proj: np.ndarray,
    dr: np.ndarray,
    shift: np.ndarray,
    prompt_idx: np.ndarray,
    pos_idx: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Prompt-level bootstrap of the primary rho, the shift-partial rho and the two-way-FE rho."""
    rng = np.random.default_rng(seed)
    prompts = np.unique(prompt_idx)
    members = [np.flatnonzero(prompt_idx == p) for p in prompts]
    primary = np.empty(n_boot)
    partial = np.empty(n_boot)
    two_way = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.integers(0, len(prompts), size=len(prompts))
        cells = np.concatenate([members[d] for d in draw])
        new_prompt = np.repeat(np.arange(len(prompts)), [len(members[d]) for d in draw])
        pos = pos_idx[cells]
        xr, yr, zr = residualize(proj[cells], pos), residualize(dr[cells], pos), residualize(shift[cells], pos)
        primary[b] = spearman(xr, yr)
        partial[b] = partial_spearman(xr, yr, zr)
        two_way[b] = spearman(residualize(proj[cells], new_prompt, pos), residualize(dr[cells], new_prompt, pos))

    def ci(samples: np.ndarray) -> list[float]:
        finite = samples[np.isfinite(samples)]
        return [float(v) for v in np.quantile(finite, (ALPHA / 2, 1 - ALPHA / 2))]

    return {"n_bootstrap": int(n_boot), "primary_ci95": ci(primary), "partial_shift_ci95": ci(partial), "two_way_fe_ci95": ci(two_way)}


def holm(p_values: Sequence[float]) -> list[float]:
    from x83_analyze import holm as _holm

    return _holm(list(p_values))


# --- leakage against the x85 crossed cohort -----------------------------------------------


def leakage_sets(x85_cohort: Mapping[str, Any]) -> tuple[set[int], set[str]]:
    if x85_cohort.get("experiment") != "x85":
        raise ValueError("not the x85 cohort")
    return {int(p["id"]) for p in x85_cohort["prompts"]}, {str(w["harm"]) for w in x85_cohort["words"]}


def leakage_free_mask(ids: Iterable[int], words: Iterable[str], prompt_ids: set[int], harm_words: set[str]) -> np.ndarray:
    """True where neither the prompt nor the word was in the x85 cohort the directions were fit on."""
    return np.asarray([int(i) not in prompt_ids and str(w) not in harm_words for i, w in zip(ids, words)], dtype=bool)


# --- behaviour ----------------------------------------------------------------------------


def load_refusal_arrays(tag: str, cohort: Mapping[str, Any]) -> tuple[list[int], dict[str, np.ndarray], list[int]]:
    """Per-prompt refusal2 arrays [n, 10] per condition, exactly as x101_analyze read them
    (x83's files for Qwen3.5-4B); prompts unjudged in any condition are dropped in all four."""
    from x83_analyze import load_generations, load_judgments
    from x101_analyze import MAX_DROPPED_PROMPTS, tag_paths, unjudged_prompts

    units = {int(u["id"]): u for u in cohort["units"]}
    paths = tag_paths(tag, cohort)
    excluded = {c: unjudged_prompts(paths["refusal"][c], c, units) for c in CONDITIONS}
    dropped = sorted(set().union(*excluded.values()))
    if len(dropped) > MAX_DROPPED_PROMPTS:
        raise SystemExit(f"FATAL: {tag} lacks labels for {len(dropped)} prompts")
    units = {i: u for i, u in units.items() if i not in dropped}
    ids = sorted(units)
    sha = paths["cohort_sha256"]
    generations = {c: load_generations(paths["generations"][c], c, units, sha) for c in CONDITIONS}
    labels = {c: load_judgments(paths["refusal"][c], c, units, generations[c], sha) for c in CONDITIONS}
    return ids, {c: np.stack([labels[c][i] for i in ids]) for c in CONDITIONS}, dropped
