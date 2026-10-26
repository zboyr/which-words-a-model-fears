"""x105: is the transferred boundary direction causally specific in the seven transfer models?

x101 showed that one inserted harm word raises `refusal2` in all eight models; x103 showed
that the shift the word makes along each model's own x88 boundary direction predicts that
rise. x105 writes that direction: a negative dose on the inserted (E) texts and a positive
dose on the original (O) texts of the frozen x101 cohort, over two dose magnitudes per sign,
against norm-matched orthogonal random directions and one KL-matched random direction per
model (the x86 control construction).

Everything x101 fixed is imported, not restated: the 130-prompt cohort and its exact O/E
strings, the per-model chat rendering verified against each generation row's recorded
`rendered_chat_sha256`, the sampling contract, and the per-(tag, prompt, condition) seeds —
reused unchanged, so every dosed arm shares its base arm's random-number stream — and the
x64 `refusal2` judge. The intervention is x88's: the consensus of the four fold directions,
written by the prefill-only `ExternalBoundaryHook` at the two trailing prefill positions of
the decoder block the frozen relative-depth rule selects, in multiples of the model's own
mean boundary shift. Dose 0 is not regenerated: the x101 (x83 for Qwen3.5-4B) generations
and judgments are the base arms.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from x86_common import (  # noqa: F401  (the x86 control construction and its KL contract)
    KL_CAP,
    KL_SLACK,
    orthogonal_random_null,
)
from x103_common import (  # noqa: F401  (re-exported for the stage scripts)
    ALL_TAGS,
    BOUNDARY_POSITIONS,
    CONDITIONS,
    DATA,
    FRESH_TAGS,
    IMPORTED_TAG,
    MODELS,
    N_RESPONSES,
    POSITIONS,
    QWEN4B_HOOK_BLOCK,
    QWEN4B_MODEL_ID,
    RESULTS,
    X83_RUN,
    X85_COHORT,
    X88_RUN,
    BoundaryReader,
    ExternalBoundaryHook,
    assert_boundary,
    atomic_json,
    bf16_ulp_bound,
    cohort_path,
    consensus_x86_vectors,
    date_candidates,
    directions_path,
    expected_renderings,
    file_record,
    format_user_chat,
    holm,
    hook_block_for,
    hook_site,
    leakage_sets,
    load_cohort,
    load_tag_model,
    manifest_path,
    n_blocks_of,
    read_json,
    render_like_x101,
    sha256_file,
    sha256_text,
    unit_directions,
)

RUN = DATA / "runs" / "x105"
X86_RUN = DATA / "runs" / "x86"
X86_MANIFEST = X86_RUN / "x86_manifest.json"

# Only the end position is dosed. It is the one insertion position whose refusal rise x101
# found in every one of the eight models (start and middle are null on Qwen3.5-0.8B), so
# "does the write remove the insertion-induced rise" is a well-posed question there in every
# model; dosing three positions would triple the generation budget for a weaker question.
TEXT_CONDITIONS = ("O", "E")

# Doses are multiples of the model's own mean boundary shift (the raw consensus vector), the
# x86/x88 unit: -1.5 is exactly the x88 transfer dose and +1.5 the x89 positive-dose point.
MAIN_MAGNITUDE = 1.5
HALF_MAGNITUDE = 0.75
DOSE_MAGNITUDES = (HALF_MAGNITUDE, MAIN_MAGNITUDE)

CONTROL_DRAWS = (1, 2)  # norm-matched orthogonal random directions per model
CONTROL_SEED = 10502  # x105 namespace for the x86 random-null construction

N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 10_500
ALPHA = 0.05

ARMS: tuple[dict[str, Any], ...] = (
    {"arm": "E_true_m075", "text": "E", "family": "true", "draw": None, "dose": -HALF_MAGNITUDE},
    {"arm": "E_true_m150", "text": "E", "family": "true", "draw": None, "dose": -MAIN_MAGNITUDE},
    {"arm": "O_true_p075", "text": "O", "family": "true", "draw": None, "dose": +HALF_MAGNITUDE},
    {"arm": "O_true_p150", "text": "O", "family": "true", "draw": None, "dose": +MAIN_MAGNITUDE},
    {"arm": "E_rand1_m150", "text": "E", "family": "random", "draw": 1, "dose": -MAIN_MAGNITUDE},
    {"arm": "E_rand2_m150", "text": "E", "family": "random", "draw": 2, "dose": -MAIN_MAGNITUDE},
    # Dose resolved from the model's own benign-KL calibration: the multiple of the random
    # direction whose benign KL equals the true direction's at -1.5 (x86's matched null).
    {"arm": "E_randkl_m150", "text": "E", "family": "random_kl", "draw": 1, "dose": None},
    {"arm": "O_rand1_p150", "text": "O", "family": "random", "draw": 1, "dose": +MAIN_MAGNITUDE},
)
ARM_NAMES = tuple(spec["arm"] for spec in ARMS)
BASE_ARMS = {"O": "O_base", "E": "E_base"}  # reused x101 / x83 artifacts, never regenerated
TRUE_ARMS = tuple(spec["arm"] for spec in ARMS if spec["family"] == "true")
RANDOM_ARMS_E = ("E_rand1_m150", "E_rand2_m150")
RANDOM_ARM_E_KL = "E_randkl_m150"
RANDOM_ARM_O = "O_rand1_p150"


def arm_spec(arm: str) -> dict[str, Any]:
    for spec in ARMS:
        if spec["arm"] == arm:
            return dict(spec)
    raise ValueError(f"unknown x105 arm {arm!r}")


# --- paths ---------------------------------------------------------------------------


def inputs_path() -> Path:
    return RUN / "x105_inputs.json"


def exclusions_path() -> Path:
    return RUN / "x105_x85_exclusions.json"


def tag_dir(tag: str) -> Path:
    if tag not in ALL_TAGS:
        raise ValueError(f"unknown x105 tag {tag!r}")
    return RUN / tag


def control_directions_path(tag: str) -> Path:
    return tag_dir(tag) / "x105_control_directions.npz"


def kl_calibration_path(tag: str) -> Path:
    return tag_dir(tag) / "x105_kl_calibration.json"


def generations_path(tag: str, arm: str) -> Path:
    arm_spec(arm)
    return tag_dir(tag) / f"x105_generations_{arm}.json"


def refusal_path(tag: str, arm: str) -> Path:
    arm_spec(arm)
    return tag_dir(tag) / f"x105_refusal_{arm}.jsonl"


def base_generations_path(tag: str, condition: str) -> Path:
    """The dose-0 generations: x101's for the seven transfer models, x83's for Qwen3.5-4B."""
    if condition not in TEXT_CONDITIONS:
        raise ValueError(f"x105 dose-0 arms are {TEXT_CONDITIONS}, not {condition!r}")
    if tag == IMPORTED_TAG:
        return X83_RUN / f"x83_generations_{condition}.json"
    from x101_common import generations_path as x101_generations_path

    return x101_generations_path(tag, condition)


def base_refusal_path(tag: str, condition: str) -> Path:
    if condition not in TEXT_CONDITIONS:
        raise ValueError(f"x105 dose-0 arms are {TEXT_CONDITIONS}, not {condition!r}")
    if tag == IMPORTED_TAG:
        return X83_RUN / f"x83_refusal_{condition}.jsonl"
    from x101_common import refusal_path as x101_refusal_path

    return x101_refusal_path(tag, condition)


def base_cohort_sha(tag: str, cohort: Mapping[str, Any]) -> str:
    """The `arms_sha256` the base generation and judgment rows carry."""
    return cohort["x83_cohort_sha256"] if tag == IMPORTED_TAG else cohort["cohort_sha256"]


def generation_seed(tag: str, prompt_id: int, text_condition: str, retry: bool = False) -> int:
    """The base arm's own seed, reused for every dose of the same text (common random numbers).

    Qwen3.5-4B's base rows are x83's, so they carry x83's seeds; the other seven carry x101's.
    """
    if text_condition not in TEXT_CONDITIONS:
        raise ValueError(f"unknown text condition {text_condition!r}")
    label = f"{text_condition}:retry4096" if retry else text_condition
    if tag == IMPORTED_TAG:
        from x83_common import stable_seed as x83_stable_seed

        return x83_stable_seed(int(prompt_id), label)
    from x101_common import stable_seed as x101_stable_seed

    return x101_stable_seed(tag, int(prompt_id), label)


# --- the x85 exclusion set (shared by the x103 and x105 analyses) -------------------------

# Every direction x103 projects onto and x105 writes was fit on the x85 crossed cohort
# (96 prompts x 24 harm words). The x101 cohort shares 47 of those prompts and 17 of those
# words, so the primary analyses run on the units that share neither; the full cohort is
# kept as a `_full` sibling result.
EXCLUSION_RULE = (
    "exclude every x101 unit whose prompt is an x85 cohort prompt or whose inserted word is "
    "an x85 harm word; the per-model boundary directions were fit on the x85 crossed cohort"
)
COHORT_MODES = ("leak_free", "full")


def build_exclusions(cohort: Mapping[str, Any], x85_cohort_path: str | Path = X85_COHORT) -> dict[str, Any]:
    """The frozen exclusion artifact, derived by `x103_common.leakage_sets` from the x85 cohort."""
    x85 = read_json(x85_cohort_path)
    x85_prompts, x85_words = leakage_sets(x85)
    units = [(int(u["id"]), str(u["word"])) for u in cohort["units"]]
    prompt_hits = sorted(i for i, _ in units if i in x85_prompts)
    word_hits = sorted({w for _, w in units if w in x85_words})
    retained = sorted(i for i, w in units if i not in x85_prompts and w not in x85_words)
    return {
        "experiment": "x105",
        "stage": "exclusions",
        "status": "frozen",
        "rule": EXCLUSION_RULE,
        "derivation": "x103_common.leakage_sets(x85 cohort), intersected with the x101 cohort's prompt ids and words",
        "x85_cohort": file_record(x85_cohort_path),
        "x85_cohort_sha256": x85["cohort_sha256"],
        "x101_cohort": file_record(cohort_path()),
        "x101_cohort_sha256": cohort["cohort_sha256"],
        "x85_prompt_ids": sorted(x85_prompts),
        "x85_harm_words": sorted(x85_words),
        "excluded_prompt_ids": prompt_hits,
        "excluded_words": word_hits,
        "retained_unit_ids": retained,
        "counts": {
            "n_units": len(units),
            "n_x85_prompts": len(x85_prompts),
            "n_x85_harm_words": len(x85_words),
            "n_units_prompt_in_x85": len(prompt_hits),
            "n_units_word_in_x85": sum(w in x85_words for _, w in units),
            "n_words_in_x85": len(word_hits),
            "n_units_excluded": len(units) - len(retained),
            "n_units_retained": len(retained),
            "n_cells_retained_x103": len(retained) * len(POSITIONS),
            "n_distinct_words_retained": len({w for i, w in units if i in set(retained)}),
        },
    }


def load_exclusions(cohort: Mapping[str, Any], path: str | Path | None = None) -> dict[str, Any]:
    """The frozen artifact, re-derived live and required to match it exactly."""
    path = Path(path) if path is not None else exclusions_path()
    frozen = read_json(path)
    if frozen.get("experiment") != "x105" or frozen.get("stage") != "exclusions" or frozen.get("status") != "frozen":
        raise SystemExit(f"FATAL: {path} is not the frozen x105 exclusion artifact")
    if frozen.get("x101_cohort_sha256") != cohort["cohort_sha256"]:
        raise SystemExit("FATAL: the exclusion artifact belongs to another x101 cohort")
    live = build_exclusions(cohort)
    for key in ("x85_cohort_sha256", "x85_prompt_ids", "x85_harm_words", "excluded_prompt_ids", "excluded_words", "retained_unit_ids", "counts"):
        if frozen.get(key) != live[key]:
            raise SystemExit(f"FATAL: the exclusion artifact's {key} differs from the live derivation")
    frozen["file"] = file_record(path)
    return frozen


def retained_ids(ids: Sequence[int], mode: str, exclusions: Mapping[str, Any]) -> list[int]:
    """The prompt ids an analysis keeps under one cohort mode."""
    if mode not in COHORT_MODES:
        raise ValueError(f"unknown cohort mode {mode!r}; expected one of {COHORT_MODES}")
    if mode == "full":
        return [int(i) for i in ids]
    keep = set(exclusions["retained_unit_ids"])
    return [int(i) for i in ids if int(i) in keep]


def result_path_for(base: str | Path, mode: str) -> Path:
    """Leak-free results are the canonical files; the full cohort gets the `_full` sibling."""
    base = Path(base)
    if mode == "leak_free":
        return base
    if mode == "full":
        return base.with_name(f"{base.stem}_full{base.suffix}")
    raise ValueError(f"unknown cohort mode {mode!r}")


def cohort_record(mode: str, exclusions: Mapping[str, Any], n_units: int) -> dict[str, Any]:
    return {
        "mode": mode,
        "rule": EXCLUSION_RULE if mode == "leak_free" else "all cohort units; no exclusion",
        "exclusions": exclusions["file"],
        "n_units_analysed": int(n_units),
        "n_units_excluded": exclusions["counts"]["n_units_excluded"] if mode == "leak_free" else 0,
        "excluded_prompt_ids": exclusions["excluded_prompt_ids"] if mode == "leak_free" else [],
        "excluded_words": exclusions["excluded_words"] if mode == "leak_free" else [],
    }


# --- control directions (the x86 construction, per model) -------------------------------


def control_direction_pair(mu: np.ndarray, tag: str, draw: int) -> np.ndarray:
    """One norm-matched random direction per boundary position, orthogonal to the true one.

    `x86_common.orthogonal_random_null` is called with the draw identity in place of x86's
    fold index, so the seed payload is unique per (tag, draw, position) and the arithmetic —
    Gaussian, projected off the true direction, rescaled to its norm — is byte-for-byte the
    construction x86's KL-matched random null used on Qwen3.5-4B.
    """
    mu = np.asarray(mu, dtype=np.float64)
    if mu.ndim != 2 or mu.shape[0] != len(BOUNDARY_POSITIONS):
        raise ValueError("expected one true direction per boundary position")
    return np.stack(
        [
            orthogonal_random_null(mu[index], f"x105|{tag}|{draw}", position, base_seed=CONTROL_SEED)
            for index, position in enumerate(BOUNDARY_POSITIONS)
        ]
    )


def control_diagnostics(mu: np.ndarray, control: np.ndarray) -> dict[str, Any]:
    mu = np.asarray(mu, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    norms_mu = np.linalg.norm(mu, axis=1)
    norms_control = np.linalg.norm(control, axis=1)
    cosines = np.einsum("td,td->t", mu, control) / (norms_mu * norms_control)
    return {
        "norm_ratio_by_token": {p: float(v) for p, v in zip(BOUNDARY_POSITIONS, norms_control / norms_mu)},
        "cosine_to_true_by_token": {p: float(v) for p, v in zip(BOUNDARY_POSITIONS, cosines)},
        "max_abs_cosine_to_true": float(np.max(np.abs(cosines))),
        "max_norm_ratio_error": float(np.max(np.abs(norms_control / norms_mu - 1.0))),
    }


def control_arrays(mu: np.ndarray, tag: str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for draw in CONTROL_DRAWS:
        pair = control_direction_pair(mu, tag, draw)
        for index, position in enumerate(BOUNDARY_POSITIONS):
            out[f"rand{draw}_{position}"] = pair[index].astype(np.float32)
    return out


def load_control_pair(path: str | Path, draw: int) -> np.ndarray:
    with np.load(path) as artifact:
        missing = [f"rand{draw}_{p}" for p in BOUNDARY_POSITIONS if f"rand{draw}_{p}" not in artifact.files]
        if missing:
            raise ValueError(f"control artifact lacks {missing}")
        return np.stack([artifact[f"rand{draw}_{p}"] for p in BOUNDARY_POSITIONS]).astype(np.float64)


# --- benign KL calibration prompts (x86's 256, model-independent texts) ------------------


def benign_calibration_prompts(path: str | Path = X86_MANIFEST) -> list[dict[str, Any]]:
    manifest = read_json(path)
    if manifest.get("experiment") != "x86" or manifest.get("status") != "frozen":
        raise ValueError("not the frozen x86 manifest")
    rows = [
        {"full_row": int(row["full_row"]), "order": int(row["order"]), "prompt": str(row["prompt"])}
        for row in manifest["benign_calibration"]
    ]
    if not rows:
        raise ValueError("x86 manifest carries no benign calibration prompts")
    return rows


def resolve_arm_vectors(
    arm: str,
    mu: np.ndarray,
    control_path: str | Path,
    calibration: Mapping[str, Any] | None,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """(direction pair, signed dose, auditable metadata) for one arm of one model."""
    spec = arm_spec(arm)
    if spec["family"] == "true":
        vectors, dose = np.asarray(mu, dtype=np.float64), float(spec["dose"])
        meta: dict[str, Any] = {"kind": "x88 consensus boundary direction"}
    else:
        vectors = load_control_pair(control_path, int(spec["draw"]))
        meta = {"kind": f"norm-matched orthogonal random direction, draw {spec['draw']}", "control": file_record(control_path)}
        meta.update(control_diagnostics(mu, vectors))
        if spec["family"] == "random":
            dose = float(spec["dose"])
        else:
            if calibration is None:
                raise SystemExit(f"FATAL: {arm} needs this model's benign-KL calibration")
            dose = matched_dose(calibration, sign=-1.0 if spec["text"] == "E" else +1.0)
            meta["kind"] = f"KL-matched random direction, draw {spec['draw']}"
            meta["kl_matching"] = kl_matching_record(calibration, dose)
    if np.asarray(vectors).shape != np.asarray(mu).shape:
        raise SystemExit(f"FATAL: {arm} direction shape {np.asarray(vectors).shape} != {np.asarray(mu).shape}")
    if dose == 0.0:
        raise SystemExit(f"FATAL: {arm} resolved to dose 0")
    return np.asarray(vectors, dtype=np.float64), dose, meta


def matched_dose(calibration: Mapping[str, Any], sign: float) -> float:
    key = "minus" if sign < 0 else "plus"
    record = calibration["random_kl_match"][key]
    if record.get("status") != "pass":
        raise SystemExit(f"FATAL: random KL match did not pass on the {key} side")
    dose = float(record["matched_signed_dose"])
    if dose * sign <= 0:
        raise SystemExit("FATAL: matched dose has the wrong sign")
    return dose


def kl_matching_record(calibration: Mapping[str, Any], dose: float) -> dict[str, Any]:
    key = "minus" if dose < 0 else "plus"
    record = calibration["random_kl_match"][key]
    return {
        "target_true_mean_kl": record["target_kl"],
        "matched_mean_kl": record["matched_mean_kl"],
        "matched_signed_dose": record["matched_signed_dose"],
        "multiple_of_true_dose": abs(float(record["matched_signed_dose"])) / MAIN_MAGNITUDE,
    }


def load_tag_tokenizer(source: str, tag: str):
    """The tokenizer x101 (x83 for Qwen3.5-4B) generated with: left padding, pad = eos."""
    from transformers import AutoTokenizer

    trust = True if tag == IMPORTED_TAG else bool(MODELS[tag]["trust_remote_code"])
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=trust)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


# --- the write site, audited exactly as x103 audited the read site ------------------------


def audit_write_site(model, tokenizer, site, block: int, vectors: np.ndarray, dose: float, chats: Sequence[str]) -> dict:
    """The hook writes `dose * vectors` into the tensor `output_hidden_states[block + 1]` holds
    at the two trailing prefill positions, once per prefill and never on a decode step, and
    removing it restores the forward exactly."""
    import torch

    minus1 = torch.from_numpy(np.asarray(vectors[0], dtype=np.float32))
    tbg = torch.from_numpy(np.asarray(vectors[1], dtype=np.float32))
    mu = torch.stack([minus1, tbg]).float()
    reports = []
    for chat in chats:
        encoded = tokenizer(chat, return_tensors="pt", add_special_tokens=False).to(model.device)

        reader = BoundaryReader()
        handle = site.register_forward_hook(reader)
        with torch.inference_mode():
            out = model(**encoded, use_cache=False, output_hidden_states=True)
        handle.remove()
        base = reader.last.clone()
        via_states = out.hidden_states[block + 1][:, -2:, :].float().cpu()
        base_logits = out.logits[:, -1, :].float().cpu()

        hook = ExternalBoundaryHook(dose, minus1, tbg)
        write_handle = site.register_forward_hook(hook)
        reader = BoundaryReader()
        handle = site.register_forward_hook(reader)
        with torch.inference_mode():
            out_edited = model(**encoded, use_cache=False)
        handle.remove()
        write_handle.remove()
        edited = reader.last.clone()

        reader = BoundaryReader()
        handle = site.register_forward_hook(reader)
        with torch.inference_mode():
            model(**encoded, use_cache=False)
        handle.remove()
        restored = reader.last.clone()

        diff = (edited - base)[0]
        target = dose * mu
        bound = bf16_ulp_bound(edited[0], target)
        cos = [float(torch.nn.functional.cosine_similarity(diff[t], target[t], dim=0)) for t in range(2)]
        norm_ratio = [float(diff[t].norm() / target[t].norm()) for t in range(2)]
        reports.append(
            {
                "n_input_tokens": int(encoded["input_ids"].shape[1]),
                "reader_equals_output_hidden_states": bool(torch.equal(base, via_states)),
                "write_hook_prefill_calls": hook.prefill_calls,
                "write_hook_decode_calls": hook.decode_calls,
                "edit_equals_dose_mu_within_bf16_ulp": bool(torch.all((diff - target).abs() <= bound)),
                "edit_vs_dose_mu_max_abs": float((diff - target).abs().max()),
                "edit_cosine_to_dose_mu": cos,
                "edit_norm_over_dose_mu_norm": norm_ratio,
                "edit_geometry_ok": bool(min(cos) >= 0.999 and max(abs(r - 1.0) for r in norm_ratio) <= 0.01),
                "logits_changed_by_write": bool((out_edited.logits[:, -1, :].float().cpu() - base_logits).abs().max() > 0),
                "restored_after_removal": bool(torch.equal(restored, base)),
                "reader_calls_per_forward": reader.calls,
            }
        )
    passed = all(
        r["reader_equals_output_hidden_states"]
        and r["edit_equals_dose_mu_within_bf16_ulp"]
        and r["edit_geometry_ok"]
        and r["logits_changed_by_write"]
        and r["restored_after_removal"]
        and r["write_hook_prefill_calls"] == 1
        and r["write_hook_decode_calls"] == 0
        and r["reader_calls_per_forward"] == 1
        for r in reports
    )
    return {"status": "pass" if passed else "fail", "dose": float(dose), "texts": reports}


# --- statistics --------------------------------------------------------------------------


def rate_matrix(labels: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-prompt refusal2 rate per arm from (prompts, 10) binary label arrays."""
    out = {}
    for arm, values in labels.items():
        array = np.asarray(values)
        if array.ndim != 2 or array.shape[1] != N_RESPONSES:
            raise ValueError(f"{arm}: labels must be (prompts, {N_RESPONSES})")
        out[arm] = array.mean(axis=1)
    return out


def estimands(rates: Mapping[str, np.ndarray]) -> dict[str, float]:
    """Every x105 quantity, from per-prompt rates. Positive = the write moved refusal the way
    the direction says it should: down on inserted texts under a negative dose, up on the
    original texts under a positive dose."""
    o0, e0 = rates[BASE_ARMS["O"]], rates[BASE_ARMS["E"]]
    mean = lambda value: float(np.mean(value))
    suppress_half = mean(e0 - rates["E_true_m075"])
    suppress_full = mean(e0 - rates["E_true_m150"])
    induce_half = mean(rates["O_true_p075"] - o0)
    induce_full = mean(rates["O_true_p150"] - o0)
    rise = mean(e0 - o0)
    random_e = [mean(e0 - rates[arm]) for arm in RANDOM_ARMS_E]
    random_e_full = float(np.mean(random_e))
    random_e_kl = mean(e0 - rates[RANDOM_ARM_E_KL])
    random_o_full = mean(rates[RANDOM_ARM_O] - o0)
    bidirectional_half = (suppress_half + induce_half) / 2.0
    bidirectional_full = (suppress_full + induce_full) / 2.0
    bidirectional_random = (random_e_full + random_o_full) / 2.0
    # Oriented dose slope over magnitudes {0, 0.75, 1.5}: per prompt, z(m) = (r_O(+m) - r_E(-m))/2
    # rises with m if the write moves both texts the way the direction predicts. Slope is the
    # per-prompt OLS coefficient on m, averaged over prompts (x86's fixed-effect dose slope).
    z = np.stack(
        [
            (o0 - e0) / 2.0,
            (rates["O_true_p075"] - rates["E_true_m075"]) / 2.0,
            (rates["O_true_p150"] - rates["E_true_m150"]) / 2.0,
        ],
        axis=1,
    )
    magnitudes = np.asarray([0.0, HALF_MAGNITUDE, MAIN_MAGNITUDE])
    centered = magnitudes - magnitudes.mean()
    slope = mean((z @ centered) / float(np.square(centered).sum()))
    return {
        "insertion_rise_E_minus_O": rise,
        "suppress_E_half": suppress_half,
        "suppress_E_full": suppress_full,
        "induce_O_half": induce_half,
        "induce_O_full": induce_full,
        "bidirectional_half": bidirectional_half,
        "bidirectional_full": bidirectional_full,
        "fraction_of_rise_removed_full": suppress_full / rise if rise > 0 else float("nan"),
        "residual_rise_full": mean(rates["E_true_m150"] - o0),
        "random_suppress_E_full": random_e_full,
        "random_kl_suppress_E_full": random_e_kl,
        "random_induce_O_full": random_o_full,
        "random_bidirectional_full": bidirectional_random,
        "true_minus_random_E": suppress_full - random_e_full,
        "true_minus_random_kl_E": suppress_full - random_e_kl,
        "true_minus_random_O": induce_full - random_o_full,
        "true_minus_random_bidirectional": bidirectional_full - bidirectional_random,
        "suppress_E_full_minus_half": suppress_full - suppress_half,
        "induce_O_full_minus_half": induce_full - induce_half,
        "oriented_dose_slope": slope,
        **{f"random_suppress_E_draw{draw}": value for draw, value in zip(CONTROL_DRAWS, random_e)},
    }


def bootstrap_estimands(
    labels: Mapping[str, np.ndarray],
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
    chunk: int = 500,
) -> dict[str, np.ndarray]:
    """x83's two-stage paired bootstrap carried over every arm at once: resample prompts, then
    redraw each arm's ten binary responses for the drawn prompt, so each replicate recomputes
    all estimands — including differences between contrasts — on one prompt sample."""
    rates = rate_matrix(labels)
    arms = sorted(rates)
    n = len(rates[arms[0]])
    if any(len(rates[arm]) != n for arm in arms):
        raise ValueError("arms are not aligned on one prompt set")
    rng = np.random.default_rng(seed)
    names = list(estimands(rates))
    out = {name: np.empty(n_bootstrap, dtype=float) for name in names}
    for start in range(0, n_bootstrap, chunk):
        stop = min(start + chunk, n_bootstrap)
        index = rng.integers(0, n, size=(stop - start, n))
        drawn = {arm: rng.binomial(N_RESPONSES, rates[arm][index]) / N_RESPONSES for arm in arms}
        for offset in range(stop - start):
            replicate = estimands({arm: drawn[arm][offset] for arm in arms})
            for name in names:
                out[name][start + offset] = replicate[name]
    return out


def interval(samples: np.ndarray, alpha: float = ALPHA) -> list[float]:
    finite = np.asarray(samples)[np.isfinite(samples)]
    if not len(finite):
        return [float("nan"), float("nan")]
    return [float(value) for value in np.quantile(finite, (alpha / 2, 1 - alpha / 2))]


def one_sided_p(samples: np.ndarray) -> float:
    """P(estimand <= 0) under the bootstrap, the direction the design predicts (x86's rule)."""
    finite = np.asarray(samples)[np.isfinite(samples)]
    if not len(finite):
        return float("nan")
    return float((1 + np.sum(finite <= 0)) / (len(finite) + 1))


def summarize(point: Mapping[str, float], samples: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        name: {
            "estimate": float(value) if np.isfinite(value) else None,
            "ci95": interval(samples[name]),
            "one_sided_p": one_sided_p(samples[name]),
        }
        for name, value in point.items()
    }


def monotone_flags(point: Mapping[str, float], samples: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Dose ordering on the point estimates and its bootstrap frequency."""
    ordered = (
        np.asarray(samples["suppress_E_full_minus_half"]) > 0
    ) & (np.asarray(samples["suppress_E_half"]) > 0) & (
        np.asarray(samples["induce_O_full_minus_half"]) > 0
    ) & (np.asarray(samples["induce_O_half"]) > 0)
    return {
        "suppress_E_ordered": bool(point["suppress_E_full"] > point["suppress_E_half"] > 0),
        "induce_O_ordered": bool(point["induce_O_full"] > point["induce_O_half"] > 0),
        "bidirectional_ordered": bool(point["bidirectional_full"] > point["bidirectional_half"] > 0),
        "bootstrap_fraction_both_orderings": float(np.mean(ordered)),
    }


def verdict(report: Mapping[str, Any]) -> str:
    """A reading of one model's three primary intervals, not a decision rule."""
    tests = report["primary"]
    passed = [
        tests["suppress_E_full"]["ci95"][0] > 0,
        tests["induce_O_full"]["ci95"][0] > 0,
        tests["true_minus_random_E"]["ci95"][0] > 0,
    ]
    if all(passed):
        return "holds"
    if any(passed):
        return "partial"
    return "absent"
