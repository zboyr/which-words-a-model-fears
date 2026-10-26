"""Contract tests for x105: the write site, the control construction, the arm table, the
common-random-number seeds, the estimands, and the bootstrap/Holm plumbing."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x83_common import stable_seed as x83_seed  # noqa: E402
from x101_common import stable_seed as x101_seed  # noqa: E402
from x105_common import (  # noqa: E402
    ARMS,
    BASE_ARMS,
    BOUNDARY_POSITIONS,
    CONTROL_DRAWS,
    HALF_MAGNITUDE,
    IMPORTED_TAG,
    MAIN_MAGNITUDE,
    N_RESPONSES,
    TEXT_CONDITIONS,
    arm_spec,
    audit_write_site,
    bootstrap_estimands,
    control_arrays,
    control_diagnostics,
    control_direction_pair,
    estimands,
    generation_seed,
    hook_block_for,
    hook_site,
    interval,
    load_control_pair,
    monotone_flags,
    one_sided_p,
    rate_matrix,
    resolve_arm_vectors,
    verdict,
)


class _Block(nn.Module):
    def __init__(self, dim: int, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.mlp = nn.Linear(dim, dim)

    def forward(self, hidden):
        return (hidden + torch.tanh(self.mlp(hidden)),)


class _Config:
    def __init__(self, n: int) -> None:
        self.num_hidden_layers = n

    def get_text_config(self):
        return self


class _Decoder(nn.Module):
    """An HF-shaped decoder whose call signature matches what the audit uses."""

    def __init__(self, n_blocks: int, dim: int) -> None:
        super().__init__()
        self.config = _Config(n_blocks)
        self.layers = nn.ModuleList([_Block(dim, seed) for seed in range(n_blocks)])
        self.head = nn.Linear(dim, 5)
        self.device = torch.device("cpu")

    def forward(self, input_ids=None, use_cache=False, output_hidden_states=False):
        hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, self.layers[0].mlp.in_features) / 10.0
        states = [hidden]
        for layer in self.layers:
            hidden = layer(hidden)[0]
            states.append(hidden)
        logits = self.head(hidden)
        if output_hidden_states:
            return type("Out", (), {"logits": logits, "hidden_states": states})()
        return type("Out", (), {"logits": logits})()


class _Tokenizer:
    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        ids = torch.tensor([[len(word) for word in text.split()]], dtype=torch.long)

        class _Batch(dict):
            def to(self, _device):
                return self

        return _Batch(input_ids=ids)


def test_arm_table_is_the_frozen_design():
    assert len(ARMS) == 8 and len({spec["arm"] for spec in ARMS}) == 8
    assert set(BASE_ARMS) == set(TEXT_CONDITIONS) == {"O", "E"}
    for spec in ARMS:
        assert spec["text"] in TEXT_CONDITIONS
        assert spec["family"] in ("true", "random", "random_kl")
        # the sign of every dose follows the text it is written on
        if spec["dose"] is not None:
            assert (spec["dose"] < 0) == (spec["text"] == "E")
            assert abs(spec["dose"]) in (HALF_MAGNITUDE, MAIN_MAGNITUDE)
        else:
            assert spec["family"] == "random_kl"
        assert (spec["draw"] is None) == (spec["family"] == "true")
    true_doses = sorted(spec["dose"] for spec in ARMS if spec["family"] == "true")
    assert true_doses == [-MAIN_MAGNITUDE, -HALF_MAGNITUDE, HALF_MAGNITUDE, MAIN_MAGNITUDE]
    assert arm_spec("E_true_m150")["dose"] == -MAIN_MAGNITUDE
    for bad in ("nope", "O_base"):
        try:
            arm_spec(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} must not be an x105 arm")


def test_control_directions_are_orthogonal_norm_matched_and_reproducible(tmp_path=None):
    mu = np.random.default_rng(0).standard_normal((2, 64)) * 3.0
    first = control_direction_pair(mu, "llama", 1)
    second = control_direction_pair(mu, "llama", 2)
    assert first.shape == mu.shape
    assert np.array_equal(first, control_direction_pair(mu, "llama", 1))  # deterministic
    assert not np.allclose(first, second)
    assert not np.allclose(first, control_direction_pair(mu, "phi", 1))  # per model
    checks = control_diagnostics(mu, first)
    assert checks["max_abs_cosine_to_true"] < 1e-12
    assert checks["max_norm_ratio_error"] < 1e-12
    for position in BOUNDARY_POSITIONS:
        assert abs(checks["norm_ratio_by_token"][position] - 1.0) < 1e-12
    arrays = control_arrays(mu, "llama")
    assert set(arrays) == {f"rand{d}_{p}" for d in CONTROL_DRAWS for p in BOUNDARY_POSITIONS}


def test_resolve_arm_vectors_picks_direction_dose_and_kl_match(tmp_path):
    mu = np.random.default_rng(1).standard_normal((2, 32)) * 2.0
    path = tmp_path / "x105_control_directions.npz"
    np.savez(path, **control_arrays(mu, "gemma"))
    vectors, dose, meta = resolve_arm_vectors("E_true_m150", mu, path, None)
    assert np.allclose(vectors, mu) and dose == -MAIN_MAGNITUDE and "consensus" in meta["kind"]
    vectors, dose, meta = resolve_arm_vectors("E_rand2_m150", mu, path, None)
    assert dose == -MAIN_MAGNITUDE and meta["max_abs_cosine_to_true"] < 1e-6
    assert np.allclose(vectors, load_control_pair(path, 2))
    calibration = {
        "random_kl_match": {
            "minus": {"status": "pass", "matched_signed_dose": -3.25, "matched_mean_kl": 0.4, "target_kl": 0.41},
            "plus": {"status": "pass", "matched_signed_dose": +3.1, "matched_mean_kl": 0.4, "target_kl": 0.4},
        }
    }
    vectors, dose, meta = resolve_arm_vectors("E_randkl_m150", mu, path, calibration)
    assert dose == -3.25 and np.allclose(vectors, load_control_pair(path, 1))
    assert abs(meta["kl_matching"]["multiple_of_true_dose"] - 3.25 / MAIN_MAGNITUDE) < 1e-12
    for bad in ({"random_kl_match": {"minus": {"status": "tolerance_miss"}}}, None):
        try:
            resolve_arm_vectors("E_randkl_m150", mu, path, bad)
        except SystemExit:
            continue
        raise AssertionError("a failed or missing KL match must abort the arm")


def test_seeds_are_the_base_arms_own_so_doses_share_random_numbers():
    assert generation_seed("llama", 7, "E") == x101_seed("llama", 7, "E")
    assert generation_seed(IMPORTED_TAG, 7, "E") == x83_seed(7, "E")
    assert generation_seed("llama", 7, "E", retry=True) == x101_seed("llama", 7, "E:retry4096")
    assert generation_seed("llama", 7, "E") != generation_seed("llama", 7, "O")
    assert generation_seed("llama", 7, "E") != generation_seed("phi", 7, "E")
    try:
        generation_seed("llama", 7, "M")
    except ValueError:
        return
    raise AssertionError("x105 only doses the O and E texts")


def test_audit_catches_a_write_that_is_not_at_the_x88_site():
    dim, n_blocks = 8, 16
    model = _Decoder(n_blocks, dim).eval()
    block = hook_block_for(n_blocks)
    vectors = np.stack([np.arange(1.0, dim + 1.0), -np.arange(1.0, dim + 1.0) / 4])
    chats = ["one two three four five", "six seven eight nine"]
    report = audit_write_site(model, _Tokenizer(), hook_site(model, block), block, vectors, -1.5, chats)
    assert report["status"] == "pass" and report["dose"] == -1.5
    assert all(r["write_hook_prefill_calls"] == 1 and r["write_hook_decode_calls"] == 0 for r in report["texts"])
    assert all(r["reader_equals_output_hidden_states"] and r["restored_after_removal"] for r in report["texts"])
    # reading one block downstream is not the site the write edits
    wrong = audit_write_site(model, _Tokenizer(), hook_site(model, block + 1), block, vectors, -1.5, chats)
    assert wrong["status"] == "fail"


def _labels(rng, n, rates):
    return {arm: (rng.random((n, N_RESPONSES)) < rate).astype(np.int8) for arm, rate in rates.items()}


def test_estimands_have_the_predicted_signs_and_the_random_contrast_is_a_difference():
    n = 60
    rng = np.random.default_rng(4)
    labels = _labels(
        rng,
        n,
        {
            "O_base": 0.20, "E_base": 0.60,
            "E_true_m075": 0.45, "E_true_m150": 0.30,
            "O_true_p075": 0.30, "O_true_p150": 0.40,
            "E_rand1_m150": 0.58, "E_rand2_m150": 0.56, "E_randkl_m150": 0.55,
            "O_rand1_p150": 0.21,
        },
    )
    point = estimands(rate_matrix(labels))
    assert point["insertion_rise_E_minus_O"] > 0.3
    assert point["suppress_E_full"] > point["suppress_E_half"] > 0
    assert point["induce_O_full"] > point["induce_O_half"] > 0
    assert point["true_minus_random_E"] > 0.2
    assert abs(point["true_minus_random_E"] - (point["suppress_E_full"] - point["random_suppress_E_full"])) < 1e-12
    assert abs(point["random_suppress_E_full"] - np.mean([point["random_suppress_E_draw1"], point["random_suppress_E_draw2"]])) < 1e-12
    assert 0 < point["fraction_of_rise_removed_full"] < 1
    assert point["oriented_dose_slope"] > 0
    assert abs(point["bidirectional_full"] - (point["suppress_E_full"] + point["induce_O_full"]) / 2) < 1e-12

    # a direction that does nothing: every effect is null and the contrast loses its sign
    flat = _labels(rng, n, {arm: (0.6 if arm.startswith("E") else 0.2) for arm in labels})
    null = estimands(rate_matrix(flat))
    assert abs(null["suppress_E_full"]) < 0.1 and abs(null["induce_O_full"]) < 0.1
    assert abs(null["true_minus_random_E"]) < 0.1


def test_bootstrap_is_seeded_paired_and_feeds_the_verdict():
    n = 50
    rng = np.random.default_rng(9)
    labels = _labels(
        rng,
        n,
        {
            "O_base": 0.20, "E_base": 0.65,
            "E_true_m075": 0.45, "E_true_m150": 0.25,
            "O_true_p075": 0.32, "O_true_p150": 0.45,
            "E_rand1_m150": 0.64, "E_rand2_m150": 0.63, "E_randkl_m150": 0.62,
            "O_rand1_p150": 0.20,
        },
    )
    point = estimands(rate_matrix(labels))
    samples = bootstrap_estimands(labels, n_bootstrap=400, seed=11)
    again = bootstrap_estimands(labels, n_bootstrap=400, seed=11)
    assert np.array_equal(samples["suppress_E_full"], again["suppress_E_full"])
    assert not np.array_equal(samples["suppress_E_full"], bootstrap_estimands(labels, n_bootstrap=400, seed=12)["suppress_E_full"])
    for name in ("suppress_E_full", "induce_O_full", "true_minus_random_E"):
        low, high = interval(samples[name])
        assert low < point[name] < high and low > 0
        assert one_sided_p(samples[name]) < 0.01
    flags = monotone_flags(point, samples)
    assert flags["suppress_E_ordered"] and flags["induce_O_ordered"] and flags["bidirectional_ordered"]
    assert flags["bootstrap_fraction_both_orderings"] > 0.5
    report = {"primary": {name: {"estimate": point[name], "ci95": interval(samples[name])} for name in
                          ("suppress_E_full", "induce_O_full", "true_minus_random_E")}}
    assert verdict(report) == "holds"
    report["primary"]["true_minus_random_E"]["ci95"] = [-0.05, 0.10]
    assert verdict(report) == "partial"
    for name in report["primary"]:
        report["primary"][name]["ci95"] = [-0.05, 0.10]
    assert verdict(report) == "absent"
