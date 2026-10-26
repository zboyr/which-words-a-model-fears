"""Contract tests for x103: the read site matches the x88 write site, projection arithmetic,
fixed-effect residualisation, the permutation/partial statistics, and the leakage mask."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x88_common import ExternalBoundaryHook, hook_block_for  # noqa: E402
from x103_common import (  # noqa: E402
    BOUNDARY_POSITIONS,
    BoundaryReader,
    bf16_ulp_bound,
    date_candidates,
    hook_site,
    leakage_free_mask,
    partial_spearman,
    permutation_p,
    primary_rho,
    projection_record,
    projections_onto,
    random_unit_directions,
    residualize,
    spearman,
)


class _Block(nn.Module):
    """An HF-style decoder block: returns a tuple whose first element is the residual stream."""

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
    def __init__(self, n_blocks: int, dim: int) -> None:
        super().__init__()
        self.config = _Config(n_blocks)
        self.layers = nn.ModuleList([_Block(dim, seed) for seed in range(n_blocks)])

    def forward(self, hidden):
        states = [hidden]
        for layer in self.layers:
            hidden = layer(hidden)[0]
            states.append(hidden)
        return hidden, states


def test_reader_sees_exactly_the_tensor_the_x88_write_edits():
    dim, n_blocks = 8, 16
    model = _Decoder(n_blocks, dim).eval()
    block = hook_block_for(n_blocks)
    site = hook_site(model, block)
    x = torch.randn(1, 7, dim)
    minus1 = torch.arange(1.0, dim + 1.0)
    tbg = -torch.arange(1.0, dim + 1.0) / 4

    reader = BoundaryReader()
    handle = site.register_forward_hook(reader)
    with torch.no_grad():
        _, states = model(x)
    handle.remove()
    base = reader.last.clone()
    # (a) the same residual tensor x88_extract_directions read: output_hidden_states[block + 1]
    assert torch.equal(base, states[block + 1][:, -2:, :])
    assert not torch.equal(base, states[block][:, -2:, :])

    # (b) the x88 write registered ahead of the reader moves the read tensor by exactly dose * mu
    dose = -1.5
    write = site.register_forward_hook(ExternalBoundaryHook(dose, minus1, tbg))
    reader = BoundaryReader()
    handle = site.register_forward_hook(reader)
    with torch.no_grad():
        model(x)
    handle.remove()
    write.remove()
    shift = (reader.last - base)[0]
    assert torch.allclose(shift[0], dose * minus1, atol=1e-6)
    assert torch.allclose(shift[1], dose * tbg, atol=1e-6)
    assert reader.calls == 1

    # a reader one block downstream does not see the pure write
    other = BoundaryReader()
    handle = hook_site(model, block + 1).register_forward_hook(other)
    write = site.register_forward_hook(ExternalBoundaryHook(dose, minus1, tbg))
    with torch.no_grad():
        model(x)
    handle.remove()
    write.remove()
    assert not torch.allclose((other.last - states[block + 2][:, -2:, :])[0, 0], dose * minus1, atol=1e-3)


def test_bf16_ulp_bound_accepts_bf16_rounding_of_the_write():
    torch.manual_seed(0)
    h = (torch.randn(2, 4096) * 40).to(torch.bfloat16)
    mu = torch.randn(2, 4096) * 3
    edited = h + mu.to(torch.bfloat16)  # the x88 hook's arithmetic: mu cast to bf16, bf16 add
    diff = edited.float() - h.float()
    assert torch.all((diff - mu).abs() <= bf16_ulp_bound(edited.float(), mu))
    assert not torch.all((diff - mu).abs() <= bf16_ulp_bound(edited.float(), torch.zeros_like(mu)) * 0.1)


def test_projection_record_sums_unit_projections_and_concatenated_norm():
    mu = np.array([[3.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    delta = np.array([[1.0, 2.0, 0.0], [0.0, 0.0, -0.5]])
    rec = projection_record(delta, mu)
    assert abs(rec["proj"] - (1.0 - 0.5)) < 1e-12
    assert abs(rec["proj_by_token"]["tbg_minus1"] - 1.0) < 1e-12
    assert abs(rec["proj_by_token"]["tbg"] + 0.5) < 1e-12
    assert abs(rec["shift"] - np.sqrt(1 + 4 + 0.25)) < 1e-12
    assert abs(rec["cosine_by_token"]["tbg"] + 1.0) < 1e-12
    assert list(rec["proj_by_token"]) == list(BOUNDARY_POSITIONS)


def test_random_directions_are_unit_seeded_and_project_like_the_record():
    dirs = random_unit_directions(16, 5, seed=7)
    assert dirs.shape == (5, 2, 16)
    assert np.allclose(np.linalg.norm(dirs, axis=2), 1.0, atol=1e-6)
    assert np.array_equal(dirs, random_unit_directions(16, 5, seed=7))
    assert not np.array_equal(dirs, random_unit_directions(16, 5, seed=8))
    deltas = np.random.default_rng(0).standard_normal((4, 2, 16))
    projected = projections_onto(deltas, dirs)
    expected = projection_record(deltas[2], dirs[3])["proj"]
    assert abs(projected[2, 3] - expected) < 1e-4


def test_residualize_removes_additive_fixed_effects_exactly():
    prompts = np.repeat(np.arange(6), 3)
    positions = np.tile(np.arange(3), 6)
    a = np.array([1.0, -2.0, 0.5, 3.0, 0.0, -1.0])[prompts]
    b = np.array([10.0, 0.0, -5.0])[positions]
    x = a + b
    assert np.allclose(residualize(x, positions), a - a.mean(), atol=1e-10)
    assert np.allclose(residualize(x, prompts, positions), 0.0, atol=1e-10)
    noise = np.random.default_rng(1).standard_normal(18)
    r = residualize(x + noise, prompts, positions)
    assert np.allclose(np.bincount(prompts, weights=r), 0.0, atol=1e-9)
    assert np.allclose(np.bincount(positions, weights=r), 0.0, atol=1e-9)


def test_primary_rho_ignores_a_pure_position_effect_and_permutation_p_is_calibrated():
    rng = np.random.default_rng(3)
    n = 60
    positions = np.tile(np.arange(3), n)
    pos_effect = np.array([0.0, 5.0, 10.0])[positions]
    proj = rng.standard_normal(3 * n) + pos_effect
    dr = rng.standard_normal(3 * n) + 0.3 * pos_effect
    raw = spearman(proj, dr)
    assert raw > 0.5  # the position effect alone makes the raw correlation large
    rho = primary_rho(proj, dr, positions)
    assert abs(rho) < 0.2
    report = permutation_p(proj, dr, positions, n_perm=400, seed=0)
    assert abs(report["rho"] - rho) < 1e-12
    assert report["p_two_sided"] > 0.05
    signal = proj + 2.0 * rng.standard_normal(3 * n)
    report = permutation_p(proj, signal, positions, n_perm=400, seed=0)
    assert report["p_two_sided"] < 0.01 and report["rho"] > 0.3


def test_partial_spearman_removes_a_shared_component():
    rng = np.random.default_rng(5)
    z = rng.standard_normal(300)
    x = z + 0.1 * rng.standard_normal(300)
    y = z + 0.1 * rng.standard_normal(300)
    assert spearman(x, y) > 0.9
    assert abs(partial_spearman(x, y, z)) < 0.3
    w = rng.standard_normal(300)
    assert partial_spearman(x, x + 0.01 * w, w) > 0.95


def test_leakage_mask_and_llama_date_window():
    mask = leakage_free_mask([1, 2, 3, 4], ["bomb", "scam", "scam", "rape"], {2}, {"bomb", "rape"})
    assert mask.tolist() == [False, False, True, False]
    dates = date_candidates("Cutting Knowledge Date: December 2023\nToday Date: 01 Oct 2026\n\n")
    assert "01 Oct 2026" in dates and "02 Oct 2026" in dates and "30 Sep 2026" in dates and len(dates) == 7
    assert date_candidates("<|im_start|>user\n{PROMPT}<|im_end|>") == []
