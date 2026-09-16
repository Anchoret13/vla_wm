#!/usr/bin/env python
"""CPU tests for the untrained random-residual control arm.

The properties that matter are that it is NOT a no-op (or it would silently duplicate
`base`), that it is bounded by its own scale, that it is reproducible, and that it
lives in the SAME coordinate system as the trained arms.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from build_v252_rand_actor import GAIN, build_rand_actor  # noqa: E402
from train_v157_residual_actor import ResidualActor  # noqa: E402

ZDIM, C, ADIM = 64, 10, 7


def test_the_random_actor_is_not_a_no_op() -> None:
    # ResidualActor zero-inits its output layer; an untouched instance IS the base arm.
    stock = ResidualActor(ZDIM, C, ADIM, scale=0.8)
    z = torch.randn(16, ZDIM)
    assert float(stock(z).abs().max()) == 0.0
    rand = build_rand_actor(ZDIM, C, ADIM, 0.8, init=7)
    assert float(rand(z).abs().mean()) > 0.0


def test_it_respects_its_own_bound() -> None:
    for scale in (0.1, 0.8, 1.6):
        act = build_rand_actor(ZDIM, C, ADIM, scale, init=7)
        z = torch.randn(64, ZDIM) * 5.0          # far outside the data range
        assert float(act(z).abs().max()) <= scale + 1e-6


def test_it_is_reproducible_and_init_separates_actors() -> None:
    a = build_rand_actor(ZDIM, C, ADIM, 0.8, init=7)
    b = build_rand_actor(ZDIM, C, ADIM, 0.8, init=7)
    c = build_rand_actor(ZDIM, C, ADIM, 0.8, init=8)
    z = torch.randn(8, ZDIM)
    assert torch.equal(a(z), b(z))
    assert not torch.equal(a(z), c(z))


def test_it_is_state_conditioned_not_a_constant_bias() -> None:
    # A constant bias would be a different (and weaker) control: the calibration's
    # interior optimum only makes sense for a function of the state.
    act = build_rand_actor(ZDIM, C, ADIM, 0.8, init=7)
    z = torch.randn(32, ZDIM)
    d = act(z)
    per_state = d.flatten(1)
    assert float(per_state.std(0).mean()) > 0.0


def test_the_calibrated_gain_reproduces_the_measured_delta_ratio() -> None:
    # collect_v250/probe_v249 measured |delta| ~ 0.536 * scale on real D0 latents at
    # GAIN = 10. The constant must not drift apart from those collectors.
    assert GAIN == 10.0
    act = build_rand_actor(ZDIM, C, ADIM, 1.0, init=0)
    z = torch.randn(512, ZDIM)
    ratio = float(act(z).abs().mean())
    assert 0.2 < ratio < 0.9        # saturating, but not a constant at the bound


def test_the_actor_output_shape_is_a_chunk() -> None:
    act = build_rand_actor(ZDIM, C, ADIM, 0.8, init=7)
    assert tuple(act(torch.randn(5, ZDIM)).shape) == (5, C, ADIM)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
