"""Mechanical tests for the blind v241 policy-pool scorer."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from score_v241_policy_pool import (  # noqa: E402
    actor_delta_from_raw,
    model_to_raw,
    raw_to_model,
    score_policy,
)
from train_v157_residual_actor import ResidualActor  # noqa: E402
from train_v220_raw_latent_wm import RawLatentWM  # noqa: E402


def actor_info(actor: ResidualActor, mu: torch.Tensor, sd: torch.Tensor,
               c: int, adim: int) -> dict:
    return {
        "actor": actor,
        "mu": mu,
        "sd": sd,
        "checkpoint": {"c": c, "adim": adim},
    }


def test_model_raw_roundtrip() -> None:
    g = torch.Generator().manual_seed(1)
    raw = torch.randn(7, 11, generator=g)
    mu = torch.randn(11, generator=g)
    sd = torch.rand(11, generator=g) + 0.1
    got = model_to_raw(raw_to_model(raw, mu, sd), mu, sd)
    torch.testing.assert_close(got, raw, rtol=1e-6, atol=1e-6)


def test_actor_delta_uses_actor_coordinates() -> None:
    torch.manual_seed(2)
    zdim, c, adim = 11, 2, 3
    actor = ResidualActor(zdim, c, adim, scale=0.04)
    with torch.no_grad():
        actor.net[-1].weight.normal_(0, 0.1)
        actor.net[-1].bias.normal_(0, 0.1)
    raw = torch.randn(5, zdim)
    mu = torch.randn(zdim)
    sd = torch.rand(zdim) + 0.1
    expected = actor((raw - mu) / sd)
    actual = actor_delta_from_raw(raw, actor, mu, sd)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_strict_controls_are_policy_invariant() -> None:
    torch.manual_seed(3)
    zdim, c, adim = 11, 2, 3
    model = RawLatentWM(zdim, c, adim, bdim=16, edim=16, hidden=32)
    prior = torch.nn.Sequential(torch.nn.Linear(zdim, c * adim))
    phi = torch.nn.Sequential(torch.nn.Linear(zdim, 1))
    starts = torch.randn(6, zdim)
    mu, sd = torch.zeros(zdim), torch.ones(zdim)
    a0 = ResidualActor(zdim, c, adim, scale=0.04)
    a1 = ResidualActor(zdim, c, adim, scale=0.04)
    with torch.no_grad():
        a1.net[-1].bias.fill_(1.0)
    i0 = actor_info(a0, mu, sd, c, adim)
    i1 = actor_info(a1, mu, sd, c, adim)

    n0 = score_policy(model, prior, phi, starts, mu, sd, i0,
                      horizon=2, gamma=0.9, use_transition=False)
    n1 = score_policy(model, prior, phi, starts, mu, sd, i1,
                      horizon=2, gamma=0.9, use_transition=False)
    assert n0["score"] == n1["score"]

    b0 = score_policy(model, prior, phi, starts, mu, sd, i0,
                      horizon=2, gamma=0.9, use_residual=False)
    b1 = score_policy(model, prior, phi, starts, mu, sd, i1,
                      horizon=2, gamma=0.9, use_residual=False)
    assert b0["score"] == b1["score"]


def test_identical_models_produce_identical_scores() -> None:
    torch.manual_seed(4)
    zdim, c, adim = 11, 2, 3
    m0 = RawLatentWM(zdim, c, adim, bdim=16, edim=16, hidden=32)
    m1 = RawLatentWM(zdim, c, adim, bdim=16, edim=16, hidden=32)
    m1.load_state_dict(m0.state_dict())
    prior = torch.nn.Sequential(torch.nn.Linear(zdim, c * adim))
    phi = torch.nn.Sequential(torch.nn.Linear(zdim, 1))
    starts = torch.randn(6, zdim)
    mu, sd = torch.zeros(zdim), torch.ones(zdim)
    actor = ResidualActor(zdim, c, adim, scale=0.04)
    info = actor_info(actor, mu, sd, c, adim)
    s0 = score_policy(m0, prior, phi, starts, mu, sd, info, horizon=3, gamma=0.9)
    s1 = score_policy(m1, prior, phi, starts, mu, sd, info, horizon=3, gamma=0.9)
    assert s0 == s1
