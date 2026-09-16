"""CPU tests for the identity-free V246 ordinal Phi head."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from lcwm.v246_count_phi import (
    CountPhiConfig,
    CountPhiHead,
    OrdinalTargets,
    masked_ordinal_bce,
    ordinal_targets_from_slots,
)


def tiny_model(input_dim: int = 7) -> CountPhiHead:
    torch.manual_seed(246)
    return CountPhiHead(
        CountPhiConfig(input_dim=input_dim, hidden_dim=11, dropout=0.0)
    )


def test_config_roundtrip_is_sufficient_to_reload_a_state_dict():
    config = CountPhiConfig(input_dim=13, hidden_dim=17, dropout=0.125)
    source = CountPhiHead(config).eval()
    z = torch.randn(2, 13)
    checkpoint = {"config": config.to_dict(), "state_dict": source.state_dict()}

    restored = CountPhiConfig.from_dict(checkpoint["config"])
    assert restored == config
    loaded = CountPhiHead(restored).eval()
    loaded.load_state_dict(checkpoint["state_dict"], strict=True)
    torch.testing.assert_close(loaded(z).logits, source(z).logits, rtol=0, atol=0)

    with pytest.raises(ValueError, match="fields mismatch"):
        CountPhiConfig.from_dict({**config.to_dict(), "extra": 1})
    with pytest.raises(ValueError, match="unsupported"):
        CountPhiConfig.from_dict({**config.to_dict(), "schema": "stale"})


def test_can_probabilities_are_monotone_by_construction():
    model = tiny_model()
    z = 100.0 * torch.randn(128, 7)
    output = model(z)
    assert bool((output.p_can_ge2 <= output.p_can_ge1).all())
    assert bool(((0.0 <= output.probabilities) & (output.probabilities <= 1.0)).all())


def test_scalar_is_exactly_the_three_probability_sum():
    output = tiny_model(5)(torch.randn(9, 5))
    expected = output.p_can_ge1 + output.p_can_ge2 + output.p_cream
    torch.testing.assert_close(output.scalar, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        output.scalar, output.probabilities.sum(dim=-1), rtol=0, atol=0
    )


def test_masked_bce_has_finite_nonzero_gradients():
    model = tiny_model()
    z = torch.randn(6, 7, requires_grad=True)
    target = OrdinalTargets(
        values=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        mask=torch.ones(6, 3, dtype=torch.bool),
    )
    loss = masked_ordinal_bce(model(z), target)
    loss.backward()
    assert torch.isfinite(loss)
    assert z.grad is not None and bool(torch.isfinite(z.grad).all())
    assert float(z.grad.abs().sum()) > 0.0
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0.0


def test_masked_labels_do_not_enter_the_loss():
    output = tiny_model()(torch.randn(4, 7))
    values = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    mask = torch.tensor(
        [
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 0, 0],
        ],
        dtype=torch.bool,
    )
    loss = masked_ordinal_bce(output, values, mask)
    manual = F.binary_cross_entropy_with_logits(
        output.logits[mask], values[mask], reduction="mean"
    )
    torch.testing.assert_close(loss, manual)

    changed = values.clone()
    changed[~mask] = 1.0 - changed[~mask]
    torch.testing.assert_close(masked_ordinal_bce(output, changed, mask), loss)

    empty = torch.zeros_like(mask)
    zero = masked_ordinal_bce(output, values, empty)
    torch.testing.assert_close(zero, torch.tensor(0.0), rtol=0, atol=0)


def test_can_slot_permutation_preserves_targets_and_scalar():
    can = torch.tensor(
        [[0, 0], [1, 0], [0, 1], [1, 1], [1, 0]], dtype=torch.float32
    )
    can_observed = torch.tensor(
        [[1, 1], [1, 1], [1, 1], [1, 1], [1, 0]], dtype=torch.bool
    )
    cream = torch.tensor([0, 0, 1, 1, 0], dtype=torch.float32)
    cream_observed = torch.tensor([1, 1, 1, 1, 0], dtype=torch.bool)
    original = ordinal_targets_from_slots(
        can,
        cream,
        can_observed=can_observed,
        cream_observed=cream_observed,
    )
    permuted = ordinal_targets_from_slots(
        can.flip(-1),
        cream,
        can_observed=can_observed.flip(-1),
        cream_observed=cream_observed,
    )
    torch.testing.assert_close(original.values, permuted.values, rtol=0, atol=0)
    assert torch.equal(original.mask, permuted.mask)
    torch.testing.assert_close(
        original.values.sum(dim=-1),
        permuted.values.sum(dim=-1),
        rtol=0,
        atol=0,
    )

    complete = original.mask.all(dim=-1)
    expected_scalar = can.sum(dim=-1) + cream
    torch.testing.assert_close(
        original.values[complete].sum(dim=-1),
        expected_scalar[complete],
        rtol=0,
        atol=0,
    )
    output = tiny_model()(torch.randn(len(can), 7))
    torch.testing.assert_close(output.scalar, output.probabilities.sum(-1), rtol=0, atol=0)


def test_invalid_shapes_and_nonfinite_values_fail_loudly():
    model = tiny_model()
    with pytest.raises(ValueError, match="shape"):
        model(torch.randn(2, 8))
    bad = torch.randn(2, 7)
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        model(bad)

    with pytest.raises(ValueError, match="shape"):
        ordinal_targets_from_slots(torch.zeros(2, 3), torch.zeros(2))
    with pytest.raises(TypeError, match="torch.bool"):
        ordinal_targets_from_slots(
            torch.zeros(2, 2),
            torch.zeros(2),
            can_observed=torch.ones(2, 2),
        )
