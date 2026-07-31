"""CPU unit tests for LC-Flow state, reduction, and adapter checkpoints."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

from lcwm.lc_flow import (
    EffectScales,
    LCFlowConfig,
    LCState,
    centered_effect_loss,
    freeze_pi05_base,
    load_lc_adapter,
    masked_branch_flow_loss,
    outcome_loss,
    save_lc_adapter,
)


def tiny_config() -> LCFlowConfig:
    return LCFlowConfig(
        d_h=16,
        d_z=12,
        state_tokens=2,
        heads=3,
        transition_layers=1,
        update_blocks=1,
        expert_width=8,
        action_dim=3,
        execution_horizon=4,
        proprio_dim=5,
        max_objects=2,
        max_atoms=5,
    )


def test_prefix_padding_is_invariant_and_chain5_shape_is_supported():
    torch.manual_seed(0)
    model = LCState(tiny_config()).eval()
    h = torch.randn(2, 7, 16)
    mask = torch.tensor(
        [[1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0]],
        dtype=torch.bool,
    )
    changed = h.clone()
    changed[~mask] = 10_000 * torch.randn_like(changed[~mask])

    z = model.posterior(h, mask)
    z_changed = model.posterior(changed, mask)
    torch.testing.assert_close(z, z_changed, rtol=0, atol=0)

    actions = torch.randn(2, 4, 3)
    future = model.transition(z, actions)
    out = model.outcome(future)
    assert out["d_q"].shape == (2, 5)
    assert out["d_obj"].shape == (2, 2, 3)
    assert out["next_bits_logits"].shape == (2, 5)


def test_branch_reduction_ignores_unexecuted_tail_loss_elements():
    raw = torch.arange(2 * 8 * 3, dtype=torch.float32).reshape(2, 8, 3)
    weights = torch.tensor([1.0, 2.0])
    lengths = torch.tensor([4, 2])
    loss, per = masked_branch_flow_loss(
        raw, weights, executed_lengths=lengths, max_executed=4
    )

    changed = raw.clone()
    changed[0, 4:] += 1e6
    changed[1, 2:] += 1e6
    loss_changed, per_changed = masked_branch_flow_loss(
        changed, weights, executed_lengths=lengths, max_executed=4
    )
    torch.testing.assert_close(per, per_changed, rtol=0, atol=0)
    torch.testing.assert_close(loss, loss_changed, rtol=0, atol=0)

    with pytest.raises(ValueError, match="nonnegative"):
        masked_branch_flow_loss(raw, torch.tensor([1.0, -1.0]))


def test_zero_weight_batch_returns_differentiable_zero():
    raw = torch.randn(2, 5, 3, requires_grad=True)
    loss, _ = masked_branch_flow_loss(raw, torch.zeros(2), max_executed=3)
    loss.backward()
    assert float(loss.detach()) == 0.0
    assert raw.grad is not None
    assert float(raw.grad.abs().max()) == 0.0


def test_outcome_loss_masks_padded_objects_and_atoms():
    predictions = {
        "d_q": torch.zeros(1, 5),
        "d_obj": torch.zeros(1, 2, 3),
        "next_bits_logits": torch.zeros(1, 5),
        "d_prog": torch.zeros(1),
        "ret": torch.zeros(1),
    }
    targets = {
        "d_q": torch.ones(1, 5),
        "d_obj": torch.tensor([[[1.0, 1.0, 1.0], [1e4, 1e4, 1e4]]]),
        "object_mask": torch.tensor([[1, 0]]),
        "next_bits": torch.tensor([[1, 0, 1, 1, 1]]),
        "atom_mask": torch.tensor([[1, 1, 1, 0, 0]]),
        "d_prog": torch.ones(1),
        "continuation_success": torch.ones(1),
    }
    total, parts = outcome_loss(predictions, targets)
    assert torch.isfinite(total)
    torch.testing.assert_close(parts["d_obj"], torch.tensor(1.0))
    expected_bce = torch.tensor(0.69314718)
    torch.testing.assert_close(parts["next_bits"], expected_bce)
    torch.testing.assert_close(parts["continuation_success"], expected_bce)


def test_centered_effect_loss_ignores_common_mode_but_not_action_mismatch():
    scales = EffectScales(
        q_position=1.0,
        q_quaternion=1.0,
        q_gripper=1.0,
        d_obj=1.0,
        d_prog=1.0,
    )
    targets = {
        "d_q": torch.arange(27, dtype=torch.float32).reshape(3, 9),
        "d_obj": torch.arange(18, dtype=torch.float32).reshape(3, 2, 3),
        "object_mask": torch.tensor(
            [[1, 0], [1, 0], [1, 0]], dtype=torch.bool
        ),
        "d_prog": torch.tensor([0.0, 1.0, 3.0]),
    }
    predictions = {
        "d_q": targets["d_q"] + 5.0,
        "d_obj": targets["d_obj"] - 7.0,
        "d_prog": targets["d_prog"] + 11.0,
    }
    loss, parts = centered_effect_loss(predictions, targets, scales)
    torch.testing.assert_close(loss, torch.tensor(0.0), rtol=0, atol=1e-12)
    assert all(abs(float(value)) < 1e-12 for value in parts.values())

    permuted = {
        key: value.roll(1, 0) if torch.is_tensor(value) else value
        for key, value in predictions.items()
    }
    mismatch, _ = centered_effect_loss(permuted, targets, scales)
    assert float(mismatch) > 0


def test_adapter_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(1)
    config = tiny_config()
    model = LCState(config)
    with torch.no_grad():
        model.wz_out.bias.fill_(0.25)
    save_lc_adapter(model, tmp_path, metadata={"base_model": "stock-test"})
    loaded, metadata = load_lc_adapter(tmp_path)
    assert loaded.config == config
    assert metadata == {"base_model": "stock-test"}
    for expected, actual in zip(model.state_dict().values(), loaded.state_dict().values()):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)


def test_freeze_pi05_base_is_explicit():
    base = nn.Sequential(nn.Linear(3, 4), nn.Dropout())
    freeze_pi05_base(base)
    assert not base.training
    assert not any(parameter.requires_grad for parameter in base.parameters())


def test_invalid_action_block_shape_fails_loudly():
    model = LCState(replace(tiny_config(), execution_horizon=4))
    z = model.initial(1, "cpu")
    with pytest.raises(ValueError, match="action block tail"):
        model.transition(z, torch.zeros(1, 3, 3))


def test_recurrent_state_remains_normalized_over_long_horizon():
    torch.manual_seed(9)
    model = LCState(tiny_config()).train()
    hidden = torch.randn(1, 7, 16)
    mask = torch.ones(1, 7, dtype=torch.bool)
    actions = torch.randn(1, 4, 3)
    z = model.posterior(hidden, mask)
    for _ in range(64):
        z = model.step(z, actions, hidden, mask)
        assert torch.isfinite(z).all()
    rms = float(z.detach().square().mean().sqrt())
    assert 0.8 < rms < 1.2


def test_transition_ignores_unexecuted_action_tail():
    torch.manual_seed(4)
    model = LCState(tiny_config()).eval()
    z = model.initial(2, "cpu")
    actions = torch.randn(2, 4, 3)
    mask = torch.tensor(
        [[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool
    )
    changed = actions.clone()
    changed[~mask] = 10_000 * torch.randn_like(changed[~mask])
    expected = model.transition(z, actions, mask)
    actual = model.transition(z, changed, mask)
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
