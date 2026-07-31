"""CPU tests for grouped LC-Flow history/outcome training semantics."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from lcwm.lc_flow import LCFlowConfig, LCState
import lcwm.lc_flow_train as flow_train
from lcwm.lc_flow_train import (
    branch_outcome_loss,
    config_from_group,
    fit_effect_scales,
    fixed_group_probe,
    outcome_effect_metrics,
    paired_flow_noise_time,
    unroll_lc_history,
)


def tiny_state() -> LCState:
    return LCState(
        LCFlowConfig(
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
            max_atoms=3,
        )
    ).eval()


def make_group() -> dict:
    n = 2
    mask = torch.tensor(
        [[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool
    )
    return {
        "history_prefix_hidden": torch.randn(3, 5, 16),
        "history_prefix_mask": torch.ones(3, 5, dtype=torch.bool),
        "history_action_norm": torch.randn(2, 4, 3),
        "executed_actions_norm": torch.randn(n, 4, 3) * mask[:, :, None],
        "execution_mask": mask,
        "targets": {
            "d_q": torch.zeros(n, 5),
            "d_obj": torch.zeros(n, 2, 3),
            "object_mask": torch.ones(n, 2, dtype=torch.bool),
            "next_bits": torch.zeros(n, 3),
            "atom_mask": torch.ones(n, 3, dtype=torch.bool),
            "d_prog": torch.zeros(n),
            "continuation_success": torch.zeros(n),
            "continuation_mask": torch.ones(n),
        },
    }


def test_history_unroll_and_masked_branch_outcome_are_finite():
    torch.manual_seed(3)
    model = tiny_state()
    group = make_group()
    z_t = unroll_lc_history(model, group, device="cpu", truncate_bptt=2)
    assert z_t.shape == (1, 2, 12)
    loss, parts = branch_outcome_loss(model, z_t, group, device="cpu")
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in parts.values())


def test_history_truncation_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        unroll_lc_history(
            tiny_state(), make_group(), device="cpu", truncate_bptt=0
        )


def test_short_history_is_not_detached_and_train_unroll_is_deterministic():
    torch.manual_seed(7)
    model = tiny_state().train()
    group = make_group()
    first = unroll_lc_history(
        model, group, device="cpu", truncate_bptt=16
    )
    second = unroll_lc_history(
        model, group, device="cpu", truncate_bptt=16
    )
    torch.testing.assert_close(first, second, rtol=0, atol=0)

    first.square().mean().backward()
    assert model.z0.grad is not None
    assert float(model.z0.grad.abs().sum()) > 0


def test_paired_flow_randomness_is_reproducible():
    class Config:
        chunk_size = 4
        max_action_dim = 6

    class Model:
        @staticmethod
        def sample_time(batch, device):
            return torch.distributions.Beta(1.5, 1.0).sample((batch,)).to(
                device
            )

    class Policy:
        config = Config()
        model = Model()

    noise_a, time_a = paired_flow_noise_time(
        Policy(), seed=10, device="cpu"
    )
    noise_b, time_b = paired_flow_noise_time(
        Policy(), seed=10, device="cpu"
    )
    torch.testing.assert_close(noise_a, noise_b, rtol=0, atol=0)
    torch.testing.assert_close(time_a, time_b, rtol=0, atol=0)


def test_config_binds_data_facing_dimensions():
    group = make_group()
    group.update(
        {
            "actions_norm": torch.zeros(2, 50, 3),
            "q_before": torch.zeros(5),
            "object_pos_before": torch.zeros(2, 3),
            "bits_before": torch.zeros(3, dtype=torch.bool),
        }
    )
    config = config_from_group(group, expert_width=9)
    assert config.d_h == 16
    assert config.action_dim == 3
    assert config.execution_horizon == 4
    assert config.proprio_dim == 5
    assert config.max_objects == 2
    assert config.max_atoms == 3
    assert config.expert_width == 9


def test_effect_metrics_remove_shared_prediction_offset_and_mask_objects():
    d_q = torch.tensor(
        [[1.0, 0.0], [2.0, 0.0], [4.0, 0.0]]
    )
    d_obj = torch.zeros(3, 2, 3)
    d_obj[:, 0, 0] = torch.tensor([0.0, 1.0, 3.0])
    d_obj[:, 1] = 10_000.0
    d_prog = torch.tensor([0.0, 1.0, 2.0])
    targets = {
        "d_q": d_q,
        "d_obj": d_obj,
        "object_mask": torch.tensor(
            [[1, 0], [1, 0], [1, 0]], dtype=torch.bool
        ),
        "d_prog": d_prog,
    }
    predictions = {
        "d_q": d_q + 5.0,
        "d_obj": d_obj + 7.0,
        "d_prog": d_prog + 3.0,
    }
    metrics = outcome_effect_metrics(predictions, targets)

    for key in ("d_q", "d_obj", "d_prog"):
        torch.testing.assert_close(
            metrics[f"{key}_centered_model_mse"], torch.tensor(0.0)
        )
        torch.testing.assert_close(
            metrics[f"{key}_centered_cosine"], torch.tensor(1.0)
        )
        assert float(metrics[f"{key}_matched_over_permutation"]) < 1e-12
        assert float(metrics[f"{key}_group_mean_mse"]) > 0
    # The masked object's huge target never enters either object metric.
    assert float(metrics["d_obj_raw_zero_mse"]) < 2.0


def test_effect_metrics_reject_branch_varying_masks():
    predictions = {
        "d_q": torch.zeros(2, 2),
        "d_obj": torch.zeros(2, 2, 3),
        "d_prog": torch.zeros(2),
    }
    targets = {
        "d_q": torch.zeros(2, 2),
        "d_obj": torch.zeros(2, 2, 3),
        "object_mask": torch.tensor([[1, 0], [0, 1]], dtype=torch.bool),
        "d_prog": torch.zeros(2),
    }
    with pytest.raises(ValueError, match="branch-constant mask"):
        outcome_effect_metrics(predictions, targets)


def test_effect_scale_fit_weights_source_episodes_not_duplicate_groups():
    def scale_group(source: str, amplitude: float) -> dict:
        q = torch.stack(
            [torch.zeros(9), torch.full((9,), amplitude)]
        )
        d_obj = torch.stack(
            [torch.zeros(2, 3), torch.full((2, 3), amplitude)]
        )
        d_prog = torch.tensor([0.0, amplitude])
        return {
            "source_trajectory_id": source,
            "targets": {
                "d_q": q,
                "d_obj": d_obj,
                "object_mask": torch.ones(2, 2, dtype=torch.bool),
                "d_prog": d_prog,
            },
            "restore_qc": {
                "records": [
                    {
                        "q": torch.zeros(9),
                        "object_pos": torch.zeros(2, 3),
                    },
                    {
                        "q": torch.zeros(9),
                        "object_pos": torch.zeros(2, 3),
                    },
                ]
            },
        }

    groups = [scale_group("source-a", 2.0)] + [
        scale_group("source-b", 0.0) for _ in range(3)
    ]
    scales, report = fit_effect_scales(groups)
    expected = 0.5**0.5
    assert scales.q_position == pytest.approx(expected)
    assert scales.q_quaternion == pytest.approx(expected)
    assert scales.q_gripper == pytest.approx(expected)
    assert scales.d_obj == pytest.approx(expected)
    assert scales.d_prog == pytest.approx(expected)
    assert report["source_episode_groups"] == {
        "source-a": 1,
        "source-b": 3,
    }


def test_fixed_probe_is_deterministic_weighted_and_reports_executed_shift(
    monkeypatch,
):
    torch.manual_seed(11)
    model = tiny_state().train()
    group = make_group()
    group.update(
        {
            "history_prefix_hidden": group["history_prefix_hidden"][:1],
            "history_prefix_mask": group["history_prefix_mask"][:1],
            "history_action_norm": group["history_action_norm"][:0],
            "actions_norm": torch.stack(
                [
                    torch.ones(6, 3),
                    torch.full((6, 3), 3.0),
                ]
            ),
            "executed_lengths": torch.tensor([4, 4]),
            "branch_weights": torch.tensor([1.0, 3.0]),
            "current_observation": {},
            "full_instruction": "test task",
        }
    )
    prefix = SimpleNamespace(
        hidden=group["history_prefix_hidden"],
        pad_masks=group["history_prefix_mask"],
        past_key_values=(),
    )
    policy = SimpleNamespace(
        config=SimpleNamespace(chunk_size=6, max_action_dim=5),
    )
    runner = SimpleNamespace(
        policy=policy,
        _obs_to_policy_batch=lambda observation, instruction: {},
    )

    monkeypatch.setattr(flow_train, "prefix_forward", lambda policy, batch: prefix)
    monkeypatch.setattr(
        flow_train,
        "paired_flow_noise_time",
        lambda policy, seed, device: (
            torch.zeros(1, 6, 5),
            torch.full((1,), 0.5),
        ),
    )

    def fake_cached_loss(
        policy,
        prefix,
        actions,
        bias,
        weights,
        **kwargs,
    ):
        return actions.mean(), {}

    monkeypatch.setattr(
        flow_train, "cached_branch_flow_loss", fake_cached_loss
    )
    monkeypatch.setattr(
        flow_train,
        "sample_chunks",
        lambda *args, **kwargs: torch.zeros(1, 6, 3),
    )

    def fake_lc_chunks(*args, **kwargs):
        shifted = torch.full((1, 6, 3), 100.0)
        shifted[:, :4] = 2.0
        return shifted

    monkeypatch.setattr(flow_train, "sample_chunks_lc", fake_lc_chunks)

    first = fixed_group_probe(
        runner, model, group, device="cpu", seed=23
    )
    second = fixed_group_probe(
        runner, model, group, device="cpu", seed=23
    )

    assert first == second
    assert model.training
    assert first["flow_loss"] == pytest.approx(2.5)
    assert first["action_shift_exec_mean_abs"] == pytest.approx(2.0)
    assert first["action_shift_exec_max_abs"] == pytest.approx(2.0)
    assert first["action_shift_full_mean_abs"] > 2.0
