"""Unit tests for grouped branch schema and derived outcome labels."""

from __future__ import annotations

import json

import pytest
import torch

from lcwm.branch_data import (
    MANIFEST,
    SCHEMA,
    BranchGroupDataset,
    derive_branch_targets,
    load_branch_group,
    positive_advantage_weights,
    save_branch_group,
    validate_branch_group,
)


def make_group() -> dict:
    n, chunk, action_dim = 3, 50, 7
    lengths = torch.tensor([10, 8, 10])
    execution_mask = torch.arange(10)[None] < lengths[:, None]
    actions_norm = torch.randn(n, chunk, action_dim)
    actions_env = torch.randn(n, chunk, action_dim)
    return {
        "schema": SCHEMA,
        "snapshot_id": "seed1000-d12",
        "source_trajectory_id": "chain3-full-seed1000",
        "decision_index": 12,
        "evaluation_goal_id": "chain3_full",
        "full_instruction": "put three objects in the basket",
        "goal_specs": {
            "chain3_full": {
                "family": "full",
                "canonical_instruction": "put three objects in the basket",
                "atoms": [["In", "a", "basket"], ["In", "b", "basket"]],
                "atom_indices": [0, 1],
            }
        },
        "history_prefix_hidden": torch.randn(2, 6, 8),
        "history_prefix_mask": torch.ones(2, 6, dtype=torch.bool),
        "history_action_norm": torch.randn(1, 10, action_dim),
        "recovery_trace": [],
        "history_behavior_prompt_id": ["full"],
        "actions_norm": actions_norm,
        "actions_env": actions_env,
        "executed_actions_norm": (
            actions_norm[:, :10] * execution_mask[:, :, None]
        ),
        "executed_actions_env": (
            actions_env[:, :10] * execution_mask[:, :, None]
        ),
        "execution_mask": execution_mask,
        "executed_lengths": lengths,
        "proposal_prompt_id": ["stock_full", "full", "remaining_atomic"],
        "proposal_family": ["stock_full", "full", "remaining_atomic"],
        "proposal_goal_id": [
            "chain3_full",
            "chain3_full",
            "chain3_full",
        ],
        "proposal_candidate_index": torch.tensor([0, 0, 0]),
        "proposal_instruction": ["full", "full", "remaining"],
        "is_stock_reference": torch.tensor([1, 0, 0], dtype=torch.bool),
        "flow_noise_id": [0, 1, 2],
        "flow_noise_seed": [0, 1, 2],
        "flow_noise": torch.randn(n, chunk, 32),
        "q_before": torch.zeros(9),
        "q_after": torch.ones(n, 9),
        "object_pos_before": torch.zeros(2, 3),
        "object_pos_after": torch.ones(n, 2, 3),
        "object_mask": torch.tensor([1, 0], dtype=torch.bool),
        "bits_before": torch.tensor([1, 1, 0, 0, 0], dtype=torch.bool),
        "bits_after": torch.tensor(
            [[1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 0, 1, 0, 0]],
            dtype=torch.bool,
        ),
        "atom_mask": torch.tensor([1, 1, 1, 0, 0], dtype=torch.bool),
        "next_observation": [{"branch": i} for i in range(n)],
        "current_observation": {"frame": torch.zeros(1)},
        "next_prefix_hidden_full": torch.randn(n, 6, 8),
        "next_prefix_mask_full": torch.ones(n, 6, dtype=torch.bool),
        "next_state_flat": torch.randn(n, 20),
        "terminated": torch.tensor([0, 0, 1], dtype=torch.bool),
        "truncated": torch.zeros(n, dtype=torch.bool),
        "info_success": torch.tensor([0, 0, 1], dtype=torch.bool),
        "atomic_continuation_success": torch.tensor(
            [0, 1, 1], dtype=torch.bool
        ),
        "continuation_success": torch.tensor([0, 1, 1], dtype=torch.bool),
        "continuation_steps": torch.tensor([160, 80, 120]),
        "continuation_mask": torch.ones(n, dtype=torch.bool),
        "continuation_final_bits": torch.tensor(
            [[1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 0, 0]],
            dtype=torch.bool,
        ),
        "continuation_policy_id": "pi05/atomic_remaining/fixed_noise",
        "continuation_noise_ids": [[10], [10], [10]],
        "branch_reward": torch.zeros(n),
        "branch_weights": torch.tensor([0.0, 2.0, 1.5]),
        "dense_goal_object_body_name": "cream_cheese_1_main",
        "dense_goal_receptacle_body_name": "basket_1_main",
        "dense_goal_object_index": 0,
        "dense_goal_receptacle_index": 1,
        "dense_goal_distance_before": torch.tensor(0.3),
        "dense_goal_distance_after": torch.tensor([0.3, 0.27, 0.3]),
        "dense_goal_progress": torch.tensor([0.0, 0.03, 0.0]),
        "dense_progress_margin": 5e-4,
        "restore_qc": {"valid": True},
        "provenance": {"max_action_dim": 32},
        "action_convention": {
            "actions_norm": "normalized",
            "actions_env": "environment",
        },
    }


def test_targets_are_physical_differences_and_full_goal_progress():
    group = make_group()
    targets = derive_branch_targets(group)
    torch.testing.assert_close(targets["d_q"], torch.ones(3, 9))
    torch.testing.assert_close(targets["d_obj"], torch.ones(3, 2, 3))
    torch.testing.assert_close(
        targets["predicate_d_prog"], torch.tensor([0.0, 1 / 3, 0.0])
    )
    torch.testing.assert_close(
        targets["d_prog"], torch.tensor([0.0, 0.1, 0.0])
    )
    assert targets["next_bits"].shape == (3, 5)
    assert targets["atom_mask"].shape == (3, 5)
    torch.testing.assert_close(
        targets["action_mask"], group["execution_mask"]
    )


def test_group_dataset_preserves_siblings(tmp_path):
    group = make_group()
    save_branch_group(group, tmp_path / "groups" / "g0.pt")
    (tmp_path / MANIFEST).write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "splits": {
                    "train": ["groups/g0.pt"],
                    "dev": [],
                    "test": [],
                },
                "source_episode_splits": {
                    "chain3-full-seed1000": "train",
                },
                "path_metadata": {
                    "groups/g0.pt": {
                        "source_trajectory_id": "chain3-full-seed1000",
                    }
                },
            }
        )
    )
    dataset = BranchGroupDataset(tmp_path, "train")
    loaded = dataset[0]
    assert loaded["actions_norm"].shape[0] == 3
    assert loaded["snapshot_id"] == "seed1000-d12"
    assert "targets" in loaded


def test_group_publish_is_atomic_and_validated(tmp_path, monkeypatch):
    group = make_group()
    target = tmp_path / "groups" / "g0.pt"
    save_branch_group(group, target)
    loaded = load_branch_group(target)
    assert loaded["snapshot_id"] == group["snapshot_id"]
    assert not list(target.parent.glob(f".{target.name}.*.tmp"))

    failed_target = tmp_path / "groups" / "failed.pt"

    def fail_save(*args, **kwargs):
        raise RuntimeError("interrupted save")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(RuntimeError, match="interrupted"):
        save_branch_group(group, failed_target)
    assert not failed_target.exists()
    assert not list(failed_target.parent.glob(f".{failed_target.name}.*.tmp"))


def test_positive_advantage_weights_never_reverse_imitate():
    positive = torch.tensor([0, 1, 1], dtype=torch.bool)
    score = torch.tensor([0.2, 0.8, 0.1])
    weights = positive_advantage_weights(positive, score)
    torch.testing.assert_close(weights, torch.tensor([0.0, 1.6, 1.0]))
    assert bool((weights >= 0).all())


def test_validation_rejects_negative_weights():
    group = make_group()
    group["branch_weights"][0] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        validate_branch_group(group)


def test_validation_rejects_execution_mask_mismatch():
    group = make_group()
    group["execution_mask"][1, 8] = True
    with pytest.raises(ValueError, match="execution_mask"):
        validate_branch_group(group)


def test_manifest_rejects_source_episode_crossing_splits(tmp_path):
    group = make_group()
    save_branch_group(group, tmp_path / "groups" / "g0.pt")
    save_branch_group(group, tmp_path / "groups" / "g1.pt")
    (tmp_path / MANIFEST).write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "splits": {
                    "train": ["groups/g0.pt"],
                    "dev": ["groups/g1.pt"],
                    "test": [],
                },
                "path_metadata": {
                    "groups/g0.pt": {
                        "source_trajectory_id": "chain3-full-seed1000",
                    },
                    "groups/g1.pt": {
                        "source_trajectory_id": "chain3-full-seed1000",
                    },
                },
            }
        )
    )
    with pytest.raises(ValueError, match="crosses train/dev"):
        BranchGroupDataset(tmp_path, "train")
