"""CPU-only tests for branch collection selection and storage helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lcwm.branch_collect import (
    _best_recovery_state_index,
    _clone_observation,
    _continuation_indices,
    _recovery_candidate_feasible,
    advance_recovery_context,
)
from scripts.collect_chain3_pilot import (
    branch_coverage_summary,
    contract_sha256,
    seed_collection_contract,
)


def test_continuation_subset_covers_stock_and_prompt_families():
    families = [
        "stock_full",
        "full",
        "full",
        "exact_pair",
        "remaining_atomic",
        "remaining_atomic",
    ]
    assert _continuation_indices(families, 4) == [0, 1, 3, 4]
    assert _continuation_indices(families, 0) == []


def test_saved_observation_is_detached_from_mutable_buffers():
    original = {
        "array": np.ones((2, 2)),
        "tensor": torch.ones(2),
        "nested": [{"value": np.ones(1)}],
    }
    cloned = _clone_observation(original)
    original["array"][0, 0] = 9
    original["tensor"][0] = 9
    original["nested"][0]["value"][0] = 9
    assert float(cloned["array"][0, 0]) == 1
    assert float(cloned["tensor"][0]) == 1
    assert float(cloned["nested"][0]["value"][0]) == 1


def test_recovery_state_selection_keeps_best_state_not_route_endpoint():
    records = [
        {
            "target_success": False,
            "target_displacement": 0.0,
            "eef_distance": 0.12,
        },
        {
            "target_success": False,
            "target_displacement": 0.0,
            "eef_distance": 0.40,
        },
    ]
    assert _best_recovery_state_index(records) == 0

    records[1]["target_displacement"] = 0.003
    assert _best_recovery_state_index(records) == 1


def test_recovery_accepts_clean_terminal_goal_success():
    assert _recovery_candidate_feasible(
        executed=7,
        execution_horizon=10,
        no_damage=True,
        target_success=True,
        terminated=True,
        truncated=False,
    )
    assert not _recovery_candidate_feasible(
        executed=7,
        execution_horizon=10,
        no_damage=True,
        target_success=False,
        terminated=True,
        truncated=False,
    )
    assert _recovery_candidate_feasible(
        executed=10,
        execution_horizon=10,
        no_damage=False,
        target_success=True,
        terminated=False,
        truncated=False,
    )


def test_negative_recovery_warmup_fails_before_touching_runtime():
    with pytest.raises(ValueError, match="nonnegative"):
        advance_recovery_context(
            None,
            None,
            None,
            "remaining task",
            blocks=-1,
            behavior_noise_seed=0,
            full_reference_noise_seed=1,
        )


def test_empty_recovery_prompt_id_fails_before_touching_runtime():
    with pytest.raises(ValueError, match="behavior_prompt_id"):
        advance_recovery_context(
            None,
            None,
            None,
            "support pair task",
            behavior_prompt_id="",
            blocks=1,
            behavior_noise_seed=0,
            full_reference_noise_seed=1,
        )


def test_collection_contract_fingerprint_covers_behavioral_parameters():
    args = SimpleNamespace(
        full_count=4,
        pair_count=4,
        remaining_count=4,
        execution_horizon=10,
        recovery_warmup_blocks=0,
        recovery_anchor_blocks=16,
        recovery_candidates=4,
        recovery_stop_distance=0.10,
        recovery_anchor_stop_distance=0.16,
        recovery_anchor_admission_distance=0.25,
        continuation_horizon=160,
        continuation_count=4,
        source_attempts=4,
        source_attempt_stride=100,
        source_noise_seed=20_000,
        proposal_noise_seed=30_000,
        continuation_noise_seed=40_000,
        recovery_noise_seed=50_000,
        recovery_anchor_noise_seed=60_000,
    )
    first = seed_collection_contract(
        args,
        1001,
        base_model_id="pi05-test",
        num_inference_steps=10,
    )
    repeated = seed_collection_contract(
        args,
        1001,
        base_model_id="pi05-test",
        num_inference_steps=10,
    )
    assert contract_sha256(first) == contract_sha256(repeated)

    changed_args = SimpleNamespace(
        **{**vars(args), "continuation_horizon": 80}
    )
    changed = seed_collection_contract(
        changed_args,
        1001,
        base_model_id="pi05-test",
        num_inference_steps=10,
    )
    assert contract_sha256(first) != contract_sha256(changed)


def test_branch_coverage_distinguishes_effect_and_eligibility():
    group = {
        "object_pos_before": torch.zeros(1, 3),
        "object_pos_after": torch.tensor(
            [[[0.0, 0.0, 0.0]], [[0.001, 0.0, 0.0]]]
        ),
        "atom_mask": torch.tensor([1, 1, 0], dtype=torch.bool),
        "bits_before": torch.tensor([1, 0, 0], dtype=torch.bool),
        "bits_after": torch.tensor(
            [[1, 0, 0], [1, 1, 0]], dtype=torch.bool
        ),
        "dense_progress_margin": 5e-4,
        "dense_goal_progress": torch.tensor([0.0, 0.001]),
        "immediate_beats_stock": torch.tensor([0, 1], dtype=torch.bool),
        "branch_weights": torch.tensor([0.0, 1.0]),
        "actions_norm": torch.zeros(2, 50, 7),
    }
    assert branch_coverage_summary(group) == {
        "branches": 2,
        "object_effectful_branches": 1,
        "predicate_flip_branches": 1,
        "absolute_dense_progress_branches": 1,
        "relative_superior_branches": 1,
        "positive_flow_targets": 1,
    }
