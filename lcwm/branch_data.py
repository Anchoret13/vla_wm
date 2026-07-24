"""Grouped crossed-branch records for the LC-Flow chain3 vertical slice.

One dataset item is one simulator snapshot with all sibling action branches.
Siblings are never flattened across dataset splits. Physical outcomes are stored
once; task-relative targets are derived without duplicating observations.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

SCHEMA = "lcwm_chain_branch_v1_1"
MANIFEST = "branch_manifest.json"


def validate_branch_group(group: dict[str, Any]) -> None:
    required = {
        "schema",
        "snapshot_id",
        "source_trajectory_id",
        "decision_index",
        "evaluation_goal_id",
        "full_instruction",
        "goal_specs",
        "history_prefix_hidden",
        "history_prefix_mask",
        "history_action_norm",
        "history_behavior_prompt_id",
        "recovery_trace",
        "actions_norm",
        "actions_env",
        "executed_actions_norm",
        "executed_actions_env",
        "execution_mask",
        "executed_lengths",
        "proposal_prompt_id",
        "proposal_family",
        "proposal_goal_id",
        "proposal_candidate_index",
        "proposal_instruction",
        "is_stock_reference",
        "flow_noise_id",
        "flow_noise_seed",
        "flow_noise",
        "q_before",
        "q_after",
        "object_pos_before",
        "object_pos_after",
        "object_mask",
        "bits_before",
        "bits_after",
        "atom_mask",
        "next_observation",
        "current_observation",
        "next_prefix_hidden_full",
        "next_prefix_mask_full",
        "next_state_flat",
        "terminated",
        "truncated",
        "info_success",
        "atomic_continuation_success",
        "continuation_success",
        "continuation_steps",
        "continuation_mask",
        "continuation_final_bits",
        "continuation_policy_id",
        "continuation_noise_ids",
        "branch_reward",
        "branch_weights",
        "dense_goal_object_body_name",
        "dense_goal_receptacle_body_name",
        "dense_goal_object_index",
        "dense_goal_receptacle_index",
        "dense_goal_distance_before",
        "dense_goal_distance_after",
        "dense_goal_progress",
        "dense_progress_margin",
        "restore_qc",
        "provenance",
        "action_convention",
    }
    missing = required.difference(group)
    if missing:
        raise KeyError(f"branch group missing keys: {sorted(missing)}")
    if group["schema"] != SCHEMA:
        raise ValueError(f"unexpected branch schema {group['schema']!r}")

    actions = group["actions_norm"]
    if actions.ndim != 3:
        raise ValueError(f"actions_norm must be [N,T,A], got {actions.shape}")
    n, steps, action_dim = actions.shape
    if group["actions_env"].shape != actions.shape:
        raise ValueError("normalized and environment action chunks differ in shape")
    horizon = group["executed_actions_norm"].shape[1]
    if horizon != 10:
        raise ValueError("branch schema v1.1 requires a 10-action horizon")
    if group["executed_actions_norm"].shape != (n, horizon, action_dim):
        raise ValueError("executed_actions_norm must be [N,H,A]")
    if group["executed_actions_env"].shape != (n, horizon, action_dim):
        raise ValueError("executed_actions_env must be [N,H,A]")
    if group["execution_mask"].shape != (n, horizon):
        raise ValueError("execution_mask must be [N,H]")
    if group["executed_lengths"].shape != (n,):
        raise ValueError("executed_lengths must have one entry per branch")
    if bool((group["executed_lengths"] < 0).any()) or bool(
        (group["executed_lengths"] > horizon).any()
    ):
        raise ValueError("executed_lengths outside action chunk")
    expected_execution_mask = (
        torch.arange(horizon)[None] < group["executed_lengths"][:, None]
    )
    if not torch.equal(group["execution_mask"].cpu(), expected_execution_mask):
        raise ValueError("execution_mask does not match executed_lengths")

    for key in (
        "proposal_prompt_id",
        "proposal_family",
        "proposal_goal_id",
        "proposal_candidate_index",
        "proposal_instruction",
        "flow_noise_id",
        "flow_noise_seed",
        "next_observation",
        "continuation_noise_ids",
    ):
        if len(group[key]) != n:
            raise ValueError(f"{key} must have one entry per branch")
    if group["is_stock_reference"].shape != (n,):
        raise ValueError("is_stock_reference must be [N]")
    if int(group["is_stock_reference"].sum()) != 1 or not bool(
        group["is_stock_reference"][0]
    ):
        raise ValueError("branch zero must be the unique stock reference")
    if (
        group["proposal_prompt_id"][0] != "stock_full"
        or group["proposal_family"][0] != "stock_full"
    ):
        raise ValueError("branch zero must carry stock_full proposal provenance")
    if any(
        prompt == "stock_full" or family == "stock_full"
        for prompt, family in zip(
            group["proposal_prompt_id"][1:],
            group["proposal_family"][1:],
            strict=True,
        )
    ):
        raise ValueError("stock_full proposal provenance must be unique")
    if group["flow_noise"].ndim != 3 or group["flow_noise"].shape[:2] != (
        n,
        steps,
    ):
        raise ValueError("flow_noise must be [N,T,PI05_INTERNAL_ACTION_DIM]")
    expected_noise_dim = group["provenance"].get("max_action_dim")
    if expected_noise_dim is not None and group["flow_noise"].shape[-1] != int(
        expected_noise_dim
    ):
        raise ValueError("flow_noise internal dimension disagrees with provenance")
    expected_norm = (
        actions[:, :horizon] * group["execution_mask"][:, :, None]
    )
    expected_env = (
        group["actions_env"][:, :horizon]
        * group["execution_mask"][:, :, None]
    )
    if not torch.equal(group["executed_actions_norm"], expected_norm):
        raise ValueError("executed_actions_norm disagrees with proposal/mask")
    if not torch.equal(group["executed_actions_env"], expected_env):
        raise ValueError("executed_actions_env disagrees with proposal/mask")
    if group["q_after"].shape != (n, group["q_before"].numel()):
        raise ValueError("q_after shape does not match q_before")
    if group["object_pos_after"].shape != (
        n,
        *group["object_pos_before"].shape,
    ):
        raise ValueError("object_pos_after shape does not match object_pos_before")
    if group["object_mask"].shape != group["object_pos_before"].shape[:1]:
        raise ValueError("object_mask must be [O]")
    if group["bits_after"].shape != (n, group["bits_before"].numel()):
        raise ValueError("bits_after shape does not match bits_before")
    if group["atom_mask"].shape != group["bits_before"].shape:
        raise ValueError("atom_mask must match bits_before")
    if group["next_prefix_hidden_full"].ndim != 3:
        raise ValueError("next_prefix_hidden_full must be [N,P,D]")
    if group["next_prefix_hidden_full"].shape[0] != n:
        raise ValueError("next prefix batch must match branches")
    if (
        group["next_prefix_hidden_full"].shape[-1]
        != group["history_prefix_hidden"].shape[-1]
    ):
        raise ValueError("history and next prefix hidden widths differ")
    if group["next_prefix_mask_full"].shape != (
        group["next_prefix_hidden_full"].shape[:2]
    ):
        raise ValueError("next_prefix_mask_full must be [N,P]")
    if group["next_state_flat"].ndim != 2 or group["next_state_flat"].shape[0] != n:
        raise ValueError("next_state_flat must be [N,S]")
    for key in (
        "terminated",
        "truncated",
        "info_success",
        "atomic_continuation_success",
        "continuation_success",
        "continuation_steps",
        "continuation_mask",
        "branch_reward",
        "branch_weights",
    ):
        if group[key].shape != (n,):
            raise ValueError(f"{key} must be [N]")
    if group["continuation_final_bits"].shape != group["bits_after"].shape:
        raise ValueError("continuation_final_bits must match bits_after")
    unlabelled = ~group["continuation_mask"].bool()
    if bool(group["atomic_continuation_success"][unlabelled].any()) or bool(
        group["continuation_success"][unlabelled].any()
    ):
        raise ValueError("unlabelled continuations cannot carry success targets")
    if bool((group["branch_weights"] < 0).any()):
        raise ValueError("branch_weights must be nonnegative")
    if group["dense_goal_distance_after"].shape != (n,):
        raise ValueError("dense_goal_distance_after must be [N]")
    if group["dense_goal_progress"].shape != (n,):
        raise ValueError("dense_goal_progress must be [N]")
    if group["dense_goal_distance_before"].numel() != 1:
        raise ValueError("dense_goal_distance_before must be scalar")
    if not isinstance(group["dense_goal_object_body_name"], str) or not isinstance(
        group["dense_goal_receptacle_body_name"], str
    ):
        raise ValueError("dense goal body names must be strings")
    if float(group["dense_progress_margin"]) < 0:
        raise ValueError("dense_progress_margin must be nonnegative")
    if not isinstance(group["goal_specs"], dict):
        raise ValueError("goal_specs must be a mapping")
    if group["evaluation_goal_id"] not in group["goal_specs"]:
        raise ValueError("evaluation_goal_id missing from goal_specs")
    unknown_goals = set(group["proposal_goal_id"]).difference(
        group["goal_specs"]
    )
    if unknown_goals:
        raise ValueError(f"proposal goals missing from goal_specs: {unknown_goals}")
    if not isinstance(group["continuation_policy_id"], str):
        raise ValueError("continuation_policy_id must be a string")
    if not isinstance(group["restore_qc"], dict):
        raise ValueError("restore_qc must be a mapping")
    if not isinstance(group["restore_qc"].get("valid"), bool):
        raise ValueError("restore_qc.valid must be a boolean")
    if not isinstance(group["current_observation"], dict):
        raise ValueError("current_observation must be a mapping")
    if not isinstance(group["provenance"], dict):
        raise ValueError("provenance must be a mapping")
    if not isinstance(group["action_convention"], dict):
        raise ValueError("action_convention must be a mapping")

    hidden = group["history_prefix_hidden"]
    prefix_mask = group["history_prefix_mask"]
    if hidden.ndim != 3 or prefix_mask.shape != hidden.shape[:2]:
        raise ValueError("history prefix tensors must be [L,P,D] and [L,P]")
    expected_history_actions = max(hidden.shape[0] - 1, 0)
    if group["history_action_norm"].shape != (
        expected_history_actions,
        min(10, steps),
        action_dim,
    ):
        raise ValueError(
            "history_action_norm must connect consecutive prefix states"
        )
    if len(group["history_behavior_prompt_id"]) != expected_history_actions:
        raise ValueError(
            "history_behavior_prompt_id must label each history action block"
        )
    if not isinstance(group["recovery_trace"], list):
        raise ValueError("recovery_trace must be a list")


def derive_branch_targets(group: dict[str, Any]) -> dict[str, Tensor]:
    """Derive outcome-head targets from one physical branch matrix."""
    validate_branch_group(group)
    bits_before = group["bits_before"].to(torch.float32)
    bits_after = group["bits_after"].to(torch.float32)
    atom_mask = group["atom_mask"].to(torch.float32)
    denom = atom_mask.sum().clamp_min(1)
    progress_before = (bits_before * atom_mask).sum() / denom
    progress_after = (bits_after * atom_mask[None]).sum(1) / denom
    return {
        "d_q": group["q_after"] - group["q_before"][None],
        "d_obj": (
            group["object_pos_after"] - group["object_pos_before"][None]
        ),
        "object_mask": group["object_mask"][None].expand(
            group["actions_norm"].shape[0], -1
        ),
        "next_bits": bits_after,
        "atom_mask": atom_mask[None].expand(bits_after.shape[0], -1),
        "predicate_d_prog": progress_after - progress_before,
        "d_prog": (
            group["dense_goal_progress"].to(torch.float32)
            / group["dense_goal_distance_before"].to(torch.float32).clamp_min(
                1e-3
            )
        ),
        "continuation_success": group["continuation_success"].to(torch.float32),
        "continuation_mask": group["continuation_mask"].to(torch.float32),
        "action_mask": group["execution_mask"].to(torch.bool),
    }


def positive_advantage_weights(
    positive: Tensor,
    score: Tensor,
    *,
    stock_index: int = 0,
    max_advantage: float = 1.0,
) -> Tensor:
    """Nonnegative action-imitation weights; failures remain outcome-only."""
    if positive.ndim != 1 or score.shape != positive.shape:
        raise ValueError("positive and score must be same-shape [N] tensors")
    if not 0 <= stock_index < positive.numel():
        raise ValueError("stock_index outside branch set")
    advantage = (score - score[stock_index]).clamp(
        min=0, max=max_advantage
    )
    return positive.to(score.dtype) * (1.0 + advantage)


def save_branch_group(group: dict[str, Any], path: str | Path) -> None:
    """Validate and atomically publish one group.

    Collection jobs can be interrupted after hundreds of GPU calls.  Writing
    through a same-directory temporary prevents a partial ``.pt`` file from
    later being mistaken for an already collected group.
    """
    validate_branch_group(group)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        torch.save(group, temporary)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_branch_group(path: str | Path) -> dict[str, Any]:
    """Load and validate a fully published grouped artifact."""
    group = torch.load(Path(path), weights_only=False)
    validate_branch_group(group)
    return group


class BranchGroupDataset(Dataset):
    """Manifest-backed dataset that preserves snapshot sibling groups."""

    def __init__(self, root: str | Path, split: str):
        self.root = Path(root)
        manifest = json.loads((self.root / MANIFEST).read_text())
        if manifest.get("schema") != SCHEMA:
            raise ValueError(f"unexpected manifest schema {manifest.get('schema')}")
        try:
            self.paths = [self.root / name for name in manifest["splits"][split]]
        except KeyError as error:
            raise KeyError(f"unknown branch split {split!r}") from error
        source_splits = manifest.get("source_episode_splits", {})
        path_metadata = manifest.get("path_metadata", {})
        seen_sources: dict[str, str] = {}
        for split_name, names in manifest.get("splits", {}).items():
            for name in names:
                metadata = path_metadata.get(name)
                if metadata is None:
                    continue
                source = metadata["source_trajectory_id"]
                previous = seen_sources.setdefault(source, split_name)
                if previous != split_name:
                    raise ValueError(
                        f"source episode {source!r} crosses {previous}/{split_name}"
                    )
                if source_splits and source_splits.get(source) != split_name:
                    raise ValueError(
                        f"manifest split mismatch for source episode {source!r}"
                    )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        group = load_branch_group(self.paths[index])
        group["targets"] = derive_branch_targets(group)
        return group
