"""Grouped LC-Flow training utilities for the first Chain-3 vertical slice."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

import torch
from torch import Tensor

from lcwm.lc_flow import (
    EffectScales,
    LCFlowConfig,
    LCState,
    cached_branch_flow_loss,
    centered_effect_loss,
    outcome_loss,
    sample_chunks_lc,
)
from lcwm.sampler import prefix_forward, sample_chunks


def config_from_group(
    group: dict[str, Any],
    *,
    expert_width: int = 1024,
) -> LCFlowConfig:
    """Bind adapter I/O dimensions to one validated branch group."""
    base = LCFlowConfig()
    return replace(
        base,
        d_h=int(group["history_prefix_hidden"].shape[-1]),
        expert_width=expert_width,
        action_dim=int(group["actions_norm"].shape[-1]),
        execution_horizon=int(group["execution_mask"].shape[1]),
        proprio_dim=int(group["q_before"].numel()),
        max_objects=int(group["object_pos_before"].shape[0]),
        max_atoms=int(group["bits_before"].numel()),
    )


def to_device_tensors(
    values: dict[str, Any],
    device: torch.device | str,
) -> dict[str, Any]:
    """Move only tensor leaves; preserve raw observations and metadata."""
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in values.items()
    }


def unroll_lc_history(
    lc_state: LCState,
    group: dict[str, Any],
    *,
    device: torch.device | str,
    truncate_bptt: int | None = None,
    current_prefix=None,
) -> Tensor:
    """Recover current ``z_t`` from L prefixes and L-1 executed blocks.

    Forward state always starts at the episode beginning.  ``truncate_bptt``
    only detaches the old recurrent graph before the final K transitions; it
    never reinitializes a stall snapshot from ``z0``.
    """
    hidden = group["history_prefix_hidden"]
    masks = group["history_prefix_mask"]
    actions = group["history_action_norm"]
    if hidden.ndim != 3 or masks.shape != hidden.shape[:2]:
        raise ValueError("history must be [L,P,D] with mask [L,P]")
    if actions.shape[0] != max(hidden.shape[0] - 1, 0):
        raise ValueError("history actions must connect consecutive prefixes")

    first_hidden = hidden[0:1].to(device)
    first_mask = masks[0:1].to(device)
    if current_prefix is not None and hidden.shape[0] == 1:
        first_hidden = current_prefix.hidden
        first_mask = current_prefix.pad_masks
    z = lc_state.posterior(first_hidden, first_mask)
    detach_before = None
    if truncate_bptt is not None:
        if truncate_bptt < 1:
            raise ValueError("truncate_bptt must be positive")
        transitions = max(hidden.shape[0] - 1, 0)
        if transitions > truncate_bptt:
            # Detach the old state immediately before the final K recurrent
            # transitions. Short histories remain fully differentiable.
            detach_before = hidden.shape[0] - truncate_bptt
    for index in range(1, hidden.shape[0]):
        if detach_before is not None and index == detach_before:
            z = z.detach()
        h_i = hidden[index : index + 1].to(device)
        mask_i = masks[index : index + 1].to(device)
        if current_prefix is not None and index == hidden.shape[0] - 1:
            h_i = current_prefix.hidden
            mask_i = current_prefix.pad_masks
        z = lc_state.step(
            z,
            actions[index - 1 : index].to(device, dtype=torch.float32),
            h_i,
            mask_i,
        )
    return z


def branch_outcome_predictions(
    lc_state: LCState,
    z_t: Tensor,
    group: dict[str, Any],
    *,
    device: torch.device | str,
) -> dict[str, Tensor]:
    """Predict every sibling's observed one-block effect."""
    actions = group["executed_actions_norm"].to(
        device, dtype=torch.float32
    )
    action_mask = group["execution_mask"].to(device)
    n = actions.shape[0]
    z_next = lc_state.transition(
        z_t.expand(n, -1, -1),
        actions,
        action_mask,
    )
    return lc_state.outcome(z_next)


def branch_outcome_loss(
    lc_state: LCState,
    z_t: Tensor,
    group: dict[str, Any],
    *,
    device: torch.device | str,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Score every sibling's observed one-block effect, failures included."""
    predictions = branch_outcome_predictions(
        lc_state,
        z_t,
        group,
        device=device,
    )
    targets = to_device_tensors(group["targets"], device)
    return outcome_loss(predictions, targets)


def outcome_effect_metrics(
    predictions: dict[str, Tensor],
    targets: dict[str, Tensor],
) -> dict[str, Tensor]:
    """Audit continuous heads against action-agnostic baselines.

    Raw zero-effect MSE detects whether predicting no physical change is already
    stronger than the model.  Branch-mean centering removes the shared
    snapshot/phase component and measures whether predictions preserve sibling
    action differences.  The corresponding target variance is an empirical
    oracle group-mean baseline; it is diagnostic, not a deployable predictor.
    """

    metrics: dict[str, Tensor] = {}
    channel_masks = {
        "d_q": targets.get("q_mask"),
        "d_obj": targets.get("object_mask"),
        "d_prog": targets.get("progress_mask"),
    }

    def masked_mean(values: Tensor, mask: Tensor | None) -> Tensor:
        if mask is None:
            return values.mean()
        expanded = mask.to(device=values.device, dtype=values.dtype)
        while expanded.ndim < values.ndim:
            expanded = expanded.unsqueeze(-1)
        expanded = expanded.expand_as(values)
        return (values * expanded).sum() / expanded.sum().clamp_min(1)

    def masked_flatten(values: Tensor, mask: Tensor | None) -> Tensor:
        if mask is None:
            return values.flatten()
        expanded = mask.to(device=values.device, dtype=torch.bool)
        while expanded.ndim < values.ndim:
            expanded = expanded.unsqueeze(-1)
        return values[expanded.expand_as(values)]

    for key, mask in channel_masks.items():
        prediction = predictions[key]
        target = targets[key].to(
            device=prediction.device, dtype=prediction.dtype
        )
        if prediction.shape != target.shape:
            raise ValueError(
                f"{key} prediction {tuple(prediction.shape)} != "
                f"target {tuple(target.shape)}"
            )
        if prediction.shape[0] < 2:
            raise ValueError("effect metrics require at least two sibling branches")
        if mask is not None and mask.shape[0] == prediction.shape[0]:
            branch_mask = mask.to(device=prediction.device, dtype=torch.bool)
            if not torch.equal(
                branch_mask,
                branch_mask[0:1].expand_as(branch_mask),
            ):
                raise ValueError(
                    f"{key} effect centering requires a branch-constant mask"
                )

        residual = prediction - target
        centered_prediction = prediction - prediction.mean(dim=0, keepdim=True)
        centered_target = target - target.mean(dim=0, keepdim=True)
        centered_residual = centered_prediction - centered_target
        permutation_losses = []
        for shift in range(1, prediction.shape[0]):
            permutation_residual = (
                centered_prediction.roll(shifts=shift, dims=0)
                - centered_target
            )
            permutation_losses.append(
                masked_mean(permutation_residual.square(), mask)
            )
        permutation_mse = torch.stack(permutation_losses).mean()
        permutation_valid = permutation_mse.detach() > 0
        prediction_flat = masked_flatten(centered_prediction, mask)
        target_flat = masked_flatten(centered_target, mask)
        cosine = torch.zeros((), device=prediction.device)
        denominator = prediction_flat.norm() * target_flat.norm()
        cosine_valid = denominator.detach() > 0
        if bool(cosine_valid):
            cosine = torch.dot(prediction_flat, target_flat) / denominator

        metrics[f"{key}_raw_model_mse"] = masked_mean(
            residual.square(), mask
        )
        metrics[f"{key}_raw_zero_mse"] = masked_mean(
            target.square(), mask
        )
        metrics[f"{key}_centered_model_mse"] = masked_mean(
            centered_residual.square(), mask
        )
        metrics[f"{key}_group_mean_mse"] = masked_mean(
            centered_target.square(), mask
        )
        metrics[f"{key}_centered_permutation_mse"] = permutation_mse
        metrics[f"{key}_matched_over_permutation"] = (
            metrics[f"{key}_centered_model_mse"]
            / permutation_mse.clamp_min(torch.finfo(permutation_mse.dtype).tiny)
        )
        metrics[f"{key}_permutation_valid"] = permutation_valid.to(
            dtype=prediction.dtype
        )
        metrics[f"{key}_centered_cosine"] = cosine
        metrics[f"{key}_centered_cosine_valid"] = cosine_valid.to(
            dtype=prediction.dtype
        )
    return metrics


def fit_effect_scales(
    groups: Iterable[dict[str, Any]],
) -> tuple[EffectScales, dict[str, Any]]:
    """Fit frozen effect scales with independent source episodes equally weighted."""

    signal_by_source: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    replay_by_source: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    group_counts: dict[str, int] = defaultdict(int)
    floors = {
        "q_position": 1e-4,
        "q_quaternion": 1e-4,
        "q_gripper": 1e-4,
        "d_obj": 1e-4,
        "d_prog": 1e-4,
    }

    def variance(value: Tensor, mask: Tensor | None = None) -> float:
        centered = value.float() - value.float().mean(
            dim=0, keepdim=True
        )
        squared = centered.square()
        if mask is None:
            return float(squared.mean())
        expanded = mask.to(dtype=torch.bool)
        while expanded.ndim < squared.ndim:
            expanded = expanded.unsqueeze(-1)
        expanded = expanded.expand_as(squared)
        return float(squared[expanded].mean()) if bool(expanded.any()) else 0.0

    for group in groups:
        source = str(group["source_trajectory_id"])
        group_counts[source] += 1
        targets = group["targets"]
        q = targets["d_q"].cpu()
        if q.ndim != 2 or q.shape[1] != 9:
            raise ValueError(
                "effect_centered_v1 requires nine-dimensional q targets"
            )
        object_mask = targets["object_mask"].cpu().bool()
        if not torch.equal(
            object_mask,
            object_mask[0:1].expand_as(object_mask),
        ):
            raise ValueError(
                "effect scale fit requires a branch-constant object mask"
            )
        block_values = {
            "q_position": variance(q[:, :3]),
            "q_quaternion": variance(q[:, 3:7]),
            "q_gripper": variance(q[:, 7:9]),
            "d_obj": variance(
                targets["d_obj"].cpu(),
                object_mask,
            ),
            "d_prog": variance(targets["d_prog"].cpu()),
        }
        for key, value in block_values.items():
            signal_by_source[source][key].append(value)

        records = group["restore_qc"].get("records", [])
        if len(records) < 2:
            raise ValueError("effect scale fit requires repeated restore records")
        replay_q = torch.stack(
            [record["q"].cpu() for record in records]
        )
        replay_obj = torch.stack(
            [record["object_pos"].cpu() for record in records]
        )
        replay_values = {
            "q_position": variance(replay_q[:, :3]),
            "q_quaternion": variance(replay_q[:, 3:7]),
            "q_gripper": variance(replay_q[:, 7:9]),
            "d_obj": variance(
                replay_obj,
                object_mask[0:1].expand(replay_obj.shape[0], -1),
            ),
            # Dense progress replay is not directly stored. Its numerical floor
            # and train-split sibling variance define the provisional scale.
            "d_prog": 0.0,
        }
        for key, value in replay_values.items():
            replay_by_source[source][key].append(value)

    if not group_counts:
        raise ValueError("cannot fit effect scales without training groups")

    def source_balanced_mean(
        values: dict[str, dict[str, list[float]]],
        key: str,
    ) -> float:
        source_means = [
            sum(source_values[key]) / len(source_values[key])
            for source_values in values.values()
        ]
        return sum(source_means) / len(source_means)

    signal_variance = {
        key: source_balanced_mean(signal_by_source, key)
        for key in floors
    }
    replay_variance = {
        key: source_balanced_mean(replay_by_source, key)
        for key in floors
    }
    scale_values = {
        key: max(
            signal_variance[key] ** 0.5,
            3.0 * replay_variance[key] ** 0.5,
            floors[key],
        )
        for key in floors
    }
    scales = EffectScales(**scale_values)
    report = {
        "rule": (
            "source-balanced train sibling RMS, floored by 3x replay RMS "
            "and fixed physical resolution"
        ),
        "source_episode_groups": dict(sorted(group_counts.items())),
        "signal_variance": signal_variance,
        "replay_variance": replay_variance,
        "floors": floors,
        "scales": scale_values,
    }
    return scales, report


def initialize_effect_v1_heads(lc_state: LCState) -> None:
    """Start continuous effects at zero; freeze currently unidentified heads."""

    for head in (
        lc_state.outcome.dq,
        lc_state.outcome.dobj,
        lc_state.outcome.dprog,
    ):
        output = head[-1]
        torch.nn.init.zeros_(output.weight)
        torch.nn.init.zeros_(output.bias)
    for head in (lc_state.outcome.dy, lc_state.outcome.ret):
        for parameter in head.parameters():
            parameter.requires_grad_(False)


def _microbatches(indices: Tensor, size: int) -> Iterable[Tensor]:
    if size < 1:
        raise ValueError("flow microbatch must be positive")
    for start in range(0, indices.numel(), size):
        yield indices[start : start + size]


def paired_flow_noise_time(
    policy,
    *,
    seed: int,
    device: torch.device | str,
) -> tuple[Tensor, Tensor]:
    """One common-random-number flow target shared by sibling branches."""
    cfg = policy.config
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(
        1,
        cfg.chunk_size,
        cfg.max_action_dim,
        generator=generator,
        dtype=torch.float32,
    ).to(device)
    # PI05's beta sampler runs on CPU and does not accept a generator.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 1)
        time = policy.model.sample_time(1, device)
    return noise, time


def backward_group_flow(
    runner,
    lc_state: LCState,
    group: dict[str, Any],
    *,
    device: torch.device | str,
    loss_scale: float,
    flow_microbatch: int,
    flow_seed: int,
    truncate_bptt: int | None = None,
    prefix=None,
) -> dict[str, float]:
    """Backprop positive-only branch flow loss with one shared prefix cache."""
    weights = group["branch_weights"].to(device, dtype=torch.float32)
    positive = torch.nonzero(weights > 0, as_tuple=False).flatten()
    weight_sum = weights[positive].sum()
    if positive.numel() == 0 or float(weight_sum) == 0.0:
        return {
            "flow_loss": 0.0,
            "positive_branches": 0.0,
            "flow_weight_sum": 0.0,
        }

    if prefix is None:
        batch = runner._obs_to_policy_batch(
            group["current_observation"],
            group["full_instruction"],
        )
        prefix = prefix_forward(runner.policy, batch)
    noise, time = paired_flow_noise_time(
        runner.policy, seed=flow_seed, device=device
    )
    weighted_loss_value = 0.0
    actions = group["actions_norm"].to(device, dtype=torch.float32)
    lengths = group["executed_lengths"].to(device, dtype=torch.long)
    for indices in _microbatches(positive, flow_microbatch):
        z_t = unroll_lc_history(
            lc_state,
            group,
            device=device,
            truncate_bptt=truncate_bptt,
            current_prefix=prefix,
        )
        local_weights = weights[indices]
        local_fraction = local_weights.sum() / weight_sum
        bias = lc_state.adarms_bias(z_t).expand(indices.numel(), -1)
        local_loss, _ = cached_branch_flow_loss(
            runner.policy,
            prefix,
            actions[indices],
            bias,
            local_weights,
            executed_lengths=lengths[indices],
            max_executed=lc_state.config.execution_horizon,
            noise=noise.expand(indices.numel(), -1, -1),
            time=time.expand(indices.numel()),
        )
        scaled = loss_scale * local_fraction * local_loss
        scaled.backward()
        weighted_loss_value += float(
            (local_fraction * local_loss).detach()
        )
    return {
        "flow_loss": weighted_loss_value,
        "positive_branches": float(positive.numel()),
        "flow_weight_sum": float(weight_sum.detach()),
    }


def gradient_norm(module) -> float:
    squared = torch.zeros((), device=next(module.parameters()).device)
    for parameter in module.parameters():
        if parameter.grad is not None:
            squared = squared + parameter.grad.detach().float().square().sum()
    return float(squared.sqrt())


@torch.no_grad()
def fixed_group_probe(
    runner,
    lc_state: LCState,
    group: dict[str, Any],
    *,
    device: torch.device | str,
    seed: int,
    effect_scales: EffectScales | None = None,
) -> dict[str, float]:
    """Deterministic pre/post probe for one grouped training example.

    The same flow time/noise and action-sampling noise are reused before and
    after optimization, so changes cannot be attributed to Monte Carlo noise.
    Positive branches are probed one at a time to preserve the 32 GB contract.
    """
    was_training = lc_state.training
    lc_state.eval()
    batch = runner._obs_to_policy_batch(
        group["current_observation"], group["full_instruction"]
    )
    prefix = prefix_forward(runner.policy, batch)
    z_t = unroll_lc_history(
        lc_state,
        group,
        device=device,
        truncate_bptt=None,
        current_prefix=prefix,
    )
    outcome, outcome_parts = branch_outcome_loss(
        lc_state, z_t, group, device=device
    )
    training_outcome = outcome
    training_outcome_parts: dict[str, Tensor] = {}
    if effect_scales is not None:
        predictions = branch_outcome_predictions(
            lc_state,
            z_t,
            group,
            device=device,
        )
        training_outcome, training_outcome_parts = centered_effect_loss(
            predictions,
            to_device_tensors(group["targets"], device),
            effect_scales,
        )

    weights = group["branch_weights"].to(device, dtype=torch.float32)
    positive = torch.nonzero(weights > 0, as_tuple=False).flatten()
    flow = 0.0
    if positive.numel():
        noise, time = paired_flow_noise_time(
            runner.policy, seed=seed, device=device
        )
        weight_sum = weights[positive].sum()
        actions = group["actions_norm"].to(device, dtype=torch.float32)
        lengths = group["executed_lengths"].to(device, dtype=torch.long)
        bias = lc_state.adarms_bias(z_t)
        for index_tensor in positive:
            index = int(index_tensor.item())
            local_weight = weights[index : index + 1]
            local_loss, _ = cached_branch_flow_loss(
                runner.policy,
                prefix,
                actions[index : index + 1],
                bias,
                local_weight,
                executed_lengths=lengths[index : index + 1],
                max_executed=lc_state.config.execution_horizon,
                noise=noise,
                time=time,
            )
            flow += float(
                (local_weight.sum() / weight_sum * local_loss).detach()
            )
    else:
        bias = lc_state.adarms_bias(z_t)

    generator = torch.Generator(device="cpu").manual_seed(seed + 2)
    sample_noise = torch.randn(
        1,
        runner.policy.config.chunk_size,
        runner.policy.config.max_action_dim,
        generator=generator,
        dtype=torch.float32,
    ).to(device)
    stock_chunk = sample_chunks(
        runner.policy,
        batch,
        n=1,
        noise=sample_noise,
        prefix=prefix,
    )
    lc_chunk = sample_chunks_lc(
        runner.policy,
        batch,
        bias,
        n=1,
        noise=sample_noise,
        prefix=prefix,
    )
    action_delta = lc_chunk - stock_chunk
    executed_delta = action_delta[
        :, : lc_state.config.execution_horizon
    ]
    result = {
        "outcome_loss": float(outcome.detach()),
        "training_outcome_loss": float(training_outcome.detach()),
        **{
            f"outcome_{name}": float(value.detach())
            for name, value in outcome_parts.items()
        },
        **{
            f"training_{name}": float(value.detach())
            for name, value in training_outcome_parts.items()
        },
        "flow_loss": flow,
        "positive_branches": float(positive.numel()),
        "bias_l2": float(bias.float().norm()),
        "wz_out_l2": float(lc_state.wz_out.weight.float().norm()),
        "action_shift_exec_mean_abs": float(executed_delta.abs().mean()),
        "action_shift_exec_max_abs": float(executed_delta.abs().max()),
        "action_shift_full_mean_abs": float(action_delta.abs().mean()),
        "action_shift_full_max_abs": float(action_delta.abs().max()),
    }
    lc_state.train(was_training)
    return result


def train_group_step(
    runner,
    lc_state: LCState,
    optimizer,
    group: dict[str, Any],
    *,
    device: torch.device | str,
    outcome_scale: float = 1.0,
    flow_scale: float = 0.1,
    flow_microbatch: int = 1,
    flow_seed: int = 0,
    truncate_bptt: int | None = 16,
    max_grad_norm: float = 1.0,
    effect_scales: EffectScales | None = None,
) -> dict[str, float]:
    """One optimizer step for one complete snapshot sibling group."""
    # AMENDMENT 1 (2026-07-24.md): admission is per-channel. A group trains
    # iff its replay is structurally valid and its object channel is
    # resolved; a q-unresolved group additionally has its q effect blocks
    # masked out below so no channel ever trains against sub-noise targets.
    # Groups collected before the amendment carry valid=True and both
    # per-channel flags True, so this is behavior-preserving for them.
    qc = group["restore_qc"]
    structural_ok = bool(qc.get("structural_valid", qc.get("valid", False)))
    object_ok = bool(qc.get("object_resolved", qc.get("valid", False)))
    if not (structural_ok and object_ok):
        raise ValueError("restore-invalid branch group cannot train LC-Flow")
    optimizer.zero_grad(set_to_none=True)
    positive_count = int((group["branch_weights"] > 0).sum())
    prefix = None
    if flow_scale > 0 and positive_count:
        batch = runner._obs_to_policy_batch(
            group["current_observation"],
            group["full_instruction"],
        )
        prefix = prefix_forward(runner.policy, batch)
    z_t = unroll_lc_history(
        lc_state,
        group,
        device=device,
        truncate_bptt=truncate_bptt,
        current_prefix=prefix,
    )
    if effect_scales is None:
        out_loss, out_parts = branch_outcome_loss(
            lc_state, z_t, group, device=device
        )
    else:
        predictions = branch_outcome_predictions(
            lc_state,
            z_t,
            group,
            device=device,
        )
        effect_targets = to_device_tensors(group["targets"], device)
        if not bool(qc.get("q_resolved", True)):
            zero_mask = torch.zeros(
                effect_targets["d_q"].shape[0],
                device=predictions["d_q"].device,
            )
            effect_targets = {
                **effect_targets,
                "q_mask_position": zero_mask,
                "q_mask_quaternion": zero_mask,
                "q_mask_gripper": zero_mask,
            }
        out_loss, out_parts = centered_effect_loss(
            predictions,
            effect_targets,
            effect_scales,
        )
    (outcome_scale * out_loss).backward()
    if flow_scale > 0:
        flow_metrics = backward_group_flow(
            runner,
            lc_state,
            group,
            device=device,
            loss_scale=flow_scale,
            flow_microbatch=flow_microbatch,
            flow_seed=flow_seed,
            truncate_bptt=truncate_bptt,
            prefix=prefix,
        )
    else:
        flow_metrics = {
            "flow_loss": 0.0,
            "positive_branches": float(
                (group["branch_weights"] > 0).sum()
            ),
            "flow_weight_sum": float(group["branch_weights"].sum()),
        }
    preclip_grad_norm = gradient_norm(lc_state)
    torch.nn.utils.clip_grad_norm_(lc_state.parameters(), max_grad_norm)
    if any(parameter.grad is not None for parameter in runner.policy.parameters()):
        raise AssertionError("frozen PI05 base received gradients")
    optimizer.step()
    return {
        "outcome_loss": float(out_loss.detach()),
        **{
            f"outcome_{name}": float(value.detach())
            for name, value in out_parts.items()
        },
        **flow_metrics,
        "preclip_grad_norm": preclip_grad_norm,
    }
