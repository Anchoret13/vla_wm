"""Chain stall-snapshot branch collection for the first LC-Flow vertical slice."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor

from lcwm.branch_data import SCHEMA, positive_advantage_weights
from lcwm.probe_data import body_positions, discover_object_bodies
from lcwm.sampler import prefix_forward, sample_chunks
from lcwm.seq_data import goal_atoms, predicate_bits
from lcwm.snapshot import SimSnapshot, restore, snap


@dataclass(frozen=True)
class ProposalSpec:
    """One prompt family used only to propose sibling action chunks."""

    proposal_prompt_id: str
    proposal_family: str
    goal_id: str
    instruction: str
    count: int


@dataclass
class StallContext:
    snapshot: SimSnapshot
    observation: dict[str, Any]
    source_trajectory_id: str
    decision_index: int
    full_instruction: str
    atoms: list[list[str]]
    history_prefix_hidden: Tensor
    history_prefix_mask: Tensor
    history_action_norm: Tensor
    history_behavior_prompt_id: list[str]
    source_policy_noise_seed: int
    next_policy_noise_seed: int
    recovery_trace: list[dict[str, Any]] = field(default_factory=list)
    mined_successor: dict[str, Any] | None = None
    # Raw decision-boundary observations of the source rollout (contact mode
    # only; None for stall/recovery contexts collected before 2026-07-24).
    # Kept so crossed-language prefixes can be recomputed over the complete
    # history through the frozen prefix model (audit correction 5).
    history_observations: list[dict[str, Any]] | None = None


def proprio_from_obs(obs: dict[str, Any]) -> Tensor:
    robot = obs["robot_state"]
    return torch.from_numpy(
        np.concatenate(
            [
                robot["eef"]["pos"],
                robot["eef"]["quat"],
                robot["gripper"]["qpos"],
            ]
        )
    ).to(torch.float32)


def _padded(values: Tensor, length: int, *tail: int) -> tuple[Tensor, Tensor]:
    if values.shape[0] > length:
        raise ValueError(f"cannot pad {values.shape[0]} entries into {length}")
    out = torch.zeros((length, *tail), dtype=values.dtype)
    out[: values.shape[0]] = values
    mask = torch.zeros(length, dtype=torch.bool)
    mask[: values.shape[0]] = True
    return out, mask


def _execute_env_block(env, actions_env: np.ndarray, horizon: int) -> tuple:
    obs = None
    last_info: dict[str, Any] = {}
    total_reward = 0.0
    terminated = truncated = False
    executed = 0
    for action in actions_env[:horizon]:
        obs, reward, term, trunc, last_info = env.step(action)
        total_reward += float(reward)
        executed += 1
        terminated, truncated = bool(term), bool(trunc)
        if terminated or truncated:
            break
    if obs is None:
        raise ValueError("execution horizon must be positive")
    return (
        obs,
        executed,
        terminated,
        truncated,
        total_reward,
        last_info,
    )


def _continuation_indices(
    proposal_families: list[str], continuation_count: int
) -> list[int]:
    """Deterministically choose stock, then one branch/family, then fill.

    Branch zero is the stock reference by contract.  Filling in branch order
    makes the selection stable across reruns as long as the proposal pool is
    unchanged.
    """
    if continuation_count < 0:
        raise ValueError("continuation_count must be nonnegative")
    if not proposal_families or continuation_count == 0:
        return []

    budget = min(continuation_count, len(proposal_families))
    selected = [0]
    represented = {proposal_families[0]}
    for index, family in enumerate(proposal_families[1:], start=1):
        if len(selected) >= budget:
            break
        if family not in represented:
            selected.append(index)
            represented.add(family)
    if len(selected) < budget:
        already_selected = set(selected)
        for index in range(1, len(proposal_families)):
            if index in already_selected:
                continue
            selected.append(index)
            if len(selected) >= budget:
                break
    return selected


def _clone_observation(value: Any) -> Any:
    """Detach an environment observation from mutable simulator buffers."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _clone_observation(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_observation(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_observation(item) for item in value)
    return copy.deepcopy(value)


def _best_recovery_state_index(records: list[dict[str, Any]]) -> int:
    """Select an effectful/contact-proximal state from an attempted route."""
    if not records:
        raise ValueError("recovery state selection requires at least one record")
    return max(
        range(len(records)),
        key=lambda index: (
            int(records[index]["target_success"]),
            int(records[index]["target_displacement"] >= 0.002),
            (
                records[index]["target_displacement"]
                if records[index]["target_displacement"] >= 0.002
                else 0.0
            ),
            -records[index]["eef_distance"],
            -index,
        ),
    )


def _recovery_candidate_feasible(
    *,
    executed: int,
    execution_horizon: int,
    no_damage: bool,
    target_success: bool,
    terminated: bool,
    truncated: bool,
) -> bool:
    """Keep target-effect actions as data even when they damage prior atoms."""
    target_effect = target_success and not truncated
    ordinary_block = (
        executed == execution_horizon
        and no_damage
        and not terminated
        and not truncated
    )
    return target_effect or ordinary_block


@torch.no_grad()
def rollout_to_stall(
    runner,
    env,
    atoms: list[list[str]] | None = None,
    *,
    seed: int,
    trigger_completed: int = 2,
    policy_noise_seed: int = 20_000,
    execution_horizon: int = 10,
) -> StallContext | None:
    """Run the full prompt until the trained two-object mode has completed.

    Prefix states and normalized executed blocks are retained so training can
    unroll the persistent LC state instead of restarting from ``z0`` at the
    selected stall snapshot.
    """
    runner.reset()
    obs, _ = env.reset(seed=seed)
    atoms = atoms or goal_atoms(env)
    full_instruction = env.task_description
    prefix_history: list[Tensor] = []
    mask_history: list[Tensor] = []
    action_history: list[Tensor] = []
    t = 0
    decision = 0
    while t < env.episode_length:
        batch = runner._obs_to_policy_batch(obs, full_instruction)
        prefix = prefix_forward(runner.policy, batch)
        prefix_history.append(prefix.hidden[0].half().cpu())
        mask_history.append(prefix.pad_masks[0].bool().cpu())
        bits = predicate_bits(env, atoms)
        trigger = (
            bool(bits[:trigger_completed].all())
            and trigger_completed < len(bits)
            and not bool(bits[trigger_completed])
        )
        if trigger:
            history_actions = (
                torch.stack(action_history)
                if action_history
                else torch.empty(0, execution_horizon, 7)
            )
            return StallContext(
                snapshot=snap(
                    env,
                    t=t,
                    suite_name="loho_inspired",
                    task_id=0,
                    source_seed=seed,
                    decision_index=decision,
                ),
                observation=obs,
                source_trajectory_id=f"{env.task}-full-seed{seed}",
                decision_index=decision,
                full_instruction=full_instruction,
                atoms=atoms,
                history_prefix_hidden=torch.stack(prefix_history),
                history_prefix_mask=torch.stack(mask_history),
                history_action_norm=history_actions,
                history_behavior_prompt_id=["full"] * len(action_history),
                source_policy_noise_seed=policy_noise_seed,
                next_policy_noise_seed=policy_noise_seed + decision,
            )

        chunk = sample_chunks(
            runner.policy,
            batch,
            n=1,
            seed=policy_noise_seed + decision,
            prefix=prefix,
        )
        action_history.append(
            chunk[0, :execution_horizon].detach().float().cpu()
        )
        actions_env = runner.chunk_to_env(chunk[:, :execution_horizon])
        (
            obs,
            executed,
            terminated,
            truncated,
            _reward,
            _info,
        ) = _execute_env_block(
            env, actions_env, execution_horizon
        )
        t += executed
        decision += 1
        if terminated or truncated:
            break
    return None


@torch.no_grad()
def rollout_to_contact(
    runner,
    env,
    atoms: list[list[str]] | None = None,
    *,
    seed: int,
    object_body_by_name: dict[str, str] | None = None,
    contact_distance: float = 0.05,
    skip_matches: int = 0,
    policy_noise_seed: int = 20_000,
    execution_horizon: int = 10,
    diagnostics: list[dict[str, Any]] | None = None,
) -> tuple[StallContext, dict[str, Any]] | None:
    """Snapshot the on-policy full-prompt rollout at a contact-proximal state.

    Same sampling law and per-decision noise seeds as ``rollout_to_stall``, so
    the trajectory prefix is the same source episode a stall run would
    produce at this seed. The trigger is physical instead of predicate-based:
    the EEF comes within ``contact_distance`` of the root body of a goal
    object whose atom is still false. The nearest incomplete object is
    reported so the caller can point the dense-progress readout at the object
    actually being manipulated. The first ``skip_matches`` triggering
    decisions are passed over so one source episode can yield several
    distinct contact-phase snapshots under separate contracts.

    Raw decision-boundary observations are kept on the returned context so
    crossed-language prefixes can later be recomputed over the complete
    history through the frozen prefix model (2026-07-24 audit correction 5).
    """
    runner.reset()
    obs, _ = env.reset(seed=seed)
    atoms = atoms or goal_atoms(env)
    if object_body_by_name is None:
        # The lazily-built raw env only exists after the reset above.
        object_body_by_name = discover_object_bodies(env)
    full_instruction = env.task_description
    prefix_history: list[Tensor] = []
    mask_history: list[Tensor] = []
    action_history: list[Tensor] = []
    observation_history: list[dict[str, Any]] = []
    t = 0
    decision = 0
    matched = 0
    while t < env.episode_length:
        batch = runner._obs_to_policy_batch(obs, full_instruction)
        prefix = prefix_forward(runner.policy, batch)
        prefix_history.append(prefix.hidden[0].half().cpu())
        mask_history.append(prefix.pad_masks[0].bool().cpu())
        observation_history.append(_clone_observation(obs))
        bits = predicate_bits(env, atoms)
        eef = np.asarray(obs["robot_state"]["eef"]["pos"], dtype=np.float64)
        incomplete = [
            atom[1] for index, atom in enumerate(atoms) if not bool(bits[index])
        ]
        nearest_object, nearest_distance = None, float("inf")
        if incomplete:
            positions = body_positions(
                env, [object_body_by_name[name] for name in incomplete]
            )
            distances = np.linalg.norm(positions - eef[None], axis=-1)
            index = int(distances.argmin())
            nearest_object = incomplete[index]
            nearest_distance = float(distances[index])
        if diagnostics is not None:
            diagnostics.append(
                {
                    "decision": decision,
                    "t": t,
                    "nearest_incomplete_object": nearest_object,
                    "nearest_distance": nearest_distance,
                    "bits": [bool(b) for b in bits],
                }
            )
        if nearest_object is not None and nearest_distance <= contact_distance:
            if matched >= skip_matches:
                history_actions = (
                    torch.stack(action_history)
                    if action_history
                    else torch.empty(0, execution_horizon, 7)
                )
                context = StallContext(
                    snapshot=snap(
                        env,
                        t=t,
                        suite_name="loho_inspired",
                        task_id=0,
                        source_seed=seed,
                        decision_index=decision,
                        contact_object=nearest_object,
                        contact_distance=nearest_distance,
                        contact_skip=skip_matches,
                    ),
                    observation=obs,
                    source_trajectory_id=f"{env.task}-full-seed{seed}",
                    decision_index=decision,
                    full_instruction=full_instruction,
                    atoms=atoms,
                    history_prefix_hidden=torch.stack(prefix_history),
                    history_prefix_mask=torch.stack(mask_history),
                    history_action_norm=history_actions,
                    history_behavior_prompt_id=["full"] * len(action_history),
                    source_policy_noise_seed=policy_noise_seed,
                    next_policy_noise_seed=policy_noise_seed + decision,
                    history_observations=observation_history,
                )
                info = {
                    "contact_object": nearest_object,
                    "contact_distance": nearest_distance,
                    "decision_index": decision,
                    "t": t,
                    "matches_skipped": matched,
                }
                return context, info
            matched += 1

        chunk = sample_chunks(
            runner.policy,
            batch,
            n=1,
            seed=policy_noise_seed + decision,
            prefix=prefix,
        )
        action_history.append(
            chunk[0, :execution_horizon].detach().float().cpu()
        )
        actions_env = runner.chunk_to_env(chunk[:, :execution_horizon])
        (
            obs,
            executed,
            terminated,
            truncated,
            _reward,
            _info,
        ) = _execute_env_block(
            env, actions_env, execution_horizon
        )
        t += executed
        decision += 1
        if terminated or truncated:
            break
    return None


@torch.no_grad()
def advance_recovery_context(
    runner,
    env,
    context: StallContext,
    behavior_instruction: str,
    *,
    behavior_prompt_id: str = "recovery",
    blocks: int,
    behavior_noise_seed: int,
    full_reference_noise_seed: int,
    target_body_name: str | None = None,
    candidates_per_block: int = 1,
    stop_eef_distance: float = 0.10,
    execution_horizon: int = 10,
    diagnostics: list[dict[str, Any]] | None = None,
) -> StallContext | None:
    """Mine a contact-proximal recovery state from PI0.5 proposals.

    Actions may be proposed by the remaining-goal prompt, but every stored
    observation prefix is recomputed under the composite full instruction.
    Thus proposal language remains behavior provenance; the learned LC state
    and all outcome labels still use the evaluation goal.  When more than one
    candidate is requested, privileged target distance is used only to choose
    a data-collection trajectory; it is never stored as a model input.
    """
    if blocks < 0:
        raise ValueError("recovery warmup blocks must be nonnegative")
    if not behavior_prompt_id:
        raise ValueError("recovery behavior_prompt_id must be nonempty")
    if candidates_per_block < 1:
        raise ValueError("recovery candidates_per_block must be positive")
    if candidates_per_block > 1 and target_body_name is None:
        raise ValueError("target_body_name is required for recovery mining")
    if stop_eef_distance <= 0:
        raise ValueError("stop_eef_distance must be positive")
    if blocks == 0:
        context.next_policy_noise_seed = full_reference_noise_seed
        return context

    obs = context.observation
    hidden = [item.clone() for item in context.history_prefix_hidden]
    masks = [item.clone() for item in context.history_prefix_mask]
    actions = [item.clone() for item in context.history_action_norm]
    behavior_ids = list(context.history_behavior_prompt_id)
    recovery_trace = list(context.recovery_trace)
    history_len_before = len(hidden)
    action_len_before = len(actions)
    trace_len_before = len(recovery_trace)
    attempted_states: list[dict[str, Any]] = []
    steps = 0
    executed_blocks = 0
    for block_index in range(blocks):
        current_snapshot = snap(
            env,
            t=context.snapshot.t + steps,
            suite_name=context.snapshot.suite_name,
            task_id=context.snapshot.task_id,
            recovery_block=block_index,
        )
        completed_before = torch.from_numpy(
            predicate_bits(env, context.atoms)
        ).bool()
        target_pos_before = (
            torch.from_numpy(body_positions(env, [target_body_name])[0]).float()
            if target_body_name is not None
            else None
        )
        proposal_batch = runner._obs_to_policy_batch(
            obs, behavior_instruction
        )
        proposal_prefix = prefix_forward(runner.policy, proposal_batch)
        candidate_seeds = [
            behavior_noise_seed
            + block_index * candidates_per_block
            + candidate_index
            for candidate_index in range(candidates_per_block)
        ]
        candidate_noise = torch.stack(
            [
                torch.randn(
                    runner.policy.config.chunk_size,
                    runner.policy.config.max_action_dim,
                    generator=torch.Generator(device="cpu").manual_seed(seed),
                    dtype=torch.float32,
                )
                for seed in candidate_seeds
            ]
        )
        chunks = sample_chunks(
            runner.policy,
            proposal_batch,
            n=candidates_per_block,
            noise=candidate_noise.to(proposal_prefix.pad_masks.device),
            prefix=proposal_prefix,
        )
        candidates: list[dict[str, Any]] = []
        for candidate_index, (seed, chunk) in enumerate(
            zip(candidate_seeds, chunks, strict=True)
        ):
            restore(env, current_snapshot)
            action_block = (
                chunk[:execution_horizon].detach().float().cpu()
            )
            actions_env = runner.chunk_to_env(action_block)
            (
                candidate_obs,
                executed,
                terminated,
                truncated,
                _reward,
                _info,
            ) = _execute_env_block(env, actions_env, execution_horizon)
            candidate_bits = torch.from_numpy(
                predicate_bits(env, context.atoms)
            ).bool()
            no_damage = bool(
                (~completed_before | candidate_bits).all()
            )
            target_success = bool(candidate_bits[-1])
            if target_body_name is not None:
                target_pos_after = torch.from_numpy(
                    body_positions(env, [target_body_name])[0]
                ).float()
                eef_distance = float(
                    torch.linalg.vector_norm(
                        proprio_from_obs(candidate_obs)[:3] - target_pos_after
                    )
                )
                target_displacement = float(
                    torch.linalg.vector_norm(
                        target_pos_after - target_pos_before
                    )
                )
            else:
                eef_distance = 0.0
                target_displacement = 0.0
            candidate_snapshot = snap(
                env,
                t=context.snapshot.t + steps + executed,
                suite_name=context.snapshot.suite_name,
                task_id=context.snapshot.task_id,
                recovery_block=block_index,
                recovery_candidate=candidate_index,
                recovery_noise_seed=seed,
            )
            feasible = _recovery_candidate_feasible(
                executed=executed,
                execution_horizon=execution_horizon,
                no_damage=no_damage,
                target_success=target_success,
                terminated=terminated,
                truncated=truncated,
            )
            candidates.append(
                {
                    "candidate_index": candidate_index,
                    "noise_seed": seed,
                    "flow_noise": candidate_noise[candidate_index].clone(),
                    "chunk_norm": chunk.detach().float().cpu(),
                    "action": action_block,
                    "observation": _clone_observation(candidate_obs),
                    "snapshot": candidate_snapshot,
                    "bits": candidate_bits,
                    "executed": executed,
                    "terminated": terminated,
                    "truncated": truncated,
                    "no_damage": no_damage,
                    "target_success": target_success,
                    "eef_distance": eef_distance,
                    "target_displacement": target_displacement,
                    "feasible": feasible,
                }
            )

        # First preserve the completed composite-goal atoms, then prefer a
        # true target effect, then proximity. Candidate order is only the final
        # deterministic tie-breaker.
        chosen = max(
            candidates,
            key=lambda item: (
                int(item["feasible"]),
                int(item["target_success"]),
                int(item["target_displacement"] >= 0.002),
                (
                    item["target_displacement"]
                    if item["target_displacement"] >= 0.002
                    else 0.0
                ),
                -item["eef_distance"],
                -item["candidate_index"],
            ),
        )
        if not chosen["feasible"]:
            if diagnostics is not None:
                diagnostics.append(
                    {
                        "status": "no_feasible_candidate",
                        "block": block_index,
                        "behavior_prompt_id": behavior_prompt_id,
                        "candidates": [
                            {
                                key: item[key]
                                for key in (
                                    "candidate_index",
                                    "noise_seed",
                                    "executed",
                                    "terminated",
                                    "truncated",
                                    "no_damage",
                                    "target_success",
                                    "eef_distance",
                                    "target_displacement",
                                    "feasible",
                                )
                            }
                            for item in candidates
                        ],
                    }
                )
            if attempted_states:
                # The route became unsafe after earlier valid progress. Fall
                # back to the best safe state already observed; the caller's
                # state-admission gate decides whether it is close/effectful
                # enough to justify branch collection.
                break
            return None
        if chosen["target_success"]:
            # The state *before* this target-effect action is exactly the
            # useful branch point. Keep the action/noise as an extra proposal
            # rather than consuming it as warmup. The later no-damage rule,
            # not the state miner, decides whether it is an imitation target.
            restore(env, current_snapshot)
            mined_successor = {
                "proposal_prompt_id": (
                    f"mined_{behavior_prompt_id}_successor"
                ),
                "proposal_family": "mined_successor",
                "goal_id": (
                    "chain3_remaining_atomic"
                    if behavior_prompt_id == "remaining_atomic_recovery"
                    else "chain3_support_pair"
                ),
                "instruction": behavior_instruction,
                "candidate_index": chosen["candidate_index"],
                "flow_noise_seed": chosen["noise_seed"],
                "flow_noise": chosen["flow_noise"],
                "action_norm": chosen["chunk_norm"],
                "candidate_noise_seeds": candidate_seeds,
                "candidate_eef_distance": [
                    item["eef_distance"] for item in candidates
                ],
                "candidate_target_displacement": [
                    item["target_displacement"] for item in candidates
                ],
                "candidate_no_damage": [
                    item["no_damage"] for item in candidates
                ],
            }
            if diagnostics is not None:
                diagnostics.append(
                    {
                        "status": "mined_target_effect_successor",
                        "block": block_index,
                        "behavior_prompt_id": behavior_prompt_id,
                        "chosen_noise_seed": chosen["noise_seed"],
                        "chosen_no_damage": chosen["no_damage"],
                        "chosen_terminated": chosen["terminated"],
                        "chosen_truncated": chosen["truncated"],
                    }
                )
            return StallContext(
                snapshot=current_snapshot,
                observation=_clone_observation(obs),
                source_trajectory_id=context.source_trajectory_id,
                decision_index=context.decision_index + executed_blocks,
                full_instruction=context.full_instruction,
                atoms=context.atoms,
                history_prefix_hidden=torch.stack(hidden),
                history_prefix_mask=torch.stack(masks),
                history_action_norm=torch.stack(actions),
                history_behavior_prompt_id=behavior_ids,
                source_policy_noise_seed=context.source_policy_noise_seed,
                next_policy_noise_seed=(
                    full_reference_noise_seed + executed_blocks
                ),
                recovery_trace=recovery_trace,
                mined_successor=mined_successor,
            )
        restore(env, chosen["snapshot"])
        obs = chosen["observation"]
        actions.append(chosen["action"])
        behavior_ids.append(
            f"{behavior_prompt_id}:seed{chosen['noise_seed']}"
        )
        recovery_trace.append(
            {
                "block": block_index,
                "history_block": len(recovery_trace),
                "behavior_prompt_id": behavior_prompt_id,
                "behavior_instruction": behavior_instruction,
                "candidate_noise_seeds": candidate_seeds,
                "candidate_eef_distance": [
                    item["eef_distance"] for item in candidates
                ],
                "candidate_target_displacement": [
                    item["target_displacement"] for item in candidates
                ],
                "candidate_no_damage": [
                    item["no_damage"] for item in candidates
                ],
                "chosen_candidate_index": chosen["candidate_index"],
                "chosen_noise_seed": chosen["noise_seed"],
                "chosen_eef_distance": chosen["eef_distance"],
                "chosen_target_displacement": chosen[
                    "target_displacement"
                ],
            }
        )
        full_batch = runner._obs_to_policy_batch(
            obs, context.full_instruction
        )
        full_prefix = prefix_forward(runner.policy, full_batch)
        hidden.append(full_prefix.hidden[0].half().cpu())
        masks.append(full_prefix.pad_masks[0].bool().cpu())
        steps += chosen["executed"]
        executed_blocks += 1
        attempted_states.append(
            {
                "snapshot": chosen["snapshot"],
                "observation": chosen["observation"],
                "steps": steps,
                "target_success": chosen["target_success"],
                "target_displacement": chosen["target_displacement"],
                "eef_distance": chosen["eef_distance"],
            }
        )
        if chosen["eef_distance"] <= stop_eef_distance:
            break

    final_bits = predicate_bits(env, context.atoms)
    if bool(final_bits.all()):
        if diagnostics is not None:
            diagnostics.append(
                {
                    "status": "full_goal_after_recovery_without_mined_successor",
                    "behavior_prompt_id": behavior_prompt_id,
                    "attempted_blocks": executed_blocks,
                    "final_bits": final_bits.tolist(),
                }
            )
        return None
    selected_local_index = _best_recovery_state_index(attempted_states)
    selected_blocks = selected_local_index + 1
    selected = attempted_states[selected_local_index]
    if selected_local_index != len(attempted_states) - 1:
        restore(env, selected["snapshot"])
    obs = selected["observation"]
    hidden = hidden[: history_len_before + selected_blocks]
    masks = masks[: history_len_before + selected_blocks]
    actions = actions[: action_len_before + selected_blocks]
    behavior_ids = behavior_ids[: action_len_before + selected_blocks]
    attempted_suffix = copy.deepcopy(
        recovery_trace[
            trace_len_before + selected_blocks :
        ]
    )
    recovery_trace = recovery_trace[
        : trace_len_before + selected_blocks
    ]
    recovery_trace[-1]["state_selection"] = {
        "attempted_blocks": executed_blocks,
        "selected_local_block": selected_local_index,
        "post_selection_search": attempted_suffix,
    }
    if diagnostics is not None:
        diagnostics.append(
            {
                "status": "selected_recovery_state",
                "behavior_prompt_id": behavior_prompt_id,
                "attempted_blocks": executed_blocks,
                "selected_local_block": selected_local_index,
                "selected_eef_distance": selected["eef_distance"],
                "selected_target_displacement": selected[
                    "target_displacement"
                ],
            }
        )
    decision_index = context.decision_index + selected_blocks
    return StallContext(
        snapshot=snap(
            env,
            t=context.snapshot.t + selected["steps"],
            suite_name=context.snapshot.suite_name,
            task_id=context.snapshot.task_id,
            source_seed=context.snapshot.meta.get("source_seed"),
            source_decision_index=context.decision_index,
            recovery_blocks=selected_blocks,
            recovery_attempted_blocks=executed_blocks,
            recovery_selected_local_block=selected_local_index,
            recovery_noise_seed=behavior_noise_seed,
        ),
        observation=_clone_observation(obs),
        # Recovery snapshots are descendants of the same source episode and
        # must remain in its split; recovery identity belongs to snapshot
        # provenance, not source_trajectory_id.
        source_trajectory_id=context.source_trajectory_id,
        decision_index=decision_index,
        full_instruction=context.full_instruction,
        atoms=context.atoms,
        history_prefix_hidden=torch.stack(hidden),
        history_prefix_mask=torch.stack(masks),
        history_action_norm=torch.stack(actions),
        history_behavior_prompt_id=behavior_ids,
        source_policy_noise_seed=context.source_policy_noise_seed,
        next_policy_noise_seed=(
            full_reference_noise_seed + selected_blocks
        ),
        recovery_trace=recovery_trace,
        mined_successor=context.mined_successor,
    )


@torch.no_grad()
def _continue_atomic_goal(
    runner,
    env,
    obs,
    atom: list[str],
    evaluation_atoms: list[list[str]],
    instruction: str,
    *,
    horizon: int,
    execution_horizon: int,
    noise_seed: int,
) -> tuple[bool, bool, int, list[int], Tensor]:
    used_seeds: list[int] = []
    steps = 0
    final_bits = torch.from_numpy(
        predicate_bits(env, evaluation_atoms)
    ).bool()
    atomic_success = bool(predicate_bits(env, [atom])[0])
    full_success = bool(final_bits.all())
    if atomic_success or full_success:
        return (
            atomic_success,
            full_success,
            steps,
            used_seeds,
            final_bits,
        )
    while steps < horizon:
        batch = runner._obs_to_policy_batch(obs, instruction)
        prefix = prefix_forward(runner.policy, batch)
        seed = noise_seed + len(used_seeds)
        used_seeds.append(seed)
        chunk = sample_chunks(
            runner.policy, batch, n=1, seed=seed, prefix=prefix
        )
        actions_env = runner.chunk_to_env(chunk[:, :execution_horizon])
        (
            obs,
            executed,
            terminated,
            truncated,
            _reward,
            _info,
        ) = _execute_env_block(
            env, actions_env, min(execution_horizon, horizon - steps)
        )
        steps += executed
        final_bits = torch.from_numpy(
            predicate_bits(env, evaluation_atoms)
        ).bool()
        atomic_success = bool(predicate_bits(env, [atom])[0])
        full_success = bool(final_bits.all())
        if atomic_success or full_success or terminated or truncated:
            break
    return atomic_success, full_success, steps, used_seeds, final_bits


@torch.no_grad()
def collect_snapshot_group(
    runner,
    env,
    context: StallContext,
    proposal_specs: list[ProposalSpec],
    body_names: list[str],
    continuation_atom: list[str],
    continuation_instruction: str,
    *,
    goal_object_body_name: str | None = None,
    goal_receptacle_body_name: str | None = None,
    execution_horizon: int = 10,
    continuation_horizon: int = 160,
    continuation_count: int = 4,
    proposal_noise_seed: int = 30_000,
    continuation_noise_seed: int = 40_000,
    max_objects: int = 8,
    max_atoms: int = 5,
    branch_advantage_margin: float = 0.0,
    dense_progress_margin: float = 5e-4,
    replay_q_tolerance: float = 1e-5,
    replay_object_tolerance: float = 1e-5,
    min_effect_noise_ratio: float = 10.0,
) -> dict[str, Any]:
    """Collect one crossed sibling group from a late-chain snapshot.

    Collection has two deliberately separate passes.  First, every sibling
    executes exactly its first ten actions and receives an immediate physical
    and full-goal label.  Then only a deterministic subset receives the
    expensive atomic-goal continuation label.  Continuation outcomes never
    decide which branches become flow-matching targets.
    """
    if execution_horizon != 10:
        raise ValueError("branch schema v1.1 requires execution_horizon=10")
    if not proposal_specs:
        raise ValueError("proposal_specs must include a stock_full branch")
    stock_spec = proposal_specs[0]
    if (
        stock_spec.proposal_prompt_id != "stock_full"
        or stock_spec.proposal_family != "stock_full"
        or stock_spec.count != 1
    ):
        raise ValueError(
            "proposal 0 must be the unique count-1 stock_full branch"
        )
    if stock_spec.instruction != context.full_instruction:
        raise ValueError("stock_full must use the composite full instruction")
    if any(
        spec.proposal_prompt_id == "stock_full"
        or spec.proposal_family == "stock_full"
        for spec in proposal_specs[1:]
    ):
        raise ValueError("stock_full may occur only as proposal branch zero")
    if any(spec.count < 0 for spec in proposal_specs):
        raise ValueError("proposal counts must be nonnegative")
    if continuation_count < 0:
        raise ValueError("continuation_count must be nonnegative")
    if branch_advantage_margin < 0:
        raise ValueError("branch_advantage_margin must be nonnegative")
    if dense_progress_margin < 0:
        raise ValueError("dense_progress_margin must be nonnegative")
    if (goal_object_body_name is None) != (
        goal_receptacle_body_name is None
    ):
        raise ValueError(
            "goal object and receptacle body names must be supplied together"
        )
    if min_effect_noise_ratio <= 0:
        raise ValueError("min_effect_noise_ratio must be positive")
    current_observation = _clone_observation(context.observation)

    action_chunks: list[Tensor] = []
    proposal_ids: list[str] = []
    proposal_families: list[str] = []
    proposal_goal_ids: list[str] = []
    proposal_candidate_indices: list[int] = []
    proposal_instructions: list[str] = []
    proposal_seeds: list[int] = []
    flow_noise_ids: list[str] = []
    proposal_noise: list[Tensor] = []
    next_seed = proposal_noise_seed
    used_seeds: set[int] = set()
    for spec_index, spec in enumerate(proposal_specs):
        if spec.count == 0:
            continue
        batch = runner._obs_to_policy_batch(
            current_observation, spec.instruction
        )
        prefix = prefix_forward(runner.policy, batch)
        spec_seeds: list[int] = []
        spec_noise: list[Tensor] = []
        for candidate_index in range(spec.count):
            if spec_index == 0 and candidate_index == 0:
                seed = context.next_policy_noise_seed
            else:
                while next_seed in used_seeds:
                    next_seed += 1
                seed = next_seed
                next_seed += 1
            if seed in used_seeds:
                raise ValueError(
                    f"duplicate proposal flow-noise seed {seed}; choose a "
                    "proposal_noise_seed that does not collide with stock"
                )
            used_seeds.add(seed)
            generator = torch.Generator(device="cpu").manual_seed(seed)
            noise = torch.randn(
                runner.policy.config.chunk_size,
                runner.policy.config.max_action_dim,
                generator=generator,
                dtype=torch.float32,
            )
            spec_seeds.append(seed)
            spec_noise.append(noise)

        # Prefix inference is done once per prompt specification and all saved
        # noise draws are denoised as one batch.  The CPU tensors below are the
        # exact Gaussian values supplied to the sampler, not regenerated IDs.
        spec_noise_cpu = torch.stack(spec_noise)
        spec_chunks = sample_chunks(
            runner.policy,
            batch,
            n=spec.count,
            noise=spec_noise_cpu.to(prefix.pad_masks.device),
            prefix=prefix,
        )
        for candidate_index, (seed, noise, chunk) in enumerate(
            zip(spec_seeds, spec_noise_cpu, spec_chunks, strict=True)
        ):
            action_chunks.append(chunk.detach().float().cpu())
            proposal_ids.append(spec.proposal_prompt_id)
            proposal_families.append(spec.proposal_family)
            proposal_goal_ids.append(spec.goal_id)
            proposal_candidate_indices.append(candidate_index)
            proposal_instructions.append(spec.instruction)
            proposal_seeds.append(seed)
            flow_noise_ids.append(f"torch_cpu_seed:{seed}")
            proposal_noise.append(noise.clone())

    if context.mined_successor is not None:
        mined = context.mined_successor
        seed = int(mined["flow_noise_seed"])
        if seed in used_seeds:
            raise ValueError(
                f"mined successor noise seed {seed} collides with proposal pool"
            )
        chunk = mined["action_norm"].detach().float().cpu()
        noise = mined["flow_noise"].detach().float().cpu()
        expected_chunk = (
            runner.policy.config.chunk_size,
            context.history_action_norm.shape[-1],
        )
        expected_noise = (
            runner.policy.config.chunk_size,
            runner.policy.config.max_action_dim,
        )
        if tuple(chunk.shape) != expected_chunk:
            raise ValueError(
                f"mined successor chunk {tuple(chunk.shape)} != "
                f"{expected_chunk}"
            )
        if tuple(noise.shape) != expected_noise:
            raise ValueError(
                f"mined successor noise {tuple(noise.shape)} != "
                f"{expected_noise}"
            )
        used_seeds.add(seed)
        action_chunks.append(chunk)
        proposal_ids.append(str(mined["proposal_prompt_id"]))
        proposal_families.append(str(mined["proposal_family"]))
        proposal_goal_ids.append(str(mined["goal_id"]))
        proposal_candidate_indices.append(int(mined["candidate_index"]))
        proposal_instructions.append(str(mined["instruction"]))
        proposal_seeds.append(seed)
        flow_noise_ids.append(f"torch_cpu_seed:{seed}")
        proposal_noise.append(noise)

    actions_norm = torch.stack(action_chunks)
    flow_noise = torch.stack(proposal_noise)
    actions_env = torch.from_numpy(
        np.stack(
            [runner.chunk_to_env(chunk) for chunk in actions_norm],
            axis=0,
        )
    ).to(torch.float32)

    # The environment is still at the uninterrupted capture state: policy
    # inference above never touches it.  Compare the stock continuation from
    # that natural state against two independent restore/replays before any
    # branch group is admitted for training.
    q_before = proprio_from_obs(current_observation)
    object_before_raw = torch.from_numpy(
        body_positions(env, body_names)
    ).to(torch.float32)
    object_before, object_mask = _padded(
        object_before_raw, max_objects, 3
    )
    bits_before_raw = torch.from_numpy(
        predicate_bits(env, context.atoms)
    ).bool()
    bits_before, atom_mask = _padded(bits_before_raw, max_atoms)

    # Restore fidelity is a QC repeat, not a duplicate training branch.
    qc_records: list[dict[str, Any]] = []
    for replay_index in range(3):
        if replay_index > 0:
            restore(env, context.snapshot)
        (
            qc_obs,
            qc_executed,
            qc_terminated,
            qc_truncated,
            _reward,
            _info,
        ) = (
            _execute_env_block(
                env, actions_env[0].numpy(), execution_horizon
            )
        )
        qc_obj_raw = torch.from_numpy(
            body_positions(env, body_names)
        ).to(torch.float32)
        qc_obj, _ = _padded(qc_obj_raw, max_objects, 3)
        qc_bits_raw = torch.from_numpy(
            predicate_bits(env, context.atoms)
        ).bool()
        qc_bits, _ = _padded(qc_bits_raw, max_atoms)
        qc_records.append(
            {
                "kind": "uninterrupted" if replay_index == 0 else "restored",
                "q": proprio_from_obs(qc_obs),
                "object_pos": qc_obj,
                "bits": qc_bits,
                "executed_length": qc_executed,
                "terminated": qc_terminated,
                "truncated": qc_truncated,
            }
        )
    reference = qc_records[0]
    q_linf = max(
        float((record["q"] - reference["q"]).abs().max())
        for record in qc_records[1:]
    )
    if bool(object_mask.any()):
        object_linf = max(
            float(
                (
                    record["object_pos"][object_mask]
                    - reference["object_pos"][object_mask]
                )
                .abs()
                .max()
            )
            for record in qc_records[1:]
        )
    else:
        object_linf = 0.0
    structural_replay_valid = all(
        torch.equal(record["bits"], reference["bits"])
        and record["executed_length"] == reference["executed_length"]
        and record["terminated"] == reference["terminated"]
        and record["truncated"] == reference["truncated"]
        for record in qc_records[1:]
    )
    restore_qc = {
        # Final validity is assigned after all sibling outcomes are available,
        # because identifiability depends on action effect / replay noise.
        "valid": False,
        "structural_valid": structural_replay_valid,
        "q_linf": q_linf,
        "object_linf": object_linf,
        "q_tolerance": replay_q_tolerance,
        "object_tolerance": replay_object_tolerance,
        "qc_repeat_count": len(qc_records) - 1,
        "qc_repeats_are_training_branches": False,
        "stock_branch_index": 0,
        "records": qc_records,
    }

    # Pass 1: all sibling chunks receive the same first-10 physical assay.
    q_after, object_after, bits_after = [], [], []
    executed_lengths, branch_rewards = [], []
    terminated_after_block, truncated_after_block = [], []
    info_success_after_block = []
    next_observations, next_states = [], []
    next_snapshots: list[SimSnapshot] = []
    next_prefix_hidden, next_prefix_mask = [], []
    for branch_index in range(actions_norm.shape[0]):
        restore(env, context.snapshot)
        (
            obs_after,
            executed,
            terminated,
            truncated,
            reward,
            info,
        ) = _execute_env_block(
            env,
            actions_env[branch_index].numpy(),
            execution_horizon,
        )
        executed_lengths.append(executed)
        branch_rewards.append(reward)
        terminated_after_block.append(terminated)
        truncated_after_block.append(truncated)
        info_success_after_block.append(bool(info.get("is_success", False)))
        next_observations.append(_clone_observation(obs_after))
        branch_snapshot = snap(
            env,
            t=context.snapshot.t + executed,
            suite_name=context.snapshot.suite_name,
            task_id=context.snapshot.task_id,
            source_snapshot_id=context.source_trajectory_id,
            source_decision_index=context.decision_index,
            branch_index=branch_index,
        )
        next_snapshots.append(branch_snapshot)
        next_states.append(torch.from_numpy(branch_snapshot.state.copy()))
        next_full_batch = runner._obs_to_policy_batch(
            obs_after, context.full_instruction
        )
        next_full_prefix = prefix_forward(
            runner.policy, next_full_batch
        )
        next_prefix_hidden.append(
            next_full_prefix.hidden[0].half().cpu()
        )
        next_prefix_mask.append(
            next_full_prefix.pad_masks[0].bool().cpu()
        )
        q_after.append(proprio_from_obs(obs_after))
        obj_raw = torch.from_numpy(
            body_positions(env, body_names)
        ).to(torch.float32)
        obj, _ = _padded(obj_raw, max_objects, 3)
        object_after.append(obj)
        bits_raw = torch.from_numpy(
            predicate_bits(env, context.atoms)
        ).bool()
        bits, _ = _padded(bits_raw, max_atoms)
        bits_after.append(bits)

    q_after_t = torch.stack(q_after)
    object_after_t = torch.stack(object_after)
    bits_after_t = torch.stack(bits_after)
    if goal_object_body_name is not None:
        try:
            goal_object_index = body_names.index(goal_object_body_name)
            goal_receptacle_index = body_names.index(
                goal_receptacle_body_name
            )
        except ValueError as error:
            raise ValueError(
                "dense-progress body is absent from collected body_names"
            ) from error
        dense_goal_distance_before = torch.linalg.vector_norm(
            object_before[goal_object_index]
            - object_before[goal_receptacle_index]
        )
        dense_goal_distance_after = torch.linalg.vector_norm(
            object_after_t[:, goal_object_index]
            - object_after_t[:, goal_receptacle_index],
            dim=1,
        )
        dense_goal_progress = (
            dense_goal_distance_before - dense_goal_distance_after
        )
    else:
        goal_object_index = goal_receptacle_index = -1
        dense_goal_distance_before = torch.tensor(0.0)
        dense_goal_distance_after = torch.zeros(actions_norm.shape[0])
        dense_goal_progress = torch.zeros(actions_norm.shape[0])

    replay_q = torch.stack([record["q"] for record in qc_records])
    replay_q_dispersion = float(torch.pdist(replay_q).mean())
    action_q_dispersion = (
        float(torch.pdist(q_after_t).mean())
        if q_after_t.shape[0] > 1
        else 0.0
    )
    q_effect_noise_ratio = action_q_dispersion / max(
        replay_q_dispersion, 1e-12
    )
    if bool(object_mask.any()):
        replay_objects = torch.stack(
            [record["object_pos"][object_mask] for record in qc_records]
        ).flatten(1)
        action_objects = object_after_t[:, object_mask].flatten(1)
        replay_object_dispersion = float(torch.pdist(replay_objects).mean())
        action_object_dispersion = (
            float(torch.pdist(action_objects).mean())
            if action_objects.shape[0] > 1
            else 0.0
        )
    else:
        replay_object_dispersion = action_object_dispersion = 0.0
    object_effect_noise_ratio = action_object_dispersion / max(
        replay_object_dispersion, 1e-12
    )
    q_resolved = (
        q_linf <= replay_q_tolerance
        or q_effect_noise_ratio >= min_effect_noise_ratio
    )
    object_resolved = (
        object_linf <= replay_object_tolerance
        or object_effect_noise_ratio >= min_effect_noise_ratio
    )
    replay_valid = bool(
        structural_replay_valid and q_resolved and object_resolved
    )
    context.snapshot.replay_valid = replay_valid
    restore_qc.update(
        {
            "valid": replay_valid,
            "min_effect_noise_ratio": min_effect_noise_ratio,
            "q_resolved": q_resolved,
            "object_resolved": object_resolved,
            "replay_q_dispersion": replay_q_dispersion,
            "action_q_dispersion": action_q_dispersion,
            "q_effect_noise_ratio": q_effect_noise_ratio,
            "replay_object_dispersion": replay_object_dispersion,
            "action_object_dispersion": action_object_dispersion,
            "object_effect_noise_ratio": object_effect_noise_ratio,
        }
    )
    executed_lengths_t = torch.tensor(executed_lengths, dtype=torch.long)
    execution_mask = (
        torch.arange(execution_horizon)[None]
        < executed_lengths_t[:, None]
    )
    executed_actions_norm = (
        actions_norm[:, :execution_horizon]
        * execution_mask[:, :, None]
    )
    executed_actions_env = (
        actions_env[:, :execution_horizon]
        * execution_mask[:, :, None]
    )

    # Pass 2: continue only stock + one representative per proposal family,
    # then fill the remaining deterministic budget in branch order.
    selected_continuations = _continuation_indices(
        proposal_families, continuation_count
    )
    continuation_mask = torch.zeros(
        actions_norm.shape[0], dtype=torch.bool
    )
    continuation_mask[selected_continuations] = True
    continuation_atomic_success = torch.zeros_like(continuation_mask)
    continuation_success = torch.zeros_like(continuation_mask)
    continuation_steps = torch.zeros(
        actions_norm.shape[0], dtype=torch.long
    )
    continuation_final_bits = bits_after_t.clone()
    continuation_seeds: list[list[int]] = [
        [] for _ in range(actions_norm.shape[0])
    ]
    for branch_index in selected_continuations:
        restore(env, next_snapshots[branch_index])
        obs_after = next_observations[branch_index]
        if (
            terminated_after_block[branch_index]
            or truncated_after_block[branch_index]
        ):
            atomic_success = bool(
                predicate_bits(env, [continuation_atom])[0]
            )
            final_bits_raw = torch.from_numpy(
                predicate_bits(env, context.atoms)
            ).bool()
            full_success = bool(final_bits_raw.all())
            steps, continuation_used_seeds = 0, []
        else:
            (
                atomic_success,
                full_success,
                steps,
                continuation_used_seeds,
                final_bits_raw,
            ) = _continue_atomic_goal(
                runner,
                env,
                obs_after,
                continuation_atom,
                context.atoms,
                continuation_instruction,
                horizon=continuation_horizon,
                execution_horizon=execution_horizon,
                noise_seed=continuation_noise_seed,
            )
        final_bits, _ = _padded(final_bits_raw, max_atoms)
        continuation_atomic_success[branch_index] = atomic_success
        continuation_success[branch_index] = full_success
        continuation_steps[branch_index] = steps
        continuation_final_bits[branch_index] = final_bits
        continuation_seeds[branch_index] = continuation_used_seeds

    # Immediate full-goal effect is the cleanest teacher.  It is often sparse
    # over ten steps, however, so the paired continuation subset may also
    # promote a branch when it improves *full-goal* success/time over the stock
    # branch under the exact same atomic continuation policy/noise schedule.
    # This is a training-target bootstrap, not an online branch ranker.
    atom_float = atom_mask.float()
    denom = atom_float.sum().clamp_min(1)
    before_progress = (bits_before.float() * atom_float).sum() / denom
    after_progress = (
        bits_after_t.float() * atom_float[None]
    ).sum(1) / denom
    d_progress = after_progress - before_progress
    completed_before = bits_before.bool() & atom_mask
    damaged_completed_atom = (
        completed_before[None] & ~bits_after_t.bool()
    ).any(dim=1)
    no_damage = ~damaged_completed_atom
    predicate_beats_stock = (
        after_progress
        > after_progress[0] + branch_advantage_margin
    )
    dense_beats_stock = (
        dense_goal_progress
        > dense_goal_progress[0] + dense_progress_margin
    )
    beats_stock = predicate_beats_stock | dense_beats_stock
    immediate_positive = no_damage & (
        (d_progress > 0)
        | (dense_goal_progress > dense_progress_margin)
        | beats_stock
    )
    immediate_score = after_progress + (
        dense_goal_progress.clamp_min(0)
        / dense_goal_distance_before.clamp_min(1e-3)
    )
    immediate_weights = positive_advantage_weights(
        immediate_positive, immediate_score, stock_index=0
    )
    continuation_score = (
        continuation_success.float()
        * (
            1.0
            + 1.0
            - continuation_steps.float()
            / max(continuation_horizon, 1)
        )
    )
    stock_continuation_score = continuation_score[0]
    paired_continuation_advantage = (
        continuation_mask
        & no_damage
        & (
            continuation_score
            > stock_continuation_score + branch_advantage_margin
        )
    )
    # The stock branch is the comparison reference, not an "improved" target.
    paired_continuation_advantage[0] = False
    continuation_weights = positive_advantage_weights(
        paired_continuation_advantage,
        continuation_score,
        stock_index=0,
    )
    branch_weights = torch.maximum(
        immediate_weights, continuation_weights
    )

    model_id = str(getattr(runner, "model_id", "unknown"))
    model_key = hashlib.sha1(model_id.encode()).hexdigest()[:8]
    recovery_key = hashlib.sha1(
        repr(
            (
                context.recovery_trace,
                (
                    context.mined_successor["flow_noise_seed"]
                    if context.mined_successor is not None
                    else None
                ),
            )
        ).encode()
    ).hexdigest()[:8]
    evaluation_goal_id = stock_spec.goal_id
    goal_specs: dict[str, dict[str, Any]] = {}
    for spec in proposal_specs:
        existing = goal_specs.get(spec.goal_id)
        if existing is not None:
            if spec.instruction not in existing["text_variants"]:
                existing["text_variants"].append(spec.instruction)
            if spec.proposal_family not in existing["proposal_families"]:
                existing["proposal_families"].append(spec.proposal_family)
        else:
            goal_specs[spec.goal_id] = {
                "canonical_instruction": spec.instruction,
                "text_variants": [spec.instruction],
                "proposal_families": [spec.proposal_family],
            }
    if context.mined_successor is not None:
        mined = context.mined_successor
        goal_specs.setdefault(
            str(mined["goal_id"]),
            {
                "canonical_instruction": str(mined["instruction"]),
                "text_variants": [str(mined["instruction"])],
                "proposal_families": [str(mined["proposal_family"])],
            },
        )
    goal_specs[evaluation_goal_id].update(
        {
            "canonical_instruction": context.full_instruction,
            "atoms": context.atoms,
        }
    )
    continuation_goal_id = next(
        (
            spec.goal_id
            for spec in proposal_specs
            if spec.instruction == continuation_instruction
        ),
        "continuation_atomic",
    )
    if continuation_goal_id not in goal_specs:
        goal_specs[continuation_goal_id] = {
            "canonical_instruction": continuation_instruction,
            "text_variants": [continuation_instruction],
            "proposal_families": ["continuation_only"],
        }
    goal_specs[continuation_goal_id]["atoms"] = [continuation_atom]

    is_stock_reference = torch.zeros(
        actions_norm.shape[0], dtype=torch.bool
    )
    is_stock_reference[0] = True
    group = {
        "schema": SCHEMA,
        "snapshot_id": (
            f"{context.source_trajectory_id}-pn"
            f"{context.source_policy_noise_seed}-d"
            f"{context.decision_index}-r{len(context.recovery_trace)}"
            f"-rk{recovery_key}-m{model_key}"
        ),
        "source_trajectory_id": context.source_trajectory_id,
        "decision_index": context.decision_index,
        "execution_horizon": execution_horizon,
        "evaluation_goal_id": evaluation_goal_id,
        "full_instruction": context.full_instruction,
        "goal_specs": goal_specs,
        "history_prefix_hidden": context.history_prefix_hidden,
        "history_prefix_mask": context.history_prefix_mask,
        "history_action_norm": context.history_action_norm,
        "history_behavior_prompt_id": context.history_behavior_prompt_id,
        "recovery_trace": context.recovery_trace,
        "actions_norm": actions_norm,
        "actions_env": actions_env,
        "executed_actions_norm": executed_actions_norm,
        "executed_actions_env": executed_actions_env,
        "execution_mask": execution_mask,
        "executed_lengths": executed_lengths_t,
        "proposal_prompt_id": proposal_ids,
        "proposal_family": proposal_families,
        "proposal_goal_id": proposal_goal_ids,
        "proposal_candidate_index": proposal_candidate_indices,
        "proposal_instruction": proposal_instructions,
        "is_stock_reference": is_stock_reference,
        "flow_noise_id": flow_noise_ids,
        "flow_noise_seed": proposal_seeds,
        "flow_noise": flow_noise,
        "q_before": q_before,
        "q_after": q_after_t,
        "object_pos_before": object_before,
        "object_pos_after": object_after_t,
        "object_mask": object_mask,
        "bits_before": bits_before,
        "bits_after": bits_after_t,
        "atom_mask": atom_mask,
        "next_observation": next_observations,
        "next_prefix_hidden_full": torch.stack(next_prefix_hidden),
        "next_prefix_mask_full": torch.stack(next_prefix_mask),
        "next_state_flat": torch.stack(next_states),
        "terminated": torch.tensor(
            terminated_after_block, dtype=torch.bool
        ),
        "truncated": torch.tensor(
            truncated_after_block, dtype=torch.bool
        ),
        "info_success": torch.tensor(
            info_success_after_block, dtype=torch.bool
        ),
        "atomic_continuation_success": continuation_atomic_success,
        "continuation_success": continuation_success,
        "continuation_steps": continuation_steps,
        "continuation_final_bits": continuation_final_bits,
        "continuation_mask": continuation_mask,
        "continuation_policy_id": (
            f"{model_id}|{continuation_goal_id}|h{continuation_horizon}|"
            f"noise{continuation_noise_seed}"
        ),
        "branch_weights": branch_weights,
        "dense_goal_object_body_name": goal_object_body_name or "",
        "dense_goal_receptacle_body_name": (
            goal_receptacle_body_name or ""
        ),
        "dense_goal_object_index": goal_object_index,
        "dense_goal_receptacle_index": goal_receptacle_index,
        "dense_goal_distance_before": dense_goal_distance_before,
        "dense_goal_distance_after": dense_goal_distance_after,
        "dense_goal_progress": dense_goal_progress,
        "dense_progress_margin": dense_progress_margin,
        "immediate_full_goal_progress": after_progress,
        "immediate_full_goal_delta": d_progress,
        "immediate_no_damage": no_damage,
        "immediate_beats_stock": beats_stock,
        "branch_weights_immediate": immediate_weights,
        "branch_weights_paired_continuation": continuation_weights,
        "paired_continuation_score": continuation_score,
        "paired_continuation_advantage": paired_continuation_advantage,
        "branch_advantage_margin": branch_advantage_margin,
        "continuation_selected_indices": selected_continuations,
        "continuation_noise_ids": continuation_seeds,
        "branch_reward": torch.tensor(branch_rewards),
        "sim_snapshot": context.snapshot,
        "current_observation": current_observation,
        "evaluation_atoms": context.atoms,
        "continuation_atom": continuation_atom,
        "restore_qc": restore_qc,
        "provenance": {
            "base_model_id": model_id,
            "num_inference_steps": int(
                runner.policy.config.num_inference_steps
            ),
            "source_policy_noise_seed": context.source_policy_noise_seed,
            "stock_policy_noise_seed": context.next_policy_noise_seed,
            "proposal_noise_seed": proposal_noise_seed,
            "continuation_noise_seed": continuation_noise_seed,
            "sampler": "lcwm.sampler.sample_chunks/euler_flow",
            "sampler_noise": (
                "torch.randn(float32,cpu_generator); exact tensor in flow_noise"
            ),
            "chunk_size": int(runner.policy.config.chunk_size),
            "max_action_dim": int(runner.policy.config.max_action_dim),
            "executed_action_dim": int(actions_norm.shape[-1]),
            "continuation_requested_count": continuation_count,
            "continuation_actual_count": len(selected_continuations),
            "continuation_selection": (
                "stock_then_first_per_family_then_branch_order"
            ),
            "continuation_common_random_numbers": True,
            "branch_weight_rule": (
                "no-damage predicate/dense-goal progress/stock advantage OR "
                "paired full-goal continuation success-time advantage"
            ),
            "mined_successor": (
                {
                    key: value
                    for key, value in context.mined_successor.items()
                    if key not in {"action_norm", "flow_noise"}
                }
                if context.mined_successor is not None
                else None
            ),
        },
        "action_convention": {
            "actions_norm": "PI05 normalized action space; transition/FM target",
            "actions_env": "postprocessed LIBERO action actually executed",
        },
    }
    return group
