"""A3.1 — p4_mixed_v1 target builder (registered 2026-07-27).

Seven separated target classes per labeled window (see the registration
table); branch siblings receive the SAME ΔΦ/no-damage semantics via the
parent episode's phase memos — never from an isolated endpoint pair.
"""

from __future__ import annotations

import torch
from torch import Tensor

from lcwm.phase_potential import (
    episode_phase_potentials,
    delta_phi_targets,
)
from lcwm.value_heads import (
    PI0_POLICY_ID,
    episode_value_targets,
    predicate_fraction_target,
    reward_target,
)


def sequential_targets(episode: dict) -> dict[str, Tensor]:
    """Per-window targets for one loader episode (T records → T-1 windows).

    Returns tensors [T-1] plus masks; ΔΦ carries its own validity mask
    (label-prefix-bounded), V^π0 is NaN outside π0 episodes.
    """
    bits = episode["predicate_bits"]
    T = bits.shape[0]
    mask = episode["label_mask"]
    window_labeled = mask[:-1] & mask[1:]

    records = episode_phase_potentials(
        episode["q"],
        episode["obj_pos"],
        bits,
        episode["object_names"],
        episode["goal_atoms"],
        mask,
    )
    dphi, dphi_valid, violation = delta_phi_targets(records)

    if "episode_truncated" in episode:  # full chain cache (A2)
        env_terminated = bool(episode.get("episode_terminated", False))
        truncated = bool(episode["episode_truncated"])
    elif "first_success_t" in episode:  # demo cache
        # Registered derivation: LIBERO terminates on success.
        env_terminated = episode["first_success_t"] is not None
        truncated = not env_terminated
    else:  # partial branch-history prefix: no terminal semantics exist
        env_terminated = truncated = False
    terminal = episode["terminal"][1:].bool()

    return {
        "window_labeled": window_labeled,
        "reward": reward_target(bits[:-1], bits[1:]),
        "predicate_fraction": predicate_fraction_target(bits[1:]),
        "dphi": dphi,
        "dphi_valid": dphi_valid & window_labeled,
        "no_damage_violation": violation,
        "success": bits[1:].bool().all(-1),
        "env_terminal": terminal & env_terminated,
        "truncated_final": terminal & (not env_terminated) & truncated,
        # A0 (2026-07-29): the transitioned value target is V^pi0(s_{t+1}) —
        # the NEXT state's value — so Q̂ = r̂_t + ΔΦ̂_t + γ·V̂_{t+1} neither
        # reuses the current-state value nor double-counts r_t. MC identity
        # V_t = r_t + γ·V_{t+1} holds exactly and is asserted in the smoke.
        "v_pi0": episode_value_targets(episode)[1:],
        "v_pi0_current": episode_value_targets(episode)[:-1],
        "generating_policy": episode.get("generating_policy", "unknown"),
        "source_id": episode["source_id"],
        "split": episode["split"],
        "n_windows": torch.tensor(T - 1),
    }


def branch_delta_phi(
    group: dict, sidecar: dict
) -> tuple[Tensor, Tensor, Tensor]:
    """Sibling ΔΦ with parent-episode memos (registered memo rule).

    Runs the potential over the sidecar-labeled prefix up to the snapshot
    decision, then evaluates each sibling's after-state as the alternative
    next record. Returns (dphi [N], valid [N], violation [N]).

    Validity: the snapshot decision must lie on the labeled pure rollout
    (contact groups). Stall-group snapshots sit after unlabeled recovery
    segments → all-invalid, by construction rather than approximation.
    """
    n = group["actions_norm"].shape[0]
    decision = int(group["decision_index"])
    dphi = torch.zeros(n)
    valid = torch.zeros(n, dtype=torch.bool)
    violation = torch.zeros(n, dtype=torch.bool)
    if decision >= sidecar["q"].shape[0]:
        return dphi, valid, violation
    # Snapshot must MATCH the sidecar state (pure-rollout check, same
    # tolerance as the bitwise alignment finding of 2026-07-25).
    if float((sidecar["q"][decision] - group["q_before"]).abs().max()) > 1e-4:
        return dphi, valid, violation

    prefix_q = sidecar["q"][: decision + 1]
    prefix_obj = sidecar["obj_pos"][: decision + 1]
    prefix_bits = sidecar["bits"][: decision + 1]
    object_names = sidecar["object_names"]
    goal_atoms = sidecar["goal_atoms"]
    object_count = prefix_obj.shape[1]

    for i in range(n):
        obj_after = group["object_pos_after"][i, :object_count]
        q_seq = torch.cat([prefix_q, group["q_after"][i][None]])
        obj_seq = torch.cat([prefix_obj, obj_after[None]])
        bits_after = group["bits_after"][i, : prefix_bits.shape[1]]
        bits_seq = torch.cat([prefix_bits, bits_after[None]])
        records = episode_phase_potentials(
            q_seq, obj_seq, bits_seq, object_names, goal_atoms
        )
        before, after = records[-2], records[-1]
        if before is None or after is None:
            continue
        dphi[i] = after.phi - before.phi
        valid[i] = True
        violation[i] = after.violation
    return dphi, valid, violation


def assert_policy_guard(targets: dict) -> None:
    """V^π0 targets may only be finite on π0-generated windows."""
    finite = ~torch.isnan(targets["v_pi0"])
    if bool(finite.any()) and targets["generating_policy"] != PI0_POLICY_ID:
        raise AssertionError(
            f"{targets['source_id']}: V^pi0 targets present under policy "
            f"{targets['generating_policy']!r}"
        )
