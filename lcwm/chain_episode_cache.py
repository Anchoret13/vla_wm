"""A2 — full chain-episode cache (contract chain_episode_cache_v1).

Deterministic replay of a stock chain source episode (recorded seed +
per-decision flow noise) capturing, at every decision boundary:

- full real-prompt prefix hidden/mask (the LC posterior's native input);
- the RAW observation dict (uint8 images + robot state) — the P5
  observation-side interface, enabling exact prefix/KV recomputation under
  any instruction without a live environment;
- the exact stock 50-step chunk (recorded noise) and its executed first-ten
  block in env and normalized conventions;
- live labels q/obj_pos/bits, ASSERTED equal to the existing relabel
  sidecars at every decision (determinism regression, free).

This is feature capture over already-used rollout states, not new
interaction (2026-07-25.md A2 registration). Split assignments come from the
branch manifest and are unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from lcwm.libero_paths import DATASETS_DIR, ensure_project_libero_config

ensure_project_libero_config()

from lcwm.branch_collect import _clone_observation, _execute_env_block  # noqa: E402
from lcwm.probe_data import body_positions, discover_object_bodies  # noqa: E402
from lcwm.sampler import prefix_forward, sample_chunks  # noqa: E402
from lcwm.seq_data import goal_atoms, predicate_bits  # noqa: E402
from lcwm.seq_prefix_cache import action_stats, normalize_actions  # noqa: E402

CONTRACT_VERSION = "chain_episode_cache_v1_20260727"
CHAIN_EPISODE_CACHE_DIR = Path(
    os.environ.get(
        "LCWM_CHAIN_EPISODE_CACHE_DIR",
        DATASETS_DIR.parent / "chain_episode_cache_v1",
    )
)
EXECUTION_HORIZON = 10


@torch.no_grad()
def build_chain_episode(
    runner,
    env,
    seed: int,
    noise_base: int,
    sidecar: dict | None = None,
) -> dict:
    """One full stock episode with prefix features, raw obs, chunks, labels.

    When `sidecar` (the chain_source_labels_v1 record) is given, live labels
    are asserted equal to it at every decision — any mismatch aborts the
    build rather than silently shipping divergent features.
    """
    runner.reset()
    obs, _ = env.reset(seed=seed)
    atoms = goal_atoms(env)
    bodies = discover_object_bodies(env)
    body_names = list(bodies.values())
    mean, std_eps = action_stats(runner)

    hiddens, masks, raw_observations = [], [], []
    chunks_norm, executed_env, executed_lengths = [], [], []
    qs, obj_positions, bits_list, terminals = [], [], [], []

    def capture(t: int, decision: int, terminal: bool) -> None:
        q_now = torch.from_numpy(
            np.concatenate(
                [
                    obs["robot_state"]["eef"]["pos"],
                    obs["robot_state"]["eef"]["quat"],
                    obs["robot_state"]["gripper"]["qpos"],
                ]
            )
        ).float()
        obj_now = torch.from_numpy(
            body_positions(env, body_names)
        ).float()
        bits_now = torch.from_numpy(predicate_bits(env, atoms))
        if sidecar is not None:
            # Codex review 2026-07-27: assert the FULL fidelity claim —
            # q, object poses, and predicate bits, at every decision.
            if decision >= sidecar["q"].shape[0]:
                raise RuntimeError(
                    f"seed {seed}: replay produced decision {decision} "
                    f"beyond sidecar length {sidecar['q'].shape[0]}"
                )
            q_error = float((sidecar["q"][decision] - q_now).abs().max())
            obj_error = float(
                (sidecar["obj_pos"][decision] - obj_now).abs().max()
            )
            if (
                q_error > 1e-4
                or obj_error > 1e-4
                or not torch.equal(
                    sidecar["bits"][decision].bool(), bits_now.bool()
                )
            ):
                raise RuntimeError(
                    f"seed {seed} decision {decision}: replay diverged from "
                    f"sidecar (q err {q_error:.2e}, obj err {obj_error:.2e})"
                    " — determinism broken"
                )
        qs.append(q_now)
        obj_positions.append(obj_now)
        bits_list.append(bits_now)
        terminals.append(terminal)
        raw_observations.append(_clone_observation(obs))

    t = decision = 0
    terminated = truncated = False
    while t < env.episode_length:
        capture(t, decision, terminal=False)
        batch = runner._obs_to_policy_batch(obs, env.task_description)
        prefix = prefix_forward(runner.policy, batch)
        hiddens.append(prefix.hidden[0].half().cpu())
        masks.append(prefix.pad_masks[0].bool().cpu())
        chunk = sample_chunks(
            runner.policy, batch, n=1,
            seed=noise_base + decision, prefix=prefix,
        )
        chunks_norm.append(chunk[0].float().cpu())
        actions_env = runner.chunk_to_env(chunk[:, :EXECUTION_HORIZON])
        obs, executed, terminated, truncated, _r, _i = _execute_env_block(
            env, actions_env, EXECUTION_HORIZON
        )
        executed_env.append(
            torch.from_numpy(np.asarray(actions_env)).float()
        )
        executed_lengths.append(int(executed))
        t += executed
        decision += 1
        if terminated or truncated:
            break
    # Terminal record: state after the final executed block, before reset.
    capture(t, decision, terminal=True)
    if sidecar is not None and len(qs) != sidecar["q"].shape[0]:
        raise RuntimeError(
            f"seed {seed}: replay produced {len(qs)} records, sidecar has "
            f"{sidecar['q'].shape[0]} — exact-length check failed"
        )
    batch = runner._obs_to_policy_batch(obs, env.task_description)
    prefix = prefix_forward(runner.policy, batch)
    hiddens.append(prefix.hidden[0].half().cpu())
    masks.append(prefix.pad_masks[0].bool().cpu())

    executed_env_t = torch.stack(executed_env)
    return {
        "schema": CONTRACT_VERSION,
        "task": "chain3_lr2",
        "seed": seed,
        "policy_noise_base": noise_base,
        "generating_policy": "pi05_full_prompt_frozen",
        "language": env.task_description,
        "goal_atoms": atoms,
        "object_names": list(bodies),
        "stride": EXECUTION_HORIZON,
        "prefix_hidden": torch.stack(hiddens),
        "prefix_mask": torch.stack(masks),
        "raw_observations": raw_observations,
        "chunks_norm": torch.stack(chunks_norm),
        "action_block_env": executed_env_t,
        "action_block_norm": normalize_actions(
            executed_env_t, mean, std_eps
        ),
        "executed_lengths": torch.tensor(
            executed_lengths, dtype=torch.long
        ),
        "action_mean": mean,
        "action_std_eps": std_eps,
        "q": torch.stack(qs),
        "obj_pos": torch.stack(obj_positions),
        "predicate_bits": torch.stack(bits_list),
        "terminal": torch.tensor(terminals),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "num_inference_steps": runner.policy.config.num_inference_steps,
        "model_id": runner.model_id,
    }
