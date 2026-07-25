"""Full-prefix sequential episode cache (P1, contract seq_prefix_cache_v1).

The v2 sequential dataset (`seq_libero_10_v2`) has the per-decision labels the
world model needs but stores only pooled 128-token features, which the LC
posterior cannot consume. This module repeats the SAME deterministic replay
(restore ``states[0]`` + step raw HDF5 actions, capture at the c=10 decision
stride, terminal record appended before any reset) but stores the full
real-prompt prefix ``PrefixCache.hidden``/``pad_masks`` per decision — the
exact tensors `LCState.posterior`/`unroll_lc_history` consume, produced by the
exact live-eval pipeline (360x360 re-renders; no 128x128 HDF5-image shift).

Also stored per decision: env-convention AND policy-normalized action blocks
(mean/std normalization verified at build time against the runner's own
`NormalizerProcessorStep`), q(9), object positions, predicate bits, success
and terminal flags. Split labels come from the SEALED v2 split manifest;
audit episodes are refused unless explicitly unsealed by the caller.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from lcwm.libero_paths import DATASETS_DIR, ensure_project_libero_config

ensure_project_libero_config()

from lcwm.chassis import make_task_suite  # noqa: E402
from lcwm.probe_data import body_positions, discover_object_bodies  # noqa: E402
from lcwm.sampler import prefix_forward  # noqa: E402
from lcwm.seq_data import STRIDE, goal_atoms, predicate_bits  # noqa: E402
from lcwm.snapshot import get_sim, reset_osc_controller  # noqa: E402

CONTRACT_VERSION = "seq_prefix_cache_v1_20260725"
CACHE_DIR = Path(
    os.environ.get(
        "LCWM_PREFIX_CACHE_DIR", DATASETS_DIR.parent / "seq_prefix_cache_v1"
    )
)
SEQ_V2_DIR = Path(
    os.environ.get(
        "LCWM_SEQ_DIR", DATASETS_DIR.parent / "seq_libero_10_v2"
    )
)
SOURCE_FILES = (
    "lcwm/seq_prefix_cache.py",
    "lcwm/seq_data.py",
    "lcwm/chassis.py",
    "lcwm/sampler.py",
)
REPO_ROOT = Path(__file__).resolve().parent.parent


def source_sha256() -> str:
    digest = hashlib.sha256()
    for relative in SOURCE_FILES:
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update((REPO_ROOT / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def action_stats(runner) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract the runner's exact ACTION normalization as (mean, std+eps).

    Probed through the verified `_normalize_action` API with basis inputs
    instead of reaching into the step's stats layout: with mean-std
    normalization, n(0) = -mean/(std+eps) and n(1) = (1-mean)/(std+eps), so
    std+eps = 1/(n(1)-n(0)) and mean = -n(0)*(std+eps). Using std+eps as the
    stored divisor reproduces the step bit-for-bit with no separate eps.
    """
    from lerobot.processor.normalize_processor import NormalizerProcessorStep

    step = next(
        s
        for s in runner.preprocessor.steps
        if isinstance(s, NormalizerProcessorStep)
    )
    zeros = step._normalize_action(torch.zeros(1, 7), inverse=False).cpu()
    ones = step._normalize_action(torch.ones(1, 7), inverse=False).cpu()
    std_eps = 1.0 / (ones - zeros)[0]
    mean = -zeros[0] * std_eps
    return mean.float(), std_eps.float()


def normalize_actions(
    actions_env: torch.Tensor, mean: torch.Tensor, std_eps: torch.Tensor
) -> torch.Tensor:
    return (actions_env - mean) / std_eps


def load_split_roles(seq_dir: Path = SEQ_V2_DIR) -> dict:
    manifest = json.loads((seq_dir / "split_manifest.json").read_text())
    return manifest["tasks"]


@torch.no_grad()
def collect_prefix_episode(
    runner,
    env,
    atoms,
    body_names,
    h5_grp,
    task_language: str,
    stride: int = STRIDE,
) -> dict:
    """One episode: identical replay/capture semantics to seq_data.collect_demo,
    but capturing the full real-prompt prefix instead of pooled taps."""
    actions = np.asarray(h5_grp["actions"])
    rec_states = np.asarray(h5_grp["states"])
    total = len(actions)

    env._env.reset()
    sim = get_sim(env)
    sim.set_state_from_flattened(rec_states[0].copy())
    sim.forward()
    reset_osc_controller(env)
    raw = env._env

    records: list[dict] = []

    def capture(t: int, action_block: torch.Tensor, terminal: bool) -> None:
        obs = env._format_raw_obs(raw.env._get_observations())
        batch = runner._obs_to_policy_batch(obs, task_language)
        prefix = prefix_forward(runner.policy, batch)
        records.append(
            {
                "t": t,
                "hidden": prefix.hidden[0].half().cpu(),
                "mask": prefix.pad_masks[0].bool().cpu(),
                "action_block_env": action_block,
                "q": torch.from_numpy(
                    np.concatenate(
                        [
                            obs["robot_state"]["eef"]["pos"],
                            obs["robot_state"]["eef"]["quat"],
                            obs["robot_state"]["gripper"]["qpos"],
                        ]
                    )
                ).float(),
                "obj_pos": torch.from_numpy(
                    body_positions(env, body_names)
                ).float(),
                "predicate_bits": torch.from_numpy(
                    predicate_bits(env, atoms)
                ),
                "success": bool(raw.check_success()),
                "terminal": terminal,
            }
        )

    first_success_t, terminal_t, success = None, total, False
    t_end = 0
    for t in range(total):
        if t % stride == 0 and t + stride <= total:
            capture(
                t,
                torch.from_numpy(actions[t : t + stride].copy()).float(),
                terminal=False,
            )
        _obs, _reward, done, _info = raw.step(actions[t])
        t_end = t + 1
        ok = bool(raw.check_success())
        if ok and first_success_t is None:
            first_success_t = t
        success = success or ok
        if done:
            terminal_t = t + 1
            break
    # Successor state of the last executed block, captured BEFORE any reset
    # (v2 terminal-record semantics, review 2026-07-22 item 8).
    capture(t_end, torch.zeros(stride, 7), terminal=True)

    return {
        "t": torch.tensor([r["t"] for r in records], dtype=torch.long),
        "prefix_hidden": torch.stack([r["hidden"] for r in records]),
        "prefix_mask": torch.stack([r["mask"] for r in records]),
        "action_block_env": torch.stack(
            [r["action_block_env"] for r in records]
        ),
        "q": torch.stack([r["q"] for r in records]),
        "obj_pos": torch.stack([r["obj_pos"] for r in records]),
        "predicate_bits": torch.stack([r["predicate_bits"] for r in records]),
        "success": torch.tensor([r["success"] for r in records]),
        "terminal": torch.tensor([r["terminal"] for r in records]),
        "episode_success": success,
        "T_actions": total,
        "first_success_t": first_success_t,
        "terminal_t": terminal_t,
    }


@torch.no_grad()
def build_task_cache(
    runner,
    suite_name: str,
    task_id: int,
    demo_roles: dict[str, list[int]],
    roles: tuple[str, ...] = ("train", "dev"),
    cache_dir: Path = CACHE_DIR,
    overwrite: bool = False,
) -> list[dict]:
    """Build cache episodes for the demos of `roles` (sealed audit excluded
    unless explicitly listed by the caller)."""
    from lcwm.chassis import make_task_env

    suite = make_task_suite(suite_name)
    task = suite.get_task(task_id)
    h5_path = DATASETS_DIR / suite_name / f"{task.name}_demo.hdf5"
    cache_dir.mkdir(parents=True, exist_ok=True)
    mean, std_eps = action_stats(runner)

    # Build-time verification that manual mean/std equals the runner's step.
    from lerobot.processor.normalize_processor import NormalizerProcessorStep

    norm_step = next(
        s
        for s in runner.preprocessor.steps
        if isinstance(s, NormalizerProcessorStep)
    )
    probe = torch.randn(3, 10, 7)
    manual = normalize_actions(probe, mean, std_eps)
    reference = norm_step._normalize_action(probe.clone(), inverse=False)
    if not torch.allclose(manual, reference.cpu(), atol=1e-6):
        raise RuntimeError(
            "manual action normalization diverges from NormalizerProcessorStep"
        )

    env = make_task_env(suite_name, task_id)
    env.reset()
    obj_bodies = discover_object_bodies(env)
    atoms = goal_atoms(env)
    entries: list[dict] = []
    try:
        with h5py.File(h5_path, "r") as f:
            for role in roles:
                for di in demo_roles.get(role, []):
                    out = cache_dir / f"task{task_id}_demo{di}.pt"
                    entry = {
                        "path": out.name,
                        "suite": suite_name,
                        "task_id": task_id,
                        "demo": di,
                        "split": role,
                        "source_id": f"{suite_name}-task{task_id}-demo{di}",
                    }
                    if out.exists() and not overwrite:
                        entries.append(entry)
                        continue
                    key = f"demo_{di}"
                    if key not in f["data"]:
                        continue
                    rec = collect_prefix_episode(
                        runner,
                        env,
                        atoms,
                        list(obj_bodies.values()),
                        f["data"][key],
                        task.language,
                    )
                    rec["action_block_norm"] = normalize_actions(
                        rec["action_block_env"], mean, std_eps
                    )
                    torch.save(
                        {
                            "schema": CONTRACT_VERSION,
                            "suite": suite_name,
                            "task_id": task_id,
                            "demo": di,
                            "split": role,
                            "source_id": entry["source_id"],
                            "language": task.language,
                            "paraphrases": [],
                            "goal_atoms": atoms,
                            "object_names": list(obj_bodies),
                            "stride": STRIDE,
                            "action_mean": mean,
                            "action_std_eps": std_eps,
                            "provenance": {
                                "model_id": runner.model_id,
                                "source_sha256": source_sha256(),
                                "seq_v2_dir": str(SEQ_V2_DIR),
                            },
                            **rec,
                        },
                        out,
                    )
                    entries.append(entry)
                    print(
                        f"[prefix-cache] task{task_id} demo{di} ({role}): "
                        f"{rec['prefix_hidden'].shape[0]} records, "
                        f"L={rec['prefix_hidden'].shape[1]}, "
                        f"success={rec['episode_success']} "
                        f"first_t={rec['first_success_t']}",
                        flush=True,
                    )
    finally:
        env.close()
    return entries
