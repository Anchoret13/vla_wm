"""Decision-rate sequential dataset for LCWM training (framework v0.2 §9 source 1).

Replays expert demos and captures at the decision stride c=10 everything the
hidden-state trainer (§11.2) needs. Fixes the audit's label gaps: per-step BDDL
predicate bits, first_success_t, actual terminal_t — and adds the action blocks
missing from the probe shards.

Per (task, demo) shard:
  meta : suite, task_id, demo, language, goal_atoms, object/body names,
         replay success, first_success_t, terminal_t, T_actions
  steps (every c env steps, captured BEFORE executing the block):
         t, {siglip, const_ll, real_ll} pooled (128,2048) fp16, e_lang fp16,
         q (9,), obj_pos (n_obj,3), action_block (c,7) f32,
         predicate_bits (n_atoms,) bool, success bool

Consecutive steps are 10 env-steps apart -> the trainer forms
(x_t, a_block_t, x_{t+1}) transition pairs directly. Trailing partial blocks are
dropped. Storage ≈ 1.5 MB/step -> ~5.5 GB for 10×10 demos.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from lcwm.libero_paths import DATASETS_DIR, ensure_project_libero_config

ensure_project_libero_config()

from lcwm.chassis import Pi05Runner, make_task_env, make_task_suite  # noqa: E402
from lcwm.probe_data import (body_positions, discover_object_bodies,  # noqa: E402
                             frame_features, valid_demo_indices)
from lcwm.snapshot import get_sim, reset_osc_controller  # noqa: E402

SEQ_DIR = DATASETS_DIR.parent / "seq_libero_10"
STRIDE = 10  # decision commitment c


def _problem_env(env):
    """The bddl problem env exposing parsed_problem / _eval_predicate."""
    return env._env.env


def goal_atoms(env) -> list[list[str]]:
    return [list(a) for a in _problem_env(env).parsed_problem["goal_state"]]


def predicate_bits(env, atoms) -> np.ndarray:
    inner = _problem_env(env)
    return np.array([bool(inner._eval_predicate(a)) for a in atoms], dtype=bool)


def collect_demo(runner, env, atoms, body_names, h5_grp, task_language: str,
                 stride: int = STRIDE) -> dict:
    actions = np.asarray(h5_grp["actions"])
    rec_states = np.asarray(h5_grp["states"])
    T = len(actions)

    env._env.reset()
    sim = get_sim(env)
    sim.set_state_from_flattened(rec_states[0].copy())
    sim.forward()
    reset_osc_controller(env)

    raw = env._env
    steps, first_success_t, terminal_t, success = [], None, T, False
    for t in range(T):
        if t % stride == 0 and t + stride <= T:
            obs = env._format_raw_obs(raw.env._get_observations())
            feats = frame_features(runner, obs, task_language)
            steps.append({
                **feats,
                "t": t,
                "q": torch.from_numpy(np.concatenate([
                    obs["robot_state"]["eef"]["pos"],
                    obs["robot_state"]["eef"]["quat"],
                    obs["robot_state"]["gripper"]["qpos"],
                ])).float(),
                "obj_pos": torch.from_numpy(
                    body_positions(env, body_names)).float(),
                "action_block": torch.from_numpy(
                    actions[t:t + stride].copy()).float(),
                "predicate_bits": torch.from_numpy(
                    predicate_bits(env, atoms)),
                "success": bool(raw.check_success()),
            })
        _o, _r, done, _i = raw.step(actions[t])
        ok = bool(raw.check_success())
        if ok and first_success_t is None:
            first_success_t = t
        success = success or ok
        if done:
            terminal_t = t + 1
            break
    return {
        "steps": steps, "success": success, "T_actions": T,
        "first_success_t": first_success_t, "terminal_t": terminal_t,
    }


def collect_task(runner, suite_name: str, task_id: int,
                 demo_indices: list[int]) -> list[Path]:
    suite = make_task_suite(suite_name)
    task = suite.get_task(task_id)
    h5_path = DATASETS_DIR / suite_name / f"{task.name}_demo.hdf5"
    SEQ_DIR.mkdir(parents=True, exist_ok=True)

    env = make_task_env(suite_name, task_id)
    env.reset()
    obj_bodies = discover_object_bodies(env)
    atoms = goal_atoms(env)
    written = []
    with h5py.File(h5_path, "r") as f:
        for di in demo_indices:
            out = SEQ_DIR / f"task{task_id}_demo{di}.pt"
            if out.exists():
                written.append(out)
                continue
            key = f"demo_{di}"
            if key not in f["data"]:
                continue
            rec = collect_demo(runner, env, atoms, list(obj_bodies.values()),
                               f["data"][key], task.language)
            torch.save({
                "suite": suite_name, "task_id": task_id, "demo": di,
                "language": task.language, "goal_atoms": atoms,
                "object_names": list(obj_bodies), "stride": STRIDE, **rec,
            }, out)
            written.append(out)
            print(f"[seq] task{task_id} demo{di}: {len(rec['steps'])} steps "
                  f"success={rec['success']} first_t={rec['first_success_t']}",
                  flush=True)
    env.close()
    return written


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="libero_10")
    p.add_argument("--tasks", type=int, nargs="*", default=list(range(10)))
    p.add_argument("--demos-per-task", type=int, default=10)
    args = p.parse_args()

    runner = Pi05Runner(suite_name=args.suite)
    manifest = {}
    for tid in args.tasks:
        idx = valid_demo_indices(args.suite, tid, args.demos_per_task)
        paths = collect_task(runner, args.suite, tid, idx)
        manifest[tid] = [p.name for p in paths]
    (SEQ_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"manifest -> {SEQ_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
