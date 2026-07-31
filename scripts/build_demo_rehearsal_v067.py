#!/usr/bin/env python
"""V6.7.5 — genuine full-50 demonstration rehearsal cache.

Iteration 1 silently substituted a partial ordered slice of LoHo stock
rollouts for the registered demonstration role. This builds the real
bank: libero_10 expert demos (the data π0.5 was finetuned on), replayed
deterministically (recorded-state reset + recorded actions); at every
50-action boundary it stores the RAW observation dict + the next 50 env
actions + their policy-normalized chunk (exact runner-inverse affine
mean/std_eps, unchanged from seq_prefix_cache). Raw observations are
ALSO stored at every 10-action decision boundary (obs_10) so the
recurrent LC state can be unrolled along the demo at the deployed
decision stride.

Frozen split: per task, the first 10 replay-valid demo indices; first 8
are train, last 2 dev (index order, no selection).

Output: datasets/libero_loho_public_v1/demo_rehearsal_v067/
  task<k>_demo<d>.pt  + manifest.json  (run_schema=v067)
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import DATASETS_DIR, ensure_project_libero_config

ensure_project_libero_config()

OUT = Path("/home/stargazer/Desktop/vla_wm/datasets/libero_loho_public_v1"
           "/demo_rehearsal_v067")
CHUNK = 50
DEMOS_PER_TASK = 10
N_DEV = 2


def main() -> None:
    from lcwm.chassis import make_task_env, make_task_suite
    from lcwm.probe_data import valid_demo_indices
    from lcwm.seq_prefix_cache import normalize_actions
    from lcwm.snapshot import get_sim, reset_osc_controller
    from lcwm.v067_lineage import RUN_SCHEMA

    ref = torch.load(Path("/home/stargazer/Desktop/vla_wm/datasets"
                          "/seq_prefix_cache_v1/task0_demo0.pt"),
                     weights_only=False)
    mean, std_eps = ref["action_mean"], ref["action_std_eps"]

    OUT.mkdir(parents=True, exist_ok=True)
    suite = make_task_suite("libero_10")
    manifest = {"schema": "v067_demo_rehearsal_v1",
                "run_schema": RUN_SCHEMA, "chunk": CHUNK,
                "split_rule": (f"first {DEMOS_PER_TASK} replay-valid "
                               f"demos; last {N_DEV} by index order are "
                               "dev"), "episodes": []}
    for task_id in range(10):
        task = suite.get_task(task_id)
        h5_path = DATASETS_DIR / "libero_10" / f"{task.name}_demo.hdf5"
        demo_idx = valid_demo_indices("libero_10", task_id,
                                      DEMOS_PER_TASK)
        split_of = {di: ("dev" if k >= DEMOS_PER_TASK - N_DEV
                         else "train")
                    for k, di in enumerate(demo_idx)}
        env = make_task_env("libero_10", task_id)
        try:
            env.reset()
            language = env.task_description
            with h5py.File(h5_path, "r") as f:
                for di in demo_idx:
                    out_path = OUT / f"task{task_id}_demo{di}.pt"
                    if out_path.exists():
                        manifest["episodes"].append(
                            {"path": out_path.name, "task_id": task_id,
                             "demo": di, "split": split_of[di]})
                        continue
                    grp = f[f"data/demo_{di}"]
                    actions = np.asarray(grp["actions"])
                    states = np.asarray(grp["states"])
                    env._env.reset()
                    sim = get_sim(env)
                    sim.set_state_from_flattened(states[0].copy())
                    sim.forward()
                    reset_osc_controller(env)
                    raw = env._env
                    rows, obs_10 = [], []
                    for t in range(len(actions)):
                        if t % 10 == 0:
                            obs = env._format_raw_obs(
                                raw.env._get_observations())
                            obs_10.append(
                                {"t": t, "obs": copy.deepcopy(obs),
                                 "chunk10_env": torch.from_numpy(
                                     actions[t:t + 10].copy()).float()})
                            if (t % CHUNK == 0
                                    and t + CHUNK <= len(actions)):
                                chunk_env = torch.from_numpy(
                                    actions[t:t + CHUNK].copy()).float()
                                rows.append({
                                    "t": t,
                                    "obs": copy.deepcopy(obs),
                                    "chunk_env": chunk_env,
                                    "chunk_norm": normalize_actions(
                                        chunk_env, mean, std_eps),
                                })
                        _o, _r, done, _i = raw.step(actions[t])
                        if done:
                            break
                    tmp = out_path.with_suffix(".tmp")
                    torch.save({
                        "schema": "v067_demo_rehearsal_v1",
                        "run_schema": RUN_SCHEMA,
                        "task_id": task_id, "demo": di,
                        "split": split_of[di],
                        "language": language,
                        "action_mean": mean,
                        "action_std_eps": std_eps,
                        "rows": rows,
                        "obs_10": obs_10,
                    }, tmp)
                    tmp.replace(out_path)
                    manifest["episodes"].append(
                        {"path": out_path.name, "task_id": task_id,
                         "demo": di, "split": split_of[di],
                         "n_rows": len(rows)})
                    print(f"[demo] task{task_id} demo{di}: "
                          f"{len(rows)} full-50 rows "
                          f"({split_of[di]})", flush=True)
        finally:
            env.close()
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
