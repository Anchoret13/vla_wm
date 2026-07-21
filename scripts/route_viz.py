#!/usr/bin/env python
"""Option-level route diversity: same state + same task, N different realizations.

For each chosen snapshot: sample N=16 candidate 50-step chunks from pi0.5
(shared prefix, lcwm.sampler), execute each OPEN-LOOP from the restored
snapshot, record the eef route + outcome, and draw all routes projected through
the calibrated camera onto the raw-oriented frame.

Outputs (results/route_viz/):
  routes_task{T}.png   frame + N projected eef routes, colored by outcome
  stats.json           pre-registered descriptive stats (2026-07-21.md)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

from lerobot.utils.constants import ACTION  # noqa: E402

from lcwm.chassis import Pi05Runner, make_task_env  # noqa: E402
from lcwm.sampler import sample_chunks  # noqa: E402
from lcwm.seg_masks import project_points  # noqa: E402
from lcwm.seq_data import goal_atoms, predicate_bits  # noqa: E402
from lcwm.snapshot import get_sim, reset_osc_controller  # noqa: E402

TASKS = [2, 0, 8]   # stove "button" | pick | failure task
N, SEED, HORIZON = 16, 7, 50


def to_env_action(runner, a_norm: torch.Tensor) -> np.ndarray:
    """One normalized action (7,) -> env action, exact chassis/select_action path."""
    action = runner.postprocessor(a_norm.unsqueeze(0))
    transition = runner.env_postprocessor({ACTION: action})
    out = transition[ACTION]
    return out[0].cpu().numpy() if torch.is_tensor(out) else np.asarray(out)[0]


def rollout_candidate(runner, env, snapshot, chunk) -> dict:
    raw = env._env
    raw.reset()                       # clear robosuite episode bookkeeping
    sim = get_sim(env)
    sim.set_state_from_flattened(snapshot.copy())
    sim.forward()
    reset_osc_controller(env)

    atoms = goal_atoms(env)
    eef = []
    success = False
    for t in range(min(HORIZON, chunk.shape[0])):
        obs_raw = raw.env._get_observations()
        eef.append(np.asarray(obs_raw["robot0_eef_pos"]).copy())
        _o, _r, done, _i = raw.step(to_env_action(runner, chunk[t]))
        success = success or bool(raw.check_success())
        if done:
            break
    obs_raw = raw.env._get_observations()
    eef.append(np.asarray(obs_raw["robot0_eef_pos"]).copy())
    return {
        "eef": np.stack(eef),
        "success": success,
        "final_bits": predicate_bits(env, atoms).tolist(),
    }


def main() -> None:
    out_dir = REPO_ROOT / "results" / "route_viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = Pi05Runner(suite_name="libero_10")
    stats = {}

    for tid in TASKS:
        env = make_task_env("libero_10", tid)
        obs, _ = env.reset(seed=1000)
        task_desc = env.task_description
        snapshot = np.asarray(get_sim(env).get_state().flatten()).copy()
        atoms0 = goal_atoms(env)
        bits0 = predicate_bits(env, atoms0).tolist()

        batch = runner._obs_to_policy_batch(obs, task_desc)
        chunks = sample_chunks(runner.policy, batch, n=N, seed=SEED)  # (N,50,7)

        rollouts = [rollout_candidate(runner, env, snapshot, chunks[i])
                    for i in range(N)]

        # --- pre-registered descriptive stats --------------------------------
        flat = chunks.reshape(N, -1)
        iu = torch.triu_indices(N, N, offset=1)
        pw_all = torch.cdist(flat, flat)[iu[0], iu[1]]
        flat10 = chunks[:, :10].reshape(N, -1)
        pw_10 = torch.cdist(flat10, flat10)[iu[0], iu[1]]
        finals = np.stack([r["eef"][-1] for r in rollouts])
        prefix_finals = np.stack([r["eef"][min(10, len(r["eef"]) - 1)]
                                  for r in rollouts])
        n_flip = sum(1 for r in rollouts if r["final_bits"] != bits0)
        stats[tid] = {
            "task": task_desc,
            "action_pairwise_l2_50": float(pw_all.mean()),
            "action_pairwise_l2_first10": float(pw_10.mean()),
            "end_eef_dispersion_cm": float(finals.std(0).mean() * 100),
            "eef_dispersion_at_t10_cm": float(prefix_finals.std(0).mean() * 100),
            "n_success": sum(r["success"] for r in rollouts),
            "n_predicate_changed": n_flip,
            "init_bits": bits0,
        }

        # --- figure ----------------------------------------------------------
        frame = obs["pixels"]["image"][::-1, ::-1]        # raw/token orientation
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(frame)
        cmap = plt.get_cmap("turbo", N)
        for i, r in enumerate(rollouts):
            pix = project_points(env, r["eef"])           # (T,2) row,col
            inb = (pix[:, 0] >= 0) & (pix[:, 0] < 360) & \
                  (pix[:, 1] >= 0) & (pix[:, 1] < 360)
            ax.plot(pix[inb, 1], pix[inb, 0], "-", color=cmap(i), lw=1.6,
                    alpha=0.85)
            if inb.any():
                last = np.where(inb)[0][-1]
                changed = r["final_bits"] != bits0
                ax.plot(pix[last, 1], pix[last, 0],
                        "*" if changed else "o", color=cmap(i),
                        ms=14 if changed else 7,
                        mec="white", mew=0.8)
        start = project_points(env, rollouts[0]["eef"][:1])[0]
        ax.plot(start[1], start[0], "s", color="white", ms=10, mec="black")
        s = stats[tid]
        ax.set_title(
            f'task {tid}: "{task_desc[:58]}"\n'
            f'N={N} same-option realizations | end-eef disp '
            f'{s["end_eef_dispersion_cm"]:.1f}cm (t10: '
            f'{s["eef_dispersion_at_t10_cm"]:.1f}cm) | '
            f'{s["n_predicate_changed"]} predicate-changed (★)', fontsize=10)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"routes_task{tid}.png", dpi=150)
        plt.close(fig)
        env.close()
        print(f"[routes] task{tid}: {json.dumps(stats[tid], default=str)[:200]}",
              flush=True)

    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
