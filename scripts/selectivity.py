#!/usr/bin/env python
"""§11.1 target-object selectivity of the task shift (framework §3).

On the same rollout frames as feature_controls, splits the per-token shift maps
δ(t1_same_scene) and δ(const) into object groups via projected patch masks:

  t0_targets  = alphabet_soup, tomato_sauce   (targets of the CANONICAL prompt)
  t1_targets  = cream_cheese, butter          (targets of the SWAPPED prompt)
  shared_goal = basket
  distractors = ketchup, orange_juice, milk
  rest        = background + robot

Task-identifier prediction: δ(t1) is elevated on t0_targets ∪ t1_targets (objects
whose ROLE changed) relative to distractors/rest; δ(const) is less selective.
Output: results/feature_controls/selectivity.json + overlay figure.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
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

from lcwm.chassis import Pi05Runner, make_task_env, make_task_suite  # noqa: E402
from lcwm.probe_data import discover_object_bodies, body_positions  # noqa: E402
from lcwm.sampler import prefix_forward  # noqa: E402
from lcwm.seg_masks import group_means, patch_disk_labels  # noqa: E402
from lcwm.taps import CONSTANT_PROMPT  # noqa: E402

SUITE, TASK_ID = "libero_10", 0
CAPTURE_STEPS = [0, 80, 160]
GRID = 16
GROUPS = {
    "t0_targets": ["alphabet_soup_1", "tomato_sauce_1"],
    "t1_targets": ["cream_cheese_1", "butter_1"],
    "shared_goal": ["basket_1"],
    "distractors": ["ketchup_1", "orange_juice_1", "milk_1"],
}


@torch.no_grad()
def agentview_tokens(runner, obs, prompt):
    pre = prefix_forward(runner.policy, runner._obs_to_policy_batch(obs, prompt))
    h = pre.hidden[0, : pre.n_img_tokens].float()
    valid = pre.pad_masks[0, : pre.n_img_tokens].bool()
    return h[valid][: GRID * GRID]  # first camera = agentview


def cosdist_map(a, b):
    c = 1 - torch.nn.functional.cosine_similarity(a, b, dim=-1)
    return c.reshape(GRID, GRID).cpu().numpy()


def main() -> None:
    out_dir = REPO_ROOT / "results" / "feature_controls"
    runner = Pi05Runner(suite_name=SUITE)
    suite = make_task_suite(SUITE)
    canon = suite.get_task(TASK_ID).language
    t1 = suite.get_task(1).language

    env = make_task_env(SUITE, TASK_ID)
    obs, _ = env.reset(seed=1000)
    obj_bodies = discover_object_bodies(env)
    body_names = list(obj_bodies.values())
    obj_names = list(obj_bodies)

    report, figs = {}, []
    runner.reset()
    for t in range(max(CAPTURE_STEPS) + 1):
        if t in CAPTURE_STEPS:
            pos = body_positions(env, body_names)
            masks = patch_disk_labels(
                env, {n: pos[i] for i, n in enumerate(obj_names)})
            ob = deepcopy(obs)
            base = agentview_tokens(runner, ob, canon)
            maps = {
                "t1_shift": cosdist_map(base, agentview_tokens(runner, ob, t1)),
                "const_shift": cosdist_map(base, agentview_tokens(runner, ob, CONSTANT_PROMPT)),
            }
            report[t] = {k: group_means(m, masks, GROUPS) for k, m in maps.items()}
            figs.append((t, ob["pixels"]["image"], maps, masks))
        obs, _r, term, trunc, _i = env.step(runner.select_action(obs, canon))
        if term or trunc:
            break
    env.close()

    # figure: frame + masks outline + the two shift maps
    fig, axes = plt.subplots(len(figs), 3, figsize=(11, 3.6 * len(figs)))
    axes = np.atleast_2d(axes)
    vmax = max(m.max() for _, _, mm, _ in figs for m in mm.values())
    for r, (t, frame, maps, masks) in enumerate(figs):
        frame = frame[::-1, ::-1]  # rotate obs to RAW/token orientation for display
        axes[r, 0].imshow(frame)
        colors = {"t0_targets": "cyan", "t1_targets": "lime",
                  "shared_goal": "yellow", "distractors": "white"}
        for gname, members in GROUPS.items():
            gm = np.zeros((GRID, GRID), bool)
            for o in members:
                gm |= masks.get(o, gm)
            ys, xs = np.where(gm)
            axes[r, 0].scatter((xs + 0.5) * frame.shape[1] / GRID,
                               (ys + 0.5) * frame.shape[0] / GRID,
                               s=10, c=colors[gname], marker="s", alpha=0.7,
                               label=gname if r == 0 else None)
        axes[r, 0].set_title(f"frame t={t} + projected masks", fontsize=9)
        axes[r, 0].axis("off")
        for j, key in enumerate(["t1_shift", "const_shift"]):
            ax = axes[r, j + 1]
            ax.imshow(frame)
            im = ax.imshow(maps[key], cmap="magma", alpha=0.55, vmin=0, vmax=vmax,
                           extent=(0, frame.shape[1], frame.shape[0], 0),
                           interpolation="bilinear")
            g = report[t][key]
            ax.set_title(f"{key}  tgt {g['t0_targets']:.2f}/{g['t1_targets']:.2f} "
                         f"dis {g['distractors']:.2f} rest {g['rest']:.2f}", fontsize=8)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046)
    axes[0, 0].legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / "selectivity.png", dpi=150)

    (out_dir / "selectivity.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"-> {out_dir}/selectivity.png")


if __name__ == "__main__":
    main()
