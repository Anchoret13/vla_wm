#!/usr/bin/env python
"""Decisive token-grid orientation test (audit finding #4, 2026-07-20).

The image path is: raw render -> LiberoEnv flips 180° ("for visualization") ->
obs pixels -> LiberoProcessorStep flips 180° AGAIN -> policy/SigLIP. So tokens
should live in RAW orientation, i.e. rotated 180° relative to obs["pixels"].

Test: black out the TOP-LEFT quadrant of the obs (flipped) agentview frame,
run both frames through the exact policy preprocessing + embed_image, and see
WHICH patch quadrant changes. If the change lands bottom-right in token space,
tokens are 180°-rotated vs obs (raw orientation) — confirming the audit.
"""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

from lcwm.chassis import Pi05Runner, make_task_env  # noqa: E402

GRID = 16


@torch.no_grad()
def agentview_siglip(runner, obs):
    batch = runner._obs_to_policy_batch(obs, "orientation check")
    images, img_masks = runner.policy._preprocess_images(batch)
    model = runner.policy.model
    toks = [model.paligemma_with_expert.embed_image(img)[0].float()
            for img, m in zip(images, img_masks) if bool(m[0])]
    return toks[0]  # first camera = agentview, (256, d)


def quadrant_energy(delta_map):
    h = GRID // 2
    return {
        "top_left": float(delta_map[:h, :h].mean()),
        "top_right": float(delta_map[:h, h:].mean()),
        "bottom_left": float(delta_map[h:, :h].mean()),
        "bottom_right": float(delta_map[h:, h:].mean()),
    }


def main() -> None:
    runner = Pi05Runner(suite_name="libero_10")
    env = make_task_env("libero_10", 0)
    obs, _ = env.reset(seed=1000)
    env.close()

    base = agentview_siglip(runner, obs)

    mod = deepcopy(obs)
    h, w, _ = mod["pixels"]["image"].shape
    mod["pixels"]["image"] = mod["pixels"]["image"].copy()
    mod["pixels"]["image"][: h // 2, : w // 2] = 0  # black top-left of OBS frame
    changed = agentview_siglip(runner, mod)

    delta = (base - changed).norm(dim=-1).reshape(GRID, GRID).cpu().numpy()
    q = quadrant_energy(delta)
    print("per-quadrant mean |Δtoken| after blacking OBS top-left:")
    for k, v in q.items():
        print(f"  {k:13s} {v:8.3f}")
    verdict = max(q, key=q.get)
    print(f"\nmax-change quadrant: {verdict}")
    if verdict == "bottom_right":
        print("=> tokens are 180° ROTATED vs obs (RAW orientation). Audit confirmed:")
        print("   seg_masks must NOT flip; all prior overlays were misaligned.")
    elif verdict == "top_left":
        print("=> tokens share obs orientation; audit's rotation concern not realized.")
    else:
        print("=> unexpected quadrant — investigate preprocessing further.")


if __name__ == "__main__":
    main()
