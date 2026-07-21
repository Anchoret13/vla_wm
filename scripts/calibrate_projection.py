#!/usr/bin/env python
"""Visual calibration of the object->patch projection (bounded diagnostic).

Projects every object's body position through lcwm.seg_masks.project_points and
draws markers on the RAW-oriented frame (the token-space view). PASS = every
marker sits on its object in the saved figure — checked by eye, per the Codex
re-review requirement ("direct visual calibration"). No policy load needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

from lcwm.chassis import make_task_env  # noqa: E402
from lcwm.probe_data import body_positions, discover_object_bodies  # noqa: E402
from lcwm.seg_masks import project_points  # noqa: E402


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, tid in zip(axes, [0, 2, 8]):
        env = make_task_env("libero_10", tid)
        obs, _ = env.reset(seed=1000)
        objs = discover_object_bodies(env)
        pos = body_positions(env, list(objs.values()))
        pix = project_points(env, pos)          # (n, 2) row,col in token space
        env.close()

        frame = obs["pixels"]["image"][::-1, ::-1]  # RAW/token orientation
        ax.imshow(frame)
        for (r, c), name in zip(pix, objs):
            ax.plot(c, r, "o", ms=10, mfc="none", mec="lime", mew=2)
            ax.annotate(name.replace("_1", ""), (c, r), color="yellow",
                        fontsize=7, xytext=(4, -4), textcoords="offset points")
        ax.set_title(f"task {tid}", fontsize=10)
        ax.axis("off")
    fig.suptitle("projection calibration — markers must sit ON objects", fontsize=11)
    fig.tight_layout()
    out = REPO_ROOT / "results" / "feature_controls" / "projection_calibration.png"
    fig.savefig(out, dpi=150)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
