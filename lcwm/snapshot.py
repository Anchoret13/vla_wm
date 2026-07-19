"""Stage-3 snapshot / restore (plan §4).

MuJoCo state round-trips through flatten/set_state_from_flattened; the OSC controller
keeps internal interpolation targets, so after every restore we must reset its goal to
the current (restored) eef pose — otherwise the arm lurches toward a stale target.
Every stored snapshot carries a `replay_valid` flag (filled by replay_validity).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _raw_env(lenv: Any):
    """lerobot LiberoEnv -> libero OffScreenRenderEnv (create lazily if needed)."""
    if getattr(lenv, "_env", None) is None:
        # LiberoEnv builds the underlying env lazily; reset() triggers it.
        lenv.reset()
    return lenv._env


def get_sim(lenv: Any):
    """robosuite MjSim handle, robust to one level of wrapping difference."""
    raw = _raw_env(lenv)
    sim = getattr(raw, "sim", None)
    if sim is None:
        sim = raw.env.sim
    return sim


def reset_osc_controller(lenv: Any) -> None:
    """Re-anchor OSC interpolation state to the *current* sim pose (robosuite 1.4)."""
    raw = _raw_env(lenv)
    robots = getattr(raw, "robots", None) or raw.env.robots
    for robot in robots:
        controller = getattr(robot, "controller", None)
        if controller is None:
            continue
        controller.update(force=True)
        controller.reset_goal()


@dataclass
class SimSnapshot:
    state: np.ndarray                 # sim.get_state().flatten()
    suite_name: str
    task_id: int
    task_description: str
    t: int                            # env step at capture time
    replay_valid: bool | None = None  # filled by replay_validity protocol
    meta: dict = field(default_factory=dict)


def snap(lenv: Any, t: int, suite_name: str, task_id: int, **meta) -> SimSnapshot:
    sim = get_sim(lenv)
    return SimSnapshot(
        state=np.asarray(sim.get_state().flatten(), dtype=np.float64).copy(),
        suite_name=suite_name,
        task_id=task_id,
        task_description=getattr(lenv, "task_description", ""),
        t=t,
        meta=meta,
    )


def restore(lenv: Any, snapshot: SimSnapshot) -> None:
    """set_state -> forward -> OSC re-anchor. No settle steps here: restoring an
    exact mid-episode state must not advance physics; callers that restore an
    *init* state (t == 0 semantics) should run their own settle noops."""
    sim = get_sim(lenv)
    sim.set_state_from_flattened(snapshot.state.copy())
    sim.forward()
    reset_osc_controller(lenv)
