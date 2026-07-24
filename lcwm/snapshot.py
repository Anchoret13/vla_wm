"""Stage-3 snapshot / restore (plan §4).

MuJoCo state round-trips through flatten/set_state_from_flattened; the OSC controller
keeps internal interpolation targets, so after every restore we must reset its goal to
the current (restored) eef pose — otherwise the arm lurches toward a stale target.
Every stored snapshot carries a `replay_valid` flag (filled by replay_validity).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import copy

import numpy as np


def _copy_sim_data(sim: Any, name: str) -> np.ndarray | None:
    value = getattr(sim.data, name, None) if hasattr(sim, "data") else None
    return None if value is None else np.asarray(value).copy()


_CONTROLLER_ARRAY_STATE = (
    "goal_pos",
    "goal_ori",
    "ori_ref",
    "relative_ori",
    "initial_joint",
    "initial_ee_pos",
    "initial_ee_ori_mat",
    "torques",
    "kp",
    "kd",
)


def _robots(lenv: Any) -> list[Any]:
    raw = _raw_env(lenv)
    return list(getattr(raw, "robots", None) or raw.env.robots)


def _snapshot_controller_state(lenv: Any) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for robot in _robots(lenv):
        controller = getattr(robot, "controller", None)
        if controller is None:
            states.append({})
            continue
        state: dict[str, Any] = {
            "new_update": bool(getattr(controller, "new_update", True)),
        }
        for name in _CONTROLLER_ARRAY_STATE:
            value = getattr(controller, name, None)
            state[name] = copy.deepcopy(value)
        for name in ("interpolator_pos", "interpolator_ori"):
            interpolator = getattr(controller, name, None)
            if interpolator is None:
                state[name] = None
            else:
                state[name] = {
                    key: copy.deepcopy(getattr(interpolator, key, None))
                    for key in ("start", "goal", "step")
                }
        states.append(state)
    return states


def _snapshot_robot_state(lenv: Any) -> list[dict[str, Any]]:
    """Capture mutable robot state that MuJoCo does not own.

    In robosuite's Panda gripper, ``current_action`` is an integrator. Leaving
    it live makes the same restored state/action drift once per replay.
    """
    states: list[dict[str, Any]] = []
    for robot in _robots(lenv):
        gripper = getattr(robot, "gripper", None)
        states.append(
            {
                "gripper_current_action": copy.deepcopy(
                    getattr(gripper, "current_action", None)
                )
            }
        )
    return states


def _restore_robot_state(
    lenv: Any, states: list[dict[str, Any]] | None
) -> None:
    if states is None:
        return
    robots = _robots(lenv)
    if len(robots) != len(states):
        raise ValueError("snapshot robot count differs from environment")
    for robot, state in zip(robots, states, strict=True):
        value = state.get("gripper_current_action")
        gripper = getattr(robot, "gripper", None)
        if value is not None and gripper is not None:
            gripper.current_action = copy.deepcopy(value)


def _restore_controller_state(
    lenv: Any, states: list[dict[str, Any]] | None
) -> bool:
    if states is None:
        return False
    robots = _robots(lenv)
    if len(robots) != len(states):
        raise ValueError("snapshot controller count differs from environment")
    for robot, state in zip(robots, states, strict=True):
        controller = getattr(robot, "controller", None)
        if controller is None:
            continue
        controller.update(force=True)
        for name in _CONTROLLER_ARRAY_STATE:
            if name in state:
                setattr(controller, name, copy.deepcopy(state[name]))
        for name in ("interpolator_pos", "interpolator_ori"):
            saved = state.get(name)
            interpolator = getattr(controller, name, None)
            if saved is None or interpolator is None:
                continue
            for key, value in saved.items():
                setattr(interpolator, key, copy.deepcopy(value))
        controller.new_update = bool(state.get("new_update", True))
    return True


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


def _robosuite_env(lenv: Any):
    """Return the env that owns robosuite's episode bookkeeping.

    In the current stack ``lenv._env`` is LIBERO's ``OffScreenRenderEnv`` and
    its ``.env`` is the robosuite environment.  Tolerating extra wrapper levels
    keeps snapshot restore usable with older / slightly different stacks.
    """
    env = _raw_env(lenv)
    seen = set()
    while id(env) not in seen:
        seen.add(id(env))
        if all(hasattr(env, name) for name in ("timestep", "cur_time", "done")):
            return env
        nested = getattr(env, "env", None)
        if nested is None:
            break
        env = nested
    return env


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
    # Appended (rather than inserted) to preserve the original positional
    # constructor.  None lets restore accept snapshots saved before these
    # robosuite bookkeeping fields existed.
    env_timestep: int | None = None
    env_cur_time: float | None = None
    env_done: bool | None = None
    # MuJoCo's flattened MjSimState does not include these data arrays.  They
    # can otherwise leak the previously executed sibling into a restored run.
    sim_ctrl: np.ndarray | None = None
    sim_qfrc_applied: np.ndarray | None = None
    sim_xfrc_applied: np.ndarray | None = None
    sim_mocap_pos: np.ndarray | None = None
    sim_mocap_quat: np.ndarray | None = None
    sim_qacc_warmstart: np.ndarray | None = None
    controller_state: list[dict[str, Any]] | None = None
    robot_state: list[dict[str, Any]] | None = None


def snap(lenv: Any, t: int, suite_name: str, task_id: int, **meta) -> SimSnapshot:
    sim = get_sim(lenv)
    episode_env = _robosuite_env(lenv)
    return SimSnapshot(
        state=np.asarray(sim.get_state().flatten(), dtype=np.float64).copy(),
        suite_name=suite_name,
        task_id=task_id,
        task_description=getattr(lenv, "task_description", ""),
        t=t,
        meta=meta,
        env_timestep=getattr(episode_env, "timestep", None),
        env_cur_time=getattr(episode_env, "cur_time", None),
        env_done=getattr(episode_env, "done", None),
        sim_ctrl=_copy_sim_data(sim, "ctrl"),
        sim_qfrc_applied=_copy_sim_data(sim, "qfrc_applied"),
        sim_xfrc_applied=_copy_sim_data(sim, "xfrc_applied"),
        sim_mocap_pos=_copy_sim_data(sim, "mocap_pos"),
        sim_mocap_quat=_copy_sim_data(sim, "mocap_quat"),
        sim_qacc_warmstart=_copy_sim_data(sim, "qacc_warmstart"),
        controller_state=_snapshot_controller_state(lenv),
        robot_state=_snapshot_robot_state(lenv),
    )


def restore(lenv: Any, snapshot: SimSnapshot) -> None:
    """Restore physics, episode bookkeeping, controller, and observables.

    No settle steps are run: restoring an exact mid-episode state must not
    advance physics. Callers that restore an *init* state (t == 0 semantics)
    should run their own settle noops.
    """
    sim = get_sim(lenv)
    sim.set_state_from_flattened(snapshot.state.copy())
    for snapshot_name, data_name in (
        ("sim_ctrl", "ctrl"),
        ("sim_qfrc_applied", "qfrc_applied"),
        ("sim_xfrc_applied", "xfrc_applied"),
        ("sim_mocap_pos", "mocap_pos"),
        ("sim_mocap_quat", "mocap_quat"),
        ("sim_qacc_warmstart", "qacc_warmstart"),
    ):
        value = getattr(snapshot, snapshot_name, None)
        if value is not None and hasattr(sim, "data"):
            getattr(sim.data, data_name)[:] = value
    sim.forward()

    episode_env = _robosuite_env(lenv)
    for snapshot_name, env_name in (
        ("env_timestep", "timestep"),
        ("env_cur_time", "cur_time"),
        ("env_done", "done"),
    ):
        # getattr(default) is intentional: old pickles / snapshot-like objects
        # do not have the fields added above.
        value = getattr(snapshot, snapshot_name, None)
        if value is not None and hasattr(episode_env, env_name):
            setattr(episode_env, env_name, value)

    restored_controller = _restore_controller_state(
        lenv, getattr(snapshot, "controller_state", None)
    )
    _restore_robot_state(lenv, getattr(snapshot, "robot_state", None))
    if not restored_controller:
        reset_osc_controller(lenv)
    update_observables = getattr(episode_env, "_update_observables", None)
    if callable(update_observables):
        # Direct sim writes bypass robosuite's normal per-step sensor update.
        # Force-refresh so sibling branches cannot inherit cached observations
        # from the branch that ran immediately before them.
        update_observables(force=True)
