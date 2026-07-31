"""Unit regressions for sibling-branch snapshot restore."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from lcwm.snapshot import SimSnapshot, restore, snap


class _State:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float64)

    def flatten(self):
        return self.value.copy()


class _Sim:
    def __init__(self, value, events):
        self.value = np.asarray(value, dtype=np.float64)
        self.events = events

    def get_state(self):
        return _State(self.value)

    def set_state_from_flattened(self, value):
        self.events.append("set_state")
        self.value = np.asarray(value, dtype=np.float64).copy()

    def forward(self):
        self.events.append("forward")


class _Controller:
    def __init__(self, events):
        self.events = events
        self.goal_pos = np.array([0.1, 0.2, 0.3])
        self.goal_ori = np.eye(3)
        self.new_update = True
        self.interpolator_pos = SimpleNamespace(
            start=np.zeros(3), goal=self.goal_pos.copy(), step=2
        )
        self.interpolator_ori = None

    def update(self, force=False):
        self.events.append(("controller_update", force))

    def reset_goal(self):
        self.events.append("controller_reset_goal")


class _EpisodeEnv:
    def __init__(self, sim, events):
        self.sim = sim
        self.events = events
        self.timestep = 17
        self.cur_time = 0.85
        self.done = False
        self.robots = [
            SimpleNamespace(
                controller=_Controller(events),
                gripper=SimpleNamespace(current_action=np.array([0.25])),
            )
        ]
        self.observation = None

    def _update_observables(self, force=False):
        self.events.append(("observables", force))
        self.observation = {
            "sim": self.sim.value.copy(),
            "timestep": self.timestep,
            "cur_time": self.cur_time,
            "done": self.done,
        }


class _RawWrapper:
    def __init__(self, episode_env):
        self.env = episode_env
        self.sim = episode_env.sim


def _make_env(initial_state=(1.0, 2.0)):
    events = []
    sim = _Sim(initial_state, events)
    episode_env = _EpisodeEnv(sim, events)
    lenv = SimpleNamespace(
        _env=_RawWrapper(episode_env),
        task_description="test task",
    )
    return lenv, episode_env, sim, events


def test_restore_round_trips_episode_state_and_refreshes_observables():
    lenv, episode_env, sim, events = _make_env()
    snapshot = snap(
        lenv,
        t=17,
        suite_name="test_suite",
        task_id=3,
        source="branch_root",
    )

    np.testing.assert_array_equal(snapshot.state, [1.0, 2.0])
    assert snapshot.env_timestep == 17
    assert snapshot.env_cur_time == 0.85
    assert snapshot.env_done is False

    sim.value[:] = [9.0, 9.0]
    controller = episode_env.robots[0].controller
    gripper = episode_env.robots[0].gripper
    controller.goal_pos[:] = 9.0
    controller.interpolator_pos.step = 99
    gripper.current_action[:] = 9.0
    episode_env.timestep = 92
    episode_env.cur_time = 4.6
    episode_env.done = True
    episode_env.observation = {"stale": True}

    restore(lenv, snapshot)

    np.testing.assert_array_equal(sim.value, [1.0, 2.0])
    assert episode_env.timestep == 17
    assert episode_env.cur_time == 0.85
    assert episode_env.done is False
    np.testing.assert_array_equal(episode_env.observation["sim"], [1.0, 2.0])
    assert episode_env.observation["timestep"] == 17
    assert episode_env.observation["cur_time"] == 0.85
    assert episode_env.observation["done"] is False
    np.testing.assert_array_equal(controller.goal_pos, [0.1, 0.2, 0.3])
    assert controller.interpolator_pos.step == 2
    np.testing.assert_array_equal(gripper.current_action, [0.25])
    assert events == [
        "set_state",
        "forward",
        ("controller_update", True),
        ("observables", True),
    ]


def test_restore_accepts_pre_bookkeeping_snapshot():
    """Old positional construction and old snapshot-like objects still work."""

    positional = SimSnapshot(
        np.array([3.0, 4.0]),
        "old_suite",
        1,
        "old task",
        5,
        True,
        {"format": "old"},
    )
    assert positional.replay_valid is True
    assert positional.meta == {"format": "old"}
    assert positional.env_timestep is None

    lenv, episode_env, sim, _events = _make_env(initial_state=(8.0, 8.0))
    episode_env.timestep = 31
    episode_env.cur_time = 1.55
    episode_env.done = True
    old_snapshot = SimpleNamespace(state=np.array([3.0, 4.0]))

    restore(lenv, old_snapshot)

    np.testing.assert_array_equal(sim.value, [3.0, 4.0])
    # Missing fields mean "leave current bookkeeping unchanged".
    assert episode_env.timestep == 31
    assert episode_env.cur_time == 1.55
    assert episode_env.done is True
    assert episode_env.observation["timestep"] == 31
