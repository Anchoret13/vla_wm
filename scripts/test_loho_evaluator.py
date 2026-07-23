"""Focused regressions for terminal-state accounting in the chained exam."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np

import lcwm.loho as loho


class _Runner:
    def __init__(self):
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1

    def select_action(self, _obs, _instruction):
        return np.zeros(7, dtype=np.float32)


class _OneStepEnv:
    task = "test_chain"
    task_description = "put both objects in the basket"
    episode_length = 1

    def reset(self, seed=None):
        del seed
        return {"frame": 0}, {}

    def step(self, _action):
        return {"frame": 1}, 0.0, True, False, {"is_success": False}


def test_chain_env_step_does_not_auto_reset_on_success():
    """The chain wrapper must leave the terminal simulator state readable."""

    inner = Mock()
    raw_terminal_obs = {"terminal": True}
    inner.step.return_value = (raw_terminal_obs, 1.0, False, {})
    inner.check_success.return_value = True

    env = object.__new__(loho.ChainEnv)
    env._env = inner
    env.task = "test_chain"
    env.task_id = 7
    env._ensure_env = Mock()
    env._format_raw_obs = Mock(return_value={"formatted_terminal": True})
    env.reset = Mock()

    obs, reward, terminated, truncated, info = env.step(
        np.zeros(7, dtype=np.float32)
    )

    assert obs == {"formatted_terminal": True}
    assert reward == 1.0
    assert terminated is True
    assert truncated is False
    assert info == {
        "task": "test_chain",
        "task_id": 7,
        "done": False,
        "is_success": True,
    }
    env.reset.assert_not_called()


def test_episode_keeps_terminal_partial_q_instead_of_rereading():
    """Final Q must use the terminal capture, not a later/reset simulator read."""

    env = _OneStepEnv()
    runner = _Runner()
    atoms = [["In", "object_a", "basket"], ["In", "object_b", "basket"]]
    reads = iter(
        [
            np.array([False, False]),  # initial state
            np.array([True, False]),  # terminal state
        ]
    )
    old_goal_atoms = loho.goal_atoms
    old_predicate_bits = loho.predicate_bits
    loho.goal_atoms = lambda _env: atoms
    loho.predicate_bits = lambda _env, _atoms: next(reads)
    try:
        result = loho.run_chain_episode(
            runner, env, condition="full", seed=1000, stride=10
        )
    finally:
        loho.goal_atoms = old_goal_atoms
        loho.predicate_bits = old_predicate_bits

    assert result.success is False
    assert result.q_score == 0.5
    assert result.bits_timeline[-1] == (1, [True, False])


def test_terminal_success_info_survives_base_env_auto_reset():
    """Compatibility guard for callers that still pass LeRobot's base env."""

    class _AutoResetSuccessEnv(_OneStepEnv):
        def step(self, _action):
            # A base LiberoEnv has already reset by the time control returns, so
            # predicate_bits() below sees the all-false initial state.
            return {"terminal_frame": 1}, 1.0, True, False, {"is_success": True}

    env = _AutoResetSuccessEnv()
    runner = _Runner()
    atoms = [["In", "object_a", "basket"], ["In", "object_b", "basket"]]
    old_goal_atoms = loho.goal_atoms
    old_predicate_bits = loho.predicate_bits
    loho.goal_atoms = lambda _env: atoms
    loho.predicate_bits = lambda _env, _atoms: np.array([False, False])
    try:
        result = loho.run_chain_episode(
            runner, env, condition="full", seed=1000, stride=10
        )
    finally:
        loho.goal_atoms = old_goal_atoms
        loho.predicate_bits = old_predicate_bits

    assert result.success is True
    assert result.q_score == 1.0
    assert result.bits_timeline[-1] == (1, [True, True])
