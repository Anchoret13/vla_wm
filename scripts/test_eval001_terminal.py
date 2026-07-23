#!/usr/bin/env python
"""EVAL-001 regression test (required repair item 3, 2026-07-23.md).

A known successful terminal transition MUST be serialized as success=True,
all goal bits true, Q=1 — i.e. the evaluator must score the PRE-reset terminal
state. Method: teleport the three chain3 goal objects into the basket, run
run_chain_episode with a null policy; is_success fires on the first decision
check and the recorded result must reflect the success state, not a reset.

Also asserts the negative control: without teleporting, a zero-action episode
records success=False with live (non-reset) bits.

No policy checkpoint is loaded — this test runs in ~1 min on env setup alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

from lcwm.loho import make_chain_env, run_chain_episode  # noqa: E402
from lcwm.probe_data import body_positions, discover_object_bodies  # noqa: E402
from lcwm.snapshot import get_sim, reset_osc_controller  # noqa: E402

GOAL_OBJS = ["alphabet_soup_1", "tomato_sauce_1", "cream_cheese_1"]


class NullRunner:
    """Zero-action stand-in for Pi05Runner: isolates the evaluator path."""

    def reset(self) -> None:
        pass

    def select_action(self, obs, task_description) -> np.ndarray:
        return np.zeros(7, dtype=np.float64)


def teleport_into_basket(env) -> None:
    sim = get_sim(env)
    objs = discover_object_bodies(env)
    basket = body_positions(env, [objs["basket_1"]])[0]
    # Side-by-side placement INSIDE the basket: vertical stacking topples the
    # third object out during the settle steps (first version of this test).
    offsets = [(-0.035, 0.0), (0.035, 0.0), (0.0, 0.035)]
    for (dx, dy), name in zip(offsets, GOAL_OBJS):
        joint = f"{name}_joint0"
        pos = basket + np.array([dx, dy, 0.05])
        sim.data.set_joint_qpos(joint, np.concatenate([pos, [1, 0, 0, 0]]))
    sim.forward()
    reset_osc_controller(env)


def main() -> None:
    runner = NullRunner()
    env = make_chain_env("chain3_lr2")

    # -- negative control: nothing placed, zero actions, truncation ----------
    env_len = 30
    env.episode_length = env_len
    res_neg = run_chain_episode(runner, env, "full", seed=1000)
    assert res_neg.success is False, "negative control must not succeed"
    assert res_neg.q_score == 0.0, f"expected Q=0, got {res_neg.q_score}"
    print(f"NEGATIVE: success={res_neg.success} q={res_neg.q_score} "
          f"steps={res_neg.steps}  ok")

    # -- known-success terminal transition -----------------------------------
    # run_chain_episode resets internally, so re-enter via a wrapper that
    # teleports right after reset: monkey-patch env.reset once.
    original_reset = env.reset

    def reset_and_teleport(*a, **kw):
        out = original_reset(*a, **kw)
        teleport_into_basket(env)
        return out

    env.reset = reset_and_teleport
    res = run_chain_episode(runner, env, "full", seed=1000)
    env.reset = original_reset
    env.close()

    print(f"TERMINAL: success={res.success} q={res.q_score} steps={res.steps} "
          f"final_bits_timeline={res.bits_timeline[-1]}")
    assert res.success is True, (
        "EVAL-001 REGRESSION: known-success terminal transition recorded as "
        "failure — the evaluator is scoring a reset state again")
    assert res.q_score == 1.0, f"expected Q=1.0, got {res.q_score}"
    assert all(res.bits_timeline[-1][1]), "terminal bits must all be true"
    assert res.steps <= 10, "success should terminate at the first check"
    print("EVAL-001 REGRESSION TEST PASSED")


if __name__ == "__main__":
    main()
