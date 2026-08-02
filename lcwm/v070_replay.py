"""V7.0.0 — strict repeat-wise replay-fidelity contract.

The V6.9 gate used `paired_preference(replay, u_0) == 0` as a fidelity
test. That helper implements both-repeats-agree CANDIDATE preference:
it returns 0 for repeat sign patterns such as [+1,-1] and [+1,0], which
are replay instabilities, not agreements. The strict rule is

    clean(s) = 1[ for every repeat r:
                  pref(y_replay_r, y_0_r) == 0 ]
               and 1[ absolute gross bounds pass ].

`paired_preference` remains valid for candidate preference only when
both-repeats-agree semantics are intended.
"""

from __future__ import annotations

from lcwm.task_automaton import pref

GATE_VERSION = "v070_strict_1"

# absolute gross restore-failure bounds (registered 2026-08-01,
# unchanged from the V6.9.2 amendment)
GROSS_BOUNDS = {"eef_pos": 0.02, "eef_quat": 7.8e-3,
                "gripper": 5e-3, "obj_pos": 2e-3}


def repeat_signs(replay_reps: list[dict], u0_reps: list[dict],
                 tolerances: dict) -> list[int]:
    """Per-repeat preference sign of the exact replay vs u_0."""
    assert len(replay_reps) == len(u0_reps) and replay_reps
    return [pref(yr, y0, tolerances)
            for yr, y0 in zip(replay_reps, u0_reps)]


def strict_replay_clean(replay_reps: list[dict], u0_reps: list[dict],
                        tolerances: dict,
                        endpoint_deltas: dict) -> tuple[bool, dict]:
    signs = repeat_signs(replay_reps, u0_reps, tolerances)
    gross = [k for k, bound in GROSS_BOUNDS.items()
             if endpoint_deltas[k] > bound]
    clean = all(s == 0 for s in signs) and not gross
    return clean, {"repeat_signs": signs,
                   "gross_failures": gross,
                   "gate_version": GATE_VERSION}
