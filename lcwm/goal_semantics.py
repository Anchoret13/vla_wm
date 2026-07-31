"""v0.6.7 GoalSpec-specific terminal semantics (2026-07-31 V6.7.1).

Iteration-1 defect: `outcome_tuple` called `terminal_success(env)` — the
environment's original BDDL `goal_state` — for EVERY GoalSpec, so the
highest-priority element of the crossed continuation tuple was wrong for
every alternative goal whose terminal predicates differ from the
canonical task.

Repair:
- every GoalSpec carries explicit terminal predicates, derived from its
  non-`pick_up` ordered subgoals (place → in/on, open/close verbatim);
  `pick_up` is an event milestone and never a terminal predicate;
- `goal_terminal_success(env, preds)` evaluates THAT GoalSpec;
- the environment BDDL evaluator is used only for the matching canonical
  GoalSpec, with equality against the derived evaluator asserted;
- `SuccessTracker` records whether the terminal predicates were jointly
  satisfied at ANY point of a continuation (per-action resolution), not
  only at the final endpoint.
"""

from __future__ import annotations

import hashlib
import json

from lcwm.seq_data import _problem_env

Predicate = tuple[str, ...]


def terminal_predicates(subgoals: list[str]) -> list[Predicate]:
    preds: list[Predicate] = []
    for sg in subgoals:
        parts = sg.split()
        kind = parts[0]
        if kind == "pick_up":
            continue
        if kind == "place":
            predicate = "on" if parts[2].endswith("top_side") else "in"
            preds.append((predicate, parts[1], parts[2]))
        elif kind in ("open", "close"):
            preds.append((kind, parts[1]))
        else:
            raise ValueError(f"unknown subgoal kind {kind!r} in {sg!r}")
    if not preds:
        raise ValueError("GoalSpec has no terminal predicates")
    return preds


def terminal_predicate_hash(preds: list[Predicate]) -> str:
    payload = json.dumps([list(p) for p in preds], sort_keys=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def env_eval_fn(env):
    inner = _problem_env(env)
    return lambda p: bool(inner._eval_predicate(list(p)))


def goal_terminal_success(env, preds: list[Predicate],
                          eval_fn=None) -> bool:
    if eval_fn is None:
        eval_fn = env_eval_fn(env)
    return all(eval_fn(p) for p in preds)


def bddl_goal_predicates(env) -> list[Predicate]:
    inner = _problem_env(env)
    return [tuple(str(x) for x in a)
            for a in inner.parsed_problem["goal_state"]]


class SuccessTracker:
    """Any-point joint satisfaction of one GoalSpec's terminal predicates.

    `achieved` (-> success_by_100) latches on first joint satisfaction;
    `final` reflects the most recent check, so transient success followed
    by invalidation keeps achieved=True while final=False.
    """

    def __init__(self, preds: list[Predicate]):
        self.preds = list(preds)
        self.achieved = False
        self.final = False
        self.first_step: int | None = None

    def update(self, env, step: int, eval_fn=None) -> bool:
        now = goal_terminal_success(env, self.preds, eval_fn)
        if now and not self.achieved:
            self.first_step = step
        self.achieved = self.achieved or now
        self.final = now
        return now

    def state(self) -> dict:
        return {"achieved": self.achieved, "final": self.final,
                "first_step": self.first_step}
