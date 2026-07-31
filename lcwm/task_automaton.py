"""v0.6 task automaton — replaces monotone ever-completed Q
(2026-07-30 V6.2 contract).

Per registered subgoal it tracks:
- EVENT milestones (pick_up; first-achievement of open/close/place):
  sticky — a grasp stays achieved after a correct release;
- CURRENT validity (place/open/close predicates re-evaluated every check):
  revocable — a placed object knocked out of its target invalidates;
- 0→1 and 1→0 flips with steps; damage = unrecovered 1→0 flips;
- ordered-prefix over current validity; valid Q; time-to-next-milestone.

Predicate names are lowercase in LIBERO's VALIDATE_PREDICATE_FN_DICT;
`*top_side` regions use On semantics (H7.1 protocol, unchanged).
Immediate object distance is a physical auxiliary only — never an
advantage component.
"""

from __future__ import annotations

import copy

import numpy as np

from lcwm.probe_data import body_positions, discover_object_bodies
from lcwm.seq_data import _problem_env

PICK_DISPLACEMENT = 0.02
PICK_LIFT = 0.03
TAU_CENSOR = 101


class GoalAutomaton:
    """One automaton per (episode-or-branch, GoalSpec)."""

    def __init__(self, subgoals: list[str]):
        self.subgoals = list(subgoals)
        self.n = len(subgoals)
        self.events_achieved: dict[int, int] = {}   # index -> step
        self.flips: list[tuple[int, int, int]] = []  # (step, index, +1/-1)
        self.prev_valid: list[bool] = [False] * self.n
        self.timeline: list[tuple[int, list[bool]]] = []
        self.bodies = None
        self.start_pos = None

    # ---- lifecycle ------------------------------------------------------
    def start(self, env) -> None:
        self.bodies = discover_object_bodies(env)
        self.start_pos = {}
        for name in {sg.split()[1] for sg in self.subgoals
                     if sg.split()[0] == "pick_up"}:
            self.start_pos[name] = body_positions(
                env, [self.bodies[name]])[0].copy()

    def fork(self) -> "GoalAutomaton":
        """Branch-time copy sharing body handles but with independent
        event/flip/validity state."""
        child = GoalAutomaton(self.subgoals)
        child.bodies = self.bodies
        child.start_pos = self.start_pos
        child.events_achieved = dict(self.events_achieved)
        child.prev_valid = list(self.prev_valid)
        return child

    # ---- evaluation -----------------------------------------------------
    def _subgoal_true(self, env, index: int) -> bool:
        inner = _problem_env(env)
        parts = self.subgoals[index].split()
        kind = parts[0]
        if kind == "place":
            predicate = "on" if parts[2].endswith("top_side") else "in"
            return bool(inner._eval_predicate([predicate, parts[1],
                                               parts[2]]))
        if kind in ("open", "close"):
            return bool(inner._eval_predicate([kind, parts[1]]))
        if kind == "pick_up":
            pos = body_positions(env, [self.bodies[parts[1]]])[0]
            start = self.start_pos[parts[1]]
            return (float(np.linalg.norm(pos - start)) >= PICK_DISPLACEMENT
                    and float(pos[2] - start[2]) >= PICK_LIFT)
        raise ValueError(f"unknown subgoal kind {kind}")

    def evaluate(self, env, step: int) -> list[bool]:
        """Current validity per subgoal; records events and flips."""
        valid = []
        for i, sg in enumerate(self.subgoals):
            true_now = self._subgoal_true(env, i)
            if true_now and i not in self.events_achieved:
                self.events_achieved[i] = step
            if sg.split()[0] == "pick_up":
                v = i in self.events_achieved   # event semantics: sticky
            else:
                v = true_now                    # predicate: revocable
            if v and not self.prev_valid[i]:
                self.flips.append((step, i, +1))
            elif not v and self.prev_valid[i]:
                self.flips.append((step, i, -1))
            valid.append(v)
        self.prev_valid = valid
        self.timeline.append((step, list(valid)))
        return valid

    # ---- outcome components --------------------------------------------
    def q_valid(self) -> float:
        return sum(self.prev_valid) / self.n

    def ordered_prefix(self) -> int:
        depth = 0
        for v in self.prev_valid:
            if v:
                depth += 1
            else:
                break
        return depth

    def p_valid(self) -> float:
        return self.ordered_prefix() / self.n

    def damage_unrecovered(self) -> int:
        """1→0 flips whose subgoal is not currently valid."""
        dropped = {i for (_s, i, d) in self.flips if d == -1}
        return sum(1 for i in dropped if not self.prev_valid[i])

    def tau_next(self, after_step: int) -> int:
        """Steps from after_step to the next 0→1 flip; censored."""
        for s, _i, d in self.flips:
            if d == +1 and s > after_step:
                return min(s - after_step, TAU_CENSOR)
        return TAU_CENSOR

    def reward_since(self, prev_valid_count: int) -> float:
        return sum(self.prev_valid) - prev_valid_count


def terminal_success(env) -> bool:
    inner = _problem_env(env)
    return all(bool(inner._eval_predicate(list(a)))
               for a in inner.parsed_problem["goal_state"])


def outcome_tuple(automaton: GoalAutomaton, env, branch_step: int,
                  q_at_horizons: dict[int, float]) -> dict:
    """Registered outcome tuple Y (lexicographic order as written)."""
    horizons = sorted(q_at_horizons)
    return {
        "success_by_100": bool(terminal_success(env)),
        "neg_damage": -automaton.damage_unrecovered(),
        "p_valid_100": automaton.p_valid(),
        "q_valid_mean": sum(q_at_horizons[h] for h in horizons)
        / len(horizons),
        "neg_tau_next": -automaton.tau_next(branch_step),
        "q_at_horizons": dict(q_at_horizons),
    }


def pref(y_i: dict, y_j: dict, tolerances: dict) -> int:
    """Lexicographic preference in the registered component order.
    Continuous components must differ beyond their frozen replay-noise
    tolerance to count; otherwise the comparison falls through (tie)."""
    order = ("success_by_100", "neg_damage", "p_valid_100",
             "q_valid_mean", "neg_tau_next")
    for key in order:
        a, b = float(y_i[key]), float(y_j[key])
        tol = float(tolerances.get(key, 0.0))
        if a > b + tol:
            return 1
        if b > a + tol:
            return -1
    return 0


def paired_preference(y_i_reps: list[dict], y_j_reps: list[dict],
                      tolerances: dict) -> int:
    """A_ij with the registered both-repeats-agree rule: ±1 only when every
    paired repeat agrees in direction beyond tolerance; else 0."""
    signs = [pref(a, b, tolerances)
             for a, b in zip(y_i_reps, y_j_reps)]
    if all(s == 1 for s in signs):
        return 1
    if all(s == -1 for s in signs):
        return -1
    return 0


def fork_env_state(automaton: GoalAutomaton) -> dict:
    """Serializable automaton state for provenance records."""
    return {
        "events_achieved": dict(automaton.events_achieved),
        "prev_valid": list(automaton.prev_valid),
        "flips": list(automaton.flips),
    }


def restore_env_state(automaton: GoalAutomaton, state: dict) -> None:
    automaton.events_achieved = dict(state["events_achieved"])
    automaton.prev_valid = list(state["prev_valid"])
    automaton.flips = [tuple(f) for f in state["flips"]]


def clone_for_branch(automaton: GoalAutomaton) -> GoalAutomaton:
    child = automaton.fork()
    child.flips = copy.deepcopy(automaton.flips)
    return child
