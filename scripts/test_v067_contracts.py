#!/usr/bin/env python
"""V6.7.0/V6.7.1 — registered contract tests + end-to-end dry fixture.

The dry fixture (tests 1 and 5) must FAIL on the two iteration-1 defects:
the candidate-specific continuation seed (`+ branch_index * 40`) and the
canonical-only distinct-goal success path. The remaining tests certify
the repaired sibling-CRN and GoalSpec terminal semantics. CPU-only; no
simulator, no π0.5.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import lcwm.task_automaton as ta  # noqa: E402
from lcwm.goal_semantics import (SuccessTracker,  # noqa: E402
                                 goal_terminal_success,
                                 terminal_predicate_hash,
                                 terminal_predicates)
from lcwm.v067_lineage import (Iter1ArtifactError,  # noqa: E402
                               assert_v067_payload, cont_seed_v067,
                               flow_noise, noise_sha)

PASS = []


def check(name: str, fn) -> None:
    fn()
    PASS.append(name)
    print(f"  [pass] {name}", flush=True)


# ---- fixtures ------------------------------------------------------------

def old_cont_seed(task_index, snap_decision, branch_index, repeat,
                  cont_decision):
    """Iteration-1 formula (defective: branch identity enters)."""
    return (50_000_000 + task_index * 2_000_000 + snap_decision * 1_000
            + branch_index * 40 + repeat * 20 + cont_decision)


class FakePredEnv:
    """Predicate-table environment for evaluator tests (no simulator)."""

    def __init__(self, table: dict):
        self.table = dict(table)

    def eval_fn(self, p):
        return bool(self.table.get(tuple(p), False))


CANON = ["pick_up butter_1", "place butter_1 cab_top_region",
         "close cab_top_region"]
DISTINCT = ["pick_up butter_2", "place butter_2 cab_top_region",
            "close cab_top_region"]


# ---- 1. dry fixture: old seed formula is candidate-specific --------------

def test_old_seed_breaks_sibling_crn():
    shas = set()
    for branch_index in range(5):
        seed = old_cont_seed(0, 12, branch_index, 0, 0)
        shas.add(noise_sha(flow_noise(seed, 50, 32)))
    assert len(shas) == 5, "old formula unexpectedly shared noise"
    # the v067 sibling assertion (bit-identical noise SHA across
    # branches) therefore REJECTS the iteration-1 stream
    def sibling_assert(sha_lists):
        first = sha_lists[0]
        for other in sha_lists[1:]:
            assert other == first, "sibling CRN violated"
    try:
        sibling_assert([[noise_sha(flow_noise(
            old_cont_seed(0, 12, b, 0, 0), 50, 32))] for b in range(5)])
    except AssertionError:
        return
    raise AssertionError("dry fixture failed to reject the old seed")


# ---- 2. repaired CRN: sibling-identical, otherwise distinct --------------

def test_new_seed_sibling_identity_and_separation():
    base = dict(run_id="v067_r1", source_id="loho_t1_drawer_stock_s2000",
                snap_decision=12, goal_spec_id="t1_canonical", repeat=0,
                cont_decision=3)
    s0 = cont_seed_v067(**base)
    # branch identity is structurally absent -> identical by construction
    assert cont_seed_v067(**base) == s0
    for field, value in [("source_id", "loho_t1_drawer_staged_s2001"),
                         ("goal_spec_id", "t1_back_butter"),
                         ("repeat", 1), ("cont_decision", 4),
                         ("snap_decision", 13), ("run_id", "v067_r2")]:
        assert cont_seed_v067(**{**base, field: value}) != s0, field
    n1 = flow_noise(s0, 50, 32)
    n2 = flow_noise(s0, 50, 32)
    assert torch.equal(n1, n2)
    assert noise_sha(n1) == noise_sha(n2)
    # replicates the sampler's seeded draw exactly
    g = torch.Generator(device="cpu").manual_seed(s0)
    ref = torch.randn((1, 50, 32), generator=g, dtype=torch.float32)
    assert torch.equal(n1, ref)


# ---- 3. terminal predicates: milestones excluded, hash stable ------------

def test_terminal_predicate_derivation():
    preds = terminal_predicates(CANON)
    assert preds == [("in", "butter_1", "cab_top_region"),
                     ("close", "cab_top_region")]
    assert all(p[0] != "pick_up" for p in preds)
    top = terminal_predicates(["place bowl_1 cab_top_side"])
    assert top == [("on", "bowl_1", "cab_top_side")]
    h1 = terminal_predicate_hash(preds)
    assert h1 == terminal_predicate_hash(
        terminal_predicates(CANON)) and len(h1) == 16
    assert h1 != terminal_predicate_hash(terminal_predicates(DISTINCT))


# ---- 4. crossed endpoint: same physics, goal-specific success ------------

def crossed_env():
    return FakePredEnv({("in", "butter_1", "cab_top_region"): True,
                        ("close", "cab_top_region"): True,
                        ("in", "butter_2", "cab_top_region"): False})


def test_crossed_success_differs_by_goal():
    env = crossed_env()
    canon = terminal_predicates(CANON)
    dist = terminal_predicates(DISTINCT)
    assert goal_terminal_success(None, canon, env.eval_fn) is True
    assert goal_terminal_success(None, dist, env.eval_fn) is False


# ---- 5. dry fixture: canonical-only success path mislabels ---------------

def test_old_success_path_fails_crossed_check():
    """Iteration-1 `outcome_tuple` called terminal_success(env) — the
    canonical BDDL — for every GoalSpec. At the crossed endpoint the old
    path assigns the DISTINCT goal the canonical verdict (True), while
    the goal-specific evaluator says False; the fixture requires the
    crossed check to reject that labeling."""
    env = crossed_env()
    old_label_for_distinct = goal_terminal_success(
        None, terminal_predicates(CANON), env.eval_fn)   # what iter-1 did
    new_label_for_distinct = goal_terminal_success(
        None, terminal_predicates(DISTINCT), env.eval_fn)
    assert old_label_for_distinct != new_label_for_distinct, (
        "fixture endpoint no longer separates the two evaluators")


# ---- 6. transient success is latched; final fields keep later state ------

def test_transient_success_latched():
    env = FakePredEnv({})
    dist = terminal_predicates(DISTINCT)
    tracker = SuccessTracker(dist)
    tracker.update(None, 1, env.eval_fn)
    assert not tracker.achieved
    env.table = {("in", "butter_2", "cab_top_region"): True,
                 ("close", "cab_top_region"): True}
    tracker.update(None, 7, env.eval_fn)
    assert tracker.achieved and tracker.final and tracker.first_step == 7
    env.table[("in", "butter_2", "cab_top_region")] = False  # knocked out
    tracker.update(None, 30, env.eval_fn)
    assert tracker.achieved, "success_by_100 must remain true"
    assert not tracker.final, "final validity must retain the later state"
    assert tracker.first_step == 7


# ---- 7. per-action resolution catches inside-chunk flips -----------------

def test_per_action_resolution():
    env = FakePredEnv({})
    auto = ta.GoalAutomaton(["place butter_2 cab_top_region",
                             "close cab_top_region"])
    auto.bodies, auto.start_pos = {}, {}
    orig = ta.GoalAutomaton._subgoal_true
    ta.GoalAutomaton._subgoal_true = (
        lambda self, _env, i: env.eval_fn(
            terminal_predicates([self.subgoals[i]])[0]))
    try:
        # chunk of 10 actions; predicate true only during steps 3..6
        chunk_auto = ta.GoalAutomaton(auto.subgoals)
        chunk_auto.bodies, chunk_auto.start_pos = {}, {}
        for step in range(1, 11):
            env.table = {("in", "butter_2", "cab_top_region"):
                         3 <= step <= 6}
            auto.evaluate(None, step)      # per action (v067)
        env.table = {("in", "butter_2", "cab_top_region"): False}
        chunk_auto.evaluate(None, 10)      # once per chunk (iter-1)
        assert len(auto.flips) == 2 and auto.flips[0][:1] == (3,)
        assert not chunk_auto.flips, "chunk-level eval saw the flip?"
    finally:
        ta.GoalAutomaton._subgoal_true = orig


# ---- 8. run_schema guard refuses iteration-1 artifacts -------------------

def test_run_schema_guard():
    for schema, kind in [("v06_continuations_v1", "continuations"),
                         ("v06_labels_v1", "semantic_labels"),
                         ("v06_teachers_v1", "teachers"),
                         ("v06_gt_manifest_v1", "teachers")]:
        try:
            assert_v067_payload({"schema": schema}, "<mem>", kind)
        except Iter1ArtifactError:
            continue
        raise AssertionError(f"guard accepted {schema}")
    # unversioned payloads (e.g. iteration-1 wz_selected.pt) refused too
    try:
        assert_v067_payload({"w_z": None, "step": 300}, "<mem>", "policy")
    except Iter1ArtifactError:
        pass
    else:
        raise AssertionError("guard accepted an unversioned payload")
    # a proper v067 payload passes
    assert_v067_payload({"schema": "v067_continuations_v1",
                         "run_schema": "v067"}, "<mem>", "continuations")


def main() -> None:
    check("dry-fixture: old candidate-specific seed rejected",
          test_old_seed_breaks_sibling_crn)
    check("sibling CRN identity + stream separation + sampler replica",
          test_new_seed_sibling_identity_and_separation)
    check("terminal predicates exclude milestones; stable hash",
          test_terminal_predicate_derivation)
    check("crossed endpoint: goal-specific success differs",
          test_crossed_success_differs_by_goal)
    check("dry-fixture: canonical-only success path rejected",
          test_old_success_path_fails_crossed_check)
    check("transient success latched; final fields keep later state",
          test_transient_success_latched)
    check("per-action resolution catches inside-chunk flips",
          test_per_action_resolution)
    check("run_schema guard refuses iteration-1 artifacts",
          test_run_schema_guard)
    print(f"ALL {len(PASS)} v067 contract tests passed", flush=True)


if __name__ == "__main__":
    main()
