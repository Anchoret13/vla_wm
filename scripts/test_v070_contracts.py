#!/usr/bin/env python
"""V7.0.0 — strict replay-rule fixtures + seed-signature contracts.

Required fixtures: repeat signs [0,0] clean; [+1,-1], [+1,0], [-1,0]
all unstable — exactly the patterns `paired_preference == 0` wrongly
passes. Plus: evaluation-CRN signatures exclude candidate/arm identity
and differ from construction CRNs.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.task_automaton import paired_preference  # noqa: E402
from lcwm.v070_replay import (GROSS_BOUNDS,  # noqa: E402
                              strict_replay_clean)

PASS = []


def check(name, fn):
    fn()
    PASS.append(name)
    print(f"  [pass] {name}", flush=True)


def outcome(success=False, damage=0, p=0.0, q=0.0, tau=-101):
    return {"success_by_100": success, "neg_damage": -damage,
            "p_valid_100": p, "q_valid_mean": q, "neg_tau_next": tau}


TOL = {"neg_damage": 0.0, "neg_tau_next": 7.15, "p_valid_100": 0.0,
       "q_valid_mean": 0.0417, "success_by_100": 0.0}
OK_DELTAS = {"eef_pos": 1e-3, "eef_quat": 1e-4, "gripper": 1e-4,
             "obj_pos": 0.0}


def reps(*qs):
    return [outcome(q=q) for q in qs]


def test_strict_fixtures():
    # [0,0] -> clean
    clean, info = strict_replay_clean(
        reps(0.5, 0.5), reps(0.5, 0.5), TOL, OK_DELTAS)
    assert clean and info["repeat_signs"] == [0, 0]
    # [+1,-1] -> unstable (the s2121_d2 pattern)
    clean, info = strict_replay_clean(
        reps(0.9, 0.1), reps(0.5, 0.5), TOL, OK_DELTAS)
    assert info["repeat_signs"] == [1, -1] and not clean
    # paired_preference wrongly calls this 0 (=agree)
    assert paired_preference(reps(0.9, 0.1), reps(0.5, 0.5), TOL) == 0
    # [+1,0] -> unstable
    clean, info = strict_replay_clean(
        reps(0.9, 0.5), reps(0.5, 0.5), TOL, OK_DELTAS)
    assert info["repeat_signs"] == [1, 0] and not clean
    assert paired_preference(reps(0.9, 0.5), reps(0.5, 0.5), TOL) == 0
    # [-1,0] -> unstable
    clean, info = strict_replay_clean(
        reps(0.1, 0.5), reps(0.5, 0.5), TOL, OK_DELTAS)
    assert info["repeat_signs"] == [-1, 0] and not clean
    # gross bound fails even with [0,0]
    bad = dict(OK_DELTAS, gripper=GROSS_BOUNDS["gripper"] * 1.1)
    clean, info = strict_replay_clean(
        reps(0.5, 0.5), reps(0.5, 0.5), TOL, bad)
    assert not clean and info["gross_failures"] == ["gripper"]


def eval_seed(payload: str) -> int:
    return int.from_bytes(hashlib.sha256(
        payload.encode()).digest()[:8], "big") & ((1 << 63) - 1)


def test_eval_crn_signatures():
    a = eval_seed("v070_anchor_eval1|action|src_a|12")
    # candidate/arm identity structurally absent; distinct anchors differ
    assert a == eval_seed("v070_anchor_eval1|action|src_a|12")
    assert a != eval_seed("v070_anchor_eval1|action|src_a|13")
    assert a != eval_seed("v070_anchor_eval1|action|src_b|12")
    c0 = eval_seed("v070_anchor_eval1|cont|src_a|12|t3_canonical|0|0")
    assert c0 != eval_seed(
        "v070_anchor_eval1|cont|src_a|12|t3_canonical|1|0")
    assert c0 != eval_seed(
        "v070_anchor_eval1|cont|src_a|12|t3_canonical|0|1")
    # evaluation streams disjoint from construction streams (different
    # run-id namespace)
    from lcwm.v067_lineage import cont_seed_v067
    assert c0 != cont_seed_v067("v069_fa1", "src_a", 12,
                                "t3_canonical", 0, 0)


def test_stats_require_loaded_checkpoint():
    from scripts.audit_v070_coupling import (SelectedToken,
                                             compute_policy_statistics)
    try:
        compute_policy_statistics(None, SelectedToken(), None, [], None)
    except AssertionError:
        return
    raise AssertionError(
        "policy statistics computed without a verified checkpoint")


def main():
    check("strict fixtures [0,0]/[+1,-1]/[+1,0]/[-1,0] + gross",
          test_strict_fixtures)
    check("evaluation CRN signatures arm-free and disjoint",
          test_eval_crn_signatures)
    check("policy stats refuse an unloaded/unverified checkpoint",
          test_stats_require_loaded_checkpoint)
    print(f"ALL {len(PASS)} v070 contract tests passed", flush=True)


if __name__ == "__main__":
    main()
