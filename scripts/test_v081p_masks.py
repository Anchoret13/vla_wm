#!/usr/bin/env python
"""Zero-environment conformance test for the Action 2P mask machinery
(daily 2026-08-20 §6 item 2).

Runs no simulator, loads no policy, spends no environment step.  It replays
stored V8.0/1R.1 episodes to prove three things before any GPU is authorized:

1. the exact eligible mask accepts and rejects the states it must;
2. the set-valued stratum separates the two historical `chain2b phase=2`
   signatures that the scalar phase aliased - the defect the 2026-08-19 audit
   found - and does so on real stored rows, not constructed ones;
3. `tau=160` is reachable and evaluable in real `chain1b` failure rollouts, and
   the fraction of them that satisfy the exact mask is reported honestly rather
   than assumed.
"""

from __future__ import annotations

import glob
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lcwm.v081p_contract import ELIGIBLE, TAU, mask_conformant  # noqa: E402
from lcwm.v08r_contract import stratum_key  # noqa: E402


def sets_at(ep: dict, upto: int):
    """(ever, valid, damaged) as of step `upto`, from a stored episode."""
    ever = sorted(i for i, s in
                  ((int(k), v) for k, v in ep["events_achieved"].items()) if s <= upto)
    valid: list[int] = []
    for step, vals in ep["milestone_timeline"]:
        if step <= upto:
            valid = [i for i, v in enumerate(vals) if v]
    dropped = {i for (s, i, d) in ep["flips"] if d == -1 and s <= upto}
    return ever, valid, sorted(dropped - set(valid))


def load(pattern: str) -> list[dict]:
    out = []
    for d in sorted(glob.glob(pattern)):
        out += [json.loads(l) for l in open(d) if l.strip()]
    return out


def main() -> int:
    fails = []

    # ---- 1. mask accept/reject ------------------------------------------
    ok, _ = mask_conformant([], [], [], [0], [0, 1])
    assert ok, "the eligible mask must accept its own definition"
    for bad in ([[0], [], [], [0], [0, 1]], [[], [], [], [0, 1], [0, 1]],
                [[], [], [0], [0], [0, 1]], [[0], [0], [], [1], [1]]):
        assert not mask_conformant(*bad)[0], f"must reject {bad}"
    print("1. exact-mask accept/reject: OK")

    # ---- 2. the aliased chain2b phase=2 signatures ----------------------
    eps = load("results/v080_screen/2026-*_confirm/episodes.jsonl")
    p2 = [e for e in eps if e["task"] == "chain2b_lr2" and not e["success"]
          and e.get("failure_phase") == 2]
    strata = Counter()
    for e in p2:
        ever, valid, dmg = sets_at(e, e["steps"])
        strata[stratum_key(e["task"], e["subgoals"], ever, valid, dmg).key()] += 1
    print(f"2. historical chain2b failures with scalar phase==2: {len(p2)}")
    for k, v in strata.most_common():
        print(f"     x{v}  {k}")
    if len(p2) and len(strata) < 2:
        fails.append("the two aliased phase=2 signatures did not separate")
    else:
        print(f"   -> {len(strata)} distinct set-valued strata from ONE scalar "
              f"phase: the aliasing is resolved on stored rows")

    # ---- 3. tau=160 reachability in real chain1b failures ---------------
    c1 = [e for e in load("results/v080r_1r1/2026-*_panel*/episodes.jsonl")
          if e["task"] == "chain1b_lr2" and e["subrole"] == "deadline_probe"]
    at_l = [e for e in c1 if not e["at_checkpoint"]["250"]["success"]]
    conform = 0
    seen = Counter()
    for e in at_l:
        ever, valid, dmg = sets_at(e, TAU)
        k = stratum_key(e["task"], e["subgoals"], ever, valid, dmg)
        seen[k.key()] += 1
        good, _why = mask_conformant(ever, valid, dmg, k.actionable, k.unresolved)
        conform += good
    print(f"3. chain1b failure@250 rollouts in 1R.1: {len(at_l)} of {len(c1)}")
    print(f"   state at tau={TAU}:")
    for k, v in seen.most_common():
        print(f"     x{v}  {k}")
    print(f"   exact-mask conformant at tau={TAU}: {conform}/{len(at_l)}"
          f" = {conform / max(1, len(at_l)):.2f}")
    print("   (1R.1 used seeds 3400-3419; Action 2P sources are 3480-3499, so "
          "this is a prevalence ESTIMATE, not the pilot's own denominator)")
    if conform == 0 and at_l:
        fails.append(f"no historical chain1b failure conforms at tau={TAU}; the "
                     f"anchor stratum would underfill by construction")

    print("\nRESULT:", "PASS" if not fails else f"FAIL {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
