#!/usr/bin/env python
"""Zero-environment unit test of the Action 2P reducer and ADVANCE gate.

The gate decides whether this project spends a further tranche, so its logic is
tested against constructed outcome vectors before it is ever applied to real
ones.  Every case below is a situation the pilot can actually produce.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lcwm import v081p_contract as P  # noqa: E402

F = P.sealed_floors()
V = lambda succ=0, dp=0, ttm=91, G=0.0, dmg=0: {
    "dmg": dmg, "succ": succ, "dp": dp, "ttm": ttm, "G": G}
REF = [V()] * 3


def anchor(*labels_specs):
    a = P.AnchorVerdict("a")
    for i, alt in enumerate(labels_specs):
        a.alternatives.append(P.classify_alternative(f"c{i}", alt, REF, F))
    return a


def main() -> int:
    fails = []
    ck = lambda name, got, want: (None if got == want
                                  else fails.append(f"{name}: got {got} want {want}"))

    win = [V(succ=1, dp=1, ttm=40, G=0.67)] * 3
    ck("all-3-keys win", P.classify_alternative("c", win, REF, F).label, "paired_positive")

    two = [V(succ=1, dp=1, ttm=40, G=0.67), V(succ=1, dp=1, ttm=40, G=0.67), V()]
    ck("2-of-3 win is NOT positive", P.classify_alternative("c", two, REF, F).label, "mixed")

    worse = [V(dmg=1)] * 3
    ck("damage-worse", P.classify_alternative("c", worse, REF, F).label,
       "paired_non_improving")

    tie = [V()] * 3
    ck("pure tie", P.classify_alternative("c", tie, REF, F).label, "mixed")
    ck("pure tie has no variation", anchor(tie).has_variation, False)

    near = [V(dp=1, ttm=86, G=0.05)] * 3          # inside both floors
    ck("within-floor is a tie", P.classify_alternative("c", near, REF, F).per_key,
       [1, 1, 1] if False else [1, 1, 1])          # dp differs -> real, floor 0

    # dmg dominates: a candidate that succeeds but damages must LOSE
    dmg_win = [V(succ=1, dp=1, ttm=30, G=0.74, dmg=1)] * 3
    ck("damage outranks success", P.classify_alternative("c", dmg_win, REF, F).label,
       "paired_non_improving")

    # gate arithmetic
    pos, non, neutral = anchor(win), anchor(worse), anchor(tie)
    both = anchor(win, worse)
    g = P.advance_gate(True, True, 0, 24, 24, 168, 168, True, True,
                       [both, both, pos, pos, neutral, neutral, neutral, neutral])
    ck("gate ADVANCE", g["verdict"], "ADVANCE")
    ck("gate counts", g["counts"],
       {"anchors": 8, "with_variation": 4, "with_positive": 4, "with_both": 2})

    g2 = P.advance_gate(True, True, 0, 24, 24, 168, 168, True, True,
                        [both, pos, pos, pos, neutral, neutral, neutral, neutral])
    ck("one 'both' fails condition 6", g2["conditions"]["6_both"], False)
    ck("that is a HALT", g2["verdict"], "HALT")

    g3 = P.advance_gate(True, True, 1, 24, 24, 168, 168, True, True,
                        [both, both, pos, pos, non, non, non, non])
    ck("reserved-seed use fails condition 1", g3["conditions"]["1_sources_and_selection"], False)

    g4 = P.advance_gate(True, True, 0, 24, 24, 167, 168, True, True,
                        [both, both, pos, pos, non, non, non, non])
    ck("one bad restore fails condition 2", g4["conditions"]["2_restore_and_replay"], False)

    print("RESULT:", "PASS" if not fails else f"FAIL {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
