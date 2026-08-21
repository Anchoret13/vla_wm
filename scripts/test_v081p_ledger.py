#!/usr/bin/env python
"""Regression test for the two accounting failures Stage 1R.1 actually hit.

1. a second run directory must NOT receive a fresh cap;
2. a segment killed between `open` and `close` must still be charged.
"""
from __future__ import annotations
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lcwm.v081p_exec import SegmentLedger  # noqa: E402

CAPS = {"source": 5000}


def main() -> int:
    fails = []
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        # ---- run 1: 20 sources, 250 steps each, exactly fills the line -----
        l1 = SegmentLedger(root / "run1" / "segment_ledger.jsonl", CAPS, action_root=root)
        for i in range(20):
            l1.open_segment(f"s{i}", "source", 250, seed=3480 + i)
            l1.close_segment(f"s{i}", "source", 250, 250, seed=3480 + i)
        if l1.total != 5000:
            fails.append(f"run1 total {l1.total} != 5000")

        # ---- run 2: a fresh directory must inherit run 1's spend -----------
        l2 = SegmentLedger(root / "run2" / "segment_ledger.jsonl", CAPS, action_root=root)
        if l2.total != 5000:
            fails.append(f"run2 did not inherit prior spend: {l2.total}")
        ok, why = l2.headroom_ok("source", 250)
        if ok:
            fails.append(f"run2 got fresh headroom on a full line: {why}")
        else:
            print(f"1. restart does NOT get a fresh cap: {why}")

        # ---- killed mid-segment: the open reservation stays charged --------
        l3 = SegmentLedger(root / "run3" / "segment_ledger.jsonl", {"source": 10000},
                           action_root=root)
        before = l3.total
        l3.open_segment("killed", "source", 250, seed=9999)   # never closed
        l4 = SegmentLedger(root / "run4" / "segment_ledger.jsonl", {"source": 10000},
                           action_root=root)
        if l4.total != before + 250:
            fails.append(f"killed segment lost: {l4.total} vs {before + 250}")
        else:
            print(f"2. segment killed between open/close still charged: "
                  f"{before} -> {l4.total}")

        # ---- a closed segment charges ACTUAL steps, not the reservation ----
        l5 = SegmentLedger(root / "run5" / "segment_ledger.jsonl", {"source": 20000},
                           action_root=root)
        b = l5.total
        l5.open_segment("short", "source", 250, seed=1)
        l5.close_segment("short", "source", 250, 137, seed=1)
        if l5.total != b + 137:
            fails.append(f"reservation not released: {l5.total} vs {b + 137}")
        else:
            print(f"3. closed segment charges actual steps, not the reservation")

    print("RESULT:", "PASS" if not fails else f"FAIL {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
