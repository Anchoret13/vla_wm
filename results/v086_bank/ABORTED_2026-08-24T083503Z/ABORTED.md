# ABORTED — cap bug, not a scientific outcome

Halted after 3 groups with `branch: worst case 80 vs headroom 41`. The
`DISCARDED_` bank's 138,723 steps were being charged against this registration
because `DISCARDED_` was not in `SegmentLedger.QUARANTINE_PREFIXES` — a
hardcoded list that fails OPEN for any label not in it.

Fixed by adding a regex that quarantines any ALL-CAPS label prefixed to a run
stamp, so inventing a new label can never silently consume a successor's cap
again. This is the second time cap accounting has bitten; the first was Action
2M.3 attempt 2.

Its ~6,860 steps stay charged and no group from it enters any split.
