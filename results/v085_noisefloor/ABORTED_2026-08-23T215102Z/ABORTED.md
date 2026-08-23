# ABORTED — tooling defect, not a scientific outcome

Second Action 2M.3 registration (38 sources / 90,500 steps, seeds 3940-3977).
Stopped at source 3948 with `HALT before segment: source: worst case 250 vs
headroom 205` after 8 sources (4 eligible) and **~1,959 steps**.

Cause is mine. `SegmentLedger` derives cumulative spend by globbing every ledger
under the action root — the cross-run accounting added after Stage 1R.1 — and it
did not distinguish registrations. It therefore deducted the FIRST, superseded
attempt's 7,336 steps from this registration's 9,500-step source line, leaving
205 steps of headroom against a 250-step reservation.

Fixed by giving the ledger a quarantine-prefix skip list, so a superseded or
aborted registration's spend stays charged to itself without consuming its
successor's cap. No decision rule, threshold, quota, or seed was changed.

Its ~1,959 steps stay charged and are counted in the project total.
