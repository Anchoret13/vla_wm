# ABORTED — incomplete smoke, not a bank

Second `V083_SMOKE=1` shakedown attempt, stopped when `validate_bootstrap_group`
rejected 17 candidates against its hard-coded `C == 7`. That defect is fixed
(the validator now requires one reference plus at least one alternative with
every per-candidate tensor agreeing on C); this directory is the evidence that
the one-group smoke caught it before the 1,088-branch run.

`status: RUNNING`, no `groups.pt`. Its **2,910 ledgered steps stay charged** and
train nothing.
