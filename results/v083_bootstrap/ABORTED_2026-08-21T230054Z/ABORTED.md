# ABORTED — incomplete smoke, not a bank

Killed by a 2-minute foreground command timeout during the `V083_SMOKE=1`
one-group-per-split shakedown. `manifest.json` still reads `status: RUNNING`
and there is no `groups.pt`, which is the registered marker of an incomplete
run.

Its **2,280 ledgered environment steps were really spent and stay charged**.
They belong to the smoke, not to the `B_boot-v2` bank: no group from this
directory enters any split, and no row here trains anything.

Cause is mine — I ran a multi-minute job in the foreground after already having
made that exact mistake once in Stage 1R.1.
