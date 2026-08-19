# ABORTED — incomplete panel 2, not a result

Killed by a 2-minute foreground command timeout after 4 of 20 episodes, before
any verdict was computed. `manifest.json` still reads `status: RUNNING` and
there is no `summary.json`, which is the registered marker of an incomplete
directory.

**Its 2,687 environment steps were really spent and are really ledgered.** They
stay charged: framework §7.6 counts every executed step, including from runs
that produced no result, and unused budget is never reallocated. The replacement
panel-2 run is charged on top of these, not instead of them.

This exposed a gap the pre-launch review did not: `record()` writes the ledger
row before charging (so a kill loses nothing there), but the cumulative
`SPEND.json` was only rewritten at the end of a successful run, so a killed run
vanished from the cap accounting. Cumulative spend is now DERIVED by summing
every `interaction_ledger.jsonl` on disk, which cannot miss a killed run.
