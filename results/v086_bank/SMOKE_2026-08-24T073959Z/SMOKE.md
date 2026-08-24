# SMOKE — mechanics only, not B_boot-v3

One group per split (3 groups, 80 branches, 7,671 steps) run under `V086_SMOKE=1`
to verify schema, per-step Phi with forked memos, restore, and ledger accounting.
Action 2M.4 states mechanics smokes are implementation checks that need no new
scientific decision.

**It must never train M0.2.** Its seeds 4000 / 4108 / 4131 are also drawn by the
real collection, so pooling would duplicate sources across splits. The `SMOKE_`
prefix also keeps its steps from consuming the real registration's cap.

What it established, and why the full run was launched: `QPhi` took **80 distinct
values across 80 branch rows**, against the legacy `G` vector's 11 on the same
rows. The continuous readout resolves candidates where the deterministic label
quantized them together.
