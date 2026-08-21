# SMOKE — a shakedown bank, not `B_boot-v2`

One group per split under `V083_SMOKE=1`, run to prove the 17-candidate path
end to end: `select_wide` at 8+8, the widened tensors, the relaxed validator,
and a `groups.pt` that loads through the M0 loader. 85 branches, 8,400 steps,
all charged.

**It must never train M0.1.** Its three groups use seeds 3600 / 3670 / 3688,
which the real collection also draws, so pooling the two would duplicate
sources across splits. The real bank is the sibling non-`SMOKE_` directory.
