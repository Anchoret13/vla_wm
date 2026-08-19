# Superseded registration d77615a5

Sealed 2026-08-19 at commit `cddf1ae`, then superseded the same day **before any
Stage 1R.1 environment step was spent under it**. Preserved, not edited: an
immutable artifact is superseded by a successor that cites it (framework §14.2).

Superseded because the Stage 1R.1 adversarial review confirmed 17 defects of 24
raised, three of which required contract changes:

1. `INTERACTION_CAP` was a hand-written literal, so the panel lines had ZERO
   headroom (`PANEL_1 x extended_horizon` exactly) while the duplicate subset
   was charged to them. The abort direction correlated with the result being
   measured - more deadline failures, longer episodes, more likely halt - which
   is the V7.6/V7.7 pattern framework §3.1.1 prohibits. The cap is now DERIVED
   from `CHECKPOINTS`, and `CAP_LINE` maps every (stage, panel, subrole) to its
   own budget line.
2. `CHECKPOINTS` and `DUPLICATE_N` lived in the unsealed runner, so the sealed
   budget could drift from the horizons it funds. Both now live in the contract.
3. No preflight existed to verify the seal. `verify_sealed()` now recomputes the
   registration hash instead of trusting the value it reports about itself, and
   re-hashes all sealed sources.

The successor registration carries the same total cap (113,030 env steps) - the
line items were redistributed, not enlarged.
