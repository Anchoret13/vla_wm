# SUPERSEDED - not a citable V8.0 calibration

This screen panel ran before two defects were found. Its per-episode records
are real and are kept, but its `summary.json` derived block is wrong and its
panel is not reproducible.

1. **Negative intervals.** `episode_record` differenced milestone completion
   steps by SUBGOAL INDEX. The chain goals are unordered conjunctions and
   pi0.5 solves them out of order (seed 3002: cream cheese at 80/160 before
   tomato sauce at 340/386), so index-differencing produced
   `chain2b_lr2|2|pick_up cream_cheese_1` q90 = -178.5, bound = -267.
   Intervals are now temporal gaps tagged by the milestone that ended them.
2. **Unseeded policy sampling.** pi0.5 is a flow policy and samples its chunk;
   torch/numpy were never seeded, so the same episode seed gave 138 and 134
   steps on two runs. Episodes are now seeded per episode.

Rates and the finalist decision were NOT affected by defect 1 (they are
computed from `success` and the achieved-milestone count, and every failure in
this panel had a prefix-shaped achieved set, so the old and new
`failure_phase` definitions agree here). They are superseded anyway because
defect 2 changes the sampling stream.

Superseded by the re-run recorded in the sibling directory.
