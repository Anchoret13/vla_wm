# DISCARDED — anchored on the wrong population

64 groups, 1,440 branches, 138,723 steps, zero restore failures. Mechanically
sound and scientifically unusable.

**33 of its 64 anchors were built from source episodes that later SUCCEEDED by
step 250.** The collector tested only the exact event mask at `tau=160` and
dropped the `failure@250` requirement that every prior action applied
(`lcwm/v081p_exec` + `scripts/run_v085_noisefloor.py`:
`s["anchor"] and s["anchor"].mask_ok and s["failure_at_L"]`).

The giveaway was the fill rate: 64 groups from 64 sources is 100% eligibility
against a measured rate of 0.538, and Action 2M.4's own source ceilings
(108/23/23 for 48/8/8 groups) are sized from that 0.538. A mask-only rule would
have made those ceilings absurdly over-provisioned.

Why this could not be kept: the method is failure-anchored (framework §3).
Training M0.2 on a bank where more than half the anchors sit inside episodes
that go on to succeed would silently redefine the population the world model
learns, and would break comparison with every earlier failure-anchored artifact.

**Its 138,723 steps stay charged** and are counted in the project total. No
group from this directory enters any split. Cause is mine, and the fix is one
line: return the anchor only when the source ends in `failure@250`.
