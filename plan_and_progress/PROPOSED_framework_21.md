# PROPOSED — not yet merged into framework_design.md
# Awaiting review. Written 2026-08-29 from the measurements cited inline.

## §21 — Amendments to §5 from measurement (2026-08-29)

§20 was written during a session the user subsequently ruled **off-mission on three
of four axes**. Its conclusion — "deployment correction works, the world model does
not" — is **withdrawn as a statement about world models**: no world model was built
in that session, only a discriminative head on the current observation. What
survives from §20 is the corrective-head result itself and the methodological
findings.

The following amend §5.1 and §5.2. Each is a measurement, not a preference.

### §21.1 — LCState must contain the controllable component (amends §5.1)

§5.1 builds the state from `PrefixVLM(o_t, ℓ)`. Measured on `chain1b_lr2`, that
pooled prefix hidden is **nearly action-invariant over c = 10**: an
action-conditioned transition beats an action-free one by **+0.81% of identity**
(CI [+0.14%, +0.94%]). It is an alignment representation and nothing ever asked it
to be predictable under actions.

Proprioception — eef pose, gripper, joints, 25 dims — carries almost all of the
action-conditioned signal:

| component | action | no-action | shuffled action | action gain |
|---|---:|---:|---:|---:|
| pooled prefix hidden (2048) | 0.552× | 0.560× | 0.597× | +0.81% |
| **proprioception (25)** | **0.117×** | 0.235× | **1.217×** | **+11.78%** |

A shuffled action makes the proprio prediction **worse than not predicting at all**
(1.217× identity) — the signature of a real dynamics model, which a smoother cannot
produce. **LCState must include it**, and its loss must be weighted per component:
under a mean-over-dims loss the 25 informative dims carry ~1.2% of the gradient and
the signal disappears (measured: +11.78% alone → +0.84% when pooled unweighted).

### §21.2 — The predictive target is not the exact next latent (amends §5.2)

§5.2 specifies predicting `z̃_{t+c} = T_θ(z_t, E_a(u^i))` and reading `D_θ` off it.
Two defects, both measured:

1. **Unweighted MSE toward an exact next latent spends capacity on
   decision-irrelevant variance.** The fix is value-equivalence: the latent need
   only be sufficient for the decision quantity. Encoder, transition and heads must
   be trained by **one objective**; §5's three-stage reading (freeze latent → fit
   transition → fit heads) produced a latent never shaped by value.
2. **Anti-collapse is unspecified and is required.** Without a decoder the latent
   collapses: a Gaussian-transition variant reached identity error 0.0004 with the
   transition scoring **5.5× worse than doing nothing**. Bounded latents (SimNorm
   groups), stop-grad targets and value/reward terms are the mechanism.

### §21.3 — One-step scoring cannot demonstrate the transition (amends §5.2)

When candidates share the state — which they do at any branch point — `T_θ(z, u)`
is a deterministic feature map of `(z, u)` and lies **inside the function class of a
critic fed `(z, E_a(u))`**. It therefore cannot beat one, at any data scale, and
more data helps the critic at least as much.

Five isolations confirmed this empirically before the argument was found:
deployment ablation p = 0.728; offline AUC 0.637 vs 0.636; joint-training variants
0.609/0.583; TD bootstrapping −0.0102 (9/21 positive) and −0.0055 (CI spans 0).

**Consequence for the framework:** `D_θ` applied to a one-step predicted latent is
a critic. Any claim that the transition contributes requires a use where it is not
replaceable — currently only *policy improvement in imagination* (a critic cannot
generate a rollout) or *optimisation over action sequences*.

### §21.4 — Model selection must not use the downstream ranking metric

Ranking AUC and predictive quality were measured **anti-correlated**:

| model | ranking AUC | h=1 vs identity | h=1 shuffled | corr(V_imag, V_real) |
|---|---:|---:|---:|---:|
| frozen projection | **0.626** | 0.935× | **0.998×** | 0.676 → 0.431 |
| learned encoder | 0.584 | **0.530×** | 0.617× | **0.947 → 0.878** |

The model that ranked best had **no action-conditioned dynamics** (a shuffled action
cost it 0.2%). Selecting by the downstream metric selected against the world model.
World-model quality must be measured directly: one-step error against the identity
floor, a shuffled-action control, compounding over horizon, and the value gap.

### §21.5 — Value extrapolation bounds imagination, and the bound is the whole story

Policy improvement in imagination (bounded residual on the VLA's chunk, VLA as
trust region) on a fresh 96-seed panel:

| residual scale | success | vs base 44/96 = 0.458 |
|---:|---:|---|
| 0.15 | 19/96 = 0.198 | +3 −28, p < 0.0001 |
| **0.01** | **47/96 = 0.490** | +9 −6, p = 0.607 |

The residual saturates its norm bound at **every** scale (0.01/0.03/0.06/0.15) and
the imagined gain grows **linearly** with it: `V` is monotone along the residual
direction with no interior optimum. The bound is the only thing limiting how far the
optimiser walks off-distribution, and 0.15 was far too loose (+0.014 imagined,
−0.26 real).

This is **not** evidence that the value gradient points the wrong way — at 0.01 the
residual is harmless and nominally positive. It is a statement about magnitude and
about the absence of any distributional penalty. An ensemble value optimised
pessimistically is the correction; whether it permits a larger useful scale is open.

### §21.6 — Testbed rules

- **Do not retire a testbed because one method failed on it.** `chain1b` and
  `chain3` were closed on the grounds that *selection* could not help them, which
  says nothing about latent state prediction. Both were reopened.
- **`chain1b` is not a low-success task.** Measured 0.39–0.60 across seed ranges.
  The genuinely low one is `chain3` at 0.031, which the goal's first axis asks for.
- **The transition needs no labels; the value does.** `T_θ` trains on `(z, u, z')`
  alone, so "can a world model be learned here" is answerable on `chain3` even
  where episode success is ~3%. Conflating the two is what closed `chain3` before.

### §21.7 — Supervision provenance

Trajectories are deployment-collected; **labels are not**. `Δw` comes from BDDL goal
predicates and success from `env.is_success` — both privileged. Candidate-level
labels additionally required **deterministic replay**, a simulator affordance a real
robot lacks. Current work uses episode success only, which a deployed system could
plausibly obtain; this is sparser and weakens the anti-collapse terms of §21.2.
