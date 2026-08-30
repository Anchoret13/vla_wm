# Framework v2 — a latent world model that improves a frozen VLA during rollout

**Status: proposed, replaces `framework_design.md` in full.**

**What is binding and what is not.** §1–§3 (paradigm, interface, constraints) and §6
(falsification) are binding. §4 and §5 are **menus of candidate methods**: each entry
carries what it requires, what would kill it, and its current status. A candidate
that fails its own test is marked dead and the next is tried — the framework does not
change when a method does.

---

## 1. The paradigm  `[BINDING]`

A pretrained VLA is a **strong prior we do not modify**. While it is deployed we
collect its trajectories, train a **latent world model** on them, and use that model
to improve the VLA's own behaviour at rollout time. Then we redeploy and repeat.

```
run π (VLA + current improvement)  →  trajectories
trajectories                       →  latent world model
world model                        →  an improvement to π
                                   →  redeploy
```

**It must work outside a simulator.** Weights and prompt frozen; deployment contract
unchanged; latent space, no reconstruction; data from deployment rollouts; **no true
reward**.

### What execution has established about this loop  `[measured 2026-08-30]`

Three findings constrain any instantiation, and none depends on which candidate is
tried next.

**The loop's middle stage does not contribute.** Replicated on two independent
held-out panels, 192 seeds:

| arm | panel A | panel B | pooled vs base |
|---|---:|---:|---|
| base (frozen VLA) | 0.469 | 0.448 | — |
| **residual, model-free (AWR)** | **0.656** | **0.573** | **+44 −14, p = 0.00010** |
| residual via the world model | 0.510 | 0.406 | +25 −25, **p = 1.000** |
| **AWR vs world model** | p = 0.0094 | p = 0.0113 | **+46 −16, p = 0.00018** |

The frozen VLA **is** improved by data collected during its own deployment —
0.458 → 0.615, replicated. The world model's contribution is **exactly zero**
(+25 −25), and routing the same reward through it is **significantly worse than not**
(p = 0.00018).

So the loop that works is `deployment data → reward → policy`. The middle stage as
§1 originally wrote it — a learned transition — is the one component measurement
removes.

**A compressed latent discards what its heads need.** The ordering signal lives in
the VLA's raw features (ρ = 0.417, bar 0.349); the learned 256-d encoder retains
**34%** of it (0.140), and shaping that encoder with the only deployable label drives
it to **−0.223** while branch-ranking AUC climbs. §5's structure — encode, roll
forward, read heads off the latent — assumes retention that does not hold under
sparse supervision. **Any candidate that reads a compressed latent inherits this.**

**The deployable label carries no progress information on the target task.** With
the identical probe and splits, binary episode success recovers 58% of the ceiling
on `chain1b` and **0%** on `chain3`, because success-vs-failure discrimination is
orthogonal to progress — R4 reached preference accuracy 1.000 while ordering failures
at −0.13. **Any candidate supervised only by episode outcome is bounded by this on
the target task**, which is why R6 (no reward at all) is the better next try than R5,
not merely the next entry in the menu.

## 2. Setting and retained VLA interface  `[LOCKED]`

The VLA produces a `K = 50` chunk from visual-action history and instruction `ℓ`;
`c = 10` is committed before replanning.

- `π_0` is the pretrained policy, the representation source, and the action prior.
- **Every candidate chunk comes from `π_k`, a frozen `π_0`, or a bounded
  perturbation of their supported chunks.** Never an unconstrained optimisation over
  a 500-dimensional action sequence.
- **The learned policy remains the final controller.** Full-prompt `N = 1`, no
  external planner.
- Policy-update interfaces are zero-initialised: stock behaviour at initialisation.
- **Only executed actions carry causal credit anywhere in the system.**

## 3. Constraints on the latent  `[BINDING]`

Not a prescribed architecture — three properties any candidate must have.

1. **It contains what actions control.** Measured over `c = 10`: the VLA's pooled
   prefix hidden gives an action gain of **+0.81%** over an action-free model, while
   proprioception alone gives **+11.78%**, and a *shuffled* action makes the proprio
   prediction **worse than not predicting at all** (1.217× identity) — the signature
   of a dynamics model. A latent without the controllable component has no dynamics
   to learn.
2. **Components are weighted in the loss.** Unweighted, 25 proprio dims carry ~1.2%
   of the gradient against 2048 hidden dims and the signal vanishes (+11.78% → +0.84%).
3. **It does not collapse.** Without a decoder this is a live failure, not a
   formality: a Gaussian-transition variant reached identity error 0.0004 with its
   transition scoring **5.5× worse than doing nothing**. Any candidate states which
   term rules collapse out, and the check runs before the latent is used.

## 4. Candidate reward constructions  `[MENU]`

**The requirement, binding:** a signal dense enough to train on, obtainable outside a
simulator, and passing §6.1. **How it is constructed is open.**

Why the obvious choice is already dead: a value learned from outcome labels needs
`env.is_success` (privileged), is nearly absent where the paradigm is aimed
(`chain3` measured **1/96 = 0.010**), and did not survive optimisation (+0.014
imagined, **−0.26** real).

| # | candidate | requires | status — **measured 2026-08-30** |
|---|---|---|---|
| R1 | quasimetric to a language anchor (QRL-style) | ≥2 instructions; an anchor budget | **dead.** Trained cleanly (separation +3.2, no collapse, stable λ) and failed §6.1 on every measurable task: chain1b +0.095, chain2b −0.107, chain3 +0.041 |
| R2 | hindsight goal from terminal states (VIP/LIV) | nothing | **dead on arrival** — 97–99% of our terminal states are failures, so this teaches "the goal is where my policy stops" |
| R3 | `G(ℓ)` compared against the VLA's `h_t` | joint text-image init | **rejected** — `h_t` is a joint `(o,ℓ)` code whose language part is a constant offset within an instruction |
| R4 | preference model from episode outcomes | binary success labels | **dead.** Preference accuracy **1.000** and ordering of failures at −0.13 / −0.12 / −0.11: success-discrimination is **orthogonal** to progress |
| R5 | a VLM as reward model | an external VLM | **unavailable locally** — PaliGemma ships no text tower, so image-text similarity needs a download |
| R6 | uncertainty/novelty, no reward | nothing | untried |
| **R7** | **plain outcome probe on RAW features** | binary success labels | **passes the primary bar on raw features (ρ = 0.417 > 0.349); fails the language null (0.327).** Usable as a *task-specific* reward, not a language-conditioned one |

**The decisive cross-cutting result is about the label and the representation, not
the objective.**

*The label.* Same probe, same splits, only the training label changes:

| trained on | chain1b | chain3 |
|---|---:|---:|
| **binary success** (the only deployable label) | +0.339 | **−0.008** |
| graded progress (privileged) | +0.579 | +0.209 |

A binary episode outcome recovers 58% of the ceiling on `chain1b` and **nothing** on
the target task. No deployable label supports a progress reward there.

*The representation.* The ordering signal is in the VLA's raw features and the
world model's encoder throws most of it away:

| representation | ρ on held-out failures | retained |
|---|---:|---:|
| raw 2073-d VLA features | **+0.417** | 100% |
| WM latent (outcome weight 0 / 1 / 5) | +0.140 / +0.128 / **−0.223** | 34% / 31% / **−53%** |

Every head that failed §6.1 was reading an already-degraded representation, and
shaping the encoder with the deployable label makes it **worse** — branch AUC climbs
(0.569 → 0.612) while failure ordering collapses. Swapping reward objectives could
never recover discarded information; **this is what five candidate failures were
actually measuring.**

**Whatever is used, the reward enters as a difference, not a level:**
`r̂ = V(z_{t+1}) − V(z_t)`. Potential-based shaping telescopes to a boundary term and
is **policy-invariant**, so a policy cannot accumulate return by walking the latent
toward the goal. Optimising a *level* is what produced our monotone-with-no-interior-
optimum pathology.

## 5. Candidate uses of the model  `[MENU]`

**What is binding:** ranking candidates is not a use of a world model. At a branch
point every candidate shares `z`, so `T(z, u)` is a deterministic feature map of
`(z, u)` and lies **inside the function class of a critic fed `(z, E_a(u))`**. It
cannot beat one at any data scale. Five independent nulls confirmed this before the
argument was found (deployment ablation p = 0.728; offline 0.637 vs 0.636; two
joint-training variants below baseline; TD bootstrapping −0.0102 and −0.0055).

| # | use | transition needed? | status — **measured 2026-08-30** |
|---|---|---|---|
| U0 | oracle best-of-N ceiling | no | **run, and it capped U1.** chain3: random 0.0039 → oracle best-of-16 **0.0625**, with 15/16 states deterministic all-fail |
| U1 | re-rank `N` sampled chunks | no | **dead on the target task.** σ = 3 proposals changed nothing (oracle 0.0625, 1/16 states disagreeing — identical to σ = 0), so changing the proposal mechanism was tried and failed |
| **U2** | bounded residual trained in imagination | **yes** | **measured harmful, replicated on two panels.** Pooled over 192 seeds: model-free AWR beats base **+44 −14 (p = 0.00010)**; imagination ties base **+25 −25 (p = 1.000)**; **AWR beats imagination +46 −16 (p = 0.00018)** |
| U3 | optimisation over action sequences | yes | untried; changes what is executed |
| U4 | model uncertainty to gate intervention | yes | untried |

**What U2 established, and it is the framework's central question.** A residual
trained on R7's reward improves the frozen VLA **significantly** — 0.469 → 0.656 —
and routing that same reward through the world model is **significantly worse than
not routing it**. The transition does not merely fail to contribute; it costs.

**§6.1 discriminates.** The reward that passed its primary bar (R7, ρ = 0.417) yielded
a significant gain; the reward that failed it (the TD value, ρ = 0.247) yielded −0.26
in deployment. The gate predicts. It simply does not require a world model.

**A note on the refuting arm for a residual.** "The same objective without `T`" is
undefinable here: a residual on the action has no offline effect without a
transition, since only a model can say where a different action leads. The
model-free counterpart is advantage-weighted regression on executed actions, which a
σ-perturbed buffer supports.

**Correction to record:** a residual "saturating its norm bound at every scale" is
**not** evidence the value is wrong — against a monotone objective a hard norm ball
puts the KKT solution on the boundary at every radius. And shrinking the bound until
the environment number stops dropping is **fitting the bound on the evaluation
panel**.

## 6. Falsification  `[LOCKED]`

**The refuting arm runs before the claim is stated.** This section does not change
when a method does.

| claim | required comparison |
|---|---|
| the reward is a reward | §6.1 |
| the improvement helps | vs the frozen VLA alone, paired, fit-disjoint panel |
| **the world model is what helped** | vs the same improvement with the model removed — same signal, same anchor, no rollout |
| the model is a model | one-step error vs the identity floor, **shuffled-action control**, compounding over horizon |

**Model quality is never inferred from a downstream metric.** Ranking does not need
dynamics, so ranking quality cannot select for it — measured anti-correlated: the
best-ranking model (AUC 0.626) had a shuffled-action cost of 0.2%, while the
worse-ranking one (0.584) predicted at 0.530× identity.

### 6.1 The reward diagnostic  `[LOCKED]`

Monotonicity within a trajectory is worthless: a clock scores perfectly on it. A
usable signal must order **failures** by how far they got — a cross-trajectory
property needing no successes, so on a 97%-failure task the whole buffer is the
sample.

```
ρ = Spearman( V(z_T; ℓ), stage_reached )   over held-out FAILED episodes
```

`stage_reached` is privileged and used **for evaluation only, never for training**.

| null | expected |
|---|---|
| shuffled instruction | ρ → 0, else the signal is not language-conditioned |
| clock | ρ → 0 |
| label-shuffled | ρ → 0, CI covering 0 |

**Bar:** ρ ≥ **0.6 × the task's supervised ceiling**, bootstrap 95% CI excluding
half the bar, nulls in [−0.1, 0.1]. The interval is the result, not the point
estimate.

The ceiling is measured by training a probe **directly on the target**, supervised
and episode-disjoint — privileged labels as a training target, an upper bound only,
the same status as the oracle in §5 U0. A fixed constant was tried first and had to
be replaced: the achievable ceiling is task-dependent (chain1b 0.617, chain2b 0.414,
chain3 0.298), so a constant 0.35 sat **above** what anything could reach on the
target task, making the test unfalsifiable there. Measure the ceiling before
judging any candidate against the bar.

**Per task, never pooled.** Pooling chain1b (L = 250, stages 0–1) with chain3
(L = 750, stages 0–5) gave ρ = 0.52 with a clock null of **+0.94** — the pooled
statistic reads task identity, not progress.

**The progress measure needs resolution.** 182 of 191 chain3 failures stop at the
same stage, so stage alone is nearly constant among failures and any rank statistic
is ≈0 by construction. Grading it by how early the last milestone landed raised the
ceiling from 0.214 to 0.298.

**Measured on the value we already have** (chain1b, 188 failed episodes): ρ =
**+0.247**, CI [+0.061, +0.412]; clock null nan (all failures run the full horizon,
so it is constant); label-shuffle −0.008. **Bar NOT met.** Some progress
information, not enough.

*Statistical note.* Use average ranks for ties. `argsort(argsort(x))` gives tied
values arbitrary distinct ranks and manufactured a spurious clock null of +0.64 from
a constant input; `stage_reached` is heavily tied.

### 6.2 Pre-registered readings

| observation | conclusion |
|---|---|
| **the supervised ceiling is below the bar** | the bar is unfalsifiable there — measure the ceiling and re-derive the bar **before** judging any candidate |
| **a head fails §6.1 while the raw features pass it** | the defect is the representation, not the objective; **stop changing candidates** |
| **shaping the encoder with the label makes ordering worse** | that label is orthogonal to the target; more of it will not help |
| the signal fails §6.1 | it is a clock, not a reward — change the candidate, do not tune it |
| no outcome variation in the proposal pool | the setting or proposal mechanism is wrong; say nothing about the model |
| model-guided selection does not beat random | the score gives no benefit; the model may still be fine |
| a correction arm improves equally without the model | supports corrective imitation, **not** a model-based claim |
| improves offline ranking but hurts behaviour | do not promote |
| round 1 improves, round 2 does not | one-shot adaptation, not continual improvement |

### 6.3 Data roles  `[LOCKED]`

Episodes carry a role — `model_train`, `model_calib`, `policy_train`, `eval` —
partitioned before collection. **No parameter or threshold may be chosen using `eval`
data**; fitting a threshold on the evaluation panel is leakage even if that panel was
never "trained on".

## 7. Method selection is literature-driven  `[BINDING]`

**Before adopting a candidate in §4 or §5, search for published work on the same
scenario** — not only on the application. Two entries above (R2, R3) were killed by
reading rather than by spending environment steps, and one earlier design failed
entirely because it was written from first principles when a standard construction
existed.

Each attempt records: the papers checked, whether any reports **our** situation (a
frozen policy as prior; a buffer that is almost all failure; no true reward), and
what they say breaks. A candidate marked dead stays dead unless a new measurement or
a new paper reopens it.

## 8. Tasks  `[LOCKED]`

| task | `π_0` | role |
|---|---:|---|
| `chain3_lr2` | **0.010–0.021** measured | **target** — and where the paradigm **fails**: the same residual that gives +0.157 on `chain1b` gives +2 −1, p = 1.000 here. 0.5% of transitions come from successful episodes, so the advantage signal does not exist |
| `chain1b_lr2` | 0.39–0.60 | mechanism testbed; **not** low-success, never reported as such |

The success band qualifies a measurement setting; **it is not the research
objective**. A testbed is not retired because one method failed on it. The
transition needs no labels, so "can a world model be learned here" is answerable even
at 1% success; only the reward is label-starved.

## 9. Provenance  `[LOCKED]`

Trajectories are deployment-collected; labels are not. `env.is_success` needs a
detector or a human outside a simulator; BDDL goal predicates and milestone times do
not exist; candidate-level labels additionally required **deterministic replay**,
which a robot cannot do. Any result depending on the last three is a **simulator
result** and is labelled as one.

## 10. Diagnostics this framework depends on  `[LOCKED]`

Each exists because its absence cost real work. Run them in this order; each one can
stop the next.

| # | question | script | cost |
|---|---|---|---|
| D1 | Is there outcome variation to act on at all? | `probe_v123_outcome_variance.py` | one panel |
| D2 | What is the oracle ceiling for selection? | `probe_v132_selection_ceiling.py` | one panel |
| D3 | **What is the supervised ceiling for the reward's target?** | `probe_v179_stage_ceiling.py` | **zero env steps** |
| D4 | Does the signal survive the encoder? | inline in D3's harness — probe raw features vs the latent | zero |
| D5 | Does the reward order held-out failures? (§6.1) | `probe_v173_reward_diagnostic.py` | zero |
| D6 | Is the world model a predictor at all? | `probe_v159_wm_quality.py` | zero |

**D3 and D4 are the ones this round proved indispensable.** D3 showed the §6.1 bar
sat above the achievable ceiling on the target task, so four candidates were judged
against a height nothing could reach. D4 showed the raw features pass the bar
(0.417) while the learned latent does not (0.140), so five candidate failures were
measuring a representation, not an objective. **Both cost zero environment steps and
both were run only after the failures they would have explained.**

## 11. Standing status

| framework element | status |
|---|---|
| §4 reward menu | R1 dead, R2/R3 dead by reading, R4 dead, R5 unavailable locally, **R7 passes the primary bar on raw features / fails the language null**, R6 untried |
| §5 use menu | U0 run and capping U1; **U1 dead on the target task under both samplers**; U2 measured on one panel with replication in flight; U3, U4 untried |
| §6.1 gate | **verified predictive.** The reward that passed its bar (R7, ρ = 0.417) produced a replicated +0.157 in deployment; the reward that failed it (TD value, ρ = 0.247) produced −0.26. Bar re-derived as 0.6 × the measured ceiling |
| §6.3 roles | attested for the U2 cycle; one recorded violation — the residual scale was chosen on an earlier panel |
| §1 paradigm | improvement from deployment data **measured and replicated** (p = 0.00010); the world model's contribution **measured at exactly zero** (+25 −25, p = 1.000) and harmful relative to model-free use (p = 0.00018) |

**The binding constraint, measured.** Every ingredient of the improvement is derived
from episode outcomes, and the low-success regime the paradigm targets supplies
almost none: `chain1b` has outcome labels on ~50% of episodes and gains +0.157;
`chain3` has 0.5% and gains nothing. **A denser deployable label — partial credit
rather than a binary outcome — is the change most likely to move the target task**,
and it attacks the constraint rather than working around it.

**What is not yet answered.** Whether any construction makes the transition
contribute. Every isolation so far — five in the ranking setting, one in the
policy-improvement setting — has returned nothing or worse than nothing, but U3
(sequence optimisation) and U4 (uncertainty gating) are untried, and R6 sidesteps the
label constraint that bounds the others.
