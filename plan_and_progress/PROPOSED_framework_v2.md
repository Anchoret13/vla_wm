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

| # | candidate | requires | dies if | status |
|---|---|---|---|---|
| R1 | **quasimetric to a language anchor** — `V(z;ℓ) = −d(z, G(ℓ))`, `d` asymmetric; local-cost + spread + instruction-mismatch + cross-trajectory terms (QRL-style) | ≥2 instructions in the buffer; an anchor budget (~10 binary or ~200 pairwise judgments per instruction) | fails §6.1, or the mismatch term is vacuous | **next to try** |
| R2 | hindsight goal from terminal states (VIP/LIV-style) | nothing | — | **dead on arrival**: 97–99% of our terminal states are failures, so this teaches "the goal is where my policy stops" |
| R3 | `G(ℓ)` compared directly against the VLA's `h_t` | a joint text-image initialisation | — | **rejected**: `h_t` is a joint `(o,ℓ)` code; within one instruction the language part is a constant offset and contributes nothing to ordering |
| R4 | preference/ranking model from pairwise episode comparisons | ~200 comparisons per instruction | fails §6.1 | untried; cheapest labels, and failure-vs-failure comparisons are abundant here |
| R5 | a VLM used as a reward model (frame-instruction similarity) | an external VLM | fails §6.1, or is hacked | untried |
| R6 | uncertainty/novelty as an intrinsic signal, no reward at all | nothing | gives no task direction | untried; would sidestep the label problem entirely |

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

| # | use | transition needed? | risk | status |
|---|---|---|---|---|
| U0 | **oracle best-of-N ceiling** (privileged, upper bound only) | no | none | **run first** — if oracle best-of-16 cannot lift the target task, every re-ranking use is capped and we learn that for the price of one panel |
| U1 | re-rank `N` sampled chunks by the §4 signal | no | low: candidates come from the VLA's own distribution, so **there is no direction to ride** | try before U2 |
| U2 | bounded residual trained in imagination | **yes** — a critic cannot generate a rollout | high: produced −0.26 | after U0/U1 |
| U3 | optimisation over action sequences (MPPI-style, VLA as proposal) | yes | changes what is executed from a VLA sample to a model-refined chunk — a decision, not a default | untried |
| U4 | model uncertainty to gate when to intervene | yes | untested | untried |

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
| `chain3_lr2` | **0.010** measured | **target** — the low-success regime the paradigm is for |
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
