# Framework v2 — a latent world model that improves a frozen VLA during rollout

**Status: proposed, replaces `framework_design.md` in full.** Written from zero
2026-08-29. Retains the substance of the old §2 (setting and VLA interface), §7.1
(goal-drift guardrail), §7.2.1 (data roles) and §8 (falsification table). Drops the
phase-potential/`QΦ`, anchor-eligibility, branch-ledger, migration, registration and
promotion-halt machinery, and the §14–§20 execution logs.

§4 and §6 were rewritten 2026-08-30 from a literature survey (VIP, LIV, QRL,
quasimetric value learning, reward-hacking mitigations) and from the diagnostic in
§7.1, which was run on the value we already had before any of this was designed.

---

## 1. The paradigm

A pretrained VLA is a **strong prior we do not modify**. While it is deployed we
collect its trajectories, train a **latent world model** on them, and use that model
to improve the VLA's own behaviour at rollout time. Then we redeploy and repeat.

```
run π (VLA + current improvement)  →  trajectories
trajectories                       →  latent WM: encoder, transition, reward
WM                                 →  an improvement to π
                                   →  redeploy
```

**The binding constraint is that this must work outside a simulator.** Weights and
prompt frozen; deployment contract unchanged (full instruction, `N = 1`, `c = 10`);
latent space, no reconstruction; data from deployment rollouts; **no true reward**.

## 2. Setting and retained VLA interface  `[LOCKED]`

The VLA produces a `K = 50` chunk from visual-action history and instruction `ℓ`;
`c = 10` is committed before replanning.

- `π_0` is the pretrained policy, the representation source, and the action prior.
- **Every candidate chunk comes from `π_k`, a frozen `π_0`, or a bounded
  perturbation of their supported chunks.** Never an unconstrained optimisation over
  a 500-dimensional action sequence.
- **The learned policy remains the final controller.** Full-prompt `N = 1`, no
  external planner.
- Policy-update interfaces are zero-initialised, so behaviour is exactly stock at
  initialisation.
- **Executed-action causal contract.** Only executed actions carry causal credit
  anywhere in the system.

The physical transition is language-independent; the predictive state is
task-conditioned, because language decides which future distinctions matter.

## 3. The latent state is language-conditioned

`z_t = Enc(h_t, p_t)` where `h_t = PrefixVLM(o_t, ℓ)` and `p_t` is proprioception.

**`h_t` is already a function of the instruction** — the VLA computes it from
observation *and* language. That is the property §4 exploits.

**Proprioception is not optional.** Measured on `chain1b_lr2` over `c = 10`:

| component | action gain over an action-free model | shuffled action |
|---|---:|---:|
| pooled prefix hidden (2048) | +0.81% | 0.597× identity |
| **proprioception (25)** | **+11.78%** | **1.217× identity** |

The hidden state is trained for language-vision alignment; nothing asked it to be
predictable under actions. A *shuffled* action makes the proprioception prediction
**worse than not predicting at all** — the signature of a dynamics model, which a
smoother cannot produce. The loss is weighted per component: unweighted, 25 dims
carry ~1.2% of the gradient and the signal vanishes (+11.78% → +0.84%).

## 4. Reward comes from language, not from labels

### 4.1 Why the value function had to go

The previous design learned `V` from outcome labels. Three measured problems:

1. **The labels are privileged.** `env.is_success` needs a detector or a human
   outside a simulator; BDDL goal predicates and milestone times do not exist at
   all; candidate-level labels additionally required **deterministic replay**.
2. **They are almost absent where the paradigm is aimed.** Measured on
   `chain3_lr2`: **1/96 = 0.010**. Ninety-six episodes buy one terminal anchor.
3. **The resulting value does not survive optimisation.** A residual trained
   against it gained +0.014 in imagination and lost **0.26** in the environment.

### 4.2 The goal is an anchor in the metric, not an observed state

Two constructions are ruled out before any objective is written.

**Not hindsight terminal states.** VIP/LIV define the goal as the last frame of the
video. **97–99% of our last frames are failures.** Hindsight relabelling on this
buffer teaches "the goal is wherever my policy usually stops" — dense, smooth,
label-free, and it scores our modal failure as success. This is the most likely way
to reproduce the +0.014/−0.26 result behind a better-looking loss curve.

**Not `G(ℓ)` compared against `h_t`.** `h_t` is a **joint** `(o_t, ℓ)` code, not a
state code. Within one instruction the language contribution is a constant offset
shared by every step of every episode, so it contributes nothing to ordering; across
instructions it dominates. LIV needed CLIP initialisation on *both* towers to make
text and images commensurate and still reported non-monotone rewards on unseen robot
data. We have no such joint initialisation — and do not need one, because the
instruction is already inside `h_t`.

**Instead:** `g_ℓ = G_ψ(e_ℓ)`, a trainable projection of a frozen sentence embedding
into the same space as `z`. `g_ℓ` is an **anchor in the metric space, never required
to equal any observed state**; its location is fixed by the objective.

`V(z; ℓ) := −d_ξ(z, g_ℓ)` with `d_ξ` an **asymmetric** quasimetric (IQE/MRN):
`d ≥ 0`, `d(z,z) = 0`, triangle inequality, all by construction. Asymmetry is not a
technicality: a block knocked off the table is one step from the pre-grasp state,
the pre-grasp state is many steps from the fallen block. A symmetric `‖z − g‖` must
average the two, flattening the value exactly around irreversible events — which is
what our failures consist of — and its only interior optimum is at `g` itself, a
plausible mechanism for the monotone-with-no-optimum pathology we measured.

### 4.3 The objective, and what each term rules out

**QRL, not VIP, as the skeleton.** VIP's TD term is anchored on the video's own last
frame, so on a failed rollout every term is fitted as though the failure end-state
were the goal. QRL's local constraint — one environment step costs at most 1 — is
**true on failed rollouts as well as successful ones** and takes no goal argument.
It is the only member of the family whose label-free terms are not actively
mis-supervised by a 97%-failure buffer.

| term | form | rules out |
|---|---|---|
| (a) local cost | `relu(d(z_t, z_{t+1}) − 1)²` | **nothing alone** — `d ≡ 0` satisfies it. It sets the scale: one chunk = one unit |
| (b) global spread | `−E[φ(d(z, g))]`, `φ(x)=x/(1+x)` | total collapse of `z`, `g_ℓ`, and `d` |
| (c) instruction mismatch | `relu(m − d(z_t, g_{ℓ′}))²`, `ℓ′ ≠ ℓ` | **the goal-agnostic timer** — the label-free analogue of the pathology we measured |
| (d) cross-trajectory | `relu(m′ − d(z_i, z_j))²`, different episodes | spuriously low cost between states no observed path connects |

Run (a)+(b) as a Lagrangian so `λ` is a **monitorable scalar**: a runaway `λ` means
the terms are incompatible on this data, which is itself a result.

**Structural precondition — a data requirement, not a detail.** Term (c) is vacuous
unless the buffer contains **≥ 2 distinct instructions**. With one task, `ℓ` is
constant, `g_ℓ` is a single free vector, "language-conditioned" degenerates to
"task-specific", and the diagnostic in §7 cannot be run. **Deployment rollouts are
collected under several chain instructions, not only the target.**

### 4.4 Reward is a difference, not a level

```
r̂(z_t, z_{t+1}; ℓ) = V(z_{t+1}; ℓ) − V(z_t; ℓ)
```

Potential-based shaping: it telescopes to a boundary term and is therefore
**policy-invariant**. A policy cannot accumulate unbounded return by walking the
latent toward `g`. Our measured pathology — value monotone along the residual
direction, no interior optimum — is what optimising a *level* produces. VIP's own
downstream use is the differenced form; we had used the level.

### 4.5 Label budget — stated honestly

**The metric is label-free. The anchor is not.** Nothing in §4.3 pins `g_ℓ` to the
*actual* goal: (b) places it maximally far from the state distribution, which on a
97%-failure buffer is an extrapolated corner no reachable state occupies. The
geometry will be well-formed and pointed at the wrong place.

Minimum anchor budget, to be **measured as a curve, not asserted**:

| form | budget per instruction | what is asked of the labeller |
|---|---|---|
| absolute | ~5–10 successful terminal latents | one binary judgment per episode |
| **relative (preferred)** | **~200 pairwise comparisons** | "which of these two got further" |

The relative form is preferred because at 1% success the absolute form needs ~1000
episodes per instruction to collect 10 anchors, while pairwise comparisons are
abundant precisely because failure-vs-failure is the modal case. Either way: **do
not call this route label-free.** The honest reduction is from per-step privileged
state predicates to ~10 binary or ~200 pairwise **episode-level** judgments.

## 5. The world model

Encoder, transition and reward trained by **one objective** on deployment
trajectories, no decoder:

| term | form | labels |
|---|---|---|
| consistency | `‖T(z_t, u_t) − sg(Enc(o_{t+1}))‖²` | none |
| language-grounded reward | §4.3 | none |

The latent is **bounded** (SimNorm-style groups) and the consistency target carries
**stop-grad**. Collapse without a decoder is real and was observed: a Gaussian
variant reached identity error 0.0004 with its transition scoring 5.5× worse than
doing nothing. The target is **not** the exact next latent — sufficiency for reward
and consistency is all that is required.

## 6. How the model is used — and how it is not

**Ranking candidates is not a use of a world model.** At a branch point every
candidate shares `z`, so `T(z, u)` is a deterministic feature map of `(z, u)` and
lies **inside the function class of a critic fed `(z, E_a(u))`**. It cannot beat one
at any data scale. This explains five independent nulls measured before the argument
was found: deployment ablation p = 0.728; offline ranking 0.637 vs 0.636; two
joint-training variants below baseline; TD bootstrapping −0.0102 and −0.0055.

**Order of work, and the ceiling comes first.**

1. **Measure the ceiling before building.** Oracle best-of-`N`: sample `N` chunks
   per step and select with privileged success (upper bound only). **If oracle
   best-of-16 does not lift the target task appreciably, no re-ranking value —
   perfect or otherwise — can help**, and this route is capped. Cheap and decisive.
2. **Re-rank, do not descend.** Score `N` sampled chunks with the potential
   difference of §4.4 and execute the argmax. The candidate set is drawn from the
   VLA's own distribution, so **no direction exists to ride however wrong the value
   is**; the worst case degrades to a bad ranking of plausible actions. This
   structurally removes the failure we measured. `N` is swept, not maximised —
   best-of-`N` reward gain turns over.
3. **Only then imagination.** A bounded residual trained on imagined rollouts is the
   use in which the transition is irreplaceable, but it is also the use that
   produced −0.26. It is attempted after (1) and (2), not before.

**Correction to an earlier reading.** That the residual "saturates its norm bound at
every scale" is **not** evidence the value is wrong: against a monotone objective, a
hard norm ball puts the KKT solution on the boundary at every radius. And shrinking
the bound until the environment number stops dropping would be **fitting the bound
on the evaluation panel** — the same class of leak already in the withdrawal table.

## 7. Falsification  `[LOCKED]`

**The refuting arm runs before the claim is stated.**

| claim | required comparison |
|---|---|
| the reward is a reward | **§7.1 failure stratification**, held-out labels used for evaluation only |
| the improvement helps | vs the frozen VLA alone, paired, fit-disjoint panel |
| **the world model is what helped** | vs the same improvement trained **without imagination** — same reward, same anchor, real transitions only |
| the model is a model | one-step error vs the identity floor, **shuffled-action control**, compounding over horizon |

**Model quality is never inferred from a downstream metric.** Ranking does not need
dynamics, so ranking quality cannot select for it — measured anti-correlated here:
the best-ranking model (AUC 0.626) had a shuffled-action cost of 0.2%, while the
worse-ranking one (0.584) predicted at 0.530× identity.

### 7.1 The reward diagnostic  `[LOCKED]`

Monotonicity within a trajectory is worthless as evidence: a clock scores perfectly
on it. What a usable value must do is order **failures** by how far they got — a
cross-trajectory property needing no successes at all, so on a 97%-failure task the
whole buffer is the sample.

```
ρ = Spearman( V(z_T; ℓ), stage_reached )   over held-out FAILED episodes
```

`stage_reached` is privileged and used **for evaluation only, never for training**.

| null | expected |
|---|---|
| shuffled instruction `g_{ℓ′}` | ρ → 0, or the value is not language-conditioned |
| clock | ρ → 0 |
| label-shuffled `stage_reached` | ρ → 0, CI covering 0 |

**Pre-registered bar:** ρ ≥ 0.35, bootstrap 95% CI excluding 0.15, all nulls in
[−0.1, 0.1]. The interval is the result, not the point estimate.

**Measured on the value we already have** (chain1b, 188 failed episodes):

| | value |
|---|---|
| ρ | **+0.247**, CI [+0.061, +0.412] |
| clock null | nan — all failures run the full horizon, so it is constant |
| label-shuffle null | −0.008 |
| **bar** | **NOT MET** (CI lower 0.061 < 0.15) |

So the current value carries **some** progress information and not enough. Run this
on any new reward **before** training a policy against it.

*Statistical note.* Ranking must use average ranks for ties. `argsort(argsort(x))`
assigns arbitrary distinct ranks to tied values and manufactured a spurious clock
null of +0.64 from a constant input; `stage_reached` is heavily tied (182 of 191
`chain3` failures share one value).

### 7.2 Pre-registered readings

| observation | conclusion |
|---|---|
| the reward falls on failures too | it is a clock, not a reward — stop |
| no outcome variation in the proposal pool | the setting or proposal mechanism is wrong; say nothing about the model |
| model-guided selection does not beat random | the score gives no benefit; the model may still be fine |
| a correction arm improves equally without the model | supports corrective imitation, **not** a model-based claim |
| improves offline ranking but hurts behaviour | do not promote |
| round 1 improves, round 2 does not | one-shot adaptation, not continual improvement |

### 7.3 Data roles  `[LOCKED]`

Episodes carry a role — `model_train`, `model_calib`, `policy_train`, `eval` —
partitioned before collection. **No parameter or threshold may be chosen using
`eval` data**; fitting a threshold on the evaluation panel is leakage even if that
panel was never "trained on".

## 8. Tasks  `[LOCKED]`

| task | `π_0` | role |
|---|---:|---|
| `chain3_lr2` | 0.031 | **target** — the low-success regime the paradigm is for |
| `chain1b_lr2` | 0.39–0.60 | mechanism testbed; **not** low-success, never reported as such |

The success band qualifies a measurement setting; **it is not the research
objective**. A testbed is not retired because one method failed on it.

## 9. Provenance  `[LOCKED]`

Trajectories are deployment-collected. Under §4 the model needs **no labels at
all**; labels appear only in evaluation. Any result that depends on BDDL predicates,
milestone times, or deterministic-replay candidate labels is a **simulator result**
and is labelled as one.
