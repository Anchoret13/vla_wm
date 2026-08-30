# Framework v2 — a latent world model that improves a frozen VLA during rollout

**Status: proposed, replaces `framework_design.md` in full.** Written from zero
2026-08-29. Retains the substance of the old §2 (setting and VLA interface), §7.1
(goal-drift guardrail), §7.2.1 (data roles) and §8 (falsification table). Drops the
phase-potential/`QΦ`, anchor-eligibility, branch-ledger, migration, registration and
promotion-halt machinery, and the §14–§20 execution logs.

**§4 is provisional** pending a literature survey on language-grounded value
learning, launched before this was written.

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

## 4. Reward comes from language, not from labels  `[PROVISIONAL]`

### 4.1 Why the value function had to go

The previous design learned `V` from outcome labels. Three measured problems:

1. **The labels are privileged.** `env.is_success` needs a detector or a human
   outside a simulator; BDDL goal predicates and milestone times do not exist at
   all; candidate-level labels additionally required **deterministic replay**, which
   a robot cannot do.
2. **They are almost absent where the paradigm is aimed.** On the target task
   success is ~3%, so 96 episodes buy ~3 terminal anchors.
3. **The resulting value does not survive optimisation.** A residual trained
   against it gained +0.014 in imagination and lost **0.26** in the environment,
   saturating its norm bound at every scale (0.01–0.15) with the imagined gain
   growing linearly — an unconstrained extrapolating value with no interior optimum.

### 4.2 The instruction is the reward

The instruction already names the goal. Embed it into the latent space and let
reward be a decreasing function of distance to it:

```
g = G(ℓ)                     the language-specified goal, in latent space
r̂(z) = −d(z, g)              dense, every step, and LABEL-FREE
```

| | value from outcome labels | language-conditioned goal distance |
|---|---|---|
| supervision | `env.is_success` | **the instruction, already present** |
| density | terminal only | **every step** |
| target task (3% success) | ~3 anchors per 96 episodes | full signal on every episode |
| outside a simulator | needs a detector or a human | **available** |

Combined with §3's consistency term, **the entire world model becomes label-free**
and therefore deployable.

### 4.3 What must prevent collapse

The degenerate solution is to map every state to `g`. Distance alone cannot rule it
out; the objective needs negatives, and two kinds are available without labels:

- **Cross-instruction.** The same observation encoded under a *different*
  instruction should not be close to `G(ℓ)`. The VLA's `h` already varies with `ℓ`,
  so these negatives are free.
- **Temporal.** Within a trajectory, later states are closer to whatever that
  trajectory reached than earlier ones — an ordering constraint, not an absolute
  distance.

The exact objective is deferred to the survey. What is fixed: **any term that can be
minimised by ignoring the observation is disqualified**, and the collapse check is
run before the reward is used, not after.

### 4.4 The measurement that decides it

A progress function that rises along **every** trajectory is worthless, and on a
task that fails 97% of the time that failure mode is the default. So:

> Train with **zero outcome labels**. Then use held-out labels **only for
> evaluation** to test whether `d(z_t, G(ℓ))` falls along successful trajectories
> and **does not** fall along failed ones.

Labels touch nothing but the evaluation. If the separation is absent, the reward is
not a reward and nothing downstream is worth running.

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
joint-training variants below baseline; TD bootstrapping −0.0102 and −0.0055 with
intervals spanning zero.

**Use it where a critic cannot substitute.** A critic scores an action; it cannot
generate a rollout.

```
Δ_φ(z)  : a bounded residual on the VLA's chunk
executed: chunk_to_env(u_VLA + Δ_φ(z))
trained : argmax_φ E[Σ r̂(z_h)] over IMAGINED rollouts, ‖Δ‖ ≤ s
```

Consistent with §2: a bounded perturbation of a supported chunk, zero at
initialisation, VLA still the controller. The prior is the **trust region**.

**The bound is load-bearing.** On a fresh 96-seed panel against a base of
44/96 = 0.458: scale 0.15 → 19/96 = 0.198; scale 0.01 → 47/96 = 0.490. Whether a
language-grounded reward is less hackable than the label-trained value is **open and
must be measured, not assumed** — a learned similarity is a classic hacking target.

## 7. Falsification  `[LOCKED]`

**The refuting arm runs before the claim is stated.**

| claim | required comparison |
|---|---|
| the reward is a reward | falls on successes, **does not fall on failures**, held-out labels used for evaluation only |
| the improvement helps | vs the frozen VLA alone, paired, fit-disjoint panel |
| **the world model is what helped** | vs the same improvement trained **without imagination** — same reward, same anchor, real transitions only |
| the model is a model | one-step error vs the identity floor, **shuffled-action control**, compounding over horizon |

**Model quality is never inferred from a downstream metric.** Ranking does not need
dynamics, so ranking quality cannot select for it — measured anti-correlated here:
the best-ranking model (AUC 0.626) had a shuffled-action cost of 0.2%, while the
worse-ranking one (0.584) predicted at 0.530× identity.

### 7.1 Pre-registered readings

| observation | conclusion |
|---|---|
| the reward falls on failures too | it is a clock, not a reward — stop |
| no outcome variation in the proposal pool | the setting or proposal mechanism is wrong; say nothing about the model |
| model-guided selection does not beat random | the score gives no benefit; the model may still be fine |
| a correction arm improves equally without the model | supports corrective imitation, **not** a model-based claim |
| improves offline ranking but hurts behaviour | do not promote |
| round 1 improves, round 2 does not | one-shot adaptation, not continual improvement |

### 7.2 Data roles  `[LOCKED]`

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
