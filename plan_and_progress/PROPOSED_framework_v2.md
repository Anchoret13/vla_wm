# Framework v2 — a latent world model that improves a frozen VLA during rollout

**Status: proposed, replaces `framework_design.md` in full.** Written from zero
2026-08-29. Retains the substance of the old §2 (setting and VLA interface), §7.1
(goal-drift guardrail), §7.2 (required arms and role sealing) and §8 (falsification
table). Drops the phase-potential/`QΦ`, anchor-eligibility, branch-ledger,
migration, registration and promotion-halt machinery, which served abandoned lines,
and the §14–§20 execution logs, which belong in `plan_and_progress/<date>.md`.

---

## 1. The paradigm

A pretrained VLA is a **strong prior we do not modify**. While it is deployed we
collect its trajectories, train a **latent world model** on them, and use that model
to improve the VLA's own behaviour at rollout time. Then we redeploy and repeat.

```
run π (VLA + current improvement)  →  trajectories
trajectories                       →  latent WM: encoder, transition, value
WM                                 →  an improvement to π
                                   →  redeploy
```

Everything here exists to make that loop work and to make its claims falsifiable.

## 2. Setting and retained VLA interface  `[LOCKED]`

The VLA produces a `K = 50` action chunk from visual-action history and the
instruction; the executed commitment is `c = 10`, after which it replans.

- `π_0` is the pretrained policy, the representation source, and the initial action
  prior.
- **Every candidate chunk comes from `π_k`, from a frozen `π_0`, or from a bounded
  perturbation of their supported chunks.** The method never optimises an
  unconstrained 500-dimensional action sequence from scratch.
- **The learned policy remains the final controller.** Deployment is full-prompt
  `N = 1` with no external planner.
- Policy-update interfaces are zero-initialised (AdaRMS/flow-boundary injection,
  stock-initialised output head) so behaviour is exactly stock at initialisation.

**Executed-action causal contract.** The VLA proposes 50 actions; at most `c = 10`
are committed before replanning. **Only executed actions may carry causal credit
anywhere in the system.**

The physical transition is language-independent; the predictive state is
task-conditioned, because language decides which future distinctions matter. The
transition therefore needs no second task-ID or raw-language bypass — language
enters through the state.

## 3. The latent state

`z = Enc(h, p)` — the VLA's pooled prefix hidden `h` and proprioception `p` (eef
pose, gripper, joints; 25 dims).

**Proprioception is not optional.** Measured on `chain1b_lr2` over `c = 10`:

| component | action gain over an action-free model | shuffled action |
|---|---:|---:|
| pooled prefix hidden (2048) | +0.81% | 0.597× identity |
| **proprioception (25)** | **+11.78%** | **1.217× identity** |

The hidden state is trained for language-vision alignment; nothing asked it to be
predictable under actions, and over ten steps it barely moves. A *shuffled* action
makes the proprioception prediction **worse than not predicting at all** — the
signature of a dynamics model, which a smoother cannot produce.

The loss is weighted per component: unweighted, the 25 informative dimensions carry
~1.2% of the gradient and the signal vanishes (+11.78% alone versus +0.84% pooled).

## 4. The world model, and what needs labels  `[LOCKED]`

**The split that matters for real deployment:**

| component | supervision | obtainable outside a simulator |
|---|---|---|
| encoder + transition | `(z, u, z′)` — **none** | **yes**, fully self-supervised from rollouts |
| value | outcome labels | **only sparse, delayed, possibly noisy ones** |

The transition is label-free. **Design so that the model's usefulness degrades
gracefully as labels become scarce**, because in the real world true reward is not
available: success must come from a detector or a human, and dense per-step progress
does not exist at all.

**Privileged signals — simulator-only, never to be required by the method:**

| signal | source | status |
|---|---|---|
| `info["is_success"]` | `env.check_success()` | needs a detector or human outside sim |
| goal predicates, `Δw` | BDDL via `predicate_bits` | **unavailable**; must not be a dependency |
| milestone times | `GoalAutomaton` | unavailable |
| candidate-level outcome labels | **deterministic replay** | unavailable — a robot cannot rewind |

Any result that depends on the last three is a simulator result and must be labelled
as one.

**Training.** Encoder, transition and value are trained by **one objective**, on
deployment trajectories, with **no reconstruction**:

| term | form | purpose |
|---|---|---|
| consistency | `‖T(z_t, u_t) − sg(Enc(o_{t+1}))‖²` | the transition |
| value | TD on executed actions; terminal = episode outcome | makes the latent task-relevant |

The latent is **bounded** (SimNorm-style groups) and the consistency target carries
**stop-grad**. Collapse without a decoder is real and was observed: a Gaussian
variant reached identity error 0.0004 with its transition scoring 5.5× worse than
doing nothing. The target is **not** the exact next latent — sufficiency for value
and consistency is all that is required.

## 5. How the model is used — and how it is not

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
trained : argmax_φ E[V(z_H)] over IMAGINED latent rollouts, ‖Δ‖ ≤ s
```

Consistent with §2: `Δ` is a bounded perturbation of a supported chunk, zero at
initialisation, and the VLA remains the controller. The prior is the **trust
region** — the residual only learns a small correction, which is why this is
attemptable at ~5k transitions.

**The bound is load-bearing.** The residual saturates it at every scale tried
(0.01/0.03/0.06/0.15) and the imagined gain grows linearly with it: `V` is monotone
along the residual direction with no interior optimum. On a fresh 96-seed panel
against a base of 44/96 = 0.458:

| scale | success |
|---:|---:|
| 0.15 | 19/96 = 0.198 |
| **0.01** | **47/96 = 0.490** |

Value extrapolation, not a wrong gradient. Pessimism over a value ensemble is the
intended correction; whether it permits a larger useful scale is open.

## 6. Falsification  `[LOCKED]`

Every claim about the world model requires the arm that removes it. **The refuting
arm runs before the claim is stated.**

| claim | required comparison |
|---|---|
| the improvement helps | vs the frozen VLA alone, paired, on a fit-disjoint panel |
| **the world model is what helped** | vs the same improvement trained **without imagination** — same value, same anchor, real transitions only |
| the model is a model | one-step error vs the identity floor, **shuffled-action control**, compounding over horizon, value gap |

**Model quality is never inferred from a downstream metric.** Ranking does not need
dynamics, so ranking quality cannot select for it — and here the two are
anti-correlated: the best-ranking model (AUC 0.626) had essentially no dynamics (a
shuffled action cost it 0.2%), while the worse-ranking one (0.584) predicted at
0.530× identity with value correlation 0.88–0.95.

### 6.1 Pre-registered readings

| observation | conclusion |
|---|---|
| no outcome variation in the proposal pool | the setting or the proposal mechanism is wrong — say nothing about the world model |
| variation exists but the model cannot predict held-out siblings | the model is inadequate for the coverage acquired |
| model-guided selection does not beat random | the score provides no benefit; the model may still be fine |
| a correction arm improves equally without the model | this supports corrective imitation, **not** a model-based claim |
| the model improves offline ranking but hurts behaviour | gating/support is insufficient; do not promote |
| round 1 improves, round 2 does not | one-shot adaptation, not continual improvement |
| the full method wins and the no-model controls do not | promote, then stress-test |

### 6.2 Data roles  `[LOCKED]`

Each episode carries a role — `model_train`, `model_calib`, `policy_train`,
`eval` — partitioned before collection. Branches inherit their source episode's
role. **No parameter, threshold or hyper-parameter may be chosen using `eval` data**;
a threshold fitted on the evaluation panel is leakage even if the panel was never
"trained on".

## 7. Tasks  `[LOCKED]`

| task | `π_0` | role |
|---|---:|---|
| `chain3_lr2` | 0.031 | **target** — the low-success regime the paradigm is for |
| `chain1b_lr2` | 0.39–0.60 | mechanism testbed; **not** low-success, never reported as such |

The success band qualifies a measurement setting; **it is not the research
objective**. A testbed is not retired because one method failed on it: the
transition needs no labels, so "can a world model be learned here" is answerable
even where success is ~3%. Only the value is label-starved.
