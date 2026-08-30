# Framework v2 — a latent world model that improves a frozen VLA during rollout

**Status: proposed, replaces `framework_design.md` in full.** Written from zero
2026-08-29. The previous document had grown to 21 sections, of which §14–§20 were
execution logs rather than design; the phase-potential, anchor-eligibility,
branch-ledger and promotion-halt machinery it specified belongs to abandoned lines
and is dropped.

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

Everything in this document exists to make that loop work and to make its claims
falsifiable. Nothing else is in scope.

**Constraints, all binding.** VLA weights and prompt are frozen. The deployment
contract is unchanged (full instruction, `N=1`, one 10-action chunk per replan). The
world model works in **latent space with no reconstruction**. Training data comes
from deployment rollouts. The target task has a **low** success rate.

## 2. The latent state

`z = Enc(h, p)` where `h` is the VLA's pooled prefix hidden and `p` is
proprioception (eef pose, gripper, joints — 25 dims).

**Proprioception is not optional.** Measured on `chain1b_lr2` over `c = 10`:

| component | action gain over an action-free model | shuffled action |
|---|---:|---:|
| pooled prefix hidden (2048) | +0.81% | 0.597× identity |
| **proprioception (25)** | **+11.78%** | **1.217× identity** |

The VLA's hidden state is trained for language-vision alignment; nothing ever asked
it to be predictable under actions, and over ten steps it barely moves. Feeding a
*shuffled* action makes the proprioception prediction **worse than not predicting at
all** — the signature of a dynamics model, which a smoother cannot produce. Without
proprioception there is no action-conditioned latent and §3 has no object.

The loss is weighted per component: under a mean-over-dims loss the 25 informative
dimensions carry ~1.2% of the gradient and the signal disappears (+11.78% alone
versus +0.84% pooled unweighted).

## 3. The world model

Encoder, transition and value are trained by **one objective**, on deployment
trajectories, with no decoder:

| term | form | purpose |
|---|---|---|
| consistency | `‖T(z_t, u_t) − sg(Enc(o_{t+1}))‖²` | the transition |
| value | TD/SARSA on executed actions, terminal = episode success | makes the latent task-relevant |

The latent is **bounded** (SimNorm-style groups) and the consistency target carries
**stop-grad**. Without a decoder, collapse is real and was observed: a Gaussian
variant reached identity error 0.0004 with its transition scoring 5.5× worse than
doing nothing. Bounding, stop-grad and the value term together are what prevent it.

The target is **not** the exact next latent. The latent only has to be sufficient
for value and consistency.

## 4. How the model is used — and how it is not

**Ranking candidates is not a use of a world model.** At a branch point every
candidate shares `z`, so `T(z, u)` is a deterministic feature map of `(z, u)` and
lies **inside the function class of a critic fed `(z, E_a(u))`**. It cannot beat one
at any data scale. This is not a post-hoc story: it explains five independent nulls
measured before the argument was found — deployment ablation p = 0.728, offline
ranking 0.637 vs 0.636, two joint-training variants below baseline, TD bootstrapping
−0.0102 and −0.0055 with intervals spanning zero.

**The model must be used where a critic cannot substitute.** A critic scores an
action; it cannot generate a rollout. So:

```
Δ_φ(z)  : a bounded residual on the VLA's chunk
executed: chunk_to_env(u_VLA + Δ_φ(z))
trained : argmax_φ E[V(z_H)] over IMAGINED latent rollouts, ‖Δ‖ ≤ s
```

The VLA is both the prior and the **trust region**: `Δ` starts at zero and is
bounded, so the residual only learns a small correction. That is why this is
attemptable at ~5k transitions when learning a policy from scratch is not.

**The bound is load-bearing.** The residual saturates it at every scale tried
(0.01/0.03/0.06/0.15) and the imagined gain grows linearly with it: `V` is monotone
along the residual direction with no interior optimum. Measured on a fresh 96-seed
panel against a base of 44/96 = 0.458:

| scale | success |
|---:|---:|
| 0.15 | 19/96 = 0.198 |
| **0.01** | **47/96 = 0.490** |

Value extrapolation, not a wrong gradient. An ensemble value optimised
pessimistically is the intended correction; whether it permits a larger useful scale
is open.

## 5. Falsification

Every claim about the world model requires the arm that removes it.

| claim | required comparison |
|---|---|
| the improvement helps | vs the frozen VLA alone, paired, on a fit-disjoint panel |
| **the world model is what helped** | vs the same improvement trained **without imagination** — same value, same anchor, real transitions only |
| the model is a model | one-step error vs the identity floor, **shuffled-action control**, error compounding over horizon, value gap |

**Model quality is never inferred from a downstream metric.** Ranking does not need
dynamics, so ranking quality cannot select for it — and here the two are
anti-correlated: the best-ranking model (AUC 0.626) had essentially no dynamics
(a shuffled action cost it 0.2%), while the worse-ranking one (0.584) predicted at
0.530× identity with value correlation 0.88–0.95.

## 6. Tasks

| task | `π_0` | role |
|---|---:|---|
| `chain3_lr2` | 0.031 | **target** — the low-success regime the paradigm is for |
| `chain1b_lr2` | 0.39–0.60 | mechanism testbed; not low-success, do not report as such |

A testbed is not retired because one method failed on it. The transition needs no
labels — it trains on `(z, u, z')` alone — so "can a world model be learned here" is
answerable even where success is ~3%. Only the value is label-starved.

## 7. Provenance

Trajectories are deployment-collected; **labels are not**. Episode success comes
from `env.is_success`, which a deployed system could plausibly obtain from a
detector or a human. Goal-predicate progress (`Δw`) and candidate-level outcome
labels are **privileged**: the latter additionally required deterministic replay,
which a real robot cannot do. Current work uses episode success only.
