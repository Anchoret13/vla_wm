# pi05-lcwm — Framework Design v0.4

Working design record, updated 2026-07-25.

This version keeps the implemented LC-Flow backbone but changes the method
center from one-step branch-effect regression to a recursively predictive world
state that is learned inside, and used to improve, a pretrained VLA.

`[LOCKED]` denotes the current method commitment. `[DEC]` denotes an unresolved
design choice. `[EST]` denotes a quantity that must be measured.

---

## 0. Research identity

### Central question

> Can world-model learning transform a pretrained VLA representation that is
> sufficient for expert-action prediction into a language-conditioned
> predictive state that is sufficient for long-horizon decision making?

The intended claim is not merely that a world model can rank actions proposed by
a VLA. The world model must alter the state used by the VLA itself, generate a
policy-improvement signal during training, and improve the final `N = 1` VLA at
deployment.

### Working thesis

Standard VLA imitation training requires the representation to predict the
demonstrated action. It does not require the representation to remain closed
under alternative actions or to retain the future distinctions needed for
long-horizon reward and value prediction.

We therefore augment π0.5 with a persistent predictive state and train a compact
world model to predict:

1. the next latent state under an executed action block;
2. instruction-independent physical effects;
3. instruction-dependent reward, progress, and continuation value.

The learned model then scores policy-supported action chunks and distills its
preferred actions back into π0.5. Deployment retains only the modified `N = 1`
VLA policy and its recurrent state update; no external MPC or best-of-N search is
required.

### Three indispensable roles

| Component | Required role |
|---|---|
| Pretrained VLA | supplies visual-language features, a coherent action prior, and the final deployed policy |
| World model | imposes action-conditioned recursive prediction and generates policy-improvement targets |
| Decision-sufficient abstraction | specifies what must be predicted without reconstructing the full visual future |

The framework fails its intended identity if any of the following is true:

- replacing π0.5 with an arbitrary action proposal mechanism leaves the main
  method and claim unchanged;
- removing world-model learning leaves policy performance unchanged under a
  matched-capacity and matched-data control;
- the final system improves only through external best-of-N selection;
- the learned state predicts task phase or success but not action-conditioned
  future differences.

### Non-goals

- pixel/video reconstruction;
- a universal transition model over arbitrary continuous actions;
- exact recovery of the physical simulator state;
- a new planning or policy-optimization operator;
- assuming arbitrary `(state, action)` generative access;
- treating language as a cause of physical dynamics.

---

## 1. Environment, VLA, and decision interface

Let

\[
H_t=(o_{\leq t},a_{<t})
\]

denote the available visual-action history and let \(\ell\) denote a natural
language instruction. The pretrained VLA produces a \(K\)-step action chunk:

\[
u_t = a_{t:t+K-1}
\sim
\pi_0(\cdot\mid H_t,\ell).
\]

For the current π0.5/LIBERO implementation:

- action chunk length: \(K=50\);
- executed commitment: \(c=10\);
- the policy replans after observing the next state;
- the main deployment interface samples one chunk per decision: `N = 1`.

The physical system remains language-independent:

\[
x_{t+c}\sim P^{(c)}(\cdot\mid x_t,u_{t,0:c}),
\]

but language determines which distinctions must be preserved in a
finite-capacity predictive state:

\[
z_t^\ell=\phi_\theta(H_t,\ell).
\]

Consequently, the induced abstract dynamics may be instruction-indexed even
though the physical transition is not:

\[
P_{\phi}^{\ell}(z_{t+c}\mid z_t,u_t).
\]

In the implementation, the π0.5 prefix KV cache and the recurrent LC state are
both internal VLA features. Conceptually, their pair is the complete internal
decision state. We do not claim that the recurrent carry alone is a minimal
sufficient statistic.

### 1.1 Working decision-sufficiency criterion

The abstraction is deliberately relative to:

- the current task/instruction family;
- the action chunks supported by the pretrained VLA;
- the decision stride and horizon relevant to chain execution.

For \(u\) in the VLA-supported candidate class
\(\mathcal C_{\pi_0}(H_t,\ell)\), the working requirements are:

\[
\left|
\mathbb E[r_t^\ell\mid H_t,u]
-
\bar r^\ell(z_t^\ell,u)
\right|
\leq \epsilon_r,
\]

\[
d\!\left(
P_\phi^\ell(\cdot\mid H_t,u),
\bar P_\theta^\ell(\cdot\mid z_t^\ell,u)
\right)
\leq \epsilon_P,
\]

\[
\left|
Q_\ell^\pi(H_t,u)
-
\bar Q_\ell^\pi(z_t^\ell,u)
\right|
\leq \epsilon_Q.
\]

Thus the state should preserve reward, action-conditioned abstract transition,
and candidate value/order within the VLA decision class. It need not preserve
every visual detail or support arbitrary off-manifold actions.

These are working functional criteria, not theoretical guarantees claimed by
v0.4. The corresponding empirical tests are reward/value calibration,
next-latent prediction, controlled action effects, and candidate-ranking regret.
The existing prefix-to-action path remains as a redundant pretrained perception
route; retaining that skip connection does not by itself certify or refute
sufficiency of the learned recurrent state.

---

## 2. Locked architecture: LC-Flow

### 2.1 Prefix representation and posterior state

The frozen π0.5 PrefixVLM processes the current observation and instruction:

\[
h_t^\ell,\,\mathrm{KV}_t^\ell
=
\operatorname{PrefixVLM}(o_t,\ell).
\]

The action-conditioned prior is propagated from the preceding posterior:

\[
\bar z_t^\ell
=
T_\theta\!\left(
z_{t-c}^\ell,
E_a(u_{t-c,0:c})
\right).
\]

The current observation corrects the prior:

\[
z_t^\ell
=
U_\theta(\bar z_t^\ell,h_t^\ell).
\]

`[LOCKED]`

- reuse the same real-prompt prefix forward and KV cache used by π0.5;
- keep `T` free of a second explicit language input;
- let language enter the transition through \(z_t^\ell\);
- reset the recurrent state only at environment reset;
- retain the current `E_a`, `T`, `U`, and LC-state token interface.

### 2.2 Direct VLA integration

The LC state changes the π0.5 flow field through the existing zero-initialized
AdaRMS adapter:

\[
c_{\mathrm{expert}}
=
c_{\mathrm{time}}
+
W_z\operatorname{Pool}(z_t^\ell),
\]

\[
v_\psi(x_\tau,\tau\mid h_t^\ell)
\longrightarrow
v_\psi(x_\tau,\tau\mid h_t^\ell,z_t^\ell).
\]

At initialization, the extension is an exact stock-policy no-op.

`[LOCKED]`

- π0.5 is the initialization, representation source, and final policy;
- the primary endpoint is full-prompt `N = 1` behavior;
- best-of-N reranking remains a diagnostic or teacher, not the deployed method.

`[DEC]` Keep the action expert frozen initially. Escalate to PEFT or selective
unfreezing only if the trained state adapter has insufficient causal influence
on the first ten generated actions.

---

## 3. Recursive predictive world state

The current one-block transition is retained, but it must be trained against the
next posterior state rather than only against manually chosen effect heads.

### 3.1 One-step predictive closure

For a real transition, compute the predicted prior

\[
\tilde z_{t+c}^\ell
=
T_\theta(z_t^\ell,E_a(u_{t,0:c}))
\]

and an observed next-state target from an EMA target encoder:

\[
z_{t+c}^{+,\ell}
=
\operatorname{sg}\!\left[
\phi_{\bar\theta}(H_{t+c},\ell)
\right].
\]

The first predictive-state loss is

\[
\mathcal L_{\mathrm{self}}^{(1)}
=
d\!\left(
p_\theta(\tilde z_{t+c}^\ell),
z_{t+c}^{+,\ell}
\right),
\]

where \(p_\theta\) is a small prediction head and
\(\bar\theta\leftarrow \operatorname{EMA}(\theta)\).

Operationally, \(\phi_{\bar\theta}\) is an EMA target recurrence over the
observed history, not a stateless projection of the next frame:

\[
\bar z_{t+c}^{+,\ell}
=
T_{\bar\theta}
\left(
z_t^{+,\ell},E_{a,\bar\theta}(u_{t,0:c})
\right),
\qquad
z_{t+c}^{+,\ell}
=
U_{\bar\theta}
\left(
\bar z_{t+c}^{+,\ell},h_{t+c}^\ell
\right).
\]

The stored `next_prefix_hidden_full` supplies \(h_{t+c}^\ell\) for the first
implementation. It is an input to the EMA posterior target, not itself the
complete target state, and the model does not regress to every nuisance
dimension of the raw PrefixVLM hidden feature.

### 3.2 Short latent rollout

After the one-step path is stable, train two- and three-block open-loop rollout
on ordinary trajectory windows:

\[
\tilde z_{t+jc}^\ell
=
T_\theta^{(j)}
\left(
z_t^\ell,
u_{t,0:c},
\ldots,
u_{t+(j-1)c,0:c}
\right),
\qquad j\in\{2,3\},
\]

\[
\mathcal L_{\mathrm{roll}}
=
\sum_{j=2}^{3}
\alpha_j
d\!\left(
p_\theta^{(j)}(\tilde z_{t+jc}^\ell),
\operatorname{sg}[z_{t+jc}^{+,\ell}]
\right).
\]

Observation correction remains active at deployment. Open-loop rollout is a
training and evaluation constraint that tests whether the state is recursively
predictive; it is not a commitment to long-horizon MPC.

### 3.3 Why self-prediction is necessary but not sufficient

Latent agreement alone can collapse or preserve irrelevant information. It is a
structural requirement for a world model, but certification still requires
observable consequences and behavior:

- controlled physical-effect prediction;
- task reward/progress prediction;
- candidate ranking and top-1 regret;
- long-horizon `N = 1` policy improvement.

The updated interpretation is therefore:

> Self-prediction defines recursive closure; grounded outcome and policy tests
> establish decision relevance.

---

## 4. Grounded world-model heads

For a posterior state and candidate action block, predict

\[
\tilde z_{t+c}^{\ell,i}
=
T_\theta(z_t^\ell,E_a(u_{t,0:c}^{i})),
\]

\[
D_\theta(\tilde z_{t+c}^{\ell,i})
=
\left(
\widehat{\Delta W}^{\,i},
\widehat{\Delta Y}^{\,\ell,i},
\hat r_t^{\,\ell,i},
\hat p_t^{\,\ell,i},
\hat V^{\,\ell}(\tilde z_{t+c}^{\ell,i})
\right).
\]

### 4.1 Physical effects

\(\Delta W\) contains instruction-independent quantities available in the
current domain:

- proprioceptive change;
- task-object displacement and contact;
- selected fixed physical predicates;
- optional object/fixture pose targets as training-only privileged supervision.

The same physical `(history, action, outcome)` relabeled under another
instruction must preserve \(\Delta W\).

### 4.2 Task-conditioned effects and reward

\(\Delta Y^\ell\), \(r^\ell\), and \(p^\ell\) represent:

- task predicate flips;
- phase-aware progress;
- one-block task reward;
- termination/success probability when that label is actually observed.

The existing `ret` output must not silently stand for all of reward, value, and
success. These targets receive separate semantics and losses.

### 4.3 Continuation value

The initial value target is explicitly

\[
V^{\pi_0}_\ell(z),
\]

estimated from ordinary stock-policy trajectories or valid recorded
continuations. This avoids pretending that a classifier trained on branch
outcomes is a policy-dependent value function.

After the LC-Flow policy changes materially, either:

1. collect one bounded rollout refresh and update \(V^{\pi}\); or
2. keep the first policy-improvement step explicitly anchored to
   \(V^{\pi_0}\).

`[DEC]` Choose Monte Carlo, TD(\(\lambda\)), or FQE-style value targets after
auditing the available continuation lengths. The value target must always record
the generating policy.

### 4.4 Absolute and contrastive grounding

Snapshot siblings provide unusually clean action contrasts. Retain the
effect-centered loss

\[
\widetilde{\Delta y}_i
=
\Delta y_i
-
\frac{1}{M}\sum_{j=1}^{M}\Delta y_j,
\]

but do not use it alone. Centering removes common-mode state and therefore
cannot anchor recursive rollout. Absolute sequential targets,
\(\mathcal L_{\mathrm{self}}\), reward/value prediction, and observed
continuations provide that anchor.

---

## 5. From world model to VLA policy improvement

The main methodological addition over v0.3 is an explicit model-generated policy
learning signal.

### 5.1 Candidate generation inside VLA support

At a recorded or rollout state, sample candidate chunks from the current VLA or
the frozen π0.5 reference:

\[
u_t^i\sim \pi_{\mathrm{cand}}(\cdot\mid H_t,\ell),
\qquad i=1,\ldots,N.
\]

Candidates remain within the VLA behavioral support; the method does not search
arbitrary 500-dimensional action chunks.

### 5.2 Model score and conservative advantage

For the executed first block of each candidate:

\[
\hat Q_\ell(z_t,u_t^i)
=
\hat r_\ell(z_t,u_t^i)
+
\gamma^c
\hat V_\ell
\left(
T_\theta(z_t,E_a(u_{t,0:c}^i))
\right).
\]

The initial advantage is

\[
\hat A_\ell(z_t,u_t^i)
=
\hat Q_\ell(z_t,u_t^i)-\hat V_\ell(z_t).
\]

`[DEC]` Compare this state-value baseline against a within-candidate-set baseline.
Use the choice with better held-out branch ranking calibration.

If an uncertainty estimate is available, use a lower-confidence score:

\[
\hat Q_{\mathrm{LCB}}
=
\hat Q-\kappa\hat\sigma_Q.
\]

The first implementation may use a small model ensemble or checkpoint
disagreement. Uncertainty is a support guard, not a separate research
contribution.

### 5.3 Model-guided flow distillation

Convert conservative positive advantages into nonnegative, clipped weights:

\[
w_i^{\mathrm{WM}}
=
\mathbf 1[
\hat A_i > m
]
\operatorname{clip}
\left(
\exp(\hat A_i/\beta),
0,
w_{\max}
\right).
\]

Train the first ten actions of the selected proposal with masked flow matching:

\[
\mathcal L_{\mathrm{WM\text{-}FM}}
=
\sum_i
w_i^{\mathrm{WM}}
\left\|
M_{0:c}\odot
\left[
v_\psi(x_\tau,\tau,h_t^\ell,z_t^\ell)
-
(\epsilon-u_t^i)
\right]
\right\|^2.
\]

Only the first \(c=10\) actions receive model-derived credit because the policy
replans afterward. Full demonstration chunks retain the ordinary unmasked π0.5
flow-matching loss.

### 5.4 Role of real branches

Existing positive-branch flow matching is retained as:

- a ground-truth branch teacher;
- a calibration set for predicted advantage;
- a control that measures how much improvement comes from directly imitating
  better observed actions.

It is no longer the only path from the world model to the policy. The decisive
comparison is:

| Arm | Interpretation |
|---|---|
| branch-FM only | improvement from observed better-action imitation |
| LC-Flow, no world loss | recurrence/capacity and shared-data control |
| world model + GT branch-FM | grounded model plus observed policy targets |
| world model + model-guided FM | full method; model creates additional policy targets |

---

## 6. Data roles

### 6.1 Ordinary demonstrations and sequential rollouts

These are the primary data for:

- posterior-state construction;
- one-step and short multi-step latent prediction;
- absolute physical/task outcome prediction;
- task reward and \(V^{\pi_0}\);
- full-chunk π0.5 rehearsal and retention.

Successful demonstrations do not identify all alternative-action effects, but
they are sufficient to learn broad nominal recursive dynamics. They must not be
reduced to rehearsal-only data.

### 6.2 Existing snapshot branches

These are used for:

- same-history alternative-action effects;
- sibling-centered effect loss;
- candidate ranking and top-1 regret;
- calibration of model-generated advantages;
- GT-branch policy-teacher controls.

Snapshot branching is a targeted coverage mechanism, not a required generative
access assumption for the entire framework.

### 6.3 Language reuse

One stored physical continuation may be evaluated under:

- paraphrases of the same goal;
- compatible alternative goals;
- the full composite instruction.

Paraphrases supervise outcome/action equivalence. Compatible goals supervise
shared physics with different reward/progress readouts. This reuse improves data
efficiency but is not the principal novelty.

### 6.4 New interaction

Do not begin a generic `128–256 branch` expansion.

New branches are collected only after the existing-data vertical slice reveals a
specific coverage failure:

- high uncertainty or ranking error at late-chain states;
- correct nominal self-prediction but incorrect alternative-action effects;
- a missing positive action inside the sampled VLA candidate pool;
- policy-induced states absent from the original stock rollouts.

The first targeted expansion should use approximately `16–32` independent
late-chain source episodes, then stop and re-evaluate.

---

## 7. Joint objectives

The world-model objective is

\[
\begin{aligned}
\mathcal L_{\mathrm{WM}}
=\;&
\lambda_{\mathrm{self}}\mathcal L_{\mathrm{self}}^{(1)}
+
\lambda_{\mathrm{roll}}\mathcal L_{\mathrm{roll}}
+
\lambda_{\mathrm{phys}}\mathcal L_{\mathrm{phys}}
+
\lambda_{\mathrm{ctr}}\mathcal L_{\mathrm{contrast}}
\\
&+
\lambda_{\mathrm{sem}}\mathcal L_{\mathrm{sem}}
+
\lambda_r\mathcal L_r
+
\lambda_V\mathcal L_V
+
\lambda_{\mathrm{shared}}\mathcal L_{\mathrm{shared}}
+
\lambda_{\mathrm{para}}\mathcal L_{\mathrm{para}}.
\end{aligned}
\]

The policy objective is

\[
\mathcal L_{\mathrm{policy}}
=
\lambda_{\mathrm{demo}}\mathcal L_{\mathrm{demo\text{-}FM}}
+
\lambda_{\mathrm{GT}}\mathcal L_{\mathrm{GT\text{-}branch\text{-}FM}}
+
\lambda_{\mathrm{model}}\mathcal L_{\mathrm{WM\text{-}FM}}.
\]

The complete objective is

\[
\mathcal L
=
\mathcal L_{\mathrm{WM}}
+
\mathcal L_{\mathrm{policy}}.
\]

Recommended optimization order:

1. train the recursive state and grounded heads with the policy adapter at its
   stock no-op;
2. train the state adapter with demonstration rehearsal and GT-branch controls;
3. freeze or slow the target world model while generating model-guided flow
   targets;
4. jointly fine-tune only after the vertical slice is stable.

This staged order avoids moving the world-model target, policy distribution, and
value target simultaneously in the first experiment.

---

## 8. Evaluation and attribution

### 8.1 World-model evidence

Report from held-out source episodes:

- one-step latent prediction against the EMA posterior target;
- two- and three-block open-loop latent prediction;
- comparison with no-action, copy-state, and mean-next-state baselines;
- task-object physical-effect error on effect-resolved branches;
- predicate-flip and phase-aware progress metrics;
- reward/value calibration;
- candidate pairwise accuracy, top-1 regret, and ranking correlation;
- paraphrase equivalence;
- cross-instruction shared-physics consistency.

Latent distance alone is never a promotion metric.

### 8.2 Policy evidence

Primary:

- chain3/4/5 full-composite-prompt `N = 1` success and Q;
- per-atom completion times and failure localization;
- standard LIBERO-10 retention.

Secondary:

- oracle best-of-N headroom;
- model reranking;
- improved-policy rollout distribution and one bounded refresh.

### 8.3 Required attribution controls

| Control | Question |
|---|---|
| stock π0.5 | original VLA baseline |
| branch-FM without persistent state | are observed better actions sufficient? |
| LC-Flow without world losses | is recurrence/extra capacity sufficient? |
| LC-Flow without self-prediction | are effect heads alone sufficient? |
| GT-branch FM only | does policy gain require model-generated targets? |
| language-free/readout-only state | must task information enter the predictive state? |
| full LC-Flow v0.4 | world-model-guided VLA learning |

The final claim requires all three:

1. the VLA improves;
2. the learned state predicts controlled action consequences;
3. matched no-world-model controls do not explain the gain.

---

## 9. Promotion logic

### Gate A — recursive prediction

Proceed when one-step prediction beats trivial baselines on held-out sequential
data and does not collapse. Add \(k=2,3\) only after the one-step target is
stable.

If Gate A fails:

- first inspect target construction, EMA update, and sequence alignment;
- then compare a smaller learned target projection;
- do not collect more branches unless the failure is localized to action
  coverage.

### Gate B — decision relevance

Proceed to model-guided distillation when the same checkpoint predicts grounded
effects and ranks held-out branch candidates better than progress-only and
action-marginal baselines.

If self-prediction succeeds but ranking fails:

- diagnose reward/value semantics and branch calibration;
- distinguish missing value supervision from missing action-effect coverage.

### Gate C — policy influence

Proceed to multi-seed chain evaluation when model-guided FM changes the first-ten
action distribution and improves paired development behavior without material
LIBERO-10 regression.

If the policy barely moves:

- measure adapter-conditioned action shift;
- only then escalate from the zero-init adapter to action-expert PEFT.

If the policy moves but behavior does not improve:

- investigate model score calibration, candidate support, and late-chain
  coverage rather than adding policy capacity.

### Gate D — targeted interaction

Collect new late-chain branches only if a Gate-B or Gate-C failure is localized
to states/actions not represented in the current data. Use uncertainty and
ranking failures to select the source states.

---

## 10. Current decisions

### Locked

- retain LCState, `E_a`, `T`, `U`, and zero-init AdaRMS injection;
- predict next latent state rather than reconstruct pixels;
- retain reward/progress and add a separately defined value head;
- train the final VLA policy on the learned internal state;
- use ordinary trajectories for recursive dynamics;
- use branches for alternative-action calibration and targeted coverage;
- deploy a full-prompt `N = 1` VLA.

### Open

- exact EMA target projection and latent distance;
- \(k=2,3\) rollout weighting;
- \(V^{\pi_0}\) target estimator and refresh strategy;
- model-advantage baseline, temperature, margin, clipping, and uncertainty
  penalty;
- candidate count during model-guided distillation;
- whether the AdaRMS adapter alone has enough policy control;
- whether explicit world/task factorization is needed;
- final evaluation seed count and retention tolerance.

### Retired as immediate priorities

- further q-channel-only diagnostics;
- treating one-step centered effect regression as the complete world model;
- collecting large crossed branch datasets before a method vertical slice;
- making frozen-policy reranking the main system;
- treating crossed-language data as the paper's central contribution;
- requiring perfect physical-state reconstruction before policy learning.

---

## 11. Immediate implementation sequence

1. Expose ordinary trajectory windows and consume
   `next_prefix_hidden_full`.
2. Add the EMA posterior target and one-step latent self-prediction.
3. Mix sequential trajectories and existing branch groups; retain both
   absolute and sibling-centered outcomes.
4. separate reward, progress, terminal success, and \(V^{\pi_0}\) semantics.
5. Train and evaluate the world-model vertical slice using existing data.
6. Add two- and three-block rollout after one-step prediction is validated.
7. Generate VLA-supported candidates and implement conservative
   model-guided masked flow matching.
8. Compare GT-branch teacher, no-world-loss, and full model-guided variants.
9. Run paired chain3 `N = 1` readouts and LIBERO-10 retention from each
   promising checkpoint.
10. Collect `16–32` targeted late-chain branch sources only if the measured
    failure satisfies Gate D.

This sequence supersedes the v0.3 immediate order centered on collecting
`128–256` branches and directly training branch-weighted flow matching.

---

## 12. Scope and current interpretation

The current chain3/4/5 suite is a self-built LoHo-inspired domain, not the
official LIBERO-LoHo benchmark. Early results are method-development evidence.

The current v3 checkpoint has shown partial held-out action-object prediction but
has not shown recursive latent prediction, model-generated policy improvement,
or better final Q/SR. It should therefore be described as an effect-conditioned
policy adapter checkpoint, not yet as evidence for a decision-sufficient VLA
world model.

LC-Flow v0.4 becomes the intended framework only when the following causal chain
is instantiated:

```text
ordinary trajectories + targeted branches
                    ↓
recursive language-conditioned predictive state
                    ↓
grounded reward/value and alternative-action prediction
                    ↓
world-model-generated policy targets
                    ↓
distillation into the pretrained VLA
                    ↓
better full-prompt N=1 long-horizon behavior
```