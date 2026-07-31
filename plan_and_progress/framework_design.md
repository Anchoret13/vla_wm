# pi05-lcwm — Framework Design v0.4

Working design record, updated 2026-07-30.

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

`[SUPERSEDED FOR v0.5]`

- reuse the same real-prompt prefix forward and KV cache used by π0.5;
- keep `T` free of a second explicit language input;
- let language enter the transition through \(z_t^\ell\);
- reset the recurrent state only at environment reset;
- retain the current `E_a`, `T`, `U`, and LC-state token interface.

H4 preserves the first three principles but invalidates free accumulated
recurrence as the next implementation. v0.5 replaces the single free state
with physical/task anchors corrected by a bounded recurrent residual. Its
matched `reset` mode runs the same modules while nulling recurrent carry at
every decision; `recurrent` carries the real prior state and action. The exact
v0.5 equations are registered in H7.3 of `2026-07-29.md`.

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

- public H-WM LIBERO-LoHo Task1–5 full-composite-prompt `N = 1` success and
  subgoal Q-score;
- per-atom completion times and failure localization;
- standard LIBERO-10 retention.

Secondary:

- local Chain3/4/5 curriculum success and final-goal-atom Q;
- oracle best-of-N headroom;
- model reranking;
- improved-policy rollout distribution and one bounded refresh.

### 8.3 Required attribution controls

| Control | Question |
|---|---|
| stock π0.5 | original VLA baseline |
| reset-state + random teacher | does score-blind policy training explain the reset arm? |
| recurrent-state + random teacher | does score-blind policy training explain the recurrent arm? |
| reset-state + WM teacher | does WM selection help without carried history? |
| recurrent state without world losses | does recurrence/extra capacity alone explain a win? |
| LC-Flow without self-prediction | are effect heads alone sufficient? |
| GT-branch FM only | does policy gain require model-generated targets? |
| language-free/readout-only state | must task information enter the predictive state? |
| full recurrent LC-Flow v0.5 | do predictive state and WM-generated targets improve the VLA? |

The final claim requires all three:

1. the VLA improves;
2. the learned state predicts controlled action consequences;
3. matched no-world-model controls do not explain the gain.

---

## 9. Promotion logic

The earlier Gate A/B/C order incorrectly let offline model diagnostics decide
whether the project could attempt its main policy hypothesis. For v0.5,
mechanical launch checks prevent invalid jobs, but grounded losses and latent
metrics do not gate policy finetuning.

The first route selector is the complete public-LoHo development factorial:

- positive WM-selection effects in both state modes localize useful learned
  candidate scoring;
- positive recurrent-minus-reset effect under the same WM teacher localizes
  useful carried visual-action history;
- WM ≈ random localizes the failure to scoring/continuation supervision;
- recurrent ≈ reset localizes it to state carry or policy coupling;
- useful candidates with negligible first-ten action shift trigger matched
  action-expert PEFT;
- missing useful candidates for one unresolved public subgoal trigger one
  bounded task-local support refresh.

A development win freezes the method and opens a sealed public confirmation;
it is not itself a promotion claim. Full LIBERO-10 retention follows a sealed
confirmation win. Language-free/readout-only and no-world-loss controls are
required before the strongest causal mechanism claim, but they are post-result
ablations rather than serial preconditions for the first training run.

---

## 10. Current decisions

### Locked

- retain π0.5 as initialization, representation source, flow-matching action
  network, and final deployed policy;
- use the real composite prompt; no oracle task-ID or active-atom bypass;
- use a current-observation-anchored, action-conditioned predictive state;
- compare reset and recurrent carry with identical modules and parameters;
- keep physical prediction language-independent and task/progress state
  language-conditioned;
- ground candidate prediction in public subgoals, damage, success, and the
  policy-provenance-preserving finite-horizon \(G^{\pi_0}_{100}\);
- keep next-latent prediction auxiliary rather than using it as a policy gate;
- train the final VLA policy on the learned internal state;
- use ordinary trajectories for recursive dynamics;
- use branches for alternative-action calibration and targeted coverage;
- use the matched
  `{reset, recurrent} × {WM-selected, random-selected}` attribution panel;
- treat public H-WM LIBERO-LoHo Task1–5 as the primary long-horizon endpoint;
- deploy a full-prompt `N = 1` VLA.

### Open

- whether reset or recurrent state improves public behavior;
- whether WM-selected training beats matched random selection;
- whether the AdaRMS adapter alone has enough policy control;
- whether action-expert PEFT is needed after measured action shift;
- longer recursive rollout weighting after the first public policy readout;
- the exact no-world-loss and task-agnostic/readout-only post-win ablations.

### Retired as immediate priorities

- further q-channel-only diagnostics;
- treating one-step centered effect regression as the complete world model;
- requiring the 360-continuation H6.1 bank before model/policy training;
- using effective rank, EMA variance, or mean-next comparison as a launch gate;
- collecting large crossed branch datasets before a method vertical slice;
- making frozen-policy reranking the main system;
- treating crossed-language data as the paper's central contribution;
- requiring perfect physical-state reconstruction before policy learning.

---

## 11. Immediate implementation sequence

H0–H5 have completed the original v0.4 vertical slice and attribution panel.
The active queue is now a direct public-LoHo training experiment:

1. Implement and source-manifest the public H-WM LIBERO-LoHo Task1–5
   protocol; run one valid stock episode per task.
2. Collect the fixed public-task batch: 20 source episodes, 40 snapshot
   groups, 160 executed sibling branches, and one policy-provenance-preserving
   100-action continuation per branch.
3. Train one observation-anchored language-conditioned predictive state.
   `reset` and `recurrent` modes use identical modules; only recurrent carry
   differs.
4. Ground candidate prediction in physical effects, public subgoal bits,
   damage, success, and finite-horizon
   \(G^{\pi_0}_{100}=Q_{t+100}+\mathbb 1[\mathrm{success}]\). Keep latent
   self-prediction auxiliary.
5. Freeze approximately 200 source/phase-balanced train decisions and their
   deterministic `N=4` full-prompt candidate pools as the policy-teacher
   manifest.
6. From the same stock π0.5 initialization, flow-finetune the full
   `state interface × teacher` factorial:
   `reset_wm`, `reset_random`, `recurrent_wm`, and `recurrent_random`.
7. Run stock plus the two reset arms first for a 75-episode public readout;
   complete the pre-registered recurrent arms regardless of that readout,
   giving a 125-episode primary factorial.
8. Run frozen v0.4 `cur_wm` as a separate 25-episode legacy reference. It is
   not a matched control.
9. Promote only through sealed public-LoHo confirmation, then run LIBERO-10
   retention. A winning method still receives task-agnostic/readout-only and
   no-world-loss ablations before the strongest causal claim.
10. Use the public behavior contrasts to select at most one localized repair:
    scoring, recurrent-state supervision, action-expert PEFT, or task-local
    support.

The exact budgets, artifacts, matching contract, and failure routing are
registered in the H7 queue of `2026-07-29.md`. The former 360-continuation
diagnostic bank is not a gate before this training/finetuning run.

---

## 12. Scope and current interpretation

The target long-horizon benchmark is the publicly specified LIBERO-LoHo
variant introduced by H-WM (arXiv:2602.11291), not a custom benchmark created
by this project. It defines five separate 5–7-step tasks.

The current local chain3/4/5 files are a development curriculum derived from
that construction, not a subset of the five published tasks: the local nested
basket object sets do not exactly match any public task. Their results remain
useful method-development evidence, but primary LIBERO-LoHo performance
requires the exact public five-task protocol or an explicitly labeled,
source-manifested specification-based reimplementation.

The persistent-recurrent v3/v0.4 path has shown partial held-out
action-object prediction but has not shown recursive latent prediction or a
promotable Chain3–5 policy improvement. The H4 factorial nevertheless provides
the first positive policy-level development result: recurrent-WM-selected
targets distilled through a current-only policy state (`cur_wm`) complete
Chain4 in 2/5 episodes, versus 0/5 for each matched arm, while every arm
remains 0/5 on Chain5. This is a promising selector/interface result, not yet
evidence for decision-sufficient language-conditioned dynamics: the teacher
still used recurrent WM scores, the gain is development-only, and the same
model regresses on Chain5. Preserve `cur_wm` as the behavior reference and
start the public-LoHo v0.5 model-training and π0.5-finetuning mainline now.
The old same-state grounded counterfactual bank is no longer a serial route
selector; any later diagnostic must repair a failure localized by the public
policy result.

The v0.5 attempt replaces free recurrent drift with a current-observation-
anchored, bounded recurrent residual. Its primary development experiment is
stock plus
`{reset, recurrent} × {WM-selected, random-selected}`. This is the minimum
complete panel that can separately measure learned selection and carried
visual-action history. Frozen v0.4 `cur_wm` remains a secondary legacy
reference, not a matched factorial arm.

LC-Flow becomes the intended framework only when the following causal chain is
instantiated:

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

---

## 13. H7 empirical correction and v0.5-r2 contract

H7 completed the first direct public-LoHo world-model training, π0.5
flow-finetuning, and closed-loop matrix. The behavior result is negative:
none of the four v0.5 arms improves stock. The result is retained, but the
implementation audit changes what it means.

### 13.1 The H7 r1 implementation is not the candidate defined here

The intended LC predictive process is:

\[
z_t^\ell=\phi_\ell(H_t),
\qquad
z_{t+1}^{\ell,i}=T_\ell(z_t^\ell,a_i),
\qquad
\hat y_{t+1}^{\ell,i}=R_\ell(z_{t+1}^{\ell,i}).
\]

H7 r1 instead transitions a language-independent physical token and combines
it with unchanged current task tokens inside the outcome head:

\[
w_{t+1}^{(i)}=T_w(w_t,a_i),
\qquad
\hat y_i=R(c_t^\ell,g_t^\ell,w_{t+1}^{(i)}).
\]

That implementation is a useful readout-style baseline, but it does not test
whether language-conditioned predictive dynamics produce a more useful task
state. In addition, the registered bounded observation-anchor, crossed
different-goal labels, next-posterior closure, multi-step rollout, damage and
success targets, and sibling-ranking loss did not enter the run. H7 therefore
cannot retire the framework's central candidate.

### 13.2 Data contract correction

The crossed dataset must identify controlled effects, not merely contain
multiple action tensors. For r2:

- sibling continuations use shared random streams or matched repeats;
- snapshot selection targets contact, object motion, recovery, and late
  unresolved subgoals rather than elapsed decisions with an unchanged
  subgoal index;
- task labels describe current-valid predicates, signed progress,
  invalidation/damage, success, and preservation at multiple horizons;
- a group is useful for ranking only when sibling outcomes differ beyond
  replay/continuation noise;
- canonical, paraphrase, compatible, and atomic instructions label the same
  physical branch, with at least some action×instruction interaction;
- these properties are measured and used for data selection, but no fixed
  dataset count becomes a serial gate before training.

The H7 statistic `dQ>0 relative to the snapshot` is retired as evidence of
action support. Ranking support is a within-snapshot contrast.

### 13.3 Locked r2 state and outcome interface

For candidate \(i\) and instruction \(\ell\):

\[
\begin{aligned}
w_{t+1}^{(i)}
  &=T_w(w_t,a_i),\\
g_{t+1}^{\ell,i}
  &=T_g(g_t^\ell,w_{t+1}^{(i)},a_i),\\
\hat W_{t+1}^{(i)}
  &=R_w(w_{t+1}^{(i)}),\\
\hat Y_{t+1}^{\ell,i}
  &=R_g(w_{t+1}^{(i)},g_{t+1}^{\ell,i}).
\end{aligned}
\]

The posterior update remains a zero-initialized bounded correction around
the real current-observation anchor. Physical readouts must agree across
instructions; task/progress readouts may differ across compatible goals and
must agree across same-goal paraphrases. Transitioned states are aligned with
re-encoded branch-after posteriors, and the same transition is rolled over
ordinary sequential windows.

A current-state policy path may remain as a safety anchor, but it cannot be
the only path receiving effective gradient. Policy coupling is reported by
the actual norm of each deployed residual and by same-checkpoint
reset-versus-carry action differences, not by scalar gate values alone.

### 13.4 Locked r2 policy-learning contract

- Use centered pairwise branch ranking and a calibrated nonzero margin.
- Treat ties as no preference; do not manufacture a teacher from max-of-noise.
- Train one grounded/data-only adapter from executed superior branches and
  one model-selected adapter from the same π0.5 initialization.
- Retain stock rehearsal and an explicit action/KL trust region.
- Pre-register checkpoint selection; an unstable final epoch is not deployed
  merely because the update budget ended there.
- Evaluate the same trained adapter in reset and carry modes for the clean
  history contrast. Independently optimized state-mode policies may be
  additional performance variants only.
- Return to public-LoHo N=1 SR/Q immediately after the first r2 policy jobs.

### 13.5 Current interpretation

The causal chain remains the project target:

```text
crossed trajectories + effectful late branches
                    ↓
language-conditioned rolled predictive state
                    ↓
task-relative controlled effects and preservation
                    ↓
calibrated model-generated π0.5 targets
                    ↓
better full-prompt N=1 public-LoHo behavior
```

H7 r1 failed at the data-identification, predictive-transition, and
policy-target stages before this chain was instantiated. It is the frozen
implementation-negative reference. H7-r2 is the active framework attempt;
another broad diagnostic phase or a 360-sample prerequisite is not.
