# pi05-lcwm — Framework Design v0.6

Working design record, updated 2026-07-30.

H7/v0.5 is complete and remains a negative implementation baseline. It did
not instantiate the predictive-state contract in Sections 2–5: candidate
actions transitioned a physical prior while current task features bypassed
the transition into the outcome heads. v0.6 therefore keeps the research
identity but replaces the implementation with one language-conditioned
recurrent predictive state used by both the world model and π0.5.

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

`[LOCKED FOR v0.6]`

- reuse the same real-prompt prefix forward and KV cache used by π0.5;
- keep `T` free of a second language/task-ID input;
- let language enter the transition through the complete
  \(z_t^\ell\);
- use an observation-anchored update, with a fixed residual bound if needed,
  so recurrence cannot freely drift;
- carry the real visual-action history in the main method;
- use reset-at-each-decision only as a matched control.

For every policy candidate, transition this complete task-conditioned state
before any outcome is decoded:

\[
\widetilde z_{t+c}^{\ell,i}
=T_\theta\!\left(z_t^\ell,E_a(u_i)\right).
\]

Physical, predicate, reward, progress, success, value, and ranking heads may
consume only \(\widetilde z_{t+c}^{\ell,i}\). They may not also consume the
current prefix, a current physical/task anchor, the true next observation,
raw language, task ID, phase ID, or a direct action bypass. The next
observation enters only afterward through
\(z_{t+c}^\ell=U(\widetilde z_{t+c}^{\ell,\mathrm{executed}},
h_{t+c}^\ell)\).

The H7 implementation is retained as a historical readout-conditioned
baseline. It is not an implementation of this locked graph.

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

`[LOCKED FOR FIRST v0.6 READOUT]` Keep the action expert frozen and train one
zero-initialized state-to-flow projection with demonstration rehearsal. H7
already showed that this interface class can produce substantial first-ten
action shifts, so action-expert capacity is not the identified repair.
Escalate to matched PEFT only if a grounded-GT teacher fails and the saved
v0.6 action shift is small.

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

### 5.2 Model score and conservative sibling preference

For every candidate, decode the registered task-automaton outcome tuple from
the transitioned state. It contains terminal success, irreversible damage,
current-valid ordered prefix, multi-horizon valid Q, and time to the next
milestone. Train one scalar utility from each predicted state and pair it
through a Bradley–Terry comparator on common-noise sibling outcomes:

\[
s_i=s_\theta(\widetilde z_{t+c}^{\ell,i}),
\qquad
p_{ij}=\sigma(s_i-s_j).
\]

Replay-noise-indistinguishable pairs are ties. Calibrate one source-disjoint
error margin \(\delta\) from grounded dev siblings. A model proposal is
eligible only when \(p_{i0}-\delta>0.5\) against the designated exchangeable
reference candidate. Immediate geometry, absolute snapshot `dQ`, or a tiny
top-1 score difference is not a candidate advantage.

With \(N=4\) exchangeable candidates, an uninformative scorer selects a
non-stock argmax with probability \(3/4\). Non-stock fraction is therefore
never evidence of learned selection.

### 5.3 Model-guided flow distillation

Select the highest-preference eligible proposal and give it one registered,
uniform generated-teacher weight:

\[
i^\star=\arg\max_i s_i,
\qquad
w_i^{\mathrm{WM}}
=
\mathbf 1[i=i^\star]\,
\mathbf 1[p_{i^\star 0}-\delta>0.5].
\]

If no proposal clears the calibrated margin, the state receives only
grounded/demonstration rehearsal. The matched-random control reuses the exact
state mask, total weight, candidate pool, optimizer update, and flow
noise/time, and changes only the candidate assignment.

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

The H7 bank is not primary candidate-ranking supervision. Although it contains
160 branches, all 40 groups tied on immediate subgoal reward, only `3/20`
train and `4/20` dev groups varied in continuation Q, the continuations were
not common-noise paired, and terminal-success support was zero. It is used
only for:

- ordinary/null-effect calibration and rehearsal;
- physical effect loss for explicitly effect-resolved groups;
- ranking loss only for explicitly variance-bearing groups whose target is
  still valid under the v0.6 label contract.

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

The H7 result has already identified the coverage failure. v0.6 therefore
collects a fixed task-balanced batch from contact, commitment, recovery,
placement, and final-two-unresolved states. State selection is semantic and
physical, never elapsed-time percentile.

Every sibling group includes repeated-action replay-noise estimation and
common-random-number continuations. Outcomes are exposed at multiple horizons,
and the labels distinguish current predicate truth, milestones, invalidation,
damage, and recovery. Crossed language variants are materialized in the
training index with prompt-conditioned features recomputed from raw
observations.

Collection runs as two bounded tranches and overlaps model implementation. It
is training-data construction, not a diagnostic gate; the complete budgets,
candidate provenance, paired-continuation rule, and artifacts are locked in
V6.2 of `2026-07-30.md`.

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
\lambda_{\mathrm{auto}}\mathcal L_{\mathrm{automaton}}
+
\lambda_{\mathrm{rank}}\mathcal L_{\mathrm{paired\text{-}rank}}
+
\lambda_{\mathrm{damage}}\mathcal L_{\mathrm{damage}}
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
3. freeze the scoring world model while generating model-guided flow targets
   and training the first v0.6 policy checkpoints;
4. consider joint fine-tuning only after a fixed behavior readout localizes a
   need for it.

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
| one GT-branch-FM checkpoint in current/reset mode | can observed useful actions improve the deployed flow policy? |
| the same GT checkpoint with recurrent carry | does carried state help under identical weights? |
| recurrent-state + matched-random generated targets | does score-blind post-training explain the model arm? |
| full recurrent LC-Flow v0.6 | do predictive state and WM-generated targets improve the VLA? |
| task-agnostic/readout-only state, post-win | must task information enter the predictive state? |
| recurrent state without world losses, post-win | does recurrence/extra capacity explain a win? |
| reset-history full method, post-win | is carried visual-action history needed? |

The final claim requires all three:

1. the VLA improves;
2. the learned state predicts controlled action consequences;
3. matched no-world-model controls do not explain the gain.

---

## 9. Promotion logic

Offline losses, rank statistics, and latent diagnostics do not decide whether
v0.6 reaches policy training. Mechanical contract checks prevent invalid jobs;
the first scientific route selector is the fixed public-LoHo `N=1` behavior
matrix:

| arm | purpose |
|---|---|
| stock | original π0.5 baseline |
| one grounded-GT checkpoint in current/reset mode | can executed useful actions train the deployed flow interface? |
| the same grounded-GT checkpoint with recurrent carry | does carried predictive state help under identical weights? |
| recurrent-state + grounded GT + matched random | score-blind generated-target control |
| recurrent-state + grounded GT + WM targets | full v0.6 method |

The routing is:

- grounded GT improves but WM does not beat matched random → repair candidate
  preference, return calibration, or crossed supervision;
- grounded GT beats the full WM arm → the observed branch targets produced
  the gain and model-generated targets reduced it; do not promote a
  world-model mechanism;
- the same GT checkpoint with carry beats its reset deployment → carried
  visual-action state contributes under grounded supervision;
- WM ≈ random and both improve → generic supported-action post-training, not
  learned world-model selection;
- grounded GT fails with small action shift → widen policy coupling with
  matched PEFT;
- grounded GT fails with large action shift → repair targets, rehearsal, or
  the flow objective rather than policy capacity;
- a support candidate helps but no full-prompt candidate does → collect one
  task/subgoal-local proposal-support tranche;
- nonzero recurrence becomes unstable → repair recurrent optimization rather
  than concluding that history is harmful.

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
- use one current-observation-anchored, language-conditioned recurrent state;
- transition the complete state under every candidate before decoding any
  candidate-dependent output;
- prohibit current-feature, true-next-observation, raw-language/task-ID, and
  direct-action bypasses into candidate heads;
- keep physical outputs invariant across compatible language relabels while
  task/progress outputs remain goal-dependent;
- train same-goal paraphrases to agree and scene-valid distinct goals to carry
  task-relative outcome labels;
- unroll recurrent state from episode reset for every teacher state and every
  crossed-language history; reset-at-snapshot scoring is forbidden;
- attach continuation value to a GoalSpec only when the continuation policy
  actually ran under that goal's canonical prompt;
- use replay-noise estimates and common-noise paired continuations;
- use task-automaton milestones, current-valid predicates, invalidation,
  damage, success, and multi-horizon progress; never immediate distance as
  advantage;
- train pairwise sibling preference rather than snapshot-group means;
- inject the same recurrent state into π0.5 through one zero-initialized
  projection without a learned scalar gate;
- freeze the world model during the first policy readout and train only the
  registered state-to-flow projection;
- use ordinary trajectories for recursive dynamics;
- use effect-resolved branches for alternative-action learning and grounded
  policy controls;
- run policy training after mechanical correctness checks without an offline
  performance gate;
- treat public H-WM LIBERO-LoHo Task1–5 as the primary long-horizon endpoint;
- deploy a full-prompt `N = 1` VLA.

### Open

- whether the faithful recurrent state improves public behavior;
- whether WM-selected training beats matched random selection;
- whether grounded executed targets improve the current-state policy;
- longer recursive rollout weighting after the first public policy readout;
- the exact no-world-loss and task-agnostic/readout-only post-win ablations.

### Resolved by H7

- v0.5 does not improve stock on the public development panel;
- the v0.5 candidate scorer is a readout-conditioned baseline, not the
  intended LC predictive-state transition;
- 100-action unpaired stock continuations are poor primary sibling-ranking
  targets for the current data;
- the existing policy adapter can move first-ten actions substantially, so
  additional action-expert PEFT is not the first repair;
- the H7 recurrent gate was behaviorally inactive, so H7 does not tell us
  whether useful carried state helps.

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

H0–H7 are complete. The only active executable sequence is v0.6:

1. freeze the H7 negative baseline and its implementation audit;
2. implement one faithful
   \(z_t^\ell\rightarrow T(z_t^\ell,a_i)\rightarrow\widehat y_i^\ell\)
   candidate path plus mechanical leakage/gradient/parity tests;
3. collect two bounded tranches of task-balanced contact/late/recovery
   branches with replay-noise repeats, common-noise continuations, task
   automata, and real crossed-language labels;
4. train one fixed seed-0 recurrent world model with latent closure, grounded
   outcome heads, and pairwise sibling preference;
5. freeze approximately 200 full-prompt `N=4` teacher states and generate
   calibrated WM and exactly matched random targets;
6. train GT-only, recurrent-random, and recurrent-WM π0.5 flow interfaces
   from the same stock initialization; evaluate the GT-only checkpoint in
   current/reset and recurrent-carry modes with identical weights;
7. run the 125-episode public-LoHo development matrix immediately;
8. either freeze a winner for sealed confirmation and LIBERO-10 retention or
   make one repair localized by the behavior contrasts.

The exact source budget, candidate provenance, automaton target, common-noise
contract, file surfaces, policy matching rules, and failure routing are in
`2026-07-30.md`. Earlier H7 checklists are historical and must not be used as
implementation instructions.

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

H7 completed the first public-task data/model/policy loop. Stock reached
`SR=.040/Q=.516`; no v0.5 arm improved it. The stable reset WM/random contrast
was approximately tied, and the recurrent-WM arm collapsed. This establishes
that the H7 teacher/readout/adapter contract is not useful.

It does not establish that language-conditioned predictive dynamics or
visual-action history are unnecessary. The H7 branch bank was mostly tied,
continuation randomness was not paired, candidate actions did not transition
the complete task-conditioned state, crossed distinct-goal labels did not
enter training, and recurrence remained gated out of the policy. v0.6 is the
first implementation required to instantiate those parts together and carry
them through a policy behavior readout.

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

## 13. H7 empirical correction and v0.6 binding

H7 completed the first direct public-LoHo model, π0.5 flow-interface, and
closed-loop matrix. None of its four v0.5 arms improved stock. That result is
retained, but the implementation audit changes its scope.

### 13.1 H7 is a readout-conditioned baseline

The intended process is:

\[
z_t^\ell=\phi(H_t,\ell),
\qquad
\widetilde z_{t+1}^{\ell,i}=T(z_t^\ell,a_i),
\qquad
\hat y_{t+1}^{\ell,i}=R(\widetilde z_{t+1}^{\ell,i}).
\]

H7 instead implemented:

\[
w_{t+1}^{(i)}=T_w(w_t,a_i),
\qquad
\hat y_i=R(c_t^\ell,g_t^\ell,w_{t+1}^{(i)}).
\]

The task-conditioned part was not itself advanced under the candidate action.
This can predict language-conditioned outcomes through a readout, but it does
not test whether a task-conditioned predictive state improves long-horizon
decisions.

### 13.2 v0.6 correction

v0.6 removes the separate current `c/g` candidate readout and binds every
candidate head to one transitioned \(z^\ell\). The same \(z^\ell\) enters the
π0.5 flow interface through one zero-initialized projection. Crossed language
labels, task automata, paired continuations, one-/multi-block latent closure,
physical/task grounding, and sibling preference are all trained in the same
fixed run.

The data batch targets contact, object motion, recovery, placement, and final
unresolved subgoals. Ranking support is a within-snapshot paired contrast
beyond replay noise; absolute `dQ>0` and a 75%-expected non-stock argmax are
not evidence.

### 13.3 Execution ownership

The detailed v0.6 contract in `2026-07-30.md` is the only active action queue.
This design document owns the architectural invariants; the daily file owns
budgets, artifacts, job order, and routing. H7 documents remain immutable
evidence. Another broad diagnostic phase or a fixed 360-sample prerequisite
is not part of the mainline.
