# pi05-lcwm — Framework Design v0.9

Working design record, updated 2026-08-14. v0.9 retains the central LC
predictive-state graph and deployment contract, and incorporates the completed
V7.4/V7.5 model-learning evidence:

1. center the recurrent history input as well as the policy readout, and keep
   the recurrent residual in a nonsaturated regime so history can receive
   gradient and reach the state;
2. require history dependence to remain near a demonstrated-live regime and
   require held-out action effects to beat copy/no-action controls; recurrence
   liveness or latent rollout separation alone is not predictive dynamics;
3. construct verified blocker-relevant positive/negative consequences across
   the public LoHo chain, using acquisition-only state reachers when stock π0.5
   cannot reach late-chain states;
4. match task/source/physical-anchor exposure across generated-target,
   grounded-only, and matched-random policy arms; language-query rotation over
   one physical state is not state diversity;
5. keep LCWM training, π0.5 flow fine-tuning, and recurrent public-LoHo
   evaluation as one bounded vertical slice whose Phase-1 endpoint is behavior.

H7/v0.5 remains a negative readout-conditioned implementation baseline. v0.6
then instantiated one language-conditioned recurrent predictive state used by
both candidate dynamics and π0.5, but its early data and teacher channels were
mostly unsupported. V7.2 completed a data -> LCWM -> policy -> public-LoHo
vertical slice. Its post-run audit found that the policy trainer reduced each
candidate separately and normalized by that candidate's scalar weight, exactly
cancelling the registered teacher weights and scaling soft FM with candidate
count. V7.2 behavior is therefore implementation-confounded, while its data,
model, teacher-assessment outcomes, videos, and failure localization remain
reusable. V7.3 then executed the complete repaired path and a fresh 125-rollout
matrix, but its deployed pooled state was nearly constant, its sole model target
had no verified outcome advantage, and its scheduler concentrated the model
term on that one anchor. V7.3 is therefore a real no-win behavior record but not
a clean negative for language-conditioned predictive state learning. V7.4 then
showed that centering restores policy-facing anchor identity but cannot recover
history absent from the recurrent state. V7.5 repaired the recurrent channel
and produced a stable live-history interval during training, but the final
checkpoint collapsed by more than two orders of magnitude from that interval
and failed to beat copy on held-out action effects. No V7.5 policy or behavior
stage has run; its current evidence is a model-learning result only.

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
AdaRMS adapter. V7.4A spent the registered fallback and committed
`TokenQueryPool`; v0.9 keeps that commitment rather than silently reopening
mean pooling. The current interface is

\[
q_t^\ell=Q_\omega(z_t^\ell),
\qquad
r_t^\ell=q_t^\ell-\mu_{Q,\mathrm{train}},
\qquad
c_t^\ell=\operatorname{Norm}_{\mathrm{bounded}}
\!\left(r_t^\ell;\sigma_{Q,\mathrm{train}}\right),
\qquad
c_{\mathrm{expert}}
=
c_{\mathrm{time}}
+
W_c c_t^\ell,
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

`[HISTORICAL v0.6 READOUT]` The first readout froze the complete action expert
and trained only the zero-initialized state-to-flow projection. V7.2's
weighted-FM implementation cancelled candidate weights and applied full-chunk
trust against first-ten corrections, so its terminal behavior cannot close the
interface question.

`[LOCKED FOR v0.9]` Freeze PrefixVLM, the LCWM, and all action-expert blocks;
train exactly three π0.5 boundary modules from matched initialization:

- the sha-seeded one-query `TokenQueryPool`;
- the zero-initialized, bias-free centered-state-to-AdaRMS projection;
- the stock-initialized `action_out_proj` weight and bias.

This is the narrow N1 boundary that transmitted the known V7.1 executable
correction. It remains the existing π0.5 generation flow, not a separate
controller. Every treatment and control uses the same trainable names,
optimizer, schedule, flow noise/time, demonstration rehearsal, and retention
loss. Two controls remain distinct:

- a **constant-token capacity control** feeds the same frozen, nonzero,
  training-statistic-matched token block to every example while keeping all
  three boundary modules trainable;
- a **zero-state grounded BC arm** forces the LC contribution to zero during
  training and tests direct correction imitation through `action_out_proj`;
- an **LC-off readout** forces the LC contribution to zero only when evaluating
  a trained state arm and measures what its action head retains without state.

Neither zero-input case is called matched-capacity because it gives the LC
projection and token query no state-dependent gradient.

The constant token block is constructed and frozen before policy targets are
read; it cannot encode task, history, source, or arm identity. Its exact
construction and scale are bound in the next daily registration, and a
two-update mechanical contract first verifies the exact step-0 stock no-op,
then verifies nonzero gradient reach to the query after the zero-initialized
projection has opened; all three modules must change under the constant-token
arm before full training.

The normalization statistics are training-only artifacts bound into every
checkpoint and eval manifest. Centering and scale control are optimization
devices, not evidence that the input contains useful state. The interface must
therefore report the raw centered residual before division, the normalized
input, the projected AdaRMS change, and the resulting flow/action change. A
small fitted scale may not promote a checkpoint by amplifying a nearly absent
raw contrast; `Norm_bounded` uses a preregistered scale floor/cap and cannot be
changed after inspecting the policy targets. Mean pooling remains a historical
V7.3/V7.4 diagnostic only; no Wz/LoRA/layer/rank interface sweep is reopened.

---

## 3. Recursive predictive world state

The one-block transition is retained, together with the V7.5 correction that
made its recurrent channel mechanically capable of carrying history. For one
observed update, the repaired core is

\[
\begin{aligned}
\bar z_t^\ell
&=T_\theta(z_{t-c}^\ell,E_a(u_{t-c,0:c})),\\
z_{t,\mathrm{hist}}^\ell
&=\frac{\bar z_t^\ell-\mu_{\mathrm{hist}}}
{\max(\sigma_{\mathrm{hist}},\epsilon_{\mathrm{hist}})},\\
c_t^\ell&=A_\theta(h_t^\ell),\\
z_t^\ell
&=\operatorname{LN}\!\left(
c_t^\ell+\beta\tanh\!\left[
\operatorname{LN}\!\left(
R_\theta(c_t^\ell,z_{t,\mathrm{hist}}^\ell)
\right)
\right]
\right).
\end{aligned}
\]

The history statistics are fitted only on the training state stream and frozen
with the checkpoint. The numerical floor `epsilon_hist` and a maximum allowed
normalized gain are registered before training; saturation, quantization SNR,
and raw pre-normalization contrasts are asserted separately. History centering
removes the shared common mode before cross-attention; the inner LayerNorm keeps
the recurrent residual out of the saturated `tanh` regime. V7.5 established
that this channel can be live, not that its training objective will keep it live
or make it action-predictive. Both properties must now be checkpoint-binding,
and normalization can never supply the evidence for raw history retention.

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

The positive rollout alone is not action-identifying: a model can use scene or
trajectory phase to approach the future posterior while ignoring which action
was supplied. Every rollout window therefore carries a matched negative made
from a policy-supported wrong action, a shuffled action order, or a same-state
sibling action:

\[
\mathcal L_{\mathrm{act\text{-}roll}}
=
\sum_j
\left[
m+d(\tilde z_{t+jc}^{\mathrm{pos}},z_{t+jc}^{+,\ell})
-d(\tilde z_{t+jc}^{\mathrm{neg}},z_{t+jc}^{+,\ell})
\right]_+.
\]

Negatives remain inside the action support of the same source/task cell; an
arbitrary out-of-distribution action cannot satisfy this contract. Separately
executed same-snapshot siblings provide the stronger physical/outcome target
whenever available. The frozen checkpoint rule is conjunctive: retain a
preregistered fraction of the demonstrated-live raw history regime and beat
copy/no-action/stock on a source-held-out action-effect or sibling test. Final
epoch, aggregate loss, task identity, or a centered distance amplified by a
small `sigma` cannot select the model.

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

V7.5 sharpens this statement: **recurrence liveness and predictive dynamics are
different axes**. A state may change with history yet still copy the current
state under every candidate action. Only controlled action-conditioned
consequences, followed by target generation and policy behavior, certify the
intended world model.

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

### 5.2 Outcome-derived conservative sibling preference

For every candidate, decode the registered task-outcome tuple directly from
the transitioned state:

\[
\hat y_i=
\left(
\hat d_i,\hat s_i,\hat m_i,\widehat{\Delta k}_i,
\hat q_i^{10:100},\hat\tau_i
\right),
\]

where the components are irreversible damage, terminal success,
next-milestone completion, ordered-prefix change, multi-horizon task-valid
progress, and time to the next milestone. Absolute targets and same-anchor
centered effects train the same heads. A standalone detached scalar rank head
does not decide v0.7 policy targets.

Freeze component-wise prediction margins on a source-disjoint
teacher-calibration bank before teacher assessment or policy training.
Candidate `i` is eligible only when its pessimistic outcome tuple
lexicographically dominates the optimistic `u0` tuple under:

```text
no additional damage
    → terminal success
    → next-milestone / ordered-prefix improvement
    → multi-horizon task-valid progress
    → shorter next-milestone time
```

Replay-noise-indistinguishable pairs are ties. Immediate geometry, absolute
snapshot `dQ`, a tiny top-1 difference, and non-stock fraction are never
candidate advantage. Pairwise labels may regularize these task-outcome heads,
but cannot introduce a separate bypass score.

The frozen rule is then evaluated once on a separate executed
teacher-assessment bank. Promotion to policy-target generation requires
preregistered precision, realized advantage over `u0`, coverage, and abstention
behavior on that bank, aggregated over unique source-held-out task/failure
cells. Assessment outcomes certify the **selection rule**; they never filter target cells one by
one and never become generated policy targets.

Model-generated policy targets are produced only afterward on outcome-blind
source histories disjoint from LCWM training/dev, teacher calibration and
assessment, grounded-correction sources, and behavior panels. Candidate
tensors, predictions, eligible masks, and matched-random assignments are sealed
without executing or inspecting candidate outcomes at those target histories.
Any action admitted because its own realized outcome is known belongs only to
the grounded ledger. The generated-teacher comparison may not recycle that
label.

### 5.3 Model-guided flow distillation

Select the strongest eligible predicted outcome. Exact prediction ties share
normalized mass:

\[
i^\star
=
\operatorname*{lexmax}_{i\in\mathcal E}\hat y_i,
\qquad
\sum_i w_i^{\mathrm{WM}}=1
\quad\text{when }\mathcal E\ne\varnothing.
\]

If no proposal is eligible, the state receives only
grounded/demonstration rehearsal. The matched-random control reuses the exact
state mask, total weight, candidate pool, optimizer update, and flow
noise/time, and changes only the candidate assignment.

Stack all candidates for one anchor into one batch and reduce the first-ten
losses over the candidate axis exactly once:

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
replans afterward. The uncredited suffix receives same-noise stock trust;
separate retention states receive full-chunk stock trust. Full demonstration
chunks retain ordinary unmasked π0.5 flow matching from recurrent demo states.

### 5.4 Role of real branches

Executed positive-branch flow matching is retained on its dedicated grounded
split as:

- a ground-truth branch teacher;
- a control that measures how much improvement comes from directly imitating
  better observed actions.

Separate teacher calibration and assessment splits measure predicted
advantage without donating their per-cell labels to policy training.

It is no longer the only path from the world model to the policy. The decisive
comparison is:

| Arm | Interpretation |
|---|---|
| zero-state grounded BC | direct correction imitation through the action head; not matched-capacity |
| constant-token grounded correction | correction imitation plus matched narrow-PEFT capacity without task/history-varying state |
| recurrent-state grounded correction | value of the LC state under identical observed targets |
| recurrent state + grounded + model-guided FM | full method; the model creates additional targets |
| recurrent state + grounded + matched-random FM | candidate-identity control for the generated targets |

---

## 6. Data roles

### 6.1 Ordinary demonstrations and sequential rollouts

These are the primary data for:

- posterior-state construction;
- one-step and short multi-step latent prediction;
- absolute physical/task outcome prediction;
- task reward and \(V^{\pi_0}\);
- full-chunk π0.5 rehearsal and retention.

Successful demonstrations provide broad nominal transition windows, but V7.5
shows that they are not by themselves evidence that the learned state uses the
action or sustains recurrence. They remain necessary closure and retention data
and must not be reduced to rehearsal-only data; action-identifying supervision
comes from matched corruptions and executed siblings.

They also cannot, by repetition alone, identify supported counterfactual
actions, crossed-goal next semantics, history hidden by the current frame, or
better-than-reference policy targets. Those require the intervention,
GoalSpec, history, and paired-continuation data roles below.

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

Every valid sibling group **must** include repeated-action replay-noise
estimation and candidate-independent common-random-number continuations.
Outcomes are exposed at multiple horizons, and the labels distinguish current
predicate truth, milestones, invalidation, damage, and recovery. Crossed
language variants are materialized in the training index with
prompt-conditioned features recomputed from raw observations.

Iteration 1 did not satisfy this normative contract: its continuation RNG
included branch identity, its distinct-goal success label used the canonical
environment goal, and numerical micro-motion could satisfy the single
`effect_resolved` flag. The repair separates physical effect, task-object
effect, semantic disagreement, policy rankability, reference improvement, and
history contrast instead of treating one flag as all six.

Collection is training-data construction, not a standalone diagnostic gate.
v0.7 specializes the new tranche to the terminal LoHo blockers observed in
V7.2: T2 butter after soup, T3 alphabet soup, T4 salad dressing after bowl,
and the T1/T5 butter-drawer chains. All scheduled attempts are retained; a
supported cell contains a next-milestone-positive correction and a same-state
nonpositive sibling, and a subset continues recovery to terminal. Recovery
runs closed-loop during acquisition instead of providing one atomic chunk
followed immediately by the failed canonical policy. Existing physical/null
rows remain retention data but cannot satisfy this positive-contrast
requirement. Missing support masks only that target family and bounds the
result's interpretation; it does not start outcome-selected recollection.

Each accepted anchor is also crossed with paraphrases and one compatible goal
whose active object/order differs. At least part of the bank must contain an
action × goal rank reversal. Static task-bit differences are not sufficient.
`2026-08-03.md` remains the historical owner of the exact V7.3 budget, support
contract, and splits.

### 6.5 Prospective source ownership for the next slice

Before candidate outcomes are inspected, assign source episodes, replay
ancestors, and all descendant snapshots to mutually exclusive roles:

| partition | may be used for |
|---|---|
| WM train | LCWM objectives and training-only normalization statistics |
| WM dev/checkpoint | source-held-out recurrence and action-effect checkpoint rule |
| teacher calibration | fit component margins and confidence/abstention thresholds |
| teacher assessment | certify the frozen selection rule's realized advantage, precision, and coverage |
| grounded correction | per-cell outcome-verified correction targets and grounded controls |
| outcome-blind policy targets | frozen WM and matched-random target generation; candidate outcomes are not queried |
| behavior panels | fresh CRN LoHo evaluation only |

No physical snapshot, source episode, or replay ancestor crosses these roles.
Goal/paraphrase relabels inherit the physical source's partition and do not
create a new independent sample. The split is written before collection and
outcome decoding, so the same siblings cannot train the LCWM, certify its
teacher, and directly supervise its policy.

---

## 7. Two-stage objectives

The v0.9 world-model objective retains the v0.7 loss families and adds the
action-identifying rollout contrast in §3.2:

\[
\begin{aligned}
\mathcal L_{\mathrm{WM}}
=\;&
\lambda_{\mathrm{cl}}\mathcal L_{\mathrm{closure}}
+
\lambda_{\mathrm{phys}}\mathcal L_{\mathrm{physical}}
+
\lambda_{\mathrm{task}}\mathcal L_{\mathrm{milestone/outcome}}
+
\lambda_{\mathrm{inv}}\mathcal L_{\mathrm{language\ contract}}
+
\lambda_{\mathrm{act\text{-}roll}}
\mathcal L_{\mathrm{act\text{-}roll}}.
\end{aligned}
\]

The policy objective is

\[
\mathcal L_{\mathrm{policy}}
=
\lambda_{\mathrm{demo}}\mathcal L_{\mathrm{demo\text{-}FM}}
+
\lambda_{\mathrm{corr}}\mathcal L_{\mathrm{grounded\text{-}correction\text{-}FM}}
+
\lambda_{\mathrm{model}}\mathcal L_{\mathrm{WM\text{-}FM}}
+
\lambda_{\mathrm{suffix}}\mathcal L_{\mathrm{suffix\text{-}trust}}
+
\lambda_{\mathrm{ret}}\mathcal L_{\mathrm{retention\text{-}trust}}.
\]

These form a **two-stage objective bundle**, not one joint backward pass. Stage
A optimizes `L_WM` while the policy boundary remains the exact stock no-op.
After the predictive checkpoint and outcome-blind teacher ledgers are frozen,
Stage B freezes the LCWM and optimizes `L_policy`. Writing the two losses next
to each other specifies the method, not simultaneous optimization.

Optimization order:

1. calibrate and freeze shared-state gradient normalization at the common
   initialization;
2. train one recursive state and its physical/task outcome heads with the
   policy boundary at its stock no-op;
3. calibrate and assess the outcome-blind selection rule on its dedicated
   splits, then freeze the world model and generated-target ledgers;
4. fine-tune the fixed π0.5 boundary with grounded corrections,
   model-generated targets, recurrent demonstration rehearsal, and trust;
5. evaluate every final arm in recurrent full-prompt `N=1` LoHo.

The five shared-state loss coefficients are set once from medians over a frozen
support-balanced calibration microbatch set. Each coefficient maps its raw
median state-gradient norm to the geometric mean of the five nonzero medians;
scaled closure, physical, task-outcome, language-contract, and action-rollout
medians must be within the registered factor-three band. They are not selected
by offline dev loss or behavior.

---

## 8. Evaluation and attribution

### 8.1 World-model evidence

Report from held-out source episodes:

- one-step latent prediction against the EMA posterior target;
- two- and three-block open-loop latent prediction;
- comparison with no-action, copy-state, and mean-next-state baselines;
- task-object physical-effect error on separately labeled meaningful-effect
  branches, always against a zero-effect baseline;
- predicate-flip and phase-aware progress metrics;
- reward/value calibration;
- candidate pairwise accuracy, top-1 regret, and ranking correlation;
- paraphrase equivalence;
- cross-instruction shared-physics consistency.

Latent distance alone is never a promotion metric.

A policy-facing checkpoint must pass a conjunctive, source-held-out contract:
raw history dependence remains within a preregistered fraction of the V7.5
live regime, and candidate action effects beat copy/no-action/stock by more
than the registered replay-noise margin. At the actual π0.5 boundary, report
raw state content, normalized content, projected flow-input differences, and
generated action differences. Task identity, paraphrase separation, aggregate
loss, and sigma-amplified centered distance cannot promote a checkpoint.

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
| zero-state grounded BC | can direct correction imitation through the action head help without an LC adapter signal? |
| constant-token grounded correction | can observed useful actions plus the same trainable boundary improve the deployed flow policy without task/history-varying state? |
| recurrent-state grounded correction | does task/history-varying state help under identical observed targets? |
| LC-off readout | what remains when a trained state arm's projected LC contribution is forced to zero only at evaluation? |
| recurrent grounded + matched-random generated targets | does outcome-blind random candidate assignment explain the model arm? |
| recurrent grounded + outcome-derived WM targets | do predictive state and WM-generated targets improve the VLA? |
| task-agnostic/readout-only state, post-win | must task information enter the predictive state? |
| recurrent state without world losses, post-win | does recurrence/extra capacity explain a win? |
| reset-history full method, post-win | is carried visual-action history needed? |

The final claim requires all three:

1. the VLA improves;
2. the learned state predicts controlled action consequences;
3. matched no-world-model controls do not explain the gain.

---

## 9. Promotion logic

Offline losses, outcome-ranking statistics, and latent diagnostics do not
choose the architecture or substitute for behavior. Two contracts have
different jobs:

| contract | required support | what it may block |
|---|---|---|
| predictive training | valid lineage, enabled predictive targets, source-held-out split, accepted recovery contrasts, and balanced state-gradient reach | promotion of an invalid predictive checkpoint to policy learning |
| generated teacher | frozen outcome-blind rule passes realized advantage/precision/coverage on the source-disjoint assessment bank and yields the registered target-history support | the full-model versus matched-random policy contrast, not LCWM training or a bounded grounded control |

Empty generated-target support does not invalidate LCWM training or a bounded
grounded-control run. It does make the full-vs-random causal contrast
unsupported. The next mainline therefore does not spend a nominal full-model
behavior matrix until the frozen selection rule clears its preregistered
teacher-assessment contract and produces enough outcome-blind target histories
across task and failure families. Counts are over unique physical histories,
not paraphrase/query repeats. Per-cell assessment winners remain assessment
evidence; they do not enter the generated ledger. This is a forward v0.9
resource and claim contract, not a retroactive V7.5 halt rule.

The behavior matrix is staged:

| arm | condition | purpose |
|---|---|---|
| stock | always | original π0.5 baseline |
| zero-state grounded BC | executed clean corrections exist | direct correction/action-head baseline; not matched-capacity |
| constant-token grounded correction | executed clean corrections exist | correction-imitation and matched-boundary control |
| recurrent-state grounded correction | with the preceding arm when grounded support exists | state value under identical correction supervision |
| recurrent grounded + matched random | only after the model-target support minimum is met | candidate-identity control |
| recurrent grounded + WM targets | only after the same model-target support minimum is met | full outcome-derived mechanism |

The routing is:

- grounded correction improves but WM does not beat matched random → repair
  candidate preference, return calibration, or crossed supervision;
- constant-token correction beats the full WM arm → the observed branch targets
  produced the gain and model-generated targets reduced it; do not promote a
  world-model mechanism;
- recurrent grounded beats constant-token grounded → task/history-varying
  predictive state contributes under identical supervision;
- WM ≈ random and both improve → generic supported-action post-training, not
  learned world-model selection;
- grounded correction fails despite accurate target reproduction and a large
  action shift → repair target coverage, rehearsal, or the flow objective;
- grounded correction cannot reproduce/execute registered corrections → the
  fixed boundary or policy objective has failed mechanically; stop rather than
  start an interface sweep;
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
- partition physical sources prospectively among WM train/dev, teacher
  calibration/assessment, grounded correction, outcome-blind policy targets,
  and behavior; relabels inherit the physical source's role;
- certify the frozen selection rule on teacher assessment, but never filter a
  generated target cell using that cell's realized future;
- use task-automaton milestones, current-valid predicates, invalidation,
  damage, success, and multi-horizon progress; never immediate distance as
  advantage;
- train absolute and same-anchor-centered milestone/outcome predictions on
  eligible clean branches; use pairwise comparisons only to constrain those
  outputs, not a detached policy-ranking head;
- retain V7.5 history centering before recurrent attention and LayerNorm before
  the bounded recurrent residual; raw-history and action-effect criteria, not
  normalized distance alone, select the checkpoint;
- remove the training-set common mode, scale-control the resulting recurrent
  state, and inject it into π0.5 through one zero-initialized, bias-free
  projection without a learned scalar gate;
- freeze the world model during policy fine-tuning and train only the
  registered LC projection plus stock-initialized `action_out_proj`;
- use a nonzero constant-token control with the same three trainable boundary
  modules, plus a separate zero-state grounded BC arm and LC-off readout; do not
  call either zero-input path matched-capacity;
- use ordinary trajectories for recursive dynamics;
- use separately labeled physical-effect, semantic-effect, rankable, and
  reference-improving branches for their corresponding losses and controls;
- run policy training after mechanical correctness and the registered
  recovery-support contract; use frozen held-out predictive criteria for model
  checkpointing but never treat an offline score as the Phase-1 endpoint;
- treat public H-WM LIBERO-LoHo Task1–5 as the primary long-horizon endpoint;
- deploy a full-prompt `N = 1` VLA.

### Open

- whether corrected counterfactual data produces source-disjoint,
  task-relevant action and reference-improvement support;
- whether the complete transitioned next-semantic heads generalize beyond
  numerical zero/copy baselines;
- whether a corrected model emits calibrated teachers that beat their exactly
  matched random assignments;
- whether the faithful recurrent state improves public behavior;
- whether WM-selected training beats matched random selection;
- whether grounded executed targets improve the current-state policy;
- whether an outcome-derived teacher adds value beyond grounded correction and
  its matched-random assignment;
- the exact weighting and live-reference fraction for the action-identifying
  recursive rollout after V7.5;
- the exact no-world-loss and task-agnostic/readout-only post-win ablations.

### Resolved by H7

- v0.5 does not improve stock on the public development panel;
- the v0.5 candidate scorer is a readout-conditioned baseline, not the
  intended LC predictive-state transition;
- 100-action unpaired stock continuations are poor primary sibling-ranking
  targets for the current data;
- the existing policy adapter can move first-ten actions substantially, so H7
  alone did not justify an action-expert sweep;
- the H7 recurrent gate was behaviorally inactive, so H7 does not tell us
  whether useful carried state helps.

### Resolved by v0.6 iteration 1

- the recurrent LC state, transitioned candidate path, no-bypass decoder,
  recurrent teacher unroll, Wz-only π0.5 interface, and public-LoHo evaluator
  can be executed end to end;
- iteration-1 model teachers were `0/208`, so its WM/random behavior contrast
  was void rather than a learned tie;
- continuation branch identity entered the RNG, so iteration-1 ranking,
  calibration, and GT targets are not clean sibling-paired labels;
- distinct-goal terminal success and four registered `D_next` semantic heads
  were not faithfully supervised;
- the existing interface can create a large action shift, but two repeated
  targets mostly learned a shared bias; PEFT was not the first repair at that
  stage. v0.7 later freezes one narrow output boundary after the V7.1
  transmission smoke and V7.2 policy-contract failure, without a sweep;
- generic `effect_resolved` counts do not measure the policy-relevant
  counterfactual support required by a world model.

### Resolved by V7.2 and its audit

- the complete data -> LCWM -> policy -> recurrent public-LoHo path can run at
  the target benchmark scale and produce complete videos;
- stock success on the consumed `{1750..1790}` panel is only `1/25`, and the
  dominant failures are concrete first-pick/later-chain blockers rather than a
  generic lack of motion;
- the outcome-blind W2 teacher was close to the proposal prior and its executed
  assessment bank had zero terminal success, so it supplied weak
  candidate-identity supervision;
- V7.2 policy training cancelled each scalar candidate weight by reducing a
  batch of one and dividing by its own weight sum; its soft teacher and matched
  permutation treatments were not implemented;
- the same bug multiplied soft-FM scale by candidate count, full-chunk trust
  opposed the credited prefix, and demo state was not recurrently unrolled;
- V7.2D is therefore an implementation-confounded behavior record, not a valid
  negative for the intended policy objective;
- merely verifying that physical/outcome/rank bundles have nonzero state
  gradients is insufficient: their V7.2 magnitudes differed by up to four
  orders of magnitude.

### Resolved by V7.3 and its audit

- the repaired LCWM -> flow-fine-tuning -> recurrent public-LoHo path can run
  at full five-arm scale with complete checkpoint and video artifacts;
- all five arms observed `0/25` terminal success on the development panel, so
  V7.3 produced no Phase-1 winner;
- aggregate LCWM dev loss is insufficient evidence of useful action-effect or
  sibling-preference learning;
- eight of nine model-teacher anchors abstained, and the sole selected target
  had no registered realized advantage over stock;
- filtering abstentions before schedule construction can turn one eligible
  anchor into the entire model-training exposure and invalidate the intended
  grounded/model/random contrast;
- a bias-free policy projection does not prevent constant adaptation when the
  uncentered pooled state itself contains a dominant common nonzero mode;
- equality between recurrent-state and constant-state arms at a zero behavior
  floor cannot establish general state-content independence.

### Resolved by V7.4/V7.5 and the 2026-08-14 audit

- training-set centering restores policy-facing anchor identity, but no
  readout can recover history that the recurrent core failed to encode;
- the original recurrent path was suppressed by both a shared history common
  mode and pre-`tanh` saturation;
- history centering plus pre-`tanh` LayerNorm makes that path mechanically live:
  V7.5 sustained `hist_rel ~= 0.237` and rollout margin `~= 0.0043` over epochs
  5--11;
- the V7.5 objective did not sustain the live solution: the final checkpoint
  fell to `hist_rel = 0.001047`, rollout margin `~= 1e-6`, and `0/72`
  action-effect wins over copy;
- the aggregate masked dev-loss decrease is optimization evidence only because
  no untrained/constant baseline or per-family dev discrimination artifact was
  persisted;
- a liveness gate anchored to a dead baseline is too weak, and a tiny fitted
  `sigma` can make centered distances large while raw relative content remains
  almost absent;
- task-balanced query visits are not state-balanced data when one task contains
  one repeatedly visited physical anchor;
- V7.4B/V7.5 still lack action × goal reversals, late-blocker coverage, and a
  usable multi-task model-teacher ledger; no V7.5 policy or LoHo behavior stage
  was executed.

### Retired as immediate priorities

- further q-channel-only diagnostics;
- treating one-step centered effect regression as the complete world model;
- requiring the 360-continuation H6.1 bank before model/policy training;
- using effective rank, EMA variance, or mean-next comparison as a launch gate;
- collecting large crossed branch datasets before a method vertical slice;
- making frozen-policy reranking the main system;
- treating crossed-language data as the paper's central contribution;
- requiring perfect physical-state reconstruction before policy learning;
- expanding the same zero-tolerance micro-effect selector before repairing
  causal pairing, GoalSpec labels, and next-semantic supervision.

---

## 11. Immediate implementation sequence

H0–H7, v0.6 iterations, and V7.0–V7.5 are historical evidence. V7.5 stopped at
the support census: its model run repaired recurrence mechanically but ended at
a collapsed checkpoint that recorded `0/72` wins over copy on the registered
action-effect readout, and no V7.5 policy or behavior stage ran. The inherited
V7.4D/E route is retired unchanged rather than executed as a nominal matrix on
four grounded rows. The next registered mainline must instantiate the project's
first principle:

> world-model learning must create a task/history-varying predictive state,
> that state must alter the π0.5 flow through a content-sensitive interface,
> model-derived supervision must change the training target, and the resulting
> recurrent `N=1` policy must improve public LoHo behavior.

The implementation sequence is one bounded train -> fine-tune -> evaluate
vertical slice. Its internal checks prevent another non-instantiation; they are
not a new open-ended diagnostic project.

1. **Seal the evidence and exact inputs.** Land the V7.5 state/trainer/gate
   implementation, the single daily amendment, lightweight manifests, and
   checkpoint/source hashes before new execution. Record that the exact
   3978-key training-cache index was overwritten and is not reproducible from
   the present 5825-key index. Future cache indices and source tensors are
   immutable inputs; no result may claim a Git SHA that predates its code.
2. **Build the next LCWM tranche for model learning, not imitation volume.**
   Before outcome inspection, partition source episodes and replay ancestors
   into WM train, WM dev/checkpoint, teacher calibration, teacher assessment,
   grounded correction, outcome-blind policy-target, and behavior roles; no
   physical source crosses roles. Then preserve ordinary source-disjoint
   trajectory windows for recursive closure, but acquire paired stock/candidate
   outcomes at the actual first-pick,
   recovery, placement, and late-chain failures of all five public LoHo tasks.
   Use provenance-labeled expert prefixes, successful-trajectory replay, or
   acquisition-only state reachers when stock π0.5 cannot reach a blocker;
   these mechanisms reach the snapshot but are not policy targets. From each
   accepted snapshot execute policy-supported sibling actions, retain the `u0`
   outcome, and cross physically compatible GoalSpecs. Bind minimum unique
   physical-anchor, positive/nonpositive sibling, source, task, and
   failure-family support before collection. Query/paraphrase rotation never
   counts as additional state coverage. Advantage is ordered milestone and
   realized continuation improvement, not immediate distance; a grasp/lift may
   move an object away from its goal while still being the correct long-horizon
   action.
3. **Train one repaired, action-identifying LCWM.** Keep V7.5
   `hist_center + r_norm`, one complete language-conditioned transitioned
   state, and the existing grounded heads. Add matched wrong/shuffled-action
   rollout controls and same-state sibling outcome losses so scene or phase
   alone cannot satisfy prediction. Report per epoch on source-held-out data:
   raw `hist_rel`, rollout margin, action-effect and sibling discrimination
   versus copy/no-action/stock, physical effects, goal-conditioned progress,
   paraphrase invariance, and action × goal reversals. Select a frozen
   checkpoint prospectively only when it jointly retains a registered fraction
   of the demonstrated-live recurrence regime and clears the action-effect
   baseline by the replay-noise margin. A `0/72` copy result, final epoch,
   aggregate loss, or sigma-amplified centered distance cannot proceed.
4. **Verify use at the real π0.5 boundary.** For the frozen checkpoint, save
   raw recurrent-state contrasts, normalized interface vectors, projected
   AdaRMS/flow differences, and generated action differences for real/reset,
   shuffled-history, same-goal paraphrase, compatible-goal, and
   candidate-action pairs. This is a bounded mechanical check of the committed
   `TokenQueryPool` interface, not an architecture sweep. Mean pooling remains
   retired and the fixed flow boundary remains unchanged.
5. **Freeze split-faithful ledgers and one arm-independent schedule.** A
   per-cell action with known realized advantage belongs only to the grounded
   ledger. Fit selection margins on teacher calibration, freeze them, and
   require realized advantage/precision/coverage on the separate assessment
   bank. Only then generate WM targets outcome-blind on the policy-target
   histories; ties under the frozen predicted tuple are abstentions. Seal the
   matched-random assignment on the same histories and eligible masks without
   querying outcomes. Compile `task -> source -> physical anchor` before arm
   identity, retain abstentions with zero model loss, and match exposure for
   grounded, model, and random terms without repeatedly turning one anchor into
   the distribution.
6. **Fine-tune the existing π0.5 flow.** From one stock initialization and the
   same schedule, train zero-state grounded BC, constant-token grounded,
   recurrent grounded, full LCWM-target, and matched-random-target arms. The
   sha-seeded `TokenQueryPool`, registered centered-state projection, and
   stock-initialized `action_out_proj` are the only trainable boundary modules;
   the constant-token arm keeps all three in graph on one frozen nondegenerate
   token block. Recurrent demonstration rehearsal, retention/suffix trust,
   optimizer, flow noise/time, and step count are matched. Also save a distinct
   LC-off readout for every trained state arm. Store results using
   execution-date names and save evaluation videos during training together
   with recurrent state and action-delta traces.
7. **Return immediately to behavior.** Run stock and all supported final arms
   on a fresh common-random-number public-LoHo panel with full prompts,
   recurrent `N=1` deployment, and one video per rollout. Task-balanced terminal
   success is primary; ordered-prefix/Q-AUC, damage, completion time, and
   failure phase localize the result. Phase 1 still requires the trained
   LCWM-conditioned π0.5 to beat stock π0.5 on long-horizon behavior; no
   model, teacher, interface, or post-training metric replaces that endpoint.

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
enter training, and recurrence remained gated out of the policy.

v0.6 iteration 1 then implemented the intended recurrent candidate graph and
reached a 125-episode behavior matrix. Stock achieved `SR=.040/Q=.560`; the
GT-current arm achieved `SR=.040/Q=.511`; all three recurrent arms shared one
byte-identical `W_z` and achieved `SR=.000/Q=.528`. This is not evidence that
WM selection ties random selection: the calibrated teacher set was empty, so
the comparison was void by construction.

The post-run audit further found candidate-specific continuation noise,
canonical-only success labels for distinct goals, silent next-semantic heads,
micro-effect admission, no explicit history contrasts, and a task-truncated
policy rehearsal schedule. Subsequent revisions repaired data semantics and
eventually executed V7.2 through a 175-rollout public matrix. That matrix found
stock `SR=.040` and every trained arm `SR=.000`, but its policy trainer had
cancelled the registered candidate weights and multiplied soft-FM scale by
candidate count. It is therefore implementation-confounded rather than a
valid LCWM negative.

V7.3 consumed the reusable physical/outcome evidence, added a recovery-crossed
tranche, trained one gradient-balanced LCWM and four π0.5 arms, and returned to
a fresh 125-rollout recurrent `N=1` matrix. All five arms, including stock,
observed `0/25` success. The execution and artifacts are real, but three defects
bound the interpretation:

- only one of nine model-teacher anchors was eligible, and its selected action
  was outcome-equivalent to stock under the registered tuple;
- the model scheduler repeated that one anchor at every optimization step while
  the grounded rows represented only six unique anchors, violating matched
  exposure;
- the policy-facing `Pool(z)` was dominated by a common nonzero mode: hundreds
  of real states had cosine above 0.99999 to the global mean and their
  state-specific projected variation was approximately `1e-4` of the common
  component.

The corrected V7.3 status is therefore **terminal behavior executed, no
Phase-1 win, intended causal comparison implementation-confounded and
supervision-coverage-deficient**. It falsifies the current
data/teacher/pooling/scheduler bundle, not the central LCWM thesis. In
particular, `lc_grounded = correction_bc` at a zero behavior floor only says
that the deployed pooled interface supplied no useful varying state; it does
not show that the internal predictive tokens or a content-preserving interface
cannot help policy learning.

V7.4A then isolated one part of that failure: centering recovered strong
policy-facing anchor identity (`T1 = 26.07`), but real/reset and
real/shuffled-history contrasts still failed on all 11 audited anchors. V7.4B
collected genuine task-object-resolved contact effects, yet decision support
remained concentrated in one t2|train terminal-recovery anchor, with no
action × goal reversals and no late-blocker family in any of the five tasks.

V7.5 localized and repaired the recurrent mechanism. `hist_center + r_norm`
made the channel live and training held `hist_rel ~= 0.237` for epochs 5--11.
The final checkpoint nevertheless collapsed to `0.001047`, its rollout margin
fell to approximately `1e-6`, and it beat copy on `0/72` held-out action-effect
rows. The registered gate passed only because its liveness threshold was
anchored to V7.3's dead baseline; normalization by `sigma = 1.27e-3` also made
small raw residuals look large at the centered interface. The current status is
therefore **recurrent mechanism repaired, final predictive state collapsed,
action-effect learning unsupported, and policy/behavior not run**. This is a
useful model-learning localization, not a Phase-1 result and not a negative on
the LCWM hypothesis.

LC-Flow becomes the intended framework only when the following causal chain is
instantiated:

```text
ordinary trajectories + paired late-blocker/crossed-goal branches
                    ↓
recursive language-conditioned predictive state
                    ↓
held-out action effects + grounded task outcome prediction
                    ↓
assessment-certified, outcome-blind WM policy targets
                    ↓
π0.5 flow fine-tuning under matched controls
                    ↓
better recurrent full-prompt N=1 public-LoHo behavior
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
π0.5 flow interface through one zero-initialized projection. A faithful
predictive checkpoint trains every currently supported crossed-language,
task-automaton, paired-continuation, closure, and physical/task-grounding
objective through that state. Every enabled head must receive a real target,
gradient, and parameter update.

Sibling preference is conditional on eligible clean pairs. When the channel
is empty, it is explicitly disabled and labeled `unsupported`; silently
leaving a nominally enabled head at initialization is forbidden. A complete
model-generated LC-Flow claim additionally requires learned sibling
preference, nonempty WM/matched-random teachers, and improved behavior.

Iteration 1 instantiated the graph but not all predictive supervision.
V6.7/V6.8 repaired pairing and GoalSpec labels and exhausted the current bank,
then found zero clean train ranking support. V6.9 therefore separates
predictive learning from targeted grounded correction acquisition without
changing the architecture.

The data batch targets contact, object motion, recovery, placement, and final
unresolved subgoals. Ranking support is a within-snapshot paired contrast
beyond replay noise; absolute `dQ>0` and a 75%-expected non-stock argmax are
not evidence.

### 13.3 Execution ownership

The detailed V7.3 execution contract in `2026-08-03.md` is now a frozen
historical preregistration. `2026-08-07.md` owns its terminal execution record
and explicit post-run audit amendment. `2026-08-09.md` is the sole daily record
for V7.4/V7.5 registration, execution, and its dated 2026-08-14 audit
amendment; no second progress-note file is created for that revision. This
design document owns the current v0.9 architectural invariants and immediate
sequence. The next experiment's exact budgets, artifacts, seeds, thresholds,
and routing are bound in its one dated daily record before execution.
Historical evidence is not silently rewritten: later-discovered errors are
marked as dated audit amendments in the same daily record. Another broad
diagnostic phase, exposure-only V7.3 replay, or fixed large-sample prerequisite
is not part of the mainline.

### 13.4 v0.7 correction

v0.7 preserves the one complete task-conditioned transitioned state and
changes the supervision and policy handoff:

- recovery-crossed branches explicitly contain blocker-relevant positive and
  negative consequences plus crossed-goal reversals;
- task-outcome predictions themselves form conservative candidate preference;
- shared-state loss contributions are normalized before training so physical
  scale cannot erase task-outcome shaping;
- weighted policy FM reduces the complete candidate axis once;
- the fixed LC projection + `action_out_proj` boundary is shared by real-state,
  constant-state, grounded, full-model, and matched-random controls;
- every revision still terminates at recurrent full-prompt `N=1` public-LoHo
  behavior, never at an offline teacher or representation metric.

### 13.5 v0.8 correction

V7.3 shows that a formally recurrent state is not sufficient when the policy
interface removes nearly all of its variation. v0.8 therefore separates the
existence of a common adapter offset from the use of decision-relevant state
content:

\[
c_t^\ell=
\operatorname{Norm}\!\left(\operatorname{Pool}(z_t^\ell)-\mu_{\mathrm{train}}
\right),
\qquad
v_\theta'=v_\theta+W_c c_t^\ell,
\]

where the normalization statistics and zero-initialized projection are frozen
in the run contract and saved with the run. This centered interface is the
first bounded candidate, not a claim that mean pooling is ultimately optimal.
If complete-anchor
measurements still show content collapse, one learned query attends directly
to the LCWM tokens and supplies the same fixed flow boundary. The alternative
changes information preservation, not deployment topology: both remain one
recurrent `N=1` policy with no external scorer.

The following are now implementation invariants:

- genuine task/history changes survive through the projection and measurably
  change the flow input; a constant offset cannot satisfy this condition;
- same-goal paraphrases remain close while compatible different goals may
  produce different semantic predictive states;
- physical predictions for one realized `(s,a,s')` remain invariant to goal
  relabeling while progress/value predictions follow the relabeled goal;
- model-teacher eligibility uses a frozen predicted-outcome margin whose
  selection rule demonstrates nontrivial paired advantage over stock on a
  separate assessment bank; target cells are never filtered by their own
  realized outcome;
- policy exposure is fixed before arm identity and includes abstentions as
  zero-loss cells;
- online recurrent states and action deltas are saved during LoHo evaluation,
  so behavioral failure can be separated from another silent interface
  collapse.

These invariants support the main experiment; they do not replace it. The only
Phase-1 success criterion remains a trained LCWM-conditioned π0.5 policy that
beats stock π0.5 on public long-horizon behavior.

### 13.6 v0.9 correction

V7.4/V7.5 separate three properties that earlier revisions treated too nearly
as one:

1. **interface identity** — centering can expose differences already present
   in the state;
2. **recurrent liveness** — `hist_center + r_norm` can transmit history through
   the recurrent core;
3. **action-conditioned prediction** — the transitioned state must distinguish
   realized consequences of supported alternative actions beyond copy/stock.

The first two cannot substitute for the third. v0.9 therefore retains the
repaired recurrent core but promotes a checkpoint only under raw,
source-held-out history retention **and** action-effect discrimination.
Normalized or centered quantities are still reported for optimization and
interface auditing, but any raw quantity from which they are derived is
reported beside them; a small fitted scale cannot turn residual numerical
variation into a scientific PASS. V7.4's fallback decision is also final:
v0.9 uses the learned `TokenQueryPool`, LC projection, and `action_out_proj` as
its three trainable boundary modules; mean pooling is not reopened.

The policy stage is likewise tied to the mechanism. An empty or nearly empty
model-teacher ledger does not make LCWM training invalid, but it cannot support
a nominal full-vs-random behavior comparison. The next slice calibrates an
outcome-blind selection rule, verifies its realized advantage and coverage on
a separate teacher-assessment bank, then freezes WM and matched-random targets
on disjoint histories whose candidate outcomes remain unobserved. Per-cell
realized winners belong only to the grounded ledger. Only then does it
fine-tune the existing π0.5 flow and run recurrent `N=1` public-LoHo. Its
first-principle chain is fixed:

```text
model-learning data
        -> recurrent language-conditioned predictive state
        -> assessment-certified, outcome-blind WM target
        -> π0.5 flow fine-tuning
        -> recurrent N=1 public-LoHo behavior
```

No intermediate diagnostic becomes the project endpoint. Conversely, running
policy compute on a checkpoint that recorded `0/72` wins over copy on the
registered action-effect readout, with only four sparse grounded rows, would
not instantiate this chain and therefore would not count as “entering the
mainline.”
