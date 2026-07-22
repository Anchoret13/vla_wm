# pi05-lcwm — Framework Design v0.2

Living design document for the language-conditioned world-model family on frozen
π0.5, LIBERO-10. Structure candidates stay comparable under one data / interface /
evaluation contract; experiments promote, modify, or delete them here. **[DEC]** =
open decision with current default. **[EST]** = measure rather than assume.

---

## 0. Objective and Phase-1 win condition (updated 2026-07-20)

Language acts as a **task identifier**, not as unconstrained text state. A task
normalizer maps paraphrases of the same task to one canonical identity:

```text
τ = g(language)                  # canonical task identity
z_t^τ = F(o_≤t, a_<t, τ)        # task-conditioned predictive state
```

Required representation behavior:

- same physical history + paraphrases of the same task → the **same** task state
- same physical history + different tasks → **different, decision-relevant** task states
- latent difference is not sufficient by itself: changing `τ` must change future
  prediction, task-relevant predicates / progress, and eventually behavior in the
  correct way

### Objective hierarchy

1. **Core research objective:** learn a language-conditioned predictive state on top
   of frozen π0.5. It must summarize physical history and canonical task identity well
   enough for action-conditioned, multi-step future prediction. Multitask sharing,
   long-horizon reasoning, and a better policy are consequences sought from this state.
2. **Representation criterion:** under identical observation / action history,
   paraphrases of one task should induce the same predictive state, while different
   tasks should differ where their future-relevant semantics differ. Predicted futures
   and behavior arbitrate this property; raw latent distance does not.
3. **Phase-1 instrumental test:** use the learned state to improve frozen π0.5 on
   LIBERO-10. Proposal reranking is the first policy interface, not the definition or
   final objective of the world model.

**Phase 1:** on standard LIBERO-10, frozen `lerobot/pi05_libero_finetuned` remains the
proposal policy; the learned module must improve its paired closed-loop success rate.
The backbone is not finetuned. Current reference is 96/100 in Stage-1 `lerobot-eval`
and 98/100 in the single-env chassis run; the final comparison must use one locked
harness, identical init-state schedule, and paired seeds.

Phase-1 policy interface = **proposal reranking**, not action generation from scratch:

```text
observation + canonical task
              │
              ├── frozen π0.5: N candidate 50-action plans
              │
              └── learned state + predictor: score each candidate
                                      │
                         conservative selector / gate
                                      │
                         execute the winning first 10 actions
                                      │
                              observe and replan
```

Branch-scale candidate ranking is therefore a **downstream diagnostic and Phase-1
interface**, not the core learning target and not a prerequisite for the first LCWM.
Restoring one snapshot and varying only the action branch creates controlled
counterfactual futures. Those branches serve three distinct purposes:

1. test and improve action-conditioned latent dynamics outside the single demonstrated
   action at a state;
2. test whether a task swap changes predicted predicates / progress for the right
   reasons under the same physical history and candidate actions;
3. derive ranking / value labels that connect the learned state to π0.5 improvement.

Purpose 3 must not replace 1–2. A direct-Q model can rank candidates without learning
a reusable predictive state; Candidate C exists precisely as that control.

Phase-1 success must report more than aggregate SR: paired ΔSR, rescue count
(`π0.5` fail / reranker success), harm count (`π0.5` success / reranker fail),
intervention rate, latency, and seed/init-level uncertainty. **[DEC]** provisional
done-check = positive paired ΔSR with rescue > harm on the full locked schedule;
statistical threshold and episode count are set before final eval.

---

## 1. Locked common contract

- Environment: LIBERO-10, all 10 tasks. Development may prioritize current weak tasks
  t6 / t8 / t9, but architecture selection and final reporting cover all tasks.
- Policy: frozen π0.5; candidate `0` is the stock action plan the baseline would have
  used, candidates `1..N-1` are additional π0.5 samples. Default `N = 16`.
- Policy chunk = 50 env actions; commitment / decision stride `c = 10`. A candidate is
  split into five 10-action blocks for latent lookahead; `K ∈ {1, 3, 5}` is measured.
- All candidate structures receive the same frozen visual features, proprio, executed
  action history, task identity, and train/validation/test split. Where branch data is
  used, they receive the same snapshot siblings and labels.
- All candidates expose the same four operations:

```python
e_task = task_encoder(language_or_task_id)
state = state_encoder(obs_features, proprio, action_history, state_prev, e_task)
future = dynamics.rollout(state, candidate_action_blocks, e_task)
score = scorer(state, future, candidate_action_blocks)
```

- Main success / progress / value heads may not receive raw language or `e_task`
  directly: task information must pass through the learned task state. Direct-language
  heads are retained only as explicit language-prior baselines.
- Branch siblings from one simulator snapshot stay in the same dataset split.
- Shared randomization schedules, pinned seeds, environment fingerprints, replay-valid
  filtering, and exact candidate noise are saved with every comparison.

---

## 2. Task normalizer

Phase-1 structural experiments use the oracle identity table first, so task parsing and
world-model structure are not changed simultaneously:

```python
e_task = nn.Embedding(num_tasks=10, embedding_dim=256)(task_id)
```

The text-facing version is then attached as:

```text
language / paraphrase → frozen text features → 10-way prototype matcher → τ → E[τ]
```

`TASKID` is therefore the Phase-1 canonical semantics / oracle upper bound, not merely
an ablation. Conditioning controls:

- `TASKID`: oracle canonical identity, used to choose the state architecture
- `LANG→TASKID`: production text interface; held-out paraphrases must map to the same `τ`
- `FREE`: no task identity, negative control
- `SHUFFLED`: wrong identity on the same physical history, causal control

**[DEC]** whether `LANG→TASKID` uses hard argmax prototypes (exact paraphrase
invariance) or a soft mixture. Default = hard identity for Phase 1; continuous/open-task
semantics are deferred.

---

## 3. Frozen feature streams and tap-source decision

The 2026-07-19 visualization establishes a heavy-tailed, spatially nonuniform prompt
effect in last-layer image tokens, but does **not yet** separate generic prompt
sensitivity from task-specific modulation. Current shift-map magnitude correlations
(`r = 0.38–0.69`) are evidence of a shared prompt-sensitive subset, not proof that the
remaining signal is content-free.

The tap decision is therefore split by **role**, rather than forced into one global tap:

```text
image ── SigLIP pre-trunk ─────────────────────────────── H_world
   └── canonical task prompt + PaliGemma last layer ──── H_task^τ
```

Terminology is kept strict: `H_world / H_task` are frozen VLA feature inputs; they are
not yet the world-model hidden state. `Z_t^τ` in A, or `(W_t, G_t^τ)` in B, is the
learned recurrent predictive state. We choose the frozen input by how well a learned
state can use it for dynamics, not by renaming the tap itself as the world model.

Candidate inputs:

- shared/world stream: SigLIP pre-trunk **or** constant-prompt last-layer
- task stream: canonical real-prompt last-layer
- monolithic control: one selected stream plus explicit `e_task`

A single canonical-prompt π0.5 prefix forward can expose SigLIP features,
task-conditioned last-layer features, and the KV cache for N-sampling. Constant + real
last-layer taps require two trunk passes and carry an online latency cost.

Tap arbitration uses two probe families:

1. **World probes:** object pose, proprio, controlled next-state prediction.
2. **Task probes:** BDDL predicate/progress, remaining-time/success value, candidate
   ranking, same-task paraphrase consistency, and same-state task intervention.

The critical same-scene control is LIBERO-10 t0 vs t1 (`LIVING_ROOM_SCENE2`): the
physical scene is shared while the target object pair changes. Compare canonical t0,
t0 paraphrases, canonical t1, unrelated t2, and constant prompt on identical frames.
Target-object vs distractor/background selectivity needs simulator segmentation or
projected object masks; heatmap appearance alone is not a decision criterion.

**[DEC]** final feature inputs remain open, but standalone probes no longer gate LCWM
training. The next arbitration happens inside matched learned-state models using a
small provisional input matrix: SigLIP-only, canonical-last-layer-only, and fused.

Evidence log (2026-07-20, controlled probes; details in `2026-07-19.md`):

- v6 "cross-layout 0.698" RETRACTED — q-only proprio control (0.641) nearly matches
  all token probes on that metric. The old result is heavily confounded by
  proprio / trajectory information; A ≈ q-only does not support a percentage
  decomposition such as "90% arm information."
- Layout probe set (50 init states/task, 40/10 init-state split): ALL streams fail
  the pass rule (siglip best at mean R² 0.035, margin over q-only 0.19 < 0.3).
  Decision-time cm-level layout reading via frozen features + light probes is NOT
  certified. SigLIP nevertheless lowers absolute error on 10/10 tasks (macro 1.71 cm
  vs q-only 1.96 cm; per-task ranges 1.35–2.30 vs 1.70–2.37 cm), so this is a weak,
  consistent visual signal rather than evidence that the frozen features contain no
  layout information. The failure is scoped to the current agentview-only, 40-shot,
  restricted LocProbe and single 40/10 split.
- Task-side prompt sensitivity and preliminary paraphrase consistency keep the
  canonical last-layer stream as a provisional `H_task` input. Object-role
  selectivity remains OPEN: the v2 projected masks are horizontally inconsistent
  with the displayed scene, so the reported ~30× group contrast is not evidence until
  projection is calibrated or direct segmentation is available.
- Consequences: choose the hidden-state input through dynamics learning rather than
  more standalone readout probes. Candidate-D pose supervision remains an explicit
  training-only ablation for A/B; the current light-probe failure does not establish
  that it is necessary.

---

## 4. Candidate A — monolithic task-conditioned carry

Minimal recurrent WM and implementation baseline:

```text
visual tokens ─────────────┐
proprio + previous actions ├── carry cross-attention ── Z_t^τ
previous carry Z_{t-1}^τ ──┤                              │
task prototype E[τ] ───────┘                 action-block dynamics
                                                          │
                                                       Z_t+k^τ
                                                          │
                                          progress / value / success
```

```python
Z_t = Carry(Z_prev, H_t, q_t, a_prev, e_task)       # (B, M, d_z)
Z_next = Transition(Z_t, encode(a_t:t+10), e_task)
score = Value(Z_next)                                # no e_task bypass
```

Initial scale: `M = 4`, `d_z = 384`, two observation cross-attention blocks, four
transition blocks. **[DEC]** match Candidate B parameter count if the initial result is
capacity-limited.

Strengths: smallest path from current plan to an end-to-end trained reranker; cheap
multi-step rollout. Risks: physical and task state are entangled; task identity can be
copied into `Z` without changing useful dynamics; counterfactual task swaps are hard to
interpret.

Status: **active baseline**.

---

## 5. Candidate B — factorized world state + task state

Primary structured hypothesis:

```text
H_world + proprio + action history + W_{t-1}
                         │
                    World Updater
                         │
                        W_t ──────────────┐
                                         │ cross-attend
E[τ] + previous G_{t-1}^τ ───────────────┘
                                         │
                                   Task State G_t^τ
```

```python
W_t = WorldUpdate(W_prev, H_world, q_t, a_prev)       # task-invariant memory
G_t = TaskUpdate(G_prev, W_t, H_task, e_task)         # task-conditioned memory

W_next = WorldTransition(W_t, encode(action_block))
G_next = TaskTransition(G_t, W_t, W_next, encode(action_block), e_task)
score = Value(G_next)                                 # no direct e_task input
```

Initial scale: `M_world = 8`, `M_task = 4`, `d_z = 384`, `d_e = 256`, two updater
blocks per stream, four transition blocks. The physical transition is shared across
tasks; task transition tracks relevant objects, goal predicates, progress, and
history-dependent subgoals.

Required causal behavior:

- fixed physical history, t0 vs t1 task identity: `W_t` stable, `G_t` changes
- task-0 canonical vs task-0 paraphrase: both `W_t` and `G_t` stable
- fixed candidate union, t0 vs t1: ranking changes toward the correct target objects

Strengths: directly represents the research hypothesis; shared physics is learned
once; language invariance and intervention have explicit locations to test. Risks:
more moving pieces; `W` can discard task-relevant detail, or `G` can reduce to a static
task code. State trajectories, prediction behavior, and ranking—not raw latent
distance—must arbitrate.

Status: **active primary candidate**; favored conceptually, not yet experimentally.

---

## 6. Candidate C — direct candidate-Q transformer

Strong policy baseline without explicit future-state rollout:

```text
history + E[τ] ── task-conditioned encoder ── Z_t^τ
                                                   │
candidate 50-action plan ──────────────────────────┤
                                                   │
                                      candidate transformer
                                                   │
                         Q^π = P(success | candidate, then π0.5)
```

It predicts the outcome of executing the candidate's committed first block and then
continuing / replanning with π0.5. Train absolute value and within-snapshot pairwise
ranking. `E[τ]` enters the history encoder; the Q head receives `Z_t^τ` and the action
plan, not a direct task-identity bypass. At inference it scores the same N candidates
as the WMs.

Strengths: objective matches Phase-1 selection directly; no compounding latent rollout
error; likely data-efficient. Risks: not a reusable world model, weaker action-OOD
generalization, no inspectable future state.

Interpretation rule: Candidate C is not the target framework, but it is the required
strong baseline. If A/B cannot match its held-out ranking or policy improvement, the
claimed benefit of explicit latent dynamics is unsupported.

Status: **downstream strong baseline**. It is implemented after the first A/B
state-learning run, not used to choose the world-model hidden state.

---

## 7. Candidate D — task-queried object / relation state

Structured extension of Candidate B:

```text
visual tokens → object/fixture slots → temporal slot dynamics
E[τ] → goal queries → relevant object/relation state → predicate/value heads
```

LIBERO goals use a small predicate vocabulary (`on`, `in`, `open`, `close`, `turnon`,
`turnoff`), so object slots plus relation tokens are a plausible inductive bias.
Simulator object poses / segmentation / predicates may be used as training-only
auxiliary targets.

Strengths: interpretable task state and explicit long-horizon relations. Risks: slot
identity drift, additional perception supervision, and substantially larger engineering
surface before the reranking hypothesis is established.

Status: **deferred**. Promote only if A/B diagnostics show that unstructured carry
cannot retain object identity or predicate state.

---

## 8. Decision-rate dynamics and scoring

One learned transition corresponds to the policy commitment `c = 10`, not one MuJoCo
step. Each π0.5 candidate supplies five coherent action blocks:

```text
A_i = [a_0:10] [a_10:20] [a_20:30] [a_30:40] [a_40:50]
                │
          K latent transitions, K ∈ {1, 3, 5}
                │
       predicted progress + terminal value
```

Candidate score default:

```python
score_i = predicted_progress_gain_i + gamma**K * terminal_value_i
```

**[DEC]** direct Q, predicate completion, failure-risk, and π0.5-prior terms may join
after calibration. Runtime selection is conservative:

```python
if best_score - baseline_score > margin and uncertainty < threshold:
    execute(best_candidate[:10])
else:
    execute(stock_candidate[:10])
```

Candidate `0` is always retained. Report intervention, rescue, and harm by confidence
bin; tune margin / uncertainty on validation snapshots only.

---

## 9. Training data

Three sources, introduced as needed rather than as prerequisite gates:

1. **Expert demonstrations (500 LIBERO-10 demos):** recurrent-state pretraining,
   observed transition targets, object/predicate/progress auxiliaries. Only replay-valid
   trajectories are used for counterfactual anchors.
2. **π0.5 on-policy trajectories:** actual state distribution, including current
   failure seeds and recovery attempts.
3. **Snapshot branch dataset:** controlled counterfactual-future supervision and the
   later policy-improvement interface. From one snapshot, generate stock + N-sampled
   candidates, restore the identical sim state, execute each first 10-action block,
   and record next state / predicates / progress. Continue a stratified subset with
   π0.5 to termination for success and remaining-time targets.

```text
branch record = {
  snapshot_id, suite, task_id, task_identity,
  obs_features, q, action_history, state_flat,
  candidates: (N, 50, 7),
  next_obs_features: (N, ...),
  predicate_bits / progress: (N, ...),
  continuation_success: (N,), remaining_steps: (N,),
  replay_valid, policy_seed, candidate_noise_seed,
}
```

Sampling is stratified by task, early/mid/late phase, success/failure trajectory, and
model confidence. All N branches of one snapshot share a split. **[DEC]** first branch
budget and full-continuation fraction are set after the formal diversity / end-state
dispersion measurement.

Expert and on-policy sequential transitions are sufficient to start the hidden-state
sweep. Branch siblings are added when the first LCWM exists, primarily to prevent an
action-ignoring predictor and to test counterfactual futures from identical starts.
The same records later supply within-snapshot value / ranking labels. Ranking success
alone does not establish a world model because Candidate C can obtain it directly.

---

## 10. Shared objectives

```text
L_world    : k-step target-state prediction (EMA or VICReg target)
L_task     : task-state prediction under the same canonical identity
L_pred     : BDDL predicate bits + progress / remaining-time prediction
L_value    : continuation success / calibrated terminal value
L_rank     : within-snapshot pairwise ranking
L_para     : same-task paraphrase state / score consistency
L_anchor   : proprio + optional object-pose reconstruction, anti-collapse
```

State-learning core for A/B:

```python
L_state = lw * L_world + lt * L_task + lp * L_pred \
        + lpara * L_para + la * L_anchor
```

Phase-1 downstream heads, attached after a viable learned state exists:

```python
L_policy = lv * L_value + lr * L_rank
```

A has no separate `L_world/L_task`; B exposes both. A/B are first promoted by
predictive-state diagnostics, then tested for decision relevance with `L_policy`.
Candidate C uses `L_value + L_rank + L_para` directly and is therefore a policy
control, not evidence for the central world-model hypothesis. Loss weights are tuned
on held-out transition/ranking metrics, never final closed-loop test SR.

Avoid trivial task-code success:

- no raw-language/task embedding bypass into the main heads
- report a task-only language-prior scorer
- measure dynamic state change, not only `G_t^τ ≠ G_t^τ'`
- require the same-state task swap to change candidate outcomes correctly

---

## 11. Architecture-selection experiments

### 11.1 Feature / identity controls (bounded screening role)

- Correct joint-PCA centering and report explained variance.
- Same frame prompts: canonical t0 / t0 paraphrases / same-scene t1 / unrelated t2 /
  constant. Report quantiles and tail fractions, not mean only.
- Compare shift-vector direction as well as shift magnitude maps.
- World/task probe ladder; target-object vs distractor/background selectivity.
- Task normalizer: held-out paraphrase classification and exact state/score consistency.

These controls set provisional inputs and catch extraction bugs; they do not choose the
world-model state. Except for one bounded projection correction / rerun, the next
decision is made through actual dynamics learning.

### 11.2 Hidden-state-first structure comparison

First hold minimal A fixed as a learning instrument and compare the provisional input
arms `{SigLIP-only, canonical-LL-only, fused}` on identical sequential splits. Then
take the top input arm(s) into a matched A-vs-B comparison; do not change the input and
state factorization in the same ablation. Primary metrics:

- one-step and `k = 2..5` action-conditioned target-state prediction
- BDDL predicate / task-progress prediction from the learned state
- degradation under action shuffle and task-ID shuffle relative to proper conditioning
- same-physical-history t0/t1 intervention and same-task paraphrase consistency
- history dependence, latent effective rank, per-dim std, and collapse diagnostics
- held-out init-state / trajectory generalization and learning curves

A hidden state is "usable for learning" only if it improves over no-action / no-task
and trivial-copy baselines, remains non-collapsed, and changes predicted futures—not
merely latent distance—under the task intervention. Static object-pose decodability is
not a prerequisite.

### 11.3 Branch-scale downstream assay

After at least one A/B state passes §11.2, add controlled snapshot siblings and attach
the shared value / ranking interface. Train Candidate C on the same branch split as the
no-world-model control. Primary metrics:

- branch next-state and predicate/progress prediction from identical starts
- within-snapshot pairwise rank accuracy, NDCG, top-1 regret
- held-out candidate noise and init-state generalization
- same-candidate-pool t0/t1 ranking intervention and paraphrase invariance
- value calibration and continuation-success prediction

Only the top two policy scorers enter expensive closed-loop comparison. Latent
self-prediction establishes the research object; branch and policy tests establish
that the object is decision-relevant for Phase 1.

### 11.4 Closed-loop Phase-1 comparison

- Stock π0.5 vs `π0.5 + scorer`, identical init-state and seed schedule.
- First report per-task and paired episode table, then aggregate.
- Report rescue, harm, intervention, confidence, steps-to-success, and wall-clock.
- Development failure states may tune data collection but never enter a separate
  cherry-picked primary score.
- **[DEC]** proposed final schedule = all 50 init states × 10 tasks (500 paired
  episodes), with additional policy-noise repetitions only if the paired confidence
  interval remains unresolved.

---

## 12. Current open-decision ledger

- State form: monolithic A vs factorized B; direct-Q C is the required baseline.
- Which provisional input arm produces the best learned predictive state:
  SigLIP-only vs canonical-last-layer-only vs fused. Static probes do not close this.
- In B, whether `H_task` is canonical-prompt last-layer or task queries over
  shared-world tokens.
- Whether one real/canonical prefix pass is sufficient for both policy and WM.
- `M_world / M_task`, recurrence form, and EMA vs VICReg target.
- Decision rollout `K = 1 / 3 / 5`; action-block encoder structure.
- Predicate/progress supervision granularity and training-only privileged pose labels.
- Branch count, continuation fraction, σ-perturbation proposal tier.
- Ensemble / uncertainty method and conservative intervention threshold.
- Final paired episode count and significance criterion.

---

## 13. Immediate order of work

1. Freeze the provisional input matrix `{SigLIP-only, canonical-LL-only, fused}` and
   oracle `TASKID`. Correct / rerun the selectivity projection once in parallel; it is
   a bounded diagnostic, not a training gate.
2. Build the decision-rate sequential dataset / trainer and implement minimal A as the
   fixed instrument for the frozen-input learning sweep.
3. Select the input that supports a learnable predictive state with multi-step
   prediction, action/task shuffles, task intervention, paraphrase consistency,
   non-collapse, and learning curves; then compare A vs B on the top input arm(s).
4. Measure formal N=16 action / end-state diversity and collect the small branch set
   needed for counterfactual action coverage and the downstream Phase-1 interface.
5. Attach the shared value / ranking head; train Candidate C on the same branches as
   the no-WM policy control.
6. Run conservative closed-loop reranking for promoted A/B states, then lock and run
   the paired Phase-1 evaluation once.

- 2026-07-22 §11.2 learning-behavior arbitration (details `2026-07-21.md`): input
  arm = **SigLIP pre-trunk** — action-shuffle sensitivity 0.96–1.16 vs real_ll
  0.23–0.25 vs fused 0.12–0.18 (language-mixed inputs let the transition partially
  ignore actions); all arms equal on predicate F1 (0.98–0.99) / progress R²
  (0.92–0.94) / copy-margin (0.69–0.79). Rank criterion re-derived from data
  references (raw-token budget 36.5 → bar 18.3; the original 96 was uncalibrated);
  siglip + var-reg passes all criteria on both seeds. Instrument objective updated:
  EMA + variance regularization.
