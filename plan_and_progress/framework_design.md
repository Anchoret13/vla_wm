# pi05-lcwm — Framework Design v0.1

Living design document for the language-conditioned world-model family on frozen
π0.5, LIBERO-10. Structure candidates stay comparable under one data / interface /
evaluation contract; experiments promote, modify, or delete them here. **[DEC]** =
open decision with current default. **[EST]** = measure rather than assume.

---

## 0. Objective and Phase-1 win condition (2026-07-19)

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
  prediction / candidate ranking in the correct way

**Phase 1:** on standard LIBERO-10, frozen `lerobot/pi05_libero_finetuned` remains the
proposal policy; the learned module must improve its paired closed-loop success rate.
The backbone is not finetuned. Current reference is 96/100 in Stage-1 `lerobot-eval`
and 98/100 in the single-env chassis run; the final comparison must use one locked
harness, identical init-state schedule, and paired seeds.

Primary policy mechanism = **proposal reranking**, not action generation from scratch:

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
  action history, task identity, branch data, and train/validation/test split.
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

**[DEC]** final feature inputs remain open until these probes. Do not extract the full
segment dataset before the decision.

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

Status: **active strong baseline**.

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

Three sources, introduced in order:

1. **Expert demonstrations (500 LIBERO-10 demos):** recurrent-state pretraining,
   observed transition targets, object/predicate/progress auxiliaries. Only replay-valid
   trajectories are used for counterfactual anchors.
2. **π0.5 on-policy trajectories:** actual state distribution, including current
   failure seeds and recovery attempts.
3. **Snapshot branch dataset:** the policy-improvement supervision. From one snapshot,
   generate stock + N-sampled candidates, restore the identical sim state, execute each
   first 10-action block, and record next state / predicates / progress. Continue a
   stratified subset with π0.5 to termination for success and remaining-time targets.

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

Default family:

```python
L = lw * L_world + lt * L_task + lp * L_pred + lv * L_value \
    + lr * L_rank + lpara * L_para + la * L_anchor
```

Not every candidate implements every term: Candidate C uses `L_value + L_rank +
L_para`; A has no separate `L_world/L_task`; B exposes both. Loss weights are tuned on
held-out transition/ranking metrics, never final closed-loop test SR.

Avoid trivial task-code success:

- no raw-language/task embedding bypass into the main heads
- report a task-only language-prior scorer
- measure dynamic state change, not only `G_t^τ ≠ G_t^τ'`
- require the same-state task swap to change candidate outcomes correctly

---

## 11. Architecture-selection experiments

### 11.1 Feature / identity controls (before full segment extraction)

- Correct joint-PCA centering and report explained variance.
- Same frame prompts: canonical t0 / t0 paraphrases / same-scene t1 / unrelated t2 /
  constant. Report quantiles and tail fractions, not mean only.
- Compare shift-vector direction as well as shift magnitude maps.
- World/task probe ladder; target-object vs distractor/background selectivity.
- Task normalizer: held-out paraphrase classification and exact state/score consistency.

### 11.2 Offline structure comparison

Train A / B / C on identical snapshot splits and approximately matched trainable
parameter / compute budgets. Primary metrics:

- `k = 1..5` decision-step transition error
- predicate/progress F1 or AUROC; remaining-time error; value calibration
- within-snapshot pairwise rank accuracy, NDCG, top-1 regret
- held-out candidate noise and init-state generalization
- same-scene t0/t1 task intervention; paraphrase invariance
- latent effective rank / per-dim std / cosine-to-init collapse diagnostics

Only the top two policy scorers enter expensive closed-loop comparison. Latent
self-prediction alone cannot promote a structure.

### 11.3 Closed-loop Phase-1 comparison

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
- Shared-world tap: SigLIP pre-trunk vs constant-prompt last-layer.
- Task tap: canonical-prompt last-layer vs task queries over shared-world tokens.
- Whether one real/canonical prefix pass is sufficient for both policy and WM.
- `M_world / M_task`, recurrence form, and EMA vs VICReg target.
- Decision rollout `K = 1 / 3 / 5`; action-block encoder structure.
- Predicate/progress supervision granularity and training-only privileged pose labels.
- Branch count, continuation fraction, σ-perturbation proposal tier.
- Ensemble / uncertainty method and conservative intervention threshold.
- Final paired episode count and significance criterion.

---

## 13. Immediate order of work

1. Same-scene/paraphrase feature controls + corrected PCA; finish the world/task probe
   ladder and resolve only the feature-input decisions supported by it.
2. Formal N=16 diversity + simulator end-state dispersion on stratified snapshots;
   lock proposal-pool composition.
3. Freeze decision-rate segment / branch schema and collect a small shared probe set.
4. Implement Candidate C as the direct-ranking floor and Candidate A as the minimal
   latent-dynamics baseline.
5. Implement Candidate B on the same interface/data; run the offline selection table.
6. Closed-loop conservative reranking for the top two, then lock and run the paired
   Phase-1 evaluation once.
