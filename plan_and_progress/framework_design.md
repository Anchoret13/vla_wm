# pi05-lcwm — Framework Design v0.3

Living design document for a language-conditioned predictive state integrated
into π0.5. Structure candidates stay comparable under one data / interface /
evaluation contract; experiments promote, modify, or delete them here.
**[DEC]** = open decision with current default. **[EST]** = measure rather than
assume.

**2026-07-23 active direction:** directly modify π0.5's flow-matching action
distribution with a persistent LC state. Frozen-policy reranking and the earlier
Candidate-A diagnostics remain useful controls, but neither is the main Phase-1
system nor a prerequisite for attempting it.

## FIRST PRINCIPLE

> **The purpose of pi05-lcwm is to improve the VLA through a learned
> understanding of real-world dynamics.**

Language-conditioned state, long-horizon tasks, branch collection, flow
fine-tuning, and reranking are mechanisms or tests—not independent objectives.
Every promoted experiment must connect to the following causal chain:

```text
action intervention at a fixed real state
                  ↓
observed difference in the physical future
                  ↓
action-conditioned predictive state learns that effect
                  ↓
the learned dynamics information influences π0.5 actions
                  ↓
closed-loop VLA performance improves
```

This creates two non-interchangeable success bars:

1. **Engineering success:** improve full-prompt chain Q/SR without destroying
   standard LIBERO-10 competence.
2. **Scientific success:** show that the gain uses learned action-conditioned
   dynamics, rather than only task/phase lookup, extra policy capacity, prompt
   routing, branch imitation, or best-of-N search.

A result may clear the engineering bar without yet clearing the scientific bar.
Such a result is retained, but it is labeled policy improvement—not LCWM
evidence—until held-out controlled-effect prediction and matched no-dynamics
controls establish attribution.

---

## 0. Objective and Phase-1 win condition (updated 2026-07-23)

Language acts as a **task identifier**, not as unconstrained text state.
Conceptually, paraphrases define one canonical task equivalence class:

```text
τ = g(language)                  # canonical task identity
z_t^τ = F(o_≤t, a_<t, τ)        # task-conditioned predictive state
```

LC-Flow v0 does not insert an oracle hard normalizer into this path. It consumes
the real prompt and learns the equivalence through shared outcome and
paraphrase-consistency supervision; `τ` states the desired semantics.

Required representation behavior:

- same physical history + paraphrases of the same task → the **same** task state
- same physical history + different tasks → **different, decision-relevant** task states
- latent difference is not sufficient by itself: changing `τ` must change future
  prediction, task-relevant predicates / progress, and eventually behavior in the
  correct way

### Objective hierarchy

1. **Core research objective:** improve π0.5 through a language-conditioned,
   action-conditioned understanding of real-world dynamics. The predictive state
   should retain the physical/history distinctions that matter for the task,
   forecast the consequences of actions, and alter the action distribution in
   the useful direction.
2. **Representation criterion:** under identical observation/action history,
   paraphrases of one task should induce equivalent predictive outcomes, while
   different tasks should differ where their future-relevant semantics differ.
   Outcome prediction and behavior arbitrate this property; raw latent distance
   does not.
3. **Dynamics criterion:** from a fixed history, different executed action blocks
   must yield correctly predicted differences in physical and task-relevant
   futures, including held-out effectful branches.
4. **Phase-1 instrumental test:** improve stock π0.5 on the self-built
   LoHo-inspired chain3/4/5 domain under the full composite instruction. The main
   result is a modified `N = 1` policy, not an external best-of-N selector.
5. **Retention criterion:** preserve the policy's existing standard LIBERO-10
   competence. LIBERO-10 is now a regression test rather than the main headroom
   benchmark.

The physical world remains language-independent. Language changes which physical
distinctions the finite-capacity latent retains and how the same physical future
is evaluated:

\[
x_{t+1}\sim P(x_{t+1}\mid x_t,a_t),
\qquad
z_t^\ell=\phi(H_t,\ell),
\qquad
\hat z_{t+1}^\ell=\hat T(z_t^\ell,a_t).
\]

Although `T` does not receive a second explicit language token in the active
implementation, the induced latent transition is language-conditioned because
its input state is `z_t^\ell`. This is deliberately different from Candidate A,
where oracle task ID was also bypassed directly into the transition.

Phase-1 primary interface:

```text
full instruction + observation + executed history
                         │
               persistent LC predictive state
                         │
          π0.5 flow-matching action expert (modified)
                         │
                one 50-action chunk, N = 1
                         │
                 execute first 10, reobserve
```

Snapshot branches still matter, but their role is now:

1. provide action variation at the same history for outcome-grounded dynamics;
2. provide language crossing / full-goal relabeling on one physical continuation;
3. identify positive-progress actions that directly supervise the flow action
   expert;
4. optionally support reranking as a bootstrap, headroom measurement, and
   secondary deployment interface.

A direct-Q model can satisfy purpose 4 without learning a reusable predictive
state; Candidate C remains the corresponding control.

Phase-1 success reports paired full-prompt ΔSR and ΔQ, per-atom completion
timelines, chain-length profile, failure localization, and standard LIBERO-10
retention. Best-of-N rescue/harm/intervention metrics are reported only for the
secondary reranking condition.

---

## 1. Locked common contract

- Main development/evaluation domain: self-built LoHo-inspired chain3/4/5 on
  `LIVING_ROOM_SCENE2`; standard LIBERO-10 is the retention suite.
- Stock baseline: frozen `lerobot/pi05_libero_finetuned`.
- Main system: LC-Flow π0.5, with the PrefixVLM initially frozen and the LC
  state/update/transition/outcome path trainable. The action distribution is
  changed through a trainable state-conditioning path; later action-expert
  unfreezing/PEFT is an explicit escalation, not assumed in v0.
- Final-policy input: the full natural-language composite instruction throughout
  the episode. No oracle predicate bits, current subgoal, task ID, or decomposition.
- Policy chunk = 50 environment actions; commitment / decision stride `c = 10`.
  Only the first 10 actions receive a branch outcome unless a continuation was
  actually executed.
- Primary evaluation uses one sampled chunk per decision (`N = 1`). `N = 8/16`
  candidate pools are allowed for data collection, oracle headroom, and secondary
  reranking.
- Main system operations:

```python
H, kv = prefix_vlm(observation, full_instruction)
z_prior = transition(z_prev, encode(executed_actions_prev[:10]))
z = observation_update(z_prior, H)
action_chunk = flow_action_expert(H, kv, lc_state=z)     # N = 1 primary
predicted_outcome = outcome_head(transition(z, encode(action_chunk[:10])))
```

- The language-conditioned prefix forward used to update `z` must reuse the same
  KV cache used for action sampling; v0 does not add a second 3B forward.
- Physical outcome heads must not change under same-history/action language swaps
  beyond tolerance. Task predicates/progress/return may change, but task
  information must pass through the learned LC state in the primary model.
- Branch siblings from one simulator snapshot stay in one data split.
- Save environment state/fingerprint, proposal provenance, exact policy flow noise,
  executed action count, pre-reset terminal labels, and all paired evaluation seeds.

---

## 2. Task normalizer

The active model consumes the real natural-language instruction through π0.5's
existing PrefixVLM. It does **not** use oracle task ID as the primary task
interface:

```text
image + proprio + full language
              │
       π0.5 prefix hidden H_t^ℓ
              │
       LC observation update U
              │
              z_t^ℓ
```

Language still acts semantically like a task identifier: paraphrases of the same
goal should produce equivalent outcome predictions and behavior, while different
goals should select different task-relevant state. We do not require literal
vector equality, because prompt surface form may occupy nuisance dimensions.

Conditioning controls:

- `REAL-LANG`: primary full-prompt interface;
- `PARAPHRASE`: same-goal invariance/equivalence test;
- `TASKID`: oracle canonical identity, diagnostic upper bound only;
- `FREE`: remove task information, negative control;
- `SHUFFLED`: wrong instruction on the same physical history, causal control;
- `READOUT-ONLY`: language-free state/transition with language-conditioned
  outcome/action readout, matched-capacity scientific baseline.

The readout-only model is not a route-selection gate. It tests the eventual claim
that putting language inside the predictive state is more useful at fixed capacity;
the LC-Flow vertical slice is attempted first.

---

## 3. Prefix feature interface

### Active v0 binding

LC-Flow v0 uses the real-prompt last hidden state from the same π0.5 prefix
forward that constructs the action sampler's KV cache:

```text
real image + full instruction
              │
        frozen PrefixVLM
              ├── H_t^ℓ ── LC observation update
              └── KV cache ── π0.5 action expert
```

This is a compute/interface decision, not a claim that the last layer has already
been certified as an LC world state. `H_t^\ell` is an input feature; the recurrent
`z_t^\ell` is the learned predictive state. Reusing the same forward avoids a
constant-prompt second trunk pass and lets the policy and model see exactly the
same language-conditioned scene representation.

SigLIP-only, constant-prompt last-layer, and fused features remain ablations if v0
fails to retain physical detail. They do not gate implementation. The critical
same-scene language control remains LIBERO-10 t0 vs t1
(`LIVING_ROOM_SCENE2`), together with same-goal paraphrases.

### Historical tap evidence

The 2026-07-19 visualization established a heavy-tailed, spatially nonuniform
prompt effect in last-layer image tokens, but did not separate generic prompt
sensitivity from task-specific modulation. Shift-map magnitude correlations
(`r = 0.38–0.69`) showed a shared prompt-sensitive subset, not proof that the
remaining signal was content-free.

That evidence motivated separate SigLIP/world and last-layer/task streams in the
older A/B candidates. The split is retained as an ablation, not as the active v0
architecture.

Evidence log:

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
  canonical last-layer stream as a provisional `H_task` input. Projection is now
  visually calibrated and v4 makes the newly relevant target group largest at all
  three sampled timesteps, but object-role selectivity remains OPEN: the only
  timestep with valid distractors has a mild 0.316 vs 0.204 contrast, and disjoint
  object-sized masks / more states are still missing. This remains a bounded
  diagnostic rather than a training gate.
- Consequence: choose the hidden-state input while training the main dynamics/policy
  system rather than through more standalone readout gates. Candidate-D pose
  supervision remains a training-only ablation; the light-probe failure does not
  establish that it is necessary.
- 2026-07-22 Candidate-A pilot (`2026-07-22.md`): SigLIP is substantially more
  sensitive to cross-batch action shuffling under the current co-adaptive latent
  metric (1.06/1.10 vs real-LL 0.25/0.23), so it becomes the provisional world-input
  default inside that diagnostic. Certification remained OPEN because the same
  `test` split was used for selection and objective tuning, future
  physical/semantic heads were not evaluated on predicted carries, and
  action-history/task-injection shortcuts remained uncontrolled.

---

## 3.1 Active primary framework — LC-Flow π0.5

### Persistent predictive state

At decision boundary `t`:

\[
H_t^\ell = \operatorname{PrefixVLM}(o_t,\ell),
\]

\[
\bar z_t^\ell =
T_\theta\!\left(
z_{t-1}^\ell,E_a(a^{\mathrm{exec}}_{t-1,0:10})
\right),
\qquad
z_t^\ell =
U_\theta(\bar z_t^\ell,H_t^\ell).
\]

```text
z_(t-1)^ℓ + previous executed 10 actions
                     │
           action-conditioned prior T
                     │
                 z̄_t^ℓ
                     │
real-prompt H_t^ℓ ── observation correction U
                     │
                  z_t^ℓ
             ┌───────┴────────┐
       outcome rollout    π0.5 flow expert
```

`z_t^\ell` persists across the episode and resets only when the environment
resets. `T` receives no explicit task embedding: language has already selected
the predictive state that it advances. `U` performs posterior correction after
the new observation. This prior/posterior split makes action history functional
without giving the target step a raw previous-action shortcut.

**[DEC] v0 state scale:** start with `M = 4` state tokens and `d_z = 384` or the
smallest width convenient for projection into the 1024-wide action expert. Scale
only if outcome learning is capacity-limited.

### Direct action-expert integration

π0.5's action expert is a flow-matching Gemma module whose AdaRMS conditioning
currently receives the flow-time embedding. LC-Flow changes:

\[
c_{\mathrm{expert}}
=
c_{\mathrm{time}}
+
W_z\operatorname{Pool}(z_t^\ell),
\]

and therefore:

\[
v_\psi(x_\tau,\tau\mid H_t^\ell)
\quad\longrightarrow\quad
v_\psi(x_\tau,\tau\mid H_t^\ell,z_t^\ell).
\]

The final layer of `W_z` is zero-initialized. At initialization the extension is
an exact stock-policy no-op; training can then change the flow field without
changing token positions, prefix length, or cache behavior. Suffix LC tokens are
a later alternative if global AdaRMS modulation is insufficient.

The first trainable boundary is:

- freeze the 2B PrefixVLM;
- train `E_a`, `T`, `U`, outcome heads, and `W_z`;
- optionally unfreeze/PEFT the 300M action expert only after measuring whether
  the conditioning adapter has enough control.

Even with the action expert initially frozen, the trainable state projection
changes its normalized activations and therefore its output distribution.

### Outcome-grounded transition

For candidate branch `i`:

\[
\hat z_{t+1}^{\ell,i}
=
T_\theta(z_t^\ell,E_a(a^{(i)}_{0:10})),
\]

\[
D_\theta(\hat z_{t+1}^{\ell,i})
=
\left(
\widehat{\Delta W}^{\,i},
\widehat{\Delta Y}^{\,\ell,i},
\widehat{\Delta \mathrm{Prog}}^{\,\ell,i},
\widehat R^{\,\ell,i}
\right).
\]

`ΔW` is an instruction-independent physical-effect block (proprio, object
displacement, and fixed physical predicates where available). `ΔY`, progress,
and return are evaluated under the full task. On the same `(history, action)`,
instruction swaps should preserve `ΔW` while changing task relevance correctly.

The state is not allowed to collapse into task progress alone. Progress memory
can help LoHo behavior, but the project requires `T(z,a)` to predict how the
executed action changes the real scene. Physical controlled effects are therefore
core supervision and held-out evaluation, not an optional interpretability head.

Latent self-prediction is an auxiliary training constraint, never the
certification metric. The model is judged by controlled outcome effects and by
the action distribution it induces.

**Status: active primary implementation.**

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

Strengths: it established the end-to-end data/training pipeline and exposed
action/history shortcuts. Risks realized in the pilot: oracle task ID FiLM enters
the encoder, and the same embedding enters `Transition` directly. Therefore
“task-swap changes the latent” is partly guaranteed by construction. Successful
expert trajectories also do not identify visual action-conditioned effects.

Status: **completed diagnostic instrument; retired as a final LCWM candidate.**
Keep it for smoke tests and historical comparisons, not certification sweeps.

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

Status: **structured factorization ablation.** Conceptually attractive and still
available if LC-Flow's monolithic state corrupts shared physics or wastes capacity,
but it is no longer a gate before the primary vertical slice.

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

Interpretation rule: Candidate C is not the target framework. It measures how much
of best-of-N improvement can be obtained without a reusable predictive transition.
It cannot decide whether the direct `N = 1` LC-Flow policy should be attempted.

Status: **secondary reranking / oracle-headroom baseline.** Implement after or
alongside the first LC-Flow vertical slice as resources allow.

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
surface before the LC-Flow vertical slice is established.

Status: **deferred**. Promote only if LC-Flow outcome errors show that an
unstructured carry cannot retain object identity or predicate state.

---

## 8. Decision-rate dynamics and control interfaces

One learned transition corresponds to the executed commitment `c = 10`, not one
MuJoCo step. π0.5 still represents a 50-step chunk internally:

```text
A = [a_0:10] [a_10:20] [a_20:30] [a_30:40] [a_40:50]
     executed + labeled       unexecuted at this decision
```

### Primary interface — learned `N = 1` action distribution

LC-Flow samples one chunk conditioned on `(H_t^\ell,z_t^\ell)`, executes the
first 10 actions, observes, updates its persistent state, and replans. No external
selector is required. This is the behavior the main experiment must improve.

### Secondary interface — best-of-N bootstrap / reranking

For collection or secondary evaluation, sample `N = 8/16` chunks and predict:

```python
score_i = predicted_progress_gain_i + gamma * predicted_short_return_i
```

Candidate `0` is the paired stock/noise reference. Reranking provides:

- a search-based teacher for branch-weighted flow learning;
- an oracle-headroom check—does a useful action exist in π0.5's support?;
- a fallback policy interface if direct distillation is initially weak;
- Candidate-C comparison against explicit dynamics.

Reranking improvement alone is not the Phase-1 claim. The final comparison
contains a full-prompt, no-oracle, `N = 1` LC-Flow arm.

---

## 9. Crossed branch data

### 9.1 Sources

1. **Original LIBERO demonstrations:** preserve generic manipulation through the
   normal full 50-step π0.5 flow-matching loss. They are rehearsal data, not
   sufficient evidence for action-conditioned dynamics.
2. **Stock π0.5 chained rollouts:** locate late-chain stall/recovery snapshots on
   the actual failure distribution.
3. **Snapshot branches:** restore one history, execute different first-10 action
   blocks, and relabel the resulting physical continuation under compatible goals.
4. **Improved-policy rollouts:** one DAgger-style recollection after the first
   closed-loop policy exposes its own state distribution.

### 9.2 Proposal pool

At each useful stall/recovery snapshot, take the union of action chunks proposed
under:

- the full composite instruction;
- the exact in-distribution pair instruction for the scene;
- compatible atomic remaining-goal instructions;
- multiple explicitly saved flow-noise samples.

The proposal prompt only expands action support. Every branch is evaluated under
the full composite evaluation goal, so a pair/atomic-prompt action with positive
full-task advantage becomes a teacher for the **full-instruction** LC state.
`proposal_prompt_id` is stored as provenance and never used as the training
target's task identity.

### 9.3 Storage contract

Physical continuation, goal specification, and semantic relabel remain
decoupled:

```text
PhysicalTransition
  snapshot_id, source_episode_id, history, state_flat
  branch_id, proposal_prompt_id, flow_noise_id
  proposed_action_50, executed_action_10
  next_observation, next_state_flat, terminal_before_reset

GoalSpec
  goal_id, canonical_instruction, paraphrases
  BDDL predicates, compatible scenes

SemanticRelabel
  snapshot_id, branch_id, evaluation_goal_id
  predicates_before/after, progress_delta
  reward, success, failure flags, optional continuation return
```

Dataset indices are `(snapshot_id, branch_id, evaluation_goal_id,
text_variant_id)`. One physical branch is not duplicated on disk for every
language label. All sibling branches and relabels from a snapshot share one
train/validation/test group.

### 9.4 Initial collection scale

Start with roughly 128–256 **useful** late-chain snapshots rather than delaying
the model for a large corpus. Contact/object-moving/recovery states are
preferable to free-space motion. During collection, repeat a small fraction of
identical `(snapshot, action)` branches to log restore noise and discard invalid
siblings. This is data quality control inside the main collection, not a
separate architecture gate.

Measure end-state/predicate dispersion, not action-space L2 alone. Fixed
environment seeds do not fix π0.5 flow noise, so proposal noise is explicitly
seeded and stored.

---

## 10. Joint objectives

### 10.1 Predictive-state / outcome losses

```text
L_phys     : instruction-independent proprio/object/fixed-predicate effects
L_sem      : task predicate flips and progress change
L_return   : short continuation return / success where actually observed
L_self     : target-state latent consistency, auxiliary only
L_shared   : same physical branch predicts the same physical future across language
L_para     : same-task paraphrase outcome/action consistency
L_anchor   : optional privileged pose/predicate reconstruction, training only
```

Controlled effects are preferred to absolute state priors:

\[
\Delta y_i^\ell =
y^\ell(s'_i)-y^\ell(s),
\qquad
\mathcal L_{\mathrm{outcome}}
=
\lambda_{\mathrm{phys}}\mathcal L_{\mathrm{phys}}
+
\lambda_{\mathrm{sem}}\mathcal L_{\mathrm{sem}}
+
\lambda_{\mathrm{return}}\mathcal L_{\mathrm{return}}.
\]

`L_phys + L_sem` is the dynamics-understanding core. A policy trained with
`L_branch-FM` but without controlled-effect learning is a matched behavior-learning
baseline, not a reduced version that can support the same scientific claim.

Overall predicate F1 is not a sufficient metric because unchanged predicates
dominate. Report flip AUPRC/precision/recall, progress-effect error, object
displacement error, and results restricted to effectful branches.

### 10.2 Branch-weighted flow matching

All branches train the state transition and outcome heads. Only
positive-progress or within-snapshot superior branches are action targets:

\[
\mathcal L_{\mathrm{branch\text{-}FM}}
=
w_i
\left\|
M_{0:10}\odot
\left[
v_\psi(x_\tau,\tau,H_t^\ell,z_t^\ell)
-
(\epsilon-a_i)
\right]
\right\|^2.
\]

`w_i` is derived from full-goal branch advantage and is nonnegative. The
first-10 mask is mandatory because the other forty proposed actions were not
executed under this label. Negative branches remain valuable outcome examples
but are not trained as reverse imitation. Original demonstrations keep the
unmasked full-chunk flow objective:

\[
\mathcal L_{\mathrm{joint}}
=
\mathcal L_{\mathrm{outcome}}
+
\lambda_{\mathrm{self}}\mathcal L_{\mathrm{self}}
+
\lambda_{\mathrm{shared}}\mathcal L_{\mathrm{shared}}
+
\lambda_{\mathrm{para}}\mathcal L_{\mathrm{para}}
+
\lambda_{\mathrm{branch}}\mathcal L_{\mathrm{branch\text{-}FM}}
+
\lambda_{\mathrm{demo}}\mathcal L_{\mathrm{demo\text{-}FM}}.
\]

LeRobot's π0.5 loss already exposes per-sample flow error; implementation still
needs a first-10 action mask and the full-goal branch-quality weighter.

### 10.3 Shortcut controls

- no oracle task ID or proposal prompt bypass into the primary transition/head;
- predict changes/effects, not only absolute task/phase labels;
- same physical continuation is reused across compatible instruction labels;
- paraphrases must preserve outcome/action behavior;
- instruction swaps may change task relevance but not predicted shared physics;
- latent distance/effective rank are monitoring signals, never primary evidence.

---

## 11. Experimental sequence

### 11.1 Primary vertical slice

1. Repair pre-reset terminal capture so the evaluation target is trustworthy.
2. Implement persistent `LCState`, `T`, `U`, outcome heads, and zero-init AdaRMS
   injection.
3. Collect the first late-chain branch set and immediately joint-train the
   predictive/outcome and masked flow objectives.
4. From the same checkpoint, measure held-out controlled physical/task effects
   and run stock π0.5 vs LC-Flow π0.5 on chain3 with the full prompt and
   `N = 1`.
5. If paired Q/SR and dynamics prediction show signal, extend to chain4/5;
   otherwise use the already
   trained outcome model and oracle best-of-N to distinguish missing action
   support from failed state/action learning.
6. Perform at most one improved-policy recollection/retrain pass before
   revisiting architecture scale.
7. Check standard LIBERO-10 retention.

The vertical slice is the focus experiment. Feature probes, Candidate-A
certification, A-vs-B comparison, and exhaustive branch diagnostics are not
preconditions.

The vertical slice deliberately produces both model evidence and behavior
evidence. We do not postpone all dynamics evaluation until after optimizing the
policy, because that would make a positive result impossible to attribute.

### 11.2 Matched explanatory controls

Run the smallest set needed to explain the main result:

| arm | question |
|---|---|
| stock π0.5, `N = 1` | baseline full-prompt policy |
| corrected oracle decomp | how much explicit goal/progress routing can help |
| branch-FM, no persistent state | are better actions alone sufficient? |
| LC-Flow, no outcome/world loss | is recurrence only acting as extra policy capacity? |
| LC-Flow, `N = 1` | primary method |
| LC-Flow + rerank, `N = 8/16` | proposal-support/headroom and bootstrap value |
| readout-only matched capacity | does language need to enter predictive state? |

Candidate A is a smoke-test/chronology arm. Candidate B is promoted only if
shared-physics or capacity results motivate explicit factorization. Candidate C
is the direct-Q reranking baseline.

The branch-FM/no-state and LC-Flow/no-world-loss arms are required before the
final scientific claim. If either matches LC-Flow, the result says the branch
actions improved π0.5, but does not yet say real-world dynamics understanding
caused the improvement.

### 11.3 Evaluation contract

- full composite instruction remains fixed for the main policy episode;
- no oracle predicate bits, current stage, or decomposition;
- held-out source-episode/snapshot groups and held-out rollout seeds;
- exact policy-noise pairing where the policy architectures permit it;
- report paired Q-score, SR, per-atom completion time, stall duration, chain
  horizon profile, and confidence intervals;
- report held-out action-effect prediction on proprio, objects/contact,
  predicate flips, and task progress from the same training checkpoint;
- report standard LIBERO-10 SR/steps as a retention check;
- separate primary `N = 1` from secondary best-of-N results.

The first chain3 run is a development result, not a final significance claim.
Final seed counts and acceptance threshold are locked after measuring runtime
and variance, before the final evaluation schedule is opened.

Final reporting separates:

- **policy result:** did the VLA improve?;
- **model result:** did it predict controlled real-world effects?;
- **attribution result:** did dynamics-grounded LC-Flow outperform
  capacity/data-matched no-dynamics policy learning?

---

## 12. Current evidence and interpretation ledger

### Candidate A

- Candidate A is a successful pipeline diagnostic, not a certified LCWM.
- Oracle task-ID FiLM plus direct task embedding into `Transition` structurally
  produces task-dependent latents.
- Successful expert trajectories bind task, scene, phase, and action; they do
  not identify visual action-conditioned dynamics.
- Removing raw `a_prev` removed a major action-sensitivity shortcut.
- Task information helps predict task-specific predicates, but that observation
  is also compatible with language-conditioned readouts.

Therefore further Candidate-A rank/variance/seed sweeps do not precede LC-Flow.

### Chained-domain correction and failure target

The 2026-07-22 evaluator reread predicates after auto-reset and hid two decomp
successes. Trajectory-audited provisional correction:

| task | full SR / Q | decomp SR / Q |
|---|---:|---:|
| chain3 | 0% / 0.67 | 20% / 0.67 |
| chain4 | 0% / 0.55 | 20% / 0.75 |
| chain5 | 0% / 0.44 | 0% / 0.60 |

**OPEN EVAL-001:** LeRobot `LiberoEnv.step()` calls `reset()` after setting
`info["is_success"]`; `run_chain_episode` ignores that field and queries
`predicate_bits(env, atoms)` from the reset simulator at its terminal paths.
The fix must preserve terminal bits/state before reset, consume that immutable
record in the runner, add a success-serialization regression test, and rerun the
affected cells. The raw artifact is still contaminated. What survives:

- full prompt is 0/15;
- all 15 full runs complete alphabet soup and 14/15 complete tomato sauce;
- butter completes twice, cream cheese once, and milk never;
- all chain3 full runs stall at 2/3 for roughly 400+ steps;
- decomp sometimes rescues, so task/progress routing matters;
- late-chain decomp still misgrounds the requested object (cream often induces
  butter), so routing text alone is not reliable.

Generic manipulation is not the first bottleneck: standard LIBERO-10 same-scene
t0/t1 are 10/10 each; atomic cream-cheese and milk tasks are 10/10, and butter
is 9/10. The first intervention should target persistent task/progress state and
its effect on action generation, not a generic low-level controller.

### Scientific interpretation

The central hypothesis remains fixed: language belongs in the learned predictive
state/dynamics. The readout-only system is the principal matched alternative, not
a go/no-go selector. Crossed branch data exists to prevent the LC system from
winning through task–trajectory correlation. Policy improvement tests whether
the learned distinctions are decision-relevant. The north star is not merely a
better latent or a better chain score: it is a better VLA **because** the VLA has
learned action-conditioned real-world dynamics.

---

## 13. Active known-problem ledger (2026-07-23)

| status | problem | consequence for the framework |
|---|---|---|
| confirmed | evaluator read predicates after auto-reset | repair pre-reset terminal capture before any new policy comparison |
| confirmed | raw chain JSON stores reset-state Q/success for two terminal episodes | preserve it as contaminated provenance; write repaired results to a versioned artifact |
| confirmed | chain5 executed at 990 rather than planned 1100 steps | treat 990 as the recorded executed horizon; do not claim the preregistered cap was run |
| confirmed | environment seed does not fix flow-matching noise | save/pair policy-noise seeds or tensors |
| confirmed | Candidate A has oracle-FiLM and direct task-to-transition shortcuts | diagnostic instrument only; no further certification sweeps |
| confirmed | old `a_prev` target path inflated action sensitivity | separate predictive prior from posterior correction |
| confirmed | expert demos permit task/phase/no-vision shortcuts | crossed fixed-snapshot branches are the main learning data |
| confirmed | old task swaps did not always recompute prompt-conditioned VLA features | every language arm must use its matching real-prompt prefix |
| open | contact-heavy replay has shown eight success disagreements | repeat identical branches and reject snapshots whose restore noise dominates |
| design-critical | only candidate actions 0:10 are executed but the proposal has length 50 | mask branch flow loss to 0:10; never propagate the label to unexecuted actions |
| design-critical | failed branches are informative but are not desirable action targets | train transition/outcome heads on them; do not use negative-weight imitation |
| open implementation | π0.5 training lacks branch-quality weighting and a first-10 mask | implement both before branch outcomes update the action expert |
| design-critical | pair/atomic prompts expand proposal support | proposal identity is provenance only; full composite goal supplies training labels |
| open | LC state may copy a static task code | outcome effects and behavior, not latent separation, are the acceptance criteria |
| open | language conditioning may corrupt predicted shared physics | enforce/report cross-language physical-effect consistency |
| open | zero-init AdaRMS adapter may have insufficient action control | measure distribution shift and escalate to PEFT/unfreezing only if needed |
| claim constraint | best-of-N may improve selection without improving the action distribution | full-prompt direct `N = 1` remains the primary endpoint |
| open implementation | branch learning may forget stock skills or a no-op extension may be wired incorrectly | demo rehearsal, LIBERO-10 retention, and fixed-noise pre-training equivalence test |
| scope | one scene family and five seeds/cell | treat initial chain3 as development; use held-out seeds and later cross-scene confirmation |

Long-continuation return is also policy-dependent. The first target is therefore
the observed one-block physical/semantic effect; continuation value is auxiliary
and used only where its generating policy is recorded.

The dated log `2026-07-23.md` records consequences and required actions in full.
Only the evaluator repair blocks scoring; the remaining items are handled inside
the main implementation/data/evaluation contract.

---

## 14. Current open-decision ledger

- LC state scale (`M`, `d_z`) and pooling used by `W_z`.
- Minimum controlled-effect and attribution result required for the final LCWM
  claim.
- Add LC conditioning to every action-expert AdaRMS layer or a selected subset.
- Whether the zero-init adapter alone moves the policy sufficiently; criterion
  for PEFT/unfreezing the action expert.
- Exact action-block encoder and whether a second/third latent transition uses
  predicted or observed correction.
- Branch-advantage definition, temperature/clipping, and demo/branch mixture.
- Initial proposal count and allocation among full/pair/atomic prompt sources.
- Which compatible alternative goals/paraphrases form the crossed language set.
- Fraction of branches receiving longer continuation-return labels.
- Physical-effect target: proprio + object pose + fixed predicate vocabulary.
- Whether explicit factorization (Candidate B) is needed after the monolithic
  LC-Flow result.
- Final evaluation seed count, paired significance rule, and acceptable
  LIBERO-10 regression.

These decisions are made from the first vertical slice and its learning curves,
not from another standalone diagnostic program.

---

## 15. Immediate order of work

1. Fix and rerun terminal evaluation capture.
2. Implement recurrent LC prior/posterior state and outcome heads.
3. Inject the zero-initialized state projection into the π0.5 action expert.
4. Collect roughly 128–256 useful late-chain snapshots with saved flow noise,
   full-goal relabels, and first-10 outcomes.
5. Joint-train outcome-grounded dynamics plus branch/demo flow matching.
6. Evaluate held-out controlled effects and run chain3, full prompt, no oracle,
   `N = 1`, from the same checkpoint.
7. Extend to chain4/5; use best-of-N only as bootstrap/headroom/secondary result.
8. Optionally recollect once under the improved policy and retrain.
9. Run standard LIBERO-10 retention and the matched no-dynamics attribution
   ablations.

This order supersedes the 2026-07-22 sequence “finish Candidate A → A-vs-B →
branch ranking → frozen-policy reranking.”

---

## 16. Scope / provenance

The chained domain is a **self-built LoHo-inspired exam**, not official
LIBERO-LoHo. Its BDDL tasks extend packaged `LIVING_ROOM_SCENE2` goals and its
evaluator/decomp controller live in this repository. Results are not directly
comparable to the unreleased benchmark or its paper numbers.
