# H7 execution record — COMPLETE

**Final status.** The initial section was written during a temporary
tool-platform outage; the work subsequently compiled and ran through H7.5.
The chronology below is preserved as contemporaneous evidence, so statements
such as `Next` describe that point in the run rather than current work. No
v0.5 arm was promoted. The active replacement contract is
`2026-07-30.md`.

## Initial outage snapshot (historical; subsequently compiled and run)

1. `scripts/analyze_behavior_factorial.py` — H7.0(d): tracked
   deterministic analysis (per-task SR/Q/per-object/D/A_D, registered
   S_prefix contrasts with fixed-seed 10k cluster bootstrap CIs, exact
   sign-flip, Holm, config block; the arbitrary-order caveat is embedded
   in the artifact itself).
2. `bddl/libero_loho_public_v1/loho_t2_basket3.bddl` — public T2
   reconstruction on the audited LIVING_ROOM_SCENE2 layout (regions
   byte-identical to the local chain3 base; language/goal/obj_of_interest
   per the public description).
3. `results/libero_loho_public_v1/task_source_manifest.json` — H6-P/H7.1
   provenance manifest: T2 status "reconstructed"; T1/T3/T4/T5
   "pending_asset_inventory" with named blockers; locked protocol block;
   Q semantics registered (unordered completed-subgoal fraction primary,
   ordered prefix as declared diagnostic — revisitable on parity
   evidence).
4. `lcwm/loho_public.py` — public-task protocol module: ordered-subgoal
   tracker (place/open/close via BDDL `_eval_predicate`; pick_up via the
   registered displacement>=0.02m AND lift>=0.03m criterion),
   `run_public_episode` with public Q, per-subgoal completion steps,
   first unresolved subgoal.

## Historical recovery queue (subsequently executed)

1. compile checks on the four files above;
2. H7.0(a) reconstruct + hash-verify the missing H4 manifest;
   (b) arm_state_mode into manifests/adapter metadata;
   (c) recompute current-arm action_shift with z=U(z0,h_t) [GPU];
3. LIBERO bddl asset inventory (chocolate_pudding / wooden_tray /
   salad_dressing / black_bowl / cabinet drawer scenes) → reconstruct
   T1/T3/T4/T5;
4. T2 one-seed stock smoke, then five-task smoke set;
5. commit + push everything above;
6. H7.2 bounded collection (20 sources / 160 branches).

## H7.1 smoke result (seed 1300, stock π0.5, full prompt)

| task | Q_public | completed | first unresolved |
|---|---|---|---|
| T1 drawer | 0.200 (1/5) | pudding picked | pick butter |
| T2 basket3 | 0.833 (5/6) | soup+sauce placed, butter picked | place butter |
| T3 tray | 0.667 (4/6) | cheese+butter placed | pick soup |
| T4 tray | 0.333 (2/6) | left bowl placed | pick dressing |
| T5 drawer+cabinet | 0.571 (4/7) | pudding in drawer, butter+bowl picked | place butter |

SR 0/5; mean Q_public 0.521. The paper's stock reference (Q=55.3%,
SR=6.4%) remains an external number — no protocol-parity claim — but the
observed magnitude is consistent with it. One evaluator defect found and
fixed before this run (predicate names must be lowercase; top_side regions
use On-semantics). H7.1 exit met: five valid stock episodes recorded in
evaluator_smoke.jsonl. Next per night plan: H7.0 (a)(b)(c) repairs, then
H7.2 overnight collection.

## H7.2 registration (operationalizations, written before implementation)

- Sources: per public task, 2 stock + 2 frozen v0.4 cur_wm rollouts
  (seeds registered: stock 1400/1410, cur_wm 1420/1430 per task) = 20
  source episodes; split by source episode at collection time.
- Snapshot rule (registered): (i) "first persistent failure" = first
  decision where the tracker's first-unresolved subgoal is unchanged for 5
  consecutive decisions; (ii) "late unresolved state" = the decision at
  80% of the episode if >= 1 subgoal is unresolved there (else the last
  unresolved decision). 2 per source = 40 snapshot groups.
- Branches: deterministic N=4 pool (candidate 0 = the source policy's own
  chunk via recorded noise; 3 fresh seeded pi0.5 samples), each executed
  once (10 actions) = 160 branches; restore/replay provenance kept.
- Continuations: one bounded 100-action stock continuation per branch =
  160; labels: public subgoal flips, damage, reward (subgoal delta),
  dQ_public, terminal success, continuation Q. Immediate object distance
  is never an advantage.
- Language: canonical instruction + ONE registered paraphrase per task
  (fixed texts in the manifest) + compatible same-scene relabels:
  T1<->T5 (shared KITCHEN_SCENE10 objects; cross-evaluable) and, for every
  task, its per-object atomic subgoal goals. Physical outcomes shared
  across labels.

## H7.2 result: collection COMPLETE (20/20 sources)

40 snapshots / 160 branches / 160 bounded continuations, per the
registered contract; splits fixed at collection (1400/1420 train,
1410/1430 dev). Positive-continuation support (dQ_public > 0):
T1 12/32, T2 14/32, T3 16/32, T4 16/32, T5 12/32 — **70/160 (43.8%)**
overall, with every task and both source policies contributing. Contrast
with the local chain data (0/52 successful continuations): public-task
value/reward targets now have real positive support. One source
(T1 cur_wm s1420) has 0/8 — recorded, not resampled.
Artifacts: data/…/libero_loho_public_v1/*.pt + collection_manifest.json.
Next: H7.3 fixed v0.5-gated world model.

## Feature recompute COMPLETE (20/20)

All sources replayed deterministically; stored candidate-0 chunks asserted
equal to the replayed source chunks on every snapshot (both stock and
cur_wm replay paths). Sidecars <source>.features.pt: full-prefix H_late +
mask, pooled SigLIP H_early (frame_features pattern), q/obj before and
after every branch. One SigLIP-capture defect found and fixed before the
successful run (per-slot validity is m[0], not mask.all()). v0.5 training
data surface is complete. Next: scripts/train_v05_wm.py (fixed objective,
seed 0, registered budget).

## H7.3 trainer scoping decisions (registered before implementation)

1. Sequential data for the first v0.5 run = the 18 cached DEMO episodes
   (tasks 0/1): H_late from seq_prefix_cache_v1, H_early from
   seq_libero_10_v2's stored pooled SigLIP, aligned by decision t values
   (asserted per episode). CHAIN sequential episodes are deferred from
   this first run (no H_early captured for them) — a recorded scoping
   choice, not a silent drop.
2. "Task-state prediction" is operationalized head-based (next public
   subgoal/predicate state, reward, ΔQ, value) — no EMA latent loss in the
   first run, consistent with the H7.3 grounded emphasis.
3. Crossed-language: paraphrase consistency = g-state + task-head
   agreement under canonical vs paraphrase H_late (captured in sidecars
   v2/v3). Shared-physics under compatible goals is enforced
   STRUCTURALLY (w never receives language), so no loss term is needed;
   compatible-goal g-states are unconstrained in run one (their
   task-relative labels are not yet captured) — recorded.
4. Branch-state w initialization: the first run initializes w at the
   snapshot from w0 + one physical update on snapshot features (no
   history unroll for public branches) — recorded limitation.
5. Value target for public branches = bounded continuation q_public
   (generating policy recorded per source; both stock and cur_wm sources
   included). Demo episodes contribute NO value targets (expert policy).
6. Budget (fixed): AdamW 3e-4 / wd 1e-4, 30 epochs, grad-norm 1.0,
   TBPTT 16 on demo episodes, source-balanced rotation (18 demo episodes
   + 40 public snapshots), seed 0, no sweeps.

## H7.3 result: v05_gated_wm training COMPLETE (fixed budget, no sweeps)

Sidecars v3 (20/20, + paraphrase & compatible-goal H_late) then one fixed
run, seed 0, 30 epochs: train surface = 14 demo episodes (t-alignment
asserted per episode) + 20 public snapshots / 80 branches. Loss families
(epoch 0 → 29): phys 35.3→12.4, subgoal 0.56→0.090, reward 0.101→0.011,
dq_public 0.065→0.011, value 0.120→0.026, paraphrase-consistency ~0.02
throughout. Policy interface (W_c/W_w/W_g, α) asserted bitwise unchanged.
Offline numbers are training diagnostics only — no gate claimed.
Artifacts: results/libero_loho_public_v1/v05_gated_wm/ (checkpoints every
5 epochs + strict final bundle + per-loss logs).
Next: H7.4 three matched adapters (v05_current_wm / v05_gated_wm /
v05_gated_random) under the locked matching contract.

## H7.4 scoping registration (before any teacher/adapter run)

Arms follow the registered table EXACTLY: `v05_reset_wm`, `v05_reset_random`,
`v05_recurrent_wm`, `v05_recurrent_random`. (An earlier working note of mine
used a three-arm naming; that note is superseded by the document.)

Registered deviations and bindings, each with reason:

1. **Teacher score Ĝ_i = v̂ (predicted continuation Q_public) only.** The
   registered formula adds P̂(terminal success); the H7.3 training data
   contains 0/160 terminal successes in the 100-action stock continuations,
   so a success head has no positive labels and was not trained (H7.3
   scoping). The omission is a deviation, not an equivalence claim.
2. **Within-sibling ranking loss: not in the fixed H7.3 run** — my H7.3
   scoping missed it; under the one-fixed-run rule I am not retraining.
   Substitute measurement (reported, non-gating): within-group rank
   agreement (Spearman) of frozen v̂ against observed continuation Q_public
   on the grounded 4-branch groups, train and dev separately.
3. **Manifest size:** spec says ~200 decisions from "15 training source
   rollouts"; the registered H7.2 collection has 10 train sources. Rule:
   20 evenly spaced decision bins per source; in each bin the median
   decision with an unresolved subgoal; bins with none are dropped
   (yield ≤200).
4. **Candidate pools at manifest states:** N=4 fresh full-prompt stock
   samples, seed = collection contract noise_seed(d) + 100000*cand.
   Candidate 0 is the designated exchangeable stock reference; for stock
   sources it reproduces the executed rollout chunk exactly (replay
   determinism asserted at the collection snapshot decisions). For cur_wm
   sources the executed v0.4 chunk is NOT placed in the pool (pools are
   pure frozen-π0.5 support; v0.4 stays a collection policy only).
5. **Scoring state = reset-computed (c, w, g) for every arm**, matching the
   WM's training distribution (branch-w init = w0 + one null-action update),
   so reset and recurrent arms share bit-identical selected chunks as the
   spec requires. Recurrent arms differ only in the policy state fed to the
   bias at training/deployment.
6. **Uniform teacher weight 1.0** (spec's default branch; no batched
   weighting implementation added).
7. **Rehearsal:** pool = the 5 stock train sources only, decisions in the
   first 2/3 of each rollout, full-50 flow loss on the executed stock chunk,
   paired 1:1 with each teacher item (cyclic, source-balanced). v0.4 cur_wm
   behavior is never a rehearsal target.
8. **Trainable set per arm:** W_c, W_w, W_g, α_w, α_g only (1.18M params);
   everything else frozen at the H7.3 checkpoint. At init the total bias is
   exactly 0 (zero-init W_c; tanh(0) gates) — asserted, giving stock parity
   at start for all four arms. Budget: AdamW lr 1e-4 / wd 1e-4, grad-norm
   1.0, 5 epochs, one optimizer step per positive teacher state
   (teacher + paired rehearsal averaged), deterministic per-item noise/time
   (base 998000) shared across arms; positive-state mask shared across all
   four arms; random arms replace only the candidate index
   (fixed generator, seed 777000 + state index, uniform over {1,2,3}).
9. **Recorded per arm:** selected-chunk sha256s, score decomposition
   (v̂, dq̂_public, r̂ per candidate), action shift at fixed states, gate
   trajectory (tanh α_w, tanh α_g, ‖W·‖ per epoch), recurrent-vs-reset
   state divergence stats. Value targets are NOT refreshed.

## H7.4 defect found by the registered scorer — bounded contract repair r1

**Observation (frozen scorer, 200 states, 40 grounded groups):** positive
teacher fraction 0/200; ALL 40 grounded groups skipped for zero prediction
variance. Cause confirmed on saved rows: v̂/dq̂/r̂ are bit-identical across
the 4 candidates of every pool. `V05State.outcomes()` feeds the
subgoal/reward/dq_public/value heads from pool(c,g) ONLY — the
action-transitioned prior enters just the physical heads. The registered
H7.3 objective says "transition the posterior state BEFORE decoding …
G^{π0}_{100}"; my implementation violated that for every grounded task
head. The H7.3 launch checks did not include a within-sibling
score-variance assert, which is why this escaped to the scorer stage.
Corrected reading of the H7.3 loss record: the reported dq_public=0.011 /
value=0.026 are fits to snapshot-group MEANS, not action-conditional
predictions.

**Registered repair (r1, bounded; no other changes):**
- `outcomes()` heads for subgoal / reward / dq_public / value consume
  concat[pool(c,g) ‖ pool(w_prior)] (input 2·d_z); physical heads
  unchanged. This is the minimal wiring that routes the action through
  T_w into every grounded head, matching the registered equation.
- Retrain with the IDENTICAL registered budget (seed 0, AdamW 3e-4/1e-4,
  30 epochs, TBPTT 16, same data) → `v05_gated_wm_r1/`. The defective
  run's artifacts stay in `v05_gated_wm/` for the record.
- New launch check: after training, assert within-sibling v̂ variance > 0
  on at least one grounded group (the class of this escape).
- Teacher manifest is rebuilt from scratch with the r1 checkpoint (its
  cached c/w/g tokens are checkpoint-dependent; the π0.5 rollouts and
  candidate pools are checkpoint-independent and reproduce bit-exactly).
- No loss-weight, architecture-width, or budget changes ride along.

## H7.5 pre-registration (before any evaluation episode)

- **Development seeds: 1500, 1510, 1520, 1530, 1540** — fresh; never used
  by smoke (1300) or collection (1400–1430).
- Noise contract: 20e6 + public_task_index·2e6 + env_seed·1e3 + decision,
  public task index 0–4 in registered T1–T5 order; stream shared bit-exactly
  across arms; every arm (stock included, bias ≡ 0) samples through the
  same seeded LC code path — stock parity holds by construction.
- Tranche 1 (75 episodes): stock + v05_reset_random + v05_reset_wm.
  Tranche 2 (50): v05_recurrent_random + v05_recurrent_wm; plus the
  25-episode frozen v0.4 cur_wm legacy reference (secondary, outside the
  factorial). Both tranches registered now; tranche-1 readout changes
  neither seeds nor whether tranche 2 runs.
- Primary metrics per episode: terminal-conjunction SR, Q_public,
  per-subgoal first-completion step, later-invalidation at terminal state,
  first unresolved subgoal, steps. Effects reported per registered H7.5
  formulas; canonical prompts only in this matrix (paraphrase panel is a
  separate frozen secondary run, not used for selection).
- Evaluator: scripts/eval_loho_public_paired.py (immutable per-run
  manifest, hash-verified resume, hash-permutation arm interleaving,
  action traces). Local chain statistics are not merged into this matrix.

## r1 training result (identical registered budget, seed 0)

Epoch 0→29: phys →9.51, subgoal →0.080, reward →0.013, dq_public →0.0051,
value →0.0099, para ~0.015. Sibling-variance launch check PASSED
(within-group v̂ std 7.6e-3 > 0). Note: value/dq_public end LOWER than the
defective run (0.026/0.011) even though they now fit action-conditional
targets rather than group means. Interface-frozen assert passed.
Artifacts: results/libero_loho_public_v1/v05_gated_wm_r1/.

## H7.4 teacher table (r1 checkpoint) — FROZEN

Scoring the 200-state manifest with r1: **155/200 states select a non-stock
candidate (77.5%)**; per-source range 0.55–0.95; advantage quantiles
(10/50/90%) = 0.0005 / 0.0023 / 0.0110 in predicted-Q units — small.
Registered rank diagnostic: 33/40 grounded groups have IDENTICAL observed
continuation Q across all four branches (verified observation-side, not a
prediction artifact), so the effective n is 3 train / 4 dev groups; on
those, Spearman 0.34 train / 0.11 dev, argmax hit 2/3 train / 2/4 dev.
Reading: the grounded evidence that WM ranking tracks observed outcomes is
weak and nearly unpowered at this bank size; recorded as registered, not a
gate. H7.5 is the behavioral test. Random-arm indices drawn at the same
155 states (seed 777000 + state index).
Artifacts: results/libero_loho_public_v1/v05_teachers/.

## H7.4 adapters COMPLETE (4/4, fixed budget) — pre-eval observations

All four arms passed init-parity (bias exactly 0) and suffix-credit
regressions; frozen-WM bitwise asserts passed. Epoch-4 readouts:
- reset_wm  teacher 0.122 / rehearsal 0.145 ; reset_random 0.115 / 0.150 ;
  recurrent_random 0.119 / 0.162 ;
- recurrent_wm 0.257 / 0.393 — its losses RISE over epochs 3–4
  (0.110→0.257 teacher); recorded as instability under the fixed budget,
  not repaired. Its max fixed-state action shift is also largest
  (12.5 L2 vs 4.3–6.2 for the others).
- Gates stayed shut in every arm: |tanh α| ≤ 0.0011.
**Structural pre-eval expectation (measured, not speculation):** with the
gates shut, deployed rec-vs-reset bias differs by 0.04–0.05% of bias norm
at manifest states. The H7.5 history contrasts therefore test whether the
memory paths earned deployment influence — currently they have not; only
environment-chaos amplification could separate rec from reset arms.
This maps to H7.6 branch 4 if it holds behaviorally. Tranches run as
registered regardless.

## H7.5a tranche 1 result (75/75 episodes, seeds 1500–1540)

Per-arm over 25 episodes each: stock SR 0.040 / Q 0.516;
v05_reset_random SR 0.040 / Q 0.472; v05_reset_wm SR 0.000 / Q 0.482.
Stock Q matches the H7.1 smoke mean (0.521) and is in the range of the
paper's external stock reference. Matched effects (25 CRN pairs each):
selection_reset dQ +0.0099 / dSR −0.040; reset_wm − stock dQ −0.0341 /
dSR −0.040. Neither reset adapter improves on stock in this tranche;
the WM-vs-random selection difference is at noise level. Registered plan:
tranche 2 (recurrent pair + v0.4 legacy reference) runs regardless;
routing decisions only after the full matrix.

## H7.5 full matrix COMPLETE (150 episodes) + H7.6 routing analysis

Per-arm (25 episodes each, seeds 1500–1540, CRN-paired):

| arm | SR | Q_public |
|---|---|---|
| stock | 0.040 | 0.516 |
| v04_cur_wm (legacy ref) | 0.040 | 0.512 |
| v05_reset_random | 0.040 | 0.472 |
| v05_reset_wm | 0.000 | 0.482 |
| v05_recurrent_random | 0.040 | 0.470 |
| v05_recurrent_wm | 0.000 | 0.070 |

Registered matched effects (25 pairs each): selection_reset +0.0099 Q /
−0.040 SR; selection_rec −0.4004 Q (confounded, see below); history_random
−0.0015 Q; history_wm −0.4118 Q (confounded); rec_wm−stock −0.4459;
reset_wm−stock −0.0341. Per-task selection_reset spans −0.10 (T3) to
+0.143 (T5): mixed at seed n=5 per task.

**Reading, in registered-branch terms:**
1. **No v0.5 arm improves on stock** in the development matrix (n=25/arm).
2. **Branch 3 fires** (WM-selected ≈ matched random for the stable reset
   pair): model selection is behaviorally ungrounded. Constraint on its
   prescribed repair: the already-collected grounded bank has observed
   continuation-Q variance in only 3/20 train groups (verified), so a
   ranking repair on existing branches is nearly unpowered — the
   100-action stock continuations mostly wash out first-ten differences.
3. **Branch 4 fires for the random pair**: history carry is behaviorally
   inactive (−0.0015 Q), with the mechanism measured pre-eval: gates shut
   (|tanh α| ≤ 0.0011; rec-vs-reset deployed bias delta 0.05%). The WM-pair
   history contrast is NOT interpretable as a history effect — it is the
   recorded recurrent_wm training instability (its collapse is uniform
   across all five tasks: Q 0.12/0.00/0.00/0.00/0.23).
4. Branch 5's capacity-escalation condition is NOT met: saved first-ten
   action shifts are substantial (L2 4.3–12.5), so the registered routing
   points at target coverage/optimization, not action-expert PEFT.
5. v0.4 cur_wm ≈ stock on the public tasks (0.512 vs 0.516) — the local
   chain3/4/5 gains did not transfer positively, and did not hurt.

**No promotion; the sealed public confirmation is NOT run.** The choice
among localized repairs — (a) grounding-signal repair beyond the current
bank's 3 variance-bearing groups (would require a new collection design,
outside current bounds), (b) shorter-horizon continuation grounding on
existing snapshots, (c) optimization repair for the unstable arm — changes
data-collection scope and is left for review, not chosen unilaterally.

## Post-H7 review and supersession

The behavioral result above is valid: the H7 teacher/adapter implementation
did not improve public-LoHo development performance and sometimes degraded it.
It does not falsify the project's language-conditioned predictive-state
hypothesis, because the executed data and model did not instantiate the
registered test:

1. all 20 persistent-failure snapshots were at decision 4 with zero completed
   subgoals, and the implemented source mix omitted the registered
   late-support source;
2. all 40 groups tied on immediate subgoal reward, only `3/20` train and
   `4/20` dev groups varied in continuation Q, and no branch reached terminal
   success;
3. sibling continuations used candidate-dependent policy-noise seeds, so
   candidate effect and continuation randomness were not paired;
4. the monotone completed-subgoal tracker did not provide current predicate
   invalidation/damage labels;
5. candidate actions transitioned only the physical prior; task/value heads
   decoded the transitioned physical feature together with current task
   features, which is a language-conditioned readout rather than
   \(z_t^\ell\rightarrow T(z_t^\ell,a)\);
6. crossed compatible-goal labels, EMA latent closure, and the registered
   within-sibling ranking loss were absent;
7. the same reset-state teachers trained both policy-state modes, while the
   recurrent gate stayed effectively closed.

The H7 data remain useful for rehearsal, null-effect calibration, and the few
explicitly variance-bearing groups. They are not the primary ranking set for
the next model. H7 checkpoints and adapters are frozen negative baselines, not
v0.6 initializations.

The authoritative next actions—effect-resolved crossed training data, a
faithful task-conditioned latent transition, direct π0.5 flow-interface
post-training, and an immediate `N=1` public-LoHo readout—are registered in
`2026-07-30.md`.

## 2026-07-30 audit correction — H7 is a pipeline negative, not an LCWM-hypothesis negative

The 150-episode matrix itself is intact. A read-only audit reconstructed all
six 25-episode arms from the raw records, found no duplicate
`(task, seed, arm)` keys, verified both run-manifest hashes, and checked every
saved trace against the registered noise formula. The SR/Q table above is
therefore a real negative result for the pipeline that ran; it is not an
evaluator, seed-routing, or result-packaging artifact.

The execution-time H7.6 reading above is retained as a chronological record,
but its mechanism attribution is superseded by the following audit.

### What the collected labels actually identify

The headline `70/160 dQ_public > 0` is not action-ranking support. It counts a
branch as positive relative to the snapshot before the subsequent stock
continuation. The relevant support for ranking is variation among sibling
actions at the same snapshot:

| grounded quantity | observed support |
|---|---:|
| 10-action subgoal/reward sibling variance | 0/40 groups |
| 100-action continuation-Q sibling variance | 7/40 groups |
| continuation-Q variance in the train split | 3/20 groups |
| groups with any non-stock Q above candidate 0 | 3/40 |
| non-stock branches above / below candidate 0 | 6 / 6 of 120 |
| terminal-success branches | 0/160 |
| negative `dQ_public` branches | 0/160 |

Three additional facts explain this:

1. Every one of the 20 `persistent_failure` snapshots is decision 4
   (`t=40`). The selector called an unchanged first-unresolved subgoal over
   five decisions a failure, although a normal first-object approach often
   lasts that long. Overall 32/40 snapshots still have subgoal 0 as their
   first unresolved phase. Train-split sibling object-outcome dispersion
   exceeds 1 mm in only 5/20 groups.
2. Continuation sampling uses a different random stream for each sibling
   (`cand * 10_000`). With one rollout per candidate, the seven varying
   continuation-Q groups mix the first-ten action effect with continuation
   policy noise; they are not clean controlled effects.
3. `SubgoalTracker.completed` is monotone. A placement or fixture state that
   is later invalidated remains credited in Q, so all 160 `dQ_public` labels
   are nonnegative and the collected records contain no actual damage label.
   This omits the preservation behavior that matters for long-horizon SR.

The policy-teacher manifest also does not extend the grounded support. Its 200
decision states come from the same source rollouts, but their exact
intersection with the 40 executed snapshot decisions is **0/200**. The frozen
WM therefore extrapolates action scores at every teacher state.

### Why the teacher statistic was misleading

The `155/200 = 77.5%` non-stock selection rate is approximately the null
expectation. With four exchangeable continuous π0.5 samples, an uninformative
scorer chooses one of candidates 1–3 rather than candidate 0 with probability
`3/4 = 75%`. The observed count is only five selections above that expectation
(about 0.82 binomial standard deviations).

On the 20 held-out grounded groups, the frozen r1 checkpoint has:

- absolute continuation-value RMSE `0.1436 Q`;
- within-sibling centered RMSE `0.0348`;
- centered RMSE `0.0778` on the four variance-bearing dev groups;
- median teacher advantage only `0.00234`.

Thus the ranking-relevant error is approximately 15–33 times the median margin,
yet H7.4 selects any non-stock candidate whose predicted score exceeds
candidate 0 by an arbitrarily small positive amount. On the grounded groups,
the frozen scorer's argmax gives one improvement, one regression, and 18 ties
in train; dev has the same 1/1/18 count. The stable reset result
`WM ≈ random` is the expected behavioral consequence.

### The implementation did not instantiate the registered LC predictive state

The repaired r1 model made task heads action-sensitive by concatenating a
language-independent physical prior:

\[
w_{t+1}^{(i)} = T_w(w_t,a_i),
\qquad
\hat y_i = R(c_t^\ell,g_t^\ell,w_{t+1}^{(i)}).
\]

It never transitions the candidate-conditioned task state
`g_t^\ell`. Candidate scoring is therefore a language-conditioned readout over
a physical action prior, not

\[
z_t^\ell
\;\xrightarrow{\,T_\ell(\cdot,a_i)\,}\;
z_{t+1}^{\ell,i}
\;\xrightarrow{\,R_\ell\,}\;
\hat y_{t+1}^{\ell,i}.
\]

Other registered components were absent or explicitly scoped out:

- the current-observation anchored `0.25 tanh` residual update was replaced by
  an unconstrained CrossUpdate; policy gates do not bound recurrent state;
- compatible-goal features were cached for T1↔T5, but no compatible-goal
  labels or loss consumed them; atomic goal relabels were not collected;
- paraphrases received state MSE only, not the registered task-head agreement;
- no next-posterior closure, multi-step latent rollout, centered sibling loss,
  damage head, success head, or within-sibling ranking loss was trained;
- sequential demo task heads read posterior `c_{t+1},g_{t+1}` after observing
  the next frame, whereas branch heads read current `c_t,g_t` plus a physical
  prior; these paths do not share one predictive-state semantics;
- `TBPTT=16` detaches once at `T-16`, producing two graph segments rather than
  truncating every 16 transitions.

The deployed policy interface also bypassed the intended memory mechanism.
`W_c(c_t)` is ungated and dominates the bias. Across manifest states, using
the same trained adapter with carried versus reset state changes the bias by
only 0.015–0.046%; all learned memory gates remain approximately `1e-3`.
Reset and recurrent arms, however, were optimized as separate adapters. They
already produce different actions at decision 0, before any history exists.
The `recurrent_wm Q=0.070` arm is therefore an independent-adapter optimization
collapse, consistent with its rising epoch-3/4 losses, not evidence that
history is harmful. The near-tied random arms show that history did not obtain
meaningful policy coupling in this implementation; they do not establish that
useful history is absent.

### Correct result statement

> H7 establishes that the implemented combination of a readout-style
> action scorer, nearly non-identifying monotone continuation-Q labels,
> zero-margin hard teacher selection, and independently optimized AdaRMS
> adapters does not improve the specification-based public-LoHo development
> panel. It does not test the registered anchored LC predictive transition,
> crossed task-identifier supervision, multi-step outcome sufficiency, or a
> clean same-policy reset-versus-carry history contrast.

Do not spend more seeds on r1 and do not run its sealed confirmation. Preserve
the artifacts and 150 episodes as the implementation-negative reference.

## Superseded H7-r2 draft — historical, do not execute

This was the first post-audit repair draft. It correctly identified paired
continuations, current-valid task labels, crossed goals, candidate-conditioned
task state, calibrated ranking, and an immediate policy readout. It is
superseded because its factorized `w/g` implementation still leaves too much
room to reproduce a physical-transition plus task-readout design. The binding
v0.6 graph instead transitions one complete \(z_t^\ell\), prohibits every
current-state candidate bypass, and sends that same state into π0.5. See
`2026-07-30.md`.

This repair is another method iteration, not a return to a serial diagnostic
program. No 360-continuation bank, latent-rank threshold, or completed offline
assay is placed in front of the next policy job.

### R2.1 — Correct and extend the training labels

- [ ] Replay the existing 20 source trajectories and stored candidate chunks;
  use one sibling-shared continuation-noise stream at each continuation
  decision. If residual policy variance is material, use matched repeated
  streams rather than one different stream per sibling.
- [ ] Replace the decision-4 persistence rule with states localized to
  contact, object motion, recovery, or an actually stalled predicate.
- [ ] In parallel, add one provenance-labeled late-support source per task
  that reaches a state with one of the final two subgoals unresolved; collect
  a small branch set around those states. This is training data and does not
  delay model implementation on the replayed sources.
- [ ] Store multi-horizon targets at the executed branch boundary and after
  bounded continuation, including current-valid predicate bits, ordered
  prefix, invalidation/damage events, terminal conjunction, and signed
  progress. Immediate Euclidean distance remains excluded as an advantage.
- [ ] Relabel each physical branch under canonical, paraphrase, compatible
  T1/T5, and per-object atomic goals. Verify at least one action×goal
  interaction in the training batch; record the statistic, but do not turn a
  fixed count into a launch gate.
- [ ] Resolve the ambiguous local canonical wording for the front/back
  butter and left-bowl tasks before r2 collection; preserve the old prompt
  hashes with H7 and register the r2 prompt hashes separately.

### R2.2 — Implement the model actually described by the formulation

For every candidate and language condition, compute:

\[
\begin{aligned}
w_{t+1}^{(i)} &= T_w(w_t,a_i),\\
g_{t+1}^{\ell,i} &= T_g(g_t^\ell,w_{t+1}^{(i)},a_i),\\
\hat y_{t+1}^{\ell,i} &= R_\ell(w_{t+1}^{(i)},g_{t+1}^{\ell,i}).
\end{aligned}
\]

- [ ] Implement the registered observation-anchor plus zero-initialized,
  `0.25 tanh`-bounded residual for both physical and task state.
- [ ] Decode physical targets from `w_{t+1}^{(i)}` and task-relative
  predicate/progress/reward/success targets from the transitioned
  `g_{t+1}^{\ell,i}`; no current-state task bypass may answer the candidate
  outcome alone.
- [ ] Align transitioned states with the re-encoded branch-after posterior
  under the same language. Add two-/multi-block rollout targets from replayed
  sequential windows rather than replacing long-horizon prediction with one
  scalar 100-step head.
- [ ] Enforce paraphrase agreement and shared physical predictions across
  instructions while supervising compatible different goals with their own
  task-relative outcomes.
- [ ] Correct TBPTT and report source-disjoint dev outcome/controlled-effect
  losses. These are training readouts, not policy-launch gates.

### R2.3 — Ground the teacher and finetune π0.5 conservatively

- [ ] Train a centered within-snapshot pairwise objective on executed
  siblings. Tied/noise-indistinguishable pairs provide no ranking update.
- [ ] Select a non-stock teacher only when its calibrated lower-confidence
  margin exceeds candidate 0; otherwise retain stock. Never use the
  exchangeability-driven non-stock fraction as evidence of teacher quality.
- [ ] Train a grounded/data-only flow adapter from actually superior executed
  branches alongside the r2 model-selected adapter. This is the policy
  interface/headroom arm, not a prerequisite assay.
- [ ] Retain stock rehearsal plus an explicit action-shift/KL trust region;
  pre-register checkpoint selection so a visibly diverging final epoch is not
  deployed automatically.
- [ ] Evaluate the **same adapter checkpoint** with reset and carried state
  when measuring history. Separately trained reset/recurrent policies may be
  performance variants, but are not a clean history contrast.

### R2.4 — Return immediately to success-rate evaluation

As soon as the first r2 checkpoints exist, run fresh development seeds for:

1. stock π0.5;
2. grounded/data-only branch flow finetuning;
3. r2 LCWM-selected flow finetuning;
4. matched score-blind finetuning if it can run concurrently.

Report public-LoHo SR/Q, terminal predicate preservation, late-subgoal
completion, and action shift. A development win freezes the checkpoint for a
sealed confirmation and then LIBERO-10 retention. A miss routes the next
repair from the grounded policy result; it does not reopen a broad diagnostic
gate.
