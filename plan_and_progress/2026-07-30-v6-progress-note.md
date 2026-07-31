# v0.6 execution log — scoping registrations and results

Companion to `2026-07-30.md` (the active queue). Every implementation
binding is registered here BEFORE the corresponding run, per the standing
discipline. Claim register: observation-first, n in the same sentence.

## V6.1 design bindings (registered before any training)

State and modules (`lcwm/v06_model.py`):
- z_t^ℓ: 4 tokens × d_z=384 (token budget matched to v0.4/v0.5; no sweep).
- h_t^ℓ = the full π0.5 prefix hidden sequence (visual+language tokens of
  the real prompt) + pad mask — ONE feature stream, no early/late split.
- E_a: MLP over flattened (10 × (7 actions + 1 mask)) → 1 action token
  (matches prior ActionTransition convention).
- T_θ(z, E_a(u)): 2-layer pre-LN transformer encoder over [z ‖ a-token],
  returns updated z tokens.
- Anchor A(h): 4 learned queries cross-attend into h (2 blocks) → c_t.
- U_θ(z̄, h) = LN(c_t + 0.25·tanh R(c_t, z̄)) — observation-anchored,
  fixed β=0.25, NO learned gate. R: cross-attention of c into z̄ tokens +
  MLP, output layer zero-initialized (at init the state is exactly the
  anchored current observation).
- z_∅: learned init tokens; a_∅ = zero action with zero mask. Reset mode
  z^reset = U(T(z_∅, a_∅), h_t) — identical to recurrent at episode start
  by construction.
- D_current(z): current automaton grounding ONLY (per-subgoal
  currently-valid bits, event-milestone bits, ordered-prefix scalar).
- D_next(z̃): every candidate-dependent output — Δq (9), Δobj (8×3),
  next currently-valid bits, next event-milestone bits, 0→1 / 1→0 flip
  bits, damage, reward (milestone delta), success-by-100,
  P_valid(100), Q_valid at h ∈ {10,30,60,100}, τ_next (censored 101,
  scaled /101), and the ranking scalar s_θ(z̃) as its own head. D_next
  consumes pooled z̃ only — no h, no z_t, no D_current features.
- Policy interface: b_t = η·tanh(W_z·Pool(z_t)), Pool = token mean,
  W_z: 384→1024 zero-init, η = 1.0 fixed (bias bounded to [−1,1]^1024;
  same AdaRMS injection path as v0.4/v0.5).
- EMA: decay 0.995 over {E_a, T, U(anchor+R), z_∅}; EMA posterior is the
  latent-closure target only; asserted unreachable from the candidate
  scorer path.

Mechanical assertion suite: `scripts/test_v06_mechanical.py` implements
the eight registered launch assertions. They certify wiring only.

## V6.2 bindings (registered before collection)

- Source seeds (fresh; disjoint from smoke 1300, H7.2 collection
  1400–1430, H7.5 dev 1500–1540, v0.6 dev 1600–1640, sealed 1700–1740):
  task k = 0..4 →
  stock_train_A 2000+10k, support_train_A 2001+10k,
  stock_train_B 2002+10k, support_dev_B 2003+10k.
- Collection noise base 40e6 + task_index·2e6 + source_seed·1e3 + decision
  (disjoint from eval 20e6 and H7.2 30e6 streams).
- Late-support sources: privileged-state scripted staging, DATA-ONLY,
  provenance `staged` in every row (the queue permits "data-only staged
  rollout"; it is never called a full-prompt policy rollout).
- Continuation contract: R=2 repeats, seeds
  50e6 + task·2e6 + snapshot_decision·1e3 + candidate·40 + repeat·20 +
  continuation_decision, shared bit-exactly across sibling candidates for
  fixed (snapshot, goal, repeat, decision); horizons 10/30/60/100.

(Concrete GoalSpec wording, scene-valid distinct goals, and staging
recipes are frozen in `goal_spec_manifest.json`, hashed before any
feature cache.)

## Results log

(appended as stages complete)

## V6.2 snapshot-search eligibility criteria (registered before collection)

- slot 1 (contact/grasp/commitment/object-moving): during the decision's
  ten executed steps, any tracked object body moved ≥ 5 mm.
- slot 2 (recovery/placement/damage-sensitive/final-two-unresolved): at
  decision start ≥1 subgoal valid AND (unresolved ≤ 2, OR a 1→0 flip in
  the previous 2 decisions, OR a 0→1 place flip in the previous 2
  decisions). Slot-2 has priority when both match. Never selected by
  elapsed-time percentile.
- ≤4 eligible decisions per slot examined chronologically; branch order
  fixed: u_support, cand0..cand3, u_replay (u_replay re-executes cand0's
  exact actions to measure restore/execution noise).
- Candidate seeds: collection contract noise_seed(d) + 100000·cand.
- Tranche-A tolerances frozen from ALL tranche-A exact repeats
  (per-component 95th percentile), then first effect_resolved group per
  slot retained; every executed rejected group saved.

## Phase-B continuation semantics (registered before phase B runs)

- Goal automata unroll from episode reset along the replayed real path;
  per-decision states stored for every registered GoalSpec (never
  initialized at the snapshot).
- Continuation flip window: at continuation start the restored automaton
  keeps validity + achieved events but zeroes the flip log, so
  damage/τ_next in the outcome tuple measure the continuation window;
  branch-window flips live in the phase-A record.
- Re-reach determinism asserted against phase-A object positions
  (atol 2e-3) at every pending snapshot decision.

## Tranche A phase-A result

10/10 sources (5 stock, 5 staged — t1/t5 re-collected once after the
registered site-anchor staging repair; the two body-anchor failures are
kept on record). 74 exact repeats; frozen 95th-pct tolerances:
eef_pos 1.97e-3, eef_quat 7.8e-4, gripper 1.5e-4, obj_pos 0.0,
valid_bits 0.0 — restore/replay is bit-exact on object positions, so any
candidate-pair object difference qualifies as an effect under the
registered rule. 57/74 audited groups effect_resolved; 18 accepted
(first per slot per source; t2/t3 have one slot each with no resolved
group). policy_rankable is determined only by phase-B paired
continuations, not by these physical micro-differences.

## V6.3 training bindings (registered before the smoke and the fixed run)

- Budget: seed 0, AdamW lr 3e-4 / wd 1e-4, grad-norm 1.0, TBPTT 16
  decisions, 25 epochs, EMA decay 0.995, one optimizer step per source
  visit (loss = mean over present families). No sweeps.
- h features: per-(source, decision, prompt-sha) fp16 sidecars from
  cache_v06_features.py; crossed rows share stored physical transitions,
  never prompt-bound features.
- Sequence pass (canonical prompt): executed-action physical prediction
  through D_next(T(z_t,u_t)) with locked v0.4 scales (Huber δ=4);
  D_current grounding (BCE valid+event bits, MSE ordered-prefix/n);
  reward = valid-count delta (MSE); 1-step latent closure MSE to the EMA
  posterior (EMA unrolled in parallel); 2-/3-block open-loop closure
  (T applied repeatedly, no U) at every 8th decision.
- Paraphrase pass: full unroll under each paraphrase h; same-goal state
  consistency MSE (symmetric, every 4th decision) + identical
  current-grounding labels.
- Distinct-goal pass: full unroll under the distinct goal's h;
  current grounding vs THAT goal's automaton labels; shared-physics:
  identical physical targets supervised from the distinct-goal state.
- Branch pass: absolute physical effects for every audited branch;
  within-sibling centered effects over candidates 0..3; both use
  D_next(predict(z_d, chunk)).
- Continuation pass (accepted groups): success BCE, P_valid MSE, Q_valid
  (4 horizons) MSE, τ/101 MSE, damage MSE — per goal, only from that
  goal's own continuations; Bradley–Terry on s_i for non-tied A_ij
  within (group, goal), ties masked. A_ij per the registered
  lexicographic + both-repeats rule; outcome components are discrete
  counters so the comparison tolerance is 0 (registered).
- Standard-LIBERO representation support: seq_prefix_cache_v1 demo
  episodes enter the sequence pass (their cached prefix hidden is h;
  physical + closure families only — no public automaton labels).
- Loss-source weighting is balanced by source episode, not by duplicated
  language rows (each source visited once per epoch; its variant passes
  averaged within the source's loss).
- The e2e SMOKE runs this trainer for 2 epochs on tranche A only and
  gates nothing but mechanics (finite losses, nonzero grads per family,
  reload). The fixed run restarts from scratch on the frozen two-tranche
  manifest.

## Results log — phase B + e2e smoke

- Phase B: 17/18 groups completed on the first pass (470 goal-conditioned
  continuation records; 6/17 groups policy_rankable by the registered
  paired-preference rule so far). The 18th group (t5_stock d=93, late
  slot) exposed two real infrastructure defects, both measured and fixed:
  (1) restore() does not clear robosuite's terminal flag — a terminal
  continuation poisoned later restores (path never exercised before:
  H7.2 had 0/160 terminal continuations); (2) the robosuite horizon is
  fixed at 1000 regardless of make_public_env's episode_length, and AT
  the horizon the env sets done while returning term=False, so the next
  step raises — late snapshot + branch + 100-action continuation
  legitimately crosses it. Fixes: done-clear on restore + direct horizon
  override in phase B only. The 17 completed groups were verified
  unaffected (any horizon crossing would have crashed, none did).
- e2e training smoke (2 epochs, tranche A): PASSED — all 12 registered
  loss families active, finite, and declining; no silent families.
  Offline numbers are mechanics only, per the queue.

## Two-tranche manifest FROZEN; fixed V6.3 launched

20/20 sources (15 train / 5 dev; all staged sources placed via region-site
anchors). 30 accepted groups with paired continuations. Per-task support
counts (descriptive, not gates): policy_rankable t1..t5 = 0/1/2/2/2;
cross_goal_disagreement = 1/2/3/3/3. History-contrast rows: NOT collected
in tranches A/B — recorded as a support gap (no matched pairs
manufactured); the D_current grounding trains memory only through real
sequential labels. The fixed seed-0 run restarts from scratch on this
manifest with the registered budget.

## V6.3 fixed run COMPLETE (seed 0, 25 epochs, registered budget)

Epoch 0→24 (train families): seq_phys 748→28.6, branch_abs 850→42.4,
shared_phys 294→8.9, closure1 1.34→0.25, closure23 1.50→0.34, current
1.21→1.15, distinct_current 0.64→0.54, reward 0.050→0.038, cont_heads
1.23→0.81, ranking 0.693→0.577, para ≈0. No silent loss families.
These are training diagnostics, not gates. Note honestly: ranking's
trainable support is thin (7 policy_rankable groups); current-grounding
loss plateaus high. Artifacts: results/libero_loho_public_v1/v06_wm/.
Next: V6.4 teachers → V6.5 three matched W_z jobs → V6.6 125-episode
development matrix (seeds 1600–1640), per the pre-registered order.
