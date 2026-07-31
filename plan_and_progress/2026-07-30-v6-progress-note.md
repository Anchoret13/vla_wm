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
