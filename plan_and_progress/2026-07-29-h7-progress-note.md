# H7 progress note (written during a tool-platform outage)

Status: the Bash execution classifier on my side has been unavailable for
an extended period this session; file reads/writes work, but compilation,
git, GPU jobs, and directory listing are blocked. Work completed within
those limits, all UNCOMPILED and UNCOMMITTED until the platform recovers
(to be verified before any is treated as done):

## Written (pending compile + run)

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

## Queued for the moment execution recovers (in order)

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
