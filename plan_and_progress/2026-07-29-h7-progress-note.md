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
