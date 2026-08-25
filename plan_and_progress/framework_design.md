# pi05-lcwm — Framework Design v1.0

Working design record, updated 2026-08-20. v1.0 opens a new method branch after the V7.3–V7.7 results. It preserves the accepted problem formulation—a pretrained VLA supplies the action prior, while a language- and action-conditioned latent world model learns decision-relevant dynamics, reward, and value—but changes the learning protocol from one fixed offline vertical slice to bounded deployment-time improvement through failure-anchored counterfactual interaction.

The new branch is provisionally called **failure-driven model-based VLA improvement**. The name is descriptive rather than a paper title.

`[LOCKED]` denotes the current method commitment. `[DEC]` denotes a bounded unresolved choice. `[EST]` denotes a quantity that must be measured before it can be fixed.

## Version lineage and identifier collision (resolved 2026-08-17, Action 0.1)

Two different designs were both labelled `v0.9`. That collision is closed here.

| identifier | content | status | retrieval |
|---|---|---|---|
| `v0.9` (pre-pivot) | one-shot LC-Flow design: one LCWM checkpoint, one π0.5 fine-tune, one public-LoHo behavior slice | frozen historical record; **not** superseded in its own evidential claims | git blob `a2e4af72026141742d2da4ce5f6735619c4b58dd`, commit `1f28a544` |
| `v1.0` (this file) | failure-driven model-based VLA improvement: bounded rounds of failure-anchored counterfactual interaction | current mainline design | working tree |

`[LOCKED]` Experiment series. The pre-pivot experiment line is `V7.x` and is closed. The new mainline experiment line is **`V8.x`**, indexed as: `V8.0` benchmark calibration (Action 1), `V8.1` bootstrap acquisition (Action 2), `V8.2` model `M_0` (Action 3), `V8.3` selector comparison (Action 4), `V8.4` first policy update (Action 5), `V8.5` round 2 and one-shot control (Action 6). `V8.1P` is the single-task Action 2P feasibility pilot of §15 and does not renumber or satisfy `V8.1`. New code lives under `lcwm/v08*_`, `scripts/*_v08*`; existing `v07*_` entry points are read-only history and are not edited into this orchestration.

No `V8.x` GPU execution is authorized by this document alone. §§13–15 state the historical contracts, their dated outcomes, and the next prospective gate.

---

## 0. Branch boundary and empirical motivation

### 0.1 What the pre-pivot line established

The V7.3–V7.7 line produced useful negative evidence and implementation infrastructure, but it did not establish a working model-based policy-improvement mechanism. All six records below are reported at their original contracts and are not reinterpreted as v1.0 evidence.

1. V7.3's policy comparison was implementation-confounded: the deployed pooled state was dominated by a common mode, the generated-target schedule concentrated almost all model mass on one anchor, and the sole selected model target had no verified realized advantage over the stock candidate.
2. V7.4A removed the pooled common mode but found that the recurrent state was functionally inert at inference: current-observation and language content varied, while real/reset/shuffled histories were nearly identical.
3. V7.4B again failed to acquire the counterfactual support required by the intended model teacher. Milestone-positive recovery appeared only in `t2|train`; late-blocker anchors were absent in all five tasks; matched positive/nonpositive and action-by-goal reversal cells remained empty. The surviving grounded policy source was 4 rows / 4 anchors / 2 tasks.
4. **V7.5 (`2026-08-10_v075_lcwm_r1`, sealed at `1f28a54`) separated architecture repair from predictive dynamics.** The `hist_center + r_norm` repair is real: it moved the different-history plumbing gain from `1.82e-10` to `0.231`, and training sustained `hist_rel ≈ 0.237` with rollout margin `≈ 0.0043` over epochs 5–11 — the first measurably live recurrence in the project. The objective did not *sustain* it: at epoch 12 the contrast fell ~17x in one epoch and never recovered, ending at `hist_rel = 0.00105`, rollout margin `+1e-6`, i.e. ~121x below its own demonstrated plateau on the complete gate set (`0.001954`). The binding gate returned `gate_pass = true` only because its `1e-4` threshold had been anchored to V7.3C's dead `~1e-6` baseline. Decisively: **held-out action-effect prediction beat the copy baseline on `0/72` rows at the final checkpoint, and on zero rows in 24 of 25 epochs.** Recurrence liveness is not action-conditioned dynamics. Stages 4–5 were retired unexecuted after the support census and must not be described as an empirical baseline.
5. **V7.6 halted at acquisition on its registered rule.** The V7.6A census relocated the yield question from the model to the snapshot: reacher-built mid-chain snapshots yielded 6/36 positives against V7.4B's 4/129 from stock-stall snapshots — a 5.4x effect, exact one-sided binomial `p = 7.8e-4` against the V7.4B rate. That still missed the pre-derived bar (≥ 8/40, equivalently ≥ 9.6% from the tranche's own quota arithmetic; Wilson 95% lower bound at 6/36 is `0.0887`). `V7.6C–G` were not run. The family breakdown was sharper than the aggregate: `recovery` 5/5, `first_pick` 1/16, **`placement` (a `pick_up` target) 0/14**; per task t1 2/9, t2 2/8, t5 2/9, t3 0/10.
6. **V7.7 showed the acquisition budget had been the instrument.** In a fresh pre-registered probe (`2026-08-15_v077_reacquire_r1`, 43 scored rollouts), **no `pick_up` milestone in the entire probe occurred at or before step 60; the earliest of twelve successes was step 66** (success steps 66, 68, 68, 68, 69, 69, 70, 83, 97, 101, 103, 106). The easiest cell — fresh episode start, first object, atomic prompt — was `0/9 @60` and `5/9 @120`. Therefore V7.6's 60-action and V7.4B's 30-action recovery budgets could not register a `pick_up` success structurally, not statistically; `pick_up 0/14` is retired as a statement about π0.5, and the V7.6 aggregate 16.7% and V7.4B 3.1% are measurements of what completes inside a truncating budget. Two findings survive as genuine: the atomic prompt *beats* the full instruction at both states (5/9 vs 3/9 fresh; 3/8 vs 1/8 mid-chain), so prompt choice was not the confound; and **π0.5 does not redirect to a later object while an earlier one is still present** (cell E: approached 3/9, contacted 0/9, against 3/8 for the same object once object 1 is removed). The reacher is also seed-sensitive at depth 3 across two independent runs.

`[AMENDED 2026-08-20; prospectively supersedes the 2026-08-19 extension; historical verdicts unchanged]` **Instrument and endpoint rule.** No yield or recovery bar in `V8.x` may be evaluated at an underived branch-continuation budget, and every such readout is reported at no fewer than two horizons. An episode deadline has a different role and must be named before execution: under a **prospectively registered experimental finite-horizon objective**, failure means non-completion by the frozen horizon and later completion is a diagnostic of speed and recoverability, not grounds to erase the fixed-horizon result; under a **capability/headroom claim**, the horizon and late-conversion tolerance must instead be established prospectively from a completion-time panel. Stage 1R.1 shows that `chain1b@250` is strongly deadline-sensitive, so it may support only claims explicitly conditional on that experimental horizon. It does not show that failure-driven counterfactual coverage is absent.

These results do not falsify language-conditioned latent dynamics. They show that (i) a fixed bank collected from a near-zero-success policy cannot be assumed to contain the interventions needed for policy improvement, (ii) architecture refinement cannot manufacture outcomes the acquisition never contained, and (iii) the acquisition instrument's own parameters can dominate the measurement they were meant to make.

V7.4 C/D/E and V7.5 Stages 4–5 may be retained as archival pre-pivot material, but they are unexecuted and no longer define the main research path.

### 0.2 What changes in v1.0

The primary scientific object is no longer one predictive-state checkpoint followed by one policy fine-tuning run. It is the bounded improvement sequence, written here in the form locked by C2 (§4.0) — with an explicit bootstrap phase, because no selector may use a model that does not yet exist:

\[
\underbrace{\pi_0\rightarrow\mathcal B_{\mathrm{boot}}\rightarrow M_0}_{\text{bootstrap: no policy update}}
\;\rightarrow\;
\underbrace{\mathcal B_0\rightarrow M_1\rightarrow\pi_1}_{\text{round }0}
\;\rightarrow\;
\underbrace{\mathcal B_1\rightarrow M_2\rightarrow\pi_2}_{\text{round }1}.
\]

Each branch set \(\mathcal B_k\) is collected around failures of the current deployed policy \(\pi_k\); \(\mathcal B_{\mathrm{boot}}\) is collected around failures of \(\pi_0\) under coverage-only selection and trains \(M_0\) without touching the policy. Failure identifies where the current data do not support the decisions required by the policy; alternative-action interaction supplies the missing action-conditioned evidence.

`[LOCKED]` Failure is a mechanism for targeted state-action coverage expansion. It is not itself a reward label, an inverse-RL signal, or merely a post-hoc diagnostic category.

---

## 1. Research identity

### Central question

> Can a pretrained imitation-learned VLA improve from its own deployment failures by using targeted counterfactual interaction to refine a decision-relevant world model and distill better actions back into the policy?

### Working thesis

Demonstrations train a VLA to reproduce expert actions on nominal histories. They do not identify the consequences of alternative actions at histories reached by the learned policy, especially near failures. Consequently, the same representation and data may be adequate for behavior cloning but insufficient for action-conditioned prediction, value comparison, and recovery.

The proposed system therefore:

1. deploys the current VLA and identifies failed or stalled histories using observable task outcomes or progress;
2. restores or reproduces those histories and executes multiple VLA-supported alternative action chunks;
3. uses the resulting positive and negative action consequences to update a language-conditioned latent world model;
4. uses the calibrated model both to choose informative future interactions and to construct conservative policy-improvement targets;
5. distills verified and model-supported improvements into the VLA while retaining its action prior;
6. repeats the process for a small, fixed number of interaction rounds and deploys the final policy with full-prompt `N=1` inference.

### Claim ladder

The branch is deliberately staged. A stronger claim is considered only after the preceding claim is supported.

1. **Coverage claim:** failure-anchored branches add action-effect and outcome support that demonstrations and nominal rollouts lack.
2. **Acquisition claim:** a learned world model finds useful counterfactual interactions more efficiently than matched random branch selection under the same interaction budget.
3. **Policy-improvement claim:** world-model-guided targets improve the deployed VLA beyond direct imitation of the same verified corrections.
4. **Continual-improvement claim:** repeated bounded rounds move the failure frontier and improve success more than a one-shot update with the same or matched total data budget.

### Non-goals

- inverse reinforcement learning;
- pixel or video reconstruction as the main world-model objective;
- arbitrary continuous-action search outside the VLA support;
- treating language as a physical cause of environment dynamics;
- replacing the VLA with a low-level model-predictive controller;
- deployment-time best-of-N reranking as the final system;
- claiming a general-purpose lifelong-learning algorithm from two or three bounded rounds;
- claiming novelty for the memory architecture, pooling layer, or uncertainty estimator alone.

---

## 2. Setting and retained VLA interface

Let

\[
H_t=(o_{\leq t},a_{<t})
\]

denote the available visual-action history and let \(\ell\) denote a natural-language instruction. At improvement round \(k\), the current VLA produces a \(K\)-step action chunk

\[
u_t \sim \pi_k(\cdot\mid H_t,\ell).
\]

For the current \(\pi0.5\)/LIBERO implementation, \(K=50\), the executed commitment is \(c=10\), and the policy replans after observing the next state.

`[LOCKED]`

- \(\pi_0\) is the pretrained imitation policy, representation source, and initial action prior.
- Every candidate action chunk is sampled from \(\pi_k\), a frozen \(\pi_0\), or a bounded perturbation of their supported chunks. The method never optimizes an unconstrained 500-dimensional action sequence from scratch.
- The learned policy remains the final controller. Deployment uses recurrent or windowed full-prompt `N=1` inference without an external planner.
- The existing zero-initialized AdaRMS/flow-boundary injection and stock-initialized action output head remain the default policy-update interface because they preserve exact stock behavior at initialization.

The physical transition is language-independent,

\[
x_{t+c}\sim P^{(c)}(\cdot\mid x_t,u_{t,0:c}),
\]

while the predictive state may be task-conditioned because language determines which future distinctions matter:

\[
z_t^\ell=\phi_\theta(H_t,\ell).
\]

Language enters the complete state. The transition need not receive a second task-ID or raw-language bypass:

\[
\widetilde z_{t+c}^{\ell,i}
=T_\theta\!\left(z_t^\ell,E_a(u_t^i)\right).
\]

Instruction-independent physical effects and instruction-dependent reward, progress, and value are decoded separately from the transitioned state.

### 2.1 Executed-action causal contract (closes C1)

`[LOCKED]` The VLA proposes \(K=50\) actions; the environment commits at most \(c=10\) before replanning. Only executed actions may carry causal credit anywhere in the system.

Define the realized execution length

\[
c_{\mathrm{eff}}
=
\min\!\left(c,\;\text{steps actually stepped before termination or truncation}\right),
\qquad
m_j=\mathbb 1[j<c_{\mathrm{eff}}],\; j=0,\ldots,K-1,
\]

and the executed chunk \(u_{\mathrm{exec}}=u_{0:c_{\mathrm{eff}}}\). Then, everywhere in v1.0:

1. **Transition input.** The latent transition consumes the executed prefix and its realized length, never the proposal:
   \[
   \widetilde z_{\tau+c_{\mathrm{eff}}}^{\ell,i}
   =T_\theta\!\left(z_\tau^\ell,\;E_a\!\left(u^i_{\mathrm{exec}},\,c_{\mathrm{eff}}^i\right)\right).
   \]
   \(c_{\mathrm{eff}}\) is embedded, so a branch truncated at 7 steps is not encoded as a 10-step branch. The unexecuted suffix \(u_{c_{\mathrm{eff}}:K}\) may never enter \(E_a\).
2. **Effect and outcome labels.** \(\Delta w,\Delta y^\ell,r^\ell\) are measured between the anchor state \(x_\tau\) and \(x_{\tau+c_{\mathrm{eff}}}\). Continuation returns \(G_{i,h}\) start their horizon clock at \(\tau+c_{\mathrm{eff}}\), and \(h\) is counted in environment steps, not in chunks.
3. **Policy credit.** Branch- and model-derived flow-matching credit is masked to the executed prefix,
   \[
   \mathcal L_{\mathrm{verified\text{-}FM}}
   =
   \frac{\sum_{j} m_j\,\big\|v_\theta^{(j)}-\hat v^{(j)}\big\|^2}{\sum_j m_j},
   \]
   with the same mask used for \(\mathcal L_{\mathrm{model\text{-}FM}}\). The suffix \(j\ge c_{\mathrm{eff}}\) receives stock trust only (\(\mathcal L_{\mathrm{suffix\text{-}trust}}\)); ordinary demonstrations keep full-chunk flow matching because their whole chunk was executed by the demonstrator.
4. **Candidate identity.** Two candidates are the same candidate iff their quantized executed prefixes agree within the measured replay-noise tolerance. Deduplication, diversity distance, and the "one candidate family" rule of §4.1 are all computed on \(u_{\mathrm{exec}}\), never on the full 50-step chunk or on diffusion noise.
5. **Ledger.** Every branch record stores \(c_{\mathrm{eff}}\), the mask, and the termination cause (`committed`, `env_terminated`, `env_truncated`, `monitor_halt`). A record without a termination cause is invalid and is dropped before any fit, with the drop counted.

---

## 3. Failure and local state-action coverage

### 3.1 Failure anchors

For a rollout of \(\pi_k\), define a failure-anchor set

\[
\mathcal F_k
=
\left\{
(H_\tau,\ell):
\text{the rollout terminates unsuccessfully or makes no task-valid progress over a registered window}
\right\}.
\]

In simulation, the task automaton or environment success/progress predicates provide this signal. In a later real-robot extension, failure may be supplied by a task monitor, human intervention, or a separately validated success/progress detector.

`[LOCKED]` Model uncertainty does not define whether a rollout failed. Uncertainty may prioritize which alternatives to execute after a failure anchor has been identified.

The stored object is the complete reproducible history state, not only the final image.

### 3.1.1 Operational anchor extractor (closes C4)

"The earliest decision after which the policy cannot recover" is **not** observable from a terminal failure. v1.0 therefore defines two named anchor kinds and forbids using the language of the second for the first.

**Retained snapshots.** During every acquisition rollout, a decision snapshot is written at each replan boundary (every \(c\) steps) containing: MuJoCo `sim.get_state()`, controller state, the runner's observation/action history window and its KV-reconstruction recipe, task-monitor predicate bits, ordered-milestone index, step index, episode seed, and the policy checkpoint hash. Snapshot stride is \(c\); a coarser stride is a registered per-run parameter, never an implicit default.

**Progress and stall.** Let \(p(t)\) be the milestone-progress index from the frozen task automaton (non-decreasing by construction) and \(t_{\mathrm{prog}}=\max\{t: p(t)>p(t-1)\}\) the last progress step (0 if none). A rollout is *stalled from* \(t\) if \(p\) does not increase over \([t,t+W]\) and the episode ends without success, where the stall window is

\[
W=\big\lceil 1.5\cdot q_{90}(\text{milestone-to-milestone completion interval})\big\rceil ,
\]

with \(q_{90}\) measured on the Action-1 calibration panel of the selected benchmark and frozen before any acquisition. `[EST]` \(W\) has no inherited value; the pre-pivot 30/60-action budgets are not evidence for it (§0.1, item 6).

`[LOCKED]` **Instantiating \(p(t)\) (amended 2026-08-18, forced by V8.0 screen data).** Two instantiations are permitted and the task decides which:

- **order-enforcing goals** (a drawer must open before a placement is valid, as in the LoHo suite): \(p(t)\) is the *ordered prefix* over the registered subgoal list;
- **unordered-conjunction goals** (the Chain family, whose BDDL goal is `(And (In a basket) (In b basket))` with no sequencing): \(p(t)\) is the *count of achieved milestones*.

The V8.0 screen showed the distinction is not academic: on `chain2b_lr2`, 6 of 10 episodes solved the two objects in the reverse of the registered subgoal order. Under an ordered-prefix \(p(t)\) those episodes register no progress at all until the late object completes, and differencing completion steps by subgoal index yields *negative* intervals — the first run of the screen produced a `pick_up cream_cheese_1` q90 of \(-178.5\). Milestone intervals are therefore always computed as gaps between successive achievements **in time**, tagged by the class of the milestone that ended the gap.

`[LOCKED]` **Per-class conditioning.** \(W\) and \(H_{\max}\) are derived per milestone class \((\text{index},\text{kind},\text{object})\) and the frozen scalar is the maximum over the classes the acquisition will anchor on, so the §5.2.1 inequality holds for every class rather than on average. The per-class vector is retained so continuations can be indexed by anchor phase. A pooled q90 may be reported but may never be the frozen budget: it mixes start-to-first-milestone against late-chain re-approach, and rungs with a single object contribute only the former.

**Tier 1 — `stall_onset` anchor (default).** \(\tau=\) the replan boundary at or immediately after \(t_{\mathrm{prog}}\). Cost: zero extra environment steps. This is a heuristic localization and is recorded under the literal name `stall_onset`. It carries **no** claim of unrecoverability.

**Tier 2 — `earliest_unrecoverable` anchor (optional, charged).** Only when explicitly registered for a sub-panel. Binary-search the retained snapshots for the earliest \(\tau^\star\) such that \(R\) repeats of the closed-loop continuation under \(\pi_k\) from \(\tau^\star\) attain the next milestone in 0 of \(R\) repeats, while the immediately preceding retained snapshot attains it in \(\ge 1\) of \(R\). `[DEC]` \(R=3\) by default. Continuation length for this test uses the same \(H_{\max}\) as branch continuations (§5.2.1) — a tier-2 search run at a budget below the measured milestone floor would reproduce the V7.7 error and is prohibited.

`[LOCKED]` **Deadline admissibility (added 2026-08-19, Action 1R).** An episode has a frozen deployment deadline \(L\). An anchor at step \(\tau\) is admissible only if it leaves a full candidate prefix and enough official continuation:

\[
c_{\mathrm{eff}}=\min\!\big(c,\max(0,L-\tau)\big),
\qquad
D=\max(0,\,L-\tau-c_{\mathrm{eff}}),
\qquad
\text{admissible}\iff c_{\mathrm{eff}}=c\ \wedge\ D\ge D_{\min}(\text{class}).
\]

\(D_{\min}\) is the achieved-state-conditioned class bound, never the global \(H_{\max}\). Applied to the V8.0 confirmation data this rule already rejects **16/16** `chain1b` last-progress anchors (20–60 steps remained at the pick event against a 71-step place bound) and 3/8 `chain2b` ones.

`[LOCKED]` **Observable stall (added 2026-08-19).** \(W_{\mathrm{used}}=\min(W,\,L-\tau)\), and a state may be called a \(W\)-step `stall_onset` only when \(L-\tau\ge W\). The V8.0-frozen \(W=345\) exceeds `chain1b`'s entire 250-step episode, so no state on that task can carry that label. FQE may replace an unidentified value target, but it does not make an unobservable window or an inadmissible anchor valid.

`[LOCKED]` Every environment step executed by a tier-2 search is charged to the interaction ledger under role `anchor_search` (§7.6). Any report using the phrase "earliest unrecoverable" must cite a tier-2 execution; otherwise the anchor is named `stall_onset` and described as such.

### 3.2 Coverage is local to the decision class

Global coverage of visual state-action space is neither measurable nor required. The relevant object is local realization coverage around histories encountered by the current policy and action chunks supported by the VLA:

\[
\mathcal C_k
=
\left\{
(H,\ell,u,y):
(H,\ell)\in\operatorname{Reach}(\pi_{0:k}),
\;u\in\mathcal C_{\pi_{0:k}}(H,\ell),
\;y\text{ is an observed transition or continuation outcome}
\right\}.
\]

Demonstrations provide dense nominal trajectory coverage but usually one action realization per history. Failure branches expand the action dimension at policy-relevant histories.

A useful branch group contains variation in at least one registered outcome: next latent state, physical effect, task progress, damage, success, or continuation return. Both successful and unsuccessful branches are retained. A positive recovery is necessary for a local policy-improvement target, but negative branches remain necessary for dynamics and ranking calibration.

### 3.3 Failure frontier

For each task, record the exact achieved, current-valid, damaged, actionable, and unresolved event sets of every rollout. Scalar phase and ordered-prefix summaries are display-only. The distribution of unresolved-set signatures is the failure frontier of \(\pi_k\). Continual improvement requires more than repeatedly solving the same stored anchors: after updating to \(\pi_{k+1}\), the unresolved sets should contract or disappear on held-out evaluation rollouts.

---

## 4. Counterfactual interaction

### 4.0 Bootstrap phase and round indexing (closes C2)

`[LOCKED]` One convention, used everywhere in v1.0. It removes two defects in the previous draft: a model-guided selector named in a round before any model existed, and a cumulative dataset \(\mathcal D_k\) that excluded the very branches \(\mathcal B_k\) that were supposed to feed the model following them.

**Phase B — bootstrap (no policy update).**

1. Deploy \(\pi_0\) on the bootstrap acquisition seeds; extract anchors by §3.1.1 tier 1.
2. Collect \(\mathcal B_{\mathrm{boot}}\) with **coverage-only** selection: \(R\) repeats of \(u^0\), farthest-point diversity over the executed-prefix/effect embedding, and a matched-random draw. No outcome-dependent selection inside the phase.
3. Fit \(M_0\) on \(\mathcal D_{\mathrm{boot}}=\mathcal D_{\mathrm{demo}}\cup\mathcal D_{\mathrm{nominal,boot}}\cup\mathcal B_{\mathrm{boot}}\); calibrate \(\beta\), the support region, and \(\delta\) on a **source-disjoint** held-out split of \(\mathcal B_{\mathrm{boot}}\).
4. \(\pi_0\) is unchanged by phase B. \(\mathcal B_{\mathrm{boot}}\) and \(M_0\) are **common to every arm** and this sharing is declared as part of every compared method (§7.2.1).

**Rounds \(k=0,1\) (and \(k=2\) only if registered before round 0).** With \(\mathcal D_0:=\mathcal D_{\mathrm{boot}}\) and \(M_0\) as above:

\[
\mathcal A_k=\operatorname{Anchors}(\pi_k),\qquad
\mathcal B_k=\operatorname{Execute}\big(S_k(\mathcal A_k,\mathcal P_k;M_k)\big),
\]
\[
\mathcal D_{k+1}
=
\mathcal D_k\cup\mathcal D_{\mathrm{nominal},k}\cup\mathcal B_k,
\qquad
M_{k+1}=\operatorname{Fit}(\mathcal D_{k+1}),
\qquad
\pi_{k+1}=\operatorname{Update}(\pi_k,M_{k+1},\mathcal B_k).
\]

`[LOCKED]` **Selector legality.** A selector \(S_k\) may consume only \(M_k\), i.e. a model trained and calibrated strictly before round \(k\) on source-disjoint data. `S_0` consumes \(M_0\) from phase B. There is no round in which a `WM-acquisition` arm uses a model that its own round produced.

`[LOCKED]` **First legal WM-vs-random comparison.** It is round 0, and only round 0 supports a clean single-component attribution: all arms enter round 0 from the identical \(\pi_0\), the identical anchor panel \(\mathcal A_0\), the identical candidate pools \(\mathcal P_0\), the identical executed-step budget, and the identical seeds; only \(S_0\in\{\text{random},\text{diversity},\text{WM}\}\) differs. From round 1 on, arms have different policies and therefore different failures by design; those results are end-to-end algorithm comparisons and are reported as such (§7.2.1).

### 4.1 Candidate pool

At a failure anchor \((H_\tau,\ell)\), form a proposal pool

\[
u^0\sim\pi_k(\cdot\mid H_\tau,\ell),
\qquad
u^i\sim q_k(\cdot\mid H_\tau,\ell),
\quad i=1,\ldots,N,
\]

where \(u^0\) is a reproducible reference chunk and \(q_k\) is a mixture of the current policy, frozen \(\pi_0\), and bounded same-policy stochastic samples. Candidate generation must preserve the VLA action prior while producing nontrivial first-\(c\)-action diversity.

Candidate diversity is measured in executed action space and predicted physical effect, not merely diffusion noise or token distance. Byte-different chunks whose first \(c\) actions are replay-noise-indistinguishable count as one candidate family.

### 4.2 Branch execution

For each selected candidate, restore the same simulator snapshot and execute the first \(c\) actions under common-random-number control where possible. Then run a registered closed-loop continuation policy for one or more horizons. Store

\[
b_i=
\left(
H_\tau,\ell,u^i,o_{\tau+c}^i,
\Delta w_i,\Delta y_i^\ell,r_i^\ell,
G_{i,h_1},\ldots,G_{i,h_m}
\right).
\]

Here \(\Delta w\) denotes instruction-independent physical effects, \(\Delta y^\ell\) denotes task-conditioned progress or predicate effects, and \(G_{i,h}\) denotes a continuation return or success/progress outcome at horizon \(h\).

Repeated execution of \(u^0\) estimates replay noise. When two acquisition selectors are compared from the same policy checkpoint, they receive the same anchor set, candidate pools, branch budget, continuation budget, and environment seeds. Round 0 therefore admits an exactly matched selector comparison. After policy arms diverge, their failures are allowed to diverge as part of the method; later rounds match task-level rollout and interaction budgets rather than forcing different policies back onto an artificial common anchor set.

### 4.3 Selectors

Phase B uses coverage-only selection, as specified in §4.0: execute \(u^0\) \(R\) times, execute farthest-point-diversity candidates over the executed-prefix/effect embedding, execute a matched-random candidate, and retain every outcome without outcome-dependent recollection inside the phase.

From round 0 on, three selectors are defined over the *same* pool \(\mathcal P_k\) and the *same* executed-step budget:

| selector | rule | uses a model? |
|---|---|---|
| `random` | uniform draw over deduplicated candidate families | no |
| `diversity` | farthest-point over the executed-prefix/effect embedding | no |
| `wm` | mixture of two model roles below | yes, \(M_k\) only |

The `wm` selector splits its budget by a fixed, pre-registered ratio between:

1. **coverage acquisition:** highest epistemic disagreement or lowest local support under \(M_k\);
2. **improvement acquisition:** highest conservative predicted advantage \(\underline\Delta_i\) over \(u^0\) (§5.4), admissible under the preference contract of §5.5.

`[DEC]` Coverage/improvement split, default `0.5/0.5`, fixed before round 0.
`[DEC]` The uncertainty estimator is a bootstrap ensemble of \(E\) outcome heads over a shared latent transition; \(E\) is fixed before phase B. Default `E = 5`.

`[LOCKED]` The `wm` selector is always accompanied by `random` and `diversity` selectors run at the same anchors, from the same pools, under the same budget and seeds (§4.0). A `wm`-only round produces no acquisition claim.

### 4.4 Bounded adaptive recollection

The pre-pivot rule forbidding new collection after a missing support cell is retired for this branch. Adaptation across rounds is the method, not post-hoc repair. To preserve a valid comparison:

- the number of rounds and per-round interaction budget are fixed before behavior evaluation;
- the acquisition rule may depend only on training-side rollout histories and model predictions available at that round;
- held-out evaluation rollouts never become training anchors;
- every attempted branch is retained and counted against the interaction budget;
- stopping early because a task is solved does not reallocate its unused budget to another method arm.

---

## 5. Decision-relevant latent world model

### 5.1 State construction

The default implementation retains the LCState components \(E_a,T,U\):

\[
h_t^\ell,\mathrm{KV}_t^\ell
=
\operatorname{PrefixVLM}(o_t,\ell),
\]

\[
\bar z_t^\ell
=
T_\theta\!\left(z_{t-c}^\ell,E_a(u_{t-c,0:c})\right),
\qquad
z_t^\ell
=
U_\theta(\bar z_t^\ell,h_t^\ell).
\]

However, v1.0 does not require the paper's contribution to be a new memory mechanism. If the recurrent carry remains functionally inert, one explicit short visual-action history window may replace it without changing the failure-driven learning claim. This is a single bounded representation choice, not another pooling/interface sweep.

### 5.2 Predictive targets

For candidate \(u^i\), predict

\[
\widetilde z_{t+c}^{\ell,i}
=T_\theta\!\left(z_t^\ell,E_a(u^i)\right),
\]

\[
D_\theta(\widetilde z_{t+c}^{\ell,i})
=
\left(
\widehat{\Delta w}_i,
\widehat{\Delta y}_i^\ell,
\hat r_i^\ell,
\hat V_k^\ell(\widetilde z_{t+c}^{\ell,i}),
\hat p_{\mathrm{succ},i}^\ell
\right).
\]

The value target is policy-indexed:

\[
V_k^\ell(z)
=
\mathbb E_{\pi_k}\!\left[G_t\mid z_t^\ell=z\right].
\]

It is updated after each bounded policy round rather than silently treating branch success, task progress, and policy value as interchangeable labels.

#### 5.2.1 Continuation protocol and policy-indexed value estimator (closes C3)

`[LOCKED]` **Provenance.** Every branch record stores four distinct identities, each as a checkpoint hash: the **proposal** policy that produced the candidate, the **execution** policy identity for the first \(c_{\mathrm{eff}}\) actions (open-loop replay of the stored chunk, recorded as `replay`), the **continuation** policy that ran the closed loop after \(\tau+c_{\mathrm{eff}}\), and the **anchor-generating** policy of the source rollout. Returns are not poolable across continuation policies.

`[LOCKED]` **Estimator: truncated current-policy Monte Carlo with policy-identity masking.** For \(M_{k+1}\), the value loss consumes only rows whose *continuation* policy identity equals \(\pi_k\):

\[
\mathcal L_{\mathrm{value}}
=
\mathbb E_{(z,G)\sim\mathcal D_{k+1}}
\Big[
\mathbb 1\!\left[\mathrm{cont\_policy}=\pi_k\right]\cdot
\big\|\hat V_k^\ell(z)-G^{(H_{\max})}\big\|^2
\Big],
\qquad
G^{(H_{\max})}=\sum_{j=0}^{H_{\max}-1}\gamma^j r_{t+j},
\]

with no bootstrap term at the truncation boundary. The head therefore predicts a **truncated** return, the advantage threshold \(\delta\) is calibrated in those same truncated units, and the truncation bias is named in every report rather than absorbed. Rows from earlier continuation policies stay in \(\mathcal D_{k+1}\) and keep their dynamics, physical-effect, reward and sibling-ranking roles; only \(\mathcal L_{\mathrm{value}}\) masks them out. Re-valuing an old branch under \(\pi_k\) requires executing a fresh continuation, which is charged to the interaction ledger like any other execution.

`[LOCKED]` **Continuation budget.** \(H_{\max}\) is derived, not inherited:

\[
H_{\max}\;\ge\;1.5\cdot q_{90}\big(\text{steps-to-next-milestone}\mid\text{milestone class at the anchor's phase}\big),
\]

measured on the Action-1 calibration panel. Continuation outcomes are recorded at every horizon in a registered set \(\mathcal H\) with \(|\mathcal H|\ge 2\) and \(\max\mathcal H=H_{\max}\), and every yield or advantage readout is reported at all of them. This is the direct instrument fix for §0.1 item 6: at \(\mathcal H=\{60\}\) the pre-pivot protocol could not observe a `pick_up` recovery at all.

`[REOPENED 2026-08-19]` **The V8.0 derivation was complete-case and context-aliased.** Class q90 was estimated only from trajectories in which that milestone was *achieved*; right-censored failures never entered the quantile. On `chain1b`, 28/30 episodes emitted the pick event but only 12/30 placed, so 16 placement times were censored at the cap. On `chain2b` the frozen 209-step cream-pick bound came from 22 cream achievements that were overwhelmingly cream-*first* states, not the modal tomato-done failure state whose 5 occurrences reached cream pick 0/5. Conditioning on \((\text{index},\text{kind},\text{object})\) is therefore insufficient: the conditioning set must be the **achieved-state stratum** of §3.1.1, and the estimator must be censoring-aware (Kaplan–Meier or an explicit lower bound), not complete-case. \(W\), \(H_{\max}\), \(\mathcal H\), the per-class bounds, and the "MC fallback does not fire" conclusion remain reopened after Action 1R HALTed. Action 2P (§15) does not estimate value and therefore uses a bounded deadline-aligned outcome window rather than pretending to reseal these quantities.

`[DEC]` **Registered fallback.** If the affordable \(H_{\max}\) falls below that bound for the selected setting, switch the value channel to FQE/TD(0) on stored continuation transitions with the bootstrap action resampled from \(\pi_k\). The switch must be decided and sealed before any value-model fit; it is not a prerequisite for the value-free coverage pilot of §15 and may not be revisited after model outcomes are seen.

### 5.3 Training data roles

The cumulative model dataset follows the C2 recursion of §4.0, so that the branches collected in round \(k\) are inside the model that follows them:

\[
\mathcal D_0=\mathcal D_{\mathrm{boot}}
=\mathcal D_{\mathrm{demo}}\cup\mathcal D_{\mathrm{nominal,boot}}\cup\mathcal B_{\mathrm{boot}},
\qquad
\mathcal D_{k+1}
=
\mathcal D_k\cup\mathcal D_{\mathrm{nominal},k}\cup\mathcal B_k,
\qquad
M_{k+1}=\operatorname{Fit}(\mathcal D_{k+1}).
\]

- demonstrations and nominal rollouts train posterior state construction, recursive dynamics, reward/value on nominal behavior, and VLA retention;
- failure branches train alternative-action effects, local ranking, and recovery outcomes;
- repeated reference branches estimate irreducible execution noise;
- source-disjoint branch groups validate counterfactual prediction and calibration.

The model objective is

\[
\mathcal L_{\mathrm{WM}}
=
\lambda_{\mathrm{dyn}}\mathcal L_{\mathrm{next\text{-}latent}}
+
\lambda_{\mathrm{phys}}\mathcal L_{\mathrm{physical}}
+
\lambda_{\mathrm{task}}\mathcal L_{\mathrm{progress/reward}}
+
\lambda_V\mathcal L_{\mathrm{value}}
+
\lambda_{\mathrm{rank}}\mathcal L_{\mathrm{sibling\ rank}}
+
\lambda_{\mathrm{inv}}\mathcal L_{\mathrm{language\ contract}}.
\]

No loss is promoted by aggregate decrease alone. The model must beat copy-state, no-action, stock-action, and empirical-frequency baselines on held-out branch groups.

### 5.4 Calibration and support

For candidate \(u^i\), define

\[
\hat Q_k(H_t,\ell,u^i)
=
\hat r_i^\ell
+
\gamma^c\hat V_k^\ell(\widetilde z_{t+c}^{\ell,i}),
\]

and relative advantage

\[
\widehat\Delta_i
=
\hat Q_k(H_t,\ell,u^i)
-
\hat Q_k(H_t,\ell,u^0).
\]

`[LOCKED]` The epistemic scale is computed on the **paired difference**, not on the candidate alone, so that reference uncertainty and the candidate/reference covariance both enter. With a bootstrap ensemble \(e=1,\ldots,E\) sharing the latent transition,

\[
\hat\sigma_i^2
=
\operatorname{Var}_{e}\!\Big[\hat Q_k^{(e)}(H_t,\ell,u^i)-\hat Q_k^{(e)}(H_t,\ell,u^0)\Big]
\;=\;
\hat\sigma^2(u^i)+\hat\sigma^2(u^0)-2\widehat{\operatorname{Cov}}\big(u^i,u^0\big),
\]

evaluated with the *same* ensemble member on both sides of the difference. A candidate-only scale is not permitted anywhere in v1.0. The conservative score is

\[
\underline\Delta_i
=
\widehat\Delta_i-\beta\hat\sigma_i.
\]

The confidence coefficient \(\beta\), the minimum improvement margin \(\delta\), and the local-support region are calibrated on source-disjoint executed sibling groups. An unsupported or uncalibrated candidate is eligible for acquisition, not for an unexecuted policy target.

### 5.5 One preference contract for both channels (closes C5)

`[LOCKED]` Verified corrections and model-generated targets are admitted by **the same ordering**. The scalar \(\hat Q\) never admits a candidate; it only ranks candidates that the ordering has already admitted. This is what keeps the trained success and safety heads inside the decision.

**Registered components, in fixed priority order.**

| rank | component | direction | source |
|---|---|---|---|
| 1 | `dmg` irreversible-damage indicator | non-inferiority required | safety head / monitor |
| 2 | `succ` task success within \(H_{\max}\) | higher better | task automaton |
| 3 | `Δp` valid progress gain / unresolved-set contraction | higher better | task automaton |
| 4 | `ttm` steps to next milestone | lower better, defined only when attained | task automaton |
| 5 | `G^{(H_max)}` truncated continuation return | higher better | reward integration |

**Executed channel (verified correction).** With \(R\) reference repeats of \(u^0\), let the reference be the *best* repeat under the same ordering — the conservative choice, because it makes a correction harder to certify. Then \(u^i\) is a verified correction iff

\[
\mathrm{dmg}_i\le\mathrm{dmg}_0^{\mathrm{best}}
\quad\text{and}\quad
\big(\mathrm{succ}_i,\Delta p_i,-\mathrm{ttm}_i,G_i\big)
\;>_{\mathrm{lex}}\;
\big(\mathrm{succ}_0^{\mathrm{best}},\Delta p_0^{\mathrm{best}},-\mathrm{ttm}_0^{\mathrm{best}},G_0^{\mathrm{best}}\big),
\]

where a difference counts at rank 4 or 5 only if it exceeds the replay-noise tolerance measured from the \(R\) repeats. Immediate object distance is not a component and cannot certify a correction.

**Unexecuted channel (model-generated target).** For every component \(j\), the model produces the paired difference \(\hat\delta_j\) and its ensemble sd \(\hat\sigma_j\) by the same difference-first estimator above, giving \(\underline\delta_j=\hat\delta_j-\beta\hat\sigma_j\) and \(\overline\delta_j=\hat\delta_j+\beta\hat\sigma_j\). Candidate \(u^i\) is admissible as a model target iff **all** of:

1. **safety:** \(\overline\delta_{\mathrm{dmg}}\le 0\) — predicted damage is not worse even at the pessimistic end;
2. **conservative lexicographic gain:** let \(j^\star\) be the highest-priority component whose \(|\hat\delta_j|\) exceeds its registered noise floor; require \(\underline\delta_{j^\star}>\delta_{j^\star}^{\min}\) and \(\overline\delta_{j}\) non-inferior for every \(j<j^\star\);
3. **support:** the anchor and the candidate lie inside the calibrated local support of \(M_k\);
4. **ranking only:** among admissible candidates, order by \(\underline\Delta_i\); \(\hat Q\) plays no other role.

Exact predicted ties, unsupported anchors, and candidates failing (1)–(3) contribute **zero** model loss — not a small loss, not a down-weighted loss.

`[LOCKED]` \(\delta_j^{\min}\), the per-component noise floors, and \(\beta\) are fixed on source-disjoint calibration data before any target is sealed, and are reported with the resulting admitted-target count. An empty admitted set is a legitimate registered outcome and is reported as such rather than relaxed.

---

## 6. From interaction to policy improvement

### 6.1 Verified correction targets

A branch action becomes a verified correction exactly when it satisfies the executed-channel rule of §5.5: safety non-inferiority against the best of the \(R\) reference repeats, then strict lexicographic gain on \(({\rm succ},\Delta p,-{\rm ttm},G^{(H_{\max})})\) beyond the measured replay-noise tolerance. Immediate object distance is not a component of the ordering and cannot certify a correction.

Verified corrections provide a grounded first-\(c\) flow-matching target. They are available to both the full method and the direct-correction control.

### 6.2 Model-generated targets

At a training-side failure anchor or a new source-disjoint nominal history, sample a fresh candidate pool. A candidate may become a model-generated target only if it is admissible under the unexecuted-channel rule of §5.5 — pessimistic safety non-inferiority, conservative lexicographic gain at the first component that clears its noise floor, and calibrated local support — with \(\underline\Delta_i\) used solely to rank the admissible set. Exact predicted ties, unsupported anchors, and inadmissible candidates produce zero model loss.

This channel tests whether the model generalizes beyond direct imitation of executed corrections. It must not reuse the held-out execution outcome of the candidate before the policy target is sealed.

### 6.3 Policy objective

The round-\(k\) policy update is

\[
\begin{aligned}
\mathcal L_{\mathrm{policy}}^{(k)}
=\;&
\lambda_{\mathrm{demo}}\mathcal L_{\mathrm{demo\text{-}FM}}
+
\lambda_{\mathrm{verified}}\mathcal L_{\mathrm{verified\text{-}FM}}
+
\lambda_{\mathrm{model}}\mathcal L_{\mathrm{model\text{-}FM}}\\
&+
\lambda_{\mathrm{suffix}}\mathcal L_{\mathrm{suffix\text{-}trust}}
+
\lambda_{\mathrm{ret}}\mathcal L_{\mathrm{retention\text{-}trust}}.
\end{aligned}
\]

Only the first \(c\) actions receive branch/model-derived credit because the policy replans afterward. The suffix remains under stock trust, and ordinary demonstrations retain full-chunk flow matching.

`[LOCKED]` The policy update begins from \(\pi_k\), not from scratch, and trains the same narrow flow boundary across all arms. A matched `zero-state` control isolates generic action-head PEFT from useful world-state conditioning.

### 6.4 Deployment contract

After the update,

\[
\pi_{k+1}
=
\operatorname{Update}(\pi_k,M_{k+1},\mathcal B_k),
\]

the next interaction and evaluation rollouts use the modified full-prompt `N=1` VLA. Candidate search, ensembles, privileged task predicates, and snapshot restoration are training-time mechanisms only.

---

## 7. Experimental design and attribution

### 7.1 Mechanism-development benchmark

The near-zero-success five-task LoHo suite is not the first mechanism-development benchmark. If the current policy cannot reach relevant late states and no supported alternative succeeds, additional failure interaction produces more negative data but no policy-improvement target.

`[LOCKED]` Select a medium-horizon task set on a frozen calibration panel such that stock \(\pi0.5\) has nonzero competence and nontrivial headroom. The initial target range is `[EST]` 20–70% success. Freeze the selected tasks, instructions, acquisition seeds, evaluation seeds, and interaction budget before running any improvement arm.

`[LOCKED 2026-08-20]` **Goal-drift guardrail.** The success band, task count, deadline, and anchor-admissibility rules qualify a measurement setting; they are not the research objective or a paper contribution. An instrument HALT supplies no positive or negative evidence about the coverage, acquisition, policy-improvement, or continual-improvement claims unless the corresponding intervention was actually executed. A small, explicitly scoped coverage pilot may precede the two-task confirmatory benchmark, but it cannot inherit the latter's claim scope.

Existing Chain3/4/5 or controlled-length LoHo variants are preferred because the repository already provides composite instructions, snapshot restoration, task automata, and failure localization. The full five-task LoHo suite becomes a post-mechanism stress test.

### 7.2 Required arms (closes C6, part 1)

Three round-0 **collections** are executed at the common \(\pi_0\), common anchors \(\mathcal A_0\), common pools \(\mathcal P_0\), common seeds, and equal executed-step budgets, differing only in selector: \(C_{\mathrm{rand}}\), \(C_{\mathrm{div}}\), \(C_{\mathrm{wm}}\). Arms are then defined over those collections.

| id | arm | collection | policy init | policy supervision | isolates |
|---|---|---|---|---|---|
| A0 | stock \(\pi0.5\) | none | — | none | pretrained VLA baseline |
| A1 | one-shot nominal FT | none | \(\pi_0\) | demo/nominal only | does generic fine-tuning explain any gain? |
| A2 | failure-random + correction | \(C_{\mathrm{rand}}\) | \(\pi_0\) | verified corrections | targeted interaction without a learned model |
| A2d | failure-diversity + correction | \(C_{\mathrm{div}}\) | \(\pi_0\) | verified corrections | is `wm` beating *random* or beating *undirected search*? |
| A3 | WM acquisition + correction | \(C_{\mathrm{wm}}\) | \(\pi_0\) | verified corrections | does the model acquire corrections more efficiently? |
| A4 | failure-random + model targets | \(C_{\mathrm{rand}}\) | \(\pi_0\) | verified + model-generated | does extra model training alone explain the gain? |
| A5 | full method | \(C_{\mathrm{wm}}\) | \(\pi_0\) | verified + model-generated | complete mechanism |
| A6 | zero-state / matched-boundary control | \(C_{\mathrm{wm}}\) | \(\pi_0\) | identical to A5, world state replaced by a constant of matched dimension | is the predictive state necessary, or is this action-head PEFT? |
| A7 | delayed one-shot, matched total data | A5's round-0 **and** round-1 branches, applied at once | \(\pi_0\) | verified + model-generated | is iteration doing anything a single larger update would not? |

A2d is required by the acquisition claim: without it, a `wm`-beats-`random` result cannot be distinguished from "any non-random search beats random". A6 is required whenever state conditioning is claimed; it shares A5's data and differs only in the state input, with identical learning rate, step count, credit mask, and schedule. A7 is required only if the continual-improvement claim (ladder item 4) is retained, and it is the only arm permitted to receive another arm's branches — declared here as an intended, symmetric part of the comparison.

The two correction-only arms (A2, A3) are indispensable. If the full method does not beat direct imitation of its verified corrections, the world model has not demonstrated a policy-improvement contribution beyond data collection.

### 7.2.1 Data ownership and role sealing (closes C6, part 2)

`[LOCKED]` The three permitted sharing patterns of C6 are resolved as follows, and no other sharing exists.

| object | owner | shared? | visible to selector before sealing? | may enter policy loss? |
|---|---|---|---|---|
| \(\mathcal D_{\mathrm{demo}}\), \(\mathcal D_{\mathrm{nominal,boot}}\) | all arms | yes, declared | n/a | yes (A1 and rehearsal in all arms) |
| \(\mathcal B_{\mathrm{boot}}\), \(M_0\) | all arms | yes, declared: part of every compared method | outcomes visible only after phase B closes | **no** — phase B never trains a policy |
| \(C_{\mathrm{rand}}\) round-0 branches | A2, A4 | yes, between those two arms only | no | yes |
| \(C_{\mathrm{div}}\) round-0 branches | A2d | no | no | yes |
| \(C_{\mathrm{wm}}\) round-0 branches | A3, A5, A6 | yes, between those three arms only | no | yes |
| round-\(k\ge1\) branches | the generating arm alone | **no** | no | yes, for that arm |
| A5 round-0 + round-1 branches | A7 | yes, by explicit design | no | yes |
| behavior-panel episodes | none | never | never | **never** |
| calibration episodes (Action 1) | none | never | never | **never** |

Arms that share a collection share it *identically*, including its interaction denominator, and this is stated in every table that reports interaction efficiency. Matched-random is therefore **an arm's own selector**, never a hidden shadow of a method arm; there is no "executed but hidden" category in v1.0.

`[LOCKED]` **Role sealing.** Each source episode carries a role \(\in\){`calibration`, `bootstrap`, `model_train`, `model_calib`, `selector_assess`, `verified_correction`, `model_target_seal`, `behavior_eval`}. Branches inherit `(source_episode_id, anchor_id)` as replay ancestry, so a branch can never enter a role its ancestor is excluded from. Roles are partitioned within each round before collection. A role transition across rounds requires a prospective written declaration before that round starts; a sealed model-target outcome may be revealed only after the corresponding policy target is fixed.

### 7.3 Round structure

Phase B runs once, before any arm exists: deploy \(\pi_0\) on the bootstrap acquisition seeds, extract anchors, collect \(\mathcal B_{\mathrm{boot}}\) under coverage-only selection, fit and calibrate \(M_0\). No policy is updated.

Then, for each arm and round \(k\):

1. deploy \(\pi_k\) on an acquisition-only rollout panel;
2. extract failure anchors using the frozen task monitor (§3.1.1);
3. build the candidate pool per anchor — common across arms at round 0;
4. select with \(S_k\), which may consume only \(M_k\) (§4.0), and seal the selection before revealing outcomes;
5. execute the arm's fixed number of branches and continuations, charging every step (§7.6);
6. update the world model on \(\mathcal D_{k+1}\);
7. seal model-generated targets before any corresponding assessment execution;
8. update the policy from \(\pi_k\);
9. evaluate \(\pi_{k+1}\) on a behavior panel never used for acquisition, calibration, or training.

The primary experiment uses two improvement rounds. `[DEC]` A third round is permitted only if fixed before round 0; it may not be added after inspecting the round-2 curve.

### 7.4 Primary metrics

- task-balanced terminal success after each round;
- success-versus-environment-interaction curve and area under that curve;
- number of verified positive corrections per executed branch;
- held-out candidate-ranking accuracy, top-1 regret, and calibration error;
- failure-frontier movement by task and milestone;
- standard-task retention and irreversible-damage rate.

Offline latent prediction and outcome-head metrics are supporting evidence, not substitutes for policy behavior.

### 7.5 Required causal conclusions

The full claim requires all of the following:

1. failure-anchored interaction adds held-out action-effect/outcome support relative to demonstrations;
2. model-guided acquisition finds verified improvements more efficiently than matched random acquisition;
3. the final `N=1` policy improves over stock and correction-only controls;
4. model-generated targets improve over the same model-guided acquisition with verified corrections only;
5. the second bounded round improves or maintains behavior while moving the failure frontier, rather than merely memorizing first-round anchors.

### 7.6 Interaction accounting (closes C7)

`[LOCKED]` The interaction denominator is counted in **environment steps** and is the sum of exactly these terms:

\[
N_{\mathrm{env}}
=
n_{\mathrm{acq\_rollout}}
+n_{\mathrm{anchor\_search}}
+n_{\mathrm{ref\_repeat}}
+n_{\mathrm{branch\_exec}}
+n_{\mathrm{branch\_cont}}
+n_{\mathrm{assess}} .
\]

- `acq_rollout` — deployment rollouts run to find failures;
- `anchor_search` — tier-2 retrospective continuations (§3.1.1);
- `ref_repeat` — the \(R\) repeated executions of \(u^0\);
- `branch_exec` — the \(c_{\mathrm{eff}}\) executed steps of every selected branch, including branches that fail, are dropped, or are later found invalid;
- `branch_cont` — every closed-loop continuation step;
- `assess` — any execution used to certify a selector or to reveal a sealed model target.

Snapshot restoration is **not** an environment step. Compute spent sampling or scoring candidates that were never executed is reported separately as \(n_{\mathrm{prop\_samples}}\) and GPU-seconds, never folded into \(N_{\mathrm{env}}\).

`[LOCKED]` Budgets are per arm per round, fixed before collection. Unused budget is never reallocated across arms, tasks, or rounds; an arm that finishes early keeps its unspent budget unspent. Overrunning a budget halts that arm rather than extending it.

**Ledger schema** (`interaction_ledger.jsonl`, one row per executed segment):

```json
{"round": 0, "phase": "boot|round", "arm": "A5", "role": "bootstrap",
 "term": "branch_cont", "task": "chain2_lr2", "source_episode_id": "s3301_t2",
 "anchor_id": "a0007", "branch_id": "b0031", "seed": 3301,
 "policy_hash": "sha256:...", "cont_policy_hash": "sha256:...",
 "n_steps": 200, "c_eff": 10, "termination": "committed",
 "replay_ancestry": ["s3301_t2", "a0007"]}
```

Every reported efficiency number cites the ledger sum it divides by, and every arm-comparison table states which arms share a denominator by construction (§7.2.1).

---

## 8. Promotion, falsification, and interpretation

The new branch should fail for interpretable reasons rather than returning to unrestricted diagnosis.

- **No outcome variation in the proposal pool:** the task/proposal generator lacks recovery support. Change the development setting or action proposal mechanism before drawing a conclusion about the world model.
- **Outcome variation exists, but the model cannot predict held-out siblings:** the dynamics/outcome model is inadequate for the acquired coverage.
- **The model predicts siblings but model-guided acquisition does not beat random:** the acquisition score or uncertainty estimate provides no interaction-efficiency benefit.
- **Verified-correction arms improve equally:** the result supports failure-driven corrective imitation, not a model-based policy-improvement claim.
- **WM acquisition improves correction efficiency, but model-generated targets add nothing:** retain the model-based active-acquisition claim and drop the stronger imagined-target claim.
- **Model-generated targets improve offline ranking but hurt behavior:** calibration/support gating is insufficient; do not promote the world-model teacher.
- **Round 1 improves but round 2 does not move the frontier:** the method is one-shot adaptation, not continual improvement.
- **The full method wins while reset/zero-state controls match it:** policy PEFT or additional corrections explain the gain; the predictive state is not yet necessary.
- **The full method wins and model/random/correction controls do not:** freeze the method and evaluate the full LoHo stress test and standard LIBERO retention.

No outcome authorizes an unconstrained scorer, pooling, layer, rank, seed, or lambda sweep inside the registered experiment.

---

## 9. Migration from the current repository

### Reuse

- \(\pi0.5\) full-prompt runner, PrefixVLM tap, action normalization, and candidate sampler;
- snapshot/restore, replay-fidelity checks, common-random-number execution, and video/trace recording;
- `LCState`, \(E_a\), \(T\), \(U\), next-latent prediction, and separate physical/task/value heads;
- zero-init AdaRMS state injection, stock-initialized action output head, first-\(c\) credit mask, suffix trust, and demonstration retention;
- task automata, ordered-progress metrics, source-disjoint splits, artifact lineage, and public LoHo evaluator;
- matched-random ledgers and outcome-blind target sealing.

### Freeze as historical pre-pivot (`v0.9`) machinery

- the fixed V7.4/V7.6 acquisition budgets and the `mask-and-name without recollection` rule — and, specifically, any budget not derived from a measured completion-step distribution;
- treating one LCWM checkpoint plus one policy fine-tune as the complete method;
- T1–T5 interface tests as the central scientific gate;
- the learned-query pooling fallback as a research contribution;
- mandatory action-by-goal reversal support before testing the basic failure-driven loop;
- V7.4 C/D/E as the mainline confirmation path;
- the claim that improving the VLA necessarily requires transforming its internal representation into a novel recurrent state.

### New modules required

1. round-indexed rollout and failure-anchor extractor;
2. common candidate-pool builder shared across acquisition arms;
3. interaction-budgeted branch selector with random, coverage, and conservative-improvement modes;
4. cumulative round-aware dataset and policy-indexed value labels;
5. bootstrap uncertainty/calibration heads;
6. round-aware policy updater starting from \(\pi_k\);
7. improvement-curve and failure-frontier reporter.

---

## 10. Immediate implementation sequence

1. **Archive the pre-pivot endpoint.** Preserve the committed `v0.9` framework, the V7.3–V7.7 records, code, and artifacts as the one-shot LC-Flow branch. Do not reinterpret its negative results as v1.0 evidence.
2. **Test the coverage mechanism directly (§15).** On a bounded, explicitly deadline-constrained setting, restore earlier histories from prospective failed source rollouts and execute outcome-blind VLA-supported sibling pools. Ask first whether matched branches create reproducible action-effect/outcome variation and verified improvements; do not train a model or policy in this pilot.
3. **Build the replicated bootstrap acquisition object (§13.2).** After the pilot advances, register a source-disjoint second-setting replication and the full two-task handoff. Only a replicated collection with both better-than-reference and non-improving siblings becomes eligible to train \(M_0\).
4. **Train the minimal counterfactual model.** Reuse the current latent model but evaluate it on source-disjoint sibling ranking, value calibration, and effect prediction. Do not begin with another representation/interface sweep.
5. **Test acquisition before full policy learning.** Under equal branch budgets, compare model-guided selection with random and diversity-only selection by the rate and magnitude of verified improvements found.
6. **Run the first policy update.** Compare direct correction, WM acquisition plus correction, and the full verified-plus-model-generated objective from matched initialization.
7. **Run round 2.** Collect failures from each updated policy, update its model and policy under the same budget, and measure success/interactions and failure-frontier movement.
8. **Promote only after a mechanism win.** A replicated development win opens full five-task LoHo and standard LIBERO retention. Otherwise follow the interpretation table in §8 and revise one named component.

---

## 11. Current locked and open decisions

### Locked

- broad problem formulation: language- and action-conditioned latent dynamics for decision-relevant reward/value and policy improvement;
- pretrained VLA as action prior, initialization, and final deployed `N=1` policy;
- failures select where to expand state-action coverage;
- alternative-action execution supplies the intervention evidence;
- both positive and negative branches are retained;
- policy improvement is evaluated behaviorally after every bounded round;
- direct verified-correction and matched-random controls share data and interaction budgets;
- inverse RL, pixel-video prediction, arbitrary-action planning, and deployment-time best-of-N remain outside scope;
- full LoHo is a later stress test, not the first mechanism-development environment.

Closed by Action 0 on 2026-08-17, and now locked:

- version lineage `v0.9` (pre-pivot) / `v1.0` (this branch), experiment series `V8.x` (header);
- executed-action causal contract on \(u_{\mathrm{exec}}\) and \(c_{\mathrm{eff}}\) (§2.1, C1);
- bootstrap phase \(\mathcal B_{\mathrm{boot}}\to M_0\), the \(\mathcal D_{k+1}\) recursion, and selector legality (§4.0, C2);
- current-policy identity masking, truncated-return semantics, and the multi-horizon reporting rule (§5.2.1, C3); the numeric \(H_{\max}\) and MC-versus-FQE choice are reopened below;
- two named anchor kinds, `stall_onset` and `earliest_unrecoverable`, and the ban on using the second name without a tier-2 execution (§3.1.1, C4);
- one preference contract — lexicographic dominance with difference-first uncertainty — for both the executed and unexecuted channels (§5.5, C5);
- the nine-arm matrix and the data-ownership/role-sealing table (§7.2, §7.2.1, C6);
- the six-term interaction denominator and ledger schema (§7.6, C7);
- the instrument rule that no yield bar may be evaluated at an underived budget or at a single horizon (§0.1).

### Open but bounded

- `[DEC]` recurrent LCState versus one explicit short history window if the former remains inert;
- `[REOPENED 2026-08-20]` exact two-task confirmatory development panel — the historical `V8.0` fixed-deadline PASS remains recorded, but it no longer supplies a valid V8.1 handoff (§13.1, §14.5);
- `[DEC]` candidate-pool size \(N\), executed branch count per anchor, ensemble size \(E\) (default 5), reference repeats \(R\) (default 3), and the `wm` coverage/improvement split (default 0.5/0.5) — all fixed before phase B;
- `[REOPENED 2026-08-19]` MC versus FQE for the value channel, together with \(W\), \(H_{\max}\), \(\mathcal H\) and the per-class bounds — the V8.0 derivation was complete-case and context-aliased, and Action 1R HALTed before a handoff was sealed (§5.2.1, §14);
- `[DEC]` whether the first paper claim stops at model-guided active acquisition or also includes unexecuted model-generated targets;
- `[EST]` \(W\), value-model \(\mathcal H\), \(H_{\max}\), and the attainable positive-branch rate. Action 2P estimates only local branch support under its own deadline-aligned readouts; it does not settle the value estimator or the two-task stock-success range.

---

## 12. Executable registration

### 12.1 Round pseudocode

```text
# ---------- Phase B: bootstrap (once, no policy update) ----------
freeze(tasks, instructions, episode_horizon, W, H_set, H_max, R, N, E, budgets, seeds)   # §13.1
A_boot = []
for seed in SEEDS.bootstrap:
    roll = deploy(pi_0, task, seed, role="bootstrap")        # charge n_acq_rollout
    A_boot += extract_anchors(roll, W, tier=1)               # §3.1.1
for anchor in A_boot:
    P = build_pool(anchor, pi_0, N)                          # §4.1, dedup on u_exec
    B_boot += execute(anchor, repeats(u0, R) + farthest_point(P) + random(P))
M_0 = fit(D_demo + D_nominal_boot + B_boot)                  # §5.3
beta, delta, support = calibrate(M_0, holdout_source_disjoint(B_boot))   # §5.4-5.5
assert selector_legality(M_0, round=0)                       # §4.0

# ---------- Round k = 0, 1 ----------
for k in (0, 1):
    if k == 0:
        A_k = common_anchor_panel(pi_0, SEEDS.acq[0])        # identical across arms
        P_k = {a: build_pool(a, pi_0, N) for a in A_k}       # identical across arms
        for sel in ("random", "diversity", "wm"):            # three collections
            S = select(sel, A_k, P_k, M_0, budget)           # sealed before execution
            C[sel] = execute(S)                              # charge §7.6 terms
    else:
        for arm in ARMS_WITH_POLICY:                         # failures now diverge
            A_k[arm] = extract_anchors(deploy(pi_1[arm], SEEDS.acq[1]), W, tier=1)
            P_k[arm] = build_pools(A_k[arm], pi_1[arm], N)
            C[arm]   = execute(select(arm.selector, A_k[arm], P_k[arm], M_1[arm], budget))
    for arm in ARMS:
        B_k[arm]   = C[arm.collection]
        D_next     = D[arm] + D_nominal_k[arm] + B_k[arm]    # §4.0
        M_next[arm]= fit(D_next)                             # value loss masked to pi_k rows
        T_ver      = verified_corrections(B_k[arm])          # §5.5 executed channel
        T_mod      = seal(model_targets(M_next[arm], fresh_pools))   # §5.5 unexecuted channel
        pi_next[arm] = update(pi_k[arm], T_ver, T_mod)       # §6.3, credit masked to c_eff
    evaluate(pi_next, panel=SEEDS.behavior, mode="full_prompt_N1")
report(success_vs_N_env, verified_per_branch, top1_regret, calibration, frontier)
```

### 12.2 Artifact schemas

One JSON object per record; every record carries `schema_version` and the producing script's source hash.

```json
// anchor.json
{"anchor_id": "a0007", "kind": "stall_onset|earliest_unrecoverable",
 "task": "chain2_lr2", "instruction": "...", "source_episode_id": "s3301_t2",
 "round": 0, "phase": "boot", "role": "bootstrap", "seed": 3301, "step": 240,
 "policy_hash": "sha256:...", "milestone_index": 1, "predicate_bits": [1,0,0],
 "last_progress_step": 188, "stall_window": 210,
 "snapshot": {"sim_state": "…npz", "controller": "…npz", "obs_window": "…npz",
              "action_history": "…npz", "kv_recipe": {...}},
 "tier2": null}

// candidate.json
{"anchor_id": "a0007", "cand_id": "c012", "proposal_policy_hash": "sha256:...",
 "u_full_ref": "…npz", "u_exec_quant": "blake2b:...", "family_id": "f003",
 "sampler": "pi_k|pi_0|perturb", "sampler_params": {...}, "executed": true}

// branch.json
{"anchor_id": "a0007", "branch_id": "b0031", "cand_id": "c012", "is_reference": false,
 "c_eff": 10, "executed_mask": [1,1,1,1,1,1,1,1,1,1], "termination": "committed",
 "exec_policy": "replay", "cont_policy_hash": "sha256:...",
 "delta_w": {...}, "delta_y": {...}, "r": 0.0,
 "outcomes": {"60": {"succ": 0, "dp": 0, "ttm": null, "G": 0.0, "dmg": 0},
              "200": {"succ": 1, "dp": 1, "ttm": 71, "G": 0.83, "dmg": 0}},
 "crn_seed": 3301, "replay_fidelity": {"max_abs_dev": 1.2e-6, "pass": true},
 "verified_correction": true, "selector": "wm", "arm_owners": ["A3","A5","A6"]}

// selector_seal.json   (written BEFORE outcomes are revealed)
{"round": 0, "selector": "wm", "model_hash": "sha256:...", "beta": 1.5,
 "delta_min": {"succ": 0.05, "dp": 0.05, "ttm": 5.0, "G": 0.02},
 "selected": [["a0007","c012"], ...], "budget_steps": 68000, "sealed_at_git": "..."}

// round_manifest.json
{"round": 0, "arm": "A5", "collection": "C_wm",
 "pi_in": "sha256:...", "pi_out": "sha256:...", "M_in": "sha256:...", "M_out": "sha256:...",
 "datasets": {"D_in": "sha256:...", "B_k": "sha256:...", "D_out": "sha256:..."},
 "ledger": "interaction_ledger.jsonl", "N_env": 91340,
 "n_prop_samples": 4200, "gpu_seconds": 38100,
 "roles_partition": {...}, "role_transitions_declared": []}
```

`[LOCKED]` PASS for Action 0 is checked mechanically against these schemas: every symbol in \(\pi_0\to\mathcal B_{\mathrm{boot}}\to M_0\to\mathcal B_0\to M_1\to\pi_1\to\mathcal B_1\to M_2\to\pi_2\) maps to a producing stage, an allowed input dataset, a checkpoint hash, and a ledger; no selector consumes a model from its own round; no value label changes policy identity silently; no branch appears under two arms unless `arm_owners` lists both and §7.2.1 declares the sharing.

---

## 13. Promotion and halt contract for the first two stages

The original §13.1–13.2 contract below was written before any `V8.x` outcome was inspected. Paragraphs carrying explicit 2026-08-19/20 dates are retrospective status or prospective successor rules; they do not masquerade as part of the original preregistration.

### 13.1 `V8.0` — benchmark calibration (Action 1)

**Instrument.** Stock π0.5 only, full-prompt `N=1`, `ChainEnv` (evaluator owns reset). No fine-tuning, no reachers, no snapshot restoration, no decomposed prompts in the primary screen.

**Horizon.** Episode length is set as \(250\times n_{\mathrm{subgoals}}\) steps, capped at 990 by the robosuite internal horizon of 1000. This is derived from the V7.7 floor (earliest `pick_up` success at step 66; twelve successes spanning 66–106) with margin for the place phase. It differs from the historical Chain3 setting of 700, so **the historical Chain3 `0/5` is not comparable to this screen** and is not used as its baseline.

**Candidates.** Chain-family tasks at 1, 2, and 3 subgoals, plus any shortened LoHo variant that can be produced without new object assets. The existing five-task LoHo suite enters only as a reference row.

**Panels.** Screen panel: 10 seeds/task. Confirmation panel: 30 seeds/task, disjoint from the screen and fixed before confirmation. Acquisition and behavior-evaluation seed families are reserved now and never executed in `V8.0`.

**PASS** requires, on the confirmation panel, at least two tasks with
- point success in \([0.20,0.70]\),
- Wilson 95% lower bound \(\ge 0.10\) (nonzero competence demonstrated, not assumed), and
- Wilson 95% upper bound \(\le 0.85\) (headroom demonstrated),

and, on those tasks, at least two distinct failure phases observed across seeds, so that failure anchors exist at more than one milestone.

**Also produced, and required before phase B may start:** the milestone-to-milestone completion-step distribution per milestone class, from which \(W\), \(\mathcal H\), and \(H_{\max}\) are computed and frozen, together with the MC-versus-FQE decision of §5.2.1. The derivation sample is restricted to the tasks that actually pass — a budget frozen partly from a rung that is then dropped describes a distribution the mechanism never samples.

`[LOCKED]` **Reproducibility.** \(\pi0.5\) is a flow policy and samples its action chunk, so every calibration, acquisition, and evaluation episode seeds torch, numpy, and CUDA from the episode seed. Unseeded, the same seed gave terminal steps 138 and 134 on two runs of identical code. A panel that cannot be reproduced is not a calibration.

`[POST-HOC STATUS 2026-08-20; stored verdict unchanged]` V8.0 remains a historical PASS under its registered fixed-deadline rule. Stage 1R.1 showed that `chain1b_lr2` succeeds on 20/20 episodes by 500 steps while 8/20 finish by its registered `L=250`; that artifact therefore measures finite-deadline performance and cannot by itself support an intrinsic capability-headroom claim. `chain2b_lr2` produced 31/40 at `L=500` on the 1R.1 panels (51/70 when descriptively pooled with V8.0), above the original screen band. These observations withdraw the old artifact's eligibility as a complete V8.1 handoff; they do not rewrite its preregistered result or test any method claim.

`[PROSPECTIVE GUARDRAIL 2026-08-20]` Endpoint semantics are fixed before a new panel. A future **capability/headroom** benchmark must prospectively bound late conversion at its horizon; a **deadline-constrained** experiment may instead treat non-completion by a meaningful frozen deadline as failure and must report later conversions separately. “Terminal” and “irreducible” are never inferred from a finite right-censored rollout. Terminality is not the scientific endpoint: the coverage endpoint is same-anchor counterfactual action-effect/outcome variation, including both verified improvement and non-improving siblings.

**HALT** if no task set clears the band. In that case change task difficulty — object count, distractor set, initial-state region — before building any world model. Do **not** compensate with privileged late-state reachers inside the primary mechanism benchmark; the reacher's seed sensitivity at depth 3 (§0.1) is a second reason to keep it out of the benchmark definition.

### 13.2 `V8.1` — bootstrap acquisition (Action 2)

Runs only on a `V8.0` PASS, with every quota below frozen before collection.

**Quotas** `[DEC]`, to be filled from the `V8.0` panel and then locked: tasks selected \(\ge 2\); anchors per task; candidate-pool size \(N\); executed branches per anchor \(= R\) reference repeats \(+\) \(n_{\mathrm{div}}\) diversity \(+\) \(n_{\mathrm{rand}}\) matched-random; continuation budget \(H_{\max}\) with reporting set \(\mathcal H\).

**PASS** requires all of:
1. reference replay fidelity passes on \(\ge 95\%\) of reference repeats, with the measured replay-noise tolerance reported and used as the §5.5 noise floor;
2. on \(\ge 2\) selected tasks, at least one verified correction and at least one non-improving sibling at the same anchor — i.e. realized outcome *variation*, not merely realized outcomes;
3. \(\ge 20\%\) of anchors have at least one strictly dominating sibling, and \(\ge 40\%\) of anchors show variation in any registered outcome component, both evaluated at \(H_{\max}\) **and** reported at every horizon in \(\mathcal H\);
4. source-disjoint groups are reserved and untouched for model calibration and for the §12.2 selector seal.

**HALT** if the proposal pool produces no realized outcome variation, or no supported positive correction, at the derived budget. Then change the benchmark or the proposal mechanism — not the budget, and not the bar. A bar may be re-derived only from a *new* independent measurement of the completion-step distribution, registered before the re-run, exactly as `V8.0` derives it here.

`[LOCKED]` Neither stage may report an aggregate rate without the per-task and per-milestone-class breakdown beside it. The V7.6 record shows an aggregate 16.7% that concealed `recovery` 5/5 against `pick_up` 0/14; that concealment is what the breakdown requirement exists to prevent.

The immediate priority is to resolve the benchmark and interaction-support decisions through one small round-0 collection. Architecture refinements are subordinate until that collection demonstrates that the new branch can actually create the counterfactual outcome variation its claim requires.

`[STATUS 2026-08-20; not part of the original preregistration]` The historical V8.0 artifact no longer supplies an accepted two-task handoff, so the §13.2 stage is paused. Action 2P in §15 is the immediate bounded interaction-support pilot. It has a different single-task estimand, cannot be reported as §13.2 PASS, and does not authorize model or policy training.
---

## 14. Action 1R / V8.0R — repairing the branch instrument

Added 2026-08-19 after the V8.0 audit. §§14.1–14.4 describe the historical sealed Action 1R contract; §14.5 records its outcome and dated interpretation. As written, §13.1's behavioral gate was taken to stand — `chain1b_lr2` 12/30 and `chain2b_lr2` 20/30 at the frozen 250/500-step deadlines, hash-bound. Stage 1R.1 later classified `chain1b` as materially deadline-sensitive and `chain2b` as indeterminate under the conservative rule, reopening the derived branch instrument — the anchor rule, budget derivation, and value-estimator choice. It executed no alternative-action coverage test, world model, or policy update.

### 14.1 What was wrong, stated once

| defect | evidence | consequence |
|---|---|---|
| scalar-phase aggregation | `chain2b` `phase = 2` covered 5 tomato-done/cream-unresolved and 2 cream-done/tomato-unresolved episodes | one number pooled mutually exclusive remaining problems; anchors, value conditioning and frontier reports would all inherit it |
| ordered-prefix assumption | all 20 `chain2b` successes ran cream→tomato; canonical listed order 0/20 | an ordered-prefix frontier reports *no movement* on every success the task actually produces |
| deadline blindness | 19/30 `chain1b` episodes end exactly at the cap; 16 post-pick failures had 20–60 steps left | the natural last-progress anchor leaves 10–50 official continuation steps against a 71-step place bound |
| complete-case q90 | 16 `chain1b` placement times censored at the cap; the `chain2b` cream bound came from cream-first states | \(W\), \(H_{\max}\), the per-class bounds and the MC conclusion are not identified for the states Action 2 would sample |

### 14.2 The repaired contract

Implemented in `lcwm/v08r_contract.py` and sealed by `scripts/register_v080r.py`; the executable definition is authoritative and this section is its description.

- **Artifact identity** is `anchor_id + snapshot content hash`. Equal content hash means the same anchor regardless of the episode that reached it; equal id with unequal content hash is a provenance failure.
- **Stratum identity** is `task + ever_achieved + current_valid + damaged`, as sets. `actionable` and `unresolved` sets are stored explicitly. `step` and `time_to_go` are reported covariate bins and never identity, so two episodes at the same physical state with different clocks still aggregate together.
- **Frontier movement** is contraction of the unresolved set, or success. Never an ordered-prefix increase.
- **Deadline arithmetic and observable stall** are as locked in §3.1.1 above.
- **Officiality**: success, correction, value labels and official environment steps exist only for `step ≤ L`. Beyond \(L\) an observation is retained and labelled `quarantined_post_deadline` — the 1R.1 extended horizons exist precisely to observe it — but may not enter any official number.
- **Decision rule**: exact one-sided Clopper–Pearson at \(\alpha=0.05\), materiality \(0.15\), panels of 20 then a conditional 20, and **no third panel**. A 0/20 result is reported as "no effect detected above 13.9%", never as "no effect".
- **Underfill** at the frozen 40-seed source cap is `STRATUM_NOT_IDENTIFIED`. Substituting an easier state or raising the cap is prohibited.
- **Interaction cap** is hard and per stage/task, totalling 113,030 environment steps for all of Action 1R, every line item derived from the protocol it funds (the 1R.2 continuation budget is sized from the registered `[60,120,240,400]` readout schedule, not guessed). An overrun halts the stage; unused budget is never reallocated.
- **The success band is closed** \([0.20,0.70]\). The `(0.2, 0.7)` rendering in V8.0 generated text is a display bug; neither confirmation rate lies on a boundary, so the PASS is unaffected.

### 14.3 Ordering, and why 1R.1 comes first

Applying §3.1.1's admissibility rule to already-collected V8.0 data gives **0/16 admissible `chain1b` anchors** and 5/8 `chain2b` ones. `chain1b`'s stratum is therefore predicted to fail 1R.2 at the current deadline, which is exactly why the deadline-sensitivity panel runs before the feasibility panel.

`[LOCKED]` Action 1R produces a new content-addressed handoff artifact and **must not mutate** `BENCHMARK_FROZEN.json`. An immutable artifact is superseded by a successor that cites it, never edited.

### 14.4 PASS / HALT

**PASS** requires all of: both frozen tasks retain at least one admissible stratum; replay and provenance pass; unordered states are never cross-pooled; official outcomes respect the deadline; the anchor rule and estimator path are identified; and a new immutable V8.1 handoff manifest is sealed.

**HALT** on any of: reserved-seed use, replay/provenance failure, state aliasing, post-deadline official stepping, source-cap underfill leaving either task without an admissible stratum, an unconfirmed horizon change, an unidentified anchor/value contract, or an unsealed interaction budget. On HALT, change exactly one registered component — task/deadline, anchor rule, or estimator path — before any new panel.

### 14.5 Action 1R outcome — HALT (2026-08-19)

Stage 1R.0 sealed (registration `673e8c23`, superseding `d77615a5`); Stage 1R.1 executed on 60 probe and 13 duplicate episodes. Stage 1R.2 was not run and is closed with Action 1R. Interaction: 29,800 of the 113,030-step cap, all role `calibration`; reserved seeds `3200–3399` untouched.

| task | pooled result | verdict |
|---|---|---|
| `chain1b_lr2` | 12/20 late conversions, CP lower 0.3936; success @250/@350/@500 = 0.40/0.95/**1.00** | `MATERIALLY_DEADLINE_SENSITIVE` |
| `chain2b_lr2` | 3/40 late, CP [0.0208, 0.1826]; official 31/40 = 0.775; **6/40 did not succeed by the registered maximum 990** | `INDETERMINATE_TREAT_AS_SENSITIVE` |

**Run status versus promotion status.** Stage 1R.1 mechanically HALTed when `1R.1_duplicate_subset | chain2b_lr2` exceeded its hard cap by 349 steps. Separately, Action 1R could not produce the §14.4 two-task handoff: the observed `chain1b` last-progress anchors leave too little official continuation under that registered rule, while Stage 1R.2 — the source-cap stratum test — never ran. Its 20 completion steps run `228…250 │ 251…282 │ 352` against `L=250`; this establishes a deadline-sensitive completion-time distribution, not that the policy “never fails” or that an earlier counterfactual anchor cannot improve deadline performance.

`[POST-HOC CLARIFICATION 2026-08-20]` The 0/16 versus 5/8 calculation in §14.3 was a prediction about the registered last-progress extractor. Because 1R.2 did not execute, it is not an empirical global claim that earlier anchors or VLA-supported corrections do not exist.

For `chain2b`, 6/40 remained unsuccessful at 990 and are right-censored in **three** exact masks: two zero-progress, three tomato-done/cream-unresolved, and one tomato-plus-cream-pick/cream-place-unresolved. They are not proven terminal or irreducible. The task also sits above the historical band and its late-conversion verdict is conservatively indeterminate.

`[LOCKED]` **Registered overrun, recorded not repaired.** `1R.1_duplicate_subset | chain2b_lr2` was charged 2,849 against a 2,500 cap, over by 349: the cap funded one duplicate subset while the runner scheduled one per panel. The cap was **not** raised; `CAP_LINE` now has no panel-2 duplicate entry so the unfunded activity cannot be scheduled again. Prefix identity matched 10/10 in panel 1 and 3/3 in panel 2 regardless.

`[LOCKED]` **Cumulative spend is derived** by summing every `interaction_ledger.jsonl` on disk, never separately maintained. An aborted run wrote 2,687 ledgered steps that a clean-exit-only counter had lost.

The “change exactly one component” rule remains binding only for a panel presented as an Action 1R comparable rerun. Action 1R is closed. The independent Action 2P below registers a different, deliberately narrower estimand — finite-deadline sibling coverage — and does not reuse Action 1R data, budget, or PASS claim.

---

## 15. Action 2P / V8.1P — finite-deadline sibling-coverage pilot

Added 2026-08-20 after rechecking the central question and claim ladder. **Status: prospective design; execution blocked until a clean, content-addressed registration is independently reviewed and sealed.** This is a subsidiary pilot inside the coverage stage, not an Action 1R rerun and not the two-task V8.1 PASS of §13.2.

### 15.1 Question and claim boundary

The pilot asks one question only:

> At histories drawn prospectively from stock-policy rollouts that miss a prospectively registered experimental horizon, do VLA-supported first-\(c\) sibling actions create reproducible local action-effect/outcome variation, including both finite-horizon improvements and non-improving alternatives?

This tests local feasibility required by the first rung of the claim ladder; it does not complete that rung. The pilot contains no world model, selector comparison, policy update, continual-learning round, or task-generalization claim. A late-converting source episode is still a valid finite-horizon failure; no row is called terminal, irreducible, or a capability failure.

### 15.2 Frozen setting, source panel, and anchor

- **Setting:** `chain1b_lr2`, stock \(\pi_0=\pi0.5\), full prompt, inference `N=1`, horizon \(L=250\). This is a post-calibration choice registered prospectively before Action 2P execution; the estimand is explicitly conditional success/progress under this experimental finite horizon, not an exogenous deployment requirement or unlimited-horizon competence.
- **Sources:** all 20 fresh seeds `3480–3499`, in a presealed order. The old `3200–3399` families remain reserved, `3400–3439` remain spent calibration, and the unexecuted but old-registered `3440–3479` family is retired. All 20 sources run; collection does not stop when the anchor quota is filled.
- **Failure:** a valid source reaches step 250 without task success. A technical-invalid source triggers `TECHNICAL_HALT`, remains charged, and is never replaced or silently counted as a non-failure. Report both unconditional deadline failures \(F/20\) and exact-mask-eligible failures \(E/20\).
- **Anchor:** one fixed `retrospective_deadline_backoff` snapshot at
  \[
  \tau=L-c-H=250-10-80=160.
  \]
  This is a pre-failure history selected because its source later misses the deadline; it is not a stall onset or earliest-unrecoverable point.
- **Primary stratum:** at \(\tau\), the exact event masks have no achieved, current-valid, or damaged event; `pick` is actionable; `pick` and `place` are unresolved. The executable registration binds the task-automaton event IDs. Scalar phase/count is display-only.
- **Selection:** after all 20 sources close, seal the first eight eligible failures in source-seed order. If fewer than eight exist, HALT. A failed restore or replay is not replaced by the ninth failure.

The original source suffix is used only to establish `failure@250`; because that suffix is selected to fail, it is never counted as a reference repeat.

### 15.3 Outcome-blind pool and paired execution

At each of the eight anchors:

1. after all sources and eight anchors are sealed, define \(u^0\) as the **first draw** from the registered hash-derived stock-\(\pi_0\) reference RNG key, with no action-geometry or outcome selection; it is outside the \(N=32\) draws and is never the source's cached action. Draw exactly \(N=32\) raw stochastic \(\pi_0\) alternative chunks from separate presealed keys;
2. deduplicate by the executed first \(c=10\) actions at the registered replay-noise tolerance; require at least 16 unique alternatives;
3. select three farthest-point-diversity alternatives using the registered standardized first-\(c\) action feature, distance, starting point, and tie-break, then draw three matched-random alternatives from the remaining unique pool. The sets are mutually exclusive and exclude \(u^0\);
4. execute \(u^0\) and each of the six alternatives under the same three presealed continuation-RNG keys. Every candidate therefore has three paired comparisons against reference, and repeats remain nested within the anchor rather than inflating \(n\);
5. globally hash-seal all eight anchors' reference/alternative chunks, deduplication, diversity/random selections, execution order, and paired continuation keys before the first branch outcome is visible; then execute. Ignore the unused chunk suffix after the first ten actions and continue with stock \(\pi_0\). Record prefix physical effect at \(c\) and task outcomes after continuation horizons \(\mathcal H_{2P}=\{20,40,80\}\). Gate decisions use only \(H=80\), which ends at deadline 250; \(H=20,40\) and prefix effects are mechanism diagnostics, and prefix effect alone cannot satisfy an outcome-variation gate. No official branch may step past 250.

No proposal feature, selection rule, or tie-break may consume the source trajectory after \(\tau\), its terminal mask, failure phase, or any branch outcome. Section 5.5 supplies the outcome components, priority order, safety rule, and replay-noise floors; its “candidate versus best reference repeat” aggregation does **not** apply here. Action 2P overrides only repeat aggregation with a predeclared same-key pairwise rule. A `paired-positive` candidate must dominate its matched reference on all three continuation keys at \(H=80\). A `paired-non-improving` candidate must never dominate reference and must be strictly worse on at least one paired key beyond the registered noise floor. Outcome variation is candidate-induced only when the preregistered deadline outcome vector and direction differ from same-key reference consistently; its components, tie rules, and noise floors are sealed before execution. Reference-repeat variability and prefix physical effect alone do not count.

All pilot rows have permanent role `calibration`, subrole `coverage_pilot`. They may support this feasibility verdict but may never enter \(M_0\), a policy loss, model calibration, selector assessment, or behavior evaluation. A later V8.1 collection uses source-disjoint fresh data.

### 15.4 Interaction cap and decision rule

The hard environment-step cap is

\[
N_{\mathrm{env}}
=20\times250
+8\times(3\text{ reference}+6\times3\text{ alternative repeats})\times(10+80)
=\mathbf{20{,}120}.
\]

Snapshot restore and candidate generation do not step the environment. Every started source, prefix, continuation, abort, and invalid attempt is nevertheless charged; there is no retry, substitute source, quota extension, or budget reallocation. The segment ledger distinguishes `source`, `ref_prefix`, `ref_cont`, `div_prefix`, `div_cont`, `rand_prefix`, and `rand_cont`, binding source/anchor/candidate/CRN/policy hashes, actual official steps, termination, and cap line. The worst case is 356 ledger segments: 20 sources plus 168 prefixes and 168 continuations. A pre-segment headroom check prevents starting work that the relevant cap line cannot fund. Proposal count and GPU-seconds are reported separately.

**ADVANCE** — explicitly not a §13.2 V8.1 PASS — requires all of:

1. all 20 sources close; report every eligible failure and select exactly the first eight in the presealed order; reserved-seed use is zero and every ledger segment reconciles within 20,120 steps;
2. the same anchor checkpoint and mask hash are restored on 168/168 branch starts, and all 24/24 reference prefixes satisfy the presealed numeric replay tolerance;
3. every anchor retains at least 16 unique first-\(c\) alternatives and executes the sealed 3+18 paired branch schedule with zero post-deadline official steps;
4. at \(H=80\), at least 4/8 independent anchors show candidate-induced registered deadline-outcome variation;
5. at \(H=80\), at least 2/8 anchors contain a `paired-positive` sibling; and
6. at \(H=80\), at least two anchors each contain both a `paired-positive` and a `paired-non-improving` sibling.

Every gate is reported per anchor at the primary \(H=80\); \(H=20,40\) are reported alongside it as secondary diagnostics. The 168 branch executions are repeated measurements inside eight independent groups, not \(n=168\). A read-only demo/nominal census reports whether any baseline contains same-history multi-action outcomes; without that comparison the conclusion is restricted to newly observed local support, not “support absent from all prior data.”

**HALT** if any ADVANCE condition is unmet, including source/stratum underfill, restore or provenance failure, pool underfill, cap overrun, any official post-deadline step, or failure of any evidence gate. Outcome underfill is interpreted at the first claim rung: the current anchor/proposal/setting did not expose adequate counterfactual support. It does not motivate a world-model architecture sweep and supplies no evidence against a model that was never trained.

### 15.5 Handoff

An ADVANCE permits one source-disjoint second-setting replication and construction of the full two-task V8.1 registration. Only that later replicated collection can PASS §13.2 and authorize \(M_0\). A HALT permits one bounded, prospectively named change to the anchor or proposal/setting; it does not reopen indefinite benchmark shopping or allow post-outcome threshold/budget tuning.

Before any Action 2P environment step, the registration must seal the actual runner and dependency closure, policy/HF snapshot and weight hashes, BDDL and automaton, source and branch RNG maps, snapshot schema, exact masks and tolerances, pool-selection code, segment-level official-step ledger, abort-safe cumulative accounting, and the 20,120-step cap from worst-case headroom. The stale 1R pooled spend field is corrected only through a hash-bound successor note citing the authoritative 29,800-step ledger sum; historical artifacts are never edited in place.

### 15.6 Action 2P outcome — HALT (2026-08-21)

Executed under registration `0979c839` with execution seal `de5dedc5` written before any outcome existed. 20 sources and 168 branch segments, **19,906 / 20,120** environment steps, reserved seeds untouched. Full record: `plan_and_progress/2026-08-21.md`; artifact `results/v081p/2026-08-21T044523Z`.

**Provenance conditions 1–3 pass, and — unlike their first implementation — they could have failed.** Restore is verified by re-snapping the live environment after `restore` and hashing that, not by hashing the stored snapshot against itself: **168/168**. Replay tolerance is measured over achieved post-prefix body positions rather than commanded actions: **max pairwise L2 = 0.00e+00** on all eight anchors, against a sealed `1e-3`. Pools are **32/32** unique after first-\(c\) deduplication.

**Decision conditions 4 and 6 fail.**

| condition | required | observed |
|---|---|---|
| anchors with candidate-induced variation | ≥ 4/8 | **3/8** |
| anchors with a paired-positive | ≥ 2/8 | 2/8 ✓ |
| anchors with both a positive and a non-improving sibling | ≥ 2/8 | **0/8** |

Three results carry forward:

1. **A genuine conversion exists.** At anchor `a3493`, sibling `rand1` beat its matched reference on all three CRN keys and turned `failure@250` into `success@250` on 3/3 keys where the reference succeeded 0/3. At `a3494`, three of six siblings are paired-positive on progress and timing without reaching success.
2. **The binding failure is the absence of non-improving siblings**, not of positives: exactly one exists across all eight anchors. The gate requires both directions because ranking signal must be two-sided; what the setting produced is one-sided.
3. **Five of eight anchors are inert** — all 18 paired comparisons are exact ties at \(H=80\).

`[LOCKED]` **The sealed floors decided this verdict, and that is the point.** Six of eight anchors differ in raw readouts while only three clear the quantization floors; reporting raw differences as variation would have passed condition 4 and produced an ADVANCE. The floors were derived from the task automaton's 10-step evaluation stride and sealed before execution, so that reading was not available after the outcomes existed. Any successor pilot inherits this rule: the discrimination threshold is derived from instrument quantization and sealed before execution, never fitted to the reference variability it must exclude.

`[LOCKED]` **Eligibility and comparison must key on the same object.** `a3480`'s source was a genuine `failure@250` and passed the registered mask, but at that anchor the freshly drawn reference succeeds on 2 of 3 CRN keys — the source failed under *its own* continuation noise only. Anchor eligibility keyed on the source's outcome while the paired comparison keys on a fresh reference; those are different objects. A successor stratum must require the reference itself to fail on a registered majority of CRN keys before the anchor is admitted.

Per §15.5, "stop the formulation" is **not** indicated: supported action diversity plainly exists (32/32 unique pools) and produces useful variation at 3/8 anchors including one full conversion. One bounded coverage-component revision is the indicated path; which component is a registered decision and is not taken here.

---

## 16. Mainline reset — model training before further diagnostics (2026-08-21)

The post-2P discussion corrected the work priority, and this section records that change in the binding design rather than leaving it only in a daily. The research object is the sequence

\[
\text{failure-driven counterfactual interaction}
\rightarrow \text{action-conditioned WM}
\rightarrow \text{model-guided acquisition}
\rightarrow \text{policy improvement},
\]

not a benchmark or an anchor diagnostic. Action 2P established that supported siblings can produce real effects, including a 3/3 failure-to-success conversion; its conservative pilot gate stands as a historical verdict but no longer inserts another diagnostic stage before model training. Action 2P rows remain permanently calibration-only and enter no bank.

`[LOCKED]` **A HALT on a pilot gate does not by itself authorize another diagnostic stage.** Where the evidence already shows the mechanism is present, the next step is to train the model the claim ladder actually requires. §14.5 and §15.6 remain valid as records of their own instruments; they are not standing blocks on model training.

### 16.1 Two M0 results, both against a capacity-matched state-only baseline

| | first M0 (v082) | M0.1 (v083, Action 2M.1) |
|---|---|---|
| bank | 28 groups, 7 candidates/anchor | **48 groups, 17 candidates/anchor**, 1,088 branches |
| test ranking support | 144 pairs: 5 better, **0 worse** | 384 pairs: **29 better, 19 worse** |
| matched prediction loss | 0.3257 vs **0.2989** state-only | 0.8168 vs **0.6965** state-only |
| rank macro accuracy | 0.500 vs 0.500 | **0.3770** vs 0.3333 |
| rank non-tie accuracy | 0 vs 0 | **0.0625** vs 0.0000 |
| action-shuffle gap | +2.168e-4 | **+4.656e-4** |
| verdict | negative | **2 of 3 registered conditions**; `prediction_loss_gain` fails |

Both runs are single fixed 30-epoch schedules with no architecture or hyperparameter sweep, validation-selected before the test split was read.

### 16.2 What is established, and what the numbers do not support

Established: the action path is used — shuffling actions moves the action model's loss and leaves the baseline's unmoved, at roughly double the first run's magnitude — and widening the proposal from 7 to 17 siblings per anchor produced the first V8 bank with **two-sided** ranking signal on held-out data.

Not established: useful sibling ranking. Non-tie accuracy of `0.0625` is **3 of 48 pairs**, far below the ~⅓ a three-way guess yields, and the baseline's `0.3333` macro is exactly what predicting "tie" everywhere produces. The action model also lost on every prediction metric — 17% higher matched prediction loss, roughly double the physical MSE, higher Brier. Adding the action path degraded prediction while barely helping ranking.

`[LOCKED]` **Two structural properties of `B_boot-v2`, recorded before its test split was opened**, because they constrain what a second negative can mean:

1. the **validation split contains zero non-tie pairs**, so checkpoint selection was blind to ranking and the rank head was chosen on physical/outcome loss alone. A weak ranking result therefore has two explanations this design cannot separate — the model did not learn it, or the selector could not pick the epoch that had;
2. the **train split carried 21 non-tie pairs of 512**, skewed 4 better to 17 worse, so the non-tie-balanced objective concentrated half the ranking gradient on 21 pairs.

Any successor must fix the validation split's ranking blindness, or state explicitly that checkpoint selection cannot see the head being evaluated.

### 16.3 Registered consequence

By Action 2M.1's completion rule this is a **second state-only result**: the indicated change is to the **candidate proposal family**. It does not authorize another benchmark screen, horizon audit, anchor diagnostic, or a sweep of this architecture. Training-side ranking signal is a live alternative diagnosis to proposal diversity, and a proposal-family change should say which of the two it targets.

### 16.4 Retraction — `B_boot-v2`'s ranking support (2026-08-23)

§16.1 credits `B_boot-v2` with "two-sided" held-out ranking support (29 better, 19 worse of 384 test pairs) and §16.2 calls that the first such bank in V8. **The pairs are real; the attribution to candidate identity is retracted.**

The bank's three test repeats make the label decomposable, and the decomposition inverts the reading:

| contrast | non-tie |
|---|---:|
| candidate vs reference, matched continuation seed | 48/384 = **12.50%** |
| identical chunk, different continuation seed | 78/408 = **19.12%** |
| reference against itself across repeats | 4/24 = **16.67%** |

Re-running the *same* action under a different continuation seed flips the label more often than swapping to a *different* action under a matched seed. **0 of 128** test candidates hold a consistent non-tie sign across all three repeats; candidate ICC is `dp −0.011`, `G 0.002`, `ttm 0.227`. A repeat-count confound hid it: `R=1` pairs are 2.98% non-tie against `R=3` pairs at 16.27%, so the test split looked richer only because it was repeated.

What survives is narrower and still useful: **at these anchors the registered outcome label is dominated by continuation noise.** M0.1's `3/48` non-tie accuracy is therefore the expected consequence of an unrankable target, not a near-miss by the model. §16.2's prediction-loss and action-shuffle conclusions are unaffected.

`[LOCKED]` **A non-tie count is not evidence of a candidate effect unless it is separated from the continuation-noise floor.** Any bank claiming sibling ranking support must report the matched-seed contrast against the same-chunk-different-seed contrast, and must report per-candidate sign consistency across repeats. A single-repeat bank cannot support a ranking-support claim at all.

---

## 17. Proposal-family diagnosis and the Action 2M.2 halt (2026-08-23)

Action 2M.2 (daily 2026-08-22) proposed effect-stratified selection over raw stock-PI0 draws as the registered repair for two state-only M0 results. It is **halted before its ~120,000-step collection** on a 1,976-step measurement, with four independent refutation attempts — one arguing explicitly to run it as written — returning zero refutations.

### 17.1 Why selection cannot be the intervention

`results/v084_probe/2026-08-23T185117Z`: 5 eligible anchors, 64 raw draws each, no selection. All 64 chunks byte-distinct at every anchor; median translation-magnitude CV **0.0121**; median within-anchor pairwise direction angle **3.05°** against a between-anchor **31.26°**; gripper saturated (`|a|>0.95`) in 100% of entries; **0/5 anchors contain both gripper classes**. In the collected bank, `post_prefix_delta` is bitwise identical across all 17 candidates at **43/48** anchors, and gripper-close timing — one of the three registered stratification axes — is unfillable at **46/48**.

By §4.1's own criterion the pool is **one candidate family**, and a quota over one family is not a stratification.

### 17.2 Scope

The supported claim is bounded, and the over-general version is explicitly not made:

> At the anchors a fixed \(\tau=160\) on `chain1b_lr2@250` admits, proposal spread is small, and where it is larger it does not move the registered outcome or ranking readout beyond continuation-seed noise.

This bounds this collection, task, \(\tau\), and \(H\). It is **not** a claim that stock PI0 proposals are effect-degenerate in general.

### 17.3 Ordering of the two successors

`[LOCKED]` **Establish whether any candidate effect exists above the noise floor before changing the proposal family.** If the outcome label at an anchor is noise-dominated, a more diverse family yields more diverse actions and the same unrankable labels — so a family change run first cannot be interpreted either way.

- **(a) Noise-floor experiment.** Raise repeats to 8–16 at ~15 anchors; far cheaper than 2M.2. Establishes whether candidate identity moves the registered outcome at all at this setting.
- **(b) Bounded-perturbation family.** Permitted by §2 `[LOCKED]` ("a bounded perturbation of their supported chunks"). Matching pooled effect spread requires per-step deltas of ~20–37% of the observed action limit on translation, or a gripper sign flip — bounded, inside support, not unconstrained search.

A negative from (a) relocates the binding constraint to the outcome readout or the anchor rather than the candidates, which is a different action item from either.

### 17.4 Action 2M.3 attempted — HALT on anchor underfill (2026-08-23)

§17.3 `[LOCKED]` the noise-floor experiment as the successor that must precede any proposal-family change. It was registered (daily 2026-08-23) and executed. It **halted on anchor underfill** and did not answer its question.

Registered: 15 anchors × (1 reference + 4 max-spread alternatives) × 12 shared continuation keys, ceiling 30 sources, cap 88,500 steps, no extension. Obtained: **13 of 15 anchors from all 30 sources**, spending **7,336 steps**. All 13 `failure@250` sources were exact-mask eligible, so the mask cost nothing and the shortfall is source yield alone. No branch executed; no \(D\), ICC, sign-consistency, or decision exists.

The interaction contract behaved as designed: the halt landed at the end of the source phase, before the 81,000-step branch phase, so an underfilled experiment cost 8.3% of its budget.

`[MEASURED]` **`failure@250` yield on `chain1b_lr2@250` is 77/143 = 0.538**, pooled over every occasion it has been observed (Action 2P 11/20, `B_boot-v2` 48/85, v084 probe 5/8, Action 2M.3 13/30). The 2M.3 run was not anomalous — \(P(\le 13 \mid 30,\,0.538)=0.166\). Any future source queue on this task must be sized from this rate rather than from a prior run's realized fill.

| sources | \(P(\ge 15\text{ anchors})\) |
|---:|---:|
| 30 (as registered) | 0.728 |
| 34 | 0.905 |
| **38** | **0.974** |
| 44 | 0.997 |

`[LOCKED]` **A source ceiling must be derived from the measured eligibility rate and stated with its fill probability.** Where a branch phase dominates the budget — as it does here, 81,000 of 88,500 — near-certain fill is nearly free: 30 → 38 sources raises the cap by 2,000 steps (2.3%) and the fill probability from 0.73 to 0.97. An underfilled registration wastes its source phase and answers nothing.

`[LOCKED]` **"No quota or cap extension" binds an underfill exactly as it binds an overrun**, and a shortfall may not be repaired by lowering the anchor requirement after the fill is known. Analysing 2M.3's 13 anchors would have widened the one-sided cluster bound by only ~7%, which is precisely why taking it after the fact would be tempting and impermissible. The remedy is a fresh registration; the halted run's anchors are not carried into it, because reusing them would make the successor's fill partly outcome-dependent.

### 17.5 Action 2M.3 completed — `NOT_IDENTIFIED` (2026-08-23)

Re-registered at 38 sources (the ceiling derived from the §17.4 measured yield; that one number was the only change) and executed: **15/15 anchors from 36 sources, 900 branches, restore verified 900/900, 89,789 / 90,500 steps**, execution seal written before any outcome existed. Record: `plan_and_progress/2026-08-23.md`.

| component | `D` | `D` lower | ICC | ICC lower | sign-consistent |
|---|---:|---:|---:|---:|---:|
| `dp` (primary) | +0.0172 | −0.0017 | +0.1265 | +0.0375 | 0/60 |
| `ttm` (primary) | +0.0132 | −0.0064 | +0.1908 | +0.0317 | 0/60 |
| `G` (primary) | +0.0174 | −0.0049 | **+0.4618** | **+0.2560** | 0/60 |
| `succ` (secondary) | +0.0042 | −0.0025 | +0.0907 | +0.0346 | 0/60 |
| `dmg` (secondary) | +0.0000 | +0.0000 | n/a | n/a | 0/60 |

**`NOT_IDENTIFIED`**: the rule requires \(D_{\text{lower}}>0\) *and* \(\mathrm{ICC}_{\text{lower}}>0\) on the same primary component. ICC passes on all three; \(D\) fails on all three.

`[ESTABLISHED]` **Candidate identity explains outcome variance but does not clear the discrimination floor.** On \(G\), ICC is 0.4618 (lower bound 0.2560) — a quarter to a half of continuation-return variance is candidate-attributable — while the per-candidate mean spread has median **0.0048** against a within-candidate seed range of median **0.0463**, roughly 10x larger, and against a registered tie floor of 0.0956. Only 3/15 anchors have candidate-mean spread above that floor. Sign consistency is 0/60 on every component across twelve shared seeds.

`[ESTABLISHED]` **The registered outcome vector is too coarse to express an effect of this size.** \(G\) takes **10 distinct values across all 900 branches**, and only 3 of 15 anchors register any `success@250` among 60 branches. The floors are not arbitrary — they follow from the automaton's 10-step evaluation stride — so the quantization limiting the readout and the floor limiting the label have the same origin.

`[LOCKED]` **The bounded-perturbation proposal family is not authorized.** §17.3 gated it on a positive noise-floor result and the result is negative. Running it now would produce more diverse actions against a readout that cannot resolve them.

`[LOCKED]` **The next change is a formulation decision — the outcome readout or the anchor/setting — not another proposal family, model, benchmark, horizon, or architecture sweep.** The evidence points at the readout: the effect exists and is sub-quantum. A progress measure resolving milestone timing more finely than 10 steps, or one that does not collapse to ~10 discrete values, would let an effect of this magnitude register. That change is not taken here.

`[LOCKED]` **An interaction cap belongs to a registration.** A superseded or aborted attempt's steps stay charged to it and stay in the project total, but must not consume its successor's budget. Attempt 2 of 2M.3 aborted because the cross-run cumulative derivation added after Stage 1R.1 did not distinguish registrations and deducted attempt 1's 7,336 superseded steps from attempt 2's 9,500-step source line. Quarantine prefixes (`ABORTED_`, `SMOKE_`, `SUPERSEDED_`, `HALTED_`) now scope the derivation. Adjusting budget accounting after a halt is permissible **only** when the halt was a tooling fault and no decision rule, threshold, quota, or seed changes — both conditions held here and both must be stated when invoked.

### 17.6 Mainline decision — expected-consequence M0.2 (2026-08-23)

`[AMENDED]` Section 17.5's `NOT_IDENTIFIED` verdict answers the registered deterministic non-tie question and continues to prohibit the bounded-perturbation inference. It does **not** establish that an action-conditioned stochastic world model is untrainable. Requiring one candidate to hold the same discrete sign under every continuation seed confuses a conservative executed-correction admission rule with the supervised object of a world model, which is the conditional consequence distribution or its expectation.

`[DECISION]` **Take the outcome-readout branch of §17.5 and return immediately to model training.** Keep `chain1b_lr2@250`, `tau=160`, `c=10`, `H=80`, and the stock-PI0 proposal family fixed. Replace the ten-step-quantized primary training label with the already registered phase-aware continuous potential evaluated every environment step. Retain damage, success, and the historical `dp/ttm/G` vector as safety/task readouts; do not use them as a pre-training support gate.

`[LOCKED]` **The phase state is causal history, not a branch-local reinitialization.** Build the phase-potential memos from the source trajectory through `tau` and fork them with the anchor snapshot for every candidate/repeat. For `γ=0.99`, the canonical score is the discounted signed sum of per-step potential increments over the executed prefix and H80 continuation: `QΦ = GΦ_exec + γ^c GΦ_cont,80`. The exact formula, terminal absorbing rule, auxiliary horizons, and top-1 regret are frozen in the Action 2M.4 daily contract.

`[LOCKED]` **Repeated continuations estimate the training target; they are not independent labels for one deterministic prediction.** The trainable target for a fixed `(state, candidate)` is its mean consequence across shared continuation keys. Candidate conditioning is trained both on the absolute expected physical/progress outcome and on the candidate-minus-reference expected effect. The per-repeat ternary tie/better/worse loss used by M0.1 is retired for M0.2 because it assigns conflicting labels to the same deterministic model output when continuation noise changes the sign.

`[LOCKED]` **M0.2 has one frozen objective and one checkpoint rule.** Train-only target statistics normalize physical and continuous targets. The anchor-balanced loss is `1.0 physical + 1.0 task + 0.25 paired-effect`; continuous terms and paired expected effects use Smooth-L1, damage/success rates use BCE. Each model's checkpoint is the epoch with minimum source-disjoint validation loss under that same formula. These definitions are training mechanics, not empirical gates, and are not revised after collection starts.

`[LOCKED]` **Action 2M.4 is one uninterrupted bank-to-checkpoint action, not a new diagnostic ladder.** It collects a fresh source-disjoint bank, trains one fixed-schedule action-conditioned M0.2 and its capacity-matched state-only baseline, freezes checkpoints on validation, and opens held-out test once. Schema, restore, finite-loss, reload, and ledger smokes are implementation checks only. There is no label-count, non-tie, sign-consistency, or noise-floor gate between a passing mechanics smoke and full model training.

The executable contract and exact collection sizes are in `plan_and_progress/2026-08-23.md`, Action 2M.4. Its mandatory deliverables are `B_boot-v3/groups.pt`, `M0.2/best.pt`, complete training curves, and one source-disjoint held-out comparison against state-only and simple prediction baselines. A useful M0.2 proceeds directly to matched-budget WM-guided acquisition; a negative M0.2 changes model/training of the observed continuous target rather than reopening proposal, anchor, horizon, or noise-floor diagnostics.

---

## 18. Action 2M.4 — continuous readout, `B_boot-v3`, and M0.2 (2026-08-24)

§17.5 sent the next change to the **outcome readout**. Action 2M.4 took it: keep the anchor, setting, and stock-PI0 proposal family fixed, replace the deterministic label with a continuous action-consequence target, collect a fresh bank, and train one M0.2. Record: `plan_and_progress/2026-08-24.md`.

### 18.1 The readout change succeeded

`[ESTABLISHED]` **The registered outcome is no longer the binding constraint.** The continuous target \(Q_\Phi\) — the phase potential evaluated at *every* environment step, discounted, and averaged over shared-seed repeats — took **80 distinct values across 80 branch rows**, where the legacy \((\mathrm{dp},\mathrm{ttm},G)\) vector took 11 on those same rows and 10 across 900 rows in Action 2M.3.

\[
G_\Phi^{\mathrm{exec}}=\sum_{j=1}^{c}\gamma^{j-1}\big(\Phi_j-\Phi_{j-1}\big),\quad
G_\Phi^{\mathrm{cont}}=\sum_{j=1}^{H}\gamma^{j-1}\big(\Phi_{c+j}-\Phi_{c+j-1}\big),\quad
Q_\Phi=G_\Phi^{\mathrm{exec}}+\gamma^{c}G_\Phi^{\mathrm{cont}}.
\]

`[LOCKED]` **Φ is built on the source episode and forked, never rebased at the anchor.** `lcwm/v086_phase.py` extracts the per-atom memos (`ever_true`, `was_lifted`, `dxy_at_lift`) and the episode baselines into a forkable object; `scripts/test_v086_phase.py` proves it byte-identical to `episode_phase_potentials` and proves a forked continuation reproduces the unforked episode tail. Reinitializing approach, attachment, lift, or transport state at \(\tau\) is prohibited.

`[LOCKED]` **The deterministic target for one `(state, candidate)` is the candidate mean across shared-seed repeats.** Repeats estimate that target and are retained for calibration; they are never converted into conflicting per-rollout labels for a single deterministic model output.

### 18.2 M0.2 is a held-out negative

| model | physical MSE | effect MAE | paired-effect err | top-1 regret | effect corr |
|---|---:|---:|---:|---:|---:|
| action-conditioned | 0.001791 | 0.063241 | 0.017362 | 0.016022 | **−0.2275** |
| state-only | **0.001298** | **0.036246** | **0.014041** | **0.012647** | 0.0000 |
| copy / no-change | 0.091660 | **0.014041** | — | — | — |

`[ESTABLISHED]` The action-conditioned model loses to every registered baseline on every metric. The action path *is* used — shuffle gap 0.01838 against the baseline's exactly 0 — and conditioning on it makes the model worse. Its candidate ranking is **anti-correlated**, not merely uninformative. Neither model beats "assume every candidate equals the reference" on effect MAE.

`[NOTE]` `state_only` and `stock_reference` tie exactly on paired effect and top-1 regret by construction: a model with no action input scores every candidate identically, so its predicted delta is zero, which *is* the stock-reference predictor. Those two registered baselines are one bar, and any future comparison must say so rather than counting them twice.

The failure mode is localized: validation selected epoch **4** for the action model against **27** for state-only, and action validation loss rises to 0.90–0.96 while the baseline sits at 0.49–0.69. With 48 training anchors against a target whose signal-to-noise on the candidate mean is roughly 1.7×, it overfits the action input immediately.

`[LOCKED]` **The next change concerns model and training of an observed continuous target** — capacity, regularization, or training-set size — and does **not** return to another proposal family, anchor, horizon, or noise-floor diagnostic. The matched-budget WM-guided-versus-random acquisition test remains locked; it required beating the state-only and stock-reference baselines.

### 18.3 Two implementation defects and their cost

`[LOCKED]` **Anchor eligibility is `failure@L` AND the exact event mask — both, always.** The first 2M.4 collection tested only the mask and put **33 of 64 anchors inside episodes that later succeeded**, discarding 138,723 steps. The method is failure-anchored (§3); training on that bank would have silently redefined the learned population and shown up in no training metric. The tell was arithmetic, not a test: 100% eligibility against a measured 0.538, on ceilings the action item had sized *from* 0.538.

`[LOCKED]` **Quarantine must fail closed.** The re-collection halted because a newly invented `DISCARDED_` label was absent from `SegmentLedger`'s hardcoded prefix list — a regression of the fix made two actions earlier for the identical failure (§17.5). Quarantine is now a regex over any ALL-CAPS label prefixed to a run stamp. A cap-accounting mechanism that enumerates its exceptions will be defeated by the next exception.

Cost: of Action 2M.4's **318,242** environment steps, **158,275 (49.7%)** bought nothing and are charged. Project total across all V8 ledgers: **611,655**.

---

## 19. Action 2M.5 — reference-centered residual WM (2026-08-25)

§18.2 sent the next change to **model and training of an observed continuous target**. Action 2M.5 took it with a fixed factorization rather than a sweep:

\[
\hat Y(i)=b(s)+\hat\Delta(i),\qquad
\hat\Delta(i)=h\big(z_s,u_0,u_i-u_0\big)-h\big(z_s,u_0,0\big),
\]

with \(b(s)\) and \(z_s\) frozen from M0.2's state-only checkpoint, one 64-unit residual head trained (32,505 parameters), and the independent rank head retired so the sibling score is the predicted consequence itself. Record: `plan_and_progress/2026-08-25.md`.

### 19.1 What the factorization fixed, and what it did not

`[ESTABLISHED]` \(\hat\Delta(\text{reference})=0\) by construction, verified before training, so the model cannot lose to the stock-reference predictor on the reference itself. Held-out effect correlation is **+0.1643**, against M0.2's **−0.2275** on the 2M.4 split.

`[ESTABLISHED]` **M0.3 is not promoted.** Effect MAE 0.006633 against 0.006195 (zero-delta/reference) and 0.006113 (M0.2); top-1 regret 0.008557 against 0.003125 and 0.006649. The bar required beating both baselines on both quantities with positive correlation; correlation passed, the other two did not. The matched-budget WM-guided acquisition test remains locked.

### 19.2 The held-out bar is underpowered — this supersedes how earlier results were read

`[LOCKED]` **An 8-anchor held-out panel cannot rank sibling-selection methods, and top-1 regret over it is not a usable promotion statistic.** On the fresh test bank a zero-information rule — "always pick candidate 4" — achieves regret **0.000930**, better than every model and every registered baseline. The same rule is the **worst of five** on the 48 dev-train anchors (**0.023339**, argmax at 3/48 against candidate 0's 28/48). The test ordering is dominated by which candidate index happened to win eight times.

`[LOCKED]` **A held-out effect correlation at this sample size is not identified in either direction.** The same M0.2 checkpoint scores **−0.2275** on one 8-anchor bank and **+0.1719** on a disjoint one, from 32 candidate points each. §18.2's statement that M0.2's ranking was "anti-correlated, not merely uninformative" is **withdrawn**: the sign does not replicate. Any future promotion claim must state the number of independent anchors behind its correlation and must not rest a directional claim on a single 8-anchor panel.

`[ESTABLISHED]` **The physical channel is exactly degenerate at these anchors.** Maximum absolute true physical delta between any candidate and its reference is **0.000e+00** across all 8 test anchors — bitwise-identical `post_prefix_delta` for all five candidates — after 43/48 in `B_boot-v3`. `L_physical` at weight 0.5 is therefore regressing a constant zero, and a near-perfect score on that channel carries no information. Any objective retaining that term at this setting is spending a third of its weight on a constant.

`[ESTABLISHED]` Effect sizes: true \(\Delta Q_\Phi\) over 32 test candidate points has mean \(|\Delta|=0.00619\), sd 0.00635; per-anchor \(S\) range 0.0047–0.0246; the reference is already the best candidate at 3/8 test and 28/48 dev anchors.

### 19.3 Consequence

The two model-side changes registered since §17.5 — a continuous readout (2M.4) and a reference-centered residual factorization (2M.5) — have each done what they were designed to do while leaving the promotion decision unresolved, because the decision is being made on a panel too small to resolve it. Before another model or training change is registered, the **evaluation panel** is the component that limits what any result can mean.

Interaction: Action 2M.5 spent **25,490** steps with no discarded or aborted run. Project total across all V8 ledgers: **637,145**.
