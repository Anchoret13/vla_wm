# pi05-lcwm — Implementation Guide v0.1

Language-conditioned world model on a frozen π0.5 backbone, LIBERO-100.
Order = order of work. Completed items are removed, not checked off. **[EST]** = measure; **[DEC]** = open decision with default.

---

## 0. Status (2026-07-18)

Done (source of truth = `vla_wm` repo, `scripts/setup_env.sh` + README gotchas):
- `vf0s` env: torch cu128 (sm_120), lerobot 0.6.1 editable `[pi,libero]`, EGL headless, imports verified
- Smoke eval end-to-end: `pi05_libero_finetuned`, libero_spatial task 0, 1 ep, success
- Checkpoint locked: **`lerobot/pi05_libero_finetuned`** (`pi05_libero_base` = community-measured 0% in this harness; never use)

Blocked: full 400-ep eval (shared GPU, ~9 GB free). GPU-free lane in §2 runs now.

---

## 1. Locked conventions

- Env: LIBERO-100 = libero_90 (short-horizon pool) + libero_10 (long-horizon reference). `action_horizon = 10`; commitment `c = 10`; imagination `K = 10` (v0).
- State: compact carry latent, `M = 4` × `d_z = 384`. No patch-grid rollout, no pixel reconstruction.
- Conditioning arms (identical graphs): `LANG` / `FREE` (learnable const) / `TASKID` (embedding table). Readout heads get the **real** language embedding in **all** arms; arm differences live only in the dynamics channel.
- Anti-collapse: EMA targets (default) + optional linear recon anchor to pooled frozen features (flag).
- Every eval: shared randomization schedule files, pinned seeds, seed-level SE, environment fingerprint (lerobot/LIBERO commits, torch/CUDA) in every results JSON.

---

## 2. Stage 1 — LIBERO-100 baseline (next actions)

**Script increments on existing `eval_*.sh` (before next run):** explicit `--seed`; per-suite invocations (separate JSONs, no shared-fate crash); fingerprint dump into results.

**1a mode-check (first GPU window):** libero_spatial alone, 10 tasks × 10 eps, batch per free VRAM. Community reproduction is bimodal — reference 97.5 avg vs one report at 77/66/89/55 (71.75). If ≈97 → run remaining three suites; if ≈77 → stop, debug in order: editable-lerobot commit, init_states, render→resize pipeline, seed, precision. Measure steady-state ep time here (smoke's 260 s included model load).

**1a targets (this checkpoint, 10 eps/task protocol):** LeRobot 97.0 / 99.0 / 98.0 / 96.0, avg 97.5; OpenPI JAX reference 98.8 / 98.2 / 98.0 / 92.4, avg 96.85.

**GPU-free lane (run now, in parallel with the blockage):**
- Stage 3 replay-validity machinery — needs no policy, one EGL context ≈ 2.5 GB
- libero_90 hdf5 → LeRobot-format conversion (CPU) **[EST 0.5 day]**
- per-scene BDDL predicate-pool inventory (CPU)

**1b libero_90 zero-shot:** `--env.task=libero_90` works directly (suite registered, max steps 400). Expected low (openpi checkpoint measured ≈18%; this variant untested). Record per-task SR.

**1c finetune on libero_90** (50 demos/task):

```bash
lerobot-train --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_libero_finetuned \
  --policy.use_relative_actions=true \
  --policy.relative_exclude_joints='["gripper"]' \
  --policy.train_expert_only=true \
  --dataset.repo_id=<libero90_lerobot>
```

Recompute norm stats on the finetune dataset. Done-check: 90-row per-task SR table, mean > 70% **[EST target]** — doubles later as chain-pool difficulty calibration.

---

## 3. Stage 2 — raw eval loop (the chassis)

```python
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

suite = benchmark.get_benchmark_dict()["libero_90"]()
task  = suite.get_task(i)
env   = OffScreenRenderEnv(bddl_file_name=bddl_path(task),
                           camera_heights=224, camera_widths=224)
env.reset(); env.set_init_state(suite.get_task_init_states(i)[k])
for _ in range(10): obs, *_ = env.step(NOOP)              # settle
for t in range(T_max):
    if t % 10 == 0:
        chunk = policy_wrapper.sample_chunk(obs, task.language)
    obs, r, done, info = env.step(chunk[t % 10])
    if done: break
```

Preprocessing parity: reuse lerobot's own pipeline (`make_pre_post_processors(policy.config, model_id)`) inside `policy_wrapper` instead of re-implementing resize/normalize/proprio assembly — kills that footgun class. Keep a raw sim handle for `set_state`. Done-check: **reproduce 1a numbers with this loop** (± noise). All later experiments run here.

---

## 4. Stage 3 — snapshot / replay

```python
s = env.env.sim.get_state().flatten()
env.env.sim.set_state_from_flattened(s); env.env.sim.forward()
reset_osc_controller(env)   # OSC keeps internal interpolation state; must reset after set_state
```

Replay-validity protocol: per suite, replay recorded demo actions from init state; compare obs/success traces; emit `replay_validity_report.json` (literature reference range: 26–28 / 30 valid). Every stored snapshot carries a `replay_valid` flag.

---

## 5. Stage 4 — batched N-sampling + diversity

In `PI05Policy` sampling path: prefix forward once → expand KV cache to N along batch → N noise draws → 10 denoise steps batched → `(N, 10, 7)`.

First measurement (immediately): ~20 states × 5 tasks, N=16 → pairwise action-space L2 histogram + open-loop end-state dispersion (via Stage-3 replay). Decides whether σ-perturbation joins ρ.

---

## 6. Stage 5 — feature tap & storage

Tap = constant-prompt prefix-only forward, hooks on:
- last-layer image-token hidden states `h_raw ∈ (N_img≈512, 2048)` (2 cams × 256)
- pooled instruction-token states from the **real**-prompt forward → `e_lang ∈ (2048,)` → linear → `(256,)`

Storage (fp16): raw ≈ 2 MB/step → LIBERO-100 expert ≈ 0.65M steps **[EST]** ≈ 1.3 TB → infeasible.
Default: 2×2 average-pool tokens → 128/cam ≈ 0.5 MB/step; **probe subset first** (50K steps ≈ 25 GB). A100 batch job. **[DEC]** stride-2 predictor steps halves everything.

Sanity: quantify feature shift real-prompt vs constant-prompt on same frames before trusting taps.

---

## 7. World model

### 7.1 Segment record schema

```
segment = {
  h:      (T, N=256, 2048) fp16   # pooled trunk tokens, T = burn_in + K + T_roll
  q:      (T, 8)                   # proprio
  a:      (T, 7)                   # env actions
  e_lang: (2048,)                  # pooled instruction embedding (pre-projection)
  task_id: int
  y_success: (T,) {0,1}            # BDDL goal check per step
  replay_valid: bool
  source: {expert | sigma_0.05 | sigma_0.15 | sigma_0.4 | random | cross_l}
}
```

Progress labels Φ and precondition labels b: **deferred** (v0 trains d̂ only).

### 7.2 Modules

```python
class CondProvider(nn.Module):            # the three arms, identical downstream graph
    def __init__(self, arm, d_e=256, num_tasks=None):
        self.proj = nn.Linear(2048, d_e)                       # LANG
        self.const = nn.Parameter(torch.zeros(d_e))            # FREE
        self.table = nn.Embedding(num_tasks or 1, d_e)         # TASKID
        self.arm = arm
    def forward(self, e_lang, task_id):
        if self.arm == "LANG":   return self.proj(e_lang)
        if self.arm == "FREE":   return self.const.expand(len(task_id), -1)
        if self.arm == "TASKID": return self.table(task_id)

class CarryAggregator(nn.Module):         # φ: (C_{t-1}, h_t, q_t, e) -> C_t
    def __init__(self, d_h=2048, d_z=384, M=4, d_e=256, n_blocks=2, heads=6):
        self.h_proj = nn.Linear(d_h, d_z)
        self.q_proj = nn.Linear(8, d_z)
        self.film   = nn.Linear(d_e, 2 * d_z)
        self.blocks = nn.ModuleList(CrossAttnBlock(d_z, heads) for _ in range(n_blocks))
        self.init_carry = nn.Parameter(0.02 * torch.randn(M, d_z))
    def forward(self, C_prev, h, q, e):
        kv = torch.cat([self.h_proj(h), self.q_proj(q)[:, None]], dim=1)
        g, b = self.film(e).chunk(2, -1)
        x = C_prev * (1 + g[:, None]) + b[:, None]      # FiLM on queries
        for blk in self.blocks: x = blk(q=x, kv=kv)
        return x                                        # (B, M, d_z)

class LatentPredictor(nn.Module):         # T̂: (C_t, a_t, e) -> C_{t+1}
    def __init__(self, d_z=384, M=4, d_e=256, layers=6, heads=6, d_a=7):
        self.a_proj = nn.Linear(d_a, d_z); self.e_proj = nn.Linear(d_e, d_z)
        self.type = nn.Parameter(0.02 * torch.randn(3, d_z))
        self.enc = PreLNTransformer(d_z, layers, heads)
    def forward(self, C, a, e):
        seq = torch.cat([C + self.type[0],
                         self.a_proj(a)[:, None] + self.type[1],
                         self.e_proj(e)[:, None] + self.type[2]], dim=1)
        return self.enc(seq)[:, :C.shape[1]]

class Heads(nn.Module):
    def __init__(self, d_z=384, M=4, d_e=256):
        self.d_head = MLP(M * d_z + d_e, 256, 1)        # success posterior d̂
        self.zs_head = nn.Linear(64, 8)                 # proprio slice decoder g_s
        self.rec_dec = nn.Linear(M * d_z, 2048)         # recon anchor (flag)
```

Param count ≈ 5M (φ) + 12M (T̂) + heads **[EST]**.

### 7.3 Training step

```python
def training_step(batch, cfg):
    h, q, a, e_lang, tid, y = batch                    # T = burn_in + K + T_roll
    e = cond(e_lang, tid)                              # dynamics-channel conditioning (arm)

    C = unroll_encoder(agg, h, q, e)                   # online, teacher-forced, (B,T,M,d_z)
    with torch.no_grad():
        Cb = unroll_encoder(agg_ema, h, q, e.detach()) # EMA target chain, stop-grad

    L_sp = 0                                           # multi-anchor short rollouts
    for t0 in sample_anchors(cfg.burn_in, T - cfg.T_roll, n=4):
        Ch = C[:, t0]
        for k in range(1, cfg.T_roll + 1):
            Ch = pred(Ch, a[:, t0 + k - 1], e)
            L_sp += cfg.w[k] * cosdist(LN(Ch), LN(Cb[:, t0 + k]))

    L_d   = bce(heads.d(C[:, cfg.burn_in:], proj(e_lang)), y[:, cfg.burn_in:])  # REAL e_lang, all arms
    L_zs  = mse(heads.zs(C[..., 0, :64]), q)
    L_rec = mse(heads.rec(C), h.mean(dim=2)) if cfg.recon_anchor else 0.0
    return L_sp + cfg.ld * L_d + cfg.ls * L_zs + cfg.lr * L_rec
# after optimizer.step(): ema.update(tau=0.996)
```

Notes: `burn_in = 4` (segments start mid-episode; carry from `init_carry`, first 4 steps excluded from losses). `T_roll` curriculum 1 → 2 → 4. Target chain uses the same (detached) conditioning as online.

### 7.4 Config defaults

```yaml
d_z: 384    M: 4    d_e: 256
K: 10       T_roll: 4       burn_in: 4
arm: [LANG, FREE, TASKID]          # sweep axis
recon_anchor: [false, true]        # sweep axis (v0: 4 configs = {EMA, EMA+rec} x {LANG, FREE})
optimizer: adamw  lr: 3e-4  schedule: cosine  wd: 0.05
batch: 256 segments   precision: bf16   ema_tau: 0.996
loss_weights: {ld: 1.0, ls: 0.1, lr: 0.5}
```

### 7.5 Diagnostics (every eval step)

- Effective rank of carry over batch (SVD entropy); per-dim std; cosine-to-init. Alert on rank collapse.
- k-step self-prediction error (k = 1..10) on held-out expert AND held-out coverage segments (report separately).
- LANG arm only — rollout invariance: same `(h, q, a)` rolled with `e` vs shuffled `e'`; report carry cos-divergence + Δ of pose-probe outputs.

### 7.6 First experiments (order)

1. **Ladder, columns 1–2** (probe subset, before full WM): linear decodability of object pose / success + 1-step controlled prediction, rows = {pooled trunk feature (paradigm-c negative control, own small predictor), SigLIP-tower pre-trunk, extracted z (LANG, FREE)}.
2. **4-config WM sweep** ({EMA, EMA+rec} × {LANG, FREE}); collapse diagnostics + 1-step error arbitrate.
3. **k-step curves + invariance** on winning config, 3 arms.
4. **Mini ranking**: 20 snapshot states × ~16 stratified candidates (8 on-policy + σ tiers + cross-ℓ + random); counterfactual GT via Stage-3 replay (open-loop, replay-valid filtered); Spearman + top-1 regret; baselines = z_t-only scoring, language-prior-only scorer.

---

## 8. Footgun checklist

- [ ] norm stats recomputed per finetune dataset; never reuse blindly
- [ ] relative-action reconstruction direction verified against one reference rollout
- [ ] OSC controller reset after every `set_state`
- [ ] 10 settle steps after reset/init-state
- [ ] constant-prompt feature-shift quantified before trusting taps
- [ ] `replay_valid` filtering applied in GT and coverage data
- [ ] shared randomization schedule files checked into repo
- [ ] environment fingerprint (lerobot/LIBERO commits, torch/CUDA) in every results JSON — editable installs drift silently
- [ ] suites evaluated in separate invocations; `--seed` pinned explicitly

## 9. Open decisions [DEC]

- Progress heads Φ (template potential functions) — deferred until d̂-only loop runs
- Precondition head b̂ — deferred until chain construction
- Predictor stride 1 vs 2 (memory)
- Recon-anchor target granularity: token-mean (current) vs 16-token subsample
- LeRobot's 6k finetune: `train_expert_only` or full? (affects wording of the probe subject only)