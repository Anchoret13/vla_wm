#!/usr/bin/env python
"""V7.0.3 — minimal hard teacher × coupling pilot (four arms).

  arms = {hard, hard-random} × {affine, centered}

Affine (current interface):  b = tanh(W·Pool(z) + c)
Centered diagnostic:         b = tanh(W_c·[Pool(z) − μ_train]), no
intercept; μ_train = equal task→source→anchor mean of recurrent
Pool(z) over the 14 strict-clean train anchors (tensor SHA bound).

Contract (2026-08-01 V7.0.3): π0.5 and predictive LCWM frozen; coupling
trained from exact zero; recurrent state only; 401-row anchor schedule
(seed 70003, fixed T1→T5 task cycle, source/anchor cycles with
seed-chosen initial offsets → task visits 81/80/80/80/80); paired with
the frozen V6.9 one-epoch demo schedule; per step
  L = 1[pos]·L_corr(10) + L_anchor_trust(50) + L_demo(50)
      + L_demo_trust(50),   all coefficients 1.0;
flow seeds 997000+k (corr + anchor trust) and 997500000+k (demo pair),
identical across arms; AdamW 1e-4/1e-4, clip 1.0; step 401 final, NO
checkpoint selection. All four finals are emitted regardless of offline
loss; offline metrics annotate, never select.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

from lcwm.v06_model import V06State  # noqa: E402
from lcwm.v067_lineage import load_v067, sha256_file  # noqa: E402

DATA = Path("/home/stargazer/Desktop/vla_wm/datasets/libero_loho_public_v1"
            "/v06_effect_crossed")
DEMO_DIR = Path("/home/stargazer/Desktop/vla_wm/datasets"
                "/libero_loho_public_v1/demo_rehearsal_v067")
RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
ORACLE = RESULTS / "v070_policy" / "oracle"
OUT = RESULTS / "v070_policy" / "hard_pilot"
TASKS = ["loho_t1_drawer", "loho_t2_basket3", "loho_t3_tray",
         "loho_t4_tray", "loho_t5_drawer_cabinet"]
N_STEPS = 401
SCHED_SEED = 70_003
LR, WD, GRAD_NORM = 1e-4, 1e-4, 1.0
T_BASE, R_BASE = 997_000, 997_500_000
ARMS = ("hard_affine", "hard_centered", "hard_random_affine",
        "hard_random_centered")


class AffineCoupling(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(384, 1024)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, pool):
        return torch.tanh(self.lin(pool))


class CenteredCoupling(nn.Module):
    def __init__(self, mu):
        super().__init__()
        self.lin = nn.Linear(384, 1024, bias=False)
        nn.init.zeros_(self.lin.weight)
        self.register_buffer("mu", mu.clone())

    def forward(self, pool):
        return torch.tanh(self.lin(pool - self.mu))


def noise_time(cfg, base, k, device):
    g = torch.Generator().manual_seed(base + k)
    noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim,
                        generator=g).to(device)
    time = torch.rand(1, generator=g).to(device)
    return noise, time


def build_anchor_schedule(manifest_rows):
    by_task = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(manifest_rows):
        by_task[r["task"]][r["source_id"]].append(i)
    rng = np.random.default_rng(SCHED_SEED)
    offsets, visits = {}, defaultdict(int)
    src_lists = {}
    for task in TASKS:
        srcs = sorted(by_task[task])
        src_lists[task] = srcs
        offsets[task] = int(rng.integers(0, len(srcs)))
        for s in srcs:
            offsets[(task, s)] = int(rng.integers(
                0, len(by_task[task][s])))
    rows = []
    for k in range(N_STEPS):
        task = TASKS[k % 5]
        v = visits[task]
        visits[task] += 1
        srcs = src_lists[task]
        src = srcs[(offsets[task] + v) % len(srcs)]
        vs = visits[(task, src)]
        visits[(task, src)] += 1
        anchors = by_task[task][src]
        rows.append(anchors[(offsets[(task, src)] + vs)
                            % len(anchors)])
    counts = defaultdict(int)
    for i in rows:
        counts[manifest_rows[i]["task"]] += 1
    assert [counts[t] for t in TASKS] == [81, 80, 80, 80, 80], counts
    return rows


def main() -> None:
    from lcwm.chassis import Pi05Runner
    from lcwm.lc_flow import (cached_branch_flow_loss,
                              denoise_step_with_lc_bias,
                              freeze_pi05_base,
                              raw_flow_losses_from_prefix)
    from lcwm.sampler import _expand_cache, prefix_forward
    from lcwm.seq_prefix_cache import normalize_actions
    from lerobot.utils.constants import ACTION
    from scripts.train_v069_grounded_wz import round_robin

    device = torch.device("cuda")
    torch.manual_seed(0)
    OUT.mkdir(parents=True, exist_ok=True)

    teacher = torch.load(ORACLE / "hard_teacher_manifest.pt",
                         weights_only=False)
    control = torch.load(ORACLE / "hard_random_manifest.pt",
                         weights_only=False)
    rows_t, rows_r = teacher["rows"], control["rows"]
    assert [r["anchor"] for r in rows_t] == \
        [r["anchor"] for r in rows_r]

    model = V06State().to(device)
    wm = torch.load(RESULTS / "v069_predictive"
                    / "checkpoint_selected.pt", weights_only=False)
    model.load_state_dict(wm["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    runner = Pi05Runner(suite_name="libero_10")
    freeze_pi05_base(runner.policy)
    cfg = runner.policy.config

    # ---- recurrent z at the 14 anchors (frozen; shared by all arms) ----
    sources, anchor_z, anchor_prefix_obs = {}, {}, {}
    with torch.no_grad():
        for r in rows_t:
            sid = r["source_id"]
            if sid not in sources:
                sources[sid] = torch.load(
                    DATA / "corrections_sources_v069" / f"{sid}.pt",
                    weights_only=False)
        for r in rows_t:
            sid, d = r["source_id"], r["decision"]
            if (sid, d) in anchor_z:
                continue
            src = sources[sid]
            z = None
            for i, row in enumerate(src["rows"]):
                if i > d:
                    break
                batch = runner._obs_to_policy_batch(
                    row["obs"], src["language_canonical"])
                prefix = prefix_forward(runner.policy, batch)
                h = prefix.hidden.float()
                m = prefix.pad_masks.bool()
                if z is None:
                    z = model.initial_state(h, m)
                else:
                    prev = src["rows"][i - 1]
                    a = prev["chunk_norm"][None, :10].float().to(
                        device)
                    am = (torch.arange(10, device=device)[None]
                          < prev["executed_len"])
                    z = model.step(z, a, h, m, action_mask=am)
            anchor_z[(sid, d)] = z.detach()
            anchor_prefix_obs[(sid, d)] = (
                src["rows"][d]["obs"], src["language_canonical"],
                src["rows"][d]["chunk_norm"])
            print(f"[z] {sid} d={d}", flush=True)

    # μ_train: equal task -> source -> anchor mean of Pool(z)
    by_task = defaultdict(lambda: defaultdict(list))
    for r in rows_t:
        by_task[r["task"]][r["source_id"]].append(
            anchor_z[(r["source_id"], r["decision"])].mean(dim=1))
    task_means = []
    for task in sorted(by_task):
        src_means = [torch.cat(v).mean(dim=0)
                     for v in by_task[task].values()]
        task_means.append(torch.stack(src_means).mean(dim=0))
    mu_train = torch.stack(task_means).mean(dim=0)
    mu_sha = hashlib.sha256(
        mu_train.cpu().numpy().tobytes()).hexdigest()

    # ---- demo bank + recurrent demo z (identical to V6.9 machinery) ----
    demo_manifest = json.loads((DEMO_DIR / "manifest.json").read_text())
    demo_eps = {}
    for e in demo_manifest["episodes"]:
        demo_eps[(e["task_id"], e["demo"])] = load_v067(
            DEMO_DIR / e["path"], "demo_rehearsal")
    demo_z = {}
    with torch.no_grad():
        for key, ep in sorted(demo_eps.items()):
            if ep["split"] != "train":
                continue
            mean, std = ep["action_mean"], ep["action_std_eps"]
            z, zs = None, {}
            for j, o in enumerate(ep["obs_10"]):
                batch = runner._obs_to_policy_batch(
                    o["obs"], ep["language"])
                prefix = prefix_forward(runner.policy, batch)
                h = prefix.hidden.float()
                m = prefix.pad_masks.bool()
                if z is None:
                    z = model.initial_state(h, m)
                else:
                    prev = ep["obs_10"][j - 1]["chunk10_env"]
                    a = normalize_actions(prev, mean, std)[None].to(
                        device)
                    am = (torch.arange(10, device=device)[None]
                          < int(prev.shape[0]))
                    z = model.step(z, a, h, m, action_mask=am)
                if o["t"] % 50 == 0:
                    zs[o["t"]] = z.detach()
            demo_z[key] = zs
    print(f"[z] demo bank {len(demo_z)} episodes", flush=True)

    rehearsal_by_task = {}
    for (tid, di), ep in sorted(demo_eps.items()):
        if ep["split"] != "train":
            continue
        for ri, _row in enumerate(ep["rows"]):
            rehearsal_by_task.setdefault(tid, []).append(
                (tid, di, ri))
    for tid in rehearsal_by_task:
        by_demo = {}
        for it in rehearsal_by_task[tid]:
            by_demo.setdefault(it[1], []).append(it)
        rehearsal_by_task[tid] = round_robin(by_demo)
    rehearsal_schedule = round_robin(rehearsal_by_task)
    assert len(rehearsal_schedule) == 401

    anchor_schedule = build_anchor_schedule(rows_t)

    prefix_lru = {}

    @torch.no_grad()
    def prefix_of(kind, *key_and_obs):
        key = (kind,) + key_and_obs[0]
        if key not in prefix_lru:
            if len(prefix_lru) > 40:
                prefix_lru.clear()
            obs, lang = key_and_obs[1]
            batch = runner._obs_to_policy_batch(obs, lang)
            prefix_lru[key] = (prefix_forward(runner.policy, batch),
                               batch)
        return prefix_lru[key]

    def velocity_trust(prefix, chunk, bias, noise, time):
        chunk = chunk[None].to(device).float()
        actions = runner.policy.prepare_action({ACTION: chunk})
        x_t = time[:, None, None] * noise \
            + (1 - time[:, None, None]) * actions
        cache = _expand_cache(prefix.past_key_values, 1)
        v_b = denoise_step_with_lc_bias(
            runner.policy.model, prefix.pad_masks, cache, x_t, time,
            bias)
        with torch.no_grad():
            v_0 = denoise_step_with_lc_bias(
                runner.policy.model, prefix.pad_masks, cache, x_t,
                time, torch.zeros_like(bias))
        return torch.nn.functional.mse_loss(v_b, v_0)

    run_manifest = {
        "schema": "v070_hard_pilot_v1", "run_schema": "v070",
        "arms": list(ARMS), "n_steps": N_STEPS,
        "sched_seed": SCHED_SEED,
        "anchor_schedule": anchor_schedule,
        "mu_train_sha256": mu_sha,
        "teacher_sha256": sha256_file(
            ORACLE / "hard_teacher_manifest.pt"),
        "control_sha256": sha256_file(
            ORACLE / "hard_random_manifest.pt"),
        "flow_seed_bases": {"anchor": T_BASE, "demo": R_BASE},
        "optimizer": {"lr": LR, "wd": WD, "grad_norm": GRAD_NORM,
                      "final_step": N_STEPS,
                      "checkpoint_selection": "disabled"},
    }
    (OUT / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2))
    torch.save({"rows_teacher": rows_t, "rows_random": rows_r,
                "mu_train": mu_train.cpu(),
                "mu_train_sha256": mu_sha},
               OUT / "train_manifest.pt")

    offline = {}
    exposure = defaultdict(lambda: defaultdict(int))
    for arm in ARMS:
        rows = rows_t if "random" not in arm else rows_r
        centered = arm.endswith("centered")
        coupling = (CenteredCoupling(mu_train) if centered
                    else AffineCoupling()).to(device)
        optimizer = torch.optim.AdamW(coupling.parameters(), lr=LR,
                                      weight_decay=WD)
        logs = []
        for k in range(N_STEPS):
            optimizer.zero_grad(set_to_none=True)
            r = rows[anchor_schedule[k]]
            sid, d = r["source_id"], r["decision"]
            exposure[arm][r["anchor"]] += 1
            z = anchor_z[(sid, d)]
            bias = coupling(z.mean(dim=1))
            obs, lang, u0_chunk = anchor_prefix_obs[(sid, d)]
            prefix, _b = prefix_of("anchor", ((sid, d),),
                                   (obs, lang))
            noise, time = noise_time(cfg, T_BASE, k, device)
            losses = {}
            if r["positive"]:
                loss_c, _ = cached_branch_flow_loss(
                    runner.policy, prefix,
                    r["target"]["chunk_norm"][None].to(
                        device).float(),
                    bias, torch.tensor([1.0], device=device),
                    max_executed=10, noise=noise, time=time)
                losses["corr"] = loss_c
            losses["anchor_trust"] = velocity_trust(
                prefix, u0_chunk, bias, noise, time)
            tid, di, ri = rehearsal_schedule[k]
            ep = demo_eps[(tid, di)]
            drow = ep["rows"][ri]
            z_d = demo_z[(tid, di)][drow["t"]]
            bias_d = coupling(z_d.mean(dim=1))
            dprefix, _b2 = prefix_of("demo", ((tid, di, ri),),
                                     (drow["obs"], ep["language"]))
            noise2, time2 = noise_time(cfg, R_BASE, k, device)
            losses["demo"] = raw_flow_losses_from_prefix(
                runner.policy,
                drow["chunk_norm"][None].to(device).float(),
                bias_d, dprefix, noise=noise2, time=time2).mean()
            losses["demo_trust"] = velocity_trust(
                dprefix, drow["chunk_norm"], bias_d, noise2, time2)
            total = sum(losses.values())
            total.backward()
            torch.nn.utils.clip_grad_norm_(coupling.parameters(),
                                           GRAD_NORM)
            optimizer.step()
            if (k + 1) % 100 == 0 or (k + 1) == N_STEPS:
                logs.append({"step": k + 1,
                             **{n: float(v)
                                for n, v in losses.items()}})
                print(f"[{arm} {k + 1}/{N_STEPS}] " + " ".join(
                    f"{n}={float(v):.4f}"
                    for n, v in losses.items()), flush=True)

        # ---- final offline metrics (annotate, never select) ------------
        with torch.no_grad():
            shifts, target_losses, demo_losses = [], [], []
            from lcwm.lc_flow import sample_chunks_lc
            for r in rows:
                sid, d = r["source_id"], r["decision"]
                z = anchor_z[(sid, d)]
                bias = coupling(z.mean(dim=1))
                obs, lang, u0_chunk = anchor_prefix_obs[(sid, d)]
                prefix, batch = prefix_of("anchor", ((sid, d),),
                                          (obs, lang))
                seed = int.from_bytes(hashlib.sha256(
                    f"v070_pilot_shift|{sid}|{d}".encode()
                ).digest()[:8], "big") & ((1 << 63) - 1)
                cb = sample_chunks_lc(runner.policy, batch, bias,
                                      n=1, seed=seed, prefix=prefix)
                c0 = sample_chunks_lc(runner.policy, batch,
                                      torch.zeros_like(bias), n=1,
                                      seed=seed, prefix=prefix)
                shifts.append(
                    (cb[0, :10] - c0[0, :10]).cpu().numpy())
                if r["positive"]:
                    nz, tm = noise_time(cfg, T_BASE, 10_000, device)
                    loss_c, _ = cached_branch_flow_loss(
                        runner.policy, prefix,
                        r["target"]["chunk_norm"][None].to(
                            device).float(),
                        bias, torch.tensor([1.0], device=device),
                        max_executed=10, noise=nz, time=tm)
                    target_losses.append(float(loss_c))
            for j, (tid, di, ri) in enumerate(
                    rehearsal_schedule[::40]):
                ep = demo_eps[(tid, di)]
                drow = ep["rows"][ri]
                z_d = demo_z[(tid, di)][drow["t"]]
                bias_d = coupling(z_d.mean(dim=1))
                dprefix, _b2 = prefix_of(
                    "demo", ((tid, di, ri),),
                    (drow["obs"], ep["language"]))
                nz, tm = noise_time(cfg, R_BASE, 20_000 + j, device)
                demo_losses.append(float(raw_flow_losses_from_prefix(
                    runner.policy,
                    drow["chunk_norm"][None].to(device).float(),
                    bias_d, dprefix, noise=nz, time=tm).mean()))
            D = np.stack(shifts).reshape(len(shifts), -1)
            mean_shift = D.mean(axis=0)
            rho = float(
                (np.linalg.norm(D - mean_shift, axis=1) ** 2).mean()
                / max((np.linalg.norm(D, axis=1) ** 2).mean(), 1e-12))
            offline[arm] = {
                "target_flow_loss": (float(np.mean(target_losses))
                                     if target_losses else None),
                "action_shift_rms": float(np.sqrt((D ** 2).mean())),
                "rho_state": rho,
                "demo_retention_flow_loss":
                    float(np.mean(demo_losses)),
                "logs": logs,
            }
        torch.save({"schema": "v070_coupling_v1",
                    "run_schema": "v070", "arm": arm,
                    "coupling_type": ("centered" if centered
                                      else "affine"),
                    "state_dict": coupling.state_dict(),
                    "mu_train_sha256": mu_sha,
                    "final_step": N_STEPS,
                    "n_params": sum(p.numel() for p in
                                    coupling.parameters())},
                   OUT / f"{arm}_final.pt")
        print(f"[{arm}] final emitted: {offline[arm]}", flush=True)

    (OUT / "offline_metrics.json").write_text(
        json.dumps(offline, indent=2))
    (OUT / "exposure_log.json").write_text(json.dumps(
        {a: dict(sorted(v.items())) for a, v in exposure.items()},
        indent=2))
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
