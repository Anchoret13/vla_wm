#!/usr/bin/env python
"""V7.2C step 2 — matched state-to-flow post-training (6 trained arms).

L_policy = 1*soft first-ten weighted FM + 1*stock trust + 1*demo FM.
Trainable: the arm WM's bias-free LC projection `w_z` ONLY (bias fixed
permanently to zero); pi0.5 base AND action expert frozen; the WM is
frozen during its policy run. 300 AdamW steps lr 1e-4 wd 1e-4, clip
1.0, checkpoints every 50, final-step rule. All arms consume the SAME
matched schedule, flow noise/time tensors, demo rows, and trust
states (matched_training_schedule.pt), differing ONLY in recurrent
state source and candidate weight ledger:

  bc_u0            frozen W2 state   all mass on u0
  p0_no_wm_soft    shared (untrained) init state   W2 teacher ledger
  w0_soft          frozen W0 state   W2 teacher ledger
  w1_soft          frozen W1 state   W2 teacher ledger
  w2_soft          frozen W2 state   W2 teacher ledger
  w2_permutation   frozen W2 state   matched-permutation ledger

Per-component gradient assertions: soft-FM, trust, and demo each
produce a finite nonzero gradient on w_z when active (a constant demo
term is a mechanical failure). Tensor-only run; no simulator.
Output: results/libero_loho_public_v1/2026-08-03_v072_policy_r1/
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
LOSS_R1 = RESULTS / "2026-08-03_v072_lcwm_loss_r1"
TEACH = RESULTS / "2026-08-03_v072_teacher_r1"
OUT = RESULTS / "2026-08-03_v072_policy_r1"
DEMO_DIR = Path("/home/stargazer/Desktop/vla_wm/datasets"
                "/libero_loho_public_v1/demo_rehearsal_v067")
RID = "v072_policy_r1"
STEPS, LR, WD, GRAD_NORM, CKPT_EVERY = 300, 1e-4, 1e-4, 1.0, 50
T_BASE, R_BASE = 972_000, 972_500_000
ARMS = {
    "bc_u0": {"state": "w2_rank_state", "weights": "bc_u0"},
    "p0_no_wm_soft": {"state": "shared_init", "weights": "teacher"},
    "w0_soft": {"state": "w0_base", "weights": "teacher"},
    "w1_soft": {"state": "w1_outcome_state", "weights": "teacher"},
    "w2_soft": {"state": "w2_rank_state", "weights": "teacher"},
    "w2_permutation": {"state": "w2_rank_state",
                       "weights": "permutation"},
}


def sha_seed(p: str) -> int:
    return int.from_bytes(hashlib.sha256(
        p.encode()).digest()[:8], "big") & ((1 << 63) - 1)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=None)
    args = ap.parse_args()
    from lcwm.chassis import Pi05Runner
    from lcwm.lc_flow import (cached_branch_flow_loss,
                              denoise_step_with_lc_bias,
                              freeze_pi05_base,
                              raw_flow_losses_from_prefix)
    from lcwm.sampler import _expand_cache, prefix_forward
    from lcwm.v067_lineage import load_v067, sha256_file
    from lcwm.v06_model import V06State
    from lerobot.utils.constants import ACTION
    from scripts.train_v069_grounded_wz import round_robin

    device = torch.device("cuda")
    torch.manual_seed(0)
    (OUT / "checkpoints").mkdir(parents=True, exist_ok=True)

    led = [json.loads(x) for x in
           (TEACH / "teacher_ledger_pre_outcome.jsonl").open()]
    perm = {f"{r['source_id']}_d{r['decision']}": r for r in
            [json.loads(x) for x in
             (TEACH / "matched_permutation_ledger.jsonl").open()]}
    chunks = torch.load(TEACH / "candidate_chunks.pt",
                        weights_only=False)
    seal = json.loads((TEACH / "ledger_seal.json").read_text())

    runner = Pi05Runner(suite_name="libero_10")
    cfg = runner.policy.config
    freeze_pi05_base(runner.policy)
    for p in runner.policy.parameters():
        p.requires_grad_(False)   # action expert frozen too

    # ---- teacher anchors: prefixes + per-arm states -----------------
    anchors = []
    for r in led:
        akey = f"{r['source_id']}_d{r['decision']}"
        anchors.append({"akey": akey, **r})
    anchors.sort(key=lambda a: (a["task"], a["source_id"],
                                a["decision"]))
    src_cache = {}

    def src(sid):
        if sid not in src_cache:
            src_cache[sid] = torch.load(
                TEACH / "teacher_sources" / f"{sid}.pt",
                weights_only=False)
        return src_cache[sid]

    with torch.no_grad():
        anchor_prefix = {}
        for a in anchors:
            row = src(a["source_id"])["rows"][a["decision"]]
            b = runner._obs_to_policy_batch(
                row["obs"], src(a["source_id"])["language_canonical"])
            anchor_prefix[a["akey"]] = (
                prefix_forward(runner.policy, b), b)

    def unroll_state(wm, a):
        s = src(a["source_id"])
        instr = s["language_canonical"]
        z = None
        with torch.no_grad():
            for dd in range(a["decision"] + 1):
                rr = s["rows"][dd]
                b = runner._obs_to_policy_batch(rr["obs"], instr)
                pfx = prefix_forward(runner.policy, b)
                h, m = pfx.hidden.float(), pfx.pad_masks.bool()
                if z is None:
                    z = wm.initial_state(h, m)
                else:
                    prev = s["rows"][dd - 1]
                    aa = prev["chunk_norm"][None, :10].float() \
                        .to(device)
                    am = (torch.arange(10, device=device)[None]
                          < prev["executed_len"])
                    z = wm.step(z, aa, h, m, action_mask=am)
        return z.detach()

    # ---- demo rehearsal rows (frozen registered schedule) -----------
    demo_manifest = json.loads(
        (DEMO_DIR / "manifest.json").read_text())
    demo_eps = {}
    for e in demo_manifest["episodes"]:
        ep = load_v067(DEMO_DIR / e["path"], "demo_rehearsal")
        if ep["split"] == "train":
            demo_eps[(e["task_id"], e["demo"])] = ep
    by_task = {}
    for (tid, di), ep in sorted(demo_eps.items()):
        for ri, _row in enumerate(ep["rows"]):
            by_task.setdefault(tid, {}).setdefault(di, []).append(
                (tid, di, ri))
    reh_by_task = {t: round_robin(v) for t, v in by_task.items()}
    rehearsal = round_robin(reh_by_task)

    # ---- matched schedule (byte-identical across arms) --------------
    sched_path = OUT / "matched_training_schedule.pt"
    if not sched_path.exists():
        order = [anchors[k % len(anchors)]["akey"]
                 for k in range(STEPS)]
        torch.save({"anchor_order": order,
                    "rehearsal": rehearsal[:STEPS] if
                    len(rehearsal) >= STEPS else
                    [rehearsal[k % len(rehearsal)]
                     for k in range(STEPS)],
                    "t_base": T_BASE, "r_base": R_BASE}, sched_path)
    sched = torch.load(sched_path, weights_only=False)

    mp = OUT / "run_manifest.json"
    if not mp.exists():
        mp.write_text(json.dumps({
            "schema": "v072_policy_manifest_v1", "run_schema":
            "v072", "run_id": RID,
            "git_sha": subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True,
                cwd=REPO_ROOT).stdout.strip(),
            "trainable": "V06State.w_z weight ONLY (bias fixed 0); "
                         "pi0.5 base + action expert frozen; WM "
                         "frozen",
            "optimizer": {"steps": STEPS, "lr": LR, "wd": WD,
                          "clip": GRAD_NORM,
                          "ckpt_every": CKPT_EVERY,
                          "selection": "final step"},
            "lambdas": {"soft": 1.0, "trust": 1.0, "demo": 1.0},
            "teacher_seal": seal, "arms": ARMS,
            "wm_checkpoints": {k: sha256_file(
                LOSS_R1 / "checkpoints" / f"{k}_final.pt")
                for k in ("w0_base", "w1_outcome_state",
                          "w2_rank_state")},
            "video": "tensor-only run"}, indent=2))

    hub_prefix_lru = {}

    def prefix_demo(key, obs, lang):
        if key not in hub_prefix_lru:
            if len(hub_prefix_lru) > 30:
                hub_prefix_lru.clear()
            with torch.no_grad():
                b = runner._obs_to_policy_batch(obs, lang)
                hub_prefix_lru[key] = prefix_forward(
                    runner.policy, b)
        return hub_prefix_lru[key]

    def velocity_trust(prefix, chunk, bias, noise, time):
        chunk = torch.as_tensor(
            np.asarray(chunk))[None].to(device).float()
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

    state_z_memo = {}

    def train_arm(arm):
        acfg = ARMS[arm]
        wm = V06State().to(device)
        if acfg["state"] == "shared_init":
            st = torch.load(LOSS_R1 / "shared_initialization.pt",
                            weights_only=False)["model"]
        else:
            st = torch.load(LOSS_R1 / "checkpoints"
                            / f"{acfg['state']}_final.pt",
                            weights_only=False)["model"]
        wm.load_state_dict(st)
        for p in wm.parameters():
            p.requires_grad_(False)
        # trainable LC projection: fresh zero-init, bias fixed at 0
        torch.nn.init.zeros_(wm.w_z.weight)
        torch.nn.init.zeros_(wm.w_z.bias)
        wm.w_z.weight.requires_grad_(True)
        optimizer = torch.optim.AdamW([wm.w_z.weight], lr=LR,
                                      weight_decay=WD)
        if acfg["state"] not in state_z_memo:
            state_z_memo[acfg["state"]] = {
                a["akey"]: unroll_state(wm, a) for a in anchors}
        z_anchor = state_z_memo[acfg["state"]]
        led_by = {a["akey"]: a for a in anchors}

        def weights_of(akey):
            if acfg["weights"] == "bc_u0":
                w = {c: 0.0 for c in led_by[akey]["weights"]}
                w["u0"] = 1.0
                return w
            if acfg["weights"] == "permutation":
                return perm[akey]["perm_weights"]
            return led_by[akey]["weights"]

        grad_checks = {}
        logs = []
        for k in range(STEPS):
            akey = sched["anchor_order"][k]
            prefix, _b = anchor_prefix[akey]
            bias = wm.policy_bias(z_anchor[akey])
            g = torch.Generator().manual_seed(T_BASE + k)
            noise = torch.randn(1, cfg.chunk_size,
                                cfg.max_action_dim,
                                generator=g).to(device)
            time = torch.rand(1, generator=g).to(device)
            w = weights_of(akey)
            soft_terms = []
            for cid, wi in sorted(w.items()):
                if wi <= 0.0:
                    continue
                tgt = torch.as_tensor(np.asarray(
                    chunks[f"{akey}_{cid}"])).float()
                loss_c, _ = cached_branch_flow_loss(
                    runner.policy, prefix, tgt[None].to(device),
                    bias, torch.tensor([float(wi)], device=device),
                    max_executed=10, noise=noise, time=time)
                soft_terms.append(loss_c)
            soft = torch.stack(soft_terms).sum()
            u0c = chunks[f"{akey}_u0"]
            trust = velocity_trust(prefix, np.asarray(u0c)[None][0],
                                   bias, noise, time)
            tid, di, ri = sched["rehearsal"][k]
            ep = demo_eps[(tid, di)]
            drow = ep["rows"][ri]
            dprefix = prefix_demo(("d", tid, di, ri), drow["obs"],
                                  ep["language"])
            with torch.no_grad():
                hd = dprefix.hidden.float()
                md = dprefix.pad_masks.bool()
                zd = wm.initial_state(hd, md)
            dbias = wm.policy_bias(zd)
            g2 = torch.Generator().manual_seed(R_BASE + k)
            noise2 = torch.randn(1, cfg.chunk_size,
                                 cfg.max_action_dim,
                                 generator=g2).to(device)
            time2 = torch.rand(1, generator=g2).to(device)
            demo_loss = raw_flow_losses_from_prefix(
                runner.policy,
                drow["chunk_norm"][None].to(device).float(),
                dbias, dprefix, noise=noise2, time=time2).mean()
            if k == 0:
                for name, term in (("soft", soft),
                                   ("trust", trust),
                                   ("demo", demo_loss)):
                    optimizer.zero_grad(set_to_none=True)
                    term.backward(retain_graph=True)
                    gn = (wm.w_z.weight.grad ** 2).sum().sqrt()
                    grad_checks[name] = float(gn)
                    assert torch.isfinite(gn), f"{name} grad"
                assert grad_checks["soft"] > 0, "soft no grad"
                assert grad_checks["demo"] > 0, \
                    "demo term constant (mechanical failure)"
                (OUT / f"grad_checks_{arm}.json").write_text(
                    json.dumps(grad_checks, indent=2))
            optimizer.zero_grad(set_to_none=True)
            total = soft + trust + demo_loss
            total.backward()
            torch.nn.utils.clip_grad_norm_([wm.w_z.weight],
                                           GRAD_NORM)
            optimizer.step()
            if (k + 1) % CKPT_EVERY == 0:
                logs.append({"step": k + 1, "soft": float(soft),
                             "trust": float(trust),
                             "demo": float(demo_loss)})
                print(f"[{arm} {k + 1}/{STEPS}] soft="
                      f"{float(soft):.4f} trust={float(trust):.4f} "
                      f"demo={float(demo_loss):.4f}", flush=True)
        torch.save({"schema": "v072_policy_ckpt_v1",
                    "run_schema": "v072", "arm": arm,
                    "wm_state": acfg["state"],
                    "w_z_weight": wm.w_z.weight.detach().cpu(),
                    "grad_checks": grad_checks, "logs": logs},
                   OUT / "checkpoints" / f"{arm}_final.pt")
        return {"final": f"{arm}_final.pt",
                "grad_checks": grad_checks}

    results = {}
    for arm in ([args.arm] if args.arm else list(ARMS)):
        results[arm] = train_arm(arm)
        (OUT / "offline_metrics.json").write_text(
            json.dumps(results, indent=2))
    print(json.dumps({k: v["grad_checks"]
                      for k, v in results.items()}, indent=1),
          flush=True)
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
