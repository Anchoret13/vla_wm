#!/usr/bin/env python
"""V6.5 — π0.5 flow-interface post-training (one of three matched
checkpoints: gt / random / wm).

Registered contract (`plan_and_progress/2026-07-30.md`, execution record):
- trainable whitelist: the zero-initialized W_z projection ONLY (π0.5,
  E_a/T/U, every WM head frozen; byte-identity asserted);
- every optimizer step in all three checkpoints: identical base batch
  {GT branch (both state modes, averaged) + rehearsal + stock-flow trust
  region}; wm/random add their matched generated loss at the SAME emitted
  states (wm: candidates[i_star]; random: candidates[i_random]); gt adds
  a literal zero;
- first-ten credit for GT/generated chunks; full-50 rehearsal; trust =
  ||v_adapt − v_stock||² on rehearsal flow noise/time (coefficient 1.0,
  frozen);
- registered budget: N_STEPS=300, AdamW lr 1e-4 / wd 1e-4, grad-norm 1.0,
  item schedules k mod |list|, noise bases 997000 (teacher/generated) and
  997500000 (rehearsal/trust) — identical across arms;
- rehearsal = stock train-source executed 50-chunks (recorded deviation:
  LIBERO demo caches store 10-action blocks, not 50-chunks; source
  rollouts are π0.5-supported full-prompt behavior);
- save every 50 steps; final selection = min dev objective (rehearsal +
  trust on the 5 dev sources), identical normalizers; tie → earliest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

from lcwm.v06_model import V06State  # noqa: E402

DATA = Path("/home/stargazer/Desktop/vla_wm/datasets/libero_loho_public_v1"
            "/v06_effect_crossed")
WM_CKPT = (REPO_ROOT / "results" / "libero_loho_public_v1" / "v06_wm"
           / "checkpoint_final.pt")
TEACHERS = (REPO_ROOT / "results" / "libero_loho_public_v1"
            / "v06_teachers")
N_STEPS = 300
LR, WD, GRAD_NORM = 1e-4, 1e-4, 1.0
TRUST_COEF = 1.0
T_BASE, R_BASE = 997_000, 997_500_000


def noise_time(policy, base, k, device):
    g = torch.Generator().manual_seed(base + k)
    cfg = policy.config
    noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim,
                        generator=g).to(device)
    time = torch.rand(1, generator=g).to(device)
    return noise, time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True,
                        choices=("gt", "random", "wm"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(0)

    from lcwm.chassis import Pi05Runner
    from lcwm.lc_flow import (cached_branch_flow_loss, freeze_pi05_base,
                              raw_flow_losses_from_prefix)
    from lcwm.sampler import prefix_forward

    model = V06State().to(device)
    bundle = torch.load(WM_CKPT, weights_only=False)
    model.load_state_dict(bundle["model"])
    model.eval()
    wm_hash = hashlib.sha256(WM_CKPT.read_bytes()).hexdigest()
    frozen_ref = {n: p.detach().clone()
                  for n, p in model.named_parameters()
                  if not n.startswith("w_z")}
    for n, p in model.named_parameters():
        p.requires_grad_(n.startswith("w_z"))
    trainable = [p for n, p in model.named_parameters()
                 if n.startswith("w_z")]
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WD)

    runner = Pi05Runner(suite_name="libero_10")
    freeze_pi05_base(runner.policy)

    teachers = torch.load(TEACHERS / "teacher_manifest.pt",
                          weights_only=False)
    gt = torch.load(TEACHERS / "gt_manifest.pt", weights_only=False)
    assert teachers["wm_checkpoint_sha256"] == wm_hash

    sources = {}
    for sp in sorted((DATA / "sources").glob("*.pt")):
        s = torch.load(sp, weights_only=False)
        sources[s["source_id"]] = s

    # ---- precompute z (recurrent + reset) at needed states (no grad) ----
    from scripts.build_v06_teachers import unroll_z
    z_cache: dict[str, list] = {}
    z_reset_cache: dict[tuple, torch.Tensor] = {}
    feats_cache: dict[str, dict] = {}
    needed = sorted({r["source_id"] for r in gt["rows"]}
                    | {r["source_id"] for r in teachers["rows"]}
                    | {sid for sid, s in sources.items()})
    with torch.no_grad():
        for sid in needed:
            fp = DATA / "features" / f"{sid}__canonical.pt"
            feats = torch.load(fp, weights_only=False)
            feats_cache[sid] = feats
            z_cache[sid] = [z.detach() for z in unroll_z(
                model, feats, sources[sid]["rows"], device)]

    def z_at(sid, d, mode):
        if mode == "recurrent":
            return z_cache[sid][d]
        key = (sid, d)
        if key not in z_reset_cache:
            feats = feats_cache[sid]
            with torch.no_grad():
                z_reset_cache[key] = model.initial_state(
                    feats["h"][d][None].float().to(device),
                    feats["mask"][d][None].to(device)).detach()
        return z_reset_cache[key]

    prefix_lru: dict[tuple, object] = {}

    @torch.no_grad()
    def prefix_at(sid, d):
        key = (sid, d)
        if key not in prefix_lru:
            if len(prefix_lru) > 40:
                prefix_lru.clear()
            src = sources[sid]
            batch = runner._obs_to_policy_batch(
                src["rows"][d]["obs"], src["language_canonical"])
            prefix_lru[key] = prefix_forward(runner.policy, batch)
        return prefix_lru[key]

    gt_items = sorted(gt["rows"], key=lambda r: (r["source_id"],
                                                 r["decision"]))
    emit_items = sorted(
        [r for r in teachers["rows"] if r["emit_model_teacher"]],
        key=lambda r: (r["source_id"], r["decision"]))
    rehearsal_items = []
    for sid, s in sorted(sources.items()):
        if s["split"] == "train" and s["provenance"] == "stock":
            for row in s["rows"]:
                rehearsal_items.append((sid, row["decision"]))
    dev_items = []
    for sid, s in sorted(sources.items()):
        if s["split"] == "dev":
            for row in s["rows"][::4]:
                dev_items.append((sid, row["decision"]))
    print(f"[{args.arm}] GT {len(gt_items)}, emitted {len(emit_items)}, "
          f"rehearsal {len(rehearsal_items)}, dev {len(dev_items)}",
          flush=True)

    b0 = model.policy_bias(z_at(gt_items[0]["source_id"],
                                gt_items[0]["decision"], "recurrent"))
    assert float(b0.abs().max()) == 0.0, "W_z bias not 0 at init"

    from lcwm.lc_flow import denoise_step_with_lc_bias
    from lcwm.sampler import _expand_cache
    from lerobot.utils.constants import ACTION

    def flow_loss_first10(sid, d, chunk, bias, noise, time):
        prefix = prefix_at(sid, d)
        loss, _ = cached_branch_flow_loss(
            runner.policy, prefix, chunk[None].to(device).float(), bias,
            torch.tensor([1.0], device=device), max_executed=10,
            noise=noise, time=time)
        return loss

    def rehearsal_and_trust(sid, d, bias, noise, time):
        src = sources[sid]
        chunk = src["rows"][d]["chunk_norm"][None].to(device).float()
        prefix = prefix_at(sid, d)
        rl = raw_flow_losses_from_prefix(
            runner.policy, chunk, bias, prefix,
            noise=noise, time=time).mean()
        policy = runner.policy
        actions = policy.prepare_action({ACTION: chunk})
        n2 = noise
        t2 = time
        x_t = t2[:, None, None] * n2 + (1 - t2[:, None, None]) * actions
        cache = _expand_cache(prefix.past_key_values, 1)
        v_a = denoise_step_with_lc_bias(
            policy.model, prefix.pad_masks, cache, x_t, t2, bias)
        with torch.no_grad():
            v_s = denoise_step_with_lc_bias(
                policy.model, prefix.pad_masks, cache, x_t, t2,
                torch.zeros_like(bias))
        trust = torch.nn.functional.mse_loss(v_a, v_s)
        return rl, trust

    logs = []
    for k in range(N_STEPS):
        optimizer.zero_grad(set_to_none=True)
        losses = {}
        # GT branch: both state modes, averaged
        g = gt_items[k % len(gt_items)]
        noise, time = noise_time(runner.policy, T_BASE, k, device)
        gt_loss = 0.0
        for mode in ("reset", "recurrent"):
            bias = model.policy_bias(
                z_at(g["source_id"], g["decision"], mode))
            gt_loss = gt_loss + flow_loss_first10(
                g["source_id"], g["decision"], g["chunk_norm"], bias,
                noise, time)
        losses["gt"] = gt_loss / 2
        # rehearsal + trust
        sid, d = rehearsal_items[k % len(rehearsal_items)]
        noise2, time2 = noise_time(runner.policy, R_BASE, k, device)
        bias_r = model.policy_bias(z_at(sid, d, "recurrent"))
        rl, trust = rehearsal_and_trust(sid, d, bias_r, noise2, time2)
        losses["rehearsal"] = rl
        losses["trust"] = TRUST_COEF * trust
        # matched generated loss
        if args.arm in ("random", "wm") and emit_items:
            e = emit_items[k % len(emit_items)]
            idx = e["i_star"] if args.arm == "wm" else e["i_random"]
            bias_g = model.policy_bias(
                z_at(e["source_id"], e["decision"], "recurrent"))
            noise3, time3 = noise_time(
                runner.policy, T_BASE + 500_000, k, device)
            losses["generated"] = flow_loss_first10(
                e["source_id"], e["decision"], e["candidates"][idx],
                bias_g, noise3, time3)
        else:
            losses["generated"] = torch.zeros((), device=device)
        total = sum(losses.values())
        total.backward()
        torch.nn.utils.clip_grad_norm_(trainable, GRAD_NORM)
        optimizer.step()
        if (k + 1) % 50 == 0:
            out = (REPO_ROOT / "results" / "libero_loho_public_v1"
                   / "v06_policy" / f"v06_{args.arm}")
            out.mkdir(parents=True, exist_ok=True)
            # dev objective: rehearsal + trust on dev sources, fixed
            # noise/time (base 996000 + index) — identical across arms
            with torch.no_grad():
                dev_vals = []
                for i, (sid, d) in enumerate(dev_items):
                    nz, tm = noise_time(runner.policy, 996_000, i,
                                        device)
                    bias_d = model.policy_bias(z_at(sid, d, "recurrent"))
                    rl_d, tr_d = rehearsal_and_trust(sid, d, bias_d,
                                                     nz, tm)
                    dev_vals.append(float(rl_d + TRUST_COEF * tr_d))
                dev_obj = sum(dev_vals) / len(dev_vals)
            torch.save({"w_z": model.w_z.state_dict(), "step": k + 1,
                        "dev_objective": dev_obj,
                        "wm_checkpoint_sha256": wm_hash},
                       out / f"wz_step{k + 1:04d}.pt")
            logs.append({"step": k + 1, "dev_objective": dev_obj,
                         **{n: float(v) for n, v in losses.items()}})
            print(f"[{args.arm} step {k + 1}] " + " ".join(
                f"{n}={float(v):.4f}" for n, v in losses.items())
                + f" dev={dev_obj:.4f} |W_z|="
                f"{float(model.w_z.weight.norm()):.4f}", flush=True)

    for n, p in model.named_parameters():
        if n in frozen_ref:
            assert torch.equal(p.detach(), frozen_ref[n]), \
                f"frozen param {n} changed"
    out = (REPO_ROOT / "results" / "libero_loho_public_v1" / "v06_policy"
           / f"v06_{args.arm}")
    best = min(logs, key=lambda r: (r["dev_objective"], r["step"]))
    selected = torch.load(out / f"wz_step{best['step']:04d}.pt",
                          weights_only=False)
    torch.save({**selected, "selection": best,
                "config": {"n_steps": N_STEPS, "lr": LR, "wd": WD,
                           "trust_coef": TRUST_COEF}},
               out / "wz_selected.pt")
    (out / "policy_logs.json").write_text(json.dumps(logs))
    print(f"[{args.arm}] selected step {best['step']} "
          f"(dev {best['dev_objective']:.4f}) -> {out}", flush=True)


if __name__ == "__main__":
    main()
