"""Candidate-A instrument v2 — trainer + repaired diagnostics.

Changes vs v1, per the 2026-07-22 review:
  item 3  heads are trained AND evaluated on rollout predictions Ẑ_{t+k}
          (future predicate bits, future proprio = physical readout, phase);
          teacher-forced readouts are kept but labeled "present".
  item 4  encoder no longer sees a_prev (shortcut removed structurally);
          matched within-task action shuffles added next to cross-batch;
          physical (future-q) readout under shuffle reported.
  item 5  task-swap metrics labeled difference-only; task-only (--zero-tokens)
          and no-task (--zero-task) baselines trainable as separate runs.
  item 9  reports serialize full config + split-manifest hash + git commit.

All outputs are DESCRIPTIVE. No pass/fail is computed here; interpretation
happens against pre-registered references in the dated progress file.

Usage: python -m lcwm.train_a --arm siglip --seed 0 [--epochs 30]
       [--var-reg 0.1] [--zero-tokens | --zero-task] [--tag name]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lcwm.candidate_a import CandidateA, cosdist
from lcwm.wm_data import SEQ_DIR, WMSeqDataset

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "wm_v2"
CKPT_DIR = Path("/home/stargazer/Desktop/vla_wm/checkpoints/wm_v2")
BURN_IN, T_ROLL = 2, 2
W_PRED, W_PROG, W_Q = 1.0, 0.5, 0.1        # present-state heads
WF_PRED, WF_PROG, WF_Q = 1.0, 0.5, 0.1     # future (rollout) heads


def to_dev(b, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in b.items()}


def head_losses(model, z_flat, bits, mask, prog, q):
    logits, prog_hat, q_hat = model.heads(z_flat)
    l_pred = (torch.nn.functional.binary_cross_entropy_with_logits(
        logits, bits, reduction="none") * mask).sum() / mask.sum().clamp(min=1)
    l_prog = torch.nn.functional.mse_loss(prog_hat, prog)
    l_q = torch.nn.functional.mse_loss(q_hat, q)
    return l_pred, l_prog, l_q


def loss_step(model, batch, dev, args):
    e = model.e_task(batch["task_id"])
    if args.zero_task:
        e = torch.zeros_like(e)
    z = model.encode(batch, e, zero_tokens=args.zero_tokens)
    with torch.no_grad():
        zt = model.encode(batch, e.detach(), ema=True,
                          zero_tokens=args.zero_tokens)
    B, L = z.shape[:2]

    l_sp = z.new_zeros(())
    lf_pred = z.new_zeros(())
    lf_prog = z.new_zeros(())
    lf_q = z.new_zeros(())
    n = 0
    for t0 in range(BURN_IN, L - T_ROLL):
        zh = z[:, t0]
        for j in range(1, T_ROLL + 1):
            zh = model.transition(zh, batch["actions"][:, t0 + j - 1], e)
            l_sp = l_sp + cosdist(zh, zt[:, t0 + j])
            lp, lg, lq = head_losses(
                model, zh.flatten(1),
                batch["bits"][:, t0 + j], batch["bits_mask"][:, t0 + j],
                batch["progress"][:, t0 + j], batch["q"][:, t0 + j])
            lf_pred, lf_prog, lf_q = lf_pred + lp, lf_prog + lg, lf_q + lq
            n += 1
    l_sp, lf_pred, lf_prog, lf_q = (x / max(n, 1)
                                    for x in (l_sp, lf_pred, lf_prog, lf_q))

    zf = z[:, BURN_IN:]
    lp_now, lg_now, lq_now = head_losses(
        model, zf.reshape(-1, zf.shape[-2] * zf.shape[-1]),
        batch["bits"][:, BURN_IN:].reshape(-1, batch["bits"].shape[-1]),
        batch["bits_mask"][:, BURN_IN:].reshape(-1, batch["bits"].shape[-1]),
        batch["progress"][:, BURN_IN:].reshape(-1),
        batch["q"][:, BURN_IN:].reshape(-1, 9))

    total = (l_sp
             + args.anchor_scale * (W_PRED * lp_now + W_PROG * lg_now
                                    + W_Q * lq_now)
             + WF_PRED * lf_pred + WF_PROG * lf_prog + WF_Q * lf_q)
    l_var = z.new_zeros(())
    if args.var_reg > 0:
        zn = torch.nn.functional.layer_norm(zf, zf.shape[-1:])
        std = zn.reshape(-1, zn.shape[-2] * zn.shape[-1]).std(0)
        l_var = torch.relu(1.0 - std).mean()
        total = total + args.var_reg * l_var
    return total, {"sp": float(l_sp.detach()),
                   "pred_now": float(lp_now.detach()),
                   "pred_fut": float(lf_pred.detach()),
                   "prog_fut": float(lf_prog.detach()),
                   "q_fut": float(lf_q.detach()),
                   "var": float(l_var.detach())}


def f1_of(logits, bits, mask):
    pred = (torch.sigmoid(logits) > 0.5).float()
    tp = ((pred == 1) & (bits == 1) & (mask == 1)).sum()
    fp = ((pred == 1) & (bits == 0) & (mask == 1)).sum()
    fn = ((pred == 0) & (bits == 1) & (mask == 1)).sum()
    return float(2 * tp / (2 * tp + fp + fn).clamp(min=1))


@torch.no_grad()
def diagnostics(model, loader, dev, args, k_max: int = 4):
    model.eval()
    keys = ([f"cos_k{j}" for j in range(1, k_max + 1)]
            + [f"cos_copy_k{j}" for j in range(1, k_max + 1)]
            + [f"qmse_fut_k{j}" for j in range(1, k_max + 1)]
            + ["cos_k1_shuf_x", "cos_k1_shuf_m", "cos_k1_zero",
               "qmse_k1_shuf_m", "tau_carry_div", "tau_dpred_fut",
               "tau_dprog_fut"])
    agg = {k: [] for k in keys}
    fut_logits = {j: [] for j in range(1, k_max + 1)}
    fut_bits = {j: [] for j in range(1, k_max + 1)}
    fut_mask = {j: [] for j in range(1, k_max + 1)}
    fut_prog = {j: [] for j in range(1, k_max + 1)}
    fut_prog_gt = {j: [] for j in range(1, k_max + 1)}
    fut_tid = []
    now_logits, now_bits, now_mask, zs = [], [], [], []

    for batch in loader:
        batch = to_dev(batch, dev)
        e = model.e_task(batch["task_id"])
        if args.zero_task:
            e = torch.zeros_like(e)
        z = model.encode(batch, e, zero_tokens=args.zero_tokens)
        zt = model.encode(batch, e, ema=True, zero_tokens=args.zero_tokens)
        B, L = z.shape[:2]
        t0 = BURN_IN
        if L - t0 <= k_max:
            continue

        zh = z[:, t0]
        for j in range(1, k_max + 1):
            zh = model.transition(zh, batch["actions"][:, t0 + j - 1], e)
            agg[f"cos_k{j}"].append(float(cosdist(zh, zt[:, t0 + j])))
            agg[f"cos_copy_k{j}"].append(float(cosdist(zt[:, t0], zt[:, t0 + j])))
            lo, pr, qh = model.heads(zh.flatten(1))
            agg[f"qmse_fut_k{j}"].append(float(
                torch.nn.functional.mse_loss(qh, batch["q"][:, t0 + j])))
            fut_logits[j].append(lo)
            fut_bits[j].append(batch["bits"][:, t0 + j])
            fut_mask[j].append(batch["bits_mask"][:, t0 + j])
            fut_prog[j].append(pr)
            fut_prog_gt[j].append(batch["progress"][:, t0 + j])
        fut_tid.append(batch["task_id"])

        # k=1 action controls: cross-batch, matched within-task, zero action
        a1 = batch["actions"][:, t0]
        perm_x = torch.randperm(B, device=dev)
        perm_m = torch.arange(B, device=dev)
        for tid in batch["task_id"].unique():
            idx = (batch["task_id"] == tid).nonzero(as_tuple=True)[0]
            perm_m[idx] = idx[torch.randperm(len(idx), device=dev)]
        for key, a_in in (("cos_k1_shuf_x", a1[perm_x]),
                          ("cos_k1_shuf_m", a1[perm_m]),
                          ("cos_k1_zero", torch.zeros_like(a1))):
            p = model.transition(z[:, t0], a_in, e)
            agg[key].append(float(cosdist(p, zt[:, t0 + 1])))
            if key == "cos_k1_shuf_m":
                _, _, qh = model.heads(p.flatten(1))
                agg["qmse_k1_shuf_m"].append(float(
                    torch.nn.functional.mse_loss(qh, batch["q"][:, t0 + 1])))

        # task swap — difference-only metric (correctness needs branch data);
        # input consistency holds for the siglip arm (prompt-free tokens)
        wrong = (batch["task_id"] + 1 + torch.randint(
            0, 9, batch["task_id"].shape, device=dev)) % 10
        ew = model.e_task(wrong)
        zw = model.encode(batch, ew, zero_tokens=args.zero_tokens)
        agg["tau_carry_div"].append(float(cosdist(zw[:, t0:], z[:, t0:])))
        zh_t = model.transition(z[:, t0], a1, e)
        zh_w = model.transition(zw[:, t0], a1, ew)
        lo_t, pr_t, _ = model.heads(zh_t.flatten(1))
        lo_w, pr_w, _ = model.heads(zh_w.flatten(1))
        agg["tau_dpred_fut"].append(float(
            (torch.sigmoid(lo_w) - torch.sigmoid(lo_t)).abs().mean()))
        agg["tau_dprog_fut"].append(float((pr_w - pr_t).abs().mean()))

        lo_n, _, _ = model.heads(z[:, t0:].reshape(-1, z.shape[-2] * z.shape[-1]))
        now_logits.append(lo_n)
        now_bits.append(batch["bits"][:, t0:].reshape(-1, lo_n.shape[-1]))
        now_mask.append(batch["bits_mask"][:, t0:].reshape(-1, lo_n.shape[-1]))
        zs.append(z[:, t0:].reshape(-1, z.shape[-2] * z.shape[-1]))

    out = {k: float(np.mean(v)) for k, v in agg.items() if v}
    out["f1_now"] = f1_of(torch.cat(now_logits), torch.cat(now_bits),
                          torch.cat(now_mask))
    for j in range(1, k_max + 1):
        out[f"f1_fut_k{j}"] = f1_of(torch.cat(fut_logits[j]),
                                    torch.cat(fut_bits[j]),
                                    torch.cat(fut_mask[j]))
        ph, pg = torch.cat(fut_prog[j]), torch.cat(fut_prog_gt[j])
        out[f"progR2_fut_k{j}"] = float(
            1 - ((ph - pg) ** 2).sum() /
            ((pg - pg.mean()) ** 2).sum().clamp(min=1e-8))
    # per-task future-F1 at k=1 (window-level task ids align 1:1 with k-rows)
    tids = torch.cat(fut_tid)
    lo1, bi1, ma1 = (torch.cat(fut_logits[1]), torch.cat(fut_bits[1]),
                     torch.cat(fut_mask[1]))
    out["f1_fut_k1_per_task"] = {
        int(t): f1_of(lo1[tids == t], bi1[tids == t], ma1[tids == t])
        for t in tids.unique()}

    Z = torch.cat(zs).float()
    s = torch.linalg.svdvals(Z - Z.mean(0, keepdim=True))
    p = (s ** 2) / (s ** 2).sum()
    out["eff_rank"] = float(torch.exp(-(p * (p + 1e-12).log()).sum()))
    out["per_dim_std"] = float(Z.std(0).mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["siglip", "real_ll", "fused"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--anchor-scale", type=float, default=1.0)
    ap.add_argument("--var-reg", type=float, default=0.0)
    ap.add_argument("--zero-tokens", action="store_true",
                    help="task-only baseline: encoder sees no vision")
    ap.add_argument("--zero-task", action="store_true",
                    help="no-task (FREE) baseline: e_task zeroed")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(args.seed)

    tr = WMSeqDataset("train", args.arm)
    te = WMSeqDataset("dev", args.arm)
    print(f"[{args.arm}/s{args.seed}] train windows {len(tr)}  dev {len(te)}")
    ltr = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=4,
                     drop_last=True)
    lte = DataLoader(te, batch_size=args.batch, num_workers=2)

    model = CandidateA().to(dev)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * len(ltr))

    curves = []
    for ep in range(args.epochs):
        model.train()
        logs = []
        for batch in ltr:
            batch = to_dev(batch, dev)
            loss, parts = loss_step(model, batch, dev, args)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            model.ema_update()
            logs.append(parts)
        mean = {k: float(np.mean([d[k] for d in logs])) for k in logs[0]}
        curves.append(mean)
        print(f"ep{ep:02d} " + " ".join(f"{k}={v:.4f}" for k, v in mean.items()),
              flush=True)

    diag = diagnostics(model, lte, dev, args)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.arm}_seed{args.seed}" + (f"_{args.tag}" if args.tag else "")
    torch.save(model.state_dict(), CKPT_DIR / f"{tag}.pt")
    report = {
        "instrument": "candidate_a_v2",
        "config": {**vars(args), "W": [W_PRED, W_PROG, W_Q],
                   "WF": [WF_PRED, WF_PROG, WF_Q],
                   "BURN_IN": BURN_IN, "T_ROLL": T_ROLL,
                   "eval_split": "dev", "params_M": n_par / 1e6},
        "split_manifest_sha": hashlib.sha256(
            (SEQ_DIR / "split_manifest.json").read_bytes()).hexdigest()[:16],
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"],
                                     capture_output=True, text=True,
                                     cwd=Path(__file__).parent).stdout.strip(),
        "train_windows": len(tr), "dev_windows": len(te),
        "curves": curves, "diagnostics": diag,
    }
    (OUT_DIR / f"{tag}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(diag, indent=2))


if __name__ == "__main__":
    main()
