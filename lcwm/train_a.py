"""Candidate-A trainer + §11.2 diagnostics (pre-registered 2026-07-21.md).

Losses: BYOL-style carry self-prediction against the EMA target chain
(multi-anchor, rollout T_ROLL), masked predicate BCE, progress MSE, proprio
anchor. Diagnostics: k=1..4 rollout cos-dist vs trivial-copy, action-shuffle /
zero-action, task-shuffle intervention, predicate F1 / progress R², effective
rank / per-dim std / cos-to-init.

Usage: python -m lcwm.train_a --arm siglip --seed 0 [--epochs 30]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lcwm.candidate_a import CandidateA, cosdist
from lcwm.wm_data import WMSeqDataset

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "wm_sweep"
CKPT_DIR = Path("/home/stargazer/Desktop/vla_wm/checkpoints/wm_sweep")
BURN_IN, T_ROLL = 2, 2
W_PRED, W_PROG, W_Q = 1.0, 0.5, 0.1


def to_dev(b, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in b.items()}


def loss_step(model, batch, dev, anchor_scale: float = 1.0, var_reg: float = 0.0):
    e = model.e_task(batch["task_id"])
    z = model.encode(batch, e)                      # (B,L,M,d) online
    with torch.no_grad():
        zt = model.encode(batch, e.detach(), ema=True)
    B, L = z.shape[:2]

    l_sp, n_anchor = 0.0, 0
    for t0 in range(BURN_IN, L - T_ROLL):
        zh = z[:, t0]
        for j in range(1, T_ROLL + 1):
            zh = model.transition(zh, batch["actions"][:, t0 + j - 1], e)
            l_sp = l_sp + cosdist(zh, zt[:, t0 + j])
            n_anchor += 1
    l_sp = l_sp / max(n_anchor, 1)

    zf = z[:, BURN_IN:]
    logits, prog, qhat = model.heads(zf.reshape(B * (L - BURN_IN), *zf.shape[2:]))
    bits = batch["bits"][:, BURN_IN:].reshape(-1, logits.shape[-1])
    mask = batch["bits_mask"][:, BURN_IN:].reshape(-1, logits.shape[-1])
    l_pred = (torch.nn.functional.binary_cross_entropy_with_logits(
        logits, bits, reduction="none") * mask).sum() / mask.sum().clamp(min=1)
    l_prog = torch.nn.functional.mse_loss(
        prog, batch["progress"][:, BURN_IN:].reshape(-1))
    l_q = torch.nn.functional.mse_loss(
        qhat, batch["q"][:, BURN_IN:].reshape(-1, 9))
    total = l_sp + anchor_scale * (W_PRED * l_pred + W_PROG * l_prog + W_Q * l_q)
    l_var = torch.tensor(0.0, device=dev)
    if var_reg > 0:  # VICReg-lite: hinge per-dim std of LN'd carry toward 1
        zf = torch.nn.functional.layer_norm(zf, zf.shape[-1:])
        std = zf.reshape(-1, zf.shape[-2] * zf.shape[-1]).std(0)
        l_var = torch.relu(1.0 - std).mean()
        total = total + var_reg * l_var
    return total, {"sp": float(l_sp.detach()), "pred": float(l_pred.detach()),
                   "prog": float(l_prog.detach()), "q": float(l_q.detach()),
                   "var": float(l_var.detach())}


@torch.no_grad()
def diagnostics(model, loader, dev):
    model.eval()
    K = 4
    agg = {f"k{j}": [] for j in range(1, K + 1)}
    agg |= {f"copy_k{j}": [] for j in range(1, K + 1)}
    agg |= {"k1_shuffle_a": [], "k1_zero_a": [], "tau_carry_div": [],
            "tau_dpred": [], "tau_dprog": []}
    logits_all, bits_all, mask_all, prog_hat, prog_gt, zs = [], [], [], [], [], []

    for batch in loader:
        batch = to_dev(batch, dev)
        e = model.e_task(batch["task_id"])
        z = model.encode(batch, e)
        zt = model.encode(batch, e, ema=True)
        B, L = z.shape[:2]
        t0 = BURN_IN
        if L - t0 <= K:
            continue
        # k-step rollout + trivial copy
        preds = model.rollout(z[:, t0], batch["actions"][:, t0:], e, K)
        for j in range(1, K + 1):
            agg[f"k{j}"].append(float(cosdist(preds[j - 1], zt[:, t0 + j])))
            agg[f"copy_k{j}"].append(float(cosdist(zt[:, t0], zt[:, t0 + j])))
        # action controls (k=1)
        perm = torch.randperm(B, device=dev)
        p1 = model.transition(z[:, t0], batch["actions"][perm, t0], e)
        agg["k1_shuffle_a"].append(float(cosdist(p1, zt[:, t0 + 1])))
        p0 = model.transition(z[:, t0], torch.zeros_like(batch["actions"][:, t0]), e)
        agg["k1_zero_a"].append(float(cosdist(p0, zt[:, t0 + 1])))
        # task intervention (SHUFFLED tau): same history, wrong id
        wrong = (batch["task_id"] + 1 + torch.randint(
            0, 9, batch["task_id"].shape, device=dev)) % 10
        ew = model.e_task(wrong)
        zw = model.encode(batch, ew)
        agg["tau_carry_div"].append(float(cosdist(zw[:, t0:], z[:, t0:])))
        lo_t, pr_t, _ = model.heads(z[:, t0:].reshape(-1, *z.shape[2:]))
        lo_w, pr_w, _ = model.heads(zw[:, t0:].reshape(-1, *z.shape[2:]))
        agg["tau_dpred"].append(float(
            (torch.sigmoid(lo_w) - torch.sigmoid(lo_t)).abs().mean()))
        agg["tau_dprog"].append(float((pr_w - pr_t).abs().mean()))
        # heads quality + collapse inputs
        logits_all.append(lo_t)
        bits_all.append(batch["bits"][:, t0:].reshape(-1, lo_t.shape[-1]))
        mask_all.append(batch["bits_mask"][:, t0:].reshape(-1, lo_t.shape[-1]))
        prog_hat.append(pr_t)
        prog_gt.append(batch["progress"][:, t0:].reshape(-1))
        zs.append(z[:, t0:].reshape(-1, z.shape[-2] * z.shape[-1]))

    lo = torch.cat(logits_all)
    bi = torch.cat(bits_all)
    ma = torch.cat(mask_all)
    ph = torch.cat(prog_hat)
    pg = torch.cat(prog_gt)
    pred = (torch.sigmoid(lo) > 0.5).float()
    tp = ((pred == 1) & (bi == 1) & (ma == 1)).sum()
    fp = ((pred == 1) & (bi == 0) & (ma == 1)).sum()
    fn = ((pred == 0) & (bi == 1) & (ma == 1)).sum()
    f1 = float(2 * tp / (2 * tp + fp + fn).clamp(min=1))
    prog_r2 = float(1 - ((ph - pg) ** 2).sum() /
                    ((pg - pg.mean()) ** 2).sum().clamp(min=1e-8))

    Z = torch.cat(zs).float()
    Zc = Z - Z.mean(0, keepdim=True)
    s = torch.linalg.svdvals(Zc)
    p = (s ** 2) / (s ** 2).sum()
    eff_rank = float(torch.exp(-(p * (p + 1e-12).log()).sum()))
    per_dim_std = float(Z.std(0).mean())

    out = {k: float(np.mean(v)) for k, v in agg.items()}
    out |= {"pred_f1": f1, "prog_r2": prog_r2,
            "eff_rank": eff_rank, "per_dim_std": per_dim_std,
            "copy_margin_k1": (out["copy_k1"] - out["k1"]) / max(out["copy_k1"], 1e-8),
            "action_sensitivity_k1":
                (out["k1_shuffle_a"] - out["k1"]) / max(out["k1"], 1e-8)}
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
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(args.seed)

    tr = WMSeqDataset("train", args.arm)
    te = WMSeqDataset("dev", args.arm)
    print(f"[{args.arm}/s{args.seed}] train windows {len(tr)}  test {len(te)}")
    ltr = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=4,
                     drop_last=True)
    lte = DataLoader(te, batch_size=args.batch, num_workers=2)

    model = CandidateA().to(dev)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params: {n_par/1e6:.1f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * len(ltr))

    curves = []
    for ep in range(args.epochs):
        model.train()
        logs = []
        for batch in ltr:
            batch = to_dev(batch, dev)
            loss, parts = loss_step(model, batch, dev,
                                    args.anchor_scale, args.var_reg)
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

    diag = diagnostics(model, lte, dev)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.arm}_seed{args.seed}" + (f"_{args.tag}" if args.tag else "")
    torch.save(model.state_dict(), CKPT_DIR / f"{tag}.pt")
    report = {"arm": args.arm, "seed": args.seed, "epochs": args.epochs,
              "params_M": n_par / 1e6, "train_windows": len(tr),
              "dev_windows": len(te), "curves": curves, "diagnostics": diag,
              "config": {**vars(args), "W_PRED": W_PRED, "W_PROG": W_PROG,
                         "W_Q": W_Q, "BURN_IN": BURN_IN, "T_ROLL": T_ROLL,
                         "eval_split": "dev"},
              "manifest_sha": __import__("hashlib").sha256(
                  (Path(__import__("lcwm.wm_data", fromlist=["SEQ_DIR"]).SEQ_DIR)
                   / "split_manifest.json").read_bytes()).hexdigest()[:16],
              "git_commit": __import__("subprocess").run(
                  ["git", "rev-parse", "HEAD"], capture_output=True,
                  text=True).stdout.strip()}
    (OUT_DIR / f"{tag}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(diag, indent=2))


if __name__ == "__main__":
    main()
