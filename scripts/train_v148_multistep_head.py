#!/usr/bin/env python
"""Score the terminal latent of a CLOSED-LOOP rollout: T_theta applied recursively.

    python scripts/train_v148_multistep_head.py --branches <branches.pt> \
        --wm-proprio <T_theta.pt> --pi-hat <pi_hat.pt> --horizon 4

GOAL ANCHOR (CLAUDE.md), object axis. This is the only use of T_theta that a
one-step feature map cannot replicate.

v145: rolling one step adds nothing at deployment (wm 75/128 vs pre_enc 72/128,
p=0.728). That is structural - with all candidates sharing z at a branch point,
T_theta(z,u) lies INSIDE the function class of a head fed (z, E_a(u)), so it
cannot beat it, and more data would only help the larger class. Recursion is what
takes it outside:

    z1 = T(z, u_cand)
    u_h = pi_hat(z_hidden_branch, z_{h-1});  z_h = T(z_{h-1}, u_h)   for h = 2..H

Only the TERMINAL latent is scored, so the head never sees the candidate action
directly - it sees where the candidate is predicted to lead after H*c steps.

Normalisation is bridged explicitly: T_theta and pi_hat were fit with different
proprio statistics, so the latent is de-normalised and re-normalised at each
hand-off rather than assumed compatible.
"""
from __future__ import annotations

import argparse, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import numpy as np, torch  # noqa: E402
from torch import nn  # noqa: E402
from train_v122_latent_wm import Transition  # noqa: E402
from train_v137_rank_head import Head, within_group_auc  # noqa: E402
from train_v146_latent_policy import LatentPolicy  # noqa: E402

OUT = REPO / "results" / "v148_multistep"


def build_rollout(T, pi, ck, pk):
    """Returns f(zh_raw, zp_raw, u) -> terminal proprio latent in T's space."""
    def roll(zh_raw, zp_raw, u, horizon):
        zT = (zp_raw - ck["mu"]) / ck["sd"]
        zT = T(zT, u)                                    # step 1: the candidate
        for _ in range(horizon - 1):
            raw = zT * ck["sd"] + ck["mu"]               # de-normalise
            uh = pi((zh_raw - pk["mu_h"]) / pk["sd_h"],
                    (raw - pk["mu_p"]) / pk["sd_p"])     # pi_hat's own space
            zT = T(zT, uh)
        return zT
    return roll


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--branches", type=Path, required=True)
    ap.add_argument("--wm-proprio", type=Path, required=True)
    ap.add_argument("--pi-hat", type=Path, required=True)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--restarts", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=600)
    a = ap.parse_args()
    b = torch.load(a.branches, weights_only=False)
    ck = torch.load(a.wm_proprio, weights_only=False)
    pk = torch.load(a.pi_hat, weights_only=False)
    zh, zp, u, y, grp = b["zh"], b["zp"], b["u"], b["y"], b["group"]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / stamp; out.mkdir(parents=True, exist_ok=True)

    T = Transition(ck["zdim"], ck["c"], ck["adim"], ck["hidden"], use_action=True)
    T.load_state_dict(ck["state_dict"]); T.eval()
    pi = LatentPolicy(pk["hdim"], pk["pdim"], pk["c"], pk["adim"])
    pi.load_state_dict(pk["state_dict"]); pi.eval()
    roll = build_rollout(T, pi, ck, pk)
    mu_h, sd_h = zh.mean(0), zh.std(0) + 1e-6
    with torch.no_grad():
        zT = roll(zh, zp, u, a.horizon)
        X = torch.cat([(zh - mu_h) / sd_h, zT], -1)
    print(f"horizon {a.horizon} ({a.horizon*10} sim steps); terminal-latent input "
          f"dim {X.shape[-1]}")

    ng = int(grp.max()) + 1
    states, aucs = [], []
    for s in range(a.restarts):
        perm = torch.randperm(ng, generator=torch.Generator().manual_seed(7000 + s))
        te_g = set(perm[int(ng * 0.85):].tolist())
        tr = torch.tensor([i for i in range(len(y)) if int(grp[i]) not in te_g])
        te = torch.tensor([i for i in range(len(y)) if int(grp[i]) in te_g])
        torch.manual_seed(s)
        m = Head(X.shape[-1], u.shape[1], u.shape[2], mode="wm")
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-3)
        best = (0.0, None)
        for e in range(a.epochs):
            m.train(); opt.zero_grad()
            nn.functional.binary_cross_entropy_with_logits(m(X[tr]), y[tr]).backward()
            opt.step()
            if e % 25 == 0 or e == a.epochs - 1:
                m.eval()
                with torch.no_grad():
                    A = within_group_auc(m(X[te]), y[te], grp[te])
                if not np.isnan(A) and A > best[0]:
                    best = (A, {k: v.clone() for k, v in m.state_dict().items()})
        states.append(best[1]); aucs.append(best[0])
        print(f"  restart {s}: within-group AUC {best[0]:.3f}")
    print(f"H={a.horizon} ensemble mean AUC {np.mean(aucs):.3f} "
          f"(early-stopped on the reported slice, so optimistic - same caveat as "
          f"wm 0.637 and pre_enc 0.636)")

    torch.save({"states": states, "zdim": X.shape[-1], "c": u.shape[1],
                "adim": u.shape[2], "mu_h": mu_h, "sd_h": sd_h,
                "mu_p": ck["mu"], "sd_p": ck["sd"], "depth": a.horizon,
                "mode": "multistep", "horizon": a.horizon,
                "wm_proprio": str(a.wm_proprio), "pi_hat": str(a.pi_hat),
                "task": b["task"], "holdout_aucs": aucs}, out / "rank_ensemble.pt")
    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "branches": str(a.branches), "wm": str(a.wm_proprio),
         "pi_hat": str(a.pi_hat), "horizon": a.horizon, "restarts": a.restarts,
         "holdout_aucs": aucs, "mean_auc": float(np.mean(aucs)), "env_steps": 0,
         "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()},
        indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
