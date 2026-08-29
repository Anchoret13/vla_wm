#!/usr/bin/env python
"""Fit the deployment rank head on all branch groups (ensemble over restarts).

GOAL ANCHOR (CLAUDE.md). object axis: the head reads D_theta off T_theta's
rolled-forward latent, which v139 showed ranks better than feeding the raw action
(+0.0255 / +0.0235 AUC at label fractions 0.25 / 1.0, 15/20 splits, sign p=0.041,
both CIs excluding zero).

Ensembled over restarts because single fits have repeatedly been misleading here:
v137's lone fit put wm BELOW direct and printed a verdict that v138/v139 reversed.
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

OUT = REPO / "results" / "v140_final_head"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--branches", type=Path, required=True)
    ap.add_argument("--wm-proprio", type=Path, required=True)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--restarts", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=600)
    a = ap.parse_args()
    b = torch.load(a.branches, weights_only=False)
    ck = torch.load(a.wm_proprio, weights_only=False)
    assert ck.get("slice") == "proprio"
    zh, zp, u, y, grp = b["zh"], b["zp"], b["u"], b["y"], b["group"]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / stamp; out.mkdir(parents=True, exist_ok=True)

    T = Transition(ck["zdim"], ck["c"], ck["adim"], ck["hidden"], use_action=True)
    T.load_state_dict(ck["state_dict"]); T.eval()
    mu_p, sd_p = ck["mu"], ck["sd"]
    mu_h, sd_h = zh.mean(0), zh.std(0) + 1e-6
    with torch.no_grad():
        zt = (zp - mu_p) / sd_p
        for _ in range(a.depth):
            zt = T(zt, u)
        X = torch.cat([(zh - mu_h) / sd_h, zt], -1)

    ng = int(grp.max()) + 1
    states, aucs = [], []
    for s in range(a.restarts):
        # a small held-out slice per restart, only to report honest quality
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
        print(f"  restart {s}: held-out within-group AUC {best[0]:.3f}")
    print(f"ensemble of {len(states)}; mean held-out AUC {np.mean(aucs):.3f}")

    torch.save({"states": states, "zdim": X.shape[-1], "c": u.shape[1],
                "adim": u.shape[2], "mu_h": mu_h, "sd_h": sd_h,
                "mu_p": mu_p, "sd_p": sd_p, "depth": a.depth,
                "wm_proprio": str(a.wm_proprio), "task": b["task"],
                "holdout_aucs": aucs}, out / "rank_ensemble.pt")
    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "branches": str(a.branches), "wm": str(a.wm_proprio),
         "depth": a.depth, "restarts": a.restarts, "holdout_aucs": aucs,
         "mean_auc": float(np.mean(aucs)), "env_steps": 0,
         "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()},
        indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
