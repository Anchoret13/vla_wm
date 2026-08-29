#!/usr/bin/env python
"""The ablation head: T_theta's FROZEN action encoder, transition never applied.

GOAL ANCHOR (CLAUDE.md). This is the arm that decides whether the deployment gain
belongs to the world model.

v137's `direct` arm built a RANDOM-INIT ActionEncoder while the wm arm used
T_theta's PRETRAINED one, so v139 compared "pretrained encoder + rollout" against
"random-init encoder" - a control for pretraining, not for rolling forward. An
audit reproduced pre_enc (pretrained encoder, no rollout) at within-group AUC
0.6600 against wm's 0.6645, i.e. the transition accounts for ~21% of a gap I had
attributed to it entirely.

This head sees exactly what the wm head sees except that T_theta is never applied:
    wm       [hidden, T_theta^depth(proprio, u)]
    pre_enc  [hidden, proprio, E_a(u)]              <- same frozen E_a, no rollout
Deployed head-to-head on identical candidate sets, this isolates the transition.
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

OUT = REPO / "results" / "v144_pre_enc_head"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--branches", type=Path, required=True)
    ap.add_argument("--wm-proprio", type=Path, required=True)
    ap.add_argument("--restarts", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=600)
    a = ap.parse_args()
    b = torch.load(a.branches, weights_only=False)
    ck = torch.load(a.wm_proprio, weights_only=False)
    zh, zp, u, y, grp = b["zh"], b["zp"], b["u"], b["y"], b["group"]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / stamp; out.mkdir(parents=True, exist_ok=True)

    T = Transition(ck["zdim"], ck["c"], ck["adim"], ck["hidden"], use_action=True)
    T.load_state_dict(ck["state_dict"]); T.eval()
    mu_p, sd_p = ck["mu"], ck["sd"]
    mu_h, sd_h = zh.mean(0), zh.std(0) + 1e-6
    with torch.no_grad():
        X = torch.cat([(zh - mu_h) / sd_h, (zp - mu_p) / sd_p, T.enc(u)], -1)
    print(f"pre_enc input dim {X.shape[-1]} (hidden + proprio + frozen E_a)")

    ng = int(grp.max()) + 1
    states, aucs = [], []
    for s in range(a.restarts):
        perm = torch.randperm(ng, generator=torch.Generator().manual_seed(7000 + s))
        te_g = set(perm[int(ng * 0.85):].tolist())
        tr = torch.tensor([i for i in range(len(y)) if int(grp[i]) not in te_g])
        te = torch.tensor([i for i in range(len(y)) if int(grp[i]) in te_g])
        torch.manual_seed(s)
        m = Head(X.shape[-1], u.shape[1], u.shape[2], mode="wm")   # no extra encoder
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
    print(f"pre_enc ensemble mean AUC {np.mean(aucs):.3f} "
          f"(note: early-stopped on the reported slice, so optimistic)")

    torch.save({"states": states, "zdim": X.shape[-1], "c": u.shape[1],
                "adim": u.shape[2], "mu_h": mu_h, "sd_h": sd_h, "mu_p": mu_p,
                "sd_p": sd_p, "depth": 0, "mode": "pre_enc",
                "wm_proprio": str(a.wm_proprio), "task": b["task"],
                "holdout_aucs": aucs}, out / "rank_ensemble.pt")
    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "branches": str(a.branches), "wm": str(a.wm_proprio),
         "mode": "pre_enc", "restarts": a.restarts, "holdout_aucs": aucs,
         "mean_auc": float(np.mean(aucs)), "env_steps": 0,
         "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()},
        indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
