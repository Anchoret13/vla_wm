#!/usr/bin/env python
"""pi_hat: predict the policy's next chunk from the latent, to close the loop.

    python scripts/train_v146_latent_policy.py --tape <tape.pt>

GOAL ANCHOR (CLAUDE.md), object axis. v145 measured that T_theta's rollout adds
nothing at deployment: wm 75/128 vs pre_enc 72/128, p = 0.728. The reason is
structural, not a tuning failure - at one branch point every candidate shares the
same z, so T_theta(z, u) is just a deterministic feature map of (z, u) and cannot
be more expressive than a head that eats (z, u) directly. One-step scoring cannot
distinguish "rolled forward" from "action features".

A forward model is only irreplaceable when used RECURSIVELY. That needs the
policy's next action at a state that exists only as a prediction, which the VLM
cannot supply - so approximate it in latent space:

    z1 = T(z, u_candidate);  u1 = pi_hat(z1);  z2 = T(z1, u1);  ...

Trained on sigma=0 rows only, since those are the frozen policy's own actions; the
sigma>0 rows exist to identify T_theta and would teach pi_hat the exploration
noise instead of the policy.

APPROXIMATION, stated because it bounds what this can show: the rollout advances
proprioception only. The pooled VLM hidden is held at its branch value, since
v134 measured its action gain at +0.81% - it barely moves in c steps - and there
is no way to advance it without running the VLM on an image that does not exist.
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

P = 25
OUT = REPO / "results" / "v146_latent_policy"


class LatentPolicy(nn.Module):
    """z_hidden (frozen at branch) + z_proprio (rolled) -> next chunk."""

    def __init__(self, hdim: int, pdim: int, c: int, adim: int, hidden: int = 512):
        super().__init__()
        self.c, self.adim = c, adim
        self.net = nn.Sequential(nn.LayerNorm(hdim + pdim),
                                 nn.Linear(hdim + pdim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, c * adim))

    def forward(self, zh, zp):
        return self.net(torch.cat([zh, zp], -1)).view(-1, self.c, self.adim)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tape", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=400)
    a = ap.parse_args()
    d = torch.load(a.tape, weights_only=False)
    z, u, ep, sg = d["z"], d["u"], d["episode"], d.get("sigma")
    assert sg is not None, "tape has no sigma column"
    keep = (sg == 0.0)
    print(f"{int(keep.sum())} of {len(z)} triples are sigma=0 (the policy's own actions)")
    z, u, ep = z[keep], u[keep], ep[keep]
    zh, zp = z[:, :-P], z[:, -P:]
    torch.manual_seed(0); np.random.seed(0)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / stamp; out.mkdir(parents=True, exist_ok=True)

    mu_h, sd_h = zh.mean(0), zh.std(0) + 1e-6
    mu_p, sd_p = zp.mean(0), zp.std(0) + 1e-6
    Zh, Zp = (zh - mu_h) / sd_h, (zp - mu_p) / sd_p

    nep = int(ep.max()) + 1
    perm = torch.randperm(nep, generator=torch.Generator().manual_seed(0))
    tr_ep = set(perm[:int(nep * 0.75)].tolist())
    tr = torch.tensor([i for i in range(len(z)) if int(ep[i]) in tr_ep])
    te = torch.tensor([i for i in range(len(z)) if int(ep[i]) not in tr_ep])
    print(f"episode-disjoint: train {len(tr)} / test {len(te)}")

    m = LatentPolicy(Zh.shape[-1], Zp.shape[-1], u.shape[1], u.shape[2])
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    base = float((u[te] ** 2).mean())          # predicting zero
    mean_pred = float(((u[te] - u[tr].mean(0)) ** 2).mean())
    best = (float("inf"), None)
    for e in range(a.epochs):
        m.train(); opt.zero_grad()
        ((m(Zh[tr], Zp[tr]) - u[tr]) ** 2).mean().backward(); opt.step()
        if e % 10 == 0 or e == a.epochs - 1:
            m.eval()
            with torch.no_grad():
                mse = float(((m(Zh[te], Zp[te]) - u[te]) ** 2).mean())
            if mse < best[0]:
                best = (mse, {k: v.clone() for k, v in m.state_dict().items()})
    m.load_state_dict(best[1])
    print(f"pi_hat test MSE {best[0]:.5f}  | zero {base:.5f}  | train-mean {mean_pred:.5f}"
          f"  -> {best[0]/mean_pred:.3f}x the mean-action baseline")

    torch.save({"state_dict": best[1], "hdim": Zh.shape[-1], "pdim": Zp.shape[-1],
                "c": u.shape[1], "adim": u.shape[2], "mu_h": mu_h, "sd_h": sd_h,
                "mu_p": mu_p, "sd_p": sd_p, "task": d["task"],
                "test_mse": best[0], "mean_baseline": mean_pred}, out / "pi_hat.pt")
    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "tape": str(a.tape), "task": d["task"],
         "sigma0_triples": int(keep.sum()), "train": len(tr), "test": len(te),
         "test_mse": best[0], "zero_baseline": base, "mean_baseline": mean_pred,
         "rel_to_mean": best[0] / mean_pred, "env_steps": 0,
         "note": "trained on sigma=0 rows only; rollout advances proprio only, the "
                 "pooled hidden is held at its branch value (action gain +0.81%)",
         "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()},
        indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
