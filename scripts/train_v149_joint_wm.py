#!/usr/bin/env python
"""Jointly learned latent world model, no reconstruction. Two transition families.

    python scripts/train_v149_joint_wm.py --variant ensemble|stochastic

WHY THIS REPLACES v122. The previous world model was built the wrong way and the
symptoms all trace to it:

  latent      FROZEN pooled VLM hidden (2048) + proprio (25), never learned. That
              representation is trained for language-vision alignment, so nothing
              ever asked it to be action-predictable - measured action gain +0.81%.
  transition  deterministic MLP regressing the EXACT next latent under MSE. Most
              of that 2048-d variance is decision-irrelevant, so capacity went
              there and the 25 informative proprio dims carried ~1.2% of the loss.
  training    three separate stages: freeze latent -> fit T_theta -> fit heads. The
              latent was never shaped by the value objective, which is why the
              heads came out flat along the direction actions move.

Here the encoder, the transition and the value/reward heads are trained TOGETHER,
and the latent only has to be sufficient for value and consistency - never to
reproduce an exact next latent, and never to reconstruct an observation.

  ensemble    K deterministic heads; the transition is represented by their spread
              (TD-MPC2 style). Uncertainty is free and usable for pessimistic
              planning.
  stochastic  diagonal-Gaussian transition trained by KL against the encoder's own
              next-state posterior (Dreamer-style prior/posterior, no decoder).

Collapse is prevented by the value and reward terms plus stop-grad on the
consistency target - there is no reconstruction loss to anchor the latent.
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
from train_v137_rank_head import within_group_auc  # noqa: E402

P = 25
OUT = REPO / "results" / "v149_joint_wm"


def mlp(i, h, o, layers=2):
    m = [nn.Linear(i, h), nn.GELU()]
    for _ in range(layers - 1):
        m += [nn.Linear(h, h), nn.GELU()]
    return nn.Sequential(*m, nn.Linear(h, o))


class JointWM(nn.Module):
    def __init__(self, obs_dim, c, adim, zdim=64, hidden=256, k=5, variant="ensemble"):
        super().__init__()
        self.variant, self.k, self.zdim = variant, k, zdim
        self.enc = nn.Sequential(nn.LayerNorm(obs_dim), mlp(obs_dim, hidden, zdim))
        self.aenc = mlp(c * adim, hidden, 64)
        if variant == "ensemble":
            self.dyn = nn.ModuleList([mlp(zdim + 64, hidden, zdim) for _ in range(k)])
        else:
            self.dyn = mlp(zdim + 64, hidden, 2 * zdim)      # mean, log-std
        self.value = mlp(zdim, hidden, 1)                    # P(branch succeeds)
        self.reward = mlp(zdim, hidden, 1)                   # P(Delta w > 0)

    def encode(self, o):
        z = self.enc(o)
        return z / (z.norm(dim=-1, keepdim=True) + 1e-6) * (self.zdim ** 0.5)

    def step(self, z, u, sample=False):
        h = torch.cat([z, self.aenc(u.flatten(1))], -1)
        if self.variant == "ensemble":
            outs = torch.stack([z + d(h) for d in self.dyn])     # residual
            return outs.mean(0), outs.std(0).mean(-1), outs
        mu, ls = self.dyn(h).chunk(2, -1)
        ls = ls.clamp(-5, 2)
        zn = z + mu
        if sample:
            zn = zn + torch.randn_like(zn) * ls.exp()
        return zn, ls.exp().mean(-1), (mu, ls)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tapes", type=Path, nargs="+", required=True)
    ap.add_argument("--branches", type=Path, required=True)
    ap.add_argument("--variant", choices=["ensemble", "stochastic"], required=True)
    ap.add_argument("--zdim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--restarts", type=int, default=5)
    a = ap.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / f"{a.variant}_{stamp}"; out.mkdir(parents=True, exist_ok=True)

    Z, U, ZN, EP, DW = [], [], [], [], []
    off = 0
    for tp in a.tapes:
        d = torch.load(tp, weights_only=False)
        meta = json.loads((tp.parent / "summary.json").read_text())
        ev = {r["idx"]: {int(k): int(v) for k, v in r["events"].items()}
              for r in meta["episode_records"]}
        Z.append(d["z"]); U.append(d["u"]); ZN.append(d["z_next"])
        EP.append(d["episode"] + off); off += int(d["episode"].max()) + 1
        DW.append(torch.tensor([float(sum(1 for s in ev.get(int(e), {}).values()
                                          if int(t) < s <= int(t) + 10))
                                for e, t in zip(d["episode"], d["t"])]))
    z, u, zn, ep = (torch.cat(Z), torch.cat(U), torch.cat(ZN), torch.cat(EP))
    dw = (torch.cat(DW) > 0).float()
    b = torch.load(a.branches, weights_only=False)
    bo = torch.cat([b["zh"], b["zp"]], -1)
    bu, by, bg = b["u"], b["y"], b["group"]
    print(f"dynamics {len(z)} triples ({int(ep.max())+1} episodes), "
          f"branch {len(by)} candidates ({int(bg.max())+1} groups), variant={a.variant}")

    mu_o, sd_o = z.mean(0), z.std(0) + 1e-6
    O, ON, BO = (z - mu_o) / sd_o, (zn - mu_o) / sd_o, (bo - mu_o) / sd_o
    ng = int(bg.max()) + 1
    aucs, states = [], []
    for s in range(a.restarts):
        torch.manual_seed(s); np.random.seed(s)
        perm = torch.randperm(ng, generator=torch.Generator().manual_seed(7000 + s))
        te_g = set(perm[int(ng * 0.85):].tolist())
        btr = torch.tensor([i for i in range(len(by)) if int(bg[i]) not in te_g])
        bte = torch.tensor([i for i in range(len(by)) if int(bg[i]) in te_g])
        m = JointWM(O.shape[-1], u.shape[1], u.shape[2], a.zdim, variant=a.variant)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        best = (0.0, None)
        for e in range(a.epochs):
            m.train(); opt.zero_grad()
            zt = m.encode(O)
            with torch.no_grad():
                tgt = m.encode(ON)                       # stop-grad target
            zp1, _, extra = m.step(zt, u)
            if a.variant == "ensemble":
                cons = ((zp1 - tgt) ** 2).mean()
            else:
                mu_, ls = extra
                var = (2 * ls).exp()
                cons = (((zt + mu_ - tgt) ** 2) / (2 * var) + ls).mean()  # Gaussian NLL
            rew = nn.functional.binary_cross_entropy_with_logits(
                m.reward(zp1).squeeze(-1), dw)
            zb = m.encode(BO[btr])
            zb1, _, _ = m.step(zb, bu[btr])
            val = nn.functional.binary_cross_entropy_with_logits(
                m.value(zb1).squeeze(-1), by[btr])
            (cons + rew + val).backward(); opt.step()
            if e % 50 == 0 or e == a.epochs - 1:
                m.eval()
                with torch.no_grad():
                    zb = m.encode(BO[bte])
                    zb1, _, _ = m.step(zb, bu[bte])
                    A = within_group_auc(m.value(zb1).squeeze(-1), by[bte], bg[bte])
                if not np.isnan(A) and A > best[0]:
                    best = (A, {k: v.clone() for k, v in m.state_dict().items()})
        aucs.append(best[0]); states.append(best[1])
        print(f"  restart {s}: within-group AUC {best[0]:.3f}")

    # is the LEARNED transition action-conditioned? shuffled-action control
    m.load_state_dict(states[int(np.argmax(aucs))]); m.eval()
    with torch.no_grad():
        zt = m.encode(O); tgt = m.encode(ON)
        zp1, _, _ = m.step(zt, u)
        g = torch.Generator().manual_seed(3)
        zsh, _, _ = m.step(zt, u[torch.randperm(len(u), generator=g)])
        ident = float(((zt - tgt) ** 2).mean())
        true_ = float(((zp1 - tgt) ** 2).mean())
        shuf = float(((zsh - tgt) ** 2).mean())
    print(f"\nlearned latent: identity {ident:.4f}  action {true_:.4f} "
          f"({true_/ident:.3f}x)  shuffled {shuf:.4f} ({shuf/ident:.3f}x)")
    print(f"{a.variant} mean within-group AUC {np.mean(aucs):.3f} "
          f"(pre_enc baseline 0.636, frozen-latent wm 0.637)")

    torch.save({"states": states, "aucs": aucs, "variant": a.variant,
                "zdim": a.zdim, "obs_dim": O.shape[-1], "c": u.shape[1],
                "adim": u.shape[2], "mu_o": mu_o, "sd_o": sd_o,
                "task": b["task"]}, out / "joint_wm.pt")
    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "variant": a.variant, "zdim": a.zdim,
         "dynamics_triples": len(z), "branch_candidates": len(by),
         "aucs": aucs, "mean_auc": float(np.mean(aucs)),
         "identity_mse": ident, "action_mse": true_, "shuffled_mse": shuf,
         "env_steps": 0,
         "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()},
        indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
