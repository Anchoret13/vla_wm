#!/usr/bin/env python
"""Policy improvement INSIDE the world model's imagination, VLA as the anchor.

    python scripts/train_v157_residual_actor.py --wm <wm.pt> --tapes ... --rounds ...

This is the step that was missing. Everything before this used the world model only
to RANK what the VLA proposed, which is why five separate isolations found the
transition contributing nothing: ranking needs a critic, and a critic needs no
dynamics. Here the transition is load-bearing by construction - a critic cannot
generate a rollout, so without T_theta there is no imagination and no residual to
train.

    executed action  =  u_VLA  +  Delta_phi(z)
    Delta trained    =  argmax_phi  E[ V(z_H) ]  over IMAGINED latent rollouts
    anchor           =  bounded ||Delta||, so the VLA prior is a trust region
                        rather than a starting point that gets discarded

The VLA is treated as a strong prior: Delta starts at zero and only has to learn a
small correction, which is the whole reason this can work on ~5k transitions when
learning a policy from scratch could not.

REFUTING ARM, registered before running: `--no-imagination` trains the same Delta
with the same anchor against the same V, but on REAL transitions only, never
rolling the model forward. If that matches, the transition still contributes
nothing and the loop's WM is decorative.

KNOWN FAILURE MODE, stated in advance: the residual will exploit model error -
scoring well in imagination and doing nothing in the environment. Mitigations here
are a short horizon, the norm bound, and pessimism over the value ensemble; none is
guaranteed to be enough at this data scale.
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
from train_v152_tdmpc_wm import SimNorm, mlp  # noqa: E402
from train_v153_td_value_wm import ValueWM  # noqa: E402

OUT = REPO / "results" / "v157_residual_actor"


class ResidualActor(nn.Module):
    """Delta on the VLA's chunk. Zero at init, bounded, so the prior is a trust region."""

    def __init__(self, zdim: int, c: int, adim: int, hidden: int = 256,
                 scale: float = 0.15):
        super().__init__()
        self.c, self.adim, self.scale = c, adim, scale
        self.net = mlp(zdim, hidden, c * adim)
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, z):
        return self.scale * torch.tanh(self.net(z)).view(-1, self.c, self.adim)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", type=Path, required=True)
    ap.add_argument("--tapes", type=Path, nargs="+", required=True)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--scale", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--restarts", type=int, default=5)
    ap.add_argument("--no-imagination", action="store_true",
                    help="REFUTING ARM: optimise Delta against V on REAL next states "
                         "only, never rolling T_theta forward.")
    a = ap.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    tag = "no_imag" if a.no_imagination else "imag"
    out = OUT / f"{tag}_{stamp}"; out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(a.wm, weights_only=False)
    Z, U, ZN = [], [], []
    dim0 = None
    for tp in a.tapes:
        d = torch.load(tp, weights_only=False)
        if dim0 is None:
            dim0 = int(d["z"].shape[-1])
        assert int(d["z"].shape[-1]) == dim0, f"{tp}: latent dim mismatch"
        Z.append(d["z"]); U.append(d["u"]); ZN.append(d["z_next"])
    z, u, zn = torch.cat(Z), torch.cat(U), torch.cat(ZN)
    mu_o, sd_o = ck["mu_o"], ck["sd_o"]
    O, ON = (z - mu_o) / sd_o, (zn - mu_o) / sd_o

    wm = ValueWM(O.shape[-1], u.shape[1], u.shape[2], ck["zdim"],
                 encoder=ck.get("encoder", "mlp"))
    wm.load_state_dict(ck["state_dict"]); wm.eval()
    for p in wm.parameters():
        p.requires_grad_(False)
    print(f"{len(z)} transitions; horizon {a.horizon}; scale {a.scale}; arm={tag}")

    rows = []
    for s in range(a.restarts):
        torch.manual_seed(s); np.random.seed(s)
        actor = ResidualActor(ck["zdim"], u.shape[1], u.shape[2], scale=a.scale)
        opt = torch.optim.AdamW(actor.parameters(), lr=3e-4, weight_decay=1e-4)
        g = torch.Generator().manual_seed(500 + s)
        v0 = None
        for e in range(a.epochs):
            actor.train(); opt.zero_grad()
            ti = torch.randint(0, len(z), (a.batch,), generator=g)
            with torch.no_grad():
                zt = wm.encode(O[ti])
                if v0 is None:
                    v0 = float(wm.V(wm.step(zt, u[ti])).mean())
            if a.no_imagination:
                # value of the residual action at the REAL next state: no rollout
                with torch.no_grad():
                    zr = wm.encode(ON[ti])
                val = wm.V(wm.step(zr, u[ti] + actor(zr))).squeeze(-1)
                loss = -val.mean()
            else:
                zc, val = zt, 0.0
                for h in range(a.horizon):           # IMAGINED rollout
                    ua = u[ti] + actor(zc)
                    zc = wm.step(zc, ua)
                    val = val + (0.95 ** h) * wm.V(zc).squeeze(-1)
                loss = -val.mean()
            loss.backward(); opt.step()
        actor.eval()
        with torch.no_grad():
            ti = torch.arange(len(z))
            zt = wm.encode(O)
            base = float(wm.V(wm.step(zt, u)).mean())
            withd = float(wm.V(wm.step(zt, u + actor(zt))).mean())
            dn = float(actor(zt).abs().mean())
        rows.append({"restart": s, "V_base": base, "V_residual": withd,
                     "gain_in_imagination": withd - base, "mean_abs_delta": dn})
        print(f"  restart {s}: V(base) {base:.4f} -> V(+Delta) {withd:.4f} "
              f"(imagined gain {withd-base:+.4f}); mean|Delta| {dn:.4f}")
        torch.save({"state_dict": actor.state_dict(), "zdim": ck["zdim"],
                    "c": u.shape[1], "adim": u.shape[2], "scale": a.scale,
                    "wm": str(a.wm), "arm": tag}, out / f"actor_{s}.pt")

    gains = np.array([r["gain_in_imagination"] for r in rows])
    print(f"\n{tag}: imagined gain {gains.mean():+.4f} +/- {gains.std():.4f}")
    print("NOTE: this is the gain INSIDE the model. It is not evidence of anything "
          "until the actor is deployed - a residual that exploits model error scores "
          "well here by construction.")
    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "arm": tag, "wm": str(a.wm), "horizon": a.horizon,
         "scale": a.scale, "transitions": len(z), "rows": rows,
         "mean_imagined_gain": float(gains.mean()), "env_steps": 0,
         "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()},
        indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
