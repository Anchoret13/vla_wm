#!/usr/bin/env python
"""Re-score the eight policies with a CLOSED-LOOP imagination. Zero env cost.

    python scripts/probe_v232_closedloop_ranking.py --tapes ... --run <v230 dir>

WHAT WENT WRONG IN v230. Policies were scored by rolling T_th forward with the
residual applied at every step - but the BASE action at each imagined step was a
random executed chunk drawn from the buffer, not what pi0.5 would emit at that
imagined latent. So the imagined trajectory was not the closed loop being scored,
and random base actions swamped the residual's contribution. That is enough on its
own to explain an imagined spread of 0.016 against a deployed spread of 0.250.

Dreamer imagines with its actor; TD-MPC rolls with a learned policy prior. This
component was simply missing. The tapes already contain what is needed: u_t IS
pi0.5's own chunk at z_t, so a latent policy prior is a behaviour clone.

    pi_hat(z) ~ u          BC on pi0.5's own deployment chunks
    imagine:  u_h = pi_hat(z_h) + Delta(z_h),   z_{h+1} = T_th(b_h, z_h, u_h)

The eight deployed rates are already measured, so this costs no environment steps:
re-score, then recompute rho against the SAME true ranking.

PRE-REGISTERED: rho over all eight, reported whatever it is, next to v230's -0.216.
A prior that fixes the roll should move rho up; if it does not, the failure is in
T_th's evaluation of sustained change and not in the missing prior.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
import numpy as np, torch
from torch import nn
from train_v157_residual_actor import ResidualActor
from train_v205_action_belief import load_episodes
from train_v220_raw_latent_wm import RawLatentWM


def rank(v):
    r = np.empty(len(v)); o = np.argsort(v, kind="mergesort"); vs = np.asarray(v)[o]; i = 0
    while i < len(vs):
        j = i
        while j + 1 < len(vs) and vs[j + 1] == vs[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tapes", type=Path, nargs="+", required=True)
    ap.add_argument("--run", type=Path, required=True, help="the v230 output dir")
    ap.add_argument("--rates", type=float, nargs="+", required=True,
                    help="the eight deployed rates, in k order")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--prior-epochs", type=int, default=3000)
    a = ap.parse_args()

    ck0 = torch.load(a.run / "actor_0.pt", weights_only=False)
    zdim, c, adim = ck0["dims"]
    m = RawLatentWM(zdim, c, adim); m.load_state_dict(ck0["rawwm"]); m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    mu, sd = ck0["mu"], ck0["sd"]

    eps = load_episodes(a.tapes, ck0["task"], keep_step=True)
    L = min(min(len(e["u"]) for e in eps), 24)
    Z = torch.stack([(e["z"][:L] - mu) / sd for e in eps])
    U = torch.stack([e["u"][:L] for e in eps])
    Zf = Z.reshape(-1, zdim); Uf = U.reshape(-1, c, adim)

    # ---- the missing component: pi0.5's own action, as a function of the latent
    torch.manual_seed(232)
    prior = nn.Sequential(nn.LayerNorm(zdim), nn.Linear(zdim, 512), nn.GELU(),
                          nn.Linear(512, 512), nn.GELU(), nn.Linear(512, c * adim))
    o = torch.optim.AdamW(prior.parameters(), lr=1e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(1)
    n = len(Zf); pm = torch.randperm(n, generator=torch.Generator().manual_seed(2))
    tr, te = pm[:int(.8 * n)], pm[int(.8 * n):]
    for it in range(a.prior_epochs):
        i = tr[torch.randint(0, len(tr), (256,), generator=g)]
        ((prior(Zf[i]).view(-1, c, adim) - Uf[i]) ** 2).mean().backward()
        o.step(); o.zero_grad()
    prior.eval()
    for p in prior.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        err = float(((prior(Zf[te]).view(-1, c, adim) - Uf[te]) ** 2).mean())
        var = float(Uf[te].var())
    print(f"pi_hat: held-out MSE {err:.5f} against action variance {var:.5f} "
          f"-> {100*(1-err/var):.1f}% of the action variance explained")

    phi_ck = torch.load(a.run / "actor_0.pt", weights_only=False)
    # Phi is not stored separately; refit it exactly as v230 did is unnecessary -
    # the ranking only needs a value, so reuse the WM's own head-free potential by
    # refitting on the same target here would diverge. Instead score with the
    # SAME potential the actors were trained against: refit deterministically.
    S = torch.tensor([float(e["succ"]) for e in eps])
    PHI = torch.zeros(len(eps), L)
    for i, e in enumerate(eps):
        if e["succ"] and e.get("succ_chunk") is not None:
            s_ = min(int(e["succ_chunk"]), L - 1)
            PHI[i, :s_ + 1] = a.gamma ** torch.arange(s_, -1, -1).float()
    torch.manual_seed(230)
    phi = nn.Sequential(nn.LayerNorm(zdim), nn.Linear(zdim, 256), nn.GELU(),
                        nn.Linear(256, 1))
    po = torch.optim.AdamW(phi.parameters(), lr=1e-3, weight_decay=1e-2)
    PHIf = PHI.reshape(-1)
    pm2 = torch.randperm(len(eps), generator=torch.Generator().manual_seed(4))
    trm = torch.zeros(len(eps), L, dtype=torch.bool); trm[pm2[:int(.8 * len(eps))]] = True
    tri = torch.nonzero(trm.reshape(-1)).flatten()
    g2 = torch.Generator().manual_seed(2301)
    for it in range(3000):
        i = tri[torch.randint(0, len(tri), (512,), generator=g2)]
        ((phi(Zf[i]).squeeze(-1) - PHIf[i]) ** 2).mean().backward()
        po.step(); po.zero_grad()
    phi.eval()
    for p in phi.parameters():
        p.requires_grad_(False)

    starts = torch.randperm(len(eps), generator=torch.Generator().manual_seed(9))
    z0 = Z[starts, 0]
    with torch.no_grad():
        b0 = m.step(torch.zeros(len(z0), m.bdim), m.enc(z0),
                    torch.zeros(len(z0), c, adim))
    vals = []
    for k in range(8):
        ck = torch.load(a.run / f"actor_{k}.pt", weights_only=False)
        act = ResidualActor(zdim, c, adim, scale=ck["scale"])
        act.load_state_dict(ck["state_dict"]); act.eval()
        with torch.no_grad():
            z, b, val = z0.clone(), b0.clone(), 0.0
            for h in range(a.horizon):
                # CLOSED LOOP: the base action is what pi0.5 would emit HERE
                u = prior(z).view(-1, c, adim) + act(z)
                z = m.predict(b, z, u)
                b = m.step(b, m.enc(z), u)
                val = val + (a.gamma ** h) * phi(z).squeeze(-1)
            vals.append(float(val.mean()))
        print(f"  policy {k} (scale {ck['scale']}): closed-loop imagined {vals[-1]:+.4f}"
              f"   deployed {a.rates[k]:.3f}")
    iv = np.array(vals); dr = np.array(a.rates)
    rho = float(np.corrcoef(rank(iv), rank(dr))[0, 1])
    print(f"\nimagined spread {iv.max()-iv.min():.4f} (v230 open loop: 0.0162)")
    print(f"deployed spread {dr.max()-dr.min():.4f}")
    print(f"** rho(closed-loop imagined, deployed) = {rho:+.3f}   "
          f"(v230 open loop: -0.216) **")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
