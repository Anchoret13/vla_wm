#!/usr/bin/env python
"""At what horizon does imagination stop being faithful? Zero environment cost.

    python scripts/probe_v233_horizon_sweep.py --tapes ... --run <v230 dir> --rates ...

v232 scored the eight policies over ten imagined chunks and came out at rho =
-0.778 with an imagined spread of 0.0076. Two explanations remain and they differ:

  (i) compounding model error over ten steps swamps a real between-policy signal,
      in which case a SHORTER horizon should rank correctly;
  (ii) the imagination genuinely converges because pi0.5 self-corrects, in which
      case no horizon helps and the spread stays at noise.

Both are measurable here. First, roll the closed loop (pi_hat + T_th) against the
REAL tapes and report the prediction error at each horizon, next to the identity
and mean baselines at that horizon. Then re-rank the eight policies at each
horizon and report rho against their already-measured deployed rates.

PRE-REGISTERED: every horizon's rho is reported, not the best one.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
import numpy as np, torch
from torch import nn
from train_v157_residual_actor import ResidualActor
from train_v205_action_belief import load_episodes
from train_v220_raw_latent_wm import RawLatentWM


def rank(v):
    v = np.asarray(v, dtype=float)
    r = np.empty(len(v)); o = np.argsort(v, kind="mergesort"); vs = v[o]; i = 0
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
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--rates", type=float, nargs="+", required=True)
    ap.add_argument("--gamma", type=float, default=0.9)
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
    Zf, Uf = Z.reshape(-1, zdim), U.reshape(-1, c, adim)

    torch.manual_seed(232)
    prior = nn.Sequential(nn.LayerNorm(zdim), nn.Linear(zdim, 512), nn.GELU(),
                          nn.Linear(512, 512), nn.GELU(), nn.Linear(512, c * adim))
    o = torch.optim.AdamW(prior.parameters(), lr=1e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(1)
    pm = torch.randperm(len(Zf), generator=torch.Generator().manual_seed(2))
    tr = pm[:int(.8 * len(Zf))]
    for it in range(3000):
        i = tr[torch.randint(0, len(tr), (256,), generator=g)]
        ((prior(Zf[i]).view(-1, c, adim) - Uf[i]) ** 2).mean().backward()
        o.step(); o.zero_grad()
    prior.eval()
    for p in prior.parameters():
        p.requires_grad_(False)

    # ---- how faithful is the closed loop against the REAL tapes? ------------
    HMAX = 10
    with torch.no_grad():
        b, _ = m.roll(Z, U)
        z = Z[:, 0].clone(); bb = b[:, 0].clone()
        print(f"{'h':>3} {'closed-loop MSE':>16} {'identity':>10} {'mean':>10} "
              f"{'x identity':>11}")
        for h in range(1, HMAX + 1):
            u = prior(z).view(-1, c, adim)
            z = m.predict(bb, z, u)
            bb = m.step(bb, m.enc(z), u)
            if h < L:
                tgt = Z[:, h]
                e_ = float(((z - tgt) ** 2).mean())
                idn = float(((Z[:, 0] - tgt) ** 2).mean())
                mn = float(((Z.reshape(-1, zdim).mean(0) - tgt) ** 2).mean())
                print(f"{h:>3} {e_:>16.5f} {idn:>10.5f} {mn:>10.5f} "
                      f"{e_/idn:>11.3f}")

    # ---- rank the eight policies at every horizon ---------------------------
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

    z0 = Z[torch.randperm(len(eps), generator=torch.Generator().manual_seed(9)), 0]
    with torch.no_grad():
        b0 = m.step(torch.zeros(len(z0), m.bdim), m.enc(z0),
                    torch.zeros(len(z0), c, adim))
    actors = []
    for k in range(8):
        ck = torch.load(a.run / f"actor_{k}.pt", weights_only=False)
        act = ResidualActor(zdim, c, adim, scale=ck["scale"])
        act.load_state_dict(ck["state_dict"]); act.eval()
        actors.append(act)
    per_h = {h: [] for h in range(1, HMAX + 1)}
    for act in actors:
        with torch.no_grad():
            z, bb, val = z0.clone(), b0.clone(), 0.0
            for h in range(1, HMAX + 1):
                u = prior(z).view(-1, c, adim) + act(z)
                z = m.predict(bb, z, u)
                bb = m.step(bb, m.enc(z), u)
                val = val + (a.gamma ** (h - 1)) * phi(z).squeeze(-1)
                per_h[h].append(float(val.mean()))
    print(f"\n{'h':>3} {'imagined spread':>16} {'rho vs deployed':>16}")
    for h in range(1, HMAX + 1):
        v = np.array(per_h[h])
        rho = float(np.corrcoef(rank(v), rank(a.rates))[0, 1])
        print(f"{h:>3} {v.max()-v.min():>16.4f} {rho:>+16.3f}")
    print(f"\ndeployed spread 0.2500; every horizon reported, none selected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
