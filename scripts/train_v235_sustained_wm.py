#!/usr/bin/env python
"""Does deployment data under DIVERSE sustained policies give T_th the ability it lacked?

    python scripts/train_v235_sustained_wm.py --base-tapes ... --policy-tapes ...

THE CHAIN THIS TESTS. v226/228/229 found no action->outcome signal in the tapes;
v230/232/233 found T_th cannot rank policies at any horizon while staying accurate
(0.087-0.243 x identity). Those are the same fact: a SINGLE-chunk perturbation is
corrected away by pi0.5, so the data contains none of the sustained bias that does
move the outcome (0.469 -> 0.743). sigma is per-chunk noise, and the v208 loop
tapes all came from near-identical scale-0.03 actors, so nothing in the buffer ever
showed that different sustained biases lead to different places.

Now four policies with deployed rates from 0.438 to 0.771 have each collected 96
episodes, so that variation exists for the first time.

HOLD-OUT DISCIPLINE, and it is the point. Training sees the tapes of exactly TWO
policies - the extremes k3 (0.771) and k7 (0.542) - plus pi0.5's own. Ranking is
then reported on SIX policies whose data the model has never seen, including k0 and
k6, which were collected and deliberately withheld. Feeding all eight in and
reporting the ranking would be handing the model its answer.

PRE-REGISTERED: rho over the six held-out policies, whatever it is, against the
-0.778 / -0.814 this same model family produced before. No horizon selection, no
subset selection.
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
import numpy as np, torch
from torch import nn
from train_v157_residual_actor import ResidualActor
from train_v205_action_belief import load_episodes
from train_v220_raw_latent_wm import RawLatentWM

OUT = REPO / "results" / "v235_sustained_wm"


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
    ap.add_argument("--base-tapes", type=Path, nargs="+", required=True)
    ap.add_argument("--policy-tapes", type=Path, nargs="+", required=True,
                    help="tapes of the TRAINING policies only (the extremes)")
    ap.add_argument("--run", type=Path, required=True, help="the v230 actor dir")
    ap.add_argument("--holdout", type=int, nargs="+", default=[0, 1, 2, 4, 5, 6])
    ap.add_argument("--rates", type=float, nargs="+", required=True,
                    help="deployed rates for the held-out policies, in --holdout order")
    ap.add_argument("--task", default="chain1b_lr2")
    ap.add_argument("--wm-epochs", type=int, default=6000)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--gamma", type=float, default=0.9)
    a = ap.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / f"{a.task}_{stamp}"; out.mkdir(parents=True, exist_ok=True)

    eps = load_episodes(list(a.base_tapes) + list(a.policy_tapes), a.task,
                        keep_step=True)
    n_base = len(load_episodes(list(a.base_tapes), a.task, keep_step=True))
    print(f"{len(eps)} episodes: {n_base} from pi0.5, "
          f"{len(eps)-n_base} under SUSTAINED training policies")
    allz = torch.cat([e["z"] for e in eps]); mu, sd = allz.mean(0), allz.std(0) + 1e-6
    L = min(min(len(e["u"]) for e in eps), 24)
    Z = torch.stack([(e["z"][:L] - mu) / sd for e in eps])
    U = torch.stack([e["u"][:L] for e in eps])
    S = torch.tensor([float(e["succ"]) for e in eps])
    zdim, c, adim = Z.shape[-1], U.shape[2], U.shape[3]
    Zf, Uf = Z.reshape(-1, zdim), U.reshape(-1, c, adim)

    torch.manual_seed(0)
    m = RawLatentWM(zdim, c, adim)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(2350)
    H = 3
    for it in range(a.wm_epochs):
        i = torch.randint(0, len(Z), (16,), generator=g)
        z, u = Z[i], U[i]
        b, e = m.roll(z, u)
        l1 = ((m.predict(b[:, :-1], z[:, :-1], u[:, :-1]) - z[:, 1:]) ** 2).mean()
        zh, bh, lh = z[:, :-H], b[:, :-H], 0.0
        for h in range(H):
            zn = m.predict(bh, zh, u[:, h:L - H + h])
            lh = lh + ((zn - z[:, h + 1:L - H + h + 1]) ** 2).mean()
            bh = m.step(bh.flatten(0, 1), m.enc(zn).flatten(0, 1),
                        u[:, h:L - H + h].flatten(0, 1)).view(*bh.shape)
            zh = zn
        linv = ((m.inv(torch.cat([b[:, :-1], b[:, 1:]], -1))
                 - u[:, :-1].flatten(2)) ** 2).mean()
        (l1 + lh / H + 0.1 * linv).backward(); opt.step(); opt.zero_grad()
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        B_, _ = m.roll(Z, U)
        p1 = float(((m.predict(B_[:, :-1], Z[:, :-1], U[:, :-1]) - Z[:, 1:]) ** 2).mean())
        idn = float(((Z[:, :-1] - Z[:, 1:]) ** 2).mean())
        perm = torch.randperm(len(Z), generator=torch.Generator().manual_seed(3))
        sh = float(((m.predict(B_[:, :-1], Z[:, :-1], U[perm][:, :-1]) - Z[:, 1:]) ** 2).mean())
    print(f"WM: 1-step {p1/idn:.3f} x identity; shuffled action {sh/p1:.2f}x worse")

    # policy prior and potential, same construction as v232/v233
    def fit(net, X, Y, iters, seed, wd=1e-4, view=None):
        o = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)
        gg = torch.Generator().manual_seed(seed)
        for it in range(iters):
            i = torch.randint(0, len(X), (256,), generator=gg)
            p = net(X[i])
            p = p.view(-1, *view) if view else p.squeeze(-1)
            ((p - Y[i]) ** 2).mean().backward(); o.step(); o.zero_grad()
        net.eval()
        for q in net.parameters():
            q.requires_grad_(False)
        return net

    torch.manual_seed(232)
    prior = fit(nn.Sequential(nn.LayerNorm(zdim), nn.Linear(zdim, 512), nn.GELU(),
                              nn.Linear(512, 512), nn.GELU(),
                              nn.Linear(512, c * adim)), Zf, Uf, 3000, 1,
                view=(c, adim))
    PHI = torch.zeros(len(eps), L)
    for i, e in enumerate(eps):
        if e["succ"] and e.get("succ_chunk") is not None:
            s_ = min(int(e["succ_chunk"]), L - 1)
            PHI[i, :s_ + 1] = a.gamma ** torch.arange(s_, -1, -1).float()
    torch.manual_seed(230)
    phi = fit(nn.Sequential(nn.LayerNorm(zdim), nn.Linear(zdim, 256), nn.GELU(),
                            nn.Linear(256, 1)), Zf, PHI.reshape(-1), 3000, 2, wd=1e-2)

    z0 = Z[torch.randperm(len(eps), generator=torch.Generator().manual_seed(9)), 0]
    with torch.no_grad():
        b0 = m.step(torch.zeros(len(z0), m.bdim), m.enc(z0),
                    torch.zeros(len(z0), c, adim))
    vals = []
    for k in a.holdout:
        ck = torch.load(a.run / f"actor_{k}.pt", weights_only=False)
        act = ResidualActor(zdim, c, adim, scale=ck["scale"])
        act.load_state_dict(ck["state_dict"]); act.eval()
        with torch.no_grad():
            z, bb, val = z0.clone(), b0.clone(), 0.0
            for h in range(a.horizon):
                u = prior(z).view(-1, c, adim) + act(z)
                z = m.predict(bb, z, u)
                bb = m.step(bb, m.enc(z), u)
                val = val + (a.gamma ** h) * phi(z).squeeze(-1)
            vals.append(float(val.mean()))
        print(f"  HELD-OUT policy {k} (scale {ck['scale']}): "
              f"imagined {vals[-1]:+.4f}   deployed {a.rates[len(vals)-1]:.3f}")
    iv = np.array(vals); dr = np.array(a.rates)
    rho = float(np.corrcoef(rank(iv), rank(dr))[0, 1])
    print(f"\nimagined spread {iv.max()-iv.min():.4f}, deployed spread {dr.max()-dr.min():.4f}")
    print(f"** rho over {len(a.holdout)} HELD-OUT policies = {rho:+.3f} **")
    print("   (same model family before sustained data: -0.778 to -0.814)")
    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "task": a.task, "episodes": len(eps), "base_episodes": n_base,
         "wm_x_identity": p1 / idn, "shuffled_ratio": sh / p1,
         "holdout": a.holdout, "imagined": vals, "deployed": a.rates,
         "rho": rho, "horizon": a.horizon, "env_steps": 0,
         "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()},
        indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
