#!/usr/bin/env python
"""Policy improvement through an ENSEMBLE of belief models, with pessimism.

    python scripts/train_v207_pessimistic_ensemble.py --tapes ... --lam 1.0

WHY THIS EXISTS. v205's residual was trained by backpropagating through a single
frozen model and pinned itself to 93% of the trust-region bound (mean|Delta| 0.0280
of 0.03) while its imagined value rose +0.1516. That signature - the optimiser
sitting on the constraint, maximising a head that is accurate on-distribution
(AUC 0.869 on predicted latents) - is model exploitation, and the literature's
answer to it is not a smaller bound but UNCERTAINTY:

  ME-TRPO (Kurutach et al., ICLR 2018): an ensemble of models regularises policy
    optimisation; the policy can no longer exploit one model's error.
  MOPO (Yu et al., 2020): penalise the value by the ensemble's disagreement, which
    makes the objective pessimistic exactly where the model is least trustworthy.

  V(u) = mean_k D_k(T_k(b_k, E_a(u)))  -  lam * std_k(D_k(T_k(b_k, E_a(u))))

K models are trained independently on the same deployment tapes. The actor sees
model 0's belief (one belief must be carried at deployment), but the objective is
evaluated under every model's own belief, so a direction that only model 0 likes
is penalised rather than rewarded.

GOAL ANCHOR: unchanged from v205 on all five axes - latent T_th rolled forward
under a candidate action, no reconstruction, deployment tapes only, model inside
the improvement loop. lam=0 recovers v205's objective exactly and is the control
that says whether pessimism, not the ensemble, is what changed the outcome.
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
from train_v157_residual_actor import ResidualActor  # noqa: E402
from train_v205_action_belief import ActionBelief, load_episodes, auc  # noqa: E402

OUT = REPO / "results" / "v207_pessimistic_ensemble"


def train_model(seed, Z, U, S, L, iters, batch):
    torch.manual_seed(seed)
    m = ActionBelief(Z.shape[-1], U.shape[2], U.shape[3])
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(2070 + seed)
    bce = nn.BCEWithLogitsLoss()
    H = 5
    for it in range(iters):
        i = torch.randint(0, len(Z), (batch,), generator=g)
        z, u, s = Z[i], U[i], S[i]
        b, e = m.roll(z, u)
        l1 = ((m.predict(b[:, :-1], u[:, :-1]) - e[:, 1:].detach()) ** 2).mean()
        bh, l5 = b[:, :-H], 0.0
        for h in range(H):
            eh = m.predict(bh, u[:, h:L - H + h])
            l5 = l5 + ((eh - e[:, h + 1:L - H + h + 1].detach()) ** 2).mean()
            bh = m.step(bh.flatten(0, 1), eh.flatten(0, 1),
                        u[:, h:L - H + h].flatten(0, 1)).view(*bh.shape)
        linv = ((m.inv(torch.cat([b[:, :-1], b[:, 1:]], -1))
                 - u[:, :-1].flatten(2)) ** 2).mean()
        lh = bce(m.head(e.detach()).squeeze(-1), s.unsqueeze(1).expand(-1, L))
        (l1 + l5 / H + 0.1 * linv + lh).backward()
        opt.step(); opt.zero_grad()
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tapes", type=Path, nargs="+", required=True)
    ap.add_argument("--task", default="chain1b_lr2")
    ap.add_argument("--wm-epochs", type=int, default=4000)
    ap.add_argument("--actor-epochs", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--ensemble", type=int, default=5)
    ap.add_argument("--lam", type=float, default=1.0,
                    help="pessimism weight; 0 recovers v205's objective exactly")
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--scale", type=float, default=0.03)
    ap.add_argument("--restarts", type=int, default=2)
    a = ap.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / f"lam{a.lam:g}_k{a.ensemble}_{stamp}"; out.mkdir(parents=True, exist_ok=True)

    eps = load_episodes(a.tapes, a.task)
    allz = torch.cat([e["z"] for e in eps])
    mu, sd = allz.mean(0), allz.std(0) + 1e-6
    L = min(min(len(e["u"]) for e in eps), 24)
    Z = torch.stack([(e["z"][:L] - mu) / sd for e in eps])
    U = torch.stack([e["u"][:L] for e in eps])
    S = torch.tensor([float(e["succ"]) for e in eps])
    print(f"{len(eps)} episodes on {a.task}, L={L}, {int(S.sum())} successful")
    print(f"ensemble K={a.ensemble}, pessimism lam={a.lam}")

    models, B_, E_ = [], [], []
    for k in range(a.ensemble):
        m = train_model(k, Z, U, S, L, a.wm_epochs, a.batch)
        with torch.no_grad():
            b, e = m.roll(Z, U)
            p1 = float(((m.predict(b[:, :-1], U[:, :-1]) - e[:, 1:]) ** 2).mean())
            idn = float(((e[:, :-1] - e[:, 1:]) ** 2).mean())
            y = S.unsqueeze(1).expand(-1, L - 1).reshape(-1).numpy()
            au = auc(m.head(m.predict(b[:, :-1], U[:, :-1])).squeeze(-1)
                     .reshape(-1).numpy(), y)
        models.append(m); B_.append(b); E_.append(e)
        print(f"  model {k}: pred1 {p1:.5f} ({p1/idn:.3f} x identity), "
              f"head AUC on predicted latents {au:.3f}", flush=True)
    B_ = torch.stack(B_); E_ = torch.stack(E_)          # (K,N,L,dim)

    # how much do the models disagree? the quantity pessimism is priced on
    with torch.no_grad():
        vk = torch.stack([models[k].head(E_[k]).squeeze(-1) for k in range(a.ensemble)])
        print(f"\nensemble disagreement on REAL latents: value std {float(vk.std(0).mean()):.4f}")

    bdim, edim = models[0].bdim, models[0].edim
    bmu = B_[0].reshape(-1, bdim).mean(0); bsd = B_[0].reshape(-1, bdim).std(0) + 1e-6
    rows = []
    for s in range(a.restarts):
        torch.manual_seed(s); np.random.seed(s)
        actor = ResidualActor(edim + bdim, U.shape[2], U.shape[3], scale=a.scale)
        aopt = torch.optim.AdamW(actor.parameters(), lr=3e-4, weight_decay=1e-4)
        gg = torch.Generator().manual_seed(940 + s)
        for it in range(a.actor_epochs):
            i = torch.randint(0, len(Z), (a.batch * 8,), generator=gg)
            t = torch.randint(0, L - a.horizon, (a.batch * 8,), generator=gg)
            bk = [B_[k][i, t] for k in range(a.ensemble)]
            ek = [E_[k][i, t] for k in range(a.ensemble)]
            val = 0.0
            for h in range(a.horizon):
                u = U[i, t + h]
                # the actor sees model 0's belief - the one carried at deployment
                d = actor(torch.cat([ek[0], (bk[0] - bmu) / bsd], -1))
                uh = u + d
                vs = []
                for k in range(a.ensemble):
                    eh = models[k].predict(bk[k], uh)
                    vs.append(models[k].head(eh).squeeze(-1))
                    bk[k] = models[k].step(bk[k], eh, uh); ek[k] = eh
                v = torch.stack(vs)
                val = val + (0.95 ** h) * (v.mean(0) - a.lam * v.std(0))
            (-val.mean()).backward(); aopt.step(); aopt.zero_grad()
        actor.eval()
        with torch.no_grad():
            b0 = B_[0][:, :-a.horizon].reshape(-1, bdim)
            e0 = E_[0][:, :-a.horizon].reshape(-1, edim)
            u0 = U[:, :-a.horizon].reshape(-1, U.shape[2], U.shape[3])
            d0 = actor(torch.cat([e0, (b0 - bmu) / bsd], -1))
            def agg(uu):
                vs = torch.stack([models[k].head(models[k].predict(
                    B_[k][:, :-a.horizon].reshape(-1, bdim), uu)).squeeze(-1)
                    for k in range(a.ensemble)])
                return float(vs.mean()), float(vs.std(0).mean())
            vb, db = agg(u0); vr, dr = agg(u0 + d0)
            dn = float(d0.abs().mean())
        rows.append({"restart": s, "V_base": vb, "V_residual": vr,
                     "imagined_gain": vr - vb, "disagree_base": db,
                     "disagree_residual": dr, "mean_abs_delta": dn,
                     "saturation": dn / a.scale})
        print(f"  restart {s}: V {vb:+.4f} -> {vr:+.4f} (gain {vr-vb:+.4f}); "
              f"disagreement {db:.4f} -> {dr:.4f}; "
              f"mean|Delta| {dn:.4f} ({100*dn/a.scale:.0f}% of bound)", flush=True)
        torch.save({"state_dict": actor.state_dict(), "zdim": edim + bdim,
                    "c": U.shape[2], "adim": U.shape[3], "scale": a.scale,
                    "model": models[0].state_dict(), "use_action": True,
                    "dims": (Z.shape[-1], U.shape[2], U.shape[3]),
                    "mu": mu, "sd": sd, "bmu": bmu, "bsd": bsd,
                    "task": a.task, "arm": f"pess_lam{a.lam:g}",
                    "lam": a.lam, "ensemble": a.ensemble}, out / f"actor_{s}.pt")
    print("\nImagined gain is NOT evidence; deployment is.")
    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "task": a.task, "lam": a.lam, "ensemble": a.ensemble,
         "horizon": a.horizon, "scale": a.scale, "episodes": len(eps),
         "rows": rows, "env_steps": 0,
         "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()},
        indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
