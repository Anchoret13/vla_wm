#!/usr/bin/env python
"""T_th rolls the VLA's OWN latent forward. No imposed bottleneck.

    python scripts/train_v220_raw_latent_wm.py --tapes ... --potential predicted

THE DESIGN ERROR THIS CORRECTS. Every world model in this repo predicted into a
256-d SimNorm code produced by an encoder I introduced. That choice was copied
from TD-MPC, which needs an encoder because it learns representations FROM PIXELS.
Here the VLA already supplies a trained representation - the 2073-d prefix-hidden
mean plus proprio - and compressing it again is pure loss. Measured, on held-out
episodes, predicting discounted time-to-success:

    raw 2073-d              Spearman +0.484   AUC 0.691
    encoder 256-d           Spearman +0.407   AUC 0.664      84% retained

and the loss is not only signal. T_th could only predict in the encoder's space, so
every head read off a PREDICTED latent was confined there, while any arm that did
not need the model could read raw. The best-performing advantage in this project
(a potential difference on raw features) was therefore structurally unavailable to
the world model. That is a property of my architecture, not of world models.

    b_t = GRU(b_{t-1}, [enc(z_t), E_a(u_{t-1})])      compact belief, history only
    T_th(b_t, E_a(u_t)) -> z^_{t+1}  in the VLA's OWN 2073-d latent
    Phi(z^_{t+1}) - Phi(z_t)                           the advantage that works,
                                                       now readable off a PREDICTION

Still a latent world model on every axis: the target is the VLA's latent, there is
no decoder and no pixel anywhere, the action conditions the roll, and the data is
deployment tapes. Compressing to 256-d was never what made it 'latent'.

The instability that forced SimNorm does not apply here. That failure (prediction
loss 0.041 -> 112 -> 64) came from targets produced by a JOINTLY TRAINED encoder
inflating along with it. These targets are fixed normalised features, so there is
nothing to inflate; the bounded code is unnecessary and, as measured above, costly.

PRE-REGISTERED CONTROLS, before any residual is trained: identity, dataset mean,
and shuffled-action, all against the same target - the same set that validated the
encoder-space model. The comparison that decides the arc is predicted vs observed
with ONE Phi in ONE space, which the encoder version could not run.
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
from train_v152_tdmpc_wm import SimNorm  # noqa: E402
from train_v157_residual_actor import ResidualActor  # noqa: E402
from train_v205_action_belief import load_episodes, auc  # noqa: E402

OUT = REPO / "results" / "v220_raw_latent_wm"


class RawLatentWM(nn.Module):
    """Belief stays compact; the PREDICTION lives in the VLA's own latent space."""

    def __init__(self, zdim, c, adim, bdim=256, edim=256, hidden=1024):
        super().__init__()
        self.enc = nn.Sequential(nn.LayerNorm(zdim), nn.Linear(zdim, 512),
                                 nn.GELU(), nn.Linear(512, edim), SimNorm(8))
        self.aenc = nn.Sequential(nn.Linear(c * adim, 128), nn.GELU(),
                                  nn.Linear(128, 64))
        self.gru = nn.GRUCell(edim + 64, bdim)
        self.bnorm = SimNorm(8)
        # residual form: predict the CHANGE from the current latent, which keeps the
        # identity control honest and makes the task well-scaled
        self.trans = nn.Sequential(nn.Linear(bdim + 64 + zdim, hidden), nn.GELU(),
                                   nn.Linear(hidden, hidden), nn.GELU(),
                                   nn.Linear(hidden, zdim))
        self.inv = nn.Sequential(nn.Linear(2 * bdim, hidden), nn.GELU(),
                                 nn.Linear(hidden, c * adim))
        self.bdim, self.edim, self.zdim = bdim, edim, zdim

    def act(self, u):
        return self.aenc(u.flatten(-2))

    def step(self, b, e, u_prev):
        return self.bnorm(self.gru(torch.cat([e, self.act(u_prev)], -1), b))

    def roll(self, zs, us):
        B, T = zs.shape[0], zs.shape[1]
        e = self.enc(zs)
        up = torch.cat([torch.zeros_like(us[:, :1]), us[:, :-1]], 1)
        b = torch.zeros(B, self.bdim, device=zs.device)
        out = []
        for t in range(T):
            b = self.step(b, e[:, t], up[:, t])
            out.append(b)
        return torch.stack(out, 1), e

    def predict(self, b, z, u):
        """z^_{t+1} = z_t + f(b_t, E_a(u_t), z_t) - in the VLA's own latent."""
        return z + self.trans(torch.cat([b, self.act(u), z], -1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tapes", type=Path, nargs="+", required=True)
    ap.add_argument("--task", default="chain1b_lr2")
    ap.add_argument("--wm-epochs", type=int, default=4000)
    ap.add_argument("--phi-epochs", type=int, default=3000)
    ap.add_argument("--actor-epochs", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--ensemble", type=int, default=3)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--alts", type=int, default=16)
    ap.add_argument("--scale", type=float, default=0.03)
    ap.add_argument("--potential", choices=["predicted", "observed", "current"],
                    required=True,
                    help="'current' is the LAST rung: A = Phi(z_t), no next latent "
                         "at all, so the weight depends only on the state the "
                         "chunk was executed in and carries no action "
                         "discrimination whatsoever. If it matches the other two, "
                         "nothing about a transition - predicted OR observed - is "
                         "doing the work.")
    ap.add_argument("--restarts", type=int, default=2)
    a = ap.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / f"{a.potential}_{stamp}"; out.mkdir(parents=True, exist_ok=True)

    eps = load_episodes(a.tapes, a.task, keep_step=True)
    allz = torch.cat([e["z"] for e in eps])
    mu, sd = allz.mean(0), allz.std(0) + 1e-6
    L = min(min(len(e["u"]) for e in eps), 24)
    Z = torch.stack([(e["z"][:L] - mu) / sd for e in eps])
    U = torch.stack([e["u"][:L] for e in eps])
    S = torch.tensor([float(e["succ"]) for e in eps])
    PHI = torch.zeros(len(eps), L)
    for i, e in enumerate(eps):
        if e["succ"] and e.get("succ_chunk") is not None:
            s_ = min(int(e["succ_chunk"]), L - 1)
            PHI[i, :s_ + 1] = a.gamma ** torch.arange(s_, -1, -1).float()
    zdim = Z.shape[-1]
    print(f"{len(eps)} episodes on {a.task}, L={L}, {int(S.sum())} successful")
    print(f"T_th predicts in the VLA's own {zdim}-d latent; advantage {a.potential}")

    models, B_ = [], []
    for k in range(a.ensemble):
        torch.manual_seed(k)
        m = RawLatentWM(zdim, U.shape[2], U.shape[3])
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        g = torch.Generator().manual_seed(2200 + k)
        H = 3
        for it in range(a.wm_epochs):
            i = torch.randint(0, len(Z), (a.batch,), generator=g)
            z, u = Z[i], U[i]
            b, e = m.roll(z, u)
            l1 = ((m.predict(b[:, :-1], z[:, :-1], u[:, :-1]) - z[:, 1:]) ** 2).mean()
            zh, bh, lh_ = z[:, :-H], b[:, :-H], 0.0
            for h in range(H):
                zn_ = m.predict(bh, zh, u[:, h:L - H + h])
                lh_ = lh_ + ((zn_ - z[:, h + 1:L - H + h + 1]) ** 2).mean()
                bh = m.step(bh.flatten(0, 1), m.enc(zn_).flatten(0, 1),
                            u[:, h:L - H + h].flatten(0, 1)).view(*bh.shape)
                zh = zn_
            linv = ((m.inv(torch.cat([b[:, :-1], b[:, 1:]], -1))
                     - u[:, :-1].flatten(2)) ** 2).mean()
            (l1 + lh_ / H + 0.1 * linv).backward()
            opt.step(); opt.zero_grad()
            if k == 0 and it % 1000 == 0:
                print(f"  it {it}: pred1 {float(l1):.4f} predH {float(lh_/H):.4f} "
                      f"inv {float(linv):.4f}", flush=True)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            b, _ = m.roll(Z, U)
        models.append(m); B_.append(b)
        print(f"  model {k} fitted", flush=True)
    B_ = torch.stack(B_)
    bdim = models[0].bdim
    bmu = B_[0].reshape(-1, bdim).mean(0); bsd = B_[0].reshape(-1, bdim).std(0) + 1e-6

    # ---- pre-registered controls, in the VLA's own latent space --------------
    m0, b0 = models[0], B_[0]
    perm = torch.randperm(len(Z), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        tgt = Z[:, 1:]
        diag = {
            "pred1_learned": float(((m0.predict(b0[:, :-1], Z[:, :-1], U[:, :-1]) - tgt) ** 2).mean()),
            "pred1_identity": float(((Z[:, :-1] - tgt) ** 2).mean()),
            "pred1_mean": float(((Z.mean((0, 1)) - tgt) ** 2).mean()),
            "pred1_shuffled_action": float(
                ((m0.predict(b0[:, :-1], Z[:, :-1], U[perm][:, :-1]) - tgt) ** 2).mean()),
        }
    print("\n1-step MSE in the VLA's own latent (identity = 1.000x)")
    base_ = diag["pred1_identity"]
    for k_ in ("pred1_learned", "pred1_identity", "pred1_mean", "pred1_shuffled_action"):
        print(f"  {k_:24s} {diag[k_]:.5f}  {diag[k_]/base_:6.3f} x identity")

    # ---- one Phi, in the SAME space as the prediction ------------------------
    torch.manual_seed(220)
    phi = nn.Sequential(nn.LayerNorm(zdim), nn.Linear(zdim, 256), nn.GELU(),
                        nn.Linear(256, 1))
    popt = torch.optim.AdamW(phi.parameters(), lr=1e-3, weight_decay=1e-2)
    Zf = Z.reshape(-1, zdim); PHIf = PHI.reshape(-1)
    gsplit = torch.Generator().manual_seed(4)
    pm = torch.randperm(len(eps), generator=gsplit)
    tr_ep, te_ep = pm[:int(0.8 * len(eps))], pm[int(0.8 * len(eps)):]
    trm = torch.zeros(len(eps), L, dtype=torch.bool); trm[tr_ep] = True
    trm = trm.reshape(-1); tem = ~trm
    g2 = torch.Generator().manual_seed(2201)
    tri = torch.nonzero(trm).flatten()
    for it in range(a.phi_epochs):
        i = tri[torch.randint(0, len(tri), (512,), generator=g2)]
        ((phi(Zf[i]).squeeze(-1) - PHIf[i]) ** 2).mean().backward()
        popt.step(); popt.zero_grad()
    phi.eval()
    for p in phi.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        pt = phi(Zf).squeeze(-1)
        ho = float(np.corrcoef(pt[tem].numpy(), PHIf[tem].numpy())[0, 1])
        ins = float(np.corrcoef(pt[trm].numpy(), PHIf[trm].numpy())[0, 1])
        hauc = auc(pt[tem].numpy(), (PHIf[tem] > 0).numpy())
    print(f"\nPhi: in-sample corr {ins:+.3f}, HELD-OUT corr {ho:+.3f}, "
          f"held-out AUC {hauc:.3f}")
    print("  (the held-out number is the honest one; a 2073-d head on 9600 samples "
          "memorises in-sample)")

    N = len(Zf)
    nxt = (torch.arange(N) + 1).clamp(max=N - 1)
    valid = ((torch.arange(N) + 1) % L != 0)
    pool = torch.nonzero(valid).flatten()
    Uf = U.reshape(-1, U.shape[2], U.shape[3])

    def advantage(idx, gen):
        with torch.no_grad():
            if a.potential == "current":
                return phi(Zf[idx]).squeeze(-1)
            if a.potential == "observed":
                return phi(Zf[nxt[idx]]).squeeze(-1) - phi(Zf[idx]).squeeze(-1)
            alt = torch.randint(0, N, (a.alts,), generator=gen)
            vs, va = [], []
            for k in range(a.ensemble):
                bk = B_[k].reshape(-1, bdim)[idx]
                vs.append(phi(models[k].predict(bk, Zf[idx], Uf[idx])).squeeze(-1))
                va.append(torch.stack(
                    [phi(models[k].predict(bk, Zf[idx], Uf[j].unsqueeze(0)
                                           .expand(len(idx), -1, -1))).squeeze(-1)
                     for j in alt]).mean(0))
            return torch.stack(vs).mean(0) - torch.stack(va).mean(0)

    rows = []
    for s in range(a.restarts):
        torch.manual_seed(s); np.random.seed(s)
        actor = ResidualActor(zdim, U.shape[2], U.shape[3], scale=a.scale)
        aopt = torch.optim.AdamW(actor.parameters(), lr=3e-4, weight_decay=1e-4)
        gg = torch.Generator().manual_seed(1000 + s)
        for it in range(a.actor_epochs):
            i = pool[torch.randint(0, len(pool), (a.batch * 8,), generator=gg)]
            adv = advantage(i, gg)
            w = torch.softmax(adv / (adv.std() + 1e-6), 0) * len(i)
            d = actor(Zf[i])
            tgt_ = Uf[i] - Uf[i].mean(0, keepdim=True)
            (w.unsqueeze(-1).unsqueeze(-1) * (d - tgt_) ** 2).mean().backward()
            aopt.step(); aopt.zero_grad()
        actor.eval()
        with torch.no_grad():
            adv = advantage(pool, gg)
            dn = float(actor(Zf[pool]).abs().mean())
            c_out = float(np.corrcoef(
                adv.numpy(), S.unsqueeze(1).expand(-1, L).reshape(-1)[pool].numpy())[0, 1])
        rows.append({"restart": s, "mean_abs_delta": dn, "adv_corr_outcome": c_out})
        print(f"  restart {s}: mean|Delta| {dn:.4f} ({100*dn/a.scale:.0f}% of bound); "
              f"advantage corr with outcome {c_out:+.3f}", flush=True)
        torch.save({"state_dict": actor.state_dict(), "zdim": zdim,
                    "condition": "raw", "c": U.shape[2], "adim": U.shape[3],
                    "scale": a.scale, "rawwm": models[0].state_dict(),
                    "dims": (zdim, U.shape[2], U.shape[3]),
                    "mu": mu, "sd": sd, "bmu": bmu, "bsd": bsd,
                    "task": a.task, "arm": f"rawwm_{a.potential}",
                    "potential": a.potential}, out / f"actor_{s}.pt")
    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "task": a.task, "potential": a.potential, "gamma": a.gamma,
         "ensemble": a.ensemble, "diagnostics": diag, "phi_heldout_corr": ho,
         "phi_insample_corr": ins, "phi_heldout_auc": hauc, "rows": rows,
         "env_steps": 0,
         "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()},
        indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
