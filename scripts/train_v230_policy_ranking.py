#!/usr/bin/env python
"""Can T_th rank POLICIES? The only counterfactual with measurable consequences.

    python scripts/train_v230_policy_ranking.py --tapes ... --k 8

WHAT v229 ESTABLISHED. From an identical state, eight genuinely different action
chunks produced identical progress in 71 of 72 cases (within-state sd 0.0067
against a between-state 0.3729). pi0.5 re-plans after the chunk and corrects the
perturbation away, so a SINGLE-CHUNK counterfactual is causally inert - which is
precisely the object 5.2 asks T_th to predict, and precisely why every use of it
failed while the model itself was accurate (action-shuffling costs it 2.35x).

But a SUSTAINED change does move the outcome: a residual applied at every chunk
took frozen pi0.5 from 0.469 to 0.531-0.562 (p = 0.0067, 288 seeds). So the unit of
counterfactual that matters here is a POLICY, not an action.

THE TEST, and it is a correlation rather than one more significance verdict.
Train K diverse residual policies. Score each by rolling T_th forward H chunks from
many real start states with that policy applied AT EVERY STEP, reading Phi off the
imagined latents. Then deploy all K and measure the true ranking.

    rho( imagined value of policy k , deployed success rate of policy k )

That number is what "can the world model evaluate a policy" means, and it is
informative whichever way it comes out - unlike a single arm comparison, it cannot
be explained away by a state-value filter, because every policy is scored at the
SAME distribution of start states.

PRE-REGISTERED: rho is reported over all K whatever its value, with the K deployed
rates. No arm is selected for reporting after the fact.
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
from train_v205_action_belief import load_episodes, auc  # noqa: E402
from train_v220_raw_latent_wm import RawLatentWM  # noqa: E402

OUT = REPO / "results" / "v230_policy_ranking"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tapes", type=Path, nargs="+", required=True)
    ap.add_argument("--task", default="chain1b_lr2")
    ap.add_argument("--wm-epochs", type=int, default=4000)
    ap.add_argument("--phi-epochs", type=int, default=3000)
    ap.add_argument("--actor-epochs", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--k", type=int, default=8, help="candidate residual policies")
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="shifts every actor init, so a FRESH pool can be built "
                         "that the world model has never been scored against")
    ap.add_argument("--horizon", type=int, default=10, help="chunks of SUSTAINED roll")
    ap.add_argument("--gamma", type=float, default=0.9)
    a = ap.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / f"{a.task}_{stamp}"; out.mkdir(parents=True, exist_ok=True)

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
    print(f"{len(eps)} episodes, L={L}, {int(S.sum())} successful")

    # ---- one world model, one potential -------------------------------------
    torch.manual_seed(0)
    m = RawLatentWM(zdim, U.shape[2], U.shape[3])
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(2300)
    H = 3
    for it in range(a.wm_epochs):
        i = torch.randint(0, len(Z), (a.batch,), generator=g)
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
        (l1 + lh / H + 0.1 * linv).backward()
        opt.step(); opt.zero_grad()
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        B_, _ = m.roll(Z, U)
        p1 = float(((m.predict(B_[:, :-1], Z[:, :-1], U[:, :-1]) - Z[:, 1:]) ** 2).mean())
        idn = float(((Z[:, :-1] - Z[:, 1:]) ** 2).mean())
    print(f"WM: 1-step {p1:.5f} = {p1/idn:.3f} x identity")

    torch.manual_seed(230)
    phi = nn.Sequential(nn.LayerNorm(zdim), nn.Linear(zdim, 256), nn.GELU(),
                        nn.Linear(256, 1))
    popt = torch.optim.AdamW(phi.parameters(), lr=1e-3, weight_decay=1e-2)
    Zf = Z.reshape(-1, zdim); PHIf = PHI.reshape(-1)
    pm = torch.randperm(len(eps), generator=torch.Generator().manual_seed(4))
    trm = torch.zeros(len(eps), L, dtype=torch.bool); trm[pm[:int(.8*len(eps))]] = True
    trm = trm.reshape(-1); tri = torch.nonzero(trm).flatten()
    g2 = torch.Generator().manual_seed(2301)
    for it in range(a.phi_epochs):
        i = tri[torch.randint(0, len(tri), (512,), generator=g2)]
        ((phi(Zf[i]).squeeze(-1) - PHIf[i]) ** 2).mean().backward()
        popt.step(); popt.zero_grad()
    phi.eval()
    for p in phi.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        pt = phi(Zf).squeeze(-1)
        ho = float(np.corrcoef(pt[~trm].numpy(), PHIf[~trm].numpy())[0, 1])
    print(f"Phi: held-out corr {ho:+.3f}")

    # ---- K deliberately DIFFERENT residual policies -------------------------
    Uf = U.reshape(-1, U.shape[2], U.shape[3])
    N = len(Zf)
    Sf = S.unsqueeze(1).expand(-1, L).reshape(-1)
    actors = []
    for k in range(a.k):
        # diversity by construction: scale, weighting target and seed all vary, so
        # the K policies really do differ rather than being restarts of one
        # widened after v235: rho over six held-out policies is underpowered
        # (exact permutation p = 0.136 at n = 6, and n = 6 needs rho >= 0.725).
        # More policies is the only thing that resolves it, so the grid is larger
        # and spans past k3's 0.05, which was the best rate seen.
        SC = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
        scale = SC[k % len(SC)]
        mode = ["value", "outcome"][k // len(SC) % 2]
        torch.manual_seed(100 + k + 1000 * (k // 16) + a.seed_offset)
        act = ResidualActor(zdim, U.shape[2], U.shape[3], scale=scale)
        o = torch.optim.AdamW(act.parameters(), lr=3e-4, weight_decay=1e-4)
        gg = torch.Generator().manual_seed(2400 + k + a.seed_offset)
        for it in range(a.actor_epochs):
            i = torch.randint(0, N, (128,), generator=gg)
            with torch.no_grad():
                adv = phi(Zf[i]).squeeze(-1) if mode == "value" else Sf[i]
            w = torch.softmax(adv / (adv.std() + 1e-6), 0) * len(i)
            d = act(Zf[i]); tgt = Uf[i] - Uf[i].mean(0, keepdim=True)
            (w.unsqueeze(-1).unsqueeze(-1) * (d - tgt) ** 2).mean().backward()
            o.step(); o.zero_grad()
        act.eval()
        actors.append({"k": k, "scale": scale, "mode": mode, "actor": act})

    # ---- score each policy by a SUSTAINED imagined roll ---------------------
    starts = torch.randperm(len(eps), generator=torch.Generator().manual_seed(9))
    z0 = Z[starts, 0]                       # every policy scored at the SAME states
    with torch.no_grad():
        b0 = torch.zeros(len(z0), m.bdim)
        b0 = m.step(b0, m.enc(z0), torch.zeros(len(z0), U.shape[2], U.shape[3]))
    rows = []
    for A in actors:
        with torch.no_grad():
            z, b, val = z0.clone(), b0.clone(), 0.0
            for h in range(a.horizon):
                # the residual is applied at EVERY step: a sustained policy change
                u = Uf[torch.randint(0, N, (len(z),),
                                     generator=torch.Generator().manual_seed(50 + h))]
                uh = u + A["actor"](z)
                z = m.predict(b, z, uh)
                b = m.step(b, m.enc(z), uh)
                val = val + (a.gamma ** h) * phi(z).squeeze(-1)
            v = float(val.mean())
            dn = float(A["actor"](Zf).abs().mean())
        rows.append({"k": A["k"], "scale": A["scale"], "mode": A["mode"],
                     "imagined_value": v, "mean_abs_delta": dn})
        print(f"  policy {A['k']} (scale {A['scale']}, {A['mode']:7s}): "
              f"imagined {v:+.4f}, mean|Delta| {dn:.4f}", flush=True)
        torch.save({"state_dict": A["actor"].state_dict(), "zdim": zdim,
                    "condition": "raw", "c": U.shape[2], "adim": U.shape[3],
                    "scale": A["scale"], "rawwm": m.state_dict(),
                    "dims": (zdim, U.shape[2], U.shape[3]),
                    "mu": mu, "sd": sd, "bmu": torch.zeros(m.bdim),
                    "bsd": torch.ones(m.bdim), "task": a.task,
                    "arm": f"polrank_k{A['k']}", "imagined_value": v},
                   out / f"actor_{A['k']}.pt")
    order = sorted(rows, key=lambda r: -r["imagined_value"])
    print("\nimagined ranking (best first): " +
          ", ".join(f"k{r['k']}" for r in order))
    print("deploy ALL of them; rho against the true rates is the measurement.")
    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "task": a.task, "k": a.k, "horizon": a.horizon,
         "wm_1step_x_identity": p1 / idn, "phi_heldout_corr": ho,
         "policies": rows, "env_steps": 0,
         "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()},
        indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
