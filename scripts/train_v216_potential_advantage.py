#!/usr/bin/env python
"""Same potential, same advantage form. The ONLY difference is whether the next
latent is PREDICTED by T_th or OBSERVED.

    python scripts/train_v216_potential_advantage.py --tapes ... --potential predicted

WHY THIS IS THE RIGHT NEXT MEASUREMENT. Closing the 2x2 at 192 seeds/cell produced
one finding larger than anything about world models: the advantage DEFINITION
dominates. Inside one AWR frame,

    0.656  potential difference of a learned reward head on REAL next states (v185)
    0.573  T_th's counterfactual over executed chunks, binary success head
    0.542  binary episode outcome

and the 0.656 arm contains no T_th at all. Framework 5.2 has two halves,

    z~_{t+c} = T_th(z_t, E_a(u))          the roll
    D_th(z~_{t+c}) = (dw, dy, r, V_k, p)  what is READ off the predicted latent

and every attempt here has varied the first half while leaving the second at a
binary success head. The measurement says the second half is where the signal is.

THE DESIGN. One potential Phi, trained once, shared by both arms:

    Phi(e_t) ~ gamma^(s - t)              discounted time-to-success, 0 on failures

    --potential observed    A = Phi(e_{t+1}) - Phi(e_t)
                            the v185 construction. Needs no model: e_{t+1} was
                            actually observed. Cannot score an action nobody took.
    --potential predicted   A = Phi(T_th(b_t, E_a(u_i))) - mean_j Phi(T_th(b_t, E_a(u_j)))
                            the same potential read off the PREDICTED latent, which
                            is 5.2 exactly, and which CAN score alternatives.

Both arms share Phi, the conditioning, the trust region, and the AWR objective.
The single difference is predicted-vs-observed, so the comparison isolates T_th.

GOAL ANCHOR (five axes): task chain1b_lr2 at 0.469; object T_th rolled forward
under a candidate action; target future latent, no reconstruction; data deployment
tapes including the actor's own; placement inside the improvement loop.

PRE-REGISTERED, before deployment. `observed` beating `predicted` says T_th's roll
adds nothing over simply looking at what happened next. `predicted` beating
`observed` is the first evidence in this project that rolling the latent forward
buys something the real transition does not - and it would have a mechanism, since
only the predicted arm can compare against actions nobody executed. Equality says
the roll is an expensive way to reproduce an observation. 192 seeds per arm.
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
from train_v207_pessimistic_ensemble import train_model  # noqa: E402

OUT = REPO / "results" / "v216_potential_advantage"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tapes", type=Path, nargs="+", required=True)
    ap.add_argument("--task", default="chain1b_lr2")
    ap.add_argument("--wm-epochs", type=int, default=4000)
    ap.add_argument("--phi-epochs", type=int, default=3000)
    ap.add_argument("--actor-epochs", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--ensemble", type=int, default=5)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--alts", type=int, default=16)
    ap.add_argument("--scale", type=float, default=0.03)
    ap.add_argument("--condition", choices=["belief", "raw"], default="belief")
    ap.add_argument("--potential", choices=["predicted", "observed"], required=True)
    ap.add_argument("--phi-space", choices=["encoder", "raw"], default="encoder",
                    help="which latent Phi reads. v185's 0.656 arm used a head on "
                         "RAW features; every arm here has used the encoder's 256-d "
                         "space, which retains only 84%% of raw's rank correlation "
                         "with progress AND is the only space T_th can predict in. "
                         "'raw' is testable for the observed arm alone - that "
                         "asymmetry is the point.")
    ap.add_argument("--restarts", type=int, default=2)
    a = ap.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / f"{a.potential}_{a.condition}_{a.phi_space}_{stamp}"; out.mkdir(parents=True, exist_ok=True)

    eps = load_episodes(a.tapes, a.task, keep_step=True)
    allz = torch.cat([e["z"] for e in eps])
    mu, sd = allz.mean(0), allz.std(0) + 1e-6
    L = min(min(len(e["u"]) for e in eps), 24)
    Z = torch.stack([(e["z"][:L] - mu) / sd for e in eps])
    U = torch.stack([e["u"][:L] for e in eps])
    S = torch.tensor([float(e["succ"]) for e in eps])
    # discounted time-to-success: a DEPLOYABLE label (the step of first success is
    # observed during deployment exactly as the binary outcome is)
    PHI = torch.zeros(len(eps), L)
    for i, e in enumerate(eps):
        if e["succ"] and e.get("succ_chunk") is not None:
            s = min(int(e["succ_chunk"]), L - 1)
            PHI[i, :s + 1] = a.gamma ** torch.arange(s, -1, -1).float()
    print(f"{len(eps)} episodes on {a.task}, L={L}, {int(S.sum())} successful")
    print(f"potential target: discounted time-to-success, gamma={a.gamma}, "
          f"mean {float(PHI.mean()):.4f}")
    print(f"advantage: {a.potential}   conditioning: {a.condition}")

    models, B_, E_ = [], [], []
    for k in range(a.ensemble):
        m = train_model(k, Z, U, S, L, a.wm_epochs, a.batch)
        with torch.no_grad():
            b, e = m.roll(Z, U)
        models.append(m); B_.append(b); E_.append(e)
        print(f"  model {k} fitted", flush=True)
    B_ = torch.stack(B_); E_ = torch.stack(E_)
    bdim, edim = models[0].bdim, models[0].edim
    bmu = B_[0].reshape(-1, bdim).mean(0); bsd = B_[0].reshape(-1, bdim).std(0) + 1e-6

    # ---- one potential, shared by both arms --------------------------------
    if a.phi_space == "raw" and a.potential == "predicted":
        raise SystemExit("T_th predicts in the encoder's space, so a raw-space Phi "
                         "cannot be read off a predicted latent. That exclusion is "
                         "the finding, not a limitation to work around here.")
    torch.manual_seed(216)
    pdim = Z.shape[-1] if a.phi_space == "raw" else edim
    phi = nn.Sequential(nn.LayerNorm(pdim), nn.Linear(pdim, 256), nn.GELU(),
                        nn.Linear(256, 1))
    popt = torch.optim.AdamW(phi.parameters(), lr=1e-3, weight_decay=1e-4)
    Ef0 = (Z.reshape(-1, Z.shape[-1]) if a.phi_space == "raw"
           else E_[0].reshape(-1, edim)).detach()
    PHIf = PHI.reshape(-1)
    g = torch.Generator().manual_seed(2160)
    for it in range(a.phi_epochs):
        i = torch.randint(0, len(Ef0), (512,), generator=g)
        loss = ((phi(Ef0[i]).squeeze(-1) - PHIf[i]) ** 2).mean()
        loss.backward(); popt.step(); popt.zero_grad()
    phi.eval()
    for p in phi.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        pr = phi(Ef0).squeeze(-1)
        r = float(np.corrcoef(pr.numpy(), PHIf.numpy())[0, 1])
        au = auc(pr.numpy(), (PHIf > 0).numpy())
    print(f"Phi fitted: corr with target {r:+.3f}, AUC vs 'on a successful "
          f"trajectory' {au:.3f}")

    Uf = U.reshape(-1, U.shape[2], U.shape[3])
    Zf = Z.reshape(-1, Z.shape[-1])
    N = len(Uf)
    # observed next latent, within-episode; the last chunk of each episode has none
    nxt = torch.arange(N) + 1
    valid = ((torch.arange(N) + 1) % L != 0)
    cdim = Z.shape[-1] if a.condition == "raw" else edim + bdim

    def cond(idx):
        if a.condition == "raw":
            return Zf[idx]
        return torch.cat([E_[0].reshape(-1, edim)[idx],
                          (B_[0].reshape(-1, bdim)[idx] - bmu) / bsd], -1)

    def advantage(idx, gen):
        with torch.no_grad():
            if a.potential == "observed":
                # exactly v185's form: what actually happened next, no model
                return (phi(Ef0[nxt[idx].clamp(max=N - 1)]).squeeze(-1)
                        - phi(Ef0[idx]).squeeze(-1))
            alt = torch.randint(0, N, (a.alts,), generator=gen)
            vs, va = [], []
            for k in range(a.ensemble):
                bk = B_[k].reshape(-1, bdim)[idx]
                vs.append(phi(models[k].predict(bk, Uf[idx])).squeeze(-1))
                va.append(torch.stack(
                    [phi(models[k].predict(bk, Uf[j].unsqueeze(0)
                                           .expand(len(idx), -1, -1))).squeeze(-1)
                     for j in alt]).mean(0))
            return torch.stack(vs).mean(0) - torch.stack(va).mean(0)

    pool = torch.nonzero(valid).flatten()
    rows = []
    for s in range(a.restarts):
        torch.manual_seed(s); np.random.seed(s)
        actor = ResidualActor(cdim, U.shape[2], U.shape[3], scale=a.scale)
        aopt = torch.optim.AdamW(actor.parameters(), lr=3e-4, weight_decay=1e-4)
        gg = torch.Generator().manual_seed(980 + s)
        for it in range(a.actor_epochs):
            i = pool[torch.randint(0, len(pool), (a.batch * 8,), generator=gg)]
            adv = advantage(i, gg)
            w = torch.softmax(adv / (adv.std() + 1e-6), 0) * len(i)
            d = actor(cond(i))
            tgt = Uf[i] - Uf[i].mean(0, keepdim=True)
            loss = (w.unsqueeze(-1).unsqueeze(-1) * (d - tgt) ** 2).mean()
            loss.backward(); aopt.step(); aopt.zero_grad()
        actor.eval()
        with torch.no_grad():
            adv = advantage(pool, gg)
            dn = float(actor(cond(pool)).abs().mean())
            c_out = float(np.corrcoef(adv.numpy(),
                                      S.unsqueeze(1).expand(-1, L)
                                      .reshape(-1)[pool].numpy())[0, 1])
            c_phi = float(np.corrcoef(adv.numpy(), PHIf[pool].numpy())[0, 1])
        rows.append({"restart": s, "mean_abs_delta": dn,
                     "adv_corr_outcome": c_out, "adv_corr_potential": c_phi})
        print(f"  restart {s}: mean|Delta| {dn:.4f} ({100*dn/a.scale:.0f}% of bound); "
              f"advantage corr with outcome {c_out:+.3f}, with Phi {c_phi:+.3f}",
              flush=True)
        torch.save({"state_dict": actor.state_dict(), "zdim": cdim,
                    "condition": a.condition,
                    "c": U.shape[2], "adim": U.shape[3], "scale": a.scale,
                    "model": models[0].state_dict(), "use_action": True,
                    "dims": (Z.shape[-1], U.shape[2], U.shape[3]),
                    "mu": mu, "sd": sd, "bmu": bmu, "bsd": bsd,
                    "task": a.task, "arm": f"pot_{a.potential}_{a.condition}",
                    "potential": a.potential}, out / f"actor_{s}.pt")
    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "task": a.task, "potential": a.potential,
         "condition": a.condition, "phi_space": a.phi_space,
         "gamma": a.gamma, "ensemble": a.ensemble,
         "phi_corr": r, "phi_auc": au, "rows": rows, "env_steps": 0,
         "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()},
        indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
