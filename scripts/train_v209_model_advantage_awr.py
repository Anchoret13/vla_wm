#!/usr/bin/env python
"""AWR whose advantage comes from T_th's counterfactual, not from the outcome.

    python scripts/train_v209_model_advantage_awr.py --tapes ... --lam 1.0

THE OBSERVATION THIS IS BUILT ON. Two facts were measured on chain1b_lr2, panel
7600-7695, all paired:

  model-free AWR              63/96 = 0.656   (+25 -7 vs base, p = 0.0021)
  every residual backpropagated through T_th   0.396 - 0.510, none above base

AWR works because it only ever imitates actions that were ACTUALLY EXECUTED - it
cannot leave the support, so there is nothing to exploit. Its weakness is that its
advantage is the OUTCOME, which is observed after the fact and says nothing about
what a different action would have done. T_th's one unique capability is exactly
that counterfactual, and every previous use of it let the actor invent a direction
no one had executed, which is what got exploited (94% saturation of the bound).

So take the half of each that works:

  actions   real, executed, in-support        (AWR's protection)
  advantage A(u_i) = mean_k D_k(T_k(b_k, E_a(u_i)))
                   - mean_j mean_k D_k(T_k(b_k, E_a(u_j)))    (T_th's counterfactual)

Every u_j is another executed chunk from the same batch, so the model is queried
ONLY at real actions - it is never asked about a direction the data does not cover.
The residual then does advantage-weighted regression exactly as the model-free arm
does. The two differ in one term: where the advantage comes from.

GOAL ANCHOR (five axes). task chain1b_lr2 at 0.469; object T_th(b_t, E_a(u))
rolled forward under a candidate action - the candidates are real chunks, which is
the change; target future latent, no reconstruction; data deployment tapes;
placement inside the improvement loop.

PRE-REGISTERED, before any deployment:
  --advantage outcome   reproduces the model-free arm's objective on this stack,
                        so the two arms differ ONLY in the advantage term.
  --shuffle-model       the counterfactual is computed by an action-blind model
                        (its transition is trained on actions from other episodes),
                        so its advantage carries no action information. If this arm
                        scores the same, the gain is not T_th's counterfactual.
  Beating 63/96 is the first evidence T_th contributes. Matching it says the
  counterfactual added nothing the outcome did not already carry. Falling below
  says the model's advantage is worse than the observed one. All get reported.
"""
from __future__ import annotations

import argparse, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import numpy as np, torch  # noqa: E402
from train_v157_residual_actor import ResidualActor  # noqa: E402
from train_v205_action_belief import load_episodes  # noqa: E402
from train_v207_pessimistic_ensemble import train_model  # noqa: E402

OUT = REPO / "results" / "v209_model_advantage"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tapes", type=Path, nargs="+", required=True)
    ap.add_argument("--task", default="chain1b_lr2")
    ap.add_argument("--wm-epochs", type=int, default=4000)
    ap.add_argument("--actor-epochs", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--ensemble", type=int, default=5)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--alts", type=int, default=16,
                    help="executed chunks used as the counterfactual comparison set")
    ap.add_argument("--scale", type=float, default=0.03)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--advantage", choices=["model", "outcome"], default="model")
    ap.add_argument("--condition", choices=["belief", "raw"], default="belief",
                    help="what the ACTOR sees. The advantage comes from T_th either "
                         "way; this only changes the conditioning input. U5 measured "
                         "belief conditioning significantly worse than raw "
                         "(p = 0.0309), and the best model-free arm (63/96) is "
                         "raw-conditioned, so 'raw' pairs T_th's counterfactual with "
                         "the conditioning that is known to work.")
    ap.add_argument("--shuffle-model", action="store_true",
                    help="REFUTING ABLATION: the ensemble's transition is trained "
                         "with actions from other episodes, so its counterfactual "
                         "is action-blind")
    a = ap.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / (f"{a.advantage}{'_shufmodel' if a.shuffle_model else ''}"
                 f"{'_raw' if a.condition == 'raw' else ''}_{stamp}"); out.mkdir(parents=True, exist_ok=True)

    eps = load_episodes(a.tapes, a.task)
    allz = torch.cat([e["z"] for e in eps])
    mu, sd = allz.mean(0), allz.std(0) + 1e-6
    L = min(min(len(e["u"]) for e in eps), 24)
    Z = torch.stack([(e["z"][:L] - mu) / sd for e in eps])
    U = torch.stack([e["u"][:L] for e in eps])
    S = torch.tensor([float(e["succ"]) for e in eps])
    print(f"{len(eps)} episodes on {a.task}, L={L}, {int(S.sum())} successful")
    print(f"advantage source: {a.advantage}")

    models, B_, E_ = [], [], []
    for k in range(a.ensemble):
        m = train_model(k, Z, U, S, L, a.wm_epochs, a.batch,
                        shuffle=a.shuffle_model)
        with torch.no_grad():
            b, e = m.roll(Z, U)
        models.append(m); B_.append(b); E_.append(e)
        print(f"  model {k} fitted", flush=True)
    B_ = torch.stack(B_); E_ = torch.stack(E_)
    bdim, edim = models[0].bdim, models[0].edim
    bmu = B_[0].reshape(-1, bdim).mean(0); bsd = B_[0].reshape(-1, bdim).std(0) + 1e-6

    Uf = U.reshape(-1, U.shape[2], U.shape[3])          # every executed chunk
    Sf = S.unsqueeze(1).expand(-1, L).reshape(-1)       # its episode outcome
    N = len(Uf)
    print(f"{N} executed chunks form the counterfactual comparison pool")

    def model_adv(idx, gen):
        """A(u_i) under the ensemble, compared against OTHER EXECUTED chunks at the
        same belief. Every query is at a real action."""
        with torch.no_grad():
            v_self, v_alt = [], []
            alt = torch.randint(0, N, (a.alts,), generator=gen)
            for k in range(a.ensemble):
                bk = B_[k].reshape(-1, bdim)[idx]
                v_self.append(models[k].head(models[k].predict(bk, Uf[idx])).squeeze(-1))
                va = [models[k].head(models[k].predict(bk, Uf[j].unsqueeze(0)
                                                       .expand(len(idx), -1, -1))
                                     ).squeeze(-1) for j in alt]
                v_alt.append(torch.stack(va).mean(0))
            vs, va = torch.stack(v_self), torch.stack(v_alt)
            return (vs.mean(0) - a.lam * vs.std(0)) - va.mean(0)

    Zf = Z.reshape(-1, Z.shape[-1])
    cdim = Z.shape[-1] if a.condition == "raw" else edim + bdim

    def cond(idx):
        if a.condition == "raw":
            return Zf[idx]
        return torch.cat([E_[0].reshape(-1, edim)[idx],
                          (B_[0].reshape(-1, bdim)[idx] - bmu) / bsd], -1)

    rows = []
    for s in range(a.restarts):
        torch.manual_seed(s); np.random.seed(s)
        actor = ResidualActor(cdim, U.shape[2], U.shape[3], scale=a.scale)
        opt = torch.optim.AdamW(actor.parameters(), lr=3e-4, weight_decay=1e-4)
        g = torch.Generator().manual_seed(960 + s)
        for it in range(a.actor_epochs):
            i = torch.randint(0, N, (a.batch * 8,), generator=g)
            adv = Sf[i] if a.advantage == "outcome" else model_adv(i, g)
            w = torch.softmax(adv / (adv.std() + 1e-6), 0) * len(i)
            d = actor(cond(i))
            tgt = Uf[i] - Uf[i].mean(0, keepdim=True)
            loss = (w.unsqueeze(-1).unsqueeze(-1) * (d - tgt) ** 2).mean()
            loss.backward(); opt.step(); opt.zero_grad()
        actor.eval()
        with torch.no_grad():
            i = torch.arange(N)
            adv = Sf if a.advantage == "outcome" else model_adv(i, g)
            dn = float(actor(cond(i)).abs().mean())
            corr = float(np.corrcoef(adv.numpy(), Sf.numpy())[0, 1])
        rows.append({"restart": s, "mean_abs_delta": dn,
                     "saturation": dn / a.scale,
                     "adv_corr_with_outcome": corr})
        print(f"  restart {s}: mean|Delta| {dn:.4f} ({100*dn/a.scale:.0f}% of bound); "
              f"advantage correlates {corr:+.3f} with the observed outcome",
              flush=True)
        torch.save({"state_dict": actor.state_dict(), "zdim": cdim,
                    "condition": a.condition,
                    "c": U.shape[2], "adim": U.shape[3], "scale": a.scale,
                    "model": models[0].state_dict(), "use_action": True,
                    "dims": (Z.shape[-1], U.shape[2], U.shape[3]),
                    "mu": mu, "sd": sd, "bmu": bmu, "bsd": bsd,
                    "task": a.task, "advantage": a.advantage,
                    "arm": f"advawr_{a.advantage}"
                           f"{'_shufmodel' if a.shuffle_model else ''}"},
                   out / f"actor_{s}.pt")
    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "task": a.task, "advantage": a.advantage, "lam": a.lam,
         "shuffle_model": a.shuffle_model, "condition": a.condition,
         "ensemble": a.ensemble, "alts": a.alts, "scale": a.scale,
         "chunks": N, "rows": rows, "env_steps": 0,
         "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()},
        indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
