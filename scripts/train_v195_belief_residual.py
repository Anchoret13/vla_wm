#!/usr/bin/env python
"""U5 — belief-conditioned policy improvement. RB-VLA's belief + the half it lacks.

    python scripts/train_v195_belief_residual.py --tapes ... [--condition raw|belief|both]

RB-VLA (arXiv 2602.20659) trains a belief state self-supervised and conditions a
BEHAVIOUR-CLONED policy on it. It does no policy improvement from deployment
experience. This work has policy improvement that measurably works - an AWR residual
lifting a frozen pi-0.5 from 0.458 to 0.615, replicated across two panels - but it
conditions on raw 2073-d features. This joins them.

**The rationale was corrected by measurement, and the original one is dead.** The
belief was expected to supply history the frame lacks; it does not - history adds
+0.011 on chain3 and -0.004 on chain1b. What it does supply is a *compact,
signal-preserving* conditioning input:

| 256-d representation | trained by | ordering signal retained |
|---|---|---:|
| WM encoder | consistency + TD | 0.140 — 34% |
| **belief** | future-embedding prediction + inverse dynamics | **0.545 — ~100%** |

So the residual moves from 2073 raw dims to 256 lossless ones: fewer parameters on
the same signal.

CONTROL: `--condition raw` is the already-deployed arm (0.656 / 0.573 on two panels).
Same AWR objective, same data, same bound, same scale - **only the conditioning
input differs**, so the belief's contribution is isolated.
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
from train_v190_belief_ceiling import Belief  # noqa: E402

OUT = REPO / "results" / "v195_belief_residual"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tapes", type=Path, nargs="+", required=True)
    ap.add_argument("--task", default="chain1b_lr2")
    ap.add_argument("--condition", choices=["raw", "belief", "both"], default="belief")
    ap.add_argument("--belief-epochs", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--scale", type=float, default=0.03)
    ap.add_argument("--restarts", type=int, default=5)
    a = ap.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / f"{a.condition}_{stamp}"; out.mkdir(parents=True, exist_ok=True)

    eps = []
    for tp in a.tapes:
        d = torch.load(tp, weights_only=False)
        if d["latent_dim"] != 2073 or d["task"] != a.task:
            continue
        meta = json.loads((tp.parent / "summary.json").read_text())
        rec = {r["idx"]: r for r in meta["episode_records"]}
        z, u, ep, tt = d["z"], d["u"], d["episode"], d["t"]
        for e in sorted(set(ep.tolist())):
            r = rec.get(int(e))
            if r is None:
                continue
            idx = torch.nonzero(ep == e).flatten()[torch.argsort(tt[ep == e])]
            if len(idx) < 8:
                continue
            eps.append({"z": z[idx], "u": u[idx], "succ": float(r["success"])})
    allz = torch.cat([e["z"] for e in eps])
    mu, sd = allz.mean(0), allz.std(0) + 1e-6
    for e in eps:
        e["zn"] = (e["z"] - mu) / sd
    print(f"{len(eps)} episodes on {a.task}, "
          f"{int(sum(e['succ'] for e in eps))} successful, {len(allz)} transitions")

    # ---- belief: self-supervised, NO outcome labels -----------------------------
    bel, bmu, bsd = None, None, None
    if a.condition in ("belief", "both"):
        torch.manual_seed(0)
        bel = Belief(allz.shape[-1], eps[0]["u"].shape[1], eps[0]["u"].shape[2])
        opt = torch.optim.AdamW(bel.parameters(), lr=1e-3, weight_decay=1e-4)
        g = torch.Generator().manual_seed(0)
        Tmin = min(len(e["zn"]) for e in eps)
        for it in range(a.belief_epochs):
            js = [int(x) for x in torch.randint(0, len(eps), (16,), generator=g)]
            zb = torch.stack([eps[k]["zn"][:Tmin] for k in js])
            ub = torch.stack([eps[k]["u"][:Tmin] for k in js])
            b, enc = bel.roll(zb, ub)
            l1 = ((bel.pred1(b[:, :-1]) - enc[:, 1:].detach()) ** 2).mean()
            l5 = (((bel.pred5(b[:, :-5]) - enc[:, 5:].detach()) ** 2).mean()
                  if Tmin > 5 else torch.zeros(()))
            li = ((bel.inv(torch.cat([b[:, :-1], b[:, 1:]], -1))
                   - ub[:, :-1].flatten(2)) ** 2).mean()
            opt.zero_grad(); (l1 + l5 + li).backward(); opt.step()
            if it % 1000 == 0:
                print(f"  belief it {it}: pred1 {float(l1):.4f} inv {float(li):.4f}",
                      flush=True)
        bel.eval()
        with torch.no_grad():
            for e in eps:
                e["b"] = bel.roll(e["zn"], e["u"])[0]
        allb = torch.cat([e["b"] for e in eps])
        bmu, bsd = allb.mean(0), allb.std(0) + 1e-6

    def cond(e):
        if a.condition == "raw":
            return e["zn"]
        bn = (e["b"] - bmu) / bsd
        return bn if a.condition == "belief" else torch.cat([e["zn"], bn], -1)

    X = torch.cat([cond(e) for e in eps])
    U = torch.cat([e["u"] for e in eps])
    S = torch.cat([torch.full((len(e["u"]),), e["succ"]) for e in eps])
    print(f"conditioning '{a.condition}': {X.shape[-1]} dims")

    rows = []
    for s in range(a.restarts):
        torch.manual_seed(s)
        actor = ResidualActor(X.shape[-1], U.shape[1], U.shape[2], scale=a.scale)
        opt = torch.optim.AdamW(actor.parameters(), lr=3e-4, weight_decay=1e-4)
        g = torch.Generator().manual_seed(900 + s)
        for e2 in range(a.epochs):
            i = torch.randint(0, len(X), (a.batch,), generator=g)
            with torch.no_grad():
                adv = S[i]
                w = torch.softmax(adv / (adv.std() + 1e-6), 0) * len(i)
            d_ = actor(X[i])
            tgt = U[i] - U[i].mean(0, keepdim=True)
            loss = (w.unsqueeze(-1).unsqueeze(-1) * (d_ - tgt) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        actor.eval()
        with torch.no_grad():
            dn = float(actor(X).abs().mean())
        rows.append({"restart": s, "mean_abs_delta": dn})
        print(f"  restart {s}: mean|Delta| {dn:.4f}")
        torch.save({"state_dict": actor.state_dict(), "zdim": X.shape[-1],
                    "c": U.shape[1], "adim": U.shape[2], "scale": a.scale,
                    "condition": a.condition, "task": a.task,
                    "mu": mu, "sd": sd,
                    "belief": bel.state_dict() if bel else None,
                    "bmu": bmu, "bsd": bsd,
                    "belief_dims": (allz.shape[-1], eps[0]["u"].shape[1],
                                    eps[0]["u"].shape[2])},
                   out / f"actor_{s}.pt")
    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "task": a.task, "condition": a.condition,
         "episodes": len(eps), "cond_dims": int(X.shape[-1]), "rows": rows,
         "env_steps": 0,
         "note": "belief is self-supervised (no outcome labels); only the residual "
                 "uses episode success. Control is --condition raw, already deployed.",
         "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()},
        indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
