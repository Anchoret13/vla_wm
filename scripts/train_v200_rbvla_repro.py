#!/usr/bin/env python
"""Reproduce RB-VLA's core ablation: belief-conditioned BC vs plain BC.

    python scripts/train_v200_rbvla_repro.py --tapes ... --condition plain|belief

WHY THIS COMES FIRST, and did not. RB-VLA reports 77.5% with belief conditioning
against 32.5% for a standard policy without it. Everything measured here about
beliefs — that history adds nothing, that belief conditioning is worse than raw —
tested a *transplanted loss* inside a different setting: 256-d GRU instead of a 150M
transformer, 192 episodes instead of 40,000 trajectories, pooled prefix hidden
instead of raw-frame embeddings, and a bounded residual on a frozen policy instead
of a 120M policy trained from scratch. **Four of four components differ**, so those
results carry no weight against the paper's claims. Establishing the reference point
should have preceded testing variants of it.

What is reproduced here is the ABLATION, not the system: a policy trained from
scratch on demonstrations, with and without belief conditioning, deployed and
compared. Scale is far below the paper's and is stated with the result.

Demonstrations are pi-0.5's own successful rollouts, which is legitimate — the
demonstrator is a competent policy — and is the only demonstration source available
without downloading LIBERO's datasets.
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
from train_v190_belief_ceiling import Belief  # noqa: E402

OUT = REPO / "results" / "v200_rbvla_repro"


class BCPolicy(nn.Module):
    """Trained from scratch. Maps the conditioning input to an action chunk."""

    def __init__(self, indim, c, adim, hidden=512):
        super().__init__()
        self.c, self.adim = c, adim
        self.net = nn.Sequential(nn.LayerNorm(indim), nn.Linear(indim, hidden),
                                 nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, c * adim))

    def forward(self, x):
        return self.net(x).view(-1, self.c, self.adim)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tapes", type=Path, nargs="+", required=True)
    ap.add_argument("--task", default="chain1b_lr2")
    ap.add_argument("--condition", choices=["plain", "belief"], required=True)
    ap.add_argument("--belief-epochs", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--restarts", type=int, default=3)
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
        sg = d.get("sigma")
        for e in sorted(set(ep.tolist())):
            r = rec.get(int(e))
            if r is None:
                continue
            idx = torch.nonzero(ep == e).flatten()[torch.argsort(tt[ep == e])]
            if len(idx) < 8:
                continue
            eps.append({"z": z[idx], "u": u[idx], "succ": bool(r["success"]),
                        "onpolicy": bool(sg is None or float(sg[idx].abs().max()) == 0)})
    allz = torch.cat([e["z"] for e in eps])
    mu, sd = allz.mean(0), allz.std(0) + 1e-6
    for e in eps:
        e["zn"] = (e["z"] - mu) / sd
    demos = [e for e in eps if e["succ"]]
    print(f"{len(eps)} episodes on {a.task}; **{len(demos)} successful used as "
          f"demonstrations** ({sum(len(e['u']) for e in demos)} chunks)")
    print(f"paper's scale: 40,000 trajectories. Here: {len(demos)}. "
          f"{100*len(demos)/40000:.2f}% of it.")
    assert len(demos) >= 20, "too few demonstrations"

    bel = bmu = bsd = None
    if a.condition == "belief":
        torch.manual_seed(0)
        bel = Belief(allz.shape[-1], eps[0]["u"].shape[1], eps[0]["u"].shape[2])
        o = torch.optim.AdamW(bel.parameters(), lr=1e-3, weight_decay=1e-4)
        g = torch.Generator().manual_seed(0)
        Tmin = min(len(e["zn"]) for e in eps)
        # belief trains on ALL rollouts, successes and failures - it is
        # self-supervised and needs no outcome labels
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
            o.zero_grad(); (l1 + l5 + li).backward(); o.step()
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
        if a.condition == "plain":
            return e["zn"]
        return torch.cat([e["zn"], (e["b"] - bmu) / bsd], -1)

    X = torch.cat([cond(e) for e in demos])
    Y = torch.cat([e["u"] for e in demos])
    print(f"BC training set: {len(X)} chunks, conditioning {X.shape[-1]} dims")

    rows = []
    for s in range(a.restarts):
        torch.manual_seed(s)
        pol = BCPolicy(X.shape[-1], Y.shape[1], Y.shape[2])
        o = torch.optim.AdamW(pol.parameters(), lr=1e-3, weight_decay=1e-4)
        g = torch.Generator().manual_seed(1300 + s)
        for e2 in range(a.epochs):
            i = torch.randint(0, len(X), (a.batch,), generator=g)
            pol.train(); o.zero_grad()
            ((pol(X[i]) - Y[i]) ** 2).mean().backward(); o.step()
        pol.eval()
        with torch.no_grad():
            mse = float(((pol(X) - Y) ** 2).mean())
        rows.append({"restart": s, "train_mse": mse})
        print(f"  restart {s}: BC train MSE {mse:.5f}")
        torch.save({"state_dict": pol.state_dict(), "indim": X.shape[-1],
                    "c": Y.shape[1], "adim": Y.shape[2], "condition": a.condition,
                    "mu": mu, "sd": sd, "task": a.task,
                    "belief": bel.state_dict() if bel else None,
                    "bmu": bmu, "bsd": bsd,
                    "belief_dims": (allz.shape[-1], eps[0]["u"].shape[1],
                                    eps[0]["u"].shape[2])},
                   out / f"policy_{s}.pt")
    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "task": a.task, "condition": a.condition,
         "episodes": len(eps), "demonstrations": len(demos),
         "chunks": int(len(X)), "cond_dims": int(X.shape[-1]), "rows": rows,
         "env_steps": 0,
         "note": "RB-VLA's core ablation at ~0.5% of its trajectory scale; "
                 "demonstrations are pi-0.5's own successful rollouts",
         "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()},
        indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
