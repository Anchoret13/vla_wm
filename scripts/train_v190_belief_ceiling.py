#!/usr/bin/env python
"""Does a RECURRENT belief recover progress that a single frame cannot?

    python scripts/train_v190_belief_ceiling.py --tapes ...

Prompted by RB-VLA (arXiv 2602.20659), which uses a world model in a role absent
from this framework's §5 menu: not scoring or imagining actions from outside a
frozen policy, but as a **belief state that conditions action generation**, carrying
history the VLA's short context loses. Its reported failure mode - "task progress
loss and action repetition" - matches what was measured here: 182 of 191 chain3
failures stop at the identical stage, which looks less like incapacity than like a
policy that has lost track of what it already did.

THE MEASUREMENT THIS CHALLENGES. The supervised ceiling on chain3 was 0.212 from a
SINGLE-TIMESTEP latent, and that number is what condemned every reward candidate.
But "how many objects are already in the basket" is a fact about history. A belief
that integrates the trajectory may carry it where one frame does not.

    belief:   b_t = GRU(b_{t-1}, [z_t, E_a(u_t)])     recurrent, action-conditioned
    trained:  predict z_{t+1} and z_{t+5} from b_t     self-supervised, NO LABELS
              + inverse dynamics: recover u_t from (b_{t-1}, b_t)

Both objectives are RB-VLA's, and both are label-free - which is why this route
sidesteps the constraint that killed the others: on chain3 only 0.5% of transitions
come from a successful episode, and every previous candidate needed that signal.

Then the same ceiling probe runs on b_t instead of z_t. If the ceiling clears the
bar, "the latent lacks progress information" was true only of single frames.
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
from probe_v167_value_stratification import spearman  # noqa: E402
from train_v152_tdmpc_wm import SimNorm  # noqa: E402

OUT = REPO / "results" / "v190_belief"


class Belief(nn.Module):
    def __init__(self, zdim, c, adim, bdim=256, hidden=512):
        super().__init__()
        # BOUNDED, per this framework's own 3 constraint 3. Without it the encoder
        # scale grew without limit and the prediction loss went 0.041 -> 112 -> 64:
        # the detached targets inflate with the encoder that produces them. The same
        # class of failure collapsed a Gaussian transition earlier to identity error
        # 0.0004.
        self.enc = nn.Sequential(nn.LayerNorm(zdim), nn.Linear(zdim, hidden),
                                 nn.GELU(), nn.Linear(hidden, 256), SimNorm(8))
        self.aenc = nn.Sequential(nn.Linear(c * adim, 128), nn.GELU(),
                                  nn.Linear(128, 64))
        self.gru = nn.GRUCell(256 + 64, bdim)
        self.bnorm = SimNorm(8)
        self.pred1 = nn.Sequential(nn.Linear(bdim, hidden), nn.GELU(),
                                   nn.Linear(hidden, 256))   # z_{t+1}
        self.pred5 = nn.Sequential(nn.Linear(bdim, hidden), nn.GELU(),
                                   nn.Linear(hidden, 256))   # z_{t+5}
        self.inv = nn.Sequential(nn.Linear(2 * bdim, hidden), nn.GELU(),
                                 nn.Linear(hidden, c * adim))  # inverse dynamics
        self.bdim = bdim

    def roll(self, zs, us):
        """(T,zdim),(T,c,adim) -> (T,bdim); or batched (B,T,...) -> (B,T,bdim).

        Batched because 300 single-episode iterations left the belief underfit -
        it scored 0.181 against a single frame's 0.232, which is not a fair test of
        the idea. RB-VLA trained on 40,000 trajectories."""
        single = zs.dim() == 2
        if single:
            zs, us = zs.unsqueeze(0), us.unsqueeze(0)
        B, T = zs.shape[0], zs.shape[1]
        e = self.enc(zs)
        a = self.aenc(us.flatten(2))
        b = torch.zeros(B, self.bdim, device=zs.device)
        out = []
        for t in range(T):
            b = self.bnorm(self.gru(torch.cat([e[:, t], a[:, t]], -1), b))
            out.append(b)
        out = torch.stack(out, 1)
        return (out[0], e[0]) if single else (out, e)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tapes", type=Path, nargs="+", required=True)
    ap.add_argument("--task", default="chain3_lr2")
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--restarts", type=int, default=4)
    a = ap.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / stamp; out.mkdir(parents=True, exist_ok=True)

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
            eps.append({"z": z[idx], "u": u[idx], "succ": bool(r["success"]),
                        "stage": len(r.get("events", {}))})
    print(f"{len(eps)} episodes on {a.task}, "
          f"{sum(e['succ'] for e in eps)} successful")
    allz = torch.cat([e["z"] for e in eps])
    mu, sd = allz.mean(0), allz.std(0) + 1e-6
    for e in eps:
        e["zn"] = (e["z"] - mu) / sd

    rows = []
    for s in range(a.restarts):
        torch.manual_seed(s); np.random.seed(s)
        m = Belief(allz.shape[-1], eps[0]["u"].shape[1], eps[0]["u"].shape[2])
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        g = torch.Generator().manual_seed(s)
        order = torch.randperm(len(eps), generator=g)
        tr_e = order[:int(len(eps) * 0.7)].tolist()
        te_e = order[int(len(eps) * 0.7):].tolist()
        Tmin = min(len(eps[k]["zn"]) for k in tr_e)
        for it in range(a.epochs):
            js = [tr_e[int(x)] for x in torch.randint(0, len(tr_e), (a.batch,),
                                                      generator=g)]
            zb = torch.stack([eps[k]["zn"][:Tmin] for k in js])
            ub = torch.stack([eps[k]["u"][:Tmin] for k in js])
            b, enc = m.roll(zb, ub)
            l1 = ((m.pred1(b[:, :-1]) - enc[:, 1:].detach()) ** 2).mean()
            l5 = (((m.pred5(b[:, :-5]) - enc[:, 5:].detach()) ** 2).mean()
                  if Tmin > 5 else torch.zeros(()))
            linv = ((m.inv(torch.cat([b[:, :-1], b[:, 1:]], -1))
                     - ub[:, :-1].flatten(2)) ** 2).mean()
            opt.zero_grad(); (l1 + l5 + linv).backward(); opt.step()
            if it % 500 == 0:
                print(f"    it {it}: pred1 {float(l1):.4f} pred5 {float(l5):.4f} "
                      f"inv {float(linv):.4f}", flush=True)
        m.eval()
        # ceiling probe on the TERMINAL BELIEF vs the terminal single-frame latent
        with torch.no_grad():
            B = torch.stack([m.roll(eps[k]["zn"], eps[k]["u"])[0][-1] for k in range(len(eps))])
            Zt = torch.stack([eps[k]["zn"][-1] for k in range(len(eps))])
        st = np.array([e["stage"] for e in eps], float)
        su = np.array([e["succ"] for e in eps], bool)

        def probe(X, tr_idx, te_idx):
            f_tr = [i for i in tr_idx if not su[i]]
            f_te = [i for i in te_idx if not su[i]]
            if len(f_te) < 15 or len(np.unique(st[f_te])) < 2:
                return float("nan")
            xm, xs = X[f_tr].mean(0), X[f_tr].std(0) + 1e-6
            Xs = (X - xm) / xs
            y = torch.tensor(st, dtype=torch.float32)
            torch.manual_seed(0)
            h = nn.Sequential(nn.LayerNorm(Xs.shape[-1]), nn.Linear(Xs.shape[-1], 128),
                              nn.GELU(), nn.Linear(128, 1))
            o = torch.optim.AdamW(h.parameters(), lr=1e-3, weight_decay=1e-2)
            best = -1.0
            ti = torch.tensor(f_tr)
            for e2 in range(800):
                h.train(); o.zero_grad()
                ((h(Xs[ti]).squeeze(-1) - y[ti]) ** 2).mean().backward(); o.step()
                if e2 % 50 == 0 or e2 == 799:
                    h.eval()
                    with torch.no_grad():
                        r_ = spearman(h(Xs[torch.tensor(f_te)]).squeeze(-1).numpy(),
                                      st[f_te])
                    if not np.isnan(r_):
                        best = max(best, r_)
            return best

        rb = probe(B, tr_e, te_e)
        rz = probe(Zt, tr_e, te_e)
        rows.append({"restart": s, "belief_rho": rb, "frame_rho": rz})
        print(f"  restart {s}: belief {rb:+.3f}   single-frame {rz:+.3f}")

    bb = float(np.nanmean([r["belief_rho"] for r in rows]))
    zz = float(np.nanmean([r["frame_rho"] for r in rows]))
    print(f"\nbelief ceiling {bb:+.3f}   single-frame ceiling {zz:+.3f}")
    print(f"chain3 bar is 0.6 x ceiling; the single-frame ceiling measured 0.212 "
          f"earlier, giving 0.127")
    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "task": a.task, "episodes": len(eps), "rows": rows,
         "belief_ceiling": bb, "frame_ceiling": zz, "env_steps": 0,
         "note": "belief trained self-supervised: 1-step and 5-step embedding "
                 "prediction plus inverse dynamics. No outcome labels.",
         "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()},
        indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
