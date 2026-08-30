#!/usr/bin/env python
"""Ceiling for §6.1: is progress recoverable from the terminal latent AT ALL?

    python scripts/probe_v179_stage_ceiling.py --tapes ...

Four reward candidates have now failed §6.1 (R1 quasimetric, R4 preference, and R2/R3
killed by reading). Before trying a fifth, measure the ceiling - the same discipline
U0 applied to selection, which capped that route for the price of one panel.

A probe is trained **directly on `stage_reached`**, supervised, episode-disjoint. If
even that cannot order held-out failures, then no reward objective can, and the
defect is UPSTREAM in the latent `z`, not in how the reward is constructed. Trying
R5 or R6 in that case would repeat four failures a fifth time.

This uses privileged labels as a TRAINING target and is therefore an upper bound
only, never a deployable method - exactly the status of the oracle in U0.

Reported per task, never pooled: pooling chain1b (L=250, stages 0-1) with chain3
(L=750, stages 0-5) gives rho = 0.5 with a clock null of +0.94, because the pooled
statistic reads task identity rather than progress.
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

OUT = REPO / "results" / "v179_stage_ceiling"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tapes", type=Path, nargs="+", required=True)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--restarts", type=int, default=5)
    a = ap.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / stamp; out.mkdir(parents=True, exist_ok=True)

    per_task = {}
    for tp in a.tapes:
        d = torch.load(tp, weights_only=False)
        meta = json.loads((tp.parent / "summary.json").read_text())
        rec = {r["idx"]: r for r in meta["episode_records"]}
        z, ep, tt = d["z"], d["episode"], d["t"]
        acc = per_task.setdefault(d["task"], {"z": [], "stage": [], "succ": []})
        for e in sorted(set(ep.tolist())):
            m_ = ep == e
            last = int(torch.nonzero(m_).flatten()[torch.argmax(tt[m_])])
            r = rec.get(int(e))
            if r is None:
                continue
            acc["z"].append(z[last]); acc["stage"].append(len(r.get("events", {})))
            acc["succ"].append(bool(r["success"]))

    rows = []
    for task, acc in sorted(per_task.items()):
        Z = torch.stack(acc["z"]); st = np.array(acc["stage"], float)
        su = np.array(acc["succ"], bool)
        f = ~su
        if f.sum() < 25 or len(np.unique(st[f])) < 2:
            print(f"{task}: {int(f.sum())} failures, {len(np.unique(st[f]))} levels "
                  f"- skipped"); continue
        Zf = Z[torch.tensor(np.flatnonzero(f))]
        y = torch.tensor(st[f], dtype=torch.float32)
        mu, sd = Zf.mean(0), Zf.std(0) + 1e-6
        X = (Zf - mu) / sd
        n = len(y)
        rhos = []
        for s in range(a.restarts):
            g = torch.Generator().manual_seed(s)
            perm = torch.randperm(n, generator=g)
            cut = int(n * 0.7)
            tr, te = perm[:cut], perm[cut:]
            if len(torch.unique(y[te])) < 2:
                continue
            torch.manual_seed(s)
            m = nn.Sequential(nn.LayerNorm(X.shape[-1]), nn.Linear(X.shape[-1], 128),
                              nn.GELU(), nn.Linear(128, 1))
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
            best = -1.0
            for e in range(a.epochs):
                m.train(); opt.zero_grad()
                ((m(X[tr]).squeeze(-1) - y[tr]) ** 2).mean().backward(); opt.step()
                if e % 50 == 0 or e == a.epochs - 1:
                    m.eval()
                    with torch.no_grad():
                        r_ = spearman(m(X[te]).squeeze(-1).numpy(), y[te].numpy())
                    if not np.isnan(r_):
                        best = max(best, r_)
            rhos.append(best)
        if not rhos:
            print(f"{task}: no usable split"); continue
        mean, sdv = float(np.mean(rhos)), float(np.std(rhos))
        rows.append({"task": task, "n_failures": int(f.sum()),
                     "stage_levels": int(len(np.unique(st[f]))),
                     "ceiling_rho": mean, "sd": sdv, "per_restart": rhos})
        print(f"{task:14s} n={int(f.sum()):4d} levels={len(np.unique(st[f]))}  "
              f"SUPERVISED ceiling rho = {mean:+.3f} +/- {sdv:.3f}   "
              f"{[round(x,2) for x in rhos]}")

    print(f"\nbar for a reward is rho >= 0.35. Where the supervised ceiling is below "
          f"that,\nno reward construction can pass and the defect is in the latent, "
          f"not the objective.")
    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "rows": rows, "env_steps": 0,
         "note": "privileged labels used as a TRAINING target - upper bound only",
         "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()},
        indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
