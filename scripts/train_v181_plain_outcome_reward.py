#!/usr/bin/env python
"""R7 — plain outcome probe. Framework v2 §4, the ablation of R1/R4's geometry.

    python scripts/train_v181_plain_outcome_reward.py --tapes ...

Specified by measurement, not chosen from the menu. On identical labels and splits,
a two-layer MLP scores rho = +0.339 on chain1b where R1/R4's quasimetric + language
anchor + Bradley-Terry scores -0.131. The metric structure did not merely fail to
help - it destroyed signal a plain network extracts. R7 removes all of it:

    V(z; l) = MLP([z ; e_l])       trained with BCE on EPISODE SUCCESS
                                   applied to every state, not only terminals

No quasimetric, no goal anchor, no preference pairs, no local/spread/mismatch terms.
The instruction embedding is concatenated so the head stays language-conditioned,
which keeps §6.1's shuffled-instruction null meaningful.

Trains and runs the §6.1 gate in one pass so the two cannot drift apart.

KNOWN CEILING, from a probe trained directly on the target: chain1b 0.581, chain3
0.212. The bar is 0.6x that, so 0.349 and 0.127. The binary-success label recovers
58% of the ceiling on chain1b and none on chain3, so R7 is expected to sit at the
bar on chain1b and fail on chain3 - stated in advance.
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
from train_v170_qrl_reward import instruction_embeddings  # noqa: E402

OUT = REPO / "results" / "v181_plain_outcome"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tapes", type=Path, nargs="+", required=True)
    ap.add_argument("--latent-dim", type=int, default=2073)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--restarts", type=int, default=6)
    a = ap.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / stamp; out.mkdir(parents=True, exist_ok=True)

    Z, EPI, TASK, SUC, STAGE, ISLAST = [], [], [], [], [], []
    gid = 0
    for tp in a.tapes:
        d = torch.load(tp, weights_only=False)
        if d["latent_dim"] != a.latent_dim:
            continue
        meta = json.loads((tp.parent / "summary.json").read_text())
        rec = {r["idx"]: r for r in meta["episode_records"]}
        z, ep, tt = d["z"], d["episode"], d["t"]
        for e in sorted(set(ep.tolist())):
            r = rec.get(int(e))
            if r is None:
                continue
            m_ = ep == e
            idx = torch.nonzero(m_).flatten()[torch.argsort(tt[m_])]
            for k, i in enumerate(idx.tolist()):
                Z.append(z[i]); EPI.append(gid); TASK.append(d["task"])
                SUC.append(float(r["success"]))
                STAGE.append(float(len(r.get("events", {}))))
                ISLAST.append(k == len(idx) - 1)
            gid += 1
    Z = torch.stack(Z)
    epi = torch.tensor(EPI); suc = torch.tensor(SUC)
    stage = np.array(STAGE); islast = np.array(ISLAST, bool)
    tasks = sorted(set(TASK)); tix = torch.tensor([tasks.index(t) for t in TASK])
    print(f"{len(Z)} states over {gid} episodes, {len(tasks)} instructions: {tasks}")

    embs = instruction_embeddings(tasks)
    E = torch.stack([embs[t] for t in tasks]); E = (E - E.mean(0)) / (E.std(0) + 1e-6)
    mu, sd = Z.mean(0), Z.std(0) + 1e-6
    X = torch.cat([(Z - mu) / sd, E[tix]], -1)
    Xalt = torch.cat([(Z - mu) / sd, E[(tix + 1) % len(tasks)]], -1)

    rows, states = [], []
    for s in range(a.restarts):
        g = torch.Generator().manual_seed(s)
        perm = torch.randperm(gid, generator=g)
        te_ep = set(perm[int(gid * 0.7):].tolist())
        tr = torch.tensor([i for i in range(len(Z)) if int(epi[i]) not in te_ep])
        torch.manual_seed(s)
        m = nn.Sequential(nn.LayerNorm(X.shape[-1]), nn.Linear(X.shape[-1], 128),
                          nn.GELU(), nn.Linear(128, 1))
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
        for e in range(a.epochs):
            i = tr[torch.randint(0, len(tr), (a.batch,), generator=g)]
            m.train(); opt.zero_grad()
            nn.functional.binary_cross_entropy_with_logits(
                m(X[i]).squeeze(-1), suc[i]).backward()
            opt.step()
        m.eval(); states.append({k: v.clone() for k, v in m.state_dict().items()})
        with torch.no_grad():
            V, Valt = m(X).squeeze(-1).numpy(), m(Xalt).squeeze(-1).numpy()
        for t in tasks:
            sel = np.array([x == t for x in TASK]) & islast & (suc.numpy() == 0) & \
                  np.array([int(epi[i]) in te_ep for i in range(len(Z))])
            if sel.sum() < 15 or len(np.unique(stage[sel])) < 2:
                continue
            rows.append({"restart": s, "task": t, "n": int(sel.sum()),
                         "rho": spearman(V[sel], stage[sel]),
                         "null_instr": spearman(Valt[sel], stage[sel])})

    print()
    summary = []
    CEIL = {"chain1b_lr2": 0.581, "chain3_lr2": 0.212, "chain2b_lr2": 0.414}
    for t in tasks:
        rs = [r["rho"] for r in rows if r["task"] == t and not np.isnan(r["rho"])]
        ns = [r["null_instr"] for r in rows if r["task"] == t]
        if not rs:
            continue
        bar = 0.6 * CEIL.get(t, float("nan"))
        ok = float(np.mean(rs)) >= bar and abs(float(np.mean(ns))) < 0.1
        summary.append({"task": t, "rho": float(np.mean(rs)), "sd": float(np.std(rs)),
                        "null_instr": float(np.mean(ns)), "bar": bar,
                        "ceiling": CEIL.get(t), "passed": bool(ok)})
        print(f"{t:14s} rho={np.mean(rs):+.3f} +/- {np.std(rs):.3f}  "
              f"instr-null={np.mean(ns):+.3f}  bar={bar:.3f} "
              f"(0.6 x ceiling {CEIL.get(t)})  -> {'PASS' if ok else 'fail'}")
    torch.save({"states": states, "mu": mu, "sd": sd, "tasks": tasks, "E": E,
                "latent_dim": a.latent_dim}, out / "reward.pt")
    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "tasks": tasks, "states": len(Z), "episodes": gid,
         "per_task": summary, "rows": rows, "env_steps": 0,
         "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()},
        indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
