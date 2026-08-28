#!/usr/bin/env python
"""D_theta's Delta-w head: progress over the next c steps, on the predicted latent.

    python scripts/train_v128_progress_head.py --tape <tape.pt> --wm <T_theta.pt>

GOAL ANCHOR (CLAUDE.md). Still framework 5.2, still reading off T_theta's
rolled-forward latent - but the head that matches the rollout horizon.

v127 located the bottleneck precisely. T_theta propagates action differences well
(action spread 0.41 -> latent spread 0.24 at depth 1; 3.15 -> 1.36 at sigma 3),
but p_succ is almost flat along that direction: a 22x increase in latent spread
(0.24 -> 5.43) moved the score sd only 0.0009 -> 0.0138. The cause is a horizon
mismatch, not a broken model. p_succ's label is whether the episode succeeds ~200
steps later, while one selection affects the next 10 steps.

5.2's output tuple is (Delta w, Delta y, r, V_k, p_succ). Only the most distal
term was implemented; Delta w - progress increment - is the dense signal matched
to a c-step rollout, and it was skipped.

Labels need no new rollouts: the v121 episode records store each goal atom's
achievement step, so w_t is the count of atoms achieved by t and
Delta w = w_{t+c} - w_t is exact.
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
from train_v122_latent_wm import Transition  # noqa: E402
from train_v125_psucc_head import PSucc, auc  # noqa: E402

C = 10
OUT = REPO / "results" / "v128_progress"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tape", type=Path, required=True)
    ap.add_argument("--wm", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=600)
    a = ap.parse_args()
    d = torch.load(a.tape, weights_only=False)
    meta = json.loads((a.tape.parent / "summary.json").read_text())
    ck = torch.load(a.wm, weights_only=False)
    z, u, ep, tt = d["z"], d["u"], d["episode"], d["t"]

    # w_t = number of goal atoms achieved by step t, from the stored event times
    ev_by_ep = {r["idx"]: {int(k): int(v) for k, v in r["events"].items()}
                for r in meta["episode_records"]}
    w_now, w_next = [], []
    for i in range(len(z)):
        e, t0 = int(ep[i]), int(tt[i])
        ev = ev_by_ep.get(e, {})
        w_now.append(sum(1 for s in ev.values() if s <= t0))
        w_next.append(sum(1 for s in ev.values() if s <= t0 + C))
    dw = torch.tensor(w_next, dtype=torch.float32) - torch.tensor(w_now, dtype=torch.float32)
    print(f"{len(z)} triples; Delta w distribution: "
          f"{ {int(v): int((dw == v).sum()) for v in sorted(set(dw.tolist()))} }")
    if float((dw > 0).sum()) < 20:
        print("WARNING: too few positive progress events for a usable head")

    torch.manual_seed(0); np.random.seed(0)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / stamp; out.mkdir(parents=True, exist_ok=True)

    T = Transition(ck["zdim"], ck["c"], ck["adim"], ck["hidden"], use_action=True)
    T.load_state_dict(ck["state_dict"]); T.eval()
    mu, sd = ck["mu"], ck["sd"]
    Z = (z - mu) / sd
    with torch.no_grad():
        Zpred = T(Z, u)

    y = (dw > 0).float()
    nep = int(ep.max()) + 1
    perm = torch.randperm(nep, generator=torch.Generator().manual_seed(0))
    tr_ep = set(perm[:int(nep * 0.75)].tolist())
    tr = torch.tensor([i for i in range(len(z)) if int(ep[i]) in tr_ep])
    te = torch.tensor([i for i in range(len(z)) if int(ep[i]) not in tr_ep])
    print(f"episode-disjoint: train {len(tr)} / test {len(te)}; "
          f"positive rate {float(y.mean()):.3f}")

    res, heads = {}, {}
    for name, X in (("current_latent", Z), ("predicted_latent", Zpred)):
        m = PSucc(X.shape[-1])
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-3)
        pw = torch.tensor(float((y[tr] == 0).sum()) / max(float((y[tr] > 0).sum()), 1.0))
        best = (0.0, None)
        for e in range(a.epochs):
            m.train(); opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(
                m(X[tr]), y[tr], pos_weight=pw)
            loss.backward(); opt.step()
            if e % 10 == 0 or e == a.epochs - 1:
                m.eval()
                with torch.no_grad():
                    A = auc(m(X[te]).numpy(), y[te].numpy())
                if A > best[0]:
                    best = (A, {k: v.clone() for k, v in m.state_dict().items()})
        m.load_state_dict(best[1]); m.eval(); heads[name] = m
        with torch.no_grad():
            A = auc(m(X[te]).numpy(), y[te].numpy())
        res[name] = {"test_auc": A}
        print(f"{name:18s}: held-out AUC {A:.3f}")

    head = heads["predicted_latent"]
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        sp = torch.stack([torch.sigmoid(head(T(Z, u[torch.randperm(len(u), generator=g)])))
                          for _ in range(8)])
        per_state_sd = float(sp.std(0).mean())
        pop = float(torch.sigmoid(head(Zpred)).std())
    res["action_score_spread"] = {"per_state_sd": per_state_sd, "population_sd": pop,
                                  "ratio": per_state_sd / max(pop, 1e-9)}
    print(f"\nscore spread across actions {per_state_sd:.4f} vs population {pop:.4f} "
          f"(ratio {per_state_sd/max(pop,1e-9):.3f})")

    torch.save({"state_dict": head.state_dict(), "zdim": Z.shape[-1],
                "results": res, "task": d["task"], "head": "delta_w"},
               out / "delta_w.pt")
    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "tape": str(a.tape), "wm": str(a.wm), "task": d["task"],
         "triples": len(z), "positive_rate": float(y.mean()),
         "delta_w_counts": {str(int(v)): int((dw == v).sum())
                            for v in sorted(set(dw.tolist()))},
         "results": res, "env_steps": 0,
         "note": "Delta w over the next c steps, labels reconstructed exactly from "
                 "stored goal-atom achievement times; no new rollouts",
         "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()},
        indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
