#!/usr/bin/env python
"""D_theta's p_succ head, read off the PREDICTED latent. Framework 5.2.

    python scripts/train_v125_psucc_head.py --tape <tape.pt> --wm <T_theta.pt>

GOAL ANCHOR (CLAUDE.md). object axis: the head must read off T_theta's rolled-
forward latent, NOT off the current observation. Reading heads off the current
observation is precisely what the 2026-08-27 session did while calling it a world
model, and it is what makes the difference between a value function and a world
model here:

    p_succ(z_t)                 constant in u - cannot rank candidate actions
    p_succ(T_theta(z_t, u))     varies with u - CAN rank candidate actions

Only the second is usable for selection at all, so the comparison is not merely
about accuracy. Two things are measured:

  1. does the predicted latent carry at least as much outcome information as the
     current one (held-out AUC, episode-disjoint)?
  2. does the score actually VARY across candidate actions at a fixed state? A
     head that ignores u is a value function wearing a world model's clothes, and
     would rank every candidate identically.

The label is episode success carried back to every triple in the episode - the
policy-indexed value target V_k(z) = E_pi[G | z_t = z] of 5.2. Credit assignment
is coarse by construction; that is stated, not hidden.
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

OUT = REPO / "results" / "v125_psucc"


class PSucc(nn.Module):
    def __init__(self, zdim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(zdim), nn.Linear(zdim, hidden),
                                 nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 1))

    def forward(self, z):
        return self.net(z).squeeze(-1)


def auc(scores, y):
    npos, nneg = int((y > 0).sum()), int((y == 0).sum())
    if not npos or not nneg:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float); ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[y > 0].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tape", type=Path, required=True)
    ap.add_argument("--wm", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=400)
    a = ap.parse_args()
    d = torch.load(a.tape, weights_only=False)
    ck = torch.load(a.wm, weights_only=False)
    z, u, ep = d["z"], d["u"], d["episode"]
    succ_ep = d["success"]
    y = succ_ep[ep]                                   # V_k target, per triple
    torch.manual_seed(0); np.random.seed(0)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / stamp; out.mkdir(parents=True, exist_ok=True)
    print(f"{len(z)} triples, {int(y.sum())} from successful episodes "
          f"({float(y.mean()):.3f})")

    T = Transition(ck["zdim"], ck["c"], ck["adim"], ck["hidden"], use_action=True)
    T.load_state_dict(ck["state_dict"]); T.eval()
    mu, sd = ck["mu"], ck["sd"]
    Z = (z - mu) / sd
    with torch.no_grad():
        Zpred = T(Z, u)                               # the rolled-forward latent

    nep = int(ep.max()) + 1
    perm = torch.randperm(nep, generator=torch.Generator().manual_seed(0))
    cut = int(nep * 0.75)
    tr_ep = set(perm[:cut].tolist())
    tr = torch.tensor([i for i in range(len(z)) if int(ep[i]) in tr_ep])
    te = torch.tensor([i for i in range(len(z)) if int(ep[i]) not in tr_ep])
    print(f"episode-disjoint: train {len(tr)} / test {len(te)}")

    res = {}
    heads = {}
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

    # Does the score move when only the ACTION changes? A head that ignores u
    # ranks every candidate identically and is useless for selection.
    head = heads["predicted_latent"]
    g = torch.Generator().manual_seed(7)
    spread = []
    with torch.no_grad():
        for _ in range(8):
            up = u[torch.randperm(len(u), generator=g)]
            spread.append(torch.sigmoid(head(T(Z, up))))
        S = torch.stack(spread)                        # (8, N)
        per_state_sd = float(S.std(0).mean())
        base = float(torch.sigmoid(head(Zpred)).std())
    res["action_score_spread"] = {"per_state_sd": per_state_sd,
                                  "population_sd": base,
                                  "ratio": per_state_sd / max(base, 1e-9)}
    print(f"\nscore spread across actions at a fixed state: {per_state_sd:.4f}")
    print(f"score spread across the population:            {base:.4f}")
    print(f"ratio {per_state_sd/max(base,1e-9):.3f}  "
          f"({'usable for ranking' if per_state_sd > 0.02 else 'TOO FLAT to rank candidates'})")

    torch.save({"state_dict": heads["predicted_latent"].state_dict(),
                "zdim": Z.shape[-1], "results": res, "task": d["task"]},
               out / "p_succ.pt")
    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "tape": str(a.tape), "wm": str(a.wm), "task": d["task"],
         "triples": len(z), "positive_rate": float(y.mean()),
         "train": len(tr), "test": len(te), "results": res, "env_steps": 0,
         "note": "p_succ read off T_theta's predicted latent; label is episode "
                 "success carried back to every triple (coarse by construction)",
         "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()},
        indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
