#!/usr/bin/env python
"""Rank candidates from candidate-level labels - with the arm that can refute T_theta.

    python scripts/train_v137_rank_head.py --branches <branches.pt> --wm-proprio <T_theta.pt>

GOAL ANCHOR (CLAUDE.md). object axis. THREE arms, registered before running:

  state_only  head(z)                          cannot depend on u at all
  direct      head(z, E_a(u))                  action-conditioned value, NO world model
  wm          head(z_hidden, T_theta(z_pro,u)) framework 5.2's object

`state_only` must land at chance within a group - it is the sanity control, since
its score is identical for every candidate at a state.

`direct` is the arm that decides the project claim. If a head fed the raw action
ranks as well as one fed T_theta's rolled-forward latent, then the world model is
not doing the work and "a latent world model helps the VLA" is not supported,
however good the numbers are. This is the same control the 2026-08-27 session
lacked when it credited a gate that a no-gate arm later beat.

Evaluation is GROUP-disjoint and the metric is within-group AUC: ranking quality
among candidates at one state, which is the only thing selection needs. Marginal
AUC is not reported as the headline because a head can score 0.997 marginally
(v128) while ranking at chance (v133, v135).
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
from train_v122_latent_wm import ActionEncoder, Transition  # noqa: E402

OUT = REPO / "results" / "v137_rank_head"


class Head(nn.Module):
    def __init__(self, zdim, c=0, adim=0, hidden=256, mode="wm"):
        super().__init__()
        self.mode = mode
        self.enc = ActionEncoder(c, adim, 128) if mode == "direct" else None
        ind = zdim + (128 if mode == "direct" else 0)
        self.net = nn.Sequential(nn.LayerNorm(ind), nn.Linear(ind, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 1))

    def forward(self, z, u=None):
        h = torch.cat([z, self.enc(u)], -1) if self.mode == "direct" else z
        return self.net(h).squeeze(-1)


def within_group_auc(scores, y, grp):
    conc = disc = tie = 0
    for g in torch.unique(grp):
        m = grp == g
        s, o = scores[m], y[m]
        if not (0 < o.sum() < len(o)):
            continue
        for i in range(len(o)):
            for j in range(len(o)):
                if o[i] > o[j]:
                    conc += float(s[i] > s[j]); disc += float(s[i] < s[j])
                    tie += float(s[i] == s[j])
    tot = conc + disc + tie
    return (conc + 0.5 * tie) / tot if tot else float("nan")


def top1(scores, y, grp):
    v = []
    for g in torch.unique(grp):
        m = grp == g
        v.append(float(y[m][int(torch.argmax(scores[m]))]))
    return float(np.mean(v))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--branches", type=Path, required=True)
    ap.add_argument("--wm-proprio", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--depth", type=int, default=1)
    a = ap.parse_args()
    b = torch.load(a.branches, weights_only=False)
    ck = torch.load(a.wm_proprio, weights_only=False)
    zh, zp, u, y, grp = b["zh"], b["zp"], b["u"], b["y"], b["group"]
    torch.manual_seed(0); np.random.seed(0)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / stamp; out.mkdir(parents=True, exist_ok=True)
    print(f"{len(y)} candidates over {int(grp.max())+1} groups, "
          f"mean outcome {float(y.mean()):.3f}")

    T = Transition(ck["zdim"], ck["c"], ck["adim"], ck["hidden"], use_action=True)
    T.load_state_dict(ck["state_dict"]); T.eval()
    mu_p, sd_p = ck["mu"], ck["sd"]
    mu_h, sd_h = zh.mean(0), zh.std(0) + 1e-6
    Zh = (zh - mu_h) / sd_h
    with torch.no_grad():
        zpn = (zp - mu_p) / sd_p
        for _ in range(a.depth):
            zpn = T(zpn, u)
        X_wm = torch.cat([Zh, zpn], -1)
        X_state = torch.cat([Zh, (zp - mu_p) / sd_p], -1)   # current, no rollout

    ng = int(grp.max()) + 1
    perm = torch.randperm(ng, generator=torch.Generator().manual_seed(0))
    tr_g = set(perm[:int(ng * 0.75)].tolist())
    tr = torch.tensor([i for i in range(len(y)) if int(grp[i]) in tr_g])
    te = torch.tensor([i for i in range(len(y)) if int(grp[i]) not in tr_g])
    print(f"group-disjoint: train {len(tr)} / test {len(te)}")

    res = {}
    for arm in ("state_only", "direct", "wm"):
        X = X_state if arm != "wm" else X_wm
        m = Head(X.shape[-1], u.shape[1], u.shape[2], mode=arm)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-3)
        best = (0.0, None)
        for e in range(a.epochs):
            m.train(); opt.zero_grad()
            sc = m(X[tr], u[tr]) if arm == "direct" else m(X[tr])
            # pairwise ranking within group: the objective selection actually needs
            loss = nn.functional.binary_cross_entropy_with_logits(sc, y[tr])
            loss.backward(); opt.step()
            if e % 20 == 0 or e == a.epochs - 1:
                m.eval()
                with torch.no_grad():
                    s_te = m(X[te], u[te]) if arm == "direct" else m(X[te])
                A = within_group_auc(s_te, y[te], grp[te])
                if not np.isnan(A) and A > best[0]:
                    best = (A, {k: v.clone() for k, v in m.state_dict().items()})
        m.load_state_dict(best[1]); m.eval()
        with torch.no_grad():
            s_te = m(X[te], u[te]) if arm == "direct" else m(X[te])
        A, T1 = within_group_auc(s_te, y[te], grp[te]), top1(s_te, y[te], grp[te])
        # bootstrap the within-group AUC over held-out GROUPS
        gs = torch.unique(grp[te]); rng = np.random.default_rng(0)
        bs = []
        for _ in range(2000):
            pick = gs[torch.tensor(rng.integers(0, len(gs), len(gs)))]
            idx = torch.cat([te[(grp[te] == g)] for g in pick])
            gg = torch.cat([torch.full(((grp[te] == g).sum(),), int(g)) for g in pick])
            with torch.no_grad():
                ss = m(X[idx], u[idx]) if arm == "direct" else m(X[idx])
            v = within_group_auc(ss, y[idx], gg)
            if not np.isnan(v):
                bs.append(v)
        lo, hi = float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))
        res[arm] = {"within_group_auc": A, "ci": [lo, hi], "top1": T1}
        print(f"{arm:11s}: within-group AUC {A:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  "
              f"top-1 {T1:.3f}")
        if arm == "wm":
            torch.save({"state_dict": m.state_dict(), "zdim": X.shape[-1],
                        "mu_h": mu_h, "sd_h": sd_h, "mu_p": mu_p, "sd_p": sd_p,
                        "depth": a.depth, "wm_proprio": str(a.wm_proprio),
                        "task": b["task"], "results": res}, out / "rank_head.pt")

    wm_beats_direct = res["wm"]["within_group_auc"] - res["direct"]["within_group_auc"]
    verdict = ("T_theta CONTRIBUTES: rolling the latent forward ranks better than "
               "feeding the raw action" if wm_beats_direct > 0 else
               "T_theta adds nothing over a direct action-conditioned value: the "
               "world model is not doing the work")
    print(f"\nwm - direct = {wm_beats_direct:+.3f}")
    print(verdict)
    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "branches": str(a.branches), "wm": str(a.wm_proprio),
         "depth": a.depth, "candidates": len(y), "groups": ng,
         "results": res, "wm_minus_direct": wm_beats_direct, "verdict": verdict,
         "env_steps": 0,
         "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()},
        indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
