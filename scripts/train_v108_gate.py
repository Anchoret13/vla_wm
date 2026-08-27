#!/usr/bin/env python
"""The deployment-time WM: predict, at t=0, whether pi_0 will pick the wrong object.

    python scripts/train_v108_gate.py --states results/v101_acq_states/<run>/summary.json

This is the component that earns the name. v099 established that a correction
applied everywhere is harmful: intervening at 226/256 boundaries dropped
cream-first success from 0.944 to 0.873. v102's corrector carries a head delta of
1.4e-01, two orders above pi_1's, so applying it unconditionally is a live risk
to the ~88% of states the policy already handles.

So the WM's job is not to choose an action - v106 showed there is nothing to
choose among (p_cream = 0.016 at 2-6x pool spread, 352 rollouts). Its job is to
predict the OUTCOME of the frozen policy from the initial state and gate the
correction on that prediction:

    pi_deploy(s) = pi_0(s) + 1[WM(s) > tau] * delta(s)   for the first 40 steps

Labels are v101's executed rollout outcomes over the held-out acquisition range;
the 3200-3399 panel is never touched. Costs no environment steps - only a reset
and one prefix forward per state.
"""
from __future__ import annotations

import argparse, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402
ensure_project_libero_config()

import numpy as np, torch  # noqa: E402
from torch import nn  # noqa: E402
from lcwm.sampler import prefix_forward  # noqa: E402
from lcwm.v080_bench import make_v080_env  # noqa: E402
from lcwm.v082_m0 import masked_prefix_mean  # noqa: E402

TASK = "chain2b_lr2"
PANEL = set(range(3200, 3400))
OUT = REPO / "results" / "v108_gate"


class OutcomeWM(nn.Module):
    """P(the frozen policy picks the WRONG object here | initial state)."""

    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden),
                                 nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, s):
        return self.net(s).squeeze(-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=400)
    a = ap.parse_args()
    v101 = json.loads(a.states.read_text())
    rows = [r for r in v101["rows"] if r["first"] in ("cream", "tomato")]
    assert rows and not ({r["seed"] for r in rows} & PANEL)
    torch.manual_seed(0); np.random.seed(0)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / stamp; out.mkdir(parents=True, exist_ok=True)
    print(f"{len(rows)} labelled states, "
          f"{sum(1 for r in rows if r['first'] == 'tomato')} positive (tomato)")

    from lcwm.chassis import DEFAULT_MODEL, Pi05Runner
    runner = Pi05Runner(model_id=DEFAULT_MODEL, suite_name="libero_10", n_action_steps=10)
    env = make_v080_env(TASK)
    X, Y = [], []
    for r in rows:
        torch.manual_seed(r["seed"]); np.random.seed(r["seed"])
        runner.reset(); obs, _ = env.reset(seed=int(r["seed"]))   # reset: 0 steps
        po = runner._obs_to_policy_batch(obs, env.task_description)
        with torch.no_grad():
            pf = prefix_forward(runner.policy, po)
            X.append(masked_prefix_mean(pf.hidden[0].detach().float().cpu(),
                                        pf.pad_masks[0].detach().cpu()))
        del pf
        Y.append(1.0 if r["first"] == "tomato" else 0.0)
    X = torch.stack(X); Y = torch.tensor(Y)
    pos = float(Y.sum())
    print(f"features {tuple(X.shape)}; positives {int(pos)}/{len(Y)}")

    # stratified k-fold: with ~10 positives a single split is not informative
    idx_p = [i for i in range(len(Y)) if Y[i] > 0]
    idx_n = [i for i in range(len(Y)) if Y[i] == 0]
    g = torch.Generator().manual_seed(0)
    idx_p = [idx_p[i] for i in torch.randperm(len(idx_p), generator=g)]
    idx_n = [idx_n[i] for i in torch.randperm(len(idx_n), generator=g)]
    folds = [([], []) for _ in range(a.folds)]
    for j, i in enumerate(idx_p): folds[j % a.folds][0].append(i)
    for j, i in enumerate(idx_n): folds[j % a.folds][1].append(i)

    aucs, accs, recalls = [], [], []
    for k in range(a.folds):
        te = folds[k][0] + folds[k][1]
        tr = [i for i in range(len(Y)) if i not in set(te)]
        m = OutcomeWM(X.shape[-1])
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-3)
        w = torch.tensor((len(idx_n) / max(len(idx_p), 1)))
        for _ in range(a.epochs):
            m.train(); opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(
                m(X[tr]), Y[tr], pos_weight=w)
            loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            p = torch.sigmoid(m(X[te]))
        yt = Y[te]
        npos, nneg = int(yt.sum()), int((yt == 0).sum())
        auc = float("nan")
        if npos and nneg:
            order = torch.argsort(p)
            ranks = torch.empty_like(order, dtype=torch.float)
            ranks[order] = torch.arange(1, len(p) + 1, dtype=torch.float)
            auc = float((ranks[yt > 0].sum() - npos * (npos + 1) / 2) / (npos * nneg))
        acc = float(((p > 0.5).float() == yt).float().mean())
        rec = float(((p > 0.5).float()[yt > 0]).mean()) if npos else float("nan")
        aucs.append(auc); accs.append(acc); recalls.append(rec)
        print(f"  fold {k}: n_te={len(te)} pos={npos} AUC={auc:.3f} "
              f"acc={acc:.3f} recall={rec:.3f}")

    m = OutcomeWM(X.shape[-1])
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-3)
    w = torch.tensor((len(idx_n) / max(len(idx_p), 1)))
    for _ in range(a.epochs):
        m.train(); opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(m(X), Y, pos_weight=w)
        loss.backward(); opt.step()
    mean = lambda v: float(np.nanmean(v))
    torch.save({"state_dict": m.state_dict(), "dim": X.shape[-1], "task": TASK,
                "cv_auc": mean(aucs), "cv_recall": mean(recalls)}, out / "gate.pt")
    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "task": TASK, "n_states": len(Y), "n_positive": int(pos),
         "folds": a.folds, "cv_auc": mean(aucs), "cv_acc": mean(accs),
         "cv_recall": mean(recalls), "per_fold_auc": aucs,
         "per_fold_recall": recalls, "env_steps": 0, "v101": str(a.states),
         "note": "gates the v102 corrector at deployment; labels are executed "
                 "outcomes over held-out acquisition states",
         "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()},
        indent=2))
    print(f"\n{a.folds}-fold CV: AUC {mean(aucs):.3f}  acc {mean(accs):.3f}  "
          f"recall {mean(recalls):.3f}")
    print(f"saved -> {out}/gate.pt   (0 environment steps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
