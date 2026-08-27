#!/usr/bin/env python
"""Fit the deployable gate at the timepoint where the state is predictive.

    python scripts/train_v115_gate_final.py --data results/v114_timed/<run>/timed.pt --tap 20

v114 measured when the frozen policy's object choice becomes predictable:

    t=0   AUC 0.745  recall@0%FP 0.037
    t=10  AUC 0.972  recall@0%FP 0.444
    t=20  AUC 0.999  recall@0%FP 0.963
    t=30  AUC 1.000  recall@0%FP 1.000

t=20 clears the bar set by v112's oracle bound (recall ~1.0 at essentially zero
false positives) while still leaving room to act, since a 20-step correction
already redirects 8/8.

Saves the whole transform - standardiser, PCA basis, logistic weights and a
threshold chosen for zero false positives on held-out folds - so deployment
applies exactly what was validated. Fitted on acquisition seeds only; the
3200-3263 panel is never seen.
"""
from __future__ import annotations

import argparse, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import numpy as np, torch  # noqa: E402

OUT = REPO / "results" / "v115_gate_final"


def fit_logreg(Z, y, alpha):
    w = torch.zeros(Z.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    Zt, yt = torch.tensor(Z), torch.tensor(y)
    pw = float((y == 0).sum()) / max(float((y > 0).sum()), 1.0)
    wgt = torch.where(yt > 0, torch.tensor(pw, dtype=torch.float64),
                      torch.tensor(1.0, dtype=torch.float64))
    opt = torch.optim.LBFGS([w, b], max_iter=2000, lr=0.1)

    def cl():
        opt.zero_grad()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(
            Zt @ w + b, yt, weight=wgt) + alpha * (w ** 2).sum())
        loss.backward(); return loss

    opt.step(cl)
    return w.detach().numpy(), float(b.detach())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--tap", type=int, default=20)
    ap.add_argument("--dims", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fp-quantile", type=float, default=1.0,
                    help="threshold = this quantile of held-out NEGATIVE scores. "
                         "1.0 = zero false positives (maximum specificity). "
                         "Lower it to buy recall: v116 measured that a false "
                         "positive is cheap - 1 fired on a good state and cost "
                         "nothing - while a missed error state costs a whole "
                         "conversion, so the loss function is asymmetric.")
    a = ap.parse_args()
    d = torch.load(a.data, weights_only=False)
    X = d["X"][a.tap].numpy().astype(np.float64); Y = d["y"].numpy().astype(np.float64)
    print(f"tap t={a.tap}: {len(Y)} states, {int(Y.sum())} positive")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / stamp; out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    ip = rng.permutation(np.flatnonzero(Y > 0)); ineg = rng.permutation(np.flatnonzero(Y == 0))
    folds = [[] for _ in range(a.folds)]
    for j, i in enumerate(ip): folds[j % a.folds].append(i)
    for j, i in enumerate(ineg): folds[j % a.folds].append(i)

    # held-out scores decide the threshold, so it is not tuned on the fit itself
    oof = np.zeros(len(Y))
    for fi in range(a.folds):
        te = np.array(folds[fi]); tr = np.setdiff1d(np.arange(len(Y)), te)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        Ztr, Zte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        _, _, Vt = np.linalg.svd(Ztr - Ztr.mean(0), full_matrices=False)
        B = Vt[:min(a.dims, Vt.shape[0])].T
        w, b = fit_logreg(Ztr @ B, Y[tr], a.alpha)
        oof[te] = 1 / (1 + np.exp(-((Zte @ B) @ w + b)))
    neg_max = float(np.quantile(oof[Y == 0], a.fp_quantile))
    rec = float((oof[Y > 0] > neg_max).mean())
    fpr = float((oof[Y == 0] > neg_max).mean())
    print(f"out-of-fold: recall = {rec:.3f} at FPR = {fpr:.3f}; "
          f"threshold = {neg_max:.4f} (quantile {a.fp_quantile})")

    mu, sd = X.mean(0), X.std(0) + 1e-8
    Z = (X - mu) / sd
    _, _, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    B = Vt[:min(a.dims, Vt.shape[0])].T
    w, b = fit_logreg(Z @ B, Y, a.alpha)
    torch.save({"mu": mu, "sd": sd, "B": B, "w": w, "b": b, "tap": a.tap,
                "threshold": neg_max, "dims": a.dims, "alpha": a.alpha,
                "oof_recall_at_0fp": rec, "oof_fpr": fpr,
                "fp_quantile": a.fp_quantile, "task": d["task"]}, out / "gate_final.pt")
    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "tap": a.tap, "dims": a.dims, "alpha": a.alpha,
         "n": len(Y), "n_positive": int(Y.sum()), "folds": a.folds,
         "oof_recall": rec, "oof_fpr": fpr, "fp_quantile": a.fp_quantile,
         "threshold": neg_max, "env_steps": 0,
         "data": str(a.data),
         "note": "fitted on acquisition seeds only; threshold chosen on held-out "
                 "folds for zero false positives",
         "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()},
        indent=2))
    print(f"saved -> {out}/gate_final.pt   (0 environment steps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
