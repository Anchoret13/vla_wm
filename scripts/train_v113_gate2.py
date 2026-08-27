#!/usr/bin/env python
"""Gate v2: regularized, dimensionality-reduced, searched on cached features.

    python scripts/train_v113_gate2.py --data results/v111_gate_data/<run>/gate_data.pt

v108 fitted 2048 dimensions to 85 states with 10 positives and reached CV AUC
0.573. v111 supplies 174 states with 27 positives and caches the features, so the
search runs on CPU.

The bar is set by v112's oracle bound, not by AUC. Correcting all 8 tomato states
converts 6; correcting a good state damages it ~5.9% of the time. With 51 good
states, a gate firing on a fraction fpr of them damages ~3*fpr. McNemar over 64
paired seeds then needs:

    gained 6, lost 0 -> p = 0.031   significant
    gained 6, lost 1 -> p = 0.125   not significant

So the operating point must reach recall ~1.0 AND essentially zero false
positives among good states - a demanding target that AUC alone does not
capture, which is why this reports recall at high-specificity thresholds rather
than AUC alone.
"""
from __future__ import annotations

import argparse, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import numpy as np, torch  # noqa: E402

OUT = REPO / "results" / "v113_gate2"


def auc(scores: np.ndarray, y: np.ndarray) -> float:
    npos, nneg = int(y.sum()), int((y == 0).sum())
    if not npos or not nneg:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float); ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[y > 0].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--dims", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.01, 0.1, 1.0, 10.0, 100.0])
    a = ap.parse_args()
    d = torch.load(a.data, weights_only=False)
    X = d["X"].numpy().astype(np.float64); Y = d["y"].numpy().astype(np.float64)
    print(f"{len(Y)} states, {int(Y.sum())} positive ({Y.mean():.3f})")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / stamp; out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    ip = rng.permutation(np.flatnonzero(Y > 0)); ineg = rng.permutation(np.flatnonzero(Y == 0))
    folds = [[] for _ in range(a.folds)]
    for j, i in enumerate(ip): folds[j % a.folds].append(i)
    for j, i in enumerate(ineg): folds[j % a.folds].append(i)

    def fit_project(Xtr, Xte, k):
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
        # PCA by SVD on the TRAINING split only; the test split is projected
        # with the training basis so no test information leaks into the fit.
        _, _, Vt = np.linalg.svd(Ztr - Ztr.mean(0), full_matrices=False)
        B = Vt[:min(k, Vt.shape[0])].T
        return Ztr @ B, Zte @ B

    def fit_logreg(Z, y, alpha, iters=2000, lr=0.1):
        w = torch.zeros(Z.shape[1], dtype=torch.float64, requires_grad=True)
        b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        Zt = torch.tensor(Z); yt = torch.tensor(y)
        pw = float((y == 0).sum()) / max(float((y > 0).sum()), 1.0)   # balanced
        wgt = torch.where(yt > 0, torch.tensor(pw, dtype=torch.float64),
                          torch.tensor(1.0, dtype=torch.float64))
        opt = torch.optim.LBFGS([w, b], max_iter=iters, lr=lr)

        def closure():
            opt.zero_grad()
            logit = Zt @ w + b
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                logit, yt, weight=wgt) + alpha * (w ** 2).sum())
            loss.backward()
            return loss

        opt.step(closure)
        return w.detach().numpy(), float(b.detach())

    rows = []
    for k in a.dims:
        for al in a.alphas:
            sc, yy = np.zeros(len(Y)), Y.copy()
            for fi in range(a.folds):
                te = np.array(folds[fi]); tr = np.setdiff1d(np.arange(len(Y)), te)
                Ztr, Zte = fit_project(X[tr], X[te], k)
                w, b = fit_logreg(Ztr, Y[tr], al)
                sc[te] = 1.0 / (1.0 + np.exp(-(Zte @ w + b)))
            A = auc(sc, yy)
            # the deployment-relevant operating point: recall when NO negative fires
            neg_max = sc[yy == 0].max()
            rec_at_zero_fp = float((sc[yy > 0] > neg_max).mean())
            thr95 = np.quantile(sc[yy == 0], 0.95)
            rec_at_5fp = float((sc[yy > 0] > thr95).mean())
            rows.append({"dims": k, "alpha": al, "auc": A,
                         "recall_at_0_fp": rec_at_zero_fp,
                         "recall_at_5pct_fp": rec_at_5fp})
            print(f"  dims={k:<3} alpha={al:<6} AUC={A:.3f}  "
                  f"recall@0%FP={rec_at_zero_fp:.3f}  recall@5%FP={rec_at_5fp:.3f}")

    best = max(rows, key=lambda r: (r["recall_at_0_fp"], r["auc"]))
    print(f"\nbest by recall@0%FP: dims={best['dims']} alpha={best['alpha']} "
          f"AUC={best['auc']:.3f} recall@0%FP={best['recall_at_0_fp']:.3f}")
    bar = ("MET: a gate at this operating point could reach gained 6 / lost 0"
           if best["recall_at_0_fp"] >= 0.99 else
           "NOT MET: significance needs recall ~1.0 at essentially zero FP; "
           "gained 6 / lost 1 is already p=0.125")
    print(bar)
    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "data": str(a.data), "n": len(Y), "n_positive": int(Y.sum()),
         "folds": a.folds, "grid": rows, "best": best, "bar": bar, "env_steps": 0,
         "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()},
        indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
