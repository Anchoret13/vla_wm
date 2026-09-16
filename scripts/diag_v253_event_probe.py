#!/usr/bin/env python
"""Does the deployed latent retain the task state, or only the clock?

    python scripts/diag_v253_event_probe.py

WHAT THIS ANSWERS.  CLAUDE.md names two zero-cost diagnostics that would have
explained four to five candidate failures had they been run first: the supervised
ceiling for the target quantity, and whether the representation retains the signal at
all.  This is the second one, for the quantity the round's whole construction rests on.

The 2026-09-12 finding was that Phi' is 97.2% predictable from the timestep alone, and
the head deployed on 09-13 scored 0.389 on the decision region against 0.844 over all
boundaries.  Both are statements about a FITTED HEAD.  Neither says whether the 8217-d
rich latent the head reads from carries the task state at all, or whether every head
fitted on it is reading a clock.  That is a property of the representation, and it is
answered with no environment steps and no model.

THE TEST.  For each of chain3's three goal-predicate bits (`labels.pt['bits']`, the
achieved-goal state at a boundary), fit a linear probe and compare three feature sets
on episode-disjoint splits:

    clock        the boundary index t alone
    latent       the 8217-d rich latent alone
    latent+clock both

A latent that merely encodes elapsed time cannot beat `clock`.  A latent that retains
the achieved-goal state should beat it, and the gap is the quantity of interest.  The
probe is a ridge regression solved in DUAL form (n = 5,800 boundaries against d = 8,217
features, so the sample-space solve is exact and cheaper), scored by AUC.

WHAT WOULD OVERTURN THE READING.  A linear probe is a lower bound on what the
representation carries: a null here is evidence about linear decodability, not about
the representation.  A win here says the state is linearly present, not that any head
trained on it will use that rather than the clock - the deployed head's 0.389 vs 0.844
gap is exactly a head that did not.  Splits are episode-disjoint; a boundary-level
split would leak the state across the split within an episode and inflate every number.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "v253_event_probe"
ROUND = REPO / "results/v250_chain3_round"
D0 = ROUND / "chain3_lr2_d0_2026-09-12T224639Z"
D1 = [ROUND / d for d in (
    "chain3_lr2_d1_i0_2026-09-12T231912Z", "chain3_lr2_d1_i1_2026-09-12T232801Z",
    "chain3_lr2_d1_i2_2026-09-12T233655Z", "chain3_lr2_d1_i3_2026-09-12T234542Z")]


def auc(scores: torch.Tensor, y: torch.Tensor) -> float:
    """Rank AUC; 0.5 when one class is absent (reported as such, not dropped)."""
    pos, neg = (y > 0.5), (y <= 0.5)
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float64)
    n1, n0 = float(pos.sum()), float(neg.sum())
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


class RidgePath:
    """Exact ridge for a whole lambda grid from ONE eigendecomposition.

    The first run of this diagnostic used a single fixed lambda and the ordering
    between the feature sets reversed when the sample was doubled - which is what a
    fixed penalty does when n changes, not what a representation does. The whole path
    is therefore computed and the whole path is reported; no lambda is selected.

    Whichever of the sample (n x n) or feature (d x d) Gram is smaller is the one
    decomposed, which keeps both the D0-only (n < d) and pooled (n > d) cases exact.
    """

    def __init__(self, xtr: torch.Tensor, xte: torch.Tensor) -> None:
        n, d = xtr.shape
        self.dual = n <= d
        self.xtr, self.xte = xtr, xte
        if self.dual:
            g = (xtr @ xtr.T).double()
        else:
            g = (xtr.T @ xtr).double()
        self.w, self.v = torch.linalg.eigh(g)
        self.w.clamp_min_(0.0)
        self.kte = (xte @ xtr.T).double() if self.dual else None

    def scores(self, y: torch.Tensor, lam: float) -> torch.Tensor:
        yd = y.double()
        if self.dual:
            a = self.v @ ((self.v.T @ yd.unsqueeze(1)) / (self.w + lam).unsqueeze(1))
            return (self.kte @ a).squeeze(1).float()
        xty = (self.xtr.T @ yd.unsqueeze(1).float()).double()
        wgt = self.v @ ((self.v.T @ xty) / (self.w + lam).unsqueeze(1))
        return (self.xte.double() @ wgt).squeeze(1).float()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--lam-grid", type=float, nargs="+",
                    default=[0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
                    help="the penalties reported. Every one is reported for every "
                         "feature set; none is selected on any split")
    ap.add_argument("--seed", type=int, default=253)
    ap.add_argument("--include-d1", action="store_true",
                    help="pool the four D1 collections. D0 alone places the third "
                         "object in 1.2% of boundaries, which is too few positives "
                         "to resolve the bit the task actually fails on")
    a = ap.parse_args()

    sources = [D0] + (D1 if a.include_d1 else [])
    zs, bs, eps_l, ts, src, atoms = [], [], [], [], [], None
    ep_offset = 0
    for si, d in enumerate(sources):
        tape = torch.load(d / "tape.pt", map_location="cpu", weights_only=False)
        lab = torch.load(d / "labels.pt", map_location="cpu", weights_only=False)
        # labels carry one boundary more per episode than the tape carries rows;
        # align on the (episode, t) pairs both sides actually have.
        key_t = {(int(e), int(t)): i for i, (e, t) in
                 enumerate(zip(tape["episode"].tolist(), tape["t"].tolist()))}
        rows_t, rows_l = [], []
        for j, (e, t) in enumerate(zip(lab["episode"].tolist(), lab["t"].tolist())):
            i = key_t.get((int(e), int(t)))
            if i is not None:
                rows_t.append(i)
                rows_l.append(j)
        rows_t, rows_l = torch.tensor(rows_t), torch.tensor(rows_l)
        atoms = atoms or lab.get("goal_atoms")
        zs.append(tape["z"][rows_t].float())
        bs.append(lab["bits"][rows_l].float())
        # episode ids restart per collection; offset so folds stay episode-disjoint
        eps_l.append(tape["episode"][rows_t] + ep_offset)
        ts.append(tape["t"][rows_t].float())
        src += [si] * int(rows_t.shape[0])
        ep_offset += int(tape["episode"].max()) + 1
        print(f"  {d.name[:34]:<36} {int(rows_t.shape[0]):>5} boundaries")
        del tape, lab
    z, bits = torch.cat(zs), torch.cat(bs)
    ep, tt = torch.cat(eps_l), torch.cat(ts)
    del zs, bs, eps_l, ts
    print(f"pooled {z.shape[0]} boundaries from {len(sources)} collections; "
          f"z {tuple(z.shape)}, bits {tuple(bits.shape)}")

    eps = torch.unique(ep)
    g = torch.Generator().manual_seed(a.seed)
    perm = eps[torch.randperm(len(eps), generator=g)]
    folds = [perm[i::a.folds] for i in range(a.folds)]

    atoms = atoms or [f"bit{i}" for i in range(bits.shape[1])]

    # one decomposition per (fold, feature set), reused by every bit and every lambda
    fold_paths = []
    for f, held in enumerate(folds):
        te = torch.isin(ep, held)
        tr = ~te
        mu = z[tr].mean(0, keepdim=True)
        sd = z[tr].std(0, keepdim=True).clamp_min(1e-6)
        ztr, zte = (z[tr] - mu) / sd, (z[te] - mu) / sd
        tmu, tsd = tt[tr].mean(), tt[tr].std().clamp_min(1e-6)
        ttr = ((tt[tr] - tmu) / tsd).unsqueeze(1)
        tte = ((tt[te] - tmu) / tsd).unsqueeze(1)
        fold_paths.append({
            "clock": RidgePath(ttr, tte),
            "latent": RidgePath(ztr, zte),
            "latent_clock": RidgePath(torch.cat([ztr, ttr], 1),
                                      torch.cat([zte, tte], 1)),
        })
        print(f"  fold {f}: {int(tr.sum())} train / {int(te.sum())} test boundaries, "
              f"{'dual' if fold_paths[-1]['latent'].dual else 'primal'} solve")

    results = []
    for b in range(bits.shape[1]):
        y = bits[:, b]
        per_fold = []
        for f, held in enumerate(folds):
            te = torch.isin(ep, held)
            tr = ~te
            ytr, yte = y[tr], y[te]
            if yte.max() == yte.min():
                per_fold.append({"fold": f, "note": "held-out fold is single-class"})
                continue
            paths = fold_paths[f]
            row = {"fold": f, "n_train": int(tr.sum()), "n_test": int(te.sum()),
                   "pos_rate_test": float(yte.mean())}
            for name, path in paths.items():
                row[name] = {f"{lam:g}": auc(path.scores(ytr, lam), yte)
                             for lam in a.lam_grid}
            per_fold.append(row)
        scored = [r for r in per_fold if "clock" in r]
        summ = {}
        for k in ("clock", "latent", "latent_clock"):
            summ[k] = {}
            for lam in a.lam_grid:
                v = torch.tensor([r[k][f"{lam:g}"] for r in scored],
                                 dtype=torch.float64)
                summ[k][f"{lam:g}"] = {"mean": float(v.mean()),
                                       "min": float(v.min()), "max": float(v.max())}
        summ["latent_minus_clock"] = {}
        for lam in a.lam_grid:
            gap = torch.tensor([r["latent"][f"{lam:g}"] - r["clock"][f"{lam:g}"]
                                for r in scored], dtype=torch.float64)
            summ["latent_minus_clock"][f"{lam:g}"] = {
                "mean": float(gap.mean()), "min": float(gap.min()),
                "max": float(gap.max()),
                "folds_latent_wins": int((gap > 0).sum()), "folds": len(scored)}
        results.append({"bit": b, "atom": str(atoms[b]) if b < len(atoms) else f"bit{b}",
                        "base_rate": float(y.mean()),
                        "per_fold": per_fold, "summary": summ})
        print(f"\nbit {b} ({results[-1]['atom']}), base rate {float(y.mean()):.3f}, "
              f"{len(scored)} folds scored")
        print(f"  {'lambda':>8} {'clock':>7} {'latent':>7} {'lat+clk':>8} "
              f"{'lat-clk':>9}  wins")
        for lam in a.lam_grid:
            k = f"{lam:g}"
            g = summ["latent_minus_clock"][k]
            print(f"  {k:>8} {summ['clock'][k]['mean']:>7.3f} "
                  f"{summ['latent'][k]['mean']:>7.3f} "
                  f"{summ['latent_clock'][k]['mean']:>8.3f} "
                  f"{g['mean']:>+9.3f}  {g['folds_latent_wins']}/{g['folds']}")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / f"chain3_lr2_{'d0d1' if a.include_d1 else 'd0'}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps({
        "utc": stamp, "task": "chain3_lr2", "env_steps": 0,
        "sources": [str(d.relative_to(REPO)) for d in sources],
        "source_sha256": {d.name: hashlib.sha256(
            (d / "tape.pt").read_bytes()).hexdigest() for d in sources},
        "n_boundaries": int(z.shape[0]), "n_episodes": int(len(eps)),
        "folds": a.folds, "lam_grid": a.lam_grid, "seed": a.seed,
        "split": "episode-disjoint", "probe": "dual-form ridge, AUC",
        "results": results,
    }, indent=1))
    print(f"\nwrote {out.relative_to(REPO)}/result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
