#!/usr/bin/env python
"""Does the value we ALREADY have separate a failure that got far from one that did not?

    python scripts/probe_v167_value_stratification.py --wm <wm.pt> --tapes ...

Run before designing a new objective, on the instruction of the survey: if the
existing value gives rho ~ 0 on stage-reached over FAILED episodes, the problem is
not the objective family and a language-conditioned quasimetric will not fix it.

WHY THIS STATISTIC. Monotonicity within a trajectory is worthless as evidence - a
clock scores perfectly on it. What a usable value must do is order failures by how
far they got, which is a CROSS-trajectory property and needs no successes at all.
On a task that fails ~97% of the time that makes the whole failure buffer usable as
the sample, instead of the handful of successes.

    rho = Spearman( V(z_T; l), stage_reached )   over FAILED episodes only

`stage_reached` is privileged (BDDL milestone times) and is used **for evaluation
only, never for training** - stated explicitly because the distinction is the whole
point of the design.

NULLS, all required:
  clock         V(t) = -(T_max - t): rho ~ 0 by construction, so any rho above it is
                signal above elapsed time
  label-shuffle stage_reached permuted: rho ~ 0, CI covering 0, or the statistic
                itself is miscalibrated

Pre-registered threshold, fixed before looking: rho >= 0.35 with a bootstrap 95% CI
excluding 0.15. Given this project's history - a p = 0.043 at 64 seeds that became
p = 0.771 at 128 - the interval is the result, not the point estimate.
"""
from __future__ import annotations

import argparse, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import numpy as np, torch  # noqa: E402
from train_v153_td_value_wm import load_valuewm  # noqa: E402

OUT = REPO / "results" / "v167_value_stratification"


def _rank(x):
    """Average ranks for ties. argsort(argsort(x)) assigns ARBITRARY distinct ranks
    to tied values, which manufactured a spurious clock-null correlation of +0.64
    from a constant input - stage_reached is heavily tied (182 of 191 chain3
    failures share one value), so this is not a corner case here."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float)
    r[order] = np.arange(len(x), dtype=float)
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, r)
    return (sums / cnt)[inv]


def spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 1e-12 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", type=Path, required=True)
    ap.add_argument("--tapes", type=Path, nargs="+", required=True)
    a = ap.parse_args()
    ck = torch.load(a.wm, weights_only=False)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / stamp; out.mkdir(parents=True, exist_ok=True)

    rows = []
    for tp in a.tapes:
        d = torch.load(tp, weights_only=False)
        meta = json.loads((tp.parent / "summary.json").read_text())
        rec = {r["idx"]: r for r in meta["episode_records"]}
        z, u, ep, tt = d["z"], d["u"], d["episode"], d["t"]
        for e in sorted(set(ep.tolist())):
            m = ep == e
            if not m.any():
                continue
            last = int(torch.nonzero(m).flatten()[torch.argmax(tt[m])])
            r = rec.get(int(e))
            if r is None:
                continue
            rows.append({"z": z[last], "u": u[last], "t_last": int(tt[last]),
                         "success": bool(r["success"]),
                         "stage": len(r.get("events", {})),
                         "steps": int(r["steps"])})
    if not rows:
        print("no episodes"); return 1

    Z = torch.stack([r["z"] for r in rows])
    O = (Z - ck["mu_o"]) / ck["sd_o"]
    m = load_valuewm(ck, O.shape[-1], rows[0]["u"].shape[0], rows[0]["u"].shape[1],
                     ck["zdim"], encoder=ck.get("encoder", "mlp"))
    m.eval()
    with torch.no_grad():
        V = m.V(m.encode(O)).numpy()

    succ = np.array([r["success"] for r in rows])
    stage = np.array([r["stage"] for r in rows], dtype=float)
    tlast = np.array([r["t_last"] for r in rows], dtype=float)
    fail = ~succ
    print(f"{len(rows)} episodes, {int(succ.sum())} success ({succ.mean():.3f}); "
          f"{int(fail.sum())} failures")
    print(f"stage_reached over failures: {dict(zip(*np.unique(stage[fail], return_counts=True)))}")
    if fail.sum() < 20 or len(np.unique(stage[fail])) < 2:
        print("not enough failure-stage variation to run the statistic"); return 1

    rho = spearman(V[fail], stage[fail])
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(fail)
    bs = np.array([spearman(V[s], stage[s])
                   for s in (rng.choice(idx, len(idx)) for _ in range(4000))])
    lo, hi = float(np.nanquantile(bs, .025)), float(np.nanquantile(bs, .975))
    rho_clock = spearman(-(tlast.max() - tlast[fail]), stage[fail])
    rho_perm = float(np.mean([spearman(V[fail], rng.permutation(stage[fail]))
                              for _ in range(200)]))
    print(f"\nPRIMARY  rho(V, stage | failures) = {rho:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"NULL     clock                     = {rho_clock:+.3f}")
    print(f"NULL     label-shuffled            = {rho_perm:+.3f}")
    passed = (rho >= 0.35 and lo > 0.15 and abs(rho_clock) < 0.1 and abs(rho_perm) < 0.1)
    print(f"\npre-registered bar (rho>=0.35, CI lower>0.15, nulls in [-0.1,0.1]): "
          f"{'MET' if passed else 'NOT MET'}")
    if not passed:
        print("If NOT MET, the existing value does not order failures by progress. "
              "That is a property of this value, not yet of the objective family.")
    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "wm": str(a.wm), "episodes": len(rows),
         "successes": int(succ.sum()), "failures": int(fail.sum()),
         "rho": rho, "ci": [lo, hi], "rho_clock": rho_clock,
         "rho_label_shuffled": rho_perm, "passed": bool(passed), "env_steps": 0,
         "note": "stage_reached is privileged and used for EVALUATION ONLY",
         "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()},
        indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
