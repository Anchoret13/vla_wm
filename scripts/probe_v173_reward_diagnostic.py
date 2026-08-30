#!/usr/bin/env python
"""§6.1 — does the reward order FAILURES by how far they got?

    python scripts/probe_v173_reward_diagnostic.py --reward <reward_k.pt> --tapes ...

The framework's gate: run before any policy is trained against this signal.

Monotonicity within a trajectory is worthless as evidence - a clock scores perfectly
on it. A usable signal must order FAILURES by progress, which is a cross-trajectory
property needing no successes, so on a 97-99%-failure task the whole buffer is the
sample.

    rho = Spearman( V(z_T; l), stage_reached )   over held-out FAILED episodes

`stage_reached` is privileged and used for EVALUATION ONLY, never for training.

NULLS (all required):
  shuffled instruction  V(z_T; l') for a different instruction -> rho ~ 0, else the
                        signal is not language-conditioned. This null needs >= 2
                        instructions and is the reason §4.3's mismatch term exists.
  clock                 -(T_max - t) -> rho ~ 0
  label-shuffled        stage permuted -> rho ~ 0, CI covering 0

BAR, pre-registered: rho >= 0.35, bootstrap 95% CI excluding 0.15, nulls in
[-0.1, 0.1]. The interval is the result, not the point estimate. The value we
already had scores rho = +0.247 with CI [+0.061, +0.412] - that is what this must
beat, and beating it is not the same as passing.

Ranks are tie-corrected: argsort(argsort(x)) gives tied values arbitrary distinct
ranks and previously manufactured a spurious null of +0.64 from a constant input.
"""
from __future__ import annotations

import argparse, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import numpy as np, torch  # noqa: E402
from probe_v167_value_stratification import spearman  # noqa: E402
from train_v170_qrl_reward import QRLReward  # noqa: E402

OUT = REPO / "results" / "v173_reward_diagnostic"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", type=Path, required=True)
    ap.add_argument("--tapes", type=Path, nargs="+", required=True)
    ap.add_argument("--per-task", action="store_true", default=True)
    a = ap.parse_args()
    ck = torch.load(a.reward, weights_only=False)
    tasks, E = ck["tasks"], ck["E"]
    m = QRLReward(ck["obs_dim"], ck["emb_dim"], ck["zdim"])
    m.load_state_dict(ck["state_dict"]); m.eval()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / stamp; out.mkdir(parents=True, exist_ok=True)

    rows = []
    for tp in a.tapes:
        d = torch.load(tp, weights_only=False)
        meta = json.loads((tp.parent / "summary.json").read_text())
        rec = {r["idx"]: r for r in meta["episode_records"]}
        z, ep, tt = d["z"], d["episode"], d["t"]
        task = d["task"]
        if task not in tasks:
            print(f"skip {tp.name}: task {task} not in the reward's instruction set")
            continue
        for e in sorted(set(ep.tolist())):
            msk = ep == e
            last = int(torch.nonzero(msk).flatten()[torch.argmax(tt[msk])])
            r = rec.get(int(e))
            if r is None:
                continue
            rows.append({"z": z[last], "task": task, "t_last": int(tt[last]),
                         "success": bool(r["success"]),
                         "stage": len(r.get("events", {}))})
    if not rows:
        print("no episodes"); return 1

    Z = torch.stack([r["z"] for r in rows])
    O = (Z - ck["mu_o"]) / ck["sd_o"]
    ti = torch.tensor([tasks.index(r["task"]) for r in rows])
    with torch.no_grad():
        zt = m.encode(O)
        V_own = m.V(zt, m.g(E[ti])).numpy()
        alt = (ti + 1) % len(tasks)
        V_alt = m.V(zt, m.g(E[alt])).numpy()

    succ = np.array([r["success"] for r in rows])
    stage = np.array([r["stage"] for r in rows], float)
    tlast = np.array([r["t_last"] for r in rows], float)
    tarr = np.array([r["task"] for r in rows])
    rng = np.random.default_rng(0)

    out_rows = []
    groups = [("ALL", np.ones(len(rows), bool))] + \
             [(t, tarr == t) for t in sorted(set(tarr))]
    for name, sel in groups:
        f = sel & ~succ
        if f.sum() < 20 or len(np.unique(stage[f])) < 2:
            print(f"{name}: {int(f.sum())} failures, "
                  f"{len(np.unique(stage[f]))} stage levels - skipped")
            continue
        rho = spearman(V_own[f], stage[f])
        idx = np.flatnonzero(f)
        bs = np.array([spearman(V_own[s], stage[s])
                       for s in (rng.choice(idx, len(idx)) for _ in range(3000))])
        lo, hi = float(np.nanquantile(bs, .025)), float(np.nanquantile(bs, .975))
        n_shuf = spearman(V_alt[f], stage[f])
        n_clock = spearman(-(tlast[f].max() - tlast[f]), stage[f])
        n_perm = float(np.mean([spearman(V_own[f], rng.permutation(stage[f]))
                                for _ in range(200)]))
        passed = bool(rho >= 0.35 and lo > 0.15 and
                      all(abs(x) < 0.1 or np.isnan(x) for x in (n_shuf, n_clock, n_perm)))
        out_rows.append({"group": name, "n_failures": int(f.sum()), "rho": rho,
                         "ci": [lo, hi], "null_shuffled_instruction": n_shuf,
                         "null_clock": n_clock, "null_label_shuffle": n_perm,
                         "passed": passed})
        print(f"{name:14s} n={int(f.sum()):4d}  rho={rho:+.3f} CI[{lo:+.3f},{hi:+.3f}]  "
              f"| nulls: instr {n_shuf:+.3f}  clock {n_clock:+.3f}  perm {n_perm:+.3f}"
              f"  -> {'PASS' if passed else 'fail'}")

    print(f"\nbar: rho>=0.35, CI lower>0.15, nulls in [-0.1,0.1]")
    print(f"prior value (chain1b): rho=+0.247, CI [+0.061,+0.412] - NOT passing")
    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "reward": str(a.reward), "tasks": tasks, "rows": out_rows,
         "env_steps": 0, "note": "stage_reached privileged, EVALUATION ONLY",
         "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()},
        indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
