#!/usr/bin/env python
"""Does sigma actually change the candidate pool? A control on the v103 null.

v103 found p_cream = 0/8 at every tomato state for sigma in {0.3, 0.6} - but the
cream controls stayed at EXACTLY 1.00, bit-identical to the ODE run. Genuine
noise injection should perturb something. If the candidate pool is not measurably
wider under sigma, the v103 null says nothing about SDE sampling; it only says my
sigma was too small to matter.

This measures pool spread directly: mean pairwise L2 between n candidate chunks
at a fixed state, swept over sigma. Costs no environment steps.
"""
from __future__ import annotations

import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402
ensure_project_libero_config()

import numpy as np, torch  # noqa: E402
from lcwm.sampler import prefix_forward, sample_chunks  # noqa: E402
from lcwm.v080_bench import make_v080_env  # noqa: E402

TASK, C = "chain2b_lr2", 10
OUT = REPO / "results" / "v105_spread"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[3207, 3225, 3200])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--sigmas", type=float, nargs="+",
                    default=[0.0, 0.3, 0.6, 1.2, 2.5, 5.0])
    a = ap.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / stamp; out.mkdir(parents=True, exist_ok=True)

    from lcwm.chassis import DEFAULT_MODEL, Pi05Runner
    runner = Pi05Runner(model_id=DEFAULT_MODEL, suite_name="libero_10", n_action_steps=10)
    env = make_v080_env(TASK)

    rows = []
    for seed in a.seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        runner.reset(); obs, _ = env.reset(seed=int(seed))
        po = runner._obs_to_policy_batch(obs, env.task_description)
        with torch.no_grad():
            pf = prefix_forward(runner.policy, po)
            for sg in a.sigmas:
                ch = sample_chunks(runner.policy, po, a.n, seed=seed * 31,
                                   prefix=pf, sigma=sg)[:, :C].detach().float().cpu()
                d = torch.cdist(ch.flatten(1), ch.flatten(1))
                iu = torch.triu_indices(a.n, a.n, offset=1)
                pw = float(d[iu[0], iu[1]].mean())
                amp = float(ch.abs().mean())
                rows.append({"seed": seed, "sigma": sg, "mean_pairwise_l2": pw,
                             "mean_abs_action": amp})
                print(f"s{seed} sigma={sg:<4}: pairwise L2 {pw:.4f}  "
                      f"mean|a| {amp:.4f}", flush=True)
        del pf

    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "task": TASK, "n": a.n, "env_steps": 0, "rows": rows},
        indent=2))
    base = {r["seed"]: r["mean_pairwise_l2"] for r in rows if r["sigma"] == 0.0}
    print("\nspread relative to the ODE (sigma=0):")
    for r in rows:
        if r["sigma"] > 0:
            print(f"  s{r['seed']} sigma={r['sigma']:<4} "
                  f"x{r['mean_pairwise_l2'] / max(base[r['seed']], 1e-9):.2f}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
