#!/usr/bin/env python
"""Stage-2 done-check (plan §3): the chassis must reproduce Stage-1a numbers.

Runs the raw chassis loop on one suite and writes chassis_parity.json (+ fingerprint)
for side-by-side comparison with the lerobot-eval eval_info.json of the same suite.

Usage (inside vf0s, GPU free):
    python scripts/parity_check.py --suite libero_spatial --tasks 10 --eps 10 --seed 1000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

from lcwm.chassis import Pi05Runner, make_task_env, run_episode  # noqa: E402
from lcwm.fingerprint import write_fingerprint  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--tasks", type=int, default=10, help="first N task ids")
    p.add_argument("--eps", type=int, default=10, help="episodes per task")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--model", default="lerobot/pi05_libero_finetuned")
    p.add_argument("--out", default=str(REPO_ROOT.parent / "eval_logs" / "stage2_parity"))
    args = p.parse_args()

    runner = Pi05Runner(model_id=args.model, suite_name=args.suite)
    per_task: dict[int, list[bool]] = {}
    seed = args.seed
    for tid in range(args.tasks):
        env = make_task_env(args.suite, tid)
        results = []
        for ep in range(args.eps):
            res = run_episode(runner, env, seed=seed)
            results.append(res.success)
            seed += 1
            print(f"[{args.suite} t{tid} ep{ep}] success={res.success} steps={res.steps}",
                  flush=True)
        env.close()
        per_task[tid] = results

    sr_per_task = {t: 100.0 * sum(r) / len(r) for t, r in per_task.items()}
    overall = sum(sr_per_task.values()) / len(sr_per_task)
    out_dir = Path(args.out) / f"{args.suite}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "chassis_parity.json").write_text(json.dumps({
        "suite": args.suite, "model": args.model, "seed_start": args.seed,
        "eps_per_task": args.eps, "pc_success_overall": overall,
        "pc_success_per_task": sr_per_task,
    }, indent=2))
    write_fingerprint(out_dir, {"run": {"kind": "stage2_parity", **vars(args)}})
    print(f"\noverall pc_success = {overall:.1f}%  -> {out_dir}")


if __name__ == "__main__":
    main()
