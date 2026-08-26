#!/usr/bin/env python
"""Action 2M.6 — frozen behavior panel for the pi_0 -> pi_1 -> pi_2 curve.

    python scripts/run_v088_behavior.py --policy stock --tag pi_0

32 seeds 3200-3231 on chain1b_lr2@250, full-prompt N=1. These seeds were
reserved for exactly this from Action 1R onward and have never been executed.
Behavior episodes never enter WM or VLA training.
"""
from __future__ import annotations

import argparse, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402
ensure_project_libero_config()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from lcwm import v086_bank as B  # noqa: E402
from lcwm.seq_data import goal_atoms, predicate_bits  # noqa: E402
from lcwm.task_automaton import GoalAutomaton  # noqa: E402
from lcwm.v080_bench import V080_TASKS  # noqa: E402
from lcwm.v080r_panel import make_env_at  # noqa: E402
from lcwm.v08r_contract import clopper_pearson_lower, clopper_pearson_upper  # noqa: E402

PANEL = tuple(range(3200, 3232))                    # 32, frozen and reserved
OUT_ROOT = REPO / "results" / "v088_behavior"
assert len(PANEL) == 32


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="stock",
                    help="'stock' or a path to a fine-tuned VLA checkpoint")
    ap.add_argument("--tag", required=True, help="pi_0 | pi_1 | pi_2")
    a = ap.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT_ROOT / f"{a.tag}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)

    from lcwm.chassis import DEFAULT_MODEL, Pi05Runner
    model_id = DEFAULT_MODEL if a.policy == "stock" else a.policy
    runner = Pi05Runner(model_id=model_id, suite_name="libero_10", n_action_steps=10)
    env = make_env_at(B.TASK, B.DEADLINE)
    subgoals = V080_TASKS[B.TASK]["ordered_subgoals"]

    rows, steps = [], 0
    for seed in PANEL:
        torch.manual_seed(seed); np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        runner.reset()
        obs, _ = env.reset(seed=seed)
        au = GoalAutomaton(subgoals); au.start(env)
        atoms = goal_atoms(env); au.evaluate(env, 0)
        t, done, succ = 0, False, None
        while not done and t < B.DEADLINE:
            obs, _r, term, trunc, info = env.step(
                runner.select_action(obs, env.task_description))
            t += 1
            done = bool(term or trunc)
            if succ is None and bool(info.get("is_success", False)):
                succ = t
            if t % 10 == 0 or done:
                au.evaluate(env, t)
                if succ is None and predicate_bits(env, atoms).all():
                    succ = t
        steps += t
        rows.append({"seed": seed, "success": succ is not None, "success_step": succ,
                     "steps": t, "milestones": len(au.events_achieved),
                     "damage": au.damage_unrecovered()})
        print(f"{a.tag} seed={seed} success={succ is not None} @{succ} steps={t}", flush=True)

    k = sum(r["success"] for r in rows)
    lo, up = clopper_pearson_lower(k, len(rows)), clopper_pearson_upper(k, len(rows))
    summary = {"action": "2M.6", "tag": a.tag, "utc": stamp,
               "policy": model_id, "task": B.TASK, "deadline": B.DEADLINE,
               "panel_seeds": [PANEL[0], PANEL[-1], len(PANEL)],
               "successes": k, "n": len(rows), "rate": k / len(rows),
               "cp95": [lo, up], "env_steps": steps,
               "mean_milestones": sum(r["milestones"] for r in rows) / len(rows),
               "damage_episodes": sum(r["damage"] > 0 for r in rows),
               "episodes": rows,
               "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                     capture_output=True, text=True).stdout.strip()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{a.tag}: {k}/{len(rows)} = {k/len(rows):.3f}  CP95 [{lo:.3f}, {up:.3f}]  "
          f"{steps} env steps -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
