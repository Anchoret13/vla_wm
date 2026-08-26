#!/usr/bin/env python
"""Testbed search: does chain3_lr2 fail for a locally-correctable reason?

chain1b@250 is exhausted - its failures are trajectory-time deficits with a
~10% local-flip ceiling at every anchor phase tried, and an oracle policy update
moved the frozen panel not at all.

V8.0 measured chain3_lr2 at 0/10 with 9/10 stopping at EXACTLY 4/6 milestones,
stuck on `pick_up cream_cheese_1`, and V7.7 identified the mechanism: pi0.5 does
not redirect to a later object while an earlier one is present. That is a
capability failure with a common cause, not a clock artifact - and a common
cause is what makes corrections transferable across states.

This measures the baseline and the failure structure in one pass: terminal
success, milestone distribution, WHERE progress stops, and how much time is left
when it stops. Nothing is corrected here.
"""
from __future__ import annotations

import argparse, json, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402
ensure_project_libero_config()

import numpy as np, torch  # noqa: E402
from lcwm.seq_data import goal_atoms, predicate_bits  # noqa: E402
from lcwm.task_automaton import GoalAutomaton  # noqa: E402
from lcwm.v080_bench import V080_TASKS, episode_length, make_v080_env  # noqa: E402
from lcwm.v08r_contract import clopper_pearson_lower, clopper_pearson_upper  # noqa: E402

TASK = "chain3_lr2"
PANEL = tuple(range(3200, 3232))          # same 32 seeds as chain1b's panel
OUT = REPO / "results" / "v091_chain3"


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=32)
    a = ap.parse_args()
    L = episode_length(TASK)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / stamp; out.mkdir(parents=True, exist_ok=True)
    from lcwm.chassis import DEFAULT_MODEL, Pi05Runner
    runner = Pi05Runner(model_id=DEFAULT_MODEL, suite_name="libero_10", n_action_steps=10)
    # v080r_panel.make_env_at is scoped to the two 1R tasks; the full
    # five-rung ladder lives in v080_bench with the derived horizon.
    env = make_v080_env(TASK)
    subgoals = V080_TASKS[TASK]["ordered_subgoals"]
    print(f"{TASK} L={L}, {len(subgoals)} milestones: {subgoals}")

    rows, steps = [], 0
    for seed in PANEL[:a.seeds]:
        torch.manual_seed(seed); np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        runner.reset(); obs, _ = env.reset(seed=seed)
        au = GoalAutomaton(subgoals); au.start(env); atoms = goal_atoms(env)
        au.evaluate(env, 0)
        t, done, succ = 0, False, None
        while not done and t < L:
            obs, _r, term, trunc, info = env.step(
                runner.select_action(obs, env.task_description))
            t += 1; done = bool(term or trunc)
            if succ is None and bool(info.get("is_success", False)):
                succ = t
            if t % 10 == 0 or done:
                au.evaluate(env, t)
                if succ is None and predicate_bits(env, atoms).all():
                    succ = t
        steps += t
        ev = {int(k): int(v) for k, v in au.events_achieved.items()}
        last = max(ev.values()) if ev else 0
        rows.append({"seed": seed, "success": succ is not None, "steps": t,
                     "milestones": len(ev), "events": ev,
                     "last_progress_step": last, "idle_tail": t - last,
                     "damage": au.damage_unrecovered()})
        print(f"seed={seed} success={succ is not None} milestones={len(ev)}/{len(subgoals)} "
              f"last_progress@{last} idle_tail={t-last} steps={t}", flush=True)

    k = sum(r["success"] for r in rows)
    ms = Counter(r["milestones"] for r in rows)
    stuck = [r for r in rows if not r["success"]]
    summary = {"task": TASK, "L": L, "utc": stamp, "n": len(rows),
               "successes": k, "rate": k / len(rows),
               "cp95": [clopper_pearson_lower(k, len(rows)),
                        clopper_pearson_upper(k, len(rows))],
               "milestone_hist": dict(sorted(ms.items())),
               "subgoals": subgoals, "env_steps": steps,
               "mean_idle_tail": sum(r["idle_tail"] for r in stuck) / max(len(stuck), 1),
               "median_last_progress": sorted(r["last_progress_step"] for r in stuck)[len(stuck)//2] if stuck else None,
               "episodes": rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{TASK}: {k}/{len(rows)} = {k/len(rows):.3f} "
          f"CP95 [{summary['cp95'][0]:.3f},{summary['cp95'][1]:.3f}]")
    print(f"milestone histogram: {dict(sorted(ms.items()))}  (of {len(subgoals)})")
    print(f"failures: median last progress @{summary['median_last_progress']}, "
          f"mean idle tail {summary['mean_idle_tail']:.0f} steps of {L}")
    print(f"{steps} env steps -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
