#!/usr/bin/env python
"""Label the HELD-OUT acquisition states by which object pi_0 picks first.

    python scripts/probe_v101_acq_states.py --seeds 90 --horizon 120

v100 established that first-object is a near-deterministic function of the
initial state (p_cream = 0.00 at eight states, ~0.94 at controls). So the states
where the policy errs are identifiable, and they are the only states where a
corrective update carries any gradient.

v097 distilled over 30 generic acquisition seeds, of which ~12% are tomato
states; at the other 88% the teacher merely confirms what the policy already
does, so the update was mostly zero-gradient. That is the DAgger principle
violated - train where the learner errs - and it explains a head delta of
8.96e-4.

This runs a short first-object probe over the acquisition range (NOT the
3200-3263 evaluation panel) to find those states. Evaluation stays frozen and
disjoint, so nothing here is train-on-test.
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
from lcwm.task_automaton import GoalAutomaton  # noqa: E402
from lcwm.v080_bench import V080_TASKS, make_v080_env  # noqa: E402

TASK = "chain2b_lr2"
ACQ = tuple(range(4900, 4990))
PANEL = set(range(3200, 3400))
OUT = REPO / "results" / "v101_acq_states"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=90)
    ap.add_argument("--horizon", type=int, default=120)
    a = ap.parse_args()
    seeds = ACQ[:a.seeds]
    assert not (set(seeds) & PANEL), "acquisition must not touch the evaluation panel"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / stamp; out.mkdir(parents=True, exist_ok=True)

    from lcwm.chassis import DEFAULT_MODEL, Pi05Runner
    runner = Pi05Runner(model_id=DEFAULT_MODEL, suite_name="libero_10", n_action_steps=10)
    env = make_v080_env(TASK); subgoals = V080_TASKS[TASK]["ordered_subgoals"]

    rows, steps = [], 0
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        runner.reset(); obs, _ = env.reset(seed=int(seed))
        au = GoalAutomaton(subgoals); au.start(env); au.evaluate(env, 0)
        t, done = 0, False
        while not done and t < a.horizon:
            obs, _r, tm, tr, _i = env.step(runner.select_action(obs, env.task_description))
            t += 1; done = bool(tm or tr)
            if t % 10 == 0 or done:
                au.evaluate(env, t)
                if au.events_achieved: break      # first object decided
        steps += t
        ev = {int(k): int(v) for k, v in au.events_achieved.items()}
        order = [i for i, _ in sorted(ev.items(), key=lambda kv: kv[1])]
        first = ("cream" if order and order[0] == 2 else
                 "tomato" if order and order[0] == 0 else "none")
        rows.append({"seed": seed, "first": first, "steps": t})
        print(f"s{seed}: {first:6s} @{t:3d}  ({steps} steps)", flush=True)

    tom = [r["seed"] for r in rows if r["first"] == "tomato"]
    non = [r["seed"] for r in rows if r["first"] == "none"]
    summary = {"utc": stamp, "task": TASK, "horizon": a.horizon,
               "acq_range": [seeds[0], seeds[-1]], "n": len(rows),
               "counts": {f: sum(1 for r in rows if r["first"] == f)
                          for f in ("cream", "tomato", "none")},
               "tomato_states": tom, "none_states": non,
               "env_steps": steps, "rows": rows,
               "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                     capture_output=True, text=True).stdout.strip()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\ncounts {summary['counts']}   {steps} steps")
    print(f"tomato states ({len(tom)}): {tom}")
    print(f"none states ({len(non)}): {non}\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
