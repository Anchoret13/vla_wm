#!/usr/bin/env python
"""Is there cream-directed mass to select at the states where pi_0 goes tomato?

    python scripts/probe_v100_diversity.py --rollouts 12 --horizon 120

Deployment-time SELECTION can only help if the policy's own stochasticity
sometimes produces the good mode. v099 showed n=8 chunk-level selection left the
tomato seeds bit-identical (8/8 overlap with pi_0), while a weight update moved
3/8 - consistent with the candidates at those states being all-tomato.

This probe measures that directly. At each decisive state it draws `--rollouts`
INDEPENDENT 40-step rollouts (4 chunks resampled autoregressively, distinct
noise per rollout), continues under the stock policy to `--horizon`, and records
which object is picked first. 40 steps because the v096 sweep measured that as
the commitment threshold; a 10-step perturbation cannot redirect (0/4).

p_cream = 0 at a state means no selector can ever fix it - the mode is absent
from the policy's support, not merely unlikely.
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
from lcwm.sampler import prefix_forward, sample_chunks  # noqa: E402
from lcwm.task_automaton import GoalAutomaton  # noqa: E402
from lcwm.v080_bench import V080_TASKS, make_v080_env  # noqa: E402

TASK, C, WINDOW = "chain2b_lr2", 10, 4
TOMATO = (3207, 3225, 3228, 3231, 3234, 3239, 3252, 3257)   # pi_0 tomato-first
CREAM = (3200, 3201, 3202, 3204)                            # cream-first controls
OUT = REPO / "results" / "v100_diversity"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=120)
    a = ap.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / stamp; out.mkdir(parents=True, exist_ok=True)

    from lcwm.chassis import DEFAULT_MODEL, Pi05Runner
    runner = Pi05Runner(model_id=DEFAULT_MODEL, suite_name="libero_10", n_action_steps=10)
    env = make_v080_env(TASK); subgoals = V080_TASKS[TASK]["ordered_subgoals"]

    rows, steps = [], 0
    for group, seeds in (("tomato", TOMATO), ("cream", CREAM)):
        for seed in seeds:
            firsts = []
            for r in range(a.rollouts):
                torch.manual_seed(seed * 1000 + r); np.random.seed(seed * 1000 + r)
                runner.reset(); obs, _ = env.reset(seed=int(seed))
                au = GoalAutomaton(subgoals); au.start(env); au.evaluate(env, 0)
                t, done = 0, False
                for b in range(WINDOW):          # 40 steps of INDEPENDENT sampling
                    if done: break
                    po = runner._obs_to_policy_batch(obs, env.task_description)
                    with torch.no_grad():
                        pf = prefix_forward(runner.policy, po)
                        ch = sample_chunks(runner.policy, po, 1,
                                           seed=seed * 7919 + r * 131 + b,
                                           prefix=pf)[0, :C].detach().float().cpu()
                    del pf
                    ech = runner.chunk_to_env(ch)
                    for i in range(C):
                        obs, _r2, tm, tr, _inf = env.step(ech[i]); t += 1
                        done = bool(tm or tr)
                        if done: break
                    au.evaluate(env, t)
                runner.reset()
                while not done and t < a.horizon:   # then stock policy
                    obs, _r2, tm, tr, _inf = env.step(
                        runner.select_action(obs, env.task_description))
                    t += 1; done = bool(tm or tr)
                    if t % 10 == 0 or done: au.evaluate(env, t)
                steps += t
                ev = {int(k): int(v) for k, v in au.events_achieved.items()}
                order = [i for i, _ in sorted(ev.items(), key=lambda kv: kv[1])]
                firsts.append("cream" if order and order[0] == 2 else
                              "tomato" if order and order[0] == 0 else "none")
            nc = sum(1 for f in firsts if f == "cream")
            rows.append({"seed": seed, "group": group, "rollouts": a.rollouts,
                         "p_cream": nc / a.rollouts, "firsts": firsts})
            print(f"{group:6s} s{seed}: p_cream={nc}/{a.rollouts}={nc/a.rollouts:.2f} "
                  f"({steps} steps)", flush=True)

    tom = [r for r in rows if r["group"] == "tomato"]
    dead = [r["seed"] for r in tom if r["p_cream"] == 0.0]
    summary = {"utc": stamp, "task": TASK, "rollouts": a.rollouts,
               "horizon": a.horizon, "window": WINDOW * C,
               "mean_p_cream_tomato_states": sum(r["p_cream"] for r in tom) / len(tom),
               "states_with_zero_cream_mass": dead,
               "n_zero": len(dead), "n_tomato_states": len(tom),
               "env_steps": steps, "rows": rows,
               "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                     capture_output=True, text=True).stdout.strip()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nmean p_cream at tomato states = {summary['mean_p_cream_tomato_states']:.3f}")
    print(f"states with ZERO cream mass: {len(dead)}/{len(tom)} {dead}")
    print(f"{steps} steps -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
