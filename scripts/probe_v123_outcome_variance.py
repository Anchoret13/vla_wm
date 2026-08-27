#!/usr/bin/env python
"""Is there outcome variance to act on at chain1b's decision points?

    python scripts/probe_v123_outcome_variance.py --task chain1b_lr2 --seeds 16 --rollouts 8

GOAL ANCHOR (CLAUDE.md). Axis check: task = chain1b_lr2 at 0.438 (low success);
data = deployment rollouts; this is a PREREQUISITE measurement for how T_theta's
p_succ head can be used, not a world model itself.

On chain2b, selection was refuted because the policy is unimodal at the decisive
states: p_cream = 0 over 352 rollouts under both ODE and SDE sampling. A world
model that scores candidate actions is useless when every candidate leads to the
same place. That result is task-specific and must NOT be assumed here.

This measures, per initial state, the spread of OUTCOMES across independent
sampling noise: within-seed success variance. If a state's rollouts are all
success or all failure, there is nothing for a p_succ head to select between at
that state, and T_theta must be used some other way (e.g. predicting failure
early enough to trigger a different behaviour). If outcomes vary within a seed,
selection is live.

Reports the within-seed variance decomposition, which is the quantity that
decides it - not the marginal success rate, which can look healthy while every
individual state is deterministic.
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
from lcwm.seq_data import goal_atoms, predicate_bits  # noqa: E402
from lcwm.v080_bench import episode_length, make_v080_env  # noqa: E402

OUT = REPO / "results" / "v123_outcome_variance"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="chain1b_lr2")
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--rollouts", type=int, default=8)
    ap.add_argument("--seed-start", type=int, default=6200)
    a = ap.parse_args()
    L = episode_length(a.task)
    seeds = list(range(a.seed_start, a.seed_start + a.seeds))
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / f"{a.task}_{stamp}"; out.mkdir(parents=True, exist_ok=True)

    from lcwm.chassis import DEFAULT_MODEL, Pi05Runner
    runner = Pi05Runner(model_id=DEFAULT_MODEL, suite_name="libero_10", n_action_steps=10)
    env = make_v080_env(a.task)

    rows, steps = [], 0
    for seed in seeds:
        outs = []
        for r in range(a.rollouts):
            torch.manual_seed(seed * 1000 + r); np.random.seed(seed * 1000 + r)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed * 1000 + r)
            runner.reset(); obs, _ = env.reset(seed=int(seed))
            atoms = goal_atoms(env)
            t, done, succ = 0, False, None
            while not done and t < L:
                obs, _r, tm, tr, inf = env.step(
                    runner.select_action(obs, env.task_description))
                t += 1; done = bool(tm or tr)
                if succ is None and bool(inf.get("is_success", False)):
                    succ = t
                if (t % 10 == 0 or done) and succ is None and predicate_bits(env, atoms).all():
                    succ = t
            steps += t
            outs.append(int(succ is not None))
        p = float(np.mean(outs))
        rows.append({"seed": seed, "successes": int(sum(outs)),
                     "rollouts": a.rollouts, "p": p, "outcomes": outs})
        print(f"s{seed}: {sum(outs)}/{a.rollouts} = {p:.2f}  ({steps} steps)", flush=True)

    ps = np.array([r["p"] for r in rows])
    pbar = float(ps.mean())
    # decomposition: how much of the outcome variance is WITHIN a state
    within = float(np.mean(ps * (1 - ps)))          # mean Bernoulli var per state
    between = float(ps.var())                        # spread of state difficulties
    total = within + between
    det = [r["seed"] for r in rows if r["p"] in (0.0, 1.0)]
    summary = {"utc": stamp, "task": a.task, "seeds": len(seeds),
               "rollouts": a.rollouts, "marginal_rate": pbar,
               "within_state_var": within, "between_state_var": between,
               "within_fraction": within / total if total else 0.0,
               "deterministic_states": det, "n_deterministic": len(det),
               "env_steps": steps, "rows": rows,
               "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                     capture_output=True, text=True).stdout.strip()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nmarginal success {pbar:.3f}")
    print(f"within-state var {within:.4f} | between-state var {between:.4f} | "
          f"within fraction {summary['within_fraction']:.3f}")
    print(f"fully deterministic states: {len(det)}/{len(rows)} {det}")
    print("selection is LIVE" if within > 0.02 else
          "selection is BLOCKED: outcomes are state-determined, as on chain2b")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
