#!/usr/bin/env python
"""Oracle ceiling: does choosing among candidates at ONE boundary change anything?

    python scripts/probe_v132_selection_ceiling.py --task chain1b_lr2 --seeds 16 --n 8

GOAL ANCHOR (CLAUDE.md). Prerequisite for whether SELECTION is a usable role for
the world model at all - measured before tuning the model further.

v126/v129/v131 all found the wm arm tied its matched random arm. Before concluding
anything about T_theta or D_theta, measure the ceiling: execute each of the n
candidates from the SAME state and see whether the outcomes differ at all. Episode
resets are deterministic given the seed, so replaying to the branch point and then
committing to candidate i is a valid counterfactual.

  all n outcomes identical  ->  zero leverage at that boundary; no scorer can help,
                                and the wm nulls say nothing about the world model
  outcomes differ           ->  leverage exists; the nulls are about the scorer

This is the chain1b analogue of the chain2b p_cream probe. v123 measured
within-state variance across WHOLE episodes (0.517), which accumulates over ~25
decision points and does NOT imply per-boundary leverage.
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
from lcwm.seq_data import goal_atoms, predicate_bits  # noqa: E402
from lcwm.v080_bench import episode_length, make_v080_env  # noqa: E402

C = 10
OUT = REPO / "results" / "v132_ceiling"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="chain1b_lr2")
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--seed-start", type=int, default=6500)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.0)
    ap.add_argument("--branch-at", type=int, default=0)
    a = ap.parse_args()
    L = episode_length(a.task)
    seeds = list(range(a.seed_start, a.seed_start + a.seeds))
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / f"{a.task}_s{a.sigma}_b{a.branch_at}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)

    from lcwm.chassis import DEFAULT_MODEL, Pi05Runner
    runner = Pi05Runner(model_id=DEFAULT_MODEL, suite_name="libero_10", n_action_steps=C)
    env = make_v080_env(a.task)

    rows, steps = [], 0
    for seed in seeds:
        # candidates drawn once at the branch point, then each committed to
        torch.manual_seed(seed); np.random.seed(seed)
        runner.reset(); obs, _ = env.reset(seed=int(seed))
        t = 0
        while t < a.branch_at:
            obs, _r, tm, tr, _i = env.step(runner.select_action(obs, env.task_description))
            t += 1
            if tm or tr: break
        steps += t
        po = runner._obs_to_policy_batch(obs, env.task_description)
        with torch.no_grad():
            pf = prefix_forward(runner.policy, po)
            cand = sample_chunks(runner.policy, po, a.n, seed=int(seed) * 7919 + a.branch_at,
                                 prefix=pf, sigma=a.sigma)[:, :C].detach().float().cpu()
        del pf

        outs = []
        for i in range(a.n):
            torch.manual_seed(seed); np.random.seed(seed)
            runner.reset(); obs2, _ = env.reset(seed=int(seed))
            atoms = goal_atoms(env)
            t2, done, succ = 0, False, None
            while t2 < a.branch_at and not done:      # deterministic replay
                obs2, _r, tm, tr, inf = env.step(
                    runner.select_action(obs2, env.task_description))
                t2 += 1; done = bool(tm or tr)
                if succ is None and bool(inf.get("is_success", False)):
                    succ = t2
            for act in runner.chunk_to_env(cand[i]):  # commit to candidate i
                if done: break
                obs2, _r, tm, tr, inf = env.step(act); t2 += 1
                done = bool(tm or tr)
                if succ is None and bool(inf.get("is_success", False)):
                    succ = t2
            runner.reset()
            while not done and t2 < L:                # then the stock policy
                obs2, _r, tm, tr, inf = env.step(
                    runner.select_action(obs2, env.task_description))
                t2 += 1; done = bool(tm or tr)
                if succ is None and bool(inf.get("is_success", False)):
                    succ = t2
                if (t2 % 10 == 0 or done) and succ is None and predicate_bits(env, atoms).all():
                    succ = t2
            steps += t2
            outs.append(int(succ is not None))
        k = sum(outs)
        rows.append({"seed": seed, "outcomes": outs, "successes": k,
                     "n": a.n, "p": k / a.n})
        print(f"s{seed}: {k}/{a.n} candidates succeed  {outs}  ({steps} steps)",
              flush=True)

    ps = np.array([r["p"] for r in rows])
    mixed = [r["seed"] for r in rows if 0 < r["successes"] < a.n]
    # ceiling: an oracle picks a succeeding candidate whenever one exists
    oracle = float(np.mean([1.0 if r["successes"] > 0 else 0.0 for r in rows]))
    randm = float(ps.mean())
    summary = {"utc": stamp, "task": a.task, "seeds": len(seeds), "n": a.n,
               "sigma": a.sigma, "branch_at": a.branch_at,
               "random_pick_rate": randm, "oracle_pick_rate": oracle,
               "oracle_gain": oracle - randm,
               "mixed_states": mixed, "n_mixed": len(mixed),
               "env_steps": steps, "rows": rows,
               "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                     capture_output=True, text=True).stdout.strip()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nrandom-pick {randm:.3f}   oracle-pick {oracle:.3f}   "
          f"ceiling gain {oracle-randm:+.3f}")
    print(f"states where candidates DISAGREE: {len(mixed)}/{len(rows)} {mixed}")
    print("selection has leverage at this boundary" if len(mixed) else
          "ZERO leverage: every candidate leads to the same outcome")
    print(f"{steps} steps -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
