#!/usr/bin/env python
"""Branch at deployment: execute every candidate, label each one by its outcome.

    python scripts/collect_v136_branch_labels.py --task chain1b_lr2 --seeds 64 --n 8

GOAL ANCHOR (CLAUDE.md). data axis: still trajectories collected while deploying
the frozen policy - branching is how a deployed agent finds out which of its
options was the right one. object/target axes unchanged: T_theta stays the
predictor and the latent stays the target.

WHY THIS IS NEEDED, from measurement rather than preference. D_theta currently
learns from episode-level success carried back to every triple. That teaches it
where a state sits in the task (Delta w AUC 0.997, p_succ 0.645) but nothing about
which of several near-identical actions at ONE state is better: v133 and v135 both
put within-state ranking AUC at ~0.5 across 28 head/depth/sigma configurations.
Meanwhile v132 showed the leverage is real - an oracle takes 0.469 -> 0.688 at
sigma=0 and 0.562 -> 0.875 at sigma=3.

Ranking needs candidate-level labels. This produces them: at a branch point, draw
n candidates, commit to each in turn via deterministic replay, run the frozen
policy to the horizon, and record whether that branch succeeded. Stores the branch
state (pooled hidden + proprioception) and every candidate with its outcome.
"""
from __future__ import annotations

import argparse, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402
ensure_project_libero_config()

import numpy as np, torch  # noqa: E402
from lcwm.sampler import prefix_forward, sample_chunks  # noqa: E402
from lcwm.seq_data import goal_atoms, predicate_bits  # noqa: E402
from lcwm.v080_bench import episode_length, make_v080_env  # noqa: E402
from lcwm.v082_m0 import masked_prefix_mean  # noqa: E402
from collect_v121_deploy_latents import proprio  # noqa: E402

C = 10
OUT = REPO / "results" / "v136_branch"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="chain1b_lr2")
    ap.add_argument("--seeds", type=int, default=64)
    ap.add_argument("--seed-start", type=int, default=6700)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 3.0])
    ap.add_argument("--branch-at", type=int, nargs="+", default=[0])
    a = ap.parse_args()
    L = episode_length(a.task)
    seeds = list(range(a.seed_start, a.seed_start + a.seeds))
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / f"{a.task}_{stamp}"; out.mkdir(parents=True, exist_ok=True)
    print(f"{len(seeds)} seeds x {len(a.sigmas)} sigmas x {len(a.branch_at)} branch "
          f"points x n={a.n} = {len(seeds)*len(a.sigmas)*len(a.branch_at)*a.n} rollouts")

    from lcwm.chassis import DEFAULT_MODEL, Pi05Runner
    runner = Pi05Runner(model_id=DEFAULT_MODEL, suite_name="libero_10", n_action_steps=C)
    env = make_v080_env(a.task)

    ZH, ZP, U, Y, GRP, META = [], [], [], [], [], []
    steps, grp = 0, 0
    for seed in seeds:
        for branch in a.branch_at:
            for sigma in a.sigmas:
                torch.manual_seed(seed); np.random.seed(seed)
                runner.reset(); obs, _ = env.reset(seed=int(seed))
                t = 0
                while t < branch:
                    obs, _r, tm, tr, _i = env.step(
                        runner.select_action(obs, env.task_description))
                    t += 1
                    if tm or tr: break
                steps += t
                po = runner._obs_to_policy_batch(obs, env.task_description)
                with torch.no_grad():
                    pf = prefix_forward(runner.policy, po)
                    cand = sample_chunks(runner.policy, po, a.n,
                                         seed=int(seed) * 7919 + branch,
                                         prefix=pf, sigma=sigma
                                         )[:, :C].detach().float().cpu()
                    zh = masked_prefix_mean(pf.hidden[0].detach().float().cpu(),
                                            pf.pad_masks[0].detach().cpu())
                del pf
                zp = proprio(obs)
                outs = []
                for i in range(a.n):
                    torch.manual_seed(seed); np.random.seed(seed)
                    runner.reset(); o2, _ = env.reset(seed=int(seed))
                    atoms = goal_atoms(env)
                    t2, done, succ = 0, False, None
                    while t2 < branch and not done:      # deterministic replay
                        o2, _r, tm, tr, inf = env.step(
                            runner.select_action(o2, env.task_description))
                        t2 += 1; done = bool(tm or tr)
                        if succ is None and bool(inf.get("is_success", False)):
                            succ = t2
                    for act in runner.chunk_to_env(cand[i]):
                        if done: break
                        o2, _r, tm, tr, inf = env.step(act); t2 += 1
                        done = bool(tm or tr)
                        if succ is None and bool(inf.get("is_success", False)):
                            succ = t2
                    runner.reset()
                    while not done and t2 < L:
                        o2, _r, tm, tr, inf = env.step(
                            runner.select_action(o2, env.task_description))
                        t2 += 1; done = bool(tm or tr)
                        if succ is None and bool(inf.get("is_success", False)):
                            succ = t2
                        if (t2 % 10 == 0 or done) and succ is None and \
                                predicate_bits(env, atoms).all():
                            succ = t2
                    steps += t2
                    outs.append(int(succ is not None))
                for i in range(a.n):
                    ZH.append(zh); ZP.append(zp); U.append(cand[i])
                    Y.append(float(outs[i])); GRP.append(grp)
                META.append({"seed": seed, "branch": branch, "sigma": sigma,
                             "group": grp, "outcomes": outs, "successes": sum(outs)})
                grp += 1
                print(f"s{seed} b{branch} sig{sigma}: {sum(outs)}/{a.n}  "
                      f"({steps} steps)", flush=True)

    torch.save({"zh": torch.stack(ZH), "zp": torch.stack(ZP), "u": torch.stack(U),
                "y": torch.tensor(Y), "group": torch.tensor(GRP),
                "task": a.task, "c": C, "n": a.n}, out / "branches.pt")
    mixed = sum(1 for m in META if 0 < m["successes"] < a.n)
    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "task": a.task, "seeds": len(seeds), "n": a.n,
         "sigmas": a.sigmas, "branch_at": a.branch_at, "groups": grp,
         "labelled_candidates": len(Y), "mixed_groups": mixed,
         "mean_rate": float(np.mean(Y)), "env_steps": steps, "meta": META,
         "note": "candidate-level outcome labels via deterministic replay; needed "
                 "because episode-level labels give within-state ranking AUC ~0.5",
         "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()},
        indent=2))
    print(f"\n{len(Y)} labelled candidates over {grp} branch groups "
          f"({mixed} mixed); {steps} steps -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
