#!/usr/bin/env python
"""The definitive ceiling: from an IDENTICAL state, does the action change what follows?

    python scripts/probe_v229_branch_ceiling.py --task chain3_lr2 --seeds 24

WHY THIS IS THE ONE THAT COUNTS. v226/v228 asked whether the action difference
predicts the outcome difference between transitions whose states are NEAR
NEIGHBOURS, and found chance on chain1b (c = 10 and c = 2) and on chain3. But
"near neighbour" in 2073-d is coarse - median matched distance 20.7 against a
random pair's 68 - so residual state differences are noise that could mask a real
action signal. That is the one weakness those probes have, and it is removable:
LIBERO is resettable, so branches can start from the SAME state exactly.

  replay deterministically to step b, then execute N DIFFERENT candidate chunks
  from that identical state, and record the progress each one reaches over a
  short horizon afterwards.

Two questions, in order:
  (a) does the action change the outcome AT ALL - is there within-group variance?
      If every candidate from an identical state gives the same progress, nothing
      can be predicted and the ceiling is zero for a structural reason.
  (b) if it does, is the change PREDICTABLE from the action?

Progress is the count of BDDL events achieved - privileged, which is what a ceiling
probe is for. It bounds what any method could reach; it never enters one.

This is the measurement that decides whether the mission is reachable on the target
task, and it should have run before any world model was built.
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
OUT = REPO / "results" / "v229_branch_ceiling"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="chain3_lr2")
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--seed-start", type=int, default=8300)
    ap.add_argument("--n", type=int, default=8, help="candidate chunks per state")
    ap.add_argument("--sigma", type=float, default=1.5,
                    help="SDE noise; pi0.5 under the ODE emits near-identical "
                         "candidates (15/16 bit-identical was measured), so a "
                         "diversity-free draw would answer (a) trivially and "
                         "wrongly")
    ap.add_argument("--branch-at", type=int, nargs="+", default=[50, 150, 250])
    ap.add_argument("--horizon", type=int, default=5, help="chunks after the branch")
    a = ap.parse_args()
    L = episode_length(a.task)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / f"{a.task}_{stamp}"; out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(a.seed_start, a.seed_start + a.seeds))
    print(f"{len(seeds)} seeds x {len(a.branch_at)} branch points x n={a.n} "
          f"= {len(seeds)*len(a.branch_at)*a.n} branches, sigma={a.sigma}")

    from lcwm.chassis import DEFAULT_MODEL, Pi05Runner
    runner = Pi05Runner(model_id=DEFAULT_MODEL, suite_name="libero_10", n_action_steps=C)
    env = make_v080_env(a.task)

    groups, steps = [], 0
    for seed in seeds:
        for branch in a.branch_at:
            torch.manual_seed(seed); np.random.seed(seed)
            runner.reset(); obs, _ = env.reset(seed=int(seed))
            atoms = goal_atoms(env)
            t, done = 0, False
            while t < branch and not done:
                obs, _r, tm, tr, _i = env.step(
                    runner.select_action(obs, env.task_description))
                t += 1; done = bool(tm or tr)
            if done:
                continue
            steps += t
            p0 = int(predicate_bits(env, atoms).sum())
            po = runner._obs_to_policy_batch(obs, env.task_description)
            with torch.no_grad():
                pf = prefix_forward(runner.policy, po)
                cand = sample_chunks(runner.policy, po, a.n,
                                     seed=int(seed) * 7919 + branch,
                                     prefix=pf, sigma=a.sigma)[:, :C].detach().float().cpu()
                zh = masked_prefix_mean(pf.hidden[0].detach().float().cpu(),
                                        pf.pad_masks[0].detach().cpu())
            del pf
            z0 = torch.cat([zh, proprio(obs)])
            outs, acts = [], []
            for i in range(a.n):
                torch.manual_seed(seed); np.random.seed(seed)
                runner.reset(); o2, _ = env.reset(seed=int(seed))
                t2, d2 = 0, False
                while t2 < branch and not d2:          # deterministic replay
                    o2, _r, tm, tr, _i = env.step(
                        runner.select_action(o2, env.task_description))
                    t2 += 1; d2 = bool(tm or tr)
                for act in runner.chunk_to_env(cand[i]):   # THE branching action
                    if d2: break
                    o2, _r, tm, tr, _i = env.step(act); t2 += 1; d2 = bool(tm or tr)
                runner.reset()
                h = 0
                while not d2 and h < a.horizon * C and t2 < L:   # short horizon
                    o2, _r, tm, tr, _i = env.step(
                        runner.select_action(o2, env.task_description))
                    t2 += 1; h += 1; d2 = bool(tm or tr)
                steps += t2
                outs.append(int(predicate_bits(env, atoms).sum()) - p0)
                acts.append(cand[i])
            groups.append({"seed": seed, "branch": branch, "p0": p0,
                           "progress": outs, "z0": z0, "actions": torch.stack(acts)})
            sp = float(np.std(outs))
            print(f"s{seed} b{branch}: p0={p0} progress {outs} sd={sp:.3f} "
                  f"({steps} steps)", flush=True)

    # ---- (a) does the action change the outcome at all? ---------------------
    W = np.array([np.std(g["progress"]) for g in groups])
    M = np.array([np.mean(g["progress"]) for g in groups])
    print(f"\n(a) WITHIN-STATE spread over {a.n} candidate actions: "
          f"mean sd {W.mean():.4f}, and {100*(W>0).mean():.0f}% of states have any")
    print(f"    BETWEEN-STATE spread of the group means: sd {M.std():.4f}")
    print(f"    ratio within/between {W.mean()/(M.std()+1e-9):.3f}")

    # ---- (b) is it predictable from the action? -----------------------------
    live = [g for g in groups if np.std(g["progress"]) > 0]
    print(f"\n(b) {len(live)} of {len(groups)} states have any within-state variance")
    r = float("nan")
    if len(live) >= 8:
        X, Y = [], []
        for g in live:                        # centre within group: state removed
            A = g["actions"].flatten(1).numpy()
            y = np.array(g["progress"], dtype=float)
            X.append(A - A.mean(0)); Y.append(y - y.mean())
        X = np.concatenate(X); Y = np.concatenate(Y)
        gsp = np.random.default_rng(229).permutation(len(X))
        tr, te = gsp[:int(0.8 * len(X))], gsp[int(0.8 * len(X)):]
        import torch.nn as nn
        def fit(Yt, name):
            torch.manual_seed(0)
            h_ = nn.Sequential(nn.LayerNorm(X.shape[1]), nn.Linear(X.shape[1], 128),
                               nn.GELU(), nn.Linear(128, 1))
            o_ = torch.optim.AdamW(h_.parameters(), lr=1e-3, weight_decay=1e-2)
            Xt = torch.tensor(X, dtype=torch.float32); Yv = torch.tensor(Yt, dtype=torch.float32)
            gg = torch.Generator().manual_seed(1)
            for it in range(1500):
                i = torch.tensor(tr)[torch.randint(0, len(tr), (128,), generator=gg)]
                ((h_(Xt[i]).squeeze(-1) - Yv[i]) ** 2).mean().backward()
                o_.step(); o_.zero_grad()
            with torch.no_grad():
                p_ = h_(Xt[torch.tensor(te)]).squeeze(-1).numpy()
            yv = Yt[te]
            rr = float(np.corrcoef(p_, yv)[0, 1]) if p_.std() > 1e-9 and yv.std() > 1e-9 else 0.0
            print(f"    {name:32s} held-out corr {rr:+.3f}")
            return rr
        r = fit(Y, "action -> progress, state exact")
        rs = fit(np.random.default_rng(3).permutation(Y), "CONTROL: labels shuffled")
        print(f"\n    signal above the shuffled control: {r - rs:+.3f}")

    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "task": a.task, "seeds": seeds, "n": a.n, "sigma": a.sigma,
         "branch_at": a.branch_at, "horizon": a.horizon, "env_steps": steps,
         "within_sd_mean": float(W.mean()) if len(W) else None,
         "between_sd": float(M.std()) if len(M) else None,
         "states_with_variance": len(live), "states": len(groups),
         "groups": [{k: v for k, v in g.items() if k not in ("z0", "actions")}
                    for g in groups],
         "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()},
        indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
