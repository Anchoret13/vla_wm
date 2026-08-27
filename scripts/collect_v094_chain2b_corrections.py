#!/usr/bin/env python
"""Verified corrections on chain2b: redirect the first-object choice.

chain2b@500 pools to cream-first 46/50 = 0.920 and tomato-first 0/10 = 0.000
across two independent panels. The failure is a DECISION - which object to
approach first - and the correct behaviour is already inside policy support,
since pi0.5 chooses cream-first unprompted ~80% of the time.

The counterfactual action at the anchor is therefore a first-10 chunk that
commits to the cream cheese. It is generated with an ATOMIC prompt naming only
that object, which framework §6.4 permits as a training-time mechanism; the
prefix is then EXECUTED and kept only if the episode succeeds under the
UNCHANGED full-prompt continuation. Nothing unexecuted becomes a target.

Acquisition seeds are disjoint from the behavior panel: corrections never come
from seeds the policy is scored on.
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
from lcwm.seq_data import goal_atoms, predicate_bits  # noqa: E402
from lcwm.task_automaton import GoalAutomaton  # noqa: E402
from lcwm.v080_bench import V080_TASKS, episode_length, make_v080_env  # noqa: E402

TASK = "chain2b_lr2"
L = episode_length(TASK)
C_PREFIX = 10
ATOMIC = "pick up the cream cheese and place it in the basket"
ACQ_SEEDS = tuple(range(4900, 4960))          # disjoint from panel 3200-3399
PANEL_BLOCK = set(range(3200, 3400))
assert not (set(ACQ_SEEDS) & PANEL_BLOCK)
OUT = REPO / "results" / "v094_chain2b_corr"
CREAM_FIRST, TOMATO_FIRST = 2, 0              # milestone indices


def seed_all(s):
    torch.manual_seed(s); np.random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def rollout(runner, env, seed, subgoals, prefix_actions=None):
    """One episode. If `prefix_actions` is given it is executed first, then the
    UNCHANGED full-prompt policy continues to the deadline."""
    seed_all(seed); runner.reset()
    obs, _ = env.reset(seed=seed)
    au = GoalAutomaton(subgoals); au.start(env); atoms = goal_atoms(env)
    au.evaluate(env, 0)
    t, done, succ = 0, False, None
    if prefix_actions is not None:
        for i in range(len(prefix_actions)):
            obs, _r, tm, tr, inf = env.step(prefix_actions[i]); t += 1
            done = bool(tm or tr)
            if succ is None and bool(inf.get("is_success", False)):
                succ = t
            if done: break
        au.evaluate(env, t)
    while not done and t < L:
        obs, _r, tm, tr, inf = env.step(runner.select_action(obs, env.task_description))
        t += 1; done = bool(tm or tr)
        if succ is None and bool(inf.get("is_success", False)):
            succ = t
        if t % 10 == 0 or done:
            au.evaluate(env, t)
            if succ is None and predicate_bits(env, atoms).all():
                succ = t
    ev = {int(k): int(v) for k, v in au.events_achieved.items()}
    order = [i for i, _ in sorted(ev.items(), key=lambda kv: kv[1])]
    return {"success": succ is not None, "success_step": succ, "steps": t,
            "events": ev, "order": order,
            "first": ("cream" if order and order[0] == CREAM_FIRST
                      else "tomato" if order and order[0] == TOMATO_FIRST else "none")}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=40)
    a = ap.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / stamp; out.mkdir(parents=True, exist_ok=True)
    from lcwm.chassis import DEFAULT_MODEL, Pi05Runner
    runner = Pi05Runner(model_id=DEFAULT_MODEL, suite_name="libero_10", n_action_steps=10)
    env = make_v080_env(TASK)
    subgoals = V080_TASKS[TASK]["ordered_subgoals"]
    print(f"{TASK}@{L}; atomic teacher: {ATOMIC!r}")

    rows, steps = [], 0
    for seed in ACQ_SEEDS[:a.seeds]:
        # 1. stock reference episode - establishes what pi_0 does on this seed
        base = rollout(runner, env, seed, subgoals)
        steps += base["steps"]

        # 2. counterfactual: atomic-cream chunk at t=0, then full-prompt continuation
        seed_all(seed); runner.reset()
        obs, _ = env.reset(seed=seed)
        po = runner._obs_to_policy_batch(obs, ATOMIC)
        pf = prefix_forward(runner.policy, po)
        chunk = sample_chunks(runner.policy, po, 1, seed=seed * 11 + 3,
                              prefix=pf)[0, :C_PREFIX].detach().float().cpu()
        del pf
        env_actions = runner.chunk_to_env(chunk)
        alt = rollout(runner, env, seed, subgoals, prefix_actions=env_actions)
        steps += alt["steps"]

        verified = alt["success"] and not base["success"]
        rows.append({"seed": seed, "base": base, "alt": alt, "verified": verified,
                     "chunk_norm": chunk.tolist()})
        print(f"s{seed}: pi_0 first={base['first']:6s} succ={base['success']!s:5s} | "
              f"atomic-cream first={alt['first']:6s} succ={alt['success']!s:5s} "
              f"| VERIFIED={verified} ({steps} steps)", flush=True)

    nb = sum(r["base"]["success"] for r in rows)
    na = sum(r["alt"]["success"] for r in rows)
    ver = [r for r in rows if r["verified"]]
    lost = [r for r in rows if r["base"]["success"] and not r["alt"]["success"]]
    tom = [r for r in rows if r["base"]["first"] == "tomato"]
    tom_fixed = [r for r in tom if r["alt"]["first"] == "cream"]
    summary = {"task": TASK, "L": L, "utc": stamp, "n": len(rows),
               "atomic_prompt": ATOMIC, "acq_seeds": [ACQ_SEEDS[0], ACQ_SEEDS[a.seeds - 1]],
               "pi0_success": nb, "atomic_success": na,
               "verified_corrections": len(ver), "regressions": len(lost),
               "pi0_tomato_first": len(tom), "tomato_redirected_to_cream": len(tom_fixed),
               "env_steps": steps,
               "corrections": [{"seed": r["seed"], "chunk_norm": r["chunk_norm"]} for r in ver],
               "rows": [{k: v for k, v in r.items() if k != "chunk_norm"} for r in rows]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    torch.save({"corrections": [(r["seed"], torch.tensor(r["chunk_norm"])) for r in ver],
                "task": TASK, "atomic_prompt": ATOMIC},
               out / "corrections.pt")
    print(f"\n=== chain2b corrections, {len(rows)} acquisition seeds ===")
    print(f"  pi_0 success            : {nb}/{len(rows)}")
    print(f"  atomic-cream prefix     : {na}/{len(rows)}")
    print(f"  VERIFIED corrections    : {len(ver)}  (alt succeeds where pi_0 failed)")
    print(f"  regressions             : {len(lost)}  (pi_0 succeeded, alt failed)")
    print(f"  pi_0 tomato-first       : {len(tom)}, redirected to cream: {len(tom_fixed)}")
    print(f"  {steps} env steps -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
