#!/usr/bin/env python
"""Can anything break the chain3 stall? Oracle ceiling at the decisional stall.

chain3_lr2@750: pi_0 = 1/32, with 30/32 stopping at exactly 4/6 milestones and
then sitting idle a median of 490 steps. The policy has two thirds of the
episode left and will not start the third object. Unlike chain1b - whose
failures needed 30-60 MORE steps - this is a decisional deficit, so a local
action change can in principle flip it.

Two candidate families are tested at the same stall states:

  A. stock full-prompt chunks (reference + 4 max-spread) - the family every V8
     action has used;
  B. ATOMIC-prompt chunks naming only the remaining object. V7.7 measured the
     atomic prompt beating the full instruction mid-chain (3/8 vs 1/8), and the
     project's standing diagnosis is that pi0.5 will not redirect to a later
     object while an earlier one is present.

Family B is a TRAINING-TIME teacher, not a deployment change: framework §6.4
allows privileged mechanisms during acquisition, and any correction it produces
is an executed action sequence to be distilled into the unchanged full-prompt
N=1 policy.

Readout is milestone 4 (`pick_up cream_cheese_1`) - the milestone 30/32
failures never reach.
"""
from __future__ import annotations

import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402
ensure_project_libero_config()

import numpy as np, torch  # noqa: E402
from lcwm.sampler import prefix_forward, sample_chunks  # noqa: E402
from lcwm.seq_data import goal_atoms  # noqa: E402
from lcwm.snapshot import restore, snap  # noqa: E402
from lcwm.task_automaton import GoalAutomaton  # noqa: E402
from lcwm.v080_bench import V080_TASKS, episode_length, make_v080_env  # noqa: E402
from lcwm.v085_noisefloor import select_max_spread  # noqa: E402

TASK, L = "chain3_lr2", episode_length("chain3_lr2")
TAU, C_PREFIX, H = 300, 10, 200          # stall is at ~260; 300 leaves 440 spare
SEEDS = tuple(range(4800, 4840))
N_ANCH, N_REP = 12, 3
ATOMIC = "pick up the cream cheese and place it in the basket"
OUT = REPO / "results" / "v092_chain3_ceiling"


def seed_all(s):
    torch.manual_seed(s); np.random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def rollout(runner, env, snapshot, chunk, key, au_src, milestone, prompt):
    restore(env, snapshot); runner.reset()
    au = au_src.fork()
    obs = env._format_raw_obs(env._env.env._get_observations())
    t, done = TAU, False
    for i in range(C_PREFIX):
        obs, _r, tm, tr, _inf = env.step(chunk[i]); t += 1
        done = bool(tm or tr)
        if done: break
    seed_all(key)
    n = 0
    while not done and n < H and t < L:
        obs, _r, tm, tr, _inf = env.step(runner.select_action(obs, prompt))
        t += 1; n += 1; done = bool(tm or tr)
        if t % 10 == 0 or done:
            au.evaluate(env, t)
    au.evaluate(env, t)
    return int(milestone in au.events_achieved), t - TAU


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--anchors", type=int, default=N_ANCH)
    a = ap.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / stamp; out.mkdir(parents=True, exist_ok=True)
    from lcwm.chassis import DEFAULT_MODEL, Pi05Runner
    runner = Pi05Runner(model_id=DEFAULT_MODEL, suite_name="libero_10", n_action_steps=10)
    env = make_v080_env(TASK)
    subgoals = V080_TASKS[TASK]["ordered_subgoals"]
    TARGET = 4                                   # pick_up cream_cheese_1
    print(f"{TASK} L={L} tau={TAU} H={H}; target milestone {TARGET} = {subgoals[TARGET]}")

    recs, steps = [], 0
    for seed in SEEDS:
        if len(recs) >= a.anchors:
            break
        seed_all(seed); runner.reset()
        obs, _ = env.reset(seed=seed)
        au = GoalAutomaton(subgoals); au.start(env); goal_atoms(env); au.evaluate(env, 0)
        t, done = 0, False
        while not done and t < TAU:
            obs, _r, tm, tr, _inf = env.step(runner.select_action(obs, env.task_description))
            t += 1; done = bool(tm or tr)
            if t % 10 == 0 or done:
                au.evaluate(env, t)
        steps += t
        ev = set(au.events_achieved)
        if done or TARGET in ev or not {0, 1, 2, 3} <= ev:
            print(f"src {seed}: milestones {sorted(ev)} -> not a stall anchor", flush=True)
            continue
        sn = snap(env, TAU, "libero_10", 0, seed=seed)
        au_src = au.fork()

        # family A: stock full-prompt pool
        restore(env, sn); runner.reset()
        o = env._format_raw_obs(env._env.env._get_observations())
        po = runner._obs_to_policy_batch(o, env.task_description)
        pf = prefix_forward(runner.policy, po)
        ref = sample_chunks(runner.policy, po, 1, seed=seed * 3 + 1, prefix=pf)[0, :C_PREFIX].detach().float().cpu()
        raw = sample_chunks(runner.policy, po, 64, seed=seed * 3 + 2, prefix=pf)[:, :C_PREFIX].detach().float().cpu()
        del pf
        refe = runner.chunk_to_env(ref); rawe = [runner.chunk_to_env(raw[i]) for i in range(64)]
        famA = [refe] + [rawe[i] for i in select_max_spread(rawe, refe)]

        # family B: atomic-prompt pool at the same state
        restore(env, sn); runner.reset()
        o2 = env._format_raw_obs(env._env.env._get_observations())
        po2 = runner._obs_to_policy_batch(o2, ATOMIC)
        pf2 = prefix_forward(runner.policy, po2)
        rawB = sample_chunks(runner.policy, po2, 8, seed=seed * 3 + 3, prefix=pf2)[:, :C_PREFIX].detach().float().cpu()
        del pf2
        famB = [runner.chunk_to_env(rawB[i]) for i in range(2)]

        r = {"seed": seed, "tau": TAU, "A": [], "B": [], "A_atomic_cont": []}
        for ci, ch in enumerate(famA):
            hits = 0
            for ri in range(N_REP):
                h, sp = rollout(runner, env, sn, ch, seed * 100 + ri, au_src,
                                TARGET, env.task_description)
                hits += h; steps += sp + C_PREFIX
            r["A"].append(hits / N_REP)
        for ci, ch in enumerate(famB):
            hits = 0
            for ri in range(N_REP):     # atomic prefix, then FULL-prompt continuation
                h, sp = rollout(runner, env, sn, ch, seed * 100 + ri, au_src,
                                TARGET, env.task_description)
                hits += h; steps += sp + C_PREFIX
            r["B"].append(hits / N_REP)
        # family B': atomic prefix AND atomic continuation (upper bound)
        hits = 0
        for ri in range(N_REP):
            h, sp = rollout(runner, env, sn, famB[0], seed * 100 + ri, au_src, TARGET, ATOMIC)
            hits += h; steps += sp + C_PREFIX
        r["B_atomic_cont"] = hits / N_REP
        r["ref"] = r["A"][0]; r["bestA"] = max(r["A"][1:]); r["bestB"] = max(r["B"])
        recs.append(r)
        print(f"s{seed}: ref={r['ref']:.2f} bestA(stock)={r['bestA']:.2f} "
              f"bestB(atomic prefix)={r['bestB']:.2f} B'(atomic cont)={r['B_atomic_cont']:.2f} "
              f"({steps} steps)", flush=True)

    n = len(recs)
    summ = {"task": TASK, "utc": stamp, "anchors": n, "tau": TAU, "H": H,
            "target_milestone": subgoals[TARGET], "env_steps": steps,
            "ref_hit_rate": sum(r["ref"] for r in recs) / max(n, 1),
            "stockA_flip": sum(r["bestA"] > r["ref"] for r in recs),
            "atomicB_flip": sum(r["bestB"] > r["ref"] for r in recs),
            "atomic_cont_flip": sum(r["B_atomic_cont"] > r["ref"] for r in recs),
            "records": recs}
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    print(f"\n=== chain3 stall ceiling, {n} anchors, target {subgoals[TARGET]} ===")
    print(f"  reference reaches it            : {summ['ref_hit_rate']:.2f} mean rate")
    print(f"  a stock alternative beats ref   : {summ['stockA_flip']}/{n}")
    print(f"  an ATOMIC-prefix cand beats ref : {summ['atomicB_flip']}/{n}")
    print(f"  atomic prefix + atomic cont     : {summ['atomic_cont_flip']}/{n}")
    print(f"  {steps} env steps -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
