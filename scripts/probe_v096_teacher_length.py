#!/usr/bin/env python
"""How long must the atomic teacher run to COMMIT the policy to cream-first?

A 10-action atomic-cream prefix redirected 0/3 tomato-first seeds: the policy
replans at step 10 under the full prompt and reverts. `c=10` binds world-model
credit assignment, not the length of an EXECUTED correction, so the teacher may
run longer and still yield a legitimately executed, verified correction.

This sweeps teacher length on seeds whose pi_0 behaviour is already known, and
reports both directions: redirection of tomato-first seeds, and preservation of
cream-first seeds that already succeed.
"""
from __future__ import annotations

import argparse, glob, json, sys
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

TASK, L = "chain2b_lr2", episode_length("chain2b_lr2")
ATOMIC = "pick up the cream cheese and place it in the basket"
CREAM_PICK = 2
LENGTHS = (10, 40, 80, 150)
OUT = REPO / "results" / "v096_teacher_len"


def seed_all(s):
    torch.manual_seed(s); np.random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def episode(runner, env, seed, subgoals, teacher_steps: int):
    """Run the ATOMIC prompt for `teacher_steps` env steps (stopping early once
    the cream is picked), then hand back to the UNCHANGED full prompt."""
    seed_all(seed); runner.reset()
    obs, _ = env.reset(seed=seed)
    au = GoalAutomaton(subgoals); au.start(env); atoms = goal_atoms(env)
    au.evaluate(env, 0)
    t, done, succ, handoff = 0, False, None, 0
    while not done and t < min(teacher_steps, L):
        obs, _r, tm, tr, inf = env.step(runner.select_action(obs, ATOMIC))
        t += 1; done = bool(tm or tr)
        if succ is None and bool(inf.get("is_success", False)):
            succ = t
        if t % 10 == 0 or done:
            au.evaluate(env, t)
            if CREAM_PICK in au.events_achieved:
                break
    handoff = t
    runner.reset()                      # clean chunk boundary at the handoff
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
    return {"success": succ is not None, "steps": t, "handoff": handoff,
            "order": order,
            "first": ("cream" if order and order[0] == CREAM_PICK
                      else "tomato" if order and order[0] == 0 else "none")}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--controls", type=int, default=5)
    a = ap.parse_args()
    prev = json.load(open(sorted(glob.glob("results/v094_chain2b_corr/2026-*"))[-1]
                          + "/summary.json"))
    tom = [r["seed"] for r in prev["rows"] if r["base"]["first"] == "tomato"]
    none = [r["seed"] for r in prev["rows"] if r["base"]["first"] == "none"]
    ctrl = [r["seed"] for r in prev["rows"]
            if r["base"]["first"] == "cream" and r["base"]["success"]][:a.controls]
    targets = tom + none
    print(f"targets (pi_0 non-cream-first): {targets}\ncontrols (pi_0 cream-first+success): {ctrl}")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / stamp; out.mkdir(parents=True, exist_ok=True)
    from lcwm.chassis import DEFAULT_MODEL, Pi05Runner
    runner = Pi05Runner(model_id=DEFAULT_MODEL, suite_name="libero_10", n_action_steps=10)
    env = make_v080_env(TASK); subgoals = V080_TASKS[TASK]["ordered_subgoals"]

    rows, steps = [], 0
    for n in LENGTHS:
        for kind, seeds in (("target", targets), ("control", ctrl)):
            for s in seeds:
                r = episode(runner, env, s, subgoals, n)
                steps += r["steps"]
                rows.append({"teacher_steps": n, "kind": kind, "seed": s, **r})
                print(f"len={n:3d} {kind:7s} s{s}: first={r['first']:6s} "
                      f"succ={r['success']!s:5s} handoff@{r['handoff']} ({steps} steps)",
                      flush=True)

    summ = {"task": TASK, "utc": stamp, "lengths": list(LENGTHS),
            "targets": targets, "controls": ctrl, "env_steps": steps, "rows": rows}
    for n in LENGTHS:
        t_ = [r for r in rows if r["teacher_steps"] == n and r["kind"] == "target"]
        c_ = [r for r in rows if r["teacher_steps"] == n and r["kind"] == "control"]
        summ[f"len_{n}"] = {
            "targets_redirected": sum(r["first"] == "cream" for r in t_), "n_targets": len(t_),
            "targets_succeeded": sum(r["success"] for r in t_),
            "controls_preserved": sum(r["success"] for r in c_), "n_controls": len(c_)}
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    print(f"\n{'teacher':>8s}{'redirect':>10s}{'target succ':>13s}{'control succ':>14s}")
    for n in LENGTHS:
        d = summ[f"len_{n}"]
        print(f"{n:>8d}{d['targets_redirected']:>6d}/{d['n_targets']:<3d}"
              f"{d['targets_succeeded']:>9d}/{d['n_targets']:<3d}"
              f"{d['controls_preserved']:>10d}/{d['n_controls']:<3d}")
    print(f"{steps} env steps -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
