#!/usr/bin/env python
"""V6.8 — frozen failure inventory from STOCK development rollouts.

Input: iteration-1 development STOCK-arm traces (25 episodes, seeds
1600–1640). Stock π0.5 diagnostics only — no repaired-method record is
read, so no method selection occurs. Recorded env actions are replayed
deterministically; the canonical automaton and grasp state are
evaluated after every action.

Per episode: first unresolved subgoal, last stable progress step,
HELD-object windows (per goal object; NOTE: a window opens only when
the grasp check first reports the object as held — attempts that never
establish a held state are NOT counted, so this is not a count of all
grasp attempts), invalidation (1→0) and recovery events WITH predicate
identity and action step, terminal-two-unresolved flag.

Output: results/libero_loho_public_v1/v069_failure_inventory.json —
the frozen input to the failure-anchored collection design (V6.9.0
amendment; the original v067 file is preserved unmodified).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
TRACES = RESULTS / "v06_eval" / "traces"
EPISODE_LENGTH = {"loho_t1_drawer": 700, "loho_t2_basket3": 900,
                  "loho_t3_tray": 900, "loho_t4_tray": 900,
                  "loho_t5_drawer_cabinet": 990}
DEV_SEEDS = [1600, 1610, 1620, 1630, 1640]
RUN_ID = "v66_dev"


def main() -> None:
    from lcwm.goal_semantics import env_eval_fn, goal_terminal_success
    from lcwm.loho_public import load_public_tasks, make_public_env
    from lcwm.task_automaton import GoalAutomaton
    from scripts.collect_v067_continuations import grasp_state

    goal_manifest = json.loads(
        (RESULTS / "goal_spec_manifest_v067.json").read_text())
    tasks_spec = load_public_tasks()

    episodes = []
    for task, spec in tasks_spec.items():
        subgoals = list(spec["ordered_subgoals"])
        entry = goal_manifest["tasks"][task]
        canon = entry["canonical_goal_spec_id"]
        term_preds = [tuple(p) for p in entry["goal_specs"][canon]
                      ["terminal_predicates"]]
        pick_objs = [sg.split()[1] for sg in subgoals
                     if sg.split()[0] == "pick_up"]
        for seed in DEV_SEEDS:
            tr_path = TRACES / f"{RUN_ID}_{task}_s{seed}_stock.npz"
            if not tr_path.exists():
                print(f"[miss] {tr_path.name}", flush=True)
                continue
            actions = np.load(tr_path)["actions_env"]
            env = make_public_env(task, EPISODE_LENGTH[task] + 200)
            try:
                env.reset(seed=seed)
                env._env.env.horizon = EPISODE_LENGTH[task] + 300
                eval_fn = env_eval_fn(env)
                auto = GoalAutomaton(subgoals)
                auto.start(env)
                auto.evaluate(env, 0)
                bodies = auto.bodies
                grasp_windows = {o: [] for o in pick_objs}
                open_grasp = {o: None for o in pick_objs}
                last_progress = 0
                progress_hi = sum(auto.prev_valid)
                terminal_step = None
                for t in range(len(actions)):
                    _o, _r, term, trunc, _i = env.step(actions[t])
                    auto.evaluate(env, t + 1)
                    n_valid = sum(auto.prev_valid)
                    if n_valid > progress_hi:
                        progress_hi = n_valid
                        last_progress = t + 1
                    g = grasp_state(env, bodies)["grasped"]
                    for o in pick_objs:
                        held = bool(g.get(o))
                        if held and open_grasp[o] is None:
                            open_grasp[o] = t + 1
                        elif not held and open_grasp[o] is not None:
                            grasp_windows[o].append(
                                (open_grasp[o], t + 1))
                            open_grasp[o] = None
                    if term and terminal_step is None:
                        terminal_step = t + 1
                    if term or trunc:
                        break
                    if goal_terminal_success(env, term_preds, eval_fn) \
                            and terminal_step is None:
                        terminal_step = t + 1
                for o in pick_objs:
                    if open_grasp[o] is not None:
                        grasp_windows[o].append((open_grasp[o], None))
                n = len(subgoals)
                valid = list(auto.prev_valid)
                invalidations = [
                    {"step": int(s), "subgoal": subgoals[i],
                     "recovered_by_end": bool(auto.prev_valid[i])}
                    for (s, i, d) in auto.flips if d == -1]
                recoveries = [
                    {"step": int(s), "subgoal": subgoals[i]}
                    for (s, i, d) in auto.flips if d == 1
                    and any(f[1] == i and f[2] == -1 and f[0] < s
                            for f in auto.flips)]
                achieved_idx = {i for i in auto.events_achieved}
                held_release = {
                    o: len(grasp_windows[o]) for o in pick_objs
                    if grasp_windows[o]
                    and subgoals.index(f"pick_up {o}")
                    not in achieved_idx}
                episodes.append({
                    "task": task, "seed": seed,
                    "n_actions": int(len(actions)),
                    "first_unresolved": next(
                        (subgoals[i] for i, v in enumerate(valid)
                         if not v), None),
                    "last_stable_progress_step": int(last_progress),
                    "max_valid": int(progress_hi), "n_subgoals": n,
                    "final_valid": valid,
                    "held_windows": {o: w for o, w in
                                     grasp_windows.items() if w},
                    "held_release_without_milestone": held_release,
                    "invalidations": invalidations,
                    "recoveries": recoveries,
                    "n_invalidations": len(invalidations),
                    "n_recoveries": len(recoveries),
                    "terminal_two_unresolved":
                        bool(sum(valid) == n - 2),
                    "terminal_step": terminal_step,
                })
                print(f"[inv] {task} s{seed}: max_valid "
                      f"{progress_hi}/{n}, first_unresolved "
                      f"{episodes[-1]['first_unresolved']}", flush=True)
            finally:
                env.close()

    by_task = {}
    for task in tasks_spec:
        eps = [e for e in episodes if e["task"] == task]
        by_task[task] = {
            "n": len(eps),
            "first_unresolved_counts": dict(Counter(
                e["first_unresolved"] for e in eps)),
            "mean_max_valid": (sum(e["max_valid"] for e in eps)
                               / len(eps)) if eps else None,
            "episodes_with_invalidation": sum(
                1 for e in eps if e["n_invalidations"]),
            "episodes_with_recovery": sum(
                1 for e in eps if e["n_recoveries"]),
            "terminal_two_unresolved": sum(
                1 for e in eps if e["terminal_two_unresolved"]),
            "episodes_with_held_release_without_milestone": sum(
                1 for e in eps
                if e["held_release_without_milestone"]),
        }
    out = {
        "schema": "v069_failure_inventory_v1", "run_schema": "v067",
        "input": ("iteration-1 development stock-arm traces, seeds "
                  "1600-1640; stock diagnostics only"),
        "note": ("held_windows: opens only when the grasp check first "
                 "reports held; attempts never establishing a held "
                 "state are NOT counted — not a count of all grasp "
                 "attempts"),
        "episodes": episodes, "by_task": by_task,
    }
    path = RESULTS / "v069_failure_inventory.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(by_task, indent=2, default=str), flush=True)
    print(f"-> {path}", flush=True)


if __name__ == "__main__":
    main()
