#!/usr/bin/env python
"""V7.0.0 — deterministic replay of the 75 saved V6.9 behavior traces
with the GoalSpec automaton evaluated after EVERY environment action.

The saved v69_dev records updated the automaton once per ten-action
chunk; their Q values are therefore provisional. This preserves the
success verdicts (expected 0/0/0) and emits a separately labeled
per-action report. Old records are not rewritten.

Output: results/libero_loho_public_v1/v070_policy/audit/
  corrected_v069_behavior.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
TRACES = RESULTS / "v069_eval" / "traces"
OUT = RESULTS / "v070_policy" / "audit" / "corrected_v069_behavior.json"
EPISODE_LENGTH = {"loho_t1_drawer": 700, "loho_t2_basket3": 900,
                  "loho_t3_tray": 900, "loho_t4_tray": 900,
                  "loho_t5_drawer_cabinet": 990}
ARMS = ("stock", "v069_grounded_reset", "v069_grounded_recurrent")
SEEDS = (1650, 1660, 1670, 1680, 1690)
RUN_ID = "v69_dev"


def main() -> None:
    from lcwm.loho_public import load_public_tasks, make_public_env
    from lcwm.task_automaton import GoalAutomaton, terminal_success

    tasks_spec = load_public_tasks()
    saved = {}
    for line in (RESULTS / "v069_eval" / "development"
                 / "records.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["run_id"] == RUN_ID:
                saved[(r["task"], r["seed"], r["arm"])] = r

    episodes = []
    for task, spec in tasks_spec.items():
        subgoals = list(spec["ordered_subgoals"])
        for seed in SEEDS:
            env = make_public_env(task, EPISODE_LENGTH[task] + 200)
            try:
                for arm in ARMS:
                    tr = np.load(
                        TRACES / f"{RUN_ID}_{task}_s{seed}_{arm}.npz")
                    actions = tr["actions_env"]
                    env.reset(seed=seed)
                    env._env.env.horizon = EPISODE_LENGTH[task] + 300
                    auto = GoalAutomaton(subgoals)
                    auto.start(env)
                    auto.evaluate(env, 0)
                    for t in range(len(actions)):
                        env.step(actions[t])
                        auto.evaluate(env, t + 1)
                    success = bool(terminal_success(env))
                    old = saved[(task, seed, arm)]
                    assert success == old["success"], (
                        f"{task} s{seed} {arm}: replay success "
                        f"{success} != saved {old['success']}")
                    episodes.append({
                        "task": task, "seed": seed, "arm": arm,
                        "success": success,
                        "q_valid_per_action": auto.q_valid(),
                        "p_valid_per_action": auto.p_valid(),
                        "ordered_prefix": auto.ordered_prefix(),
                        "damage_unrecovered":
                            auto.damage_unrecovered(),
                        "n_flips_down": sum(
                            1 for f in auto.flips if f[2] == -1),
                        "q_saved_chunk_resolution": old["q_valid"],
                        "n_actions": int(len(actions)),
                    })
                    print(f"[replay] {task} s{seed} {arm}: "
                          f"q_per_action={auto.q_valid():.3f} "
                          f"(saved {old['q_valid']:.3f}) "
                          f"success={success}", flush=True)
            finally:
                env.close()

    summary = {}
    for arm in ARMS:
        rs = [e for e in episodes if e["arm"] == arm]
        n = len(rs)
        summary[arm] = {
            "n": n,
            "SR": sum(e["success"] for e in rs) / n,
            "Q_per_action": sum(e["q_valid_per_action"]
                                for e in rs) / n,
            "P_per_action": sum(e["p_valid_per_action"]
                                for e in rs) / n,
            "damage": sum(e["damage_unrecovered"] for e in rs) / n,
            "Q_saved_chunk_resolution": sum(
                e["q_saved_chunk_resolution"] for e in rs) / n,
        }
    OUT.write_text(json.dumps({
        "schema": "v070_corrected_behavior_v1",
        "resolution": "per_action",
        "note": ("success verdicts preserved from v69_dev; Q/P/damage "
                 "re-derived at action resolution; saved "
                 "chunk-resolution Q kept alongside, not overwritten"),
        "episodes": episodes, "summary": summary}, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
