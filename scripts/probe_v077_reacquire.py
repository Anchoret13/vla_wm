#!/usr/bin/env python
"""V7.7 — is pi0.5's `pick_up` 0/14 a capability, a prompt, an object, or a
state effect?

Registered in `plan_and_progress/archive/daily/2026-08-15.md` (V7.7) before execution.

The V7.6 yield pilot ended with one exact residual: from reacher-built
mid-chain states, pi0.5 under the ATOMIC prompt hit 0/14 on `pick_up`
targets in 60 actions. Four readings of that number survive, and two of them
are retroactively serious -- V7.4B (4/129) and V7.6 (6/36) BOTH used the
atomic prompt, so if the prompt or the pick capability is the problem then
neither run was measuring recovery at all.

Five cells separate them. Three tasks x three seeds each = 45 rollouts.

    A  episode start   object 1   atomic     trivial baseline: can it pick?
    B  episode start   object 1   full       prompt effect at fresh state
    C  reacher d2      object 2   atomic     the pilot's own condition
    D  reacher d2      object 2   full       prompt effect mid-chain
    E  episode start   object 2   atomic     THE decisive control

E exists because "mid-chain" and "second object" are perfectly confounded in
the pilot. Asking for object 2 from a fresh state is what separates them.

Graded outcome. Binary milestone is what produced an uninformative zero, so
every rollout records the FIRST step at which each level is reached:
approached / contacted / moved / displaced / lifted / milestone. That
distinguishes "never went near it" from "grasped but never lifted".

Budget 120 actions, reported at BOTH 60 and 120. Because each level stores
its first step, the registered-60 figure is recoverable from the same
rollouts -- a superset, not a moved threshold.

`milestone` uses the GoalAutomaton's own criterion, identical to the pilot's
positivity rule, so cell C is a true replication.

Output: `<date>_v077_reacquire_r1/reacquire_probe.json`.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
RID = "v077_reacquire_r1"

# Registered before execution --------------------------------------------
TASKS = ("loho_t1_drawer", "loho_t2_basket3", "loho_t5_drawer_cabinet")
SEEDS = (2620, 2621, 2622)          # fresh family, disjoint from 2600-2614
MAX_ACT = 120                       # reported at 60 AND 120
MAX_DEC = 12
REPORT_AT = (60, 120)
PHASE_BUDGET = 80                   # reacher, as in the pre-check
SUPPORTED_MIN = 3                   # ">= 3/9" in the decision rule
EPISODE_LENGTH = {"loho_t1_drawer": 700, "loho_t2_basket3": 900,
                  "loho_t3_tray": 900, "loho_t4_tray": 900,
                  "loho_t5_drawer_cabinet": 990}

# cell -> (build_depth, target subgoal index, prompt style)
CELLS = {
    "A": (0, 0, "atomic"),
    "B": (0, 0, "full"),
    "C": (2, 2, "atomic"),
    "D": (2, 2, "full"),
    "E": (0, 2, "atomic"),
}

LEVELS = ("approached", "contacted", "moved", "displaced", "lifted",
          "milestone")
THRESH = {"approached": 0.05, "contacted": 0.03, "moved": 0.005,
          "displaced": 0.02, "lifted": 0.03}


def sha_seed(p: str) -> int:
    return int(hashlib.sha256(p.encode()).hexdigest()[:8], 16)


def run_date() -> str:
    return subprocess.run(["date", "+%F"], capture_output=True, text=True,
                          env={"TZ": "America/Chicago"}).stdout.strip()


def close_env(env) -> None:
    try:
        env.close()
    except Exception:
        pass


@torch.no_grad()
def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from lcwm.chassis import Pi05Runner
    from lcwm.loho_public import make_public_env
    from lcwm.probe_data import body_positions
    from lcwm.sampler import prefix_forward, sample_chunks
    from lcwm.task_automaton import GoalAutomaton
    from lcwm.v067_lineage import flow_noise
    from lcwm.v076_reacher import ScriptedReacher, eef_pos
    from scripts.collect_v069_corrections import OBJ_DISPLAY

    man = json.loads((RESULTS / "task_source_manifest.json").read_text())
    device = torch.device("cuda")
    runner = Pi05Runner(suite_name="libero_10")
    cfg = runner.policy.config

    def obs_frame(env):
        return env._format_raw_obs(env._env.env._get_observations())

    plan = [(c, t, s) for c in CELLS for t in TASKS for s in SEEDS]
    if args.limit:
        plan = plan[:args.limit]
    print(f"[v077] {len(plan)} rollouts", flush=True)

    rows = []
    for i, (cell, task, seed) in enumerate(plan):
        depth, tgt_idx, style = CELLS[cell]
        subgoals = man["tasks"][task]["ordered_subgoals"]
        tgt_obj = subgoals[tgt_idx].split()[1]

        env = make_public_env(task, EPISODE_LENGTH[task] + 400)
        runner.reset()
        env.reset(seed=seed)
        auto = GoalAutomaton(subgoals)
        auto.start(env)
        auto.evaluate(env, 0)
        full_prompt = env.task_description
        od = OBJ_DISPLAY.get((task, tgt_obj),
                             "the " + tgt_obj.rsplit("_", 1)[0]
                             .replace("_", " "))
        prompt = f"pick up {od}" if style == "atomic" else full_prompt

        # ---- state construction (acquisition instrument) -------------
        steps, built = 0, True
        if depth:
            reacher = ScriptedReacher(auto.bodies)
            for j in range(depth):
                parts = subgoals[j].split()
                mode, obj = parts[0], parts[1]
                region = parts[2] if len(parts) > 2 else None
                reacher.reset_phase()
                done = False
                for _ in range(PHASE_BUDGET):
                    a = reacher.act(env, mode, obj, region,
                                    auto.start_pos.get(obj))
                    env.step(a)
                    steps += 1
                    auto.evaluate(env, steps)
                    env._env.env.done = False
                    if auto.prev_valid[j]:
                        done = True
                        break
                if not done:
                    built = False
                    break
        if not built:
            rows.append({"cell": cell, "task": task, "seed": seed,
                         "state_built": False, "excluded": True,
                         "note": "reacher failed to build the d2 state"})
            print(f"[v077] {i + 1}/{len(plan)} {cell} {task} s{seed}"
                  f"  STATE FAILED", flush=True)
            close_env(env)
            continue

        # ---- pi0.5 alone from here ----------------------------------
        ref = body_positions(env, [auto.bodies[tgt_obj]])[0].copy()
        first = {k: None for k in LEVELS}
        rec, frame = 0, obs_frame(env)
        for cd in range(MAX_DEC):
            b = runner._obs_to_policy_batch(frame, prompt)
            pfx = prefix_forward(runner.policy, b)
            nz = flow_noise(sha_seed(f"{RID}|{cell}|{task}|{seed}|{cd}"),
                            cfg.chunk_size, cfg.max_action_dim)
            ch = sample_chunks(runner.policy, b, n=1,
                               noise=nz.to(device), prefix=pfx)
            for a_env in runner.chunk_to_env(ch[:, :10]):
                env.step(a_env)
                rec += 1
                auto.evaluate(env, steps + rec)
                env._env.env.done = False
                frame = obs_frame(env)

                o = body_positions(env, [auto.bodies[tgt_obj]])[0]
                e = eef_pos(env)
                lat = float(np.linalg.norm((o - e)[:2]))
                dist = float(np.linalg.norm(o - e))
                disp = float(np.linalg.norm(o - ref))
                dz = float(o[2] - ref[2])
                hit = {
                    "approached": lat < THRESH["approached"],
                    "contacted": dist < THRESH["contacted"],
                    "moved": disp >= THRESH["moved"],
                    "displaced": disp >= THRESH["displaced"],
                    "lifted": dz >= THRESH["lifted"],
                    "milestone": bool(auto.prev_valid[tgt_idx]),
                }
                for k, v in hit.items():
                    if v and first[k] is None:
                        first[k] = rec
                if rec >= MAX_ACT:
                    break
            if rec >= MAX_ACT:
                break

        row = {"cell": cell, "task": task, "seed": seed,
               "state_built": True, "excluded": False,
               "build_actions": int(steps),
               "target_object": tgt_obj, "target_subgoal": subgoals[tgt_idx],
               "prompt_style": style, "prompt": prompt,
               "policy_actions": int(rec),
               "first_step": first}
        for h in REPORT_AT:
            row[f"at_{h}"] = {k: bool(first[k] is not None and first[k] <= h)
                              for k in LEVELS}
        rows.append(row)
        got = [k for k in LEVELS if first[k] is not None]
        print(f"[v077] {i + 1}/{len(plan)} {cell} {task} s{seed}"
              f"  '{prompt[:42]}'  reached={got or ['nothing']}", flush=True)
        close_env(env)

    # ---- aggregate ---------------------------------------------------
    scored = [r for r in rows if not r["excluded"]]

    def cell_counts(cell, horizon):
        rs = [r for r in scored if r["cell"] == cell]
        return {k: sum(r[f"at_{horizon}"][k] for r in rs) for k in LEVELS} \
            | {"n": len(rs)}

    summary = {str(h): {c: cell_counts(c, h) for c in CELLS}
               for h in REPORT_AT}

    def sup(c, h=120):
        return summary[str(h)][c]["milestone"] >= SUPPORTED_MIN

    def zero(c, h=120):
        return summary[str(h)][c]["milestone"] == 0

    readings = []
    if sup("A") and zero("C"):
        readings.append("(d) STATE-SPECIFIC: pi0.5 picks from a fresh "
                        "state but not from a reacher-built mid-chain "
                        "state; the V7.6 acquisition mechanism is "
                        "implicated, not the policy")
    if zero("A") and sup("B"):
        readings.append("(b) PROMPT-SPECIFIC: the atomic acquisition "
                        "prompt is the wrong instrument; V7.4B's 4/129 "
                        "and V7.6's 6/36 are confounded by it and must "
                        "be re-read")
    if zero("A") and zero("B"):
        readings.append("(a) CAPABILITY: pi0.5 cannot execute an isolated "
                        "pick under either prompt within 120 actions; the "
                        "recovery protocol was never measuring recovery "
                        "in ANY run that used it")
    if sup("E") and zero("C"):
        readings.append("state, not object (reinforces (d))")
    if zero("E") and sup("A"):
        readings.append("(c) OBJECT-SPECIFIC: the pilot's pick_up 0/14 is "
                        "partly an artifact of always targeting object 2")
    if not readings:
        readings.append("no registered pattern matched; reported as "
                        "measured with no forced reading")

    out = Path(args.out) if args.out else RESULTS / f"{run_date()}_{RID}"
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "v077_reacquire_probe_v1", "run_schema": "v077",
        "run_id": RID,
        "registered_by": "plan_and_progress/archive/daily/2026-08-15.md — V7.7, "
                         "registered before execution",
        "question": "Is the V7.6 pilot's pick_up 0/14 a pi0.5 capability "
                    "limit (a), a prompt artifact (b), an object effect "
                    "(c), or specific to reacher-built states (d)?",
        "registered": {
            "cells": {c: {"build_depth": v[0], "target_subgoal_index":
                          v[1], "prompt": v[2]} for c, v in CELLS.items()},
            "tasks": list(TASKS), "seeds": list(SEEDS),
            "max_actions": MAX_ACT, "max_decisions": MAX_DEC,
            "report_at": list(REPORT_AT),
            "supported_min": SUPPORTED_MIN,
            "level_thresholds": THRESH,
            "note": "milestone uses the GoalAutomaton criterion, "
                    "identical to the V7.6 pilot positivity rule, so "
                    "cell C replicates the pilot",
        },
        "rows": rows,
        "summary_by_horizon": summary,
        "n_state_failures": sum(1 for r in rows if r["excluded"]),
        "registered_readings": readings,
    }
    dest = out / "reacquire_probe.json"
    dest.write_text(json.dumps(report, indent=2, default=float))

    for h in REPORT_AT:
        print(f"\n=== at {h} actions ===")
        print(f"{'cell':<5} {'n':>3}  " +
              "  ".join(f"{k[:9]:>9}" for k in LEVELS))
        for c in CELLS:
            cc = summary[str(h)][c]
            print(f"{c:<5} {cc['n']:>3}  " +
                  "  ".join(f"{cc[k]:>9}" for k in LEVELS))
    print("\nregistered reading(s):")
    for r in readings:
        print(f"  - {r}")
    print(f"[v077] wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
