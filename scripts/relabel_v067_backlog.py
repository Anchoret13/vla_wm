#!/usr/bin/env python
"""V6.8 step 1 — corrected immediate crossed labels for the executed
BACKLOG: every phase-A audit group NOT in the accepted set (105 groups).

Same machinery as the v067 collector (deterministic re-reach with
per-action evaluation of every GoalSpec automaton; branch replay with
done-clearing; qpos incl. fixture joints; replay-vs-candidate-0
endpoint deltas measured and flagged) but NO continuations and NO π0.5
— this is a sim-only relabeling pass that runs before any new source
collection, per the registered V6.8 order.

Output: v06_effect_crossed/backlog_relabels_v067/<source_id>.pt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

DATA = Path("/home/stargazer/Desktop/vla_wm/datasets/libero_loho_public_v1"
            "/v06_effect_crossed")
RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
EPISODE_LENGTH = {"loho_t1_drawer": 700, "loho_t2_basket3": 900,
                  "loho_t3_tray": 900, "loho_t4_tray": 900,
                  "loho_t5_drawer_cabinet": 990}
REREACH_ATOL = 2e-3


def main() -> None:
    from lcwm.goal_semantics import env_eval_fn, goal_terminal_success
    from lcwm.loho_public import make_public_env
    from lcwm.probe_data import body_positions
    from lcwm.task_automaton import (GoalAutomaton, fork_env_state,
                                     restore_env_state)
    from lcwm.snapshot import restore, snap
    from lcwm.v067_lineage import RUN_SCHEMA, assert_v067_payload
    from scripts.collect_loho_v06 import obs_q, stage_support
    from scripts.collect_v067_continuations import (full_qpos,
                                                    grasp_state,
                                                    qpos_joint_map)

    goal_manifest = json.loads(
        (RESULTS / "goal_spec_manifest_v067.json").read_text())
    assert_v067_payload(goal_manifest, "goal_spec_manifest_v067.json",
                        "goal_manifest")
    selection = json.loads(
        (RESULTS / "v06_group_selection.json").read_text())
    accepted: dict[str, set[int]] = {}
    for g in selection["accepted"]:
        accepted.setdefault(g["source_id"], set()).add(g["decision"])
    tol = selection["tolerances_95pct"]

    out_dir = DATA / "backlog_relabels_v067"
    out_dir.mkdir(exist_ok=True)

    for sp in sorted((DATA / "sources").glob("*.pt")):
        source = torch.load(sp, weights_only=False)
        sid = source["source_id"]
        out_path = out_dir / sp.name
        if out_path.exists():
            print(f"[skip] {sid}", flush=True)
            continue
        backlog = [a for a in source["audits"]
                   if a["decision"] not in accepted.get(sid, set())]
        if not backlog:
            torch.save({"schema": "v067_backlog_labels_v1",
                        "run_schema": RUN_SCHEMA, "source_id": sid,
                        "task": source["task"],
                        "split": source["split"], "groups": []},
                       out_path)
            print(f"[empty] {sid}", flush=True)
            continue
        task_name = source["task"]
        entry = goal_manifest["tasks"][task_name]
        canon_id = entry["canonical_goal_spec_id"]
        goal_specs = entry["goal_specs"]
        goal_ids = [canon_id] + [g for g in goal_specs if g != canon_id]
        term_preds = {gid: [tuple(p) for p in
                            goal_specs[gid]["terminal_predicates"]]
                      for gid in goal_ids}

        env = make_public_env(task_name, EPISODE_LENGTH[task_name] + 200)
        try:
            env.reset(seed=source["seed"])
            env._env.env.horizon = EPISODE_LENGTH[task_name] + 300
            eval_fn = env_eval_fn(env)
            automata = {}
            for gid in goal_ids:
                a = GoalAutomaton(goal_specs[gid]["ordered_subgoals"])
                a.start(env)
                automata[gid] = a
            if source["provenance"] == "staged":
                stage_support(env, automata[canon_id], task_name,
                              recorded=source["staging_info"])
            for a in automata.values():
                a.evaluate(env, 0)
            bodies = automata[canon_id].bodies

            backlog_decs = {a["decision"] for a in backlog}
            max_dec = max(backlog_decs)
            snaps, auto_states = {}, {}
            t = 0
            for row in source["rows"]:
                d = row["decision"]
                if d > max_dec:
                    break
                snaps[d] = snap(env, t=t, suite_name="loho_public",
                                task_id=0)
                auto_states[d] = {gid: fork_env_state(a)
                                  for gid, a in automata.items()}
                for a_env in row["actions_env"]:
                    env.step(a_env)
                    t += 1
                    for a in automata.values():
                        a.evaluate(env, t)
                if d in backlog_decs:
                    err = float(np.abs(body_positions(
                        env, list(bodies.values()))
                        - row["obj_after"]).max())
                    assert err < REREACH_ATOL, (
                        f"{sid} d={d}: re-reach diverges {err:.2e}")

            out_groups = []
            for audit in sorted(backlog, key=lambda a: a["decision"]):
                d = audit["decision"]
                branch_summaries = []
                cand0_end = {}
                for branch in audit["branches"]:
                    restore(env, snaps[d])
                    env._env.env.done = False
                    branch_autos = {}
                    for gid in goal_ids:
                        ba = GoalAutomaton(
                            goal_specs[gid]["ordered_subgoals"])
                        base = automata[gid]
                        ba.bodies, ba.start_pos = base.bodies, \
                            base.start_pos
                        restore_env_state(ba, auto_states[d][gid])
                        branch_autos[gid] = ba
                    flips_at_fork = {gid: len(ba.flips)
                                     for gid, ba in branch_autos.items()}
                    qpos_before = full_qpos(env)
                    steps_b = 0
                    for a_env in branch["actions_env"]:
                        _o, _r, tb, tr, _i = env.step(a_env)
                        steps_b += 1
                        for ba in branch_autos.values():
                            ba.evaluate(env, steps_b)
                        if tb:
                            env._env.env.done = False
                        if tr:
                            break
                    q_after = obs_q(env._format_raw_obs(
                        env._env.env._get_observations()))
                    obj_b_after = body_positions(
                        env, list(bodies.values()))
                    qpos_after = full_qpos(env)
                    immediate = {}
                    for gid, ba in branch_autos.items():
                        window = ba.flips[flips_at_fork[gid]:]
                        before = auto_states[d][gid]
                        immediate[gid] = {
                            "valid_before": list(before["prev_valid"]),
                            "valid_after": list(ba.prev_valid),
                            "events_before": sorted(
                                before["events_achieved"]),
                            "events_after": sorted(ba.events_achieved),
                            "flips_01": [f for f in window
                                         if f[2] == 1],
                            "flips_10": [f for f in window
                                         if f[2] == -1],
                            "ordered_prefix_after": ba.ordered_prefix(),
                            "q_valid_after": ba.q_valid(),
                            "reward_valid": (
                                sum(ba.prev_valid)
                                - sum(before["prev_valid"])),
                            "terminal_now": goal_terminal_success(
                                env, term_preds[gid], eval_fn),
                        }
                    summary = {
                        "kind": branch["kind"],
                        "candidate": branch.get("candidate"),
                        "steps": steps_b,
                        "q_after": q_after,
                        "obj_after": obj_b_after,
                        "qpos_before": qpos_before,
                        "qpos_after": qpos_after,
                        "grasp_after": grasp_state(env, bodies),
                        "immediate": immediate,
                    }
                    if branch["kind"] == "candidate" and \
                            branch["candidate"] == 0:
                        cand0_end = {"q": q_after, "obj": obj_b_after,
                                     "immediate": immediate}
                    if branch["kind"] == "replay" and cand0_end:
                        dq = (q_after - cand0_end["q"]).abs()
                        deltas = {
                            "eef_pos": float(dq[:3].max()),
                            "eef_quat": float(dq[3:7].max()),
                            "gripper": float(dq[7:].max()),
                            "obj_pos": float(np.abs(
                                obj_b_after
                                - cand0_end["obj"]).max())}
                        exceeds = [k for k in deltas
                                   if deltas[k] > tol[k]]
                        mism = {g_: int(sum(
                            x != y for x, y in zip(
                                immediate[g_]["valid_after"],
                                cand0_end["immediate"][g_][
                                    "valid_after"])))
                            for g_ in goal_ids}
                        if any(mism.values()):
                            exceeds.append("valid_bits")
                        summary["replay_endpoint"] = {
                            "deltas": deltas,
                            "valid_bits_mismatch": mism,
                            "exceeds_tolerance": exceeds}
                    branch_summaries.append(summary)
                out_groups.append({
                    "decision": d, "slot": audit["slot"],
                    "auto_states_snapshot": auto_states[d],
                    "obj_before_snapshot": np.asarray(
                        next(r_ for r_ in source["rows"]
                             if r_["decision"] == d)["obj_before"]),
                    "branch_summaries": branch_summaries,
                })
                print(f"  [backlog] {sid} d={d}", flush=True)
            tmp = out_path.with_suffix(".tmp")
            torch.save({
                "schema": "v067_backlog_labels_v1",
                "run_schema": RUN_SCHEMA,
                "source_id": sid, "task": task_name,
                "task_index": source["task_index"],
                "split": source["split"],
                "provenance_source": source["provenance"],
                "goals": goal_ids,
                "canonical_goal_spec_id": canon_id,
                "goal_manifest_sha256":
                    goal_manifest["manifest_sha256"],
                "qpos_joint_map": qpos_joint_map(env),
                "groups": out_groups,
            }, tmp)
            tmp.replace(out_path)
            print(f"[backlog] {sid}: {len(out_groups)} groups",
                  flush=True)
        finally:
            env.close()
    print("v067 backlog relabel complete", flush=True)


if __name__ == "__main__":
    main()
