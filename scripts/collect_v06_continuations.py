#!/usr/bin/env python
"""V6.2 phase B — paired continuations for ACCEPTED groups only.

For each accepted group: deterministically re-reach the snapshot (reset →
re-stage if staged → replay recorded env actions, verifying object
positions against the phase-A record), snap; for each branch (u_support +
policy_candidates[0:4]): replay its recorded actions to the branch
endpoint, snap; then for every registered GoalSpec of the scene
(canonical + distinct) × repeat r ∈ {0,1}: restore the branch endpoint
and run frozen stock π0.5 under THAT GoalSpec's canonical prompt for 100
actions or terminal.

Registered contracts enforced here:
- continuation seed = 50e6 + task_index·2e6 + snapshot_decision·1e3 +
  branch_index·40 + repeat·20 + continuation_decision, branch_index
  0-3 = candidates, 4 = u_support — identical across sibling candidates
  for fixed (snapshot, goal, repeat, decision) by construction;
- a continuation collected under goal ℓa is never relabeled to ℓb;
- outcomes exposed at 10/30/60/100 actions via each goal's automaton
  (unrolled along the replayed path from episode start, never initialized
  at the snapshot);
- support-branch continuations are stored as provenance `support`
  (G_support), never mixed with policy-candidate records.

Output: v06_effect_crossed/continuations/<source>_d<dec>.pt
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
GOAL_SPECS = json.loads(
    (REPO_ROOT / "results" / "libero_loho_public_v1"
     / "goal_spec_manifest.json").read_text())
SELECTION = json.loads(
    (REPO_ROOT / "results" / "libero_loho_public_v1"
     / "v06_group_selection.json").read_text())
CONT_BASE = 50_000_000
EPISODE_LENGTH = {"loho_t1_drawer": 700, "loho_t2_basket3": 900,
                  "loho_t3_tray": 900, "loho_t4_tray": 900,
                  "loho_t5_drawer_cabinet": 990}
HORIZONS = (10, 30, 60, 100)
R = 2
REREACH_ATOL = 2e-3


def cont_seed(task_index: int, snap_decision: int, branch_index: int,
              repeat: int, cont_decision: int) -> int:
    return (CONT_BASE + task_index * 2_000_000 + snap_decision * 1_000
            + branch_index * 40 + repeat * 20 + cont_decision)


@torch.no_grad()
def main() -> None:
    from lcwm.chassis import Pi05Runner
    from lcwm.loho_public import make_public_env
    from lcwm.sampler import prefix_forward, sample_chunks
    from lcwm.snapshot import snap, restore
    from lcwm.task_automaton import (GoalAutomaton, fork_env_state,
                                     outcome_tuple, restore_env_state,
                                     terminal_success)
    from scripts.collect_loho_v06 import stage_support

    (DATA / "continuations").mkdir(exist_ok=True)
    runner = Pi05Runner(suite_name="libero_10")

    by_source: dict[str, list[dict]] = {}
    for g in SELECTION["accepted"]:
        by_source.setdefault(g["source_id"], []).append(g)

    for source_id, groups in sorted(by_source.items()):
        source = torch.load(DATA / "sources" / f"{source_id}.pt",
                            weights_only=False)
        task_name, task_index = source["task"], source["task_index"]
        spec = GOAL_SPECS["tasks"][task_name]
        goals = [{"goal_spec_id": spec["canonical"]["goal_spec_id"],
                  "language": spec["canonical"]["language"],
                  "ordered_subgoals":
                      spec["canonical"]["ordered_subgoals"]}]
        goals += [g for g in spec["distinct_goals"]]
        pending = [g for g in groups if not (
            DATA / "continuations"
            / f"{source_id}_d{g['decision']}.pt").exists()]
        if not pending:
            print(f"[skip] {source_id}", flush=True)
            continue

        env = make_public_env(task_name, EPISODE_LENGTH[task_name])
        try:
            runner.reset()
            obs, _ = env.reset(seed=source["seed"])
            # goal automata unroll from episode reset along the real path
            automata = {}
            for goal in goals:
                a = GoalAutomaton(goal["ordered_subgoals"])
                a.start(env)
                automata[goal["goal_spec_id"]] = a
            if source["provenance"] == "staged":
                canon = automata[spec["canonical"]["goal_spec_id"]]
                stage_support(env, canon, task_name,
                              recorded=source["staging_info"])
            for a in automata.values():
                a.evaluate(env, 0)

            max_dec = max(g["decision"] for g in pending)
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
                # re-reach determinism check against the phase-A record
                for a_env in row["actions_env"]:
                    env.step(a_env)
                    t += 1
                for a in automata.values():
                    a.evaluate(env, t)
                if d in {g["decision"] for g in pending}:
                    from lcwm.probe_data import body_positions
                    err = float(np.abs(
                        body_positions(
                            env, list(automata[goals[0][
                                "goal_spec_id"]].bodies.values()))
                        - row["obj_after"]).max())
                    assert err < REREACH_ATOL, (
                        f"{source_id} d={d}: re-reach diverges {err:.2e}")

            # automaton states at each pending snapshot come from the
            # phase-A record (verified path)
            for g in sorted(pending, key=lambda x: x["decision"]):
                d = g["decision"]
                audit = next(a for a in source["audits"]
                             if a["decision"] == d)
                row = next(r_ for r_ in source["rows"]
                           if r_["decision"] == d)
                records = []
                branch_list = [(b, {"support": 4, "candidate":
                                    b["candidate"], "replay": None}[
                                        b["kind"]])
                               for b in audit["branches"]
                               if b["kind"] != "replay"]
                for branch, branch_index in branch_list:
                    restore(env, snaps[d])
                    for a_env in branch["actions_env"]:
                        env.step(a_env)
                    branch_end = snap(env, t=row["t_start"] + 10,
                                      suite_name="loho_public", task_id=0)
                    for goal in goals:
                        gid = goal["goal_spec_id"]
                        for repeat in range(R):
                            restore(env, branch_end)
                            auto = GoalAutomaton(
                                goal["ordered_subgoals"])
                            base = automata[gid]
                            auto.bodies = base.bodies
                            auto.start_pos = base.start_pos
                            restore_env_state(auto, auto_states[d][gid])
                            # continuation-relative flip window: keep
                            # validity/events, zero the flip log so
                            # damage/tau_next measure THIS continuation
                            # (branch-window flips are in the phase-A
                            # record). Registered in the v6 note.
                            auto.flips = []
                            obs_c = env._format_raw_obs(
                                env._env.env._get_observations())
                            auto.evaluate(env, 0)
                            q_at = {}
                            steps_c, cd = 0, 0
                            done = False
                            while steps_c < 100 and not done:
                                batch = runner._obs_to_policy_batch(
                                    obs_c, goal["language"])
                                prefix = prefix_forward(
                                    runner.policy, batch)
                                chunk = sample_chunks(
                                    runner.policy, batch, n=1,
                                    seed=cont_seed(task_index, d,
                                                   branch_index,
                                                   repeat, cd),
                                    prefix=prefix)
                                for a_env in runner.chunk_to_env(
                                        chunk[:, :10]):
                                    obs_c, _r, term, trunc, _i = env.step(
                                        a_env)
                                    steps_c += 1
                                    if term or trunc or steps_c >= 100:
                                        break
                                auto.evaluate(env, steps_c)
                                if steps_c in HORIZONS:
                                    q_at[steps_c] = auto.q_valid()
                                done = bool(term or trunc)
                                cd += 1
                            for h in HORIZONS:
                                q_at.setdefault(h, auto.q_valid())
                            records.append({
                                "branch_kind": branch["kind"],
                                "branch_index": branch_index,
                                "goal_spec_id": gid,
                                "goal_language": goal["language"],
                                "repeat": repeat,
                                "steps": steps_c,
                                "terminal_success":
                                    bool(terminal_success(env))
                                    if gid == goals[0]["goal_spec_id"]
                                    else None,
                                "outcome": outcome_tuple(
                                    auto, env, 0, q_at),
                                "provenance": ("support"
                                               if branch["kind"] ==
                                               "support" else "policy"),
                            })
                    print(f"  [cont] {source_id} d={d} "
                          f"branch={branch['kind']}"
                          f"{branch.get('candidate')}", flush=True)
                torch.save({
                    "schema": "v06_continuations_v1",
                    "source_id": source_id, "task": task_name,
                    "decision": d, "slot": g["slot"],
                    "split": source["split"],
                    "goals": [gg["goal_spec_id"] for gg in goals],
                    "records": records,
                    "R": R, "horizons": list(HORIZONS),
                }, DATA / "continuations" / f"{source_id}_d{d}.pt")
        finally:
            env.close()
    print("phase B complete", flush=True)


if __name__ == "__main__":
    main()
