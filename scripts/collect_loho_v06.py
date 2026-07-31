#!/usr/bin/env python
"""V6.2 phase A — source rollouts + bounded snapshot search + first-ten /
replay audits (NO continuations here; phase B runs them for accepted
groups only, after tranche-A tolerances are frozen).

Registered contract (2026-07-30 + v6 progress note):
- tranche A sources: per task k: stock_train_A seed 2000+10k (stock π0.5,
  canonical prompt), support_train_A seed 2001+10k (privileged pose-set
  staging, DATA-ONLY, provenance `staged`, then stock π0.5 rollout);
- collection noise: 40e6 + task_index*2e6 + seed*1e3 + decision;
  candidate i seed = noise_seed(d) + 100000*i;
- slot criteria as registered; ≤4 eligible decisions per slot,
  chronological; branch order u_support, cand0..3, u_replay;
- everything executed is recorded, including rejected audits;
- raw observations and env actions from episode reset through every
  branch outcome are stored (crossed relabels re-unroll from reset).

Output: datasets/libero_loho_public_v1/v06_effect_crossed/
  sources/<source_id>.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

OUT = Path("/home/stargazer/Desktop/vla_wm/datasets/libero_loho_public_v1"
           "/v06_effect_crossed")
GOAL_SPECS = json.loads(
    (REPO_ROOT / "results" / "libero_loho_public_v1"
     / "goal_spec_manifest.json").read_text())
NOISE_BASE = 40_000_000
EPISODE_LENGTH = {"loho_t1_drawer": 700, "loho_t2_basket3": 900,
                  "loho_t3_tray": 900, "loho_t4_tray": 900,
                  "loho_t5_drawer_cabinet": 990}
TASK_ORDER = ["loho_t1_drawer", "loho_t2_basket3", "loho_t3_tray",
              "loho_t4_tray", "loho_t5_drawer_cabinet"]
MOVE_MM = 0.005
MAX_PER_SLOT = 4
SETTLE_STEPS = 120
STAGE_Z_TRIES = (0.10, 0.15, 0.06)


def noise_seed(task_index: int, seed: int, decision: int) -> int:
    return NOISE_BASE + task_index * 2_000_000 + seed * 1_000 + decision


def obs_q(obs) -> torch.Tensor:
    return torch.from_numpy(np.concatenate([
        obs["robot_state"]["eef"]["pos"],
        obs["robot_state"]["eef"]["quat"],
        obs["robot_state"]["gripper"]["qpos"],
    ])).float()


def stage_support(env, automaton, task_name: str) -> dict:
    """Privileged pose-set of the registered objects into the target
    region; asserts staged predicates true. DATA-ONLY provenance."""
    from lcwm.seq_data import _problem_env
    spec = GOAL_SPECS["tasks"][task_name]["staging"]
    inner = _problem_env(env)
    sim = inner.sim
    fixture = spec["target"].rsplit("_", 2)[0]  # e.g. basket_1
    if "cabinet" in spec["target"]:
        candidates = [b for b in sim.model.body_names
                      if fixture in b and "drawer" in b]
        anchor_body = candidates[0] if candidates else fixture + "_base"
        if anchor_body not in sim.model.body_names:
            anchor_body = next(b for b in sim.model.body_names
                               if fixture in b)
    else:
        anchor_body = next(b for b in sim.model.body_names
                           if b == fixture or b.startswith(fixture))
    anchor = sim.data.get_body_xpos(anchor_body).copy()
    staged, info = [], {"anchor_body": anchor_body}
    for k, obj in enumerate(spec["place_objects"]):
        joint = f"{obj}_joint0"
        placed = False
        idx = next(i for i, sg in enumerate(automaton.subgoals)
                   if sg == f"place {obj} {spec['target']}")
        dz_used = None
        for dz in STAGE_Z_TRIES:
            offset = np.array([0.03 * (k - 0.5), 0.0, dz])
            sim.data.set_joint_qpos(
                joint, np.concatenate([anchor + offset, [1, 0, 0, 0]]))
            sim.forward()
            for _ in range(SETTLE_STEPS):
                sim.step()
            # state-free predicate probe: no flip-log mutation
            if automaton._subgoal_true(env, idx):
                placed = True
                dz_used = dz
                break
        staged.append({"object": obj, "placed": placed, "z": dz_used})
        if not placed:
            info["staging_failed"] = obj
            break
    info["staged"] = staged
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tranche", choices=("A", "B"), default="A")
    args = parser.parse_args()

    from lcwm.chassis import Pi05Runner
    from lcwm.loho_public import make_public_env
    from lcwm.sampler import prefix_forward, sample_chunks
    from lcwm.snapshot import snap, restore
    from lcwm.task_automaton import (GoalAutomaton, fork_env_state,
                                     restore_env_state)

    (OUT / "sources").mkdir(parents=True, exist_ok=True)
    runner = Pi05Runner(suite_name="libero_10")
    offsets = {"A": (0, 1), "B": (2, 3)}[args.tranche]
    split_of = {0: "train", 1: "train", 2: "train", 3: "dev"}

    for task_index, task_name in enumerate(TASK_ORDER):
        spec = GOAL_SPECS["tasks"][task_name]
        subgoals = spec["canonical"]["ordered_subgoals"]
        for off in offsets:
            seed = 2000 + 10 * task_index + off
            provenance = "stock" if off in (0, 2) else "staged"
            source_id = f"{task_name}_{provenance}_s{seed}"
            out_path = OUT / "sources" / f"{source_id}.pt"
            if out_path.exists():
                print(f"[skip] {source_id}", flush=True)
                continue
            env = make_public_env(task_name, EPISODE_LENGTH[task_name])
            try:
                runner.reset()
                obs, _ = env.reset(seed=seed)
                automaton = GoalAutomaton(subgoals)
                automaton.start(env)
                staging_info = None
                if provenance == "staged":
                    staging_info = stage_support(env, automaton, task_name)
                    obs = env._format_raw_obs(
                        env._env.env._get_observations())
                instruction = env.task_description
                automaton.evaluate(env, 0)

                from lcwm.probe_data import body_positions
                bodies = list(automaton.bodies.values())

                rows, snaps = [], {}
                t, decision = 0, 0
                term = trunc = False
                while t < env.episode_length:
                    snaps[decision] = snap(
                        env, t=t, suite_name="loho_public", task_id=0)
                    obs_now = copy.deepcopy(obs)
                    automaton_before = fork_env_state(automaton)
                    batch = runner._obs_to_policy_batch(obs, instruction)
                    prefix = prefix_forward(runner.policy, batch)
                    chunk = sample_chunks(
                        runner.policy, batch, n=1,
                        seed=noise_seed(task_index, seed, decision),
                        prefix=prefix)
                    obj_before = body_positions(env, bodies).copy()
                    valid_before = list(automaton.prev_valid)
                    flips_before = len(automaton.flips)
                    actions_env = []
                    executed = 0
                    for a in runner.chunk_to_env(chunk[:, :10]):
                        obs, _r, term, trunc, _i = env.step(a)
                        actions_env.append(np.asarray(a))
                        t += 1
                        executed += 1
                        if term or trunc:
                            break
                    automaton.evaluate(env, t)
                    obj_after = body_positions(env, bodies).copy()
                    rows.append({
                        "decision": decision, "t_start": t - executed,
                        "obs": obs_now,
                        "automaton_before": automaton_before,
                        "chunk_norm": chunk[0].float().cpu(),
                        "actions_env": np.stack(actions_env),
                        "executed_len": executed,
                        "q": obs_q(obs_now),
                        "obj_before": obj_before, "obj_after": obj_after,
                        "valid_before": valid_before,
                        "valid_after": list(automaton.prev_valid),
                        "new_flips": automaton.flips[flips_before:],
                        "moved_mm": float(np.abs(
                            obj_after - obj_before).max()),
                    })
                    decision += 1
                    if term or trunc:
                        break

                # ---- slot eligibility (registered criteria) -------------
                n_dec = len(rows)
                slot2, slot1 = [], []
                for d in range(n_dec):
                    row = rows[d]
                    n_valid = sum(row["valid_before"])
                    unresolved = len(subgoals) - n_valid
                    recent = [f for dd in range(max(0, d - 2), d)
                              for f in rows[dd]["new_flips"]]
                    drop_recent = any(f[2] == -1 for f in recent)
                    place_recent = any(
                        f[2] == +1 and
                        subgoals[f[1]].startswith("place")
                        for f in recent)
                    if n_valid >= 1 and (unresolved <= 2 or drop_recent
                                         or place_recent):
                        slot2.append(d)
                    elif np.abs(row["obj_after"]
                                - row["obj_before"]).max() >= MOVE_MM:
                        slot1.append(d)
                audit_plan = ([("slot1", d) for d in slot1[:MAX_PER_SLOT]]
                              + [("slot2", d) for d in slot2[:MAX_PER_SLOT]])

                # ---- first-ten / replay audits --------------------------
                audits = []
                for slot, d in audit_plan:
                    branches = []
                    for kind, cand in ([("support", None)]
                                       + [("candidate", i)
                                          for i in range(4)]
                                       + [("replay", 0)]):
                        restore(env, snaps[d])
                        branch_automaton = GoalAutomaton(subgoals)
                        branch_automaton.bodies = automaton.bodies
                        branch_automaton.start_pos = automaton.start_pos
                        restore_env_state(branch_automaton,
                                          rows[d]["automaton_before"])
                        obs_b = env._format_raw_obs(
                            env._env.env._get_observations())
                        if kind == "support":
                            chunk_b = rows[d]["chunk_norm"][None].cuda()
                            acts = None
                        elif kind == "candidate":
                            batch = runner._obs_to_policy_batch(
                                obs_b, instruction)
                            prefix = prefix_forward(runner.policy, batch)
                            chunk_b = sample_chunks(
                                runner.policy, batch, n=1,
                                seed=noise_seed(task_index, seed, d)
                                + 100_000 * cand, prefix=prefix)
                            acts = None
                        else:  # replay of candidate 0
                            chunk_b = None
                            acts = branches[1]["actions_env"]
                        executed_actions = []
                        if acts is None:
                            it = runner.chunk_to_env(chunk_b[:, :10])
                        else:
                            it = list(acts)
                        for a in it:
                            env.step(a)
                            executed_actions.append(np.asarray(a))
                        branch_automaton.evaluate(
                            env, rows[d]["t_start"] + 10)
                        obs_end = env._format_raw_obs(
                            env._env.env._get_observations())
                        branches.append({
                            "kind": kind, "candidate": cand,
                            "chunk_norm": (chunk_b[0].float().cpu()
                                           if chunk_b is not None
                                           else None),
                            "actions_env": np.stack(executed_actions),
                            "q_after": obs_q(obs_end),
                            "obj_after": body_positions(
                                env, bodies).copy(),
                            "valid_after": list(
                                branch_automaton.prev_valid),
                            "flips": list(branch_automaton.flips),
                            "obs_after": copy.deepcopy(obs_end),
                        })
                    audits.append({"slot": slot, "decision": d,
                                   "branches": branches})
                    print(f"  [audit] {source_id} d={d} ({slot})",
                          flush=True)

                torch.save({
                    "schema": "v06_source_v1",
                    "source_id": source_id, "task": task_name,
                    "task_index": task_index, "seed": seed,
                    "provenance": provenance,
                    "tranche": args.tranche,
                    "split": split_of[off],
                    "language_canonical": instruction,
                    "goal_spec_id": spec["canonical"]["goal_spec_id"],
                    "subgoals": subgoals,
                    "staging_info": staging_info,
                    "n_decisions": n_dec,
                    "rows": rows,
                    "slot1_eligible": slot1, "slot2_eligible": slot2,
                    "audits": audits,
                    "goal_spec_manifest_sha256":
                        GOAL_SPECS["manifest_sha256"],
                }, out_path)
                print(f"[source] {source_id}: {n_dec} decisions, "
                      f"{len(audits)} audits "
                      f"(slot1 {len(slot1[:MAX_PER_SLOT])}, "
                      f"slot2 {len(slot2[:MAX_PER_SLOT])}), "
                      f"staging={staging_info}", flush=True)
            finally:
                env.close()
    print("phase A complete", flush=True)


if __name__ == "__main__":
    main()
