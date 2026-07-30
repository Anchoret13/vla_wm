#!/usr/bin/env python
"""H7.2 — bounded public-task training collection (registered contract in
2026-07-29-h7-progress-note.md; operationalizations fixed before this run).

20 sources (5 tasks x [stock 1400/1410, cur_wm 1420/1430]) -> 40 snapshots
(first-persistent-failure + late-unresolved per source) -> 160 branches
(N=4: source chunk + 3 stock samples, each executed 10 actions) -> 160
bounded 100-action stock continuations. Labels: physical effects, public
subgoal flips/damage, reward (subgoal delta), dQ_public, continuation
Q/terminal, generating policy, canonical+paraphrase+compatible language.
Split by source episode at collection time. Immediate object distance is
never an advantage.
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

OUT = Path("/home/stargazer/Desktop/vla_wm/datasets/libero_loho_public_v1")
NOISE_BASE = 30_000_000  # collection stream, disjoint from evaluation's 20e6
STOCK_SEEDS = (1400, 1410)
CURWM_SEEDS = (1420, 1430)
PERSIST_K = 5
LATE_FRACTION = 0.8
CONTINUATION_STEPS = 100
EPISODE_LENGTH = {"loho_t1_drawer": 700, "loho_t2_basket3": 900,
                  "loho_t3_tray": 900, "loho_t4_tray": 900,
                  "loho_t5_drawer_cabinet": 990}
PARAPHRASE = {
    "loho_t1_drawer": "place the front butter and the chocolate pudding into the cabinet's top drawer, then shut the drawer",
    "loho_t2_basket3": "place the alphabet soup, the butter and the tomato sauce into the basket",
    "loho_t3_tray": "place the alphabet soup, the cream cheese and the butter onto the tray",
    "loho_t4_tray": "place the left black bowl, the salad dressing and the chocolate pudding onto the tray",
    "loho_t5_drawer_cabinet": "place the back butter and the chocolate pudding into the top drawer, shut it, and set the black bowl on top of the cabinet",
}
COMPATIBLE = {  # same-scene cross-evaluable public goals
    "loho_t1_drawer": ["loho_t5_drawer_cabinet"],
    "loho_t5_drawer_cabinet": ["loho_t1_drawer"],
    "loho_t2_basket3": [], "loho_t3_tray": [], "loho_t4_tray": [],
}
TASK_ORDER = ["loho_t1_drawer", "loho_t2_basket3", "loho_t3_tray",
              "loho_t4_tray", "loho_t5_drawer_cabinet"]


def subgoal_states(tracker):
    return dict(tracker.completed)


@torch.no_grad()
def main() -> None:
    from lcwm.chassis import Pi05Runner
    from lcwm.lc_flow import LCFlowRunner, load_lc_adapter
    from lcwm.loho_public import (
        SubgoalTracker, load_public_tasks, make_public_env,
    )
    from lcwm.sampler import prefix_forward, sample_chunks
    from lcwm.seq_data import _problem_env
    from lcwm.snapshot import snap, restore

    tasks = load_public_tasks("reconstructed")
    manifest_tasks = json.loads(
        (REPO_ROOT / "results" / "libero_loho_public_v1"
         / "task_source_manifest.json").read_text())["tasks"]
    runner = Pi05Runner(suite_name="libero_10")
    cur_wm_lc, _ = load_lc_adapter(
        REPO_ROOT / "results" / "lc_flow_v04_e2e_v1_cur_wm"
        / "eval_adapter", device="cuda")

    OUT.mkdir(parents=True, exist_ok=True)
    source_manifest = []

    for task_index, task_name in enumerate(TASK_ORDER):
        spec = manifest_tasks[task_name]
        subgoals = spec["ordered_subgoals"]
        for policy_name, seeds in (("stock", STOCK_SEEDS),
                                   ("cur_wm", CURWM_SEEDS)):
            for seed in seeds:
                out_path = OUT / f"{task_name}_{policy_name}_s{seed}.pt"
                if out_path.exists():
                    print(f"[skip] {out_path.name}", flush=True)
                    source_manifest.append(out_path.name)
                    continue
                env = make_public_env(
                    task_name, EPISODE_LENGTH[task_name])
                try:
                    record = collect_source(
                        runner, cur_wm_lc, env, task_name, task_index,
                        policy_name, seed, subgoals,
                        SubgoalTracker, prefix_forward, sample_chunks,
                        _problem_env, snap, restore, LCFlowRunner,
                    )
                finally:
                    env.close()
                torch.save(record, out_path)
                source_manifest.append(out_path.name)
                n_pos = sum(
                    1 for s in record["snapshots"]
                    for b in s["branches"]
                    if b["continuation"]["dq_public"] > 0
                )
                print(
                    f"[source] {task_name} {policy_name} s{seed}: "
                    f"{len(record['snapshots'])} snapshots, "
                    f"{sum(len(s['branches']) for s in record['snapshots'])}"
                    f" branches, dQ>0 continuations: {n_pos}",
                    flush=True,
                )
    (OUT / "collection_manifest.json").write_text(json.dumps({
        "schema": "loho_public_collection_v1",
        "sources": sorted(source_manifest),
        "split_rule": "by source episode: seeds 1400/1420 train, 1410/1430 dev",
        "noise_base": NOISE_BASE,
        "registered": "2026-07-29-h7-progress-note.md H7.2 registration",
    }, indent=2))
    print(f"-> {OUT}", flush=True)


def collect_source(runner, cur_wm_lc, env, task_name, task_index,
                   policy_name, seed, subgoals, SubgoalTracker,
                   prefix_forward, sample_chunks, _problem_env,
                   snap, restore, LCFlowRunner):
    def noise_seed(decision):
        return NOISE_BASE + task_index * 2_000_000 + seed * 1_000 + decision

    if policy_name == "cur_wm":
        flow = LCFlowRunner(runner, cur_wm_lc)
    runner.reset()
    obs, _ = env.reset(seed=seed)
    tracker = SubgoalTracker(list(subgoals))
    tracker.start(env)
    inner = _problem_env(env)
    goal_atoms = [list(a) for a in inner.parsed_problem["goal_state"]]
    instruction = env.task_description

    decisions = []  # (decision, snapshot, obs_chunk_norm, first_unresolved)
    unresolved_streak, streak_target = 0, None
    t, decision = 0, 0
    horizon = 10
    while t < env.episode_length:
        snapshot = snap(env, t=t, suite_name="loho_public", task_id=0,
                        source_seed=seed, decision_index=decision)
        batch = runner._obs_to_policy_batch(obs, instruction)
        prefix = prefix_forward(runner.policy, batch)
        if policy_name == "cur_wm":
            from lcwm.lc_flow import sample_chunks_lc
            z = cur_wm_lc.posterior(prefix.hidden, prefix.pad_masks)
            bias = cur_wm_lc.adarms_bias(z)
            chunk = sample_chunks_lc(
                runner.policy, batch, bias, n=1,
                seed=noise_seed(decision), prefix=prefix)
        else:
            chunk = sample_chunks(
                runner.policy, batch, n=1,
                seed=noise_seed(decision), prefix=prefix)
        first_unresolved = next(
            (i for i in range(len(subgoals)) if i not in tracker.completed),
            None)
        decisions.append((decision, snapshot, chunk[0].float().cpu(),
                          first_unresolved))
        if first_unresolved == streak_target and first_unresolved is not None:
            unresolved_streak += 1
        else:
            streak_target, unresolved_streak = first_unresolved, 1
        actions_env = runner.chunk_to_env(chunk[:, :horizon])
        for a in actions_env:
            obs, _r, term, trunc, _i = env.step(a)
            t += 1
            if term or trunc:
                break
        tracker.update(env, t)
        decision += 1
        if term or trunc:
            break

    # Registered snapshot selection
    persist_idx = None
    streak_target, streak = None, 0
    for d, _snap, _chunk, fu in decisions:
        if fu is not None and fu == streak_target:
            streak += 1
            if streak >= PERSIST_K and persist_idx is None:
                persist_idx = d
        else:
            streak_target, streak = fu, 1
    late_d = int(len(decisions) * LATE_FRACTION)
    late_idx = None
    for d in range(late_d, len(decisions)):
        if decisions[d][3] is not None:
            late_idx = d
            break
    if late_idx is None:
        for d in range(len(decisions) - 1, -1, -1):
            if decisions[d][3] is not None:
                late_idx = d
                break
    if persist_idx is None:
        persist_idx = max(0, (late_idx or 0) - 10)
    selected = sorted({persist_idx, late_idx if late_idx is not None
                       else persist_idx})

    snapshots_out = []
    for sel in selected:
        d, snapshot, source_chunk, fu = decisions[sel]
        branches = []
        for cand in range(4):
            restore(env, snapshot)
            tracker_b = SubgoalTracker(list(subgoals))
            tracker_b.completed = {
                i: s for i, s in tracker.completed.items() if s <= snapshot.t
            }
            tracker_b.bodies = tracker.bodies
            tracker_b.start_pos = tracker.start_pos
            obs_b = env._format_raw_obs(env._env.env._get_observations())
            if cand == 0:
                chunk = source_chunk[None].cuda()
            else:
                batch = runner._obs_to_policy_batch(obs_b, instruction)
                prefix = prefix_forward(runner.policy, batch)
                chunk = sample_chunks(
                    runner.policy, batch, n=1,
                    seed=noise_seed(d) + 100_000 * cand, prefix=prefix)
            before_q = len(tracker_b.completed) / len(subgoals)
            actions_env = runner.chunk_to_env(chunk[:, :10])
            for a in actions_env:
                o2, _r, term_b, trunc_b, _i = env.step(a)
                if term_b or trunc_b:
                    break
            tracker_b.update(env, snapshot.t + 10)
            after_branch = dict(tracker_b.completed)
            # bounded stock continuation, full prompt
            steps_c = 0
            runner.reset()
            obs_c = env._format_raw_obs(env._env.env._get_observations())
            while steps_c < CONTINUATION_STEPS:
                batch = runner._obs_to_policy_batch(obs_c, instruction)
                prefix = prefix_forward(runner.policy, batch)
                cchunk = sample_chunks(
                    runner.policy, batch, n=1,
                    seed=noise_seed(d) + 500_000 + cand * 10_000 + steps_c,
                    prefix=prefix)
                for a in runner.chunk_to_env(cchunk[:, :10]):
                    obs_c, _r, term_c, trunc_c, _i = env.step(a)
                    steps_c += 1
                    if term_c or trunc_c or steps_c >= CONTINUATION_STEPS:
                        break
                tracker_b.update(env, snapshot.t + 10 + steps_c)
                if term_c or trunc_c:
                    break
            inner_env = _problem_env(env)
            terminal = all(bool(inner_env._eval_predicate(a))
                           for a in goal_atoms)
            q_cont = len(tracker_b.completed) / len(subgoals)
            branches.append({
                "candidate": cand,
                "is_source_chunk": cand == 0,
                "chunk_norm": chunk[0].float().cpu(),
                "subgoals_after_branch": after_branch,
                "reward_subgoal_delta": (
                    len(after_branch) / len(subgoals) - before_q),
                "continuation": {
                    "q_public": q_cont,
                    "dq_public": q_cont - before_q,
                    "terminal_success": bool(terminal),
                    "steps": steps_c,
                },
            })
        snapshots_out.append({
            "decision": d, "snapshot_t": snapshot.t,
            "first_unresolved": fu,
            "rule": "persistent_failure" if sel == persist_idx else "late",
            "branches": branches,
        })
    return {
        "schema": "loho_public_collection_v1",
        "task": task_name, "policy": policy_name, "seed": seed,
        "split": "train" if seed in (1400, 1420) else "dev",
        "generating_policy": (
            "pi05_full_prompt_frozen" if policy_name == "stock"
            else "v0.4_cur_wm_frozen"),
        "language": {
            "canonical": instruction,
            "paraphrase": PARAPHRASE[task_name],
            "compatible_goals": COMPATIBLE[task_name],
        },
        "subgoals": subgoals,
        "source_subgoal_completion": dict(tracker.completed),
        "snapshots": snapshots_out,
    }


if __name__ == "__main__":
    main()
