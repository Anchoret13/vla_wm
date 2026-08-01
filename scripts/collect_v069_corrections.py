#!/usr/bin/env python
"""V6.9.2 — executable failure-anchored corrections (mainline training
data, not a diagnostic).

Fresh source-disjoint stock rollouts (seeds registered before
collection); anchors from the frozen failure table at stable
pre-boundary states; at each anchor the fixed proposal set

  u_0              deployed canonical chunk (collection-noise seed)
  u_canon[1..15]   +100000*i canonical full-prompt draws
  u_subgoal[0..3]  2 atomic-subgoal + 2 compatible distinct-goal draws
                   (behavior_goal_id recorded; outside teacher pools)
  u_support        scripted privileged-state greedy servo (closed-loop;
                   competence judged ONLY by the registered outcomes)
  u_replay         exact replay of u_0 (audit, never a target)

is executed once (first ten actions) from the same restored state;
every GoalSpec automaton evaluates per action; every branch gets
immediate crossed relabels and R=2 canonical continuations under
sibling CRN (cont_seed_v067, run_id v069_fa1). Replay endpoint deltas
are measured and flagged (gross-failure bounds hard); a flagged anchor
retries once at the preceding stable decision. Every executed row is
retained; clean robust winners over u_0 are marked grounded
corrections; null/tied rows keep predictive-calibration value.

Output:
  datasets/libero_loho_public_v1/v06_effect_crossed/
    corrections_sources_v069/<source_id>.pt   (fresh rollout rows)
    corrections_v069/<source_id>_d<dec>.pt    (anchor groups)
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

DATA = Path("/home/stargazer/Desktop/vla_wm/datasets/libero_loho_public_v1"
            "/v06_effect_crossed")
RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
TASK_ORDER = ["loho_t1_drawer", "loho_t2_basket3", "loho_t3_tray",
              "loho_t4_tray", "loho_t5_drawer_cabinet"]
EPISODE_LENGTH = {"loho_t1_drawer": 700, "loho_t2_basket3": 900,
                  "loho_t3_tray": 900, "loho_t4_tray": 900,
                  "loho_t5_drawer_cabinet": 990}
NOISE_BASE = 40_000_000
RUN_ID = "v069_fa1"
R = 2
CONT_MAX = 100
HORIZONS = (10, 30, 60, 100)
N_CANON = 16          # u_0 + 15 extra
MAX_ANCHORS_PER_SOURCE = 2
MAX_ATTEMPTS_PER_TASK = 8
ANCHOR_MIN_GAP = 5
GROSS_FACTOR = 10.0
REREACH_ATOL = 2e-3

# frozen anchor families (from the V6.9 failure table)
ANCHOR_OBJ = {"loho_t1_drawer": "butter_1",
              "loho_t2_basket3": "butter_1",
              "loho_t3_tray": "alphabet_soup_1",
              "loho_t4_tray": "new_salad_dressing_1",
              "loho_t5_drawer_cabinet": "butter_2"}
ANCHOR_FORMS = {"loho_t1_drawer": ("pick_up", "place"),
                "loho_t2_basket3": ("pick_up", "place"),
                "loho_t3_tray": ("pick_up",),
                "loho_t4_tray": ("pick_up",),
                "loho_t5_drawer_cabinet": ("pick_up", "place")}
OBJ_DISPLAY = {("loho_t1_drawer", "butter_1"): "the butter at the front",
               ("loho_t2_basket3", "butter_1"): "the butter",
               ("loho_t3_tray", "alphabet_soup_1"): "the alphabet soup",
               ("loho_t4_tray", "new_salad_dressing_1"):
                   "the salad dressing",
               ("loho_t5_drawer_cabinet", "butter_2"):
                   "the butter at the back"}
REGION_DISPLAY = {"wooden_cabinet_1_top_region":
                  "the top drawer of the cabinet",
                  "basket_1_contain_region": "the basket",
                  "wooden_tray_1_contain_region": "the tray"}
# AMENDED (registered 2026-08-01, after the t3 zero-anchor diagnosis):
# the stock stall mode is WRONG-OBJECT engagement (t3: arm hovers 5 mm
# from cream_cheese while the soup sits 22 cm away), so a narrow
# approach window around the anchor object never fires. The queue's
# criterion is reachability — "a stable pre-contact state whose next
# ten actions can cross the relevant outcome boundary" — and ten servo
# actions cover ~25-30 cm. Window widened accordingly; a stall
# condition (first-unresolved unchanged for >=3 consecutive decisions)
# replaces proximity as the anchor-quality signal.
PICK_NEAR, PICK_FAR = 0.03, 0.30
PLACE_NEAR, PLACE_FAR = 0.05, 0.30
STABLE_MM = 0.005
STALL_DECS = 3


def noise_seed(task_index: int, seed: int, decision: int) -> int:
    return NOISE_BASE + task_index * 2_000_000 + seed * 1_000 + decision


def site_pos(env, name):
    return env._env.env.sim.data.get_site_xpos(name).copy()


def scripted_servo_action(env, phase_obj, region, bodies, mode):
    """One closed-loop env action toward the grasp point / region.
    Position servo (unit action ~ 5 cm command), orientation deltas 0."""
    from lcwm.probe_data import body_positions
    inner = env._env.env
    eef = np.asarray(
        inner.sim.data.site_xpos[inner.robots[0].eef_site_id]).copy() \
        if hasattr(inner.robots[0], "eef_site_id") else None
    if eef is None:
        obs = env._format_raw_obs(inner._get_observations())
        eef = np.asarray(obs["robot_state"]["eef"]["pos"])
    obj = body_positions(env, [bodies[phase_obj]])[0]
    a = np.zeros(7, dtype=np.float64)
    if mode == "pick_up":
        lateral = np.linalg.norm((obj - eef)[:2])
        if lateral > 0.015:
            target = obj + np.array([0.0, 0.0, 0.06])
            grip = -1.0
        elif eef[2] - obj[2] > 0.02:
            target = obj + np.array([0.0, 0.0, 0.005])
            grip = -1.0
        else:
            target = obj + np.array([0.0, 0.0, 0.01])
            grip = 1.0
    else:  # place
        tgt = site_pos(env, region)
        lateral = np.linalg.norm((tgt - obj)[:2])
        if lateral > 0.03:
            target = tgt + np.array([0.0, 0.0, 0.10]) + (eef - obj)
            grip = 1.0
        else:
            target = eef
            grip = -1.0
    a[:3] = np.clip((target - eef) / 0.05, -1.0, 1.0)
    a[6] = grip
    return a


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="*", default=TASK_ORDER)
    args = parser.parse_args()

    from lcwm.chassis import Pi05Runner
    from lcwm.goal_semantics import SuccessTracker, env_eval_fn
    from lcwm.loho_public import make_public_env
    from lcwm.probe_data import body_positions
    from lcwm.sampler import prefix_forward, sample_chunks
    from lcwm.snapshot import restore, snap
    from lcwm.task_automaton import (GoalAutomaton, fork_env_state,
                                     paired_preference,
                                     restore_env_state)
    from lcwm.v067_lineage import cont_seed_v067, flow_noise, noise_sha
    from scripts.collect_loho_v06 import obs_q
    from scripts.collect_v067_continuations import (full_qpos,
                                                    grasp_state,
                                                    qpos_joint_map)

    goal_manifest = json.loads(
        (RESULTS / "goal_spec_manifest_v067.json").read_text())
    support = json.loads(
        (RESULTS / "v067_support_report.json").read_text())
    tolerances = support["frozen_outcome_tolerances"]
    sel = json.loads(
        (RESULTS / "v06_group_selection.json").read_text())
    tol95 = sel["tolerances_95pct"]

    (DATA / "corrections_sources_v069").mkdir(exist_ok=True)
    (DATA / "corrections_v069").mkdir(exist_ok=True)
    runner = Pi05Runner(suite_name="libero_10")
    cfg = runner.policy.config
    chunk_size, max_dim = cfg.chunk_size, cfg.max_action_dim

    attempts_by_task: dict[str, int] = {t: 0 for t in TASK_ORDER}

    for task_name in args.tasks:
        task_index = TASK_ORDER.index(task_name)
        entry = goal_manifest["tasks"][task_name]
        canon_id = entry["canonical_goal_spec_id"]
        goal_specs = entry["goal_specs"]
        goal_ids = [canon_id] + [g for g in goal_specs if g != canon_id]
        term_preds = {gid: [tuple(p) for p in
                            goal_specs[gid]["terminal_predicates"]]
                      for gid in goal_ids}
        subgoals = goal_specs[canon_id]["ordered_subgoals"]
        anchor_obj = ANCHOR_OBJ[task_name]
        anchor_sgs = {}
        for form in ANCHOR_FORMS[task_name]:
            for i, sg in enumerate(subgoals):
                parts = sg.split()
                if parts[0] == form and parts[1] == anchor_obj:
                    anchor_sgs[sg] = (form, parts[2] if form == "place"
                                      else None)
        obj_disp = OBJ_DISPLAY[(task_name, anchor_obj)]
        place_region = next((r for (f, r) in anchor_sgs.values()
                             if f == "place"), None)
        atomic_prompts = [f"pick up {obj_disp}"]
        if place_region:
            atomic_prompts.append(
                f"put {obj_disp} in {REGION_DISPLAY[place_region]}")
        else:
            atomic_prompts.append(f"lift {obj_disp} off the table")
        distinct_langs = [
            (gid, goal_specs[gid]["language"]) for gid in goal_ids
            if gid != canon_id][:2]

        for off in (0, 1, 2):
            seed = 2100 + 10 * task_index + off
            split = "dev" if off == 2 else "train"
            source_id = f"{task_name}_correction_s{seed}"
            src_path = (DATA / "corrections_sources_v069"
                        / f"{source_id}.pt")
            if attempts_by_task[task_name] >= MAX_ATTEMPTS_PER_TASK:
                print(f"[cap] {task_name}: attempt cap reached",
                      flush=True)
                break

            env = make_public_env(task_name,
                                  EPISODE_LENGTH[task_name] + 200)
            try:
                runner.reset()
                obs, _ = env.reset(seed=seed)
                env._env.env.horizon = EPISODE_LENGTH[task_name] + 300
                eval_fn = env_eval_fn(env)
                automata = {}
                for gid in goal_ids:
                    a = GoalAutomaton(
                        goal_specs[gid]["ordered_subgoals"])
                    a.start(env)
                    automata[gid] = a
                for a in automata.values():
                    a.evaluate(env, 0)
                bodies = automata[canon_id].bodies
                instruction = env.task_description

                def anchor_ok(first_unres, obj_positions, obj_prev,
                              eef, stall_count):
                    """AMENDED rule: reachability window + stall
                    condition (>=3 consecutive decisions on the same
                    first-unresolved anchor subgoal)."""
                    if first_unres not in anchor_sgs:
                        return None
                    if stall_count < STALL_DECS:
                        return None
                    form, region = anchor_sgs[first_unres]
                    gsp = grasp_state(env, bodies)["grasped"]
                    opos = body_positions(
                        env, [bodies[anchor_obj]])[0]
                    moved = float(np.abs(
                        obj_positions - obj_prev).max())
                    if form == "pick_up":
                        dist = float(np.linalg.norm(eef - opos))
                        if (not gsp.get(anchor_obj)
                                and PICK_NEAR <= dist <= PICK_FAR
                                and moved < STABLE_MM):
                            return {"form": form, "region": region,
                                    "dist": dist}
                    else:
                        tgt = site_pos(env, region)
                        dist = float(np.linalg.norm(opos - tgt))
                        if (bool(gsp.get(anchor_obj))
                                and PLACE_NEAR <= dist <= PLACE_FAR):
                            return {"form": form, "region": region,
                                    "dist": dist}
                    return None

                rows, snaps, auto_states = [], {}, {}
                anchor_candidates = []
                obj_prev = body_positions(env, list(bodies.values()))
                stall_count, prev_unres = 0, "___"
                term = trunc = False
                if src_path.exists():
                    # ---- deterministic re-reach of a saved source ------
                    saved = torch.load(src_path, weights_only=False)
                    assert saved["schema"] == "v069_correction_source_v1"
                    rows = saved["rows"]
                    t = 0
                    for row in rows:
                        d = row["decision"]
                        snaps[d] = snap(env, t=t,
                                        suite_name="loho_public",
                                        task_id=0)
                        auto_states[d] = {
                            gid: fork_env_state(a)
                            for gid, a in automata.items()}
                        first_unres = row["first_unresolved"]
                        stall_count = (stall_count + 1
                                       if first_unres == prev_unres
                                       else 1)
                        prev_unres = first_unres
                        obj_positions = body_positions(
                            env, list(bodies.values()))
                        err = float(np.abs(
                            obj_positions - row["obj_before"]).max())
                        assert err < REREACH_ATOL, (
                            f"{source_id} d={d}: re-reach diverges "
                            f"{err:.2e}")
                        hit = anchor_ok(
                            first_unres, obj_positions, obj_prev,
                            np.asarray(
                                row["obs"]["robot_state"]["eef"]
                                ["pos"]), stall_count)
                        if hit:
                            anchor_candidates.append(
                                {"decision": d, **hit})
                        obj_prev = obj_positions
                        for a_env in row["actions_env"]:
                            env.step(a_env)
                            t += 1
                            for au in automata.values():
                                au.evaluate(env, t)
                else:
                    # ---- fresh stock rollout with per-action eval ------
                    t, decision = 0, 0
                    while t < EPISODE_LENGTH[task_name]:
                        snaps[decision] = snap(env, t=t,
                                               suite_name="loho_public",
                                               task_id=0)
                        auto_states[decision] = {
                            gid: fork_env_state(a)
                            for gid, a in automata.items()}
                        obs_now = copy.deepcopy(obs)
                        canon_auto = automata[canon_id]
                        first_unres = next(
                            (subgoals[i] for i, v in
                             enumerate(canon_auto.prev_valid)
                             if not v), None)
                        stall_count = (stall_count + 1
                                       if first_unres == prev_unres
                                       else 1)
                        prev_unres = first_unres
                        obj_positions = body_positions(
                            env, list(bodies.values()))
                        hit = anchor_ok(
                            first_unres, obj_positions, obj_prev,
                            np.asarray(
                                obs_now["robot_state"]["eef"]["pos"]),
                            stall_count)
                        if hit:
                            anchor_candidates.append(
                                {"decision": decision, **hit})
                        obj_prev = obj_positions
                        batch = runner._obs_to_policy_batch(
                            obs, instruction)
                        prefix = prefix_forward(runner.policy, batch)
                        chunk = sample_chunks(
                            runner.policy, batch, n=1,
                            seed=noise_seed(task_index, seed,
                                            decision),
                            prefix=prefix)
                        actions_env, executed = [], 0
                        for a_env in runner.chunk_to_env(
                                chunk[:, :10]):
                            obs, _r, term, trunc, _i = env.step(a_env)
                            actions_env.append(np.asarray(a_env))
                            t += 1
                            executed += 1
                            for au in automata.values():
                                au.evaluate(env, t)
                            if term or trunc:
                                break
                        rows.append({
                            "decision": decision,
                            "t_start": t - executed,
                            "obs": obs_now,
                            "chunk_norm": chunk[0].float().cpu(),
                            "actions_env": np.stack(actions_env),
                            "executed_len": executed,
                            "q": obs_q(obs_now),
                            "obj_before": obj_positions,
                            "first_unresolved": first_unres,
                        })
                        decision += 1
                        if term or trunc:
                            break
                if not src_path.exists():
                    tmp = src_path.with_suffix(".tmp")
                    torch.save({
                        "schema": "v069_correction_source_v1",
                        "run_schema": "v069",
                        "source_id": source_id, "task": task_name,
                        "task_index": task_index, "seed": seed,
                        "split": split, "provenance": "stock_fresh",
                        "language_canonical": instruction,
                        "n_decisions": len(rows), "rows": rows,
                        "anchor_candidates": anchor_candidates,
                    }, tmp)
                    tmp.replace(src_path)
                print(f"[source] {source_id}: {len(rows)} decisions, "
                      f"{len(anchor_candidates)} anchor candidates, "
                      f"success={bool(term)}", flush=True)

                # ---- pick anchors (chronological, min gap) -------------
                chosen, last_d = [], -10**9
                for c in anchor_candidates:
                    if len(chosen) >= MAX_ANCHORS_PER_SOURCE:
                        break
                    if c["decision"] - last_d >= ANCHOR_MIN_GAP:
                        chosen.append(c)
                        last_d = c["decision"]

                def run_anchor(anchor) -> str:
                    """Returns 'ok' | 'unstable' | 'skip'."""
                    d = anchor["decision"]
                    out_path = (DATA / "corrections_v069"
                                / f"{source_id}_d{d}.pt")
                    if out_path.exists():
                        return "ok"
                    row = rows[d]
                    # proposal chunks
                    batch = runner._obs_to_policy_batch(
                        row["obs"], instruction)
                    prefix = prefix_forward(runner.policy, batch)
                    proposals = [
                        {"kind": "candidate", "key": 0,
                         "provenance": "policy",
                         "behavior_goal_id": canon_id,
                         "chunk": row["chunk_norm"]}]
                    for i in range(1, N_CANON):
                        ch = sample_chunks(
                            runner.policy, batch, n=1,
                            seed=noise_seed(task_index, seed, d)
                            + 100_000 * i, prefix=prefix)
                        proposals.append(
                            {"kind": "candidate", "key": i,
                             "provenance": "policy",
                             "behavior_goal_id": canon_id,
                             "chunk": ch[0].float().cpu()})
                    sub_prompts = (
                        [("atomic", p) for p in atomic_prompts]
                        + [("distinct", lang)
                           for _g, lang in distinct_langs])[:4]
                    for j, (pk, lang) in enumerate(sub_prompts):
                        b2 = runner._obs_to_policy_batch(
                            row["obs"], lang)
                        p2 = prefix_forward(runner.policy, b2)
                        ch = sample_chunks(
                            runner.policy, b2, n=1,
                            seed=noise_seed(task_index, seed, d)
                            + 100_000 * (20 + j), prefix=p2)
                        proposals.append(
                            {"kind": "subgoal", "key": f"sub{j}",
                             "provenance": f"subgoal_{pk}",
                             "behavior_goal_id": lang,
                             "chunk": ch[0].float().cpu()})
                    proposals.append(
                        {"kind": "support", "key": "support",
                         "provenance": "scripted_servo",
                         "behavior_goal_id": "privileged_script",
                         "chunk": None})
                    proposals.append(
                        {"kind": "replay", "key": "replay",
                         "provenance": "replay_audit",
                         "behavior_goal_id": canon_id,
                         "chunk": None})

                    branch_summaries, records = [], []
                    crn_log: dict[tuple, dict] = {}
                    u0_actions, u0_end = None, {}
                    unstable = False
                    for prop in proposals:
                        restore(env, snaps[d])
                        env._env.env.done = False
                        branch_autos = {}
                        for gid in goal_ids:
                            ba = GoalAutomaton(
                                goal_specs[gid]["ordered_subgoals"])
                            base = automata[gid]
                            ba.bodies = base.bodies
                            ba.start_pos = base.start_pos
                            restore_env_state(ba, auto_states[d][gid])
                            branch_autos[gid] = ba
                        flips0 = {gid: len(ba.flips)
                                  for gid, ba in branch_autos.items()}
                        qpos_before = full_qpos(env)
                        term_b = trunc_b = False
                        executed_actions = []
                        steps_b = 0
                        if prop["kind"] == "replay":
                            action_iter = list(u0_actions)
                        elif prop["kind"] == "support":
                            action_iter = None
                        else:
                            action_iter = list(runner.chunk_to_env(
                                prop["chunk"][None, :10].cuda()))
                        for k in range(10):
                            if action_iter is not None:
                                if k >= len(action_iter):
                                    break
                                a_env = action_iter[k]
                            else:
                                a_env = scripted_servo_action(
                                    env, anchor_obj,
                                    anchor.get("region"), bodies,
                                    anchor["form"])
                            _o, _r, tb, tr, _i = env.step(a_env)
                            executed_actions.append(np.asarray(a_env))
                            steps_b += 1
                            for ba in branch_autos.values():
                                ba.evaluate(env, steps_b)
                            if tb:
                                term_b = True
                                env._env.env.done = False
                            if tr:
                                trunc_b = True
                                break
                        if prop["key"] == 0:
                            u0_actions = executed_actions
                        q_after = obs_q(env._format_raw_obs(
                            env._env.env._get_observations()))
                        obj_after = body_positions(
                            env, list(bodies.values()))
                        qpos_after = full_qpos(env)
                        immediate = {}
                        for gid, ba in branch_autos.items():
                            window = ba.flips[flips0[gid]:]
                            before = auto_states[d][gid]
                            immediate[gid] = {
                                "valid_before":
                                    list(before["prev_valid"]),
                                "valid_after": list(ba.prev_valid),
                                "events_before": sorted(
                                    before["events_achieved"]),
                                "events_after": sorted(
                                    ba.events_achieved),
                                "flips_01": [f for f in window
                                             if f[2] == 1],
                                "flips_10": [f for f in window
                                             if f[2] == -1],
                                "ordered_prefix_after":
                                    ba.ordered_prefix(),
                                "q_valid_after": ba.q_valid(),
                                "reward_valid": (
                                    sum(ba.prev_valid)
                                    - sum(before["prev_valid"])),
                            }
                        summary = {
                            "kind": prop["kind"], "key": prop["key"],
                            "provenance": prop["provenance"],
                            "behavior_goal_id":
                                prop["behavior_goal_id"],
                            "chunk_norm": prop["chunk"],
                            "actions_env": np.stack(executed_actions),
                            "steps": steps_b,
                            "term_branch_canonical": term_b,
                            "trunc_branch": trunc_b,
                            "q_after": q_after,
                            "obj_after": obj_after,
                            "qpos_before": qpos_before,
                            "qpos_after": qpos_after,
                            "grasp_after": grasp_state(env, bodies),
                            "immediate": immediate,
                        }
                        if prop["key"] == 0:
                            u0_end = {"q": q_after, "obj": obj_after,
                                      "immediate": immediate}
                        if prop["kind"] == "replay":
                            dq = (q_after - u0_end["q"]).abs()
                            deltas = {
                                "eef_pos": float(dq[:3].max()),
                                "eef_quat": float(dq[3:7].max()),
                                "gripper": float(dq[7:].max()),
                                "obj_pos": float(np.abs(
                                    obj_after
                                    - u0_end["obj"]).max())}
                            exceeds = [k_ for k_ in
                                       ("eef_pos", "eef_quat",
                                        "gripper", "obj_pos")
                                       if deltas[k_] > tol95[k_]]
                            mism = {g_: int(sum(
                                x != y for x, y in zip(
                                    immediate[g_]["valid_after"],
                                    u0_end["immediate"][g_]
                                    ["valid_after"])))
                                for g_ in goal_ids}
                            if any(mism.values()):
                                exceeds.append("valid_bits")
                            summary["replay_endpoint"] = {
                                "deltas": deltas,
                                "valid_bits_mismatch": mism,
                                "exceeds_tolerance": exceeds}
                            unstable = bool(exceeds)
                            for k_ in ("eef_pos", "eef_quat",
                                       "gripper"):
                                assert deltas[k_] <= \
                                    GROSS_FACTOR * tol95[k_], \
                                    f"gross restore failure {k_}"
                            assert deltas["obj_pos"] <= REREACH_ATOL,\
                                "gross restore failure obj_pos"
                        branch_end = snap(env,
                                          t=row["t_start"] + steps_b,
                                          suite_name="loho_public",
                                          task_id=0)

                        # canonical continuations R=2, sibling CRN
                        for repeat in range(R):
                            restore(env, branch_end)
                            env._env.env.done = False
                            auto = GoalAutomaton(
                                goal_specs[canon_id]
                                ["ordered_subgoals"])
                            base = automata[canon_id]
                            auto.bodies = base.bodies
                            auto.start_pos = base.start_pos
                            restore_env_state(auto, fork_env_state(
                                branch_autos[canon_id]))
                            auto.flips = []
                            tracker = SuccessTracker(
                                term_preds[canon_id])
                            obs_c = env._format_raw_obs(
                                env._env.env._get_observations())
                            auto.evaluate(env, 0)
                            tracker.update(env, 0, eval_fn)
                            q_at, crn_entries = {}, []
                            steps_c, cd = 0, 0
                            stop = term_b or trunc_b
                            truncated = trunc_b
                            while steps_c < CONT_MAX and not stop:
                                s_ = cont_seed_v067(
                                    RUN_ID, source_id, d, canon_id,
                                    repeat, cd)
                                nz = flow_noise(s_, chunk_size,
                                                max_dim)
                                sha = noise_sha(nz)
                                keyc = (repeat, cd)
                                if keyc in crn_log:
                                    assert crn_log[keyc] == {
                                        "seed": s_, "sha": sha}
                                else:
                                    crn_log[keyc] = {"seed": s_,
                                                     "sha": sha}
                                crn_entries.append(
                                    {"cont_decision": cd, "seed": s_,
                                     "noise_sha256": sha})
                                bc = runner._obs_to_policy_batch(
                                    obs_c, instruction)
                                pc = prefix_forward(runner.policy, bc)
                                ch = sample_chunks(
                                    runner.policy, bc, n=1,
                                    noise=nz.to("cuda"), prefix=pc)
                                for a_env in runner.chunk_to_env(
                                        ch[:, :10]):
                                    obs_c, _r, tm, tr2, _i = env.step(
                                        a_env)
                                    steps_c += 1
                                    auto.evaluate(env, steps_c)
                                    gt = tracker.update(env, steps_c,
                                                        eval_fn)
                                    assert gt == bool(tm), \
                                        "canonical evaluator != BDDL"
                                    if steps_c in HORIZONS:
                                        q_at[steps_c] = auto.q_valid()
                                    if tm:
                                        stop = True
                                        break
                                    if tr2:
                                        truncated = stop = True
                                        break
                                    if steps_c >= CONT_MAX:
                                        break
                                cd += 1
                            for h in HORIZONS:
                                q_at.setdefault(h, auto.q_valid())
                            records.append({
                                "branch_key": prop["key"],
                                "provenance": prop["provenance"],
                                "goal_spec_id": canon_id,
                                "repeat": repeat,
                                "steps": steps_c,
                                "truncated": truncated,
                                "outcome": {
                                    "success_by_100":
                                        bool(tracker.achieved),
                                    "neg_damage":
                                        -auto.damage_unrecovered(),
                                    "p_valid_100": auto.p_valid(),
                                    "q_valid_mean": sum(
                                        q_at[h] for h in HORIZONS)
                                    / len(HORIZONS),
                                    "neg_tau_next": -auto.tau_next(0),
                                    "q_at_horizons": dict(q_at),
                                    "success_final":
                                        bool(tracker.final),
                                    "first_success_step":
                                        tracker.first_step,
                                    "cont_steps": int(steps_c)},
                                "crn": crn_entries,
                            })
                        branch_summaries.append(summary)
                        print(f"  [fa] {source_id} d={d} "
                              f"{prop['provenance']}:{prop['key']}",
                              flush=True)

                    # judge vs u_0 (frozen tolerances)
                    def outs(key):
                        return [r_["outcome"] for r_ in sorted(
                            (r_ for r_ in records
                             if r_["branch_key"] == key),
                            key=lambda r_: r_["repeat"])]
                    ref = outs(0)
                    judged = {}
                    for prop in proposals:
                        if prop["key"] in (0, "replay"):
                            continue
                        a_i0 = paired_preference(outs(prop["key"]),
                                                 ref, tolerances)
                        judged[str(prop["key"])] = a_i0
                    corrections = [k_ for k_, v in judged.items()
                                   if v == 1 and not unstable]
                    tmp = out_path.with_suffix(".tmp")
                    torch.save({
                        "schema": "v069_corrections_v1",
                        "run_schema": "v069", "run_id": RUN_ID,
                        "source_id": source_id, "task": task_name,
                        "task_index": task_index, "decision": d,
                        "split": split,
                        "anchor": anchor,
                        "goals": goal_ids,
                        "canonical_goal_spec_id": canon_id,
                        "auto_states_snapshot": auto_states[d],
                        "obj_before_snapshot": rows[d]["obj_before"],
                        "qpos_joint_map": qpos_joint_map(env),
                        "branch_summaries": branch_summaries,
                        "records": records,
                        "judged_vs_u0": judged,
                        "replay_unstable": bool(unstable),
                        "grounded_corrections": corrections,
                        "frozen_outcome_tolerances": tolerances,
                        "R": R, "horizons": list(HORIZONS),
                    }, tmp)
                    tmp.replace(out_path)
                    print(f"[anchor] {source_id} d={d} "
                          f"({anchor['form']}): "
                          f"corrections={corrections} "
                          f"unstable={unstable}", flush=True)
                    return "unstable" if unstable else "ok"

                for anchor in chosen:
                    if attempts_by_task[task_name] >= \
                            MAX_ATTEMPTS_PER_TASK:
                        break
                    attempts_by_task[task_name] += 1
                    status = run_anchor(anchor)
                    if status == "unstable":
                        # one retry at the preceding stable candidate
                        prev = [c for c in anchor_candidates
                                if c["decision"] < anchor["decision"]
                                and c not in chosen]
                        if prev and attempts_by_task[task_name] < \
                                MAX_ATTEMPTS_PER_TASK:
                            attempts_by_task[task_name] += 1
                            run_anchor(prev[-1])
            finally:
                env.close()
    print("v069 failure-anchored collection complete", flush=True)
    print(f"attempts: {attempts_by_task}", flush=True)


if __name__ == "__main__":
    main()
