#!/usr/bin/env python
"""V6.7.1/V6.7.2 — rebuild paired continuations + crossed next labels for
the 30 accepted groups, on the REUSED phase-A physical bank.

Repairs relative to iteration 1 (registered in 2026-07-31.md):
- sibling common randomness: continuation seed = stable hash of
  (run_id, source_id, snapshot_decision, goal_spec_id, repeat,
  continuation_decision); branch/candidate identity structurally absent;
  integer seed + full noise-tensor SHA stored for every continuation
  decision; bit-identity asserted across sibling branches;
- GoalSpec-specific terminal success: each GoalSpec's explicit terminal
  predicates (goal_spec_manifest_v067) are evaluated per action;
  success_by_100 = jointly satisfied at ANY point of the continuation;
  the environment BDDL evaluator is consulted only for the canonical
  GoalSpec and asserted equal to the goal evaluator there;
- a canonical env term stops only the canonical-goal continuation; a
  distinct-goal continuation continues (done flag cleared) and stops on
  its OWN GoalSpec success, true truncation, or 100 actions;
- every GoalSpec automaton is evaluated after every environment action
  (flips, milestone times, transient success at action resolution);
- the exact first-ten replay of candidate 0 runs an audit continuation
  (canonical goal, both repeats) under the SAME restored state and CRN
  noise; its branch endpoint must match candidate 0 within the frozen
  tranche-A tolerances; it is provenance `replay_audit`, never ranked;
- immediate crossed labels (valid/event bits, flips, reward components)
  for EVERY compatible GoalSpec from the same stored physical branch.

Output: v06_effect_crossed/continuations_v067/<source>_d<dec>.pt
(run_schema=v067). Iteration-1 continuations are never read.
"""

from __future__ import annotations

import argparse
import hashlib
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
HORIZONS = (10, 30, 60, 100)
R = 2
REREACH_ATOL = 2e-3
CONT_MAX = 100
POLICY_ID = "lerobot/pi05_libero_finetuned"


def load_inputs():
    from lcwm.v067_lineage import assert_v067_payload, sha256_file
    selection = json.loads(
        (RESULTS / "v06_group_selection.json").read_text())
    goal_manifest = json.loads(
        (RESULTS / "goal_spec_manifest_v067.json").read_text())
    assert_v067_payload(goal_manifest, "goal_spec_manifest_v067.json",
                        "goal_manifest")
    lineage = json.loads((DATA / "manifest_v067.json").read_text())
    assert lineage["run_schema"] == "v067"
    # the reused selection is referenced by hash in the lineage manifest
    sel_sha = sha256_file(RESULTS / "v06_group_selection.json")
    assert lineage["reused_json_by_hash"]["v06_group_selection.json"] \
        == sel_sha, "group selection changed since lineage freeze"
    return selection, goal_manifest, lineage


def grasp_state(env, bodies: dict) -> dict:
    """Grasp per tracked object + contact count, where available."""
    out = {"ncon": None, "grasped": {}}
    try:
        inner = env._env.env
        out["ncon"] = int(inner.sim.data.ncon)
        gripper = inner.robots[0].gripper
        objs = getattr(inner, "objects_dict", {})
        for name in bodies:
            obj = objs.get(name)
            if obj is None:
                out["grasped"][name] = None
                continue
            try:
                out["grasped"][name] = bool(
                    inner._check_grasp(gripper=gripper, object_geoms=obj))
            except Exception:
                out["grasped"][name] = None
    except Exception:
        pass
    return out


def outcome_v067(auto, tracker, q_at: dict, cont_steps: int) -> dict:
    """Registered outcome tuple with GOAL-SPECIFIC any-point success."""
    horizons = sorted(q_at)
    return {
        "success_by_100": bool(tracker.achieved),
        "neg_damage": -auto.damage_unrecovered(),
        "p_valid_100": auto.p_valid(),
        "q_valid_mean": sum(q_at[h] for h in horizons) / len(horizons),
        "neg_tau_next": -auto.tau_next(0),
        "q_at_horizons": dict(q_at),
        "success_final": bool(tracker.final),
        "first_success_step": tracker.first_step,
        "cont_steps": int(cont_steps),
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="v067_pb1")
    parser.add_argument("--sources", nargs="*", default=None,
                        help="optional source_id filter")
    args = parser.parse_args()

    from lcwm.chassis import Pi05Runner
    from lcwm.goal_semantics import (SuccessTracker, env_eval_fn,
                                     goal_terminal_success)
    from lcwm.loho_public import make_public_env
    from lcwm.probe_data import body_positions
    from lcwm.sampler import prefix_forward, sample_chunks
    from lcwm.snapshot import snap, restore
    from lcwm.task_automaton import (GoalAutomaton, fork_env_state,
                                     restore_env_state)
    from lcwm.v067_lineage import (RUN_SCHEMA, cont_seed_v067, flow_noise,
                                   noise_sha, sha256_file)
    from scripts.collect_loho_v06 import obs_q, stage_support

    selection, goal_manifest, lineage = load_inputs()
    tol = selection["tolerances_95pct"]
    out_dir = DATA / "continuations_v067"
    out_dir.mkdir(exist_ok=True)

    runner = Pi05Runner(suite_name="libero_10")
    cfg = runner.policy.config
    chunk_size, max_dim = cfg.chunk_size, cfg.max_action_dim

    run_manifest_path = (RESULTS / "v067_lineage"
                         / f"phase_b_manifest_{args.run_id}.json")
    run_manifest = {
        "schema": "v067_phase_b_manifest", "run_schema": RUN_SCHEMA,
        "run_id": args.run_id, "policy_id": POLICY_ID,
        "crn_contract": ("seed = sha256(v067|run_id|source_id|snap_dec|"
                         "goal_spec_id|repeat|cont_dec)[:8] & (2^63-1); "
                         "branch identity structurally absent"),
        "goal_manifest_sha256": goal_manifest["manifest_sha256"],
        "tolerances_95pct": tol, "R": R, "horizons": list(HORIZONS),
        "cont_max_actions": CONT_MAX,
        "code_files": {f: sha256_file(REPO_ROOT / f) for f in (
            "scripts/collect_v067_continuations.py",
            "lcwm/v067_lineage.py", "lcwm/goal_semantics.py",
            "lcwm/task_automaton.py", "lcwm/snapshot.py")},
    }
    payload = json.dumps(run_manifest, sort_keys=True)
    run_manifest["manifest_sha256"] = hashlib.sha256(
        payload.encode()).hexdigest()
    if run_manifest_path.exists():
        prev = json.loads(run_manifest_path.read_text())
        assert prev["manifest_sha256"] == run_manifest["manifest_sha256"],\
            "phase-B manifest hash mismatch on resume"
    else:
        run_manifest_path.write_text(json.dumps(run_manifest, indent=2))

    by_source: dict[str, list[dict]] = {}
    for g in selection["accepted"]:
        by_source.setdefault(g["source_id"], []).append(g)

    for source_id, groups in sorted(by_source.items()):
        if args.sources and source_id not in args.sources:
            continue
        pending = [g for g in groups if not (
            out_dir / f"{source_id}_d{g['decision']}.pt").exists()]
        if not pending:
            print(f"[skip] {source_id}", flush=True)
            continue
        source = torch.load(DATA / "sources" / f"{source_id}.pt",
                            weights_only=False)
        assert source["schema"] == "v06_source_v1"  # reused physical bank
        task_name, task_index = source["task"], source["task_index"]
        entry = goal_manifest["tasks"][task_name]
        canon_id = entry["canonical_goal_spec_id"]
        goal_specs = entry["goal_specs"]
        goal_ids = [canon_id] + [g for g in goal_specs if g != canon_id]
        term_preds = {gid: [tuple(p) for p in
                            goal_specs[gid]["terminal_predicates"]]
                      for gid in goal_ids}

        env = make_public_env(task_name, EPISODE_LENGTH[task_name] + 200)
        try:
            runner.reset()
            obs, _ = env.reset(seed=source["seed"])
            # robosuite horizon is a step budget, not dynamics (measured
            # in iteration 1): raise it so late snapshot + branch +
            # 100-action continuation never hits the silent done flag.
            env._env.env.horizon = EPISODE_LENGTH[task_name] + 300
            eval_fn = env_eval_fn(env)
            automata: dict[str, GoalAutomaton] = {}
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

            # ---- deterministic re-reach, per-action evaluation ---------
            max_dec = max(g["decision"] for g in pending)
            pending_decs = {g["decision"] for g in pending}
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
                if d in pending_decs:
                    err = float(np.abs(body_positions(
                        env, list(bodies.values()))
                        - row["obj_after"]).max())
                    assert err < REREACH_ATOL, (
                        f"{source_id} d={d}: re-reach diverges {err:.2e}")

            for g in sorted(pending, key=lambda x: x["decision"]):
                d = g["decision"]
                audit = next(a for a in source["audits"]
                             if a["decision"] == d)
                row = next(r_ for r_ in source["rows"]
                           if r_["decision"] == d)
                obj_before = body_positions(env, list(bodies.values()))
                # branch_index is provenance labeling only — it NEVER
                # enters the continuation seed.
                branch_list = [(b, {"support": 4, "candidate":
                                    b["candidate"],
                                    "replay": "replay_audit"}[b["kind"]])
                               for b in audit["branches"]]
                records, branch_summaries = [], []
                crn_log: dict[tuple, dict] = {}
                cand0_end = {}
                for branch, branch_key in branch_list:
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
                    obj_b_before = body_positions(
                        env, list(bodies.values()))
                    grasp_before = grasp_state(env, bodies)
                    term_branch = trunc_branch = False
                    steps_b = 0
                    for a_env in branch["actions_env"]:
                        _o, _r, tb, tr, _i = env.step(a_env)
                        steps_b += 1
                        for ba in branch_autos.values():
                            ba.evaluate(env, steps_b)
                        if tb:
                            term_branch = True
                            env._env.env.done = False
                        if tr:
                            trunc_branch = True
                            break
                    q_after = obs_q(env._format_raw_obs(
                        env._env.env._get_observations()))
                    obj_b_after = body_positions(
                        env, list(bodies.values()))
                    grasp_after = grasp_state(env, bodies)
                    branch_end = snap(env, t=row["t_start"] + steps_b,
                                      suite_name="loho_public", task_id=0)

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
                            "flips_01": [f for f in window if f[2] == 1],
                            "flips_10": [f for f in window if f[2] == -1],
                            "ordered_prefix_after": ba.ordered_prefix(),
                            "q_valid_after": ba.q_valid(),
                            "reward_valid": (sum(ba.prev_valid)
                                             - sum(before["prev_valid"])),
                            "reward_milestone": (
                                len(ba.events_achieved)
                                - len(before["events_achieved"])),
                            "terminal_now": goal_terminal_success(
                                env, term_preds[gid], eval_fn),
                        }
                    branch_summaries.append({
                        "kind": branch["kind"],
                        "candidate": branch.get("candidate"),
                        "branch_key": branch_key,
                        "steps": steps_b,
                        "term_branch_canonical": term_branch,
                        "trunc_branch": trunc_branch,
                        "q_after": q_after,
                        "obj_before": obj_b_before,
                        "obj_after": obj_b_after,
                        "grasp_before": grasp_before,
                        "grasp_after": grasp_after,
                        "immediate": immediate,
                    })
                    if branch["kind"] == "candidate" and \
                            branch["candidate"] == 0:
                        cand0_end = {"q": q_after, "obj": obj_b_after}
                    if branch["kind"] == "replay":
                        # frozen tranche-A replay agreement (registered)
                        dq = (q_after - cand0_end["q"]).abs()
                        assert float(dq[:3].max()) <= tol["eef_pos"], \
                            f"{source_id} d={d} replay eef_pos"
                        assert float(dq[3:7].max()) <= tol["eef_quat"], \
                            f"{source_id} d={d} replay eef_quat"
                        assert float(dq[7:].max()) <= tol["gripper"], \
                            f"{source_id} d={d} replay gripper"
                        assert float(np.abs(
                            obj_b_after - cand0_end["obj"]).max()) \
                            <= tol["obj_pos"], \
                            f"{source_id} d={d} replay obj_pos"

                    # ---- continuations under every compatible GoalSpec -
                    if branch["kind"] == "replay":
                        cont_goals = [canon_id]        # audit only
                        provenance = "replay_audit"
                    elif branch["kind"] == "support":
                        cont_goals = goal_ids
                        provenance = "support"
                    else:
                        cont_goals = goal_ids
                        provenance = "policy"
                    for gid in cont_goals:
                        language = goal_specs[gid]["language"]
                        for repeat in range(R):
                            restore(env, branch_end)
                            env._env.env.done = False
                            auto = GoalAutomaton(
                                goal_specs[gid]["ordered_subgoals"])
                            base = automata[gid]
                            auto.bodies, auto.start_pos = base.bodies, \
                                base.start_pos
                            restore_env_state(
                                auto, fork_env_state(branch_autos[gid]))
                            # continuation-relative flip window
                            # (registered in iteration 1, kept)
                            auto.flips = []
                            tracker = SuccessTracker(term_preds[gid])
                            obs_c = env._format_raw_obs(
                                env._env.env._get_observations())
                            auto.evaluate(env, 0)
                            tracker.update(env, 0, eval_fn)
                            q_at = {}
                            steps_c, cd = 0, 0
                            stop = term_branch and gid == canon_id
                            stop = stop or trunc_branch
                            truncated = trunc_branch
                            canon_term_steps = []
                            crn_entries = []
                            while steps_c < CONT_MAX and not stop \
                                    and not tracker.final:
                                seed = cont_seed_v067(
                                    args.run_id, source_id, d, gid,
                                    repeat, cd)
                                noise = flow_noise(seed, chunk_size,
                                                   max_dim)
                                sha = noise_sha(noise)
                                key = (gid, repeat, cd)
                                if key in crn_log:
                                    assert crn_log[key] == {
                                        "seed": seed, "sha": sha}, (
                                        "sibling CRN violated at "
                                        f"{source_id} d={d} {key}")
                                else:
                                    crn_log[key] = {"seed": seed,
                                                    "sha": sha}
                                crn_entries.append(
                                    {"cont_decision": cd, "seed": seed,
                                     "noise_sha256": sha})
                                batch = runner._obs_to_policy_batch(
                                    obs_c, language)
                                prefix = prefix_forward(
                                    runner.policy, batch)
                                chunk = sample_chunks(
                                    runner.policy, batch, n=1,
                                    noise=noise.to("cuda"),
                                    prefix=prefix)
                                for a_env in runner.chunk_to_env(
                                        chunk[:, :10]):
                                    obs_c, _r, term, trunc, _i = \
                                        env.step(a_env)
                                    steps_c += 1
                                    auto.evaluate(env, steps_c)
                                    goal_term = tracker.update(
                                        env, steps_c, eval_fn)
                                    if gid == canon_id:
                                        assert goal_term == bool(term), (
                                            f"{source_id} d={d} step "
                                            f"{steps_c}: canonical "
                                            "evaluator != env BDDL")
                                    if steps_c in HORIZONS:
                                        q_at[steps_c] = auto.q_valid()
                                    if term:
                                        canon_term_steps.append(steps_c)
                                        if gid == canon_id:
                                            stop = True
                                            break
                                        env._env.env.done = False
                                    if trunc:
                                        truncated = stop = True
                                        break
                                    if tracker.final and gid != canon_id:
                                        stop = True
                                        break
                                    if steps_c >= CONT_MAX:
                                        break
                                cd += 1
                            for h in HORIZONS:
                                q_at.setdefault(h, auto.q_valid())
                            records.append({
                                "branch_kind": branch["kind"],
                                "candidate": branch.get("candidate"),
                                "branch_key": branch_key,
                                "goal_spec_id": gid,
                                "goal_language": language,
                                "terminal_predicate_hash":
                                    goal_specs[gid][
                                        "terminal_predicate_hash"],
                                "prompt_sha256":
                                    goal_specs[gid]["prompt_sha256"],
                                "repeat": repeat,
                                "provenance": provenance,
                                "steps": steps_c,
                                "truncated": truncated,
                                "zero_step_terminal_branch":
                                    bool(term_branch and
                                         gid == canon_id and
                                         steps_c == 0),
                                "canonical_term_steps": canon_term_steps,
                                "automaton_after": fork_env_state(auto),
                                "outcome": outcome_v067(
                                    auto, tracker, q_at, steps_c),
                                "crn": crn_entries,
                            })
                    print(f"  [cont] {source_id} d={d} "
                          f"branch={branch['kind']}"
                          f"{branch.get('candidate')}", flush=True)

                # sibling CRN audit across ALL branches of this group
                for (gid, repeat, cd), entry_ in sorted(crn_log.items()):
                    seeds = {r_["crn"][cd]["seed"] for r_ in records
                             if r_["goal_spec_id"] == gid
                             and r_["repeat"] == repeat
                             and len(r_["crn"]) > cd}
                    assert len(seeds) <= 1, (
                        f"sibling seed divergence {source_id} d={d} "
                        f"{(gid, repeat, cd)}")
                torch.save({
                    "schema": "v067_continuations_v1",
                    "run_schema": RUN_SCHEMA, "run_id": args.run_id,
                    "source_id": source_id, "task": task_name,
                    "task_index": task_index,
                    "decision": d, "slot": g["slot"],
                    "split": source["split"],
                    "provenance_source": source["provenance"],
                    "goals": goal_ids,
                    "canonical_goal_spec_id": canon_id,
                    "goal_manifest_sha256":
                        goal_manifest["manifest_sha256"],
                    "policy_id": POLICY_ID,
                    "obj_before_snapshot": obj_before,
                    "auto_states_snapshot": auto_states[d],
                    "branch_summaries": branch_summaries,
                    "records": records,
                    "R": R, "horizons": list(HORIZONS),
                    "run_manifest_sha256":
                        run_manifest["manifest_sha256"],
                }, out_dir / f"{source_id}_d{d}.pt")
                print(f"[group] {source_id} d={d}: "
                      f"{len(records)} continuation records", flush=True)
        finally:
            env.close()
    print("v067 phase B complete", flush=True)


if __name__ == "__main__":
    main()
