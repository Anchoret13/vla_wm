#!/usr/bin/env python
"""V7.1.1 — materialize accumulated decision-transition replay.

Replay unit = one policy decision (c=10) with raw observations before
every executed action and after the last. Bindings registered in
2026-08-02.md before launch.

Phases:
  bank    — re-reach all 26 correction anchors; execute every stored
            proposal branch once with per-action obs capture; EVERY
            anchor gets a u_0 fidelity repeat (stored env actions
            twice) with measured endpoint deltas (the V7 strict-rule
            physical deltas; V7.0 piggyback). At the 3
            legacy-calibrated dev positive anchors, fresh u_0 and the
            stored winner get R=2 canonical continuations under fresh
            v071 SHA-CRNs.
  acquire — frozen budget: per task one fresh train source
            (seed 2200+10k), stock rollout, ≤4 anchors by the amended
            reachability rule over ALL registered subgoals with
            state-type diversity (stall/recovery/late_chain/
            milestone_boundary); proposals u_0 + 7 canonical +
            2 atomic + 2 distinct + servo + u_0 fidelity repeat;
            per-action obs; R=2 canonical continuations for the 12
            policy-eligible branches.

Output root: results/libero_loho_public_v1/<DATE>_v071_replay_r1/
(DATE = run start in America/Chicago, frozen in the manifest).
"""

from __future__ import annotations

import argparse
import copy
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

DATA = Path("/home/stargazer/Desktop/vla_wm/datasets/libero_loho_public_v1"
            "/v06_effect_crossed")
RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
STAGE = "replay"
# bank ran as r1 (frozen); the acquire phase carries gen/servo code
# additions = a new configuration = a new run_id per the registered
# date-root contract
PHASE_RUN_ID = {"bank": "r1", "acquire": "r2"}
TASK_ORDER = ["loho_t1_drawer", "loho_t2_basket3", "loho_t3_tray",
              "loho_t4_tray", "loho_t5_drawer_cabinet"]
EPISODE_LENGTH = {"loho_t1_drawer": 700, "loho_t2_basket3": 900,
                  "loho_t3_tray": 900, "loho_t4_tray": 900,
                  "loho_t5_drawer_cabinet": 990}
NOISE_BASE = 40_000_000
HORIZONS = (10, 30, 60, 100)
R = 2
CONT_MAX = 100
REREACH_ATOL = 2e-3
N_EXTRA_CANON = 7
MAX_ANCHORS_ACQ = 4
ANCHOR_MIN_GAP = 5
PICK_NEAR, PICK_FAR = 0.03, 0.30
PLACE_NEAR, PLACE_FAR = 0.05, 0.30
STABLE_MM = 0.005
STALL_DECS = 3
DEV_ORACLE_ANCHORS = {
    "loho_t2_basket3_correction_s2112_d75": ["sub0", "sub1", "sub2"],
    "loho_t3_tray_correction_s2122_d2": ["sub1", "support"],
    "loho_t5_drawer_cabinet_correction_s2142_d2": ["sub0"],
}


def run_date_chicago() -> str:
    return subprocess.run(
        ["date", "+%F"], capture_output=True, text=True,
        env={"TZ": "America/Chicago"}).stdout.strip()


def sha_seed(payload: str) -> int:
    return int.from_bytes(hashlib.sha256(
        payload.encode()).digest()[:8], "big") & ((1 << 63) - 1)


def obs_frame(env):
    return copy.deepcopy(env._format_raw_obs(
        env._env.env._get_observations()))


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("bank", "acquire"),
                        required=True)
    args = parser.parse_args()
    global RUN_ID
    RUN_ID = PHASE_RUN_ID[args.phase]

    from lcwm.chassis import Pi05Runner
    from lcwm.goal_semantics import SuccessTracker, env_eval_fn
    from lcwm.loho_public import make_public_env
    from lcwm.probe_data import body_positions
    from lcwm.sampler import prefix_forward, sample_chunks
    from lcwm.snapshot import restore, snap
    from lcwm.task_automaton import (GoalAutomaton, fork_env_state,
                                     paired_preference,
                                     restore_env_state)
    from lcwm.v067_lineage import flow_noise, noise_sha, sha256_file
    from scripts.collect_loho_v06 import obs_q
    from scripts.collect_v067_continuations import (full_qpos,
                                                    grasp_state,
                                                    qpos_joint_map)
    from scripts.collect_v069_corrections import scripted_servo_action

    goal_manifest = json.loads(
        (RESULTS / "goal_spec_manifest_v067.json").read_text())
    support = json.loads(
        (RESULTS / "v067_support_report.json").read_text())
    tolerances = support["frozen_outcome_tolerances"]
    git_sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True,
                             cwd=REPO_ROOT).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain", "-uno"], capture_output=True,
        text=True, cwd=REPO_ROOT).stdout.strip())

    root = None
    for p in sorted(RESULTS.glob(f"*_v071_{STAGE}_{RUN_ID}")):
        root = p
    if root is None:
        root = RESULTS / f"{run_date_chicago()}_v071_{STAGE}_{RUN_ID}"
    root.mkdir(exist_ok=True)
    (root / "shards").mkdir(exist_ok=True)
    manifest_path = root / "run_manifest.json"
    manifest = {
        "schema": "v071_replay_manifest_v1", "run_schema": "v071",
        "run_id": RUN_ID, "stage": STAGE,
        "run_date": root.name.split("_")[0],
        "timezone": "America/Chicago",
        "git_sha": git_sha, "dirty_worktree": dirty,
        "crn_contract": ("v071_replay_r1|cont|src|dec|gid|r|c and "
                         "|action| SHA signatures; arm/candidate "
                         "identity structurally absent"),
        "acquisition_budget": {
            "sources_per_task": 1, "seed_rule": "2200+10*task",
            "max_anchors_per_source": MAX_ANCHORS_ACQ,
            "proposals": "u0+7canon+2atomic+2distinct+servo+fidelity"},
        "goal_manifest_sha256": goal_manifest["manifest_sha256"],
        "frozen_outcome_tolerances": tolerances,
        "code_files": {f: sha256_file(REPO_ROOT / f) for f in (
            "scripts/materialize_v071_replay.py",
            "scripts/collect_v069_corrections.py",
            "lcwm/task_automaton.py", "lcwm/goal_semantics.py",
            "lcwm/snapshot.py")},
    }
    payload = json.dumps(manifest, sort_keys=True)
    manifest["manifest_sha256"] = hashlib.sha256(
        payload.encode()).hexdigest()
    if manifest_path.exists():
        prev = json.loads(manifest_path.read_text())
        assert prev["manifest_sha256"] == manifest["manifest_sha256"],\
            "immutable manifest mismatch — start a new run_id"
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2))

    runner = Pi05Runner(suite_name="libero_10")
    cfg = runner.policy.config

    idx_path = root / "physical_transition_index.jsonl"
    fid_path = root / "replay_fidelity.jsonl"
    rel_path = root / "semantic_relabels.jsonl"
    gs_path = root / "goal_specs.jsonl"
    if not gs_path.exists():
        with gs_path.open("w") as f:
            for task, entry in goal_manifest["tasks"].items():
                for gid, spec in entry["goal_specs"].items():
                    f.write(json.dumps({
                        "task": task, "goal_id": gid,
                        "language": spec["language"],
                        "terminal_predicate_hash":
                            spec["terminal_predicate_hash"],
                        "ordered_subgoals":
                            spec["ordered_subgoals"]}) + "\n")

    done_shards = {p.stem for p in (root / "shards").glob("*.pt")}

    def execute_branch(env, runner, actions_or_chunk, mode, autos,
                       canon_id, eval_fn, term_preds):
        """Execute ≤10 actions with per-action obs + predicate capture.
        Returns transition payload pieces."""
        frames = [obs_frame(env)]
        eef_seq = [obs_q(frames[0])]
        valid_seq = {gid: [list(a.prev_valid)]
                     for gid, a in autos.items()}
        actions_env = []
        term_b = trunc_b = False
        if mode == "actions":
            it = list(actions_or_chunk)[:10]
        elif mode == "servo":
            it = None                      # closed-loop scripted servo
        else:
            it = list(runner.chunk_to_env(
                actions_or_chunk[:, :10]))[:10]
        flips0 = {gid: len(a.flips) for gid, a in autos.items()}
        steps = 0
        any_bodies = next(iter(autos.values())).bodies
        for k in range(10):
            if it is None:
                s_obj, s_region, s_form = actions_or_chunk
                a_env = scripted_servo_action(
                    env, s_obj, s_region, any_bodies, s_form)
            elif k >= len(it):
                break
            else:
                a_env = it[k]
            _o, _r, tb, tr, _i = env.step(a_env)
            steps += 1
            actions_env.append(np.asarray(a_env))
            for gid, a in autos.items():
                a.evaluate(env, steps)
                valid_seq[gid].append(list(a.prev_valid))
            frames.append(obs_frame(env))
            eef_seq.append(obs_q(frames[-1]))
            if tb:
                term_b = True
                env._env.env.done = False
            if tr:
                trunc_b = True
                break
        imm = {}
        for gid, a in autos.items():
            window = a.flips[flips0[gid]:]
            imm[gid] = {
                "valid_after": list(a.prev_valid),
                "events_after": sorted(a.events_achieved),
                "flips_01": [f for f in window if f[2] == 1],
                "flips_10": [f for f in window if f[2] == -1],
                "ordered_prefix_after": a.ordered_prefix(),
                "q_valid_after": a.q_valid(),
            }
        return {"frames": frames, "eef_seq": eef_seq,
                "valid_seq_canon": valid_seq[canon_id],
                "actions_env": (np.stack(actions_env)
                                if actions_env else
                                np.zeros((0, 7))),
                "steps": steps, "term": term_b, "trunc": trunc_b,
                "immediate": imm}

    def run_continuation(env, runner, src_lang, auto_proto, canon_id,
                         term_preds, eval_fn, crn_prefix, rep):
        auto = auto_proto
        auto.flips = []
        tracker = SuccessTracker(term_preds)
        obs_c = obs_frame(env)
        auto.evaluate(env, 0)
        tracker.update(env, 0, eval_fn)
        q_at, steps_c, cd = {}, 0, 0
        stop = False
        while steps_c < CONT_MAX and not stop:
            seed = sha_seed(f"{crn_prefix}|{rep}|{cd}")
            nz = flow_noise(seed, cfg.chunk_size, cfg.max_action_dim)
            bc = runner._obs_to_policy_batch(obs_c, src_lang)
            pc = prefix_forward(runner.policy, bc)
            ch = sample_chunks(runner.policy, bc, n=1,
                               noise=nz.to("cuda"), prefix=pc)
            for a_env in runner.chunk_to_env(ch[:, :10]):
                obs_c, _r, tm, tr2, _i = env.step(a_env)
                steps_c += 1
                auto.evaluate(env, steps_c)
                gt = tracker.update(env, steps_c, eval_fn)
                assert gt == bool(tm)
                if steps_c in HORIZONS:
                    q_at[steps_c] = auto.q_valid()
                if tm or tr2:
                    stop = True
                    break
                if steps_c >= CONT_MAX:
                    break
            cd += 1
        for h in HORIZONS:
            q_at.setdefault(h, auto.q_valid())
        return {"success_by_100": bool(tracker.achieved),
                "neg_damage": -auto.damage_unrecovered(),
                "p_valid_100": auto.p_valid(),
                "q_valid_mean": sum(q_at[h] for h in HORIZONS)
                / len(HORIZONS),
                "neg_tau_next": -auto.tau_next(0),
                "q_at_horizons": dict(q_at),
                "cont_steps": steps_c}

    def process_anchor(task_name, source, d, branch_specs, akey,
                       split, dev_oracle_keys=None):
        """Re-reach to d, then execute every branch spec; save shard."""
        if akey in done_shards:
            print(f"[skip] {akey}", flush=True)
            return
        entry = goal_manifest["tasks"][task_name]
        canon_id = entry["canonical_goal_spec_id"]
        goal_specs = entry["goal_specs"]
        goal_ids = [canon_id] + [g for g in goal_specs
                                 if g != canon_id]
        term_preds = {gid: [tuple(p) for p in
                            goal_specs[gid]["terminal_predicates"]]
                      for gid in goal_ids}
        env = make_public_env(task_name,
                              EPISODE_LENGTH[task_name] + 200)
        try:
            runner.reset()
            env.reset(seed=source["seed"])
            env._env.env.horizon = EPISODE_LENGTH[task_name] + 300
            eval_fn = env_eval_fn(env)
            automata = {}
            for gid in goal_ids:
                a = GoalAutomaton(goal_specs[gid]["ordered_subgoals"])
                a.start(env)
                automata[gid] = a
            for a in automata.values():
                a.evaluate(env, 0)
            bodies = automata[canon_id].bodies
            t = 0
            for i, row in enumerate(source["rows"]):
                if i >= d:
                    break
                for a_env in row["actions_env"]:
                    env.step(a_env)
                    t += 1
                    for a in automata.values():
                        a.evaluate(env, t)
            row = source["rows"][d]
            err = float(np.abs(body_positions(
                env, list(bodies.values()))
                - row["obj_before"]).max())
            assert err < REREACH_ATOL, f"{akey}: re-reach {err:.2e}"
            snap_a = snap(env, t=t, suite_name="loho_public",
                          task_id=0)
            anchor_states = {gid: fork_env_state(a)
                             for gid, a in automata.items()}
            anchor_obs = obs_frame(env)
            for spec in branch_specs:
                if spec["mode"] == "fresh_chunk":
                    batch = runner._obs_to_policy_batch(
                        anchor_obs, source["language_canonical"])
                    prefix = prefix_forward(runner.policy, batch)
                    spec["payload"] = sample_chunks(
                        runner.policy, batch, n=1,
                        noise=spec["payload"].to("cuda"),
                        prefix=prefix)
                    spec["mode"] = "chunk"
                elif spec["mode"] == "gen":
                    _kind, lang, gseed = spec["gen"]
                    batch = runner._obs_to_policy_batch(
                        anchor_obs, lang)
                    prefix = prefix_forward(runner.policy, batch)
                    ch = sample_chunks(runner.policy, batch, n=1,
                                       seed=gseed, prefix=prefix)
                    spec["payload"] = ch
                    spec["chunk_norm"] = ch[0].float().cpu()
                    spec["mode"] = "chunk"
                elif spec["mode"] == "servo":
                    spec["payload"] = spec["servo"]

            transitions, fid_rows, rel_rows = [], [], []
            u0_endpoint = {}
            for spec in branch_specs:
                restore(env, snap_a)
                env._env.env.done = False
                autos = {}
                for gid in goal_ids:
                    ba = GoalAutomaton(
                        goal_specs[gid]["ordered_subgoals"])
                    base = automata[gid]
                    ba.bodies, ba.start_pos = base.bodies, \
                        base.start_pos
                    restore_env_state(ba, anchor_states[gid])
                    autos[gid] = ba
                res = execute_branch(env, runner, spec["payload"],
                                     spec["mode"], autos, canon_id,
                                     eval_fn, term_preds)
                tid = f"{akey}_{spec['key']}"
                obj_after = body_positions(env, list(bodies.values()))
                qpos_after = full_qpos(env)
                if spec["key"] == "u0":
                    u0_endpoint = {"eef": res["eef_seq"][-1],
                                   "obj": obj_after,
                                   "qpos": qpos_after}
                if spec["key"] == "u0_rep2":
                    dq = (res["eef_seq"][-1]
                          - u0_endpoint["eef"]).abs()
                    fid_rows.append({
                        "anchor": akey,
                        "eef_pos": float(dq[:3].max()),
                        "eef_quat": float(dq[3:7].max()),
                        "gripper": float(dq[7:].max()),
                        "obj_pos": float(np.abs(
                            obj_after - u0_endpoint["obj"]).max()),
                        "qpos": float(np.abs(
                            qpos_after
                            - u0_endpoint["qpos"]).max())})
                transitions.append({
                    "transition_id": tid, "anchor": akey,
                    "task": task_name, "split": split,
                    "source_id": source["source_id"], "decision": d,
                    "branch_key": spec["key"],
                    "family": spec["family"],
                    "provenance": spec["provenance"],
                    "behavior_goal_id": spec["behavior_goal_id"],
                    "chunk_norm": spec.get("chunk_norm"),
                    **{k: res[k] for k in
                       ("frames", "eef_seq", "valid_seq_canon",
                        "actions_env", "steps", "term", "trunc")},
                    "obj_after": obj_after, "qpos_after": qpos_after,
                    "grasp_after": grasp_state(env, bodies),
                    "immediate_canonical":
                        res["immediate"][canon_id],
                    "continuations": [],
                })
                for gid in goal_ids:
                    rel_rows.append({
                        "transition_id": tid, "goal_id": gid,
                        "immediate": res["immediate"][gid]})
                # continuations for eligible branches
                if spec.get("continuations"):
                    branch_end = snap(env, t=t + res["steps"],
                                      suite_name="loho_public",
                                      task_id=0)
                    for rep in range(R):
                        restore(env, branch_end)
                        env._env.env.done = False
                        ca = GoalAutomaton(
                            goal_specs[canon_id]["ordered_subgoals"])
                        base = automata[canon_id]
                        ca.bodies, ca.start_pos = base.bodies, \
                            base.start_pos
                        restore_env_state(ca, fork_env_state(
                            autos[canon_id]))
                        out = run_continuation(
                            env, runner,
                            source["language_canonical"], ca,
                            canon_id, term_preds[canon_id], eval_fn,
                            f"v071_{STAGE}_{RUN_ID}|cont|"
                            f"{source['source_id']}|{d}|{canon_id}",
                            rep)
                        transitions[-1]["continuations"].append(out)
                print(f"  [{akey}] {spec['key']}", flush=True)

            # dev-oracle piggyback judgment
            dev_judged = None
            if dev_oracle_keys:
                by_key = {tr["branch_key"]: tr for tr in transitions}
                ref = by_key["u0_fresh"]["continuations"]
                dev_judged = {}
                for k in dev_oracle_keys:
                    kk = f"stored_{k}"
                    if kk in by_key:
                        dev_judged[k] = paired_preference(
                            by_key[kk]["continuations"], ref,
                            tolerances)
            shard = {"schema": "v071_transition_shard_v1",
                     "run_schema": "v071", "anchor": akey,
                     "task": task_name, "split": split,
                     "decision": d,
                     "source_seed": source["seed"],
                     "manifest_sha256": manifest["manifest_sha256"],
                     "anchor_states": anchor_states,
                     "qpos_joint_map": qpos_joint_map(env),
                     "transitions": transitions,
                     "dev_oracle_fresh_judged": dev_judged}
            tmp = (root / "shards" / f"{akey}.tmp")
            torch.save(shard, tmp)
            tmp.replace(root / "shards" / f"{akey}.pt")
            with idx_path.open("a") as f:
                for tr in transitions:
                    f.write(json.dumps({
                        "transition_id": tr["transition_id"],
                        "anchor": akey, "task": task_name,
                        "split": split,
                        "branch_key": str(tr["branch_key"]),
                        "family": tr["family"],
                        "provenance": tr["provenance"],
                        "steps": tr["steps"],
                        "n_continuations":
                            len(tr["continuations"])}) + "\n")
            with rel_path.open("a") as f:
                for r_ in rel_rows:
                    f.write(json.dumps(
                        {k: (v if k != "immediate" else v)
                         for k, v in r_.items()},
                        default=str) + "\n")
            with fid_path.open("a") as f:
                for r_ in fid_rows:
                    f.write(json.dumps(r_) + "\n")
            if dev_judged is not None:
                print(f"[dev-oracle] {akey}: {dev_judged}",
                      flush=True)
            print(f"[shard] {akey}: {len(transitions)} transitions",
                  flush=True)
        finally:
            env.close()

    if args.phase == "bank":
        for p in sorted((DATA / "corrections_v069").glob("*.pt")):
            grp = torch.load(p, weights_only=False)
            akey = f"{grp['source_id']}_d{grp['decision']}"
            source = torch.load(
                DATA / "corrections_sources_v069"
                / f"{grp['source_id']}.pt", weights_only=False)
            canon_id = grp["canonical_goal_spec_id"]
            specs = []
            u0_actions = None
            for b in grp["branch_summaries"]:
                key = str(b["key"])
                if b["kind"] == "replay":
                    continue
                fam = ("canonical" if key.isdigit() else
                       "atomic" if key in ("sub0", "sub1") else
                       "distinct" if key in ("sub2", "sub3") else
                       "servo")
                if key == "0":
                    key = "u0"
                    u0_actions = b["actions_env"]
                specs.append({
                    "key": key, "family": fam,
                    "provenance": b["provenance"],
                    "behavior_goal_id": b["behavior_goal_id"],
                    "chunk_norm": b.get("chunk_norm"),
                    "payload": b["actions_env"], "mode": "actions",
                    "continuations": False})
            specs.append({"key": "u0_rep2", "family": "canonical",
                          "provenance": "fidelity_repeat",
                          "behavior_goal_id": canon_id,
                          "payload": u0_actions, "mode": "actions",
                          "continuations": False})
            dev_keys = None
            if akey in DEV_ORACLE_ANCHORS:
                dev_keys = DEV_ORACLE_ANCHORS[akey]
                a_seed = sha_seed(
                    f"v071_{STAGE}_{RUN_ID}|action|"
                    f"{grp['source_id']}|{grp['decision']}")
                nz = flow_noise(a_seed, cfg.chunk_size,
                                cfg.max_action_dim)
                specs.append({
                    "key": "u0_fresh", "family": "canonical",
                    "provenance": "fresh_u0",
                    "behavior_goal_id": canon_id,
                    "fresh_noise": (a_seed, noise_sha(nz)),
                    "payload": nz, "mode": "fresh_chunk",
                    "continuations": True})
                by_key = {str(b["key"]): b
                          for b in grp["branch_summaries"]}
                for k in dev_keys:
                    specs.append({
                        "key": f"stored_{k}",
                        "family": ("servo" if k == "support"
                                   else "atomic"
                                   if k in ("sub0", "sub1")
                                   else "distinct"),
                        "provenance": "stored_dev_winner",
                        "behavior_goal_id":
                            by_key[k]["behavior_goal_id"],
                        "payload": by_key[k]["actions_env"],
                        "mode": "actions", "continuations": True})
            process_anchor(grp["task"], source, grp["decision"],
                           specs, akey, grp["split"],
                           dev_oracle_keys=dev_keys)
    else:
        # ---- ACQUIRE: frozen budget, fresh train sources ----------------
        from scripts.collect_v069_corrections import (
            OBJ_DISPLAY, REGION_DISPLAY, scripted_servo_action)
        (root / "acquire_sources").mkdir(exist_ok=True)

        def display_obj(task, obj):
            return OBJ_DISPLAY.get((task, obj),
                                   "the " + obj.rsplit("_", 1)[0]
                                   .replace("_", " "))

        for task_index, task_name in enumerate(TASK_ORDER):
            seed = 2200 + 10 * task_index
            source_id = f"{task_name}_acquire_s{seed}"
            src_path = root / "acquire_sources" / f"{source_id}.pt"
            entry = goal_manifest["tasks"][task_name]
            canon_id = entry["canonical_goal_spec_id"]
            goal_specs = entry["goal_specs"]
            goal_ids = [canon_id] + [g for g in goal_specs
                                     if g != canon_id]
            subgoals = goal_specs[canon_id]["ordered_subgoals"]
            distinct_langs = [goal_specs[g]["language"]
                              for g in goal_ids if g != canon_id][:2]
            env = make_public_env(task_name,
                                  EPISODE_LENGTH[task_name] + 200)
            try:
                runner.reset()
                obs, _ = env.reset(seed=seed)
                env._env.env.horizon = EPISODE_LENGTH[task_name] + 300
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
                canon_auto = automata[canon_id]

                if src_path.exists():
                    saved = torch.load(src_path, weights_only=False)
                    rows = saved["rows"]
                    cands = saved["anchor_candidates"]
                else:
                    rows, cands = [], []
                    obj_prev = body_positions(
                        env, list(bodies.values()))
                    stall, prev_unres = 0, "___"
                    t, decision = 0, 0
                    term = trunc = False
                    while t < EPISODE_LENGTH[task_name]:
                        obs_now = copy.deepcopy(obs)
                        first_unres = next(
                            (subgoals[i] for i, v in enumerate(
                                canon_auto.prev_valid) if not v),
                            None)
                        stall = (stall + 1
                                 if first_unres == prev_unres else 1)
                        prev_unres = first_unres
                        obj_positions = body_positions(
                            env, list(bodies.values()))
                        recent = [f for f in canon_auto.flips
                                  if f[0] > t - 20]
                        n_unres = sum(
                            1 for v in canon_auto.prev_valid
                            if not v)
                        state_type = None
                        if any(f[2] == -1 for f in recent):
                            state_type = "recovery"
                        elif n_unres <= 2 and n_unres > 0:
                            state_type = "late_chain"
                        elif any(f[2] == 1 for f in recent):
                            state_type = "milestone_boundary"
                        elif stall >= STALL_DECS:
                            state_type = "stall"
                        ok = None
                        if state_type and first_unres:
                            parts = first_unres.split()
                            form = parts[0]
                            obj = parts[1]
                            gsp = grasp_state(
                                env, bodies)["grasped"]
                            eef = np.asarray(
                                obs_now["robot_state"]["eef"]
                                ["pos"])
                            opos = body_positions(
                                env, [bodies[obj]])[0] \
                                if obj in bodies else None
                            moved = float(np.abs(
                                obj_positions - obj_prev).max())
                            if form == "pick_up" and opos is not None:
                                dist = float(np.linalg.norm(
                                    eef - opos))
                                if (not gsp.get(obj)
                                        and PICK_NEAR <= dist
                                        <= PICK_FAR
                                        and moved < STABLE_MM):
                                    ok = {"form": form, "obj": obj,
                                          "region": None,
                                          "dist": dist}
                            elif form in ("place", "close", "open"):
                                region = (parts[2] if form == "place"
                                          else parts[1])
                                try:
                                    tgt = env._env.env.sim.data \
                                        .get_site_xpos(region).copy()
                                except Exception:
                                    tgt = None
                                ref = (opos if form == "place"
                                       and opos is not None else eef)
                                if tgt is not None:
                                    dist = float(np.linalg.norm(
                                        ref - tgt))
                                    held_ok = (bool(gsp.get(obj))
                                               if form == "place"
                                               else True)
                                    if held_ok and PLACE_NEAR <= \
                                            dist <= PLACE_FAR:
                                        ok = {"form": form,
                                              "obj": obj,
                                              "region": region,
                                              "dist": dist}
                        if ok:
                            cands.append({"decision": decision,
                                          "state_type": state_type,
                                          **ok})
                        obj_prev = obj_positions
                        batch = runner._obs_to_policy_batch(
                            obs, instruction)
                        prefix = prefix_forward(runner.policy, batch)
                        chunk = sample_chunks(
                            runner.policy, batch, n=1,
                            seed=NOISE_BASE
                            + task_index * 2_000_000
                            + seed * 1_000 + decision,
                            prefix=prefix)
                        actions_env, executed = [], 0
                        for a_env in runner.chunk_to_env(
                                chunk[:, :10]):
                            obs, _r, term, trunc, _i = env.step(
                                a_env)
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
                    tmp = src_path.with_suffix(".tmp")
                    torch.save({
                        "schema": "v071_acquire_source_v1",
                        "run_schema": "v071",
                        "source_id": source_id, "task": task_name,
                        "task_index": task_index, "seed": seed,
                        "split": "train",
                        "language_canonical": instruction,
                        "rows": rows,
                        "anchor_candidates": cands}, tmp)
                    tmp.replace(src_path)
                    print(f"[acq-source] {source_id}: {len(rows)} "
                          f"decisions, {len(cands)} candidates "
                          f"({[c['state_type'] for c in cands[:8]]})",
                          flush=True)
            finally:
                env.close()

            # diversity-first anchor selection
            chosen, last_d, seen_types = [], -10**9, set()
            for want_new_type in (True, False):
                for c in cands:
                    if len(chosen) >= MAX_ANCHORS_ACQ:
                        break
                    if c in chosen or \
                            c["decision"] - last_d < ANCHOR_MIN_GAP:
                        continue
                    if want_new_type and \
                            c["state_type"] in seen_types:
                        continue
                    chosen.append(c)
                    seen_types.add(c["state_type"])
                    last_d = c["decision"]

            source = torch.load(src_path, weights_only=False)
            for c in chosen:
                d = c["decision"]
                akey = f"{source_id}_d{d}"
                if akey in done_shards:
                    continue
                row = source["rows"][d]
                specs = [{"key": "u0", "family": "canonical",
                          "provenance": "policy",
                          "behavior_goal_id": canon_id,
                          "chunk_norm": row["chunk_norm"],
                          "payload": row["actions_env"],
                          "mode": "actions", "continuations": True}]
                for i in range(1, 1 + N_EXTRA_CANON):
                    a_seed = (NOISE_BASE + task_index * 2_000_000
                              + seed * 1_000 + d + 100_000 * i)
                    specs.append({
                        "key": f"c{i}", "family": "canonical",
                        "provenance": "policy",
                        "behavior_goal_id": canon_id,
                        "gen": ("canonical", instruction, a_seed),
                        "payload": None, "mode": "gen",
                        "continuations": True})
                if c["form"] in ("close", "open"):
                    rdisp = REGION_DISPLAY.get(
                        c["obj"], c["obj"].replace("_", " "))
                    atomic1 = f"{c['form']} {rdisp}"
                    atomic2 = f"{c['form']} the drawer"
                else:
                    obj_disp = display_obj(task_name, c["obj"])
                    atomic1 = f"pick up {obj_disp}"
                    atomic2 = (
                        f"put {obj_disp} in "
                        f"{REGION_DISPLAY.get(c['region'], 'place')}"
                        if c["region"] else
                        f"lift {obj_disp} off the table")
                for j, lang in enumerate(
                        [atomic1, atomic2] + distinct_langs):
                    fam = "atomic" if j < 2 else "distinct"
                    a_seed = (NOISE_BASE + task_index * 2_000_000
                              + seed * 1_000 + d
                              + 100_000 * (20 + j))
                    specs.append({
                        "key": f"sub{j}", "family": fam,
                        "provenance": f"subgoal_{fam}",
                        "behavior_goal_id": lang,
                        "gen": ("prompt", lang, a_seed),
                        "payload": None, "mode": "gen",
                        "continuations": True})
                if c["form"] in ("pick_up", "place"):
                    specs.append({
                        "key": "servo", "family": "servo",
                        "provenance": "scripted_servo",
                        "behavior_goal_id": "privileged_script",
                        "servo": (c["obj"], c["region"],
                                  c["form"]),
                        "payload": None, "mode": "servo",
                        "continuations": False})
                # close/open anchors: the servo primitive is not
                # applicable (no graspable object) — recorded, skipped
                specs.append({
                    "key": "u0_rep2", "family": "canonical",
                    "provenance": "fidelity_repeat",
                    "behavior_goal_id": canon_id,
                    "payload": row["actions_env"],
                    "mode": "actions", "continuations": False})
                process_anchor(task_name, source, d, specs, akey,
                               "train")
    print("v071 materialization phase complete", flush=True)


if __name__ == "__main__":
    main()
