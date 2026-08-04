#!/usr/bin/env python
"""V7.3B — build one recovery-crossed training tranche (2026-08-03.md).

Frozen budget (registered before collection):
  seeds per task k (fresh 2400 family, disjoint from every prior
  family and all behavior panels):
    train sources  2400+10k, 2401+10k, 2402+10k   (2 anchors each)
    dev sources    2403+10k, 2404+10k             (1 anchor each)
    teacher source 2405+10k                        (2 anchors)
  = 30 train + 10 dev acquisition anchors, 10 outcome-blind teacher
  anchors.

Blocker-targeted anchors: candidate decisions where the canonical
first_unresolved matches the task's registered V7.2D terminal-blocker
milestones AND the reachability rule holds; two per train source, one
per dev source, >=5 decisions apart, state-type diversity preferred.

Bank per acquisition anchor (8 executed branches + u0 audit replay):
  u0            deployed full-prompt chunk (stored actions replay)
  c1..c3        fresh full-prompt canonical samples (SHA seeds)
  rec_canon     first chunk of the canonical next-milestone recovery
  rec_alt       first chunk of the compatible-alt-goal recovery
  m1, m2        matched policy-supported alternatives (c4, c5)
Recovery runs CLOSED-LOOP for <=3 decisions (<=30 actions) under the
atomic acquisition prompt (atomic pi0.5 first; privileged scripted
servo fallback for pick/place, provenance-bound, acquisition
instrument only). Recovery trajectories land in recovery_ledger; the
canonical-recovery branch and its matched nonpositive sibling (c1)
additionally get one recovery-to-terminal deployed-pi0.5
continuation. Every branch gets R=2 common-noise canonical
continuations with q@{10,30,60,100}; per-action physical+semantic
traces under BOTH frozen GoalSpecs; all winners/losers/ties/nulls/
failed recoveries retained; order-independent u0 fidelity replay.

Language crossing: 2 scene-compatible GoalSpecs x 3 verified texts
per anchor, indexed separately from the physical rows.

Teacher anchors: bank chunks GENERATED AND STORED UNEXECUTED
(u0 + 7 canonical + 2 atomic + <=2 distinct); outcomes hidden until
the V7.3C ledgers are sealed.

Output root: results/<DATE>_v073_data_r1/ per the required artifact
list. Resumable; per-anchor shards; videos for every executed branch
and recovery trajectory.
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
SEMWIN = RESULTS / "2026-08-02_v071_semwin_r1"
RID = "v073_data_r1"
TASK_ORDER = ["loho_t1_drawer", "loho_t2_basket3", "loho_t3_tray",
              "loho_t4_tray", "loho_t5_drawer_cabinet"]
EPISODE_LENGTH = {"loho_t1_drawer": 700, "loho_t2_basket3": 900,
                  "loho_t3_tray": 900, "loho_t4_tray": 900,
                  "loho_t5_drawer_cabinet": 990}
# registered V7.2D terminal-blocker milestones per task (primary
# training states); a candidate anchor's canonical first_unresolved
# must start with one of these prefixes
BLOCKERS = {
    "loho_t1_drawer": ["pick_up butter_1", "place butter_1"],
    "loho_t2_basket3": ["pick_up butter_1", "place butter_1"],
    "loho_t3_tray": ["pick_up alphabet_soup_1",
                     "place alphabet_soup_1"],
    "loho_t4_tray": ["pick_up new_salad_dressing_1",
                     "place new_salad_dressing_1"],
    "loho_t5_drawer_cabinet": ["pick_up butter_2",
                               "place butter_2"],
}
# the frozen second (alternative) GoalSpec per task for crossing
ALT_GOAL = {"loho_t1_drawer": "t1_back_butter",
            "loho_t2_basket3": "t2_cheese_butter_milk",
            "loho_t3_tray": "t3_sauce_ketchup_butter",
            "loho_t4_tray": "t4_right_bowl",
            "loho_t5_drawer_cabinet": "t5_front_butter"}
GAP, STALL_DECS = 5, 3
PICK_NEAR, PICK_FAR, PLACE_NEAR, PLACE_FAR = 0.03, 0.30, 0.05, 0.30
STABLE_MM = 0.005
HORIZONS, R, CONT_MAX = (10, 30, 60, 100), 2, 100
REC_MAX_DEC, REC_MAX_ACT = 3, 30
# recovery-to-terminal must be able to run PAST the ordinary q@100
# horizon (review finding: 100 made "terminal" unreachable)
REC_TERM_MAX = 300
REREACH_ATOL = 2e-3
ROLES = [("train", 2400, 2, "acq"), ("train", 2401, 2, "acq"),
         ("train", 2402, 2, "acq"), ("dev", 2403, 1, "acq"),
         ("dev", 2404, 1, "acq"), ("teacher", 2405, 2, "teach")]


def sha_seed(p: str) -> int:
    return int.from_bytes(hashlib.sha256(
        p.encode()).digest()[:8], "big") & ((1 << 63) - 1)


def obs_frame(env):
    return copy.deepcopy(env._format_raw_obs(
        env._env.env._get_observations()))


def run_date() -> str:
    return subprocess.run(["date", "+%F"], capture_output=True,
                          text=True,
                          env={"TZ": "America/Chicago"}
                          ).stdout.strip()


@torch.no_grad()
def main() -> None:
    from lcwm.chassis import Pi05Runner
    from lcwm.loho_public import make_public_env
    from lcwm.probe_data import body_positions
    from lcwm.sampler import prefix_forward, sample_chunks
    from lcwm.snapshot import restore, snap
    from lcwm.task_automaton import (GoalAutomaton, fork_env_state,
                                     outcome_tuple,
                                     restore_env_state)
    from lcwm.v067_lineage import flow_noise, sha256_file
    from lcwm.video_recorder import VideoRecorder, write_index_row
    from scripts.collect_loho_v06 import obs_q
    from scripts.collect_v067_continuations import (full_qpos,
                                                    grasp_state)
    from scripts.collect_v069_corrections import (
        OBJ_DISPLAY, REGION_DISPLAY, scripted_servo_action)

    device = torch.device("cuda")
    goal_manifest = json.loads(
        (RESULTS / "goal_spec_manifest_v067.json").read_text())
    tv = json.loads((SEMWIN / "text_variants.json").read_text())
    texts = defaultdict(dict)
    for row in tv["rows"]:
        texts[row["goal_id"]][row["text_variant_id"]] = \
            row["language"]

    root = None
    for p in sorted(RESULTS.glob(f"*_{RID}")):
        root = p
    root = root or RESULTS / f"{run_date()}_{RID}"
    for sub in ("sources", "shards", "videos", "teacher_banks"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    git_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True,
        text=True, cwd=REPO_ROOT).stdout.strip()
    mp = root / "run_manifest.json"
    if not mp.exists():
        mp.write_text(json.dumps({
            "schema": "v073_data_manifest_v1", "run_schema": "v073",
            "run_id": RID, "git_sha": git_sha,
            "seed_roles": ROLES, "blockers": BLOCKERS,
            "alt_goals": ALT_GOAL,
            "bank": "u0+c1..c3+rec_canon+rec_alt+m1+m2 (8 executed) "
                    "+ u0 audit replay; teacher banks generated "
                    "unexecuted (u0+7canon+2atomic+<=2distinct)",
            "recovery": {"closed_loop_decisions": REC_MAX_DEC,
                         "max_actions": REC_MAX_ACT,
                         "order": "atomic pi0.5 first; privileged "
                                  "scripted servo fallback for "
                                  "pick/place, provenance-bound, "
                                  "acquisition instrument only"},
            "crossing": "2 GoalSpecs x 3 verified texts per anchor; "
                        "physical rows stored once",
            "proposal_seed": f"SHA256('{RID}|prop|src|dec|kind|i')",
            "cont_seed": f"SHA256('{RID}|cont|src|dec|goal|rep|cd')",
            "recovery_seed": f"SHA256('{RID}|rec|src|dec|goal|cd')",
            "order_seed": f"SHA256('{RID}|order|src|dec|branch')",
            "goal_manifest_sha256": goal_manifest["manifest_sha256"],
            "text_variants_sha256": sha256_file(
                SEMWIN / "text_variants.json"),
            "panels_excluded": "{1700..1740} sealed, {1750..1790} "
                               "consumed, {1800..1840} reserved dev",
        }, indent=2))
    manifest_sha = sha256_file(mp)

    runner = Pi05Runner(suite_name="libero_10")
    cfg = runner.policy.config
    vindex = root / "video_index.jsonl"
    rec_ledger = root / "recovery_ledger.jsonl"
    fid_path = root / "replay_fidelity.json"
    fids = json.loads(fid_path.read_text()) if fid_path.exists() \
        else {}

    def atomic_prompt(task, first_unres):
        parts = first_unres.split()
        form, obj = parts[0], parts[1]
        if form in ("close", "open"):
            rd = REGION_DISPLAY.get(parts[1],
                                    parts[1].replace("_", " "))
            return f"{form} {rd}"
        od = OBJ_DISPLAY.get((task, obj),
                             "the " + obj.rsplit("_", 1)[0]
                             .replace("_", " "))
        if form == "pick_up":
            return f"pick up {od}"
        region = parts[2] if len(parts) > 2 else None
        return (f"put {od} in "
                f"{REGION_DISPLAY.get(region, 'place')}"
                if region else f"lift {od} off the table")

    # ---------------- Phase A: sources + frozen anchors --------------
    anchor_path = root / "anchor_manifest.json"
    if not anchor_path.exists():
        all_anchors = []
        for ti, task in enumerate(TASK_ORDER):
            for role, base, n_anchor, kind in ROLES:
                seed = base + 10 * ti
                sid = f"{task}_v073_s{seed}"
                spath = root / "sources" / f"{sid}.pt"
                entry = goal_manifest["tasks"][task]
                canon_id = entry["canonical_goal_spec_id"]
                gspecs = entry["goal_specs"]
                subgoals = gspecs[canon_id]["ordered_subgoals"]
                if not spath.exists():
                    env = make_public_env(
                        task, EPISODE_LENGTH[task] + 200)
                    try:
                        runner.reset()
                        obs, _ = env.reset(seed=seed)
                        env._env.env.horizon = \
                            EPISODE_LENGTH[task] + 300
                        auto = GoalAutomaton(list(subgoals))
                        auto.start(env)
                        auto.evaluate(env, 0)
                        bodies = auto.bodies
                        instr = env.task_description
                        rows, cands = [], []
                        obj_prev = body_positions(
                            env, list(bodies.values()))
                        stall, prev_u, t, dec = 0, "___", 0, 0
                        term = trunc = False
                        while t < EPISODE_LENGTH[task]:
                            obs_now = copy.deepcopy(obs)
                            first_u = next(
                                (subgoals[i] for i, v in enumerate(
                                    auto.prev_valid) if not v), None)
                            stall = (stall + 1
                                     if first_u == prev_u else 1)
                            prev_u = first_u
                            opos_all = body_positions(
                                env, list(bodies.values()))
                            recent = [f for f in auto.flips
                                      if f[0] > t - 20]
                            n_un = sum(1 for v in auto.prev_valid
                                       if not v)
                            st = None
                            if any(f[2] == -1 for f in recent):
                                st = "recovery"
                            elif 0 < n_un <= 2:
                                st = "late_chain"
                            elif any(f[2] == 1 for f in recent):
                                st = "milestone_boundary"
                            elif stall >= STALL_DECS:
                                st = "stall"
                            ok = None
                            blocked = first_u is not None and any(
                                first_u.startswith(b)
                                for b in BLOCKERS[task])
                            if st and first_u and blocked:
                                parts = first_u.split()
                                form, obj = parts[0], parts[1]
                                gsp = grasp_state(
                                    env, bodies)["grasped"]
                                eef = np.asarray(
                                    obs_now["robot_state"]["eef"]
                                    ["pos"])
                                opos = body_positions(
                                    env, [bodies[obj]])[0] \
                                    if obj in bodies else None
                                moved = float(np.abs(
                                    opos_all - obj_prev).max())
                                if form == "pick_up" \
                                        and opos is not None:
                                    dist = float(np.linalg.norm(
                                        eef - opos))
                                    if (not gsp.get(obj)
                                            and PICK_NEAR <= dist
                                            <= PICK_FAR
                                            and moved < STABLE_MM):
                                        ok = {"form": form,
                                              "obj": obj,
                                              "region": None,
                                              "dist": dist}
                                elif form == "place":
                                    region = parts[2]
                                    try:
                                        tgt = env._env.env.sim.data \
                                            .get_site_xpos(
                                                region).copy()
                                    except Exception:
                                        tgt = None
                                    if tgt is not None \
                                            and opos is not None:
                                        dist = float(
                                            np.linalg.norm(
                                                opos - tgt))
                                        if bool(gsp.get(obj)) and \
                                                PLACE_NEAR <= dist \
                                                <= PLACE_FAR:
                                            ok = {"form": form,
                                                  "obj": obj,
                                                  "region": region,
                                                  "dist": dist}
                            if ok:
                                cands.append({"decision": dec,
                                              "state_type": st,
                                              "first_unresolved":
                                                  first_u, **ok})
                            obj_prev = opos_all
                            batch = runner._obs_to_policy_batch(
                                obs, instr)
                            pfx = prefix_forward(runner.policy,
                                                 batch)
                            nz = flow_noise(sha_seed(
                                f"{RID}|prop|{sid}|{dec}|deploy|0"),
                                cfg.chunk_size, cfg.max_action_dim)
                            chunk = sample_chunks(
                                runner.policy, batch, n=1,
                                noise=nz.to(device), prefix=pfx)
                            a_env_l, ex = [], 0
                            for a_env in runner.chunk_to_env(
                                    chunk[:, :10]):
                                obs, _r, term, trunc, _i = env.step(
                                    a_env)
                                a_env_l.append(np.asarray(a_env))
                                t += 1
                                ex += 1
                                auto.evaluate(env, t)
                                if term or trunc:
                                    break
                            rows.append({
                                "decision": dec, "t_start": t - ex,
                                "obs": obs_now,
                                "chunk_norm":
                                    chunk[0].float().cpu(),
                                "actions_env": np.stack(a_env_l),
                                "executed_len": ex,
                                "q": obs_q(obs_now),
                                "obj_before": opos_all,
                                "first_unresolved": first_u})
                            dec += 1
                            if term or trunc:
                                break
                        tmp = spath.with_suffix(".tmp")
                        torch.save({
                            "schema": "v073_source_v1",
                            "run_schema": "v073",
                            "source_id": sid, "task": task,
                            "task_index": ti, "seed": seed,
                            "split": role,
                            "language_canonical": instr,
                            "rows": rows,
                            "anchor_candidates": cands}, tmp)
                        tmp.replace(spath)
                        print(f"[src] {sid} ({role}): {len(rows)} "
                              f"dec, {len(cands)} blocker-cands",
                              flush=True)
                    finally:
                        env.close()
                src = torch.load(spath, weights_only=False)
                chosen, seen = [], set()

                def gap_ok(c):
                    return all(abs(c["decision"] - x["decision"])
                               >= GAP for x in chosen)
                for want_new in (True, False):
                    for c in src["anchor_candidates"]:
                        if len(chosen) >= n_anchor:
                            break
                        if c in chosen or not gap_ok(c):
                            continue
                        if want_new and c["state_type"] in seen:
                            continue
                        chosen.append(c)
                        seen.add(c["state_type"])
                for c in chosen:
                    all_anchors.append({
                        "source_id": sid, "task": task,
                        "seed": seed, "role": role, "kind": kind,
                        **c})
                if len(chosen) < n_anchor:
                    # explicit shortfall record (review finding:
                    # never silently under-fill the frozen budget)
                    all_anchors.append({
                        "source_id": sid, "task": task,
                        "seed": seed, "role": role,
                        "kind": "SHORTFALL",
                        "scheduled": n_anchor,
                        "found": len(chosen)})
                    print(f"[anchors] SHORTFALL {sid}: "
                          f"{len(chosen)}/{n_anchor}", flush=True)
        anchor_path.write_text(json.dumps({
            "schema": "v073_anchor_manifest_v1",
            "note": "blocker-targeted; frozen from pre-action stock "
                    "state before any candidate generation; "
                    "SHORTFALL rows record under-filled budgets "
                    "explicitly",
            "anchors": all_anchors}, indent=2))
        by_role = defaultdict(int)
        n_short = 0
        for a in all_anchors:
            if a["kind"] == "SHORTFALL":
                n_short += 1
            else:
                by_role[a["role"]] += 1
        if n_short:
            by_role["SHORTFALL_rows"] = n_short
        print(f"[anchors] frozen: {dict(by_role)}", flush=True)
    anchors = json.loads(anchor_path.read_text())["anchors"]

    # ---------------- helpers ----------------------------------------
    def rereach(env, src, d, autos):
        t = 0
        for i in range(d):
            for a_env in src["rows"][i]["actions_env"]:
                env.step(a_env)
                t += 1
                for a in autos.values():
                    a.evaluate(env, t)
        return t

    def cont_run(env, instr, autos_c, branch_steps, crn_prefix, rep,
                 max_steps=CONT_MAX):
        """Deployed-pi0.5 continuation on the UNIFIED time axis:
        branch actions occupy steps 1..branch_steps, continuation
        actions branch_steps+1..branch_steps+H (V7.2A time-axis
        repair). Tracks EVERY automaton in autos_c so crossed-goal
        horizon labels exist (review finding). q@h keys stay
        continuation-local (registered label_version note). Returns
        {goal_id: outcome_tuple} and continuation steps."""
        obs_c = obs_frame(env)
        q_at = {g: {} for g in autos_c}
        sc, cd, stop = 0, 0, False
        while sc < max_steps and not stop:
            nz = flow_noise(sha_seed(f"{crn_prefix}|{rep}|{cd}"),
                            cfg.chunk_size, cfg.max_action_dim)
            b = runner._obs_to_policy_batch(obs_c, instr)
            pfx = prefix_forward(runner.policy, b)
            ch = sample_chunks(runner.policy, b, n=1,
                               noise=nz.to(device), prefix=pfx)
            for a_env in runner.chunk_to_env(ch[:, :10]):
                _o, _r, tm, tr2, _i = env.step(a_env)
                sc += 1
                for g, ca in autos_c.items():
                    ca.evaluate(env, branch_steps + sc)
                obs_c = obs_frame(env)
                if sc in HORIZONS:
                    for g, ca in autos_c.items():
                        q_at[g][sc] = ca.q_valid()
                if tm or tr2:
                    stop = True
                    break
                if sc >= max_steps:
                    break
            cd += 1
        out = {}
        for g, ca in autos_c.items():
            for hz in HORIZONS:
                q_at[g].setdefault(hz, ca.q_valid())
            out[g] = outcome_tuple(ca, env, branch_steps, q_at[g])
        return out, sc

    # ---------------- Phase B: acquisition banks ---------------------
    done = {p.stem for p in (root / "shards").glob("*.pt")}
    for a in anchors:
        if a["kind"] != "acq":
            continue
        sid, d, task = a["source_id"], a["decision"], a["task"]
        akey = f"{sid}_d{d}"
        if akey in done:
            continue
        src = torch.load(root / "sources" / f"{sid}.pt",
                         weights_only=False)
        entry = goal_manifest["tasks"][task]
        canon_id = entry["canonical_goal_spec_id"]
        alt_id = ALT_GOAL[task]
        gspecs = entry["goal_specs"]
        instr = src["language_canonical"]
        env = make_public_env(task, EPISODE_LENGTH[task] + 400)
        try:
            runner.reset()
            env.reset(seed=src["seed"])
            env._env.env.horizon = EPISODE_LENGTH[task] + 500
            autos0 = {}
            for gid in (canon_id, alt_id):
                g = GoalAutomaton(gspecs[gid]["ordered_subgoals"])
                g.start(env)
                g.evaluate(env, 0)
                autos0[gid] = g
            bodies = autos0[canon_id].bodies
            t = rereach(env, src, d, autos0)
            row = src["rows"][d]
            err = float(np.abs(body_positions(
                env, list(bodies.values()))
                - row["obj_before"]).max())
            assert err < REREACH_ATOL, f"{akey} rereach {err:.2e}"
            snap_a = snap(env, t=t, suite_name="loho_public",
                          task_id=0)
            states_a = {g: fork_env_state(x)
                        for g, x in autos0.items()}
            anchor_obs = obs_frame(env)
            obj_before = body_positions(env, list(bodies.values()))
            qpos_before = full_qpos(env)

            def fresh_autos():
                out = {}
                for gid in (canon_id, alt_id):
                    g = GoalAutomaton(
                        gspecs[gid]["ordered_subgoals"])
                    base = autos0[gid]
                    g.bodies, g.start_pos = base.bodies, \
                        base.start_pos
                    restore_env_state(g, states_a[gid])
                    out[gid] = g
                return out


            def cont_automata(autos_branch):
                """Continuation automata forked from the branch automata with
                inherited flip TIMESTAMPS rebased to -1: prefix/branch flips
                keep their (index, direction) — damage_unrecovered is
                timestamp-independent — but can no longer shadow real
                continuation milestones inside tau_next's post-branch window
                (fix-verification seam defect: prefix flips carried absolute
                episode steps)."""
                out = {}
                for gid in (canon_id, alt_id):
                    ca = GoalAutomaton(gspecs[gid]["ordered_subgoals"])
                    ca.bodies = autos0[gid].bodies
                    ca.start_pos = autos0[gid].start_pos
                    restore_env_state(ca, fork_env_state(autos_branch[gid]))
                    ca.flips = [(-1, i_, d_) for (_s, i_, d_) in ca.flips]
                    out[gid] = ca
                return out

            # ---- recovery rollouts (closed-loop, <=3 decisions) ----
            def _run_recovery_attempt(goal_id, tag, mode, prompt,
                                      servo_target):
                """One closed-loop recovery attempt from snap_a with
                full per-action physical+semantic traces (both
                goals), its own video, and its own ledger row —
                failed attempts are retained, never discarded
                (review finding)."""
                restore(env, snap_a)
                env._env.env.done = False
                autos = fresh_autos()
                ga = autos[goal_id]
                flips0 = len(ga.flips)
                vr = VideoRecorder(root / "videos"
                                   / f"{akey}_rec_{tag}_{mode}.mp4")
                frames = [obs_frame(env)]
                vr.add(frames[0])
                eefs = [obs_q(frames[0])]
                vseq = {g: [list(x.prev_valid)]
                        for g, x in autos.items()}
                acts, steps = [], 0
                positive = False
                if mode == "atomic_pi05":
                    for cd in range(REC_MAX_DEC):
                        b = runner._obs_to_policy_batch(
                            frames[-1], prompt)
                        pfx = prefix_forward(runner.policy, b)
                        nz = flow_noise(sha_seed(
                            f"{RID}|rec|{sid}|{d}|{goal_id}|{cd}"),
                            cfg.chunk_size, cfg.max_action_dim)
                        ch = sample_chunks(runner.policy, b, n=1,
                                           noise=nz.to(device),
                                           prefix=pfx)
                        for a_env in runner.chunk_to_env(
                                ch[:, :10]):
                            _o, _r, tb, tr2, _i = env.step(a_env)
                            steps += 1
                            acts.append(np.asarray(a_env))
                            for g in autos.values():
                                g.evaluate(env, steps)
                            for g, x in autos.items():
                                vseq[g].append(
                                    list(x.prev_valid))
                            f = obs_frame(env)
                            frames.append(f)
                            vr.add(f)
                            eefs.append(obs_q(f))
                            if tb:
                                env._env.env.done = False
                            if tr2 or steps >= REC_MAX_ACT:
                                break
                        if len(ga.flips) > flips0 and any(
                                fl[2] == 1
                                for fl in ga.flips[flips0:]):
                            positive = True
                            break
                        if steps >= REC_MAX_ACT:
                            break
                else:   # privileged_servo toward THIS GOAL's target
                    s_form, s_obj, s_region = servo_target
                    for _k in range(REC_MAX_ACT):
                        a_env = scripted_servo_action(
                            env, s_obj, s_region,
                            autos[goal_id].bodies, s_form)
                        _o, _r, tb, tr2, _i = env.step(a_env)
                        steps += 1
                        acts.append(np.asarray(a_env))
                        for g in autos.values():
                            g.evaluate(env, steps)
                        for g, x in autos.items():
                            vseq[g].append(list(x.prev_valid))
                        f = obs_frame(env)
                        frames.append(f)
                        vr.add(f)
                        eefs.append(obs_q(f))
                        if tb:
                            env._env.env.done = False
                        if len(ga.flips) > flips0 and any(
                                fl[2] == 1
                                for fl in ga.flips[flips0:]):
                            positive = True
                            break
                        if tr2:
                            break
                vm = vr.close(completed=True)
                rec = {"anchor": akey, "goal_id": goal_id,
                       "tag": tag, "prompt": prompt, "mode": mode,
                       "servo_target": (list(servo_target)
                                        if servo_target else None),
                       "positive": bool(positive), "steps": steps,
                       "actions_env": [x.tolist() for x in acts],
                       "eef_seq": [np.asarray(e).tolist()
                                   for e in eefs],
                       "valid_seq": {g: v for g, v in vseq.items()},
                       "video": vm["video_path"],
                       "acquisition_instrument_only": True}
                with rec_ledger.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                write_index_row(
                    vindex, vm, run_id=RID,
                    checkpoint_tag=f"recovery_{mode}",
                    checkpoint_path=None, checkpoint_sha256=None,
                    manifest_sha256=manifest_sha, task=task,
                    seed=src["seed"], arm=f"rec_{tag}_{mode}",
                    split=a["role"], steps=steps,
                    success=bool(positive),
                    ordered_progress=0, damage=0,
                    termination="recovery_end", root=root)
                return rec, acts

            def run_recovery(goal_id, tag):
                """Atomic pi0.5 first; privileged servo fallback
                toward THIS goal's own first-unresolved target
                (review critical: never the canonical anchor's).
                Both attempts retained. Returns
                (final_record, first_chunk_actions or None)."""
                g0 = fresh_autos()[goal_id]
                sub = gspecs[goal_id]["ordered_subgoals"]
                first_u = next((sub[i] for i, v in
                                enumerate(g0.prev_valid)
                                if not v), None)
                if first_u is None:
                    return None, None
                prompt = atomic_prompt(task, first_u)
                rec, acts = _run_recovery_attempt(
                    goal_id, tag, "atomic_pi05", prompt, None)
                if not rec["positive"]:
                    parts = first_u.split()
                    g_form, g_obj = parts[0], parts[1]
                    g_region = parts[2] if len(parts) > 2 else None
                    if g_form in ("pick_up", "place") \
                            and g_obj in g0.bodies:
                        rec2, acts2 = _run_recovery_attempt(
                            goal_id, tag, "privileged_servo",
                            prompt, (g_form, g_obj, g_region))
                        if rec2["positive"]:
                            rec, acts = rec2, acts2
                if not acts:
                    return rec, None
                first = np.stack(acts[:10])
                return rec, first

            rec_c, rc_first = run_recovery(canon_id, "canon")
            rec_a2, ra_first = run_recovery(alt_id, "alt")

            # ---- bank definition -------------------------------------
            # each spec: (key, provenance, env_actions, chunk_norm)
            # chunk_norm stored ONCE at generation (exactly-once
            # contract; review critical: 7/8 branches previously had
            # no stored normalized chunk)
            specs = [("u0", "deployed",
                      list(row["actions_env"])[:10],
                      row["chunk_norm"])]
            b = runner._obs_to_policy_batch(anchor_obs, instr)
            pfx = prefix_forward(runner.policy, b)
            for i in range(1, 6):
                nz = flow_noise(sha_seed(
                    f"{RID}|prop|{sid}|{d}|canonical|{i}"),
                    cfg.chunk_size, cfg.max_action_dim)
                ch = sample_chunks(runner.policy, b, n=1,
                                   noise=nz.to(device), prefix=pfx)
                key = f"c{i}" if i <= 3 else f"m{i - 3}"
                specs.append((key, "canonical_sample",
                              list(runner.chunk_to_env(
                                  ch[:, :10]))[:10],
                              ch[0].float().cpu()))
            if rc_first is not None:
                specs.append(("rec_canon",
                              f"recovery_{rec_c['mode']}",
                              [x for x in rc_first], None))
            if ra_first is not None:
                specs.append(("rec_alt",
                              f"recovery_{rec_a2['mode']}",
                              [x for x in ra_first], None))
            specs.append(("u0_repeat", "fidelity_audit",
                          list(row["actions_env"])[:10], None))
            bank_report = {
                "registered": 8,
                "executed_policy_branches":
                    sum(1 for s in specs
                        if s[0] != "u0_repeat"),
                "missing": [k for k, present in
                            (("rec_canon", rc_first is not None),
                             ("rec_alt", ra_first is not None))
                            if not present]}
            if bank_report["missing"]:
                print(f"  [{akey}] DEGRADED bank (recorded): "
                      f"missing {bank_report['missing']}",
                      flush=True)
            order = sorted(specs, key=lambda s: sha_seed(
                f"{RID}|order|{sid}|{d}|{s[0]}"))

            transitions = []
            branch_ends = {}    # key -> (b_end snap, automata forks)
            for key, prov, acts, chunk_norm in order:
                restore(env, snap_a)
                env._env.env.done = False
                chk = float(np.abs(body_positions(
                    env, list(bodies.values()))
                    - row["obj_before"]).max())
                assert chk < REREACH_ATOL
                autos = fresh_autos()
                vr = VideoRecorder(root / "videos"
                                   / f"{akey}_{key}.mp4")
                frames = [obs_frame(env)]
                vr.add(frames[0])
                eef_seq = [obs_q(frames[0])]
                valid_seq = {g: [list(x.prev_valid)]
                             for g, x in autos.items()}
                a_env_l, steps = [], 0
                for a_env in acts:
                    _o, _r, tb, tr2, _i = env.step(a_env)
                    steps += 1
                    a_env_l.append(np.asarray(a_env))
                    for g in autos.values():
                        g.evaluate(env, steps)
                    for g, x in autos.items():
                        valid_seq[g].append(list(x.prev_valid))
                    f = obs_frame(env)
                    frames.append(f)
                    vr.add(f)
                    eef_seq.append(obs_q(f))
                    if tb:
                        env._env.env.done = False
                    if tr2:
                        break
                obj_after = body_positions(env,
                                           list(bodies.values()))
                qpos_after = full_qpos(env)
                conts = []
                if key != "u0_repeat":
                    b_end = snap(env, t=t + steps,
                                 suite_name="loho_public",
                                 task_id=0)
                    branch_ends[key] = (
                        b_end, {g: fork_env_state(x)
                                for g, x in autos.items()},
                        steps)
                    for rep in range(R):
                        restore(env, b_end)
                        env._env.env.done = False
                        autos_c = cont_automata(autos)
                        ys, _sc = cont_run(
                            env, instr, autos_c, steps,
                            f"{RID}|cont|{sid}|{d}|{canon_id}",
                            rep)
                        conts.append(ys)
                vm = vr.close(completed=True)
                write_index_row(
                    vindex, vm, run_id=RID,
                    checkpoint_tag="bank",
                    checkpoint_path=None, checkpoint_sha256=None,
                    manifest_sha256=manifest_sha, task=task,
                    seed=src["seed"], arm=key, split=a["role"],
                    steps=steps,
                    success=bool(conts and any(
                        ys[canon_id]["success_by_100"]
                        for ys in conts)),
                    ordered_progress=autos[
                        canon_id].ordered_prefix(),
                    damage=autos[canon_id].damage_unrecovered(),
                    termination="branch_end", root=root)
                transitions.append({
                    "transition_id": f"{akey}_{key}",
                    "branch_key": key, "provenance": prov,
                    "anchor": akey, "task": task,
                    "source_id": sid, "decision": d,
                    "role": a["role"],
                    "frames": frames, "eef_seq": eef_seq,
                    "valid_seq": valid_seq,
                    "actions_env": (np.stack(a_env_l) if a_env_l
                                    else np.zeros((0, 7))),
                    "chunk_norm": chunk_norm,
                    "steps": steps,
                    "obj_after": obj_after,
                    "qpos_after": qpos_after,
                    "grasp_after": grasp_state(env, bodies),
                    "continuations": conts,
                    "recovery_terminal": None})
                print(f"  [{akey}] {key} steps={steps} "
                      f"conts={len(conts)}", flush=True)

            # ---- second pass (order-independent by construction) ----
            trs_by = {tr["branch_key"]: tr for tr in transitions}

            # (1) fidelity from STORED endpoints — never capture-order
            if "u0" in trs_by and "u0_repeat" in trs_by:
                ta_, tb_ = trs_by["u0"], trs_by["u0_repeat"]
                dq = np.abs(np.asarray(ta_["eef_seq"][-1])
                            - np.asarray(tb_["eef_seq"][-1]))
                fids[akey] = {
                    "eef_pos": float(dq[:3].max()),
                    "eef_quat": float(dq[3:7].max()),
                    "gripper": float(dq[7:].max()),
                    "obj_pos": float(np.abs(
                        np.asarray(ta_["obj_after"])
                        - np.asarray(tb_["obj_after"])).max()),
                    "qpos": float(np.abs(
                        np.asarray(ta_["qpos_after"])
                        - np.asarray(tb_["qpos_after"])).max())}
            # (2) matched NONPOSITIVE sibling: first canonical sample
            # with no positive canonical flip in its executed actions
            # (measured, not assumed — review finding)
            def branch_positive(tr):
                vs = tr["valid_seq"][canon_id]
                v0 = np.asarray(vs[0], dtype=bool)
                v1 = np.asarray(vs[-1], dtype=bool)
                return bool((v1 & ~v0).any())
            sibling = next(
                (k for k in ("c1", "c2", "c3", "m1", "m2")
                 if k in trs_by
                 and not branch_positive(trs_by[k])), None)
            # (3) recovery-to-terminal for the canonical correction
            # and the measured sibling, from their saved end states,
            # allowed to run past q@100 (REC_TERM_MAX)
            for rk in ([x for x in ("rec_canon",) if x in trs_by]
                       + ([sibling] if sibling else [])):
                if rk not in branch_ends:
                    continue
                b_end, forks, bsteps = branch_ends[rk]
                restore(env, b_end)
                env._env.env.done = False
                _fa = {}
                for gid in (canon_id, alt_id):
                    ca = GoalAutomaton(
                        gspecs[gid]["ordered_subgoals"])
                    ca.bodies = autos0[gid].bodies
                    ca.start_pos = autos0[gid].start_pos
                    restore_env_state(ca, forks[gid])
                    _fa[gid] = ca
                autos_c = cont_automata(_fa)
                ys, sc2 = cont_run(
                    env, instr, autos_c, bsteps,
                    f"{RID}|recterm|{sid}|{d}|{canon_id}", 0,
                    max_steps=REC_TERM_MAX)
                trs_by[rk]["recovery_terminal"] = {
                    "outcomes": ys,
                    "steps": sc2,
                    "role": ("canonical_correction"
                             if rk == "rec_canon"
                             else "matched_nonpositive_sibling")}
            shard = {
                "schema": "v073_shard_v1", "run_schema": "v073",
                "anchor": akey, "task": task, "source_id": sid,
                "decision": d, "role": a["role"],
                "goal_ids": [canon_id, alt_id],
                "first_unresolved": a["first_unresolved"],
                "obj_before": obj_before,
                "qpos_before": qpos_before,
                "bank_report": bank_report,
                "nonpositive_sibling": sibling,
                "transitions": transitions}
            tmp = root / "shards" / f"{akey}.pt.tmp"
            torch.save(shard, tmp)
            tmp.replace(root / "shards" / f"{akey}.pt")
            fid_path.write_text(json.dumps(fids, indent=2))
            done.add(akey)
        finally:
            env.close()

    # ---------------- Phase C: teacher banks (UNEXECUTED) ------------
    for a in anchors:
        if a["kind"] != "teach":
            continue
        sid, d, task = a["source_id"], a["decision"], a["task"]
        akey = f"{sid}_d{d}"
        tb_path = root / "teacher_banks" / f"{akey}.pt"
        if tb_path.exists():
            continue
        src = torch.load(root / "sources" / f"{sid}.pt",
                         weights_only=False)
        entry = goal_manifest["tasks"][task]
        canon_id = entry["canonical_goal_spec_id"]
        gspecs = entry["goal_specs"]
        instr = src["language_canonical"]
        row = src["rows"][d]
        anchor_obs = row["obs"]
        bank = {"u0": ("canonical", instr, row["chunk_norm"])}
        b = runner._obs_to_policy_batch(anchor_obs, instr)
        pfx = prefix_forward(runner.policy, b)
        for i in range(1, 8):
            nz = flow_noise(sha_seed(
                f"{RID}|prop|{sid}|{d}|canonical|{i}"),
                cfg.chunk_size, cfg.max_action_dim)
            ch = sample_chunks(runner.policy, b, n=1,
                               noise=nz.to(device), prefix=pfx)
            bank[f"c{i}"] = ("canonical", instr,
                             ch[0].float().cpu())
        first_u = a["first_unresolved"]
        at1 = atomic_prompt(task, first_u)
        at2 = (f"{first_u.split()[0].replace('_', ' ')} "
               f"{first_u.split()[1].rsplit('_', 1)[0]}"
               .replace("_", " "))
        dl = [gspecs[g]["language"] for g in gspecs
              if g != canon_id][:2]
        for j, lang in enumerate([at1, at2] + dl):
            fam = "atomic" if j < 2 else "distinct"
            b2 = runner._obs_to_policy_batch(anchor_obs, lang)
            p2 = prefix_forward(runner.policy, b2)
            nz = flow_noise(sha_seed(
                f"{RID}|prop|{sid}|{d}|{fam}|{j}"),
                cfg.chunk_size, cfg.max_action_dim)
            ch = sample_chunks(runner.policy, b2, n=1,
                               noise=nz.to(device), prefix=p2)
            bank[f"sub{j}"] = (fam, lang, ch[0].float().cpu())
        torch.save({"schema": "v073_teacher_bank_v1",
                    "anchor": akey, "task": task,
                    "source_id": sid, "decision": d,
                    "bank": bank,
                    "note": "UNEXECUTED; outcomes hidden until "
                            "V7.3C ledgers sealed"}, tb_path)
        print(f"[teach-bank] {akey}: {len(bank)} candidates stored "
              f"unexecuted", flush=True)

    # ---------------- indices, crossing, support ---------------------
    with (root / "physical_transition_index.jsonl").open("w") as f:
        for p in sorted((root / "shards").glob("*.pt")):
            s = torch.load(p, weights_only=False)
            for tr in s["transitions"]:
                f.write(json.dumps({
                    "transition_id": tr["transition_id"],
                    "anchor": s["anchor"], "task": s["task"],
                    "role": s["role"],
                    "branch_key": tr["branch_key"],
                    "provenance": tr["provenance"],
                    "steps": tr["steps"],
                    "n_continuations":
                        len(tr["continuations"]),
                    "has_recovery_terminal":
                        tr["recovery_terminal"] is not None,
                }) + "\n")
    with (root / "semantic_query_index.jsonl").open("w") as f:
        for p in sorted((root / "shards").glob("*.pt")):
            s = torch.load(p, weights_only=False)
            for gid in s["goal_ids"]:
                for pv in ("p0", "p1", "p2"):
                    tvid = f"{gid}_{pv}"
                    if tvid in texts[gid]:
                        f.write(json.dumps({
                            "anchor": s["anchor"],
                            "goal_id": gid,
                            "text_variant_id": tvid,
                            "language": texts[gid][tvid],
                            "role": s["role"]}) + "\n")
    (root / "source_split.json").write_text(json.dumps({
        "rule": "source-disjoint; split frozen before language "
                "expansion; teacher sources disjoint from "
                "train/dev",
        "sources": {f"{t}_v073_s{b + 10 * ti}": r
                    for ti, t in enumerate(TASK_ORDER)
                    for r, b, _n, _k in ROLES}}, indent=2))

    # ---- outcome support: registered cells, masks only --------------
    def per_subgoal_flip(tr, gid):
        vs = tr["valid_seq"][gid]
        v0 = np.asarray(vs[0], dtype=bool)
        v1 = np.asarray(vs[-1], dtype=bool)
        return bool((v1 & ~v0).any())      # real milestone flip

    def cont_pref(tr_a, tr_b, gid):
        """Both-repeats-agree preference on the frozen tuple
        p_valid_100 -> mean(q@h) under goal gid; 0 = tie/unstable."""
        signs = []
        for r_ in range(min(len(tr_a["continuations"]),
                            len(tr_b["continuations"]))):
            ya = tr_a["continuations"][r_][gid]
            yb = tr_b["continuations"][r_][gid]
            qa = np.mean(list(ya["q_at_horizons"].values()))
            qb = np.mean(list(yb["q_at_horizons"].values()))
            if ya["p_valid_100"] > yb["p_valid_100"]:
                signs.append(1)
            elif yb["p_valid_100"] > ya["p_valid_100"]:
                signs.append(-1)
            elif qa > qb + 0.0417:
                signs.append(1)
            elif qb > qa + 0.0417:
                signs.append(-1)
            else:
                signs.append(0)
        if signs and all(s_ == 1 for s_ in signs):
            return 1
        if signs and all(s_ == -1 for s_ in signs):
            return -1
        return 0

    ALL_CELLS = [f"{tk}|{rl}" for tk in TASK_ORDER
                 for rl in ("train", "dev")]
    sup = {c: defaultdict(int) for c in ALL_CELLS}
    sup_src = {c: defaultdict(set) for c in ALL_CELLS}
    rev_anchors = {c: set() for c in ALL_CELLS}
    recterm_contrast_anchors = {c: set() for c in ALL_CELLS}
    for p_ in sorted((root / "shards").glob("*.pt")):
        s = torch.load(p_, weights_only=False)
        canon, alt = s["goal_ids"]
        cell = f"{s['task']}|{s['role']}"
        pos_branch, nonpos_branch = None, None

        policy_trs = [tr for tr in s["transitions"]
                      if tr["branch_key"] != "u0_repeat"]
        n_recterm = 0
        for tr in policy_trs:
            sup[cell]["branches"] += 1
            if per_subgoal_flip(tr, canon):
                sup[cell]["milestone_positive"] += 1
                pos_branch = tr
                sup_src[cell]["milestone_pos_sources"].add(
                    s["source_id"])
            else:
                nonpos_branch = tr
            if tr["recovery_terminal"] is not None:
                sup[cell]["recovery_terminal"] += 1
                n_recterm += 1
        if pos_branch is not None and nonpos_branch is not None:
            sup[cell]["pos_plus_matched_nonpos_anchors"] += 1
            sup_src[cell]["pos_nonpos_pair_sources"].add(
                s["source_id"])
        if n_recterm >= 2:
            # a REAL per-anchor contrast: rec_canon AND the measured
            # sibling both carried to terminal at this anchor
            recterm_contrast_anchors[cell].add(s["anchor"])
        # action x goal rank reversal, tracked PER ANCHOR (the
        # registered cell counts anchors, not pairs)
        found_rev = False
        for i in range(len(policy_trs)):
            for j in range(i + 1, len(policy_trs)):
                pc = cont_pref(policy_trs[i], policy_trs[j], canon)
                pa = cont_pref(policy_trs[i], policy_trs[j], alt)
                if pc != 0 and pa != 0 and pc != pa:
                    found_rev = True
        if found_rev:
            rev_anchors[cell].add(s["anchor"])
    support = {
        "cells": {c: dict(sup[c]) for c in ALL_CELLS},
        "independent_sources": {
            c: {kk: sorted(vv) for kk, vv in sup_src[c].items()}
            for c in ALL_CELLS},
        "rank_reversal_anchors": {
            c: sorted(rev_anchors[c]) for c in ALL_CELLS},
        "recterm_contrast_anchors": {
            c: sorted(recterm_contrast_anchors[c])
            for c in ALL_CELLS},
        "registered_cell_checks": {
            c: {"pos_nonpos_at_2train_sources_or_1dev":
                    len(sup_src[c]["pos_nonpos_pair_sources"])
                    >= (2 if c.endswith("train") else 1),
                "rank_reversal_anchor_quota":
                    len(rev_anchors[c])
                    >= (2 if c.endswith("train") else 1),
                "has_recovery_terminal_contrast":
                    len(recterm_contrast_anchors[c]) >= 1}
            for c in ALL_CELLS},
        "note": "masks only; ALL cells listed explicitly incl. "
                "empty ones; support-empty cells are recorded, "
                "never refilled by outcome (bounded fallback rule)",
    }
    (root / "outcome_support.json").write_text(
        json.dumps(support, indent=2))
    print(json.dumps({"cells": support["cells"],
                      "checks":
                          support["registered_cell_checks"]},
                     indent=1), flush=True)
    print(f"-> {root}", flush=True)


if __name__ == "__main__":
    main()
