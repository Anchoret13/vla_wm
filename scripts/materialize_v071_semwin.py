#!/usr/bin/env python
"""V7.1.2b — materialize semantic milestone-crossing windows.

Frozen budget registered in plan_and_progress/2026-08-02.md BEFORE this
run: 22 simulator windows total, one indexed video each.

  Tranche A (17) — re-reach each registered natural crossing decision
    and execute that decision's STORED stock actions (c=10) with
    per-action raw obs + automaton state under ALL scene-compatible
    GoalSpecs. If the stored crossing does not reproduce, the window is
    kept with its MEASURED labels and non-reproduction is recorded.
  Tranche B (5) — cells with no natural crossing (t3 train/dev,
    t5 dev): stored branch chunk replay + fresh canonical continuation
    in 10-action decisions, <=5 continuation decisions, stopping at the
    END of the first decision containing a canonical flip. No flip
    within horizon = failed candidate, kept as negative.
  t1 dev has neither natural crossings nor continuation-crossing
  branches: recorded ABSENT, no new proposal sweep.

CRN namespace v071_semwin_r1|cont|{source_id}|{decision}|{canon_id}
|{rep}|{cd} — branch identity structurally absent.

Offline in the same run: positive-support labeling under the frozen
rule (nonzero per-goal delta AND a differing scene-compatible pair),
and 24 hand-authored field-verified same-goal paraphrases
(text_variants.json). Output root:
results/libero_loho_public_v1/2026-08-02_v071_semwin_r1/
"""

from __future__ import annotations

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
R2_REPLAY = RESULTS / "2026-08-02_v071_replay_r2"
UNION = RESULTS / "2026-08-02_v071_union_f1"
STAGE, RUN_ID = "semwin", "r1"
EPISODE_LENGTH = {"loho_t1_drawer": 700, "loho_t2_basket3": 900,
                  "loho_t3_tray": 900, "loho_t4_tray": 900,
                  "loho_t5_drawer_cabinet": 990}
REREACH_ATOL = 2e-3
B_MAX_CONT_DECISIONS = 5

# ---- frozen window budget (registered 2026-08-02.md) -----------------------
TRANCHE_A = [
    ("loho_t1_drawer_acquire_s2200", 37),
    ("loho_t1_drawer_acquire_s2200", 47),
    ("loho_t2_basket3_correction_s2110", 6),
    ("loho_t2_basket3_correction_s2110", 13),
    ("loho_t2_basket3_correction_s2111", 6),
    ("loho_t2_basket3_correction_s2111", 14),
    ("loho_t2_basket3_acquire_s2210", 7),
    ("loho_t2_basket3_acquire_s2210", 14),
    ("loho_t2_basket3_acquire_s2210", 61),
    ("loho_t2_basket3_acquire_s2210", 62),
    ("loho_t2_basket3_correction_s2112", 6),
    ("loho_t2_basket3_correction_s2112", 13),
    ("loho_t4_tray_correction_s2130", 5),
    ("loho_t4_tray_correction_s2130", 11),
    ("loho_t4_tray_correction_s2132", 5),
    ("loho_t4_tray_correction_s2132", 11),
    ("loho_t5_drawer_cabinet_acquire_s2240", 93),
]
TRANCHE_B = [
    ("loho_t3_tray_acquire_s2220", 2,
     "loho_t3_tray_acquire_s2220_d2_u0"),
    ("loho_t3_tray_acquire_s2220", 2,
     "loho_t3_tray_acquire_s2220_d2_c1"),
    ("loho_t3_tray_correction_s2122", 2,
     "loho_t3_tray_correction_s2122_d2_u0_fresh"),
    ("loho_t3_tray_correction_s2122", 2,
     "loho_t3_tray_correction_s2122_d2_stored_support"),
    ("loho_t5_drawer_cabinet_correction_s2142", 2,
     "loho_t5_drawer_cabinet_correction_s2142_d2_stored_sub0"),
]
ABSENT_CELLS = [["loho_t1_drawer", "dev"]]

# ---- 24 hand-authored same-goal paraphrases (field-verified below) ---------
# check = ordered surface mentions + required close verb + forbidden
# opposite spatial token
PARAPHRASES = {
    "t1_canonical": [
        "place the front butter and the chocolate pudding into the "
        "cabinet's top drawer, then close the drawer",
        "move the butter at the front and the chocolate pudding into "
        "the top drawer of the cabinet and shut it",
    ],
    "t1_back_butter": [
        "place the back butter and the chocolate pudding into the "
        "cabinet's top drawer, then close the drawer",
        "move the butter at the back and the chocolate pudding into "
        "the top drawer of the cabinet and shut it",
    ],
    "t5_canonical": [
        "place the back butter and the chocolate pudding into the "
        "cabinet's top drawer, close the drawer, then set the black "
        "bowl on the cabinet top",
        "move the butter at the back and the chocolate pudding into "
        "the top drawer of the cabinet, shut it, and put the black "
        "bowl onto the top of the cabinet",
    ],
    "t5_front_butter": [
        "place the front butter and the chocolate pudding into the "
        "cabinet's top drawer, close the drawer, then set the black "
        "bowl on the cabinet top",
        "move the butter at the front and the chocolate pudding into "
        "the top drawer of the cabinet, shut it, and put the black "
        "bowl onto the top of the cabinet",
    ],
    "t2_canonical": [
        "place the alphabet soup, the butter, and the tomato sauce "
        "into the basket",
        "move the alphabet soup and the butter and the tomato sauce "
        "into the basket",
    ],
    "t2_cheese_butter_milk": [
        "place the cream cheese, the butter, and the milk into the "
        "basket",
        "move the cream cheese and the butter and the milk into the "
        "basket",
    ],
    "t2_milk_ketchup_oj": [
        "place the milk, the ketchup, and the orange juice into the "
        "basket",
        "move the milk and the ketchup and the orange juice into the "
        "basket",
    ],
    "t3_canonical": [
        "place the alphabet soup, the cream cheese, and the butter "
        "into the tray",
        "move the alphabet soup and the cream cheese and the butter "
        "into the tray",
    ],
    "t3_sauce_ketchup_butter": [
        "place the tomato sauce, the ketchup, and the butter into "
        "the tray",
        "move the tomato sauce and the ketchup and the butter into "
        "the tray",
    ],
    "t3_soup_sauce_ketchup": [
        "place the alphabet soup, the tomato sauce, and the ketchup "
        "into the tray",
        "move the alphabet soup and the tomato sauce and the ketchup "
        "into the tray",
    ],
    "t4_canonical": [
        "place the left black bowl, the salad dressing, and the "
        "chocolate pudding into the tray",
        "move the black bowl at the left and the salad dressing and "
        "the chocolate pudding into the tray",
    ],
    "t4_right_bowl": [
        "place the right black bowl, the salad dressing, and the "
        "chocolate pudding into the tray",
        "move the black bowl at the right and the salad dressing and "
        "the chocolate pudding into the tray",
    ],
}
SURFACE = {"butter_1": "butter", "butter_2": "butter",
           "chocolate_pudding_1": "chocolate pudding",
           "alphabet_soup_1": "alphabet soup",
           "cream_cheese_1": "cream cheese", "milk_1": "milk",
           "ketchup_1": "ketchup", "orange_juice_1": "orange juice",
           "tomato_sauce_1": "tomato sauce",
           "akita_black_bowl_1": "black bowl",
           "akita_black_bowl_2": "black bowl",
           "new_salad_dressing_1": "salad dressing"}
SPATIAL = {"t1_canonical": ("front", "back"),
           "t1_back_butter": ("back", "front"),
           "t5_canonical": ("back", "front"),
           "t5_front_butter": ("front", "back"),
           "t4_canonical": ("left", "right"),
           "t4_right_bowl": ("right", "left")}


def verify_paraphrases(goal_manifest: dict) -> dict:
    """Mechanical field check: goal-object mentions appear in subgoal
    order; close verb present when the goal closes the drawer;
    distinguishing spatial token present and its opposite absent."""
    specs = {}
    for entry in goal_manifest["tasks"].values():
        specs.update(entry["goal_specs"])
    rows = []
    for gid, texts in PARAPHRASES.items():
        spec = specs[gid]
        objs = [sg.split()[1] for sg in spec["ordered_subgoals"]
                if sg.startswith("pick_up")]
        needs_close = any(sg.startswith("close")
                          for sg in spec["ordered_subgoals"])
        for i, text in enumerate(texts, start=1):
            pos = -1
            for o in objs:
                p = text.find(SURFACE[o], pos + 1)
                assert p > pos, f"{gid} p{i}: '{SURFACE[o]}' out of order"
                pos = p
            if needs_close:
                assert ("close" in text or "shut" in text), \
                    f"{gid} p{i}: close verb missing"
            if gid in SPATIAL:
                want, forbid = SPATIAL[gid]
                assert want in text and forbid not in text, \
                    f"{gid} p{i}: spatial token check failed"
            rows.append({"goal_id": gid,
                         "text_variant_id": f"{gid}_p{i}",
                         "language": text,
                         "canonical_language": spec["language"],
                         "verified": True})
        rows.append({"goal_id": gid, "text_variant_id": f"{gid}_p0",
                     "language": spec["language"],
                     "canonical_language": spec["language"],
                     "verified": True})
    return {"schema": "v071_text_variants_v1",
            "verification": ("mechanical field check: goal-object "
                             "mention order == subgoal order; close "
                             "verb required when goal closes drawer; "
                             "distinguishing spatial token present, "
                             "opposite absent"),
            "rows": rows}


def sha_seed(payload: str) -> int:
    return int.from_bytes(hashlib.sha256(
        payload.encode()).digest()[:8], "big") & ((1 << 63) - 1)


def obs_frame(env):
    return copy.deepcopy(env._format_raw_obs(
        env._env.env._get_observations()))


def run_date_chicago() -> str:
    return subprocess.run(
        ["date", "+%F"], capture_output=True, text=True,
        env={"TZ": "America/Chicago"}).stdout.strip()


def load_source(source_id: str) -> dict:
    if "_acquire_" in source_id:
        return torch.load(R2_REPLAY / "acquire_sources"
                          / f"{source_id}.pt", weights_only=False)
    return torch.load(DATA / "corrections_sources_v069"
                      / f"{source_id}.pt", weights_only=False)


def load_branch_actions(union: dict, transition_id: str):
    row = next(r for r in union["rows"]
               if r["transition_id"] == transition_id)
    roots = {k: Path(v) for k, v in union["replay_roots"].items()}
    shard = torch.load(roots[row["run"]] / "shards" / row["shard"],
                       weights_only=False)
    tr = next(t for t in shard["transitions"]
              if t["transition_id"] == transition_id)
    return tr, row


def goal_signature(imm: dict) -> tuple:
    """Per-goal delta signature for the frozen support rule (imm holds
    both the *_before and *_after fields of one window)."""
    flips = tuple(sorted(
        (int(i), int(d)) for (_s, i, d) in imm["window_flips"]))
    dp = imm["ordered_prefix_after"] - imm["ordered_prefix_before"]
    dq = round(imm["q_valid_after"] - imm["q_valid_before"], 6)
    return (flips, dp, dq)


@torch.no_grad()
def main() -> None:
    from lcwm.chassis import Pi05Runner
    from lcwm.probe_data import body_positions
    from lcwm.sampler import prefix_forward, sample_chunks
    from lcwm.snapshot import restore, snap
    from lcwm.task_automaton import (GoalAutomaton, fork_env_state,
                                     restore_env_state)
    from lcwm.v067_lineage import flow_noise, sha256_file
    from lcwm.video_recorder import VideoRecorder, write_index_row
    from lcwm.loho_public import make_public_env
    from scripts.collect_loho_v06 import obs_q
    from scripts.collect_v067_continuations import (full_qpos,
                                                    grasp_state)

    goal_manifest = json.loads(
        (RESULTS / "goal_spec_manifest_v067.json").read_text())
    union = json.loads((UNION / "union_manifest.json").read_text())
    git_sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True,
                             cwd=REPO_ROOT).stdout.strip()

    root = None
    for p in sorted(RESULTS.glob(f"*_v071_{STAGE}_{RUN_ID}")):
        root = p
    if root is None:
        root = RESULTS / f"{run_date_chicago()}_v071_{STAGE}_{RUN_ID}"
    root.mkdir(exist_ok=True)
    (root / "shards").mkdir(exist_ok=True)
    (root / "videos").mkdir(exist_ok=True)

    manifest_path = root / "run_manifest.json"
    manifest = {
        "schema": "v071_semwin_manifest_v1", "run_schema": "v071",
        "run_id": RUN_ID, "stage": STAGE,
        "run_date": root.name.split("_")[0],
        "timezone": "America/Chicago", "git_sha": git_sha,
        "crn_contract": (f"v071_{STAGE}_{RUN_ID}|cont|src|dec|gid|rep"
                         "|cd; branch identity structurally absent"),
        "tranche_a": TRANCHE_A, "tranche_b": TRANCHE_B,
        "absent_cells": ABSENT_CELLS,
        "b_max_cont_decisions": B_MAX_CONT_DECISIONS,
        "union_manifest_sha256": sha256_file(
            UNION / "union_manifest.json"),
        "goal_manifest_sha256": goal_manifest["manifest_sha256"],
        "code_files": {f: sha256_file(REPO_ROOT / f) for f in (
            "scripts/materialize_v071_semwin.py",
            "lcwm/task_automaton.py", "lcwm/snapshot.py",
            "lcwm/video_recorder.py")},
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

    tv = verify_paraphrases(goal_manifest)
    (root / "text_variants.json").write_text(json.dumps(tv, indent=2))
    print(f"text_variants.json: {len(tv['rows'])} rows "
          f"(24 paraphrases + 12 canonical)", flush=True)

    runner = Pi05Runner(suite_name="libero_10")
    cfg = runner.policy.config
    vindex = root / "video_index.jsonl"
    done_shards = {p.stem for p in (root / "shards").glob("*.pt")}

    # group windows by source, ascending decision
    by_source: dict[str, dict] = {}
    for sid, d in TRANCHE_A:
        by_source.setdefault(sid, {"windows": []})["windows"].append(
            {"tranche": "A", "decision": d, "branch": "stock",
             "expected_crossing": True})
    for sid, d, tid in TRANCHE_B:
        tr, row = load_branch_actions(union, tid)
        by_source.setdefault(sid, {"windows": []})["windows"].append(
            {"tranche": "B", "decision": d,
             "branch": tid.rsplit(f"_d{d}_", 1)[1],
             "branch_transition_id": tid,
             "branch_actions": np.asarray(tr["actions_env"]),
             "branch_family": tr["family"],
             "branch_provenance": tr["provenance"],
             "branch_audit_only": bool(row["audit_only"]),
             "expected_crossing": False})

    def execute_window(env, autos, canon_id, source, w, video,
                       bodies, crn_prefix):
        """Stored actions (A: decision row / B: branch chunk), then for
        B fresh canonical continuation decisions until the first
        canonical flip's decision completes."""
        frames = [obs_frame(env)]
        video.add(frames[0])
        eef_seq = [obs_q(frames[0])]
        valid_seq = {gid: [list(a.prev_valid)]
                     for gid, a in autos.items()}
        flips0 = {gid: len(a.flips) for gid, a in autos.items()}
        imm_before = {gid: {
            "valid_before": list(a.prev_valid),
            "ordered_prefix_before": a.ordered_prefix(),
            "q_valid_before": a.q_valid()}
            for gid, a in autos.items()}
        actions_env, segments = [], []
        steps = 0

        def step_actions(acts, seg_name):
            nonlocal steps
            n = 0
            for a_env in acts:
                _o, _r, tb, tr2, _i = env.step(a_env)
                steps += 1
                n += 1
                actions_env.append(np.asarray(a_env))
                for gid, a in autos.items():
                    a.evaluate(env, steps)
                    valid_seq[gid].append(list(a.prev_valid))
                f = obs_frame(env)
                frames.append(f)
                video.add(f)
                eef_seq.append(obs_q(f))
                if tb:
                    env._env.env.done = False
                if tr2:
                    break
            segments.append({"segment": seg_name, "actions": n})

        if w["tranche"] == "A":
            step_actions(
                list(source["rows"][w["decision"]]["actions_env"])[:10],
                "stored_decision")
        else:
            step_actions(list(w["branch_actions"])[:10], "stored_branch")
            cd = 0
            while (len(autos[canon_id].flips) == flips0[canon_id]
                   and cd < B_MAX_CONT_DECISIONS):
                seed = sha_seed(f"{crn_prefix}|0|{cd}")
                nz = flow_noise(seed, cfg.chunk_size,
                                cfg.max_action_dim)
                batch = runner._obs_to_policy_batch(
                    obs_frame(env), source["language_canonical"])
                pc = prefix_forward(runner.policy, batch)
                ch = sample_chunks(runner.policy, batch, n=1,
                                   noise=nz.to("cuda"), prefix=pc)
                step_actions(list(runner.chunk_to_env(ch[:, :10]))[:10],
                             f"fresh_cont_{cd}")
                cd += 1

        imm = {}
        for gid, a in autos.items():
            window = a.flips[flips0[gid]:]
            imm[gid] = {
                **imm_before[gid],
                "valid_after": list(a.prev_valid),
                "window_flips": [tuple(f) for f in window],
                "flips_01": [tuple(f) for f in window if f[2] == 1],
                "flips_10": [tuple(f) for f in window if f[2] == -1],
                "ordered_prefix_after": a.ordered_prefix(),
                "q_valid_after": a.q_valid(),
            }
        return {"frames": frames, "eef_seq": eef_seq,
                "valid_seq": valid_seq,
                "actions_env": (np.stack(actions_env)
                                if actions_env else np.zeros((0, 7))),
                "steps": steps, "segments": segments,
                "immediate": imm}

    for sid in sorted(by_source):
        source = load_source(sid)
        task_name = source["task"]
        split = source["split"]
        wins = sorted(by_source[sid]["windows"],
                      key=lambda w: (w["decision"], w["branch"]))
        akeys = {f"{sid}_d{w['decision']}_semwin_{w['branch']}"
                 for w in wins}
        if akeys <= done_shards:
            print(f"[skip] {sid} (all windows done)", flush=True)
            continue
        entry = goal_manifest["tasks"][task_name]
        canon_id = entry["canonical_goal_spec_id"]
        goal_specs = entry["goal_specs"]
        goal_ids = [canon_id] + [g for g in goal_specs
                                 if g != canon_id]
        env = make_public_env(task_name,
                              EPISODE_LENGTH[task_name] + 200)
        try:
            runner.reset()
            env.reset(seed=source["seed"])
            env._env.env.horizon = EPISODE_LENGTH[task_name] + 400
            automata = {}
            for gid in goal_ids:
                a = GoalAutomaton(goal_specs[gid]["ordered_subgoals"])
                a.start(env)
                automata[gid] = a
            for a in automata.values():
                a.evaluate(env, 0)
            bodies = automata[canon_id].bodies
            t, next_i = 0, 0
            for w in wins:
                d = w["decision"]
                for i in range(next_i, d):
                    for a_env in source["rows"][i]["actions_env"]:
                        env.step(a_env)
                        t += 1
                        for a in automata.values():
                            a.evaluate(env, t)
                next_i = d
                row = source["rows"][d]
                err = float(np.abs(body_positions(
                    env, list(bodies.values()))
                    - row["obj_before"]).max())
                assert err < REREACH_ATOL, \
                    f"{sid}_d{d}: re-reach {err:.2e}"
                akey = f"{sid}_d{d}_semwin_{w['branch']}"
                if akey in done_shards:
                    print(f"[skip] {akey}", flush=True)
                    continue
                snap_a = snap(env, t=t, suite_name="loho_public",
                              task_id=0)
                anchor_states = {gid: fork_env_state(a)
                                 for gid, a in automata.items()}
                obj_before = body_positions(env, list(bodies.values()))
                qpos_before = full_qpos(env)

                autos = {}
                for gid in goal_ids:
                    ba = GoalAutomaton(
                        goal_specs[gid]["ordered_subgoals"])
                    base = automata[gid]
                    ba.bodies, ba.start_pos = base.bodies, \
                        base.start_pos
                    restore_env_state(ba, anchor_states[gid])
                    autos[gid] = ba
                video = VideoRecorder(root / "videos" / f"{akey}.mp4")
                crn_prefix = (f"v071_{STAGE}_{RUN_ID}|cont|{sid}|{d}"
                              f"|{canon_id}")
                res = execute_window(env, autos, canon_id, source, w,
                                     video, bodies, crn_prefix)
                vmeta = video.close(completed=True)
                obj_after = body_positions(env, list(bodies.values()))
                qpos_after = full_qpos(env)
                canon_imm = res["immediate"][canon_id]
                crossed = bool(canon_imm["window_flips"])
                shard = {
                    "schema": "v071_semwin_shard_v1",
                    "run_schema": "v071",
                    "window_id": akey, "task": task_name,
                    "split": split, "source_id": sid, "decision": d,
                    "tranche": w["tranche"], "branch": w["branch"],
                    "branch_transition_id":
                        w.get("branch_transition_id"),
                    "branch_family": w.get("branch_family",
                                           "stock_decision"),
                    "branch_provenance": w.get("branch_provenance",
                                               "stock_replay"),
                    "branch_audit_only": w.get("branch_audit_only",
                                               False),
                    "quat_gross_flag": sid.endswith("_s2142"),
                    "expected_crossing": w["expected_crossing"],
                    "crossed_canonical": crossed,
                    "crossing_reproduced": (crossed
                                            if w["expected_crossing"]
                                            else None),
                    "goal_ids": goal_ids, "canonical_goal": canon_id,
                    "frames": res["frames"],
                    "eef_seq": res["eef_seq"],
                    "valid_seq": res["valid_seq"],
                    "actions_env": res["actions_env"],
                    "steps": res["steps"],
                    "segments": res["segments"],
                    "immediate": res["immediate"],
                    "obj_before": obj_before, "obj_after": obj_after,
                    "qpos_before": qpos_before,
                    "qpos_after": qpos_after,
                    "grasp_after": grasp_state(env, bodies),
                    "video": vmeta,
                    "manifest_sha256": manifest["manifest_sha256"],
                }
                tmp = root / "shards" / f"{akey}.pt.tmp"
                torch.save(shard, tmp)
                tmp.replace(root / "shards" / f"{akey}.pt")
                done_shards.add(akey)
                write_index_row(
                    vindex, vmeta, run_id=RUN_ID,
                    checkpoint_tag="stored_actions+stock_pi05",
                    checkpoint_path=None, checkpoint_sha256=None,
                    manifest_sha256=manifest["manifest_sha256"],
                    task=task_name, seed=source["seed"],
                    arm=f"{w['tranche']}_{w['branch']}", split=split,
                    steps=res["steps"], success=crossed,
                    ordered_progress=canon_imm[
                        "ordered_prefix_after"],
                    damage=len(canon_imm["flips_10"]),
                    termination="window_end", root=root)
                print(f"[{akey}] steps={res['steps']} "
                      f"crossed={crossed} "
                      f"flips={canon_imm['window_flips']}", flush=True)
                restore(env, snap_a)
                env._env.env.done = False
                for gid in goal_ids:
                    restore_env_state(automata[gid],
                                      anchor_states[gid])
        finally:
            env.close()

    # ---- support report under the frozen rule --------------------------
    rows = []
    for p in sorted((root / "shards").glob("*.pt")):
        s = torch.load(p, weights_only=False)
        sigs = {gid: goal_signature(s["immediate"][gid])
                for gid in s["goal_ids"]}
        support = {}
        for gid in s["goal_ids"]:
            nonzero = (sigs[gid][0] != () or sigs[gid][1] != 0
                       or abs(sigs[gid][2]) > 0)
            discriminating = any(sigs[g2] != sigs[gid]
                                 for g2 in s["goal_ids"] if g2 != gid)
            support[gid] = (
                "positive" if (nonzero and discriminating) else
                "nonzero_nondiscriminative" if nonzero else
                "negative")
        rows.append({
            "window_id": s["window_id"], "task": s["task"],
            "split": s["split"], "source_id": s["source_id"],
            "decision": s["decision"], "tranche": s["tranche"],
            "branch": s["branch"],
            "branch_family": s["branch_family"],
            "branch_provenance": s["branch_provenance"],
            "steps": s["steps"],
            "crossed_canonical": s["crossed_canonical"],
            "crossing_reproduced": s["crossing_reproduced"],
            "goal_signatures": {g: repr(sigs[g])
                                for g in s["goal_ids"]},
            "support": support})
    cells = {}
    for r in rows:
        c = cells.setdefault(f"{r['task']}|{r['split']}", {
            "windows": 0, "crossed": 0, "positive_any_goal": 0,
            "sources": set(), "anchors": set(), "non_servo_crossed": 0})
        c["windows"] += 1
        c["sources"].add(r["source_id"])
        c["anchors"].add(f"{r['source_id']}_d{r['decision']}")
        if r["crossed_canonical"]:
            c["crossed"] += 1
            if r["branch_family"] != "servo":
                c["non_servo_crossed"] += 1
        if any(v == "positive" for v in r["support"].values()):
            c["positive_any_goal"] += 1
    report = {
        "schema": "v071_semwin_support_v1",
        "rule": ("positive iff nonzero per-goal delta AND a "
                 "scene-compatible goal with a different signature; "
                 "zero-delta rows negative; 724 union windows are "
                 "bulk negatives (0 flips, verified offline)"),
        "absent_cells": ABSENT_CELLS,
        "cells": {k: {"windows": v["windows"], "crossed": v["crossed"],
                      "positive_any_goal": v["positive_any_goal"],
                      "non_servo_crossed": v["non_servo_crossed"],
                      "independent_sources": len(v["sources"]),
                      "independent_anchors": len(v["anchors"])}
                  for k, v in sorted(cells.items())},
        "windows": rows,
    }
    (root / "support_report.json").write_text(
        json.dumps(report, indent=2))
    print(json.dumps(report["cells"], indent=1), flush=True)
    print(f"-> {root}", flush=True)


if __name__ == "__main__":
    main()
