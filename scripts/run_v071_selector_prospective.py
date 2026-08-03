#!/usr/bin/env python
"""V7.1.4B — prospective frozen-selector intervention (2026-08-03.md).

Phases (sequential, resumable; ledgers sealed BEFORE any candidate
branch executes — mechanically asserted):
  A sources+anchors: fresh stock rollouts seeds 2300+10k (2301+10k
    reserved, never touched); amended V7.1.1 state-diverse anchor rule;
    anchors frozen pre-scoring.
  B generate+score+seal: full bank chunks (u0 deployed + 7 canon +
    2 atomic + 2 distinct) under SHA proposal seeds; frozen selector +
    designated matched-permutation + 32 secondary derangements;
    ledgers sealed by SHA before execution.
  C execute: complete bank + u0_repeat audit, hash-randomized order,
    videos for every branch, R=2 sibling-CRN stock continuations.
  D analysis: registered gate aggregates + headroom + null safety.
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
UNION = RESULTS / "2026-08-02_v071_union_f1"
LCWM2C = RESULTS / "2026-08-02_v071_lcwm2c_r1"
OUTCOME = RESULTS / "2026-08-02_v071_outcome_r1"
ROOT = RESULTS / "2026-08-03_v071_selector_r1"
RID = "v071_selector_r1"
TASK_ORDER = ["loho_t1_drawer", "loho_t2_basket3", "loho_t3_tray",
              "loho_t4_tray", "loho_t5_drawer_cabinet"]
EPISODE_LENGTH = {"loho_t1_drawer": 700, "loho_t2_basket3": 900,
                  "loho_t3_tray": 900, "loho_t4_tray": 900,
                  "loho_t5_drawer_cabinet": 990}
MAX_ANCHORS, GAP, STALL_DECS = 4, 5, 3
PICK_NEAR, PICK_FAR, PLACE_NEAR, PLACE_FAR = 0.03, 0.30, 0.05, 0.30
STABLE_MM = 0.005
HORIZONS, R, CONT_MAX = (10, 30, 60, 100), 2, 100
REREACH_ATOL = 2e-3
N_SECONDARY = 32


def sha_seed(p: str) -> int:
    return int.from_bytes(hashlib.sha256(
        p.encode()).digest()[:8], "big") & ((1 << 63) - 1)


def obs_frame(env):
    return copy.deepcopy(env._format_raw_obs(
        env._env.env._get_observations()))


@torch.no_grad()
def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("AB", "C", "D", "ALL"),
                    default="ALL")
    ap.add_argument("--run", choices=("r1", "r2"), default="r1")
    args = ap.parse_args()
    global ROOT, RID
    seed_base = 2300
    scorer_path = OUTCOME / "scorer_ensembles.pt"
    if args.run == "r2":
        # the ONE final post-refresh prospective test on the reserved
        # 2301+10k family (pre-registered route; consumes the reserve)
        seed_base = 2301
        RID = "v071_selector_r2"
        ROOT = RESULTS / "2026-08-03_v071_selector_r2"
        scorer_path = (RESULTS / "2026-08-03_v071_outcome_refresh_r1"
                       / "scorer_ensembles.pt")
    from lcwm.chassis import Pi05Runner
    from lcwm.loho_public import make_public_env
    from lcwm.probe_data import body_positions
    from lcwm.sampler import prefix_forward, sample_chunks
    from lcwm.seq_prefix_cache import normalize_actions
    from lcwm.snapshot import restore, snap
    from lcwm.task_automaton import (GoalAutomaton, fork_env_state,
                                     outcome_tuple, paired_preference,
                                     restore_env_state)
    from lcwm.v067_lineage import flow_noise, sha256_file
    from lcwm.v06_model import V06State
    from lcwm.v071_selector import MASKED, SELECT_ORDER, select
    from lcwm.video_recorder import VideoRecorder, write_index_row
    from scripts.collect_loho_v06 import obs_q
    from scripts.collect_v067_continuations import (full_qpos,
                                                    grasp_state)
    from scripts.collect_v069_corrections import (OBJ_DISPLAY,
                                                  REGION_DISPLAY)
    from scripts.train_v071_3_outcome import Head

    device = torch.device("cuda")
    goal_manifest = json.loads(
        (RESULTS / "goal_spec_manifest_v067.json").read_text())
    tolerances = json.loads(
        (RESULTS / "v067_support_report.json").read_text()
    )["frozen_outcome_tolerances"]
    ref = torch.load(Path("/home/stargazer/Desktop/vla_wm/datasets"
                          "/seq_prefix_cache_v1/task0_demo0.pt"),
                     weights_only=False)
    ref_mean, ref_std = ref["action_mean"], ref["action_std_eps"]
    ROOT.mkdir(exist_ok=True)
    for sub in ("prospective_sources", "shards", "videos", "analysis"):
        (ROOT / sub).mkdir(exist_ok=True)
    git_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True,
        text=True, cwd=REPO_ROOT).stdout.strip()
    mp = ROOT / "run_manifest.json"
    if not mp.exists():
        mp.write_text(json.dumps({
            "schema": "v071_selector_manifest_v1",
            "run_schema": "v071", "run_id": RID,
            "git_sha": git_sha,
            "seeds": f"{seed_base}+10k",
            "scorer_path": str(scorer_path),
            "bank": "u0(deployed)+7canon+2atomic+2distinct"
                    "+u0_repeat(audit); no servo",
            "proposal_seed": f"SHA256('{RID}|proposal|src|dec|fam|i')",
            "cont_seed": f"SHA256('{RID}|cont|src|dec|goal|rep|cd')",
            "order_seed": f"SHA256('{RID}|order|src|dec')",
            "perm_seed": f"SHA256('{RID}|perm|j|src|dec')",
            "selector": {"masked": sorted(MASKED),
                         "order": [n for n, _i, _t in SELECT_ORDER]},
            "frozen": {
                "lc_full_selected": sha256_file(
                    LCWM2C / "checkpoints" / "lc_full_selected.pt"),
                "scorer_ensembles": sha256_file(scorer_path),
                "selector_code": sha256_file(
                    REPO_ROOT / "lcwm" / "v071_selector.py"),
                "union_manifest": sha256_file(
                    UNION / "union_manifest.json")},
            "tolerances": tolerances}, indent=2))

    runner = Pi05Runner(suite_name="libero_10")
    cfg = runner.policy.config
    wm = V06State().to(device)
    wm.load_state_dict(torch.load(
        LCWM2C / "checkpoints" / "lc_full_selected.pt",
        weights_only=False)["model"])
    wm.eval()
    heads = []
    for sd in torch.load(scorer_path,
                         weights_only=False)["lc_full"]:
        h = Head(768).to(device)
        h.load_state_dict(sd)
        h.eval()
        heads.append(h)

    def norm_chunk(a):
        ae = torch.as_tensor(np.asarray(a)).float()
        cn = normalize_actions(ae, ref_mean, ref_std)[None].to(device)
        am = (torch.arange(10, device=device)[None] < ae.shape[0])
        if cn.shape[1] < 10:
            cn = torch.cat([cn, torch.zeros(
                1, 10 - cn.shape[1], 7, device=device)], dim=1)
        return cn, am

    # ---------------- Phase A: sources + frozen anchors -----------------
    anchor_path = ROOT / "anchor_manifest.json"
    if not anchor_path.exists():
        all_anchors = []
        for ti, task in enumerate(TASK_ORDER):
            seed = seed_base + 10 * ti
            sid = f"{task}_prospect_s{seed}"
            spath = ROOT / "prospective_sources" / f"{sid}.pt"
            entry = goal_manifest["tasks"][task]
            canon_id = entry["canonical_goal_spec_id"]
            gspecs = entry["goal_specs"]
            gids = [canon_id] + [g for g in gspecs if g != canon_id]
            subgoals = gspecs[canon_id]["ordered_subgoals"]
            if not spath.exists():
                env = make_public_env(task, EPISODE_LENGTH[task] + 200)
                try:
                    runner.reset()
                    obs, _ = env.reset(seed=seed)
                    env._env.env.horizon = EPISODE_LENGTH[task] + 300
                    autos = {}
                    for g in gids:
                        a = GoalAutomaton(gspecs[g]["ordered_subgoals"])
                        a.start(env)
                        autos[g] = a
                    for a in autos.values():
                        a.evaluate(env, 0)
                    bodies = autos[canon_id].bodies
                    instr = env.task_description
                    ca = autos[canon_id]
                    rows, cands = [], []
                    obj_prev = body_positions(env,
                                              list(bodies.values()))
                    stall, prev_u, t, dec = 0, "___", 0, 0
                    term = trunc = False
                    while t < EPISODE_LENGTH[task]:
                        obs_now = copy.deepcopy(obs)
                        first_u = next(
                            (subgoals[i] for i, v in
                             enumerate(ca.prev_valid) if not v), None)
                        stall = stall + 1 if first_u == prev_u else 1
                        prev_u = first_u
                        opos_all = body_positions(
                            env, list(bodies.values()))
                        recent = [f for f in ca.flips if f[0] > t - 20]
                        n_un = sum(1 for v in ca.prev_valid if not v)
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
                        if st and first_u:
                            parts = first_u.split()
                            form, obj = parts[0], parts[1]
                            gsp = grasp_state(env, bodies)["grasped"]
                            eef = np.asarray(
                                obs_now["robot_state"]["eef"]["pos"])
                            opos = body_positions(
                                env, [bodies[obj]])[0] \
                                if obj in bodies else None
                            moved = float(np.abs(
                                opos_all - obj_prev).max())
                            if form == "pick_up" and opos is not None:
                                dist = float(np.linalg.norm(eef - opos))
                                if (not gsp.get(obj) and PICK_NEAR
                                        <= dist <= PICK_FAR
                                        and moved < STABLE_MM):
                                    ok = {"form": form, "obj": obj,
                                          "region": None, "dist": dist}
                            elif form in ("place", "close", "open"):
                                region = (parts[2] if form == "place"
                                          else parts[1])
                                try:
                                    tgt = env._env.env.sim.data \
                                        .get_site_xpos(region).copy()
                                except Exception:
                                    tgt = None
                                refp = (opos if form == "place"
                                        and opos is not None else eef)
                                if tgt is not None:
                                    dist = float(np.linalg.norm(
                                        refp - tgt))
                                    held = (bool(gsp.get(obj))
                                            if form == "place"
                                            else True)
                                    if held and PLACE_NEAR <= dist \
                                            <= PLACE_FAR:
                                        ok = {"form": form, "obj": obj,
                                              "region": region,
                                              "dist": dist}
                        if ok:
                            cands.append({"decision": dec,
                                          "state_type": st, **ok})
                        obj_prev = opos_all
                        batch = runner._obs_to_policy_batch(obs, instr)
                        pfx = prefix_forward(runner.policy, batch)
                        nz = flow_noise(sha_seed(
                            f"{RID}|proposal|{sid}|{dec}|deploy|0"),
                            cfg.chunk_size, cfg.max_action_dim)
                        chunk = sample_chunks(runner.policy, batch,
                                              n=1, noise=nz.to(device),
                                              prefix=pfx)
                        a_env_l, ex = [], 0
                        for a_env in runner.chunk_to_env(chunk[:, :10]):
                            obs, _r, term, trunc, _i = env.step(a_env)
                            a_env_l.append(np.asarray(a_env))
                            t += 1
                            ex += 1
                            for au in autos.values():
                                au.evaluate(env, t)
                            if term or trunc:
                                break
                        rows.append({"decision": dec,
                                     "t_start": t - ex,
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
                    torch.save({"schema": "v071_prospect_source_v1",
                                "run_schema": "v071",
                                "source_id": sid, "task": task,
                                "task_index": ti, "seed": seed,
                                "split": "train",
                                "language_canonical": instr,
                                "rows": rows,
                                "anchor_candidates": cands}, tmp)
                    tmp.replace(spath)
                    print(f"[src] {sid}: {len(rows)} dec, "
                          f"{len(cands)} cands", flush=True)
                finally:
                    env.close()
            src = torch.load(spath, weights_only=False)
            chosen, seen = [], set()

            def gap_ok(c):
                return all(abs(c["decision"] - x["decision"]) >= GAP
                           for x in chosen)
            for want_new in (True, False):
                for c in src["anchor_candidates"]:
                    if len(chosen) >= MAX_ANCHORS:
                        break
                    if c in chosen or not gap_ok(c):
                        continue
                    if want_new and c["state_type"] in seen:
                        continue
                    chosen.append(c)
                    seen.add(c["state_type"])
            for c in chosen:
                all_anchors.append({"source_id": sid, "task": task,
                                    "seed": seed, **c})
        anchor_path.write_text(json.dumps({
            "schema": "v071_selector_anchors_v1",
            "note": "frozen from pre-action stock state ONLY, before "
                    "any candidate scoring", "anchors": all_anchors},
            indent=2))
        print(f"[anchors] frozen: {len(all_anchors)}", flush=True)

    anchors = json.loads(anchor_path.read_text())["anchors"]

    # ---------------- Phase B: generate + score + seal ------------------
    cand_path = ROOT / "candidate_manifest.jsonl"
    led_path = ROOT / "selector_ledger_pre_outcome.jsonl"
    perm_path = ROOT / "matched_permutation_ledger.jsonl"
    seal_path = ROOT / "ledger_seal.json"
    chunks_path = ROOT / "candidate_chunks.pt"
    if not seal_path.exists():
        cand_rows, led_rows, perm_rows = [], [], []
        chunk_store = {}
        for a in anchors:
            sid, d, task = a["source_id"], a["decision"], a["task"]
            src = torch.load(ROOT / "prospective_sources"
                             / f"{sid}.pt", weights_only=False)
            entry = goal_manifest["tasks"][task]
            canon_id = entry["canonical_goal_spec_id"]
            gspecs = entry["goal_specs"]
            instr = src["language_canonical"]
            env = make_public_env(task, EPISODE_LENGTH[task] + 200)
            try:
                runner.reset()
                env.reset(seed=src["seed"])
                env._env.env.horizon = EPISODE_LENGTH[task] + 400
                auto = GoalAutomaton(
                    gspecs[canon_id]["ordered_subgoals"])
                auto.start(env)
                auto.evaluate(env, 0)
                bodies = auto.bodies
                t = 0
                for i in range(d):
                    for a_env in src["rows"][i]["actions_env"]:
                        env.step(a_env)
                        t += 1
                        auto.evaluate(env, t)
                row = src["rows"][d]
                err = float(np.abs(body_positions(
                    env, list(bodies.values()))
                    - row["obj_before"]).max())
                assert err < REREACH_ATOL, f"{sid}_d{d} {err:.2e}"
                anchor_obs = obs_frame(env)
                batch = runner._obs_to_policy_batch(anchor_obs, instr)
                pfx = prefix_forward(runner.policy, batch)
                bank = {"u0": ("deployed", None,
                               row["chunk_norm"])}
                for i in range(1, 8):
                    nz = flow_noise(sha_seed(
                        f"{RID}|proposal|{sid}|{d}|canonical|{i}"),
                        cfg.chunk_size, cfg.max_action_dim)
                    ch = sample_chunks(runner.policy, batch, n=1,
                                       noise=nz.to(device),
                                       prefix=pfx)
                    bank[f"c{i}"] = ("canonical", instr,
                                     ch[0].float().cpu())
                if a["form"] in ("close", "open"):
                    rd = REGION_DISPLAY.get(a["obj"],
                                            a["obj"].replace("_", " "))
                    at = [f"{a['form']} {rd}",
                          f"{a['form']} the drawer"]
                else:
                    od = OBJ_DISPLAY.get(
                        (task, a["obj"]),
                        "the " + a["obj"].rsplit("_", 1)[0]
                        .replace("_", " "))
                    at = [f"pick up {od}",
                          (f"put {od} in "
                           f"{REGION_DISPLAY.get(a['region'], 'place')}"
                           if a["region"] else
                           f"lift {od} off the table")]
                dl = [gspecs[g]["language"] for g in gspecs
                      if g != canon_id][:2]
                for j, lang in enumerate(at + dl):
                    fam = "atomic" if j < 2 else "distinct"
                    b = runner._obs_to_policy_batch(anchor_obs, lang)
                    p2 = prefix_forward(runner.policy, b)
                    nz = flow_noise(sha_seed(
                        f"{RID}|proposal|{sid}|{d}|{fam}|{j}"),
                        cfg.chunk_size, cfg.max_action_dim)
                    ch = sample_chunks(runner.policy, b, n=1,
                                       noise=nz.to(device), prefix=p2)
                    bank[f"sub{j}"] = (fam, lang,
                                       ch[0].float().cpu())
                # frozen LC state + ensemble preds
                z = None
                for dd in range(d + 1):
                    rr = src["rows"][dd]
                    b = runner._obs_to_policy_batch(rr["obs"], instr)
                    p2 = prefix_forward(runner.policy, b)
                    h, m = p2.hidden.float(), p2.pad_masks.bool()
                    if z is None:
                        z = wm.initial_state(h, m)
                    else:
                        prev = src["rows"][dd - 1]
                        aa = prev["chunk_norm"][None, :10].float() \
                            .to(device)
                        am = (torch.arange(10, device=device)[None]
                              < prev["executed_len"])
                        z = wm.step(z, aa, h, m, action_mask=am)
                preds = {}
                for cid, (fam, lang, ch) in bank.items():
                    cn, am = norm_chunk(ch[:10])
                    zt = wm.predict(z, cn, action_mask=am)
                    x = torch.cat([z.mean(dim=1)[0],
                                   zt.mean(dim=1)[0]])[None]
                    outs = []
                    for hd in heads:
                        o = hd(x)[0]
                        o = torch.cat([torch.sigmoid(o[:1]), o[1:]])
                        outs.append(o.cpu().numpy())
                    preds[cid] = np.stack(outs)
                    cand_rows.append({
                        "source_id": sid, "decision": d,
                        "candidate_id": cid, "family": fam,
                        "prompt": lang,
                        "chunk_sha": hashlib.sha256(
                            np.asarray(ch).tobytes())
                        .hexdigest()[:16]})
                    chunk_store[f"{sid}_d{d}_{cid}"] = ch
                led = select(preds, "u0", run_id=RID,
                             source_id=sid, decision=d)
                led["anchor_state_type"] = a["state_type"]
                led_rows.append(led)
                # matched permutation controls (designated j=0 + 32)
                fam_of = {c: bank[c][0] for c in bank}
                perms = {}
                for j in range(N_SECONDARY + 1):
                    if not led["intervention"]:
                        perms[j] = "u0"
                        continue
                    fam = fam_of[led["selected"]]
                    pool = sorted(c for c in bank
                                  if fam_of[c] == fam and c != "u0")
                    if len(pool) < 2:
                        perms[j] = led["selected"]
                        continue
                    rot = 1 + sha_seed(
                        f"{RID}|perm|{j}|{sid}|{d}") \
                        % (len(pool) - 1)
                    perms[j] = pool[(pool.index(led["selected"])
                                     + rot) % len(pool)]
                    assert perms[j] != led["selected"]
                perm_rows.append({
                    "source_id": sid, "decision": d,
                    "designated": perms[0],
                    "secondary": {str(j): perms[j]
                                  for j in range(1, N_SECONDARY + 1)},
                    "lc_selected": led["selected"],
                    "intervention": led["intervention"]})
                print(f"[score] {sid}_d{d}: sel={led['selected']} "
                      f"int={led['intervention']} "
                      f"({led['selection_reason']})", flush=True)
            finally:
                env.close()
        with cand_path.open("w") as f:
            for r in cand_rows:
                f.write(json.dumps(r) + "\n")
        with led_path.open("w") as f:
            for r in led_rows:
                f.write(json.dumps(r) + "\n")
        with perm_path.open("w") as f:
            for r in perm_rows:
                f.write(json.dumps(r) + "\n")
        torch.save(chunk_store, chunks_path)
        seal_path.write_text(json.dumps({
            "sealed_before_any_branch_execution": True,
            "selector_ledger_sha256": sha256_file(led_path),
            "permutation_ledger_sha256": sha256_file(perm_path),
            "candidate_manifest_sha256": sha256_file(cand_path),
            "candidate_chunks_sha256": sha256_file(chunks_path),
        }, indent=2))
        print("[seal] ledgers sealed", flush=True)
    if args.phase == "AB":
        return

    # ---------------- Phase C: execute the complete bank ----------------
    seal = json.loads(seal_path.read_text())
    assert seal["selector_ledger_sha256"] == sha256_file(led_path) \
        and seal["permutation_ledger_sha256"] == sha256_file(perm_path), \
        "ledger modified after seal"
    chunk_store = torch.load(chunks_path, weights_only=False)
    out_path = ROOT / "outcome_ledger.jsonl"
    fid_path = ROOT / "replay_fidelity.json"
    vindex = ROOT / "video_index.jsonl"
    done = {p.stem for p in (ROOT / "shards").glob("*.pt")}
    fids = json.loads(fid_path.read_text()) if fid_path.exists() \
        else {}
    for a in anchors:
        sid, d, task = a["source_id"], a["decision"], a["task"]
        akey = f"{sid}_d{d}"
        if akey in done:
            continue
        src = torch.load(ROOT / "prospective_sources" / f"{sid}.pt",
                         weights_only=False)
        entry = goal_manifest["tasks"][task]
        canon_id = entry["canonical_goal_spec_id"]
        gspecs = entry["goal_specs"]
        instr = src["language_canonical"]
        term_preds = [tuple(p) for p in
                      gspecs[canon_id]["terminal_predicates"]]
        env = make_public_env(task, EPISODE_LENGTH[task] + 200)
        try:
            runner.reset()
            env.reset(seed=src["seed"])
            env._env.env.horizon = EPISODE_LENGTH[task] + 400
            auto0 = GoalAutomaton(
                gspecs[canon_id]["ordered_subgoals"])
            auto0.start(env)
            auto0.evaluate(env, 0)
            bodies = auto0.bodies
            t = 0
            for i in range(d):
                for a_env in src["rows"][i]["actions_env"]:
                    env.step(a_env)
                    t += 1
                    auto0.evaluate(env, t)
            row = src["rows"][d]
            err = float(np.abs(body_positions(
                env, list(bodies.values()))
                - row["obj_before"]).max())
            assert err < REREACH_ATOL
            snap_a = snap(env, t=t, suite_name="loho_public",
                          task_id=0)
            a_state = fork_env_state(auto0)
            cids = sorted(c for c in
                          {k.rsplit("_", 1)[-1]
                           for k in chunk_store
                           if k.startswith(f"{akey}_")})
            branches = [(c, "chunk") for c in cids] \
                + [("u0_repeat", "audit")]
            branches.sort(key=lambda b: sha_seed(
                f"{RID}|order|{sid}|{d}|{b[0]}"))
            transitions, u0_end = [], None
            for cid, kind in branches:
                restore(env, snap_a)
                env._env.env.done = False
                chk = float(np.abs(body_positions(
                    env, list(bodies.values()))
                    - row["obj_before"]).max())
                assert chk < REREACH_ATOL, "post-restore checksum"
                ba = GoalAutomaton(
                    gspecs[canon_id]["ordered_subgoals"])
                ba.bodies, ba.start_pos = auto0.bodies, \
                    auto0.start_pos
                restore_env_state(ba, a_state)
                flips0 = len(ba.flips)
                if kind == "audit":
                    acts = list(row["actions_env"])[:10]
                else:
                    ch = chunk_store[f"{akey}_{cid}"]
                    acts = (list(row["actions_env"])[:10]
                            if cid == "u0" else
                            list(runner.chunk_to_env(
                                torch.as_tensor(ch)[None, :10]
                                .to(device)))[:10])
                vr = VideoRecorder(ROOT / "videos"
                                   / f"{akey}_{cid}.mp4")
                frames = [obs_frame(env)]
                vr.add(frames[0])
                eef_seq = [obs_q(frames[0])]
                a_env_l, steps = [], 0
                for a_env in acts:
                    _o, _r, tb, tr2, _i = env.step(a_env)
                    steps += 1
                    a_env_l.append(np.asarray(a_env))
                    ba.evaluate(env, steps)
                    f = obs_frame(env)
                    frames.append(f)
                    vr.add(f)
                    eef_seq.append(obs_q(f))
                    if tb:
                        env._env.env.done = False
                    if tr2:
                        break
                obj_after = body_positions(env, list(bodies.values()))
                qpos_after = full_qpos(env)
                if cid == "u0":
                    u0_end = (np.asarray(eef_seq[-1]), obj_after,
                              qpos_after)
                if kind == "audit" and u0_end is not None:
                    dq = np.abs(np.asarray(eef_seq[-1]) - u0_end[0])
                    fids[akey] = {
                        "eef_pos": float(dq[:3].max()),
                        "eef_quat": float(dq[3:7].max()),
                        "gripper": float(dq[7:].max()),
                        "obj_pos": float(np.abs(
                            obj_after - u0_end[1]).max()),
                        "qpos": float(np.abs(
                            qpos_after - u0_end[2]).max())}
                conts = []
                if kind != "audit":
                    b_end = snap(env, t=t + steps,
                                 suite_name="loho_public", task_id=0)
                    for rep in range(R):
                        restore(env, b_end)
                        env._env.env.done = False
                        ca = GoalAutomaton(
                            gspecs[canon_id]["ordered_subgoals"])
                        ca.bodies, ca.start_pos = auto0.bodies, \
                            auto0.start_pos
                        restore_env_state(ca, fork_env_state(ba))
                        obs_c = obs_frame(env)
                        q_at, sc, cd2, stop = {}, 0, 0, False
                        while sc < CONT_MAX and not stop:
                            nz = flow_noise(sha_seed(
                                f"{RID}|cont|{sid}|{d}|{canon_id}"
                                f"|{rep}|{cd2}"), cfg.chunk_size,
                                cfg.max_action_dim)
                            bcc = runner._obs_to_policy_batch(
                                obs_c, instr)
                            pcc = prefix_forward(runner.policy, bcc)
                            ch2 = sample_chunks(
                                runner.policy, bcc, n=1,
                                noise=nz.to(device), prefix=pcc)
                            for a_env in runner.chunk_to_env(
                                    ch2[:, :10]):
                                _o2, _r, tm, tr3, _i = env.step(a_env)
                                sc += 1
                                ca.evaluate(env, sc)
                                obs_c = obs_frame(env)
                                if sc in HORIZONS:
                                    q_at[sc] = ca.q_valid()
                                if tm or tr3:
                                    stop = True
                                    break
                                if sc >= CONT_MAX:
                                    break
                            cd2 += 1
                        for hz in HORIZONS:
                            q_at.setdefault(hz, ca.q_valid())
                        conts.append(outcome_tuple(ca, env, steps,
                                                   q_at))
                vm = vr.close(completed=True)
                write_index_row(
                    vindex, vm, run_id=RID,
                    checkpoint_tag="frozen_bank",
                    checkpoint_path=None, checkpoint_sha256=None,
                    manifest_sha256=seal["selector_ledger_sha256"],
                    task=task, seed=src["seed"], arm=cid,
                    split="train", steps=steps,
                    success=bool(conts and any(
                        c["success_by_100"] for c in conts)),
                    ordered_progress=ba.ordered_prefix(),
                    damage=len([f for f in ba.flips[flips0:]
                                if f[2] == -1]),
                    termination="branch_end", root=ROOT)
                transitions.append({
                    "transition_id": f"{akey}_{cid}",
                    "candidate_id": cid, "kind": kind,
                    "anchor": akey, "task": task, "source_id": sid,
                    "decision": d, "frames": frames,
                    "eef_seq": eef_seq,
                    "actions_env": (np.stack(a_env_l) if a_env_l
                                    else np.zeros((0, 7))),
                    "steps": steps, "obj_after": obj_after,
                    "qpos_after": qpos_after,
                    "grasp_after": grasp_state(env, bodies),
                    "continuations": conts})
                with out_path.open("a") as f:
                    f.write(json.dumps({
                        "anchor": akey, "candidate_id": cid,
                        "kind": kind, "steps": steps,
                        "continuations": [
                            {k: (v if k != "q_at_horizons" else
                                 {str(kk): vv
                                  for kk, vv in v.items()})
                             for k, v in c.items()}
                            for c in conts]}) + "\n")
                print(f"  [{akey}] {cid} steps={steps} "
                      f"conts={len(conts)}", flush=True)
            tmp = ROOT / "shards" / f"{akey}.pt.tmp"
            torch.save({"schema": "v071_selector_shard_v1",
                        "run_schema": "v071", "anchor": akey,
                        "task": task, "source_id": sid,
                        "decision": d, "transitions": transitions},
                       tmp)
            tmp.replace(ROOT / "shards" / f"{akey}.pt")
            fid_path.write_text(json.dumps(fids, indent=2))
            done.add(akey)
        finally:
            env.close()
    if args.phase == "C":
        return

    # ---------------- Phase D: gate analysis ----------------------------
    led_rows = [json.loads(x) for x in led_path.open()]
    perm_rows = [json.loads(x) for x in perm_path.open()]
    led_by, perm_by = {}, {}
    for r in led_rows:
        led_by[f"{r['source_id']}_d{r['decision']}"] = r
    for r in perm_rows:
        perm_by[f"{r['source_id']}_d{r['decision']}"] = r
    shards = {p.stem: torch.load(p, weights_only=False)
              for p in sorted((ROOT / "shards").glob("*.pt"))}

    def conts_of(akey, cid):
        for tr in shards[akey]["transitions"]:
            if tr["candidate_id"] == cid and tr["kind"] != "audit":
                return tr["continuations"]
        return None

    def margin(akey, cid_a, cid_b):
        ca, cb = conts_of(akey, cid_a), conts_of(akey, cid_b)
        if not ca or not cb:
            return None
        return paired_preference(ca, cb, tolerances)

    per_anchor = []
    for akey, led in led_by.items():
        if akey not in shards:
            continue
        sel = led["selected"]
        des = perm_by[akey]["designated"]
        cids = [tr["candidate_id"] for tr in
                shards[akey]["transitions"] if tr["kind"] != "audit"]
        gt = {}
        for i, ci in enumerate(cids):
            for cj in cids[i + 1:]:
                gt[f"{ci}|{cj}"] = margin(akey, ci, cj)
        oracle = max(cids, key=lambda c: sum(
            1 for o in cids if o != c
            and margin(akey, c, o) == 1))
        canon = [c for c in cids
                 if c == "u0" or c.startswith("c")]
        oracle_canon = max(canon, key=lambda c: sum(
            1 for o in canon if o != c
            and margin(akey, c, o) == 1))
        per_anchor.append({
            "anchor": akey, "task": shards[akey]["task"],
            "source_id": shards[akey]["source_id"],
            "state_type": led.get("anchor_state_type"),
            "lc_selected": sel,
            "intervention": led["intervention"],
            "designated_control": des,
            "lc_vs_stock": margin(akey, sel, "u0")
            if sel != "u0" else 0,
            "control_vs_stock": margin(akey, des, "u0")
            if des != "u0" else 0,
            "lc_vs_control": margin(akey, sel, des)
            if sel != des else 0,
            "oracle_full": oracle,
            "oracle_full_beats_stock": margin(akey, oracle, "u0"),
            "oracle_canon": oracle_canon,
            "oracle_canon_beats_stock":
                margin(akey, oracle_canon, "u0"),
            "gt_pairs": gt})
    # equal task->source->anchor aggregation
    def agg(key):
        by_task = defaultdict(lambda: defaultdict(list))
        for r in per_anchor:
            v = r[key]
            if v is not None:
                by_task[r["task"]][r["source_id"]].append(v)
        task_means = []
        for tsk, srcs in by_task.items():
            task_means.append(float(np.mean(
                [np.mean(v) for v in srcs.values()])))
        return float(np.mean(task_means)) if task_means else None

    n_int = sum(1 for r in per_anchor if r["intervention"])
    null_anchors = [r for r in per_anchor
                    if r["oracle_full_beats_stock"] != 1]
    src_dirs = defaultdict(list)
    for r in per_anchor:
        src_dirs[r["source_id"]].append(r["lc_vs_stock"] or 0)
    pos_sources = [s for s, v in src_dirs.items()
                   if float(np.mean(v)) > 0]
    report = {
        "n_anchors": len(per_anchor),
        "n_interventions": n_int,
        "aggregate_lc_vs_stock_all_anchors": agg("lc_vs_stock"),
        "aggregate_lc_vs_control_interventions": (
            agg("lc_vs_control") if n_int else None),
        "sources_positive_direction": pos_sources,
        "n_sources_positive": len(pos_sources),
        "null_anchor_lc_vs_stock_mean": (float(np.mean(
            [r["lc_vs_stock"] or 0 for r in null_anchors]))
            if null_anchors else None),
        "n_null_anchors": len(null_anchors),
        "oracle_headroom_full": sum(
            1 for r in per_anchor
            if r["oracle_full_beats_stock"] == 1),
        "oracle_headroom_canonical": sum(
            1 for r in per_anchor
            if r["oracle_canon_beats_stock"] == 1),
        "per_anchor": per_anchor,
    }
    (ROOT / "analysis" / "selector_report.json").write_text(
        json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items()
                      if k != "per_anchor"}, indent=1), flush=True)
    print(f"-> {ROOT}", flush=True)


if __name__ == "__main__":
    main()
