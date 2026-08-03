#!/usr/bin/env python
"""V7.2C step 1 — freeze the ordinal-soft W2 teacher ledger.

Phases (sequential, resumable; ledger sealed BEFORE any candidate
branch executes — mechanically asserted):
  A sources+anchors: one stock rollout per task on the UNTOUCHED
    2302+10k family; ≤4 anchors/source, ≥5 decisions apart, the
    registered stall/recovery/late-chain/milestone-boundary priority;
    anchors frozen from pre-action state before W2 scoring.
  B bank+weights+seal: u0 (deployed) + 7 canonical + 2 atomic + ≤2
    scene-compatible distinct proposals (manifest defines M, no
    servo); frozen W2 scores via the scalar rank head only:
      r_i = (1/(M-1)) * sum_{j != i} sigmoid(s_i - s_j)
      w_i(beta) = pi_mix(i) exp(r_i/beta) / Z,  pi_mix = 1/M
    beta: start at 0.25 and only INCREASE by deterministic bisection
    on [0.25, 1000] (tol 1e-6) until ESS/M >= 0.5; +inf if the upper
    endpoint cannot satisfy the floor (ties keep the proposal prior).
    Matched within-family permutation ledger (weight multiset
    hash-permuted inside each family; singleton families unpermutable,
    recorded, mass excluded from the identity claim). Everything
    sealed outcome-blind.
  C execute: complete bank + u0_repeat audit, hash-randomized order,
    videos for every branch, R=2 sibling-CRN stock continuations —
    an assessment/replay tranche; teacher targets are NEVER filtered,
    relabeled, reweighted, or deleted by observed outcomes.

Output: results/libero_loho_public_v1/2026-08-03_v072_teacher_r1/
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

RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
LOSS_R1 = RESULTS / "2026-08-03_v072_lcwm_loss_r1"
DATA_R1 = RESULTS / "2026-08-03_v072_data_r1"
ROOT = RESULTS / "2026-08-03_v072_teacher_r1"
RID = "v072_teacher_r1"
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
BETA0, BETA_HI, BETA_TOL, ESS_FLOOR = 0.25, 1000.0, 1e-6, 0.5
N_SECONDARY_PERM = 0   # one designated permutation; no secondaries


def sha_seed(p: str) -> int:
    return int.from_bytes(hashlib.sha256(
        p.encode()).digest()[:8], "big") & ((1 << 63) - 1)


def obs_frame(env):
    return copy.deepcopy(env._format_raw_obs(
        env._env.env._get_observations()))


def soft_weights(r, beta):
    x = np.asarray(r, dtype=np.float64)
    if np.isinf(beta):
        w = np.ones_like(x)
    else:
        e = np.exp((x - x.max()) / beta)
        w = e
    w = w / w.sum()
    return w


def ess_frac(w):
    return float(1.0 / (np.sum(np.asarray(w) ** 2) * len(w)))


def freeze_beta(r):
    """Start at BETA0, only increase, deterministic bisection until
    ESS/M >= ESS_FLOOR; +inf if unreachable at BETA_HI."""
    if ess_frac(soft_weights(r, BETA0)) >= ESS_FLOOR:
        return BETA0
    if ess_frac(soft_weights(r, BETA_HI)) < ESS_FLOOR:
        return float("inf")
    lo, hi = BETA0, BETA_HI
    while hi - lo > BETA_TOL:
        mid = 0.5 * (lo + hi)
        if ess_frac(soft_weights(r, mid)) >= ESS_FLOOR:
            hi = mid
        else:
            lo = mid
    return hi


@torch.no_grad()
def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("AB", "C", "ALL"),
                    default="ALL")
    args = ap.parse_args()
    from lcwm.chassis import Pi05Runner
    from lcwm.loho_public import make_public_env
    from lcwm.probe_data import body_positions
    from lcwm.sampler import prefix_forward, sample_chunks
    from lcwm.snapshot import restore, snap
    from lcwm.task_automaton import (GoalAutomaton, fork_env_state,
                                     outcome_tuple,
                                     restore_env_state)
    from lcwm.v067_lineage import flow_noise, sha256_file
    from lcwm.v06_model import V06State
    from lcwm.video_recorder import VideoRecorder, write_index_row
    from scripts.collect_loho_v06 import obs_q
    from scripts.collect_v067_continuations import (full_qpos,
                                                    grasp_state)
    from scripts.collect_v069_corrections import (OBJ_DISPLAY,
                                                  REGION_DISPLAY)

    device = torch.device("cuda")
    goal_manifest = json.loads(
        (RESULTS / "goal_spec_manifest_v067.json").read_text())
    ROOT.mkdir(exist_ok=True)
    for sub in ("teacher_sources", "shards", "videos"):
        (ROOT / sub).mkdir(exist_ok=True)
    w2_path = LOSS_R1 / "checkpoints" / "w2_rank_state_final.pt"
    git_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True,
        text=True, cwd=REPO_ROOT).stdout.strip()
    mp = ROOT / "run_manifest.json"
    if not mp.exists():
        mp.write_text(json.dumps({
            "schema": "v072_teacher_manifest_v1",
            "run_schema": "v072", "run_id": RID, "git_sha": git_sha,
            "seeds": "2302+10k (previously untouched)",
            "bank": "u0(deployed)+7canon+2atomic+<=2distinct"
                    "+u0_repeat(audit); no servo; M manifest-defined",
            "teacher": {"score": "rank head only: r_i = mean_j "
                                 "sigmoid(s_i - s_j)",
                        "pi_mix": "uniform 1/M over manifest-valid "
                                  "candidates",
                        "beta": {"beta0": BETA0, "hi": BETA_HI,
                                 "tol": BETA_TOL,
                                 "ess_floor": ESS_FLOOR,
                                 "rule": "increase-only bisection; "
                                         "+inf keeps the prior"}},
            "w2_checkpoint": sha256_file(w2_path),
            "proposal_seed": f"SHA256('{RID}|proposal|src|dec|fam|i')",
            "cont_seed": f"SHA256('{RID}|cont|src|dec|goal|rep|cd')",
            "order_seed": f"SHA256('{RID}|order|src|dec|branch')",
            "perm_seed": f"SHA256('{RID}|perm|src|dec|family')",
        }, indent=2))

    runner = Pi05Runner(suite_name="libero_10")
    cfg = runner.policy.config
    wm = V06State().to(device)
    wm.load_state_dict(torch.load(w2_path,
                                  weights_only=False)["model"])
    wm.eval()

    # ---------------- Phase A: sources + frozen anchors -------------
    anchor_path = ROOT / "anchor_manifest.json"
    if not anchor_path.exists():
        all_anchors = []
        for ti, task in enumerate(TASK_ORDER):
            seed = 2302 + 10 * ti
            sid = f"{task}_teacher_s{seed}"
            spath = ROOT / "teacher_sources" / f"{sid}.pt"
            entry = goal_manifest["tasks"][task]
            canon_id = entry["canonical_goal_spec_id"]
            gspecs = entry["goal_specs"]
            gids = [canon_id] + [g for g in gspecs if g != canon_id]
            subgoals = gspecs[canon_id]["ordered_subgoals"]
            if not spath.exists():
                env = make_public_env(task,
                                      EPISODE_LENGTH[task] + 200)
                try:
                    runner.reset()
                    obs, _ = env.reset(seed=seed)
                    env._env.env.horizon = EPISODE_LENGTH[task] + 300
                    autos = {}
                    for g in gids:
                        a = GoalAutomaton(
                            gspecs[g]["ordered_subgoals"])
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
                             enumerate(ca.prev_valid) if not v),
                            None)
                        stall = stall + 1 if first_u == prev_u else 1
                        prev_u = first_u
                        opos_all = body_positions(
                            env, list(bodies.values()))
                        recent = [f for f in ca.flips
                                  if f[0] > t - 20]
                        n_un = sum(1 for v in ca.prev_valid
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
                        if st and first_u:
                            parts = first_u.split()
                            form, obj = parts[0], parts[1]
                            gsp = grasp_state(env,
                                              bodies)["grasped"]
                            eef = np.asarray(
                                obs_now["robot_state"]["eef"]["pos"])
                            opos = body_positions(
                                env, [bodies[obj]])[0] \
                                if obj in bodies else None
                            moved = float(np.abs(
                                opos_all - obj_prev).max())
                            if form == "pick_up" \
                                    and opos is not None:
                                dist = float(np.linalg.norm(
                                    eef - opos))
                                if (not gsp.get(obj) and PICK_NEAR
                                        <= dist <= PICK_FAR
                                        and moved < STABLE_MM):
                                    ok = {"form": form, "obj": obj,
                                          "region": None,
                                          "dist": dist}
                            elif form in ("place", "close", "open"):
                                region = (parts[2]
                                          if form == "place"
                                          else parts[1])
                                try:
                                    tgt = env._env.env.sim.data \
                                        .get_site_xpos(region).copy()
                                except Exception:
                                    tgt = None
                                refp = (opos if form == "place"
                                        and opos is not None
                                        else eef)
                                if tgt is not None:
                                    dist = float(np.linalg.norm(
                                        refp - tgt))
                                    held = (bool(gsp.get(obj))
                                            if form == "place"
                                            else True)
                                    if held and PLACE_NEAR <= dist \
                                            <= PLACE_FAR:
                                        ok = {"form": form,
                                              "obj": obj,
                                              "region": region,
                                              "dist": dist}
                        if ok:
                            cands.append({"decision": dec,
                                          "state_type": st, **ok})
                        obj_prev = opos_all
                        batch = runner._obs_to_policy_batch(obs,
                                                            instr)
                        pfx = prefix_forward(runner.policy, batch)
                        nz = flow_noise(sha_seed(
                            f"{RID}|proposal|{sid}|{dec}|deploy|0"),
                            cfg.chunk_size, cfg.max_action_dim)
                        chunk = sample_chunks(runner.policy, batch,
                                              n=1,
                                              noise=nz.to(device),
                                              prefix=pfx)
                        a_env_l, ex = [], 0
                        for a_env in runner.chunk_to_env(
                                chunk[:, :10]):
                            obs, _r, term, trunc, _i = env.step(
                                a_env)
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
                                     "actions_env":
                                         np.stack(a_env_l),
                                     "executed_len": ex,
                                     "q": obs_q(obs_now),
                                     "obj_before": opos_all,
                                     "first_unresolved": first_u})
                        dec += 1
                        if term or trunc:
                            break
                    tmp = spath.with_suffix(".tmp")
                    torch.save({"schema": "v072_teacher_source_v1",
                                "run_schema": "v072",
                                "source_id": sid, "task": task,
                                "task_index": ti, "seed": seed,
                                "split": "teacher",
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
            "schema": "v072_teacher_anchors_v1",
            "note": "frozen from pre-action stock state before W2 "
                    "scoring", "anchors": all_anchors}, indent=2))
        print(f"[anchors] frozen: {len(all_anchors)}", flush=True)
    anchors = json.loads(anchor_path.read_text())["anchors"]

    # ---------------- Phase B: bank + weights + seal ----------------
    cand_path = ROOT / "candidate_manifest.jsonl"
    led_path = ROOT / "teacher_ledger_pre_outcome.jsonl"
    perm_path = ROOT / "matched_permutation_ledger.jsonl"
    seal_path = ROOT / "ledger_seal.json"
    chunks_path = ROOT / "candidate_chunks.pt"
    if not seal_path.exists():
        cand_rows, led_rows, perm_rows = [], [], []
        chunk_store = {}
        for a in anchors:
            sid, d, task = a["source_id"], a["decision"], a["task"]
            src = torch.load(ROOT / "teacher_sources"
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
                batch = runner._obs_to_policy_batch(anchor_obs,
                                                    instr)
                pfx = prefix_forward(runner.policy, batch)
                bank = {"u0": ("canonical", instr,
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
                    rd = REGION_DISPLAY.get(
                        a["obj"], a["obj"].replace("_", " "))
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
                    b = runner._obs_to_policy_batch(anchor_obs,
                                                    lang)
                    p2 = prefix_forward(runner.policy, b)
                    nz = flow_noise(sha_seed(
                        f"{RID}|proposal|{sid}|{d}|{fam}|{j}"),
                        cfg.chunk_size, cfg.max_action_dim)
                    ch = sample_chunks(runner.policy, b, n=1,
                                       noise=nz.to(device),
                                       prefix=p2)
                    bank[f"sub{j}"] = (fam, lang,
                                       ch[0].float().cpu())
                # frozen W2 recurrent state (canonical prompt, full
                # history from reset) + rank-head scores.
                # Candidate tensors go DIRECTLY to E_a (already
                # pi0.5-normalized; the corrected one-normalization
                # contract).
                z = None
                for dd in range(d + 1):
                    rr = src["rows"][dd]
                    b = runner._obs_to_policy_batch(rr["obs"],
                                                    instr)
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
                cids = sorted(bank)
                scores = {}
                for cid in cids:
                    ch = torch.as_tensor(
                        np.asarray(bank[cid][2])).float() \
                        .to(device)[None, :10]
                    am = torch.ones(1, 10, dtype=torch.bool,
                                    device=device)
                    zt = wm.predict(z, ch, action_mask=am)
                    scores[cid] = float(wm.d_next(zt)["s"][0])
                M = len(cids)
                r_i = {c: float(np.mean(
                    [1.0 / (1.0 + np.exp(-(scores[c]
                                           - scores[o])))
                     for o in cids if o != c])) for c in cids}
                beta = freeze_beta([r_i[c] for c in cids])
                w = soft_weights([r_i[c] for c in cids], beta)
                weights = {c: float(w[k])
                           for k, c in enumerate(cids)}
                # matched within-family permutation of the weight
                # multiset (candidate identity control)
                fam_of = {c: bank[c][0] for c in cids}
                perm_w = dict(weights)
                unpermutable = []
                for fam in sorted(set(fam_of.values())):
                    pool = [c for c in cids if fam_of[c] == fam]
                    if len(pool) < 2:
                        unpermutable.append(fam)
                        continue
                    rot = 1 + sha_seed(
                        f"{RID}|perm|{sid}|{d}|{fam}") \
                        % (len(pool) - 1)
                    vals = [weights[c] for c in pool]
                    for k, c in enumerate(pool):
                        perm_w[c] = vals[(k + rot) % len(pool)]
                for cid in cids:
                    chunk_store[f"{sid}_d{d}_{cid}"] = bank[cid][2]
                    cand_rows.append({
                        "source_id": sid, "decision": d,
                        "candidate_id": cid,
                        "family": fam_of[cid],
                        "prompt": bank[cid][1],
                        "chunk_sha": hashlib.sha256(np.asarray(
                            bank[cid][2]).tobytes())
                        .hexdigest()[:16]})
                led_rows.append({
                    "source_id": sid, "decision": d, "task": task,
                    "state_type": a["state_type"], "M": M,
                    "scores": scores, "r": r_i, "beta": beta,
                    "ess_frac": ess_frac(w), "pi_mix": 1.0 / M,
                    "weights": weights})
                perm_rows.append({
                    "source_id": sid, "decision": d,
                    "perm_weights": perm_w,
                    "unpermutable_families": unpermutable,
                    "unpermutable_mass": float(sum(
                        weights[c] for c in cids
                        if fam_of[c] in unpermutable))})
                print(f"[teach] {sid}_d{d}: M={M} beta="
                      f"{beta if not np.isinf(beta) else 'inf'} "
                      f"ess={ess_frac(w):.3f} top="
                      f"{max(weights, key=weights.get)}",
                      flush=True)
            finally:
                env.close()
        with cand_path.open("w") as f:
            for r_ in cand_rows:
                f.write(json.dumps(r_) + "\n")
        with led_path.open("w") as f:
            for r_ in led_rows:
                f.write(json.dumps(r_) + "\n")
        with perm_path.open("w") as f:
            for r_ in perm_rows:
                f.write(json.dumps(r_) + "\n")
        torch.save(chunk_store, chunks_path)
        seal_path.write_text(json.dumps({
            "sealed_before_any_branch_execution": True,
            "teacher_ledger_sha256": sha256_file(led_path),
            "permutation_ledger_sha256": sha256_file(perm_path),
            "candidate_manifest_sha256": sha256_file(cand_path),
            "candidate_chunks_sha256": sha256_file(chunks_path),
        }, indent=2))
        print("[seal] teacher ledgers sealed (outcome-blind)",
              flush=True)
    if args.phase == "AB":
        return

    # ---------------- Phase C: execute assessment tranche -----------
    seal = json.loads(seal_path.read_text())
    from lcwm.v067_lineage import sha256_file as shaf
    assert seal["teacher_ledger_sha256"] == shaf(led_path), \
        "teacher ledger modified after seal"
    chunk_store = torch.load(chunks_path, weights_only=False)
    out_path = ROOT / "outcome_ledger.jsonl"
    vindex = ROOT / "video_index.jsonl"
    done = {p.stem for p in (ROOT / "shards").glob("*.pt")}
    for a in anchors:
        sid, d, task = a["source_id"], a["decision"], a["task"]
        akey = f"{sid}_d{d}"
        if akey in done:
            continue
        src = torch.load(ROOT / "teacher_sources" / f"{sid}.pt",
                         weights_only=False)
        entry = goal_manifest["tasks"][task]
        canon_id = entry["canonical_goal_spec_id"]
        gspecs = entry["goal_specs"]
        instr = src["language_canonical"]
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
            transitions = []
            for cid, kind in branches:
                restore(env, snap_a)
                env._env.env.done = False
                chk = float(np.abs(body_positions(
                    env, list(bodies.values()))
                    - row["obj_before"]).max())
                assert chk < REREACH_ATOL
                ba = GoalAutomaton(
                    gspecs[canon_id]["ordered_subgoals"])
                ba.bodies, ba.start_pos = auto0.bodies, \
                    auto0.start_pos
                restore_env_state(ba, a_state)
                flips0 = len(ba.flips)
                if kind == "audit":
                    acts = list(row["actions_env"])[:10]
                elif cid == "u0":
                    acts = list(row["actions_env"])[:10]
                else:
                    ch = chunk_store[f"{akey}_{cid}"]
                    acts = list(runner.chunk_to_env(
                        torch.as_tensor(np.asarray(ch))[None, :10]
                        .to(device)))[:10]
                vr = VideoRecorder(ROOT / "videos"
                                   / f"{akey}_{cid}.mp4")
                frames = [obs_frame(env)]
                vr.add(frames[0])
                eef_seq = [obs_q(frames[0])]
                a_env_l, steps = [], 0
                # per-step valid capture: the semantic-state gap that
                # blocked selector-row semantics is not repeated here
                valid_seq = [list(ba.prev_valid)]
                for a_env in acts:
                    _o, _r, tb, tr2, _i = env.step(a_env)
                    steps += 1
                    a_env_l.append(np.asarray(a_env))
                    ba.evaluate(env, steps)
                    valid_seq.append(list(ba.prev_valid))
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
                conts = []
                if kind != "audit":
                    b_end = snap(env, t=t + steps,
                                 suite_name="loho_public",
                                 task_id=0)
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
                            pcc = prefix_forward(runner.policy,
                                                 bcc)
                            ch2 = sample_chunks(
                                runner.policy, bcc, n=1,
                                noise=nz.to(device), prefix=pcc)
                            for a_env in runner.chunk_to_env(
                                    ch2[:, :10]):
                                _o2, _r, tm, tr3, _i = env.step(
                                    a_env)
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
                    checkpoint_tag="w2_teacher_bank",
                    checkpoint_path=None, checkpoint_sha256=None,
                    manifest_sha256=seal["teacher_ledger_sha256"],
                    task=task, seed=src["seed"], arm=cid,
                    split="teacher", steps=steps,
                    success=bool(conts and any(
                        c["success_by_100"] for c in conts)),
                    ordered_progress=ba.ordered_prefix(),
                    damage=len([f for f in ba.flips[flips0:]
                                if f[2] == -1]),
                    termination="branch_end", root=ROOT)
                transitions.append({
                    "transition_id": f"{akey}_{cid}",
                    "candidate_id": cid, "kind": kind,
                    "anchor": akey, "task": task,
                    "source_id": sid, "decision": d,
                    "frames": frames, "eef_seq": eef_seq,
                    "valid_seq_canon": valid_seq,
                    "actions_env": (np.stack(a_env_l) if a_env_l
                                    else np.zeros((0, 7))),
                    "steps": steps, "obj_after": obj_after,
                    "qpos_after": full_qpos(env),
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
            torch.save({"schema": "v072_teacher_shard_v1",
                        "run_schema": "v072", "anchor": akey,
                        "task": task, "source_id": sid,
                        "decision": d,
                        "transitions": transitions}, tmp)
            tmp.replace(ROOT / "shards" / f"{akey}.pt")
            done.add(akey)
        finally:
            env.close()
    with (ROOT / "replay_index.jsonl").open("w") as f:
        for p in sorted((ROOT / "shards").glob("*.pt")):
            s = torch.load(p, weights_only=False)
            for tr in s["transitions"]:
                f.write(json.dumps({
                    "transition_id": tr["transition_id"],
                    "anchor": s["anchor"], "task": s["task"],
                    "kind": tr["kind"], "steps": tr["steps"],
                    "n_continuations":
                        len(tr["continuations"])}) + "\n")
    print(f"-> {ROOT}", flush=True)


if __name__ == "__main__":
    main()
