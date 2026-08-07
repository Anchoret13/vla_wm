#!/usr/bin/env python
"""V7.3C teacher step — outcome-derived model teacher + grounded
ledger (2026-08-03.md "Outcome-derived model teacher").

Phase A (tensor-only, outcome-blind):
  - assert teacher-source disjointness: no source ID, transition ID,
    or (history_hash, candidate_chunk_hash) overlap with any LCWM
    train/dev or grounded-correction source;
  - freeze component-wise source-dev error margins of the frozen
    final.pt on the v073 dev anchors (per outcome component:
    median absolute error) BEFORE any teacher decoding is inspected;
  - for each of the 9 unexecuted teacher banks: unroll the frozen
    state, decode the predicted tuple per candidate, apply the
    registered eligibility rule with masked components skipped
    (damage, tau masked → the lexicographic chain is
    terminal success → next-milestone improvement → multi-horizon
    task-valid progress), pessimistic(i) = pred - margin vs
    optimistic(u0) = pred + margin;
  - strongest eligible tuple gets the mass; exact prediction ties
    share; no eligible candidate → abstention (grounded/demo/
    retention only);
  - matched-random ledger: same eligible-state mask, mass, family
    composition, and candidate count, hash-permuted identity;
  - GROUNDED ledger: the milestone-positive replay-clean V7.3B
    branches (t2|train cell) + released V7.2 teacher-assessment
    candidates whose FULL recomputed tuple beats u0 in both repeats;
  - seal everything by SHA before Phase B.

Phase B (simulator): execute the 9 teacher banks once as an
assessment/replay tranche with videos and R=2 sibling-CRN
continuations. Outcomes may enter a future revision, never V7.3
model or policy weights.

Output: results/libero_loho_public_v1/2026-08-07_v073_teacher_r1/
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

from lcwm import v073_schedule as S  # noqa: E402
from lcwm.v06_model import V06State  # noqa: E402

RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
LCWM = RESULTS / "2026-08-06_v073_lcwm_r1"
V73 = RESULTS / "2026-08-04_v073_data_r1"
V72T = RESULTS / "2026-08-03_v072_teacher_r1"
OUT = RESULTS / "2026-08-07_v073_teacher_r1"
RID = "v073_teacher_r1"
EPISODE_LENGTH = {"loho_t1_drawer": 700, "loho_t2_basket3": 900,
                  "loho_t3_tray": 900, "loho_t4_tray": 900,
                  "loho_t5_drawer_cabinet": 990}
HORIZONS, R, CONT_MAX = (10, 30, 60, 100), 2, 100
REREACH_ATOL = 2e-3
# both-repeat q tolerance (frozen replay-noise tolerance)
TOL_QM = 0.041666666666666664


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
    ap.add_argument("--phase", choices=("A", "B", "ALL"),
                    default="ALL")
    args = ap.parse_args()
    from lcwm.chassis import Pi05Runner
    from lcwm.loho_public import make_public_env
    from lcwm.probe_data import body_positions
    from lcwm.sampler import prefix_forward, sample_chunks
    from lcwm.seq_prefix_cache import normalize_actions
    from lcwm.snapshot import restore, snap
    from lcwm.task_automaton import (GoalAutomaton, fork_env_state,
                                     outcome_tuple,
                                     restore_env_state)
    from lcwm.v067_lineage import flow_noise, sha256_file
    from lcwm.video_recorder import VideoRecorder, write_index_row
    from scripts.collect_loho_v06 import obs_q
    from scripts.collect_v067_continuations import grasp_state

    device = torch.device("cuda")
    OUT.mkdir(exist_ok=True)
    for sub in ("shards", "videos"):
        (OUT / sub).mkdir(exist_ok=True)
    goal_manifest = json.loads(
        (RESULTS / "goal_spec_manifest_v067.json").read_text())
    ref = torch.load(Path("/home/stargazer/Desktop/vla_wm/datasets"
                          "/seq_prefix_cache_v1/task0_demo0.pt"),
                     weights_only=False)
    ref_mean, ref_std = ref["action_mean"], ref["action_std_eps"]

    anchors73 = json.loads(
        (V73 / "anchor_manifest.json").read_text())["anchors"]
    teach_anchors = [a for a in anchors73 if a["kind"] == "teach"]
    acq_sids = {a["source_id"] for a in anchors73
                if a["kind"] == "acq"}
    teach_sids = {a["source_id"] for a in teach_anchors}
    assert not (teach_sids & acq_sids), "teacher source overlap"
    # extended registered disjointness: teacher candidate chunk
    # hashes must not appear among ANY LCWM-training chunks, and
    # teacher source IDs must be absent from every training universe
    train_sids = set(acq_sids)
    union72 = json.loads((RESULTS / "2026-08-02_v071_union_f1"
                          / "union_manifest.json").read_text())
    train_sids |= set(union72["source_histories"])
    for p_ in (V72T / "teacher_sources").glob("*.pt"):
        train_sids.add(p_.stem)
    assert not (teach_sids & train_sids), \
        f"teacher sources appear in training universes: " \
        f"{teach_sids & train_sids}"
    train_chunk_shas = set()
    for p_ in sorted((V73 / "shards").glob("*.pt")):
        sh_ = torch.load(p_, weights_only=False)
        for tr_ in sh_["transitions"]:
            if tr_.get("chunk_norm") is not None:
                train_chunk_shas.add(hashlib.sha256(
                    np.asarray(tr_["chunk_norm"]).tobytes())
                    .hexdigest())
    teach_chunk_shas = set()
    for a_ in teach_anchors:
        tb_ = torch.load(
            V73 / "teacher_banks"
            / f"{a_['source_id']}_d{a_['decision']}.pt",
            weights_only=False)
        for _cid, (_f, _l, ch_) in tb_["bank"].items():
            teach_chunk_shas.add(hashlib.sha256(
                np.asarray(ch_).tobytes()).hexdigest())
    overlap = train_chunk_shas & teach_chunk_shas
    assert not overlap, f"chunk-hash overlap: {len(overlap)}"

    runner = Pi05Runner(suite_name="libero_10")
    cfg = runner.policy.config
    wm = V06State().to(device)
    wm.load_state_dict(torch.load(LCWM / "checkpoints" / "final.pt",
                                  weights_only=False)["model"])
    wm.eval()

    def norm_chunk(cn_or_env, is_norm):
        if is_norm:
            cn = torch.as_tensor(np.asarray(cn_or_env),
                                 dtype=torch.float32,
                                 device=device)[None, :10]
        else:
            ae = torch.from_numpy(np.asarray(cn_or_env)).float()
            cn = normalize_actions(
                ae, ref_mean, ref_std)[None].to(device)[:, :10]
        if cn.shape[1] < 10:
            cn = torch.cat([cn, torch.zeros(
                1, 10 - cn.shape[1], 7, device=device)], dim=1)
        am = torch.ones(1, 10, dtype=torch.bool, device=device)
        return cn, am

    def unroll_live(src, d, instr):
        z = None
        for dd in range(d + 1):
            b = runner._obs_to_policy_batch(src["rows"][dd]["obs"],
                                            instr)
            pfx = prefix_forward(runner.policy, b)
            h, m = pfx.hidden.float(), pfx.pad_masks.bool()
            if z is None:
                z = wm.initial_state(h, m)
            else:
                prev = src["rows"][dd - 1]
                aa = prev["chunk_norm"][None, :10].float().to(device)
                am = (torch.arange(10, device=device)[None]
                      < prev["executed_len"])
                z = wm.step(z, aa, h, m, action_mask=am)
        return z

    def n_sub_of(task):
        entry = goal_manifest["tasks"][task]
        cg = entry["canonical_goal_spec_id"]
        return len(entry["goal_specs"][cg]["ordered_subgoals"])

    def decode(z, cn, am, task):
        zt = wm.predict(z, cn, action_mask=am)
        o = wm.d_next(zt)
        q = torch.sigmoid(o["q_valid"][0]).cpu().numpy()
        n = n_sub_of(task)
        return {
            "success": float(torch.sigmoid(
                o["success_logit"]).reshape(-1)[0]),
            "milestone": float(torch.sigmoid(
                o["flips_01"][0, :n]).max()),
            "q_mean": float(q.mean()),
            "p_valid": float(torch.sigmoid(
                o["p_valid"].reshape(-1)[0])),
        }

    # ---- Phase A ----------------------------------------------------
    seal_path = OUT / "ledger_seal.json"
    if not seal_path.exists():
        # (1) frozen dev error margins per component
        margins_path = OUT / "dev_margins.json"
        anchors, tq = S.load_universe()
        hidx = json.loads((LCWM / "hcache_index.json").read_text())

        def hget(key):
            dd = torch.load(hidx[key], weights_only=False)
            return (dd["h"][None].float().to(device),
                    dd["mask"][None].to(device))

        errs = defaultdict(list)
        for ak in sorted(anchors):
            a = anchors[ak]
            if a["split"] != "dev" or a["universe"] != "v073":
                continue
            sid, d = a["source_id"], a["decision"]
            sh = torch.load(V73 / "shards" / f"{sid}_d{d}.pt",
                            weights_only=False)
            canon = sh["goal_ids"][0]
            tvv = canon + "_p0"
            z = None
            src_rows = torch.load(V73 / "sources" / f"{sid}.pt",
                                  weights_only=False)["rows"]
            for dd in range(d + 1):
                h, m = hget(f"{tvv}::src::{sid}::d{dd}")
                if z is None:
                    z = wm.initial_state(h, m)
                else:
                    prev = src_rows[dd - 1]
                    aa = prev["chunk_norm"][None, :10].float() \
                        .to(device)
                    am = (torch.arange(10, device=device)[None]
                          < prev["executed_len"])
                    z = wm.step(z, aa, h, m, action_mask=am)
            for tr in sh["transitions"]:
                if tr["branch_key"] == "u0_repeat" \
                        or not tr["continuations"]:
                    continue
                cn, am = norm_chunk(
                    tr["chunk_norm"] if tr["chunk_norm"] is not None
                    else tr["actions_env"],
                    tr["chunk_norm"] is not None)
                pred = decode(z, cn, am, a["task"])
                gs = []
                for ys in tr["continuations"]:
                    c = ys[canon]
                    qv = {int(k): v for k, v in
                          c["q_at_horizons"].items()}
                    gs.append([c["p_valid_100"],
                               np.mean([qv[h_] for h_ in HORIZONS])])
                gt_p = float(np.mean([g[0] for g in gs]))
                gt_q = float(np.mean([g[1] for g in gs]))
                vs = tr["valid_seq"][canon]
                gt_m = float((np.asarray(vs[-1], dtype=bool)
                              & ~np.asarray(vs[0],
                                            dtype=bool)).any())
                errs["p_valid"].append(abs(pred["p_valid"] - gt_p))
                errs["q_mean"].append(abs(pred["q_mean"] - gt_q))
                errs["milestone"].append(
                    abs(pred["milestone"] - gt_m))
                errs["success"].append(pred["success"])  # gt 0 dev
        margins = {k: float(np.median(v)) for k, v in errs.items()}
        margins_path.write_text(json.dumps(
            {"margins": margins,
             "n_dev_rows": {k: len(v) for k, v in errs.items()},
             "note": "median |error| per component on v073 dev "
                     "anchors under the frozen final.pt; frozen "
                     "BEFORE any teacher decoding"}, indent=2))
        print(f"[margins] {margins}", flush=True)

        # (2) model + matched-random ledgers on the 9 teacher banks
        led_rows, rnd_rows = [], []
        chunk_store = {}
        for a in teach_anchors:
            sid, d = a["source_id"], a["decision"]
            akey = f"{sid}_d{d}"
            tb = torch.load(V73 / "teacher_banks" / f"{akey}.pt",
                            weights_only=False)
            src = torch.load(V73 / "sources" / f"{sid}.pt",
                             weights_only=False)
            instr = src["language_canonical"]
            z = unroll_live(src, d, instr)
            preds = {}
            fams = {}
            for cid, (fam, _lang, ch) in tb["bank"].items():
                cn, am = norm_chunk(ch, True)
                preds[cid] = decode(z, cn, am, a["task"])
                fams[cid] = fam
                chunk_store[f"{akey}_{cid}"] = ch
            m = margins
            u0p = preds["u0"]

            def dominates(ci):
                """Registered chain with MASKED/UNSUPPORTED rungs
                skipped: damage + tau masked (no labels), and
                terminal SUCCESS is dropped from teacher eligibility
                because its dev margin is one-sided against a
                constant-0 ground truth (median raw prediction 0.58
                on never-successful states — review critical: an
                untrained head must not grant or veto eligibility).
                Effective chain: milestone -> q_mean. Milestone's
                margin is likewise one-sided (dev milestone GT all
                zero); it is a FALSE-POSITIVE scale, recorded as
                such."""
                p = preds[ci]
                pess = {k: p[k] - m.get(k, 0.0) for k in p}
                opti = {k: u0p[k] + m.get(k, 0.0) for k in u0p}
                if pess["milestone"] > opti["milestone"]:
                    return ("milestone", 0,
                            pess["milestone"] - opti["milestone"])
                if pess["milestone"] < opti["milestone"] \
                        - 2 * m.get("milestone", 0.0):
                    return None
                if pess["q_mean"] > opti["q_mean"]:
                    return ("q_mean", 1,
                            pess["q_mean"] - opti["q_mean"])
                return None

            elig = {}
            for cid in preds:
                if cid == "u0":
                    continue
                r_ = dominates(cid)
                if r_ is not None:
                    elig[cid] = r_
            if elig:
                # lexicographic winner: earliest chain component
                # first, then max margin WITHIN that component
                # (review: cross-component margin magnitudes are
                # not comparable)
                best_rank = min(v[1] for v in elig.values())
                pool_l = {c: v for c, v in elig.items()
                          if v[1] == best_rank}
                best_margin = max(v[2] for v in pool_l.values())
                winners = sorted(
                    c for c, v in pool_l.items()
                    if abs(v[2] - best_margin) < 1e-9)
                w = {c: (1.0 / len(winners) if c in winners
                         else 0.0) for c in preds}
                mode = "model_teacher"
            else:
                w = {c: 0.0 for c in preds}
                mode = "abstain"
            led_rows.append({
                "anchor": akey, "task": a["task"],
                "source_id": sid, "decision": d, "mode": mode,
                "predictions": preds, "eligible": {
                    c: {"component": v[0], "margin": v[2]}
                    for c, v in elig.items()},
                "weights": w, "families": fams})
            # matched-random: permute identity within eligible-mass
            # support, preserving family composition of the winners
            unpermutable = False
            if elig:
                winners = [c for c, ww in w.items() if ww > 0]
                win_fams = {fams[c] for c in winners}
                pool = [c for c in preds if c != "u0"
                        and fams[c] in win_fams]
                alt_pool = [c for c in pool if c not in winners]
                if not alt_pool:
                    # no alternative identity exists in the winner
                    # families: the anchor is UNPERMUTABLE — keep the
                    # weights, mark the row, and exclude its mass
                    # from the identity-control claim (registered
                    # V7.2C precedent; review: never silently copy
                    # the model's winner as its own control)
                    unpermutable = True
                    rw = dict(w)
                else:
                    rot = sha_seed(f"{RID}|rnd|{akey}") \
                        % len(alt_pool)
                    rnd_winners = [
                        alt_pool[(rot + k) % len(alt_pool)]
                        for k in range(len(winners))]
                    rw = {c: (sum(1 for x in rnd_winners
                                  if x == c) / len(rnd_winners))
                          for c in preds}
            else:
                rw = {c: 0.0 for c in preds}
            rnd_rows.append({"anchor": akey, "weights": rw,
                             "mode": mode,
                             "unpermutable": unpermutable})
            print(f"[teach] {akey}: {mode} "
                  f"{[c for c, ww in w.items() if ww > 0]}",
                  flush=True)

        # (3) grounded ledger
        grounded = []
        # V7.3B milestone-positive replay-clean branches
        fid = json.loads((V73 / "replay_fidelity.json").read_text())
        for p_ in sorted((V73 / "shards").glob("*.pt")):
            sh = torch.load(p_, weights_only=False)
            if sh["role"] != "train":
                continue
            canon = sh["goal_ids"][0]
            for tr in sh["transitions"]:
                if tr["branch_key"] == "u0_repeat":
                    continue
                vs = tr["valid_seq"][canon]
                pos = (np.asarray(vs[-1], dtype=bool)
                       & ~np.asarray(vs[0], dtype=bool)).any()
                if pos and sh["anchor"] in fid:
                    grounded.append({
                        "origin": "v073B",
                        "anchor": sh["anchor"], "task": sh["task"],
                        "branch_key": tr["branch_key"],
                        "provenance": tr["provenance"],
                        "chunk_norm_present":
                            tr["chunk_norm"] is not None})
        # released V7.2 teacher candidates: full-tuple both-repeat
        # beat over u0
        for p_ in sorted((V72T / "shards").glob("*.pt")):
            sh = torch.load(p_, weights_only=False)
            trs = {t_["candidate_id"]: t_ for t_ in
                   sh["transitions"] if t_["kind"] != "audit"}
            if "u0" not in trs:
                continue

            def tolerances():
                return json.loads((RESULTS
                                   / "v067_support_report.json")
                                  .read_text()
                                  )["frozen_outcome_tolerances"]

            _tol = tolerances()
            from lcwm.task_automaton import paired_preference
            for cid, t_ in trs.items():
                if cid == "u0" or len(t_["continuations"]) < 2:
                    continue
                # the REGISTERED lexicographic comparator (success ->
                # neg_damage -> p_valid -> q_valid_mean -> neg_tau)
                # with frozen tolerances, both-repeats rule — the
                # hand-rolled chain dropped neg_damage/neg_tau and
                # lost real corrections (review critical)
                verdict = paired_preference(
                    t_["continuations"],
                    trs["u0"]["continuations"], _tol)
                if verdict == 1:
                    grounded.append({
                        "origin": "v072T_released",
                        "anchor": sh["anchor"], "task": sh["task"],
                        "branch_key": cid,
                        "provenance": "released_assessment",
                        "chunk_norm_present": True})
        with (OUT / "grounded_ledger.jsonl").open("w") as f:
            for g_ in grounded:
                f.write(json.dumps(g_) + "\n")
        with (OUT / "model_ledger_pre_outcome.jsonl").open("w") as f:
            for r_ in led_rows:
                f.write(json.dumps(r_) + "\n")
        with (OUT / "matched_random_ledger.jsonl").open("w") as f:
            for r_ in rnd_rows:
                f.write(json.dumps(r_) + "\n")
        torch.save(chunk_store, OUT / "candidate_chunks.pt")
        git_sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
            cwd=REPO_ROOT).stdout.strip()
        seal_path.write_text(json.dumps({
            "git_sha": git_sha,
            "lcwm_final_sha": sha256_file(
                LCWM / "checkpoints" / "final.pt"),
            "dev_margins_sha": sha256_file(margins_path),
            "model_ledger_sha": sha256_file(
                OUT / "model_ledger_pre_outcome.jsonl"),
            "matched_random_sha": sha256_file(
                OUT / "matched_random_ledger.jsonl"),
            "grounded_sha": sha256_file(
                OUT / "grounded_ledger.jsonl"),
            "chunks_sha": sha256_file(OUT / "candidate_chunks.pt"),
            "sealed_before_any_execution": True}, indent=2))
        n_model = sum(1 for r_ in led_rows
                      if r_["mode"] == "model_teacher")
        print(f"[seal] model-teacher anchors {n_model}/9, "
              f"grounded rows {len(grounded)}", flush=True)
    if args.phase == "A":
        return

    # ---- Phase B: execute teacher banks (assessment only) ----------
    seal = json.loads(seal_path.read_text())
    from lcwm.v067_lineage import sha256_file as shaf
    assert seal["model_ledger_sha"] == shaf(
        OUT / "model_ledger_pre_outcome.jsonl"), "ledger modified"
    chunk_store = torch.load(OUT / "candidate_chunks.pt",
                             weights_only=False)
    done = {p.stem for p in (OUT / "shards").glob("*.pt")}
    vindex = OUT / "video_index.jsonl"
    for a in teach_anchors:
        sid, d, task = a["source_id"], a["decision"], a["task"]
        akey = f"{sid}_d{d}"
        if akey in done:
            continue
        src = torch.load(V73 / "sources" / f"{sid}.pt",
                         weights_only=False)
        entry = goal_manifest["tasks"][task]
        canon_id = entry["canonical_goal_spec_id"]
        gspecs = entry["goal_specs"]
        instr = src["language_canonical"]
        env = make_public_env(task, EPISODE_LENGTH[task] + 400)
        try:
            runner.reset()
            env.reset(seed=src["seed"])
            env._env.env.horizon = EPISODE_LENGTH[task] + 500
            alt_id = next(g for g in gspecs if g != canon_id)
            auto0 = GoalAutomaton(
                gspecs[canon_id]["ordered_subgoals"])
            auto0.start(env)
            auto0.evaluate(env, 0)
            alt0 = GoalAutomaton(gspecs[alt_id]["ordered_subgoals"])
            alt0.start(env)          # start_pos from EPISODE RESET
            alt0.evaluate(env, 0)
            bodies = auto0.bodies
            t = 0
            for i in range(d):
                for a_env in src["rows"][i]["actions_env"]:
                    env.step(a_env)
                    t += 1
                    auto0.evaluate(env, t)
                    alt0.evaluate(env, t)
            row = src["rows"][d]
            err = float(np.abs(body_positions(
                env, list(bodies.values()))
                - row["obj_before"]).max())
            assert err < REREACH_ATOL
            snap_a = snap(env, t=t, suite_name="loho_public",
                          task_id=0)
            a_state = fork_env_state(auto0)
            alt_state = fork_env_state(alt0)
            cids = sorted(c.rsplit("_", 1)[-1] if False else
                          c[len(akey) + 1:]
                          for c in chunk_store
                          if c.startswith(f"{akey}_"))
            order = sorted(cids, key=lambda c: sha_seed(
                f"{RID}|order|{sid}|{d}|{c}"))
            transitions = []
            for cid in order:
                restore(env, snap_a)
                env._env.env.done = False
                ba = GoalAutomaton(
                    gspecs[canon_id]["ordered_subgoals"])
                ba.bodies, ba.start_pos = auto0.bodies, \
                    auto0.start_pos
                restore_env_state(ba, a_state)
                b_alt = GoalAutomaton(
                    gspecs[alt_id]["ordered_subgoals"])
                b_alt.bodies, b_alt.start_pos = alt0.bodies, \
                    alt0.start_pos
                restore_env_state(b_alt, alt_state)
                ch = chunk_store[f"{akey}_{cid}"]
                if cid == "u0":
                    acts = list(row["actions_env"])[:10]
                else:
                    acts = list(runner.chunk_to_env(
                        torch.as_tensor(np.asarray(ch))[None, :10]
                        .to(device)))[:10]
                vr = VideoRecorder(OUT / "videos"
                                   / f"{akey}_{cid}.mp4")
                frames = [obs_frame(env)]
                vr.add(frames[0])
                eef_seq = [obs_q(frames[0])]
                valid_seq = [list(ba.prev_valid)]
                valid_seq_alt = [list(b_alt.prev_valid)]
                a_env_l, steps = [], 0
                for a_env in acts:
                    _o, _r, tb, tr2, _i = env.step(a_env)
                    steps += 1
                    a_env_l.append(np.asarray(a_env))
                    ba.evaluate(env, steps)
                    b_alt.evaluate(env, steps)
                    valid_seq.append(list(ba.prev_valid))
                    valid_seq_alt.append(list(b_alt.prev_valid))
                    f = obs_frame(env)
                    frames.append(f)
                    vr.add(f)
                    eef_seq.append(obs_q(f))
                    if tb:
                        env._env.env.done = False
                    if tr2:
                        break
                conts = []
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
                    ca.flips = [(-1, i_, d_) for (_s, i_, d_)
                                in ca.flips]
                    obs_c = obs_frame(env)
                    q_at, sc, cd2, stop = {}, 0, 0, False
                    while sc < CONT_MAX and not stop:
                        nz = flow_noise(sha_seed(
                            f"{RID}|cont|{sid}|{d}|{canon_id}"
                            f"|{rep}|{cd2}"), cfg.chunk_size,
                            cfg.max_action_dim)
                        bcc = runner._obs_to_policy_batch(obs_c,
                                                          instr)
                        pcc = prefix_forward(runner.policy, bcc)
                        ch2 = sample_chunks(runner.policy, bcc,
                                            n=1,
                                            noise=nz.to(device),
                                            prefix=pcc)
                        for a_env in runner.chunk_to_env(
                                ch2[:, :10]):
                            _o2, _r, tm, tr3, _i = env.step(a_env)
                            sc += 1
                            ca.evaluate(env, steps + sc)
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
                    checkpoint_tag="teacher_assessment",
                    checkpoint_path=None, checkpoint_sha256=None,
                    manifest_sha256=seal["model_ledger_sha"],
                    task=task, seed=src["seed"], arm=cid,
                    split="teacher", steps=steps,
                    success=bool(any(c["success_by_100"]
                                     for c in conts)),
                    ordered_progress=ba.ordered_prefix(),
                    damage=ba.damage_unrecovered(),
                    termination="branch_end", root=OUT)
                transitions.append({
                    "transition_id": f"{akey}_{cid}",
                    "candidate_id": cid, "anchor": akey,
                    "task": task, "frames": frames,
                    "eef_seq": eef_seq,
                    "valid_seq_canon": valid_seq,
                    "valid_seq_alt": {alt_id: valid_seq_alt},
                    "actions_env": (np.stack(a_env_l) if a_env_l
                                    else np.zeros((0, 7))),
                    "steps": steps,
                    "grasp_after": grasp_state(env, bodies),
                    "continuations": conts})
                print(f"  [{akey}] {cid} steps={steps}",
                      flush=True)
            tmp = OUT / "shards" / f"{akey}.pt.tmp"
            torch.save({"schema": "v073_teacher_shard_v1",
                        "anchor": akey, "task": task,
                        "transitions": transitions}, tmp)
            tmp.replace(OUT / "shards" / f"{akey}.pt")
            done.add(akey)
        finally:
            env.close()
    with (OUT / "outcome_ledger.jsonl").open("w") as f:
        for p_ in sorted((OUT / "shards").glob("*.pt")):
            sh = torch.load(p_, weights_only=False)
            for tr in sh["transitions"]:
                f.write(json.dumps({
                    "anchor": sh["anchor"],
                    "candidate_id": tr["candidate_id"],
                    "steps": tr["steps"],
                    "continuations": [
                        {k: (v if k != "q_at_horizons" else
                             {str(kk): vv for kk, vv in v.items()})
                         for k, v in c.items()}
                        for c in tr["continuations"]]}) + "\n")
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
