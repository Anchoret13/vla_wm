#!/usr/bin/env python
"""V7.0.4 — paired anchor-replay behavior gate.

run_id=v070_anchor_eval1; R=2; first-ten branch credit; CONT_MAX=100;
horizons {10,30,60,100}. Seed contract (candidate/arm identity
structurally absent; evaluation streams disjoint from construction):

  action noise seed = SHA256("v070_anchor_eval1|action|src|dec")
  cont noise seed   = SHA256("v070_anchor_eval1|cont|src|dec|gid|r|c")
  (int.from_bytes(digest[:8],'big') & (2^63-1))

Per strict-clean anchor (train + source-held-out dev):
 1. reset recorded source seed, replay stored source actions, verify
    proprio/object/semantic checksums (re-reach contract);
 2. one in-memory SimSnapshot at the verified anchor, restored across
    arms (file restore forbidden — none is stored);
 3. fresh u_0 generated ONCE from the action-noise tensor + its exact
    replay executed first; anchor excluded under the repeat-wise strict
    rule if fresh replay is unstable;
 4. recurrent LC state unrolled from the stored source episode reset;
 5. each learned arm's full-prompt N=1 chunk generated under the SAME
    action-noise tensor; stored oracle chunk RE-EXECUTED (positive
    anchors);
 6. first ten actions executed, then stock π0.5 continuations under
    sibling common randomness; automaton evaluated after EVERY action;
 7. every stored seed and noise SHA independently recomputed before a
    record is accepted.

Branches: fresh_u0, u0_replay(audit), stored_oracle(+anchors only),
hard_affine, hard_centered, hard_random_affine, hard_random_centered.
"""

from __future__ import annotations

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
AUDIT = RESULTS / "v070_policy" / "audit"
ORACLE = RESULTS / "v070_policy" / "oracle"
PILOT = RESULTS / "v070_policy" / "hard_pilot"
OUT = RESULTS / "v070_policy" / "anchor_replay"
RUN_ID = "v070_anchor_eval1"
EPISODE_LENGTH = {"loho_t1_drawer": 700, "loho_t2_basket3": 900,
                  "loho_t3_tray": 900, "loho_t4_tray": 900,
                  "loho_t5_drawer_cabinet": 990}
HORIZONS = (10, 30, 60, 100)
R = 2
CONT_MAX = 100
REREACH_ATOL = 2e-3
LEARNED_ARMS = ("hard_affine", "hard_centered", "hard_random_affine",
                "hard_random_centered")


def sha_seed(payload: str) -> int:
    return int.from_bytes(hashlib.sha256(
        payload.encode()).digest()[:8], "big") & ((1 << 63) - 1)


@torch.no_grad()
def main() -> None:
    from lcwm.chassis import Pi05Runner
    from lcwm.goal_semantics import SuccessTracker, env_eval_fn
    from lcwm.lc_flow import sample_chunks_lc
    from lcwm.loho_public import make_public_env
    from lcwm.probe_data import body_positions
    from lcwm.sampler import prefix_forward, sample_chunks
    from lcwm.snapshot import restore, snap
    from lcwm.task_automaton import GoalAutomaton, fork_env_state, \
        restore_env_state
    from lcwm.v067_lineage import flow_noise, noise_sha
    from lcwm.v06_model import V06State
    from lcwm.v070_replay import strict_replay_clean
    from scripts.train_v070_hard_pilot import (AffineCoupling,
                                               CenteredCoupling)

    OUT.mkdir(parents=True, exist_ok=True)
    records_path = OUT / "records.jsonl"
    done = set()
    if records_path.exists():
        for line in records_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add(r["anchor"])

    strict = json.loads(
        (AUDIT / "strict_replay_report.json").read_text())
    clean = {s: sorted(strict["clean_anchor_lists"][s])
             for s in ("train", "dev")}
    teacher = torch.load(ORACLE / "hard_teacher_manifest.pt",
                         weights_only=False)
    trow_by_anchor = {r["anchor"]: r for r in teacher["rows"]}
    goal_manifest = json.loads(
        (RESULTS / "goal_spec_manifest_v067.json").read_text())

    device = torch.device("cuda")
    model = V06State().to(device)
    wm = torch.load(RESULTS / "v069_predictive"
                    / "checkpoint_selected.pt", weights_only=False)
    model.load_state_dict(wm["model"])
    model.eval()
    train_man = torch.load(PILOT / "train_manifest.pt",
                           weights_only=False)
    couplings = {}
    for arm in LEARNED_ARMS:
        ck = torch.load(PILOT / f"{arm}_final.pt", weights_only=False)
        assert ck["run_schema"] == "v070"
        c = (CenteredCoupling(train_man["mu_train"].to(device))
             if ck["coupling_type"] == "centered"
             else AffineCoupling()).to(device)
        c.load_state_dict(ck["state_dict"])
        c.eval()
        couplings[arm] = c
    runner = Pi05Runner(suite_name="libero_10")
    cfg = runner.policy.config

    manifest = {
        "schema": "v070_anchor_eval_v1", "run_schema": "v070",
        "run_id": RUN_ID, "R": R, "cont_max": CONT_MAX,
        "horizons": list(HORIZONS),
        "arm_checkpoints": {a: str(PILOT / f"{a}_final.pt")
                            for a in LEARNED_ARMS},
        "seed_contract": ("SHA256 action/cont signatures; arm "
                          "identity structurally absent"),
    }
    mf = OUT / "run_manifest.json"
    if not mf.exists():
        mf.write_text(json.dumps(manifest, indent=2))

    tolerances = None
    for split in ("train", "dev"):
        for akey in clean[split]:
            if akey in done:
                print(f"[skip] {akey}", flush=True)
                continue
            sid, d = akey.rsplit("_d", 1)
            d = int(d)
            grp = torch.load(DATA / "corrections_v069"
                             / f"{akey}.pt", weights_only=False)
            tolerances = grp["frozen_outcome_tolerances"]
            src = torch.load(DATA / "corrections_sources_v069"
                             / f"{sid}.pt", weights_only=False)
            task = grp["task"]
            entry = goal_manifest["tasks"][task]
            canon_id = entry["canonical_goal_spec_id"]
            subgoals = entry["goal_specs"][canon_id][
                "ordered_subgoals"]
            term_preds = [tuple(p) for p in entry["goal_specs"][
                canon_id]["terminal_predicates"]]
            env = make_public_env(task, EPISODE_LENGTH[task] + 200)
            try:
                runner.reset()
                env.reset(seed=src["seed"])
                env._env.env.horizon = EPISODE_LENGTH[task] + 300
                eval_fn = env_eval_fn(env)
                auto = GoalAutomaton(list(subgoals))
                auto.start(env)
                auto.evaluate(env, 0)
                bodies = auto.bodies
                # 1. re-reach + checksums + 4. recurrent z unroll
                z = None
                t = 0
                for i, row in enumerate(src["rows"]):
                    if i > d:
                        break
                    batch = runner._obs_to_policy_batch(
                        row["obs"], src["language_canonical"])
                    prefix = prefix_forward(runner.policy, batch)
                    h = prefix.hidden.float()
                    m = prefix.pad_masks.bool()
                    if z is None:
                        z = model.initial_state(h, m)
                    else:
                        prev = src["rows"][i - 1]
                        aa = prev["chunk_norm"][None, :10].float().to(
                            device)
                        am = (torch.arange(10, device=device)[None]
                              < prev["executed_len"])
                        z = model.step(z, aa, h, m, action_mask=am)
                    if i == d:
                        anchor_prefix, anchor_batch = prefix, batch
                        break_row = row
                        break          # anchor is PRE-decision-d
                    for a_env in row["actions_env"]:
                        env.step(a_env)
                        t += 1
                        auto.evaluate(env, t)
                obj_err = float(np.abs(body_positions(
                    env, list(bodies.values()))
                    - break_row["obj_before"]).max())
                assert obj_err < REREACH_ATOL, f"{akey}: {obj_err:.2e}"
                obs_now = env._format_raw_obs(
                    env._env.env._get_observations())
                eef_live = np.concatenate([
                    obs_now["robot_state"]["eef"]["pos"],
                    obs_now["robot_state"]["eef"]["quat"],
                    obs_now["robot_state"]["gripper"]["qpos"]])
                assert float(np.abs(
                    eef_live - np.asarray(break_row["q"])).max()) \
                    < 5e-3, f"{akey}: proprio checksum"
                canon_state = grp["auto_states_snapshot"][canon_id]
                assert list(auto.prev_valid) == list(
                    canon_state["prev_valid"]), \
                    f"{akey}: semantic checksum"
                anchor_state = fork_env_state(auto)
                # 2. one in-memory snapshot
                snap_anchor = snap(env, t=t, suite_name="loho_public",
                                   task_id=0)

                # branch chunks
                a_seed = sha_seed(f"{RUN_ID}|action|{sid}|{d}")
                a_noise = flow_noise(a_seed, cfg.chunk_size,
                                     cfg.max_action_dim)
                a_sha = noise_sha(a_noise)
                chunk_u0 = sample_chunks(
                    runner.policy, anchor_batch, n=1,
                    noise=a_noise.to(device), prefix=anchor_prefix)
                branches = [("fresh_u0", chunk_u0),
                            ("u0_replay", chunk_u0)]
                trow = trow_by_anchor.get(akey)
                if trow and trow["positive"] and split == "train":
                    branches.append((
                        "stored_oracle",
                        trow["target"]["chunk_norm"][None].to(
                            device).float()))
                pool = z.mean(dim=1)
                for arm in LEARNED_ARMS:
                    bias = couplings[arm](pool)
                    branches.append((arm, sample_chunks_lc(
                        runner.policy, anchor_batch, bias, n=1,
                        noise=a_noise.to(device),
                        prefix=anchor_prefix)))

                results = {}
                crn_log = {}
                for bname, chunk in branches:
                    reps = []
                    for rep in range(R):
                        restore(env, snap_anchor)
                        env._env.env.done = False
                        ba = GoalAutomaton(list(subgoals))
                        ba.bodies, ba.start_pos = auto.bodies, \
                            auto.start_pos
                        restore_env_state(ba, anchor_state)
                        term_b = trunc_b = False
                        steps_b = 0
                        for a_env in runner.chunk_to_env(
                                chunk[:, :10]):
                            _o, _r, tb, tr, _i = env.step(a_env)
                            steps_b += 1
                            ba.evaluate(env, steps_b)
                            if tb:
                                term_b = True
                                env._env.env.done = False
                            if tr:
                                trunc_b = True
                                break
                        ba.flips = []
                        tracker = SuccessTracker(term_preds)
                        obs_c = env._format_raw_obs(
                            env._env.env._get_observations())
                        ba.evaluate(env, 0)
                        tracker.update(env, 0, eval_fn)
                        q_at, steps_c, cd = {}, 0, 0
                        stop = term_b or trunc_b
                        while steps_c < CONT_MAX and not stop:
                            c_seed = sha_seed(
                                f"{RUN_ID}|cont|{sid}|{d}"
                                f"|{canon_id}|{rep}|{cd}")
                            c_noise = flow_noise(
                                c_seed, cfg.chunk_size,
                                cfg.max_action_dim)
                            key = (rep, cd)
                            entry_ = {"seed": c_seed,
                                      "sha": noise_sha(c_noise)}
                            if key in crn_log:
                                assert crn_log[key] == entry_, \
                                    "sibling CRN violated"
                            else:
                                crn_log[key] = entry_
                            bc = runner._obs_to_policy_batch(
                                obs_c, src["language_canonical"])
                            pc = prefix_forward(runner.policy, bc)
                            ch = sample_chunks(
                                runner.policy, bc, n=1,
                                noise=c_noise.to(device), prefix=pc)
                            for a_env in runner.chunk_to_env(
                                    ch[:, :10]):
                                obs_c, _r, tm, tr2, _i = env.step(
                                    a_env)
                                steps_c += 1
                                ba.evaluate(env, steps_c)
                                gt = tracker.update(env, steps_c,
                                                    eval_fn)
                                assert gt == bool(tm)
                                if steps_c in HORIZONS:
                                    q_at[steps_c] = ba.q_valid()
                                if tm:
                                    stop = True
                                    break
                                if tr2:
                                    stop = True
                                    break
                                if steps_c >= CONT_MAX:
                                    break
                            cd += 1
                        for hh in HORIZONS:
                            q_at.setdefault(hh, ba.q_valid())
                        reps.append({
                            "success_by_100": bool(tracker.achieved),
                            "neg_damage": -ba.damage_unrecovered(),
                            "p_valid_100": ba.p_valid(),
                            "q_valid_mean": sum(
                                q_at[hh] for hh in HORIZONS)
                            / len(HORIZONS),
                            "neg_tau_next": -ba.tau_next(0),
                            "q_at_horizons": dict(q_at)})
                    results[bname] = reps
                    print(f"  [{akey}] {bname}", flush=True)

                # 3. strict fresh-replay rule
                clean_fresh, info = strict_replay_clean(
                    results["u0_replay"], results["fresh_u0"],
                    tolerances,
                    {"eef_pos": 0, "eef_quat": 0, "gripper": 0,
                     "obj_pos": 0})
                record = {
                    "anchor": akey, "task": task, "split": split,
                    "source_id": sid, "decision": d,
                    "positive": bool(trow and trow["positive"]),
                    "fresh_replay_clean": bool(clean_fresh),
                    "fresh_replay_signs": info["repeat_signs"],
                    "action_noise_seed": a_seed,
                    "action_noise_sha256": a_sha,
                    "outcomes": results,
                    "tolerances": tolerances,
                }
                with records_path.open("a") as f:
                    f.write(json.dumps(record) + "\n")
                print(f"[anchor] {akey} clean={clean_fresh} "
                      f"signs={info['repeat_signs']}", flush=True)
            finally:
                env.close()
    print(f"-> {records_path}", flush=True)


if __name__ == "__main__":
    main()
