#!/usr/bin/env python
"""V7.2D — fresh public-LoHo recurrent N=1 behavior matrix.

7 arms x 5 public tasks x seeds {1750,1760,1770,1780,1790}. Exact
public five-task specifications, full composite prompt, recurrent
carry from episode reset, deployment N=1 — no external scorer,
best-of-N, GoalSpec, task ID, atomic prompt, or planner. Same env
seeds, pre-generated shared per-decision action noise (CRN namespace
structurally arm-free), same horizon/evaluator, hash-sorted arm
execution order. A complete or partial/failure video for EVERY
(arm, task, seed) rollout.

Primary metric: task-balanced success (average the five seeds within
each task, then equal task mass). Phase-1 `A > B` requires strictly
higher task-balanced success, positive success direction on >= 2
tasks, no task-level success regression, and no damage increase.
Exact SR ties remain ties (paired Q-AUC is secondary evidence only).
Only frozen final-step checkpoints; nothing selects after the fact.
Output: results/libero_loho_public_v1/2026-08-03_v072_loho_dev_r1/
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
LOSS_R1 = RESULTS / "2026-08-03_v072_lcwm_loss_r1"
POLICY = RESULTS / "2026-08-03_v072_policy_r1"
ROOT = RESULTS / "2026-08-03_v072_loho_dev_r1"
RID = "v072_loho_dev_r1"
TASK_ORDER = ["loho_t1_drawer", "loho_t2_basket3", "loho_t3_tray",
              "loho_t4_tray", "loho_t5_drawer_cabinet"]
EPISODE_LENGTH = {"loho_t1_drawer": 700, "loho_t2_basket3": 900,
                  "loho_t3_tray": 900, "loho_t4_tray": 900,
                  "loho_t5_drawer_cabinet": 990}
SEEDS = [1750, 1760, 1770, 1780, 1790]
ARM_LIST = ["stock", "bc_u0", "p0_no_wm_soft", "w0_soft", "w1_soft",
            "w2_soft", "w2_permutation"]


def sha_seed(p: str) -> int:
    return int.from_bytes(hashlib.sha256(
        p.encode()).digest()[:8], "big") & ((1 << 63) - 1)


def obs_frame(env):
    return copy.deepcopy(env._format_raw_obs(
        env._env.env._get_observations()))


@torch.no_grad()
def main() -> None:
    from lcwm.chassis import Pi05Runner
    from lcwm.lc_flow import sample_chunks_lc
    from lcwm.loho_public import make_public_env
    from lcwm.sampler import prefix_forward, sample_chunks
    from lcwm.task_automaton import GoalAutomaton, terminal_success
    from lcwm.v067_lineage import flow_noise, sha256_file
    from lcwm.v06_model import V06State
    from lcwm.video_recorder import (VideoRecorder, video_name,
                                     write_index_row)

    device = torch.device("cuda")
    goal_manifest = json.loads(
        (RESULTS / "goal_spec_manifest_v067.json").read_text())
    ROOT.mkdir(exist_ok=True)
    (ROOT / "videos").mkdir(exist_ok=True)
    rec_path = ROOT / "records.jsonl"
    vindex = ROOT / "video_index.jsonl"
    git_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True,
        text=True, cwd=REPO_ROOT).stdout.strip()
    ckpt_shas = {}
    for arm in ARM_LIST:
        if arm != "stock":
            ckpt_shas[arm] = sha256_file(
                POLICY / "checkpoints" / f"{arm}_final.pt")
    mp = ROOT / "run_manifest.json"
    if not mp.exists():
        mp.write_text(json.dumps({
            "schema": "v072_loho_dev_manifest_v1",
            "run_schema": "v072", "run_id": RID, "git_sha": git_sha,
            "arms": ARM_LIST, "seeds": SEEDS,
            "deployment": "recurrent full-prompt N=1; no scorer/"
                          "best-of-N/GoalSpec/taskID/atomic/planner",
            "noise_crn": f"SHA256('{RID}|act|task|seed|decision')"
                         " — arm identity structurally absent",
            "arm_order": "hash-sorted per (task, seed)",
            "policy_checkpoints": ckpt_shas,
            "wm_checkpoints": {k: sha256_file(
                LOSS_R1 / "checkpoints" / f"{k}_final.pt")
                for k in ("w0_base", "w1_outcome_state",
                          "w2_rank_state")},
        }, indent=2))

    runner = Pi05Runner(suite_name="libero_10")
    cfg = runner.policy.config

    def load_arm(arm):
        """Returns (wm_with_trained_wz or None)."""
        if arm == "stock":
            return None
        ck = torch.load(POLICY / "checkpoints" / f"{arm}_final.pt",
                        weights_only=False)
        state_name = ck["wm_state"]
        wm = V06State().to(device)
        if state_name == "shared_init":
            st = torch.load(LOSS_R1 / "shared_initialization.pt",
                            weights_only=False)["model"]
        else:
            st = torch.load(LOSS_R1 / "checkpoints"
                            / f"{state_name}_final.pt",
                            weights_only=False)["model"]
        wm.load_state_dict(st)
        wm.w_z.weight.copy_(ck["w_z_weight"].to(device))
        torch.nn.init.zeros_(wm.w_z.bias)
        wm.eval()
        return wm

    done = set()
    if rec_path.exists():
        for line in rec_path.open():
            r = json.loads(line)
            done.add((r["arm"], r["task"], r["seed"]))

    arms_loaded = {}
    for task in TASK_ORDER:
        entry = goal_manifest["tasks"][task]
        canon_id = entry["canonical_goal_spec_id"]
        subgoals = entry["goal_specs"][canon_id]["ordered_subgoals"]
        for seed in SEEDS:
            arm_order = sorted(ARM_LIST, key=lambda a_: sha_seed(
                f"{RID}|order|{task}|{seed}|{a_}"))
            for arm in arm_order:
                if (arm, task, seed) in done:
                    continue
                if arm not in arms_loaded:
                    arms_loaded[arm] = load_arm(arm)
                wm = arms_loaded[arm]
                env = make_public_env(task, EPISODE_LENGTH[task])
                try:
                    runner.reset()
                    env.reset(seed=seed)
                    env._env.env.horizon = EPISODE_LENGTH[task] + 50
                    instr = env.task_description
                    auto = GoalAutomaton(list(subgoals))
                    auto.start(env)
                    auto.evaluate(env, 0)
                    vr = VideoRecorder(ROOT / "videos" / video_name(
                        RID, "final", task, seed, arm))
                    obs = obs_frame(env)
                    vr.add(obs)
                    z = None
                    prev_chunk = None
                    t, dec = 0, 0
                    success = False
                    q_traj = []
                    term = trunc = False
                    while t < EPISODE_LENGTH[task]:
                        batch = runner._obs_to_policy_batch(obs,
                                                            instr)
                        pfx = prefix_forward(runner.policy, batch)
                        if wm is not None:
                            h = pfx.hidden.float()
                            m = pfx.pad_masks.bool()
                            if z is None:
                                z = wm.initial_state(h, m)
                            else:
                                am = torch.ones(
                                    1, 10, dtype=torch.bool,
                                    device=device)
                                z = wm.step(z, prev_chunk, h, m,
                                            action_mask=am)
                        nz = flow_noise(sha_seed(
                            f"{RID}|act|{task}|{seed}|{dec}"),
                            cfg.chunk_size, cfg.max_action_dim)
                        if wm is not None:
                            bias = wm.policy_bias(z)
                            ch = sample_chunks_lc(
                                runner.policy, batch, bias, n=1,
                                noise=nz.to(device), prefix=pfx)
                        else:
                            ch = sample_chunks(
                                runner.policy, batch, n=1,
                                noise=nz.to(device), prefix=pfx)
                        prev_chunk = ch[:, :10].float()
                        for a_env in runner.chunk_to_env(
                                ch[:, :10]):
                            _o, _r, term, trunc, _i = env.step(
                                a_env)
                            t += 1
                            auto.evaluate(env, t)
                            q_traj.append(auto.q_valid())
                            obs = obs_frame(env)
                            vr.add(obs)
                            if terminal_success(env):
                                success = True
                            if term or trunc \
                                    or t >= EPISODE_LENGTH[task]:
                                break
                        dec += 1
                        if success or term or trunc:
                            break
                    vm = vr.close(completed=True)
                    first_unres = next(
                        (subgoals[i] for i, v in
                         enumerate(auto.prev_valid) if not v), None)
                    rec = {
                        "arm": arm, "task": task, "seed": seed,
                        "success": bool(success),
                        "steps": t, "decisions": dec,
                        "ordered_prefix": auto.ordered_prefix(),
                        "q_final": auto.q_valid(),
                        "q_auc": float(np.mean(q_traj))
                        if q_traj else 0.0,
                        "damage": auto.damage_unrecovered(),
                        "first_unresolved": first_unres,
                        "video": vm["video_path"],
                    }
                    with rec_path.open("a") as f:
                        f.write(json.dumps(rec) + "\n")
                    write_index_row(
                        vindex, vm, run_id=RID,
                        checkpoint_tag="final",
                        checkpoint_path=(str(
                            POLICY / "checkpoints"
                            / f"{arm}_final.pt")
                            if arm != "stock" else None),
                        checkpoint_sha256=ckpt_shas.get(arm),
                        manifest_sha256=json.loads(
                            mp.read_text())["git_sha"],
                        task=task, seed=seed, arm=arm,
                        split="dev", steps=t,
                        success=bool(success),
                        ordered_progress=auto.ordered_prefix(),
                        damage=auto.damage_unrecovered(),
                        termination=("success" if success else
                                     "horizon"),
                        root=ROOT)
                    print(f"[{task} s{seed}] {arm}: "
                          f"success={success} steps={t} "
                          f"q_auc={rec['q_auc']:.3f} "
                          f"damage={rec['damage']}", flush=True)
                finally:
                    env.close()

    # ---------------- paired report ---------------------------------
    recs = [json.loads(x) for x in rec_path.open()]
    table = defaultdict(dict)
    for r in recs:
        table[(r["task"], r["seed"])][r["arm"]] = r

    def task_balanced(arm, key="success"):
        by_task = defaultdict(list)
        for (task, seed), row in table.items():
            if arm in row:
                by_task[task].append(float(row[arm][key]))
        return {t: float(np.mean(v)) for t, v in by_task.items()}

    def phase1_gt(a, b):
        sa, sb = task_balanced(a), task_balanced(b)
        tasks = sorted(set(sa) & set(sb))
        bal_a = float(np.mean([sa[t] for t in tasks]))
        bal_b = float(np.mean([sb[t] for t in tasks]))
        pos = sum(1 for t in tasks if sa[t] > sb[t])
        reg = [t for t in tasks if sa[t] < sb[t]]
        da = float(np.mean([r["damage"] for r in recs
                            if r["arm"] == a]))
        db = float(np.mean([r["damage"] for r in recs
                            if r["arm"] == b]))
        verdict = (bal_a > bal_b and pos >= 2 and not reg
                   and da <= db)
        return {"a": a, "b": b, "bal_a": bal_a, "bal_b": bal_b,
                "tasks_positive": pos, "task_regressions": reg,
                "damage_a": da, "damage_b": db,
                "phase1_greater": bool(verdict),
                "sr_tie": bal_a == bal_b}

    pairs = [("w0_soft", "p0_no_wm_soft"), ("w1_soft", "w0_soft"),
             ("w2_soft", "w1_soft"), ("w2_soft", "w2_permutation"),
             ("w2_soft", "stock"), ("w2_soft", "bc_u0"),
             ("bc_u0", "stock"), ("p0_no_wm_soft", "stock")]
    report = {
        "task_seed_success": {
            f"{t}|{s}": {a: r["success"]
                         for a, r in table[(t, s)].items()}
            for (t, s) in sorted(table)},
        "task_balanced_success": {a: task_balanced(a)
                                  for a in ARM_LIST},
        "q_auc_task_balanced": {a: task_balanced(a, "q_auc")
                                for a in ARM_LIST},
        "paired": [phase1_gt(a, b) for a, b in pairs],
    }
    (ROOT / "paired_report.json").write_text(
        json.dumps(report, indent=2))
    (ROOT / "task_seed_table.json").write_text(json.dumps(
        report["task_seed_success"], indent=2))
    fail = defaultdict(list)
    for r in recs:
        if not r["success"]:
            fail[r["arm"]].append(
                {"task": r["task"], "seed": r["seed"],
                 "first_unresolved": r["first_unresolved"],
                 "ordered_prefix": r["ordered_prefix"]})
    (ROOT / "failure_localization.json").write_text(
        json.dumps(fail, indent=2))
    print(json.dumps(report["task_balanced_success"], indent=1),
          flush=True)
    print(f"-> {ROOT}", flush=True)


if __name__ == "__main__":
    main()
