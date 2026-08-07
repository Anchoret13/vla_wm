#!/usr/bin/env python
"""V7.3E — fresh public-LoHo recurrent N=1 endpoint.

5 arms x 5 public tasks x seeds {1800,1810,1820,1830,1840}
= 125 rollouts, 125 videos. Arms: stock (untouched pi0.5) +
correction_bc / lc_grounded / lc_full / lc_random (each = stock pi0.5
with its trained bias-free LCProj and trained action_out_proj; the
LCWM is frozen and drives the recurrent state, except correction_bc
which uses the frozen global-mean state).

Deployment contract: exact public task specs, full composite prompt,
recurrent carry from episode reset, N=1 generation. No scorer,
best-of-N, atomic prompt, GoalSpec, task ID, milestone, privileged
state, or planner. Shared environment seeds, task order, per-decision
flow noise (CRN namespace structurally arm-free), horizon, evaluator;
hash-sorted arm execution order. Every rollout indexed with a video.

Primary metric: task-balanced terminal success (average the five
seeds within a task, then equal task mass). Secondary: Q-AUC,
ordered prefix, first unresolved milestone, time, damage. Only the
frozen final-step checkpoints are used.
Output: results/libero_loho_public_v1/<DATE>_v073_loho_dev_r1/
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

from lcwm.v06_model import V06State  # noqa: E402

RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
LCWM = RESULTS / "2026-08-06_v073_lcwm_r1"
POLICY = RESULTS / "2026-08-07_v073_policy_r1"
RID = "v073_loho_dev_r1"
TASK_ORDER = ["loho_t1_drawer", "loho_t2_basket3", "loho_t3_tray",
              "loho_t4_tray", "loho_t5_drawer_cabinet"]
EPISODE_LENGTH = {"loho_t1_drawer": 700, "loho_t2_basket3": 900,
                  "loho_t3_tray": 900, "loho_t4_tray": 900,
                  "loho_t5_drawer_cabinet": 990}
SEEDS = [1800, 1810, 1820, 1830, 1840]
ARMS = ["stock", "correction_bc", "lc_grounded", "lc_full",
        "lc_random"]
GLOBAL_MEAN_ARMS = {"correction_bc"}


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
    from lcwm.lc_flow import freeze_pi05_base, sample_chunks_lc
    from lcwm.loho_public import make_public_env
    from lcwm.sampler import prefix_forward, sample_chunks
    from lcwm.task_automaton import GoalAutomaton, terminal_success
    from lcwm.v067_lineage import flow_noise, sha256_file
    from lcwm.video_recorder import (VideoRecorder, video_name,
                                     write_index_row)

    device = torch.device("cuda")
    root = None
    for p in sorted(RESULTS.glob(f"*_{RID}")):
        root = p
    root = root or RESULTS / f"{run_date()}_{RID}"
    (root / "videos").mkdir(parents=True, exist_ok=True)
    goal_manifest = json.loads(
        (RESULTS / "goal_spec_manifest_v067.json").read_text())

    runner = Pi05Runner(suite_name="libero_10")
    cfg = runner.policy.config
    freeze_pi05_base(runner.policy)
    for p in runner.policy.parameters():
        p.requires_grad_(False)
    aop = runner.policy.model.action_out_proj
    stock_aop = {k: v.detach().clone()
                 for k, v in aop.state_dict().items()}
    wm = V06State().to(device)
    wm.load_state_dict(torch.load(LCWM / "checkpoints" / "final.pt",
                                  weights_only=False)["model"])
    wm.eval()
    states = torch.load(POLICY / "anchor_states.pt",
                        weights_only=False)
    gmean = states["global_mean"].to(device)

    ck_shas = {}
    arm_blobs = {}
    for arm in ARMS:
        if arm == "stock":
            continue
        pth = POLICY / "checkpoints" / f"{arm}_final.pt"
        ck_shas[arm] = sha256_file(pth)
        arm_blobs[arm] = torch.load(pth, weights_only=False)

    mp = root / "run_manifest.json"
    if not mp.exists():
        mp.write_text(json.dumps({
            "schema": "v073_loho_dev_manifest_v1",
            "run_schema": "v073", "run_id": RID,
            "git_sha": subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True,
                cwd=REPO_ROOT).stdout.strip(),
            "arms": ARMS, "seeds": SEEDS,
            "deployment": "recurrent full-prompt N=1; no scorer/"
                          "best-of-N/GoalSpec/taskID/atomic/planner",
            "noise_crn": f"SHA256('{RID}|act|task|seed|decision')"
                         " — arm identity structurally absent",
            "arm_order": "hash-sorted per (task, seed)",
            "policy_checkpoints": ck_shas,
            "lcwm_final_sha": sha256_file(
                LCWM / "checkpoints" / "final.pt"),
            "global_mean_arms": sorted(GLOBAL_MEAN_ARMS),
        }, indent=2))

    rec_path = root / "records.jsonl"
    vindex = root / "video_index.jsonl"
    done = set()
    if rec_path.exists():
        for line in rec_path.open():
            r = json.loads(line)
            done.add((r["arm"], r["task"], r["seed"]))

    lc_cache = {}

    def lc_of(arm):
        if arm not in lc_cache:
            lin = nn.Linear(384, 1024, bias=False).to(device)
            lin.load_state_dict(
                {"weight": arm_blobs[arm]["lc_proj"]["lin.weight"]
                 .to(device)})
            lin.eval()
            lc_cache[arm] = lin
        return lc_cache[arm]

    for task in TASK_ORDER:
        entry = goal_manifest["tasks"][task]
        canon = entry["canonical_goal_spec_id"]
        subgoals = entry["goal_specs"][canon]["ordered_subgoals"]
        for seed in SEEDS:
            order = sorted(ARMS, key=lambda a: sha_seed(
                f"{RID}|order|{task}|{seed}|{a}"))
            for arm in order:
                if (arm, task, seed) in done:
                    continue
                if arm == "stock":
                    aop.load_state_dict(stock_aop)
                else:
                    aop.load_state_dict(
                        arm_blobs[arm]["action_out_proj"])
                env = make_public_env(task, EPISODE_LENGTH[task])
                try:
                    runner.reset()
                    env.reset(seed=seed)
                    env._env.env.horizon = EPISODE_LENGTH[task] + 50
                    instr = env.task_description
                    auto = GoalAutomaton(list(subgoals))
                    auto.start(env)
                    auto.evaluate(env, 0)
                    vr = VideoRecorder(root / "videos" / video_name(
                        RID, "final", task, seed, arm))
                    obs = obs_frame(env)
                    vr.add(obs)
                    z, prev_chunk = None, None
                    t, dec, success = 0, 0, False
                    q_traj = []
                    term = trunc = False
                    while t < EPISODE_LENGTH[task]:
                        batch = runner._obs_to_policy_batch(obs,
                                                            instr)
                        pfx = prefix_forward(runner.policy, batch)
                        nz = flow_noise(sha_seed(
                            f"{RID}|act|{task}|{seed}|{dec}"),
                            cfg.chunk_size, cfg.max_action_dim)
                        if arm == "stock":
                            ch = sample_chunks(
                                runner.policy, batch, n=1,
                                noise=nz.to(device), prefix=pfx)
                        else:
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
                            pool = (gmean
                                    if arm in GLOBAL_MEAN_ARMS
                                    else z.mean(dim=1))
                            bias = lc_of(arm)(pool)
                            ch = sample_chunks_lc(
                                runner.policy, batch, bias, n=1,
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
                            if success or term or trunc \
                                    or t >= EPISODE_LENGTH[task]:
                                break
                        dec += 1
                        if success or term or trunc:
                            break
                    vm = vr.close(completed=True)
                    first_un = next(
                        (subgoals[i] for i, v in
                         enumerate(auto.prev_valid) if not v), None)
                    rec = {"arm": arm, "task": task, "seed": seed,
                           "success": bool(success), "steps": t,
                           "decisions": dec,
                           "ordered_prefix": auto.ordered_prefix(),
                           "q_final": auto.q_valid(),
                           "q_auc": float(np.mean(q_traj))
                           if q_traj else 0.0,
                           "damage": auto.damage_unrecovered(),
                           "first_unresolved": first_un,
                           "video": vm["video_path"]}
                    with rec_path.open("a") as f:
                        f.write(json.dumps(rec) + "\n")
                    write_index_row(
                        vindex, vm, run_id=RID,
                        checkpoint_tag="final",
                        checkpoint_path=(
                            str(POLICY / "checkpoints"
                                / f"{arm}_final.pt")
                            if arm != "stock" else None),
                        checkpoint_sha256=ck_shas.get(arm),
                        manifest_sha256=sha256_file(mp),
                        task=task, seed=seed, arm=arm, split="dev",
                        steps=t, success=bool(success),
                        ordered_progress=auto.ordered_prefix(),
                        damage=auto.damage_unrecovered(),
                        termination=("success" if success
                                     else "horizon"), root=root)
                    print(f"[{task} s{seed}] {arm}: "
                          f"success={success} steps={t} "
                          f"q_auc={rec['q_auc']:.3f} "
                          f"dmg={rec['damage']}", flush=True)
                finally:
                    env.close()
    aop.load_state_dict(stock_aop)

    # ---------------- paired report ---------------------------------
    recs = [json.loads(x) for x in rec_path.open()]
    table = defaultdict(dict)
    for r in recs:
        table[(r["task"], r["seed"])][r["arm"]] = r

    def per_task_counts(arm, key="success"):
        """EXACT per-task tallies (numerator, denominator) — the
        registered comparison is on task-balanced rates and must not
        be decided by float summation order (review: np.mean over 5
        per-task rates gives different floats for the SAME total, so
        genuine ties were reported as wins)."""
        by = defaultdict(list)
        for (task, _s), row in table.items():
            if arm in row:
                by[task].append(row[arm][key])
        return {t_: (sum(Fraction(int(x) if isinstance(x, bool)
                                  else x).limit_denominator(10**6)
                         for x in v), len(v))
                for t_, v in by.items()}

    def balanced(arm, key="success"):
        c = per_task_counts(arm, key)
        if not c:
            return Fraction(0)
        return sum(Fraction(n, d) for n, d in c.values()) \
            / len(c)

    def per_task(arm, key="success"):
        return {t_: float(Fraction(n, d))
                for t_, (n, d) in per_task_counts(arm, key).items()}

    def gt(a, b):
        ca, cb = per_task_counts(a), per_task_counts(b)
        tasks = sorted(set(ca) & set(cb))
        bal_a, bal_b = balanced(a), balanced(b)
        pos = [t_ for t_ in tasks
               if Fraction(*ca[t_]) > Fraction(*cb[t_])]
        reg = [t_ for t_ in tasks
               if Fraction(*ca[t_]) < Fraction(*cb[t_])]
        da, db = per_task_counts(a, "damage"), \
            per_task_counts(b, "damage")
        dmg_up = [t_ for t_ in tasks
                  if Fraction(*da[t_]) > Fraction(*db[t_])]
        out = {"a": a, "b": b, "bal_a": float(bal_a),
               "bal_b": float(bal_b),
               "bal_a_exact": str(bal_a), "bal_b_exact": str(bal_b),
               "tasks_positive": pos, "task_regressions": reg,
               "damage_increase_tasks": dmg_up,
               "strictly_greater": bool(bal_a > bal_b),
               "exact_tie": bool(bal_a == bal_b)}
        if b == "stock":
            # the >=2-positive / no-regression / no-damage-increase
            # safety rule is registered RELATIVE TO STOCK only
            out["phase1_win_vs_stock"] = bool(
                bal_a > bal_b and len(pos) >= 2 and not reg
                and not dmg_up)
        return out

    pairs = [("lc_full", "stock"), ("lc_full", "correction_bc"),
             ("lc_full", "lc_grounded"), ("lc_full", "lc_random"),
             ("lc_grounded", "correction_bc"),
             ("lc_grounded", "stock"), ("correction_bc", "stock"),
             ("lc_random", "stock")]
    verdicts = {f"{a}>{b}": gt(a, b) for a, b in pairs}
    winner = bool(
        verdicts["lc_full>stock"].get("phase1_win_vs_stock")
        and verdicts["lc_full>correction_bc"]["strictly_greater"]
        and verdicts["lc_full>lc_grounded"]["strictly_greater"]
        and verdicts["lc_full>lc_random"]["strictly_greater"])
    report = {
        "v73_development_winner": winner,
        "winner_rule": "lc_full > stock under the stock-only safety "
                       "rule AND strictly greater task-balanced "
                       "success than correction_bc, lc_grounded, "
                       "lc_random (exact-rational comparison)",
        "task_balanced_success": {a: per_task(a) for a in ARMS},
        "task_balanced_overall": {a: str(balanced(a))
                                  for a in ARMS},
        "task_balanced_q_auc": {a: per_task(a, "q_auc")
                                for a in ARMS},
        "task_balanced_damage": {a: per_task(a, "damage")
                                 for a in ARMS},
        "paired": list(verdicts.values()),
        "n_rollouts": len(recs),
    }
    (root / "paired_report.json").write_text(
        json.dumps(report, indent=2))
    (root / "task_seed_table.json").write_text(json.dumps(
        {f"{t}|{s}": {a: r["success"] for a, r in row.items()}
         for (t, s), row in sorted(table.items())}, indent=2))
    fail = defaultdict(list)
    for r in recs:
        if not r["success"]:
            fail[r["arm"]].append({
                "task": r["task"], "seed": r["seed"],
                "first_unresolved": r["first_unresolved"],
                "ordered_prefix": r["ordered_prefix"]})
    (root / "failure_localization.json").write_text(
        json.dumps(fail, indent=2))
    print(json.dumps(report["task_balanced_success"], indent=1),
          flush=True)
    print(f"-> {root}", flush=True)


if __name__ == "__main__":
    main()
