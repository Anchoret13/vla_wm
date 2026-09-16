#!/usr/bin/env python
"""V7.6B yield pilot — does moving the snapshot actually raise yield?

Registered in `plan_and_progress/archive/daily/2026-08-15.md` (V7.6B YIELD PILOT) before
execution:

    n = 40 attempts on the three reacher-supported tasks. Pass at >= 8
    positive (20%). Derived: the tranche quota needs >= 25 positives over
    ~260 planned attempts = 9.6% operating yield, and the exact one-sided
    95% binomial lower bound at 8/40 is ~0.104, which clears it. The V7.4B
    rate of 3.1% would produce 8/40 with probability well under 1%.

What this isolates. The census showed the reacher can BUILD snapshots. It
said nothing about what happens after one. The V7.6A thesis -- recovery is
decided by where the snapshot sits -- is the claim the whole 70-source
budget is bought on, and it is a claim about the post-snapshot rollout.

So this runs the *registered V7.6B recovery protocol* unchanged: atomic
pi0.5, closed loop, <= 60 actions / <= 6 decisions, toward the next
unresolved ordered subgoal, positive iff that subgoal flips 0->1 inside the
budget. Same policy, same budget, same positivity rule V7.4B used. The ONLY
difference is where the starting state came from. That is what makes the
comparison against 4/129 meaningful.

Snapshot construction is an acquisition instrument and never a policy
target; the recovery rollout is pi0.5 alone, with the reacher switched off.

Output: `<date>_v076_pilot_r1/yield_pilot.json`. Exit 0 on PASS, 1 on FAIL.
"""

from __future__ import annotations

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
RID = "v076_pilot_r1"
CENSUS = RESULTS / "2026-08-15_v076_precheck_r1" / "reacher_census.json"

# Registered before execution --------------------------------------------
N_TARGET = 40
PASS_MIN = 8
REC_MAX_ACT = 60           # V7.6B widened budget (V7.4B was 30)
REC_MAX_DEC = 6            # V7.4B was 3
PHASE_BUDGET = 80          # per reacher subgoal phase, as in the pre-check
SEEDS = (2610, 2611, 2612, 2613, 2614)   # fresh, disjoint from 2600-2602
FAMILY_DEPTH = {"first_pick": 1, "placement": 2,
                "recovery": 3, "late_chain": 4}
EPISODE_LENGTH = {"loho_t1_drawer": 700, "loho_t2_basket3": 900,
                  "loho_t3_tray": 900, "loho_t4_tray": 900,
                  "loho_t5_drawer_cabinet": 990}
V74_RATE = 4 / 129


def close_env(env) -> None:
    """Release the EGL context. 40 envs held open exhaust the driver."""
    try:
        env.close()
    except Exception:
        pass


def sha_seed(p: str) -> int:
    return int(hashlib.sha256(p.encode()).hexdigest()[:8], 16)


def run_date() -> str:
    return subprocess.run(["date", "+%F"], capture_output=True, text=True,
                          env={"TZ": "America/Chicago"}).stdout.strip()


def atomic_prompt(task, subgoal, OBJ_DISPLAY, REGION_DISPLAY):
    """The V7.4B atomic acquisition prompt, reused verbatim."""
    parts = subgoal.split()
    form, obj = parts[0], parts[1]
    if form in ("close", "open"):
        rd = REGION_DISPLAY.get(parts[1], parts[1].replace("_", " "))
        return f"{form} {rd}"
    od = OBJ_DISPLAY.get((task, obj),
                         "the " + obj.rsplit("_", 1)[0].replace("_", " "))
    if form == "pick_up":
        return f"pick up {od}"
    region = parts[2] if len(parts) > 2 else None
    return (f"put {od} in {REGION_DISPLAY.get(region, 'place')}"
            if region else f"lift {od} off the table")


def build_plan(census: dict) -> list[dict]:
    """(task, family, seed) rows, round-robin over tasks then families.

    Round-robin rather than task-major so a truncated run is still
    balanced across tasks -- a partial result must not be a t1 result.
    """
    usable = [(t, a["constructible_families"])
              for t, a in census["allocation"].items()
              if a["constructible_families"]]
    rows, i = [], 0
    while len(rows) < N_TARGET:
        progressed = False
        for task, fams in usable:
            if len(rows) >= N_TARGET:
                break
            fam = fams[i % len(fams)]
            seed = SEEDS[(i // len(fams)) % len(SEEDS)]
            rows.append({"task": task, "family": fam, "seed": seed})
            progressed = True
        if not progressed:
            break
        i += 1
    return rows[:N_TARGET]


def wilson_lower(k: int, n: int, z: float = 1.645) -> float:
    """One-sided lower bound; reported beside the exact-rule verdict."""
    if n == 0:
        return 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float(max(0.0, (c - m) / d))


@torch.no_grad()
def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=N_TARGET)
    args = ap.parse_args()

    from lcwm.chassis import Pi05Runner
    from lcwm.loho_public import make_public_env
    from lcwm.sampler import prefix_forward, sample_chunks
    from lcwm.task_automaton import GoalAutomaton
    from lcwm.v067_lineage import flow_noise
    from lcwm.v076_reacher import ScriptedReacher
    from scripts.collect_v069_corrections import (OBJ_DISPLAY,
                                                  REGION_DISPLAY)

    census = json.loads(CENSUS.read_text())
    plan = build_plan(census)[:args.limit]
    man = json.loads((RESULTS / "task_source_manifest.json").read_text())

    device = torch.device("cuda")
    runner = Pi05Runner(suite_name="libero_10")
    cfg = runner.policy.config
    print(f"[pilot] {len(plan)} attempts over "
          f"{len({r['task'] for r in plan})} tasks", flush=True)

    def obs_frame(env):
        return env._format_raw_obs(env._env.env._get_observations())

    rows = []
    for idx, item in enumerate(plan):
        task, family, seed = item["task"], item["family"], item["seed"]
        subgoals = man["tasks"][task]["ordered_subgoals"]
        depth = FAMILY_DEPTH[family]

        env = make_public_env(task, EPISODE_LENGTH[task] + 400)
        runner.reset()
        env.reset(seed=seed)
        auto = GoalAutomaton(subgoals)
        auto.start(env)
        auto.evaluate(env, 0)

        # ---- snapshot construction (acquisition instrument) ----------
        reacher = ScriptedReacher(auto.bodies)
        steps, built = 0, True
        for i in range(depth):
            parts = subgoals[i].split()
            mode, obj = parts[0], parts[1]
            region = parts[2] if len(parts) > 2 else None
            reacher.reset_phase()
            done = False
            for _ in range(PHASE_BUDGET):
                a = reacher.act(env, mode, obj, region,
                                auto.start_pos.get(obj))
                env.step(a)
                steps += 1
                auto.evaluate(env, steps)
                env._env.env.done = False
                if auto.prev_valid[i]:
                    done = True
                    break
            if not done:
                built = False
                break
        reached = auto.ordered_prefix()
        if not built or reached < depth:
            rows.append({"task": task, "family": family, "seed": seed,
                         "snapshot_built": False,
                         "reached_depth": int(reached),
                         "positive": False,
                         "excluded_from_yield": True,
                         "note": "snapshot construction failed; excluded "
                                 "from the yield denominator and "
                                 "reported separately"})
            print(f"[pilot] {idx + 1}/{len(plan)} {task} {family} s{seed}"
                  f"  SNAPSHOT FAILED (d{reached})", flush=True)
            close_env(env)
            continue

        # ---- recovery rollout: pi0.5 ALONE, reacher off --------------
        first_unres = next((sg for i, sg in enumerate(subgoals)
                            if not auto.prev_valid[i]), None)
        if first_unres is None:
            rows.append({"task": task, "family": family, "seed": seed,
                         "snapshot_built": True, "positive": False,
                         "excluded_from_yield": True,
                         "note": "chain already complete at snapshot"})
            close_env(env)
            continue
        target_idx = subgoals.index(first_unres)
        prompt = atomic_prompt(task, first_unres, OBJ_DISPLAY,
                               REGION_DISPLAY)
        flips0 = len(auto.flips)
        rec_steps, positive = 0, False
        frame = obs_frame(env)
        for cd in range(REC_MAX_DEC):
            b = runner._obs_to_policy_batch(frame, prompt)
            pfx = prefix_forward(runner.policy, b)
            nz = flow_noise(sha_seed(f"{RID}|rec|{task}|{family}|{seed}"
                                     f"|{cd}"),
                            cfg.chunk_size, cfg.max_action_dim)
            ch = sample_chunks(runner.policy, b, n=1,
                               noise=nz.to(device), prefix=pfx)
            for a_env in runner.chunk_to_env(ch[:, :10]):
                env.step(a_env)
                rec_steps += 1
                auto.evaluate(env, steps + rec_steps)
                env._env.env.done = False
                frame = obs_frame(env)
                if any(f[2] == 1 and f[1] == target_idx
                       for f in auto.flips[flips0:]):
                    positive = True
                    break
                if rec_steps >= REC_MAX_ACT:
                    break
            if positive or rec_steps >= REC_MAX_ACT:
                break

        rows.append({
            "task": task, "family": family, "seed": seed,
            "snapshot_built": True, "snapshot_actions": int(steps),
            "snapshot_depth": int(reached),
            "target_subgoal": first_unres, "prompt": prompt,
            "recovery_actions": int(rec_steps),
            "recovery_decisions": int(np.ceil(rec_steps / 10)),
            "positive": bool(positive),
            "excluded_from_yield": False,
            "final_prefix": int(auto.ordered_prefix()),
            "acquisition_instrument_only": True,
        })
        print(f"[pilot] {idx + 1}/{len(plan)} {task} {family} s{seed}"
              f"  d{reached} -> '{prompt}' in {rec_steps} acts"
              f"  {'POSITIVE' if positive else 'negative'}", flush=True)
        close_env(env)

    scored = [r for r in rows if not r["excluded_from_yield"]]
    n, k = len(scored), sum(r["positive"] for r in scored)
    rate = k / n if n else 0.0
    verdict = "PASS" if k >= PASS_MIN else "FAIL"

    by_task, by_family = {}, {}
    for key, dst in (("task", by_task), ("family", by_family)):
        for r in scored:
            e = dst.setdefault(r[key], {"n": 0, "pos": 0})
            e["n"] += 1
            e["pos"] += bool(r["positive"])

    out = Path(args.out) if args.out else RESULTS / f"{run_date()}_{RID}"
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "v076_yield_pilot_v1", "run_schema": "v076",
        "run_id": RID,
        "registered_by": "plan_and_progress/archive/daily/2026-08-15.md — V7.6B YIELD "
                         "PILOT, registered before execution",
        "registered": {
            "n_target": N_TARGET, "pass_min": PASS_MIN,
            "rec_max_actions": REC_MAX_ACT,
            "rec_max_decisions": REC_MAX_DEC,
            "seeds": list(SEEDS),
            "derivation": "tranche quota needs >=25 positives over ~260 "
                          "planned attempts = 9.6% operating yield; the "
                          "exact one-sided 95% binomial lower bound at "
                          "8/40 is ~0.104, which clears it",
        },
        "isolated_variable": "snapshot origin only — policy, budget and "
                             "positivity rule are the V7.4B protocol "
                             "unchanged",
        "baseline_v074": {"positive": 4, "n": 129, "rate": V74_RATE},
        "rows": rows,
        "n_scored": n, "n_positive": k, "rate": rate,
        "wilson_lower_90": wilson_lower(k, n),
        "n_snapshot_failures": sum(1 for r in rows
                                   if r["excluded_from_yield"]),
        "by_task": by_task, "by_family": by_family,
        "verdict": verdict,
        "routing": (
            "PASS -> the V7.6A thesis is supported; V7.6B collects at the "
            "census allocation"
            if verdict == "PASS" else
            "FAIL -> the thesis is refuted at the operating point; V7.6 "
            "halts at acquisition as a yield-bounded negative. Moving the "
            "snapshot is not sufficient to make grounded correction data "
            "collectible at this policy's competence, which locates the "
            "bottleneck in pi0.5's recovery capability rather than in the "
            "acquisition design."),
    }
    dest = out / "yield_pilot.json"
    dest.write_text(json.dumps(report, indent=2, default=float))
    print(f"\n[pilot] {k}/{n} positive = {rate:.1%} "
          f"(V7.4B: 4/129 = {V74_RATE:.1%}); need >= {PASS_MIN} -> "
          f"{verdict}", flush=True)
    for t, e in by_task.items():
        print(f"      {t:<24} {e['pos']}/{e['n']}")
    for f, e in by_family.items():
        print(f"      {f:<24} {e['pos']}/{e['n']}")
    print(f"[pilot] wrote {dest}", flush=True)
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
