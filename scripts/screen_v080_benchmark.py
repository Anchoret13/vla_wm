#!/usr/bin/env python
"""V8.0 - mechanism-development benchmark calibration (framework v1.0 §13.1,
daily 2026-08-17 Action 1).

Runs STOCK pi0.5 only: full-prompt, `N=1`, ChainEnv (the evaluator owns reset).
No fine-tuning, no reachers, no snapshot restoration, no decomposed prompts.
Nothing here trains anything or writes a policy.

What it must produce, in order of importance:

1. the per-task stock success rate with a Wilson 95% interval, against the
   registered `[0.20, 0.70]` band with LB >= 0.10 and UB <= 0.85;
2. the milestone completion-step distribution, from which the stall window `W`,
   the continuation budget `H_max`, and the reporting horizon set `H` are
   derived and frozen for every later V8 stage;
3. the failure-phase census, because a task with a single failure phase gives
   failure anchors at one milestone only.

Every decision rule below was written before any episode ran.  NOTE (audit
2026-08-19): they were written before execution but the tree was not committed
until afterwards, so the run manifest records `git_dirty=true`.  The claim this
comment may be read as making - commit-before-execution - is not supported by
the git tree, and Stage 1R.0 seals provenance properly instead.
The horizon is derived (250 steps/subgoal, cap 990) rather than inherited, per
the instrument rule locked in framework §0.1 after V7.7 showed a 60-action
budget could not register a `pick_up` success at all.

Usage:
    python scripts/screen_v080_benchmark.py --panel screen
    python scripts/screen_v080_benchmark.py --panel confirm --tasks chain2_lr2,...
    python scripts/screen_v080_benchmark.py --panel screen --tasks chain1_lr2 \
        --limit 1 --dry-run          # harness smoke, 1 episode
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

from lcwm import v080_bench as B  # noqa: E402
from lcwm.chassis import DEFAULT_MODEL, Pi05Runner  # noqa: E402
from lcwm.v080_bench import run_v080_episode  # noqa: E402

MODEL_ID = DEFAULT_MODEL   # lerobot/pi05_libero_finetuned

# --------------------------------------------------------------------------
# Registered decision rules (framework §13.1).  Do not edit after first run.
# --------------------------------------------------------------------------

#: A task advances from the screen panel to the confirmation panel iff its
#: screen point estimate lies here.  Deliberately WIDER than the PASS band so
#: the confirmation panel is not pre-selected onto the band it must test.
FINALIST_BAND = (0.10, 0.80)

#: PASS band on the confirmation panel.
PASS_BAND = (0.20, 0.70)
PASS_WILSON_LB_MIN = 0.10          # nonzero competence demonstrated
PASS_WILSON_UB_MAX = 0.85          # headroom demonstrated
PASS_MIN_TASKS = 2                 # >= 2 tasks must clear all of the above
PASS_MIN_FAILURE_PHASES = 2        # distinct stall milestones per passing task

#: W and H_max are both 1.5 * q90(milestone-to-milestone interval): framework
#: §3.1.1 for the stall window, §5.2.1 for the continuation budget.
DERIVE_MULTIPLIER = 1.5
DERIVE_QUANTILE = 0.90

#: W and H_max are derived ONLY from the tasks the mechanism will actually
#: use - finalists on the screen panel, PASSING tasks on the confirmation
#: panel.  Registered here before execution: a budget frozen partly from a
#: rung that is then dropped describes a distribution the mechanism never
#: samples.  Milestone class is `(index, kind, object)` from the frozen
#: `ordered_subgoals`, which is fixed per task and does not depend on the
#: order a given episode happened to solve the chain in.
DERIVE_SCOPE = {"screen": "finalists", "confirm": "passing"}

#: V7.7 measured the `pick_up` milestone class directly: twelve successes
#: spanning steps 66..106 from a fresh episode start.  The screen re-measures
#: that class and reports the comparison, so the 250-steps/subgoal horizon can
#: be checked against the evidence it was derived from.
V77_PICKUP_REFERENCE = {"earliest": 66, "latest": 106, "n": 12,
                        "source": "2026-08-15 v077_reacquire_r1, cell A"}

#: Reporting horizons.  60 is kept deliberately: it is the pre-pivot V7.6
#: recovery budget, and carrying it makes the truncation effect visible in
#: every later table instead of arguable.
LEGACY_HORIZON = 60

WILSON_Z = 1.959963984540054       # two-sided 95%


def wilson(k: int, n: int, z: float = WILSON_Z) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    d = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    rad = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (centre - rad) / d), min(1.0, (centre + rad) / d))


def quantile(xs: list[float], q: float) -> float | None:
    """Linear-interpolated quantile; None on an empty sample."""
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return float(s[0])
    pos = q * (len(s) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(s[lo])
    return float(s[lo] + (s[hi] - s[lo]) * (pos - lo))


def checkpoint_identity(model_id: str) -> dict:
    """Pin the exact weights this calibration was produced with.

    `chassis.DEFAULT_MODEL` is a Hub repo id with no revision, and it is a
    LIBERO *fine-tune* of pi0.5 - calling it "stock pi0.5, no fine-tuning"
    would attribute this calibration to the wrong model.  Every later V8 stage
    stores four checkpoint hashes per branch (framework §5.2.1); they are
    anchored to whatever this run fixes here.
    """
    info = {"model_id": model_id,
            "description": ("LIBERO-finetuned pi0.5 as published; not further "
                            "tuned in this project. 'stock' below always means "
                            "this checkpoint, unmodified by us.")}
    try:
        from huggingface_hub import HfApi
        info["hub_revision_sha"] = HfApi().model_info(model_id).sha
    except Exception as e:
        info["hub_revision_sha"] = f"unavailable: {type(e).__name__}"
    try:
        from huggingface_hub import snapshot_download
        snap = Path(snapshot_download(model_id))
        info["local_snapshot"] = str(snap)
        info["weight_files"] = {
            f.name: {"sha256": sha256_file(f), "bytes": f.stat().st_size}
            for f in sorted(snap.glob("*.safetensors"))}
        cfg = snap / "config.json"
        if cfg.exists():
            info["config_sha256"] = sha256_file(cfg)
    except Exception as e:
        info["local_snapshot"] = f"unavailable: {type(e).__name__}: {e}"
    return info


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                             capture_output=True, text=True,
                             check=True).stdout.strip()
        return bool(out)
    except Exception:
        return True


# --------------------------------------------------------------------------
# Episode -> record
# --------------------------------------------------------------------------

def episode_record(res, task: str, panel: str, wall_s: float) -> dict:
    """V080Result -> the flat row written to the JSONL.

    Progress is read from the frozen `GoalAutomaton`, not from the BDDL goal
    conjunction.  Each object contributes two milestones (`pick_up`, `place`),
    so a 1-subgoal rung has two milestones and three reachable failure phases;
    driving this off goal atoms would have given it exactly one and excluded it
    from PASS_MIN_FAILURE_PHASES before a single episode ran.
    """
    subgoals = res.subgoals
    achieved = {int(k): int(v) for k, v in res.events_achieved.items()}

    # The chain goals are UNORDERED conjunctions - the BDDL says
    # `(And (In a basket) (In b basket))` with no sequencing - and the screen
    # data shows pi0.5 does solve them out of order (chain2b seed 3002 took
    # cream cheese at 80/160 before tomato sauce at 340/386).  So:
    #
    #  * p(t) is instantiated as the COUNT of achieved milestones, not the
    #    ordered prefix.  §3.1.1's "ordered-milestone index" is only
    #    well-defined when the task enforces an order, which these do not.
    #  * intervals are gaps between successive achievements IN TIME, tagged by
    #    the class of the milestone that ENDED the gap.  Differencing by
    #    subgoal index instead produces negative intervals on out-of-order
    #    episodes and poisons the q90 that freezes W and H_max.
    ordered = sorted(achieved.items(), key=lambda kv: (kv[1], kv[0]))
    out_of_order = [i for i, _ in ordered] != sorted(achieved)

    intervals, prev = [], 0
    for rank, (i, step) in enumerate(ordered):
        kind, obj = subgoals[i].split()[0], subgoals[i].split()[1]
        intervals.append({
            "milestone_index": i, "kind": kind, "object": obj,
            "class_key": f"{i}|{kind} {obj}",
            "achieved_at": step, "achieved_rank": rank,
            "interval": step - prev,
            "interval_class": "start_to_first" if rank == 0
                              else "milestone_to_milestone"})
        prev = step

    unachieved = [sg for i, sg in enumerate(subgoals) if i not in achieved]

    return {
        "task": task,
        "panel": panel,
        "condition": res.condition,
        "seed": res.seed,
        "success": bool(res.success),
        "q_score": float(res.q_score),
        "steps": int(res.steps),
        "episode_length": B.episode_length(task),
        "n_subgoals": B.V080_TASKS[task]["n_subgoals"],
        "n_milestones": len(subgoals),
        "subgoals": subgoals,
        "milestones_achieved": len(achieved),
        "events_achieved": {str(k): v for k, v in sorted(achieved.items())},
        "solved_out_of_order": out_of_order,
        # failure phase = p(t) at termination = number of achieved milestones;
        # unsuccessful episodes only.  Identical to "first unresolved index"
        # whenever the achieved set is a prefix, which every failure in the
        # screen panel was.
        "failure_phase": None if res.success else len(achieved),
        "failure_signature": None if res.success else unachieved,
        "ordered_prefix": int(res.ordered_prefix),
        "q_valid": float(res.q_valid),
        "damage_unrecovered": int(res.damage_unrecovered),
        "milestone_intervals": intervals,
        "flips": res.flips,
        "milestone_timeline": res.milestone_timeline,
        # goal-atom view retained for comparability with the V7.x records
        "subgoal_completion_t": res.subgoal_completion_t,
        "bits_timeline": res.bits_timeline,
        "instruction": res.instruction,
        "wall_s": round(wall_s, 2),
    }


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

def _class_stats(samples: list[int]) -> dict:
    q90 = quantile([float(x) for x in samples], DERIVE_QUANTILE)
    out = {"n": len(samples), "q90": q90,
           "median": quantile([float(x) for x in samples], 0.5),
           "min": min(samples) if samples else None,
           "max": max(samples) if samples else None}
    out["bound"] = int(math.ceil(DERIVE_MULTIPLIER * q90)) if q90 is not None else None
    return out


def derive_budgets(rows: list[dict], scope_tasks: list[str]) -> dict:
    """W and H_max, per milestone class, over the in-scope tasks only.

    framework §5.2.1 binds H_max to q90(steps-to-next-milestone | milestone
    class at the anchor's phase) and §13 forbids reporting an aggregate without
    the per-class breakdown beside it.  The pooled number is therefore computed
    and reported but explicitly NOT the frozen budget: pooling mixes
    start-to-first-milestone (fresh scene, arm at home) with late-chain
    re-approach, and the 1-subgoal rungs contribute only the former.
    """
    in_scope = [r for r in rows if r["task"] in scope_tasks]
    per_class: dict[str, list[int]] = {}
    by_kind: dict[str, list[int]] = {}
    by_interval_class: dict[str, list[int]] = {}
    pooled: list[int] = []
    undefined = 0
    for r in in_scope:
        for iv in r["milestone_intervals"]:
            if iv["interval"] is None:
                undefined += 1
                continue
            key = f"{r['task']}|{iv['class_key']}"
            per_class.setdefault(key, []).append(iv["interval"])
            by_kind.setdefault(iv["kind"], []).append(iv["interval"])
            by_interval_class.setdefault(iv["interval_class"], []).append(iv["interval"])
            pooled.append(iv["interval"])

    cls_stats = {k: _class_stats(v) for k, v in sorted(per_class.items())}
    bounds = [c["bound"] for c in cls_stats.values() if c["bound"] is not None]
    frozen = max(bounds) if bounds else None
    horizons = (sorted({LEGACY_HORIZON, max(1, frozen // 2), frozen})
                if frozen is not None else None)

    derived = {
        "scope_tasks": scope_tasks,
        "scope_rule": ("W and H_max derived only from the tasks the mechanism "
                       "will use; registered before execution"),
        "per_class": cls_stats,
        "by_kind": {k: _class_stats(v) for k, v in sorted(by_kind.items())},
        "by_interval_class": {k: _class_stats(v)
                              for k, v in sorted(by_interval_class.items())},
        "undefined_intervals": undefined,
        "pooled_REPORTED_NOT_FROZEN": _class_stats(pooled),
        # frozen scalars: the max over classes, so no class is continued below
        # its own 1.5*q90 floor (§5.2.1 is an inequality)
        "W_stall_window": frozen,
        "H_max": frozen,
        "H_max_by_class": {k: c["bound"] for k, c in cls_stats.items()},
        "H_reporting_set": horizons,
        "value_estimator": "MC" if frozen is not None else None,
        "note": ("W (§3.1.1) and H_max (§5.2.1) are the same 1.5*q90 bound on "
                 "the interval preceding a milestone; the frozen scalar is the "
                 "MAX over classes so the inequality holds for every class, and "
                 "H_max_by_class is retained so continuations can be indexed by "
                 "anchor phase.  Horizon 60 is kept in H so the V7.6 truncation "
                 "stays visible in every later table."),
    }
    pick = derived["by_kind"].get("pick_up")
    derived["v77_crosscheck"] = {
        "reference": V77_PICKUP_REFERENCE,
        "measured_pick_up": pick,
        "reading": (None if not pick or pick["q90"] is None else
                    ("consistent with the V7.7 floor" if pick["min"] is not None
                     and pick["min"] >= 40 else
                     "measured pick_up faster than V7.7 - investigate before freezing")),
    }
    return derived


def summarize(rows: list[dict], panel: str) -> dict:
    by_task: dict[str, list[dict]] = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)

    per_task = {}
    for task, rs in sorted(by_task.items()):
        n = len(rs)
        k = sum(r["success"] for r in rs)
        rate = k / n if n else 0.0
        lb, ub = wilson(k, n)
        phases = sorted({r["failure_phase"] for r in rs
                         if r["failure_phase"] is not None})
        ivs = [iv["interval"] for r in rs for iv in r["milestone_intervals"]
               if iv["interval"] is not None]
        by_index: dict[int, list[int]] = {}
        for r in rs:
            for iv in r["milestone_intervals"]:
                if iv["interval"] is not None:
                    by_index.setdefault(iv["milestone_index"], []).append(iv["interval"])
        per_task[task] = {
            "n": n, "successes": k, "rate": rate,
            "wilson95": [lb, ub],
            "in_pass_band": PASS_BAND[0] <= rate <= PASS_BAND[1],
            "lb_ok": lb >= PASS_WILSON_LB_MIN,
            "ub_ok": ub <= PASS_WILSON_UB_MAX,
            "distinct_failure_phases": phases,
            "n_distinct_failure_phases": len(phases),
            "in_finalist_band": FINALIST_BAND[0] <= rate <= FINALIST_BAND[1],
            "mean_q": sum(r["q_score"] for r in rs) / n if n else 0.0,
            "mean_q_valid": sum(r["q_valid"] for r in rs) / n if n else 0.0,
            "mean_ordered_prefix": sum(r["ordered_prefix"] for r in rs) / n if n else 0.0,
            "n_milestones": rs[0]["n_milestones"],
            "damage_episodes": sum(1 for r in rs if r["damage_unrecovered"] > 0),
            "milestone_interval_stats_by_index": {
                str(i): _class_stats(v) for i, v in sorted(by_index.items())},
            "milestone_intervals_n": len(ivs),
            "total_env_steps": sum(r["steps"] for r in rs),
            "wall_s": round(sum(r["wall_s"] for r in rs), 1),
        }

    passing = [t for t, s in per_task.items()
               if s["in_pass_band"] and s["lb_ok"] and s["ub_ok"]
               and s["n_distinct_failure_phases"] >= PASS_MIN_FAILURE_PHASES]
    finalists = [t for t, s in per_task.items() if s["in_finalist_band"]]

    scope = passing if DERIVE_SCOPE[panel] == "passing" else finalists
    derived = derive_budgets(rows, scope)
    # The full table is always emitted for information, even when the scope set
    # is empty (every rung out of band).  Only the scoped block is frozen.
    derived["all_tasks_REPORTED_NOT_FROZEN"] = derive_budgets(
        rows, sorted(by_task))

    if panel == "confirm":
        ok = len(passing) >= PASS_MIN_TASKS
        decision = {
            "rule": (f"PASS iff >= {PASS_MIN_TASKS} tasks with rate in "
                     f"{PASS_BAND}, Wilson LB >= {PASS_WILSON_LB_MIN}, "
                     f"UB <= {PASS_WILSON_UB_MAX}, and >= "
                     f"{PASS_MIN_FAILURE_PHASES} distinct failure phases"),
            "passing_tasks": passing,
            "verdict": "PASS" if ok else "HALT",
            "halt_action": None if ok else (
                "framework §13.1: change task difficulty (object count, "
                "distractor set, initial-state region) before building any "
                "world model.  Do NOT move the bar and do NOT introduce "
                "privileged late-state reachers into the benchmark."),
        }
    else:
        decision = {
            "rule": f"advance to confirmation iff screen rate in {FINALIST_BAND}",
            "finalists": finalists,
            "verdict": "ADVANCE" if finalists else "HALT",
            "halt_action": None if finalists else (
                "no candidate is even near the band; author a different "
                "difficulty ladder before spending the confirmation panel"),
        }

    return {"panel": panel, "per_task": per_task, "derived": derived,
            "decision": decision,
            "totals": {"episodes": len(rows),
                       "env_steps": sum(r["steps"] for r in rows),
                       "wall_s": round(sum(r["wall_s"] for r in rows), 1)}}


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", choices=B.EXECUTABLE_PANELS, required=True)
    ap.add_argument("--tasks", default=",".join(B.V080_TASKS))
    ap.add_argument("--limit", type=int, default=0,
                    help="cap episodes per task (smoke tests only)")
    ap.add_argument("--dry-run", action="store_true",
                    help="do not write results/ artifacts")
    ap.add_argument("--out-root", default="results/v080_screen")
    args = ap.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in tasks if t not in B.V080_TASKS]
    if unknown:
        raise SystemExit(f"unknown task(s): {unknown}")

    seeds = list(B.SEED_FAMILIES[args.panel])
    if args.limit:
        seeds = seeds[: args.limit]

    # Hard guard: reserved families must never be executed here.
    reserved = set(B.SEED_FAMILIES["behavior"]) | set(B.SEED_FAMILIES["acquisition"])
    if reserved & set(seeds):
        raise SystemExit("refusing to execute reserved behavior/acquisition seeds")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out_dir = REPO_ROOT / args.out_root / f"{stamp}_{args.panel}"
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "stage": "V8.0", "action": "Action 1 - benchmark calibration",
        "framework": "v1.0 §13.1", "panel": args.panel,
        "utc": stamp, "git_sha": git_sha(), "git_dirty": git_dirty(),
        "tasks": tasks, "seeds": seeds,
        "condition": "full",           # full-prompt N=1 only
        "policy": checkpoint_identity(MODEL_ID),
        "policy_runtime": {"n_action_steps": 10, "device": "cuda",
                           "suite_name": "libero_10",
                           "no_finetuning_in_this_project": True},
        "horizon_rule": "min(990, 250 * n_subgoals)",
        "episode_lengths": {t: B.episode_length(t) for t in tasks},
        "n_action_steps": 10,
        "predicate_stride": 10,
        "registered_rules": {
            "finalist_band": FINALIST_BAND, "pass_band": PASS_BAND,
            "pass_wilson_lb_min": PASS_WILSON_LB_MIN,
            "pass_wilson_ub_max": PASS_WILSON_UB_MAX,
            "pass_min_tasks": PASS_MIN_TASKS,
            "pass_min_failure_phases": PASS_MIN_FAILURE_PHASES,
            "derive_multiplier": DERIVE_MULTIPLIER,
            "derive_quantile": DERIVE_QUANTILE,
            "legacy_horizon_retained": LEGACY_HORIZON,
        },
        "source_hashes": {
            p: sha256_file(REPO_ROOT / p) for p in
            ["scripts/screen_v080_benchmark.py", "lcwm/v080_bench.py",
             "lcwm/loho.py", "lcwm/chassis.py"]
            + [f"bddl/chains/{t}.bddl" for t in tasks]
        },
        "not_comparable_to": ("historical Chain3 0/5 at episode_length=700; "
                              "this screen uses a different derived horizon"),
    }
    print(json.dumps({k: manifest[k] for k in
                      ("stage", "panel", "tasks", "episode_lengths",
                       "git_sha", "git_dirty")}, indent=2), flush=True)

    if not args.dry_run:
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    jl = out_dir / "episodes.jsonl"
    if not args.dry_run:
        jl.write_text("")

    runner = Pi05Runner(model_id=MODEL_ID, suite_name="libero_10",
                        n_action_steps=10)

    rows: list[dict] = []
    for task in tasks:
        env = B.make_v080_env(task)
        for seed in seeds:
            t0 = time.time()
            res = run_v080_episode(runner, env, seed=seed, stride=10)
            rec = episode_record(res, task, args.panel, time.time() - t0)
            rows.append(rec)
            if not args.dry_run:
                with jl.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
            print(f"{task} seed={seed} success={rec['success']} "
                  f"q={rec['q_score']:.3f} steps={rec['steps']} "
                  f"milestones={rec['milestones_achieved']}/{rec['n_milestones']} "
                  f"phase={rec['failure_phase']} dmg={rec['damage_unrecovered']} "
                  f"events={rec['events_achieved']} "
                  f"({rec['wall_s']:.0f}s)", flush=True)
        try:
            env.close()
        except Exception:
            pass

    summary = summarize(rows, args.panel)
    summary["manifest"] = manifest
    if not args.dry_run:
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== V8.0 summary ===")
    for task, s in summary["per_task"].items():
        print(f"{task:14s} {s['successes']:>3}/{s['n']:<3} "
              f"rate={s['rate']:.3f} wilson=[{s['wilson95'][0]:.3f},"
              f"{s['wilson95'][1]:.3f}] q={s['mean_q']:.3f} "
              f"phases={s['distinct_failure_phases']} "
              f"milestones={s['n_milestones']}")
    print(json.dumps({"derived": summary["derived"],
                      "decision": summary["decision"],
                      "totals": summary["totals"]}, indent=2))
    if not args.dry_run:
        print(f"\nartifacts: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
