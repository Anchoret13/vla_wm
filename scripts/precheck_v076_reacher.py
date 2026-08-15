#!/usr/bin/env python
"""V7.6B reacher feasibility pre-check — the gate before 2-3 GPU-days.

Registered in `plan_and_progress/2026-08-15.md` (V7.6B amendment):

    Before collecting 70 sources, run the reacher on one task x the four
    failure families x 3 attempts (12 attempts). Registered pass
    condition: the reacher places the arm in the designated
    configuration on >= 8 of 12, with replay fidelity inside the V7.4B
    audit tolerance.

Why this exists. The V7.6A census showed V7.4B recovery yield at 4/129
(3.1%), with 127 of 129 attempts consuming the whole 30-action budget and
the two fastest attempts both succeeding. Recovery is decided by where the
snapshot SITS, not by the recovery policy -- so V7.6B moves the snapshot,
and after the `expert_prefix` retraction (LoHo has no demonstrations) the
scripted servo is the only general mechanism left to move it with. The
entire V7.6B budget rests on one unproven instrument. This checks it for
minutes of CPU instead of discovering it after two days of collection.

The four failure families map to target ordered-prefix depths on the
registered subgoal chain. They are monotone -- deeper is strictly harder --
which is the property that makes the result interpretable: a pass at
`late_chain` implies the reacher can construct snapshots anywhere the
V7.4B tranche could not.

    first_pick   depth 1   pick_up  <first object>
    placement    depth 2   place    <first object> -> region
    recovery     depth 3   pick_up  <second object>   (mid-chain snapshot)
    late_chain   depth 4   place    <second object> -> region

Each attempt is INDEPENDENT (its own seed, its own target), faithful to
the registered "4 x 3 = 12 attempts" rather than scoring one deep rollout
four times.

No policy forward is involved -- the servo is scripted -- so this needs no
GPU and runs in minutes.

Provenance. Every action this script emits is an acquisition instrument.
Nothing here is a policy target, and the output records that on every row.

Output: `<date>_v076_precheck_r1/reacher_precheck.json`. Exit 0 on PASS,
1 on FAIL; both write the report.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
RID = "v076_precheck_r1"

# Registered before execution --------------------------------------------
TASK = "loho_t1_drawer"
EPISODE_LENGTH = 700
# Fresh seed family, disjoint from every prior acquisition family
# ({2500..2545} = v074) and from all behavior panels ({1700..1890}).
SEEDS = (2600, 2601, 2602)
FAMILY_DEPTH = {"first_pick": 1, "placement": 2,
                "recovery": 3, "late_chain": 4}
PHASE_BUDGET = 80          # env actions per subgoal phase (~5 cm/action)
PASS_MIN = 8               # of 12
# V7.4B replay_fidelity audit measured obj_pos 0.0, qpos 0.0,
# eef_pos <= 2.2e-3 over 33 anchors. Registered tolerance is generous
# against that, so a failure here means a real divergence.
FIDELITY_TOL = {"eef_pos": 5e-3, "obj_pos": 5e-3, "qpos": 5e-3}


def run_date() -> str:
    return subprocess.run(["date", "+%F"], capture_output=True, text=True,
                          env={"TZ": "America/Chicago"}).stdout.strip()


def subgoals_for(task: str) -> list[str]:
    man = json.loads((RESULTS / "task_source_manifest.json").read_text())
    return list(man["tasks"][task]["ordered_subgoals"])


def eef_pos(env) -> np.ndarray:
    inner = env._env.env
    robot = inner.robots[0]
    if hasattr(robot, "eef_site_id"):
        return np.asarray(
            inner.sim.data.site_xpos[robot.eef_site_id]).copy()
    obs = env._format_raw_obs(inner._get_observations())
    return np.asarray(obs["robot_state"]["eef"]["pos"])


def state_vector(env, bodies: dict) -> dict:
    """The three quantities the V7.4B fidelity audit compares."""
    from lcwm.probe_data import body_positions
    names = sorted(bodies)
    return {
        "eef_pos": eef_pos(env),
        "obj_pos": np.asarray(body_positions(
            env, [bodies[n] for n in names])).reshape(-1),
        "qpos": np.asarray(env._env.env.sim.data.qpos).copy(),
    }


def max_abs_diff(a: dict, b: dict) -> dict:
    out = {}
    for k in a:
        u, v = np.asarray(a[k]).ravel(), np.asarray(b[k]).ravel()
        out[k] = float(np.max(np.abs(u - v))) if u.shape == v.shape \
            else float("inf")
    return out


def make_actor(instrument: str, automaton):
    """(callable(env, mode, obj, region, start_pos) -> action, reset).

    Two instruments, both runnable, so the v069 FAIL stays reproducible
    alongside the v076 repair rather than being overwritten by it.
    """
    if instrument == "v069":
        from scripts.collect_v069_corrections import scripted_servo_action

        def act(env, mode, obj, region, start_pos):
            return scripted_servo_action(env, obj, region,
                                         automaton.bodies, mode)

        return act, (lambda: None)

    from lcwm.v076_reacher import ScriptedReacher
    reacher = ScriptedReacher(automaton.bodies)

    def act(env, mode, obj, region, start_pos):
        return reacher.act(env, mode, obj, region, start_pos)

    return act, reacher.reset_phase


def drive_to_depth(env, automaton, subgoals, depth, actor, reset_phase,
                   step0=0):
    """Drive through subgoals[0:depth]; return (actions, log, steps).

    Each phase stops as soon as ITS subgoal evaluates true, so a phase
    that finishes early hands its remaining budget to nothing -- the
    budget is per-phase by registration, not shared.
    """
    actions, phases, steps = [], [], step0
    for i in range(depth):
        parts = subgoals[i].split()
        mode, obj = parts[0], parts[1]
        region = parts[2] if len(parts) > 2 else None
        if mode not in ("pick_up", "place"):
            phases.append({"subgoal": subgoals[i], "mode": mode,
                           "status": "UNSUPPORTED_MODE", "used": 0})
            break
        start_pos = automaton.start_pos.get(obj)
        reset_phase()
        used, done = 0, False
        for _ in range(PHASE_BUDGET):
            a_env = actor(env, mode, obj, region, start_pos)
            env.step(a_env)
            steps += 1
            used += 1
            actions.append(np.asarray(a_env, dtype=np.float64))
            automaton.evaluate(env, steps)
            # env.done can latch on a terminal predicate; this is an
            # acquisition rollout, so it keeps going.
            env._env.env.done = False
            if automaton.prev_valid[i]:
                done = True
                break
        phases.append({"subgoal": subgoals[i], "mode": mode,
                       "status": "reached" if done else "budget_exhausted",
                       "used": used})
        if not done:
            break
    return actions, phases, steps


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--out", default=None)
    ap.add_argument("--instrument", choices=("v076", "v069"),
                    default="v076",
                    help="v076: lcwm.v076_reacher.ScriptedReacher (the "
                         "repaired instrument). v069: the original "
                         "scripted_servo_action, kept runnable so its "
                         "0/12 stays reproducible.")
    args = ap.parse_args()

    from lcwm.loho_public import make_public_env
    from lcwm.task_automaton import GoalAutomaton

    subgoals = subgoals_for(args.task)
    print(f"[pre] {args.task}: {len(subgoals)} ordered subgoals", flush=True)
    for i, sg in enumerate(subgoals):
        print(f"      {i + 1}. {sg}")

    max_depth = max(FAMILY_DEPTH.values())
    assert len(subgoals) >= max_depth, \
        f"{args.task} has {len(subgoals)} subgoals, need >= {max_depth}"

    attempts = []
    for family in sorted(FAMILY_DEPTH, key=lambda f: FAMILY_DEPTH[f]):
        depth = FAMILY_DEPTH[family]
        for seed in SEEDS:
            env = make_public_env(args.task, EPISODE_LENGTH)
            env.reset(seed=seed)
            auto = GoalAutomaton(subgoals)
            auto.start(env)
            auto.evaluate(env, 0)

            actor, reset_phase = make_actor(args.instrument, auto)
            actions, phases, steps = drive_to_depth(
                env, auto, subgoals, depth, actor, reset_phase)
            reached = auto.ordered_prefix()
            end_state = state_vector(env, auto.bodies)
            ok = reached >= depth

            # replay fidelity: same seed, replay the recorded env actions,
            # compare the three audited quantities at the endpoint
            fid, fid_ok = None, None
            if ok:
                renv = make_public_env(args.task, EPISODE_LENGTH)
                renv.reset(seed=seed)
                rauto = GoalAutomaton(subgoals)
                rauto.start(renv)
                for a in actions:
                    renv.step(a)
                    renv._env.env.done = False
                fid = max_abs_diff(end_state,
                                   state_vector(renv, rauto.bodies))
                fid_ok = all(fid[k] <= FIDELITY_TOL[k] for k in FIDELITY_TOL)
                renv.close() if hasattr(renv, "close") else None

            attempts.append({
                "family": family, "target_depth": depth, "seed": seed,
                "reached_depth": int(reached),
                "target_met": bool(ok),
                "steps": int(steps), "n_actions": len(actions),
                "phases": phases,
                "replay_fidelity": fid,
                "replay_fidelity_ok": fid_ok,
                "pass": bool(ok and fid_ok),
                "acquisition_instrument_only": True,
                "is_policy_target": False,
            })
            mark = "PASS" if attempts[-1]["pass"] else "fail"
            print(f"[pre] {family:<11} seed {seed} target d{depth} "
                  f"-> reached d{reached} in {steps} actions [{mark}]",
                  flush=True)
            env.close() if hasattr(env, "close") else None

    n_pass = sum(a["pass"] for a in attempts)
    by_family = {}
    for f in FAMILY_DEPTH:
        rows = [a for a in attempts if a["family"] == f]
        by_family[f] = {"target_depth": FAMILY_DEPTH[f],
                        "n": len(rows),
                        "n_pass": sum(r["pass"] for r in rows),
                        "reached": [r["reached_depth"] for r in rows]}
    verdict = "PASS" if n_pass >= PASS_MIN else "FAIL"

    out = Path(args.out) if args.out else RESULTS / f"{run_date()}_{RID}"
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "v076_reacher_precheck_v1",
        "run_schema": "v076", "run_id": RID,
        "registered_by": "plan_and_progress/2026-08-15.md — V7.6B "
                         "amendment, reacher feasibility pre-check",
        "task": args.task, "subgoals": subgoals,
        "registered": {
            "families": FAMILY_DEPTH, "seeds": list(SEEDS),
            "phase_budget_actions": PHASE_BUDGET,
            "pass_min": PASS_MIN, "n_attempts": len(attempts),
            "fidelity_tolerance": FIDELITY_TOL,
            "fidelity_basis": "V7.4B replay_fidelity audit measured "
                              "obj_pos 0.0, qpos 0.0, eef_pos <= 2.2e-3 "
                              "over 33 anchors",
        },
        "instrument": {
            "selected": args.instrument,
            "v069": "scripts.collect_v069_corrections."
                    "scripted_servo_action — no lift phase; the pick_up "
                    "target is defined relative to the object being "
                    "held, so the controller converges to a fixed point",
            "v076": "lcwm.v076_reacher.ScriptedReacher — explicit "
                    "approach/descend/grasp/lift phase machine; the "
                    "lift target is anchored to the object's "
                    "episode-start z, the same reference PICK_LIFT uses",
            "policy_forward": False,
            "note": "every action here is an acquisition instrument; "
                    "none is a policy target",
        },
        "attempts": attempts,
        "by_family": by_family,
        "n_pass": n_pass, "verdict": verdict,
        "routing": (
            "PASS -> V7.6B proceeds at the registered 70-source budget"
            if verdict == "PASS" else
            "FAIL -> V7.6 halts at acquisition and is reported as an "
            "instrument-bounded negative: the project cannot construct "
            "snapshots at blockers its policy cannot reach, and source "
            "volume does not fix that"),
    }
    dest = out / f"reacher_precheck_{args.instrument}.json"
    dest.write_text(json.dumps(report, indent=2, default=float))
    print(f"\n[pre] {n_pass}/{len(attempts)} pass "
          f"(need >= {PASS_MIN}) -> {verdict}", flush=True)
    for f, b in by_family.items():
        print(f"      {f:<11} d{b['target_depth']}  "
              f"{b['n_pass']}/{b['n']}  reached {b['reached']}")
    print(f"[pre] wrote {dest}", flush=True)
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
