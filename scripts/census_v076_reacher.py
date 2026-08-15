#!/usr/bin/env python
"""V7.6B per-task reacher feasibility census — sizes the collection.

The registered gate (`scripts/precheck_v076_reacher.py`) is one task x
four families x three attempts, and it PASSED on t1 at 9/12. That verdict
stands. This census is BEYOND the registered scope: it runs the same check
on the other four tasks, before collection, so the V7.6B budget is
allocated against measured per-task feasibility instead of an assumption
that one task generalizes.

It does not re-adjudicate the gate. It answers a different question --
"where can snapshots actually be constructed?" -- and turns the answer into
a per-task mechanism/family allocation that V7.6B must respect.

Reads the five `reacher_precheck_v076.json` reports and writes
`reacher_census.json`: per-task family support, the derived allocation, and
the named causes for every unsupported cell.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
ROOT = RESULTS / "2026-08-15_v076_precheck_r1"
FAMILIES = ("first_pick", "placement", "recovery", "late_chain")
# a family is usable for collection only if it cleared every attempt;
# 1/3 or 2/3 is not a mechanism you can plan a tranche around
SUPPORT_MIN = 3

# Causes established by step traces, not inferred from scores.
CAUSES = {
    "late_chain_confined":
        "depth 4 is a SECOND placement into the same region. It succeeds "
        "in t2 (basket, open container, 3/3) and fails in t1/t5 (cabinet "
        "drawer) and t3 (tray). The discriminator is container geometry, "
        "not occupancy -- t2 disproves the occupancy reading.",
    "t4_bowl_grasp":
        "t4's first subgoal is pick_up akita_black_bowl_1. The phase "
        "machine runs correctly (retreat/approach/descend/grasp/lift all "
        "transition), the arm descends to dz_eef_obj 0.019, closes, and "
        "lifts to z 0.866 -- while the bowl stays put at |disp| 0.0001, "
        "dz 0.0000. The GRASP fails: a body-centre top-down close on a "
        "bowl closes inside it, not on the rim. This is per-object "
        "geometry, a different problem class from the missing lift and "
        "the missing retreat, and it is NOT patched here.",
    "t3_second_pick":
        "t3 depth 3 (pick_up cream_cheese_1 after placing into the tray) "
        "clears 1/3. Marginal, so it is not planned around.",
}


def run_date() -> str:
    return subprocess.run(["date", "+%F"], capture_output=True, text=True,
                          env={"TZ": "America/Chicago"}).stdout.strip()


def load_reports() -> list[dict]:
    paths = [ROOT / "reacher_precheck_v076.json"]
    paths += sorted(ROOT.glob("*/reacher_precheck_v076.json"))
    out = []
    for p in paths:
        if p.is_file():
            out.append(json.loads(p.read_text()))
    return out


def main() -> None:
    reports = load_reports()
    assert reports, f"no precheck reports under {ROOT}"

    per_task, alloc = {}, {}
    for r in reports:
        task = r["task"]
        bf = r["by_family"]
        supported = [f for f in FAMILIES
                     if bf[f]["n_pass"] >= SUPPORT_MIN]
        per_task[task] = {
            "n_pass": r["n_pass"],
            "n_attempts": len(r["attempts"]),
            "gate_bar_met": r["n_pass"] >= r["registered"]["pass_min"],
            "by_family": {f: f"{bf[f]['n_pass']}/{bf[f]['n']}"
                          for f in FAMILIES},
            "supported_families": supported,
            "n_supported": len(supported),
        }
        if not supported:
            mech, note = ["stock_stall"], CAUSES["t4_bowl_grasp"]
        else:
            mech = ["stock_stall", "state_reacher"]
            note = None
            if "late_chain" not in supported:
                note = CAUSES["late_chain_confined"]
            if task == "loho_t3_tray":
                note = (note or "") + " " + CAUSES["t3_second_pick"]
        alloc[task] = {
            "snapshot_mechanisms": mech,
            "constructible_families": supported,
            "limitation": note,
        }

    n_full = sum(1 for v in per_task.values() if v["n_supported"] == 4)
    n_none = sum(1 for v in per_task.values() if v["n_supported"] == 0)
    fid = {"eef_pos": 0.0, "obj_pos": 0.0, "qpos": 0.0}
    n_fid = 0
    for r in reports:
        for a in r["attempts"]:
            if a.get("replay_fidelity"):
                n_fid += 1
                for k in fid:
                    fid[k] = max(fid[k], float(a["replay_fidelity"][k]))

    census = {
        "schema": "v076_reacher_census_v1",
        "run_schema": "v076", "run_id": "v076_precheck_r1",
        "scope": "BEYOND the registered gate. The gate is one task x 4 "
                 "families x 3 attempts and PASSED on loho_t1_drawer at "
                 "9/12; that verdict stands unchanged. This census sizes "
                 "the collection and does not re-adjudicate the gate.",
        "support_min_per_family": SUPPORT_MIN,
        "per_task": per_task,
        "totals": {
            "attempts": sum(v["n_attempts"] for v in per_task.values()),
            "pass": sum(v["n_pass"] for v in per_task.values()),
            "tasks_all_four_families": n_full,
            "tasks_no_constructible_family": n_none,
        },
        "replay_fidelity_max_over_passing": fid,
        "replay_fidelity_n": n_fid,
        "causes": CAUSES,
        "allocation": alloc,
        "binding_consequence":
            "V7.6B collects with state_reacher only where a family is "
            "3/3. Elsewhere it falls back to stock_stall, which is the "
            "V7.4B mechanism -- so no task is worse off than V7.4B, and "
            "the tasks that are not improved are named rather than "
            "silently thin.",
        "reopens_t4":
            "A per-object grasp-point table (rim offset for bowls) is the "
            "specific, bounded work that would reopen t4 to the reacher. "
            "Identified, not done, and not part of this slice.",
    }
    out = ROOT / "reacher_census.json"
    out.write_text(json.dumps(census, indent=2))

    print(f"{'task':<24} {'gate':>6}  families supported")
    for t, v in per_task.items():
        print(f"{t:<24} {v['n_pass']:>2}/12  "
              f"{v['n_supported']}/4  {v['supported_families']}")
    print(f"\ntotal {census['totals']['pass']}/"
          f"{census['totals']['attempts']} attempts; "
          f"{n_full} task(s) with all four families, "
          f"{n_none} with none")
    print(f"replay fidelity max over {n_fid} passing attempts: {fid}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
