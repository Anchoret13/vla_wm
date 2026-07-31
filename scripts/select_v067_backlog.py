#!/usr/bin/env python
"""V6.8 step 2 — apply the FROZEN backlog selection rule (registered in
2026-07-31.md BEFORE any backlog label was computed):

Per task, rank backlog groups by (a) any-goal semantic effect
(candidate-pair valid/event/flip/terminal disagreement), then (b) max
task-object dispersion (absolute; fixture-joint dispersion reported
alongside, not a rank key); take top 4 train-source + top 2 dev-source
groups per task; total cap 30; never by elapsed-time percentile.

Output: results/libero_loho_public_v1/v067_backlog_selection.json
(consumed by collect_v067_continuations.py --groups-json, run v067_pb2)
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATA = Path("/home/stargazer/Desktop/vla_wm/datasets/libero_loho_public_v1"
            "/v06_effect_crossed")
RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
TASKS = ["loho_t1_drawer", "loho_t2_basket3", "loho_t3_tray",
         "loho_t4_tray", "loho_t5_drawer_cabinet"]
TOP_TRAIN, TOP_DEV, CAP = 4, 2, 30


def goal_objects(subgoals):
    out = set()
    for sg in subgoals:
        parts = sg.split()
        if parts[0] in ("place", "pick_up"):
            out.add(parts[1])
    return out


def main() -> None:
    from lcwm.v067_lineage import load_v067

    goal_manifest = json.loads(
        (RESULTS / "goal_spec_manifest_v067.json").read_text())

    rows = []
    for p in sorted((DATA / "backlog_relabels_v067").glob("*.pt")):
        import torch
        lab = load_v067(p, "semantic_labels")  # same guard semantics
        task = lab["task"]
        specs = goal_manifest["tasks"][task]["goal_specs"]
        objs = set()
        for gid in lab["goals"]:
            objs |= goal_objects(specs[gid]["ordered_subgoals"])
        fixture_addrs = {
            n: a for n, a in lab.get("qpos_joint_map", {}).items()
            if not n.startswith(("robot0", "gripper0"))
            and not n.endswith("_joint0")}
        for g in lab["groups"]:
            cands = [b for b in g["branch_summaries"]
                     if b["kind"] == "candidate"]
            body_names = list(cands[0]["grasp_after"]["grasped"])
            sem = False
            max_task_disp, max_fix = 0.0, 0.0
            for i, j in itertools.combinations(range(len(cands)), 2):
                for gid in lab["goals"]:
                    a = cands[i]["immediate"][gid]
                    b = cands[j]["immediate"][gid]
                    if (a["valid_after"] != b["valid_after"]
                            or a["events_after"] != b["events_after"]
                            or bool(a["flips_10"]) != bool(b["flips_10"])
                            or a["terminal_now"] != b["terminal_now"]):
                        sem = True
                d_all = np.abs(np.asarray(cands[i]["obj_after"])
                               - np.asarray(cands[j]["obj_after"]))
                for bi, name in enumerate(body_names):
                    if name in objs and bi < d_all.shape[0]:
                        max_task_disp = max(max_task_disp,
                                            float(d_all[bi].max()))
                qa = np.asarray(cands[i]["qpos_after"])
                qb = np.asarray(cands[j]["qpos_after"])
                for addr in fixture_addrs.values():
                    max_fix = max(max_fix, float(abs(qa[addr]
                                                     - qb[addr])))
            rep = next((b for b in g["branch_summaries"]
                        if b["kind"] == "replay"), {})
            flags = (rep.get("replay_endpoint") or {}).get(
                "exceeds_tolerance", [])
            rows.append({
                "source_id": lab["source_id"], "task": task,
                "split": lab["split"], "decision": g["decision"],
                "slot": g["slot"], "semantic_effect": bool(sem),
                "max_task_object_dispersion_m": max_task_disp,
                "max_fixture_joint_dispersion": max_fix,
                "endpoint_flags": flags,
            })

    selected = []
    for task in TASKS:
        for split, top_k in (("train", TOP_TRAIN), ("dev", TOP_DEV)):
            pool = [r for r in rows if r["task"] == task
                    and r["split"] == split]
            pool.sort(key=lambda r: (
                not r["semantic_effect"],
                -r["max_task_object_dispersion_m"]))
            selected.extend(pool[:top_k])
    selected = selected[:CAP]

    out = {
        "schema": "v067_backlog_selection_v1", "run_schema": "v067",
        "rule": ("per task: rank (semantic_effect desc, task-object "
                 "dispersion desc); top 4 train + top 2 dev; cap 30"),
        "n_backlog_groups_scanned": len(rows),
        "n_selected": len(selected),
        "groups": [{"source_id": r["source_id"],
                    "decision": r["decision"], "slot": r["slot"]}
                   for r in selected],
        "selected_detail": selected,
        "all_backlog_labels": rows,
    }
    path = RESULTS / "v067_backlog_selection.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"scanned {len(rows)} backlog groups; selected "
          f"{len(selected)}", flush=True)
    for r in selected:
        print(f"  {r['source_id']} d={r['decision']} "
              f"sem={r['semantic_effect']} "
              f"obj={r['max_task_object_dispersion_m'] * 1000:.1f}mm "
              f"fix={r['max_fixture_joint_dispersion']:.4f}",
              flush=True)
    print(f"-> {path}", flush=True)


if __name__ == "__main__":
    main()
