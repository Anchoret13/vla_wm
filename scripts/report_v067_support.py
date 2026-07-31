#!/usr/bin/env python
"""V6.7.2 — clean support report over the rebuilt (v067) crossed
outcomes. A routing fact, not a model-performance gate.

Registered label definitions (2026-07-31 execution record):
- physical_effect     any candidate-pair object-position difference at
                      the branch endpoint (> frozen replay tol = 0.0);
                      NUMERICAL ONLY — kept for null calibration, never
                      support;
- task_object_effect  candidate-pair dispersion of a GOAL-REFERENCED
                      object >= 5 mm absolute floor (the registered
                      slot-1 floor), reported with its magnitude;
- semantic_effect     any candidate pair disagrees on immediate valid
                      bits, event bits, or window flips for ANY
                      registered GoalSpec (discrete, tol 0);
- policy_rankable     >= 1 non-tied paired preference among candidate
                      pairs under the canonical goal (both-repeats rule,
                      FROZEN outcome tolerances, see below);
- reference_improving some candidate i != 0 with A_i0 = +1 (canonical);
- support_improving   u_support with A_support,0 = +1 (canonical);
- history_contrast    matched-frame pairs across decisions: eef_pos
                      <= 5 mm, gripper <= 0.02, all tracked objects
                      <= 5 mm, equal current valid bits, DIFFERENT flip
                      history (damage/recovery) or event steps.

Outcome tolerances are frozen HERE, from the replay-audit continuations
(candidate-0 exact replay under identical restored state + CRN noise):
- discrete components (success_by_100, neg_damage): tolerance 0; any
  replay disagreement flags the group `replay_unstable` (excluded from
  teacher support);
- continuous components (p_valid_100, q_valid_mean, neg_tau_next):
  tolerance = 95th percentile of |replay - cand0| across all groups x
  repeats (tranche-A convention).

Output: results/libero_loho_public_v1/v067_support_report.json
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATA = Path("/home/stargazer/Desktop/vla_wm/datasets/libero_loho_public_v1"
            "/v06_effect_crossed")
RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
TASKS = ["loho_t1_drawer", "loho_t2_basket3", "loho_t3_tray",
         "loho_t4_tray", "loho_t5_drawer_cabinet"]
ABS_FLOOR_MM = 0.005
HC_EEF = 0.005
HC_GRIP = 0.02
HC_OBJ = 0.005
CONTINUOUS = ("p_valid_100", "q_valid_mean", "neg_tau_next")
DISCRETE = ("success_by_100", "neg_damage")


def goal_objects(subgoals: list[str]) -> set[str]:
    out = set()
    for sg in subgoals:
        parts = sg.split()
        if parts[0] in ("place", "pick_up"):
            out.add(parts[1])
    return out


def outcomes_by(records, branch_key, gid):
    reps = [r for r in records
            if r["branch_key"] == branch_key and r["goal_spec_id"] == gid]
    return [r["outcome"] for r in sorted(reps, key=lambda r: r["repeat"])]


def main() -> None:
    from lcwm.task_automaton import paired_preference
    from lcwm.v067_lineage import load_v067

    goal_manifest = json.loads(
        (RESULTS / "goal_spec_manifest_v067.json").read_text())
    files = sorted((DATA / "continuations_v067").glob("*.pt"))
    assert files, "no v067 continuation groups found"
    groups = [load_v067(p, "continuations") for p in files]
    print(f"loaded {len(groups)} v067 groups", flush=True)

    # ---- freeze outcome tolerances from replay audits -------------------
    deltas = {k: [] for k in CONTINUOUS + DISCRETE}
    per_group_replay = {}
    for g in groups:
        canon = g["canonical_goal_spec_id"]
        y_c0 = outcomes_by(g["records"], 0, canon)
        y_rep = outcomes_by(g["records"], "replay_audit", canon)
        gid_key = f"{g['source_id']}_d{g['decision']}"
        rep_branch = next((b for b in g["branch_summaries"]
                           if b["kind"] == "replay"), None)
        endpoint = (rep_branch or {}).get("replay_endpoint")
        entry = {"endpoint_exceeds":
                 list(endpoint["exceeds_tolerance"]) if endpoint else [],
                 "endpoint_deltas":
                 endpoint["deltas"] if endpoint else None,
                 "outcome_deltas": None}
        if y_rep:
            dd = {}
            for k in CONTINUOUS + DISCRETE:
                dd[k] = max(abs(float(a[k]) - float(b[k]))
                            for a, b in zip(y_c0, y_rep))
                deltas[k].append(dd[k])
            entry["outcome_deltas"] = dd
        per_group_replay[gid_key] = entry
    tolerances = {k: float(np.percentile(deltas[k], 95))
                  if deltas[k] else 0.0 for k in CONTINUOUS}
    for k in DISCRETE:
        tolerances[k] = 0.0
    # replay_unstable: endpoint exceeds a frozen tranche-A tolerance
    # channel (incl. valid_bits) OR a discrete outcome component flips
    # between candidate 0 and its exact replay
    replay_unstable = {
        k: v for k, v in per_group_replay.items()
        if v["endpoint_exceeds"]
        or (v["outcome_deltas"]
            and any(v["outcome_deltas"][c] > 0 for c in DISCRETE))}
    print(f"frozen outcome tolerances: {tolerances}", flush=True)
    print(f"replay_unstable groups: {sorted(replay_unstable)}", flush=True)

    # ---- per-group labels ----------------------------------------------
    rows = []
    for g in groups:
        task = g["task"]
        canon = g["canonical_goal_spec_id"]
        specs = goal_manifest["tasks"][task]["goal_specs"]
        gid_key = f"{g['source_id']}_d{g['decision']}"
        cands = [b for b in g["branch_summaries"]
                 if b["kind"] == "candidate"]
        cand_pairs = list(itertools.combinations(range(len(cands)), 2))
        fixture_addrs = {
            name: addr for name, addr in
            g.get("qpos_joint_map", {}).items()
            if not name.startswith(("robot0", "gripper0"))
            and not name.endswith("_joint0")}

        # physical / task-object dispersion at branch endpoints
        objs = sorted(goal_objects(
            list(itertools.chain.from_iterable(
                specs[gid]["ordered_subgoals"] for gid in g["goals"]))))
        body_names = list(
            g["branch_summaries"][0]["grasp_after"]["grasped"])
        max_disp, max_task_disp, max_fixture_disp = 0.0, 0.0, 0.0
        for i, j in cand_pairs:
            d_all = np.abs(np.asarray(cands[i]["obj_after"])
                           - np.asarray(cands[j]["obj_after"]))
            max_disp = max(max_disp, float(d_all.max()))
            for bi, name in enumerate(body_names):
                if name in objs and bi < d_all.shape[0]:
                    max_task_disp = max(max_task_disp,
                                        float(d_all[bi].max()))
            if fixture_addrs and "qpos_after" in cands[i]:
                qa = np.asarray(cands[i]["qpos_after"])
                qb = np.asarray(cands[j]["qpos_after"])
                for addr in fixture_addrs.values():
                    max_fixture_disp = max(
                        max_fixture_disp, float(abs(qa[addr] - qb[addr])))
        physical_effect = max_disp > 0.0 or max_fixture_disp > 0.0
        task_object_effect = max_task_disp >= ABS_FLOOR_MM

        # semantic effect: any candidate pair, any goal, discrete
        semantic_effect = False
        sem_detail = []
        for i, j in cand_pairs:
            for gid in g["goals"]:
                a = cands[i]["immediate"][gid]
                b = cands[j]["immediate"][gid]
                if (a["valid_after"] != b["valid_after"]
                        or a["events_after"] != b["events_after"]
                        or bool(a["flips_10"]) != bool(b["flips_10"])
                        or a["terminal_now"] != b["terminal_now"]):
                    semantic_effect = True
                    sem_detail.append((i, j, gid))

        # paired preferences under the canonical goal, frozen tolerances
        y = {b["candidate"]: outcomes_by(g["records"], b["candidate"],
                                         canon) for b in cands}
        y_sup = outcomes_by(g["records"], 4, canon)
        nontied, ref_improving = [], []
        for i, j in cand_pairs:
            a_ij = paired_preference(y[i], y[j], tolerances)
            if a_ij != 0:
                nontied.append((i, j, a_ij))
        for i in y:
            if i == 0:
                continue
            if paired_preference(y[i], y[0], tolerances) == 1:
                ref_improving.append(i)
        support_improving = bool(
            y_sup and paired_preference(y_sup, y[0], tolerances) == 1)

        # cross-goal disagreement in continuation outcomes
        cross_goal = False
        for b in cands:
            outs = {gid: outcomes_by(g["records"], b["candidate"], gid)
                    for gid in g["goals"]}
            vals = {gid: (o[0]["success_by_100"],
                          round(o[0]["q_valid_mean"], 3))
                    for gid, o in outs.items() if o}
            if len(set(vals.values())) > 1:
                cross_goal = True
        rows.append({
            "group": gid_key, "task": task,
            "source_id": g["source_id"], "decision": g["decision"],
            "slot": g["slot"], "split": g["split"],
            "provenance_source": g["provenance_source"],
            "n_valid_before": sum(
                g["auto_states_snapshot"][canon]["prev_valid"]),
            "physical_effect": bool(physical_effect),
            "max_dispersion_m": max_disp,
            "task_object_effect": bool(task_object_effect),
            "max_task_object_dispersion_m": max_task_disp,
            "max_fixture_joint_dispersion": max_fixture_disp,
            "semantic_effect": bool(semantic_effect),
            "semantic_detail": sem_detail[:8],
            "policy_rankable": bool(nontied),
            "nontied_pairs": nontied,
            "reference_improving": ref_improving,
            "support_improving": support_improving,
            "cross_goal_disagreement": bool(cross_goal),
            "replay_unstable": gid_key in replay_unstable,
        })

    # ---- history contrasts across the full source bank ------------------
    hc_pairs = []
    label_files = sorted(
        (DATA / "semantic_relabels_v067").glob("*.pt"))
    per_task_rows: dict[str, list] = {}
    for p in label_files:
        lab = load_v067(p, "semantic_labels")
        src = torch.load(DATA / "sources" / f"{lab['source_id']}.pt",
                         weights_only=False)
        canon = lab["canonical_goal_spec_id"]
        for dec, row in zip(lab["per_decision"], src["rows"]):
            q = np.asarray(row["q"], dtype=np.float64)
            entry = dec["goals"][canon]
            per_task_rows.setdefault(lab["task"], []).append({
                "source_id": lab["source_id"], "split": lab["split"],
                "decision": dec["decision"],
                "eef": q[:3], "grip": q[7:],
                "obj": np.asarray(row["obj_before"]),
                "valid": tuple(entry["valid_before"]),
                "events": tuple(entry["events_before"]),
                "flips10": tuple(tuple(f) for f in
                                 entry["state_before"]["flips"]
                                 if f[2] == -1),
            })
    for task, items in per_task_rows.items():
        for a, b in itertools.combinations(items, 2):
            if a["obj"].shape != b["obj"].shape:
                continue
            if (np.abs(a["eef"] - b["eef"]).max() <= HC_EEF
                    and np.abs(a["grip"] - b["grip"]).max() <= HC_GRIP
                    and np.abs(a["obj"] - b["obj"]).max() <= HC_OBJ
                    and a["valid"] == b["valid"]
                    and (a["events"] != b["events"]
                         or bool(a["flips10"]) != bool(b["flips10"]))):
                hc_pairs.append({
                    "task": task,
                    "a": (a["source_id"], a["decision"]),
                    "b": (b["source_id"], b["decision"]),
                    "events_a": a["events"], "events_b": b["events"],
                    "n_flips10_a": len(a["flips10"]),
                    "n_flips10_b": len(b["flips10"]),
                })

    # ---- aggregation + V6.8 triggers ------------------------------------
    def agg(pred, split=None):
        out = {}
        for t in TASKS:
            sel = [r for r in rows if r["task"] == t
                   and (split is None or r["split"] == split)]
            out[t] = sum(1 for r in sel if pred(r))
        return out

    clean = [r for r in rows if not r["replay_unstable"]]
    summary = {
        "n_groups": len(rows),
        "n_replay_unstable": sum(r["replay_unstable"] for r in rows),
        "physical_effect_by_task": agg(lambda r: r["physical_effect"]),
        "task_object_effect_by_task":
            agg(lambda r: r["task_object_effect"]),
        "semantic_effect_by_task": agg(lambda r: r["semantic_effect"]),
        "policy_rankable_by_task": agg(lambda r: r["policy_rankable"]),
        "reference_improving_by_task_train":
            agg(lambda r: bool(r["reference_improving"])
                and not r["replay_unstable"], split="train"),
        "support_improving_by_task":
            agg(lambda r: r["support_improving"]),
        "cross_goal_by_task":
            agg(lambda r: r["cross_goal_disagreement"]),
        "dev_nontied_groups": sorted(
            r["group"] for r in clean
            if r["split"] == "dev" and r["policy_rankable"]),
        "dev_nontied_sources": sorted({
            r["source_id"] for r in clean
            if r["split"] == "dev" and r["policy_rankable"]}),
        "history_contrast_pairs": len(hc_pairs),
        "max_task_object_dispersion_by_group": {
            r["group"]: round(r["max_task_object_dispersion_m"], 5)
            for r in rows},
    }
    ref_by_task = summary["reference_improving_by_task_train"]
    triggers = {
        "task_without_reference_improving_train_group": sorted(
            t for t in TASKS if ref_by_task[t] == 0),
        "dev_calibration_single_group":
            len(summary["dev_nontied_groups"]) <= 1,
        "support_improves_but_policy_does_not": sorted(
            r["group"] for r in clean if r["support_improving"]
            and not r["reference_improving"]),
        "history_contrast_unsupported": len(hc_pairs) == 0,
    }
    report = {
        "schema": "v067_support_report_v1", "run_schema": "v067",
        "frozen_outcome_tolerances": tolerances,
        "replay_delta_distributions": {
            k: {"max": float(max(v)) if v else None,
                "p95": float(np.percentile(v, 95)) if v else None,
                "n": len(v)} for k, v in deltas.items()},
        "replay_unstable_groups": {
            k: v for k, v in replay_unstable.items()},
        "abs_floor_m": ABS_FLOOR_MM,
        "history_contrast_thresholds": {
            "eef": HC_EEF, "grip": HC_GRIP, "obj": HC_OBJ},
        "groups": rows,
        "history_contrasts": hc_pairs,
        "summary": summary,
        "v68_triggers": triggers,
    }
    out = RESULTS / "v067_support_report.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True,
                              default=str))
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(json.dumps(triggers, indent=2, sort_keys=True), flush=True)
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
