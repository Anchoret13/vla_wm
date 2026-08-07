"""V7.3C — merged deterministic schedule over three anchor universes.

  v072B  the V7.2 repaired union's 96 anchors (delegated wholesale to
         lcwm.v072_schedule; existing caches reused via its index);
  v072T  the 18 RELEASED V7.2 teacher anchors (canonical-only
         semantics from stored per-branch valid_seq_canon);
  v073   the 33 V7.3B acquisition anchors (dual-goal valid_seq,
         2 GoalSpecs x 3 texts crossing, per-goal continuations,
         recovery-terminal rows).

Shared by the cache builder and the trainer — no re-derivation. One
(goal, text_variant) query per anchor visit, rotated by
(epoch + index); dev = full valid pairs. Same EPOCHS/rotation
contract as V7.2B.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch

from lcwm import v072_schedule as S72

RESULTS = Path(__file__).resolve().parent.parent / "results" \
    / "libero_loho_public_v1"
V72T = RESULTS / "2026-08-03_v072_teacher_r1"
V73 = RESULTS / "2026-08-04_v073_data_r1"
EPOCHS = 25
TASKS = S72.TASKS
ALT_GOAL = {"loho_t1_drawer": "t1_back_butter",
            "loho_t2_basket3": "t2_cheese_butter_milk",
            "loho_t3_tray": "t3_sauce_ketchup_butter",
            "loho_t4_tray": "t4_right_bowl",
            "loho_t5_drawer_cabinet": "t5_front_butter"}


def canon_goal(task):
    return S72.canonical_goal(task)


def load_universe():
    """Returns (anchors, task_queries, origins) where anchors is a
    dict anchor_id -> entry with origin in {v072B, v072T, v073}."""
    transitions = S72.load_union()
    anchors = {}
    for ak, a in S72.build_anchors(transitions).items():
        anchors[ak] = {**a, "universe": "v072B"}
    tq = S72.load_queries()

    t72 = json.loads((V72T / "anchor_manifest.json").read_text())
    for a in t72["anchors"]:
        sid, d = a["source_id"], a["decision"]
        ak = f"v072T::{sid}_d{d}"
        anchors[ak] = {
            "anchor_id": ak, "task": a["task"], "source_id": sid,
            "decision": d, "origin": "v072T", "universe": "v072T",
            "split": "train", "rows": None}

    t73 = json.loads((V73 / "anchor_manifest.json").read_text())
    for a in t73["anchors"]:
        if a["kind"] != "acq":
            continue
        sid, d = a["source_id"], a["decision"]
        ak = f"v073::{sid}_d{d}"
        anchors[ak] = {
            "anchor_id": ak, "task": a["task"], "source_id": sid,
            "decision": d, "origin": "v073", "universe": "v073",
            "split": a["role"], "rows": None}
    return anchors, tq


def queries_for(anchor, task_queries):
    """(goal_id, text_variant_id) pairs valid at this anchor."""
    task = anchor["task"]
    if anchor["universe"] == "v072B":
        return S72.queries_for(anchor, task_queries)
    if anchor["universe"] == "v072T":
        cg = canon_goal(task)
        return [(cg, f"{cg}_p0")]
    # v073: the two frozen GoalSpecs x three verified texts
    pairs = []
    for gid in (canon_goal(task), ALT_GOAL[task]):
        for pv in ("p0", "p1", "p2"):
            pairs.append((gid, f"{gid}_{pv}"))
    return pairs


def rr_schedule(anchors, split, epoch):
    by_task = defaultdict(lambda: defaultdict(list))
    for ak in sorted(anchors):
        a = anchors[ak]
        if a["split"] != split:
            continue
        by_task[a["task"]][a["source_id"]].append(ak)
    queues = []
    for task in TASKS:
        if task not in by_task:
            continue
        srcs = sorted(by_task[task])
        rot = srcs[epoch % len(srcs):] + srcs[:epoch % len(srcs)]
        merged, i = [], 0
        while any(i < len(by_task[task][s]) for s in rot):
            for s in rot:
                if i < len(by_task[task][s]):
                    merged.append(by_task[task][s][i])
            i += 1
        queues.append(merged)
    order, j = [], 0
    while any(j < len(q) for q in queues):
        for q in queues:
            if j < len(q):
                order.append(q[j])
        j += 1
    return order


def visit_query(anchors, task_queries, ak, epoch, idx):
    pairs = queries_for(anchors[ak], task_queries)
    return pairs[(epoch + idx) % len(pairs)]


def needed_visits(anchors, task_queries):
    visits = set()
    for ep in range(EPOCHS):
        order = rr_schedule(anchors, "train", ep)
        for i, ak in enumerate(order):
            g, tvv = visit_query(anchors, task_queries, ak, ep, i)
            visits.add((ak, g, tvv))
    for ak in sorted(anchors):
        if anchors[ak]["split"] != "dev":
            continue
        for g, tvv in queries_for(anchors[ak], task_queries):
            visits.add((ak, g, tvv))
    return sorted(visits)


def sources_for(universe):
    if universe == "v072T":
        return V72T / "teacher_sources"
    if universe == "v073":
        return V73 / "sources"
    raise ValueError(universe)


def shards_for(universe):
    if universe == "v072T":
        return V72T / "shards"
    if universe == "v073":
        return V73 / "shards"
    raise ValueError(universe)
