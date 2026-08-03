"""V7.2B — deterministic visit schedule shared by the h-cache builder
and the three matched loss arms.

One anchor visit per schedule slot; exactly ONE (goal, text_variant)
query per visit, rotated by (epoch + anchor index) so language
replication never multiplies a physical anchor's mass. The cache
builder enumerates exactly the (observation, text) keys this schedule
will touch over the frozen epoch budget; the trainer asserts every key
exists. Both sides import THIS module — no independent re-derivation.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch

RESULTS = Path(__file__).resolve().parent.parent / "results" \
    / "libero_loho_public_v1"
DATA_R1 = RESULTS / "2026-08-03_v072_data_r1"
EPOCHS = 25
TASKS = ["loho_t1_drawer", "loho_t2_basket3", "loho_t3_tray",
         "loho_t4_tray", "loho_t5_drawer_cabinet"]


def load_union():
    return torch.load(DATA_R1 / "physical_transitions"
                      / "transitions.pt",
                      weights_only=False)["transitions"]


def load_queries():
    q = defaultdict(list)
    for line in (DATA_R1 / "goal_queries.jsonl").open():
        row = json.loads(line)
        q[row["task"]].append(row)
    for t in q:
        q[t].sort(key=lambda r: r["text_variant_id"])
    return q


def anchor_key(rec) -> str:
    if rec["origin"] == "v071_semwin":
        return "sw::" + rec["raw_key"].split("|")[0]
    return f"{rec['origin']}::{rec['source_id']}_d{rec['decision']}"


def build_anchors(transitions):
    """Group non-audit rows into anchor entries (semwin windows are one
    anchor each; their segments chain inside the visit)."""
    anchors = {}
    for rec in transitions:
        if rec["audit_only"]:
            continue
        ak = anchor_key(rec)
        a = anchors.setdefault(ak, {
            "anchor_id": ak, "task": rec["task"],
            "source_id": rec["source_id"],
            "decision": rec["decision"], "origin": rec["origin"],
            "split": rec["split"], "rows": []})
        a["rows"].append(rec["pt_id"])
    for a in anchors.values():
        a["rows"].sort()
    return anchors


def queries_for(anchor, task_queries):
    """Query (goal_id, text_variant_id) pairs valid at this anchor.
    Full crossed set only where exact per-goal semantics were stored
    (v071 union r1 relabels, semwin valid_seq); canonical p0 only for
    selector/acquire rows."""
    full = anchor["origin"] == "v071_semwin" or (
        anchor["origin"] == "v071_union"
        and "acquire" not in anchor["source_id"])
    rows = task_queries[anchor["task"]]
    if full:
        return [(r["goal_id"], r["text_variant_id"]) for r in rows]
    return [(r["goal_id"], r["text_variant_id"]) for r in rows
            if r["role"] == "canonical"
            and r["goal_id"] == canonical_goal(anchor["task"])]


_goal_manifest = None


def canonical_goal(task):
    global _goal_manifest
    if _goal_manifest is None:
        _goal_manifest = json.loads(
            (RESULTS / "goal_spec_manifest_v067.json").read_text())
    return _goal_manifest["tasks"][task]["canonical_goal_spec_id"]


def rr_schedule(anchors, split, epoch):
    """task -> source -> anchor round-robin, source rotation by epoch
    (the registered 2a hierarchy), deterministic."""
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


def text_id(goal_id, tvid, canonical_p0_as="canon"):
    """Cache text identity. Canonical p0 == the behavior prompt ==
    the v071 canonical cache namespace."""
    if tvid.endswith("_p0") and goal_id == tvid[:-3]:
        # p0 of any goal is that goal's registered language; only the
        # TASK-canonical goal's p0 equals the behavior prompt
        pass
    return tvid


def needed_visits(anchors, task_queries, splits=("train", "dev")):
    """Every (anchor, goal, tvid) pair the frozen budget will touch:
    train = 25-epoch rotation; dev = ALL valid pairs (full dev
    coverage at eval, as registered in 2a/2c)."""
    visits = set()
    for split in splits:
        if split == "train":
            for ep in range(EPOCHS):
                order = rr_schedule(anchors, "train", ep)
                for i, ak in enumerate(order):
                    g, tv = visit_query(anchors, task_queries, ak,
                                        ep, i)
                    visits.add((ak, g, tv))
        else:
            for ak in sorted(anchors):
                if anchors[ak]["split"] != "dev":
                    continue
                for g, tv in queries_for(anchors[ak], task_queries):
                    visits.add((ak, g, tv))
    return sorted(visits)
