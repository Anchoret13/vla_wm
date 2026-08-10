"""V7.4C — merged deterministic schedule over five anchor universes.

Extends lcwm.v073_schedule (never forks it — 2026-08-09.md V7.4C):

  v072B/v072T/v073  the V7.3 merged universe, delegated wholesale;
  v074   the V7.4B acquisition anchors (2500 seed family; shard
         layout identical to v073). Manifest SHORTFALL and
         family_shortfall rows are budget records, not anchors —
         tolerated and skipped, never silently refilled;
  v073T_released  the 9 executed V7.3 teacher-assessment anchors
         (214 sealed continuations), released to ordinary
         provenance-labeled training per the 2026-08-03 release
         rule. Canonical-p0 queries only — a design choice following
         the v072T precedent, not a data constraint (the shards do
         store per-branch valid_seq_alt alongside valid_seq_canon);
         candidate chunks live in candidate_chunks.pt.

Registered exposure change: rr_schedule_balanced wraps short task
queues so every epoch delivers EQUAL per-task visit mass (V7.3C's
700/725/550/425/625 imbalance is the defect repaired). Source
rotation and visit_query rotation are unchanged; a wrapped repeat
visit reuses its anchor's (goal, text) pair because visit_query is
slot-independent by construction.

hcache key contract (build_v074_hcache.py): unroll keys
{tvv}::src::{sid}::d{dd}; next-obs keys {tvv}::next::{pt} with
pt = "{universe}::{transition_id}" for v073/v074/v073T_released.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from lcwm import v073_schedule as S73

RESULTS = Path(__file__).resolve().parent.parent / "results" \
    / "libero_loho_public_v1"
V73TEACH = RESULTS / "2026-08-07_v073_teacher_r1"
EPOCHS = S73.EPOCHS
TASKS = S73.TASKS
ALT_GOAL = S73.ALT_GOAL


def canon_goal(task):
    return S73.canon_goal(task)


def v074_root() -> Path:
    """Latest *_v074_data_r1 root by sorted glob (the build_v074_data
    discovery rule)."""
    root = None
    for p in sorted(RESULTS.glob("*_v074_data_r1")):
        root = p
    if root is None:
        raise FileNotFoundError("no *_v074_data_r1 root under "
                                f"{RESULTS}")
    return root


def load_universe():
    """(anchors, task_queries): the V7.3 merged universe plus the
    V7.4B tranche plus the released V7.3 teacher-assessment anchors,
    universe in {v072B, v072T, v073, v074, v073T_released}."""
    anchors, tq = S73.load_universe()

    t74 = json.loads(
        (v074_root() / "anchor_manifest.json").read_text())
    for a in t74["anchors"]:
        if a["kind"] != "acq":     # SHORTFALL / family_shortfall
            continue
        sid, d = a["source_id"], a["decision"]
        ak = f"v074::{sid}_d{d}"
        anchors[ak] = {
            "anchor_id": ak, "task": a["task"], "source_id": sid,
            "decision": d, "origin": "v074", "universe": "v074",
            "split": a["role"], "rows": None}

    t73 = json.loads(
        (S73.V73 / "anchor_manifest.json").read_text())
    for a in t73["anchors"]:
        if a["kind"] != "teach":
            continue
        sid, d = a["source_id"], a["decision"]
        shard = V73TEACH / "shards" / f"{sid}_d{d}.pt"
        assert shard.exists(), f"released shard missing: {shard}"
        ak = f"v073T_released::{sid}_d{d}"
        anchors[ak] = {
            "anchor_id": ak, "task": a["task"], "source_id": sid,
            "decision": d, "origin": "v073T_released",
            "universe": "v073T_released", "split": "train",
            "rows": None}
    return anchors, tq


def queries_for(anchor, task_queries):
    """(goal_id, text_variant_id) pairs valid at this anchor. v074
    inherits the v073 crossing (2 GoalSpecs x 3 verified texts);
    the released assessment anchors are canonical-p0 only — a
    design choice following the v072T release precedent, not a data
    constraint: their shards store valid_seq_alt alongside
    valid_seq_canon, but only the canonical branch is queried."""
    u = anchor["universe"]
    if u == "v074":
        pairs = []
        for gid in (canon_goal(anchor["task"]),
                    ALT_GOAL[anchor["task"]]):
            for pv in ("p0", "p1", "p2"):
                pairs.append((gid, f"{gid}_{pv}"))
        return pairs
    if u == "v073T_released":
        cg = canon_goal(anchor["task"])
        return [(cg, f"{cg}_p0")]
    return S73.queries_for(anchor, task_queries)


def _task_queues(anchors, split, epoch):
    """Per-task anchor queues recovered from S73.rr_schedule's
    merged order (delegation, never a re-derived copy — a parent
    patch propagates here). The round-robin interleave preserves
    per-task relative order, so grouping its flat output by task is
    an exact inverse of the per-task queue construction."""
    by_task = defaultdict(list)
    for ak in S73.rr_schedule(anchors, split, epoch):
        by_task[anchors[ak]["task"]].append(ak)
    return [by_task[t] for t in TASKS if t in by_task]


def rr_schedule_balanced(anchors, split, epoch):
    """The registered V7.4C schedule: v073 rr_schedule task queues,
    with SHORT queues wrapped cyclically so every task contributes
    max-queue-length visits per epoch (equal per-task visit mass;
    per-anchor visits within a task differ by at most one)."""
    queues = _task_queues(anchors, split, epoch)
    if not queues:
        return []
    length = max(len(q) for q in queues)
    order = []
    for j in range(length):
        for q in queues:
            order.append(q[j % len(q)])
    return order


def visit_query(anchors, task_queries, ak, epoch, idx):
    """v073 rule verbatim — (epoch + anchor_hash) mod len, slot
    independent — over THIS module's queries_for (v073's would miss
    the v074 / released universes)."""
    del idx
    pairs = queries_for(anchors[ak], task_queries)
    return pairs[(epoch + S73._ak_off(ak)) % len(pairs)]


def needed_visits(anchors, task_queries):
    """Every (anchor, goal, tvid) the frozen budget touches: train =
    25 balanced epochs (wrapped repeats add no new tuples because
    visit_query is slot-independent); dev = full valid pairs."""
    visits = set()
    for ep in range(EPOCHS):
        order = rr_schedule_balanced(anchors, "train", ep)
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
    if universe == "v074":
        return v074_root() / "sources"
    if universe == "v073T_released":
        # released teacher sources live in the V7.3B data root
        return S73.V73 / "sources"
    return S73.sources_for(universe)


def shards_for(universe):
    if universe == "v074":
        return v074_root() / "shards"
    if universe == "v073T_released":
        return V73TEACH / "shards"
    return S73.shards_for(universe)


if __name__ == "__main__":
    # CPU-only self-test on an injected synthetic universe. The
    # goal manifest is monkeypatched so no results-tree file is
    # read; only the v074 / v073T_released queries_for branches are
    # exercised (v072B/v073 delegate to real caches by design).
    from collections import Counter

    from lcwm import v072_schedule as S72
    S72._goal_manifest = {
        "tasks": {t: {"canonical_goal_spec_id": f"cg_{t}"}
                  for t in TASKS}}

    counts = dict(zip(TASKS, (4, 5, 2, 1, 3)))
    anchors = {}
    for ti, (t, n) in enumerate(counts.items()):
        for k in range(n):
            u = "v073T_released" if k == 0 and n >= 3 else "v074"
            sid = f"s{ti}{k % 2}"
            ak = f"{u}::{t}_{sid}_d{k}"
            anchors[ak] = {
                "anchor_id": ak, "task": t, "source_id": sid,
                "decision": k, "origin": u, "universe": u,
                "split": "train", "rows": None}
    for t in TASKS[:2]:
        ak = f"v074::{t}_dev_d0"
        anchors[ak] = {
            "anchor_id": ak, "task": t, "source_id": "sdev",
            "decision": 0, "origin": "v074", "universe": "v074",
            "split": "dev", "rows": None}

    train = sorted(ak for ak in anchors
                   if anchors[ak]["split"] == "train")
    peak = max(counts.values())
    sched_visits = set()
    for ep in range(EPOCHS):
        order = rr_schedule_balanced(anchors, "train", ep)
        assert order == rr_schedule_balanced(anchors, "train", ep)
        per_task = defaultdict(list)
        for ak in order:
            per_task[anchors[ak]["task"]].append(ak)
        assert set(per_task) == set(TASKS)
        for t, visits in per_task.items():
            assert len(visits) == peak, (ep, t, len(visits))
            per_anchor = Counter(visits)
            assert set(per_anchor) == {
                ak for ak in train if anchors[ak]["task"] == t}
            assert (max(per_anchor.values())
                    - min(per_anchor.values())) <= 1, (ep, t)
        for ak in order:
            sched_visits.add(
                (ak,) + visit_query(anchors, None, ak, ep, 0))
    for ak in train:
        seen = {visit_query(anchors, None, ak, ep, 0)
                for ep in range(EPOCHS)}
        assert seen == set(queries_for(anchors[ak], None)), ak
    nv = needed_visits(anchors, None)
    assert nv == needed_visits(anchors, None)
    assert nv == sorted(set(nv))
    dev_visits = set()
    for ak in sorted(anchors):
        if anchors[ak]["split"] != "dev":
            continue
        for g, tvv in queries_for(anchors[ak], None):
            dev_visits.add((ak, g, tvv))
    assert sched_visits | dev_visits == set(nv)
    print(f"self-test OK: {len(train)} train anchors, "
          f"{len(dev_visits)} dev pairs, "
          f"{len(nv)} needed visits over {EPOCHS} epochs")
