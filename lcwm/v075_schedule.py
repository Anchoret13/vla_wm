"""V7.5C — deterministic schedule over the surviving anchor universe.

Registered universe (plan_and_progress/archive/daily/2026-08-09.md V7.5). The v072B/v072T/v073 tensors
and the V7.2 shared initialization were deliberately deleted; only
their JSON/JSONL contracts survive. Two consequences, both verified
against the tree rather than assumed:

  * `v073T_released` is ALSO retired. Its 9 teacher shards survive, but
    the recurrent unroll needs `sources/<sid>.pt` from the v073 data
    root, which does not. A shard carries only
    {schema, anchor, task, transitions} -- no `rows` -- so those
    anchors cannot be unrolled and are not anchors here.
  * The universe is therefore `v074` ALONE: 24 train + 9 dev acquisition
    anchors over 22 sources, decision index 2..65. SHORTFALL and
    family_shortfall manifest rows remain budget records, never anchors.

Registered schedule change vs `lcwm/v074_schedule.py`. V7.4C's
`rr_schedule_balanced` wraps short task queues for equal per-task visit
mass, but `visit_query` was slot-independent, so a wrapped repeat
reused its anchor's `(goal, text)` pair verbatim. On this universe t4
has exactly ONE train anchor against a longest queue of 6, so t4 would
have contributed six IDENTICAL samples per epoch -- V7.3's
exposure-concentration defect relocated into the LCWM schedule.

  `rr_schedule_balanced` now yields `(anchor_key, repeat_index)` and
  `visit_query` rotates on the repeat index. Each anchor exposes
  2 GoalSpecs x 3 verified texts = 6 pairs, so t4's six visits land on
  six DISTINCT queries instead of one repeated six times. The residual
  per-anchor imbalance is real and is reported by the trainer, never
  presented as balance.

Per-task queue construction delegates to `v073_schedule.rr_schedule`
(the V7.4 process lesson: delegate, never re-derive -- a parent patch
must propagate).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from lcwm import v073_schedule as S73

RESULTS = Path(__file__).resolve().parent.parent / "results" \
    / "libero_loho_public_v1"
EPOCHS = S73.EPOCHS
TASKS = S73.TASKS
ALT_GOAL = S73.ALT_GOAL
TEXT_VARIANTS = ("p0", "p1", "p2")
UNIVERSE = "v074"


def canon_goal(task):
    return S73.canon_goal(task)


def v074_root() -> Path:
    root = None
    for p in sorted(RESULTS.glob("*_v074_data_r1")):
        root = p
    if root is None:
        raise FileNotFoundError(f"no *_v074_data_r1 root under {RESULTS}")
    return root


def load_universe():
    """(anchors, task_queries) over the surviving v074 tranche.

    Every anchor is asserted loadable -- both its source and its shard
    must exist -- so a deleted tensor surfaces here as a loud failure
    rather than as a silently smaller training set.
    """
    root = v074_root()
    src_dir, shard_dir = root / "sources", root / "shards"
    manifest = json.loads((root / "anchor_manifest.json").read_text())

    anchors = {}
    for a in manifest["anchors"]:
        if a["kind"] != "acq":        # SHORTFALL / family_shortfall
            continue
        sid, d = a["source_id"], a["decision"]
        src, shard = src_dir / f"{sid}.pt", shard_dir / f"{sid}_d{d}.pt"
        assert src.exists(), f"missing source for {sid}_d{d}: {src}"
        assert shard.exists(), f"missing shard for {sid}_d{d}: {shard}"
        ak = f"v074::{sid}_d{d}"
        anchors[ak] = {
            "anchor_id": ak, "task": a["task"], "source_id": sid,
            "decision": d, "origin": UNIVERSE, "universe": UNIVERSE,
            "split": a["role"], "rows": None}
    assert anchors, f"no loadable acq anchors under {root}"
    return anchors, S73.S72.load_queries()


def queries_for(anchor, task_queries=None):
    """(goal_id, text_variant_id) pairs: 2 GoalSpecs x 3 verified
    texts, the v073/v074 crossing, unchanged."""
    del task_queries
    task = anchor["task"]
    return [(gid, f"{gid}_{pv}")
            for gid in (canon_goal(task), ALT_GOAL[task])
            for pv in TEXT_VARIANTS]


def _task_queues(anchors, split, epoch):
    """Per-task queues recovered from the delegated merged order. The
    round-robin interleave preserves per-task relative order, so
    grouping its flat output by task inverts the construction exactly."""
    by_task = defaultdict(list)
    for ak in S73.rr_schedule(anchors, split, epoch):
        by_task[anchors[ak]["task"]].append(ak)
    return [by_task[t] for t in TASKS if t in by_task]


def rr_schedule_balanced(anchors, split, epoch):
    """[(anchor_key, repeat_index)] with equal per-task visit mass.

    Short queues wrap cyclically; `repeat_index` counts prior
    occurrences of that anchor within THIS epoch and drives the query
    rotation, so a wrapped repeat is a different query rather than a
    duplicated sample.
    """
    queues = _task_queues(anchors, split, epoch)
    if not queues:
        return []
    length = max(len(q) for q in queues)
    order, seen = [], defaultdict(int)
    for j in range(length):
        for q in queues:
            ak = q[j % len(q)]
            order.append((ak, seen[ak]))
            seen[ak] += 1
    return order


def visit_query(anchors, task_queries, ak, epoch, rep):
    """(epoch + anchor_hash + repeat_index) mod len -- the v073 rule
    plus the registered repeat rotation."""
    pairs = queries_for(anchors[ak], task_queries)
    return pairs[(epoch + S73._ak_off(ak) + int(rep)) % len(pairs)]


def needed_visits(anchors, task_queries):
    """Every (anchor, goal, tvid) the frozen budget touches: train over
    EPOCHS balanced epochs, dev over its full valid pairs."""
    visits = set()
    for ep in range(EPOCHS):
        for ak, rep in rr_schedule_balanced(anchors, "train", ep):
            visits.add((ak,) + visit_query(anchors, task_queries,
                                           ak, ep, rep))
    for ak in sorted(anchors):
        if anchors[ak]["split"] != "dev":
            continue
        for g, tvv in queries_for(anchors[ak], task_queries):
            visits.add((ak, g, tvv))
    return sorted(visits)


def sources_for(universe):
    assert universe == UNIVERSE, f"retired universe: {universe}"
    return v074_root() / "sources"


def shards_for(universe):
    assert universe == UNIVERSE, f"retired universe: {universe}"
    return v074_root() / "shards"


def exposure_census(anchors, split="train"):
    """Per-task visit mass and per-anchor visit counts over the frozen
    epoch budget. The trainer reports this instead of asserting that
    the schedule is balanced at the anchor level -- it is not."""
    per_task = defaultdict(int)
    per_anchor = defaultdict(int)
    for ep in range(EPOCHS):
        for ak, _ in rr_schedule_balanced(anchors, split, ep):
            per_task[anchors[ak]["task"]] += 1
            per_anchor[ak] += 1
    return dict(per_task), dict(per_anchor)


if __name__ == "__main__":
    from collections import Counter

    anchors, tq = load_universe()
    train = sorted(a for a in anchors if anchors[a]["split"] == "train")
    dev = sorted(a for a in anchors if anchors[a]["split"] == "dev")
    print(f"universe: {len(train)} train + {len(dev)} dev anchors, "
          f"{len({anchors[a]['source_id'] for a in anchors})} sources")

    per_task_counts = Counter(anchors[a]["task"] for a in train)
    peak = max(per_task_counts.values())
    for ep in range(EPOCHS):
        order = rr_schedule_balanced(anchors, "train", ep)
        assert order == rr_schedule_balanced(anchors, "train", ep), \
            f"schedule not deterministic at epoch {ep}"
        by_task = defaultdict(list)
        for ak, rep in order:
            by_task[anchors[ak]["task"]].append((ak, rep))
        assert set(by_task) == set(per_task_counts), ep
        for t, visits in by_task.items():
            assert len(visits) == peak, (ep, t, len(visits))
            # repeat indices are dense per anchor within the epoch
            for ak, reps in defaultdict(
                    list, {k: [r for a_, r in visits if a_ == k]
                           for k, _ in visits}).items():
                assert sorted(reps) == list(range(len(reps))), (ep, ak)
        # the registered point of the change: a wrapped repeat is a
        # DIFFERENT query, never the same sample twice
        qs = defaultdict(set)
        for ak, rep in order:
            qs[ak].add(visit_query(anchors, tq, ak, ep, rep))
        for ak, seen in qs.items():
            n_rep = sum(1 for a_, _ in order if a_ == ak)
            assert len(seen) == min(n_rep, len(queries_for(anchors[ak]))), \
                (ep, ak, n_rep, len(seen))

    nv = needed_visits(anchors, tq)
    assert nv == needed_visits(anchors, tq) and nv == sorted(set(nv))
    pt, pa = exposure_census(anchors)
    print(f"per-task visit mass over {EPOCHS} epochs: {pt}")
    print(f"per-anchor visits: min={min(pa.values())} "
          f"max={max(pa.values())} (imbalance is REPORTED, not asserted "
          f"away)")
    print(f"needed visits: {len(nv)}")
    print("self-test OK")
