"""V7.4D — matched policy schedule with anchor-level exposure caps.

Repairs the V7.3E exposure defect: that model stream cycled the
ELIGIBLE-only key list (one eligible anchor -> all 300 visits) while
grounded anchors saw 30-60. Here the grounded and model streams are
both built task -> source -> anchor round-robin (the
lcwm/v073_schedule.py::rr_schedule construction) over their FULL
anchor sets — abstentions included as (ak, active=False)
zero-model-loss cells — with wrap-around cycling so per-task mass
stays equal to +-1 for the whole run and no anchor can exceed
ceil(task_slots / n_task_anchors) visits. Demo and retention streams
keep the train_v073_policy.py interleave (retention offset by half
the row list). Pure round-robin, no RNG; `rid` is recorded only.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict

SCHEMA = "v074_policy_sched_v1"
TEACHER_MODES = {"model_teacher", "abstain"}


def _demo_key(row):
    """JSON-safe histogram key for a (task_id, demo, row) triple —
    the same encoding train_v073_policy.py uses for demo_pools."""
    return json.dumps(list(row))


def _task_queue(by_source):
    """Source-interleaved per-task queue (rr_schedule merge, epoch
    rotation fixed at 0: this schedule is built exactly once)."""
    srcs = sorted(by_source)
    merged, i = [], 0
    while any(i < len(by_source[s]) for s in srcs):
        for s in srcs:
            if i < len(by_source[s]):
                merged.append(by_source[s][i])
        i += 1
    return merged


def _cycled_stream(anchors, steps, name):
    """Task-interleaved stream over CYCLING per-task queues.

    rr_schedule stops when its queues empty; this wraps each queue
    (q[pos % len(q)]) so the stream reaches exactly `steps` and every
    task keeps equal presence throughout. Tasks in sorted order, as
    in the train_v073_policy.py grounded loop.
    """
    if not anchors:
        raise ValueError(f"{name}: empty anchor set")
    by_task = defaultdict(lambda: defaultdict(list))
    for ak in sorted(anchors):
        a = anchors[ak]
        by_task[a["task"]][a["source_id"]].append(ak)
    tasks = sorted(by_task)
    queues = {t: _task_queue(by_task[t]) for t in tasks}
    pos = {t: 0 for t in tasks}
    order = []
    while len(order) < steps:
        for t in tasks:
            q = queues[t]
            order.append(q[pos[t] % len(q)])
            pos[t] += 1
            if len(order) >= steps:
                break
    return order, queues


def _assert_caps(order, queues, anchors, name):
    """Registered exposure invariants: per-anchor visit count <=
    ceil(task_slots / n_task_anchors); per-task mass equal +-1.
    The histogram lists EVERY anchor, zero-visit ones included."""
    hist = Counter(order)
    task_slots = Counter(anchors[ak]["task"] for ak in order)
    for t, q in queues.items():
        cap = math.ceil(task_slots[t] / len(q)) if task_slots[t] \
            else 0
        for ak in q:
            assert hist[ak] <= cap, (
                f"{name}: anchor {ak} visited {hist[ak]}x, cap "
                f"{cap} (task {t}: {task_slots[t]} slots / "
                f"{len(q)} anchors)")
    masses = sorted(task_slots[t] for t in queues)
    assert masses[-1] - masses[0] <= 1, (
        f"{name}: per-task mass imbalance {dict(task_slots)}")
    return {ak: hist.get(ak, 0) for ak in sorted(anchors)}


def build_matched_policy_schedule(grounded_anchors, teacher_anchors,
                                  demo_rows, steps, rid):
    """Build the V7.4D matched schedule consumed by all four arms.

    grounded_anchors: dict ak -> {task, source_id, ...}
    teacher_anchors:  dict ak -> {task, source_id, mode, ...} — the
        FULL teacher set, eligible AND abstain
    demo_rows: list of (task_id, demo, row) triples
    steps: stream length; rid: recorded seed namespace (no RNG here)

    Returns {schema, rid, grounded: [ak]*steps, model: [(ak,
    active)]*steps with active = (mode == "model_teacher"), demo,
    retention, histograms: {stream: {key: count}}}.
    """
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if not demo_rows:
        raise ValueError("empty demo_rows")
    demo_rows = [tuple(r) for r in demo_rows]
    if any(len(r) != 3 for r in demo_rows):
        raise ValueError(
            "demo_rows must be (task_id, demo, row) triples")
    bad_modes = {ak: teacher_anchors[ak].get("mode")
                 for ak in teacher_anchors
                 if teacher_anchors[ak].get("mode")
                 not in TEACHER_MODES}
    if bad_modes:
        raise ValueError(
            f"unrecognized teacher modes {bad_modes} — refusing to "
            f"treat them as abstention silently")
    # the registered equal-mass-per-task contract is cross-stream:
    # a task missing from either anchor set must fail loudly (a
    # V7.4B shortfall is handled by an explicit decision upstream,
    # never by silently unbalanced streams)
    g_tasks = {a["task"] for a in grounded_anchors.values()}
    t_tasks = {a["task"] for a in teacher_anchors.values()}
    assert g_tasks == t_tasks, (
        f"task coverage differs between grounded {sorted(g_tasks)} "
        f"and teacher {sorted(t_tasks)} anchor sets")
    g_order, g_queues = _cycled_stream(grounded_anchors, steps,
                                       "grounded")
    m_aks, m_queues = _cycled_stream(teacher_anchors, steps, "model")
    m_order = [(ak, teacher_anchors[ak]["mode"] == "model_teacher")
               for ak in m_aks]
    g_hist = _assert_caps(g_order, g_queues, grounded_anchors,
                          "grounded")
    m_hist = _assert_caps(m_aks, m_queues, teacher_anchors, "model")
    d_order = [demo_rows[k % len(demo_rows)] for k in range(steps)]
    r_order = [demo_rows[(k + len(demo_rows) // 2) % len(demo_rows)]
               for k in range(steps)]
    d_hist = Counter(d_order)
    r_hist = Counter(r_order)
    return {"schema": SCHEMA, "rid": rid,
            "grounded": g_order, "model": m_order,
            "demo": d_order, "retention": r_order,
            "histograms": {"grounded": g_hist, "model": m_hist,
                           "demo": {_demo_key(k): v
                                    for k, v in d_hist.items()},
                           "retention": {_demo_key(k): v
                                         for k, v
                                         in r_hist.items()}}}


def verify_matched_policy_schedule(sched, grounded_anchors,
                                   teacher_anchors, demo_rows):
    """Re-assert a LOADED schedule before step 0: deterministic
    rebuild from the same inputs must reproduce it exactly (the
    builder's cap/mass asserts re-run inside)."""
    if sched.get("schema") != SCHEMA:
        raise ValueError(
            f"schedule schema {sched.get('schema')!r} != {SCHEMA!r}")
    fresh = build_matched_policy_schedule(
        grounded_anchors, teacher_anchors, demo_rows,
        len(sched["grounded"]), sched["rid"])
    for stream in ("grounded", "model", "demo", "retention",
                   "histograms"):
        assert fresh[stream] == sched[stream], (
            f"loaded schedule diverges from deterministic rebuild "
            f"in stream {stream!r}")
    return True


def _selftest():
    # rr_schedule merge semantics locked: sources interleaved.
    assert _task_queue({"s0": ["a", "b"], "s1": ["c"]}) \
        == ["a", "c", "b"]

    steps = 300
    demo_rows = [(0, 0, r) for r in range(4)]

    # Case A — the V7.3E defect universe: 1 eligible + 8 abstention
    # teacher anchors over 3 tasks. The eligible anchor must receive
    # only its round-robin share, never the whole model budget.
    teach = {}
    for ti, task in enumerate(("t1", "t2", "t3")):
        for j in range(3):
            teach[f"{task}_s{j}_d5"] = {
                "task": task, "source_id": f"{task}_s{j}",
                "mode": ("model_teacher" if (ti, j) == (0, 0)
                         else "abstain")}
    ground_a = {f"{t}_g{j}": {"task": t, "source_id": f"{t}_gs{j}"}
                for t in ("t1", "t2", "t3") for j in range(2)}
    s = build_matched_policy_schedule(ground_a, teach,
                                      demo_rows, steps, "v074_test")
    for stream in ("grounded", "model", "demo", "retention"):
        assert len(s[stream]) == steps
    mh = s["histograms"]["model"]
    elig = "t1_s0_d5"
    # task t1 owns 100 slots over 3 anchors -> cap ceil(100/3) = 34
    assert mh[elig] <= 34, mh[elig]
    assert mh[elig] < steps        # V7.3 defect: was steps (300)
    assert set(mh) == set(teach)   # every abstention present
    n_active = sum(1 for _ak, act in s["model"] if act)
    assert n_active == mh[elig]
    assert all(act == (ak == elig) for ak, act in s["model"])
    tmass = Counter(teach[ak]["task"] for ak, _ in s["model"])
    assert set(tmass.values()) == {100}

    # Case B — 6 grounded anchors over 5 tasks (one task has 2):
    # V7.3-style balanced counts, 60 per task -> visits in {30, 60}.
    ground_b = {"t1_a0": {"task": "t1", "source_id": "t1_s0"},
                "t1_a1": {"task": "t1", "source_id": "t1_s1"},
                "t2_a0": {"task": "t2", "source_id": "t2_s0"},
                "t3_a0": {"task": "t3", "source_id": "t3_s0"},
                "t4_a0": {"task": "t4", "source_id": "t4_s0"},
                "t5_a0": {"task": "t5", "source_id": "t5_s0"}}
    teach_b = {f"{t}_ts0_d5": {"task": t, "source_id": f"{t}_ts0",
                               "mode": "abstain"}
               for t in ("t1", "t2", "t3", "t4", "t5")}
    s2 = build_matched_policy_schedule(ground_b, teach_b, demo_rows,
                                       steps, "v074_test")
    gh = s2["histograms"]["grounded"]
    assert gh["t1_a0"] == gh["t1_a1"] == 30, gh
    assert all(gh[k] == 60 for k in gh if not k.startswith("t1")), gh
    gmass = Counter(ground_b[ak]["task"] for ak in s2["grounded"])
    assert set(gmass.values()) == {60}

    # steps not divisible by n_tasks -> per-task mass equal +-1
    s3 = build_matched_policy_schedule(ground_a, teach, demo_rows,
                                       301, "v074_test")
    m3 = sorted(Counter(ground_a[ak]["task"]
                        for ak in s3["grounded"]).values())
    assert m3 == [100, 100, 101], m3

    # demo/retention: identical interleave semantics, half offset
    assert s["demo"][:5] == [(0, 0, 0), (0, 0, 1), (0, 0, 2),
                             (0, 0, 3), (0, 0, 0)]
    assert s["retention"][0] == demo_rows[len(demo_rows) // 2]
    assert all(s["retention"][k]
               == demo_rows[(k + 2) % 4] for k in range(steps))

    # histograms must be JSON-serializable end to end
    json.dumps(s["histograms"])
    # zero-visit anchors stay listed (degenerate small-steps build)
    s4 = build_matched_policy_schedule(ground_a, teach, demo_rows,
                                       2, "v074_test")
    assert set(s4["histograms"]["model"]) == set(teach)
    # list-typed demo rows are normalized, not crashed on
    s5 = build_matched_policy_schedule(ground_a, teach,
                                       [list(r) for r in demo_rows],
                                       steps, "v074_test")
    assert s5["demo"] == s["demo"]
    # loaded-schedule re-assert path
    assert verify_matched_policy_schedule(s, ground_a, teach,
                                          demo_rows)
    # loud failures: unknown mode, cross-stream task mismatch
    try:
        build_matched_policy_schedule(
            ground_a, {"x_s0_d1": {"task": "t1", "source_id": "x_s0",
                                   "mode": "eligible"}},
            demo_rows, steps, "v074_test")
        raise AssertionError("accepted unknown teacher mode")
    except ValueError:
        pass
    try:
        build_matched_policy_schedule(ground_b, teach, demo_rows,
                                      steps, "v074_test")
        raise AssertionError("accepted task-coverage mismatch")
    except AssertionError as e:
        assert "task coverage differs" in str(e), e

    print("model hist (defect case):", dict(sorted(mh.items())))
    print("grounded hist (6-anchor case):", dict(sorted(gh.items())))
    print("v074_policy_schedule self-test OK")


if __name__ == "__main__":
    _selftest()
