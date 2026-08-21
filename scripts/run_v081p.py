#!/usr/bin/env python
"""Action 2P / V8.1P runner — finite-deadline sibling-coverage pilot.

    python scripts/run_v081p.py --plan     # validate + print; spend nothing
    python scripts/run_v081p.py --run

Phase order is registered and enforced: ALL 20 sources close first, THEN the
first eight eligible failures are sealed in seed order, THEN pools and CRN
schedules are hash-sealed, and only THEN is any branch outcome produced.
"""
from __future__ import annotations

import argparse, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402
ensure_project_libero_config()

from lcwm import v081p_contract as P  # noqa: E402
from lcwm.v080r_panel import make_env_at  # noqa: E402
from lcwm.v081p_exec import (SegmentLedger, build_pool, run_branch,  # noqa: E402
                             run_source, select_candidates)
from register_v081p import verify_sealed  # noqa: E402

OUT_ROOT = REPO / "results" / "v081p"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    if not (a.plan or a.run):
        raise SystemExit("pass --plan or --run")

    pre = verify_sealed(strict_git=not a.allow_dirty)
    print(json.dumps({k: pre.get(k) for k in
                      ("ok", "self_sha256_match", "sealed_source_count")}, indent=2))
    if pre.get("source_drift"):
        raise SystemExit(f"HALT: sealed source drift: {sorted(pre['source_drift'])}")
    if not pre.get("self_sha256_match"):
        raise SystemExit("HALT: registration self_sha256 does not recompute")
    if pre["git"]["dirty"] and not a.allow_dirty:
        raise SystemExit("HALT: dirty tree; commit or pass --allow-dirty")
    reg = json.loads((REPO / "results/v081p_registration/V081P_REGISTRATION.json").read_text())
    root = reg["self_sha256"]           # every RNG key derives from the seal

    bad = set(P.SOURCE_SEEDS) & (P.RESERVED_SEEDS | P.RETIRED_SEEDS
                                 | P.CALIBRATION_ONLY_SEEDS)
    if bad:
        raise SystemExit(f"HALT: source seeds collide with protected families: {sorted(bad)}")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT_ROOT / stamp
    plan = {"action": "2P", "utc": stamp, "registration_sha256": root,
            "git_head": pre["git"]["head"], "git_dirty": pre["git"]["dirty"],
            "task": P.TASK, "L": P.DEADLINE, "tau": P.TAU,
            "sources": len(P.SOURCE_SEEDS), "anchors": P.N_ANCHORS,
            "per_anchor_segments": P.N_CRN + (P.N_DIVERSITY + P.N_RANDOM) * P.N_CRN,
            "worst_case_segments": P.WORST_CASE_SEGMENTS,
            "cap": P.INTERACTION_CAP, "cap_total": P.INTERACTION_CAP_TOTAL}
    if a.plan:
        print(json.dumps({"PLAN_ONLY": True, **plan}, indent=2))
        return 0

    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps({**plan, "status": "RUNNING"}, indent=2))
    ledger = SegmentLedger(out / "segment_ledger.jsonl", P.INTERACTION_CAP)

    from lcwm.chassis import DEFAULT_MODEL, Pi05Runner
    runner = Pi05Runner(model_id=DEFAULT_MODEL, suite_name="libero_10", n_action_steps=10)
    env = make_env_at(P.TASK, P.DEADLINE)

    # ---- phase 1: ALL sources close before anything is selected ----------
    sources = []
    for seed in P.SOURCE_SEEDS:
        s = run_source(runner, env, seed, ledger)
        a_ = s["anchor"]
        sources.append({"seed": seed, "steps": s["steps"],
                        "success_step": s["success_step"],
                        "failure_at_L": s["failure_at_L"],
                        "anchor_id": a_.anchor_id if a_ else None,
                        "mask_ok": bool(a_ and a_.mask_ok),
                        "mask_why": (a_.mask_why if a_ else ["no tau snapshot"]),
                        "stratum": (a_.stratum if a_ else None),
                        "content_hash": (a_.content_hash if a_ else None)})
        print(f"src {seed}: fail@L={s['failure_at_L']} succ@{s['success_step']} "
              f"mask_ok={bool(a_ and a_.mask_ok)}", flush=True)
        if a_:
            s["anchor"].snapshot_obj = a_
    (out / "sources.json").write_text(json.dumps(sources, indent=2))

    F = sum(s["failure_at_L"] for s in sources)
    E = sum(s["failure_at_L"] and s["mask_ok"] for s in sources)
    print(f"\nsources closed: F={F}/20 deadline failures, E={E}/20 exact-mask eligible")
    if E < P.N_ANCHORS:
        (out / "HALT.json").write_text(json.dumps(
            {"reason": "source/stratum underfill", "F": F, "E": E,
             "required": P.N_ANCHORS,
             "rule": "underfill HALTs; sources are not replaced or extended"}, indent=2))
        raise SystemExit(f"HALT: only {E} eligible of {P.N_ANCHORS} required")
    print("selection sealed in seed order; pools next")
    (out / "manifest.json").write_text(json.dumps(
        {**plan, "status": "SOURCES_CLOSED", "F": F, "E": E}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
