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
from lcwm.v081p_exec import (SegmentLedger, TechnicalHalt, build_pool,  # noqa: E402
                             run_branch, run_source, select_candidates)
import numpy as np  # noqa: E402
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
    # action-scoped, not invocation-scoped: a restart must NOT receive a fresh cap
    ledger = SegmentLedger(out / "segment_ledger.jsonl", P.INTERACTION_CAP,
                           action_root=OUT_ROOT)
    if ledger.total:
        print(f"prior ledgered spend across this action: {ledger.total}"
              f"/{P.INTERACTION_CAP_TOTAL}", flush=True)

    from lcwm.chassis import DEFAULT_MODEL, Pi05Runner
    runner = Pi05Runner(model_id=DEFAULT_MODEL, suite_name="libero_10", n_action_steps=10)
    env = make_env_at(P.TASK, P.DEADLINE)

    # ---- phase 1: ALL sources close before anything is selected ----------
    sources, anchors_by_seed = [], {}
    for seed in P.SOURCE_SEEDS:
        try:
            s = run_source(runner, env, seed, ledger)
        except TechnicalHalt as e:
            (out / "HALT.json").write_text(json.dumps(
                {"reason": "TECHNICAL_HALT", "seed": seed, "detail": str(e),
                 "rule": ("charged, not replaced, and not counted as a normal "
                          "non-failure")}, indent=2))
            raise
        a_ = s["anchor"]
        if a_ is not None:
            anchors_by_seed[seed] = a_
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
    # ---- phase 2: seal the eight anchors, then the pools ----------------
    eligible = [s_ for s_ in sources if s_["failure_at_L"] and s_["mask_ok"]]
    selected = sorted(eligible, key=lambda r: r["seed"])[:P.N_ANCHORS]
    anchors = [anchors_by_seed[r["seed"]] for r in selected]
    print(f"sealed anchors (first {P.N_ANCHORS} eligible in seed order): "
          f"{[a.seed for a in anchors]}")

    pools, sel = {}, {}
    for a_ in anchors:
        pool = build_pool(runner, env, a_, root)
        if pool["n_unique"] < P.MIN_UNIQUE_ALTS:
            (out / "HALT.json").write_text(json.dumps(
                {"reason": "pool underfill", "anchor": a_.anchor_id,
                 "n_unique": pool["n_unique"], "required": P.MIN_UNIQUE_ALTS},
                indent=2))
            raise SystemExit(f"HALT: {a_.anchor_id} pool has {pool['n_unique']} "
                             f"unique < {P.MIN_UNIQUE_ALTS}")
        pools[a_.anchor_id] = pool
        sel[a_.anchor_id] = select_candidates(pool, root, a_.anchor_id)
        print(f"  {a_.anchor_id}: unique={pool['n_unique']} "
              f"div={sel[a_.anchor_id]['diversity']} rand={sel[a_.anchor_id]['random']}")

    # Global hash-seal of pools, chunks, selections, execution order and CRN
    # schedules BEFORE any branch outcome exists.
    schedule = []
    for a_ in anchors:
        keys = P.crn_keys(root, a_.anchor_id)
        cands = ([("reference", None, P.Subrole.REF_PREFIX.value, P.Subrole.REF_CONT.value)]
                 + [(f"div{i}", j, P.Subrole.DIV_PREFIX.value, P.Subrole.DIV_CONT.value)
                    for i, j in enumerate(sel[a_.anchor_id]["diversity"])]
                 + [(f"rand{i}", j, P.Subrole.RAND_PREFIX.value, P.Subrole.RAND_CONT.value)
                    for i, j in enumerate(sel[a_.anchor_id]["random"])])
        for cid, jdx, cp, cc in cands:
            for k in keys:
                schedule.append({"anchor_id": a_.anchor_id, "candidate_id": cid,
                                 "alt_index": jdx, "crn_key": k,
                                 "cap_prefix": cp, "cap_cont": cc})
    seal = {"root": root, "anchors": [{"anchor_id": a_.anchor_id, "seed": a_.seed,
                                       "content_hash": a_.content_hash,
                                       "stratum": a_.stratum} for a_ in anchors],
            "selections": sel, "schedule": schedule,
            "pool_hashes": {aid: hashlib.sha256(
                np.concatenate([pools[aid]["reference"].reshape(-1)]
                               + [x.reshape(-1) for x in pools[aid]["alternatives"]]
                               ).tobytes()).hexdigest() for aid in pools},
            "floors": P.SEALED_FLOORS_KWARGS}
    seal["seal_sha256"] = hashlib.sha256(
        json.dumps(seal, sort_keys=True, default=str).encode()).hexdigest()
    (out / "execution_seal.json").write_text(json.dumps(seal, indent=2, default=str))
    print(f"execution seal {seal['seal_sha256'][:16]} written "
          f"({len(schedule)} segments) BEFORE any outcome", flush=True)

    # ---- phase 3: paired execution -------------------------------------
    branches = []
    for step in schedule:
        a_ = next(x for x in anchors if x.anchor_id == step["anchor_id"])
        pool = pools[a_.anchor_id]
        pre = (pool["reference"] if step["candidate_id"] == "reference"
               else pool["alternatives"][step["alt_index"]])
        r = run_branch(runner, env, a_, pre, step["crn_key"], step["cap_prefix"],
                       step["cap_cont"], ledger, step["candidate_id"])
        r.update({"anchor_id": a_.anchor_id, "anchor_hash": a_.content_hash})
        branches.append(r)
        with (out / "branches.jsonl").open("a") as fh:
            fh.write(json.dumps(r) + "\n")
    print(f"executed {len(branches)}/{len(schedule)} branch segments", flush=True)

    # ---- phase 4: reduce + ADVANCE gate --------------------------------
    floors = P.sealed_floors()
    verdicts, per_anchor = [], {}
    for a_ in anchors:
        keys = P.crn_keys(root, a_.anchor_id)
        by = lambda cid: [next(b for b in branches if b["anchor_id"] == a_.anchor_id
                               and b["candidate_id"] == cid and b["crn_key"] == k)
                          for k in keys]
        ref = [b["readouts"][str(P.H_PRIMARY)] for b in by("reference")]
        av = P.AnchorVerdict(a_.anchor_id)
        for cid in [f"div{i}" for i in range(P.N_DIVERSITY)] + \
                   [f"rand{i}" for i in range(P.N_RANDOM)]:
            alt = [b["readouts"][str(P.H_PRIMARY)] for b in by(cid)]
            av.alternatives.append(P.classify_alternative(cid, alt, ref, floors))
        verdicts.append(av)
        per_anchor[a_.anchor_id] = {
            "seed": a_.seed, "variation": av.has_variation,
            "n_positive": av.n_positive, "n_non_improving": av.n_non_improving,
            "both": av.has_both,
            "alternatives": [{"id": x.candidate_id, "per_key": x.per_key,
                              "label": x.label} for x in av.alternatives]}

    # restore verification now compares a RE-SNAPPED live env against the
    # sealed anchor hash, so this can actually fail
    restore_ok = sum(1 for b in branches if b["restore_verified"])

    # replay tolerance over ACHIEVED post-prefix physics across the three
    # reference repeats at each anchor
    replay_ok = 0
    replay_detail = {}
    for a_ in anchors:
        sigs = [np.asarray(b["post_prefix_signature"]) for b in branches
                if b["anchor_id"] == a_.anchor_id and b["candidate_id"] == "reference"]
        dev = max((float(np.linalg.norm(x - y)) for i, x in enumerate(sigs)
                   for y in sigs[i + 1:]), default=0.0)
        replay_detail[a_.anchor_id] = {"max_pairwise_l2": dev,
                                       "tol": P.REPLAY_TOL_L2,
                                       "pass": dev <= P.REPLAY_TOL_L2,
                                       "n_repeats": len(sigs)}
        replay_ok += P.N_CRN if dev <= P.REPLAY_TOL_L2 else 0
    gate = P.advance_gate(
        sources_closed=len(sources) == len(P.SOURCE_SEEDS), selected_in_order=True,
        reserved_used=0, replay_ok=replay_ok,
        replay_total=P.N_ANCHORS * P.N_CRN, restore_ok=restore_ok,
        restore_total=len(branches), pools_ok=all(
            pools[a_.anchor_id]["n_unique"] >= P.MIN_UNIQUE_ALTS for a_ in anchors),
        within_cap=ledger.total <= P.INTERACTION_CAP_TOTAL, anchors=verdicts)

    spend = {"by_cap_line": ledger.spent, "total": ledger.total,
             "cap_total": P.INTERACTION_CAP_TOTAL,
             "note": "DERIVED from every segment_ledger.jsonl under results/v081p"}
    (OUT_ROOT / "SPEND.json").write_text(json.dumps(spend, indent=2))
    summary = {"manifest": {**plan, "status": "COMPLETE"}, "F": F, "E": E,
               "restore_verified": f"{restore_ok}/{len(branches)}",
               "replay": replay_detail, "spend": spend,
               "selected_seeds": [a_.seed for a_ in anchors],
               "execution_seal": seal["seal_sha256"], "per_anchor": per_anchor,
               "gate": gate, "env_steps": ledger.total,
               "cap_total": P.INTERACTION_CAP_TOTAL,
               "secondary_horizons": [h for h in P.H_READOUTS if h != P.H_PRIMARY]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "manifest.json").write_text(json.dumps({**plan, "status": "COMPLETE"}, indent=2))
    print("\n=== Action 2P ===")
    for aid, v in per_anchor.items():
        print(f"{aid}: var={v['variation']} pos={v['n_positive']} "
              f"non_imp={v['n_non_improving']} both={v['both']}")
    print(json.dumps(gate, indent=2))
    print(f"env steps {ledger.total}/{P.INTERACTION_CAP_TOTAL}")
    return 0 if gate["verdict"] == "ADVANCE" else 3


if __name__ == "__main__":
    raise SystemExit(main())
