#!/usr/bin/env python
"""Action 2M.3 — separate candidate effect from continuation noise.

    python scripts/run_v085_noisefloor.py --plan
    python scripts/run_v085_noisefloor.py --run

Phase order is registered: sources close and anchors seal first; then the 64-draw
pool and the four max-spread alternatives are selected and hash-sealed with the
12 shared continuation keys; only then does any branch produce an outcome.
"""
from __future__ import annotations

import argparse, hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402
ensure_project_libero_config()

import numpy as np  # noqa: E402
from lcwm import v085_noisefloor as N  # noqa: E402
from lcwm.v080r_panel import make_env_at  # noqa: E402
from lcwm.v081p_exec import SegmentLedger, TechnicalHalt, run_branch, run_source  # noqa: E402

OUT_ROOT = REPO / "results" / "v085_noisefloor"
SRC_LINE, BR_LINE = "source", "branch"


def git_state():
    g = lambda *a: subprocess.run(["git", *a], cwd=REPO, capture_output=True,
                                  text=True).stdout.strip()
    return {"head": g("rev-parse", "HEAD"),
            "dirty": bool(g("status", "--porcelain"))}


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    if not (a.plan or a.run):
        raise SystemExit("pass --plan or --run")

    g = git_state()
    if g["dirty"] and not a.allow_dirty:
        raise SystemExit("HALT: dirty tree; commit or pass --allow-dirty")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT_ROOT / stamp
    manifest = {
        "schema": N.SCHEMA_VERSION, "action": "2M.3", "utc": stamp,
        "daily": "plan_and_progress/archive/daily/2026-08-23.md (ACTION ITEM 2M.3)",
        "objective": ("does candidate identity move the registered H80 outcome "
                      "above continuation-seed variability at "
                      f"{N.TASK}@{N.DEADLINE}, tau={N.TAU}?"),
        "git": g,
        "source_hashes": {p: sha(REPO / p) for p in
                          ("lcwm/v085_noisefloor.py",
                           "scripts/run_v085_noisefloor.py",
                           "lcwm/v081p_exec.py", "lcwm/v080r_panel.py")},
        "locked": {
            "task": N.TASK, "deadline": N.DEADLINE, "tau": N.TAU,
            "c_prefix": N.C_PREFIX, "H": N.H,
            "anchors": N.N_ANCHORS, "raw_draws": N.N_RAW_DRAWS,
            "alternatives": N.N_ALTERNATIVES, "candidates": N.N_CANDIDATES,
            "repeats": N.N_REPEATS, "max_sources": N.MAX_SOURCES,
            "source_seeds": list(N.SOURCE_SEEDS),
            "selection_rule": ("standardize summed executed action over the 64-draw "
                               "pool; seed the greedy set at the draw farthest from "
                               "the reference; add farthest-point candidates; ties "
                               "broken by index. Outcome-blind and deterministic."),
            "continuation_keys": ("12 per anchor, derived from the manifest root; "
                                  "every candidate AND the reference uses the same 12"),
            "floors": N.FLOORS, "primary_continuous": list(N.PRIMARY_CONTINUOUS),
            "secondary": list(N.SECONDARY),
            "statistic": ("D = matched-seed candidate/reference non-tie rate "
                          "- same-chunk cross-seed non-tie rate, per component"),
            "bootstrap": {"B": N.BOOTSTRAP_B, "alpha": 0.05,
                          "unit": "anchor (repeats and pairs nested, never resampled)",
                          "key_root": N.BOOTSTRAP_KEY_ROOT},
            "decision": ("CANDIDATE_EFFECT_IDENTIFIED iff, for the SAME primary "
                         "continuous component, D_lower > 0 AND ICC_lower > 0"),
            "role": N.ROLE, "subrole": N.SUBROLE, "data_use": N.DATA_USE,
            "interaction_cap": N.INTERACTION_CAP,
            "interaction_cap_total": N.INTERACTION_CAP_TOTAL,
            "no_extension": "no quota or cap extension; underfill HALTs",
        },
    }
    manifest["root"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()).hexdigest()

    if a.plan:
        print(json.dumps({"PLAN_ONLY": True, **{k: manifest[k] for k in
              ("action", "utc", "git", "root")},
              "locked": {k: manifest["locked"][k] for k in
                         ("anchors", "candidates", "repeats", "max_sources",
                          "interaction_cap", "interaction_cap_total")}}, indent=2))
        return 0

    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps({**manifest, "status": "RUNNING"}, indent=2))
    ledger = SegmentLedger(out / "segment_ledger.jsonl", N.INTERACTION_CAP,
                           action_root=OUT_ROOT)
    root = manifest["root"]

    from lcwm.chassis import DEFAULT_MODEL, Pi05Runner
    from lcwm.sampler import prefix_forward, sample_chunks
    from lcwm.snapshot import restore
    runner = Pi05Runner(model_id=DEFAULT_MODEL, suite_name="libero_10", n_action_steps=10)
    env = make_env_at(N.TASK, N.DEADLINE)

    # ---- phase 1: sources -> anchors, stop at N_ANCHORS or the source ceiling
    anchors, sources = [], []
    for seed in N.SOURCE_SEEDS:
        if len(anchors) >= N.N_ANCHORS:
            break
        try:
            s = run_source(runner, env, seed, ledger)
        except TechnicalHalt as e:
            (out / "HALT.json").write_text(json.dumps(
                {"reason": "TECHNICAL_HALT", "seed": seed, "detail": str(e)}, indent=2))
            raise
        ok = bool(s["anchor"] and s["anchor"].mask_ok and s["failure_at_L"])
        sources.append({"seed": seed, "failure_at_L": s["failure_at_L"],
                        "eligible": ok, "steps": s["steps"]})
        if ok:
            anchors.append(s["anchor"])
        print(f"src {seed}: eligible={ok}  anchors={len(anchors)}/{N.N_ANCHORS}", flush=True)
    (out / "sources.json").write_text(json.dumps(sources, indent=2))
    if len(anchors) < N.N_ANCHORS:
        (out / "HALT.json").write_text(json.dumps(
            {"reason": "anchor underfill", "got": len(anchors),
             "required": N.N_ANCHORS, "sources_spent": len(sources),
             "rule": "no quota or cap extension"}, indent=2))
        raise SystemExit(f"HALT: {len(anchors)}/{N.N_ANCHORS} anchors from "
                         f"{len(sources)} sources")

    # ---- phase 2: pools, max-spread selection, shared keys — all sealed first
    seal = {"root": root, "anchors": []}
    pools = {}
    for anc in anchors:
        restore(env, anc.snapshot)
        runner.reset()
        obs = env._format_raw_obs(env._env.env._get_observations())
        po = runner._obs_to_policy_batch(obs, env.task_description)
        pf = prefix_forward(runner.policy, po)
        kref = int.from_bytes(hashlib.sha256(f"{root}|ref|{anc.anchor_id}".encode())
                              .digest()[:8], "big") % (2**31 - 1)
        kalt = int.from_bytes(hashlib.sha256(f"{root}|alt|{anc.anchor_id}".encode())
                              .digest()[:8], "big") % (2**31 - 1)
        ref = sample_chunks(runner.policy, po, 1, seed=kref, prefix=pf)[0, :N.C_PREFIX]
        raw = sample_chunks(runner.policy, po, N.N_RAW_DRAWS, seed=kalt,
                            prefix=pf)[:, :N.C_PREFIX]
        del pf
        ref_env = runner.chunk_to_env(ref.detach().float().cpu())
        raw_env = [runner.chunk_to_env(raw[i].detach().float().cpu())
                   for i in range(N.N_RAW_DRAWS)]
        pick = N.select_max_spread(raw_env, ref_env)
        chunks = [ref_env] + [raw_env[i] for i in pick]
        ids = ["reference"] + [f"alt{i}" for i in range(N.N_ALTERNATIVES)]
        keys = [int.from_bytes(hashlib.sha256(
            f"{root}|crn|{anc.anchor_id}|{j}".encode()).digest()[:8], "big")
            % (2**31 - 1) for j in range(N.N_REPEATS)]
        S = np.stack([np.asarray(c).sum(0) for c in chunks])
        pools[anc.anchor_id] = {"chunks": chunks, "ids": ids, "keys": keys}
        seal["anchors"].append({
            "anchor_id": anc.anchor_id, "seed": anc.seed,
            "content_hash": anc.content_hash, "selected": [int(i) for i in pick],
            "crn_keys": keys,
            "achieved_action_spread": {
                "max_pairwise_summed": float(np.abs(S[:, None] - S[None]).max()),
                "per_dim_max_gap": np.abs(S[:, None] - S[None]).max(axis=(0, 1)).tolist()}})
    seal["seal_sha256"] = hashlib.sha256(
        json.dumps(seal, sort_keys=True, default=str).encode()).hexdigest()
    (out / "execution_seal.json").write_text(json.dumps(seal, indent=2, default=str))
    print(f"seal {seal['seal_sha256'][:16]} — {len(anchors)} anchors x "
          f"{N.N_CANDIDATES} candidates x {N.N_REPEATS} repeats", flush=True)

    # ---- phase 3: execute
    rows = []
    for anc in anchors:
        p = pools[anc.anchor_id]
        for ci, (cid, chunk) in enumerate(zip(p["ids"], p["chunks"])):
            for ri, key in enumerate(p["keys"]):
                r = run_branch(runner, env, anc, np.asarray(chunk), key,
                               BR_LINE, BR_LINE, ledger, cid)
                row = {"anchor_id": anc.anchor_id, "candidate_id": cid,
                       "candidate_index": ci, "repeat": ri, "crn_key": key,
                       "restore_verified": r["restore_verified"],
                       "outcome": r["readouts"][str(N.H)]}
                rows.append(row)
                with (out / "outcomes.jsonl").open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
        print(f"{anc.anchor_id}: {N.N_CANDIDATES * N.N_REPEATS} branches done "
              f"({ledger.total}/{N.INTERACTION_CAP_TOTAL} steps)", flush=True)

    # ---- phase 4: statistics
    per_component = {}
    for comp in N.ALL_COMPONENTS:
        d_vals, icc_vals, sc = [], [], {"ever": 0, "consistent": 0, "candidates": 0}
        for anc in anchors:
            grid = [[None] * N.N_REPEATS for _ in range(N.N_CANDIDATES)]
            for row in rows:
                if row["anchor_id"] == anc.anchor_id:
                    grid[row["candidate_index"]][row["repeat"]] = row["outcome"][comp]
            ar = N.anchor_rates(grid, comp)
            d_vals.append(ar.d)
            icc = N.icc_one_way(grid)
            if icc is not None:
                icc_vals.append(icc)
            s = N.sign_consistency(grid, comp)
            sc["ever"] += s["ever_nontie"]; sc["consistent"] += s["consistent_all_repeats"]
            sc["candidates"] += s["candidates"]
        per_component[comp] = {
            "D": N.cluster_bootstrap_lower(d_vals, key=N.bootstrap_key(root, comp, "D")),
            "ICC": N.cluster_bootstrap_lower(icc_vals, key=N.bootstrap_key(root, comp, "ICC")),
            "per_anchor_D": d_vals, "sign_consistency": sc,
            "primary": comp in N.PRIMARY_CONTINUOUS}

    decision = N.decide(per_component)
    summary = {"manifest": {**manifest, "status": "COMPLETE"},
               "anchors": [a.anchor_id for a in anchors],
               "sources_spent": len(sources), "branches": len(rows),
               "execution_seal": seal["seal_sha256"],
               "restore_verified": f"{sum(r['restore_verified'] for r in rows)}/{len(rows)}",
               "per_component": per_component, "decision": decision,
               "interaction": {"by_line": ledger.spent, "total": ledger.total,
                               "cap": N.INTERACTION_CAP_TOTAL}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "manifest.json").write_text(json.dumps({**manifest, "status": "COMPLETE"}, indent=2))

    # A degenerate component (e.g. no damage anywhere) yields ICC None by
    # design; format defensively so the console never crashes AFTER the
    # statistics are already on disk.
    fmt = lambda v: "  n/a  " if v is None else f"{v:+.4f}"
    print("\n=== Action 2M.3 ===")
    for comp in N.ALL_COMPONENTS:
        r = per_component[comp]
        tag = "PRIMARY" if r["primary"] else "secondary"
        print(f"{comp:5s} [{tag:9s}] D={fmt(r['D']['point'])} "
              f"(lower {fmt(r['D']['lower'])})  ICC={fmt(r['ICC']['point'])} "
              f"(lower {fmt(r['ICC']['lower'])})  sign-consistent "
              f"{r['sign_consistency']['consistent']}/{r['sign_consistency']['candidates']}")
    print(json.dumps(decision, indent=2))
    print(f"steps {ledger.total}/{N.INTERACTION_CAP_TOTAL} | {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
