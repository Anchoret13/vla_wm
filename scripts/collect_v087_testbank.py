#!/usr/bin/env python
"""Action 2M.5 — collect and SEAL the fresh held-out test bank before training.

    python scripts/collect_v087_testbank.py --plan
    python scripts/collect_v087_testbank.py --run

Reuses `scripts/collect_v086_bank.py`'s `run_source` / `build_group` verbatim.
That is deliberate: the 2M.4 collection was discarded because a
reimplementation dropped the `failure@250` half of anchor eligibility, so this
action reuses the corrected function rather than writing a second copy of the
rule.

Its outcomes must not be read until the M0.3 validation checkpoint is fixed.
"""
from __future__ import annotations

import argparse, hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402
ensure_project_libero_config()

import torch  # noqa: E402
from lcwm import v086_bank as B  # noqa: E402
from lcwm.v080r_panel import make_env_at  # noqa: E402
from lcwm.v081p_exec import SegmentLedger  # noqa: E402

import collect_v086_bank as C  # noqa: E402

SEEDS = tuple(range(4200, 4223))                 # 23, frozen by the action item
N_GROUPS = 8
REPEATS = 6                                      # shared continuation keys
BRANCHES = N_GROUPS * B.N_CANDIDATES * REPEATS   # 240
CAP = {"source": len(SEEDS) * B.DEADLINE,        # 5,750
       "branch": BRANCHES * (B.C_PREFIX + B.H)}  # 21,600
CAP_TOTAL = sum(CAP.values())                    # 27,350
assert BRANCHES == 240 and CAP_TOTAL == 27_350
assert REPEATS == B.REPEATS["test"], "reusing build_group's test-split repeats"
_prior = (set(range(3200, 3400)) | set(range(3400, 3440)) | set(range(3480, 3500))
          | set(range(3500, 3576)) | set(range(3600, 3706)) | set(range(3800, 3816))
          | set(range(3900, 3930)) | set(range(3940, 3978)) | set(range(4000, 4154)))
assert not (set(SEEDS) & _prior), "test seeds must be fresh"

OUT_ROOT = REPO / "results" / "v087_testbank"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    g = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                       text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                                capture_output=True, text=True).stdout.strip())
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    plan = {"action": "2M.5", "bank": "fresh held-out test", "utc": stamp,
            "git": g, "git_dirty": dirty, "seeds": [SEEDS[0], SEEDS[-1], len(SEEDS)],
            "groups": N_GROUPS, "candidates": B.N_CANDIDATES, "repeats": REPEATS,
            "branches": BRANCHES, "cap": CAP, "cap_total": CAP_TOTAL,
            "eligibility": "failure@250 AND the exact event mask at tau=160",
            "sealed": ("outcomes must not be read until the M0.3 validation "
                       "checkpoint is fixed")}
    if a.plan:
        print(json.dumps(plan, indent=2)); return 0
    if dirty:
        raise SystemExit("commit before the environment run")

    out = OUT_ROOT / stamp
    out.mkdir(parents=True, exist_ok=True)
    root = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    plan["root"] = root
    (out / "manifest.json").write_text(json.dumps({**plan, "status": "RUNNING"}, indent=2))
    ledger = SegmentLedger(out / "segment_ledger.jsonl", CAP, action_root=OUT_ROOT)

    from lcwm.chassis import DEFAULT_MODEL, Pi05Runner
    runner = Pi05Runner(model_id=DEFAULT_MODEL, suite_name="libero_10", n_action_steps=10)
    env = make_env_at(B.TASK, B.DEADLINE)
    scene = C.Scene(env)

    groups, sources = [], []
    for seed in SEEDS:
        if len(groups) >= N_GROUPS:
            break
        s = C.run_source(runner, env, scene, seed, ledger)
        sources.append({"seed": seed, "failure_at_L": s["failure_at_L"],
                        "eligible": s["anchor"] is not None})
        if s["anchor"] is None:
            continue
        grp = C.build_group(runner, env, scene, s["anchor"], "test", root, ledger)
        B.validate_group(grp)
        groups.append(grp)
        print(f"test {grp['anchor_id']} {len(groups)}/{N_GROUPS} "
              f"restores {grp['restores_ok']}/{B.N_CANDIDATES * REPEATS} "
              f"({ledger.total}/{CAP_TOTAL})", flush=True)
    (out / "sources.json").write_text(json.dumps(sources, indent=2))
    if len(groups) < N_GROUPS:
        (out / "HALT.json").write_text(json.dumps(
            {"reason": "source underfill", "got": len(groups), "need": N_GROUPS,
             "sources_spent": len(sources)}, indent=2))
        raise SystemExit(f"HALT: {len(groups)}/{N_GROUPS} groups")
    torch.save({"schema": B.SCHEMA, "groups": groups, "root": root}, out / "groups.pt")
    sha = hashlib.sha256((out / "groups.pt").read_bytes()).hexdigest()
    (out / "SEALED.json").write_text(json.dumps(
        {"groups_sha256": sha, "sealed_utc": stamp, "root": root,
         "rule": ("outcomes not to be read until the M0.3 validation checkpoint "
                  "is fixed")}, indent=2))
    (out / "manifest.json").write_text(json.dumps(
        {**plan, "status": "COMPLETE", "groups_sha256": sha}, indent=2))
    bad = sum(x["eligible"] and not x["failure_at_L"] for x in sources)
    print(json.dumps({"groups": len(groups), "sources": len(sources),
                      "branches": BRANCHES, "steps": ledger.total, "cap": CAP_TOTAL,
                      "anchors_from_succeeding_sources": bad,
                      "groups_sha256": sha[:16], "out": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
