#!/usr/bin/env python
"""One-command project status: GPU, running jobs, evaluation ledgers,
latest records, H7 queue state. Read-only; safe to run any time.
Usage: python scripts/status.py    (or: watch -n 30 python scripts/status.py)
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FACTORIAL = REPO_ROOT / "results" / "behavior_factorial_v1"
PUBLIC = REPO_ROOT / "results" / "libero_loho_public_v1"


def sh(cmd: str) -> str:
    try:
        return subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception as error:  # noqa: BLE001 — status must never crash
        return f"(unavailable: {error})"


def main() -> None:
    print(f"=== status @ {datetime.now():%m-%d %H:%M:%S} ===")
    print("[gpu]", sh(
        "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu "
        "--format=csv,noheader"
    ))
    procs = sh(
        "ps -o pid,etime,cmd -C python --no-headers 2>/dev/null | "
        "grep -E 'train_|collect_|eval_|build_' | grep -v grep"
    )
    print("[jobs]", procs if procs else "none running")

    for panel_dir in sorted(FACTORIAL.glob("*/records.jsonl")):
        records = [
            json.loads(line)
            for line in panel_dir.read_text().splitlines()
            if line.strip()
        ]
        by_run: dict[str, list] = {}
        for r in records:
            by_run.setdefault(r["run_id"], []).append(r)
        for run_id, rows in sorted(by_run.items()):
            succ = sum(r["success"] for r in rows)
            print(
                f"[{panel_dir.parent.name}] {run_id}: "
                f"{len(rows)} episodes, {succ} successes; latest: "
                f"{rows[-1]['task']} s{rows[-1]['seed']} "
                f"{rows[-1]['arm']} q={rows[-1]['q_final']:.2f}"
            )

    manifest_path = PUBLIC / "task_source_manifest.json"
    if manifest_path.exists():
        tasks = json.loads(manifest_path.read_text())["tasks"]
        states = {name: t["status"] for name, t in tasks.items()}
        done = sum(1 for s in states.values() if s == "reconstructed")
        print(f"[public tasks] {done}/{len(states)} reconstructed:",
              {k.split('_', 1)[1]: v for k, v in states.items()})
    smoke = PUBLIC / "evaluator_smoke.jsonl"
    if smoke.exists():
        rows = [json.loads(l) for l in smoke.read_text().splitlines() if l.strip()]
        print(f"[public smoke] {len(rows)} episodes;",
              [(r['task'], round(r['q_public'], 2)) for r in rows[-5:]])
    print("[git]", sh("cd " + str(REPO_ROOT) + " && git log --oneline -1"))


if __name__ == "__main__":
    main()
