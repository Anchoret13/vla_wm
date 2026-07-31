#!/usr/bin/env python
"""Repair the pre-fix v1.1 recovery-prompt labels without losing the original."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.branch_data import (  # noqa: E402
    MANIFEST,
    load_branch_group,
    save_branch_group,
)

SUPPORT_PAIR_INSTRUCTION = (
    "put both the cream cheese box and the butter in the basket"
)
REPAIR_ID = "recovery_behavior_prompt_id_v1"


def prompt_id_for_trace(group: dict, trace: dict) -> str:
    existing = trace.get("behavior_prompt_id")
    if existing:
        return str(existing)
    instruction = str(trace["behavior_instruction"])
    if instruction == SUPPORT_PAIR_INSTRUCTION:
        return "support_pair_recovery"
    remaining = group["goal_specs"]["chain3_remaining_atomic"][
        "canonical_instruction"
    ]
    if instruction == remaining:
        return "remaining_atomic_recovery"
    digest = hashlib.sha1(instruction.encode()).hexdigest()[:10]
    return f"recovery_instruction_{digest}"


def repair_group(group: dict) -> tuple[dict, int]:
    traces = group["recovery_trace"]
    if not traces:
        return group, 0
    ids = list(group["history_behavior_prompt_id"])
    start = len(ids) - len(traces)
    if start < 0:
        raise ValueError("more recovery records than history action labels")
    changed = 0
    for offset, trace in enumerate(traces):
        prompt_id = prompt_id_for_trace(group, trace)
        expected = f"{prompt_id}:seed{int(trace['chosen_noise_seed'])}"
        index = start + offset
        if ids[index] != expected:
            ids[index] = expected
            changed += 1
        trace["behavior_prompt_id"] = prompt_id
    if changed:
        group["history_behavior_prompt_id"] = ids
        repairs = list(group["provenance"].get("provenance_repairs", []))
        if REPAIR_ID not in repairs:
            repairs.append(REPAIR_ID)
        group["provenance"]["provenance_repairs"] = repairs
    return group, changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT.parent / "datasets" / "chain_branches_v1_1",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="atomically replace repaired groups after hard-linking a backup",
    )
    args = parser.parse_args()
    root = args.data.resolve()
    manifest = json.loads((root / MANIFEST).read_text())
    names = sorted(
        {
            name
            for split_names in manifest["splits"].values()
            for name in split_names
        }
    )
    total = 0
    for name in names:
        path = root / name
        group = load_branch_group(path)
        group, changed = repair_group(group)
        print(f"{name}: {changed} labels to repair")
        total += changed
        if changed and args.apply:
            backup = path.with_name(f"{path.name}.before_prompt_id_fix")
            if not backup.exists():
                os.link(path, backup)
            save_branch_group(group, path)
    mode = "applied" if args.apply else "dry-run"
    print(f"{mode}: {total} labels across {len(names)} groups")


if __name__ == "__main__":
    main()
