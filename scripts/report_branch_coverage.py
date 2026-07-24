#!/usr/bin/env python
"""Effect/noise coverage report (pre-registered 2026-07-24).

Read-only over the branch dataset. Per group, reports the pre-registered
quantities with task-object effects separated from incidental motion:

1. task-object sibling displacement (the group's dense-goal object),
2. incidental displacement (max over all other tracked objects),
3. restore-QC replay residuals as the noise floor (object_linf, q dispersion),
4. positive flow targets, predicate flips, dense-progress branches,
5. the effect-resolved verdict:
   max task-object displacement >= max(2 mm, 10 x replay object residual).

Aggregates: per split, groups / effect-resolved groups / positive targets /
independent source episodes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

ABS_FLOOR_M = 0.002
NOISE_MULTIPLE = 10.0


def group_row(path: Path, split: str) -> dict:
    group = torch.load(path, weights_only=False)
    mask = group["object_mask"].bool()
    disp = torch.linalg.vector_norm(
        group["object_pos_after"] - group["object_pos_before"][None], dim=-1
    )
    goal_index = int(group["dense_goal_object_index"])
    task_disp = float(disp[:, goal_index].max()) if goal_index >= 0 else 0.0
    incidental_indices = [
        index
        for index in range(disp.shape[1])
        if bool(mask[index]) and index != goal_index
    ]
    incidental_disp = (
        float(disp[:, incidental_indices].max())
        if incidental_indices
        else 0.0
    )
    qc = group["restore_qc"]
    floor = max(ABS_FLOOR_M, NOISE_MULTIPLE * float(qc["object_linf"]))
    active = group["atom_mask"].bool()
    flips = int(
        (
            group["bits_after"][:, active]
            != group["bits_before"][active][None]
        )
        .any(dim=1)
        .sum()
    )
    margin = float(group["dense_progress_margin"])
    contact = group["provenance"].get("contact_info") or {}
    return {
        "path": path.name,
        "split": split,
        "source_trajectory_id": group["source_trajectory_id"],
        "source_seed": group.get("source_seed"),
        "snapshot_mode": (
            group["provenance"]
            .get("collection_contract", {})
            .get("snapshot_mode", "stall_recovery")
        ),
        "contact_object": contact.get("contact_object"),
        "decision_index": int(group["decision_index"]),
        "branches": int(group["actions_norm"].shape[0]),
        "task_object_max_disp_mm": 1000 * task_disp,
        "incidental_max_disp_mm": 1000 * incidental_disp,
        "replay_object_linf_mm": 1000 * float(qc["object_linf"]),
        "replay_q_dispersion": float(qc["replay_q_dispersion"]),
        "q_resolved": bool(qc["q_resolved"]),
        "object_resolved": bool(qc["object_resolved"]),
        "structural_valid": bool(qc["structural_valid"]),
        "positive_flow_targets": int((group["branch_weights"] > 0).sum()),
        "predicate_flip_branches": flips,
        "dense_progress_branches": int(
            (group["dense_goal_progress"] > margin).sum()
        ),
        "effect_resolved": bool(task_disp >= floor),
        "effect_floor_mm": 1000 * floor,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            "/home/stargazer/Desktop/vla_wm/datasets/chain_branches_v1_1"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = json.loads((args.data / "branch_manifest.json").read_text())
    rows: list[dict] = []
    for split, paths in manifest["splits"].items():
        for relative in paths:
            rows.append(group_row(args.data / relative, split))

    columns = (
        "split",
        "source_seed",
        "snapshot_mode",
        "contact_object",
        "branches",
        "task_object_max_disp_mm",
        "incidental_max_disp_mm",
        "replay_object_linf_mm",
        "effect_floor_mm",
        "effect_resolved",
        "q_resolved",
        "positive_flow_targets",
        "predicate_flip_branches",
    )
    print(" | ".join(columns))
    for row in sorted(rows, key=lambda r: (r["split"], str(r["source_seed"]))):
        print(
            " | ".join(
                f"{row[c]:.2f}" if isinstance(row[c], float) else str(row[c])
                for c in columns
            )
        )

    summary: dict[str, dict] = {}
    for split in ("train", "dev", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        resolved_rows = [row for row in split_rows if row["effect_resolved"]]
        summary[split] = {
            "groups": len(split_rows),
            "effect_resolved_groups": len(resolved_rows),
            "positive_flow_targets": sum(
                row["positive_flow_targets"] for row in split_rows
            ),
            "independent_source_episodes": len(
                {row["source_trajectory_id"] for row in split_rows}
            ),
            "independent_effect_resolved_episodes": len(
                {row["source_trajectory_id"] for row in resolved_rows}
            ),
        }
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "schema": "branch_coverage_report_v1",
                    "effect_resolved_rule": (
                        f"max sibling task-object displacement >= "
                        f"max({ABS_FLOOR_M} m, {NOISE_MULTIPLE} x replay "
                        f"object linf)"
                    ),
                    "rows": rows,
                    "summary": summary,
                },
                indent=2,
            )
        )
        print(f"-> {args.output}")


if __name__ == "__main__":
    main()
