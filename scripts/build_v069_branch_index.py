#!/usr/bin/env python
"""V6.9.0 — one deduplicated branch index over ALL 135 audited groups
(60 sibling-CRN continued + 75 immediate-only backlog), plus the v069
lineage binding (pb1/pb2 manifests, backlog selection, support report,
all by hash; non-overwriting).

Asserts exactly one immediate physical/crossed-semantic row per
(source, decision, branch, GoalSpec); continued and backlog group sets
are disjoint and their union covers every phase-A audit.

Output:
  results/libero_loho_public_v1/v069_branch_index.json
  results/libero_loho_public_v1/v069_lineage.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.v067_lineage import load_v067, sha256_file  # noqa: E402

DATA = Path("/home/stargazer/Desktop/vla_wm/datasets/libero_loho_public_v1"
            "/v06_effect_crossed")
RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"


def main() -> None:
    # ---- audit universe from phase-A sources ---------------------------
    audit_universe = set()
    for sp in sorted((DATA / "sources").glob("*.pt")):
        src = torch.load(sp, weights_only=False)
        for a in src["audits"]:
            audit_universe.add((src["source_id"], a["decision"]))

    seen_rows = set()
    continued, backlog = {}, {}
    for cp in sorted((DATA / "continuations_v067").glob("*.pt")):
        cont = load_v067(cp, "continuations")
        key = (cont["source_id"], cont["decision"])
        assert key not in continued, f"duplicate continued group {key}"
        continued[key] = {"file": cp.name, "run_id": cont["run_id"],
                          "split": cont["split"],
                          "goals": cont["goals"]}
        for b in cont["branch_summaries"]:
            for gid in cont["goals"]:
                row = (key[0], key[1], b["kind"], b.get("candidate"),
                       gid)
                assert row not in seen_rows, f"duplicate row {row}"
                seen_rows.add(row)
    n_shadowed = 0
    for bp in sorted((DATA / "backlog_relabels_v067").glob("*.pt")):
        lab = load_v067(bp, "semantic_labels")
        for g in lab["groups"]:
            key = (lab["source_id"], g["decision"])
            if key in continued:
                # pb2-selected group: the continuation file is the
                # single source of its immediate rows; the backlog copy
                # is shadowed (dedup, not error)
                n_shadowed += 1
                continue
            assert key not in backlog, f"duplicate backlog group {key}"
            backlog[key] = {"file": bp.name, "split": lab["split"],
                            "goals": lab["goals"]}
            for b in g["branch_summaries"]:
                for gid in lab["goals"]:
                    row = (key[0], key[1], b["kind"],
                           b.get("candidate"), gid)
                    assert row not in seen_rows, f"duplicate row {row}"
                    seen_rows.add(row)

    union = set(continued) | set(backlog)
    missing = audit_universe - union
    extra = union - audit_universe
    assert not missing, f"audited groups without labels: {missing}"
    assert not extra, f"labeled groups outside audit universe: {extra}"
    assert len(union) == len(audit_universe)

    index = {
        "schema": "v069_branch_index_v1", "run_schema": "v067",
        "n_audited_groups": len(audit_universe),
        "n_continued": len(continued), "n_backlog": len(backlog),
        "n_backlog_rows_shadowed_by_continuations": n_shadowed,
        "n_branch_goal_rows": len(seen_rows),
        "continued": {f"{k[0]}_d{k[1]}": v
                      for k, v in sorted(continued.items())},
        "backlog": {f"{k[0]}_d{k[1]}": v
                    for k, v in sorted(backlog.items())},
    }
    idx_path = RESULTS / "v069_branch_index.json"
    idx_path.write_text(json.dumps(index, indent=2, sort_keys=True))
    print(f"branch index: {len(audit_universe)} groups "
          f"({len(continued)} continued + {len(backlog)} backlog), "
          f"{len(seen_rows)} (source,decision,branch,goal) rows",
          flush=True)

    lineage = {
        "schema": "v069_lineage_v1", "run_schema": "v067",
        "bound_by_hash": {
            str(p.relative_to(REPO_ROOT)): sha256_file(p) for p in [
                RESULTS / "v067_lineage" / "iter1_freeze.json",
                RESULTS / "v067_lineage"
                / "phase_b_manifest_v067_pb1.json",
                RESULTS / "v067_lineage"
                / "phase_b_manifest_v067_pb2.json",
                RESULTS / "v067_support_report.json",
                RESULTS / "v067_backlog_selection.json",
                RESULTS / "goal_spec_manifest_v067.json",
                idx_path,
            ]},
        "policy": "non-overwriting; v069 artifacts get new namespaces "
                  "(v069_predictive/, v069_policy/, v069_eval/)",
    }
    lin_path = RESULTS / "v069_lineage.json"
    lin_path.write_text(json.dumps(lineage, indent=2, sort_keys=True))
    print(f"-> {idx_path}\n-> {lin_path}", flush=True)


if __name__ == "__main__":
    main()
