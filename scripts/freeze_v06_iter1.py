#!/usr/bin/env python
"""V6.7.0 — freeze iteration-1 evidence and write the v067 lineage
manifest.

Freezes SHA-256 for every iteration-1 source, continuation, label,
feature, WM, teacher, policy, and evaluation artifact
(-> results/libero_loho_public_v1/v067_lineage/iter1_freeze.json).

Writes manifest_v067.json into the dataset namespace: every REUSED
phase-A source/group/feature artifact referenced by hash; every
REGENERATED downstream namespace listed; iteration-1 schemas that a
repaired job must refuse. Historical artifacts are not deleted or
overwritten.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.v067_lineage import (ITER1_FORBIDDEN_SCHEMAS,  # noqa: E402
                               RUN_SCHEMA, sha256_file)

RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
DATA = Path("/home/stargazer/Desktop/vla_wm/datasets/libero_loho_public_v1"
            "/v06_effect_crossed")
LINEAGE = RESULTS / "v067_lineage"

ITER1_GLOBS = {
    "sources": (DATA / "sources", "*.pt"),
    "continuations_iter1": (DATA / "continuations", "*.pt"),
    "labels_iter1": (DATA / "labels", "*.pt"),
    "features": (DATA / "features", "*.pt"),
    "v06_wm": (RESULTS / "v06_wm", "**/*"),
    "v06_teachers": (RESULTS / "v06_teachers", "**/*"),
    "v06_policy": (RESULTS / "v06_policy", "**/*"),
    "v06_eval": (RESULTS / "v06_eval", "**/*"),
}
ITER1_JSON = [
    RESULTS / "goal_spec_manifest.json",
    RESULTS / "v06_group_selection.json",
    RESULTS / "v06_collection_manifest.json",
]
REUSED_KEYS = ("sources", "features")   # reusable by hash reference
REGENERATED_NAMESPACES = {
    "continuations": str(DATA / "continuations_v067"),
    "semantic_relabels": str(DATA / "semantic_relabels_v067"),
    "history_contrasts": str(DATA / "history_contrasts_v067"),
    "wm": str(RESULTS / "v067_wm"),
    "teachers": str(RESULTS / "v067_teachers"),
    "policy": str(RESULTS / "v067_policy"),
    "eval": str(RESULTS / "v067_eval"),
}


def main() -> None:
    LINEAGE.mkdir(parents=True, exist_ok=True)
    freeze: dict[str, dict] = {}
    for key, (root, pattern) in ITER1_GLOBS.items():
        if not root.exists():
            raise SystemExit(f"missing iteration-1 namespace: {root}")
        entries = {}
        for p in sorted(root.glob(pattern)):
            if p.is_file():
                entries[str(p.relative_to(root))] = {
                    "sha256": sha256_file(p), "bytes": p.stat().st_size}
        assert entries, f"empty iteration-1 namespace: {root}"
        freeze[key] = {"root": str(root), "files": entries,
                       "n_files": len(entries)}
        print(f"[freeze] {key}: {len(entries)} files", flush=True)
    for p in ITER1_JSON:
        freeze[p.name] = {"root": str(p.parent),
                          "files": {p.name: {"sha256": sha256_file(p),
                                             "bytes": p.stat().st_size}},
                          "n_files": 1}

    freeze_path = LINEAGE / "iter1_freeze.json"
    freeze_path.write_text(json.dumps(
        {"schema": "v067_iter1_freeze", "run_schema": RUN_SCHEMA,
         "frozen_on": date.today().isoformat(),
         "namespaces": freeze}, indent=2, sort_keys=True))
    print(f"-> {freeze_path}", flush=True)

    manifest = {
        "schema": "v067_lineage_manifest", "run_schema": RUN_SCHEMA,
        "written_on": date.today().isoformat(),
        "iter1_freeze_sha256": sha256_file(freeze_path),
        "reused_by_hash": {
            key: {rel: meta["sha256"]
                  for rel, meta in freeze[key]["files"].items()}
            for key in REUSED_KEYS},
        "reused_json_by_hash": {
            p.name: freeze[p.name]["files"][p.name]["sha256"]
            for p in ITER1_JSON},
        "regenerated_namespaces": REGENERATED_NAMESPACES,
        "historical_only": ["continuations_iter1", "labels_iter1",
                            "v06_wm", "v06_teachers", "v06_policy",
                            "v06_eval"],
        "forbidden_schemas": sorted(ITER1_FORBIDDEN_SCHEMAS),
        "policy": ("historical artifacts are never deleted or "
                   "overwritten; any reused physical/feature artifact is "
                   "referenced here by hash, not copied and renamed"),
    }
    out = DATA / "manifest_v067.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
