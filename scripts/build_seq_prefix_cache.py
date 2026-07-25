#!/usr/bin/env python
"""Build the full-prefix sequential episode cache (P1, pre-registered
2026-07-25.md). Default scope: LIBERO-10 tasks 0 and 1 (LIVING_ROOM_SCENE2,
the chain3 scene family), train+dev roles from the SEALED v2 split manifest.
Audit demos are never built unless --roles explicitly includes audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--tasks", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--roles", nargs="+", default=["train", "dev"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if "audit" in args.roles:
        print(
            "REFUSING: audit split is sealed until the frozen final "
            "evaluation (split_manifest v2 rule).",
            file=sys.stderr,
        )
        raise SystemExit(2)

    from lcwm.chassis import Pi05Runner
    from lcwm.seq_prefix_cache import (
        CACHE_DIR,
        CONTRACT_VERSION,
        build_task_cache,
        load_split_roles,
        source_sha256,
    )

    roles_by_task = load_split_roles()
    runner = Pi05Runner(suite_name=args.suite)
    manifest_entries: list[dict] = []
    for task_id in args.tasks:
        task_roles = roles_by_task[str(task_id)]
        manifest_entries += build_task_cache(
            runner,
            args.suite,
            task_id,
            task_roles,
            roles=tuple(args.roles),
            overwrite=args.overwrite,
        )

    manifest = {
        "schema": CONTRACT_VERSION,
        "suite": args.suite,
        "tasks": args.tasks,
        "roles": args.roles,
        "source_sha256": source_sha256(),
        "episodes": manifest_entries,
    }
    out = CACHE_DIR / "cache_manifest.json"
    existing = {}
    if out.exists():
        existing = json.loads(out.read_text())
    merged = {
        entry["path"]: entry
        for entry in existing.get("episodes", []) + manifest_entries
    }
    manifest["episodes"] = sorted(merged.values(), key=lambda e: e["path"])
    out.write_text(json.dumps(manifest, indent=2))
    print(f"cache manifest -> {out} ({len(manifest['episodes'])} episodes)")


if __name__ == "__main__":
    main()
