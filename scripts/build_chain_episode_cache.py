#!/usr/bin/env python
"""A2 cache build (registered 2026-07-27). Deterministic feature capture over
the already-used stock chain episodes — NOT new interaction. Train+dev
sources by default; the test-split source requires --include-test."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

BRANCH_DIR = Path(
    "/home/stargazer/Desktop/vla_wm/datasets/chain_branches_v1_1"
)
SIDECAR_DIR = Path(
    "/home/stargazer/Desktop/vla_wm/datasets/chain_source_labels_v1"
)
SOURCE_RE = re.compile(r"chain3_lr2-full-seed(\d+)-src(\d+)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from lcwm.chain_episode_cache import (
        CHAIN_EPISODE_CACHE_DIR,
        CONTRACT_VERSION,
        build_chain_episode,
    )
    from lcwm.chassis import Pi05Runner
    from lcwm.loho import make_chain_env

    manifest = json.loads(
        (BRANCH_DIR / "branch_manifest.json").read_text()
    )
    source_splits = manifest["source_episode_splits"]
    CHAIN_EPISODE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    runner = Pi05Runner(suite_name="libero_10")
    built = []
    for source_id in sorted(source_splits):
        split = source_splits[source_id]
        if split == "test" and not args.include_test:
            print(f"[skip] {source_id}: test split sealed", flush=True)
            continue
        match = SOURCE_RE.fullmatch(source_id)
        if match is None:
            print(f"[skip] unparseable {source_id}", flush=True)
            continue
        out = CHAIN_EPISODE_CACHE_DIR / f"{source_id}.pt"
        if out.exists() and not args.overwrite:
            print(f"[skip] exists {out.name}", flush=True)
            built.append(out.name)
            continue
        seed, noise = int(match.group(1)), int(match.group(2))
        sidecar_path = SIDECAR_DIR / f"{source_id}.pt"
        sidecar = (
            torch.load(sidecar_path, weights_only=False)
            if sidecar_path.exists()
            else None
        )
        env = make_chain_env("chain3_lr2")
        try:
            record = build_chain_episode(
                runner, env, seed, noise, sidecar=sidecar
            )
        finally:
            env.close()
        record["source_trajectory_id"] = source_id
        record["split"] = split
        torch.save(record, out)
        built.append(out.name)
        print(
            f"[cache] {source_id} ({split}): "
            f"{record['prefix_hidden'].shape[0]} records, "
            f"sidecar-checked={sidecar is not None} -> {out.name}",
            flush=True,
        )
    (CHAIN_EPISODE_CACHE_DIR / "cache_manifest.json").write_text(
        json.dumps(
            {
                "schema": CONTRACT_VERSION,
                "episodes": sorted(built),
                "split_source": "branch_manifest.source_episode_splits",
            },
            indent=2,
        )
    )
    print(f"-> {CHAIN_EPISODE_CACHE_DIR} ({len(built)} episodes)")


if __name__ == "__main__":
    main()
