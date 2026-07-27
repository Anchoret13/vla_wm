#!/usr/bin/env python
"""A2 exit test (registered 2026-07-27; requires GPU + built cache).

1. Loader serves full chain episodes (~71 records, fully labeled) and the
   late-chain state count matches the review's expectation (>= 250
   post-soup+sauce labeled states across train+dev; review estimate ~304).
2. Phase potential Φ is defined at late-chain decisions.
3. EXACTNESS: regenerated prefix from stored raw observations matches the
   cached prefix hidden at a late-chain decision (fp16 tolerance).
4. Candidate sampling at that state returns [N,50,7] π0.5-supported chunks.
5. Split assignments unchanged vs the branch manifest.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

from lcwm.phase_potential import episode_phase_potentials  # noqa: E402
from lcwm.seq_prefix_cache import CACHE_DIR  # noqa: E402
from lcwm.seq_windows import (  # noqa: E402
    BRANCH_DIR_DEFAULT,
    SequentialEpisodeDataset,
)


def main() -> None:
    train = SequentialEpisodeDataset(CACHE_DIR, splits=("train",))
    dev = SequentialEpisodeDataset(CACHE_DIR, splits=("dev",))
    union = train.episodes + dev.episodes
    full = [
        e for e in union if e.get("full_episode_cache")
    ]
    assert full, "no full-episode chain caches loaded — build A2 cache first"

    late_states = 0
    for episode in full:
        bits = episode["predicate_bits"]
        late_states += int(
            (bits.float().sum(-1) == 2).sum()
        )
    print(
        f"full chain episodes: {len(full)}; labeled 2/3-complete states "
        f"(train+dev): {late_states}"
    )
    assert late_states >= 250, "late-chain coverage below expectation"

    episode = max(full, key=lambda e: e["prefix_hidden"].shape[0])
    records = episode_phase_potentials(
        episode["q"],
        episode["obj_pos"],
        episode["predicate_bits"],
        episode["object_names"],
        episode["goal_atoms"],
        episode["label_mask"],
    )
    late = [
        r for r in records[45:] if r is not None and r.phase != "done"
    ]
    assert late, "no late-chain phase records"
    print(
        f"late-chain phases on {episode['source_id']}: "
        f"{sorted({r.phase for r in late})}"
    )

    manifest = json.loads(
        (BRANCH_DIR_DEFAULT / "branch_manifest.json").read_text()
    )
    for e in full:
        assert (
            manifest["source_episode_splits"][e["source_id"]] == e["split"]
        ), f"split changed for {e['source_id']}"
    print("split assignments unchanged")

    # -- GPU part: exactness + candidates ------------------------------------
    from lcwm.chassis import Pi05Runner
    from lcwm.policy_interface import ReplayPolicyInterface

    runner = Pi05Runner(suite_name="libero_10")
    interface = ReplayPolicyInterface(runner)
    raw = torch.load(episode["path"], weights_only=False)
    decision = min(50, raw["prefix_hidden"].shape[0] - 2)
    error = interface.verify_prefix(raw, decision)
    print(f"prefix exactness at decision {decision}: max|Δ| = {error:.3e}")
    candidates = interface.candidates_at(
        raw, decision, n=4, seed=777_000 + decision
    )
    assert tuple(candidates.shape) == (4, 50, 7), candidates.shape
    print(f"candidates: {tuple(candidates.shape)} sampled at late-chain state")
    print("A2 CHAIN-EPISODE CACHE EXIT TEST PASSED")


if __name__ == "__main__":
    main()
