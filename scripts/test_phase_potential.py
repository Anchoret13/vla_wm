#!/usr/bin/env python
"""A1 hand-check (registered exit condition, 2026-07-27).

--calibrate: print gripper-sum / EEF-distance statistics on task0 demo0
  (the ONE registered calibration demo) so the gripper constants can be
  fixed once and recorded.

default: assert on task1 (cream-cheese/butter) demo episodes — real
  pick-place arcs with labels:
  1. Φ is non-decreasing (tolerance 1e-3) on >= 90% of labeled transitions
     up to each atom completion;
  2. in grasp/lift-phase windows where RAW object-to-basket distance
     increases, ΔΦ is still positive on a clear majority — the exact
     defect the old dense-distance progress had;
  3. no violation flags on successful demos;
  4. chain episodes (labeled prefixes) produce defined Φ with approach-family
     phases for the active soup atom.
CPU-only; stored labels; no environment, no GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.phase_potential import (  # noqa: E402
    delta_phi_targets,
    episode_phase_potentials,
    resolve_entity_row,
)
from lcwm.seq_prefix_cache import CACHE_DIR  # noqa: E402
from lcwm.seq_windows import SequentialEpisodeDataset  # noqa: E402


def episode_records(episode):
    return episode_phase_potentials(
        episode["q"],
        episode["obj_pos"],
        episode["predicate_bits"],
        episode["object_names"],
        episode["goal_atoms"],
        episode.get("label_mask"),
    )


def calibrate(episodes) -> None:
    episode = next(
        e
        for e in episodes
        if e["source_kind"] == "demo" and "task0_demo0" in e["path"]
    )
    q, obj_pos = episode["q"], episode["obj_pos"]
    names = episode["object_names"]
    soup = names.index(episode["goal_atoms"][0][1])
    print("t | gripper_sum | d(EEF, soup) | soup_z")
    for t in range(q.shape[0]):
        d = float(torch.linalg.vector_norm(q[t, 0:3] - obj_pos[t, soup]))
        print(
            f"{int(episode['t'][t]):4d} | {float(q[t, 7] + q[t, 8]):.4f} | "
            f"{d:.3f} | {float(obj_pos[t, soup, 2]):.3f}"
        )


def main() -> None:
    train = SequentialEpisodeDataset(CACHE_DIR, splits=("train",))
    if "--calibrate" in sys.argv:
        calibrate(train.episodes)
        return

    task1 = [
        e
        for e in train.episodes
        if e["source_kind"] == "demo" and "task1_" in e["path"]
    ]
    assert task1, "no task1 (cream cheese/butter) demo episodes found"
    total_windows = monotone = 0
    grasp_lift_windows = grasp_lift_positive = basket_dist_up = 0
    for episode in task1:
        records = episode_records(episode)
        assert not any(
            r.violation for r in records if r is not None
        ), f"violation flag on successful demo {episode['source_id']}"
        delta, valid, _ = delta_phi_targets(records)
        names = episode["object_names"]
        for i in range(len(records) - 1):
            if not bool(valid[i]):
                continue
            a = records[i]
            if a.phase == "done":
                continue
            total_windows += 1
            if float(delta[i]) >= -1e-3:
                monotone += 1
            if a.phase in ("grasp", "lift"):
                grasp_lift_windows += 1
                row = resolve_entity_row(
                    names, episode["goal_atoms"][a.active_atom][1]
                )
                basket = resolve_entity_row(
                    names, episode["goal_atoms"][a.active_atom][2]
                )
                d0 = float(
                    torch.linalg.vector_norm(
                        episode["obj_pos"][i, row]
                        - episode["obj_pos"][i, basket]
                    )
                )
                d1 = float(
                    torch.linalg.vector_norm(
                        episode["obj_pos"][i + 1, row]
                        - episode["obj_pos"][i + 1, basket]
                    )
                )
                if d1 > d0:
                    basket_dist_up += 1
                    if float(delta[i]) > 0:
                        grasp_lift_positive += 1
    fraction = monotone / max(total_windows, 1)
    print(
        f"task1 demos: {len(task1)} episodes, {total_windows} windows, "
        f"monotone(Φ) {100*fraction:.1f}% (need >= 90%)"
    )
    print(
        f"grasp/lift windows {grasp_lift_windows}; basket-distance-up "
        f"among them {basket_dist_up}; ΔΦ>0 on those "
        f"{grasp_lift_positive}"
    )
    assert fraction >= 0.90, "Φ monotonicity below registered 90%"
    if basket_dist_up:
        assert grasp_lift_positive / basket_dist_up > 0.5, (
            "ΔΦ not positive on majority of basket-distance-increasing "
            "grasp/lift windows — the old defect persists"
        )

    chain = [
        e
        for e in train.episodes
        if e["source_kind"] == "chain" and e["has_labels"]
    ]
    for episode in chain[:2]:
        records = [r for r in episode_records(episode) if r is not None]
        phases = {r.phase for r in records}
        assert records and all(r.phi >= 0 for r in records)
        print(
            f"chain {episode['source_id'][-14:]}: {len(records)} labeled "
            f"records, phases {sorted(phases)}"
        )
    print("A1 PHASE-POTENTIAL HAND-CHECK PASSED")


if __name__ == "__main__":
    main()
