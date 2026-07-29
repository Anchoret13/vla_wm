#!/usr/bin/env python
"""H2 — score-blind matched control teacher (v0.4-e2e-v1-random-selected-uniform).

Differs from the WM arm in exactly one place: candidate selection.
- Regenerates each deterministic N=4 pool from the saved candidate_seed and
  ASSERTS the regenerated stock and WM-selected chunks reproduce the
  recorded ones before constructing the control.
- Reuses the exact positive-state mask on which the WM arm selected a
  non-stock teacher (expected 152 states).
- At each positive state picks candidate 1-3 by a fixed hash of
  (source_id, decision_index, control_seed=0); model scores are never read.
- Advantages are stored as 0.0 so the shared trainer's 1+clip(A) rule
  yields exactly uniform weight 1.0 (recorded matching choice).
- Reports selected first-ten action distance/diversity vs the WM arm so an
  action-magnitude mismatch would be visible.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

CACHE_DIR = Path(
    "/home/stargazer/Desktop/vla_wm/datasets/chain_episode_cache_v1"
)
OUT = REPO_ROOT / "results" / "lc_flow_v04_e2e_v1_random_control"


def control_choice(source_id: str, decision: int) -> int:
    digest = hashlib.sha256(
        f"{source_id}|{decision}|control_seed=0".encode()
    ).hexdigest()
    return 1 + (int(digest, 16) % 3)


@torch.no_grad()
def main() -> None:
    from lcwm.chassis import Pi05Runner
    from lcwm.policy_interface import ReplayPolicyInterface

    teacher = torch.load(
        REPO_ROOT / "results" / "lc_flow_v04_e2e_v1"
        / "candidate_teacher.pt",
        weights_only=False,
    )
    rows = teacher["rows"]
    runner = Pi05Runner(suite_name="libero_10")
    interface = ReplayPolicyInterface(runner)
    episodes = {}
    for path in sorted(CACHE_DIR.glob("*.pt")):
        data = torch.load(path, weights_only=False)
        if data["split"] == "train":
            episodes[data["source_trajectory_id"]] = data

    control_rows = []
    positive = 0
    distances_control, distances_wm = [], []
    for row in rows:
        episode = episodes[row["source_id"]]
        candidates = interface.candidates_at(
            episode, row["decision"], n=3, seed=row["candidate_seed"]
        ).cpu().float()
        stock = episode["chunks_norm"][row["decision"]].float()
        pool = torch.cat([stock[None], candidates])
        # Regeneration assertions against the recorded artifact.
        assert torch.allclose(
            pool[0], row["stock_chunk"].float(), atol=1e-5
        ), f"stock chunk mismatch at {row['source_id']}:{row['decision']}"
        assert torch.allclose(
            pool[row["selected"]], row["selected_chunk"].float(), atol=1e-5
        ), f"selected chunk mismatch at {row['source_id']}:{row['decision']}"

        new = dict(row)
        if row["selected"] != 0:  # exact WM-positive mask reuse
            positive += 1
            choice = control_choice(row["source_id"], row["decision"])
            new["selected"] = choice
            new["selected_chunk"] = pool[choice]
            new["advantage"] = 0.0  # -> trainer weight exactly 1.0
            distances_control.append(
                float((pool[choice, :10] - pool[0, :10]).norm())
            )
            distances_wm.append(
                float(
                    (row["selected_chunk"].float()[:10] - pool[0, :10])
                    .norm()
                )
            )
        else:
            new["selected"] = 0
            new["advantage"] = 0.0
        control_rows.append(new)

    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "candidate_teacher_v1",
            "rows": control_rows,
            "wm_checkpoint_sha256": teacher["wm_checkpoint_sha256"],
            "control": "score_blind_hash_choice_on_wm_positive_mask",
        },
        OUT / "teacher_manifest.pt",
    )
    diagnostics = {
        "positive_states": positive,
        "mean_first10_distance_control": sum(distances_control)
        / max(len(distances_control), 1),
        "mean_first10_distance_wm": sum(distances_wm)
        / max(len(distances_wm), 1),
        "regeneration_assertions": "all passed",
    }
    (OUT / "control_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2)
    )
    print(json.dumps(diagnostics, indent=2))
    print(f"-> {OUT / 'teacher_manifest.pt'}")


if __name__ == "__main__":
    main()
