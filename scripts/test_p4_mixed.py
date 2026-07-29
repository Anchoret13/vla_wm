#!/usr/bin/env python
"""A3.1 p4_mixed_v1 smoke (registered artifact): aligned targets from both
sequential sources + branch-sibling ΔΦ under the memo rule. CPU-only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.p4_targets import (  # noqa: E402
    assert_policy_guard,
    branch_delta_phi,
    sequential_targets,
)
from lcwm.seq_prefix_cache import CACHE_DIR  # noqa: E402
from lcwm.seq_windows import (  # noqa: E402
    BRANCH_DIR_DEFAULT,
    CHAIN_LABELS_DIR_DEFAULT,
    SequentialEpisodeDataset,
)


def main() -> None:
    train = SequentialEpisodeDataset(CACHE_DIR, splits=("train",))
    for kind in ("demo", "chain"):
        episode = next(
            e
            for e in train.episodes
            if e["source_kind"] == kind and e["has_labels"]
        )
        targets = sequential_targets(episode)
        assert_policy_guard(targets)
        finite_v = int((~torch.isnan(targets["v_pi0"])).sum())
        print(
            f"[{kind}] {targets['source_id']}: windows "
            f"{int(targets['n_windows'])}, labeled "
            f"{int(targets['window_labeled'].sum())}, dphi_valid "
            f"{int(targets['dphi_valid'].sum())}, r!=0 "
            f"{int((targets['reward'] != 0).sum())}, success@end "
            f"{bool(targets['success'][-1])}, env_terminal "
            f"{int(targets['env_terminal'].sum())}, truncated_final "
            f"{int(targets['truncated_final'].sum())}, V targets {finite_v} "
            f"({targets['generating_policy']})"
        )
        if kind == "demo":
            assert finite_v == 0, "expert episode must have no V^pi0"
            assert int(targets["env_terminal"].sum()) == 1
        else:
            assert finite_v > 0, "pi0 chain episode must have V^pi0"
            # A0 assertions (2026-07-29): transition-level alignment.
            from lcwm.value_heads import GAMMA_DECISION, value_target_mc

            v_next = targets["v_pi0"]
            v_curr = targets["v_pi0_current"]
            r = targets["reward"]
            for t in (0, 30, int(targets["n_windows"]) - 1):
                if torch.isnan(v_next[t]):
                    continue
                expected = value_target_mc(episode["sidecar_bits"], t + 1)
                assert abs(float(v_next[t]) - expected) < 1e-5, (
                    f"v_pi0[{t}] != V(s_{t+1})"
                )
                identity = float(r[t]) + GAMMA_DECISION * float(v_next[t])
                assert abs(float(v_curr[t]) - identity) < 1e-5, (
                    f"MC identity V_t = r_t + gamma*V_{{t+1}} broken at {t}"
                )
            last = int(targets["n_windows"]) - 1
            assert abs(float(v_next[last])) < 1e-9, (
                "terminal next-state value must be zero"
            )
            print(
                "A0 value-alignment assertions PASSED "
                "(V_next indexing, MC identity, zero terminal)"
            )
            assert int(targets["env_terminal"].sum()) == 0, (
                "stalled chain episode must not emit env termination"
            )
            expected_truncated = (
                1 if episode.get("full_episode_cache") else 0
            )
            assert (
                int(targets["truncated_final"].sum()) == expected_truncated
            ), "partial prefixes carry no terminal semantics"

    manifest = json.loads(
        (BRANCH_DIR_DEFAULT / "branch_manifest.json").read_text()
    )
    contact = stall = None
    for split, paths in manifest["splits"].items():
        if split == "test":
            continue
        for rel in paths:
            group = torch.load(BRANCH_DIR_DEFAULT / rel, weights_only=False)
            mode = (
                group["provenance"]
                .get("collection_contract", {})
                .get("snapshot_mode", "stall_recovery")
            )
            sidecar_path = (
                CHAIN_LABELS_DIR_DEFAULT
                / f"{group['source_trajectory_id']}.pt"
            )
            if not sidecar_path.exists():
                continue
            sidecar = torch.load(sidecar_path, weights_only=False)
            if mode == "on_policy_contact" and contact is None:
                contact = (group, sidecar, rel)
            if mode != "on_policy_contact" and stall is None:
                stall = (group, sidecar, rel)
    assert contact is not None and stall is not None

    group, sidecar, rel = contact
    dphi, valid, violation = branch_delta_phi(group, sidecar)
    print(
        f"[branch contact] {rel.split('/')[-1][:40]}: valid "
        f"{int(valid.sum())}/{len(valid)}, dphi range "
        f"[{float(dphi[valid].min()):.3f}, {float(dphi[valid].max()):.3f}], "
        f"violations {int(violation.sum())}"
    )
    assert int(valid.sum()) == len(valid), "contact siblings must be valid"
    assert float(dphi[valid].abs().max()) > 0, "sibling dphi must vary"

    group, sidecar, rel = stall
    dphi, valid, _ = branch_delta_phi(group, sidecar)
    print(
        f"[branch stall] {rel.split('/')[-1][:40]}: valid "
        f"{int(valid.sum())}/{len(valid)} (expected 0 — memo rule)"
    )
    assert int(valid.sum()) == 0, "stall siblings must be masked unavailable"
    print("A3.1 P4-MIXED TARGET SMOKE PASSED")


if __name__ == "__main__":
    main()
