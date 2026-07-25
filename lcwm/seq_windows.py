"""Sequential predictive-state episodes and windows (P1).

Serves episodes in the LC posterior's native format from two existing
sources, with split integrity enforced by source episode:

1. prefix-cache episodes (`lcwm/seq_prefix_cache.py` artifacts): full labels;
2. chain branch-group histories (`chain_branches_v1_1`): the on-policy chain
   rollout prefix sequences already stored per group, deduplicated to ONE
   longest history per `source_trajectory_id` (several groups share a source
   episode); per-decision labels are absent there and flagged as such.

Unified episode dict fields:
  source_id, split, source_kind ("demo" | "chain"), language,
  prefix_hidden [T,P,2048] fp16, prefix_mask [T,P] bool,
  actions_norm [T-1,10,7] fp32 (block i connects record i -> i+1),
  action_exec_mask [T-1,10] bool (partial final block before a terminal),
  has_labels bool, and when has_labels: t [T], q [T,9], obj_pos [T,n,3],
  predicate_bits [T,A], success [T], terminal [T], first_success_t,
  terminal_t, stride, goal_atoms, object_names.

`iter_windows(episode, k)` yields k-step windows
(h_t, a_{t..t+k-1}, h_{t+k}, labels at both ends) for P2's self-prediction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import torch

BRANCH_DIR_DEFAULT = Path(
    "/home/stargazer/Desktop/vla_wm/datasets/chain_branches_v1_1"
)
SEALED_SPLITS = ("audit", "test")


def _episode_from_cache(path: Path) -> dict[str, Any]:
    data = torch.load(path, weights_only=False)
    T = data["prefix_hidden"].shape[0]
    stride = int(data["stride"])
    t = data["t"]
    # Block i connects record i -> i+1. The final connecting block may be
    # partial when the episode terminated inside it: executed = t[i+1]-t[i].
    executed = (t[1:] - t[:-1]).clamp(max=stride)
    exec_mask = (
        torch.arange(stride)[None, :] < executed[:, None]
    )
    return {
        "source_id": data["source_id"],
        "split": data["split"],
        "source_kind": "demo",
        "language": data["language"],
        "prefix_hidden": data["prefix_hidden"],
        "prefix_mask": data["prefix_mask"],
        "actions_norm": data["action_block_norm"][:-1].float(),
        "action_exec_mask": exec_mask,
        "has_labels": True,
        "t": t,
        "q": data["q"],
        "obj_pos": data["obj_pos"],
        "predicate_bits": data["predicate_bits"],
        "success": data["success"],
        "terminal": data["terminal"],
        "first_success_t": data["first_success_t"],
        "terminal_t": data["terminal_t"],
        "stride": stride,
        "goal_atoms": data["goal_atoms"],
        "object_names": data["object_names"],
        "path": str(path),
    }


def load_cache_episodes(
    cache_dir: Path,
    splits: tuple[str, ...],
    allow_sealed: bool = False,
) -> list[dict[str, Any]]:
    for split in splits:
        if split in SEALED_SPLITS and not allow_sealed:
            raise PermissionError(
                f"split {split!r} is sealed; pass allow_sealed=True only for "
                "a frozen final evaluation"
            )
    episodes = []
    for path in sorted(Path(cache_dir).glob("task*_demo*.pt")):
        episode = _episode_from_cache(path)
        if episode["split"] in splits:
            episodes.append(episode)
    return episodes


def load_chain_history_episodes(
    branch_dir: Path = BRANCH_DIR_DEFAULT,
    splits: tuple[str, ...] = ("train",),
    allow_sealed: bool = False,
) -> list[dict[str, Any]]:
    """One episode per unique chain source: the LONGEST stored history."""
    for split in splits:
        if split == "test" and not allow_sealed:
            raise PermissionError(
                "chain test split is sealed; pass allow_sealed=True only for "
                "a frozen final evaluation"
            )
    manifest = json.loads(
        (Path(branch_dir) / "branch_manifest.json").read_text()
    )
    source_split = manifest["source_episode_splits"]
    best: dict[str, dict[str, Any]] = {}
    for split_name, paths in manifest["splits"].items():
        if split_name not in splits:
            continue
        for relative in paths:
            group = torch.load(
                Path(branch_dir) / relative, weights_only=False
            )
            source_id = group["source_trajectory_id"]
            registered = source_split.get(source_id)
            if registered is not None and registered != split_name:
                raise ValueError(
                    f"{source_id}: manifest split {registered!r} disagrees "
                    f"with group location {split_name!r}"
                )
            hidden = group["history_prefix_hidden"]
            current = best.get(source_id)
            if current is not None and (
                current["prefix_hidden"].shape[0] >= hidden.shape[0]
            ):
                continue
            length = hidden.shape[0]
            best[source_id] = {
                "source_id": source_id,
                "split": split_name,
                "source_kind": "chain",
                "language": group["full_instruction"],
                "prefix_hidden": hidden,
                "prefix_mask": group["history_prefix_mask"],
                "actions_norm": group["history_action_norm"].float(),
                "action_exec_mask": torch.ones(
                    max(length - 1, 0),
                    group["history_action_norm"].shape[1]
                    if length > 1
                    else 10,
                    dtype=torch.bool,
                ),
                "has_labels": False,
                "stride": 10,
                "goal_atoms": group.get("goal_specs", {})
                .get("chain3_full", {})
                .get("atoms"),
                "path": str(Path(branch_dir) / relative),
            }
    return list(best.values())


def assert_split_integrity(episodes: list[dict[str, Any]]) -> dict[str, str]:
    """One source episode -> exactly one split, across all sources."""
    assignment: dict[str, str] = {}
    for episode in episodes:
        source, split = episode["source_id"], episode["split"]
        if assignment.get(source, split) != split:
            raise ValueError(
                f"source {source!r} appears in splits "
                f"{assignment[source]!r} and {split!r}"
            )
        assignment[source] = split
    return assignment


class SequentialEpisodeDataset(torch.utils.data.Dataset):
    """Episode-shaped items; windows are formed by the consumer so the LC
    state can be unrolled with cross-window context."""

    def __init__(
        self,
        cache_dir: Path | None,
        branch_dir: Path | None = BRANCH_DIR_DEFAULT,
        splits: tuple[str, ...] = ("train",),
        allow_sealed: bool = False,
    ):
        episodes: list[dict[str, Any]] = []
        if cache_dir is not None and Path(cache_dir).exists():
            episodes += load_cache_episodes(
                Path(cache_dir), splits, allow_sealed
            )
        if branch_dir is not None and Path(branch_dir).exists():
            chain_splits = tuple(
                "dev" if s == "dev" else s for s in splits
            )
            episodes += load_chain_history_episodes(
                Path(branch_dir), chain_splits, allow_sealed
            )
        episodes = [
            e for e in episodes if e["prefix_hidden"].shape[0] >= 2
        ]
        self.split_assignment = assert_split_integrity(episodes)
        self.episodes = episodes

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.episodes[index]

    def transition_count(self) -> int:
        return sum(
            e["prefix_hidden"].shape[0] - 1 for e in self.episodes
        )


def iter_windows(
    episode: dict[str, Any], k: int = 1
) -> Iterator[dict[str, Any]]:
    """k-step windows: (h_t, actions t..t+k-1, h_{t+k})."""
    T = episode["prefix_hidden"].shape[0]
    for i in range(T - k):
        window = {
            "source_id": episode["source_id"],
            "split": episode["split"],
            "index": i,
            "h_t": episode["prefix_hidden"][i],
            "mask_t": episode["prefix_mask"][i],
            "actions_norm": episode["actions_norm"][i : i + k],
            "action_exec_mask": episode["action_exec_mask"][i : i + k],
            "h_next": episode["prefix_hidden"][i + k],
            "mask_next": episode["prefix_mask"][i + k],
        }
        if episode["has_labels"]:
            window.update(
                {
                    "q_t": episode["q"][i],
                    "q_next": episode["q"][i + k],
                    "obj_pos_t": episode["obj_pos"][i],
                    "obj_pos_next": episode["obj_pos"][i + k],
                    "bits_t": episode["predicate_bits"][i],
                    "bits_next": episode["predicate_bits"][i + k],
                    "success_next": episode["success"][i + k],
                    "terminal_next": episode["terminal"][i + k],
                }
            )
        yield window
