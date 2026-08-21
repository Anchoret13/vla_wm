"""Tensor contract for the fresh V8.2 bootstrap world-model bank.

The Action-2P rows are calibration-only summaries and cannot train a model.
This module defines the small, source-disjoint tensor object collected afresh
for M0.  One record is one anchor with all of its sibling candidates and
nested continuation repeats.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import torch


SCHEMA = "v082_bootstrap_group_v1"
GAMMA = 0.99
SPLIT_ROLES = {
    "train": "model_train",
    "val": "model_calib",
    "test": "selector_assess",
}
REPEATS = {"train": 1, "val": 1, "test": 3}
REQUIRED_FIELDS = {
    "schema", "anchor_id", "source_id", "split", "role",
    "prefix_hidden", "prefix_mask", "anchor_signature", "actions_norm",
    "actions_env", "candidate_ids", "is_reference", "post_prefix_delta",
    "immediate_bits", "outcome_h80", "repeat_mask",
}


def continuation_outcome(
    achieved_steps: Mapping[int, int],
    success_step: int | None,
    damage: int | float,
    tau: int,
    c_eff: int,
    horizon: int,
) -> dict[str, float]:
    """Outcome whose clock starts after the executed action prefix.

    Events during ``tau < t <= tau+c_eff`` are immediate transition effects
    and deliberately excluded from the continuation progress/return target.
    """

    start = int(tau) + int(c_eff)
    end = start + int(horizon)
    gains = {i: int(step) for i, step in achieved_steps.items()
             if start < int(step) <= end}
    ttm = min(gains.values()) - start if gains else int(horizon) + 1
    ret = sum(GAMMA ** (step - start) for step in gains.values())
    return {
        "dmg": float(damage),
        "succ": float(success_step is not None and int(success_step) <= end),
        "dp": float(len(gains)),
        "ttm": float(ttm),
        "G": float(ret),
    }


def assert_role_allowed(role: str, *, subrole: str | None = None) -> None:
    if role not in set(SPLIT_ROLES.values()):
        detail = f"/{subrole}" if subrole else ""
        raise ValueError(f"role {role}{detail} cannot enter the bootstrap bank")


def assert_split_role(split: str, role: str) -> None:
    if split not in SPLIT_ROLES:
        raise ValueError(f"unknown bootstrap split {split!r}")
    assert_role_allowed(role)
    if SPLIT_ROLES[split] != role:
        raise ValueError(
            f"split {split!r} requires role {SPLIT_ROLES[split]!r}, got {role!r}"
        )


def _tensor(group: dict, key: str) -> torch.Tensor:
    value = group[key]
    if not torch.is_tensor(value):
        raise TypeError(f"{key} must be a torch.Tensor")
    return value


def validate_bootstrap_group(group: dict) -> None:
    missing = REQUIRED_FIELDS - set(group)
    if missing:
        raise KeyError(f"missing bootstrap fields: {sorted(missing)}")
    if group["schema"] != SCHEMA:
        raise ValueError(f"unexpected schema {group['schema']!r}")
    split, role = str(group["split"]), str(group["role"])
    assert_split_role(split, role)

    hidden = _tensor(group, "prefix_hidden")
    mask = _tensor(group, "prefix_mask")
    signature = _tensor(group, "anchor_signature")
    actions_norm = _tensor(group, "actions_norm")
    actions_env = _tensor(group, "actions_env")
    is_reference = _tensor(group, "is_reference")
    delta = _tensor(group, "post_prefix_delta")
    bits = _tensor(group, "immediate_bits")
    outcomes = _tensor(group, "outcome_h80")
    repeat_mask = _tensor(group, "repeat_mask")

    if hidden.ndim != 2 or hidden.shape[1] != 2048:
        raise ValueError(f"prefix_hidden must be [P,2048], got {tuple(hidden.shape)}")
    if mask.shape != hidden.shape[:1] or mask.dtype != torch.bool or not bool(mask.any()):
        raise ValueError("prefix_mask must be nonempty BoolTensor[P]")
    if signature.shape != (24,):
        raise ValueError(f"anchor_signature must be [24], got {tuple(signature.shape)}")
    if actions_norm.ndim != 3 or tuple(actions_norm.shape[1:]) != (10, 7):
        raise ValueError(f"actions_norm must be [C,10,7], got {tuple(actions_norm.shape)}")
    if actions_env.shape != actions_norm.shape:
        raise ValueError("actions_env shape must match actions_norm exactly")
    candidates = int(actions_norm.shape[0])
    if candidates != 7:
        raise ValueError(f"expected 7 sibling candidates, got {candidates}")
    if len(group["candidate_ids"]) != candidates:
        raise ValueError("candidate_ids length does not match actions")
    if is_reference.shape != (candidates,) or int(is_reference.bool().sum()) != 1:
        raise ValueError("is_reference must identify exactly one of C candidates")
    repeats = REPEATS[split]
    if delta.shape != (candidates, repeats, 24):
        raise ValueError(f"post_prefix_delta must be [C,{repeats},24]")
    if bits.shape != (candidates, repeats, 2):
        raise ValueError(f"immediate_bits must be [C,{repeats},2]")
    if outcomes.shape != (candidates, repeats, 5):
        raise ValueError(f"outcome_h80 must be [C,{repeats},5]")
    if repeat_mask.shape != (candidates, repeats) or repeat_mask.dtype != torch.bool:
        raise ValueError(f"repeat_mask must be BoolTensor[C,{repeats}]")
    for key in ("prefix_hidden", "anchor_signature", "actions_norm", "actions_env",
                "post_prefix_delta", "outcome_h80"):
        value = _tensor(group, key)
        if not bool(torch.isfinite(value.float()).all()):
            raise ValueError(f"{key} contains non-finite values")


def validate_groups(groups: Iterable[dict]) -> list[dict]:
    groups = list(groups)
    seen: dict[str, str] = {}
    anchors: set[str] = set()
    for group in groups:
        validate_bootstrap_group(group)
        source, split = str(group["source_id"]), str(group["split"])
        if source in seen and seen[source] != split:
            raise ValueError(f"source {source!r} leaks across {seen[source]!r}/{split!r}")
        seen[source] = split
        anchor = str(group["anchor_id"])
        if anchor in anchors:
            raise ValueError(f"duplicate anchor_id {anchor!r}")
        anchors.add(anchor)
    return groups


def split_by_source(groups: Iterable[dict], split_sources: Mapping[str, set[str]]) -> list[dict]:
    owners: dict[str, str] = {}
    for split, sources in split_sources.items():
        if split not in SPLIT_ROLES:
            raise ValueError(f"unknown split {split!r}")
        for source in sources:
            if source in owners:
                raise ValueError(f"source {source!r} assigned to two splits")
            owners[source] = split
    selected = []
    for group in groups:
        source = str(group["source_id"])
        if source not in owners:
            continue
        item = dict(group)
        item["split"] = owners[source]
        item["role"] = SPLIT_ROLES[owners[source]]
        selected.append(item)
    return validate_groups(selected)


def save_bootstrap_group(group: dict, path: Path) -> None:
    validate_bootstrap_group(group)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(group, path)


def load_bootstrap_group(path: Path) -> dict:
    group = torch.load(Path(path), map_location="cpu", weights_only=False)
    validate_bootstrap_group(group)
    return group

