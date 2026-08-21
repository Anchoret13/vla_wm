"""Minimal V8.2 counterfactual world model and ``groups.pt`` loader.

The module intentionally has no simulator or PI0 dependency.  The collector
stores frozen PI0 prefix hidden states; this trainer consumes only their
masked mean and the first ten normalized executed actions.  Splits are owned
by source episodes, never by candidate rows or repeats.

Canonical collector group (one dict per anchor)::

    anchor_id: str
    source_id: str
    split: "train" | "calib" | "val" | "test"
    prefix_hidden: FloatTensor[P, D]       # frozen PI0 prefix output
    prefix_mask: BoolTensor[P]
    actions_norm: FloatTensor[C, 10, A]
    candidate_ids: list[str]
    is_reference: BoolTensor[C]            # exactly one reference candidate
    post_prefix_delta: FloatTensor[C, R, F]
    outcome_h80: FloatTensor[C, R, 5]      # dmg, succ, dp, ttm, G
    repeat_mask: BoolTensor[C, R]

``anchor_signature``, ``actions_env`` and ``immediate_bits`` may be present
but are deliberately not model inputs.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


OUTCOME_NAMES = ("dmg", "succ", "dp", "ttm", "G")
# Damage is handled separately as a one-sided safety veto.  Less damage does
# not win by itself; worse damage loses before task components are compared.
OUTCOME_DIRECTIONS = (0.0, 1.0, 1.0, -1.0, 1.0)
# Same task-automaton resolution used by the finite-horizon collector.  These
# are not fitted to train or validation outcomes.
DEFAULT_RANK_FLOORS = (0.0, 0.0, 0.0, 10.0, 1.0 - 0.99**10)
SPLIT_ALIASES = {
    "train": "train",
    "model_train": "train",
    "calib": "val",
    "model_calib": "val",
    "validation": "val",
    "valid": "val",
    "dev": "val",
    "val": "val",
    "test": "test",
}


def _tensor(value: Any, *, dtype: torch.dtype | None = None) -> Tensor:
    value = value if torch.is_tensor(value) else torch.as_tensor(value)
    return value.to(dtype=dtype) if dtype is not None else value


def masked_prefix_mean(hidden: Tensor, mask: Tensor) -> Tensor:
    """Return one frozen PI0 state vector from ``[P,D]`` prefix tokens.

    A collector may retain a history as ``[L,P,D]``.  The model input is the
    current (last) prefix only; earlier prefixes are not silently pooled into
    the token axis.
    """

    hidden = _tensor(hidden, dtype=torch.float32)
    mask = _tensor(mask).bool()
    if hidden.ndim == 3:
        hidden = hidden[-1]
        if mask.ndim == 2:
            mask = mask[-1]
    if hidden.ndim != 2 or mask.ndim != 1:
        raise ValueError(
            f"prefix_hidden/mask must be [P,D]/[P], got "
            f"{tuple(hidden.shape)}/{tuple(mask.shape)}"
        )
    if hidden.shape[0] != mask.shape[0]:
        raise ValueError("prefix mask length does not match token count")
    if not bool(mask.any()):
        raise ValueError("prefix mask has no valid token")
    if not torch.isfinite(hidden[mask]).all():
        raise ValueError("valid prefix tokens contain non-finite values")
    weights = mask.to(hidden.dtype)[:, None]
    return (hidden * weights).sum(0) / weights.sum()


@dataclass
class M0Group:
    anchor_id: str
    source_id: str
    split: str
    state: Tensor                         # [D]
    actions: Tensor                       # [C,10,A]
    physical: Tensor                      # [C,R,F]
    outcome: Tensor                       # [C,R,5]
    repeat_mask: Tensor                   # [C,R]
    is_reference: Tensor                  # [C]
    candidate_ids: tuple[str, ...]
    rank_floors: Tensor                   # [5]

    @property
    def candidates(self) -> int:
        return int(self.actions.shape[0])

    @property
    def repeats(self) -> int:
        return int(self.physical.shape[1])

    @property
    def reference_index(self) -> int:
        indices = torch.nonzero(self.is_reference, as_tuple=False).flatten()
        if indices.numel() != 1:
            raise ValueError(
                f"{self.anchor_id}: expected exactly one reference candidate, "
                f"got {indices.numel()}"
            )
        return int(indices.item())

    def to(self, device: torch.device | str) -> "M0Group":
        return M0Group(
            anchor_id=self.anchor_id,
            source_id=self.source_id,
            split=self.split,
            state=self.state.to(device),
            actions=self.actions.to(device),
            physical=self.physical.to(device),
            outcome=self.outcome.to(device),
            repeat_mask=self.repeat_mask.to(device),
            is_reference=self.is_reference.to(device),
            candidate_ids=self.candidate_ids,
            rank_floors=self.rank_floors.to(device),
        )


def _field(record: dict[str, Any], names: Sequence[str], *, required=True):
    for name in names:
        if name in record:
            return record[name]
    if required:
        raise KeyError(f"missing required field; tried {list(names)}")
    return None


def _split_name(value: Any) -> str:
    key = str(value).lower()
    if key not in SPLIT_ALIASES:
        raise ValueError(f"unknown split {value!r}")
    return SPLIT_ALIASES[key]


def _candidate_repeat_tensor(value: Any, candidates: int, name: str) -> Tensor:
    value = _tensor(value, dtype=torch.float32)
    if value.ndim == 1 and candidates == 1:
        value = value[None, None]
    elif value.ndim == 2:
        # Canonical R=1 shorthand is [C,F].  Candidate-row shorthand is
        # [R,F] with C=1.
        value = value[None] if candidates == 1 and value.shape[0] != 1 \
            else value[:, None]
    if value.ndim != 3 or value.shape[0] != candidates:
        raise ValueError(
            f"{name} must be [C,R,F], got {tuple(value.shape)} for C={candidates}"
        )
    return value


def _normalize_repeat_mask(value: Any, candidates: int, repeats: int) -> Tensor:
    if value is None:
        return torch.ones(candidates, repeats, dtype=torch.bool)
    mask = _tensor(value).bool()
    if mask.ndim == 1:
        if mask.numel() == candidates and repeats == 1:
            mask = mask[:, None]
        elif mask.numel() == repeats and candidates == 1:
            mask = mask[None]
    if tuple(mask.shape) != (candidates, repeats):
        raise ValueError(
            f"repeat_mask must be {(candidates, repeats)}, got {tuple(mask.shape)}"
        )
    return mask


def _parse_group(record: dict[str, Any]) -> M0Group:
    anchor_id = str(_field(record, ("anchor_id", "group_id", "snapshot_id")))
    source_id = str(_field(record, ("source_id", "source_episode_id",
                                    "source_trajectory_id")))
    split = _split_name(_field(record, ("split", "role")))
    hidden = _field(record, ("prefix_hidden", "anchor_prefix_hidden",
                             "history_prefix_hidden"))
    mask = _field(record, ("prefix_mask", "anchor_prefix_mask",
                           "history_prefix_mask"))
    state = masked_prefix_mean(hidden, mask)

    actions = _tensor(_field(record, ("actions_norm", "executed_actions_norm",
                                             "first10_actions_norm")),
                      dtype=torch.float32)
    if actions.ndim == 2:
        actions = actions[None]
    if actions.ndim != 3 or actions.shape[1] < 10:
        raise ValueError(
            f"{anchor_id}: actions_norm must be [C,T>=10,A], got "
            f"{tuple(actions.shape)}"
        )
    actions = actions[:, :10].contiguous()
    candidates = int(actions.shape[0])

    physical = _candidate_repeat_tensor(
        _field(record, ("post_prefix_delta", "physical_delta",
                        "post_prefix_physical_delta")),
        candidates,
        "post_prefix_delta",
    )
    outcome = _candidate_repeat_tensor(
        _field(record, ("outcome_h80", "h80_outcome", "outcomes_h80")),
        candidates,
        "outcome_h80",
    )
    if physical.shape[:2] != outcome.shape[:2]:
        raise ValueError(f"{anchor_id}: physical and outcome repeat shapes differ")
    if outcome.shape[-1] != len(OUTCOME_NAMES):
        raise ValueError(
            f"{anchor_id}: outcome_h80 must end in {OUTCOME_NAMES}, got "
            f"dimension {outcome.shape[-1]}"
        )
    repeat_mask = _normalize_repeat_mask(
        record.get("repeat_mask"), candidates, int(physical.shape[1])
    )
    if not torch.isfinite(physical[repeat_mask]).all():
        raise ValueError(f"{anchor_id}: valid physical targets are non-finite")
    if not torch.isfinite(outcome[repeat_mask]).all():
        raise ValueError(f"{anchor_id}: valid H80 targets are non-finite")

    candidate_ids = record.get("candidate_ids")
    if candidate_ids is None:
        candidate_ids = [f"c{i}" for i in range(candidates)]
    if len(candidate_ids) != candidates:
        raise ValueError(f"{anchor_id}: candidate_ids length != C")
    is_reference = record.get("is_reference")
    if is_reference is None:
        is_reference = [str(x).lower() in {"reference", "ref", "u0"}
                        for x in candidate_ids]
    is_reference = _tensor(is_reference).bool().flatten()
    if is_reference.numel() != candidates:
        raise ValueError(f"{anchor_id}: is_reference length != C")

    floors = _tensor(record.get("rank_floors", DEFAULT_RANK_FLOORS),
                     dtype=torch.float32).flatten()
    if floors.numel() != len(OUTCOME_NAMES) or bool((floors < 0).any()):
        raise ValueError(f"{anchor_id}: invalid rank_floors")

    group = M0Group(
        anchor_id=anchor_id,
        source_id=source_id,
        split=split,
        state=state,
        actions=actions,
        physical=physical,
        outcome=outcome,
        repeat_mask=repeat_mask,
        is_reference=is_reference,
        candidate_ids=tuple(map(str, candidate_ids)),
        rank_floors=floors,
    )
    _ = group.reference_index
    if group.candidates < 2:
        raise ValueError(f"{anchor_id}: ranking needs reference + alternative")
    return group


def load_groups(path: str | Path) -> list[M0Group]:
    """Load, validate, and canonicalize a collector ``groups.pt`` artifact."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "groups" in payload:
        records = payload["groups"]
    elif isinstance(payload, dict) and any(k in payload for k in
                                           ("train", "calib", "val", "test")):
        records = []
        for split, values in payload.items():
            if split not in SPLIT_ALIASES:
                continue
            for value in values:
                value = dict(value)
                value.setdefault("split", split)
                records.append(value)
    elif isinstance(payload, (list, tuple)):
        records = list(payload)
    else:
        raise TypeError("groups.pt must contain a list or {'groups': list}")
    if not records or not all(isinstance(x, dict) for x in records):
        raise ValueError("groups.pt contains no group dictionaries")
    groups = [_parse_group(dict(x)) for x in records]
    validate_groups(groups)
    return groups


def validate_groups(groups: Sequence[M0Group], *, require_test: bool = True) -> None:
    if not groups:
        raise ValueError("empty group collection")
    expected = (
        groups[0].state.numel(),
        groups[0].actions.shape[-1],
        groups[0].physical.shape[-1],
    )
    seen_anchors: set[str] = set()
    sources: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for group in groups:
        got = (group.state.numel(), group.actions.shape[-1],
               group.physical.shape[-1])
        if got != expected:
            raise ValueError(f"{group.anchor_id}: inconsistent dimensions {got} != {expected}")
        if group.anchor_id in seen_anchors:
            raise ValueError(f"duplicate anchor_id {group.anchor_id!r}")
        seen_anchors.add(group.anchor_id)
        sources[group.split].add(group.source_id)
    required = ("train", "val", "test") if require_test else ("train", "val")
    for split in required:
        if not sources[split]:
            raise ValueError(f"split {split!r} has no source episodes")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sources[left] & sources[right]
        if overlap:
            raise ValueError(
                f"source leakage between {left}/{right}: {sorted(overlap)}"
            )


def split_groups(groups: Sequence[M0Group]) -> dict[str, list[M0Group]]:
    out = {"train": [], "val": [], "test": []}
    for group in groups:
        out[group.split].append(group)
    return out


@dataclass(frozen=True)
class M0Config:
    state_dim: int
    action_dim: int
    physical_dim: int
    hidden_dim: int = 256
    action_steps: int = 10
    outcome_dim: int = len(OUTCOME_NAMES)


class M0Predictor(nn.Module):
    """Small state/action/transition MLP with physical, outcome, rank heads.

    ``use_action=False`` retains the identical parameterization but replaces
    the action with zero.  It is therefore a capacity-matched state-only
    baseline rather than a separately designed control.
    """

    def __init__(self, config: M0Config, *, use_action: bool = True):
        super().__init__()
        self.config = config
        self.use_action = bool(use_action)
        h = config.hidden_dim
        self.state_mlp = nn.Sequential(
            nn.LayerNorm(config.state_dim),
            nn.Linear(config.state_dim, h), nn.GELU(), nn.Linear(h, h),
        )
        self.action_mlp = nn.Sequential(
            nn.LayerNorm(config.action_steps * config.action_dim),
            nn.Linear(config.action_steps * config.action_dim, h),
            nn.GELU(), nn.Linear(h, h),
        )
        self.transition_mlp = nn.Sequential(
            nn.LayerNorm(2 * h), nn.Linear(2 * h, h), nn.GELU(),
            nn.Linear(h, h), nn.GELU(),
        )
        self.physical_head = nn.Linear(h, config.physical_dim)
        self.outcome_head = nn.Linear(h, config.outcome_dim)
        self.rank_head = nn.Linear(h, 1)

    def forward(self, state: Tensor, actions: Tensor) -> dict[str, Tensor]:
        if state.ndim != 2 or state.shape[-1] != self.config.state_dim:
            raise ValueError(f"state must be [B,{self.config.state_dim}]")
        expected = (self.config.action_steps, self.config.action_dim)
        if actions.ndim != 3 or tuple(actions.shape[-2:]) != expected:
            raise ValueError(f"actions must be [B,{expected[0]},{expected[1]}]")
        if state.shape[0] != actions.shape[0]:
            raise ValueError("state/action batch mismatch")
        action_input = actions if self.use_action else torch.zeros_like(actions)
        s = self.state_mlp(state)
        a = self.action_mlp(action_input.flatten(1))
        z = self.transition_mlp(torch.cat([s, a], dim=-1))
        return {
            "physical_norm": self.physical_head(z),
            # indices 0/1 are logits; 2: are normalized regressions
            "outcome_raw": self.outcome_head(z),
            "rank_score": self.rank_head(z).squeeze(-1),
        }


@dataclass
class TargetStats:
    physical_scale: Tensor
    outcome_mean: Tensor
    outcome_scale: Tensor

    def to(self, device: torch.device | str) -> "TargetStats":
        return TargetStats(
            self.physical_scale.to(device),
            self.outcome_mean.to(device),
            self.outcome_scale.to(device),
        )

    def state_dict(self) -> dict[str, Tensor]:
        return {
            "physical_scale": self.physical_scale.cpu(),
            "outcome_mean": self.outcome_mean.cpu(),
            "outcome_scale": self.outcome_scale.cpu(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Tensor]) -> "TargetStats":
        return cls(state["physical_scale"], state["outcome_mean"],
                   state["outcome_scale"])


def fit_target_stats(groups: Sequence[M0Group]) -> TargetStats:
    """Fit target scaling on training sources only."""

    if not groups or any(g.split != "train" for g in groups):
        raise ValueError("target statistics must be fit on train groups only")
    physical, outcome = [], []
    for group in groups:
        physical.append(group.physical[group.repeat_mask])
        outcome.append(group.outcome[group.repeat_mask])
    physical_t = torch.cat(physical).float()
    outcome_t = torch.cat(outcome).float()
    physical_scale = physical_t.square().mean(0).sqrt().clamp_min(1e-4)
    outcome_mean = torch.zeros(len(OUTCOME_NAMES))
    outcome_scale = torch.ones(len(OUTCOME_NAMES))
    outcome_mean[2:] = outcome_t[:, 2:].mean(0)
    outcome_scale[2:] = outcome_t[:, 2:].std(0, unbiased=False).clamp_min(1e-4)
    return TargetStats(physical_scale, outcome_mean, outcome_scale)


@dataclass(frozen=True)
class LossWeights:
    physical: float = 1.0
    outcome: float = 1.0
    rank: float = 0.25


def lexicographic_label(candidate: Tensor, reference: Tensor,
                        floors: Tensor) -> int:
    """Return -1/0/+1 under the registered H80 component ordering."""

    if float(candidate[0] - reference[0]) > float(floors[0]):
        return -1
    for index, direction in enumerate(OUTCOME_DIRECTIONS[1:], start=1):
        delta = float((candidate[index] - reference[index]) * direction)
        if abs(delta) <= float(floors[index]):
            continue
        return 1 if delta > 0 else -1
    return 0


def pair_indices_and_labels(group: M0Group) -> list[tuple[int, int, int, int]]:
    """Candidate/reference pairs as ``(candidate, reference, repeat, label)``."""

    reference = group.reference_index
    pairs = []
    for candidate in range(group.candidates):
        if candidate == reference:
            continue
        for repeat in range(group.repeats):
            if not (bool(group.repeat_mask[candidate, repeat]) and
                    bool(group.repeat_mask[reference, repeat])):
                continue
            label = lexicographic_label(
                group.outcome[candidate, repeat],
                group.outcome[reference, repeat],
                group.rank_floors,
            )
            pairs.append((candidate, reference, repeat, label))
    return pairs


def _expanded_inputs(group: M0Group, actions: Tensor | None = None
                     ) -> tuple[Tensor, Tensor]:
    actions = group.actions if actions is None else actions
    c, r = group.candidates, group.repeats
    state = group.state[None, None].expand(c, r, -1).reshape(c * r, -1)
    action = actions[:, None].expand(c, r, -1, -1).reshape(
        c * r, actions.shape[1], actions.shape[2]
    )
    return state, action


def decode_predictions(predictions: dict[str, Tensor], stats: TargetStats
                      ) -> tuple[Tensor, Tensor]:
    physical = predictions["physical_norm"] * stats.physical_scale
    raw = predictions["outcome_raw"]
    outcome = raw.clone()
    outcome[:, :2] = torch.sigmoid(raw[:, :2])
    outcome[:, 2:] = raw[:, 2:] * stats.outcome_scale[2:] + stats.outcome_mean[2:]
    return physical, outcome


def group_loss(model: M0Predictor, group: M0Group, stats: TargetStats,
               weights: LossWeights = LossWeights(), *,
               actions_override: Tensor | None = None,
               balance_nontie: bool = False,
               ) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
    """Source-balanced loss: every returned component is one anchor mean."""

    state, action = _expanded_inputs(group, actions_override)
    predictions = model(state, action)
    c, r = group.candidates, group.repeats
    valid = group.repeat_mask.reshape(-1)
    physical_target = group.physical.reshape(c * r, -1)[valid]
    outcome_target = group.outcome.reshape(c * r, -1)[valid]
    physical_pred_norm = predictions["physical_norm"][valid]
    raw = predictions["outcome_raw"][valid]

    physical = F.smooth_l1_loss(
        physical_pred_norm,
        physical_target / stats.physical_scale,
    )
    binary = F.binary_cross_entropy_with_logits(raw[:, :2], outcome_target[:, :2])
    continuous = F.smooth_l1_loss(
        raw[:, 2:],
        (outcome_target[:, 2:] - stats.outcome_mean[2:])
        / stats.outcome_scale[2:],
    )
    outcome = binary + continuous

    scores = predictions["rank_score"].reshape(c, r)
    tie_terms, nontie_terms = [], []
    for candidate, reference, repeat, label in pair_indices_and_labels(group):
        difference = scores[candidate, repeat] - scores[reference, repeat]
        if label == 0:
            tie_terms.append(difference.square())
        else:
            target = difference.new_tensor(float(label))
            nontie_terms.append(F.softplus(0.2 - target * difference))
    if balance_nontie:
        # Group-level, non-tie-balanced: ties and non-ties each contribute one
        # mean, so rare informative pairs are not numerically erased.  The first
        # M0 run had 139/144 held-out pairs tied; under a flat mean the five
        # non-tie pairs carried ~3% of the ranking gradient.
        halves = [torch.stack(t).mean() for t in (tie_terms, nontie_terms) if t]
        rank = torch.stack(halves).mean() if halves else scores.sum() * 0.0
    else:
        pair_terms = tie_terms + nontie_terms
        rank = torch.stack(pair_terms).mean() if pair_terms else scores.sum() * 0.0
    total = (weights.physical * physical + weights.outcome * outcome
             + weights.rank * rank)
    parts = {"loss_total": total, "loss_physical": physical,
             "loss_outcome": outcome, "loss_rank": rank}
    return total, parts, predictions


def evaluate_model(model: M0Predictor, groups: Sequence[M0Group],
                   stats: TargetStats, weights: LossWeights = LossWeights(),
                   *, rank_tie_epsilon: float = 0.05) -> dict[str, float | int]:
    """Evaluate one model without pooling away independent anchor groups."""

    if not groups:
        raise ValueError("cannot evaluate an empty split")
    model.eval()
    loss_parts: dict[str, list[float]] = {
        "loss_total": [], "loss_physical": [],
        "loss_outcome": [], "loss_rank": [],
    }
    shuffled_prediction_losses = []
    phys_sq = phys_abs = binary_sq = binary_nll = 0.0
    phys_n = binary_n = 0
    cont_abs = torch.zeros(3)
    cont_n = 0
    labels: list[int] = []
    predicted_labels: list[int] = []
    nontie_correct = nontie_n = 0
    with torch.no_grad():
        for cpu_group in groups:
            group = cpu_group.to(next(model.parameters()).device)
            _, parts, predictions = group_loss(model, group, stats, weights)
            for key, value in parts.items():
                loss_parts[key].append(float(value))
            shuffled = group.actions.roll(shifts=1, dims=0)
            _, shuffled_parts, _ = group_loss(
                model, group, stats, weights, actions_override=shuffled
            )
            shuffled_prediction_losses.append(float(
                shuffled_parts["loss_physical"] + shuffled_parts["loss_outcome"]
            ))

            c, r = group.candidates, group.repeats
            valid = group.repeat_mask.reshape(-1)
            physical_pred, outcome_pred = decode_predictions(predictions, stats)
            physical_target = group.physical.reshape(c * r, -1)[valid]
            outcome_target = group.outcome.reshape(c * r, -1)[valid]
            physical_pred = physical_pred[valid]
            outcome_pred = outcome_pred[valid]
            difference = physical_pred - physical_target
            phys_sq += float(difference.square().sum())
            phys_abs += float(difference.abs().sum())
            phys_n += difference.numel()
            binary_difference = outcome_pred[:, :2] - outcome_target[:, :2]
            binary_sq += float(binary_difference.square().sum())
            probabilities = outcome_pred[:, :2].clamp(1e-6, 1 - 1e-6)
            binary_nll += float(F.binary_cross_entropy(
                probabilities, outcome_target[:, :2], reduction="sum"
            ))
            binary_n += binary_difference.numel()
            cont_abs += (outcome_pred[:, 2:] - outcome_target[:, 2:]).abs().sum(0).cpu()
            cont_n += outcome_target.shape[0]

            scores = predictions["rank_score"].reshape(c, r)
            for candidate, reference, repeat, label in pair_indices_and_labels(group):
                score_delta = float(scores[candidate, repeat]
                                    - scores[reference, repeat])
                pred_label = (0 if abs(score_delta) <= rank_tie_epsilon
                              else (1 if score_delta > 0 else -1))
                labels.append(label)
                predicted_labels.append(pred_label)
                if label != 0:
                    nontie_n += 1
                    nontie_correct += int(pred_label == label)

    metric: dict[str, float | int] = {
        key: sum(values) / len(values) for key, values in loss_parts.items()
    }
    matched_prediction = metric["loss_physical"] + metric["loss_outcome"]
    shuffled_prediction = sum(shuffled_prediction_losses) / len(
        shuffled_prediction_losses
    )
    metric.update({
        "anchors": len(groups),
        "sources": len({g.source_id for g in groups}),
        "valid_branch_repeats": sum(int(g.repeat_mask.sum()) for g in groups),
        "physical_mse": phys_sq / max(phys_n, 1),
        "physical_mae": phys_abs / max(phys_n, 1),
        "outcome_binary_brier": binary_sq / max(binary_n, 1),
        "outcome_binary_nll": binary_nll / max(binary_n, 1),
        "outcome_dp_mae": float(cont_abs[0] / max(cont_n, 1)),
        "outcome_ttm_mae": float(cont_abs[1] / max(cont_n, 1)),
        "outcome_G_mae": float(cont_abs[2] / max(cont_n, 1)),
        "matched_prediction_loss": matched_prediction,
        "shuffled_prediction_loss": shuffled_prediction,
        "action_shuffle_gap": shuffled_prediction - matched_prediction,
        "rank_pairs": len(labels),
        "rank_better": labels.count(1),
        "rank_tie": labels.count(0),
        "rank_worse": labels.count(-1),
        "rank_accuracy": (sum(int(a == b) for a, b in
                              zip(labels, predicted_labels)) / max(len(labels), 1)),
        "rank_nontie_accuracy": nontie_correct / max(nontie_n, 1),
    })
    per_class = []
    for label in (-1, 0, 1):
        indices = [i for i, value in enumerate(labels) if value == label]
        if indices:
            per_class.append(sum(predicted_labels[i] == label for i in indices)
                             / len(indices))
    metric["rank_macro_accuracy"] = sum(per_class) / max(len(per_class), 1)
    return metric


def infer_config(groups: Sequence[M0Group], hidden_dim: int = 256) -> M0Config:
    group = groups[0]
    return M0Config(
        state_dim=int(group.state.numel()),
        action_dim=int(group.actions.shape[-1]),
        physical_dim=int(group.physical.shape[-1]),
        hidden_dim=int(hidden_dim),
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_smoke_groups(seed: int = 0) -> list[M0Group]:
    """Small source-disjoint synthetic collection for ``--smoke --device cpu``."""

    generator = torch.Generator().manual_seed(seed)
    groups: list[M0Group] = []
    layout = (("train", 4, 1), ("val", 2, 1), ("test", 2, 3))
    action_weight = torch.randn(70, 24, generator=generator) * 0.08
    for split, count, repeats in layout:
        for index in range(count):
            candidates, tokens, state_dim = 5, 6, 16
            hidden = torch.randn(tokens, state_dim, generator=generator)
            mask = torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.bool)
            state = masked_prefix_mean(hidden, mask)
            actions = torch.randn(candidates, 10, 7, generator=generator)
            actions[0].zero_()
            base_physical = actions.flatten(1) @ action_weight
            physical = base_physical[:, None].expand(-1, repeats, -1).clone()
            physical += 0.005 * torch.randn(
                candidates, repeats, 24, generator=generator
            )
            signal = actions[:, :, 0].mean(1)
            outcome = torch.zeros(candidates, repeats, len(OUTCOME_NAMES))
            for repeat in range(repeats):
                jitter = 0.03 * torch.randn(candidates, generator=generator)
                s = signal + jitter
                outcome[:, repeat, 0] = 0.0
                outcome[:, repeat, 1] = (s > 0.25).float()
                outcome[:, repeat, 2] = (s > 0.0).float()
                outcome[:, repeat, 3] = 60.0 - 20.0 * s
                outcome[:, repeat, 4] = s
            groups.append(M0Group(
                anchor_id=f"smoke-{split}-{index}",
                source_id=f"smoke-source-{split}-{index}",
                split=split,
                state=state,
                actions=actions,
                physical=physical,
                outcome=outcome,
                repeat_mask=torch.ones(candidates, repeats, dtype=torch.bool),
                is_reference=torch.tensor([1, 0, 0, 0, 0], dtype=torch.bool),
                candidate_ids=("reference", "c1", "c2", "c3", "c4"),
                rank_floors=torch.tensor(DEFAULT_RANK_FLOORS),
            ))
    validate_groups(groups)
    return groups


def serializable_config(config: M0Config, weights: LossWeights) -> dict[str, Any]:
    return {"model": asdict(config), "loss_weights": asdict(weights),
            "outcome_names": list(OUTCOME_NAMES),
            "rank_floors_default": list(DEFAULT_RANK_FLOORS)}
