"""Identity-free ordinal progress head for the V246 development branch.

This module is deliberately independent from the frozen V242 teacher and Gate
implementation.  It defines only a deployable model/loss interface; importing
or using it does not inspect an experiment artifact and does not imply that any
promotion Gate has passed.

The two target cans are represented by cumulative, permutation-invariant
events rather than named identities::

    P(number of target cans achieved >= 1)
    P(number of target cans achieved >= 2)
    P(cream cheese achieved)

The first two logits share a nonnegative, structurally parameterized gap, so
``p_can_ge2 <= p_can_ge1`` holds for every input without a soft penalty.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn


CONFIG_SCHEMA = "v246_count_phi_config_v1"
OUTPUT_NAMES = ("can_ge1", "can_ge2", "cream")


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class CountPhiConfig:
    """Serializable architecture configuration for :class:`CountPhiHead`."""

    input_dim: int
    hidden_dim: int = 256
    dropout: float = 0.0

    def __post_init__(self) -> None:
        _positive_int(self.input_dim, "input_dim")
        _positive_int(self.hidden_dim, "hidden_dim")
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not math.isfinite(float(self.dropout))
            or not 0.0 <= float(self.dropout) < 1.0
        ):
            raise ValueError("dropout must be finite and in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-compatible reconstruction config."""

        return {"schema": CONFIG_SCHEMA, **asdict(self)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CountPhiConfig":
        """Strictly reconstruct a config, rejecting stale or extra fields."""

        if not isinstance(payload, Mapping):
            raise TypeError("CountPhiConfig payload must be a mapping")
        expected = {"schema", "input_dim", "hidden_dim", "dropout"}
        actual = set(payload)
        if actual != expected:
            raise ValueError(
                "CountPhiConfig fields mismatch: "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
        if payload["schema"] != CONFIG_SCHEMA:
            raise ValueError(f"unsupported CountPhiConfig schema: {payload['schema']!r}")
        return cls(
            input_dim=payload["input_dim"],
            hidden_dim=payload["hidden_dim"],
            dropout=payload["dropout"],
        )


@dataclass(frozen=True)
class CountPhiOutput:
    """Ordered logits/probabilities and their identity-free scalar sum."""

    logits: Tensor
    probabilities: Tensor
    scalar: Tensor

    @property
    def p_can_ge1(self) -> Tensor:
        return self.probabilities[..., 0]

    @property
    def p_can_ge2(self) -> Tensor:
        return self.probabilities[..., 1]

    @property
    def p_cream(self) -> Tensor:
        return self.probabilities[..., 2]


@dataclass(frozen=True)
class OrdinalTargets:
    """Binary cumulative targets and a same-shaped confidence mask."""

    values: Tensor
    mask: Tensor


def _require_binary(name: str, value: Tensor) -> None:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a tensor")
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains a non-finite value")
    if value.dtype == torch.bool:
        return
    if not bool(((value == 0) | (value == 1)).all()):
        raise ValueError(f"{name} must contain only binary values")


def _require_bool_mask(name: str, mask: Tensor, shape: torch.Size) -> None:
    if not torch.is_tensor(mask):
        raise TypeError(f"{name} must be a tensor")
    if mask.dtype != torch.bool:
        raise TypeError(f"{name} must have dtype torch.bool")
    if mask.shape != shape:
        raise ValueError(f"{name} shape {tuple(mask.shape)} != expected {tuple(shape)}")


def ordinal_targets_from_slots(
    can_inside: Tensor,
    cream_achieved: Tensor,
    *,
    can_observed: Tensor | None = None,
    cream_observed: Tensor | None = None,
) -> OrdinalTargets:
    """Convert two anonymous can slots into permutation-invariant targets.

    ``can_inside`` must have shape ``[batch, 2]`` and ``cream_achieved`` shape
    ``[batch]``.  Observation masks mark reliable labels.  Partial evidence is
    retained when logically sufficient: one observed inside can proves
    ``can_ge1``; one observed outside can disproves ``can_ge2``.  No slot name
    or slot order enters the result.
    """

    if not torch.is_tensor(can_inside) or can_inside.ndim != 2:
        raise ValueError("can_inside must be a rank-2 tensor with shape [batch, 2]")
    if can_inside.shape[0] == 0 or can_inside.shape[1] != 2:
        raise ValueError("can_inside must have nonempty shape [batch, 2]")
    if not torch.is_tensor(cream_achieved) or cream_achieved.ndim != 1:
        raise ValueError("cream_achieved must be a rank-1 tensor with shape [batch]")
    if cream_achieved.shape[0] != can_inside.shape[0]:
        raise ValueError("can_inside and cream_achieved batch dimensions differ")
    if cream_achieved.device != can_inside.device:
        raise ValueError("can_inside and cream_achieved must be on the same device")
    _require_binary("can_inside", can_inside)
    _require_binary("cream_achieved", cream_achieved)

    if can_observed is None:
        can_observed = torch.ones_like(can_inside, dtype=torch.bool)
    if cream_observed is None:
        cream_observed = torch.ones_like(cream_achieved, dtype=torch.bool)
    _require_bool_mask("can_observed", can_observed, can_inside.shape)
    _require_bool_mask("cream_observed", cream_observed, cream_achieved.shape)
    if can_observed.device != can_inside.device:
        raise ValueError("can_observed and can_inside must be on the same device")
    if cream_observed.device != can_inside.device:
        raise ValueError("cream_observed and can_inside must be on the same device")

    can_bits = can_inside.to(dtype=torch.bool)
    known_inside = can_observed & can_bits
    known_outside = can_observed & ~can_bits
    both_observed = can_observed.all(dim=-1)

    can_ge1 = known_inside.any(dim=-1)
    can_ge1_mask = can_ge1 | both_observed
    can_ge2 = known_inside.all(dim=-1)
    can_ge2_mask = known_outside.any(dim=-1) | both_observed

    dtype = torch.float32
    if can_inside.is_floating_point():
        dtype = can_inside.dtype
    elif cream_achieved.is_floating_point():
        dtype = cream_achieved.dtype
    values = torch.stack(
        (
            can_ge1.to(dtype=dtype),
            can_ge2.to(dtype=dtype),
            cream_achieved.to(dtype=dtype),
        ),
        dim=-1,
    )
    mask = torch.stack((can_ge1_mask, can_ge2_mask, cream_observed), dim=-1)
    return OrdinalTargets(values=values, mask=mask)


class CountPhiHead(nn.Module):
    """MLP ordinal head whose can probabilities are ordered by construction."""

    def __init__(self, config: CountPhiConfig):
        super().__init__()
        if not isinstance(config, CountPhiConfig):
            raise TypeError("config must be a CountPhiConfig")
        self.config = config
        self.network = nn.Sequential(
            nn.LayerNorm(config.input_dim),
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(config.hidden_dim, 3),
        )

    def forward(self, z: Tensor) -> CountPhiOutput:
        if not torch.is_tensor(z):
            raise TypeError("z must be a tensor")
        if not z.is_floating_point():
            raise TypeError("z must have a floating-point dtype")
        if z.ndim != 2 or z.shape[0] == 0 or z.shape[1] != self.config.input_dim:
            raise ValueError(
                f"z must have nonempty shape [batch, {self.config.input_dim}], "
                f"got {tuple(z.shape)}"
            )
        if not bool(torch.isfinite(z).all()):
            raise ValueError("z contains a non-finite value")

        raw = self.network(z)
        if raw.shape != (z.shape[0], 3) or not bool(torch.isfinite(raw).all()):
            raise RuntimeError("CountPhiHead produced invalid raw outputs")

        center = raw[:, 0]
        gap = F.softplus(raw[:, 1])
        logits = torch.stack(
            (center + 0.5 * gap, center - 0.5 * gap, raw[:, 2]), dim=-1
        )
        probabilities = torch.sigmoid(logits)
        scalar = probabilities.sum(dim=-1)
        if not bool(torch.isfinite(probabilities).all()) or not bool(
            torch.isfinite(scalar).all()
        ):
            raise RuntimeError("CountPhiHead produced non-finite probabilities")
        if not bool((probabilities[:, 1] <= probabilities[:, 0]).all()):
            raise RuntimeError("internal error: ordinal can probabilities are unordered")
        return CountPhiOutput(logits=logits, probabilities=probabilities, scalar=scalar)


def masked_ordinal_bce(
    output: CountPhiOutput,
    targets: OrdinalTargets | Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Mean BCE over confident ordinal labels only.

    Masked values contribute exactly zero.  A fully uncertain batch returns a
    differentiable zero connected to ``output.logits``, so it cannot create a
    gradient or a NaN.  Targets remain strictly binary even when masked; the
    mask, rather than a magic target value, represents uncertainty.
    """

    if not isinstance(output, CountPhiOutput):
        raise TypeError("output must be a CountPhiOutput")
    logits = output.logits
    if not torch.is_tensor(logits) or logits.ndim != 2 or logits.shape[1] != 3:
        raise ValueError("output.logits must have shape [batch, 3]")
    if not logits.is_floating_point() or not bool(torch.isfinite(logits).all()):
        raise ValueError("output.logits must be finite floating-point values")

    if isinstance(targets, OrdinalTargets):
        if mask is not None:
            raise ValueError("mask must be omitted when targets is OrdinalTargets")
        values, confidence = targets.values, targets.mask
    else:
        if mask is None:
            raise ValueError("mask is required when targets is a tensor")
        values, confidence = targets, mask
    if not torch.is_tensor(values) or values.shape != logits.shape:
        raise ValueError(
            f"target shape must equal logits shape {tuple(logits.shape)}"
        )
    if values.device != logits.device:
        raise ValueError("targets and logits must be on the same device")
    _require_binary("targets", values)
    _require_bool_mask("mask", confidence, logits.shape)
    if confidence.device != logits.device:
        raise ValueError("mask and logits must be on the same device")

    losses = F.binary_cross_entropy_with_logits(
        logits, values.to(dtype=logits.dtype), reduction="none"
    )
    if not bool(confidence.any()):
        return logits.sum() * 0.0
    loss = losses[confidence].mean()
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("masked ordinal BCE produced a non-finite loss")
    return loss


__all__ = [
    "CONFIG_SCHEMA",
    "OUTPUT_NAMES",
    "CountPhiConfig",
    "CountPhiHead",
    "CountPhiOutput",
    "OrdinalTargets",
    "masked_ordinal_bce",
    "ordinal_targets_from_slots",
]
