"""Identity-free causal can counting with a fixed basket-rim allowance."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Mapping, Sequence


SCHEMA = "v247_causal_can_count_v1"


class CanStateError(ValueError):
    """The anonymous can-count contract was violated."""


@dataclass(frozen=True)
class CanStateConfig:
    schema: str = SCHEMA
    rim_margin_px: float = 8.0
    expected_instances: int = 2

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise CanStateError("can-state schema mismatch")
        if not isfinite(self.rim_margin_px) or self.rim_margin_px < 0.0:
            raise CanStateError("rim margin must be finite and nonnegative")
        if self.expected_instances != 2:
            raise CanStateError("the chain3 quotient requires exactly two cans")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CanStateConfig":
        if set(value) != {"schema", "rim_margin_px", "expected_instances"}:
            raise CanStateError("can-state config field closure mismatch")
        return cls(**dict(value))


def _finite_vector(value: Sequence[float], size: int, label: str) -> list[float]:
    if len(value) != size:
        raise CanStateError(f"{label} must have length {size}")
    output = [float(item) for item in value]
    if not all(isfinite(item) for item in output):
        raise CanStateError(f"{label} is not finite")
    return output


def rim_expanded_inside_count(
    basket_xyxy: Sequence[float] | None,
    can_centers_xy: Sequence[Sequence[float]],
    config: CanStateConfig | None = None,
) -> int | None:
    """Count anonymous can centroids in a basket expanded only above its rim."""
    cfg = config or CanStateConfig()
    if basket_xyxy is None or len(can_centers_xy) != cfg.expected_instances:
        return None
    x0, y0, x1, y1 = _finite_vector(basket_xyxy, 4, "basket box")
    if not x0 < x1 or not y0 < y1:
        raise CanStateError("basket box is degenerate")
    count = 0
    for index, raw_center in enumerate(can_centers_xy):
        x, y = _finite_vector(raw_center, 2, f"can center {index}")
        count += int(x0 <= x <= x1 and y0 - cfg.rim_margin_px <= y <= y1)
    return count


@dataclass(frozen=True)
class CanFrameState:
    raw_count: int | None
    sticky_count: int
    used_carry: bool
    transition_size: int


class CausalCanCount:
    def __init__(self, config: CanStateConfig | None = None) -> None:
        self.config = config or CanStateConfig()
        self.sticky_count = 0

    def step(
        self,
        basket_xyxy: Sequence[float] | None,
        can_centers_xy: Sequence[Sequence[float]],
    ) -> CanFrameState:
        raw = rim_expanded_inside_count(basket_xyxy, can_centers_xy, self.config)
        previous = self.sticky_count
        if raw is not None:
            self.sticky_count = max(self.sticky_count, raw)
        return CanFrameState(
            raw_count=raw,
            sticky_count=self.sticky_count,
            used_carry=raw is None and self.sticky_count > 0,
            transition_size=self.sticky_count - previous,
        )
