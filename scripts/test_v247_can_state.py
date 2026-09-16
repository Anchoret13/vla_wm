#!/usr/bin/env python3
"""CPU-only tests for the anonymous causal can-count state."""
from __future__ import annotations

import pytest

from lcwm.v247_can_state import (
    CanStateConfig,
    CanStateError,
    CausalCanCount,
    rim_expanded_inside_count,
)


def test_eight_pixel_top_rim_allowance_is_inclusive() -> None:
    basket = [0.0, 100.0, 80.0, 200.0]
    assert rim_expanded_inside_count(basket, [[20.0, 92.0], [100.0, 150.0]]) == 1
    assert rim_expanded_inside_count(basket, [[20.0, 91.999], [100.0, 150.0]]) == 0


def test_expansion_is_not_applied_to_sides_or_bottom() -> None:
    basket = [0.0, 100.0, 80.0, 200.0]
    assert rim_expanded_inside_count(basket, [[-0.1, 150.0], [40.0, 200.1]]) == 0


def test_missing_basket_or_instance_is_explicit_and_carries_only_past() -> None:
    counter = CausalCanCount()
    first = counter.step([0.0, 100.0, 80.0, 200.0], [[20.0, 120.0], [90.0, 120.0]])
    assert first.raw_count == 1 and first.sticky_count == 1
    missing = counter.step(None, [[20.0, 120.0], [90.0, 120.0]])
    assert missing.raw_count is None and missing.sticky_count == 1
    assert missing.used_carry is True and missing.transition_size == 0
    assert rim_expanded_inside_count([0.0, 100.0, 80.0, 200.0], [[20.0, 120.0]]) is None


def test_count_is_anonymous_monotone_and_can_jump_two() -> None:
    counter = CausalCanCount()
    basket = [0.0, 100.0, 80.0, 200.0]
    zero = counter.step(basket, [[90.0, 120.0], [100.0, 130.0]])
    two = counter.step(basket, [[10.0, 120.0], [20.0, 130.0]])
    later_zero = counter.step(basket, [[90.0, 120.0], [100.0, 130.0]])
    assert zero.sticky_count == 0
    assert two.sticky_count == 2 and two.transition_size == 2
    assert later_zero.sticky_count == 2 and later_zero.transition_size == 0
    forward = rim_expanded_inside_count(
        basket, [[10.0, 120.0], [20.0, 130.0]]
    )
    reverse = rim_expanded_inside_count(
        basket, [[20.0, 130.0], [10.0, 120.0]]
    )
    assert forward == reverse


def test_config_and_geometry_fail_closed() -> None:
    config = CanStateConfig()
    assert CanStateConfig.from_dict(config.to_dict()) == config
    with pytest.raises(CanStateError):
        CanStateConfig.from_dict({**config.to_dict(), "success": True})
    with pytest.raises(CanStateError):
        CanStateConfig(rim_margin_px=float("nan"))
    with pytest.raises(CanStateError):
        rim_expanded_inside_count([0.0, 100.0, 0.0, 200.0], [[1.0, 1.0], [2.0, 2.0]])
