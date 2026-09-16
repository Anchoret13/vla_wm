#!/usr/bin/env python3
"""CPU-only tests for the causal cream-placement state machine."""
from __future__ import annotations

from dataclasses import replace

import pytest

from lcwm.v247_cream_state import (
    CreamFrameEvidence,
    CreamStateConfig,
    CreamStateError,
    run_causal_cream_state,
)


def frame(
    index: int,
    *,
    present: bool = True,
    rim: bool = False,
    collision: bool = False,
    sift: bool = True,
    scale: float = 1.0,
) -> CreamFrameEvidence:
    return CreamFrameEvidence(
        frame_index=index,
        t=index * 10,
        initial_track_present=present,
        near_basket_rim=rim,
        cross_class_collision=collision,
        sift_valid=sift,
        sift_scale=scale if sift else None,
    )


def test_rim_then_one_scale_collapse_signal_commits_once() -> None:
    evidence = [
        frame(0),
        frame(1),
        frame(2, rim=True, scale=3.0),
        frame(3, rim=True, scale=3.1),
        frame(4, present=False, sift=True, scale=1.0),
        frame(5, present=False, sift=False),
        frame(6, present=False, sift=False),
    ]
    states = run_causal_cream_state(evidence)
    assert states[2].mode == "candidate"
    assert states[4].occlusion_streak == 1
    assert states[4].transition_now is True
    assert states[4].cream_achieved is True
    assert states[5].transition_now is False
    assert states[5].cream_achieved is True
    assert states[6].transition_now is False
    assert states[6].cream_achieved is True


def test_one_frame_missing_sift_dropout_then_reappearance_does_not_commit() -> None:
    evidence = [
        frame(0),
        frame(1),
        frame(2, rim=True, scale=3.0),
        frame(3, present=False, sift=False),
        frame(4, rim=True, scale=3.0),
    ]
    states = run_causal_cream_state(evidence)
    assert states[3].missing_sift_streak == 1
    assert states[3].cream_achieved is False
    assert states[4].mode == "candidate"
    assert states[4].missing_sift_streak == 0
    assert not any(state.transition_now for state in states)


def test_terminal_scale_collapse_can_arrive_one_environment_step_later() -> None:
    evidence = [
        replace(frame(0), t=0),
        replace(frame(1), t=10),
        replace(frame(2, rim=True, scale=3.0), t=20),
        replace(frame(3, present=False, sift=True, scale=1.0), t=21),
    ]
    states = run_causal_cream_state(evidence)
    assert states[-1].transition_now is True
    assert states[-1].t == 21


def test_object_held_at_rim_expires_without_false_increment() -> None:
    evidence = [frame(0), frame(1)] + [
        frame(index, rim=True, scale=3.0) for index in range(2, 10)
    ]
    states = run_causal_cream_state(evidence)
    assert not any(state.transition_now for state in states)
    assert states[7].mode == "rejected"
    assert states[-1].cream_achieved is False


def test_cross_class_collision_fails_closed() -> None:
    states = run_causal_cream_state(
        [frame(0), frame(1), frame(2, rim=True, collision=True)]
    )
    assert states[-1].mode == "search"
    assert states[-1].cream_achieved is False
    assert "cross_class_collision_fail_closed" in states[-1].reasons


def test_track_churn_without_authenticated_rim_never_commits() -> None:
    evidence = [frame(0), frame(1)] + [
        frame(index, present=False, sift=False) for index in range(2, 12)
    ]
    states = run_causal_cream_state(evidence)
    assert not any(state.transition_now for state in states)
    assert all(state.mode == "search" for state in states)


def test_visible_failed_attempt_requires_outside_rearm() -> None:
    evidence = [
        frame(0),
        frame(1),
        frame(2, rim=True, scale=2.0),
        frame(3, rim=False, scale=2.0),
        frame(4, rim=False, scale=1.0),
        frame(5, rim=False, scale=1.0),
        frame(6, rim=True, scale=3.0),
    ]
    states = run_causal_cream_state(evidence)
    assert states[3].mode == "rejected"
    assert states[5].mode == "search"
    assert states[6].mode == "candidate"


def test_future_replacement_cannot_change_prefix() -> None:
    prefix = [frame(0), frame(1), frame(2, rim=True, scale=3.0)]
    future_a = [
        frame(3, present=False, sift=True, scale=1.0),
        frame(4, present=False, sift=False),
    ]
    future_b = [frame(3), frame(4)]
    prefix_states = [state.to_dict() for state in run_causal_cream_state(prefix)]
    full_a = [state.to_dict() for state in run_causal_cream_state(prefix + future_a)]
    full_b = [state.to_dict() for state in run_causal_cream_state(prefix + future_b)]
    assert full_a[: len(prefix)] == prefix_states
    assert full_b[: len(prefix)] == prefix_states


def test_config_and_stream_contracts_fail_closed() -> None:
    config = CreamStateConfig()
    assert CreamStateConfig.from_dict(config.to_dict()) == config
    with pytest.raises(CreamStateError):
        CreamStateConfig.from_dict({**config.to_dict(), "success": True})
    with pytest.raises(CreamStateError):
        replace(config, scale_collapse_ratio_lte=1.0)
    with pytest.raises(CreamStateError):
        CreamFrameEvidence(0, 0, True, False, False, False, 1.0)
    with pytest.raises(CreamStateError):
        run_causal_cream_state([frame(0), replace(frame(1), frame_index=2)])
