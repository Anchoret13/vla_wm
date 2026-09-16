"""Causal, identity-locked cream-placement state for Gate-0 development.

The state machine consumes already-derived observation evidence.  It never
looks ahead and has no access to task events or success.  A placement is
latched only after the cream track approaches the basket rim and then supplies
an occlusion signal inside a short, fixed confirmation window.  One signal is
used because a successful deployment can terminate on the first post-placement
observation; requiring a future frame would make the signal undeployable there.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


SCHEMA = "v247_causal_cream_state_v2"


class CreamStateError(ValueError):
    """The causal cream-state contract was violated."""


@dataclass(frozen=True)
class CreamStateConfig:
    """Frozen development constants; future panels must not tune these."""

    schema: str = SCHEMA
    outside_frames_to_arm: int = 2
    max_confirmation_frames: int = 4
    scale_collapse_frames_required: int = 1
    missing_sift_frames_required: int = 2
    scale_collapse_ratio_lte: float = 0.40

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise CreamStateError("cream-state schema mismatch")
        if self.outside_frames_to_arm < 1:
            raise CreamStateError("outside_frames_to_arm must be positive")
        required = max(
            self.scale_collapse_frames_required,
            self.missing_sift_frames_required,
        )
        if self.max_confirmation_frames < required:
            raise CreamStateError("confirmation window is too short")
        if self.scale_collapse_frames_required < 1:
            raise CreamStateError("scale_collapse_frames_required must be positive")
        if self.missing_sift_frames_required < 1:
            raise CreamStateError("missing_sift_frames_required must be positive")
        if not 0.0 < self.scale_collapse_ratio_lte < 1.0:
            raise CreamStateError("scale-collapse ratio must be in (0,1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CreamStateConfig":
        expected = {
            "schema",
            "outside_frames_to_arm",
            "max_confirmation_frames",
            "scale_collapse_frames_required",
            "missing_sift_frames_required",
            "scale_collapse_ratio_lte",
        }
        if set(value) != expected:
            raise CreamStateError("cream-state config field closure mismatch")
        return cls(**dict(value))


@dataclass(frozen=True)
class CreamFrameEvidence:
    """Observation-only evidence for one monotonically ordered frame."""

    frame_index: int
    t: int
    initial_track_present: bool
    near_basket_rim: bool
    cross_class_collision: bool
    sift_valid: bool
    sift_scale: float | None

    def __post_init__(self) -> None:
        if self.frame_index < 0 or self.t < 0:
            raise CreamStateError("frame index/time must be nonnegative")
        if self.sift_valid:
            if self.sift_scale is None or not 0.0 < self.sift_scale < float("inf"):
                raise CreamStateError("valid SIFT evidence requires a finite scale")
        elif self.sift_scale is not None:
            raise CreamStateError("invalid SIFT evidence cannot expose a scale")


@dataclass(frozen=True)
class CreamFrameState:
    schema: str
    frame_index: int
    t: int
    mode: str
    cream_achieved: bool
    transition_now: bool
    candidate_start_frame: int | None
    candidate_start_t: int | None
    candidate_age_frames: int | None
    occlusion_streak: int
    scale_collapse_streak: int
    missing_sift_streak: int
    armed_outside_streak: int
    observation_reliable: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["reasons"] = list(self.reasons)
        return value


class CausalCreamState:
    """Streaming cream-placement recognizer with explicit failed-attempt state."""

    MODES = frozenset({"search", "candidate", "rejected", "achieved"})

    def __init__(self, config: CreamStateConfig | None = None) -> None:
        self.config = config or CreamStateConfig()
        self.mode = "search"
        self.last_frame_index: int | None = None
        self.last_t: int | None = None
        self.outside_streak = 0
        self.candidate_start_frame: int | None = None
        self.candidate_start_t: int | None = None
        self.candidate_reference_scale: float | None = None
        self.occlusion_streak = 0
        self.scale_collapse_streak = 0
        self.missing_sift_streak = 0

    def _reset_candidate(self) -> None:
        self.candidate_start_frame = None
        self.candidate_start_t = None
        self.candidate_reference_scale = None
        self.occlusion_streak = 0
        self.scale_collapse_streak = 0
        self.missing_sift_streak = 0

    def _reject(self, reasons: list[str], reason: str) -> None:
        self.mode = "rejected"
        self._reset_candidate()
        self.outside_streak = 0
        reasons.append(reason)

    def step(self, evidence: CreamFrameEvidence) -> CreamFrameState:
        if self.last_frame_index is not None:
            if evidence.frame_index != self.last_frame_index + 1:
                raise CreamStateError("frame indices must be contiguous")
            if self.last_t is None or evidence.t <= self.last_t:
                raise CreamStateError("frame times must be strictly increasing")
        self.last_frame_index = evidence.frame_index
        self.last_t = evidence.t
        reasons: list[str] = []
        transition_now = False

        if self.mode == "achieved":
            reasons.append("past_only_achievement_latch")
        elif self.mode == "search":
            was_armed = self.outside_streak >= self.config.outside_frames_to_arm
            outside = (
                evidence.initial_track_present
                and not evidence.near_basket_rim
                and not evidence.cross_class_collision
                and evidence.sift_valid
            )
            self.outside_streak = self.outside_streak + 1 if outside else 0
            armed = was_armed or self.outside_streak >= self.config.outside_frames_to_arm
            eligible_rim = (
                evidence.initial_track_present
                and evidence.near_basket_rim
                and not evidence.cross_class_collision
                and evidence.sift_valid
            )
            if eligible_rim and armed:
                self.mode = "candidate"
                self.candidate_start_frame = evidence.frame_index
                self.candidate_start_t = evidence.t
                self.candidate_reference_scale = evidence.sift_scale
                self.occlusion_streak = 0
                self.scale_collapse_streak = 0
                self.missing_sift_streak = 0
                reasons.append("armed_identity_track_reached_rim")
            elif evidence.cross_class_collision:
                reasons.append("cross_class_collision_fail_closed")
            elif not evidence.initial_track_present:
                reasons.append("initial_cream_track_absent")
            elif evidence.near_basket_rim and not evidence.sift_valid:
                reasons.append("rim_without_sift_identity")
            else:
                reasons.append("searching")
        elif self.mode == "candidate":
            if self.candidate_start_frame is None or self.candidate_reference_scale is None:
                raise CreamStateError("candidate state is internally incomplete")
            age = evidence.frame_index - self.candidate_start_frame
            if evidence.cross_class_collision:
                self._reject(reasons, "candidate_cross_class_collision")
            elif age > self.config.max_confirmation_frames:
                self._reject(reasons, "candidate_confirmation_timeout")
            elif evidence.initial_track_present:
                self.occlusion_streak = 0
                self.scale_collapse_streak = 0
                self.missing_sift_streak = 0
                if evidence.near_basket_rim and evidence.sift_valid:
                    reasons.append("candidate_object_still_visible_at_rim")
                else:
                    self._reject(reasons, "candidate_object_left_rim_visible")
            else:
                scale_collapsed = bool(
                    evidence.sift_valid
                    and evidence.sift_scale is not None
                    and evidence.sift_scale
                    <= self.candidate_reference_scale
                    * self.config.scale_collapse_ratio_lte
                )
                occluded = (not evidence.sift_valid) or scale_collapsed
                self.occlusion_streak = self.occlusion_streak + 1 if occluded else 0
                if not evidence.sift_valid:
                    self.missing_sift_streak += 1
                    self.scale_collapse_streak = 0
                    reasons.append("identity_track_absent_and_sift_missing")
                elif scale_collapsed:
                    self.scale_collapse_streak += 1
                    self.missing_sift_streak = 0
                    reasons.append("identity_track_absent_and_sift_scale_collapsed")
                else:
                    self.scale_collapse_streak = 0
                    self.missing_sift_streak = 0
                    reasons.append("identity_track_absent_without_occlusion_support")
                collapse_confirmed = (
                    self.scale_collapse_streak
                    >= self.config.scale_collapse_frames_required
                )
                missing_confirmed = (
                    self.missing_sift_streak
                    >= self.config.missing_sift_frames_required
                )
                if collapse_confirmed or missing_confirmed:
                    self.mode = "achieved"
                    transition_now = True
                    reasons.append("causal_rim_then_occlusion_confirmed")
        elif self.mode == "rejected":
            outside = (
                evidence.initial_track_present
                and not evidence.near_basket_rim
                and not evidence.cross_class_collision
                and evidence.sift_valid
            )
            self.outside_streak = self.outside_streak + 1 if outside else 0
            if self.outside_streak >= self.config.outside_frames_to_arm:
                self.mode = "search"
                reasons.append("failed_attempt_left_rim_and_rearmed")
            else:
                reasons.append("failed_attempt_not_rearmed")
        else:
            raise CreamStateError(f"unknown mode {self.mode}")

        candidate_age = (
            evidence.frame_index - self.candidate_start_frame
            if self.candidate_start_frame is not None
            else None
        )
        reliable = bool(
            self.mode == "achieved"
            or evidence.initial_track_present
            or (self.mode == "candidate" and self.occlusion_streak > 0)
        )
        return CreamFrameState(
            schema=SCHEMA,
            frame_index=evidence.frame_index,
            t=evidence.t,
            mode=self.mode,
            cream_achieved=self.mode == "achieved",
            transition_now=transition_now,
            candidate_start_frame=self.candidate_start_frame,
            candidate_start_t=self.candidate_start_t,
            candidate_age_frames=candidate_age,
            occlusion_streak=self.occlusion_streak,
            scale_collapse_streak=self.scale_collapse_streak,
            missing_sift_streak=self.missing_sift_streak,
            armed_outside_streak=self.outside_streak,
            observation_reliable=reliable,
            reasons=tuple(reasons),
        )


def run_causal_cream_state(
    evidence: Sequence[CreamFrameEvidence],
    config: CreamStateConfig | None = None,
) -> list[CreamFrameState]:
    machine = CausalCreamState(config)
    return [machine.step(frame) for frame in evidence]
