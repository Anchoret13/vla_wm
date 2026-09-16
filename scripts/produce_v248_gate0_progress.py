#!/usr/bin/env python3
# RETIRED 2026-09-16. Nothing supersedes it: the gate it serves was removed as a
# prerequisite at plan_and_progress/2026-09-12.md:105, and the round used a supplied
# Phi' annotation instead. It has produced zero artifacts - none of
# results/v248_gate0_{registration_bundle,materialized_models,formal_campaign}_v1 exists.
# Blockers never cleared: WORM ledger, runtime fingerprint, registration bundle, and
# non-deterministic cuDNN/TF32 (plan_and_progress/2026-09-02.md:108-129).
# Do not extend, do not import, do not cite as capability. Recoverable at d35e2832.
"""Produce the sealed observation-only v248 Gate-0 visual progress tape.

The two formal RGB runs and their pre-registrations are validated completely
before any JPEG is decoded.  Oracle summaries are never stat'ed, hashed, or
opened by this program.  The heavy SAM3/SIFT engines are imported lazily only
after the metadata boundary has passed.

The public frame surface contains anonymous ordinal progress, reliability, and
current-frame evidence.  It deliberately contains no paths, proprioception,
outcome, success, event, or episode-length fields.  A recovered t=651 frame is
an observation for Phi only and is explicitly dynamics-ineligible.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
import random
import stat
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lcwm.v247_can_state import CanStateConfig, CausalCanCount  # noqa: E402
from lcwm.v247_cream_state import (  # noqa: E402
    CreamFrameEvidence,
    CreamStateConfig,
    run_causal_cream_state,
)
from lcwm.v248_gate0_contract import (  # noqa: E402
    CAMERAS,
    Gate0ContractError,
    PANEL_IDS,
    RGB_RESULT_SCHEMA,
    TASK,
    canonical_bytes,
    canonical_sha256,
    is_sha256,
    load_and_validate_campaign_bundle,
    load_and_validate_registration,
    loads_strict_json,
    require,
    require_exact_keys,
    sam3_config_sha256,
    sam3_processor_sha256,
    sha256_file,
)


CONFIG_SCHEMA = "v248_gate0_semantic_config_v1"
RESULT_SCHEMA = "v248_gate0_progress_result_v1"
FRAME_SCHEMA = "v248_gate0_progress_frame_v1"
COMPLETE_SCHEMA = "v248_gate0_progress_complete_v1"
EXPECTED_CONFIG_SHA256 = (
    "6f40e1199e2ffe81c001e4acf341dad85af705783941f550f0c8d40729112267"
)

RESULT_FILENAME = "RESULT.json"
FRAMES_FILENAME = "frames.jsonl"
COMPLETE_FILENAME = "COMPLETE.json"
PRODUCER_FILENAME = "PRODUCER.py"
CONFIG_FILENAME = "SEMANTIC_CONFIG.json"

TOP_CONFIG_FIELDS = frozenset(
    {
        "schema",
        "status",
        "task",
        "observation_contract",
        "sam3",
        "sift",
        "geometry",
        "can_state",
        "cream_state",
        "progress",
        "schemas",
        "screen",
        "promotion_gates",
    }
)
OBSERVATION_CONFIG_FIELDS = frozenset(
    {
        "cameras",
        "causal",
        "forbidden_inputs",
        "primary_tracking_camera",
        "cream_identity_camera",
    }
)
SAM3_CONFIG_FIELDS = frozenset(
    {
        "prompts",
        "detection_score_threshold",
        "new_track_score_threshold",
        "detector_nms_iou_threshold",
        "duplicate_mask_iou_gte",
        "duplicate_box_iou_gte",
        "max_can_instances",
        "mask_area_bounds_inclusive",
        "device",
        "state_device",
        "dtype",
        "global_seed",
    }
)
SIFT_CONFIG_FIELDS = frozenset(
    {
        "upscale",
        "contrast_threshold",
        "lowe_ratio",
        "cluster_eps_px",
        "min_cluster_matches",
        "ransac_reprojection_threshold_px",
        "ransac_max_iters",
        "ransac_confidence",
        "ransac_refine_iters",
        "ransac_min_inliers",
        "ransac_min_inlier_ratio",
        "scale_range_inclusive",
        "opencv_rng_seed",
    }
)
GEOMETRY_CONFIG_FIELDS = frozenset(
    {
        "cream_rim_margin_px",
        "cream_rim_max_depth_fraction",
        "cream_can_cross_class_box_ios_gte",
    }
)
PROGRESS_CONFIG_FIELDS = frozenset(
    {
        "atoms",
        "scalar",
        "sticky",
        "can_identity_quotient",
        "prefix_check_frames_per_episode",
        "prefix_float_round_decimals",
    }
)
SCHEMA_CONFIG_FIELDS = frozenset(
    {"result", "frame", "complete", "evaluation"}
)
SCREEN_CONFIG_FIELDS = frozenset(
    {"claim", "promotion_authorized", "required_primary_chain3_episodes"}
)
PROMOTION_CONFIG_FIELDS = frozenset(
    {
        "required_primary_chain3_episodes",
        "minimum_evidence_coverage",
        "maximum_uncertainty_rate",
        "minimum_exact_scalar_accuracy",
        "minimum_macro_f1",
        "maximum_missing_transitions",
        "maximum_extra_transitions",
        "maximum_median_timing_error_steps",
        "maximum_p95_timing_error_steps",
        "maximum_timing_error_steps",
        "minimum_never_achieved_terminal_specificity",
        "minimum_held_cream_specificity",
        "minimum_chain3_cream_positive_transitions",
        "maximum_carry_created_transitions",
        "require_all_prefix_checks_exact",
        "require_terminal_recovery_pass",
    }
)

RESULT_FIELDS = frozenset(
    {
        "schema",
        "status",
        "created_utc",
        "claim_scope",
        "phase",
        "task",
        "authority",
        "registrations",
        "rgb_inputs",
        "episode_ids",
        "observation_contract",
        "engines",
        "prefix_checks",
        "recovered_endpoint",
        "output",
        "safety",
    }
)
AUTHORITY_FIELDS = frozenset(
    {
        "freeze_complete_sha256",
        "campaign_manifest_sha256",
        "campaign_body_sha256",
        "campaign_projection_sha256",
        "campaign_id",
        "registration_slots",
        "output_complete_slot",
        "confirm_authorization_complete_sha256",
    }
)
REGISTRATION_RESULT_FIELDS = frozenset(
    {"registration_id", "file_sha256", "canonical_self_sha256"}
)
RGB_INPUT_FIELDS = frozenset(
    {
        "result_sha256",
        "manifest_sha256",
        "complete_sha256",
        "producer_sha256",
        "image_tree_sha256",
        "frames",
    }
)
ENGINE_FIELDS = frozenset(
    {
        "sam3_engine_sha256",
        "sift_engine_sha256",
        "can_state_sha256",
        "cream_state_sha256",
        "sam3_checkpoint_sha256",
        "clip_tokenizer_tree_sha256",
        "sam3_config_sha256",
        "sam3_processor_sha256",
        "cream_template_image_sha256",
        "cream_template_metadata_sha256",
        "semantic_config_sha256",
        "transformers_source_sha256s",
    }
)
PREFIX_CHECK_FIELDS = frozenset(
    {
        "frames",
        "reference_sha256",
        "fresh_session_sha256",
        "exact_after_rounding_6_decimals",
    }
)
RECOVERED_RESULT_FIELDS = frozenset(
    {
        "required",
        "present",
        "panel_id",
        "episode_id",
        "t",
        "phi_only",
        "dynamics_transition_eligible",
        "rgb_recovery_prerequisites_passed",
    }
)
OUTPUT_RESULT_FIELDS = frozenset(
    {"frames", "frames_sha256", "producer_sha256", "semantic_config_sha256"}
)
SAFETY_RESULT_FIELDS = frozenset(
    {
        "oracle_opened_or_hashed",
        "privileged_inputs_read",
        "simulator_imported_or_run",
        "model_input_fields",
        "forbidden_public_field_scan_passed",
        "source_images_mutated",
        "dynamics_rows_written",
    }
)

FRAME_FIELDS = frozenset(
    {
        "schema",
        "panel_id",
        "phase",
        "task",
        "episode_id",
        "env_seed",
        "rgb_frame_index",
        "episode_frame_index",
        "t",
        "frame_id",
        "boundary_role",
        "observation_hashes",
        "atoms",
        "scalar",
        "reliability",
        "transitions",
        "causal_state",
        "evidence",
        "phi_only",
        "dynamics_transition_eligible",
    }
)
OBSERVATION_HASH_FIELDS = frozenset(
    {"agentview_sha256", "eye_in_hand_sha256"}
)
ATOM_FIELDS = frozenset({"can_ge1", "can_ge2", "cream"})
RELIABILITY_FIELDS = frozenset({"can_ge1", "can_ge2", "cream", "scalar"})
TRANSITION_FIELDS = ATOM_FIELDS
CAUSAL_STATE_FIELDS = frozenset(
    {"sticky_can_count", "cream_mode", "can_used_carry", "cream_used_carry"}
)
EVIDENCE_FIELDS = frozenset(
    {
        "basket_bbox_xyxy",
        "anonymous_can_centers_xy",
        "raw_can_inside_count",
        "initial_cream_track_present",
        "cream_near_basket_rim",
        "cream_cross_class_collision",
        "sift_valid",
        "sift_affine_scale",
        "sift_invalid_reasons",
        "cream_state_reasons",
    }
)
COMPLETE_FIELDS = frozenset(
    {
        "schema",
        "status",
        "atomic_commit",
        "phase",
        "authority",
        "registration_sha256s",
        "result_sha256",
        "frames_sha256",
        "producer_sha256",
        "semantic_config_sha256",
    }
)

PUBLIC_FORBIDDEN_KEYS = frozenset(
    {
        "done",
        "episode_length",
        "events",
        "event",
        "is_success",
        "outcome",
        "path",
        "paths",
        "policy_id",
        "proprio",
        "reward",
        "simulator_state",
        "source_summary",
        "success",
        "summary",
        "terminal",
        "terminated",
        "truncated",
    }
)


@dataclass(frozen=True)
class RgbPanel:
    root: Path
    panel_id: str
    phase: str
    result: dict[str, Any]
    complete: dict[str, Any]
    rows: tuple[dict[str, Any], ...]
    seals: dict[str, Any]


@dataclass(frozen=True)
class PreparedInputs:
    campaign_bundle: dict[str, Any]
    authority: dict[str, Any]
    registrations: dict[int, dict[str, Any]]
    registration_hashes: dict[int, str]
    registration_paths: dict[int, Path]
    panels: dict[int, RgbPanel]
    config: dict[str, Any]
    config_path: Path
    config_bytes: bytes
    config_sha256: str
    expected_rgb_complete_sha256s: dict[int, str]
    expected_confirm_authorization_complete_sha256: str | None
    output_dir: Path


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _absolute_lexical(path: Path) -> Path:
    """Return an absolute path without following a dangling final symlink."""
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _require_no_symlink_parent(path: Path, where: str) -> None:
    """Reject an output path whose existing ancestry contains a symlink."""
    absolute = _absolute_lexical(path)
    current = Path(absolute.anchor)
    for component in absolute.parts[1:-1]:
        current /= component
        if os.path.lexists(current):
            require(not current.is_symlink(), f"{where} has symlink parent: {current}")


def _check_output_separation(
    output: Path,
    input_paths: Sequence[Path],
    *,
    lexical_only_paths: Sequence[Path] = (),
) -> Path:
    """Reject equality/nesting against every input before staging can mutate it."""
    destination = _absolute_lexical(output)
    _require_no_symlink_parent(destination, "output")
    require(not os.path.lexists(destination), f"output already exists: {destination}")
    resolved_destination = destination.resolve(strict=False)
    for raw in input_paths:
        source = raw.expanduser().resolve()
        require(resolved_destination != source, "output aliases an input")
        require(
            not resolved_destination.is_relative_to(source),
            f"output is nested in input: {source}",
        )
        require(
            not source.is_relative_to(resolved_destination),
            f"output would contain input: {source}",
        )
    for raw in lexical_only_paths:
        source = _absolute_lexical(raw)
        require(destination != source, "output lexically aliases protected input")
        require(
            not destination.is_relative_to(source),
            f"output is lexically nested in protected input: {source}",
        )
        require(
            not source.is_relative_to(destination),
            f"output would lexically contain protected input: {source}",
        )
    return destination


def _read_sealed_regular_bytes(
    path: Path,
    expected_sha256: str,
    where: str,
    *,
    expected_bytes: int | None = None,
) -> bytes:
    """Hash and consume one regular file from the same no-follow descriptor."""
    require(is_sha256(expected_sha256), f"{where} expected SHA invalid")
    absolute = _absolute_lexical(path)
    _require_no_symlink_parent(absolute, where)
    require(not absolute.is_symlink(), f"{where} is a symlink")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(absolute, flags)
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"{where} is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
        after = os.fstat(descriptor)
        require(
            (before.st_dev, before.st_ino, before.st_size)
            == (after.st_dev, after.st_ino, after.st_size),
            f"{where} changed while being read",
        )
    finally:
        os.close(descriptor)
    current = os.stat(absolute, follow_symlinks=False)
    require(
        (current.st_dev, current.st_ino, current.st_size)
        == (before.st_dev, before.st_ino, before.st_size),
        f"{where} path was replaced while being read",
    )
    if expected_bytes is not None:
        require(len(payload) == expected_bytes, f"{where} byte-size drift")
    require(
        hashlib.sha256(payload).hexdigest() == expected_sha256,
        f"{where} SHA-256 drift",
    )
    return payload


def _strict_json_bytes(payload: bytes, where: str) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise Gate0ContractError(f"{where} is not UTF-8") from error
    value = loads_strict_json(text, where=where)
    require(isinstance(value, dict), f"{where} must be an object")
    return dict(value)


def _strict_json_file(path: Path, where: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"{where} missing/symlink")
    value = loads_strict_json(path.read_text(encoding="utf-8"), where=where)
    require(isinstance(value, dict), f"{where} must be an object")
    return dict(value)


def _strict_jsonl(path: Path, where: str) -> list[dict[str, Any]]:
    require(path.is_file() and not path.is_symlink(), f"{where} missing/symlink")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            require(bool(line.strip()), f"{where}:{line_number}: blank row")
            value = loads_strict_json(line, where=f"{where}:{line_number}")
            require(isinstance(value, dict), f"{where}:{line_number}: object required")
            rows.append(dict(value))
    require(bool(rows), f"{where} is empty")
    return rows


def _strict_jsonl_bytes(payload: bytes, where: str) -> list[dict[str, Any]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise Gate0ContractError(f"{where} is not UTF-8") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        require(bool(line.strip()), f"{where}:{line_number}: blank row")
        value = loads_strict_json(line, where=f"{where}:{line_number}")
        require(isinstance(value, dict), f"{where}:{line_number}: object required")
        rows.append(dict(value))
    require(bool(rows), f"{where} is empty")
    return rows


def _load_module(path: Path, name: str) -> ModuleType:
    require(path.is_file(), f"registered module missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_registered_module(
    path: Path, expected_sha256: str, name: str
) -> ModuleType:
    payload = _read_sealed_regular_bytes(
        path, expected_sha256, f"registered source {name}"
    )
    module = ModuleType(name)
    module.__file__ = path.as_posix()
    sys.modules[name] = module
    exec(compile(payload, path.as_posix(), "exec"), module.__dict__)  # noqa: S102
    return module


def _reject_public_leakage(value: Any, where: str = "$public") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            require(
                normalized not in PUBLIC_FORBIDDEN_KEYS,
                f"{where}.{key}: forbidden public field",
            )
            _reject_public_leakage(child, f"{where}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_public_leakage(child, f"{where}[{index}]")


def validate_semantic_config(value: Any) -> dict[str, Any]:
    config = dict(require_exact_keys(value, TOP_CONFIG_FIELDS, "semantic config"))
    require(config["schema"] == CONFIG_SCHEMA, "semantic config schema")
    require(
        config["status"] == "candidate_frozen_before_any_8000_8100_rgb",
        "semantic config status",
    )
    require(config["task"] == TASK, "semantic config task")

    observation = require_exact_keys(
        config["observation_contract"],
        OBSERVATION_CONFIG_FIELDS,
        "semantic config observation_contract",
    )
    require(observation["cameras"] == ["agentview", "eye_in_hand"], "camera order")
    require(observation["causal"] is True, "semantic producer must be causal")
    required_forbidden = {
        "done",
        "episode_length",
        "events",
        "is_success",
        "proprio",
        "simulator_state",
        "source_summary",
        "success",
        "terminal",
    }
    require(
        set(observation["forbidden_inputs"]) == required_forbidden,
        "semantic forbidden-input set drift",
    )
    require(observation["primary_tracking_camera"] == "agentview", "tracking camera")
    require(observation["cream_identity_camera"] == "eye_in_hand", "cream camera")

    sam3 = require_exact_keys(config["sam3"], SAM3_CONFIG_FIELDS, "semantic config sam3")
    require(
        sam3["prompts"]
        == {
            "basket": "white woven basket",
            "can": "tomato sauce can",
            "cream": "cream cheese box",
        },
        "SAM3 prompts drift",
    )
    expected_sam3 = {
        "detection_score_threshold": 0.1,
        "new_track_score_threshold": 0.1,
        "detector_nms_iou_threshold": 0.1,
        "duplicate_mask_iou_gte": 0.5,
        "duplicate_box_iou_gte": 0.7,
        "max_can_instances": 2,
        "device": "cuda",
        "state_device": "cpu",
        "dtype": "bfloat16",
        "global_seed": 0,
    }
    for key, expected in expected_sam3.items():
        require(sam3[key] == expected, f"semantic config sam3.{key} drift")
    require(
        sam3["mask_area_bounds_inclusive"]
        == {"basket": [3000, 15000], "can": [64, 5000], "cream": [64, 5000]},
        "SAM3 area bounds drift",
    )

    sift = require_exact_keys(config["sift"], SIFT_CONFIG_FIELDS, "semantic config sift")
    expected_sift = {
        "upscale": 3,
        "contrast_threshold": 0.01,
        "lowe_ratio": 0.7,
        "cluster_eps_px": 22.0,
        "min_cluster_matches": 4,
        "ransac_reprojection_threshold_px": 3.0,
        "ransac_max_iters": 2000,
        "ransac_confidence": 0.99,
        "ransac_refine_iters": 10,
        "ransac_min_inliers": 4,
        "ransac_min_inlier_ratio": 0.5,
        "scale_range_inclusive": [0.5, 4.0],
        "opencv_rng_seed": 0,
    }
    require(dict(sift) == expected_sift, "SIFT config drift")
    geometry = require_exact_keys(
        config["geometry"], GEOMETRY_CONFIG_FIELDS, "semantic config geometry"
    )
    require(
        dict(geometry)
        == {
            "cream_rim_margin_px": 8.0,
            "cream_rim_max_depth_fraction": 0.3,
            "cream_can_cross_class_box_ios_gte": 0.5,
        },
        "cream geometry drift",
    )
    CanStateConfig.from_dict(config["can_state"])
    CreamStateConfig.from_dict(config["cream_state"])

    progress = require_exact_keys(
        config["progress"], PROGRESS_CONFIG_FIELDS, "semantic config progress"
    )
    require(progress["atoms"] == ["can_ge1", "can_ge2", "cream"], "atom order")
    require(progress["scalar"] == "can_ge1+can_ge2+cream", "scalar rule")
    require(progress["sticky"] is True, "progress must be sticky")
    require(progress["can_identity_quotient"] == "anonymous_top_two", "can quotient")
    require(progress["prefix_check_frames_per_episode"] == 8, "prefix length")
    require(progress["prefix_float_round_decimals"] == 6, "prefix precision")

    schemas = require_exact_keys(
        config["schemas"], SCHEMA_CONFIG_FIELDS, "semantic config schemas"
    )
    require(schemas["result"] == RESULT_SCHEMA, "result schema drift")
    require(schemas["frame"] == FRAME_SCHEMA, "frame schema drift")
    require(schemas["complete"] == COMPLETE_SCHEMA, "complete schema drift")
    require(
        schemas["evaluation"] == "v248_gate0_progress_evaluation_v1",
        "evaluation schema drift",
    )
    screen = require_exact_keys(
        config["screen"], SCREEN_CONFIG_FIELDS, "semantic config screen"
    )
    require(screen["promotion_authorized"] is False, "screen may never promote")
    require(screen["required_primary_chain3_episodes"] == 24, "screen size")
    require(isinstance(screen["claim"], str) and screen["claim"], "screen claim")
    gates = require_exact_keys(
        config["promotion_gates"],
        PROMOTION_CONFIG_FIELDS,
        "semantic config promotion_gates",
    )
    require(gates["required_primary_chain3_episodes"] == 192, "confirm size")
    for key in (
        "minimum_evidence_coverage",
        "maximum_uncertainty_rate",
        "minimum_exact_scalar_accuracy",
        "minimum_macro_f1",
        "minimum_never_achieved_terminal_specificity",
        "minimum_held_cream_specificity",
    ):
        require(_finite_number(gates[key]), f"non-finite promotion gate {key}")
    for key in (
        "maximum_missing_transitions",
        "maximum_extra_transitions",
        "maximum_median_timing_error_steps",
        "maximum_p95_timing_error_steps",
        "maximum_timing_error_steps",
        "minimum_chain3_cream_positive_transitions",
        "maximum_carry_created_transitions",
    ):
        require(type(gates[key]) is int and gates[key] >= 0, f"invalid gate {key}")
    require(gates["require_all_prefix_checks_exact"] is True, "prefix gate")
    require(gates["require_terminal_recovery_pass"] is True, "recovery gate")
    return config


def load_semantic_config(path: Path) -> tuple[dict[str, Any], str, bytes]:
    payload = _read_sealed_regular_bytes(
        path,
        EXPECTED_CONFIG_SHA256,
        "semantic config",
    )
    return (
        validate_semantic_config(_strict_json_bytes(payload, "semantic config")),
        EXPECTED_CONFIG_SHA256,
        payload,
    )


def _expected_episode_schedule(spec: Mapping[str, Any]) -> tuple[list[int], list[str]]:
    if spec["mode"] == "ordinary":
        return list(range(0, 750, 10)) + [750], ["pre_action"] * 75 + ["n_plus_1"]
    return (
        list(range(0, 650, 10)) + [650, 651],
        ["pre_action"] * 65 + ["n_plus_1", "recovered_endpoint"],
    )


def _validate_cross_registration_bindings(
    registrations: Mapping[int, Mapping[str, Any]],
) -> None:
    require(set(registrations) == {8000, 8100}, "exactly panels 8000 and 8100 required")
    left = registrations[8000]
    right = registrations[8100]
    require(left["phase"] == right["phase"], "registration phases differ")
    for key in (
        "models",
        "templates",
        "sources",
        "runtime",
        "semantic_contract",
    ):
        require(left[key] == right[key], f"cross-panel registration {key} drift")
    for key in ("schema", "campaign_id", "protocol_sha256", "projection_sha256"):
        require(
            left["campaign"][key] == right["campaign"][key],
            f"cross-panel campaign {key} drift",
        )


def _load_registration_pair(
    paths: Sequence[Path], expected_hashes: Sequence[str], role: str
) -> tuple[dict[int, dict[str, Any]], dict[int, str]]:
    require(len(paths) == len(expected_hashes) == 2, "two registrations/hashes required")
    registrations: dict[int, dict[str, Any]] = {}
    hashes: dict[int, str] = {}
    for path, expected in zip(paths, expected_hashes, strict=True):
        require(is_sha256(expected), "invalid expected registration SHA")
        registration, actual = load_and_validate_registration(
            path,
            expected_source_role=role,
            expected_file_sha256=expected,
        )
        require(actual == expected, "external registration SHA anchor mismatch")
        seed_start = int(registration["panel"]["seed_start"])
        require(seed_start not in registrations, "duplicate registration panel")
        registrations[seed_start] = registration
        hashes[seed_start] = actual
    _validate_cross_registration_bindings(registrations)
    return registrations, hashes


def _load_rgb_support(registration: Mapping[str, Any]) -> ModuleType:
    reference = registration["sources"]["formal_replay"]
    relative = reference["relative_path"]
    path = (REPO / relative).resolve()
    require(path.is_relative_to(REPO), "formal replay path escapes repository")
    return _load_registered_module(
        path, reference["sha256"], "v248_registered_rgb_support"
    )


def _validate_rgb_panel(
    root: Path,
    registration: Mapping[str, Any],
    registration_sha256: str,
    expected_complete_sha256: str,
    expected_authority: Mapping[str, Any],
    support: ModuleType,
) -> RgbPanel:
    lexical_root = _absolute_lexical(root)
    require(not lexical_root.is_symlink(), f"RGB run root is a symlink: {lexical_root}")
    root = lexical_root.resolve()
    require(root.is_dir(), f"missing RGB run: {root}")
    result_path = root / "RESULT.json"
    manifest_path = root / "manifest.jsonl"
    complete_path = root / "COMPLETE.json"
    producer_path = root / "PRODUCER.py"
    complete_bytes = _read_sealed_regular_bytes(
        complete_path,
        expected_complete_sha256,
        "externally anchored RGB COMPLETE",
    )
    complete = _strict_json_bytes(complete_bytes, "RGB COMPLETE")
    require_exact_keys(complete, support.COMPLETE_FIELDS, "RGB COMPLETE")
    support._validate_result_and_complete(root, registration, expected_authority)
    _read_sealed_regular_bytes(
        complete_path,
        expected_complete_sha256,
        "RGB COMPLETE after panel validation",
    )
    result_bytes = _read_sealed_regular_bytes(
        result_path, complete["result_sha256"], "sealed RGB RESULT"
    )
    manifest_bytes = _read_sealed_regular_bytes(
        manifest_path, complete["manifest_sha256"], "sealed RGB manifest"
    )
    producer_bytes = _read_sealed_regular_bytes(
        producer_path, complete["producer_sha256"], "sealed RGB producer"
    )
    result = _strict_json_bytes(result_bytes, "RGB RESULT")
    rows = _strict_jsonl_bytes(manifest_bytes, "RGB manifest")
    panel_id = registration["panel"]["panel_id"]
    phase = registration["phase"]
    expected_episodes = [
        int(item["episode_id"]) for item in registration["panel"]["episodes"]
    ]
    require(result["schema"] == RGB_RESULT_SCHEMA, "RGB result schema")
    require(
        result.get("authority")
        == complete.get("authority")
        == dict(expected_authority),
        "captured RGB authority differs from frozen campaign",
    )
    require(result["panel_id"] == panel_id and result["phase"] == phase, "RGB phase")
    require(result["task"] == TASK, "RGB task")
    require(result["episode_ids"] == expected_episodes, "RGB episode set/order")
    require(
        result["registration"]["file_sha256"] == registration_sha256,
        "RGB registration file binding",
    )
    require(
        result["registration"]["canonical_self_sha256"]
        == registration["self_sha256"],
        "RGB registration self binding",
    )
    require(
        complete["registration_sha256"] == registration_sha256
        and complete["registration_self_sha256"] == registration["self_sha256"],
        "RGB COMPLETE registration binding",
    )
    require(complete["panel_id"] == panel_id and complete["phase"] == phase, "RGB COMPLETE phase")
    require(
        result["inputs"]["sanitized_tape_sha256"]
        == registration["sanitized_source"]["tape_sha256"],
        "RGB sanitized tape binding",
    )
    require(
        result["inputs"]["source_tape_sha256"]
        == registration["sanitized_source"]["source_tape_sha256"],
        "RGB source tape binding",
    )
    require(
        result["output"]["manifest_sha256"] == complete["manifest_sha256"]
        and result["output"]["producer_sha256"] == complete["producer_sha256"],
        "captured RGB RESULT/COMPLETE seal drift",
    )

    by_episode: dict[int, list[dict[str, Any]]] = {
        episode_id: [] for episode_id in expected_episodes
    }
    seen_image_paths: set[str] = set()
    image_tree_records: list[dict[str, Any]] = []
    for global_index, raw in enumerate(rows):
        row = dict(support.validate_manifest_row(raw))
        require(row["frame_index"] == global_index, "RGB global frame index")
        require(row["panel_id"] == panel_id and row["phase"] == phase, "RGB row phase")
        require(row["task"] == TASK, "RGB row task")
        episode_id = int(row["episode_id"])
        require(episode_id in by_episode, "RGB row outside registration episode set")
        require(
            row["env_seed"] == registration["panel"]["seed_start"] + episode_id,
            "RGB row env seed",
        )
        require(
            row["source"]["tape_sha256"]
            == registration["sanitized_source"]["tape_sha256"],
            "RGB row tape binding",
        )
        by_episode[episode_id].append(row)
    flattened_order = [int(row["episode_id"]) for row in rows]
    expected_flattened = []
    for spec in registration["panel"]["episodes"]:
        episode_id = int(spec["episode_id"])
        times, roles = _expected_episode_schedule(spec)
        episode_rows = by_episode[episode_id]
        require([int(row["t"]) for row in episode_rows] == times, "RGB time schedule")
        require(
            [row["source"]["boundary_role"] for row in episode_rows] == roles,
            "RGB boundary-role schedule",
        )
        expected_flattened.extend([episode_id] * len(times))
        for local_index, row in enumerate(episode_rows):
            expected_id = f"{panel_id}:{phase}:ep{episode_id:02d}:t{row['t']:04d}"
            require(row["frame_id"] == expected_id, "RGB frame ID binding")
            for camera in CAMERAS:
                image = row["images"][camera]
                relative = str(image["path"])
                require(relative not in seen_image_paths, "duplicate RGB image path")
                seen_image_paths.add(relative)
                image_path = _absolute_lexical(root / relative)
                require(image_path.is_relative_to(root), "RGB image escapes run")
                captured_image = _read_sealed_regular_bytes(
                    image_path,
                    str(image["sha256"]),
                    f"sealed RGB {camera} {row['frame_id']}",
                )
                image_tree_records.append(
                    {
                        "path": relative,
                        "sha256": hashlib.sha256(captured_image).hexdigest(),
                        "bytes": len(captured_image),
                    }
                )
            require(local_index < len(times), "internal RGB local index")
    require(flattened_order == expected_flattened, "RGB episode ordering")

    recovery = result["terminal_recovery_audit"]
    if registration["panel"]["seed_start"] == 8100:
        expected_recovery = {
            "enabled": True,
            "episode_id": 77,
            "stored_chunks": 65,
            "stored_chunks_exact_bitwise": True,
            "two_fresh_replays": 2,
            "cross_run_boundary_hashes_exact": True,
            "cross_run_decoded_actions_exact": True,
            "halted_at_registered_t": True,
            "phi_only": True,
            "dynamics_transition_eligible": False,
            "recovered_dynamics_rows_written": 0,
        }
        for key, expected in expected_recovery.items():
            require(recovery[key] == expected, f"RGB recovery {key}")
        require(
            is_sha256(recovery["recovered_chunk_tensor_sha256"]),
            "RGB recovered chunk hash",
        )
    else:
        require(recovery["enabled"] is False, "unexpected RGB recovery")

    image_tree_records.sort(key=lambda item: item["path"])
    image_tree_sha = canonical_sha256(image_tree_records)
    require(
        image_tree_sha
        == complete["image_tree_sha256"]
        == result["output"]["image_tree_sha256"],
        "captured RGB image tree does not match sealed artifact",
    )
    seals = {
        "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "complete_sha256": expected_complete_sha256,
        "producer_sha256": hashlib.sha256(producer_bytes).hexdigest(),
        "image_tree_sha256": image_tree_sha,
        "frames": len(rows),
    }
    require(
        seals["producer_sha256"]
        == registration["sources"]["formal_replay"]["sha256"],
        "RGB producer snapshot is not the registered formal replay source",
    )
    return RgbPanel(
        root=root,
        panel_id=panel_id,
        phase=phase,
        result=result,
        complete=complete,
        rows=tuple(rows),
        seals=seals,
    )


def validate_authority(
    value: Any,
    registrations: Mapping[int, Mapping[str, Any]],
    phase: str,
    *,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    authority = dict(require_exact_keys(value, AUTHORITY_FIELDS, "progress authority"))
    for key in (
        "freeze_complete_sha256",
        "campaign_manifest_sha256",
        "campaign_body_sha256",
        "campaign_projection_sha256",
    ):
        require(is_sha256(authority[key]), f"progress authority {key} invalid")
    require(phase in {"screen", "confirm"}, "progress authority phase")
    campaigns = [registrations[seed]["campaign"] for seed in (8000, 8100)]
    require(
        all(item["campaign_id"] == authority["campaign_id"] for item in campaigns),
        "progress authority campaign ID drift",
    )
    require(
        all(
            item["projection_sha256"]
            == authority["campaign_projection_sha256"]
            for item in campaigns
        ),
        "progress authority projection drift",
    )
    expected_slots = [
        f"{PANEL_IDS[seed]}_{phase}.json" for seed in (8000, 8100)
    ]
    require(
        authority["registration_slots"] == expected_slots,
        "progress authority registration slots drift",
    )
    require(
        authority["output_complete_slot"] == f"{phase}/progress/COMPLETE.json",
        "progress authority output slot drift",
    )
    if phase == "screen":
        require(
            authority["confirm_authorization_complete_sha256"] is None,
            "screen progress must not consume confirm authorization",
        )
    else:
        require(
            is_sha256(authority["confirm_authorization_complete_sha256"]),
            "confirm progress lacks external confirm-authorization anchor",
        )
    if expected is not None:
        require(authority == dict(expected), "progress authority differs from campaign")
    return authority


def _phase_from_frozen_rgb_slots(
    bundle: Mapping[str, Any], rgb_roots: Sequence[Path]
) -> tuple[str, dict[Path, int]]:
    """Identify the phase from frozen paths without touching an RGB artifact."""
    require(len(rgb_roots) == 2, "exactly two RGB run paths required")
    supplied = [_absolute_lexical(root) for root in rgb_roots]
    require(len(set(supplied)) == 2, "duplicate RGB run path")
    target_root = Path(bundle["target_root"])
    matches: list[tuple[str, dict[Path, int]]] = []
    for phase in ("screen", "confirm"):
        expected = {
            _absolute_lexical(
                (
                    target_root
                    / bundle["target_output_slots"][
                        f"{phase}_rgb_complete_by_panel"
                    ][PANEL_IDS[seed]]
                ).parent
            ): seed
            for seed in (8000, 8100)
        }
        if set(supplied) == set(expected):
            matches.append((phase, expected))
    require(
        len(matches) == 1,
        "RGB run paths do not exactly equal one frozen screen/confirm panel pair",
    )
    return matches[0]


def _validate_confirm_authorization(
    bundle: Mapping[str, Any],
    expected_complete_sha256: str,
    rgb_support: ModuleType,
) -> Path:
    """Run the registered replay/authorizer's complete A->F/P/S/Q/E validator."""
    require(is_sha256(expected_complete_sha256), "confirm authorization SHA invalid")
    validated = rgb_support._validate_confirm_authorization(
        bundle, expected_complete_sha256
    )
    require(
        isinstance(validated, Mapping),
        "registered authorizer returned a non-object validation result",
    )
    require(
        "root" in validated and "complete_sha256" in validated,
        "registered authorizer validation result is incomplete",
    )
    slot = bundle["target_output_slots"]["confirm_authorization_complete"]
    expected_root = _absolute_lexical((Path(bundle["target_root"]) / slot).parent)
    require(
        _absolute_lexical(Path(validated["root"])) == expected_root,
        "registered authorizer returned a different frozen slot",
    )
    require(
        validated["complete_sha256"] == expected_complete_sha256,
        "registered authorizer returned a different COMPLETE anchor",
    )
    return expected_root


def prepare_inputs(
    *,
    freeze_bundle_root: Path,
    expected_freeze_complete_sha256: str,
    rgb_roots: Sequence[Path],
    expected_rgb_complete_sha256s: Sequence[str],
    output_dir: Path | None = None,
    expected_confirm_authorization_complete_sha256: str | None = None,
) -> PreparedInputs:
    bundle = load_and_validate_campaign_bundle(
        freeze_bundle_root,
        expected_freeze_complete_sha256,
        expected_source_role="semantic_producer",
    )
    require(
        len(rgb_roots) == len(expected_rgb_complete_sha256s) == 2,
        "exactly two RGB runs and external COMPLETE anchors required",
    )
    phase, frozen_root_to_seed = _phase_from_frozen_rgb_slots(bundle, rgb_roots)
    for expected_complete in expected_rgb_complete_sha256s:
        require(is_sha256(expected_complete), "invalid expected RGB COMPLETE SHA")
    if phase == "screen":
        require(
            expected_confirm_authorization_complete_sha256 is None,
            "screen progress must not accept confirm authorization",
        )
    else:
        require(
            is_sha256(expected_confirm_authorization_complete_sha256),
            "confirm progress requires external confirm authorization COMPLETE SHA",
        )

    # Loading the registered replay source touches only the frozen source bundle.
    # For confirm, its complete authorization validator must finish before even
    # stat'ing, resolving, or opening a confirm RGB root.
    support = _load_rgb_support(
        bundle["registrations"][f"chain3_8000_{phase}.json"]["registration"]
    )
    authorization_root: Path | None = None
    if phase == "confirm":
        assert expected_confirm_authorization_complete_sha256 is not None
        authorization_root = _validate_confirm_authorization(
            bundle,
            expected_confirm_authorization_complete_sha256,
            support,
        )

    registrations: dict[int, dict[str, Any]] = {}
    registration_hashes: dict[int, str] = {}
    registration_path_by_seed: dict[int, Path] = {}
    panels: dict[int, RgbPanel] = {}
    expected_by_seed: dict[int, str] = {}
    for root, expected_complete in zip(
        rgb_roots, expected_rgb_complete_sha256s, strict=True
    ):
        lexical_candidate = _absolute_lexical(root)
        require(
            lexical_candidate in frozen_root_to_seed,
            "RGB run path escaped the frozen phase slots",
        )
        seed_start = frozen_root_to_seed[lexical_candidate]
        require(not lexical_candidate.is_symlink(), "RGB run root is a symlink")
        candidate = lexical_candidate.resolve()
        require(seed_start not in panels, "duplicate RGB panel")
        panel_id = PANEL_IDS[seed_start]
        slot = f"{panel_id}_{phase}.json"
        require(slot in bundle["registrations"], "RGB registration slot not in campaign")
        entry = bundle["registrations"][slot]
        registration = entry["registration"]
        output_slot = bundle["target_output_slots"][
            f"{phase}_rgb_complete_by_panel"
        ][panel_id]
        expected_root = (Path(bundle["target_root"]) / output_slot).parent
        require(candidate == expected_root, "RGB run is not in its frozen campaign slot")
        expected_authority = {
            "freeze_complete_sha256": bundle["freeze_complete_sha256"],
            "campaign_manifest_sha256": bundle["campaign_manifest_sha256"],
            "campaign_body_sha256": bundle["campaign"]["body_sha256"],
            "campaign_projection_sha256": bundle["campaign"]["projection_sha256"],
            "campaign_id": bundle["campaign"]["campaign_id"],
            "registration_slot": slot,
            "output_complete_slot": output_slot,
            "confirm_authorization_complete_sha256": (
                expected_confirm_authorization_complete_sha256
            ),
        }
        panels[seed_start] = _validate_rgb_panel(
            candidate,
            registration,
            entry["file_sha256"],
            expected_complete,
            expected_authority,
            support,
        )
        registrations[seed_start] = registration
        registration_hashes[seed_start] = entry["file_sha256"]
        registration_path_by_seed[seed_start] = Path(entry["path"])
        expected_by_seed[seed_start] = expected_complete
    require(set(panels) == {8000, 8100}, "both RGB panels are required")
    _validate_cross_registration_bindings(registrations)
    semantic_ref = registrations[8000]["templates"]["semantic_config"]
    require(
        semantic_ref == registrations[8100]["templates"]["semantic_config"],
        "cross-panel semantic config reference drift",
    )
    require(semantic_ref["sha256"] == EXPECTED_CONFIG_SHA256, "registered config SHA")
    config_path = Path(semantic_ref["path"])
    config, config_sha, config_bytes = load_semantic_config(config_path)
    protected_oracles = [
        Path(registrations[seed]["oracle"]["summary"]["path"])
        for seed in (8000, 8100)
    ]
    progress_slot = bundle["target_output_slots"][f"{phase}_progress_complete"]
    expected_output = (Path(bundle["target_root"]) / progress_slot).parent
    if output_dir is not None:
        require(
            _absolute_lexical(output_dir).resolve(strict=False) == expected_output,
            "progress output is not its frozen campaign slot",
        )
    destination = _check_output_separation(
        expected_output,
        [
            Path(bundle["root"]),
            *registration_path_by_seed.values(),
            *(panel.root for panel in panels.values()),
            config_path,
            *([authorization_root] if authorization_root is not None else []),
        ],
        lexical_only_paths=protected_oracles,
    )
    registration_slots = [
        f"{PANEL_IDS[seed]}_{phase}.json" for seed in (8000, 8100)
    ]
    authority = {
        "freeze_complete_sha256": bundle["freeze_complete_sha256"],
        "campaign_manifest_sha256": bundle["campaign_manifest_sha256"],
        "campaign_body_sha256": bundle["campaign"]["body_sha256"],
        "campaign_projection_sha256": bundle["campaign"]["projection_sha256"],
        "campaign_id": bundle["campaign"]["campaign_id"],
        "registration_slots": registration_slots,
        "output_complete_slot": progress_slot,
        "confirm_authorization_complete_sha256": (
            expected_confirm_authorization_complete_sha256
        ),
    }
    validate_authority(authority, registrations, phase, expected=authority)
    return PreparedInputs(
        campaign_bundle=bundle,
        authority=authority,
        registrations=registrations,
        registration_hashes=registration_hashes,
        registration_paths=registration_path_by_seed,
        panels=panels,
        config=config,
        config_path=config_path,
        config_bytes=config_bytes,
        config_sha256=config_sha,
        expected_rgb_complete_sha256s=expected_by_seed,
        expected_confirm_authorization_complete_sha256=(
            expected_confirm_authorization_complete_sha256
        ),
        output_dir=destination,
    )


def revalidate_static_inputs(
    prepared: PreparedInputs,
) -> tuple[ModuleType, Path, Path]:
    """Recheck inputs and return fresh executable/output publication authority."""
    authority = dict(
        require_exact_keys(prepared.authority, AUTHORITY_FIELDS, "progress authority")
    )
    bundle = load_and_validate_campaign_bundle(
        prepared.campaign_bundle["root"],
        authority["freeze_complete_sha256"],
        expected_source_role="semantic_producer",
    )
    phase_matches = [
        phase
        for phase in ("screen", "confirm")
        if authority["registration_slots"]
        == [f"{PANEL_IDS[seed]}_{phase}.json" for seed in (8000, 8100)]
    ]
    require(
        len(phase_matches) == 1,
        "prepared authority does not identify one frozen progress phase",
    )
    phase = phase_matches[0]
    registration_slots = [
        f"{PANEL_IDS[seed]}_{phase}.json" for seed in (8000, 8100)
    ]
    entries = {
        seed: bundle["registrations"][registration_slots[index]]
        for index, seed in enumerate((8000, 8100))
    }
    registrations = {
        seed: entries[seed]["registration"] for seed in (8000, 8100)
    }
    hashes = {
        seed: entries[seed]["file_sha256"] for seed in (8000, 8100)
    }
    registration_paths = {
        seed: Path(entries[seed]["path"]) for seed in (8000, 8100)
    }
    _validate_cross_registration_bindings(registrations)
    require(registrations == prepared.registrations, "registration changed during inference")
    require(hashes == prepared.registration_hashes, "registration hash changed during inference")
    require(
        registration_paths == prepared.registration_paths,
        "prepared registration paths differ from frozen bundle",
    )

    expected_authorization = authority["confirm_authorization_complete_sha256"]
    require(
        expected_authorization
        == prepared.expected_confirm_authorization_complete_sha256,
        "prepared confirm authorization anchor drift",
    )
    progress_slot = bundle["target_output_slots"][f"{phase}_progress_complete"]
    expected_authority = {
        "freeze_complete_sha256": bundle["freeze_complete_sha256"],
        "campaign_manifest_sha256": bundle["campaign_manifest_sha256"],
        "campaign_body_sha256": bundle["campaign"]["body_sha256"],
        "campaign_projection_sha256": bundle["campaign"]["projection_sha256"],
        "campaign_id": bundle["campaign"]["campaign_id"],
        "registration_slots": registration_slots,
        "output_complete_slot": progress_slot,
        "confirm_authorization_complete_sha256": expected_authorization,
    }
    validate_authority(
        authority,
        registrations,
        phase,
        expected=expected_authority,
    )
    target_root = Path(bundle["target_root"])
    expected_output = _absolute_lexical((target_root / progress_slot).parent)
    require(
        prepared.output_dir == expected_output,
        "prepared output path differs from frozen progress slot",
    )

    semantic_ref = registrations[8000]["templates"]["semantic_config"]
    require(
        semantic_ref == registrations[8100]["templates"]["semantic_config"],
        "cross-panel semantic config reference drift",
    )
    require(semantic_ref["sha256"] == EXPECTED_CONFIG_SHA256, "registered config SHA")
    config_path = Path(semantic_ref["path"])
    require(
        config_path == prepared.config_path,
        "prepared semantic config path differs from frozen registration",
    )

    support = _load_rgb_support(registrations[8000])
    authorization_root: Path | None = None
    if phase == "confirm":
        require(
            is_sha256(expected_authorization),
            "confirm authorization anchor disappeared",
        )
        assert isinstance(expected_authorization, str)
        authorization_root = _validate_confirm_authorization(
            bundle,
            expected_authorization,
            support,
        )
    config, config_sha, config_bytes = load_semantic_config(config_path)
    require(config == prepared.config, "semantic config value changed")
    require(config_sha == prepared.config_sha256, "semantic config hash changed")
    require(config_bytes == prepared.config_bytes, "semantic config bytes changed")
    require(set(prepared.panels) == {8000, 8100}, "prepared RGB panel set drift")
    require(
        set(prepared.expected_rgb_complete_sha256s) == {8000, 8100},
        "prepared RGB anchor set drift",
    )
    panel_roots: dict[int, Path] = {}
    for seed in (8000, 8100):
        panel = prepared.panels[seed]
        panel_id = PANEL_IDS[seed]
        registration_slot = f"{panel_id}_{phase}.json"
        output_slot = bundle["target_output_slots"][
            f"{phase}_rgb_complete_by_panel"
        ][panel_id]
        panel_root = _absolute_lexical((target_root / output_slot).parent)
        require(
            panel.root == panel_root,
            "prepared RGB root differs from frozen campaign slot",
        )
        panel_roots[seed] = panel_root
        expected_rgb_authority = {
            "freeze_complete_sha256": bundle["freeze_complete_sha256"],
            "campaign_manifest_sha256": bundle["campaign_manifest_sha256"],
            "campaign_body_sha256": bundle["campaign"]["body_sha256"],
            "campaign_projection_sha256": bundle["campaign"]["projection_sha256"],
            "campaign_id": bundle["campaign"]["campaign_id"],
            "registration_slot": registration_slot,
            "output_complete_slot": output_slot,
            "confirm_authorization_complete_sha256": (
                expected_authorization
            ),
        }
        checked = _validate_rgb_panel(
            panel_root,
            registrations[seed],
            hashes[seed],
            prepared.expected_rgb_complete_sha256s[seed],
            expected_rgb_authority,
            support,
        )
        require(checked.rows == panel.rows, f"RGB manifest changed for panel {seed}")
        require(checked.seals == panel.seals, f"RGB seals changed for panel {seed}")

    destination = _check_output_separation(
        expected_output,
        [
            *registration_paths.values(),
            Path(bundle["root"]),
            *panel_roots.values(),
            config_path,
            *([authorization_root] if authorization_root is not None else []),
        ],
        lexical_only_paths=[
            Path(registrations[seed]["oracle"]["summary"]["path"])
            for seed in (8000, 8100)
        ],
    )
    require(destination == expected_output, "fresh output path changed")
    return support, destination, target_root


def _assert_engine_constants(
    sam3_engine: ModuleType,
    sift_engine: ModuleType,
    config: Mapping[str, Any],
    registration: Mapping[str, Any],
) -> None:
    sam3 = config["sam3"]
    require(sam3_engine.PROMPTS == sam3["prompts"], "SAM3 prompt drift")
    sam3_constants = {
        "DETECTION_SCORE_THRESHOLD": sam3["detection_score_threshold"],
        "NEW_TRACK_SCORE_THRESHOLD": sam3["new_track_score_threshold"],
        "DETECTOR_NMS_IOU_THRESHOLD": sam3["detector_nms_iou_threshold"],
        "DOWNSTREAM_DUPLICATE_MASK_IOU_GTE": sam3["duplicate_mask_iou_gte"],
        "DOWNSTREAM_DUPLICATE_BOX_IOU_GTE": sam3["duplicate_box_iou_gte"],
        "MAX_CAN_INSTANCES": sam3["max_can_instances"],
    }
    for name, expected in sam3_constants.items():
        require(getattr(sam3_engine, name) == expected, f"SAM3 engine {name} drift")
    require(
        {key: list(value) for key, value in sam3_engine.MASK_AREA_BOUNDS.items()}
        == sam3["mask_area_bounds_inclusive"],
        "SAM3 engine area bounds drift",
    )
    sift = config["sift"]
    sift_constants = {
        "SIFT_UPSCALE": sift["upscale"],
        "SIFT_CONTRAST_THRESHOLD": sift["contrast_threshold"],
        "SIFT_LOWE_RATIO": sift["lowe_ratio"],
        "SIFT_CLUSTER_EPS_PX": sift["cluster_eps_px"],
        "SIFT_MIN_CLUSTER_MATCHES": sift["min_cluster_matches"],
        "SIFT_RANSAC_REPROJECTION_THRESHOLD_PX": sift[
            "ransac_reprojection_threshold_px"
        ],
        "SIFT_RANSAC_MAX_ITERS": sift["ransac_max_iters"],
        "SIFT_RANSAC_CONFIDENCE": sift["ransac_confidence"],
        "SIFT_RANSAC_REFINE_ITERS": sift["ransac_refine_iters"],
        "SIFT_RANSAC_MIN_INLIERS": sift["ransac_min_inliers"],
        "SIFT_RANSAC_MIN_INLIER_RATIO": sift["ransac_min_inlier_ratio"],
    }
    for name, expected in sift_constants.items():
        require(getattr(sift_engine, name) == expected, f"SIFT engine {name} drift")
    require(
        list(sift_engine.SIFT_SCALE_RANGE) == sift["scale_range_inclusive"],
        "SIFT scale range drift",
    )
    geometry = config["geometry"]
    require(sift_engine.RIM_MARGIN_PX == geometry["cream_rim_margin_px"], "rim margin")
    require(
        sift_engine.RIM_MAX_DEPTH_FRACTION
        == geometry["cream_rim_max_depth_fraction"],
        "rim depth",
    )
    require(
        sift_engine.CROSS_CLASS_BOX_IOS_GTE
        == geometry["cream_can_cross_class_box_ios_gte"],
        "cross-class threshold",
    )
    templates = registration["templates"]
    require(
        sift_engine.EXPECTED_TEMPLATE_RESULT_SHA256
        == templates["cream_template_metadata"]["sha256"],
        "SIFT template metadata binding",
    )
    require(
        sift_engine.EXPECTED_TEMPLATE_IMAGE_SHA256
        == templates["cream_template_image"]["sha256"],
        "SIFT template image binding",
    )


def _assert_transformers_source_bindings(
    sam3_engine: ModuleType, registration: Mapping[str, Any]
) -> None:
    classes = {
        "transformers_sam3_video_configuration": sam3_engine.Sam3VideoConfig,
        "transformers_sam3_video_modeling": sam3_engine.Sam3VideoModel,
        "transformers_sam3_video_processing": sam3_engine.Sam3VideoProcessor,
        "transformers_sam3_image_processing": sam3_engine.Sam3ImageProcessor,
        "transformers_sam2_video_processing": sam3_engine.Sam2VideoVideoProcessor,
        "transformers_clip_tokenization": sam3_engine.CLIPTokenizer,
    }
    registered = registration["runtime"]["runtime_sources"]
    for role, class_object in classes.items():
        raw_path = inspect.getsourcefile(class_object)
        require(raw_path is not None, f"cannot locate live Transformers source {role}")
        live_path = Path(raw_path).resolve()
        frozen_path = Path(registered[role]["path"])
        require(live_path == frozen_path, f"live Transformers source path drift: {role}")
        require(
            sha256_file(live_path) == registered[role]["sha256"],
            f"live Transformers source SHA drift: {role}",
        )


def _rows_for_episode(panel: RgbPanel, episode_id: int) -> list[dict[str, Any]]:
    return [row for row in panel.rows if int(row["episode_id"]) == episode_id]


def _engine_rows(panel: RgbPanel, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        image = row["images"]["agentview"]
        image_path = _absolute_lexical(panel.root / image["path"])
        sealed_bytes = _read_sealed_regular_bytes(
            image_path,
            str(image["sha256"]),
            f"agentview {row['frame_id']}",
        )
        output.append(
            {
                "episode_id": int(row["episode_id"]),
                "env_seed": int(row["env_seed"]),
                "t": int(row["t"]),
                "frame_id": str(row["frame_id"]),
                "image_path": str(image_path),
                "image_sha256": str(image["sha256"]),
                "sealed_image_bytes": sealed_bytes,
                "formal_same_buffer": True,
                "height": int(image["height"]),
                "width": int(image["width"]),
            }
        )
    return output


def _locate_sealed_eye_frame(
    locator: Any,
    panel: RgbPanel,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    image = row["images"]["eye_in_hand"]
    path = _absolute_lexical(panel.root / image["path"])
    payload = _read_sealed_regular_bytes(
        path,
        str(image["sha256"]),
        f"eye-in-hand {row['frame_id']}",
    )
    return locator.locate(
        None,
        sealed_bytes=payload,
        expected_sha256=str(image["sha256"]),
    )


def _selected_geometry(
    sam3_frame: Mapping[str, Any],
) -> tuple[list[float] | None, list[list[float]]]:
    selection = sam3_frame.get("selection")
    require(isinstance(selection, Mapping), "SAM3 selection missing")
    baskets = selection.get("basket")
    cans = selection.get("can")
    require(isinstance(baskets, list) and len(baskets) <= 1, "SAM3 basket selection")
    require(isinstance(cans, list) and len(cans) <= 2, "SAM3 can selection")
    basket_box = None
    if baskets:
        raw = baskets[0].get("mask_bbox_xyxy")
        require(isinstance(raw, list) and len(raw) == 4, "SAM3 basket box")
        basket_box = [float(value) for value in raw]
    centers = []
    for item in cans:
        raw = item.get("mask_centroid_xy")
        require(isinstance(raw, list) and len(raw) == 2, "SAM3 can centroid")
        centers.append([float(value) for value in raw])
    return basket_box, centers


def _round_floats(value: Any, decimals: int) -> Any:
    if isinstance(value, float):
        return round(value, decimals)
    if isinstance(value, Mapping):
        return {key: _round_floats(child, decimals) for key, child in value.items()}
    if isinstance(value, list):
        return [_round_floats(child, decimals) for child in value]
    return value


def compose_episode_frames(
    *,
    panel: RgbPanel,
    manifest_rows: Sequence[Mapping[str, Any]],
    sam3_frames: Sequence[Mapping[str, Any]],
    sift_rows: Sequence[Mapping[str, Any]],
    derive_evidence_fn: Callable[
        [Sequence[dict[str, Any]], Sequence[dict[str, Any]]],
        tuple[list[CreamFrameEvidence], int],
    ],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    require(
        len(manifest_rows) == len(sam3_frames) == len(sift_rows),
        "episode semantic input lengths differ",
    )
    sam3_materialized = [dict(frame) for frame in sam3_frames]
    sift_materialized = [dict(row) for row in sift_rows]
    evidence_rows, _initial_track_id = derive_evidence_fn(
        sam3_materialized, sift_materialized
    )
    cream_states = run_causal_cream_state(
        evidence_rows, CreamStateConfig.from_dict(config["cream_state"])
    )
    can_machine = CausalCanCount(CanStateConfig.from_dict(config["can_state"]))
    previous_atoms = {"can_ge1": False, "can_ge2": False, "cream": False}
    output = []
    for local_index, (manifest, sam3_frame, sift, cream_evidence, cream_state) in enumerate(
        zip(
            manifest_rows,
            sam3_frames,
            sift_rows,
            evidence_rows,
            cream_states,
            strict=True,
        )
    ):
        require(sam3_frame["frame_id"] == manifest["frame_id"], "SAM3 frame ID")
        require(int(sam3_frame["t"]) == int(manifest["t"]), "SAM3 time")
        require(
            sam3_frame["image_sha256"]
            == manifest["images"]["agentview"]["sha256"],
            "SAM3 image hash",
        )
        basket_box, can_centers = _selected_geometry(sam3_frame)
        can_state = can_machine.step(basket_box, can_centers)
        atoms = {
            "can_ge1": can_state.sticky_count >= 1,
            "can_ge2": can_state.sticky_count >= 2,
            "cream": bool(cream_state.cream_achieved),
        }
        reliability = {
            "can_ge1": can_state.raw_count is not None or atoms["can_ge1"],
            "can_ge2": can_state.raw_count is not None or atoms["can_ge2"],
            "cream": bool(cream_state.observation_reliable),
        }
        reliability["scalar"] = all(reliability.values())
        transitions = {
            name: bool(atoms[name] and not previous_atoms[name]) for name in ATOM_FIELDS
        }
        previous_atoms = dict(atoms)
        cream_used_carry = bool(
            "past_only_achievement_latch" in cream_state.reasons
        )
        role = manifest["source"]["boundary_role"]
        is_recovered = role == "recovered_endpoint"
        record = {
            "schema": FRAME_SCHEMA,
            "panel_id": panel.panel_id,
            "phase": panel.phase,
            "task": TASK,
            "episode_id": int(manifest["episode_id"]),
            "env_seed": int(manifest["env_seed"]),
            "rgb_frame_index": int(manifest["frame_index"]),
            "episode_frame_index": local_index,
            "t": int(manifest["t"]),
            "frame_id": str(manifest["frame_id"]),
            "boundary_role": role,
            "observation_hashes": {
                "agentview_sha256": manifest["images"]["agentview"]["sha256"],
                "eye_in_hand_sha256": manifest["images"]["eye_in_hand"]["sha256"],
            },
            "atoms": atoms,
            "scalar": sum(int(atoms[name]) for name in ("can_ge1", "can_ge2", "cream")),
            "reliability": reliability,
            "transitions": transitions,
            "causal_state": {
                "sticky_can_count": can_state.sticky_count,
                "cream_mode": cream_state.mode,
                "can_used_carry": can_state.used_carry,
                "cream_used_carry": cream_used_carry,
            },
            "evidence": {
                "basket_bbox_xyxy": basket_box,
                "anonymous_can_centers_xy": can_centers,
                "raw_can_inside_count": can_state.raw_count,
                "initial_cream_track_present": cream_evidence.initial_track_present,
                "cream_near_basket_rim": cream_evidence.near_basket_rim,
                "cream_cross_class_collision": cream_evidence.cross_class_collision,
                "sift_valid": bool(sift["valid"]),
                "sift_affine_scale": (
                    float(sift["affine_scale"]) if sift["valid"] else None
                ),
                "sift_invalid_reasons": list(sift.get("invalid_reasons", [])),
                "cream_state_reasons": list(cream_state.reasons),
            },
            "phi_only": is_recovered,
            "dynamics_transition_eligible": not is_recovered,
        }
        validate_progress_frame(record)
        output.append(record)
    return output


def validate_progress_frame(value: Any) -> dict[str, Any]:
    row = dict(require_exact_keys(value, FRAME_FIELDS, "progress frame"))
    require(row["schema"] == FRAME_SCHEMA, "progress frame schema")
    require(row["panel_id"] in set(PANEL_IDS.values()), "progress panel ID")
    require(row["phase"] in {"screen", "confirm"}, "progress phase")
    require(row["task"] == TASK, "progress task")
    for key in (
        "episode_id",
        "env_seed",
        "rgb_frame_index",
        "episode_frame_index",
        "t",
        "scalar",
    ):
        require(type(row[key]) is int and row[key] >= 0, f"progress frame {key}")
    require(
        row["boundary_role"] in {"pre_action", "n_plus_1", "recovered_endpoint"},
        "progress frame role",
    )
    require_exact_keys(row["observation_hashes"], OBSERVATION_HASH_FIELDS, "observation hashes")
    require(
        all(is_sha256(item) for item in row["observation_hashes"].values()),
        "observation hash invalid",
    )
    atoms = require_exact_keys(row["atoms"], ATOM_FIELDS, "progress atoms")
    require(all(type(value) is bool for value in atoms.values()), "progress atoms bool")
    require(not atoms["can_ge2"] or atoms["can_ge1"], "ordinal can invariant")
    require(row["scalar"] == sum(int(value) for value in atoms.values()), "scalar sum")
    reliability = require_exact_keys(
        row["reliability"], RELIABILITY_FIELDS, "progress reliability"
    )
    require(all(type(value) is bool for value in reliability.values()), "reliability bool")
    require(
        reliability["scalar"]
        == all(reliability[name] for name in ("can_ge1", "can_ge2", "cream")),
        "scalar reliability",
    )
    transitions = require_exact_keys(
        row["transitions"], TRANSITION_FIELDS, "progress transitions"
    )
    require(all(type(value) is bool for value in transitions.values()), "transition bool")
    state = require_exact_keys(
        row["causal_state"], CAUSAL_STATE_FIELDS, "progress causal_state"
    )
    require(state["sticky_can_count"] in {0, 1, 2}, "sticky can count")
    require(state["cream_mode"] in {"search", "candidate", "rejected", "achieved"}, "cream mode")
    require(type(state["can_used_carry"]) is bool, "can carry bool")
    require(type(state["cream_used_carry"]) is bool, "cream carry bool")
    evidence = require_exact_keys(row["evidence"], EVIDENCE_FIELDS, "progress evidence")
    require(
        evidence["raw_can_inside_count"] in {None, 0, 1, 2},
        "raw can count",
    )
    require(
        atoms["can_ge1"] is (state["sticky_can_count"] >= 1)
        and atoms["can_ge2"] is (state["sticky_can_count"] >= 2),
        "can atoms do not equal causal sticky count",
    )
    require(
        atoms["cream"] is (state["cream_mode"] == "achieved"),
        "cream atom does not equal causal cream mode",
    )
    raw_can_count = evidence["raw_can_inside_count"]
    require(
        raw_can_count is None or raw_can_count <= state["sticky_can_count"],
        "raw can count exceeds sticky count",
    )
    require(
        state["can_used_carry"] is (
            raw_can_count is None and state["sticky_can_count"] > 0
        ),
        "can carry flag disagrees with evidence/state",
    )
    require(
        reliability["can_ge1"] is (raw_can_count is not None or atoms["can_ge1"])
        and reliability["can_ge2"] is (
            raw_can_count is not None or atoms["can_ge2"]
        ),
        "can reliability disagrees with evidence/state",
    )
    require(
        state["cream_used_carry"]
        is ("past_only_achievement_latch" in evidence["cream_state_reasons"]),
        "cream carry flag disagrees with causal evidence reasons",
    )
    require(isinstance(evidence["anonymous_can_centers_xy"], list), "can centers")
    require(len(evidence["anonymous_can_centers_xy"]) <= 2, "too many can centers")
    require(type(row["phi_only"]) is bool, "phi_only bool")
    require(type(row["dynamics_transition_eligible"]) is bool, "dynamics bool")
    recovered = row["boundary_role"] == "recovered_endpoint"
    require(row["phi_only"] is recovered, "recovered phi-only binding")
    require(row["dynamics_transition_eligible"] is (not recovered), "recovered dynamics binding")
    _reject_public_leakage(row)
    return row


def _episode_prefix_view(frames: Sequence[Mapping[str, Any]], decimals: int) -> list[Any]:
    return [_round_floats(dict(frame), decimals) for frame in frames]


def _rgb_recovery_passed(panel: RgbPanel) -> bool:
    audit = panel.result["terminal_recovery_audit"]
    return bool(
        audit["enabled"] is True
        and audit["episode_id"] == 77
        and audit["stored_chunks_exact_bitwise"] is True
        and audit["two_fresh_replays"] == 2
        and audit["cross_run_boundary_hashes_exact"] is True
        and audit["cross_run_decoded_actions_exact"] is True
        and audit["halted_at_registered_t"] is True
        and audit["phi_only"] is True
        and audit["dynamics_transition_eligible"] is False
        and audit["recovered_dynamics_rows_written"] == 0
    )


def run_semantic_engines(prepared: PreparedInputs) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    registration = prepared.registrations[8000]
    sam3_ref = registration["sources"]["sam3_engine"]
    sift_ref = registration["sources"]["sift_engine"]
    sam3_path = (REPO / sam3_ref["relative_path"]).resolve()
    sift_path = (REPO / sift_ref["relative_path"]).resolve()
    sam3_engine = _load_registered_module(
        sam3_path, sam3_ref["sha256"], "v248_registered_sam3_engine"
    )
    sift_engine = _load_registered_module(
        sift_path, sift_ref["sha256"], "v248_registered_sift_engine"
    )
    _assert_engine_constants(sam3_engine, sift_engine, prepared.config, registration)
    _assert_transformers_source_bindings(sam3_engine, registration)

    import numpy as np
    import torch

    seed = int(prepared.config["sam3"]["global_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
    device = prepared.config["sam3"]["device"]
    require(device != "cuda" or torch.cuda.is_available(), "registered CUDA unavailable")
    dtype = getattr(torch, prepared.config["sam3"]["dtype"])
    checkpoint_ref = registration["models"]["sam3"]["checkpoint"]
    checkpoint = Path(checkpoint_ref["path"])
    tokenizer_root = Path(registration["models"]["sam3"]["clip_tokenizer"]["root"])
    _read_sealed_regular_bytes(
        checkpoint,
        checkpoint_ref["sha256"],
        "SAM3 checkpoint immediately before model load",
        expected_bytes=checkpoint_ref["bytes"],
    )
    model = sam3_engine.load_model(checkpoint, device, dtype)
    processor = sam3_engine.build_processor(tokenizer_root)
    _read_sealed_regular_bytes(
        checkpoint,
        checkpoint_ref["sha256"],
        "SAM3 checkpoint immediately after model load",
        expected_bytes=checkpoint_ref["bytes"],
    )
    model_config_hash = sam3_config_sha256(model.config.to_dict())
    processor_hash = sam3_processor_sha256(
        processor.image_processor.to_dict(),
        processor.video_processor.to_dict(),
        registration["models"]["sam3"]["clip_tokenizer"]["tree_sha256"],
    )
    require(
        model_config_hash == registration["models"]["sam3"]["config_sha256"],
        "live SAM3 config hash drift",
    )
    require(
        processor_hash == registration["models"]["sam3"]["processor_sha256"],
        "live SAM3 processor hash drift",
    )
    template_metadata_ref = registration["templates"]["cream_template_metadata"]
    template_image_ref = registration["templates"]["cream_template_image"]
    template_metadata_bytes = _read_sealed_regular_bytes(
        Path(template_metadata_ref["path"]),
        template_metadata_ref["sha256"],
        "registered cream template metadata",
        expected_bytes=template_metadata_ref["bytes"],
    )
    template_image_bytes = _read_sealed_regular_bytes(
        Path(template_image_ref["path"]),
        template_image_ref["sha256"],
        "registered cream template image",
        expected_bytes=template_image_ref["bytes"],
    )
    locator = sift_engine.EyeCreamSift(
        None,
        sealed_metadata_bytes=template_metadata_bytes,
        sealed_image_bytes=template_image_bytes,
        expected_metadata_sha256=template_metadata_ref["sha256"],
        expected_image_sha256=template_image_ref["sha256"],
    )
    all_frames: list[dict[str, Any]] = []
    prefix_checks: dict[str, Any] = {}
    prefix_count = int(prepared.config["progress"]["prefix_check_frames_per_episode"])
    decimals = int(prepared.config["progress"]["prefix_float_round_decimals"])
    for seed_start in (8000, 8100):
        panel = prepared.panels[seed_start]
        for spec in prepared.registrations[seed_start]["panel"]["episodes"]:
            episode_id = int(spec["episode_id"])
            manifest_rows = _rows_for_episode(panel, episode_id)
            engine_rows = _engine_rows(panel, manifest_rows)
            sam3_frames, _diagnostics = sam3_engine.process_rows(
                model,
                processor,
                engine_rows,
                device=device,
                state_device=prepared.config["sam3"]["state_device"],
                dtype=dtype,
            )
            sift_rows = [
                _locate_sealed_eye_frame(locator, panel, row)
                for row in manifest_rows
            ]
            semantic_frames = compose_episode_frames(
                panel=panel,
                manifest_rows=manifest_rows,
                sam3_frames=sam3_frames,
                sift_rows=sift_rows,
                derive_evidence_fn=sift_engine.derive_evidence,
                config=prepared.config,
            )
            all_frames.extend(semantic_frames)

            count = min(prefix_count, len(manifest_rows))
            repeated_sam3, _ = sam3_engine.process_rows(
                model,
                processor,
                engine_rows[:count],
                device=device,
                state_device=prepared.config["sam3"]["state_device"],
                dtype=dtype,
            )
            repeated_sift = [
                _locate_sealed_eye_frame(locator, panel, row)
                for row in manifest_rows[:count]
            ]
            repeated_semantic = compose_episode_frames(
                panel=panel,
                manifest_rows=manifest_rows[:count],
                sam3_frames=repeated_sam3,
                sift_rows=repeated_sift,
                derive_evidence_fn=sift_engine.derive_evidence,
                config=prepared.config,
            )
            reference_view = _episode_prefix_view(semantic_frames[:count], decimals)
            repeated_view = _episode_prefix_view(repeated_semantic, decimals)
            key = f"{panel.panel_id}:ep{episode_id:02d}"
            prefix_checks[key] = {
                "frames": count,
                "reference_sha256": canonical_sha256(reference_view),
                "fresh_session_sha256": canonical_sha256(repeated_view),
                "exact_after_rounding_6_decimals": reference_view == repeated_view,
            }
    return all_frames, prefix_checks


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_bytes(path, canonical_bytes(dict(value)))


def _registration_result(prepared: PreparedInputs, seed_start: int) -> dict[str, Any]:
    registration = prepared.registrations[seed_start]
    return {
        "registration_id": registration["registration_id"],
        "file_sha256": prepared.registration_hashes[seed_start],
        "canonical_self_sha256": registration["self_sha256"],
    }


def _engine_result(prepared: PreparedInputs) -> dict[str, Any]:
    registration = prepared.registrations[8000]
    return {
        "sam3_engine_sha256": registration["sources"]["sam3_engine"]["sha256"],
        "sift_engine_sha256": registration["sources"]["sift_engine"]["sha256"],
        "can_state_sha256": registration["sources"]["can_state"]["sha256"],
        "cream_state_sha256": registration["sources"]["cream_state"]["sha256"],
        "sam3_checkpoint_sha256": registration["models"]["sam3"]["checkpoint"]["sha256"],
        "clip_tokenizer_tree_sha256": registration["models"]["sam3"]
        ["clip_tokenizer"]["tree_sha256"],
        "sam3_config_sha256": registration["models"]["sam3"]["config_sha256"],
        "sam3_processor_sha256": registration["models"]["sam3"]["processor_sha256"],
        "cream_template_image_sha256": registration["templates"]
        ["cream_template_image"]["sha256"],
        "cream_template_metadata_sha256": registration["templates"]
        ["cream_template_metadata"]["sha256"],
        "semantic_config_sha256": prepared.config_sha256,
        "transformers_source_sha256s": {
            role: registration["runtime"]["runtime_sources"][role]["sha256"]
            for role in (
                "transformers_sam3_video_configuration",
                "transformers_sam3_video_modeling",
                "transformers_sam3_video_processing",
                "transformers_sam3_image_processing",
                "transformers_sam2_video_processing",
                "transformers_clip_tokenization",
            )
        },
    }


def _validate_frame_against_rgb_manifest(
    frame_value: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    local_index: int,
    previous_atoms: Mapping[str, bool],
) -> dict[str, Any]:
    frame = validate_progress_frame(frame_value)
    require(frame["frame_id"] == manifest["frame_id"], "progress/RGB ID")
    require(frame["panel_id"] == manifest["panel_id"], "progress/RGB panel")
    require(frame["phase"] == manifest["phase"], "progress/RGB phase")
    require(frame["task"] == manifest["task"], "progress/RGB task")
    require(frame["episode_id"] == manifest["episode_id"], "progress/RGB episode")
    require(frame["env_seed"] == manifest["env_seed"], "progress/RGB seed")
    require(frame["rgb_frame_index"] == manifest["frame_index"], "RGB index")
    require(frame["episode_frame_index"] == local_index, "episode index")
    require(frame["t"] == manifest["t"], "progress/RGB t")
    require(
        frame["boundary_role"] == manifest["source"]["boundary_role"],
        "progress/RGB boundary role",
    )
    require(
        frame["observation_hashes"]
        == {
            "agentview_sha256": manifest["images"]["agentview"]["sha256"],
            "eye_in_hand_sha256": manifest["images"]["eye_in_hand"]["sha256"],
        },
        "progress observation hashes are not the sealed RGB pair",
    )
    expected_transitions = {
        name: bool(frame["atoms"][name] and not previous_atoms[name])
        for name in ATOM_FIELDS
    }
    require(
        frame["transitions"] == expected_transitions,
        "progress transition does not equal causal state delta",
    )
    require(
        frame["causal_state"]["cream_used_carry"]
        is bool(frame["atoms"]["cream"] and previous_atoms["cream"]),
        "cream carry flag does not equal the past-only achieved latch",
    )
    require(
        all(not previous_atoms[name] or frame["atoms"][name] for name in ATOM_FIELDS),
        "progress atom latch regressed",
    )
    return frame


def _validate_frame_sequence(
    frames: Sequence[Mapping[str, Any]], prepared: PreparedInputs
) -> None:
    expected_total = sum(len(panel.rows) for panel in prepared.panels.values())
    require(len(frames) == expected_total, "progress/RGB frame count mismatch")
    position = 0
    seen_ids = set()
    for seed_start in (8000, 8100):
        panel = prepared.panels[seed_start]
        for spec in prepared.registrations[seed_start]["panel"]["episodes"]:
            manifest_rows = _rows_for_episode(panel, int(spec["episode_id"]))
            previous_atoms = {name: False for name in ATOM_FIELDS}
            for local_index, manifest in enumerate(manifest_rows):
                frame = _validate_frame_against_rgb_manifest(
                    frames[position],
                    manifest,
                    local_index=local_index,
                    previous_atoms=previous_atoms,
                )
                previous_atoms = dict(frame["atoms"])
                require(frame["frame_id"] not in seen_ids, "duplicate progress frame")
                seen_ids.add(frame["frame_id"])
                position += 1
    require(position == len(frames), "unconsumed progress frames")


def validate_progress_stage(stage: Path, prepared: PreparedInputs) -> None:
    result_path = stage / RESULT_FILENAME
    frames_path = stage / FRAMES_FILENAME
    complete_path = stage / COMPLETE_FILENAME
    producer_path = stage / PRODUCER_FILENAME
    config_path = stage / CONFIG_FILENAME
    for path in (result_path, frames_path, complete_path, producer_path, config_path):
        require(path.is_file() and not path.is_symlink(), f"missing snapshot {path.name}")
    require(
        {path.name for path in stage.iterdir()}
        == {
            RESULT_FILENAME,
            FRAMES_FILENAME,
            COMPLETE_FILENAME,
            PRODUCER_FILENAME,
            CONFIG_FILENAME,
        },
        "progress artifact root closure mismatch",
    )
    result = _strict_json_file(result_path, "progress RESULT")
    complete = _strict_json_file(complete_path, "progress COMPLETE")
    frames = _strict_jsonl(frames_path, "progress frames")
    require_exact_keys(result, RESULT_FIELDS, "progress RESULT")
    require(result["schema"] == RESULT_SCHEMA, "progress result schema")
    require(result["status"] == "observation_only_sealed", "progress result status")
    require(result["claim_scope"] == "visual_progress_producer_only", "claim scope")
    require(result["phase"] == prepared.registrations[8000]["phase"], "result phase")
    require(result["task"] == TASK, "result task")
    require(
        validate_authority(
            result["authority"],
            prepared.registrations,
            result["phase"],
            expected=prepared.authority,
        )
        == prepared.authority,
        "result authority binding",
    )
    require_exact_keys(
        result["registrations"], frozenset(PANEL_IDS.values()), "result registrations"
    )
    require_exact_keys(
        result["rgb_inputs"], frozenset(PANEL_IDS.values()), "result rgb_inputs"
    )
    for seed_start, panel_id in PANEL_IDS.items():
        require_exact_keys(
            result["registrations"][panel_id],
            REGISTRATION_RESULT_FIELDS,
            f"result registrations.{panel_id}",
        )
        require(
            result["registrations"][panel_id]
            == _registration_result(prepared, seed_start),
            "result registration binding",
        )
        require_exact_keys(
            result["rgb_inputs"][panel_id],
            RGB_INPUT_FIELDS,
            f"result rgb_inputs.{panel_id}",
        )
        require(
            result["rgb_inputs"][panel_id] == prepared.panels[seed_start].seals,
            "RGB seal binding",
        )
    require_exact_keys(result["engines"], ENGINE_FIELDS, "result engines")
    require(result["engines"] == _engine_result(prepared), "engine binding")
    expected_episode_ids = {
        PANEL_IDS[seed]: [
            int(item["episode_id"])
            for item in prepared.registrations[seed]["panel"]["episodes"]
        ]
        for seed in (8000, 8100)
    }
    require(result["episode_ids"] == expected_episode_ids, "result episode IDs")
    require(
        result["observation_contract"]
        == {
            "decoded_cameras": ["agentview", "eye_in_hand"],
            "semantic_inputs": ["current_agentview_rgb", "current_eye_in_hand_rgb"],
            "causal_past_state_only": True,
        },
        "result observation contract",
    )
    expected_prefix_keys = {
        f"{PANEL_IDS[seed]}:ep{episode_id:02d}"
        for seed in (8000, 8100)
        for episode_id in expected_episode_ids[PANEL_IDS[seed]]
    }
    require(set(result["prefix_checks"]) == expected_prefix_keys, "prefix episode set")
    for key, check in result["prefix_checks"].items():
        require_exact_keys(check, PREFIX_CHECK_FIELDS, f"prefix {key}")
        require(check["frames"] == 8, f"prefix {key} frame count")
        require(is_sha256(check["reference_sha256"]), f"prefix {key} reference hash")
        require(is_sha256(check["fresh_session_sha256"]), f"prefix {key} fresh hash")
        require(type(check["exact_after_rounding_6_decimals"]) is bool, "prefix bool")
    require_exact_keys(result["recovered_endpoint"], RECOVERED_RESULT_FIELDS, "recovered result")
    require(
        result["recovered_endpoint"]
        == {
            "required": True,
            "present": True,
            "panel_id": "chain3_8100",
            "episode_id": 77,
            "t": 651,
            "phi_only": True,
            "dynamics_transition_eligible": False,
            "rgb_recovery_prerequisites_passed": True,
        },
        "recovered endpoint result",
    )
    require_exact_keys(result["output"], OUTPUT_RESULT_FIELDS, "progress output")
    require(result["output"]["frames"] == len(frames), "result frame count")
    require(result["output"]["frames_sha256"] == sha256_file(frames_path), "frames seal")
    require(result["output"]["producer_sha256"] == sha256_file(producer_path), "producer seal")
    require(
        sha256_file(producer_path)
        == prepared.registrations[8000]["sources"]["semantic_producer"]["sha256"],
        "progress producer snapshot is not the registered source",
    )
    require(
        result["output"]["semantic_config_sha256"] == sha256_file(config_path),
        "config seal",
    )
    require(
        result["output"]["semantic_config_sha256"]
        == prepared.config_sha256
        == result["engines"]["semantic_config_sha256"]
        == prepared.registrations[8000]["templates"]["semantic_config"]["sha256"],
        "config is not bound to the captured registered bytes",
    )
    require(
        config_path.read_bytes() == prepared.config_bytes,
        "published semantic config differs from captured bytes",
    )
    require_exact_keys(result["safety"], SAFETY_RESULT_FIELDS, "progress safety")
    require(
        result["safety"]
        == {
            "oracle_opened_or_hashed": False,
            "privileged_inputs_read": False,
            "simulator_imported_or_run": False,
            "model_input_fields": ["agentview_rgb", "eye_in_hand_rgb"],
            "forbidden_public_field_scan_passed": True,
            "source_images_mutated": False,
            "dynamics_rows_written": 0,
        },
        "progress safety declaration",
    )
    _reject_public_leakage(result)
    _validate_frame_sequence(frames, prepared)
    recovered = [
        frame
        for frame in frames
        if frame["panel_id"] == "chain3_8100"
        and frame["episode_id"] == 77
        and frame["boundary_role"] == "recovered_endpoint"
    ]
    require(len(recovered) == 1 and recovered[0]["t"] == 651, "missing ep77 t651")
    require_exact_keys(complete, COMPLETE_FIELDS, "progress COMPLETE")
    require(complete["schema"] == COMPLETE_SCHEMA, "progress COMPLETE schema")
    require(complete["status"] == "atomic_success", "progress COMPLETE status")
    require(complete["atomic_commit"] is True, "progress atomic flag")
    require(complete["phase"] == result["phase"], "COMPLETE phase")
    require(
        validate_authority(
            complete["authority"],
            prepared.registrations,
            result["phase"],
            expected=prepared.authority,
        )
        == result["authority"],
        "RESULT/COMPLETE authority drift",
    )
    require(
        complete["registration_sha256s"]
        == {PANEL_IDS[key]: prepared.registration_hashes[key] for key in (8000, 8100)},
        "COMPLETE registrations",
    )
    require(complete["result_sha256"] == sha256_file(result_path), "result seal")
    require(complete["frames_sha256"] == sha256_file(frames_path), "frames seal")
    require(complete["producer_sha256"] == sha256_file(producer_path), "producer seal")
    require(complete["semantic_config_sha256"] == sha256_file(config_path), "config seal")


def publish_progress(
    prepared: PreparedInputs,
    frames: Sequence[Mapping[str, Any]],
    prefix_checks: Mapping[str, Any],
) -> None:
    _validate_frame_sequence(frames, prepared)
    require(_rgb_recovery_passed(prepared.panels[8100]), "RGB recovery prerequisite")
    support, destination, target_root = revalidate_static_inputs(prepared)
    source_path = Path(__file__).resolve()
    source_bytes = _read_sealed_regular_bytes(
        source_path,
        prepared.registrations[8000]["sources"]["semantic_producer"]["sha256"],
        "registered semantic producer source",
    )
    phase = prepared.registrations[8000]["phase"]

    def build(stage: Path) -> None:
        _write_bytes(stage / PRODUCER_FILENAME, source_bytes)
        _write_bytes(stage / CONFIG_FILENAME, prepared.config_bytes)
        _write_bytes(
            stage / FRAMES_FILENAME,
            b"".join(canonical_bytes(dict(frame)) for frame in frames),
        )
        episode_ids = {
            PANEL_IDS[seed]: [
                int(item["episode_id"])
                for item in prepared.registrations[seed]["panel"]["episodes"]
            ]
            for seed in (8000, 8100)
        }
        result = {
            "schema": RESULT_SCHEMA,
            "status": "observation_only_sealed",
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "claim_scope": "visual_progress_producer_only",
            "phase": phase,
            "task": TASK,
            "authority": dict(prepared.authority),
            "registrations": {
                PANEL_IDS[seed]: _registration_result(prepared, seed)
                for seed in (8000, 8100)
            },
            "rgb_inputs": {
                PANEL_IDS[seed]: prepared.panels[seed].seals for seed in (8000, 8100)
            },
            "episode_ids": episode_ids,
            "observation_contract": {
                "decoded_cameras": ["agentview", "eye_in_hand"],
                "semantic_inputs": ["current_agentview_rgb", "current_eye_in_hand_rgb"],
                "causal_past_state_only": True,
            },
            "engines": _engine_result(prepared),
            "prefix_checks": dict(prefix_checks),
            "recovered_endpoint": {
                "required": True,
                "present": True,
                "panel_id": "chain3_8100",
                "episode_id": 77,
                "t": 651,
                "phi_only": True,
                "dynamics_transition_eligible": False,
                "rgb_recovery_prerequisites_passed": True,
            },
            "output": {
                "frames": len(frames),
                "frames_sha256": sha256_file(stage / FRAMES_FILENAME),
                "producer_sha256": sha256_file(stage / PRODUCER_FILENAME),
                "semantic_config_sha256": sha256_file(stage / CONFIG_FILENAME),
            },
            "safety": {
                "oracle_opened_or_hashed": False,
                "privileged_inputs_read": False,
                "simulator_imported_or_run": False,
                "model_input_fields": ["agentview_rgb", "eye_in_hand_rgb"],
                "forbidden_public_field_scan_passed": True,
                "source_images_mutated": False,
                "dynamics_rows_written": 0,
            },
        }
        _reject_public_leakage(result)
        _write_json(stage / RESULT_FILENAME, result)
        complete = {
            "schema": COMPLETE_SCHEMA,
            "status": "atomic_success",
            "atomic_commit": True,
            "phase": phase,
            "authority": dict(prepared.authority),
            "registration_sha256s": {
                PANEL_IDS[key]: prepared.registration_hashes[key]
                for key in (8000, 8100)
            },
            "result_sha256": sha256_file(stage / RESULT_FILENAME),
            "frames_sha256": sha256_file(stage / FRAMES_FILENAME),
            "producer_sha256": sha256_file(stage / PRODUCER_FILENAME),
            "semantic_config_sha256": sha256_file(stage / CONFIG_FILENAME),
        }
        _write_json(stage / COMPLETE_FILENAME, complete)

    support.atomic_publish_directory(
        destination,
        build,
        lambda stage: validate_progress_stage(stage, prepared),
        trusted_target_root=target_root,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-bundle", type=Path, required=True)
    parser.add_argument("--expected-freeze-complete-sha256", required=True)
    parser.add_argument("--rgb-run", action="append", type=Path, required=True)
    parser.add_argument(
        "--expected-rgb-complete-sha256",
        action="append",
        required=True,
        help="External SHA-256 anchor paired with each --rgb-run",
    )
    parser.add_argument(
        "--expected-confirm-authorization-complete-sha256",
        help=(
            "Externally recorded confirm authorization COMPLETE SHA; required "
            "for confirm RGB and forbidden for screen RGB"
        ),
    )
    args = parser.parse_args(argv)
    if len(args.rgb_run) != 2:
        parser.error("exactly two --rgb-run arguments are required")
    if len(args.expected_rgb_complete_sha256) != 2:
        parser.error("exactly two external RGB COMPLETE anchors are required")
    if (
        args.expected_confirm_authorization_complete_sha256 is not None
        and not is_sha256(args.expected_confirm_authorization_complete_sha256)
    ):
        parser.error("confirm authorization anchor must be 64 lowercase hex")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    prepared = prepare_inputs(
        freeze_bundle_root=args.freeze_bundle,
        expected_freeze_complete_sha256=args.expected_freeze_complete_sha256,
        rgb_roots=args.rgb_run,
        expected_rgb_complete_sha256s=args.expected_rgb_complete_sha256,
        expected_confirm_authorization_complete_sha256=(
            args.expected_confirm_authorization_complete_sha256
        ),
    )
    frames, prefix_checks = run_semantic_engines(prepared)
    publish_progress(prepared, frames, prefix_checks)
    print(
        json.dumps(
            {
                "status": "SEALED",
                "claim_scope": "visual_progress_producer_only",
                "phase": prepared.registrations[8000]["phase"],
                "frames": len(frames),
                "output_sha256": sha256_file(
                    prepared.output_dir / COMPLETE_FILENAME
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Gate0ContractError, OSError, ValueError) as error:
        print(f"STOP: {error}", file=sys.stderr)
        raise SystemExit(2) from error
