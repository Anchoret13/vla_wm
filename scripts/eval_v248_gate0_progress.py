#!/usr/bin/env python3
"""Evaluate the sealed v248 observation-only progress tape after its seal.

This is only the visual subgate of Gate-0.  It cannot certify the world model,
Phi training, or policy improvement.  Screen runs are unconditionally
non-promotional even when every visual metric passes.

The ``seal`` command validates the complete progress artifact without touching
either oracle file, atomically publishes a pre-oracle directory, and exits.
A separate ``evaluate`` invocation must receive externally recorded hashes of
both that seal and its COMPLETE marker before any oracle is touched.  There is
intentionally no summary-path, registration-path, or output-path CLI argument;
ordered publication itself still belongs to the external WORM/signed ledger.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import produce_v248_gate0_progress as producer  # noqa: E402
from lcwm.v248_gate0_contract import (  # noqa: E402
    Gate0ContractError,
    PANEL_IDS,
    TASK,
    canonical_bytes,
    is_sha256,
    load_and_validate_registration,
    loads_strict_json,
    require,
    require_exact_keys,
    sha256_file,
)


EVALUATION_SCHEMA = "v248_gate0_progress_evaluation_v1"
PRE_ORACLE_SCHEMA = "v248_gate0_progress_pre_oracle_seal_v1"
PRE_ORACLE_COMPLETE_SCHEMA = "v248_gate0_progress_pre_oracle_complete_v1"
VISUAL_SUBGATE_SCHEMA = "v248_gate0_visual_subgate_v1"
MARKER_SCHEMA = "v248_gate0_visual_subgate_marker_v1"
COMPLETE_SCHEMA = "v248_gate0_progress_evaluation_complete_v1"

RESULT_FILENAME = "RESULT.json"
SUBGATE_FILENAME = "VISUAL_SUBGATE.json"
PRE_ORACLE_FILENAME = "PRE_ORACLE_SEAL.json"
EVALUATOR_FILENAME = "EVALUATOR.py"
COMPLETE_FILENAME = "COMPLETE.json"
PRE_ORACLE_COMPLETE_FIELDS = frozenset(
    {
        "schema",
        "status",
        "atomic_commit",
        "pre_oracle_seal_sha256",
        "evaluator_sha256",
        "authority",
    }
)

PRE_ORACLE_FIELDS = frozenset(
    {
        "schema",
        "status",
        "created_utc",
        "oracle_opened_or_hashed",
        "phase",
        "registration_sha256s",
        "progress_artifact",
        "evaluator_sha256",
        "authority",
    }
)
PROGRESS_SEAL_FIELDS = frozenset(
    {
        "complete_sha256",
        "result_sha256",
        "frames_sha256",
        "producer_sha256",
        "semantic_config_sha256",
        "authority",
    }
)
EVALUATION_FIELDS = frozenset(
    {
        "schema",
        "status",
        "created_utc",
        "claim_scope",
        "phase",
        "decision",
        "authority",
        "pre_oracle_seal_sha256",
        "pre_oracle_complete_sha256",
        "registrations",
        "progress_artifact",
        "oracle_inputs",
        "metrics",
        "visual_subgate",
        "provenance",
    }
)
ORACLE_INPUT_FIELDS = frozenset(
    {"file_sha256", "bytes", "fields_used", "success_values_accessed"}
)
PROVENANCE_FIELDS = frozenset(
    {
        "evaluator_sha256",
        "registered_evaluator_sha256",
        "producer_sha256",
        "semantic_config_sha256",
    }
)
SUBGATE_FIELDS = frozenset(
    {
        "schema",
        "claim_scope",
        "phase",
        "decision",
        "promotion_authorized",
        "screen_metrics_passed",
        "metric_checks_passed",
        "thresholds",
        "values",
        "checks",
    }
)
MARKER_FIELDS = frozenset(
    {"schema", "decision", "claim_scope", "result_sha256", "subgate_sha256"}
)
COMPLETE_FIELDS = frozenset(
    {
        "schema",
        "status",
        "atomic_commit",
        "decision",
        "authority",
        "result_sha256",
        "subgate_sha256",
        "marker_filename",
        "marker_sha256",
        "pre_oracle_seal_sha256",
        "pre_oracle_complete_sha256",
        "evaluator_sha256",
    }
)

ATOM_NAMES = ("can_ge1", "can_ge2", "cream")
CAN_EVENT_INDICES = (1, 3)
CREAM_EVENT_INDEX = 5
HELD_CREAM_EVENT_INDEX = 4


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


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
    require(bool(rows), f"{where} empty")
    return rows


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_bytes(path, canonical_bytes(dict(value)))


def campaign_node_authority(
    bundle: Mapping[str, Any],
    phase: str,
    node: str,
    confirm_authorization_complete_sha256: str | None,
) -> dict[str, Any]:
    require(phase in {"screen", "confirm"}, "campaign authority phase")
    require(node in {"pre_oracle", "evaluation"}, "campaign authority node")
    slots = [f"{PANEL_IDS[seed]}_{phase}.json" for seed in (8000, 8100)]
    output_key = f"{phase}_{node}_complete"
    output_slot = bundle["target_output_slots"][output_key]
    authority = {
        "freeze_complete_sha256": bundle["freeze_complete_sha256"],
        "campaign_manifest_sha256": bundle["campaign_manifest_sha256"],
        "campaign_body_sha256": bundle["campaign"]["body_sha256"],
        "campaign_projection_sha256": bundle["campaign"]["projection_sha256"],
        "campaign_id": bundle["campaign"]["campaign_id"],
        "registration_slots": slots,
        "output_complete_slot": output_slot,
        "confirm_authorization_complete_sha256": (
            confirm_authorization_complete_sha256
        ),
    }
    require_exact_keys(authority, producer.AUTHORITY_FIELDS, "campaign node authority")
    return authority


def validate_combined_authority(
    value: Any,
    registrations: Mapping[int, Mapping[str, Any]],
    phase: str,
    *,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    authority = dict(
        require_exact_keys(value, producer.AUTHORITY_FIELDS, "combined authority")
    )
    for key in (
        "freeze_complete_sha256",
        "campaign_manifest_sha256",
        "campaign_body_sha256",
        "campaign_projection_sha256",
    ):
        require(is_sha256(authority[key]), f"combined authority {key} invalid")
    expected_slots = [
        f"{PANEL_IDS[seed]}_{phase}.json" for seed in (8000, 8100)
    ]
    require(authority["registration_slots"] == expected_slots, "authority slots drift")
    require(
        all(
            registrations[seed]["campaign"]["campaign_id"]
            == authority["campaign_id"]
            and registrations[seed]["campaign"]["projection_sha256"]
            == authority["campaign_projection_sha256"]
            for seed in (8000, 8100)
        ),
        "combined authority campaign drift",
    )
    if phase == "screen":
        require(
            authority["confirm_authorization_complete_sha256"] is None,
            "screen authority consumed confirm authorization",
        )
    else:
        require(
            is_sha256(authority["confirm_authorization_complete_sha256"]),
            "confirm authority lacks authorization anchor",
        )
    if expected is not None:
        require(authority == dict(expected), "combined authority differs from campaign")
    return authority


def load_registration_pair_without_oracle(
    paths: Sequence[Path], expected_hashes: Sequence[str]
) -> tuple[dict[int, dict[str, Any]], dict[int, str]]:
    return producer._load_registration_pair(
        paths, expected_hashes, role="semantic_producer"
    )


def _phase_from_frozen_progress_slot(
    bundle: Mapping[str, Any], progress_root: Path
) -> tuple[str, Path]:
    """Identify a progress phase by frozen lexical path, without target I/O."""
    supplied = producer._absolute_lexical(progress_root)
    matches = []
    for phase in ("screen", "confirm"):
        expected = producer._absolute_lexical(
            (
                Path(bundle["target_root"])
                / bundle["target_output_slots"][f"{phase}_progress_complete"]
            ).parent
        )
        if supplied == expected:
            matches.append((phase, expected))
    require(
        len(matches) == 1,
        "progress path does not equal one frozen screen/confirm slot",
    )
    return matches[0]


def load_anchored_campaign_progress(
    *,
    freeze_bundle_root: Path,
    expected_freeze_complete_sha256: str,
    progress_root: Path,
    expected_progress_complete_sha256: str,
    expected_confirm_authorization_complete_sha256: str | None = None,
) -> tuple[
    dict[str, Any],
    dict[int, dict[str, Any]],
    dict[int, str],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    bundle = producer.load_and_validate_campaign_bundle(
        freeze_bundle_root,
        expected_freeze_complete_sha256,
        expected_source_role="semantic_producer",
    )
    phase, expected_progress_root = _phase_from_frozen_progress_slot(
        bundle, progress_root
    )
    if phase == "screen":
        require(
            expected_confirm_authorization_complete_sha256 is None,
            "screen progress must not consume confirm authorization",
        )
    else:
        require(
            is_sha256(expected_confirm_authorization_complete_sha256),
            "confirm progress requires an external confirm-authorization anchor",
        )
        replay = producer._load_rgb_support(
            bundle["registrations"]["chain3_8000_confirm.json"]["registration"]
        )
        assert expected_confirm_authorization_complete_sha256 is not None
        producer._validate_confirm_authorization(
            bundle,
            expected_confirm_authorization_complete_sha256,
            replay,
        )

    # For confirm, all A->F/P/S/Q/E validation above completes before this first
    # stat/open of the progress target.
    progress_complete_path = expected_progress_root / producer.COMPLETE_FILENAME
    complete = producer._strict_json_bytes(
        producer._read_sealed_regular_bytes(
            progress_complete_path,
            expected_progress_complete_sha256,
            "progress COMPLETE phase locator",
        ),
        "progress COMPLETE phase locator",
    )
    require(complete.get("phase") == phase, "progress phase differs from frozen slot")
    slots = [f"{PANEL_IDS[seed]}_{phase}.json" for seed in (8000, 8100)]
    registrations = {
        seed: bundle["registrations"][slots[index]]["registration"]
        for index, seed in enumerate((8000, 8100))
    }
    registration_hashes = {
        seed: bundle["registrations"][slots[index]]["file_sha256"]
        for index, seed in enumerate((8000, 8100))
    }
    locator_authority = complete.get("authority")
    require(isinstance(locator_authority, Mapping), "progress authority locator missing")
    confirm_authorization_sha = expected_confirm_authorization_complete_sha256
    require(
        locator_authority.get("confirm_authorization_complete_sha256")
        == confirm_authorization_sha,
        "progress authorization differs from the external anchor",
    )
    expected_authority = {
        "freeze_complete_sha256": bundle["freeze_complete_sha256"],
        "campaign_manifest_sha256": bundle["campaign_manifest_sha256"],
        "campaign_body_sha256": bundle["campaign"]["body_sha256"],
        "campaign_projection_sha256": bundle["campaign"]["projection_sha256"],
        "campaign_id": bundle["campaign"]["campaign_id"],
        "registration_slots": slots,
        "output_complete_slot": bundle["target_output_slots"][
            f"{phase}_progress_complete"
        ],
        "confirm_authorization_complete_sha256": confirm_authorization_sha,
    }
    require(
        producer._absolute_lexical(progress_root) == expected_progress_root,
        "progress artifact is not in its frozen campaign slot",
    )
    result, frames, config, seals = validate_progress_artifact_without_oracle(
        progress_root,
        registrations,
        registration_hashes,
        expected_progress_complete_sha256,
        expected_authority=expected_authority,
    )
    return (
        bundle,
        registrations,
        registration_hashes,
        result,
        frames,
        config,
        seals,
    )


def revalidate_anchored_campaign_progress(
    *,
    freeze_bundle_root: Path,
    expected_freeze_complete_sha256: str,
    progress_root: Path,
    expected_progress_complete_sha256: str,
    expected_bundle: Mapping[str, Any],
    expected_result: Mapping[str, Any],
    expected_seals: Mapping[str, Any],
    expected_confirm_authorization_complete_sha256: str | None = None,
) -> None:
    current = load_anchored_campaign_progress(
        freeze_bundle_root=freeze_bundle_root,
        expected_freeze_complete_sha256=expected_freeze_complete_sha256,
        progress_root=progress_root,
        expected_progress_complete_sha256=expected_progress_complete_sha256,
        expected_confirm_authorization_complete_sha256=(
            expected_confirm_authorization_complete_sha256
        ),
    )
    bundle, _registrations, _hashes, result, _frames, _config, seals = current
    require(
        bundle["freeze_complete_sha256"]
        == expected_bundle["freeze_complete_sha256"]
        and bundle["campaign_manifest_sha256"]
        == expected_bundle["campaign_manifest_sha256"],
        "campaign changed during evaluation",
    )
    require(result == dict(expected_result), "progress RESULT changed during evaluation")
    require(seals == dict(expected_seals), "progress seals changed during evaluation")


def _expected_frames_for_registration(registration: Mapping[str, Any]) -> int:
    return sum(
        len(producer._expected_episode_schedule(spec)[0])
        for spec in registration["panel"]["episodes"]
    )


def _validate_semantic_sequence(
    frames: Sequence[Mapping[str, Any]], registrations: Mapping[int, Mapping[str, Any]]
) -> None:
    position = 0
    seen_ids: set[str] = set()
    for seed_start in (8000, 8100):
        registration = registrations[seed_start]
        panel_id = PANEL_IDS[seed_start]
        rgb_index = 0
        for spec in registration["panel"]["episodes"]:
            episode_id = int(spec["episode_id"])
            times, roles = producer._expected_episode_schedule(spec)
            previous = {name: False for name in ATOM_NAMES}
            for local_index, (expected_t, expected_role) in enumerate(
                zip(times, roles, strict=True)
            ):
                require(position < len(frames), "progress frame sequence truncated")
                frame = producer.validate_progress_frame(frames[position])
                require(frame["panel_id"] == panel_id, "progress panel sequence")
                require(frame["phase"] == registration["phase"], "progress phase")
                require(frame["episode_id"] == episode_id, "progress episode sequence")
                require(frame["env_seed"] == seed_start + episode_id, "progress seed")
                require(frame["rgb_frame_index"] == rgb_index, "progress RGB index")
                require(frame["episode_frame_index"] == local_index, "progress local index")
                require(frame["t"] == expected_t, "progress time schedule")
                require(frame["boundary_role"] == expected_role, "progress role schedule")
                expected_id = (
                    f"{panel_id}:{registration['phase']}:"
                    f"ep{episode_id:02d}:t{expected_t:04d}"
                )
                require(frame["frame_id"] == expected_id, "progress frame ID")
                require(frame["frame_id"] not in seen_ids, "duplicate progress frame ID")
                seen_ids.add(frame["frame_id"])
                current = frame["atoms"]
                require(
                    all(not previous[name] or current[name] for name in ATOM_NAMES),
                    "progress atoms are not sticky",
                )
                expected_transitions = {
                    name: bool(current[name] and not previous[name])
                    for name in ATOM_NAMES
                }
                require(
                    frame["transitions"] == expected_transitions,
                    "progress transition/state mismatch",
                )
                require(
                    frame["causal_state"]["cream_used_carry"]
                    is bool(current["cream"] and previous["cream"]),
                    "progress cream carry/state mismatch",
                )
                previous = dict(current)
                position += 1
                rgb_index += 1
    require(position == len(frames), "extra progress frames")


def validate_progress_artifact_without_oracle(
    root: Path,
    registrations: Mapping[int, Mapping[str, Any]],
    registration_hashes: Mapping[int, str],
    expected_complete_sha256: str,
    expected_authority: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    lexical_root = producer._absolute_lexical(root)
    require(not lexical_root.is_symlink(), "progress run root is a symlink")
    root = lexical_root.resolve()
    require(root.is_dir(), f"missing progress run: {root}")
    result_path = root / producer.RESULT_FILENAME
    frames_path = root / producer.FRAMES_FILENAME
    complete_path = root / producer.COMPLETE_FILENAME
    producer_path = root / producer.PRODUCER_FILENAME
    config_path = root / producer.CONFIG_FILENAME
    for path in (result_path, frames_path, complete_path, producer_path, config_path):
        require(path.is_file() and not path.is_symlink(), f"missing progress {path.name}")
    require(is_sha256(expected_complete_sha256), "invalid external progress COMPLETE SHA")
    complete_bytes = producer._read_sealed_regular_bytes(
        complete_path,
        expected_complete_sha256,
        "externally anchored progress COMPLETE",
    )
    complete = producer._strict_json_bytes(complete_bytes, "progress COMPLETE")
    require_exact_keys(complete, producer.COMPLETE_FIELDS, "progress COMPLETE")
    require(
        {path.name for path in root.iterdir()}
        == {
            producer.RESULT_FILENAME,
            producer.FRAMES_FILENAME,
            producer.COMPLETE_FILENAME,
            producer.PRODUCER_FILENAME,
            producer.CONFIG_FILENAME,
        },
        "progress artifact root closure mismatch",
    )
    result_bytes = producer._read_sealed_regular_bytes(
        result_path, complete["result_sha256"], "sealed progress RESULT"
    )
    frames_bytes = producer._read_sealed_regular_bytes(
        frames_path, complete["frames_sha256"], "sealed progress frames"
    )
    producer_bytes = producer._read_sealed_regular_bytes(
        producer_path, complete["producer_sha256"], "sealed progress producer"
    )
    config_bytes = producer._read_sealed_regular_bytes(
        config_path,
        complete["semantic_config_sha256"],
        "sealed progress semantic config",
    )
    result = producer._strict_json_bytes(result_bytes, "progress RESULT")
    frames = producer._strict_jsonl_bytes(frames_bytes, "progress frames")
    config = producer.validate_semantic_config(
        producer._strict_json_bytes(config_bytes, "progress semantic config")
    )
    config_sha = hashlib.sha256(config_bytes).hexdigest()

    require_exact_keys(result, producer.RESULT_FIELDS, "progress RESULT")
    require(result["schema"] == producer.RESULT_SCHEMA, "progress result schema")
    require(result["status"] == "observation_only_sealed", "progress status")
    require(result["claim_scope"] == "visual_progress_producer_only", "progress scope")
    phase = registrations[8000]["phase"]
    require(result["phase"] == registrations[8100]["phase"] == phase, "progress phase")
    require(result["task"] == TASK, "progress task")
    require_exact_keys(
        result["authority"], producer.AUTHORITY_FIELDS, "progress authority"
    )
    require(
        all(
            is_sha256(result["authority"][key])
            for key in (
                "freeze_complete_sha256",
                "campaign_manifest_sha256",
                "campaign_body_sha256",
                "campaign_projection_sha256",
            )
        ),
        "progress authority hashes",
    )
    validated_progress_authority = producer.validate_authority(
        result["authority"],
        registrations,
        result["phase"],
        expected=expected_authority,
    )
    if expected_authority is not None:
        require(
            validated_progress_authority == dict(expected_authority),
            "progress authority campaign binding",
        )
    require_exact_keys(
        result["registrations"], frozenset(PANEL_IDS.values()), "progress registrations"
    )
    require_exact_keys(
        result["rgb_inputs"], frozenset(PANEL_IDS.values()), "progress RGB inputs"
    )
    for seed_start, panel_id in PANEL_IDS.items():
        registration = registrations[seed_start]
        expected_registration = {
            "registration_id": registration["registration_id"],
            "file_sha256": registration_hashes[seed_start],
            "canonical_self_sha256": registration["self_sha256"],
        }
        require(
            result["registrations"][panel_id] == expected_registration,
            "progress registration binding",
        )
        require_exact_keys(
            result["rgb_inputs"][panel_id],
            producer.RGB_INPUT_FIELDS,
            f"progress RGB inputs.{panel_id}",
        )
        for hash_key in (
            "result_sha256",
            "manifest_sha256",
            "complete_sha256",
            "producer_sha256",
            "image_tree_sha256",
        ):
            require(
                is_sha256(result["rgb_inputs"][panel_id][hash_key]),
                f"invalid RGB provenance {panel_id}.{hash_key}",
            )
        require(
            result["rgb_inputs"][panel_id]["producer_sha256"]
            == registration["sources"]["formal_replay"]["sha256"],
            "RGB producer prereg binding",
        )
        require(
            type(result["rgb_inputs"][panel_id]["frames"]) is int
            and result["rgb_inputs"][panel_id]["frames"]
            == _expected_frames_for_registration(registration),
            "RGB frame count provenance",
        )

    expected_episode_ids = {
        PANEL_IDS[seed]: [
            int(item["episode_id"])
            for item in registrations[seed]["panel"]["episodes"]
        ]
        for seed in (8000, 8100)
    }
    require(result["episode_ids"] == expected_episode_ids, "progress episode IDs")
    require(
        result["observation_contract"]
        == {
            "decoded_cameras": ["agentview", "eye_in_hand"],
            "semantic_inputs": ["current_agentview_rgb", "current_eye_in_hand_rgb"],
            "causal_past_state_only": True,
        },
        "progress observation contract",
    )
    require_exact_keys(result["engines"], producer.ENGINE_FIELDS, "progress engines")
    expected_engine = {
        "sam3_engine_sha256": registrations[8000]["sources"]["sam3_engine"]["sha256"],
        "sift_engine_sha256": registrations[8000]["sources"]["sift_engine"]["sha256"],
        "can_state_sha256": registrations[8000]["sources"]["can_state"]["sha256"],
        "cream_state_sha256": registrations[8000]["sources"]["cream_state"]["sha256"],
        "sam3_checkpoint_sha256": registrations[8000]["models"]["sam3"]
        ["checkpoint"]["sha256"],
        "clip_tokenizer_tree_sha256": registrations[8000]["models"]["sam3"]
        ["clip_tokenizer"]["tree_sha256"],
        "sam3_config_sha256": registrations[8000]["models"]["sam3"]["config_sha256"],
        "sam3_processor_sha256": registrations[8000]["models"]["sam3"]["processor_sha256"],
        "cream_template_image_sha256": registrations[8000]["templates"]
        ["cream_template_image"]["sha256"],
        "cream_template_metadata_sha256": registrations[8000]["templates"]
        ["cream_template_metadata"]["sha256"],
        "semantic_config_sha256": config_sha,
        "transformers_source_sha256s": {
            role: registrations[8000]["runtime"]["runtime_sources"][role]["sha256"]
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
    require(result["engines"] == expected_engine, "progress engine prereg binding")
    require(
        config_sha
        == registrations[8000]["templates"]["semantic_config"]["sha256"]
        == registrations[8100]["templates"]["semantic_config"]["sha256"],
        "progress config is not the registered semantic config",
    )
    expected_prefix = {
        f"{PANEL_IDS[seed]}:ep{episode_id:02d}"
        for seed in (8000, 8100)
        for episode_id in expected_episode_ids[PANEL_IDS[seed]]
    }
    require(set(result["prefix_checks"]) == expected_prefix, "prefix episode set")
    for key, check in result["prefix_checks"].items():
        require_exact_keys(check, producer.PREFIX_CHECK_FIELDS, f"prefix {key}")
        require(check["frames"] == 8, "prefix frame count")
        require(is_sha256(check["reference_sha256"]), "prefix reference hash")
        require(is_sha256(check["fresh_session_sha256"]), "prefix fresh hash")
        require(type(check["exact_after_rounding_6_decimals"]) is bool, "prefix bool")
    require_exact_keys(
        result["recovered_endpoint"],
        producer.RECOVERED_RESULT_FIELDS,
        "progress recovered endpoint",
    )
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
        "progress recovered endpoint declaration",
    )
    require_exact_keys(result["output"], producer.OUTPUT_RESULT_FIELDS, "progress output")
    require(result["output"]["frames"] == len(frames), "progress frame count")
    require(
        result["output"]["frames_sha256"]
        == hashlib.sha256(frames_bytes).hexdigest(),
        "frame seal",
    )
    require(
        result["output"]["producer_sha256"]
        == hashlib.sha256(producer_bytes).hexdigest(),
        "producer seal",
    )
    require(result["output"]["semantic_config_sha256"] == config_sha, "config seal")
    registered_producer = registrations[8000]["sources"]["semantic_producer"]["sha256"]
    require(
        hashlib.sha256(producer_bytes).hexdigest()
        == registered_producer,
        "producer prereg binding",
    )
    require_exact_keys(result["safety"], producer.SAFETY_RESULT_FIELDS, "progress safety")
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
        "progress safety",
    )
    producer._reject_public_leakage(result)
    _validate_semantic_sequence(frames, registrations)
    expected_total = sum(
        _expected_frames_for_registration(registrations[seed])
        for seed in (8000, 8100)
    )
    require(len(frames) == expected_total, "progress total frame count")
    recovered = [
        frame
        for frame in frames
        if frame["panel_id"] == "chain3_8100"
        and frame["episode_id"] == 77
        and frame["boundary_role"] == "recovered_endpoint"
    ]
    require(
        len(recovered) == 1
        and recovered[0]["t"] == 651
        and recovered[0]["phi_only"] is True
        and recovered[0]["dynamics_transition_eligible"] is False,
        "missing valid ep77 recovered endpoint",
    )

    require_exact_keys(complete, producer.COMPLETE_FIELDS, "progress COMPLETE")
    require(complete["schema"] == producer.COMPLETE_SCHEMA, "progress COMPLETE schema")
    require(complete["status"] == "atomic_success", "progress COMPLETE status")
    require(complete["atomic_commit"] is True, "progress atomic commit")
    require(complete["phase"] == phase, "progress COMPLETE phase")
    require_exact_keys(
        complete["authority"], producer.AUTHORITY_FIELDS, "progress COMPLETE authority"
    )
    require(
        complete["authority"] == result["authority"],
        "progress RESULT/COMPLETE authority drift",
    )
    require(
        complete["registration_sha256s"]
        == {PANEL_IDS[key]: registration_hashes[key] for key in (8000, 8100)},
        "progress COMPLETE registration binding",
    )
    result_sha = hashlib.sha256(result_bytes).hexdigest()
    frames_sha = hashlib.sha256(frames_bytes).hexdigest()
    producer_sha = hashlib.sha256(producer_bytes).hexdigest()
    require(complete["result_sha256"] == result_sha, "result seal")
    require(complete["frames_sha256"] == frames_sha, "frames seal")
    require(complete["producer_sha256"] == producer_sha, "producer seal")
    require(complete["semantic_config_sha256"] == config_sha, "config seal")
    producer._read_sealed_regular_bytes(
        complete_path,
        expected_complete_sha256,
        "progress COMPLETE after unprivileged validation",
    )
    seals = {
        "complete_sha256": expected_complete_sha256,
        "result_sha256": result_sha,
        "frames_sha256": frames_sha,
        "producer_sha256": producer_sha,
        "semantic_config_sha256": config_sha,
        "authority": dict(result["authority"]),
    }
    return result, frames, config, seals


def make_pre_oracle_seal(
    *,
    stage: Path,
    phase: str,
    registration_hashes: Mapping[int, str],
    progress_seals: Mapping[str, Any],
    evaluator_sha256: str,
    authority: Mapping[str, Any],
) -> tuple[Path, str]:
    payload = {
        "schema": PRE_ORACLE_SCHEMA,
        "status": "sealed_before_oracle_access",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "oracle_opened_or_hashed": False,
        "phase": phase,
        "registration_sha256s": {
            PANEL_IDS[key]: registration_hashes[key] for key in (8000, 8100)
        },
        "progress_artifact": dict(progress_seals),
        "evaluator_sha256": evaluator_sha256,
        "authority": dict(authority),
    }
    require_exact_keys(payload, PRE_ORACLE_FIELDS, "pre-oracle seal")
    require_exact_keys(
        payload["progress_artifact"], PROGRESS_SEAL_FIELDS, "pre-oracle progress"
    )
    path = stage / PRE_ORACLE_FILENAME
    _write_json(path, payload)
    digest = sha256_file(path)
    require(sha256_file(path) == digest, "pre-oracle seal instability")
    return path, digest


def validate_published_pre_oracle(
    *,
    root: Path,
    expected_pre_oracle_sha256: str,
    expected_complete_sha256: str,
    phase: str,
    registration_hashes: Mapping[int, str],
    progress_seals: Mapping[str, Any],
    evaluator_sha256: str,
    expected_authority: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], bytes]:
    """Validate the separately published, externally anchored oracle boundary."""
    require(is_sha256(expected_pre_oracle_sha256), "invalid external pre-oracle SHA")
    require(is_sha256(expected_complete_sha256), "invalid pre-oracle COMPLETE anchor")
    lexical_root = producer._absolute_lexical(root)
    require(not lexical_root.is_symlink(), "pre-oracle root is a symlink")
    published = lexical_root.resolve()
    require(
        published.is_dir() and not published.is_symlink(),
        "published pre-oracle directory missing/symlink",
    )
    require(
        {path.name for path in published.iterdir()}
        == {PRE_ORACLE_FILENAME, EVALUATOR_FILENAME, COMPLETE_FILENAME},
        "pre-oracle artifact root closure mismatch",
    )
    preseal_path = published / PRE_ORACLE_FILENAME
    evaluator_path = published / EVALUATOR_FILENAME
    complete_path = published / COMPLETE_FILENAME
    complete_bytes = producer._read_sealed_regular_bytes(
        complete_path,
        expected_complete_sha256,
        "externally anchored pre-oracle COMPLETE",
    )
    preseal_bytes = producer._read_sealed_regular_bytes(
        preseal_path,
        expected_pre_oracle_sha256,
        "externally anchored PRE_ORACLE_SEAL",
    )
    preseal = producer._strict_json_bytes(preseal_bytes, "published pre-oracle seal")
    require_exact_keys(preseal, PRE_ORACLE_FIELDS, "published pre-oracle seal")
    require(preseal["schema"] == PRE_ORACLE_SCHEMA, "published pre-oracle schema")
    require(
        preseal["status"] == "sealed_before_oracle_access"
        and preseal["oracle_opened_or_hashed"] is False,
        "published pre-oracle status",
    )
    require(preseal["phase"] == phase, "published pre-oracle phase")
    require(
        preseal["registration_sha256s"]
        == {PANEL_IDS[seed]: registration_hashes[seed] for seed in (8000, 8100)},
        "published pre-oracle registration binding",
    )
    require(
        preseal["progress_artifact"] == progress_seals,
        "published pre-oracle progress binding",
    )
    require(preseal["evaluator_sha256"] == evaluator_sha256, "pre-oracle evaluator")
    require(
        preseal["authority"] == dict(expected_authority),
        "published pre-oracle campaign authority",
    )
    producer._read_sealed_regular_bytes(
        evaluator_path,
        evaluator_sha256,
        "published pre-oracle evaluator snapshot",
    )
    complete = producer._strict_json_bytes(complete_bytes, "pre-oracle COMPLETE")
    require_exact_keys(complete, PRE_ORACLE_COMPLETE_FIELDS, "pre-oracle COMPLETE")
    require(
        complete
        == {
            "schema": PRE_ORACLE_COMPLETE_SCHEMA,
            "status": "atomic_pre_oracle_seal_complete",
            "atomic_commit": True,
            "pre_oracle_seal_sha256": expected_pre_oracle_sha256,
            "evaluator_sha256": evaluator_sha256,
            "authority": dict(expected_authority),
        },
        "pre-oracle COMPLETE content drift",
    )
    return preseal_path, preseal, preseal_bytes


def publish_pre_oracle_seal(
    *,
    freeze_bundle_root: Path,
    expected_freeze_complete_sha256: str,
    progress_root: Path,
    expected_progress_complete_sha256: str,
    expected_confirm_authorization_complete_sha256: str | None = None,
) -> dict[str, Any]:
    """Phase A: publish and stop without stat'ing, hashing, or opening an oracle."""
    (
        bundle,
        registrations,
        registration_hashes,
        progress_result,
        _frames,
        _config,
        progress_seals,
    ) = load_anchored_campaign_progress(
        freeze_bundle_root=freeze_bundle_root,
        expected_freeze_complete_sha256=expected_freeze_complete_sha256,
        progress_root=progress_root,
        expected_progress_complete_sha256=expected_progress_complete_sha256,
        expected_confirm_authorization_complete_sha256=(
            expected_confirm_authorization_complete_sha256
        ),
    )
    phase = progress_result["phase"]
    authority = campaign_node_authority(
        bundle,
        phase,
        "pre_oracle",
        progress_result["authority"]["confirm_authorization_complete_sha256"],
    )
    evaluator_source = Path(__file__).resolve()
    evaluator_bytes, evaluator_sha = capture_live_evaluator_source(
        registrations, evaluator_source
    )
    protected_oracles = [
        Path(registrations[seed]["oracle"]["summary"]["path"])
        for seed in (8000, 8100)
    ]
    destination = (
        Path(bundle["target_root"]) / authority["output_complete_slot"]
    ).parent
    destination = producer._check_output_separation(
        destination,
        [
            Path(bundle["root"]),
            *(
                Path(bundle["registrations"][slot]["path"])
                for slot in authority["registration_slots"]
            ),
            progress_root,
        ],
        lexical_only_paths=protected_oracles,
    )
    support = producer._load_rgb_support(registrations[8000])
    final_digest = ""

    def build(stage: Path) -> None:
        nonlocal final_digest
        _write_bytes(stage / EVALUATOR_FILENAME, evaluator_bytes)
        _path, final_digest = make_pre_oracle_seal(
            stage=stage,
            phase=phase,
            registration_hashes=registration_hashes,
            progress_seals=progress_seals,
            evaluator_sha256=evaluator_sha,
            authority=authority,
        )
        _write_json(
            stage / COMPLETE_FILENAME,
            {
                "schema": PRE_ORACLE_COMPLETE_SCHEMA,
                "status": "atomic_pre_oracle_seal_complete",
                "atomic_commit": True,
                "pre_oracle_seal_sha256": final_digest,
                "evaluator_sha256": evaluator_sha,
                "authority": dict(authority),
            },
        )

    def validate(stage: Path) -> None:
        revalidate_anchored_campaign_progress(
            freeze_bundle_root=freeze_bundle_root,
            expected_freeze_complete_sha256=expected_freeze_complete_sha256,
            progress_root=progress_root,
            expected_progress_complete_sha256=expected_progress_complete_sha256,
            expected_bundle=bundle,
            expected_result=progress_result,
            expected_seals=progress_seals,
            expected_confirm_authorization_complete_sha256=(
                expected_confirm_authorization_complete_sha256
            ),
        )
        validate_published_pre_oracle(
            root=stage,
            expected_pre_oracle_sha256=final_digest,
            expected_complete_sha256=sha256_file(stage / COMPLETE_FILENAME),
            phase=phase,
            registration_hashes=registration_hashes,
            progress_seals=progress_seals,
            evaluator_sha256=evaluator_sha,
            expected_authority=authority,
        )

    support.atomic_publish_directory(
        destination,
        build,
        validate,
        trusted_target_root=bundle["target_root"],
    )
    return {
        "root": destination,
        "phase": phase,
        "pre_oracle_seal_sha256": final_digest,
        "pre_oracle_complete_sha256": sha256_file(destination / COMPLETE_FILENAME),
        "authority": authority,
    }


def load_registrations_after_preseal(
    *,
    paths: Sequence[Path],
    expected_hashes: Sequence[str],
    preseal_path: Path,
    preseal_sha256: str,
    unprivileged_registrations: Mapping[int, Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    producer._read_sealed_regular_bytes(
        preseal_path,
        preseal_sha256,
        "published pre-oracle seal immediately before oracle access",
    )
    require(len(paths) == len(expected_hashes) == 2, "two oracle registrations")
    output: dict[int, dict[str, Any]] = {}
    for path, expected in zip(paths, expected_hashes, strict=True):
        registration, actual = load_and_validate_registration(
            path,
            expected_source_role="semantic_evaluator",
            expected_file_sha256=expected,
        )
        require(actual == expected, "postseal registration SHA anchor")
        seed_start = int(registration["panel"]["seed_start"])
        require(seed_start not in output, "duplicate postseal registration")
        require(
            registration == unprivileged_registrations[seed_start],
            "registration changed across oracle boundary",
        )
        output[seed_start] = registration
    require(set(output) == {8000, 8100}, "postseal registration panel set")
    return output


def load_oracle_records(
    registration: Mapping[str, Any]
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    reference = registration["oracle"]["summary"]
    path = Path(reference["path"])
    payload = producer._strict_json_bytes(
        producer._read_sealed_regular_bytes(
            path,
            reference["sha256"],
            "registered oracle",
            expected_bytes=reference["bytes"],
        ),
        "registered oracle",
    )
    raw_records = payload.get("episode_records")
    require(
        isinstance(raw_records, list) and len(raw_records) == 96,
        "oracle must contain 96 episode records",
    )
    records: dict[int, dict[str, Any]] = {}
    for expected_episode, raw in enumerate(raw_records):
        require(isinstance(raw, Mapping), "oracle episode record must be object")
        require(
            {"idx", "seed", "steps", "events"}.issubset(raw),
            "oracle episode record is missing a required field",
        )
        episode_id = raw["idx"]
        env_seed = raw["seed"]
        steps = raw["steps"]
        events_raw = raw["events"]
        require(type(episode_id) is int, "oracle idx type")
        require(episode_id == expected_episode, "oracle episode order")
        require(
            env_seed == registration["panel"]["seed_start"] + episode_id,
            "oracle seed rule",
        )
        require(type(steps) is int and 0 < steps <= 750, "oracle steps")
        require(isinstance(events_raw, Mapping), "oracle events object")
        events: dict[int, int] = {}
        for raw_key, raw_step in events_raw.items():
            require(
                isinstance(raw_key, str) and raw_key == str(int(raw_key)),
                "oracle event key",
            )
            event_index = int(raw_key)
            require(0 <= event_index <= 5 and event_index not in events, "oracle event index")
            require(
                type(raw_step) is int and 0 <= raw_step <= steps,
                "oracle event step",
            )
            events[event_index] = raw_step
        records[episode_id] = {
            "env_seed": env_seed,
            "steps": steps,
            "events": events,
        }
    provenance = {
        "file_sha256": reference["sha256"],
        "bytes": reference["bytes"],
        "fields_used": ["episode_records.idx", "seed", "steps", "events"],
        "success_values_accessed": False,
    }
    require_exact_keys(provenance, ORACLE_INPUT_FIELDS, "oracle provenance")
    return records, provenance


def ordinal_truth_times(events: Mapping[int, int]) -> dict[str, int | None]:
    can_times = sorted(events[index] for index in CAN_EVENT_INDICES if index in events)
    return {
        "can_ge1": can_times[0] if can_times else None,
        "can_ge2": can_times[1] if len(can_times) == 2 else None,
        "cream": events.get(CREAM_EVENT_INDEX),
    }


def _f1(cell: Mapping[str, int]) -> float | None:
    if cell["certain_support"] == 0:
        return None
    denominator = 2 * cell["tp"] + cell["fp"] + cell["fn"]
    return 1.0 if denominator == 0 else 2 * cell["tp"] / denominator


def _nearest_rank(values: Sequence[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return float(ordered[rank - 1])


def _median(values: Sequence[int]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def compute_metrics(
    frames: Sequence[Mapping[str, Any]],
    records: Mapping[tuple[str, int], Mapping[str, Any]],
    expected_frames: int | None = None,
) -> dict[str, Any]:
    by_episode: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for frame in frames:
        by_episode[(str(frame["panel_id"]), int(frame["episode_id"]))].append(frame)
    for episode_frames in by_episode.values():
        episode_frames.sort(key=lambda item: int(item["t"]))
    require(set(by_episode) == set(records), "metric frame/oracle episode set")

    confusion = {
        name: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "uncertain": 0,
            "certain_support": 0,
        }
        for name in ATOM_NAMES
    }
    atom_labels = len(frames) * len(ATOM_NAMES)
    reliable_atom_labels = 0
    certain_scalar_frames = 0
    correct_scalar_frames = 0
    all_scalar_frames = len(frames)
    for frame in frames:
        key = (str(frame["panel_id"]), int(frame["episode_id"]))
        truth_times = ordinal_truth_times(records[key]["events"])
        truth_atoms = {
            name: truth_times[name] is not None and truth_times[name] <= int(frame["t"])
            for name in ATOM_NAMES
        }
        for name in ATOM_NAMES:
            cell = confusion[name]
            if not frame["reliability"][name]:
                cell["uncertain"] += 1
                continue
            reliable_atom_labels += 1
            cell["certain_support"] += 1
            prediction = bool(frame["atoms"][name])
            truth = bool(truth_atoms[name])
            if prediction and truth:
                cell["tp"] += 1
            elif prediction and not truth:
                cell["fp"] += 1
            elif not prediction and truth:
                cell["fn"] += 1
            else:
                cell["tn"] += 1
        if frame["reliability"]["scalar"]:
            certain_scalar_frames += 1
            correct_scalar_frames += int(
                int(frame["scalar"]) == sum(int(value) for value in truth_atoms.values())
            )

    per_atom_f1 = {name: _f1(cell) for name, cell in confusion.items()}
    f1_values = [value for value in per_atom_f1.values() if value is not None]
    macro_f1 = sum(f1_values) / 3.0 if len(f1_values) == 3 else None
    evidence_coverage = reliable_atom_labels / atom_labels if atom_labels else None
    uncertainty_rate = (
        (atom_labels - reliable_atom_labels) / atom_labels if atom_labels else None
    )
    exact_scalar_accuracy = (
        correct_scalar_frames / certain_scalar_frames if certain_scalar_frames else None
    )
    effective_scalar_accuracy = (
        correct_scalar_frames / all_scalar_frames if all_scalar_frames else None
    )

    timing_errors: list[int] = []
    transition_rows = []
    missing = 0
    extra = 0
    right_censored = 0
    observable = 0
    cream_positive = 0
    reverse_order = 0
    for key, episode_frames in sorted(by_episode.items()):
        record = records[key]
        events = record["events"]
        truth_times = ordinal_truth_times(events)
        last_t = int(episode_frames[-1]["t"])
        if 1 in events and 3 in events and events[3] < events[1]:
            reverse_order += 1
        for name in ATOM_NAMES:
            predicted_times = [
                int(frame["t"]) for frame in episode_frames if frame["transitions"][name]
            ]
            require(len(predicted_times) <= 1, "non-monotone transition multiplicity")
            predicted = predicted_times[0] if predicted_times else None
            truth = truth_times[name]
            if truth is not None and truth <= last_t:
                observable += 1
                cream_positive += int(name == "cream")
                if predicted is None:
                    missing += 1
                    status = "missing"
                    error = None
                else:
                    error = abs(predicted - truth)
                    timing_errors.append(error)
                    status = "matched"
            elif truth is not None:
                right_censored += 1
                if predicted is None:
                    status = "right_censored"
                    error = None
                else:
                    extra += 1
                    status = "extra_before_right_censored_truth"
                    error = None
            elif predicted is not None:
                extra += 1
                status = "extra_without_truth"
                error = None
            else:
                status = "negative_no_transition"
                error = None
            transition_rows.append(
                {
                    "panel_id": key[0],
                    "episode_id": key[1],
                    "atom": name,
                    "oracle_t": truth,
                    "last_observed_t": last_t,
                    "predicted_t": predicted,
                    "absolute_error_steps": error,
                    "status": status,
                }
            )

    never_total = 0
    never_true_negative = 0
    never_rows = []
    held_total = 0
    held_true_negative = 0
    held_rows = []
    carry_created = 0
    carry_rows = []
    for key, episode_frames in sorted(by_episode.items()):
        events = records[key]["events"]
        truth_times = ordinal_truth_times(events)
        final = episode_frames[-1]
        for name in ATOM_NAMES:
            if truth_times[name] is not None:
                continue
            never_total += 1
            correct = bool(final["reliability"][name] and not final["atoms"][name])
            never_true_negative += int(correct)
            never_rows.append(
                {
                    "panel_id": key[0],
                    "episode_id": key[1],
                    "atom": name,
                    "reliable": bool(final["reliability"][name]),
                    "predicted": bool(final["atoms"][name]),
                    "true_negative": correct,
                }
            )
        if HELD_CREAM_EVENT_INDEX in events and CREAM_EVENT_INDEX not in events:
            held_total += 1
            correct = bool(
                final["reliability"]["cream"] and not final["atoms"]["cream"]
            )
            held_true_negative += int(correct)
            held_rows.append(
                {
                    "panel_id": key[0],
                    "episode_id": key[1],
                    "reliable": bool(final["reliability"]["cream"]),
                    "predicted": bool(final["atoms"]["cream"]),
                    "true_negative": correct,
                }
            )
        for frame in episode_frames:
            can_carry = bool(frame["causal_state"]["can_used_carry"])
            cream_carry = bool(frame["causal_state"]["cream_used_carry"])
            created_atoms = [
                name
                for name in ATOM_NAMES
                if frame["transitions"][name]
                and (cream_carry if name == "cream" else can_carry)
            ]
            carry_created += len(created_atoms)
            if created_atoms:
                carry_rows.append(
                    {
                        "panel_id": key[0],
                        "episode_id": key[1],
                        "t": int(frame["t"]),
                        "atoms": created_atoms,
                    }
                )

    expected = expected_frames if expected_frames is not None else len(frames)
    return {
        "coverage": {
            "expected_rgb_frames": expected,
            "progress_frames": len(frames),
            "frame_coverage": len(frames) / expected if expected else None,
            "atom_labels": atom_labels,
            "reliable_atom_labels": reliable_atom_labels,
            "evidence_coverage": evidence_coverage,
            "uncertainty_rate": uncertainty_rate,
        },
        "classification": {
            "confusion": confusion,
            "per_atom_positive_f1": per_atom_f1,
            "macro_f1": macro_f1,
            "certain_scalar_frames": certain_scalar_frames,
            "correct_scalar_frames": correct_scalar_frames,
            "exact_scalar_accuracy_certain_only": exact_scalar_accuracy,
            "effective_scalar_accuracy_uncertain_incorrect": effective_scalar_accuracy,
            "empty_positive_f1_convention": (
                "1.0 iff certain support exists and tp=fp=fn=0"
            ),
        },
        "transitions": {
            "identity_free_oracle": (
                "sort present chain3 place events {1,3} into can_ge1/can_ge2; "
                "cream uses event 5"
            ),
            "p95_convention": "nearest-rank ceil(0.95*n)",
            "observable_truth_transitions": observable,
            "matched_transitions": len(timing_errors),
            "missing_transitions": missing,
            "extra_transitions": extra,
            "right_censored_truth_transitions": right_censored,
            "median_absolute_error_steps": _median(timing_errors),
            "p95_absolute_error_steps": _nearest_rank(timing_errors, 0.95),
            "maximum_absolute_error_steps": (
                float(max(timing_errors)) if timing_errors else None
            ),
            "chain3_cream_positive_transitions": cream_positive,
            "reverse_order_episodes": reverse_order,
            "rows": transition_rows,
        },
        "never_achieved_terminal_specificity": {
            "denominator": never_total,
            "true_negatives": never_true_negative,
            "specificity": never_true_negative / never_total if never_total else None,
            "rows": never_rows,
        },
        "held_cream_specificity": {
            "definition": "chain3 event 4 present and event 5 absent",
            "denominator": held_total,
            "true_negatives": held_true_negative,
            "specificity": held_true_negative / held_total if held_total else None,
            "rows": held_rows,
        },
        "carry_audit": {
            "carry_created_transitions": carry_created,
            "rows": carry_rows,
        },
    }


def _compare(value: Any, operation: str, threshold: float) -> bool:
    if not _finite_number(value):
        return False
    if operation == "gte":
        return float(value) >= threshold
    if operation == "lte":
        return float(value) <= threshold
    raise ValueError(operation)


def visual_subgate_decision(
    *,
    metrics: Mapping[str, Any],
    config: Mapping[str, Any],
    phase: str,
    primary_episode_count: int,
    all_prefix_exact: bool,
    recovery_passed: bool,
) -> dict[str, Any]:
    require(phase in {"screen", "confirm"}, "visual subgate phase")
    gates = config["promotion_gates"]
    coverage = metrics["coverage"]
    classification = metrics["classification"]
    transitions = metrics["transitions"]
    never = metrics["never_achieved_terminal_specificity"]
    held = metrics["held_cream_specificity"]
    carry = metrics["carry_audit"]
    expected_episodes = (
        config["screen"]["required_primary_chain3_episodes"]
        if phase == "screen"
        else gates["required_primary_chain3_episodes"]
    )
    values = {
        "primary_chain3_episodes": primary_episode_count,
        "frame_coverage": coverage["frame_coverage"],
        "evidence_coverage": coverage["evidence_coverage"],
        "uncertainty_rate": coverage["uncertainty_rate"],
        # Uncertain frames count as incorrect here; the separate uncertainty
        # threshold cannot be used to hide up to 15% abstentions.
        "exact_scalar_accuracy": classification[
            "effective_scalar_accuracy_uncertain_incorrect"
        ],
        "macro_f1": classification["macro_f1"],
        "missing_transitions": transitions["missing_transitions"],
        "extra_transitions": transitions["extra_transitions"],
        "median_timing_error_steps": transitions["median_absolute_error_steps"],
        "p95_timing_error_steps": transitions["p95_absolute_error_steps"],
        "maximum_timing_error_steps": transitions["maximum_absolute_error_steps"],
        "never_achieved_terminal_specificity": never["specificity"],
        "held_cream_specificity": held["specificity"],
        "chain3_cream_positive_transitions": transitions[
            "chain3_cream_positive_transitions"
        ],
        "carry_created_transitions": carry["carry_created_transitions"],
        "all_prefix_checks_exact": all_prefix_exact,
        "terminal_recovery_passed": recovery_passed,
    }
    checks = {
        "exact_primary_episode_count": primary_episode_count == expected_episodes,
        "complete_progress_to_rgb_frame_bijection": coverage["frame_coverage"] == 1.0,
        "minimum_evidence_coverage": _compare(
            values["evidence_coverage"], "gte", gates["minimum_evidence_coverage"]
        ),
        "maximum_uncertainty_rate": _compare(
            values["uncertainty_rate"], "lte", gates["maximum_uncertainty_rate"]
        ),
        "minimum_exact_scalar_accuracy": _compare(
            values["exact_scalar_accuracy"],
            "gte",
            gates["minimum_exact_scalar_accuracy"],
        ),
        "minimum_macro_f1": _compare(
            values["macro_f1"], "gte", gates["minimum_macro_f1"]
        ),
        "maximum_missing_transitions": (
            values["missing_transitions"] <= gates["maximum_missing_transitions"]
        ),
        "maximum_extra_transitions": (
            values["extra_transitions"] <= gates["maximum_extra_transitions"]
        ),
        "maximum_median_timing_error_steps": _compare(
            values["median_timing_error_steps"],
            "lte",
            gates["maximum_median_timing_error_steps"],
        ),
        "maximum_p95_timing_error_steps": _compare(
            values["p95_timing_error_steps"],
            "lte",
            gates["maximum_p95_timing_error_steps"],
        ),
        "maximum_timing_error_steps": _compare(
            values["maximum_timing_error_steps"],
            "lte",
            gates["maximum_timing_error_steps"],
        ),
        "minimum_never_achieved_terminal_specificity": _compare(
            values["never_achieved_terminal_specificity"],
            "gte",
            gates["minimum_never_achieved_terminal_specificity"],
        ),
        "never_achieved_support_present": never["denominator"] > 0,
        "minimum_held_cream_specificity": _compare(
            values["held_cream_specificity"],
            "gte",
            gates["minimum_held_cream_specificity"],
        ),
        "held_cream_support_present": held["denominator"] > 0,
        "minimum_chain3_cream_positive_transitions": (
            values["chain3_cream_positive_transitions"]
            >= gates["minimum_chain3_cream_positive_transitions"]
        ),
        "maximum_carry_created_transitions": (
            values["carry_created_transitions"]
            <= gates["maximum_carry_created_transitions"]
        ),
        "all_prefix_checks_exact": (
            all_prefix_exact if gates["require_all_prefix_checks_exact"] else True
        ),
        "terminal_recovery_passed": (
            recovery_passed if gates["require_terminal_recovery_pass"] else True
        ),
    }
    metric_checks_passed = all(checks.values())
    promotion_authorized = bool(phase == "confirm" and metric_checks_passed)
    decision = "PASS" if promotion_authorized else "STOP"
    return {
        "schema": VISUAL_SUBGATE_SCHEMA,
        "claim_scope": "gate0_visual_subgate_only",
        "phase": phase,
        "decision": decision,
        "promotion_authorized": promotion_authorized,
        "screen_metrics_passed": metric_checks_passed if phase == "screen" else None,
        "metric_checks_passed": metric_checks_passed,
        "thresholds": {
            "phase_required_primary_chain3_episodes": expected_episodes,
            **dict(gates),
        },
        "values": values,
        "checks": checks,
    }


def _combined_records(
    registrations: Mapping[int, Mapping[str, Any]]
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
    combined = {}
    provenance = {}
    for seed_start in (8000, 8100):
        records, source = load_oracle_records(registrations[seed_start])
        panel_id = PANEL_IDS[seed_start]
        selected = {
            int(item["episode_id"])
            for item in registrations[seed_start]["panel"]["episodes"]
        }
        for episode_id in selected:
            combined[(panel_id, episode_id)] = records[episode_id]
        provenance[panel_id] = source
    return combined, provenance


def _validate_evaluation_stage(
    stage: Path,
    registrations: Mapping[int, Mapping[str, Any]],
    registration_hashes: Mapping[int, str],
    progress_seals: Mapping[str, Any],
    expected_authority: Mapping[str, Any] | None = None,
    expected_payload: Mapping[str, Any] | None = None,
) -> None:
    result_path = stage / RESULT_FILENAME
    subgate_path = stage / SUBGATE_FILENAME
    complete_path = stage / COMPLETE_FILENAME
    preseal_path = stage / PRE_ORACLE_FILENAME
    evaluator_path = stage / EVALUATOR_FILENAME
    for path in (result_path, subgate_path, complete_path, preseal_path, evaluator_path):
        require(path.is_file() and not path.is_symlink(), f"missing evaluation {path.name}")
    result = _strict_json_file(result_path, "evaluation RESULT")
    subgate = _strict_json_file(subgate_path, "visual subgate")
    complete = _strict_json_file(complete_path, "evaluation COMPLETE")
    preseal = _strict_json_file(preseal_path, "pre-oracle seal")
    require_exact_keys(preseal, PRE_ORACLE_FIELDS, "pre-oracle seal")
    require(preseal["schema"] == PRE_ORACLE_SCHEMA, "pre-oracle schema")
    require(preseal["status"] == "sealed_before_oracle_access", "pre-oracle status")
    require(preseal["oracle_opened_or_hashed"] is False, "pre-oracle access flag")
    require(preseal["progress_artifact"] == progress_seals, "pre-oracle progress seal")
    require(
        preseal["registration_sha256s"]
        == {PANEL_IDS[key]: registration_hashes[key] for key in (8000, 8100)},
        "pre-oracle registration binding",
    )
    require_exact_keys(result, EVALUATION_FIELDS, "evaluation RESULT")
    require(result["schema"] == EVALUATION_SCHEMA, "evaluation schema")
    require(result["status"] == "postseal_visual_evaluation_complete", "evaluation status")
    require(result["claim_scope"] == "gate0_visual_subgate_only", "evaluation scope")
    require(result["phase"] in {"screen", "confirm"}, "evaluation phase")
    require(preseal["phase"] == result["phase"], "pre-oracle/evaluation phase drift")
    require(result["decision"] in {"PASS", "STOP"}, "evaluation decision")
    validate_combined_authority(
        result["authority"],
        registrations,
        result["phase"],
        expected=expected_authority,
    )
    require(
        result["authority"]["output_complete_slot"]
        == f"{result['phase']}/evaluation/COMPLETE.json",
        "evaluation authority output slot drift",
    )
    require(
        {path.name for path in stage.iterdir()}
        == {
            RESULT_FILENAME,
            SUBGATE_FILENAME,
            COMPLETE_FILENAME,
            PRE_ORACLE_FILENAME,
            EVALUATOR_FILENAME,
            f"{result['decision']}.json",
        },
        "evaluation artifact root closure mismatch",
    )
    require(result["pre_oracle_seal_sha256"] == sha256_file(preseal_path), "preseal hash")
    require(
        is_sha256(result["pre_oracle_complete_sha256"]),
        "pre-oracle COMPLETE external anchor missing",
    )
    require(result["progress_artifact"] == progress_seals, "evaluation progress seal")
    require_exact_keys(
        result["registrations"], frozenset(PANEL_IDS.values()), "evaluation registrations"
    )
    require_exact_keys(
        result["oracle_inputs"], frozenset(PANEL_IDS.values()), "evaluation oracle inputs"
    )
    for seed_start, panel_id in PANEL_IDS.items():
        registration = registrations[seed_start]
        require_exact_keys(
            result["registrations"][panel_id],
            producer.REGISTRATION_RESULT_FIELDS,
            f"evaluation registrations.{panel_id}",
        )
        require(
            result["registrations"][panel_id]
            == {
                "registration_id": registration["registration_id"],
                "file_sha256": registration_hashes[seed_start],
                "canonical_self_sha256": registration["self_sha256"],
            },
            "evaluation registration prereg binding",
        )
        oracle = result["oracle_inputs"][panel_id]
        require_exact_keys(oracle, ORACLE_INPUT_FIELDS, f"oracle inputs.{panel_id}")
        require(
            oracle
            == {
                "file_sha256": registration["oracle"]["summary"]["sha256"],
                "bytes": registration["oracle"]["summary"]["bytes"],
                "fields_used": [
                    "episode_records.idx",
                    "seed",
                    "steps",
                    "events",
                ],
                "success_values_accessed": False,
            },
            "evaluation oracle prereg binding",
        )
    require_exact_keys(subgate, SUBGATE_FIELDS, "visual subgate")
    require(subgate["schema"] == VISUAL_SUBGATE_SCHEMA, "subgate schema")
    require(subgate["claim_scope"] == "gate0_visual_subgate_only", "subgate scope")
    require(result["visual_subgate"] == subgate, "embedded/subgate mismatch")
    require(result["decision"] == subgate["decision"], "decision mismatch")
    if result["phase"] == "screen":
        require(subgate["decision"] == "STOP", "screen promoted")
        require(subgate["promotion_authorized"] is False, "screen authorization")
    evaluator_sha = sha256_file(evaluator_path)
    registered_sha = registrations[8000]["sources"]["semantic_evaluator"]["sha256"]
    require(evaluator_sha == registered_sha, "evaluator snapshot prereg binding")
    require_exact_keys(result["provenance"], PROVENANCE_FIELDS, "evaluation provenance")
    require(result["provenance"]["evaluator_sha256"] == evaluator_sha, "evaluator seal")
    require(
        result["provenance"]["registered_evaluator_sha256"] == registered_sha,
        "registered evaluator seal",
    )
    require(
        result["provenance"]["producer_sha256"]
        == progress_seals["producer_sha256"],
        "evaluation producer provenance",
    )
    require(
        result["provenance"]["semantic_config_sha256"]
        == progress_seals["semantic_config_sha256"],
        "evaluation semantic-config provenance",
    )
    require(preseal["evaluator_sha256"] == evaluator_sha, "pre-oracle evaluator seal")
    expected_pre_oracle_authority = {
        **dict(result["authority"]),
        "output_complete_slot": f"{result['phase']}/preoracle/COMPLETE.json",
    }
    require(
        preseal["authority"] == expected_pre_oracle_authority,
        "pre-oracle/evaluation authority DAG drift",
    )
    if expected_payload is not None:
        require(result == expected_payload, "evaluation RESULT differs from builder payload")
    require_exact_keys(complete, COMPLETE_FIELDS, "evaluation COMPLETE")
    require(complete["schema"] == COMPLETE_SCHEMA, "evaluation COMPLETE schema")
    require(complete["status"] == "atomic_evaluation_complete", "COMPLETE status")
    require(complete["atomic_commit"] is True, "COMPLETE atomic flag")
    validate_combined_authority(
        complete["authority"],
        registrations,
        result["phase"],
        expected=expected_authority,
    )
    require(complete["authority"] == result["authority"], "evaluation authority drift")
    require(complete["decision"] == result["decision"], "COMPLETE decision")
    require(complete["result_sha256"] == sha256_file(result_path), "result seal")
    require(complete["subgate_sha256"] == sha256_file(subgate_path), "subgate seal")
    marker_filename = complete["marker_filename"]
    require(marker_filename == f"{result['decision']}.json", "marker filename")
    marker_path = stage / marker_filename
    require(marker_path.is_file() and not marker_path.is_symlink(), "marker missing")
    marker = _strict_json_file(marker_path, "decision marker")
    require_exact_keys(marker, MARKER_FIELDS, "decision marker")
    require(marker["schema"] == MARKER_SCHEMA, "marker schema")
    require(marker["decision"] == result["decision"], "marker decision")
    require(marker["result_sha256"] == sha256_file(result_path), "marker result seal")
    require(marker["subgate_sha256"] == sha256_file(subgate_path), "marker subgate seal")
    require(complete["marker_sha256"] == sha256_file(marker_path), "marker seal")
    require(
        complete["pre_oracle_seal_sha256"] == sha256_file(preseal_path),
        "COMPLETE preseal",
    )
    require(
        complete["pre_oracle_complete_sha256"]
        == result["pre_oracle_complete_sha256"],
        "evaluation pre-oracle COMPLETE anchor drift",
    )
    require(complete["evaluator_sha256"] == evaluator_sha, "COMPLETE evaluator")


def validate_live_evaluator_source(
    registrations: Mapping[int, Mapping[str, Any]], source_path: Path
) -> str:
    registered = {
        registrations[seed]["sources"]["semantic_evaluator"]["sha256"]
        for seed in (8000, 8100)
    }
    require(len(registered) == 1, "cross-panel evaluator source drift")
    expected = next(iter(registered))
    producer._read_sealed_regular_bytes(
        source_path, expected, "live registered semantic evaluator"
    )
    return expected


def capture_live_evaluator_source(
    registrations: Mapping[int, Mapping[str, Any]], source_path: Path
) -> tuple[bytes, str]:
    registered = {
        registrations[seed]["sources"]["semantic_evaluator"]["sha256"]
        for seed in (8000, 8100)
    }
    require(len(registered) == 1, "cross-panel evaluator source drift")
    expected = next(iter(registered))
    payload = producer._read_sealed_regular_bytes(
        source_path, expected, "captured registered semantic evaluator"
    )
    return payload, expected


def evaluate_after_published_preseal(
    *,
    freeze_bundle_root: Path,
    expected_freeze_complete_sha256: str,
    progress_root: Path,
    expected_progress_complete_sha256: str,
    pre_oracle_root: Path,
    expected_pre_oracle_sha256: str,
    expected_pre_oracle_complete_sha256: str,
    expected_confirm_authorization_complete_sha256: str | None = None,
) -> dict[str, Any]:
    """Phase B: consume an external preseal, then and only then read oracles."""
    (
        bundle,
        registrations,
        registration_hashes,
        progress_result,
        frames,
        config,
        progress_seals,
    ) = load_anchored_campaign_progress(
        freeze_bundle_root=freeze_bundle_root,
        expected_freeze_complete_sha256=expected_freeze_complete_sha256,
        progress_root=progress_root,
        expected_progress_complete_sha256=expected_progress_complete_sha256,
        expected_confirm_authorization_complete_sha256=(
            expected_confirm_authorization_complete_sha256
        ),
    )
    phase = progress_result["phase"]
    confirm_authorization_sha = progress_result["authority"][
        "confirm_authorization_complete_sha256"
    ]
    pre_oracle_authority = campaign_node_authority(
        bundle, phase, "pre_oracle", confirm_authorization_sha
    )
    expected_pre_oracle_root = (
        Path(bundle["target_root"])
        / pre_oracle_authority["output_complete_slot"]
    ).parent
    require(
        producer._absolute_lexical(pre_oracle_root).resolve()
        == expected_pre_oracle_root,
        "pre-oracle artifact is not in its frozen campaign slot",
    )
    evaluator_source = Path(__file__).resolve()
    evaluator_bytes, evaluator_sha = capture_live_evaluator_source(
        registrations, evaluator_source
    )
    registered_evaluator_sha = registrations[8000]["sources"]["semantic_evaluator"][
        "sha256"
    ]
    preseal_path, _preseal, preseal_bytes = validate_published_pre_oracle(
        root=pre_oracle_root,
        expected_pre_oracle_sha256=expected_pre_oracle_sha256,
        expected_complete_sha256=expected_pre_oracle_complete_sha256,
        phase=progress_result["phase"],
        registration_hashes=registration_hashes,
        progress_seals=progress_seals,
        evaluator_sha256=evaluator_sha,
        expected_authority=pre_oracle_authority,
    )
    evaluation_authority = campaign_node_authority(
        bundle, phase, "evaluation", confirm_authorization_sha
    )
    destination = (
        Path(bundle["target_root"]) / evaluation_authority["output_complete_slot"]
    ).parent
    destination = producer._check_output_separation(
        destination,
        [
            Path(bundle["root"]),
            *(
                Path(bundle["registrations"][slot]["path"])
                for slot in evaluation_authority["registration_slots"]
            ),
            progress_root,
            pre_oracle_root,
        ],
        lexical_only_paths=[
            Path(registrations[seed]["oracle"]["summary"]["path"])
            for seed in (8000, 8100)
        ],
    )

    # This is the first operation in phase B that is permitted to stat/hash an
    # oracle.  It is unreachable until the externally anchored, separately
    # published preseal above has validated.
    privileged_registrations = load_registrations_after_preseal(
        paths=[
            Path(bundle["registrations"][slot]["path"])
            for slot in evaluation_authority["registration_slots"]
        ],
        expected_hashes=[
            bundle["registrations"][slot]["file_sha256"]
            for slot in evaluation_authority["registration_slots"]
        ],
        preseal_path=preseal_path,
        preseal_sha256=expected_pre_oracle_sha256,
        unprivileged_registrations=registrations,
    )
    records, oracle_provenance = _combined_records(privileged_registrations)
    expected_frames = sum(
        _expected_frames_for_registration(privileged_registrations[seed])
        for seed in (8000, 8100)
    )
    metrics = compute_metrics(frames, records, expected_frames=expected_frames)
    prefix_exact = all(
        bool(check["exact_after_rounding_6_decimals"])
        for check in progress_result["prefix_checks"].values()
    )
    recovery_passed = bool(
        progress_result["recovered_endpoint"]["present"]
        and progress_result["recovered_endpoint"][
            "rgb_recovery_prerequisites_passed"
        ]
    )
    episode_count = sum(
        len(registrations[seed]["panel"]["episodes"]) for seed in (8000, 8100)
    )
    subgate = visual_subgate_decision(
        metrics=metrics,
        config=config,
        phase=progress_result["phase"],
        primary_episode_count=episode_count,
        all_prefix_exact=prefix_exact,
        recovery_passed=recovery_passed,
    )
    registrations_public = {
        PANEL_IDS[seed]: {
            "registration_id": registrations[seed]["registration_id"],
            "file_sha256": registration_hashes[seed],
            "canonical_self_sha256": registrations[seed]["self_sha256"],
        }
        for seed in (8000, 8100)
    }
    support = producer._load_rgb_support(registrations[8000])
    final_payload: dict[str, Any] = {}

    def build(stage: Path) -> None:
        nonlocal final_payload
        _write_bytes(stage / EVALUATOR_FILENAME, evaluator_bytes)
        _write_bytes(stage / PRE_ORACLE_FILENAME, preseal_bytes)
        payload = {
            "schema": EVALUATION_SCHEMA,
            "status": "postseal_visual_evaluation_complete",
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "claim_scope": "gate0_visual_subgate_only",
            "phase": progress_result["phase"],
            "decision": subgate["decision"],
            "authority": dict(evaluation_authority),
            "pre_oracle_seal_sha256": expected_pre_oracle_sha256,
            "pre_oracle_complete_sha256": expected_pre_oracle_complete_sha256,
            "registrations": registrations_public,
            "progress_artifact": progress_seals,
            "oracle_inputs": oracle_provenance,
            "metrics": metrics,
            "visual_subgate": subgate,
            "provenance": {
                "evaluator_sha256": evaluator_sha,
                "registered_evaluator_sha256": registered_evaluator_sha,
                "producer_sha256": progress_seals["producer_sha256"],
                "semantic_config_sha256": progress_seals[
                    "semantic_config_sha256"
                ],
            },
        }
        _write_json(stage / RESULT_FILENAME, payload)
        _write_json(stage / SUBGATE_FILENAME, subgate)
        marker_filename = f"{subgate['decision']}.json"
        marker = {
            "schema": MARKER_SCHEMA,
            "decision": subgate["decision"],
            "claim_scope": "gate0_visual_subgate_only",
            "result_sha256": sha256_file(stage / RESULT_FILENAME),
            "subgate_sha256": sha256_file(stage / SUBGATE_FILENAME),
        }
        _write_json(stage / marker_filename, marker)
        complete = {
            "schema": COMPLETE_SCHEMA,
            "status": "atomic_evaluation_complete",
            "atomic_commit": True,
            "decision": subgate["decision"],
            "authority": dict(evaluation_authority),
            "result_sha256": sha256_file(stage / RESULT_FILENAME),
            "subgate_sha256": sha256_file(stage / SUBGATE_FILENAME),
            "marker_filename": marker_filename,
            "marker_sha256": sha256_file(stage / marker_filename),
            "pre_oracle_seal_sha256": expected_pre_oracle_sha256,
            "pre_oracle_complete_sha256": expected_pre_oracle_complete_sha256,
            "evaluator_sha256": evaluator_sha,
        }
        _write_json(stage / COMPLETE_FILENAME, complete)
        final_payload = payload

    revalidate_anchored_campaign_progress(
        freeze_bundle_root=freeze_bundle_root,
        expected_freeze_complete_sha256=expected_freeze_complete_sha256,
        progress_root=progress_root,
        expected_progress_complete_sha256=expected_progress_complete_sha256,
        expected_bundle=bundle,
        expected_result=progress_result,
        expected_seals=progress_seals,
        expected_confirm_authorization_complete_sha256=(
            expected_confirm_authorization_complete_sha256
        ),
    )
    validate_published_pre_oracle(
        root=pre_oracle_root,
        expected_pre_oracle_sha256=expected_pre_oracle_sha256,
        expected_complete_sha256=expected_pre_oracle_complete_sha256,
        phase=phase,
        registration_hashes=registration_hashes,
        progress_seals=progress_seals,
        evaluator_sha256=evaluator_sha,
        expected_authority=pre_oracle_authority,
    )
    support.atomic_publish_directory(
        destination,
        build,
        lambda stage: _validate_evaluation_stage(
            stage,
            registrations,
            registration_hashes,
            progress_seals,
            expected_authority=evaluation_authority,
            expected_payload=final_payload,
        ),
        trusted_target_root=bundle["target_root"],
    )
    return final_payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--freeze-bundle", type=Path, required=True
        )
        subparser.add_argument("--expected-freeze-complete-sha256", required=True)
        subparser.add_argument("--progress-run", type=Path, required=True)
        subparser.add_argument(
            "--expected-progress-complete-sha256",
            required=True,
            help="Externally recorded SHA-256 of progress COMPLETE.json",
        )
        subparser.add_argument(
            "--expected-confirm-authorization-complete-sha256",
            help=(
                "Externally recorded confirm authorization COMPLETE SHA; "
                "required for confirm progress and forbidden for screen progress"
            ),
        )

    seal = subparsers.add_parser(
        "seal", help="Phase A: publish PRE_ORACLE_SEAL and stop"
    )
    add_common(seal)
    evaluate = subparsers.add_parser(
        "evaluate", help="Phase B: consume an externally anchored preseal"
    )
    add_common(evaluate)
    evaluate.add_argument("--pre-oracle-run", type=Path, required=True)
    evaluate.add_argument("--expected-pre-oracle-seal-sha256", required=True)
    evaluate.add_argument("--expected-pre-oracle-complete-sha256", required=True)
    args = parser.parse_args(argv)
    for name in ("expected_freeze_complete_sha256", "expected_progress_complete_sha256"):
        if not is_sha256(getattr(args, name)):
            parser.error(f"--{name.replace('_', '-')} must be 64 lowercase hex")
    if args.command == "evaluate":
        for name in (
            "expected_pre_oracle_seal_sha256",
            "expected_pre_oracle_complete_sha256",
        ):
            if not is_sha256(getattr(args, name)):
                parser.error(f"--{name.replace('_', '-')} must be 64 lowercase hex")
    authorization = args.expected_confirm_authorization_complete_sha256
    if authorization is not None and not is_sha256(authorization):
        parser.error(
            "--expected-confirm-authorization-complete-sha256 must be 64 lowercase hex"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "seal":
        sealed = publish_pre_oracle_seal(
            freeze_bundle_root=args.freeze_bundle,
            expected_freeze_complete_sha256=args.expected_freeze_complete_sha256,
            progress_root=args.progress_run,
            expected_progress_complete_sha256=(
                args.expected_progress_complete_sha256
            ),
            expected_confirm_authorization_complete_sha256=(
                args.expected_confirm_authorization_complete_sha256
            ),
        )
        print(
            json.dumps(
                {
                    "status": "PRE_ORACLE_SEAL_PUBLISHED_STOP",
                    "oracle_opened_or_hashed": False,
                    "phase": sealed["phase"],
                    "output": str(sealed["root"]),
                    "pre_oracle_seal_sha256": sealed[
                        "pre_oracle_seal_sha256"
                    ],
                    "pre_oracle_complete_sha256": sealed[
                        "pre_oracle_complete_sha256"
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    payload = evaluate_after_published_preseal(
        freeze_bundle_root=args.freeze_bundle,
        expected_freeze_complete_sha256=args.expected_freeze_complete_sha256,
        progress_root=args.progress_run,
        expected_progress_complete_sha256=args.expected_progress_complete_sha256,
        pre_oracle_root=args.pre_oracle_run,
        expected_pre_oracle_sha256=args.expected_pre_oracle_seal_sha256,
        expected_pre_oracle_complete_sha256=(
            args.expected_pre_oracle_complete_sha256
        ),
        expected_confirm_authorization_complete_sha256=(
            args.expected_confirm_authorization_complete_sha256
        ),
    )
    output_root = (
        Path(args.freeze_bundle).expanduser().resolve()
    )
    # The actual path is committed by the evaluation authority, never selected
    # by a caller.  Derive it again only for the printed external anchor.
    bundle = producer.load_and_validate_campaign_bundle(
        args.freeze_bundle,
        args.expected_freeze_complete_sha256,
        expected_source_role="semantic_producer",
    )
    output_root = (
        Path(bundle["target_root"]) / payload["authority"]["output_complete_slot"]
    ).parent
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "claim_scope": payload["claim_scope"],
                "phase": payload["phase"],
                "output_complete_sha256": sha256_file(
                    output_root / COMPLETE_FILENAME
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["decision"] == "PASS" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Gate0ContractError, OSError, ValueError) as error:
        print(f"STOP: {error}", file=sys.stderr)
        raise SystemExit(2) from error
