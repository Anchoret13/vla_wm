#!/usr/bin/env python3
# RETIRED 2026-09-16. Nothing supersedes it: the gate it serves was removed as a
# prerequisite at plan_and_progress/2026-09-12.md:105, and the round used a supplied
# Phi' annotation instead. It has produced zero artifacts - none of
# results/v248_gate0_{registration_bundle,materialized_models,formal_campaign}_v1 exists.
# Blockers never cleared: WORM ledger, runtime fingerprint, registration bundle, and
# non-deterministic cuDNN/TF32 (plan_and_progress/2026-09-02.md:108-129).
# Do not extend, do not import, do not cite as capability. Recoverable at d35e2832.
"""Freeze the four v248 Gate-0 registrations without rendering any RGB.

The default mode is read-only ``--validate-only``.  It computes a canonical
closure over every input and source, proves that a five-file CLIP tokenizer
materialization preserves the three frozen SAM3 prompts, and prints the
closure digest.  A later, explicit ``--freeze`` invocation must provide that
digest via ``--expected-input-closure-sha256``.  Thus any source or artifact
change between review and freeze invalidates the authorization.

The reviewed closure includes an explicit UTC timestamp and a canonical,
initially absent formal target root.  All four screen/confirm registrations
are committed together through an acyclic campaign intent before that target
root may be created.  Model materialization is a separate, non-authorizing
atomic stage that can be reused only through its externally supplied COMPLETE
digest.

This program never imports LIBERO, robosuite, MuJoCo, or LeRobot and never
constructs or resets an environment.  Oracle summaries are hashed as opaque
files; their JSON and success values are never decoded.
"""
from __future__ import annotations

import argparse
import ctypes
import errno
import inspect
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lcwm import v248_gate0_contract as C  # noqa: E402


FREEZE_RESULT_SCHEMA = "v248_gate0_registration_freeze_result_v1"
FREEZE_COMPLETE_SCHEMA = "v248_gate0_registration_freeze_complete_v1"
CAMPAIGN_MANIFEST_SCHEMA = "v248_gate0_campaign_manifest_v1"
CAMPAIGN_PROJECTION_SCHEMA = "v248_gate0_campaign_projection_v1"
CAMPAIGN_INTENT_SCHEMA = "v248_gate0_campaign_intent_v1"
MATERIALIZATION_RESULT_SCHEMA = "v248_gate0_model_materialization_result_v1"
MATERIALIZATION_COMPLETE_SCHEMA = "v248_gate0_model_materialization_complete_v1"
INPUT_CLOSURE_SCHEMA = "v248_gate0_registration_input_closure_v1"
TOKENIZATION_PROOF_SCHEMA = "v248_gate0_clip_tokenization_proof_v1"

MATERIALIZATION_RESULT_FIELDS = frozenset(
    {
        "schema",
        "status",
        "input_closure_sha256",
        "final_root",
        "revisions",
        "regular_hardlinks",
        "tokenization_proof_sha256",
        "producer_sha256",
        "rgb_authorized",
    }
)
MATERIALIZATION_COMPLETE_FIELDS = frozenset(
    {
        "schema",
        "status",
        "atomic_commit",
        "input_closure_sha256",
        "result_sha256",
        "producer_sha256",
        "rgb_authorized",
    }
)
FREEZE_RESULT_FIELDS = frozenset(
    {
        "schema",
        "status",
        "created_utc",
        "input_closure_sha256",
        "output_root",
        "materialized_root",
        "target_root",
        "materialization_complete_sha256",
        "campaign_manifest_sha256",
        "campaign_body_sha256",
        "campaign_id",
        "protocol_sha256",
        "projection_sha256",
        "registrations",
        "materialized_roots",
        "tokenization_proof_sha256",
        "producer_sha256",
        "environment_constructed_or_reset",
        "target_rgb_opened_or_generated",
        "oracle_json_decoded",
        "oracle_success_values_accessed",
    }
)
FREEZE_COMPLETE_FIELDS = frozenset(
    {
        "schema",
        "status",
        "atomic_commit",
        "input_closure_sha256",
        "materialization_complete_sha256",
        "campaign_manifest_sha256",
        "campaign_body_sha256",
        "campaign_id",
        "projection_sha256",
        "result_sha256",
        "producer_sha256",
        "registration_sha256s",
    }
)
CAMPAIGN_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "status",
        "created_utc",
        "campaign_id",
        "protocol_sha256",
        "projection_sha256",
        "projection",
        "screen_episode_ids",
        "confirm_episode_ids",
        "automatic_expansion_rule",
        "confirm_authorization_required",
        "registrations",
        "body_sha256",
    }
)
CAMPAIGN_PROJECTION_FIELDS = frozenset({"schema", "intent", "campaign_id"})
CAMPAIGN_INTENT_FIELDS = frozenset(
    {
        "schema",
        "input_closure_sha256",
        "protocol_sha256",
        "registration_core_sha256s",
        "screen_episode_ids",
        "confirm_episode_ids",
        "automatic_expansion_rule",
        "confirm_authorization_required",
        "target_root",
        "target_output_slots",
        "dag",
    }
)
CAMPAIGN_REGISTRATION_FIELDS = frozenset(
    {
        "file_sha256",
        "self_sha256",
        "core_sha256",
        "panel_id",
        "seed_start",
        "phase",
    }
)
REGISTRATION_SEAL_FIELDS = frozenset({"sha256", "self_sha256"})
REGISTRATION_CAMPAIGN_FIELDS = frozenset(
    {
        "schema",
        "campaign_id",
        "protocol_sha256",
        "projection_sha256",
        "manifest_relative_path",
        "registration_slot",
    }
)

SAM3_PROMPTS = (
    "white woven basket",
    "tomato sauce can",
    "cream cheese box",
)
CLIP_TOKENIZER_MEMBER_PATHS = (
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)

TARGET_OUTPUT_SLOTS = {
    "screen_rgb_complete_by_panel": {
        "chain3_8000": "screen/rgb/chain3_8000/COMPLETE.json",
        "chain3_8100": "screen/rgb/chain3_8100/COMPLETE.json",
    },
    "screen_progress_complete": "screen/progress/COMPLETE.json",
    "screen_pre_oracle_complete": "screen/preoracle/COMPLETE.json",
    "screen_evaluation_complete": "screen/evaluation/COMPLETE.json",
    "confirm_authorization_complete": "confirm/authorization/COMPLETE.json",
    "confirm_rgb_complete_by_panel": {
        "chain3_8000": "confirm/rgb/chain3_8000/COMPLETE.json",
        "chain3_8100": "confirm/rgb/chain3_8100/COMPLETE.json",
    },
    "confirm_progress_complete": "confirm/progress/COMPLETE.json",
    "confirm_pre_oracle_complete": "confirm/preoracle/COMPLETE.json",
    "confirm_evaluation_complete": "confirm/evaluation/COMPLETE.json",
}
CAMPAIGN_DAG = [
    "freeze_complete->screen_rgb",
    "screen_rgb->screen_progress",
    "screen_progress->screen_preoracle",
    "screen_preoracle->screen_evaluation",
    "screen_evaluation->confirm_authorization",
    "confirm_authorization->confirm_rgb",
    "confirm_rgb->confirm_progress",
    "confirm_progress->confirm_preoracle",
    "confirm_preoracle->confirm_evaluation",
]

SANITIZER_RESULT_SCHEMA = "v245_sanitized_deploy_tape_result_v1"
SANITIZER_COMPLETE_SCHEMA = "v245_sanitized_deploy_tape_complete_v1"
SANITIZER_RESULT_FIELDS = frozenset(
    {
        "schema",
        "status",
        "utc",
        "source",
        "field_policy",
        "validation",
        "metadata",
        "tensor_sha256",
        "metadata_sha256",
        "sanitized_tape_sha256",
        "producer_snapshot_sha256",
        "git_head",
        "torch_version",
    }
)
SANITIZER_COMPLETE_FIELDS = frozenset(
    {
        "schema",
        "status",
        "atomic_commit",
        "source_tape_sha256",
        "sanitized_tape_sha256",
        "result_sha256",
        "producer_snapshot_sha256",
    }
)
SANITIZER_SOURCE_FIELDS = frozenset(
    {"path", "sha256", "bytes", "summary_opened"}
)
SANITIZER_VALIDATION_FIELDS = frozenset(
    {"rows", "latent_dim", "episodes", "episode_ids", "episode_row_counts", "c"}
)
SANITIZER_METADATA_FIELDS = frozenset(
    {"task", "c", "latent_dim", "proprio_dim", "proprio_keys"}
)
SANITIZER_FIELD_POLICY_FIELDS = frozenset(
    {"kept", "dropped_names_without_value_access", "declared_outcome_fields"}
)
SANITIZED_TAPE_FIELDS = frozenset(
    {
        "z",
        "u",
        "z_next",
        "episode",
        "t",
        "sigma",
        "task",
        "c",
        "latent_dim",
        "proprio_dim",
        "proprio_keys",
    }
)
SEQUENCE_TENSOR_FIELDS = frozenset({"z", "u", "z_next", "episode", "t", "sigma"})

RUNTIME_RELATIVE_PATHS = {
    "lerobot_policy_config": "lerobot/configs/policies.py",
    "lerobot_policy_factory": "lerobot/policies/factory.py",
    "lerobot_processor": "lerobot/processor/pipeline.py",
    "lerobot_env_factory": "lerobot/envs/factory.py",
    "lerobot_libero_env": "lerobot/envs/libero.py",
    "pi05_configuration": "lerobot/policies/pi05/configuration_pi05.py",
    "pi05_modeling": "lerobot/policies/pi05/modeling_pi05.py",
    "libero_bddl_env": "libero/libero/envs/bddl_base_domain.py",
    "robosuite_environment": "robosuite/environments/base.py",
    "transformers_sam3_video_configuration": (
        "transformers/models/sam3_video/configuration_sam3_video.py"
    ),
    "transformers_sam3_video_modeling": (
        "transformers/models/sam3_video/modeling_sam3_video.py"
    ),
    "transformers_sam3_video_processing": (
        "transformers/models/sam3_video/processing_sam3_video.py"
    ),
    "transformers_sam3_image_processing": (
        "transformers/models/sam3/image_processing_sam3.py"
    ),
    "transformers_sam2_video_processing": (
        "transformers/models/sam2_video/video_processing_sam2_video.py"
    ),
    "transformers_clip_tokenization": (
        "transformers/models/clip/tokenization_clip.py"
    ),
}

DEFAULT_SANITIZED = {
    8000: REPO / "results/v248_gate0_sanitized/chain3_8000_v1",
    8100: REPO / "results/v248_gate0_sanitized/chain3_8100_v1",
}
DEFAULT_HF_CACHE = Path.home() / ".cache/huggingface/hub"
DEFAULT_PI05_CACHE = DEFAULT_HF_CACHE / "models--lerobot--pi05_libero_finetuned"
DEFAULT_PALIGEMMA_CACHE = DEFAULT_HF_CACHE / "models--google--paligemma-3b-pt-224"
DEFAULT_SAM3_CACHE = DEFAULT_HF_CACHE / "models--1038lab--sam3"
DEFAULT_CLIP_CACHE = DEFAULT_HF_CACHE / "models--openai--clip-vit-base-patch32"
DEFAULT_CLIP_SOURCE = Path(
    "/home/stargazer/Desktop/UW_Madison/RoboOPE/rebuttal/ckpts/clip-vit-base-patch32"
)
DEFAULT_CREAM_TEMPLATE_IMAGE = (
    REPO
    / "results/v242_rgb_replay/chain3_lr2_ep0_stride1_v242/images/"
    "ep0000_t0000_eye_in_hand.jpg"
)
DEFAULT_CREAM_TEMPLATE_METADATA = (
    REPO / "results/v245_sam3_open_vocab/ep0_t0_eye_in_hand_v1/result.json"
)


@dataclass(frozen=True)
class SanitizedArtifact:
    root: Path
    registration_ref: dict[str, str]
    source_tape_path: Path
    episode_row_counts: tuple[int, ...]
    input_refs: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class ModelSource:
    repo_id: str
    revision: str
    snapshot_root: Path
    members: tuple[str, ...]
    source_refs: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class FreezeOptions:
    output_root: Path
    materialized_root: Path
    target_root: Path
    sanitized_roots: Mapping[int, Path]
    oracle_paths: Mapping[int, Path | None]
    pi05_cache_root: Path
    paligemma_cache_root: Path
    clip_source_root: Path
    clip_revision: str | None
    sam3_checkpoint: Path | None
    sam3_model_id: str
    sam3_revision: str | None
    cream_template_image: Path
    cream_template_metadata: Path
    semantic_config: Path
    semantic_producer: Path
    semantic_evaluator: Path
    runtime_source_overrides: Mapping[str, Path]
    created_utc: str


@dataclass(frozen=True)
class FreezePlan:
    options: FreezeOptions
    sanitized: Mapping[int, SanitizedArtifact]
    oracles: Mapping[int, dict[str, Any]]
    pi05: ModelSource
    paligemma: ModelSource
    clip_revision: str
    clip_source_refs: tuple[dict[str, Any], ...]
    clip_tree_sha256: str
    tokenization_proof: dict[str, Any]
    sam3_model_id: str
    sam3_revision: str
    sam3_checkpoint: dict[str, Any]
    sam3_config_sha256: str
    sam3_processor_sha256: str
    templates: Mapping[str, dict[str, Any]]
    sources: Mapping[str, dict[str, Any]]
    runtime: Mapping[str, Any]
    freezer_ref: dict[str, Any]
    input_closure: dict[str, Any]
    input_closure_sha256: str


def require(condition: bool, message: str) -> None:
    C.require(condition, message)


def _canonical_absolute(path: Path, where: str, *, must_exist: bool = True) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    resolved = expanded.resolve(strict=must_exist)
    if must_exist:
        require(resolved.exists(), f"{where} is missing: {resolved}")
    return resolved


def _validate_created_utc(value: str) -> str:
    require(
        isinstance(value, str)
        and bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value)),
        "--created-utc must be an explicit UTC second (YYYY-MM-DDTHH:MM:SSZ)",
    )
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise C.Gate0ContractError("--created-utc is not a real UTC timestamp") from error
    require(
        parsed.strftime("%Y-%m-%dT%H:%M:%SZ") == value,
        "--created-utc is not canonical",
    )
    return value


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _flatten_target_slots(value: Mapping[str, Any]) -> tuple[str, ...]:
    paths: list[str] = []
    for key in sorted(value):
        item = value[key]
        if isinstance(item, Mapping):
            paths.extend(_flatten_target_slots(item))
            continue
        require(isinstance(item, str) and item, f"target slot {key} must be a path")
        relative = Path(item)
        require(
            not relative.is_absolute()
            and "." not in relative.parts
            and ".." not in relative.parts
            and relative.as_posix() == item,
            f"target slot {key} is not canonical relative syntax",
        )
        paths.append(item)
    require(len(paths) == len(set(paths)), "target output slots contain duplicate paths")
    return tuple(paths)


def _assert_target_root_absent(target_root: Path) -> None:
    require(
        not os.path.lexists(target_root),
        f"formal target root already exists; refusing post-screen freeze: {target_root}",
    )
    for relative in _flatten_target_slots(TARGET_OUTPUT_SLOTS):
        slot = target_root / relative
        require(
            not os.path.lexists(slot),
            f"formal target output slot already exists: {slot}",
        )


def _assert_target_root_isolated(
    target_root: Path, protected_paths: Sequence[Path]
) -> None:
    for raw_path in protected_paths:
        path = raw_path.expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        canonical = path.resolve(strict=False)
        require(
            not _paths_overlap(target_root, canonical),
            f"formal target root overlaps protected input/output path: {canonical}",
        )


def _strict_json(path: Path, where: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"{where} missing/symlink: {path}")
    value = C.loads_strict_json(path.read_text(encoding="utf-8"), where=where)
    require(isinstance(value, dict), f"{where} must contain an object")
    return value


def _external_ref(path: Path, where: str) -> dict[str, Any]:
    resolved = _canonical_absolute(path, where)
    require(resolved.is_file() and not resolved.is_symlink(), f"{where} must be regular")
    return {
        "path": resolved.as_posix(),
        "sha256": C.sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _opaque_oracle_ref(path: Path, where: str) -> dict[str, Any]:
    """Hash an oracle as bytes without decoding or inspecting any JSON value."""
    return _external_ref(path, where)


def _member_ref(root: Path, relative: str, where: str) -> dict[str, Any]:
    relative_path = Path(relative)
    require(
        not relative_path.is_absolute() and ".." not in relative_path.parts,
        f"{where} has unsafe relative path",
    )
    unresolved = root / relative_path
    require(unresolved.is_file(), f"{where} missing: {unresolved}")
    resolved = unresolved.resolve(strict=True)
    require(resolved.is_file() and not resolved.is_symlink(), f"{where} target is not regular")
    return {
        "relative_path": relative_path.as_posix(),
        "sha256": C.sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _repo_ref(path: Path, where: str) -> dict[str, Any]:
    resolved = _canonical_absolute(path, where)
    require(resolved.is_file() and not resolved.is_symlink(), f"{where} must be regular")
    require(resolved.is_relative_to(REPO), f"{where} must be inside repository")
    return {
        "relative_path": resolved.relative_to(REPO).as_posix(),
        "sha256": C.sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _cache_revision(cache_root: Path, where: str) -> str:
    root = _canonical_absolute(cache_root, where)
    ref = root / "refs/main"
    require(ref.is_file() and not ref.is_symlink(), f"{where} refs/main missing")
    revision = ref.read_text(encoding="utf-8").strip()
    require(
        bool(re.fullmatch(r"[0-9a-f]{40}", revision)),
        f"{where} revision is not 40 lowercase hex",
    )
    return revision


def _model_source(
    cache_root: Path,
    repo_id: str,
    members: Sequence[str],
    where: str,
) -> ModelSource:
    root = _canonical_absolute(cache_root, where)
    revision = _cache_revision(root, where)
    snapshot = root / "snapshots" / revision
    require(snapshot.is_dir(), f"{where} snapshot missing: {snapshot}")
    refs = tuple(_member_ref(snapshot, relative, where) for relative in members)
    return ModelSource(repo_id, revision, snapshot, tuple(members), refs)


def _validate_sanitized(root: Path, seed_start: int) -> SanitizedArtifact:
    artifact_root = _canonical_absolute(root, f"sanitized[{seed_start}]")
    paths = {
        "tape": artifact_root / "sanitized_tape.pt",
        "result": artifact_root / "RESULT.json",
        "complete": artifact_root / "COMPLETE.json",
        "producer": artifact_root / "PRODUCER.py",
    }
    require(
        {path.name for path in artifact_root.iterdir()}
        == {"sanitized_tape.pt", "RESULT.json", "COMPLETE.json", "PRODUCER.py"},
        f"sanitized[{seed_start}] file closure mismatch",
    )
    for role, path in paths.items():
        require(path.is_file() and not path.is_symlink(), f"sanitized {role} missing/symlink")
    refs = {
        role: _external_ref(path, f"sanitized[{seed_start}].{role}")
        for role, path in paths.items()
    }
    result = _strict_json(paths["result"], f"sanitized[{seed_start}] RESULT")
    complete = _strict_json(paths["complete"], f"sanitized[{seed_start}] COMPLETE")
    C.require_exact_keys(result, SANITIZER_RESULT_FIELDS, "sanitizer RESULT")
    C.require_exact_keys(complete, SANITIZER_COMPLETE_FIELDS, "sanitizer COMPLETE")
    require(
        result["schema"] == SANITIZER_RESULT_SCHEMA and result["status"] == "PASS",
        f"sanitized[{seed_start}] RESULT status",
    )
    require(
        complete["schema"] == SANITIZER_COMPLETE_SCHEMA
        and complete["status"] == "atomic_success"
        and complete["atomic_commit"] is True,
        f"sanitized[{seed_start}] COMPLETE status",
    )
    source = C.require_exact_keys(result["source"], SANITIZER_SOURCE_FIELDS, "source")
    policy = C.require_exact_keys(
        result["field_policy"], SANITIZER_FIELD_POLICY_FIELDS, "field_policy"
    )
    validation = C.require_exact_keys(
        result["validation"], SANITIZER_VALIDATION_FIELDS, "validation"
    )
    metadata = C.require_exact_keys(
        result["metadata"], SANITIZER_METADATA_FIELDS, "metadata"
    )
    tensor_hashes = C.require_exact_keys(
        result["tensor_sha256"], SEQUENCE_TENSOR_FIELDS, "tensor_sha256"
    )
    require(source["summary_opened"] is False, "sanitizer opened source summary")
    require(C.is_sha256(source["sha256"]), "sanitizer source tape SHA invalid")
    require(
        type(source["bytes"]) is int and source["bytes"] >= 0,
        "sanitizer source bytes invalid",
    )
    source_tape = Path(source["path"])
    require(source_tape.is_absolute(), "sanitizer source tape path is not absolute")
    require(
        policy["kept"] == sorted(SANITIZED_TAPE_FIELDS)
        and policy["dropped_names_without_value_access"] == ["success"],
        "sanitizer field policy drift",
    )
    expected_metadata = {
        "task": C.TASK,
        "c": C.C,
        "latent_dim": C.LATENT_DIM,
        "proprio_dim": C.PROPRIO_DIM,
        "proprio_keys": list(C.PROPRIO_KEYS),
    }
    require(dict(metadata) == expected_metadata, "sanitizer metadata drift")
    require(
        result["metadata_sha256"] == C.canonical_sha256(expected_metadata),
        "sanitizer metadata seal drift",
    )
    require(
        validation["episodes"] == 96
        and validation["episode_ids"] == list(range(96))
        and validation["latent_dim"] == C.LATENT_DIM
        and validation["c"] == C.C,
        "sanitizer validation contract drift",
    )
    counts = validation["episode_row_counts"]
    require(
        isinstance(counts, list)
        and len(counts) == 96
        and all(type(value) is int and 1 <= value <= 75 for value in counts),
        "sanitizer episode-row counts invalid",
    )
    expected_rows = 7200 if seed_start == 8000 else 7190
    require(validation["rows"] == sum(counts) == expected_rows, "sanitizer row total")
    require(
        counts == ([75] * 96 if seed_start == 8000 else [75] * 77 + [65] + [75] * 18),
        f"sanitized[{seed_start}] episode-row contract drift",
    )
    require(
        all(C.is_sha256(digest) for digest in tensor_hashes.values()),
        "sanitizer tensor digest invalid",
    )
    tape_sha = refs["tape"]["sha256"]
    result_sha = refs["result"]["sha256"]
    producer_sha = refs["producer"]["sha256"]
    require(result["sanitized_tape_sha256"] == tape_sha, "sanitizer RESULT tape seal")
    require(result["producer_snapshot_sha256"] == producer_sha, "sanitizer producer seal")
    require(complete["sanitized_tape_sha256"] == tape_sha, "sanitizer COMPLETE tape seal")
    require(complete["result_sha256"] == result_sha, "sanitizer COMPLETE result seal")
    require(
        complete["producer_snapshot_sha256"] == producer_sha,
        "sanitizer COMPLETE producer seal",
    )
    require(
        complete["source_tape_sha256"] == source["sha256"],
        "sanitizer source tape seal",
    )
    registration_ref = {
        "tape_sha256": tape_sha,
        "result_sha256": result_sha,
        "complete_sha256": refs["complete"]["sha256"],
        "producer_sha256": producer_sha,
        "source_tape_sha256": source["sha256"],
    }
    return SanitizedArtifact(
        artifact_root,
        registration_ref,
        source_tape,
        tuple(counts),
        refs,
    )


def _find_runtime_source(relative: str, where: str) -> Path:
    matches = []
    for raw in sys.path:
        if not raw:
            continue
        candidate = (Path(raw) / relative).resolve()
        if candidate.is_file() and candidate not in matches:
            matches.append(candidate)
    require(len(matches) == 1, f"{where}: expected one runtime source, got {matches}")
    return matches[0]


def _runtime_sources(overrides: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    require(
        set(overrides).issubset(C.RUNTIME_SOURCE_ROLES),
        "unknown runtime-source override role",
    )
    result = {}
    for role in sorted(C.RUNTIME_SOURCE_ROLES):
        path = overrides.get(role)
        if path is None:
            path = _find_runtime_source(RUNTIME_RELATIVE_PATHS[role], role)
        result[role] = _external_ref(path, f"runtime source {role}")
    return result


def _transformer_fingerprints(
    clip_tree_sha256: str,
    runtime_sources: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str]:
    from transformers import (
        CLIPTokenizer,
        Sam2VideoVideoProcessor,
        Sam3ImageProcessor,
        Sam3VideoConfig,
        Sam3VideoModel,
        Sam3VideoProcessor,
    )

    classes = {
        "transformers_sam3_video_configuration": Sam3VideoConfig,
        "transformers_sam3_video_modeling": Sam3VideoModel,
        "transformers_sam3_video_processing": Sam3VideoProcessor,
        "transformers_sam3_image_processing": Sam3ImageProcessor,
        "transformers_sam2_video_processing": Sam2VideoVideoProcessor,
        "transformers_clip_tokenization": CLIPTokenizer,
    }
    for role, class_object in classes.items():
        raw_path = inspect.getsourcefile(class_object)
        require(raw_path is not None, f"cannot inspect runtime source for {role}")
        path = Path(raw_path).resolve()
        registered = runtime_sources[role]
        require(path.as_posix() == registered["path"], f"runtime path drift for {role}")
        require(C.sha256_file(path) == registered["sha256"], f"runtime SHA drift for {role}")
    config = Sam3VideoConfig()
    image_processor = Sam3ImageProcessor()
    video_processor = Sam2VideoVideoProcessor(
        size={"height": 1008, "width": 1008},
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
    )
    return (
        C.sam3_config_sha256(config.to_dict()),
        C.sam3_processor_sha256(
            image_processor.to_dict(),
            video_processor.to_dict(),
            clip_tree_sha256,
        ),
    )


def _tokenizer_signature(tokenizer: Any, prompt: str) -> dict[str, Any]:
    encoded = tokenizer(
        prompt,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_attention_mask=True,
    )
    return {
        "input_ids": [int(value) for value in encoded["input_ids"]],
        "attention_mask": [int(value) for value in encoded["attention_mask"]],
        "encode": [int(value) for value in tokenizer.encode(prompt)],
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token_id": tokenizer.unk_token_id,
        "model_max_length": int(tokenizer.model_max_length),
    }


def prove_clip_tokenization(source_root: Path, materialized_root: Path) -> dict[str, Any]:
    from transformers import CLIPTokenizer

    source = CLIPTokenizer.from_pretrained(source_root, local_files_only=True)
    materialized = CLIPTokenizer.from_pretrained(materialized_root, local_files_only=True)
    source_signatures = {
        prompt: _tokenizer_signature(source, prompt) for prompt in SAM3_PROMPTS
    }
    materialized_signatures = {
        prompt: _tokenizer_signature(materialized, prompt) for prompt in SAM3_PROMPTS
    }
    require(
        source_signatures == materialized_signatures,
        "materialized CLIP tokenizer changes frozen prompt tokenization",
    )
    return {
        "schema": TOKENIZATION_PROOF_SCHEMA,
        "tokenizer_class": "CLIPTokenizer",
        "prompts": list(SAM3_PROMPTS),
        "signatures": source_signatures,
        "exact": True,
    }


def _hardlink_member(source_root: Path, destination_root: Path, relative: str) -> None:
    source_lexical = source_root / relative
    require(source_lexical.is_file(), f"hardlink source missing: {source_lexical}")
    source = source_lexical.resolve(strict=True)
    require(source.is_file() and not source.is_symlink(), "hardlink source target not regular")
    destination = destination_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    require(not os.path.lexists(destination), f"hardlink destination exists: {destination}")
    os.link(source, destination, follow_symlinks=False)
    require(destination.is_file() and not destination.is_symlink(), "materialized file invalid")
    source_stat = source.stat()
    destination_stat = destination.stat()
    require(
        (source_stat.st_dev, source_stat.st_ino)
        == (destination_stat.st_dev, destination_stat.st_ino),
        f"materialized file is not a hardlink: {relative}",
    )


def materialize_snapshot(
    source_root: Path,
    bundle_model_root: Path,
    revision: str,
    members: Sequence[str],
) -> Path:
    snapshot = bundle_model_root / "snapshots" / revision
    require(not os.path.lexists(bundle_model_root), f"model output exists: {bundle_model_root}")
    snapshot.mkdir(parents=True)
    for relative in members:
        _hardlink_member(source_root, snapshot, relative)
    refs = bundle_model_root / "refs"
    refs.mkdir()
    (refs / "main").write_text(revision, encoding="utf-8")
    require(not (refs / "main").is_symlink(), "materialized refs/main is a symlink")
    return snapshot


def _temporary_clip_proof(
    source_root: Path, revision: str, materialized_root: Path
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    scratch_parent = materialized_root.parent
    while not scratch_parent.exists():
        require(scratch_parent != scratch_parent.parent, "no existing scratch parent")
        scratch_parent = scratch_parent.parent
    with tempfile.TemporaryDirectory(
        prefix=".v248-clip-proof-", dir=scratch_parent
    ) as raw:
        temporary = Path(raw) / "clip"
        snapshot = materialize_snapshot(
            source_root,
            temporary,
            revision,
            CLIP_TOKENIZER_MEMBER_PATHS,
        )
        proof = prove_clip_tokenization(source_root, snapshot)
        refs = tuple(
            _member_ref(snapshot, relative, "temporary CLIP tokenizer")
            for relative in CLIP_TOKENIZER_MEMBER_PATHS
        )
    return proof, refs


def _source_refs(
    semantic_producer: Path, semantic_evaluator: Path
) -> dict[str, dict[str, Any]]:
    paths = dict(C.KNOWN_SOURCE_PATHS)
    paths["semantic_producer"] = _repo_ref(
        semantic_producer, "semantic producer"
    )["relative_path"]
    paths["semantic_evaluator"] = _repo_ref(
        semantic_evaluator, "semantic evaluator"
    )["relative_path"]
    require(set(paths) == set(C.SOURCE_ROLES), "registration source-role closure drift")
    return {
        role: _repo_ref(REPO / paths[role], f"source {role}")
        for role in sorted(C.SOURCE_ROLES)
    }


def _resolve_oracle_path(
    override: Path | None, artifact: SanitizedArtifact, seed_start: int
) -> Path:
    if override is not None:
        return _canonical_absolute(override, f"oracle[{seed_start}]")
    return _canonical_absolute(
        artifact.source_tape_path.parent / "summary.json",
        f"oracle[{seed_start}] derived from sanitizer source",
    )


def _sam3_checkpoint(options: FreezeOptions) -> tuple[dict[str, Any], str]:
    revision = options.sam3_revision or _cache_revision(DEFAULT_SAM3_CACHE, "SAM3 cache")
    if options.sam3_checkpoint is None:
        path = DEFAULT_SAM3_CACHE / "snapshots" / revision / "sam3.safetensors"
    else:
        path = options.sam3_checkpoint
    return _external_ref(path.resolve(), "SAM3 checkpoint"), revision


def collect_plan(
    options: FreezeOptions,
    expected_materialization_complete_sha256: str | None = None,
) -> FreezePlan:
    output_root = _canonical_absolute(
        options.output_root, "output root", must_exist=False
    )
    materialized_root = _canonical_absolute(
        options.materialized_root, "materialized root", must_exist=False
    )
    target_root = _canonical_absolute(
        options.target_root, "formal target root", must_exist=False
    )
    created_utc = _validate_created_utc(options.created_utc)
    require(not os.path.lexists(output_root), f"refusing existing output: {output_root}")
    materialized_exists = os.path.lexists(materialized_root)
    if materialized_exists:
        require(
            C.is_sha256(expected_materialization_complete_sha256),
            "existing stage A requires --expected-materialization-complete-sha256",
        )
    else:
        require(
            expected_materialization_complete_sha256 is None,
            "stage-A anchor supplied but materialized root does not exist",
        )
    require(
        not _paths_overlap(output_root, materialized_root),
        "output and materialized roots must not overlap",
    )
    _assert_target_root_absent(target_root)
    early_protected_paths = [
        output_root,
        materialized_root,
        *options.sanitized_roots.values(),
        *(path for path in options.oracle_paths.values() if path is not None),
        options.pi05_cache_root,
        options.paligemma_cache_root,
        options.clip_source_root,
        *(path for path in (options.sam3_checkpoint,) if path is not None),
        options.cream_template_image,
        options.cream_template_metadata,
        options.semantic_config,
        options.semantic_producer,
        options.semantic_evaluator,
        *options.runtime_source_overrides.values(),
    ]
    _assert_target_root_isolated(target_root, early_protected_paths)
    normalized_options = FreezeOptions(
        output_root=output_root,
        materialized_root=materialized_root,
        target_root=target_root,
        sanitized_roots=options.sanitized_roots,
        oracle_paths=options.oracle_paths,
        pi05_cache_root=options.pi05_cache_root,
        paligemma_cache_root=options.paligemma_cache_root,
        clip_source_root=options.clip_source_root,
        clip_revision=options.clip_revision,
        sam3_checkpoint=options.sam3_checkpoint,
        sam3_model_id=options.sam3_model_id,
        sam3_revision=options.sam3_revision,
        cream_template_image=options.cream_template_image,
        cream_template_metadata=options.cream_template_metadata,
        semantic_config=options.semantic_config,
        semantic_producer=options.semantic_producer,
        semantic_evaluator=options.semantic_evaluator,
        runtime_source_overrides=options.runtime_source_overrides,
        created_utc=created_utc,
    )
    sanitized = {
        seed: _validate_sanitized(options.sanitized_roots[seed], seed)
        for seed in (8000, 8100)
    }
    oracles = {
        seed: _opaque_oracle_ref(
            _resolve_oracle_path(options.oracle_paths.get(seed), sanitized[seed], seed),
            f"oracle[{seed}]",
        )
        for seed in (8000, 8100)
    }
    pi05 = _model_source(
        options.pi05_cache_root,
        "lerobot/pi05_libero_finetuned",
        tuple(C.PI05_MEMBER_PATHS.values()),
        "Pi05 cache",
    )
    paligemma = _model_source(
        options.paligemma_cache_root,
        "google/paligemma-3b-pt-224",
        tuple(sorted(C.TOKENIZER_MEMBER_PATHS)),
        "PaliGemma tokenizer cache",
    )
    clip_source = _canonical_absolute(options.clip_source_root, "CLIP tokenizer source")
    require(clip_source.is_dir(), "CLIP tokenizer source must be a directory")
    clip_revision = options.clip_revision or _cache_revision(
        DEFAULT_CLIP_CACHE, "CLIP tokenizer cache"
    )
    require(
        bool(re.fullmatch(r"[0-9a-f]{40}", clip_revision)),
        "CLIP tokenizer revision must be 40 lowercase hex",
    )
    clip_source_refs = tuple(
        _member_ref(clip_source, relative, "CLIP tokenizer source")
        for relative in CLIP_TOKENIZER_MEMBER_PATHS
    )
    tokenization_proof, proof_refs = _temporary_clip_proof(
        clip_source, clip_revision, materialized_root
    )
    require(
        list(clip_source_refs) == list(proof_refs),
        "temporary CLIP hardlinks changed file inventory",
    )
    clip_tree_sha = C.file_tree_sha256(clip_source_refs)
    runtime_sources = _runtime_sources(options.runtime_source_overrides)
    config_sha, processor_sha = _transformer_fingerprints(
        clip_tree_sha, runtime_sources
    )
    checkpoint, sam3_revision = _sam3_checkpoint(options)
    templates = {
        "cream_template_image": _external_ref(
            options.cream_template_image, "cream template image"
        ),
        "cream_template_metadata": _external_ref(
            options.cream_template_metadata, "cream template metadata"
        ),
        "semantic_config": _external_ref(options.semantic_config, "semantic config"),
    }
    sources = _source_refs(options.semantic_producer, options.semantic_evaluator)
    for seed in (8000, 8100):
        require(
            sanitized[seed].registration_ref["producer_sha256"]
            == sources["sanitizer"]["sha256"],
            f"sanitized[{seed}] producer snapshot is not the registered sanitizer",
        )
    runtime = C.current_runtime_fingerprint(device="cuda")
    runtime["runtime_sources"] = runtime_sources
    freezer_ref = _repo_ref(Path(__file__), "registration freezer")
    protected_paths = [
        output_root,
        materialized_root,
        *(artifact.root for artifact in sanitized.values()),
        *(artifact.source_tape_path for artifact in sanitized.values()),
        *(Path(reference["path"]) for reference in oracles.values()),
        pi05.snapshot_root,
        paligemma.snapshot_root,
        clip_source,
        Path(checkpoint["path"]),
        *(Path(reference["path"]) for reference in templates.values()),
        *(REPO / reference["relative_path"] for reference in sources.values()),
        *(Path(reference["path"]) for reference in runtime_sources.values()),
    ]
    _assert_target_root_isolated(target_root, protected_paths)
    input_closure = {
        "schema": INPUT_CLOSURE_SCHEMA,
        "created_utc": created_utc,
        "output_root": output_root.as_posix(),
        "materialized_root": materialized_root.as_posix(),
        "target_root": target_root.as_posix(),
        "target_output_slots": TARGET_OUTPUT_SLOTS,
        "sanitized": {
            str(seed): {
                "registration_ref": sanitized[seed].registration_ref,
                "input_refs": sanitized[seed].input_refs,
            }
            for seed in (8000, 8100)
        },
        "oracles": {str(seed): oracles[seed] for seed in (8000, 8100)},
        "model_sources": {
            "pi05": {
                "repo_id": pi05.repo_id,
                "revision": pi05.revision,
                "root": pi05.snapshot_root.as_posix(),
                "files": list(pi05.source_refs),
            },
            "paligemma": {
                "repo_id": paligemma.repo_id,
                "revision": paligemma.revision,
                "root": paligemma.snapshot_root.as_posix(),
                "files": list(paligemma.source_refs),
            },
            "clip": {
                "revision": clip_revision,
                "root": clip_source.as_posix(),
                "tree_sha256": clip_tree_sha,
                "files": list(clip_source_refs),
                "tokenization_proof": tokenization_proof,
            },
            "sam3": {
                "model_id": options.sam3_model_id,
                "revision": sam3_revision,
                "checkpoint": checkpoint,
                "config_sha256": config_sha,
                "processor_sha256": processor_sha,
            },
        },
        "templates": templates,
        "sources": sources,
        "runtime": runtime,
        "freezer": freezer_ref,
    }
    closure_sha = C.canonical_sha256(input_closure)
    plan = FreezePlan(
        normalized_options,
        sanitized,
        oracles,
        pi05,
        paligemma,
        clip_revision,
        clip_source_refs,
        clip_tree_sha,
        tokenization_proof,
        options.sam3_model_id,
        sam3_revision,
        checkpoint,
        config_sha,
        processor_sha,
        templates,
        sources,
        runtime,
        freezer_ref,
        input_closure,
        closure_sha,
    )
    if materialized_exists:
        materialization_complete = materialized_root / "MATERIALIZATION_COMPLETE.json"
        require(
            materialization_complete.is_file()
            and not materialization_complete.is_symlink(),
            "existing stage A lacks a regular MATERIALIZATION_COMPLETE.json",
        )
        require(
            C.sha256_file(materialization_complete)
            == expected_materialization_complete_sha256,
            "existing stage-A COMPLETE does not equal the external anchor",
        )
        _validate_materialized_stage(materialized_root, plan)
    return plan


def _episodes(seed_start: int, phase: str) -> list[dict[str, Any]]:
    episode_ids = C.SCREEN_EPISODES[seed_start] if phase == "screen" else range(96)
    return [
        {
            "episode_id": episode_id,
            "env_seed": seed_start + episode_id,
            "stored_rows": 65 if seed_start == 8100 and episode_id == 77 else 75,
            "mode": (
                "recover_last_action"
                if seed_start == 8100 and episode_id == 77
                else "ordinary"
            ),
        }
        for episode_id in episode_ids
    ]


def _recovery(seed_start: int, counts: Sequence[int]) -> dict[str, Any]:
    if seed_start == 8000:
        return {
            "enabled": False,
            "episode_id": None,
            "env_seed": None,
            "stored_chunks": None,
            "stored_steps": None,
            "source_terminal_t": None,
            "missing_chunk_t": None,
            "missing_raw_sigma_choice": None,
            "sigma_rng": None,
            "sampler": None,
            "phi_only": True,
            "dynamics_transition_eligible": False,
        }
    proposed = list(counts[:78])
    require(len(proposed) == 78 and proposed[77] == 65, "ep77 source row counts")
    proposed[77] = 66
    return {
        "enabled": True,
        "episode_id": 77,
        "env_seed": 8177,
        "stored_chunks": 65,
        "stored_steps": 650,
        "source_terminal_t": 651,
        "missing_chunk_t": 650,
        "missing_raw_sigma_choice": 0.6,
        "sigma_rng": {
            "algorithm": "numpy.random.Generator(PCG64)",
            "seed": 8100,
            "choices": [0.0, 0.6, 1.5, 3.0],
            "source_proposed_chunks_by_episode": proposed,
            "missing_draw_offset_within_episode": 65,
        },
        "sampler": {
            "zero_sigma_path": "Pi05Runner.sample_chunk",
            "positive_sigma_path": "lcwm.sampler.sample_chunks",
            "seed_rule": "env_seed*7919+t",
            "n": 1,
            "normalized_dtype": "torch.float32",
            "exact_comparator": "torch.equal",
            "prerequisite_chunks": 65,
            "missing_chunk_actions": 10,
            "required_halt_action_offset": 0,
            "required_halt_t": 651,
        },
        "phi_only": True,
        "dynamics_transition_eligible": False,
    }


def _models(
    plan: FreezePlan,
    pi05_root: Path,
    paligemma_root: Path,
    clip_root: Path,
) -> dict[str, Any]:
    pi_refs = {item["relative_path"]: item for item in plan.pi05.source_refs}
    return {
        "pi05": {
            "repo_id": plan.pi05.repo_id,
            "revision": plan.pi05.revision,
            "local_root": pi05_root.resolve().as_posix(),
            **{
                role: pi_refs[relative]
                for role, relative in C.PI05_MEMBER_PATHS.items()
            },
        },
        "tokenizer": {
            "repo_id": plan.paligemma.repo_id,
            "revision": plan.paligemma.revision,
            "local_root": paligemma_root.resolve().as_posix(),
            "files": list(plan.paligemma.source_refs),
        },
        "sam3": {
            "model_id": plan.sam3_model_id,
            "revision": plan.sam3_revision,
            "checkpoint": plan.sam3_checkpoint,
            "clip_tokenizer": {
                "root": clip_root.resolve().as_posix(),
                "tree_sha256": plan.clip_tree_sha256,
                "files": list(plan.clip_source_refs),
            },
            "config_sha256": plan.sam3_config_sha256,
            "processor_sha256": plan.sam3_processor_sha256,
        },
    }


def make_registration_core(
    plan: FreezePlan,
    seed_start: int,
    phase: str,
    models: Mapping[str, Any],
    created_utc: str,
) -> dict[str, Any]:
    return {
        "schema": C.REGISTRATION_SCHEMA,
        "status": C.REGISTRATION_STATUS,
        "registration_id": f"v248_gate0_{C.PANEL_IDS[seed_start]}_{phase}_v2",
        "created_utc": created_utc,
        "phase": phase,
        "panel": {
            "panel_id": C.PANEL_IDS[seed_start],
            "seed_start": seed_start,
            "source_episode_count": 96,
            "episodes": _episodes(seed_start, phase),
        },
        "replay": {
            "task": C.TASK,
            "c": C.C,
            "latent_layout": C.LATENT_LAYOUT,
            "latent_dim": C.LATENT_DIM,
            "proprio_dim": C.PROPRIO_DIM,
            "proprio_keys": list(C.PROPRIO_KEYS),
            "cameras": C.CAMERAS,
            "jpeg_quality": 95,
            "orientation": "rotate_180_to_lerobot_policy_view",
            "frame_roles": list(C.FRAME_ROLES),
            "two_fresh_recovery_replays": True,
        },
        "terminal_recovery": _recovery(
            seed_start, plan.sanitized[seed_start].episode_row_counts
        ),
        "sanitized_source": plan.sanitized[seed_start].registration_ref,
        "oracle": {
            "summary": plan.oracles[seed_start],
            "access": "post_progress_seal_evaluator_only",
            "fields": ["episode_records.idx", "seed", "steps", "events"],
            "success_values_accessed": False,
        },
        "models": dict(models),
        "templates": dict(plan.templates),
        "sources": dict(plan.sources),
        "runtime": dict(plan.runtime),
        "semantic_contract": {
            "manifest_schema": C.RGB_MANIFEST_SCHEMA,
            "result_schema": C.RGB_RESULT_SCHEMA,
            "complete_schema": C.RGB_COMPLETE_SCHEMA,
            "allowed_frame_roles": list(C.FRAME_ROLES),
            "recovered_endpoint": {
                "phi_only": True,
                "dynamics_transition_eligible": False,
            },
        },
    }


def make_registrations(
    plan: FreezePlan,
    models: Mapping[str, Any],
    created_utc: str,
) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
    cores = {
        (seed, phase): make_registration_core(
            plan, seed, phase, models, created_utc
        )
        for phase in ("screen", "confirm")
        for seed in (8000, 8100)
    }
    protocols = [C.campaign_protocol_payload(core) for core in cores.values()]
    require(
        protocols and all(protocol == protocols[0] for protocol in protocols),
        "four registration cores do not share one protocol",
    )
    protocol = protocols[0]
    protocol_sha = C.canonical_sha256(protocol)
    core_sha256s = {
        _registration_filename(seed, phase): C.canonical_sha256(core)
        for (seed, phase), core in cores.items()
    }
    intent = {
        "schema": CAMPAIGN_INTENT_SCHEMA,
        "input_closure_sha256": plan.input_closure_sha256,
        "protocol_sha256": protocol_sha,
        "registration_core_sha256s": core_sha256s,
        "screen_episode_ids": protocol["screen_episode_ids"],
        "confirm_episode_ids": protocol["confirm_episode_ids"],
        "automatic_expansion_rule": C.AUTOMATIC_CONFIRM_EXPANSION_RULE,
        "confirm_authorization_required": True,
        "target_root": plan.options.target_root.as_posix(),
        "target_output_slots": TARGET_OUTPUT_SLOTS,
        "dag": CAMPAIGN_DAG,
    }
    anchor_sha = C.canonical_sha256(intent)
    campaign_id = f"v248-gate0-{anchor_sha[:16]}"
    projection = {
        "schema": CAMPAIGN_PROJECTION_SCHEMA,
        "intent": intent,
        "campaign_id": campaign_id,
    }
    projection_sha = C.canonical_sha256(projection)
    registrations = {}
    for key, core in cores.items():
        seed, phase = key
        campaign = {
            "schema": C.CAMPAIGN_SCHEMA,
            "campaign_id": campaign_id,
            "protocol_sha256": protocol_sha,
            "projection_sha256": projection_sha,
            "manifest_relative_path": "CAMPAIGN.json",
            "registration_slot": _registration_filename(seed, phase),
        }
        registration = {**core, "campaign": campaign}
        registration["self_sha256"] = C.canonical_sha256(registration)
        registrations[key] = registration
    return registrations, projection


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    require(renameat2 is not None, "renameat2 unavailable; refusing unsafe fallback")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(destination)
        raise OSError(error_number, os.strerror(error_number), destination)


def _write_bytes(path: Path, payload: bytes) -> None:
    require(not os.path.lexists(path), f"refusing to overwrite staged file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _registration_filename(seed_start: int, phase: str) -> str:
    return f"{C.PANEL_IDS[seed_start]}_{phase}.json"


def _registration_core_payload(registration: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in registration.items()
        if key not in {"campaign", "self_sha256"}
    }


def _validate_projection(
    projection: Mapping[str, Any],
    registrations: Mapping[tuple[int, str], Mapping[str, Any]],
) -> Mapping[str, Any]:
    value = C.require_exact_keys(
        projection, CAMPAIGN_PROJECTION_FIELDS, "campaign projection"
    )
    require(
        value["schema"] == CAMPAIGN_PROJECTION_SCHEMA,
        "campaign projection schema mismatch",
    )
    intent = C.require_exact_keys(
        value["intent"], CAMPAIGN_INTENT_FIELDS, "campaign intent"
    )
    require(intent["schema"] == CAMPAIGN_INTENT_SCHEMA, "campaign intent schema")
    require(C.is_sha256(intent["input_closure_sha256"]), "campaign input closure SHA")
    require(C.is_sha256(intent["protocol_sha256"]), "campaign protocol SHA")
    expected_names = {
        _registration_filename(seed, phase)
        for phase in ("screen", "confirm")
        for seed in (8000, 8100)
    }
    require(
        set(intent["registration_core_sha256s"]) == expected_names,
        "campaign core slot closure mismatch",
    )
    for (seed, phase), registration in registrations.items():
        filename = _registration_filename(seed, phase)
        require(
            intent["registration_core_sha256s"][filename]
            == C.canonical_sha256(_registration_core_payload(registration)),
            f"campaign core SHA mismatch: {filename}",
        )
    protocols = [
        C.campaign_protocol_payload(registration)
        for registration in registrations.values()
    ]
    require(
        protocols and all(protocol == protocols[0] for protocol in protocols),
        "campaign registration protocol drift",
    )
    protocol = protocols[0]
    require(
        intent["protocol_sha256"] == C.canonical_sha256(protocol),
        "campaign protocol digest mismatch",
    )
    require(
        intent["screen_episode_ids"] == protocol["screen_episode_ids"]
        and intent["confirm_episode_ids"] == protocol["confirm_episode_ids"],
        "campaign episode schedules drift",
    )
    require(
        intent["automatic_expansion_rule"] == C.AUTOMATIC_CONFIRM_EXPANSION_RULE,
        "campaign automatic expansion rule drift",
    )
    require(
        intent["confirm_authorization_required"] is True,
        "campaign must require confirm authorization",
    )
    require(intent["target_output_slots"] == TARGET_OUTPUT_SLOTS, "target slots drift")
    require(intent["dag"] == CAMPAIGN_DAG, "campaign DAG drift")
    target_root_text = intent["target_root"]
    require(
        isinstance(target_root_text, str)
        and Path(target_root_text).is_absolute()
        and Path(target_root_text).as_posix() == target_root_text,
        "campaign target root syntax mismatch",
    )
    anchor_sha = C.canonical_sha256(dict(intent))
    require(
        value["campaign_id"] == f"v248-gate0-{anchor_sha[:16]}",
        "campaign ID does not bind its full intent",
    )
    return intent


def _campaign_manifest(
    registrations: Mapping[tuple[int, str], Mapping[str, Any]],
    registration_seals: Mapping[str, Mapping[str, str]],
    projection: Mapping[str, Any],
    created_utc: str,
) -> dict[str, Any]:
    expected_names = {
        _registration_filename(seed, phase)
        for phase in ("screen", "confirm")
        for seed in (8000, 8100)
    }
    require(
        set(registration_seals) == expected_names,
        "campaign registration seal closure mismatch",
    )
    for filename, seal in registration_seals.items():
        C.require_exact_keys(seal, REGISTRATION_SEAL_FIELDS, f"seal {filename}")
        require(
            C.is_sha256(seal["sha256"]) and C.is_sha256(seal["self_sha256"]),
            f"invalid registration seal: {filename}",
        )
    campaigns = [registration["campaign"] for registration in registrations.values()]
    require(campaigns, "campaign registrations are empty")
    for (seed, phase), registration in registrations.items():
        filename = _registration_filename(seed, phase)
        campaign_value = C.require_exact_keys(
            registration["campaign"],
            REGISTRATION_CAMPAIGN_FIELDS,
            f"registration campaign {filename}",
        )
        require(
            campaign_value["manifest_relative_path"] == "CAMPAIGN.json"
            and campaign_value["registration_slot"] == filename,
            f"registration campaign slot drift: {filename}",
        )
    campaign_ids = {item["campaign_id"] for item in campaigns}
    protocol_shas = {item["protocol_sha256"] for item in campaigns}
    projection_shas = {item["projection_sha256"] for item in campaigns}
    require(
        len(campaign_ids) == len(protocol_shas) == len(projection_shas) == 1,
        "campaign identity drift",
    )
    campaign_id = next(iter(campaign_ids))
    protocol_sha = next(iter(protocol_shas))
    projection_sha = next(iter(projection_shas))
    require(
        C.canonical_sha256(projection) == projection_sha,
        "campaign projection SHA drift",
    )
    intent = _validate_projection(projection, registrations)
    require(projection["campaign_id"] == campaign_id, "projection campaign ID drift")
    require(intent["protocol_sha256"] == protocol_sha, "projection protocol drift")
    body = {
        "schema": CAMPAIGN_MANIFEST_SCHEMA,
        "status": "frozen_before_any_formal_rgb",
        "created_utc": created_utc,
        "campaign_id": campaign_id,
        "protocol_sha256": protocol_sha,
        "projection_sha256": projection_sha,
        "projection": dict(projection),
        "screen_episode_ids": intent["screen_episode_ids"],
        "confirm_episode_ids": intent["confirm_episode_ids"],
        "automatic_expansion_rule": intent["automatic_expansion_rule"],
        "confirm_authorization_required": True,
        "registrations": {
            _registration_filename(seed, phase): {
                "file_sha256": registration_seals[
                    _registration_filename(seed, phase)
                ]["sha256"],
                "self_sha256": registration["self_sha256"],
                "core_sha256": intent["registration_core_sha256s"][
                    _registration_filename(seed, phase)
                ],
                "panel_id": registration["panel"]["panel_id"],
                "seed_start": seed,
                "phase": phase,
            }
            for (seed, phase), registration in registrations.items()
        },
    }
    return {**body, "body_sha256": C.canonical_sha256(body)}


def _atomic_publish_directory(
    destination: Path,
    builder: Any,
    validator: Any,
) -> Path:
    require(not os.path.lexists(destination), f"refusing to overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.partial-", dir=destination.parent
        )
    )
    try:
        builder(stage)
        validator(stage)
        require(not os.path.lexists(destination), f"output appeared: {destination}")
        _fsync_directory(stage)
        _rename_noreplace(stage, destination)
        _fsync_directory(destination.parent)
        return destination
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        raise


def _materialized_roots(root: Path, plan: FreezePlan) -> dict[str, Path]:
    return {
        "pi05": root / "pi05/snapshots" / plan.pi05.revision,
        "paligemma": root / "paligemma/snapshots" / plan.paligemma.revision,
        "clip": root / "clip_tokenizer/snapshots" / plan.clip_revision,
    }


def _validate_materialized_stage(stage: Path, plan: FreezePlan) -> None:
    require(
        {path.name for path in stage.iterdir()}
        == {
            "pi05",
            "paligemma",
            "clip_tokenizer",
            "PRODUCER.py",
            "MATERIALIZATION.json",
            "MATERIALIZATION_COMPLETE.json",
        },
        "materialized bundle root closure mismatch",
    )
    require(
        not any(path.is_symlink() for path in stage.rglob("*")),
        "materialized bundle contains a symlink",
    )
    roots = _materialized_roots(stage, plan)
    source_by_role = {
        "pi05": plan.pi05.snapshot_root,
        "paligemma": plan.paligemma.snapshot_root,
        "clip": plan.options.clip_source_root,
    }
    members_by_role = {
        "pi05": plan.pi05.members,
        "paligemma": plan.paligemma.members,
        "clip": CLIP_TOKENIZER_MEMBER_PATHS,
    }
    for role, destination_root in roots.items():
        source_root = source_by_role[role]
        expected_members = set(members_by_role[role])
        expected_directories = {
            parent.as_posix()
            for relative in expected_members
            for parent in Path(relative).parents
            if parent != Path(".")
        }
        observed_files = {
            path.relative_to(destination_root).as_posix()
            for path in destination_root.rglob("*")
            if path.is_file()
        }
        observed_directories = {
            path.relative_to(destination_root).as_posix()
            for path in destination_root.rglob("*")
            if path.is_dir()
        }
        require(
            observed_files == expected_members
            and observed_directories == expected_directories,
            f"materialized {role} snapshot inventory drift",
        )
        model_root = destination_root.parent.parent
        require(
            {path.name for path in model_root.iterdir()} == {"snapshots", "refs"},
            f"materialized {role} root inventory drift",
        )
        require(
            {path.name for path in (model_root / "snapshots").iterdir()}
            == {destination_root.name},
            f"materialized {role} revision inventory drift",
        )
        require(
            {path.name for path in (model_root / "refs").iterdir()} == {"main"},
            f"materialized {role} refs inventory drift",
        )
        require(
            (model_root / "refs/main").read_text(encoding="utf-8")
            == destination_root.name,
            f"materialized {role} main ref drift",
        )
        for relative in members_by_role[role]:
            source = (source_root / relative).resolve(strict=True)
            destination = destination_root / relative
            require(
                destination.is_file() and not destination.is_symlink(),
                f"materialized {role}/{relative} is not a regular file",
            )
            require(
                (source.stat().st_dev, source.stat().st_ino)
                == (destination.stat().st_dev, destination.stat().st_ino),
                f"materialized {role}/{relative} is not a source hardlink",
            )
    proof = prove_clip_tokenization(plan.options.clip_source_root, roots["clip"])
    require(proof == plan.tokenization_proof, "materialized CLIP proof drift")
    clip_tree = {
        "root": roots["clip"].resolve().as_posix(),
        "tree_sha256": plan.clip_tree_sha256,
        "files": list(plan.clip_source_refs),
    }
    C.validate_file_tree(clip_tree, "materialized CLIP tokenizer")
    producer = stage / "PRODUCER.py"
    result = _strict_json(stage / "MATERIALIZATION.json", "materialization RESULT")
    complete = _strict_json(
        stage / "MATERIALIZATION_COMPLETE.json", "materialization COMPLETE"
    )
    C.require_exact_keys(result, MATERIALIZATION_RESULT_FIELDS, "materialization RESULT")
    C.require_exact_keys(
        complete, MATERIALIZATION_COMPLETE_FIELDS, "materialization COMPLETE"
    )
    require(
        C.sha256_file(producer) == plan.freezer_ref["sha256"],
        "materialization producer snapshot drift",
    )
    require(result["schema"] == MATERIALIZATION_RESULT_SCHEMA, "materialization schema")
    require(
        result["status"] == "materialized_only_not_rgb_authorization",
        "materialization status",
    )
    require(
        result["input_closure_sha256"] == plan.input_closure_sha256
        and result["final_root"] == plan.options.materialized_root.as_posix(),
        "materialization plan binding drift",
    )
    require(
        result["revisions"]
        == {
            "pi05": plan.pi05.revision,
            "paligemma": plan.paligemma.revision,
            "clip_tokenizer": plan.clip_revision,
        },
        "materialization revision binding drift",
    )
    require(
        result["regular_hardlinks"]
        == {
            "pi05": len(plan.pi05.members),
            "paligemma": len(plan.paligemma.members),
            "clip_tokenizer": len(CLIP_TOKENIZER_MEMBER_PATHS),
        },
        "materialization hardlink-count drift",
    )
    require(
        result["tokenization_proof_sha256"]
        == C.canonical_sha256(plan.tokenization_proof),
        "materialization tokenization proof seal",
    )
    require(
        result["producer_sha256"] == plan.freezer_ref["sha256"]
        and result["rgb_authorized"] is False,
        "materialization authority boundary drift",
    )
    require(
        complete["schema"] == MATERIALIZATION_COMPLETE_SCHEMA,
        "materialization COMPLETE schema",
    )
    require(
        complete["status"] == "materialized_only_not_rgb_authorization"
        and complete["atomic_commit"] is True,
        "materialization COMPLETE status",
    )
    require(
        complete["input_closure_sha256"] == plan.input_closure_sha256
        and complete["rgb_authorized"] is False,
        "materialization COMPLETE authority boundary drift",
    )
    require(
        complete["result_sha256"] == C.sha256_file(stage / "MATERIALIZATION.json"),
        "materialization result seal",
    )
    require(
        complete["producer_sha256"] == C.sha256_file(producer),
        "materialization producer seal",
    )


def _publish_materialized_models(plan: FreezePlan) -> Path:
    destination = plan.options.materialized_root

    def build(stage: Path) -> None:
        roots = {
            "pi05": materialize_snapshot(
                plan.pi05.snapshot_root,
                stage / "pi05",
                plan.pi05.revision,
                plan.pi05.members,
            ),
            "paligemma": materialize_snapshot(
                plan.paligemma.snapshot_root,
                stage / "paligemma",
                plan.paligemma.revision,
                plan.paligemma.members,
            ),
            "clip": materialize_snapshot(
                plan.options.clip_source_root,
                stage / "clip_tokenizer",
                plan.clip_revision,
                CLIP_TOKENIZER_MEMBER_PATHS,
            ),
        }
        proof = prove_clip_tokenization(plan.options.clip_source_root, roots["clip"])
        require(proof == plan.tokenization_proof, "CLIP tokenization changed in stage A")
        _write_bytes(stage / "PRODUCER.py", Path(__file__).read_bytes())
        require(
            C.sha256_file(stage / "PRODUCER.py") == plan.freezer_ref["sha256"],
            "freezer changed after input closure was computed",
        )
        result = {
            "schema": MATERIALIZATION_RESULT_SCHEMA,
            "status": "materialized_only_not_rgb_authorization",
            "input_closure_sha256": plan.input_closure_sha256,
            "final_root": destination.as_posix(),
            "revisions": {
                "pi05": plan.pi05.revision,
                "paligemma": plan.paligemma.revision,
                "clip_tokenizer": plan.clip_revision,
            },
            "regular_hardlinks": {
                "pi05": len(plan.pi05.members),
                "paligemma": len(plan.paligemma.members),
                "clip_tokenizer": len(CLIP_TOKENIZER_MEMBER_PATHS),
            },
            "tokenization_proof_sha256": C.canonical_sha256(plan.tokenization_proof),
            "producer_sha256": plan.freezer_ref["sha256"],
            "rgb_authorized": False,
        }
        _write_bytes(stage / "MATERIALIZATION.json", C.canonical_bytes(result))
        complete = {
            "schema": MATERIALIZATION_COMPLETE_SCHEMA,
            "status": "materialized_only_not_rgb_authorization",
            "atomic_commit": True,
            "input_closure_sha256": plan.input_closure_sha256,
            "result_sha256": C.sha256_file(stage / "MATERIALIZATION.json"),
            "producer_sha256": C.sha256_file(stage / "PRODUCER.py"),
            "rgb_authorized": False,
        }
        _write_bytes(
            stage / "MATERIALIZATION_COMPLETE.json", C.canonical_bytes(complete)
        )

    return _atomic_publish_directory(
        destination,
        build,
        lambda stage: _validate_materialized_stage(stage, plan),
    )


def _revalidate_non_model_inputs(plan: FreezePlan) -> None:
    for seed in (8000, 8100):
        current = _validate_sanitized(plan.sanitized[seed].root, seed)
        require(current == plan.sanitized[seed], f"sanitized[{seed}] changed before stage B")
        current_oracle = _opaque_oracle_ref(
            Path(plan.oracles[seed]["path"]), f"oracle[{seed}] revalidation"
        )
        require(current_oracle == plan.oracles[seed], f"oracle[{seed}] changed")
    for role, reference in plan.templates.items():
        current = _external_ref(Path(reference["path"]), f"template {role} revalidation")
        require(current == reference, f"template {role} changed")
    require(
        _repo_ref(Path(__file__), "registration freezer") == plan.freezer_ref,
        "registration freezer changed",
    )


def _validate_registration_stage(
    stage: Path,
    plan: FreezePlan,
    registrations: Mapping[tuple[int, str], Mapping[str, Any]],
    projection: Mapping[str, Any],
    registration_seals: Mapping[str, Mapping[str, str]],
    materialization_complete_sha256: str,
) -> None:
    require(
        {path.name for path in stage.iterdir()}
        == {
            "registrations",
            "PRODUCER.py",
            "CAMPAIGN.json",
            "FREEZE_RESULT.json",
            "COMPLETE.json",
        },
        "registration bundle root closure mismatch",
    )
    require(
        not any(path.is_symlink() for path in stage.rglob("*")),
        "registration bundle contains a symlink",
    )
    expected_names = {
        _registration_filename(seed, phase)
        for seed in (8000, 8100)
        for phase in ("screen", "confirm")
    }
    require(
        {path.name for path in (stage / "registrations").iterdir()} == expected_names,
        "registration file closure mismatch",
    )
    actual_registration_seals = {}
    for (seed, phase), expected in registrations.items():
        path = stage / "registrations" / _registration_filename(seed, phase)
        loaded, file_sha = C.load_and_validate_registration(
            path, expected_source_role="formal_replay"
        )
        require(loaded == expected, f"registration value drift: {path.name}")
        actual_registration_seals[path.name] = {
            "sha256": file_sha,
            "self_sha256": loaded["self_sha256"],
        }
    require(
        registration_seals == actual_registration_seals,
        "registration seal mapping drift",
    )
    result_path = stage / "FREEZE_RESULT.json"
    producer_path = stage / "PRODUCER.py"
    campaign_path = stage / "CAMPAIGN.json"
    result = _strict_json(result_path, "freeze RESULT")
    complete = _strict_json(stage / "COMPLETE.json", "freeze COMPLETE")
    campaign = _strict_json(campaign_path, "campaign manifest")
    C.require_exact_keys(result, FREEZE_RESULT_FIELDS, "freeze RESULT")
    C.require_exact_keys(complete, FREEZE_COMPLETE_FIELDS, "freeze COMPLETE")
    C.require_exact_keys(campaign, CAMPAIGN_MANIFEST_FIELDS, "campaign manifest")
    campaign_body = {
        key: value for key, value in campaign.items() if key != "body_sha256"
    }
    require(
        C.canonical_sha256(campaign_body) == campaign["body_sha256"],
        "campaign manifest body seal",
    )
    expected_campaign = _campaign_manifest(
        registrations,
        actual_registration_seals,
        projection,
        plan.options.created_utc,
    )
    require(campaign == expected_campaign, "campaign manifest content drift")
    intent = _validate_projection(projection, registrations)
    require(
        intent["input_closure_sha256"] == plan.input_closure_sha256
        and intent["target_root"] == plan.options.target_root.as_posix(),
        "campaign intent plan binding drift",
    )
    roots = _materialized_roots(plan.options.materialized_root, plan)
    models = _models(plan, roots["pi05"], roots["paligemma"], roots["clip"])
    expected_result = {
        "schema": FREEZE_RESULT_SCHEMA,
        "status": "PASS",
        "created_utc": plan.options.created_utc,
        "input_closure_sha256": plan.input_closure_sha256,
        "output_root": plan.options.output_root.as_posix(),
        "materialized_root": plan.options.materialized_root.as_posix(),
        "target_root": plan.options.target_root.as_posix(),
        "materialization_complete_sha256": materialization_complete_sha256,
        "campaign_manifest_sha256": C.sha256_file(campaign_path),
        "campaign_body_sha256": campaign["body_sha256"],
        "campaign_id": campaign["campaign_id"],
        "protocol_sha256": campaign["protocol_sha256"],
        "projection_sha256": campaign["projection_sha256"],
        "registrations": actual_registration_seals,
        "materialized_roots": {
            "pi05": models["pi05"]["local_root"],
            "paligemma": models["tokenizer"]["local_root"],
            "clip_tokenizer": models["sam3"]["clip_tokenizer"]["root"],
        },
        "tokenization_proof_sha256": C.canonical_sha256(plan.tokenization_proof),
        "producer_sha256": plan.freezer_ref["sha256"],
        "environment_constructed_or_reset": False,
        "target_rgb_opened_or_generated": False,
        "oracle_json_decoded": False,
        "oracle_success_values_accessed": False,
    }
    require(result == expected_result, "freeze RESULT content drift")
    expected_complete = {
        "schema": FREEZE_COMPLETE_SCHEMA,
        "status": "atomic_success",
        "atomic_commit": True,
        "input_closure_sha256": plan.input_closure_sha256,
        "materialization_complete_sha256": materialization_complete_sha256,
        "campaign_manifest_sha256": C.sha256_file(campaign_path),
        "campaign_body_sha256": campaign["body_sha256"],
        "campaign_id": campaign["campaign_id"],
        "projection_sha256": campaign["projection_sha256"],
        "result_sha256": C.sha256_file(result_path),
        "producer_sha256": C.sha256_file(producer_path),
        "registration_sha256s": {
            name: seal["sha256"] for name, seal in actual_registration_seals.items()
        },
    }
    require(complete == expected_complete, "freeze COMPLETE content drift")
    require(
        complete["producer_sha256"] == plan.freezer_ref["sha256"],
        "freeze producer is not the reviewed freezer",
    )
    authority = C.load_and_validate_campaign_bundle(
        stage,
        C.sha256_file(stage / "COMPLETE.json"),
        expected_source_role="formal_replay",
    )
    require(
        set(authority["registrations"]) == expected_names
        and authority["target_root"] == plan.options.target_root,
        "shared campaign loader returned a different authority view",
    )
    _assert_target_root_absent(plan.options.target_root)


def freeze_bundle(
    plan: FreezePlan,
    expected_input_closure_sha256: str,
    expected_materialization_complete_sha256: str | None = None,
) -> Path:
    require(
        C.is_sha256(expected_input_closure_sha256),
        "--expected-input-closure-sha256 must be 64 lowercase hex",
    )
    require(
        expected_input_closure_sha256 == plan.input_closure_sha256,
        "input closure differs from reviewed validate-only digest",
    )
    _assert_target_root_absent(plan.options.target_root)
    require(
        not os.path.lexists(plan.options.output_root),
        "registration stage-B root already exists",
    )
    if os.path.lexists(plan.options.materialized_root):
        require(
            C.is_sha256(expected_materialization_complete_sha256),
            "reusing stage A requires --expected-materialization-complete-sha256",
        )
        materialized = plan.options.materialized_root
        materialization_complete_path = materialized / "MATERIALIZATION_COMPLETE.json"
        require(
            C.sha256_file(materialization_complete_path)
            == expected_materialization_complete_sha256,
            "existing stage-A COMPLETE does not equal the external anchor",
        )
        _validate_materialized_stage(materialized, plan)
    else:
        require(
            expected_materialization_complete_sha256 is None,
            "stage-A anchor supplied but materialized root does not exist",
        )
        materialized = _publish_materialized_models(plan)
    roots = _materialized_roots(materialized, plan)
    final_models = _models(plan, roots["pi05"], roots["paligemma"], roots["clip"])
    created_utc = plan.options.created_utc
    registrations, projection = make_registrations(
        plan, final_models, created_utc
    )
    _revalidate_non_model_inputs(plan)
    _assert_target_root_absent(plan.options.target_root)
    for registration in registrations.values():
        C.validate_registration(registration, expected_source_role="formal_replay")
    materialization_complete = materialized / "MATERIALIZATION_COMPLETE.json"
    materialization_complete_sha = C.sha256_file(materialization_complete)
    output = plan.options.output_root

    def build(stage: Path) -> None:
        registration_seals = {}
        for (seed, phase), registration in registrations.items():
            filename = _registration_filename(seed, phase)
            path = stage / "registrations" / filename
            _write_bytes(path, C.canonical_bytes(registration))
            registration_seals[filename] = {
                "sha256": C.sha256_file(path),
                "self_sha256": registration["self_sha256"],
            }
        _write_bytes(stage / "PRODUCER.py", Path(__file__).read_bytes())
        require(
            C.sha256_file(stage / "PRODUCER.py") == plan.freezer_ref["sha256"],
            "freezer changed before stage B",
        )
        campaign = _campaign_manifest(
            registrations, registration_seals, projection, created_utc
        )
        _write_bytes(stage / "CAMPAIGN.json", C.canonical_bytes(campaign))
        result = {
            "schema": FREEZE_RESULT_SCHEMA,
            "status": "PASS",
            "created_utc": created_utc,
            "input_closure_sha256": plan.input_closure_sha256,
            "output_root": output.as_posix(),
            "materialized_root": materialized.as_posix(),
            "target_root": plan.options.target_root.as_posix(),
            "materialization_complete_sha256": materialization_complete_sha,
            "campaign_manifest_sha256": C.sha256_file(stage / "CAMPAIGN.json"),
            "campaign_body_sha256": campaign["body_sha256"],
            "campaign_id": campaign["campaign_id"],
            "protocol_sha256": campaign["protocol_sha256"],
            "projection_sha256": campaign["projection_sha256"],
            "registrations": registration_seals,
            "materialized_roots": {
                "pi05": final_models["pi05"]["local_root"],
                "paligemma": final_models["tokenizer"]["local_root"],
                "clip_tokenizer": final_models["sam3"]["clip_tokenizer"]["root"],
            },
            "tokenization_proof_sha256": C.canonical_sha256(plan.tokenization_proof),
            "producer_sha256": plan.freezer_ref["sha256"],
            "environment_constructed_or_reset": False,
            "target_rgb_opened_or_generated": False,
            "oracle_json_decoded": False,
            "oracle_success_values_accessed": False,
        }
        _write_bytes(stage / "FREEZE_RESULT.json", C.canonical_bytes(result))
        complete = {
            "schema": FREEZE_COMPLETE_SCHEMA,
            "status": "atomic_success",
            "atomic_commit": True,
            "input_closure_sha256": plan.input_closure_sha256,
            "materialization_complete_sha256": materialization_complete_sha,
            "campaign_manifest_sha256": C.sha256_file(stage / "CAMPAIGN.json"),
            "campaign_body_sha256": campaign["body_sha256"],
            "campaign_id": campaign["campaign_id"],
            "projection_sha256": campaign["projection_sha256"],
            "result_sha256": C.sha256_file(stage / "FREEZE_RESULT.json"),
            "producer_sha256": C.sha256_file(stage / "PRODUCER.py"),
            "registration_sha256s": {
                name: value["sha256"] for name, value in registration_seals.items()
            },
        }
        _write_bytes(stage / "COMPLETE.json", C.canonical_bytes(complete))

    def validate(stage: Path) -> None:
        result = _strict_json(stage / "FREEZE_RESULT.json", "freeze RESULT")
        _validate_registration_stage(
            stage,
            plan,
            registrations,
            projection,
            result["registrations"],
            materialization_complete_sha,
        )

    published = _atomic_publish_directory(output, build, validate)
    external_anchor = C.sha256_file(published / "COMPLETE.json")
    authority = C.load_and_validate_campaign_bundle(
        published,
        external_anchor,
        expected_source_role="formal_replay",
    )
    require(
        set(authority["registrations"])
        == {
            _registration_filename(seed, phase)
            for phase in ("screen", "confirm")
            for seed in (8000, 8100)
        }
        and authority["target_root"] == plan.options.target_root,
        "published campaign failed downstream authority round-trip",
    )
    return published


def _parse_runtime_overrides(values: Sequence[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        role, separator, raw_path = value.partition("=")
        require(separator == "=" and role and raw_path, "runtime override must be ROLE=PATH")
        require(role not in result, f"duplicate runtime override: {role}")
        result[role] = Path(raw_path)
    return result


def options_from_args(args: argparse.Namespace) -> FreezeOptions:
    return FreezeOptions(
        output_root=args.output_root,
        materialized_root=args.materialized_root,
        target_root=args.target_root,
        sanitized_roots={8000: args.sanitized_8000, 8100: args.sanitized_8100},
        oracle_paths={8000: args.oracle_8000, 8100: args.oracle_8100},
        pi05_cache_root=args.pi05_cache_root,
        paligemma_cache_root=args.paligemma_cache_root,
        clip_source_root=args.clip_tokenizer_source,
        clip_revision=args.clip_tokenizer_revision,
        sam3_checkpoint=args.sam3_checkpoint,
        sam3_model_id=args.sam3_model_id,
        sam3_revision=args.sam3_revision,
        cream_template_image=args.cream_template_image,
        cream_template_metadata=args.cream_template_metadata,
        semantic_config=args.semantic_config,
        semantic_producer=args.semantic_producer,
        semantic_evaluator=args.semantic_evaluator,
        runtime_source_overrides=_parse_runtime_overrides(args.runtime_source),
        created_utc=args.created_utc,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--freeze", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO / "results/v248_gate0_registration_bundle_v1",
    )
    parser.add_argument(
        "--materialized-root",
        type=Path,
        default=REPO / "results/v248_gate0_materialized_models_v1",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=REPO / "results/v248_gate0_formal_campaign_v1",
    )
    parser.add_argument("--sanitized-8000", type=Path, default=DEFAULT_SANITIZED[8000])
    parser.add_argument("--sanitized-8100", type=Path, default=DEFAULT_SANITIZED[8100])
    parser.add_argument("--oracle-8000", type=Path)
    parser.add_argument("--oracle-8100", type=Path)
    parser.add_argument("--pi05-cache-root", type=Path, default=DEFAULT_PI05_CACHE)
    parser.add_argument(
        "--paligemma-cache-root", type=Path, default=DEFAULT_PALIGEMMA_CACHE
    )
    parser.add_argument(
        "--clip-tokenizer-source", type=Path, default=DEFAULT_CLIP_SOURCE
    )
    parser.add_argument("--clip-tokenizer-revision")
    parser.add_argument("--sam3-checkpoint", type=Path)
    parser.add_argument("--sam3-model-id", default="1038lab/sam3")
    parser.add_argument("--sam3-revision")
    parser.add_argument(
        "--cream-template-image", type=Path, default=DEFAULT_CREAM_TEMPLATE_IMAGE
    )
    parser.add_argument(
        "--cream-template-metadata",
        type=Path,
        default=DEFAULT_CREAM_TEMPLATE_METADATA,
    )
    parser.add_argument(
        "--semantic-config",
        type=Path,
        default=REPO / "plan_and_progress/v248_gate0_semantic_config.json",
    )
    parser.add_argument(
        "--semantic-producer",
        type=Path,
        default=REPO / "scripts/produce_v248_gate0_progress.py",
    )
    parser.add_argument(
        "--semantic-evaluator",
        type=Path,
        default=REPO / "scripts/eval_v248_gate0_progress.py",
    )
    parser.add_argument("--runtime-source", action="append", default=[], metavar="ROLE=PATH")
    parser.add_argument("--created-utc", required=True)
    parser.add_argument("--expected-input-closure-sha256")
    parser.add_argument("--expected-materialization-complete-sha256")
    args = parser.parse_args(argv)
    if args.freeze and not args.expected_input_closure_sha256:
        parser.error("--freeze requires --expected-input-closure-sha256 from validate-only")
    if not args.freeze:
        args.validate_only = True
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan = collect_plan(
        options_from_args(args),
        expected_materialization_complete_sha256=(
            args.expected_materialization_complete_sha256
        ),
    )
    if args.validate_only:
        C.assert_no_simulator_modules_imported()
        print(
            json.dumps(
                {
                    "status": "VALID",
                    "mode": "validate_only",
                    "input_closure_sha256": plan.input_closure_sha256,
                    "output_root": plan.options.output_root.as_posix(),
                    "materialized_root": plan.options.materialized_root.as_posix(),
                    "target_root": plan.options.target_root.as_posix(),
                    "output_created": False,
                    "registrations_planned": 4,
                    "environment_constructed_or_reset": False,
                    "target_rgb_opened_or_generated": False,
                    "oracle_json_decoded": False,
                    "oracle_success_values_accessed": False,
                    "simulator_modules_imported": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    output = freeze_bundle(
        plan,
        args.expected_input_closure_sha256,
        args.expected_materialization_complete_sha256,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "mode": "freeze",
                "input_closure_sha256": plan.input_closure_sha256,
                "freeze_complete_sha256": C.sha256_file(output / "COMPLETE.json"),
                "output_root": output.as_posix(),
                "materialized_root": plan.options.materialized_root.as_posix(),
                "target_root": plan.options.target_root.as_posix(),
                "registrations": 4,
                "environment_constructed_or_reset": False,
                "target_rgb_opened_or_generated": False,
                "oracle_json_decoded": False,
                "oracle_success_values_accessed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
