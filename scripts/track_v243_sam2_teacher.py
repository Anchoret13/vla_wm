#!/usr/bin/env python3
"""Frozen-registration SAM2 RGB teacher for chain3 Gate 0.

The only runtime inputs are the frozen registration, one or more v242 replay
JSONL manifests, the JPEGs referenced by those manifests, their sibling v242
mechanical replay-completion summaries, and the registered SAM2 checkpoint/
conversion code.  It never follows the recorded source-summary path and never
opens a tape, simulator state, semantic event, success label, or task definition.

``objects`` intentionally implements the exact label contract consumed by
``train_v242_vlm_phi_gate.py``.  Rich tracker measurements live in the parallel
``object_diagnostics`` field so the existing gate can consume labels unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import transformers
from PIL import Image, ImageDraw
from transformers import Sam2ImageProcessor, Sam2VideoProcessor, Sam2VideoVideoProcessor

import probe_v242_sam2_tracking as registered_probe


REPO = Path(__file__).resolve().parent.parent
EXPECTED_REGISTRATION_SHA256 = "3d892616b97442a080e16f50f226217eb00dd3d2e19f6fb7e048750a251363ab"
REGISTRATION_SCHEMA = "v242_chain3_sam2_registration_v1"
REPLAY_SCHEMA = "v242_rgb_replay_v1"
# Kept for direct compatibility with the existing Gate-0 loader.
LABEL_SCHEMA = "v242_qwen_label_v1"
TEACHER_LABEL_SCHEMA = "v243_sam2_teacher_label_v1"
RUN_SCHEMA = "v243_sam2_teacher_run_v1"
EPISODE_SCHEMA = "v243_sam2_teacher_episode_summary_v1"
COMPLETE_SCHEMA = "v243_sam2_teacher_complete_v1"
REPLAY_SUMMARY_SCHEMA = "v242_rgb_replay_summary_v1"
TARGET_OBJECTS = ("alphabet_soup_1", "tomato_sauce_1", "cream_cheese_1")
HEX = frozenset("0123456789abcdef")
FORBIDDEN_MECHANICAL_SUMMARY_KEY_FRAGMENTS = ("semantic", "event", "success", "bddl")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json_rows(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_fields(mapping: dict[str, Any], fields: Iterable[str], where: str) -> None:
    missing = [field for field in fields if field not in mapping]
    if missing:
        raise ValueError(f"{where}: missing fields {missing}")


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value.lower()) <= HEX


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_json_strict(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def load_jsonl_strict(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL row")
            row = json.loads(line, object_pairs_hook=reject_duplicate_keys)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: empty manifest")
    return rows


def forbidden_key_paths(value: Any, prefix: str = "$") -> list[str]:
    """Find privileged semantic key names without inspecting referenced files."""
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(fragment in lowered for fragment in FORBIDDEN_MECHANICAL_SUMMARY_KEY_FRAGMENTS):
                found.append(f"{prefix}.{key}")
            found.extend(forbidden_key_paths(child, f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(forbidden_key_paths(child, f"{prefix}[{index}]"))
    return found


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def parse_partition(registration: dict[str, Any]) -> tuple[dict[int, str], dict[str, list[int]]]:
    partition = registration["episode_partition"]
    names = {
        "tracker_calibration": "tracker_and_initializer_calibration_episode_ids",
        "phi_fit": "phi_fit_episode_ids",
        "phi_calibration": "phi_calibration_episode_ids",
        "phi_test": "phi_test_episode_ids",
    }
    groups: dict[str, list[int]] = {}
    episode_to_partition: dict[int, str] = {}
    for short_name, field in names.items():
        values = partition.get(field)
        require(isinstance(values, list) and values, f"registration: invalid {field}")
        ids = [int(value) for value in values]
        require(ids == sorted(set(ids)), f"registration: {field} must be sorted and unique")
        groups[short_name] = ids
        for episode_id in ids:
            require(episode_id not in episode_to_partition, "registration partitions overlap")
            episode_to_partition[episode_id] = short_name
    require(partition.get("disjoint") is True, "registration does not declare disjoint partitions")
    require(
        sorted(episode_to_partition) == list(range(96)),
        "registration episode partitions must cover exactly episode IDs 0..95",
    )
    require(
        partition.get("environment_seed_rule") == "env_seed = 8700 + episode_id",
        "unexpected environment seed rule",
    )
    return episode_to_partition, groups


def validate_registration(path: Path) -> dict[str, Any]:
    actual_sha = sha256_file(path)
    require(
        actual_sha == EXPECTED_REGISTRATION_SHA256,
        f"registration SHA mismatch: {actual_sha} != {EXPECTED_REGISTRATION_SHA256}",
    )
    registration = load_json_strict(path)
    require(registration.get("schema") == REGISTRATION_SCHEMA, "registration schema mismatch")
    require(
        registration.get("status") == "frozen_before_any_episode_16_to_95_tracking_output",
        "registration is not in the expected frozen state",
    )
    target = registration.get("target")
    require(isinstance(target, dict), "registration target must be an object")
    expected_target = {
        "task": "chain3_lr2",
        "camera": "agentview",
        "image_width": 360,
        "image_height": 360,
        "source_tape_sha256": "7dbff743ea89db5fd792d317111129d188d439122a46db7384f4549fd947f86c",
        "chunk_steps": 10,
        "horizon_steps": 750,
    }
    for field, expected in expected_target.items():
        require(target.get(field) == expected, f"registration target.{field} mismatch")
    require(is_sha256(target["source_tape_sha256"]), "registration tape SHA is invalid")
    episode_to_partition, groups = parse_partition(registration)

    initializer = registration.get("fixed_t0_initializer")
    require(isinstance(initializer, dict), "missing fixed_t0_initializer")
    require(
        initializer.get("method")
        == (
            "four task-specific fixed xyxy box prompts; no per-episode detector, "
            "adaptation, manual correction, or later-frame prompt"
        ),
        "unexpected initializer method",
    )
    objects = initializer.get("objects")
    require(isinstance(objects, dict), "registration initializer objects must be a mapping")
    expected_names = {"basket_1", *TARGET_OBJECTS}
    require(set(objects) == expected_names, f"initializer objects must be exactly {expected_names}")
    seen_ids: set[int] = set()
    for name, config in objects.items():
        require(isinstance(config, dict), f"initializer {name} must be an object")
        require_fields(config, ("sam_object_id", "xyxy"), f"initializer {name}")
        obj_id = int(config["sam_object_id"])
        require(obj_id not in seen_ids, "SAM object IDs must be unique")
        seen_ids.add(obj_id)
        box = config["xyxy"]
        require(
            isinstance(box, list)
            and len(box) == 4
            and all(isinstance(value, (int, float)) and math.isfinite(value) for value in box),
            f"initializer {name}: invalid xyxy",
        )
        x0, y0, x1, y1 = (float(value) for value in box)
        require(
            0 <= x0 < x1 <= 360 and 0 <= y0 < y1 <= 360,
            f"initializer {name}: box out of bounds",
        )
    require(seen_ids == {1, 2, 3, 4}, "registered SAM IDs must be exactly 1..4")

    tracker = registration.get("tracker")
    require(isinstance(tracker, dict), "registration tracker must be an object")
    require(tracker.get("no_prompt_after_frame_zero") is True, "later prompts are not forbidden")
    require(tracker.get("inference_dtype") == "bfloat16", "unexpected inference dtype")
    require(is_sha256(tracker.get("checkpoint_sha256")), "invalid checkpoint SHA")
    require(
        int(tracker.get("checkpoint_tensor_count", -1)) == 903,
        "checkpoint tensor count mismatch",
    )
    checkpoint = Path(tracker["checkpoint_path"]).resolve()
    require(checkpoint.is_file(), f"registered checkpoint missing: {checkpoint}")
    require(sha256_file(checkpoint) == tracker["checkpoint_sha256"], "checkpoint SHA mismatch")

    probe_path = (REPO / tracker["probe_code_path"]).resolve()
    require(probe_path == Path(registered_probe.__file__).resolve(), "imported probe path mismatch")
    require(sha256_file(probe_path) == tracker["probe_code_sha256"], "probe code SHA mismatch")

    rule = registration.get("visual_label_rule")
    require(isinstance(rule, dict), "registration visual_label_rule must be an object")
    require(
        rule.get("basket_relation")
        == "target-mask centroid lies inside the current tracked basket-mask bounding box",
        "unexpected basket relation",
    )
    require(rule.get("temporal_smoothing") == "none", "temporal smoothing must be disabled")
    require(rule.get("labels") == ["inside", "outside", "uncertain"], "label set mismatch")
    uncertain = rule.get("uncertain_if")
    require(isinstance(uncertain, dict), "missing uncertainty thresholds")
    expected_uncertain = {
        "object_score_logit_lte": 0.0,
        "target_mask_area_px_lt": 64,
        "target_mask_area_px_gt": 5000,
        "basket_mask_area_px_lt": 3000,
        "basket_mask_area_px_gt": 15000,
        "nonfinite_output": True,
    }
    require(
        uncertain == expected_uncertain,
        "uncertainty thresholds differ from frozen registration",
    )
    return {
        "payload": registration,
        "sha256": actual_sha,
        "checkpoint": checkpoint,
        "probe_path": probe_path,
        "episode_to_partition": episode_to_partition,
        "partition_groups": groups,
        "objects": objects,
        "thresholds": uncertain,
    }


def validate_image_record(
    image_record: dict[str, Any], manifest_dir: Path, where: str
) -> tuple[Path, str, int, int]:
    require(isinstance(image_record, dict), f"{where}: image record must be an object")
    require_fields(image_record, ("path", "sha256", "width", "height"), where)
    require(is_sha256(image_record["sha256"]), f"{where}: invalid image SHA")
    image_path = (manifest_dir / str(image_record["path"])).resolve()
    require(image_path.is_relative_to(manifest_dir), f"{where}: image escapes manifest directory")
    require(image_path.is_file(), f"{where}: missing image {image_path}")
    require(sha256_file(image_path) == image_record["sha256"], f"{where}: image SHA mismatch")
    width, height = int(image_record["width"]), int(image_record["height"])
    return image_path, str(image_record["sha256"]), width, height


def validate_replay_completion_summary(
    summary_path: Path,
    manifest_path: Path,
    manifest_sha256: str,
    source_rows: list[dict[str, Any]],
    target: dict[str, Any],
) -> dict[str, Any]:
    """Validate only v242 replay mechanics; never follow source-summary paths."""
    summary = load_json_strict(summary_path)
    forbidden = forbidden_key_paths(summary)
    require(not forbidden, f"{summary_path}: privileged/semantic fields forbidden: {forbidden}")
    require(summary.get("schema") == REPLAY_SUMMARY_SCHEMA, "replay summary schema mismatch")
    require(summary.get("task") == target["task"], "replay summary task mismatch")
    require(int(summary.get("c", -1)) == target["chunk_steps"], "replay summary c mismatch")
    require(
        summary.get("tape_sha256") == target["source_tape_sha256"],
        "replay summary tape SHA mismatch",
    )
    require(
        summary.get("manifest_sha256") == manifest_sha256,
        "replay summary manifest SHA mismatch",
    )
    declared_manifest = (summary_path.parent / str(summary.get("manifest_path", ""))).resolve()
    require(declared_manifest == manifest_path, "replay summary manifest path mismatch")

    episode_ids = sorted({int(row["episode_id"]) for row in source_rows})
    env_seed_by_episode = {
        episode_id: {
            int(row["env_seed"])
            for row in source_rows
            if int(row["episode_id"]) == episode_id
        }
        for episode_id in episode_ids
    }
    require(
        all(len(seeds) == 1 for seeds in env_seed_by_episode.values()),
        "manifest episode crosses env seeds",
    )
    expected_env_seeds = [next(iter(env_seed_by_episode[episode_id])) for episode_id in episode_ids]
    require(summary.get("episode_ids") == episode_ids, "replay summary episode IDs mismatch")
    require(summary.get("env_seeds") == expected_env_seeds, "replay summary env seeds mismatch")
    require(
        int(summary.get("episodes_replayed", -1)) == len(episode_ids),
        "episodes_replayed mismatch",
    )
    require(int(summary.get("frames_saved", -1)) == len(source_rows), "frames_saved mismatch")
    require(int(summary.get("chunks_replayed", -1)) == len(source_rows), "chunks_replayed mismatch")
    require(
        int(summary.get("env_steps_replayed", -1))
        == len(source_rows) * target["chunk_steps"],
        "env_steps_replayed mismatch",
    )
    sampling = summary.get("sampling")
    require(
        isinstance(sampling, dict)
        and sampling.get("chunk_stride") == 1
        and sampling.get("pre_action") is True,
        "replay summary must declare dense pre-action stride-1 sampling",
    )

    validation = summary.get("proprio_validation")
    require(isinstance(validation, dict), "replay summary missing proprio validation")
    require(
        int(validation.get("frames_checked", -1)) == len(source_rows),
        "proprio frame count mismatch",
    )
    threshold = float(validation.get("threshold_max_abs_error", float("nan")))
    observed = float(validation.get("observed_max_abs_error", float("nan")))
    mean_max = float(validation.get("mean_of_frame_max_abs_error", float("nan")))
    require(
        all(
            math.isfinite(value) and value >= 0
            for value in (threshold, observed, mean_max)
        ),
        "invalid proprio summary values",
    )
    row_errors = [float(row["proprio"]["max_abs_error"]) for row in source_rows]
    require(
        all(math.isfinite(value) and 0 <= value <= threshold for value in row_errors),
        "manifest proprio error exceeds threshold",
    )
    require(
        abs(observed - max(row_errors, default=0.0)) <= 1e-12,
        "observed proprio maximum mismatch",
    )

    recovery = summary.get("terminal_action_recovery")
    require(isinstance(recovery, dict), "replay summary missing terminal recovery metadata")
    require(recovery.get("recoverable") is False, "unstored terminal action must be unrecoverable")
    episodes_with_unstored = int(
        recovery.get("episodes_with_unstored_terminal_steps", -1)
    )
    unstored_steps = int(recovery.get("unstored_terminal_env_steps", -1))
    require(
        0 <= episodes_with_unstored <= len(episode_ids),
        "invalid count of episodes with unstored terminal steps",
    )
    require(
        (episodes_with_unstored == 0 and unstored_steps == 0)
        or (
            episodes_with_unstored > 0
            and episodes_with_unstored <= unstored_steps
            <= episodes_with_unstored * target["chunk_steps"]
        ),
        "invalid aggregate count of unstored terminal environment steps",
    )
    source_summary_sha = summary.get("summary_sha256")
    require(is_sha256(source_summary_sha), "replay summary source-summary SHA is invalid")
    declared_row_shas = {str(row["summary_sha256"]).lower() for row in source_rows}
    require(declared_row_shas == {source_summary_sha}, "manifest/source-summary SHA mismatch")
    return {
        "path": str(summary_path),
        "sha256": sha256_file(summary_path),
        "schema": REPLAY_SUMMARY_SCHEMA,
        "mechanical_metadata_only": True,
        "forbidden_privileged_field_paths": [],
        "manifest_sha256": manifest_sha256,
        "source_summary_sha256_declared_only_not_opened": source_summary_sha,
        "episode_ids": episode_ids,
        "frames_saved": len(source_rows),
        "chunks_replayed": len(source_rows),
        "sampling": {"chunk_stride": 1, "pre_action": True},
        "proprio_validation": {
            "threshold_max_abs_error": threshold,
            "observed_max_abs_error": observed,
            "frames_checked": len(source_rows),
        },
        "episodes_with_unstored_terminal_steps": episodes_with_unstored,
        "unstored_terminal_env_steps": unstored_steps,
    }


def validate_manifests(
    manifest_paths: list[Path],
    requested_episode_ids: set[int],
    frozen: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target = frozen["payload"]["target"]
    all_rows: list[dict[str, Any]] = []
    manifest_audit: list[dict[str, Any]] = []
    seen_manifest_paths: set[Path] = set()
    seen_manifest_shas: set[str] = set()
    seen_frame_ids: set[str] = set()
    seen_row_indices: set[int] = set()
    seen_episodes: set[int] = set()

    for raw_path in manifest_paths:
        manifest = raw_path.resolve()
        require(manifest.is_file(), f"missing manifest: {manifest}")
        require(manifest not in seen_manifest_paths, f"duplicate manifest path: {manifest}")
        seen_manifest_paths.add(manifest)
        manifest_sha = sha256_file(manifest)
        require(
            manifest_sha not in seen_manifest_shas,
            f"duplicate manifest content: {manifest_sha}",
        )
        seen_manifest_shas.add(manifest_sha)
        source_rows = load_jsonl_strict(manifest)
        manifest_dir = manifest.parent.resolve()
        manifest_episode_ids: set[int] = set()
        verified_image_records = 0
        verified_image_digest = hashlib.sha256()
        normalized_rows: list[dict[str, Any]] = []

        for row_number, row in enumerate(source_rows, start=1):
            where = f"{manifest}:{row_number}"
            require_fields(
                row,
                (
                    "schema",
                    "frame_id",
                    "task",
                    "tape_sha256",
                    "episode_id",
                    "env_seed",
                    "t",
                    "row_index",
                    "chunk_index",
                    "summary_sha256",
                    "proprio",
                    "sampling",
                    "images",
                ),
                where,
            )
            require(row["schema"] == REPLAY_SCHEMA, f"{where}: replay schema mismatch")
            require(row["task"] == target["task"], f"{where}: task mismatch")
            require(
                row["tape_sha256"] == target["source_tape_sha256"],
                f"{where}: tape SHA mismatch",
            )
            source_summary_sha = str(row["summary_sha256"]).lower()
            require(is_sha256(source_summary_sha), f"{where}: invalid source summary SHA")
            episode_id = int(row["episode_id"])
            env_seed = int(row["env_seed"])
            t_value = int(row["t"])
            row_index = int(row["row_index"])
            chunk_index = int(row["chunk_index"])
            require(episode_id in frozen["episode_to_partition"], f"{where}: unregistered episode")
            require(env_seed == 8700 + episode_id, f"{where}: env_seed rule mismatch")
            require(
                row_index >= 0 and row_index not in seen_row_indices,
                f"{where}: duplicate/invalid row_index",
            )
            require(chunk_index >= 0, f"{where}: negative chunk_index")
            require(
                t_value == chunk_index * target["chunk_steps"],
                f"{where}: t/chunk mismatch",
            )
            require(0 <= t_value < target["horizon_steps"], f"{where}: t outside horizon")
            frame_id = str(row["frame_id"])
            expected_frame_id = f"{target['source_tape_sha256'][:12]}:ep{episode_id}:t{t_value}"
            require(frame_id == expected_frame_id, f"{where}: frame_id mismatch")
            require(frame_id not in seen_frame_ids, f"{where}: duplicate frame_id")
            sampling = row["sampling"]
            require(isinstance(sampling, dict), f"{where}: sampling must be an object")
            require(sampling.get("chunk_stride") == 1, f"{where}: dense chunk_stride=1 required")
            require(sampling.get("pre_action") is True, f"{where}: pre_action=true required")
            proprio = row["proprio"]
            require(isinstance(proprio, dict), f"{where}: proprio must be an object")
            require_fields(
                proprio,
                ("max_abs_error", "mean_abs_error"),
                f"{where}:proprio",
            )
            proprio_max = float(proprio["max_abs_error"])
            proprio_mean = float(proprio["mean_abs_error"])
            require(
                math.isfinite(proprio_max)
                and math.isfinite(proprio_mean)
                and 0 <= proprio_mean <= proprio_max,
                f"{where}: invalid proprio errors",
            )
            images = row["images"]
            require(
                isinstance(images, dict) and target["camera"] in images,
                f"{where}: missing registered camera",
            )
            verified_images: dict[str, dict[str, Any]] = {}
            for camera_name in sorted(images):
                image_path, image_sha, width, height = validate_image_record(
                    images[camera_name], manifest_dir, f"{where}:{camera_name}"
                )
                require(
                    width == target["image_width"] and height == target["image_height"],
                    f"{where}:{camera_name}: dimensions mismatch",
                )
                with Image.open(image_path) as opened:
                    require(
                        opened.size == (width, height),
                        f"{where}:{camera_name}: JPEG dimensions mismatch",
                    )
                    opened.verify()
                verified_images[camera_name] = {
                    "path": image_path,
                    "sha256": image_sha,
                    "width": width,
                    "height": height,
                }
                verified_image_digest.update(f"{frame_id}\0{camera_name}\0{image_sha}\n".encode())
                verified_image_records += 1
            normalized_rows.append(
                {
                    "frame_id": frame_id,
                    "task": str(row["task"]),
                    "episode_id": episode_id,
                    "env_seed": env_seed,
                    "t": t_value,
                    "row_index": row_index,
                    "chunk_index": chunk_index,
                    "source_summary_sha256": source_summary_sha,
                    "proprio_max_abs_error": proprio_max,
                    "manifest_path": manifest,
                    "manifest_sha256": manifest_sha,
                    "images": verified_images,
                }
            )
            manifest_episode_ids.add(episode_id)
            seen_episodes.add(episode_id)
            seen_frame_ids.add(frame_id)
            seen_row_indices.add(row_index)

        replay_summary_path = manifest_dir / "summary.json"
        require(
            replay_summary_path.is_file(),
            f"missing mechanical replay summary: {replay_summary_path}",
        )
        replay_completion = validate_replay_completion_summary(
            replay_summary_path,
            manifest,
            manifest_sha,
            source_rows,
            target,
        )
        for normalized in normalized_rows:
            normalized["replay_completion_aggregate"] = {
                "manifest_episode_count": len(manifest_episode_ids),
                "episodes_with_unstored_terminal_steps": replay_completion[
                    "episodes_with_unstored_terminal_steps"
                ],
                "unstored_terminal_env_steps": replay_completion[
                    "unstored_terminal_env_steps"
                ],
            }

        manifest_audit.append(
            {
                "path": str(manifest),
                "sha256": manifest_sha,
                "row_count": len(source_rows),
                "episode_ids": sorted(manifest_episode_ids),
                "verified_image_records": verified_image_records,
                "verified_image_set_sha256": verified_image_digest.hexdigest(),
                "replay_completion": replay_completion,
            }
        )
        all_rows.extend(normalized_rows)

    require(
        seen_episodes == requested_episode_ids,
        f"manifest episodes {seen_episodes} != requested {requested_episode_ids}",
    )
    expected_frame_count = target["horizon_steps"] // target["chunk_steps"]
    by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        by_episode[row["episode_id"]].append(row)
    for episode_id, episode_rows in by_episode.items():
        episode_rows.sort(key=lambda row: row["t"])
        times = [row["t"] for row in episode_rows]
        chunks = [row["chunk_index"] for row in episode_rows]
        require(
            1 <= len(episode_rows) <= expected_frame_count,
            f"episode {episode_id}: invalid dense replay prefix length",
        )
        require(
            times == [index * target["chunk_steps"] for index in range(len(episode_rows))],
            f"episode {episode_id}: time grid must be a gap-free prefix from t0",
        )
        require(
            chunks == list(range(len(episode_rows))),
            f"episode {episode_id}: chunk grid mismatch",
        )
        require(
            len({row["manifest_sha256"] for row in episode_rows}) == 1,
            f"episode {episode_id}: split across manifests",
        )
        require(
            len({row["source_summary_sha256"] for row in episode_rows}) == 1,
            f"episode {episode_id}: inconsistent source-summary SHA declarations",
        )
    all_rows.sort(key=lambda row: (row["episode_id"], row["t"]))
    return all_rows, manifest_audit


def mask_stats(mask: np.ndarray) -> dict[str, Any]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return {"area_px": 0, "bbox_xyxy": None, "centroid_xy": None}
    return {
        "area_px": int(len(xs)),
        "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
        "centroid_xy": [float(xs.mean()), float(ys.mean())],
    }


def centroid_inside_bbox(centroid: list[float] | None, bbox: list[int] | None) -> bool | None:
    if centroid is None or bbox is None:
        return None
    x_value, y_value = centroid
    x0, y0, x1, y1 = bbox
    return bool(x0 <= x_value < x1 and y0 <= y_value < y1)


def finite_float(value: torch.Tensor) -> tuple[float | None, bool]:
    scalar = float(value.detach().float().cpu().reshape(-1)[0])
    return (scalar, True) if math.isfinite(scalar) else (None, False)


def normalize_masks(processed: torch.Tensor, object_count: int) -> torch.Tensor:
    while processed.ndim > 3 and processed.shape[0] == 1:
        processed = processed[0]
    if processed.ndim == 4 and processed.shape[1] == 1:
        processed = processed[:, 0]
    if processed.ndim != 3 or processed.shape[0] != object_count:
        raise RuntimeError(f"unexpected processed mask shape {tuple(processed.shape)}")
    return processed


def per_object_raw_finite(raw_masks: torch.Tensor, object_count: int) -> list[bool]:
    masks = raw_masks
    while masks.ndim > 3 and masks.shape[0] == 1:
        masks = masks[0]
    if masks.shape[0] != object_count:
        return [False] * object_count
    return [bool(torch.isfinite(masks[index]).all()) for index in range(object_count)]


def render_overlay(
    frame: Image.Image,
    masks: dict[int, np.ndarray],
    id_to_name: dict[int, str],
    diagnostics: dict[str, dict[str, Any]],
) -> Image.Image:
    palette = {1: (255, 255, 0), 2: (0, 255, 0), 3: (0, 160, 255), 4: (255, 0, 255)}
    pixels = np.asarray(frame).astype(np.float32)
    for obj_id, mask in masks.items():
        color = np.asarray(palette[obj_id], dtype=np.float32)
        pixels[mask] = 0.55 * pixels[mask] + 0.45 * color
    output = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(output)
    for obj_id, name in id_to_name.items():
        bbox = diagnostics[name]["mask_bbox_xyxy"]
        if bbox is None:
            continue
        draw.rectangle(bbox, outline=palette[obj_id], width=2)
        label = diagnostics[name].get("label", name)
        draw.text((bbox[0] + 2, max(0, bbox[1] - 12)), f"{name}:{label}", fill=palette[obj_id])
    return output


def classify_frame(
    output: Any,
    processed_masks: torch.Tensor,
    id_to_name: dict[int, str],
    thresholds: dict[str, Any],
) -> tuple[dict[str, Any], dict[int, np.ndarray]]:
    output_ids = [int(value) for value in output.object_ids]
    require(set(output_ids) == set(id_to_name), f"SAM2 object IDs changed: {output_ids}")
    raw_finite = per_object_raw_finite(output.pred_masks, len(output_ids))
    by_name: dict[str, dict[str, Any]] = {}
    masks: dict[int, np.ndarray] = {}
    for index, obj_id in enumerate(output_ids):
        name = id_to_name[obj_id]
        mask = processed_masks[index].detach().cpu().numpy().astype(bool)
        masks[obj_id] = mask
        stats = mask_stats(mask)
        score, score_finite = finite_float(output.object_score_logits[index])
        by_name[name] = {
            "sam_object_id": obj_id,
            "mask_area_px": stats["area_px"],
            "mask_bbox_xyxy": stats["bbox_xyxy"],
            "mask_centroid_xy": stats["centroid_xy"],
            "object_score_logit": score,
            "finite_output": bool(raw_finite[index] and score_finite),
        }

    basket = by_name["basket_1"]
    basket_reasons: list[str] = []
    if not basket["finite_output"]:
        basket_reasons.append("nonfinite_output")
    if (
        basket["object_score_logit"] is None
        or basket["object_score_logit"] <= thresholds["object_score_logit_lte"]
    ):
        basket_reasons.append("basket_object_score_logit_lte")
    if basket["mask_area_px"] < thresholds["basket_mask_area_px_lt"]:
        basket_reasons.append("basket_mask_area_px_lt")
    if basket["mask_area_px"] > thresholds["basket_mask_area_px_gt"]:
        basket_reasons.append("basket_mask_area_px_gt")
    basket["uncertain_reasons"] = basket_reasons
    basket["reliable"] = not basket_reasons

    for name in TARGET_OBJECTS:
        diagnostics = by_name[name]
        relation = centroid_inside_bbox(
            diagnostics["mask_centroid_xy"], basket["mask_bbox_xyxy"]
        )
        reasons = list(basket_reasons)
        if not diagnostics["finite_output"]:
            reasons.append("nonfinite_output")
        if (
            diagnostics["object_score_logit"] is None
            or diagnostics["object_score_logit"] <= thresholds["object_score_logit_lte"]
        ):
            reasons.append("object_score_logit_lte")
        if diagnostics["mask_area_px"] < thresholds["target_mask_area_px_lt"]:
            reasons.append("target_mask_area_px_lt")
        if diagnostics["mask_area_px"] > thresholds["target_mask_area_px_gt"]:
            reasons.append("target_mask_area_px_gt")
        reasons = sorted(set(reasons))
        label = "uncertain" if reasons else ("inside" if relation else "outside")
        diagnostics["centroid_in_tracked_basket_bbox"] = relation
        diagnostics["uncertain_reasons"] = reasons
        diagnostics["label"] = label
    return by_name, masks


def transition_summary(labels: list[dict[str, Any]], object_name: str) -> dict[str, Any]:
    sequence = [(row["t"], row["objects"][object_name]["label"]) for row in labels]
    first_certain_inside = next((t_value for t_value, label in sequence if label == "inside"), None)
    transition = None
    previous_certain = None
    for t_value, label in sequence:
        if label == "uncertain":
            continue
        if label == "inside" and previous_certain == "outside":
            transition = t_value
            break
        previous_certain = label
    counts = {
        label: sum(current == label for _, current in sequence)
        for label in ("inside", "outside", "uncertain")
    }
    return {
        "label_counts": counts,
        "first_certain_inside_t": first_certain_inside,
        "first_certain_outside_to_inside_transition_t": transition,
        "terminal_label": sequence[-1][1],
    }


def track_episode(
    model: Any,
    processor: Sam2VideoProcessor,
    episode_rows: list[dict[str, Any]],
    frozen: dict[str, Any],
    overlay_dir: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    registration = frozen["payload"]
    camera = registration["target"]["camera"]
    episode_id = episode_rows[0]["episode_id"]
    manifest_sha = episode_rows[0]["manifest_sha256"]
    frames = [Image.open(row["images"][camera]["path"]).convert("RGB") for row in episode_rows]
    width, height = frames[0].size
    object_config = frozen["objects"]
    id_to_name = {int(config["sam_object_id"]): name for name, config in object_config.items()}
    object_ids = tuple(sorted(id_to_name))
    boxes = [[object_config[id_to_name[obj_id]]["xyxy"] for obj_id in object_ids]]
    device = model.device
    session = processor.init_video_session(
        video=frames,
        inference_device=device,
        inference_state_device=device,
        processing_device=device,
        video_storage_device="cpu",
        dtype=torch.bfloat16,
    )
    processor.add_inputs_to_inference_session(
        inference_session=session,
        frame_idx=0,
        obj_ids=list(object_ids),
        input_boxes=boxes,
    )
    output_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        # This is the only prompted frame.  No add/refinement call occurs later.
        model(inference_session=session, frame_idx=0)
        iterator = model.propagate_in_video_iterator(session, show_progress_bar=True)
        for output in iterator:
            frame_index = int(output.frame_idx)
            require(0 <= frame_index < len(episode_rows), "SAM2 returned an invalid frame index")
            processed = processor.post_process_masks(
                [output.pred_masks],
                original_sizes=[[height, width]],
                binarize=True,
            )[0]
            processed = normalize_masks(processed, len(output.object_ids))
            diagnostics, masks = classify_frame(
                output, processed, id_to_name, frozen["thresholds"]
            )
            source = episode_rows[frame_index]
            objects = {
                name: {"label": diagnostics[name]["label"]} for name in TARGET_OBJECTS
            }
            object_diagnostics = {name: diagnostics[name] for name in TARGET_OBJECTS}
            row = {
                "schema": LABEL_SCHEMA,
                "teacher_schema": TEACHER_LABEL_SCHEMA,
                "producer": "frozen_registration_sam2_rgb_teacher",
                "frame_id": source["frame_id"],
                "task": source["task"],
                "replay_manifest_sha256": manifest_sha,
                "parse_ok": True,
                "objects": objects,
                "object_diagnostics": object_diagnostics,
                "basket_diagnostics": diagnostics["basket_1"],
                "episode_id": episode_id,
                "env_seed": source["env_seed"],
                "t": source["t"],
                "row_index": source["row_index"],
                "chunk_index": source["chunk_index"],
                "registration_sha256": frozen["sha256"],
                "checkpoint_sha256": registration["tracker"]["checkpoint_sha256"],
                "camera": camera,
                "prompt_frame_index": 0,
                "post_t0_prompt_count": 0,
                "source_image_sha256": source["images"][camera]["sha256"],
                "source_summary_sha256_declared_only": source["source_summary_sha256"],
            }
            if overlay_dir is not None:
                overlay_path = overlay_dir / f"ep{episode_id:04d}_t{source['t']:04d}.jpg"
                render_overlay(frames[frame_index], masks, id_to_name, diagnostics).save(
                    overlay_path, quality=95
                )
                row["overlay_relative_path"] = str(overlay_path.relative_to(overlay_dir.parent))
            output_rows.append(row)
    output_rows.sort(key=lambda row: row["t"])
    require(
        [row["t"] for row in output_rows] == [row["t"] for row in episode_rows],
        f"episode {episode_id}: SAM2 did not return each frame exactly once",
    )
    uncertain_count = sum(
        row["objects"][name]["label"] == "uncertain"
        for row in output_rows
        for name in TARGET_OBJECTS
    )
    summary = {
        "schema": EPISODE_SCHEMA,
        "episode_id": episode_id,
        "env_seed": episode_rows[0]["env_seed"],
        "partition": frozen["episode_to_partition"][episode_id],
        "manifest_sha256": manifest_sha,
        "frame_count": len(output_rows),
        "expected_frame_count": len(episode_rows),
        "frame_coverage": len(output_rows) / len(episode_rows),
        "nominal_horizon_frame_count": (
            registration["target"]["horizon_steps"]
            // registration["target"]["chunk_steps"]
        ),
        "nominal_horizon_frame_coverage": (
            len(output_rows)
            / (
                registration["target"]["horizon_steps"]
                // registration["target"]["chunk_steps"]
            )
        ),
        "manifest_is_gap_free_prefix": True,
        "mechanical_replay_right_truncated": len(output_rows)
        < registration["target"]["horizon_steps"] // registration["target"]["chunk_steps"],
        "unstored_terminal_env_steps_from_replay_metadata": (
            episode_rows[0]["replay_completion_aggregate"][
                "unstored_terminal_env_steps"
            ]
            if episode_rows[0]["replay_completion_aggregate"]["manifest_episode_count"] == 1
            else None
        ),
        "source_summary_sha256_declared_only_not_opened": episode_rows[0][
            "source_summary_sha256"
        ],
        "object_label_count": len(output_rows) * len(TARGET_OBJECTS),
        "uncertain_object_labels": uncertain_count,
        "uncertain_fraction": uncertain_count / (len(output_rows) * len(TARGET_OBJECTS)),
        "objects": {name: transition_summary(output_rows, name) for name in TARGET_OBJECTS},
        "prompt_frames": [0],
        "post_t0_prompt_count": 0,
        "runtime_seconds": time.perf_counter() - started,
        "labels_canonical_sha256": sha256_json_rows(output_rows),
    }
    del session, frames
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registration",
        type=Path,
        default=REPO / "plan_and_progress" / "v242_chain3_sam2_registration.json",
    )
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument(
        "--episode-id",
        type=int,
        action="append",
        required=True,
        help="Exact episode IDs expected across the input manifests; repeat for multiple episodes.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-overlays", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_started = time.perf_counter()
    requested_episode_ids = set(args.episode_id)
    require(len(requested_episode_ids) == len(args.episode_id), "duplicate --episode-id")
    require(requested_episode_ids, "at least one episode ID is required")
    output_dir = args.output_dir.resolve()
    require(not output_dir.exists(), f"output path already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    ).resolve()
    try:
        frozen = validate_registration(args.registration.resolve())
        rows, manifest_audit = validate_manifests(
            args.manifest, requested_episode_ids, frozen
        )
        if args.device.startswith("cuda"):
            require(torch.cuda.is_available(), "CUDA requested but unavailable")
        device = torch.device(args.device)
        model, checkpoint_tensor_count = registered_probe.load_model(
            frozen["checkpoint"], device, torch.bfloat16
        )
        require(checkpoint_tensor_count == 903, "strict loader returned wrong tensor count")
        processor = Sam2VideoProcessor(
            image_processor=Sam2ImageProcessor(),
            video_processor=Sam2VideoVideoProcessor(),
        )
        overlay_dir = temporary / "overlays" if args.save_overlays else None
        if overlay_dir is not None:
            overlay_dir.mkdir()
        episode_summary_dir = temporary / "episode_summaries"
        episode_summary_dir.mkdir()
        by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_episode[row["episode_id"]].append(row)

        labels: list[dict[str, Any]] = []
        episode_summaries: dict[str, dict[str, Any]] = {}
        for episode_id in sorted(by_episode):
            episode_labels, episode_summary = track_episode(
                model,
                processor,
                by_episode[episode_id],
                frozen,
                overlay_dir,
            )
            labels.extend(episode_labels)
            episode_path = episode_summary_dir / f"ep{episode_id:04d}.json"
            write_json(episode_path, episode_summary)
            episode_summaries[str(episode_id)] = {
                **episode_summary,
                "path": str(episode_path.relative_to(temporary)),
                "file_sha256": sha256_file(episode_path),
            }

        labels.sort(key=lambda row: (row["episode_id"], row["t"]))
        labels_path = temporary / "labels.jsonl"
        write_jsonl(labels_path, labels)
        per_manifest_dir = temporary / "labels_by_manifest"
        per_manifest_dir.mkdir()
        per_manifest_outputs: list[dict[str, Any]] = []
        for manifest in manifest_audit:
            manifest_labels = [
                row for row in labels if row["replay_manifest_sha256"] == manifest["sha256"]
            ]
            per_manifest_path = per_manifest_dir / f"{manifest['sha256']}.jsonl"
            write_jsonl(per_manifest_path, manifest_labels)
            per_manifest_outputs.append(
                {
                    "manifest_sha256": manifest["sha256"],
                    "labels_relative_path": str(per_manifest_path.relative_to(temporary)),
                    "labels_sha256": sha256_file(per_manifest_path),
                    "label_rows": len(manifest_labels),
                }
            )

        uncertain_labels = sum(
            row["objects"][name]["label"] == "uncertain"
            for row in labels
            for name in TARGET_OBJECTS
        )
        result = {
            "schema": RUN_SCHEMA,
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "claim_scope": (
                "task-specific frozen-initializer RGB teacher; not a general open-world detector"
            ),
            "runtime_data_boundary": (
                "registration + replay manifest JSONL + referenced JPEGs + registered "
                "SAM2 checkpoint/conversion code + sibling v242 mechanical replay-completion "
                "summary only; never opens the latent tape, referenced v121 source summary, "
                "simulator state, semantic/event/success/BDDL data"
            ),
            "registration": {
                "path": str(args.registration.resolve()),
                "sha256": frozen["sha256"],
                "expected_sha256_compiled_into_code": EXPECTED_REGISTRATION_SHA256,
            },
            "checkpoint": {
                "path": str(frozen["checkpoint"]),
                "sha256": frozen["payload"]["tracker"]["checkpoint_sha256"],
                "tensor_count": checkpoint_tensor_count,
                "strict_load": True,
                "dtype": "torch.bfloat16",
            },
            "registered_probe": {
                "path": str(frozen["probe_path"]),
                "sha256": frozen["payload"]["tracker"]["probe_code_sha256"],
            },
            "runtime": {
                "argv": sys.argv,
                "device": str(device),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "git_head": git_head(),
                "seconds_before_atomic_commit": time.perf_counter() - run_started,
            },
            "inputs": {
                "requested_episode_ids": sorted(requested_episode_ids),
                "episode_partitions": {
                    str(episode_id): frozen["episode_to_partition"][episode_id]
                    for episode_id in sorted(requested_episode_ids)
                },
                "manifests": manifest_audit,
                "verified_frame_rows": len(rows),
            },
            "label_rule": frozen["payload"]["visual_label_rule"],
            "prompt_policy": {
                "source": "fixed_t0_initializer in frozen registration",
                "prompt_frames": [0],
                "post_t0_prompts": 0,
                "temporal_smoothing": "none",
            },
            "outputs": {
                "labels_relative_path": "labels.jsonl",
                "labels_sha256": sha256_file(labels_path),
                "label_rows": len(labels),
                "object_labels": len(labels) * len(TARGET_OBJECTS),
                "uncertain_object_labels": uncertain_labels,
                "uncertain_fraction": uncertain_labels / (len(labels) * len(TARGET_OBJECTS)),
                "per_manifest": per_manifest_outputs,
                "episode_summaries": episode_summaries,
                "overlays_saved": bool(args.save_overlays),
            },
            "code": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        }
        result_path = temporary / "result.json"
        write_json(result_path, result)
        complete = {
            "schema": COMPLETE_SCHEMA,
            "status": "complete",
            "result_sha256": sha256_file(result_path),
            "labels_sha256": sha256_file(labels_path),
            "code_sha256": result["code"]["sha256"],
            "registration_sha256": frozen["sha256"],
        }
        write_json(temporary / "COMPLETE.json", complete)
        os.replace(temporary, output_dir)
        print(
            json.dumps(
                {
                    "output_dir": str(output_dir),
                    "result_sha256": complete["result_sha256"],
                    "labels_sha256": complete["labels_sha256"],
                    "episodes": episode_summaries,
                },
                indent=2,
                sort_keys=True,
            )
        )
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
