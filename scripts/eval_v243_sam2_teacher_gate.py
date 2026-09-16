#!/usr/bin/env python3
"""Fail-closed privileged evaluator for the frozen V243 SAM2 RGB teacher.

This program has an explicit two-phase boundary.

Phase A reads only the frozen registration, completed teacher artifacts, replay
manifests, and the RGB files named by those manifests.  It verifies the complete
SHA/provenance chain and writes an immutable ``pre_privileged_seal.json``.

Only after that seal exists and its SHA256 has been re-verified may Phase B hash
and open the deployment ``summary.json``.  Phase B uses only episode ``steps``
and first-achievement ``events``.  It never consults episode success.  The
result is an atomic PASS/STOP directory containing a detailed ``result.json``,
a minimal exact-contract ``GATE.json`` for the Phi trainer, and ``COMPLETE.json``.

The tracker gate is over exactly episodes 16--95.  A replay is an observable
prefix: trailing frames after an early termination are right-censored, whereas
an internal hole in the prefix is a contract error.  A place event is observable
for transition timing iff a recorded RGB frame exists at or after its event
step.  Events after the last recorded frame are disclosed as right-censored and
are not silently imputed.
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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


REPO = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRATION = REPO / "plan_and_progress" / "v242_chain3_sam2_registration.json"
EXPECTED_REGISTRATION_SHA256 = (
    "3d892616b97442a080e16f50f226217eb00dd3d2e19f6fb7e048750a251363ab"
)
EXPECTED_TEACHER_CODE_PATH = REPO / "scripts" / "track_v243_sam2_teacher.py"
EXPECTED_TEACHER_CODE_SHA256 = (
    "205c31169f014e9f5ee4dd607a68f359ae1a2d844ef93fbaf6d830660d407adf"
)

REGISTRATION_SCHEMA = "v242_chain3_sam2_registration_v1"
REPLAY_SCHEMA = "v242_rgb_replay_v1"
REPLAY_SUMMARY_SCHEMA = "v242_rgb_replay_summary_v1"
LEGACY_LABEL_SCHEMA = "v242_qwen_label_v1"
TEACHER_LABEL_SCHEMA = "v243_sam2_teacher_label_v1"
TEACHER_RUN_SCHEMA = "v243_sam2_teacher_run_v1"
TEACHER_COMPLETE_SCHEMA = "v243_sam2_teacher_complete_v1"

PRESEAL_SCHEMA = "v243_sam2_teacher_gate_pre_privileged_seal_v1"
RESULT_SCHEMA = "v243_sam2_teacher_gate_result_v1"
GATE_SCHEMA = "v243_sam2_teacher_gate_certificate_v1"
COMPLETE_SCHEMA = "v243_sam2_teacher_gate_complete_v1"

TASK = "chain3_lr2"
CAMERA = "agentview"
TAPE_SHA256 = "7dbff743ea89db5fd792d317111129d188d439122a46db7384f4549fd947f86c"
CHUNK_STEPS = 10
HORIZON_STEPS = 750
EXPECTED_EPISODES = tuple(range(16, 96))
TARGET_OBJECTS = ("alphabet_soup_1", "tomato_sauce_1", "cream_cheese_1")
PLACE_EVENT_INDEX = {
    "alphabet_soup_1": 1,
    "tomato_sauce_1": 3,
    "cream_cheese_1": 5,
}
HEX = frozenset("0123456789abcdef")

EXPECTED_GATE_THRESHOLDS = {
    "frame_coverage_gte": 0.95,
    "uncertain_fraction_lte": 0.05,
    "evaluation_only_framewise_place_accuracy_gte": 0.95,
    "evaluation_only_macro_f1_gte": 0.90,
    "positive_place_transition_median_absolute_error_steps_lte": 20,
    "never_placed_object_terminal_specificity_gte": 0.95,
    "thresholds_may_not_be_changed_after_unsealing": True,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_fields(mapping: dict[str, Any], fields: Iterable[str], where: str) -> None:
    missing = [field for field in fields if field not in mapping]
    if missing:
        raise ValueError(f"{where}: missing fields {missing}")


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value.lower()) <= HEX
        and value == value.lower()
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def parse_json(text: str, where: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite_json,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{where}: invalid strict JSON: {exc}") from exc


def load_json_strict(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON file: {path}")
    payload = parse_json(path.read_text(encoding="utf-8"), str(path))
    require(isinstance(payload, dict), f"{path}: expected a JSON object")
    return payload


def load_jsonl_strict(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"missing JSONL file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            require(bool(line.strip()), f"{path}:{line_number}: blank JSONL row")
            row = parse_json(line, f"{path}:{line_number}")
            require(isinstance(row, dict), f"{path}:{line_number}: expected object")
            rows.append(row)
    require(bool(rows), f"{path}: empty JSONL")
    return rows


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


def safe_child(root: Path, relative: Any, where: str) -> Path:
    require(isinstance(relative, str) and relative, f"{where}: invalid relative path")
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), f"{where}: path escapes artifact root")
    return path


def validate_registration(path: Path) -> dict[str, Any]:
    actual_sha = sha256_file(path)
    require(
        actual_sha == EXPECTED_REGISTRATION_SHA256,
        f"registration SHA mismatch: {actual_sha}",
    )
    registration = load_json_strict(path)
    require(registration.get("schema") == REGISTRATION_SCHEMA, "registration schema mismatch")
    require(
        registration.get("status")
        == "frozen_before_any_episode_16_to_95_tracking_output",
        "registration is not frozen at the expected boundary",
    )
    target = registration.get("target")
    require(isinstance(target, dict), "registration target must be an object")
    expected_target = {
        "task": TASK,
        "camera": CAMERA,
        "image_width": 360,
        "image_height": 360,
        "source_tape_sha256": TAPE_SHA256,
        "chunk_steps": CHUNK_STEPS,
        "horizon_steps": HORIZON_STEPS,
    }
    for field, expected in expected_target.items():
        require(target.get(field) == expected, f"registration target.{field} mismatch")

    partition = registration.get("episode_partition")
    require(isinstance(partition, dict), "registration episode_partition missing")
    expected_partition = {
        "tracker_and_initializer_calibration_episode_ids": list(range(0, 16)),
        "phi_fit_episode_ids": list(range(16, 64)),
        "phi_calibration_episode_ids": list(range(64, 80)),
        "phi_test_episode_ids": list(range(80, 96)),
        "environment_seed_rule": "env_seed = 8700 + episode_id",
        "disjoint": True,
    }
    require(partition == expected_partition, "registration partition differs from frozen split")
    episode_partition = {
        **{episode: "phi_fit" for episode in range(16, 64)},
        **{episode: "phi_calibration" for episode in range(64, 80)},
        **{episode: "phi_test" for episode in range(80, 96)},
    }

    thresholds = registration.get("tracker_promotion_gate_on_episodes_16_to_95")
    require(
        thresholds == EXPECTED_GATE_THRESHOLDS,
        "tracker promotion thresholds differ from frozen registration",
    )
    visual_rule = registration.get("visual_label_rule")
    require(isinstance(visual_rule, dict), "registration visual_label_rule missing")
    require(
        visual_rule.get("basket_relation")
        == "target-mask centroid lies inside the current tracked basket-mask bounding box",
        "registered relation rule mismatch",
    )
    require(visual_rule.get("temporal_smoothing") == "none", "smoothing must be none")
    require(
        visual_rule.get("labels") == ["inside", "outside", "uncertain"],
        "registered label set mismatch",
    )
    uncertainty = visual_rule.get("uncertain_if")
    expected_uncertainty = {
        "object_score_logit_lte": 0.0,
        "target_mask_area_px_lt": 64,
        "target_mask_area_px_gt": 5000,
        "basket_mask_area_px_lt": 3000,
        "basket_mask_area_px_gt": 15000,
        "nonfinite_output": True,
    }
    require(uncertainty == expected_uncertainty, "registered uncertainty rule mismatch")

    initializer = registration.get("fixed_t0_initializer")
    require(isinstance(initializer, dict), "fixed_t0_initializer missing")
    objects = initializer.get("objects")
    require(isinstance(objects, dict), "initializer objects missing")
    require(set(objects) == {"basket_1", *TARGET_OBJECTS}, "initializer object set mismatch")
    ids = {name: int(config["sam_object_id"]) for name, config in objects.items()}
    require(set(ids.values()) == {1, 2, 3, 4}, "initializer SAM IDs mismatch")

    tracker = registration.get("tracker")
    require(isinstance(tracker, dict), "registration tracker missing")
    require(is_sha256(tracker.get("checkpoint_sha256")), "invalid checkpoint SHA")
    require(is_sha256(tracker.get("probe_code_sha256")), "invalid probe SHA")
    probe_path = (REPO / str(tracker.get("probe_code_path", ""))).resolve()
    require(probe_path.is_file(), "registered probe code is missing")
    require(sha256_file(probe_path) == tracker["probe_code_sha256"],
            "registered probe code SHA mismatch")
    return {
        "payload": registration,
        "sha256": actual_sha,
        "target": target,
        "thresholds": thresholds,
        "uncertainty": uncertainty,
        "sam_ids": ids,
        "episode_partition": episode_partition,
        "probe_path": probe_path,
    }


def validate_image_record(record: Any, manifest_dir: Path, target: dict[str, Any],
                          where: str) -> dict[str, Any]:
    require(isinstance(record, dict), f"{where}: image record must be an object")
    require_fields(record, ("path", "sha256", "width", "height"), where)
    require(is_sha256(record["sha256"]), f"{where}: invalid image SHA")
    image_path = safe_child(manifest_dir, record["path"], where)
    require(image_path.is_file(), f"{where}: missing image")
    require(sha256_file(image_path) == record["sha256"], f"{where}: image SHA mismatch")
    width, height = int(record["width"]), int(record["height"])
    require(
        (width, height) == (target["image_width"], target["image_height"]),
        f"{where}: declared dimensions mismatch",
    )
    with Image.open(image_path) as opened:
        require(opened.size == (width, height), f"{where}: decoded dimensions mismatch")
        opened.verify()
    return {
        "path": str(image_path),
        "sha256": str(record["sha256"]),
        "width": width,
        "height": height,
    }


def canonical_relation(centroid: Any, bbox: Any) -> bool | None:
    if centroid is None or bbox is None:
        return None
    require(
        isinstance(centroid, list)
        and len(centroid) == 2
        and all(is_finite_number(value) for value in centroid),
        "invalid diagnostic centroid",
    )
    require(
        isinstance(bbox, list)
        and len(bbox) == 4
        and all(is_int(value) for value in bbox),
        "invalid diagnostic bbox",
    )
    x_value, y_value = (float(value) for value in centroid)
    x0, y0, x1, y1 = (int(value) for value in bbox)
    require(x0 < x1 and y0 < y1, "invalid diagnostic bbox extent")
    return bool(x0 <= x_value < x1 and y0 <= y_value < y1)


def diagnostic_scalar(value: Any, where: str) -> float | None:
    if value is None:
        return None
    require(is_finite_number(value), f"{where}: expected finite number or null")
    return float(value)


def validate_label_diagnostics(row: dict[str, Any], frozen: dict[str, Any],
                               where: str) -> None:
    require_fields(row, ("object_diagnostics", "basket_diagnostics"), where)
    object_diagnostics = row["object_diagnostics"]
    basket = row["basket_diagnostics"]
    require(
        isinstance(object_diagnostics, dict) and set(object_diagnostics) == set(TARGET_OBJECTS),
        f"{where}: object_diagnostics set mismatch",
    )
    require(isinstance(basket, dict), f"{where}: basket_diagnostics must be object")
    thresholds = frozen["uncertainty"]

    def validate_common(diag: dict[str, Any], name: str) -> tuple[int, float | None, bool]:
        require_fields(
            diag,
            (
                "sam_object_id", "mask_area_px", "mask_bbox_xyxy", "mask_centroid_xy",
                "object_score_logit", "finite_output", "uncertain_reasons",
            ),
            f"{where}:{name}",
        )
        require(int(diag["sam_object_id"]) == frozen["sam_ids"][name],
                f"{where}:{name}: SAM ID mismatch")
        area = diag["mask_area_px"]
        require(is_int(area) and area >= 0, f"{where}:{name}: invalid area")
        bbox, centroid = diag["mask_bbox_xyxy"], diag["mask_centroid_xy"]
        if area == 0:
            require(bbox is None and centroid is None,
                    f"{where}:{name}: empty mask must have null geometry")
        else:
            canonical_relation(centroid, bbox)
        score = diagnostic_scalar(diag["object_score_logit"], f"{where}:{name}:score")
        require(isinstance(diag["finite_output"], bool),
                f"{where}:{name}: finite_output must be boolean")
        require(
            isinstance(diag["uncertain_reasons"], list)
            and all(isinstance(value, str) for value in diag["uncertain_reasons"]),
            f"{where}:{name}: invalid uncertain_reasons",
        )
        require(
            len(diag["uncertain_reasons"]) == len(set(diag["uncertain_reasons"])),
            f"{where}:{name}: uncertainty reasons must be unique",
        )
        return int(area), score, bool(diag["finite_output"])

    basket_area, basket_score, basket_finite = validate_common(basket, "basket_1")
    basket_reasons: list[str] = []
    if not basket_finite:
        basket_reasons.append("nonfinite_output")
    if basket_score is None or basket_score <= thresholds["object_score_logit_lte"]:
        basket_reasons.append("basket_object_score_logit_lte")
    if basket_area < thresholds["basket_mask_area_px_lt"]:
        basket_reasons.append("basket_mask_area_px_lt")
    if basket_area > thresholds["basket_mask_area_px_gt"]:
        basket_reasons.append("basket_mask_area_px_gt")
    basket_reasons = sorted(set(basket_reasons))
    require(set(basket["uncertain_reasons"]) == set(basket_reasons),
            f"{where}: basket uncertainty reasons mismatch")
    require(basket.get("reliable") is (not basket_reasons),
            f"{where}: basket reliability mismatch")

    for name in TARGET_OBJECTS:
        diag = object_diagnostics[name]
        require(isinstance(diag, dict), f"{where}:{name}: diagnostic must be object")
        area, score, finite = validate_common(diag, name)
        relation = canonical_relation(diag["mask_centroid_xy"], basket["mask_bbox_xyxy"])
        require(diag.get("centroid_in_tracked_basket_bbox") is relation,
                f"{where}:{name}: relation mismatch")
        reasons = list(basket_reasons)
        if not finite:
            reasons.append("nonfinite_output")
        if score is None or score <= thresholds["object_score_logit_lte"]:
            reasons.append("object_score_logit_lte")
        if area < thresholds["target_mask_area_px_lt"]:
            reasons.append("target_mask_area_px_lt")
        if area > thresholds["target_mask_area_px_gt"]:
            reasons.append("target_mask_area_px_gt")
        reasons = sorted(set(reasons))
        expected_label = "uncertain" if reasons else ("inside" if relation else "outside")
        require(set(diag["uncertain_reasons"]) == set(reasons),
                f"{where}:{name}: uncertainty reasons mismatch")
        require(diag.get("label") == expected_label,
                f"{where}:{name}: diagnostic label mismatch")
        require(row["objects"][name]["label"] == expected_label,
                f"{where}:{name}: public label mismatch")


def validate_manifest(
    manifest_audit: dict[str, Any],
    requested_episodes: set[int],
    frozen: dict[str, Any],
    global_frame_ids: set[str],
    global_row_indices: set[int],
    global_episode_manifest: dict[int, str],
) -> tuple[list[dict[str, Any]], dict[str, Any], set[str]]:
    where = "teacher result manifest audit"
    require(isinstance(manifest_audit, dict), f"{where}: expected object")
    require_fields(manifest_audit, ("path", "sha256", "row_count", "episode_ids"), where)
    manifest_path = Path(str(manifest_audit["path"])).resolve()
    require(manifest_path.is_file(), f"missing replay manifest: {manifest_path}")
    manifest_sha = sha256_file(manifest_path)
    require(manifest_sha == manifest_audit["sha256"], "replay manifest SHA mismatch")
    require(is_sha256(manifest_sha), "invalid replay manifest SHA")
    source_rows = load_jsonl_strict(manifest_path)
    require(len(source_rows) == int(manifest_audit["row_count"]),
            f"{manifest_path}: row_count audit mismatch")
    manifest_dir = manifest_path.parent.resolve()
    rows: list[dict[str, Any]] = []
    episode_ids: set[int] = set()
    summary_hashes: set[str] = set()
    target = frozen["target"]
    for line_number, row in enumerate(source_rows, start=1):
        row_where = f"{manifest_path}:{line_number}"
        require_fields(
            row,
            (
                "schema", "frame_id", "task", "tape_sha256", "episode_id",
                "env_seed", "t", "row_index", "chunk_index", "sampling", "images",
                "summary_sha256",
            ),
            row_where,
        )
        require(row["schema"] == REPLAY_SCHEMA, f"{row_where}: schema mismatch")
        require(row["task"] == TASK, f"{row_where}: task mismatch")
        require(row["tape_sha256"] == TAPE_SHA256, f"{row_where}: tape SHA mismatch")
        episode_id = row["episode_id"]
        env_seed, t_value = row["env_seed"], row["t"]
        row_index, chunk_index = row["row_index"], row["chunk_index"]
        require(all(is_int(value) for value in (
            episode_id, env_seed, t_value, row_index, chunk_index
        )),
                f"{row_where}: integer field encoded with wrong type")
        require(episode_id in requested_episodes, f"{row_where}: unexpected episode")
        require(env_seed == 8700 + episode_id, f"{row_where}: seed rule mismatch")
        require(chunk_index >= 0 and t_value == chunk_index * CHUNK_STEPS,
                f"{row_where}: t/chunk mismatch")
        require(0 <= t_value < HORIZON_STEPS, f"{row_where}: t outside horizon")
        require(row_index >= 0 and row_index not in global_row_indices,
                f"{row_where}: duplicate/invalid row_index")
        expected_frame_id = f"{TAPE_SHA256[:12]}:ep{episode_id}:t{t_value}"
        require(row["frame_id"] == expected_frame_id, f"{row_where}: frame_id mismatch")
        require(row["frame_id"] not in global_frame_ids, f"{row_where}: duplicate frame_id")
        sampling = row["sampling"]
        require(isinstance(sampling, dict), f"{row_where}: sampling must be object")
        require(sampling.get("chunk_stride") == 1, f"{row_where}: stride must be one")
        require(sampling.get("pre_action") is True, f"{row_where}: pre_action must be true")
        images = row["images"]
        require(isinstance(images, dict) and CAMERA in images,
                f"{row_where}: registered camera missing")
        image = validate_image_record(images[CAMERA], manifest_dir, target,
                                      f"{row_where}:{CAMERA}")
        summary_sha = row["summary_sha256"]
        require(is_sha256(summary_sha), f"{row_where}: invalid declared summary SHA")
        summary_hashes.add(summary_sha)
        if episode_id in global_episode_manifest:
            require(global_episode_manifest[episode_id] == manifest_sha,
                    f"episode {episode_id} is split across manifests")
        global_episode_manifest[episode_id] = manifest_sha
        global_row_indices.add(row_index)
        global_frame_ids.add(str(row["frame_id"]))
        episode_ids.add(episode_id)
        rows.append(
            {
                "frame_id": str(row["frame_id"]),
                "task": TASK,
                "episode_id": episode_id,
                "env_seed": env_seed,
                "t": t_value,
                "row_index": row_index,
                "chunk_index": chunk_index,
                "manifest_path": str(manifest_path),
                "manifest_sha256": manifest_sha,
                "source_image_sha256": image["sha256"],
                "source_summary_sha256": summary_sha,
            }
        )
    declared_episode_ids = manifest_audit["episode_ids"]
    require(
        isinstance(declared_episode_ids, list)
        and all(is_int(value) for value in declared_episode_ids)
        and declared_episode_ids == sorted(episode_ids),
        f"{manifest_path}: episode_ids audit mismatch",
    )
    require(episode_ids <= requested_episodes, f"{manifest_path}: episode set mismatch")

    # The frozen teacher independently validates the replay's mechanical completion
    # summary without following/opening the privileged source-summary path.  Bind
    # that proof here as an immutable Phase-A artifact; never silently trust a bare
    # boolean copied into result.json.
    require(len(summary_hashes) == 1,
            f"{manifest_path}: rows must bind exactly one source summary SHA")
    replay_completion = manifest_audit.get("replay_completion")
    replay_where = f"{where}:replay_completion"
    require(isinstance(replay_completion, dict), f"{replay_where}: missing object")
    require_fields(
        replay_completion,
        (
            "path", "sha256", "schema", "mechanical_metadata_only",
            "forbidden_privileged_field_paths", "manifest_sha256",
            "source_summary_sha256_declared_only_not_opened", "episode_ids",
            "frames_saved", "chunks_replayed", "sampling", "proprio_validation",
            "episodes_with_unstored_terminal_steps", "unstored_terminal_env_steps",
        ),
        replay_where,
    )
    replay_path = Path(str(replay_completion["path"])).resolve()
    require(replay_path == (manifest_dir / "summary.json").resolve(),
            f"{replay_where}: path is not the manifest's mechanical summary")
    require(replay_path.is_file(), f"{replay_where}: artifact is missing")
    replay_sha = replay_completion["sha256"]
    require(is_sha256(replay_sha), f"{replay_where}: invalid SHA")
    require(sha256_file(replay_path) == replay_sha,
            f"{replay_where}: artifact SHA mismatch")
    require(replay_completion["schema"] == REPLAY_SUMMARY_SCHEMA,
            f"{replay_where}: schema mismatch")
    require(replay_completion["mechanical_metadata_only"] is True,
            f"{replay_where}: proof is not mechanical-only")
    require(replay_completion["forbidden_privileged_field_paths"] == [],
            f"{replay_where}: privileged fields were reported")
    require(replay_completion["manifest_sha256"] == manifest_sha,
            f"{replay_where}: manifest SHA mismatch")
    require(
        replay_completion["source_summary_sha256_declared_only_not_opened"]
        == next(iter(summary_hashes)),
        f"{replay_where}: source-summary declaration mismatch",
    )
    require(replay_completion["episode_ids"] == sorted(episode_ids),
            f"{replay_where}: episode IDs mismatch")
    require(replay_completion["frames_saved"] == len(rows),
            f"{replay_where}: frames_saved mismatch")
    require(replay_completion["chunks_replayed"] == len(rows),
            f"{replay_where}: chunks_replayed mismatch")
    require(replay_completion["sampling"] == {"chunk_stride": 1, "pre_action": True},
            f"{replay_where}: sampling mismatch")
    proprio = replay_completion["proprio_validation"]
    require(isinstance(proprio, dict), f"{replay_where}: proprio proof missing")
    require_fields(
        proprio,
        ("threshold_max_abs_error", "observed_max_abs_error", "frames_checked"),
        f"{replay_where}:proprio_validation",
    )
    require(
        is_finite_number(proprio["threshold_max_abs_error"])
        and float(proprio["threshold_max_abs_error"]) >= 0
        and is_finite_number(proprio["observed_max_abs_error"])
        and 0 <= float(proprio["observed_max_abs_error"])
        <= float(proprio["threshold_max_abs_error"])
        and proprio["frames_checked"] == len(rows),
        f"{replay_where}: invalid proprio proof",
    )
    episodes_with_unstored = replay_completion[
        "episodes_with_unstored_terminal_steps"
    ]
    unstored_steps = replay_completion["unstored_terminal_env_steps"]
    require(is_int(episodes_with_unstored) and is_int(unstored_steps),
            f"{replay_where}: terminal recovery counters must be integers")
    require(
        0 <= episodes_with_unstored <= len(episode_ids)
        and (
            (episodes_with_unstored == 0 and unstored_steps == 0)
            or (
                episodes_with_unstored > 0
                and episodes_with_unstored <= unstored_steps
                <= episodes_with_unstored * CHUNK_STEPS
            )
        ),
        f"{replay_where}: invalid terminal recovery counters",
    )
    return rows, {
        "path": str(manifest_path),
        "sha256": manifest_sha,
        "row_count": len(rows),
        "episode_ids": sorted(episode_ids),
        "source_summary_sha256": next(iter(summary_hashes)),
        "replay_completion": {
            "path": str(replay_path),
            "sha256": replay_sha,
            "manifest_sha256": manifest_sha,
            "mechanical_metadata_only": True,
            "episodes_with_unstored_terminal_steps": episodes_with_unstored,
            "unstored_terminal_env_steps": unstored_steps,
        },
    }, summary_hashes


def validate_label_row(row: dict[str, Any], frame: dict[str, Any],
                       frozen: dict[str, Any], checkpoint_sha: str,
                       where: str) -> None:
    require_fields(
        row,
        (
            "schema", "teacher_schema", "producer", "frame_id", "task",
            "replay_manifest_sha256", "parse_ok", "objects", "episode_id",
            "env_seed", "t", "row_index", "chunk_index", "registration_sha256",
            "checkpoint_sha256", "camera", "prompt_frame_index",
            "post_t0_prompt_count", "source_image_sha256", "object_diagnostics",
            "basket_diagnostics", "source_summary_sha256_declared_only",
        ),
        where,
    )
    require(row["schema"] == LEGACY_LABEL_SCHEMA, f"{where}: legacy schema mismatch")
    require(row["teacher_schema"] == TEACHER_LABEL_SCHEMA,
            f"{where}: teacher schema mismatch")
    require(row["producer"] == "frozen_registration_sam2_rgb_teacher",
            f"{where}: producer mismatch")
    require(row["parse_ok"] is True, f"{where}: parse_ok must be true")
    expected = {
        "frame_id": frame["frame_id"],
        "task": TASK,
        "replay_manifest_sha256": frame["manifest_sha256"],
        "episode_id": frame["episode_id"],
        "env_seed": frame["env_seed"],
        "t": frame["t"],
        "row_index": frame["row_index"],
        "chunk_index": frame["chunk_index"],
        "registration_sha256": frozen["sha256"],
        "checkpoint_sha256": checkpoint_sha,
        "camera": CAMERA,
        "prompt_frame_index": 0,
        "post_t0_prompt_count": 0,
        "source_image_sha256": frame["source_image_sha256"],
        "source_summary_sha256_declared_only": frame["source_summary_sha256"],
    }
    for field, value in expected.items():
        require(row[field] == value, f"{where}: {field} mismatch")
    objects = row["objects"]
    require(isinstance(objects, dict) and set(objects) == set(TARGET_OBJECTS),
            f"{where}: object set mismatch")
    for name in TARGET_OBJECTS:
        require(
            isinstance(objects[name], dict)
            and set(objects[name]) == {"label"}
            and objects[name]["label"] in {"inside", "outside", "uncertain"},
            f"{where}:{name}: invalid public label object",
        )
    validate_label_diagnostics(row, frozen, where)


def validate_teacher_run(
    run_dir: Path,
    frozen: dict[str, Any],
    global_frame_ids: set[str],
    global_row_indices: set[int],
    global_episode_manifest: dict[int, str],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], set[str]]:
    run_dir = run_dir.resolve()
    require(run_dir.is_dir(), f"teacher run is not a directory: {run_dir}")
    complete_path = run_dir / "COMPLETE.json"
    result_path = run_dir / "result.json"
    complete = load_json_strict(complete_path)
    result = load_json_strict(result_path)
    require(complete.get("schema") == TEACHER_COMPLETE_SCHEMA,
            f"{complete_path}: schema mismatch")
    require(complete.get("status") == "complete", f"{complete_path}: incomplete")
    result_sha = sha256_file(result_path)
    require(complete.get("result_sha256") == result_sha,
            f"{run_dir}: result SHA not sealed")
    require(result.get("schema") == TEACHER_RUN_SCHEMA, f"{result_path}: schema mismatch")
    require(result.get("status") == "complete", f"{result_path}: incomplete run")
    require(result.get("registration", {}).get("sha256") == frozen["sha256"],
            f"{result_path}: registration SHA mismatch")
    require(complete.get("registration_sha256") == frozen["sha256"],
            f"{complete_path}: registration SHA mismatch")

    checkpoint_sha = frozen["payload"]["tracker"]["checkpoint_sha256"]
    probe_sha = frozen["payload"]["tracker"]["probe_code_sha256"]
    require(result.get("checkpoint", {}).get("sha256") == checkpoint_sha,
            f"{result_path}: checkpoint SHA mismatch")
    require(result.get("checkpoint", {}).get("strict_load") is True,
            f"{result_path}: checkpoint was not strict-loaded")
    require(result.get("checkpoint", {}).get("tensor_count") == 903,
            f"{result_path}: checkpoint tensor count mismatch")
    require(result.get("checkpoint", {}).get("dtype") == "torch.bfloat16",
            f"{result_path}: checkpoint dtype mismatch")
    require(result.get("registered_probe", {}).get("sha256") == probe_sha,
            f"{result_path}: registered probe SHA mismatch")
    require(
        Path(str(result.get("registered_probe", {}).get("path", ""))).resolve()
        == frozen["probe_path"],
        f"{result_path}: registered probe path mismatch",
    )

    code = result.get("code")
    require(isinstance(code, dict), f"{result_path}: missing code provenance")
    require_fields(code, ("path", "sha256"), f"{result_path}:code")
    code_path = Path(str(code["path"])).resolve()
    require(code_path == EXPECTED_TEACHER_CODE_PATH.resolve(),
            f"{result_path}: teacher code path is not the frozen implementation")
    require(code_path.is_file(), f"{result_path}: frozen teacher code missing")
    require(code["sha256"] == EXPECTED_TEACHER_CODE_SHA256,
            f"{result_path}: teacher code is not the frozen SHA")
    require(sha256_file(code_path) == EXPECTED_TEACHER_CODE_SHA256,
            f"{result_path}: frozen teacher code SHA mismatch")
    require(complete.get("code_sha256") == code["sha256"],
            f"{complete_path}: code SHA mismatch")

    inputs = result.get("inputs")
    require(isinstance(inputs, dict), f"{result_path}: inputs missing")
    requested = inputs.get("requested_episode_ids")
    require(
        isinstance(requested, list)
        and requested
        and all(is_int(value) for value in requested)
        and requested == sorted(set(requested))
        and set(requested) <= set(EXPECTED_EPISODES),
        f"{result_path}: invalid requested episodes",
    )
    declared_partitions = inputs.get("episode_partitions")
    require(isinstance(declared_partitions, dict), f"{result_path}: partitions missing")
    expected_partitions = {
        str(episode): frozen["episode_partition"][episode] for episode in requested
    }
    require(declared_partitions == expected_partitions,
            f"{result_path}: registered partition mismatch")
    manifest_audits = inputs.get("manifests")
    require(isinstance(manifest_audits, list) and manifest_audits,
            f"{result_path}: manifests missing")

    frames: list[dict[str, Any]] = []
    normalized_manifests: list[dict[str, Any]] = []
    summary_hashes: set[str] = set()
    for audit in manifest_audits:
        current, normalized, declared = validate_manifest(
            audit,
            set(requested),
            frozen,
            global_frame_ids,
            global_row_indices,
            global_episode_manifest,
        )
        frames.extend(current)
        normalized_manifests.append(normalized)
        summary_hashes.update(declared)
    require({frame["episode_id"] for frame in frames} == set(requested),
            f"{result_path}: manifests do not cover requested episodes")
    require(inputs.get("verified_frame_rows") == len(frames),
            f"{result_path}: verified_frame_rows mismatch")

    by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        by_episode[frame["episode_id"]].append(frame)
    for episode, episode_frames in by_episode.items():
        episode_frames.sort(key=lambda item: item["t"])
        require(len(episode_frames) <= HORIZON_STEPS // CHUNK_STEPS,
                f"episode {episode}: too many frames")
        times = [item["t"] for item in episode_frames]
        require(times == list(range(0, len(times) * CHUNK_STEPS, CHUNK_STEPS)),
                f"episode {episode}: replay is not a contiguous observable prefix")

    outputs = result.get("outputs")
    require(isinstance(outputs, dict), f"{result_path}: outputs missing")
    labels_path = safe_child(run_dir, outputs.get("labels_relative_path"),
                             f"{result_path}:labels")
    labels_sha = sha256_file(labels_path)
    require(outputs.get("labels_sha256") == labels_sha,
            f"{result_path}: labels SHA mismatch")
    require(complete.get("labels_sha256") == labels_sha,
            f"{complete_path}: labels SHA mismatch")
    labels = load_jsonl_strict(labels_path)
    require(outputs.get("label_rows") == len(labels), f"{result_path}: label_rows mismatch")

    frames_by_id = {frame["frame_id"]: frame for frame in frames}
    require(len(frames_by_id) == len(frames), f"{result_path}: duplicate frames")
    labels_by_id: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(labels, start=1):
        where = f"{labels_path}:{line_number}"
        frame_id = row.get("frame_id")
        require(isinstance(frame_id, str) and frame_id in frames_by_id,
                f"{where}: label has no replay frame")
        require(frame_id not in labels_by_id, f"{where}: duplicate label frame_id")
        validate_label_row(row, frames_by_id[frame_id], frozen, checkpoint_sha, where)
        labels_by_id[frame_id] = row
    require(set(labels_by_id) == set(frames_by_id),
            f"{labels_path}: missing or extra labels are fatal")

    uncertain_count = sum(
        row["objects"][name]["label"] == "uncertain"
        for row in labels
        for name in TARGET_OBJECTS
    )
    require(outputs.get("object_labels") == len(labels) * len(TARGET_OBJECTS),
            f"{result_path}: object_labels mismatch")
    require(outputs.get("uncertain_object_labels") == uncertain_count,
            f"{result_path}: uncertain count mismatch")
    require(
        math.isclose(
            float(outputs.get("uncertain_fraction", -1)),
            uncertain_count / (len(labels) * len(TARGET_OBJECTS)),
            rel_tol=0,
            abs_tol=1e-15,
        ),
        f"{result_path}: uncertain fraction mismatch",
    )

    per_manifest = outputs.get("per_manifest")
    require(isinstance(per_manifest, list), f"{result_path}: per_manifest missing")
    declared_per_manifest = {item["manifest_sha256"]: item for item in per_manifest}
    require(len(declared_per_manifest) == len(per_manifest),
            f"{result_path}: duplicate per_manifest declaration")
    require(set(declared_per_manifest) == {item["sha256"] for item in normalized_manifests},
            f"{result_path}: per_manifest set mismatch")
    for manifest in normalized_manifests:
        declaration = declared_per_manifest[manifest["sha256"]]
        per_path = safe_child(run_dir, declaration.get("labels_relative_path"),
                              f"{result_path}:per_manifest")
        require(sha256_file(per_path) == declaration.get("labels_sha256"),
                f"{per_path}: SHA mismatch")
        per_rows = load_jsonl_strict(per_path)
        expected_rows = [
            row for row in labels if row["replay_manifest_sha256"] == manifest["sha256"]
        ]
        require(per_rows == expected_rows, f"{per_path}: rows differ from combined labels")
        require(declaration.get("label_rows") == len(expected_rows),
                f"{per_path}: label_rows mismatch")

    labels_with_frame = []
    for frame_id, frame in frames_by_id.items():
        labels_with_frame.append({**frame, "label_row": labels_by_id[frame_id]})
    labels_with_frame.sort(key=lambda item: (item["episode_id"], item["t"]))
    artifact = {
        "run_path": str(run_dir),
        "complete_sha256": sha256_file(complete_path),
        "result_sha256": result_sha,
        "labels_sha256": labels_sha,
        "code_sha256": str(code["sha256"]),
        "checkpoint_sha256": checkpoint_sha,
        "registration_sha256": frozen["sha256"],
        "manifest_sha256s": sorted(item["sha256"] for item in normalized_manifests),
    }
    return labels_with_frame, artifact, normalized_manifests, summary_hashes


def validate_unprivileged_inputs(registration_path: Path,
                                 teacher_runs: list[Path]) -> dict[str, Any]:
    frozen = validate_registration(registration_path.resolve())
    require(bool(teacher_runs), "at least one --teacher-run is required")
    resolved = [path.resolve() for path in teacher_runs]
    require(len(resolved) == len(set(resolved)), "duplicate teacher run path")
    global_frame_ids: set[str] = set()
    global_row_indices: set[int] = set()
    global_episode_manifest: dict[int, str] = {}
    all_frames: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    summary_hashes: set[str] = set()
    seen_episodes: set[int] = set()
    for run_dir in resolved:
        frames, artifact, run_manifests, declared = validate_teacher_run(
            run_dir,
            frozen,
            global_frame_ids,
            global_row_indices,
            global_episode_manifest,
        )
        run_episodes = {frame["episode_id"] for frame in frames}
        require(not (seen_episodes & run_episodes), "teacher runs overlap episodes")
        seen_episodes.update(run_episodes)
        all_frames.extend(frames)
        artifacts.append(artifact)
        manifests.extend(run_manifests)
        summary_hashes.update(declared)
    require(seen_episodes == set(EXPECTED_EPISODES),
            f"teacher episodes must be exactly 16..95, got {sorted(seen_episodes)}")
    require(len(summary_hashes) == 1,
            f"all replay rows must bind one source summary SHA, got {summary_hashes}")
    require(len({item["sha256"] for item in manifests}) == len(manifests),
            "duplicate replay manifest SHA across teacher runs")
    require(len({item["code_sha256"] for item in artifacts}) == 1,
            "teacher code drift across runs")
    all_frames.sort(key=lambda item: (item["episode_id"], item["t"]))
    artifacts.sort(key=lambda item: item["run_path"])
    manifests.sort(key=lambda item: item["sha256"])
    return {
        "frozen": frozen,
        "frames": all_frames,
        "teacher_artifacts": artifacts,
        "manifests": manifests,
        "declared_summary_sha256": next(iter(summary_hashes)),
    }


def write_pre_privileged_seal(temporary: Path, phase_a: dict[str, Any],
                              evaluator_code_sha: str) -> tuple[Path, str]:
    frames = phase_a["frames"]
    payload = {
        "schema": PRESEAL_SCHEMA,
        "status": "sealed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "privileged_summary_opened": False,
        "registration_sha256": phase_a["frozen"]["sha256"],
        "frozen_thresholds": phase_a["frozen"]["thresholds"],
        "episode_ids": list(EXPECTED_EPISODES),
        "observable_frame_rows": len(frames),
        "teacher_label_rows": len(frames),
        "manifest_sha256s": sorted(item["sha256"] for item in phase_a["manifests"]),
        "replay_completion_artifacts": [
            item["replay_completion"] for item in phase_a["manifests"]
        ],
        "teacher_artifacts": phase_a["teacher_artifacts"],
        "declared_source_summary_sha256": phase_a["declared_summary_sha256"],
        "evaluator_code_sha256": evaluator_code_sha,
    }
    path = temporary / "pre_privileged_seal.json"
    write_json(path, payload)
    digest = sha256_file(path)
    require(sha256_file(path) == digest, "pre-privileged seal was not stable")
    return path, digest


def load_privileged_summary(summary_path: Path, expected_sha: str,
                            preseal_path: Path, preseal_sha: str) -> tuple[dict[str, Any], str]:
    # This guard must run before even hashing the privileged file.
    require(preseal_path.is_file(), "privileged summary access attempted before pre-seal")
    require(sha256_file(preseal_path) == preseal_sha,
            "pre-privileged seal changed before summary access")
    actual_sha = sha256_file(summary_path)
    require(actual_sha == expected_sha,
            f"source summary SHA mismatch: {actual_sha} != {expected_sha}")
    return load_json_strict(summary_path), actual_sha


def validate_summary(summary: dict[str, Any], frames: list[dict[str, Any]]) -> dict[
    int, dict[str, Any]
]:
    require(summary.get("task") == TASK, "source summary task mismatch")
    require(summary.get("c") == CHUNK_STEPS, "source summary chunk mismatch")
    require(summary.get("episodes") == 96, "source summary must contain 96 episodes")
    raw_records = summary.get("episode_records")
    require(isinstance(raw_records, list) and len(raw_records) == 96,
            "source summary episode_records mismatch")
    records: dict[int, dict[str, Any]] = {}
    for position, raw in enumerate(raw_records):
        where = f"source summary episode_records[{position}]"
        require(isinstance(raw, dict), f"{where}: expected object")
        require_fields(raw, ("idx", "seed", "steps", "events"), where)
        episode, seed, steps = raw["idx"], raw["seed"], raw["steps"]
        require(all(is_int(value) for value in (episode, seed, steps)),
                f"{where}: invalid integer field")
        require(0 <= episode < 96 and episode not in records,
                f"{where}: duplicate/out-of-range episode")
        require(seed == 8700 + episode, f"{where}: seed rule mismatch")
        require(0 < steps <= HORIZON_STEPS, f"{where}: invalid steps")
        raw_events = raw["events"]
        require(isinstance(raw_events, dict), f"{where}: events must be object")
        events: dict[int, int] = {}
        for raw_key, raw_step in raw_events.items():
            require(isinstance(raw_key, str) and raw_key == str(int(raw_key)),
                    f"{where}: non-canonical event key")
            event_index = int(raw_key)
            require(event_index not in events and 0 <= event_index <= 5,
                    f"{where}: duplicate/out-of-range event index")
            require(is_int(raw_step) and 0 <= raw_step <= steps,
                    f"{where}: invalid event step")
            events[event_index] = raw_step
        records[episode] = {"seed": seed, "steps": steps, "events": events}
    require(set(records) == set(range(96)), "source summary episode IDs must be 0..95")

    by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        by_episode[frame["episode_id"]].append(frame)
    require(set(by_episode) == set(EXPECTED_EPISODES), "frames do not cover ep16..95")
    for episode in EXPECTED_EPISODES:
        current = sorted(by_episode[episode], key=lambda item: item["t"])
        require(current[-1]["t"] < records[episode]["steps"],
                f"episode {episode}: frame is not pre-terminal/pre-action")
        require(current[0]["env_seed"] == records[episode]["seed"],
                f"episode {episode}: frame/summary seed mismatch")
        expected_chunk_starts = math.ceil(records[episode]["steps"] / CHUNK_STEPS)
        require(
            0 <= expected_chunk_starts - len(current) <= 1,
            f"episode {episode}: observable prefix loses more than the one "
            "unrecoverable terminal chunk",
        )
    return records


def first_certain_outside_to_inside(rows: list[dict[str, Any]], name: str) -> int | None:
    previous: str | None = None
    for row in rows:
        label = row["label_row"]["objects"][name]["label"]
        if label == "uncertain":
            continue
        if label == "inside" and previous == "outside":
            return int(row["t"])
        previous = label
    return None


def compute_metrics(frames: list[dict[str, Any]], records: dict[int, dict[str, Any]],
                    expected_episodes: Iterable[int]) -> dict[str, Any]:
    episode_ids = tuple(expected_episodes)
    by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        by_episode[frame["episode_id"]].append(frame)
    for rows in by_episode.values():
        rows.sort(key=lambda item: item["t"])

    last_t_by_episode = {
        episode: by_episode[episode][-1]["t"] for episode in episode_ids
    }
    right_censored_pairs = {
        (episode, name)
        for episode in episode_ids
        for name in TARGET_OBJECTS
        if (
            (event_step := records[episode]["events"].get(PLACE_EVENT_INDEX[name]))
            is not None
            and event_step > last_t_by_episode[episode]
        )
    }

    confusion = {
        name: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "uncertain": 0,
               "certain_support": 0}
        for name in TARGET_OBJECTS
    }
    correct_certain = 0
    certain_total = 0
    all_object_labels = len(frames) * len(TARGET_OBJECTS)
    uncertain_total = 0
    evaluable_uncertain_total = 0
    excluded_censored_frame_labels = 0
    for frame in frames:
        events = records[frame["episode_id"]]["events"]
        for name in TARGET_OBJECTS:
            label = frame["label_row"]["objects"][name]["label"]
            if label == "uncertain":
                # The uncertainty gate deliberately covers every teacher output,
                # including object-episodes later excluded as right-censored.
                uncertain_total += 1
            if (frame["episode_id"], name) in right_censored_pairs:
                excluded_censored_frame_labels += 1
                continue
            truth = int(
                PLACE_EVENT_INDEX[name] in events
                and events[PLACE_EVENT_INDEX[name]] <= frame["t"]
            )
            cell = confusion[name]
            if label == "uncertain":
                cell["uncertain"] += 1
                evaluable_uncertain_total += 1
                continue
            prediction = int(label == "inside")
            cell["certain_support"] += 1
            certain_total += 1
            correct_certain += int(prediction == truth)
            if prediction and truth:
                cell["tp"] += 1
            elif prediction and not truth:
                cell["fp"] += 1
            elif not prediction and truth:
                cell["fn"] += 1
            else:
                cell["tn"] += 1

    per_object_f1: dict[str, float | None] = {}
    for name, cell in confusion.items():
        if cell["certain_support"] == 0:
            per_object_f1[name] = None
            continue
        denominator = 2 * cell["tp"] + cell["fp"] + cell["fn"]
        per_object_f1[name] = (
            1.0 if denominator == 0 else 2 * cell["tp"] / denominator
        )
    f1_values = [value for value in per_object_f1.values() if value is not None]
    macro_f1 = float(np.mean(f1_values)) if len(f1_values) == len(TARGET_OBJECTS) else None
    accuracy = correct_certain / certain_total if certain_total else None
    evaluable_object_labels = all_object_labels - excluded_censored_frame_labels
    effective_accuracy = (
        correct_certain / evaluable_object_labels if evaluable_object_labels else None
    )
    uncertain_fraction = uncertain_total / all_object_labels if all_object_labels else None

    transition_rows: list[dict[str, Any]] = []
    matched_errors: list[int] = []
    observable_positive_events = 0
    missing_observable_transitions = 0
    right_censored_positive_events = 0
    for episode in episode_ids:
        rows = by_episode[episode]
        last_t = rows[-1]["t"]
        events = records[episode]["events"]
        for name in TARGET_OBJECTS:
            event_step = events.get(PLACE_EVENT_INDEX[name])
            if event_step is None:
                continue
            predicted = first_certain_outside_to_inside(rows, name)
            if event_step > last_t:
                right_censored_positive_events += 1
                status = "right_censored_after_last_observed_frame"
                error = None
            else:
                observable_positive_events += 1
                if predicted is None:
                    missing_observable_transitions += 1
                    status = "observable_event_missing_predicted_transition"
                    error = None
                else:
                    error = abs(int(predicted) - int(event_step))
                    matched_errors.append(error)
                    status = "matched"
            transition_rows.append(
                {
                    "episode_id": episode,
                    "object": name,
                    "event_step": event_step,
                    "last_observed_t": last_t,
                    "predicted_transition_t": predicted,
                    "absolute_error_steps": error,
                    "status": status,
                }
            )
    transition_mae_median: float | None
    if observable_positive_events == 0 or missing_observable_transitions:
        transition_mae_median = None
    else:
        require(len(matched_errors) == observable_positive_events,
                "internal transition accounting mismatch")
        transition_mae_median = float(np.median(np.asarray(matched_errors, dtype=float)))

    terminal_rows: list[dict[str, Any]] = []
    terminal_denominator = 0
    terminal_outside = 0
    for episode in episode_ids:
        rows = by_episode[episode]
        events = records[episode]["events"]
        for name in TARGET_OBJECTS:
            if PLACE_EVENT_INDEX[name] in events:
                continue
            terminal_denominator += 1
            label = rows[-1]["label_row"]["objects"][name]["label"]
            is_outside = label == "outside"
            terminal_outside += int(is_outside)
            terminal_rows.append(
                {
                    "episode_id": episode,
                    "object": name,
                    "last_observed_t": rows[-1]["t"],
                    "terminal_observable_label": label,
                    "counts_as_true_negative": is_outside,
                }
            )
    specificity = (
        terminal_outside / terminal_denominator if terminal_denominator else None
    )
    expected_source_chunk_frames = sum(
        math.ceil(records[episode]["steps"] / CHUNK_STEPS) for episode in episode_ids
    )
    source_chunk_observation_fraction = (
        len(frames) / expected_source_chunk_frames
        if expected_source_chunk_frames else None
    )
    # Label/manifest bijection is a Phase-A contract.  This is the registered
    # frame-coverage gate denominator: only RGB rows which are actually observable.
    teacher_to_manifest_coverage = 1.0 if frames else None
    frame_coverage = teacher_to_manifest_coverage
    horizon_observation_fraction = len(frames) / (
        len(episode_ids) * (HORIZON_STEPS // CHUNK_STEPS)
    )
    return {
        "coverage": {
            "observable_manifest_frames": len(frames),
            "valid_teacher_frames": len(frames),
            "teacher_to_manifest_coverage": teacher_to_manifest_coverage,
            "expected_source_chunk_start_frames_from_privileged_steps": (
                expected_source_chunk_frames
            ),
            "source_chunk_observation_fraction": source_chunk_observation_fraction,
            "frame_coverage": frame_coverage,
            "registered_horizon_frames": len(episode_ids)
            * (HORIZON_STEPS // CHUNK_STEPS),
            "horizon_observation_fraction": horizon_observation_fraction,
            "note": (
                "gate frame_coverage is valid teacher labels divided by observable "
                "manifest RGB rows and is fail-closed by label/manifest bijection; "
                "source-chunk and registered-horizon observation fractions are reported "
                "separately, so an unrecoverable terminal chunk is right-censoring, not "
                "a missing teacher label"
            ),
        },
        "classification": {
            "certain_object_labels": certain_total,
            "uncertain_object_labels": uncertain_total,
            "evaluable_uncertain_object_labels": evaluable_uncertain_total,
            "all_object_labels": all_object_labels,
            "evaluable_object_labels": evaluable_object_labels,
            "excluded_censored_frame_labels": excluded_censored_frame_labels,
            "uncertain_fraction": uncertain_fraction,
            "framewise_place_accuracy_certain_only": accuracy,
            "effective_accuracy_uncertain_counted_incorrect": effective_accuracy,
            "per_object_confusion": confusion,
            "per_object_inside_positive_f1": per_object_f1,
            "macro_inside_positive_f1": macro_f1,
            "empty_positive_f1_convention": (
                "1.0 iff confident support exists and TP=FP=FN=0"
            ),
        },
        "transitions": {
            "ground_truth": "sticky first-achievement place events 1/3/5",
            "prediction": "first certain outside-to-inside transition; uncertain skipped",
            "observable_positive_events": observable_positive_events,
            "matched_observable_positive_events": len(matched_errors),
            "missing_observable_transitions": missing_observable_transitions,
            "right_censored_positive_events": right_censored_positive_events,
            "median_absolute_error_steps": transition_mae_median,
            "rows": transition_rows,
        },
        "terminal_specificity": {
            "definition": (
                "among object-episodes with no place event, only an outside label at "
                "the last observable frame counts as a true negative; uncertain is failure"
            ),
            "denominator": terminal_denominator,
            "true_negatives": terminal_outside,
            "specificity": specificity,
            "rows": terminal_rows,
        },
    }


def finite_compare(value: Any, operation: str, threshold: float) -> bool:
    if not is_finite_number(value):
        return False
    if operation == "gte":
        return float(value) >= threshold
    if operation == "lte":
        return float(value) <= threshold
    raise ValueError(operation)


def gate_decision(metrics: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    coverage = metrics["coverage"]["frame_coverage"]
    classification = metrics["classification"]
    transitions = metrics["transitions"]
    terminal = metrics["terminal_specificity"]
    values = {
        "frame_coverage": coverage,
        "uncertain_fraction": classification["uncertain_fraction"],
        "evaluation_only_framewise_place_accuracy": (
            classification["framewise_place_accuracy_certain_only"]
        ),
        "evaluation_only_macro_f1": classification["macro_inside_positive_f1"],
        "positive_place_transition_median_absolute_error_steps": (
            transitions["median_absolute_error_steps"]
        ),
        "never_placed_object_terminal_specificity": terminal["specificity"],
    }
    checks = {
        "frame_coverage_gte": finite_compare(
            values["frame_coverage"], "gte", thresholds["frame_coverage_gte"]
        ),
        "uncertain_fraction_lte": finite_compare(
            values["uncertain_fraction"], "lte", thresholds["uncertain_fraction_lte"]
        ),
        "evaluation_only_framewise_place_accuracy_gte": finite_compare(
            values["evaluation_only_framewise_place_accuracy"], "gte",
            thresholds["evaluation_only_framewise_place_accuracy_gte"],
        ),
        "evaluation_only_macro_f1_gte": finite_compare(
            values["evaluation_only_macro_f1"], "gte",
            thresholds["evaluation_only_macro_f1_gte"],
        ),
        "positive_place_transition_median_absolute_error_steps_lte": finite_compare(
            values["positive_place_transition_median_absolute_error_steps"], "lte",
            thresholds["positive_place_transition_median_absolute_error_steps_lte"],
        ),
        "never_placed_object_terminal_specificity_gte": finite_compare(
            values["never_placed_object_terminal_specificity"], "gte",
            thresholds["never_placed_object_terminal_specificity_gte"],
        ),
    }
    # A median over a cherry-picked subset is forbidden: any observable event
    # without a predicted certain transition makes the transition metric invalid.
    checks["all_observable_positive_transitions_matched"] = (
        transitions["observable_positive_events"] > 0
        and transitions["missing_observable_transitions"] == 0
        and transitions["matched_observable_positive_events"]
        == transitions["observable_positive_events"]
    )
    checks["terminal_specificity_has_support"] = terminal["denominator"] > 0
    passed = all(checks.values())
    return {
        "thresholds": thresholds,
        "values": values,
        "checks": checks,
        "passed": passed,
        "decision": "PASS" if passed else "STOP",
    }


def evaluate_atomic(registration_path: Path, teacher_runs: list[Path],
                    source_summary: Path, output_dir: Path) -> tuple[Path, dict[str, Any]]:
    output_dir = output_dir.resolve()
    require(not output_dir.exists(), f"output path already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    ).resolve()
    try:
        evaluator_code_sha = sha256_file(Path(__file__).resolve())
        phase_a = validate_unprivileged_inputs(registration_path, teacher_runs)
        preseal_path, preseal_sha = write_pre_privileged_seal(
            temporary, phase_a, evaluator_code_sha
        )
        summary, summary_sha = load_privileged_summary(
            source_summary.resolve(),
            phase_a["declared_summary_sha256"],
            preseal_path,
            preseal_sha,
        )
        records = validate_summary(summary, phase_a["frames"])
        metrics = compute_metrics(phase_a["frames"], records, EXPECTED_EPISODES)
        metrics["by_registered_partition"] = {
            partition_name: compute_metrics(
                [
                    frame for frame in phase_a["frames"]
                    if frame["episode_id"] in set(episode_ids)
                ],
                records,
                episode_ids,
            )
            for partition_name, episode_ids in (
                ("phi_fit", tuple(range(16, 64))),
                ("phi_calibration", tuple(range(64, 80))),
                ("phi_test", tuple(range(80, 96))),
            )
        }
        gate = gate_decision(metrics, phase_a["frozen"]["thresholds"])

        result = {
            "schema": RESULT_SCHEMA,
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "passed": gate["passed"],
            "decision": gate["decision"],
            "scientific_scope": (
                "privileged evaluation of the frozen task-specific SAM2 teacher on "
                "exactly chain3_lr2 episodes 16..95"
            ),
            "phase_boundary": {
                "pre_privileged_seal_path": "pre_privileged_seal.json",
                "pre_privileged_seal_sha256": preseal_sha,
                "summary_hashed_and_opened_only_after_preseal_reverification": True,
                "success_field_read_or_used": False,
            },
            "registration": {
                "path": str(registration_path.resolve()),
                "sha256": phase_a["frozen"]["sha256"],
            },
            "source_summary": {
                "path": str(source_summary.resolve()),
                "sha256": summary_sha,
                "used_fields": ["task", "c", "episodes", "episode_records.idx",
                                "episode_records.seed", "episode_records.steps",
                                "episode_records.events"],
            },
            "episode_ids": list(EXPECTED_EPISODES),
            "teacher_artifacts": phase_a["teacher_artifacts"],
            "manifest_sha256s": sorted(item["sha256"] for item in phase_a["manifests"]),
            "metrics": metrics,
            "gate": gate,
            "evaluator": {
                "path": str(Path(__file__).resolve()),
                "code_sha256": evaluator_code_sha,
                "git_head": git_head(),
                "argv": list(sys.argv),
            },
        }
        result_path = temporary / "result.json"
        write_json(result_path, result)
        result_sha = sha256_file(result_path)

        # Exact-key trainer contract: privileged details stay in result.json.
        teacher_artifacts = [
            {
                "run_path": item["run_path"],
                "result_sha256": item["result_sha256"],
                "labels_sha256": item["labels_sha256"],
                "code_sha256": item["code_sha256"],
                "checkpoint_sha256": item["checkpoint_sha256"],
                "registration_sha256": item["registration_sha256"],
                "manifest_sha256s": item["manifest_sha256s"],
            }
            for item in phase_a["teacher_artifacts"]
        ]
        certificate = {
            "schema": GATE_SCHEMA,
            "status": "complete",
            "passed": gate["passed"],
            "decision": gate["decision"],
            "registration_sha256": phase_a["frozen"]["sha256"],
            "source_summary_sha256": summary_sha,
            "episode_ids": list(EXPECTED_EPISODES),
            "manifest_sha256s": sorted(item["sha256"] for item in phase_a["manifests"]),
            "teacher_artifacts": teacher_artifacts,
            "pre_privileged_seal_sha256": preseal_sha,
            "result_sha256": result_sha,
            "evaluator_code_sha256": evaluator_code_sha,
        }
        require(
            set(certificate)
            == {
                "schema", "status", "passed", "decision", "registration_sha256",
                "source_summary_sha256", "episode_ids", "manifest_sha256s",
                "teacher_artifacts", "pre_privileged_seal_sha256", "result_sha256",
                "evaluator_code_sha256",
            },
            "internal GATE certificate key drift",
        )
        gate_path = temporary / "GATE.json"
        write_json(gate_path, certificate)
        gate_sha = sha256_file(gate_path)
        complete = {
            "schema": COMPLETE_SCHEMA,
            "status": "complete",
            "passed": gate["passed"],
            "decision": gate["decision"],
            "gate_sha256": gate_sha,
            "result_sha256": result_sha,
            "pre_privileged_seal_sha256": preseal_sha,
            "evaluator_code_sha256": evaluator_code_sha,
            "registration_sha256": phase_a["frozen"]["sha256"],
            "source_summary_sha256": summary_sha,
        }
        write_json(temporary / "COMPLETE.json", complete)
        os.replace(temporary, output_dir)
        return output_dir, certificate
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def make_diagnostic(name: str, label: str, frozen: dict[str, Any]) -> dict[str, Any]:
    inside = label == "inside"
    uncertain = label == "uncertain"
    area = 10 if uncertain else 100
    centroid = [50.0, 50.0] if inside else [200.0, 200.0]
    reasons = ["target_mask_area_px_lt"] if uncertain else []
    return {
        "sam_object_id": frozen["sam_ids"][name],
        "mask_area_px": area,
        "mask_bbox_xyxy": [45, 45, 55, 55] if inside else [195, 195, 205, 205],
        "mask_centroid_xy": centroid,
        "object_score_logit": 2.0,
        "finite_output": True,
        "centroid_in_tracked_basket_bbox": inside,
        "uncertain_reasons": reasons,
        "label": label,
    }


def make_synthetic_fixture(root: Path, registration_path: Path,
                           bad_metrics: bool = False) -> tuple[list[Path], Path]:
    frozen = validate_registration(registration_path)
    summary_records = []
    event_map: dict[int, dict[int, int]] = {}
    for episode in range(96):
        events: dict[int, int] = {}
        if episode >= 16:
            events[1] = 7 if episode == 17 else 10  # off-grid events remain valid
            if episode % 2 == 0:
                events[3] = 20
            if episode == 16:
                events[5] = 30  # no frame at/after 30: one explicit right-censored event
        event_map[episode] = events
        summary_records.append(
            {
                "idx": episode,
                "seed": 8700 + episode,
                "steps": 30,
                "events": events,
                # Deliberately arbitrary and unused.  Metrics must depend only on events/steps.
                "success": bool(episode % 3 == 0),
            }
        )
    summary = {
        "task": TASK,
        "c": CHUNK_STEPS,
        "episodes": 96,
        "episode_records": summary_records,
    }
    summary_path = root / "synthetic_source_summary.json"
    write_json(summary_path, summary)
    summary_sha = sha256_file(summary_path)

    manifest_specs = [("a", range(16, 56)), ("b", range(56, 96))]
    manifest_rows: dict[str, list[dict[str, Any]]] = {}
    manifest_replay_completions: dict[str, dict[str, Any]] = {}
    manifest_paths: list[Path] = []
    row_index = 0
    for tag, episodes in manifest_specs:
        manifest_dir = root / f"replay_{tag}"
        manifest_dir.mkdir(parents=True)
        image_path = manifest_dir / "synthetic_agentview.jpg"
        Image.new("RGB", (360, 360), color=(32, 64, 96)).save(image_path, quality=90)
        image_sha = sha256_file(image_path)
        rows: list[dict[str, Any]] = []
        for episode in episodes:
            for chunk_index, t_value in enumerate((0, 10, 20)):
                rows.append(
                    {
                        "schema": REPLAY_SCHEMA,
                        "frame_id": f"{TAPE_SHA256[:12]}:ep{episode}:t{t_value}",
                        "task": TASK,
                        "tape_sha256": TAPE_SHA256,
                        "episode_id": episode,
                        "env_seed": 8700 + episode,
                        "t": t_value,
                        "row_index": row_index,
                        "chunk_index": chunk_index,
                        "sampling": {"chunk_stride": 1, "pre_action": True},
                        "images": {
                            CAMERA: {
                                "path": image_path.name,
                                "sha256": image_sha,
                                "width": 360,
                                "height": 360,
                            }
                        },
                        "summary_sha256": summary_sha,
                    }
                )
                row_index += 1
        manifest_path = manifest_dir / "manifest.jsonl"
        write_jsonl(manifest_path, rows)
        manifest_sha = sha256_file(manifest_path)
        replay_summary_path = manifest_dir / "summary.json"
        episode_ids = sorted({row["episode_id"] for row in rows})
        write_json(
            replay_summary_path,
            {
                "schema": REPLAY_SUMMARY_SCHEMA,
                "task": TASK,
                "c": CHUNK_STEPS,
                "tape_sha256": TAPE_SHA256,
                "manifest_path": manifest_path.name,
                "manifest_sha256": manifest_sha,
                "summary_sha256": summary_sha,
                "episode_ids": episode_ids,
                "env_seeds": [8700 + episode for episode in episode_ids],
                "episodes_replayed": len(episode_ids),
                "frames_saved": len(rows),
                "chunks_replayed": len(rows),
                "env_steps_replayed": len(rows) * CHUNK_STEPS,
                "sampling": {"chunk_stride": 1, "pre_action": True},
                "proprio_validation": {
                    "frames_checked": len(rows),
                    "threshold_max_abs_error": 1e-5,
                    "observed_max_abs_error": 0.0,
                    "mean_of_frame_max_abs_error": 0.0,
                },
                "terminal_action_recovery": {
                    "recoverable": False,
                    "episodes_with_unstored_terminal_steps": 0,
                    "unstored_terminal_env_steps": 0,
                },
            },
        )
        manifest_replay_completions[manifest_sha] = {
            "path": str(replay_summary_path.resolve()),
            "sha256": sha256_file(replay_summary_path),
            "schema": REPLAY_SUMMARY_SCHEMA,
            "mechanical_metadata_only": True,
            "forbidden_privileged_field_paths": [],
            "manifest_sha256": manifest_sha,
            "source_summary_sha256_declared_only_not_opened": summary_sha,
            "episode_ids": episode_ids,
            "frames_saved": len(rows),
            "chunks_replayed": len(rows),
            "sampling": {"chunk_stride": 1, "pre_action": True},
            "proprio_validation": {
                "threshold_max_abs_error": 1e-5,
                "observed_max_abs_error": 0.0,
                "frames_checked": len(rows),
            },
            "episodes_with_unstored_terminal_steps": 0,
            "unstored_terminal_env_steps": 0,
        }
        manifest_paths.append(manifest_path)
        manifest_rows[manifest_sha] = rows

    run_dir = root / ("teacher_bad" if bad_metrics else "teacher_pass")
    run_dir.mkdir()
    code_path = EXPECTED_TEACHER_CODE_PATH.resolve()
    code_sha = sha256_file(code_path)
    require(code_sha == EXPECTED_TEACHER_CODE_SHA256,
            "installed frozen teacher code SHA drifted")
    checkpoint_sha = frozen["payload"]["tracker"]["checkpoint_sha256"]
    labels: list[dict[str, Any]] = []
    manifest_audits: list[dict[str, Any]] = []
    for manifest_path in manifest_paths:
        manifest_sha = sha256_file(manifest_path)
        rows = manifest_rows[manifest_sha]
        episodes = sorted({row["episode_id"] for row in rows})
        manifest_audits.append(
            {
                "path": str(manifest_path.resolve()),
                "sha256": manifest_sha,
                "row_count": len(rows),
                "episode_ids": episodes,
                "verified_image_records": len(rows),
                "verified_image_set_sha256": "0" * 64,
                "replay_completion": manifest_replay_completions[manifest_sha],
            }
        )
        for source in rows:
            episode, t_value = source["episode_id"], source["t"]
            public: dict[str, dict[str, str]] = {}
            diagnostics: dict[str, dict[str, Any]] = {}
            for name in TARGET_OBJECTS:
                event_step = event_map[episode].get(PLACE_EVENT_INDEX[name])
                inside = event_step is not None and event_step <= t_value
                if bad_metrics and name == "cream_cheese_1":
                    inside = True
                label = "inside" if inside else "outside"
                public[name] = {"label": label}
                diagnostics[name] = make_diagnostic(name, label, frozen)
            basket = {
                "sam_object_id": frozen["sam_ids"]["basket_1"],
                "mask_area_px": 5000,
                "mask_bbox_xyxy": [0, 0, 100, 100],
                "mask_centroid_xy": [50.0, 50.0],
                "object_score_logit": 2.0,
                "finite_output": True,
                "uncertain_reasons": [],
                "reliable": True,
            }
            labels.append(
                {
                    "schema": LEGACY_LABEL_SCHEMA,
                    "teacher_schema": TEACHER_LABEL_SCHEMA,
                    "producer": "frozen_registration_sam2_rgb_teacher",
                    "frame_id": source["frame_id"],
                    "task": TASK,
                    "replay_manifest_sha256": manifest_sha,
                    "parse_ok": True,
                    "objects": public,
                    "object_diagnostics": diagnostics,
                    "basket_diagnostics": basket,
                    "episode_id": episode,
                    "env_seed": source["env_seed"],
                    "t": t_value,
                    "row_index": source["row_index"],
                    "chunk_index": source["chunk_index"],
                    "registration_sha256": frozen["sha256"],
                    "checkpoint_sha256": checkpoint_sha,
                    "camera": CAMERA,
                    "prompt_frame_index": 0,
                    "post_t0_prompt_count": 0,
                    "source_image_sha256": source["images"][CAMERA]["sha256"],
                    "source_summary_sha256_declared_only": source["summary_sha256"],
                }
            )
    labels.sort(key=lambda row: (row["episode_id"], row["t"]))
    labels_path = run_dir / "labels.jsonl"
    write_jsonl(labels_path, labels)
    labels_sha = sha256_file(labels_path)
    per_dir = run_dir / "labels_by_manifest"
    per_dir.mkdir()
    per_manifest = []
    for manifest_sha in sorted(manifest_rows):
        rows = [row for row in labels if row["replay_manifest_sha256"] == manifest_sha]
        path = per_dir / f"{manifest_sha}.jsonl"
        write_jsonl(path, rows)
        per_manifest.append(
            {
                "manifest_sha256": manifest_sha,
                "labels_relative_path": str(path.relative_to(run_dir)),
                "labels_sha256": sha256_file(path),
                "label_rows": len(rows),
            }
        )
    uncertain = sum(
        row["objects"][name]["label"] == "uncertain"
        for row in labels for name in TARGET_OBJECTS
    )
    requested = list(EXPECTED_EPISODES)
    result = {
        "schema": TEACHER_RUN_SCHEMA,
        "status": "complete",
        "registration": {"path": str(registration_path.resolve()), "sha256": frozen["sha256"]},
        "checkpoint": {
            "sha256": checkpoint_sha,
            "strict_load": True,
            "tensor_count": 903,
            "dtype": "torch.bfloat16",
        },
        "registered_probe": {
            "path": str(frozen["probe_path"]),
            "sha256": frozen["payload"]["tracker"]["probe_code_sha256"],
        },
        "code": {"path": str(code_path.resolve()), "sha256": code_sha},
        "inputs": {
            "requested_episode_ids": requested,
            "episode_partitions": {
                str(episode): frozen["episode_partition"][episode] for episode in requested
            },
            "manifests": manifest_audits,
            "verified_frame_rows": len(labels),
        },
        "outputs": {
            "labels_relative_path": "labels.jsonl",
            "labels_sha256": labels_sha,
            "label_rows": len(labels),
            "object_labels": len(labels) * len(TARGET_OBJECTS),
            "uncertain_object_labels": uncertain,
            "uncertain_fraction": uncertain / (len(labels) * len(TARGET_OBJECTS)),
            "per_manifest": per_manifest,
        },
    }
    result_path = run_dir / "result.json"
    write_json(result_path, result)
    complete = {
        "schema": TEACHER_COMPLETE_SCHEMA,
        "status": "complete",
        "result_sha256": sha256_file(result_path),
        "labels_sha256": labels_sha,
        "code_sha256": code_sha,
        "registration_sha256": frozen["sha256"],
    }
    write_json(run_dir / "COMPLETE.json", complete)
    return [run_dir], summary_path


def golden_metric_smoke() -> None:
    frames: list[dict[str, Any]] = []
    for index, t_value in enumerate(range(0, 750, 10)):
        labels = {
            "alphabet_soup_1": "inside" if t_value >= 110 else "outside",
            "tomato_sauce_1": "inside" if t_value >= 240 else "outside",
            "cream_cheese_1": "outside",
        }
        frames.append(
            {
                "episode_id": 16,
                "env_seed": 8716,
                "t": t_value,
                "label_row": {
                    "objects": {
                        name: {"label": label} for name, label in labels.items()
                    }
                },
            }
        )
    records = {16: {"seed": 8716, "steps": 750, "events": {1: 120, 3: 250}}}
    metrics = compute_metrics(frames, records, [16])
    classification = metrics["classification"]
    require(math.isclose(classification["framewise_place_accuracy_certain_only"],
                         223 / 225, abs_tol=1e-15), "golden accuracy mismatch")
    f1 = classification["per_object_inside_positive_f1"]
    require(math.isclose(f1["alphabet_soup_1"], 126 / 127, abs_tol=1e-15),
            "golden alphabet F1 mismatch")
    require(math.isclose(f1["tomato_sauce_1"], 100 / 101, abs_tol=1e-15),
            "golden tomato F1 mismatch")
    require(f1["cream_cheese_1"] == 1.0, "golden empty-positive F1 mismatch")
    require(metrics["transitions"]["median_absolute_error_steps"] == 10.0,
            "golden transition median mismatch")
    require(metrics["terminal_specificity"]["specificity"] == 1.0,
            "golden terminal specificity mismatch")


def reason_set_contract_smoke(frozen: dict[str, Any]) -> None:
    """Accept teacher construction order, but reject duplicate/missing reasons."""
    basket_reasons = [
        "basket_object_score_logit_lte",
        "basket_mask_area_px_lt",
    ]
    basket = {
        "sam_object_id": frozen["sam_ids"]["basket_1"],
        "mask_area_px": 1000,
        "mask_bbox_xyxy": [0, 0, 100, 100],
        "mask_centroid_xy": [50.0, 50.0],
        "object_score_logit": 0.0,
        "finite_output": True,
        "uncertain_reasons": basket_reasons,
        "reliable": False,
    }
    diagnostics = {}
    objects = {}
    for name in TARGET_OBJECTS:
        diagnostics[name] = {
            "sam_object_id": frozen["sam_ids"][name],
            "mask_area_px": 100,
            "mask_bbox_xyxy": [10, 10, 20, 20],
            "mask_centroid_xy": [15.0, 15.0],
            "object_score_logit": 2.0,
            "finite_output": True,
            "uncertain_reasons": list(reversed(basket_reasons)),
            "centroid_in_tracked_basket_bbox": True,
            "label": "uncertain",
        }
        objects[name] = {"label": "uncertain"}
    row = {
        "objects": objects,
        "object_diagnostics": diagnostics,
        "basket_diagnostics": basket,
    }
    validate_label_diagnostics(row, frozen, "reason-order smoke")
    duplicate = json.loads(json.dumps(row))
    duplicate["basket_diagnostics"]["uncertain_reasons"].append(
        "basket_mask_area_px_lt"
    )
    try:
        validate_label_diagnostics(duplicate, frozen, "duplicate-reason smoke")
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate uncertainty reason was accepted")


def synthetic_smoke(registration_path: Path) -> dict[str, Any]:
    golden_metric_smoke()
    reason_set_contract_smoke(validate_registration(registration_path))
    with tempfile.TemporaryDirectory(prefix="v243_teacher_gate_smoke_") as raw:
        root = Path(raw)
        pass_root = root / "pass_fixture"
        pass_root.mkdir()
        runs, summary = make_synthetic_fixture(pass_root, registration_path, bad_metrics=False)
        pass_out, pass_gate = evaluate_atomic(
            registration_path, runs, summary, root / "pass_output"
        )
        require(pass_gate["passed"] is True and pass_gate["decision"] == "PASS",
                "synthetic PASS fixture did not pass")
        pass_result = load_json_strict(pass_out / "result.json")
        require(pass_result["metrics"]["transitions"]["right_censored_positive_events"] == 1,
                "right-censor accounting mismatch")
        classification = pass_result["metrics"]["classification"]
        require(classification["excluded_censored_frame_labels"] == 3,
                "right-censored object-episode leaked into framewise metrics")
        require(classification["evaluable_object_labels"] == 717,
                "right-censored classification denominator mismatch")
        coverage = pass_result["metrics"]["coverage"]
        require(coverage["frame_coverage"] == 1.0
                and coverage["teacher_to_manifest_coverage"] == 1.0,
                "observable label/manifest coverage mismatch")
        require(sha256_file(pass_out / "GATE.json")
                == load_json_strict(pass_out / "COMPLETE.json")["gate_sha256"],
                "PASS GATE seal mismatch")

        stop_root = root / "stop_fixture"
        stop_root.mkdir()
        stop_runs, stop_summary = make_synthetic_fixture(
            stop_root, registration_path, bad_metrics=True
        )
        stop_out, stop_gate = evaluate_atomic(
            registration_path, stop_runs, stop_summary, root / "stop_output"
        )
        require(stop_gate["passed"] is False and stop_gate["decision"] == "STOP",
                "synthetic STOP fixture did not stop")
        require((stop_out / "COMPLETE.json").is_file(), "STOP artifact was not atomic/complete")

        tamper_root = root / "tamper_fixture"
        tamper_root.mkdir()
        tamper_runs, tamper_summary = make_synthetic_fixture(
            tamper_root, registration_path, bad_metrics=False
        )
        labels_path = tamper_runs[0] / "labels.jsonl"
        labels_path.write_text(labels_path.read_text(encoding="utf-8") + "\n",
                               encoding="utf-8")
        tamper_out = root / "tamper_output"
        try:
            evaluate_atomic(registration_path, tamper_runs, tamper_summary, tamper_out)
        except ValueError:
            pass
        else:
            raise AssertionError("tampered teacher labels were accepted")
        require(not tamper_out.exists(), "failed evaluation left a committed output")

        missing_preseal = root / "missing_preseal.json"
        try:
            load_privileged_summary(summary, sha256_file(summary), missing_preseal, "0" * 64)
        except ValueError as exc:
            require("before pre-seal" in str(exc), "wrong preseal guard failure")
        else:
            raise AssertionError("privileged summary opened without a pre-seal")
        return {
            "status": "PASS",
            "golden_metric_fixture": "PASS",
            "formal_pass_fixture": "PASS",
            "right_censored_positive_events": 1,
            "excluded_censored_frame_labels": 3,
            "formal_stop_fixture": "STOP artifact committed",
            "tampered_labels": "rejected atomically",
            "summary_before_preseal": "rejected",
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    parser.add_argument("--teacher-run", type=Path, action="append")
    parser.add_argument("--source-summary", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args()
    if args.synthetic_smoke:
        require(
            args.teacher_run is None
            and args.source_summary is None
            and args.output_dir is None,
            "--synthetic-smoke cannot be mixed with formal inputs",
        )
    else:
        require(bool(args.teacher_run) and args.source_summary is not None
                and args.output_dir is not None,
                "formal run requires --teacher-run, --source-summary and --output-dir")
    return args


def main() -> int:
    args = parse_args()
    if args.synthetic_smoke:
        print(json.dumps(synthetic_smoke(args.registration.resolve()), indent=2, sort_keys=True))
        return 0
    output, certificate = evaluate_atomic(
        args.registration.resolve(),
        list(args.teacher_run),
        args.source_summary.resolve(),
        args.output_dir.resolve(),
    )
    print(json.dumps({"output_dir": str(output), **certificate}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
