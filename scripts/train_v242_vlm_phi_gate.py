#!/usr/bin/env python3
"""Distil a frozen SAM2 RGB teacher into an observation-only latent Phi.

Formal mode is registration driven.  Before the latent tape is opened, this
program validates a minimal PASS certificate emitted by
``eval_v243_sam2_teacher_gate.py``.  It then verifies every replay/teacher row,
including the row's originating manifest SHA and the frozen registration,
checkpoint and producer-code provenance.  Labels and manifests must form an
exact bijection; an ``uncertain`` visual label is masked, while a missing label
is fatal.

The frozen episode partition is fit=16..63, calibration=64..79 and test=80..95.
The privileged v121 summary (simulator/BDDL placement events) remains evaluation
only: it is not resolved, hashed or opened until ``phi.pt`` has been written and
its SHA256 sealed.  ``--synthetic-smoke`` exercises the same boundary without
reading real replay episodes.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "v242_vlm_phi_gate"
OBJECTS = ("alphabet_soup_1", "tomato_sauce_1", "cream_cheese_1")
REPLAY_SCHEMA = "v242_rgb_replay_v1"
LABEL_COMPAT_SCHEMA = "v242_qwen_label_v1"
TEACHER_SCHEMA = "v243_sam2_teacher_label_v1"
TEACHER_PRODUCER = "frozen_registration_sam2_rgb_teacher"
TEACHER_RUN_SCHEMA = "v243_sam2_teacher_run_v1"
TEACHER_COMPLETE_SCHEMA = "v243_sam2_teacher_complete_v1"
REGISTRATION_SCHEMA = "v242_chain3_sam2_registration_v1"
REGISTRATION_SHA256 = "3d892616b97442a080e16f50f226217eb00dd3d2e19f6fb7e048750a251363ab"
FROZEN_TEACHER_CODE_SHA256 = (
    "205c31169f014e9f5ee4dd607a68f359ae1a2d844ef93fbaf6d830660d407adf"
)
TRACKER_EVALUATOR_PATH = REPO / "scripts" / "eval_v243_sam2_teacher_gate.py"
TRACKER_GATE_SCHEMA = "v243_sam2_teacher_gate_certificate_v1"
TRACKER_GATE_COMPLETE_SCHEMA = "v243_sam2_teacher_gate_complete_v1"
PHI_SCHEMA = "v242_vlm_phi_v1"
RESULT_SCHEMA = "v242_vlm_phi_gate_result_v1"
ARM_NAMES = ("full", "time_only", "proprio_only", "label_shuffle")

# Frozen before labels are inspected.
PRIMARY_RHO_FLOOR = 0.157
PRIMARY_CI_FLOOR = 0.078
NULL_ABS_CEILING = 0.100


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()


def json_safe(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, torch.Tensor):
        return json_safe(x.detach().cpu().tolist())
    return x


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Strictly read either JSONL rows or a JSON object/list of rows."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{path}: empty file")
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL row")
            rows.append(json.loads(line, object_pairs_hook=reject_duplicate_keys))
    else:
        obj = json.loads(text, object_pairs_hook=reject_duplicate_keys)
        if isinstance(obj, list):
            rows = obj
        elif isinstance(obj, dict) and isinstance(obj.get("rows"), list):
            rows = obj["rows"]
        else:
            # A one-row JSON file is useful for tiny contract fixtures.
            rows = [obj]
    if not rows or not all(isinstance(r, dict) for r in rows):
        raise ValueError(f"{path}: expected non-empty JSON object rows")
    return rows


@dataclass
class Dataset:
    z: torch.Tensor
    y: torch.Tensor
    mask: torch.Tensor
    frame_ids: list[str]
    row_indices: np.ndarray
    episode_ids: np.ndarray
    env_seeds: np.ndarray
    t: np.ndarray
    t_norm: np.ndarray
    task: str
    tape_sha256: str
    proprio_dim: int
    coverage: dict[str, Any]

    def __len__(self) -> int:
        return len(self.frame_ids)


def _require(row: dict[str, Any], fields: Iterable[str], where: str) -> None:
    missing = [k for k in fields if k not in row]
    if missing:
        raise ValueError(f"{where}: missing fields {missing}")


def validate_registration(
    path: Path, synthetic_contract: bool = False
) -> dict[str, Any]:
    actual_sha = sha256_file(path)
    if not synthetic_contract and actual_sha != REGISTRATION_SHA256:
        raise ValueError(
            f"registration SHA mismatch: {actual_sha} != {REGISTRATION_SHA256}"
        )
    registration = load_json_object(path)
    if registration.get("schema") != REGISTRATION_SCHEMA:
        raise ValueError("registration schema mismatch")
    if registration.get("status") != "frozen_before_any_episode_16_to_95_tracking_output":
        raise ValueError("registration is not frozen at the required boundary")
    target = registration.get("target")
    if not isinstance(target, dict):
        raise ValueError("registration target must be an object")
    expected_target: dict[str, Any] = {
        "task": "chain3_lr2",
        "camera": "agentview",
        "chunk_steps": 10,
        "horizon_steps": 750,
    }
    if synthetic_contract:
        if registration.get("synthetic_contract_fixture") is not True:
            raise ValueError("synthetic registration lacks fixture marker")
        if not is_sha256(target.get("source_tape_sha256")):
            raise ValueError("synthetic registration tape SHA is invalid")
    else:
        expected_target["source_tape_sha256"] = (
            "7dbff743ea89db5fd792d317111129d188d439122a46db7384f4549fd947f86c"
        )
    for field, expected in expected_target.items():
        if target.get(field) != expected:
            raise ValueError(f"registration target.{field} mismatch")
    partition = registration.get("episode_partition")
    if not isinstance(partition, dict) or partition.get("disjoint") is not True:
        raise ValueError("invalid registration partition")
    groups = {
        "train": partition.get("phi_fit_episode_ids"),
        "calib": partition.get("phi_calibration_episode_ids"),
        "test": partition.get("phi_test_episode_ids"),
    }
    expected_groups = {
        "train": list(range(16, 64)),
        "calib": list(range(64, 80)),
        "test": list(range(80, 96)),
    }
    if groups != expected_groups:
        raise ValueError(f"registration Phi partition mismatch: {groups}")
    if partition.get("environment_seed_rule") != "env_seed = 8700 + episode_id":
        raise ValueError("registration environment seed rule mismatch")
    tracker = registration.get("tracker")
    if not isinstance(tracker, dict):
        raise ValueError("registration tracker must be an object")
    for field in ("checkpoint_sha256", "probe_code_sha256"):
        if not is_sha256(tracker.get(field)):
            raise ValueError(f"registration tracker.{field} is invalid")
    return {
        "payload": registration,
        "sha256": actual_sha,
        "groups": expected_groups,
    }


def validate_tracker_gate(
    gate_path: Path,
    registration: dict[str, Any],
    manifest_paths: list[Path],
    teacher_output_paths: list[Path],
) -> dict[str, Any]:
    """Validate the non-privileged, minimal tracker PASS certificate first."""
    gate_path = gate_path.resolve()
    gate = load_json_object(gate_path)
    expected_keys = {
        "schema",
        "status",
        "passed",
        "decision",
        "registration_sha256",
        "source_summary_sha256",
        "episode_ids",
        "manifest_sha256s",
        "teacher_artifacts",
        "pre_privileged_seal_sha256",
        "result_sha256",
        "evaluator_code_sha256",
    }
    if set(gate) != expected_keys:
        raise ValueError(
            f"tracker GATE.json keys differ from minimal certificate: {set(gate)}"
        )
    if gate.get("schema") != TRACKER_GATE_SCHEMA:
        raise ValueError("tracker gate certificate schema mismatch")
    if not (
        gate.get("status") == "complete"
        and gate.get("passed") is True
        and gate.get("decision") == "PASS"
    ):
        raise ValueError("tracker gate did not pass; STOP before tape access")
    for field in (
        "source_summary_sha256",
        "pre_privileged_seal_sha256",
        "result_sha256",
        "evaluator_code_sha256",
    ):
        if not is_sha256(gate.get(field)):
            raise ValueError(f"tracker gate {field} is invalid")
    if gate.get("registration_sha256") != registration["sha256"]:
        raise ValueError("tracker gate registration SHA mismatch")
    if gate.get("episode_ids") != list(range(16, 96)):
        raise ValueError("tracker gate must cover exactly episode IDs 16..95")

    actual_manifest_shas = sorted(sha256_file(path.resolve()) for path in manifest_paths)
    if len(actual_manifest_shas) != len(set(actual_manifest_shas)):
        raise ValueError("duplicate replay manifest content")
    if gate.get("manifest_sha256s") != actual_manifest_shas:
        raise ValueError("tracker gate manifest SHA set mismatch")
    run_paths = sorted(str(path.resolve()) for path in teacher_output_paths)
    declared_artifacts = gate.get("teacher_artifacts")
    if not isinstance(declared_artifacts, list) or not declared_artifacts:
        raise ValueError("tracker gate teacher_artifacts must be non-empty")
    if [artifact.get("run_path") for artifact in declared_artifacts] != run_paths:
        raise ValueError("tracker gate teacher artifact paths must be canonical and exact")
    artifact_keys = {
        "run_path",
        "result_sha256",
        "labels_sha256",
        "code_sha256",
        "checkpoint_sha256",
        "registration_sha256",
        "manifest_sha256s",
    }
    for index, artifact in enumerate(declared_artifacts):
        if not isinstance(artifact, dict) or set(artifact) != artifact_keys:
            raise ValueError(f"tracker gate teacher_artifacts[{index}] schema mismatch")
        for field in (
            "result_sha256",
            "labels_sha256",
            "code_sha256",
            "checkpoint_sha256",
            "registration_sha256",
        ):
            if not is_sha256(artifact.get(field)):
                raise ValueError(f"tracker artifact {index} has invalid {field}")
        manifest_shas = artifact.get("manifest_sha256s")
        if not isinstance(manifest_shas, list) or manifest_shas != sorted(set(manifest_shas)):
            raise ValueError(f"tracker artifact {index} manifest SHAs not canonical")
    artifact_manifest_union = sorted(
        manifest_sha
        for artifact in declared_artifacts
        for manifest_sha in artifact["manifest_sha256s"]
    )
    if artifact_manifest_union != actual_manifest_shas:
        raise ValueError("tracker artifact manifest subsets do not partition input manifests")

    complete_path = gate_path.parent / "COMPLETE.json"
    complete = load_json_object(complete_path)
    if complete.get("schema") != TRACKER_GATE_COMPLETE_SCHEMA:
        raise ValueError("tracker gate COMPLETE schema mismatch")
    if complete.get("status") != "complete" or complete.get("passed") is not True:
        raise ValueError("tracker gate COMPLETE is not a successful atomic commit")
    if complete.get("decision") != "PASS":
        raise ValueError("tracker gate COMPLETE decision mismatch")
    if complete.get("gate_sha256") != sha256_file(gate_path):
        raise ValueError("tracker gate certificate SHA mismatch")
    for field in (
        "result_sha256",
        "pre_privileged_seal_sha256",
        "evaluator_code_sha256",
    ):
        if complete.get(field) != gate[field]:
            raise ValueError(f"tracker gate COMPLETE {field} mismatch")
    if complete.get("registration_sha256") != gate["registration_sha256"]:
        raise ValueError("tracker gate COMPLETE registration SHA mismatch")
    if complete.get("source_summary_sha256") != gate["source_summary_sha256"]:
        raise ValueError("tracker gate COMPLETE source summary SHA mismatch")

    evaluator_result_path = gate_path.parent / "result.json"
    preseal_path = gate_path.parent / "pre_privileged_seal.json"
    if not evaluator_result_path.is_file() or not preseal_path.is_file():
        raise ValueError("tracker gate result/pre-privileged seal artifact is missing")
    if sha256_file(evaluator_result_path) != gate["result_sha256"]:
        raise ValueError("tracker evaluator result SHA mismatch")
    if sha256_file(preseal_path) != gate["pre_privileged_seal_sha256"]:
        raise ValueError("tracker evaluator pre-privileged seal SHA mismatch")
    evaluator_path = TRACKER_EVALUATOR_PATH.resolve()
    if not evaluator_path.is_file():
        raise ValueError("tracker evaluator code is missing")
    if sha256_file(evaluator_path) != gate["evaluator_code_sha256"]:
        raise ValueError("tracker evaluator code SHA mismatch")
    return {
        "payload": gate,
        "path": str(gate_path),
        "sha256": sha256_file(gate_path),
        "complete_path": str(complete_path),
        "complete_sha256": sha256_file(complete_path),
        "result_path": str(evaluator_result_path),
        "result_sha256": gate["result_sha256"],
        "pre_privileged_seal_path": str(preseal_path),
        "pre_privileged_seal_sha256": gate["pre_privileged_seal_sha256"],
        "evaluator_code_path": str(evaluator_path),
        "evaluator_code_sha256": gate["evaluator_code_sha256"],
        "manifest_sha256s": actual_manifest_shas,
    }


def validate_visual_sources(
    manifest_paths: list[Path],
    teacher_output_paths: list[Path],
    registration: dict[str, Any],
    gate_audit: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    """Validate all observation-only inputs before the latent tape is opened."""
    target = registration["payload"]["target"]
    gate = gate_audit["payload"]
    manifest_by_sha: dict[str, Path] = {}
    replay_by_frame: dict[str, dict[str, Any]] = {}
    seen_row_indices: set[int] = set()
    manifest_audit: list[dict[str, Any]] = []
    for input_path in manifest_paths:
        path = input_path.resolve()
        manifest_sha = sha256_file(path)
        if manifest_sha in manifest_by_sha:
            raise ValueError(f"duplicate manifest SHA {manifest_sha}")
        manifest_by_sha[manifest_sha] = path
        rows = read_rows(path)
        manifest_episodes: set[int] = set()
        for index, row in enumerate(rows):
            where = f"{path}:row {index}"
            _require(
                row,
                (
                    "schema",
                    "frame_id",
                    "task",
                    "tape_sha256",
                    "summary_sha256",
                    "episode_id",
                    "env_seed",
                    "t",
                    "row_index",
                    "chunk_index",
                    "images",
                ),
                where,
            )
            if row["schema"] != REPLAY_SCHEMA or row["task"] != target["task"]:
                raise ValueError(f"{where}: replay schema/task mismatch")
            if row["tape_sha256"] != target["source_tape_sha256"]:
                raise ValueError(f"{where}: tape SHA differs from registration")
            if row["summary_sha256"] != gate["source_summary_sha256"]:
                raise ValueError(f"{where}: source summary SHA differs from tracker gate")
            episode_id = int(row["episode_id"])
            env_seed = int(row["env_seed"])
            t_value = int(row["t"])
            row_index = int(row["row_index"])
            chunk_index = int(row["chunk_index"])
            if episode_id not in range(16, 96) or env_seed != 8700 + episode_id:
                raise ValueError(f"{where}: episode partition/seed mismatch")
            if t_value != chunk_index * target["chunk_steps"]:
                raise ValueError(f"{where}: t/chunk mismatch")
            expected_frame_id = (
                f"{target['source_tape_sha256'][:12]}:ep{episode_id}:t{t_value}"
            )
            frame_id = str(row["frame_id"])
            if frame_id != expected_frame_id or frame_id in replay_by_frame:
                raise ValueError(f"{where}: invalid/duplicate frame_id")
            if row_index < 0 or row_index in seen_row_indices:
                raise ValueError(f"{where}: invalid/duplicate row_index")
            images = row["images"]
            camera = target["camera"]
            if not isinstance(images, dict) or camera not in images:
                raise ValueError(f"{where}: registered camera image missing")
            image = images[camera]
            if not isinstance(image, dict) or not is_sha256(image.get("sha256")):
                raise ValueError(f"{where}: registered image SHA missing")
            replay_by_frame[frame_id] = {
                **row,
                "episode_id": episode_id,
                "env_seed": env_seed,
                "t": t_value,
                "row_index": row_index,
                "chunk_index": chunk_index,
                "_manifest_sha256": manifest_sha,
                "_manifest_path": str(path),
            }
            seen_row_indices.add(row_index)
            manifest_episodes.add(episode_id)
        manifest_audit.append(
            {
                "path": str(path),
                "sha256": manifest_sha,
                "row_count": len(rows),
                "episode_ids": sorted(manifest_episodes),
            }
        )

    by_episode: dict[int, list[dict[str, Any]]] = {}
    for row in replay_by_frame.values():
        by_episode.setdefault(row["episode_id"], []).append(row)
    if sorted(by_episode) != list(range(16, 96)):
        raise ValueError("replay manifests must cover exactly episode IDs 16..95")
    nominal_frames = target["horizon_steps"] // target["chunk_steps"]
    for episode_id, rows in by_episode.items():
        ordered = sorted(rows, key=lambda row: row["t"])
        if not 1 <= len(ordered) <= nominal_frames:
            raise ValueError(f"episode {episode_id}: invalid prefix length")
        expected_t = [index * target["chunk_steps"] for index in range(len(ordered))]
        if [row["t"] for row in ordered] != expected_t:
            raise ValueError(f"episode {episode_id}: manifest is not a gap-free t0 prefix")
        if [row["chunk_index"] for row in ordered] != list(range(len(ordered))):
            raise ValueError(f"episode {episode_id}: chunk grid mismatch")
        if len({row["_manifest_sha256"] for row in ordered}) != 1:
            raise ValueError(f"episode {episode_id}: split across manifests")

    labels_by_frame: dict[str, dict[str, Any]] = {}
    artifact_audit: list[dict[str, Any]] = []
    declared_gate_artifacts = {
        artifact["run_path"]: artifact for artifact in gate["teacher_artifacts"]
    }
    tracker = registration["payload"]["tracker"]
    for output_path in sorted(path.resolve() for path in teacher_output_paths):
        run_path = str(output_path)
        declared = declared_gate_artifacts.get(run_path)
        if declared is None:
            raise ValueError(f"teacher output absent from gate certificate: {run_path}")
        complete_path = output_path / "COMPLETE.json"
        result_path = output_path / "result.json"
        complete = load_json_object(complete_path)
        result = load_json_object(result_path)
        if complete.get("schema") != TEACHER_COMPLETE_SCHEMA:
            raise ValueError(f"{output_path}: teacher COMPLETE schema mismatch")
        if complete.get("status") != "complete" or result.get("status") != "complete":
            raise ValueError(f"{output_path}: teacher artifact is incomplete")
        if result.get("schema") != TEACHER_RUN_SCHEMA:
            raise ValueError(f"{output_path}: teacher result schema mismatch")
        if complete.get("result_sha256") != sha256_file(result_path):
            raise ValueError(f"{output_path}: teacher result SHA mismatch")
        labels_relative = result.get("outputs", {}).get("labels_relative_path")
        labels_path = (output_path / str(labels_relative)).resolve()
        if not labels_path.is_relative_to(output_path) or not labels_path.is_file():
            raise ValueError(f"{output_path}: invalid teacher labels path")
        labels_sha = sha256_file(labels_path)
        result_sha = sha256_file(result_path)
        if complete.get("labels_sha256") != labels_sha:
            raise ValueError(f"{output_path}: teacher labels SHA mismatch")
        if result.get("outputs", {}).get("labels_sha256") != labels_sha:
            raise ValueError(f"{output_path}: teacher result labels SHA mismatch")
        reg_info = result.get("registration")
        checkpoint = result.get("checkpoint")
        code = result.get("code")
        if not all(isinstance(value, dict) for value in (reg_info, checkpoint, code)):
            raise ValueError(f"{output_path}: teacher provenance objects missing")
        if reg_info.get("sha256") != registration["sha256"]:
            raise ValueError(f"{output_path}: teacher registration SHA mismatch")
        if complete.get("registration_sha256") != registration["sha256"]:
            raise ValueError(f"{output_path}: teacher COMPLETE registration mismatch")
        if checkpoint.get("sha256") != tracker["checkpoint_sha256"]:
            raise ValueError(f"{output_path}: teacher checkpoint SHA mismatch")
        strict_checkpoint_proof = (
            checkpoint.get("strict_load") is True
            and checkpoint.get("tensor_count") == 903
            and checkpoint.get("dtype") == "torch.bfloat16"
        )
        if (
            not strict_checkpoint_proof
            and registration["payload"].get("synthetic_contract_fixture") is not True
        ):
            raise ValueError(f"{output_path}: teacher strict checkpoint proof missing")
        code_path = Path(str(code.get("path", ""))).resolve()
        expected_code_path = (REPO / "scripts" / "track_v243_sam2_teacher.py").resolve()
        if code_path != expected_code_path:
            raise ValueError(f"{output_path}: unexpected teacher producer code path")
        if not code_path.is_file() or sha256_file(code_path) != code.get("sha256"):
            raise ValueError(f"{output_path}: teacher producer code SHA mismatch")
        if code.get("sha256") != FROZEN_TEACHER_CODE_SHA256:
            raise ValueError(f"{output_path}: teacher code is not the frozen v243 version")
        if complete.get("code_sha256") != code.get("sha256"):
            raise ValueError(f"{output_path}: teacher COMPLETE code SHA mismatch")
        for field, value in (
            ("result_sha256", result_sha),
            ("labels_sha256", labels_sha),
            ("code_sha256", code.get("sha256")),
            ("checkpoint_sha256", checkpoint.get("sha256")),
            ("registration_sha256", reg_info.get("sha256")),
        ):
            if declared.get(field) != value:
                raise ValueError(f"{output_path}: tracker certificate {field} mismatch")
        result_manifest_shas = sorted(
            manifest.get("sha256") for manifest in result.get("inputs", {}).get("manifests", [])
        )
        if declared.get("manifest_sha256s") != result_manifest_shas:
            raise ValueError(f"{output_path}: teacher manifest provenance mismatch")
        if any(manifest_sha not in manifest_by_sha for manifest_sha in result_manifest_shas):
            raise ValueError(f"{output_path}: teacher references an unknown manifest")

        label_rows = read_rows(labels_path)
        if result.get("outputs", {}).get("label_rows") != len(label_rows):
            raise ValueError(f"{output_path}: teacher label row count mismatch")
        for index, label in enumerate(label_rows):
            where = f"{labels_path}:row {index}"
            required = (
                "schema",
                "teacher_schema",
                "producer",
                "frame_id",
                "task",
                "replay_manifest_sha256",
                "parse_ok",
                "objects",
                "episode_id",
                "env_seed",
                "t",
                "row_index",
                "chunk_index",
                "registration_sha256",
                "checkpoint_sha256",
                "camera",
                "prompt_frame_index",
                "post_t0_prompt_count",
                "source_image_sha256",
                "source_summary_sha256_declared_only",
            )
            _require(label, required, where)
            if label["replay_manifest_sha256"] not in result_manifest_shas:
                raise ValueError(
                    f"{where}: label belongs to a manifest outside this teacher run"
                )
            frame_id = str(label["frame_id"])
            replay = replay_by_frame.get(frame_id)
            if replay is None or frame_id in labels_by_frame:
                raise ValueError(f"{where}: absent/duplicate replay frame")
            expected_pairs = {
                "task": target["task"],
                "replay_manifest_sha256": replay["_manifest_sha256"],
                "episode_id": replay["episode_id"],
                "env_seed": replay["env_seed"],
                "t": replay["t"],
                "row_index": replay["row_index"],
                "chunk_index": replay["chunk_index"],
                "registration_sha256": registration["sha256"],
                "checkpoint_sha256": tracker["checkpoint_sha256"],
                "camera": target["camera"],
                "source_image_sha256": replay["images"][target["camera"]]["sha256"],
                "source_summary_sha256_declared_only": gate["source_summary_sha256"],
            }
            if any(label.get(field) != value for field, value in expected_pairs.items()):
                raise ValueError(f"{where}: row-level visual provenance mismatch")
            if not (
                label["schema"] == LABEL_COMPAT_SCHEMA
                and label["teacher_schema"] == TEACHER_SCHEMA
                and label["producer"] == TEACHER_PRODUCER
                and label["parse_ok"] is True
                and label["prompt_frame_index"] == 0
                and label["post_t0_prompt_count"] == 0
            ):
                raise ValueError(f"{where}: teacher schema/prompt contract mismatch")
            objects = label["objects"]
            if not isinstance(objects, dict) or set(objects) != set(OBJECTS):
                raise ValueError(f"{where}: visual objects mismatch")
            for object_name in OBJECTS:
                value = objects[object_name]
                if not isinstance(value, dict) or set(value) != {"label"}:
                    raise ValueError(f"{where}: invalid {object_name} payload")
                if value["label"] not in ("inside", "outside", "uncertain"):
                    raise ValueError(f"{where}: invalid {object_name} label")
            labels_by_frame[frame_id] = label
        artifact_audit.append(
            {
                "run_path": run_path,
                "result_sha256": result_sha,
                "labels_path": str(labels_path),
                "labels_sha256": labels_sha,
                "label_rows": len(label_rows),
                "code_sha256": code["sha256"],
                "checkpoint_sha256": checkpoint["sha256"],
                "manifest_sha256s": result_manifest_shas,
            }
        )

    if set(labels_by_frame) != set(replay_by_frame):
        missing = sorted(set(replay_by_frame) - set(labels_by_frame))[:5]
        extra = sorted(set(labels_by_frame) - set(replay_by_frame))[:5]
        raise ValueError(
            f"teacher/replay bijection failed: missing={missing}, extra={extra}"
        )
    ordered_rows = sorted(
        replay_by_frame.values(), key=lambda row: (row["episode_id"], row["t"])
    )
    return ordered_rows, labels_by_frame, {
        "manifests": sorted(manifest_audit, key=lambda value: value["path"]),
        "teacher_artifacts": artifact_audit,
        "manifest_frames": len(ordered_rows),
        "teacher_rows": len(labels_by_frame),
        "exact_frame_bijection": True,
        "episode_ids": sorted(by_episode),
        "source_summary_sha256_declared_only": gate["source_summary_sha256"],
    }


def load_and_join(
    tape_path: Path,
    replay_rows: list[dict[str, Any]],
    labels_by_frame: dict[str, dict[str, Any]],
    expected_task: str,
) -> tuple[Dataset, dict[str, Any]]:
    """Join already validated visual sources to the tape; never open summary.json."""
    tape_sha = sha256_file(tape_path)
    tape = torch.load(tape_path, weights_only=False, map_location="cpu")
    _require(tape, ("z", "episode", "t", "task", "latent_dim"), str(tape_path))
    if tape["task"] != expected_task:
        raise ValueError(f"tape task={tape['task']!r}, expected {expected_task!r}")
    z_all = tape["z"].detach().float().cpu()
    ep_all = tape["episode"].detach().long().cpu()
    t_all = tape["t"].detach().long().cpu()
    if z_all.ndim != 2 or len(z_all) != len(ep_all) or len(z_all) != len(t_all):
        raise ValueError("malformed v121 tensors")
    if int(tape["latent_dim"]) != int(z_all.shape[1]):
        raise ValueError("latent_dim metadata disagrees with z")
    proprio_dim = int(tape.get("proprio_dim", 25))
    if not (0 < proprio_dim < z_all.shape[1]):
        raise ValueError(f"invalid proprio_dim={proprio_dim}")

    # Manifest order is irrelevant; row_index gives one canonical order.
    ordered = sorted(
        ((str(row["frame_id"]), row) for row in replay_rows),
        key=lambda item: int(item[1]["row_index"]),
    )
    frame_ids, indices, eps, seeds, times = [], [], [], [], []
    ys, masks = [], []
    object_counts = {
        object_name: {"inside": 0, "outside": 0, "uncertain": 0}
        for object_name in OBJECTS
    }
    for fid, rr in ordered:
        ri, episode_id, t_value = (
            int(rr["row_index"]),
            int(rr["episode_id"]),
            int(rr["t"]),
        )
        if not 0 <= ri < len(z_all):
            raise ValueError(f"frame {fid}: tape row_index out of bounds")
        if int(ep_all[ri]) != episode_id or int(t_all[ri]) != t_value:
            raise ValueError(f"frame {fid}: tape episode/t mismatch")
        frame_ids.append(fid)
        indices.append(ri)
        eps.append(episode_id)
        seeds.append(int(rr["env_seed"]))
        times.append(t_value)
        label_row = labels_by_frame[fid]
        yrow, mrow = [], []
        for object_name in OBJECTS:
            label = label_row["objects"][object_name]["label"]
            object_counts[object_name][label] += 1
            yrow.append(1.0 if label == "inside" else 0.0)
            mrow.append(label != "uncertain")
        ys.append(yrow)
        masks.append(mrow)

    seeds_np, t_np = np.asarray(seeds, np.int64), np.asarray(times, np.int64)
    t_norm = np.zeros(len(times), dtype=np.float64)
    for seed in sorted(set(seeds)):
        ii = np.flatnonzero(seeds_np == seed)
        denom = max(1, int(t_np[ii].max()))
        t_norm[ii] = t_np[ii] / denom
    y = torch.tensor(ys, dtype=torch.float32)
    mask = torch.tensor(masks, dtype=torch.bool)
    coverage = {
        "manifest_frames": len(ordered),
        "visual_teacher_rows": len(labels_by_frame),
        "joined_visual_teacher_rows": len(labels_by_frame),
        "exact_frame_bijection": len(labels_by_frame) == len(ordered),
        "objects": object_counts,
        "confident_labels": int(mask.sum()),
        "possible_object_labels": int(mask.numel()),
        "confident_fraction": float(mask.float().mean()),
    }
    ds = Dataset(
        z=z_all[torch.tensor(indices)], y=y, mask=mask, frame_ids=frame_ids,
        row_indices=np.asarray(indices), episode_ids=np.asarray(eps),
        env_seeds=seeds_np, t=t_np, t_norm=t_norm, task=expected_task,
        tape_sha256=tape_sha, proprio_dim=proprio_dim, coverage=coverage,
    )
    audit = {
        "tape_rows": len(z_all), "joined_rows": len(ds),
        "latent_dim": int(z_all.shape[1]), "proprio_dim": proprio_dim,
        "episodes": len(set(eps)), "env_seeds": len(set(seeds)),
        "row_indices_unique": len(indices) == len(set(indices)),
        "frame_ids_unique": len(frame_ids) == len(set(frame_ids)),
        "declared_eval_summary_sha256": registration_summary_sha(replay_rows),
    }
    return ds, audit


def registration_summary_sha(replay_rows: list[dict[str, Any]]) -> str:
    declared = {str(row["summary_sha256"]) for row in replay_rows}
    if len(declared) != 1:
        raise ValueError("replay rows must declare exactly one source summary SHA")
    value = next(iter(declared))
    if not is_sha256(value):
        raise ValueError("replay source summary SHA is invalid")
    return value


def registration_split(
    episode_ids: np.ndarray, registration: dict[str, Any]
) -> tuple[np.ndarray, dict[str, list[int]]]:
    groups = copy.deepcopy(registration["groups"])
    assignment = {
        episode_id: name
        for name, group_episode_ids in groups.items()
        for episode_id in group_episode_ids
    }
    if sorted(set(int(value) for value in episode_ids)) != list(range(16, 96)):
        raise ValueError("joined dataset must cover exactly episode IDs 16..95")
    split = np.asarray([assignment[int(value)] for value in episode_ids], dtype=object)
    for name, expected_ids in groups.items():
        actual_ids = sorted(set(int(value) for value in episode_ids[split == name]))
        if actual_ids != expected_ids:
            raise ValueError(f"{name} registration split mismatch")
    return split, groups


class ObjectPhi(nn.Module):
    """Three object logits; calibrated ordinal Phi is sum(sigmoid(logit/T))."""

    def __init__(self, input_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden), nn.GELU(),
            nn.Dropout(0.05), nn.Linear(hidden, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class FitResult:
    name: str
    model: ObjectPhi
    feature_kind: str
    mu: torch.Tensor
    sd: torch.Tensor
    temperatures: torch.Tensor
    history: list[dict[str, float]]
    best_epoch: int
    best_calib_bce: float
    pos_weight: torch.Tensor
    transformed_label_sha256: str


def raw_features(ds: Dataset, kind: str) -> torch.Tensor:
    if kind == "full":
        return ds.z
    if kind == "proprio":
        return ds.z[:, -ds.proprio_dim:]
    if kind == "time":
        t = torch.tensor(ds.t_norm, dtype=torch.float32)
        return torch.stack((t, t.square()), dim=1)
    raise ValueError(kind)


def tensor_sha256(*items: torch.Tensor) -> str:
    h = hashlib.sha256()
    for x in items:
        a = x.detach().cpu().contiguous().numpy()
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def permuted_targets(ds: Dataset, split: np.ndarray, mode: str,
                     seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    y, mask = ds.y.clone(), ds.mask.clone()
    if mode == "none":
        return y, mask
    rng = np.random.default_rng(seed)
    y2, m2 = y.clone(), mask.clone()
    for split_name in ("train", "calib", "test"):
        base = np.flatnonzero(split == split_name)
        if mode == "label_shuffle":
            for obj in range(3):
                perm = base[rng.permutation(len(base))]
                y2[torch.tensor(base), obj] = y[torch.tensor(perm), obj]
                m2[torch.tensor(base), obj] = mask[torch.tensor(perm), obj]
        else:
            raise ValueError(mode)
    return y2, m2


def masked_bce(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor,
               pos_weight: torch.Tensor | None = None) -> torch.Tensor:
    if not bool(mask.any()):
        raise ValueError("batch has no confident visual-teacher labels")
    loss = F.binary_cross_entropy_with_logits(
        logits, y, reduction="none", pos_weight=pos_weight
    )
    return loss[mask].mean()


@torch.no_grad()
def logits_for(model: nn.Module, raw_x: torch.Tensor, mu: torch.Tensor,
               sd: torch.Tensor, indices: np.ndarray, device: torch.device,
               batch_size: int) -> torch.Tensor:
    model.eval()
    chunks = []
    for lo in range(0, len(indices), batch_size):
        ii = torch.tensor(indices[lo:lo + batch_size], dtype=torch.long)
        x = ((raw_x[ii] - mu) / sd).to(device)
        chunks.append(model(x).cpu())
    return torch.cat(chunks) if chunks else torch.empty(0, 3)


def calibrate_temperatures(logits: torch.Tensor, y: torch.Tensor,
                           mask: torch.Tensor) -> torch.Tensor:
    temps = torch.ones(3)
    grid = torch.logspace(math.log10(0.20), math.log10(5.0), 61)
    for obj in range(3):
        m = mask[:, obj]
        if int(m.sum()) < 2 or len(torch.unique(y[m, obj])) < 2:
            continue
        losses = torch.stack([
            F.binary_cross_entropy_with_logits(logits[m, obj] / t, y[m, obj])
            for t in grid
        ])
        temps[obj] = grid[int(torch.argmin(losses))]
    return temps


def fit_head(name: str, ds: Dataset, split: np.ndarray, y: torch.Tensor,
             mask: torch.Tensor, feature_kind: str, args: argparse.Namespace,
             seed_offset: int) -> FitResult:
    raw_x = raw_features(ds, feature_kind).float().cpu()
    tr = np.flatnonzero((split == "train") & mask.any(1).numpy())
    ca = np.flatnonzero((split == "calib") & mask.any(1).numpy())
    if len(tr) < 8 or len(ca) < 4:
        raise ValueError(f"{name}: insufficient labelled train/calib rows")
    mu = raw_x[torch.tensor(tr)].mean(0)
    sd = raw_x[torch.tensor(tr)].std(0, unbiased=False).clamp_min(1e-5)
    positives = (y[torch.tensor(tr)] * mask[torch.tensor(tr)]).sum(0)
    negatives = ((1 - y[torch.tensor(tr)]) * mask[torch.tensor(tr)]).sum(0)
    if bool((positives == 0).any()) or bool((negatives == 0).any()):
        raise ValueError(f"{name}: every object needs inside and outside train labels")
    pos_weight = (negatives / positives).clamp(0.25, 8.0)

    seed = int(args.model_seed + seed_offset)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ObjectPhi(raw_x.shape[1], args.hidden).to(args.device_obj)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(seed + 1)
    best_loss, best_epoch, wait = float("inf"), -1, 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    for epoch in range(args.epochs):
        model.train()
        order = torch.tensor(tr)[torch.randperm(len(tr), generator=generator)]
        train_losses = []
        for lo in range(0, len(order), args.batch):
            ii = order[lo:lo + args.batch]
            xb = ((raw_x[ii] - mu) / sd).to(args.device_obj)
            yb, mb = y[ii].to(args.device_obj), mask[ii].to(args.device_obj)
            logits = model(xb)
            loss = masked_bce(logits, yb, mb, pos_weight.to(args.device_obj))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_losses.append(float(loss.detach().cpu()))
        calib_logits = logits_for(model, raw_x, mu, sd, ca, args.device_obj,
                                  args.eval_batch)
        calib_loss = float(masked_bce(calib_logits, y[torch.tensor(ca)],
                                      mask[torch.tensor(ca)]))
        history.append({
            "epoch": epoch,
            "train_weighted_bce": float(np.mean(train_losses)),
            "calib_visual_teacher_bce": calib_loss,
        })
        if calib_loss < best_loss - 1e-6:
            best_loss, best_epoch, wait = calib_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            wait += 1
        if wait >= args.patience:
            break
    if best_state is None:
        raise AssertionError("no model selected")
    model.load_state_dict(best_state)
    model.to(args.device_obj).eval()
    calib_logits = logits_for(model, raw_x, mu, sd, ca, args.device_obj,
                              args.eval_batch)
    temperatures = calibrate_temperatures(
        calib_logits, y[torch.tensor(ca)], mask[torch.tensor(ca)]
    )
    return FitResult(
        name=name, model=model, feature_kind=feature_kind, mu=mu, sd=sd,
        temperatures=temperatures, history=history, best_epoch=best_epoch,
        best_calib_bce=best_loss, pos_weight=pos_weight,
        transformed_label_sha256=tensor_sha256(y, mask),
    )


def predict_ordinal(
    fit: FitResult, ds: Dataset, args: argparse.Namespace
) -> tuple[np.ndarray, np.ndarray]:
    raw_x = raw_features(ds, fit.feature_kind).float().cpu()
    idx = np.arange(len(ds))
    logits = logits_for(fit.model, raw_x, fit.mu, fit.sd, idx,
                        args.device_obj, args.eval_batch)
    probs = torch.sigmoid(logits / fit.temperatures.view(1, 3))
    return probs.sum(1).numpy(), probs.numpy()


def _rank_average(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts), dtype=float)
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra, rb = _rank_average(a), _rank_average(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = math.sqrt(float(np.dot(ra, ra) * np.dot(rb, rb)))
    return float(np.dot(ra, rb) / denom) if denom > 1e-12 else 0.0


def partial_spearman_time(pred: np.ndarray, stage: np.ndarray,
                          t_norm: np.ndarray) -> float:
    """Spearman residual correlation after controlling normalized episode time."""
    if len(pred) < 4:
        return float("nan")
    rp, rs, rt = _rank_average(pred), _rank_average(stage), _rank_average(t_norm)
    design = np.stack((np.ones(len(rt)), rt), axis=1)
    xp = rp - design @ np.linalg.lstsq(design, rp, rcond=None)[0]
    xs = rs - design @ np.linalg.lstsq(design, rs, rcond=None)[0]
    denom = math.sqrt(float(np.dot(xp, xp) * np.dot(xs, xs)))
    # A pure clock has no residual variance and is exactly the intended zero null.
    return float(np.dot(xp, xs) / denom) if denom > 1e-12 else 0.0


def exact_time_pairwise(pred: np.ndarray, stage: np.ndarray, t: np.ndarray,
                        groups: np.ndarray) -> dict[str, float | int]:
    """Cross-episode signed concordance at exactly equal recorded time bins."""
    numerator, comparable = 0.0, 0
    for tt in sorted(set(int(x) for x in t)):
        ii = np.flatnonzero(t == tt)
        for j, left in enumerate(ii[:-1]):
            right = ii[j + 1:]
            keep = groups[right] != groups[left]
            right = right[keep]
            if not len(right):
                continue
            dy = stage[right] - stage[left]
            valid = dy != 0
            if not np.any(valid):
                continue
            dx = pred[right[valid]] - pred[left]
            numerator += float(np.sign(dx).dot(np.sign(dy[valid])))
            comparable += int(valid.sum())
    rho = numerator / comparable if comparable else float("nan")
    return {
        "pairwise_rho": rho,
        "concordance": (rho + 1.0) / 2.0 if math.isfinite(rho) else float("nan"),
        "comparable_pairs": comparable,
    }


def metric_set(pred: np.ndarray, stage: np.ndarray, t: np.ndarray,
               t_norm: np.ndarray, groups: np.ndarray) -> dict[str, float | int]:
    pair = exact_time_pairwise(pred, stage, t, groups)
    means, terminals, finals = [], [], []
    for group in sorted(set(int(x) for x in groups)):
        ii = np.flatnonzero(groups == group)
        last = ii[np.argmax(t[ii])]
        means.append(float(np.mean(pred[ii])))
        terminals.append(float(pred[last]))
        finals.append(float(stage[last]))
    return {
        "partial_spearman_time": partial_spearman_time(pred, stage, t_norm),
        **pair,
        "episode_mean_rho": spearman(np.asarray(means), np.asarray(finals)),
        "episode_terminal_rho": spearman(np.asarray(terminals), np.asarray(finals)),
        "rows": len(pred),
        "episode_groups": len(set(int(x) for x in groups)),
    }


def episode_bootstrap(pred: np.ndarray, stage: np.ndarray, t: np.ndarray,
                      t_norm: np.ndarray, groups: np.ndarray, n_boot: int,
                      seed: int) -> dict[str, Any]:
    point = metric_set(pred, stage, t, t_norm, groups)
    keys = ("partial_spearman_time", "pairwise_rho", "episode_mean_rho",
            "episode_terminal_rho")
    draws = {k: [] for k in keys}
    unique = np.asarray(sorted(set(int(x) for x in groups)))
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        sampled = rng.choice(unique, len(unique), replace=True)
        chunks, boot_groups = [], []
        for new_group, old_group in enumerate(sampled):
            ii = np.flatnonzero(groups == old_group)
            chunks.append(ii)
            boot_groups.extend([new_group] * len(ii))
        jj = np.concatenate(chunks)
        m = metric_set(pred[jj], stage[jj], t[jj], t_norm[jj],
                       np.asarray(boot_groups))
        for key in keys:
            value = float(m[key])
            if math.isfinite(value):
                draws[key].append(value)
    ci: dict[str, Any] = {}
    for key in keys:
        values = np.asarray(draws[key], dtype=float)
        ci[key] = {
            "low": float(np.quantile(values, .025)) if len(values) else float("nan"),
            "high": float(np.quantile(values, .975)) if len(values) else float("nan"),
            "finite_draws": len(values),
        }
    return {"point": point, "episode_bootstrap_95ci": ci,
            "bootstrap_resamples": n_boot, "bootstrap_seed": seed}


def load_eval_stages(summary_path: Path, ds: Dataset) -> dict[str, np.ndarray]:
    """The only privileged-data loader.  It is called strictly after sealing."""
    summary = load_json_object(summary_path)
    if summary.get("task") != ds.task:
        raise ValueError("summary task mismatch")
    records = {int(r["idx"]): r for r in summary["episode_records"]}
    object_stage = np.zeros(len(ds), dtype=float)
    milestone_stage = np.zeros(len(ds), dtype=float)
    per_object = np.zeros((len(ds), 3), dtype=float)
    for i, (episode_id, env_seed, tt) in enumerate(
            zip(ds.episode_ids, ds.env_seeds, ds.t)):
        if int(episode_id) not in records:
            raise ValueError(f"summary missing episode {episode_id}")
        rec = records[int(episode_id)]
        if int(rec["seed"]) != int(env_seed):
            raise ValueError(f"summary env_seed mismatch for episode {episode_id}")
        events = {int(k): int(v) for k, v in rec.get("events", {}).items()}
        milestone_stage[i] = sum(step <= int(tt) for step in events.values())
        # V080 interleaves pick_up (even event index) and place (odd index).
        for obj in range(3):
            place_index = 2 * obj + 1
            per_object[i, obj] = float(place_index in events and
                                       events[place_index] <= int(tt))
        object_stage[i] = per_object[i].sum()
    return {"object_stage": object_stage, "milestone_stage": milestone_stage,
            "per_object_inside": per_object}


def visual_teacher_agreement(
    ds: Dataset,
    stages: dict[str, np.ndarray],
    test: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    y, mask = ds.y.numpy(), ds.mask.numpy()
    idx = np.flatnonzero(test)
    complete = idx[mask[idx].all(1)]
    teacher = y[complete].sum(1)
    stage = stages["object_stage"][complete]
    metrics = episode_bootstrap(
        teacher, stage, ds.t[complete], ds.t_norm[complete],
        ds.env_seeds[complete], n_boot, seed
    ) if len(complete) >= 4 else None
    obj_rows: dict[str, Any] = {}
    truth = stages["per_object_inside"]
    for obj_i, obj in enumerate(OBJECTS):
        jj = idx[mask[idx, obj_i]]
        correct = (y[jj, obj_i] == truth[jj, obj_i]).astype(float)
        by_episode = []
        for env_seed in sorted(set(int(x) for x in ds.env_seeds[jj])):
            kk = jj[ds.env_seeds[jj] == env_seed]
            by_episode.append(float(np.mean(y[kk, obj_i] == truth[kk, obj_i])))
        rng = np.random.default_rng(seed + obj_i + 1)
        draws = (
            [
                float(np.mean(rng.choice(by_episode, len(by_episode), replace=True)))
                for _ in range(n_boot)
            ]
            if by_episode
            else []
        )
        obj_rows[obj] = {
            "confident_test_labels": len(jj),
            "accuracy_vs_eval_only_inside_event": (
                float(correct.mean()) if len(correct) else float("nan")
            ),
            "episode_bootstrap_95ci": (
                [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]
                if draws
                else [float("nan"), float("nan")]
            ),
        }
    return {
        "complete_confident_test_frames": len(complete),
        "exact_ordinal_agreement": (
            float(np.mean(teacher == stage)) if len(complete) else float("nan")
        ),
        "teacher_stage_metrics": metrics,
        "per_object": obj_rows,
        "note": "agreement is against simulator placement events and is evaluation-only",
    }


def object_identity_permutation_diagnostic(
        probabilities: np.ndarray, stages: dict[str, np.ndarray], ds: Dataset,
        test: np.ndarray, n_boot: int, seed: int) -> dict[str, Any]:
    """Identity-sensitive object null, separate from the invariant ordinal sum.

    A channel permutation cannot be a null for ``sum_j p_j``: the sum is exactly
    permutation invariant.  Instead, compare each named probability channel with
    its corresponding evaluation-only place milestone, then cyclically permute
    the probability-to-object mapping.  This is mandatory diagnostic evidence,
    but it is deliberately not included in the scalar-Phi promotion conjunction.
    """
    idx = np.flatnonzero(test)
    truth = stages["per_object_inside"][idx]
    probs = probabilities[idx]
    times, times_norm = ds.t[idx], ds.t_norm[idx]
    episode_groups = ds.env_seeds[idx]

    def mapped_metrics(mapping: tuple[int, int, int], take: np.ndarray,
                       groups: np.ndarray) -> dict[str, Any]:
        rows: dict[str, Any] = {}
        partials, pairs, briers = [], [], []
        for target_obj, source_obj in enumerate(mapping):
            partial = partial_spearman_time(
                probs[take, source_obj], truth[take, target_obj], times_norm[take]
            )
            pair = exact_time_pairwise(
                probs[take, source_obj], truth[take, target_obj], times[take], groups,
            )
            brier = float(np.mean(
                (probs[take, source_obj] - truth[take, target_obj]) ** 2
            ))
            rows[OBJECTS[target_obj]] = {
                "prediction_channel": OBJECTS[source_obj],
                "partial_spearman_time": partial,
                "pairwise_rho": pair["pairwise_rho"],
                "comparable_pairs": pair["comparable_pairs"],
                "brier": brier,
            }
            partials.append(partial)
            pairs.append(float(pair["pairwise_rho"]))
            briers.append(brier)
        finite_partial = [v for v in partials if math.isfinite(float(v))]
        finite_pairs = [v for v in pairs if math.isfinite(float(v))]
        return {
            "per_target_object": rows,
            "macro_partial_spearman_time": (
                float(np.mean(finite_partial)) if finite_partial else float("nan")
            ),
            "macro_pairwise_rho": (
                float(np.mean(finite_pairs)) if finite_pairs else float("nan")
            ),
            "macro_brier": float(np.mean(briers)),
        }

    take_all = np.arange(len(idx))
    correct = mapped_metrics((0, 1, 2), take_all, episode_groups)
    permuted = mapped_metrics((1, 2, 0), take_all, episode_groups)
    draws = {"correct_partial": [], "correct_pairwise": [],
             "permuted_partial": [], "permuted_pairwise": [],
             "delta_partial": [], "delta_pairwise": []}
    unique = np.asarray(sorted(set(int(x) for x in episode_groups)))
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        sampled = rng.choice(unique, len(unique), replace=True)
        chunks, boot_groups = [], []
        for new_group, old_group in enumerate(sampled):
            jj = np.flatnonzero(episode_groups == old_group)
            chunks.append(jj)
            boot_groups.extend([new_group] * len(jj))
        take = np.concatenate(chunks)
        groups = np.asarray(boot_groups)
        c = mapped_metrics((0, 1, 2), take, groups)
        p = mapped_metrics((1, 2, 0), take, groups)
        values = {
            "correct_partial": c["macro_partial_spearman_time"],
            "correct_pairwise": c["macro_pairwise_rho"],
            "permuted_partial": p["macro_partial_spearman_time"],
            "permuted_pairwise": p["macro_pairwise_rho"],
            "delta_partial": (c["macro_partial_spearman_time"] -
                              p["macro_partial_spearman_time"]),
            "delta_pairwise": c["macro_pairwise_rho"] - p["macro_pairwise_rho"],
        }
        for key, value in values.items():
            if math.isfinite(float(value)):
                draws[key].append(float(value))
    ci = {
        key: {
            "low": float(np.quantile(values, .025)) if values else float("nan"),
            "high": float(np.quantile(values, .975)) if values else float("nan"),
            "finite_draws": len(values),
        }
        for key, values in draws.items()
    }
    invariant_delta = float(np.max(np.abs(probs.sum(1) - probs[:, (1, 2, 0)].sum(1))))
    return {
        "correct_object_mapping": correct,
        "cyclic_object_mapping_null": permuted,
        "correct_minus_permuted_macro_partial":
            correct["macro_partial_spearman_time"] - permuted["macro_partial_spearman_time"],
        "correct_minus_permuted_macro_pairwise":
            correct["macro_pairwise_rho"] - permuted["macro_pairwise_rho"],
        "ordinal_sum_max_abs_change_under_permutation": invariant_delta,
        "ordinal_sum_is_permutation_invariant": invariant_delta <= 1e-7,
        "episode_bootstrap_95ci": ci,
        "bootstrap_resamples": n_boot,
        "bootstrap_seed": seed,
        "gate_role": "mandatory identity diagnostic; not a scalar-Phi null gate",
    }


def fit_metadata(fit: FitResult) -> dict[str, Any]:
    return {
        "name": fit.name, "feature_kind": fit.feature_kind,
        "best_epoch": fit.best_epoch, "epochs_ran": len(fit.history),
        "best_calib_visual_teacher_bce": fit.best_calib_bce,
        "temperatures": fit.temperatures, "pos_weight": fit.pos_weight,
        "transformed_label_sha256": fit.transformed_label_sha256,
        "history": fit.history,
    }


def run(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    tape_path = args.tape.resolve()
    registration_path = args.registration.resolve()
    manifest_paths = [path.resolve() for path in args.replay_manifest]
    teacher_output_paths = [path.resolve() for path in args.teacher_output]
    if len(manifest_paths) != len(set(manifest_paths)):
        raise ValueError("duplicate --replay-manifest path")
    if len(teacher_output_paths) != len(set(teacher_output_paths)):
        raise ValueError("duplicate --teacher-output path")

    # This fail-closed PASS check is deliberately first.  In particular, a
    # failed tracker gate must stop before torch.load or any source-summary read.
    registration = validate_registration(
        registration_path,
        synthetic_contract=bool(getattr(args, "_synthetic_contract", False)),
    )
    tracker_gate = validate_tracker_gate(
        args.tracker_gate,
        registration,
        manifest_paths,
        teacher_output_paths,
    )
    replay_rows, labels_by_frame, visual_audit = validate_visual_sources(
        manifest_paths,
        teacher_output_paths,
        registration,
        tracker_gate,
    )
    if not visual_audit["exact_frame_bijection"]:
        raise ValueError("visual teacher coverage gate failed before tape access")

    ds, join_audit = load_and_join(
        tape_path, replay_rows, labels_by_frame, args.task
    )
    if ds.tape_sha256 != registration["payload"]["target"]["source_tape_sha256"]:
        raise ValueError("loaded tape SHA differs from frozen registration")
    if join_audit["declared_eval_summary_sha256"] != tracker_gate["payload"][
        "source_summary_sha256"
    ]:
        raise ValueError("joined source-summary declaration differs from tracker gate")
    split, split_groups = registration_split(ds.episode_ids, registration)
    for name in ("train", "calib", "test"):
        confident = int(ds.mask[torch.tensor(split == name)].sum())
        if confident == 0:
            raise ValueError(f"{name} split has no confident visual-teacher labels")

    if args.out is None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
        out = OUT / f"{args.task}_{stamp}"
    else:
        out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{out.name}.tmp-", dir=out.parent))
    try:
        targets: dict[str, tuple[torch.Tensor, torch.Tensor, str]] = {
            "full": (*permuted_targets(ds, split, "none", args.model_seed), "full"),
            "time_only": (
                *permuted_targets(ds, split, "none", args.model_seed),
                "time",
            ),
            "proprio_only": (
                *permuted_targets(ds, split, "none", args.model_seed),
                "proprio",
            ),
            "label_shuffle": (
                *permuted_targets(
                    ds, split, "label_shuffle", args.model_seed + 31
                ),
                "full",
            ),
        }

        # PHASE A: observation-only visual-teacher fitting and calibration.
        # Never move any eval-summary filesystem access above PHI_SEALED.json.
        fits: dict[str, FitResult] = {}
        predictions: dict[str, np.ndarray] = {}
        object_probabilities: dict[str, np.ndarray] = {}
        for name in ARM_NAMES:
            y, mask, feature_kind = targets[name]
            print(f"fit {name:20s} feature={feature_kind}", flush=True)
            fits[name] = fit_head(
                name,
                ds,
                split,
                y,
                mask,
                feature_kind,
                args,
                seed_offset=0,
            )
            predictions[name], object_probabilities[name] = predict_ordinal(
                fits[name], ds, args
            )
            print(
                f"  best epoch {fits[name].best_epoch}, visual-teacher calib BCE "
                f"{fits[name].best_calib_bce:.4f}",
                flush=True,
            )

        input_hashes = {
            "tape": sha256_file(tape_path),
            "registration": registration["sha256"],
            "tracker_gate": tracker_gate["sha256"],
            "replay_manifests": {
                value["path"]: value["sha256"]
                for value in visual_audit["manifests"]
            },
            "visual_teacher_labels": {
                value["labels_path"]: value["labels_sha256"]
                for value in visual_audit["teacher_artifacts"]
            },
            "visual_teacher_results": {
                value["run_path"]: value["result_sha256"]
                for value in visual_audit["teacher_artifacts"]
            },
            "code": sha256_file(Path(__file__).resolve()),
        }
        primary = fits["full"]
        checkpoint = {
            "schema": PHI_SCHEMA,
            "task": ds.task,
            "objects": list(OBJECTS),
            "latent_dim": int(ds.z.shape[1]),
            "proprio_dim": ds.proprio_dim,
            "hidden": args.hidden,
            "state_dict": {
                key: value.detach().cpu()
                for key, value in primary.model.state_dict().items()
            },
            "mu": primary.mu.detach().cpu(),
            "sd": primary.sd.detach().cpu(),
            "temperatures": primary.temperatures.detach().cpu(),
            "output_contract": "sum_j sigmoid(object_logit_j / temperature_j)",
            "training": fit_metadata(primary),
            "provenance": {
                "utc_sealed": datetime.now(timezone.utc).isoformat(),
                "input_paths": {
                    "tape": str(tape_path),
                    "registration": str(registration_path),
                    "tracker_gate": tracker_gate["path"],
                    "replay_manifests": [str(path) for path in manifest_paths],
                    "visual_teacher_outputs": [
                        str(path) for path in teacher_output_paths
                    ],
                    "eval_summary": "deferred until after this checkpoint seal",
                },
                "input_sha256": input_hashes,
                "argv": list(sys.argv),
                "git": git_head(),
                "model_seed": args.model_seed,
                "registration_episode_ids": split_groups,
                "selection_metric": (
                    "masked calibration BCE on observation-only SAM2 visual labels"
                ),
                "calibration_source": (
                    "SAM2 RGB inside/outside visual labels from episodes 64..79 only"
                ),
                "uncertain_policy": "masked; missing label fatal",
                "tracker_gate_passed_before_tape_load": True,
                "privileged_summary_opened_before_seal": False,
                "success_or_events_used_for_training_early_stop_or_selection": False,
            },
        }
        checkpoint_path = temporary / "phi.pt"
        torch.save(checkpoint, checkpoint_path)
        checkpoint_sha = sha256_file(checkpoint_path)
        phi_seal = {
            "schema": "v242_vlm_phi_pre_privileged_seal_v1",
            "status": "sealed_before_privileged_summary_access",
            "phi_relative_path": "phi.pt",
            "phi_sha256": checkpoint_sha,
            "source_summary_sha256_declared_only": tracker_gate["payload"][
                "source_summary_sha256"
            ],
            "registration_sha256": registration["sha256"],
            "tracker_gate_sha256": tracker_gate["sha256"],
        }
        phi_seal_path = temporary / "PHI_SEALED.json"
        write_json(phi_seal_path, phi_seal)
        phi_seal_sha = sha256_file(phi_seal_path)
        print(f"sealed phi.pt sha256={checkpoint_sha[:16]}...", flush=True)

        # PHASE B: the first privileged-summary filesystem access in this run.
        # Nothing learned or selected above can be changed after this point.
        summary_path = args.eval_summary.resolve()
        if not summary_path.is_file():
            raise FileNotFoundError(
                f"evaluation-only source summary missing: {summary_path}"
            )
        summary_sha = sha256_file(summary_path)
        if summary_sha != tracker_gate["payload"]["source_summary_sha256"]:
            raise ValueError("evaluation summary differs from tracker gate declaration")
        if summary_sha != join_audit["declared_eval_summary_sha256"]:
            raise ValueError("evaluation summary differs from replay declarations")
        stages = load_eval_stages(summary_path, ds)
        test = split == "test"
        eval_rows: dict[str, Any] = {}
        for arm_i, name in enumerate(ARM_NAMES):
            idx = np.flatnonzero(test)
            eval_rows[name] = episode_bootstrap(
                predictions[name][idx],
                stages["object_stage"][idx],
                ds.t[idx],
                ds.t_norm[idx],
                ds.env_seeds[idx],
                args.bootstrap,
                args.bootstrap_seed + arm_i * 1009,
            )

        signal = eval_rows["full"]
        signal_checks = {}
        for key in ("partial_spearman_time", "pairwise_rho"):
            point = float(signal["point"][key])
            lower = float(signal["episode_bootstrap_95ci"][key]["low"])
            signal_checks[key] = {
                "point": point,
                "lower": lower,
                "point_at_least_0.157": point >= PRIMARY_RHO_FLOOR,
                "lower_above_0.078": lower > PRIMARY_CI_FLOOR,
                "passed": (
                    point >= PRIMARY_RHO_FLOOR and lower > PRIMARY_CI_FLOOR
                ),
            }
        null_checks = {}
        for name in ARM_NAMES[1:]:
            values = {
                key: float(eval_rows[name]["point"][key])
                for key in ("partial_spearman_time", "pairwise_rho")
            }
            null_checks[name] = {
                **values,
                "abs_each_at_most_0.1": all(
                    math.isfinite(value) and abs(value) <= NULL_ABS_CEILING
                    for value in values.values()
                ),
            }
        gate_passed = (
            all(value["passed"] for value in signal_checks.values())
            and all(
                value["abs_each_at_most_0.1"] for value in null_checks.values()
            )
        )

        final_checkpoint_path = out / "phi.pt"
        result = {
            "schema": RESULT_SCHEMA,
            "utc": datetime.now(timezone.utc).isoformat(),
            "task": ds.task,
            "env_steps": 0,
            "checkpoint": str(final_checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_schema": PHI_SCHEMA,
            "pre_privileged_phi_seal_sha256": phi_seal_sha,
            "tracker_gate": {
                "path": tracker_gate["path"],
                "sha256": tracker_gate["sha256"],
                "passed": True,
            },
            "registration": {
                "path": str(registration_path),
                "sha256": registration["sha256"],
            },
            "visual_source_audit": visual_audit,
            "join_audit": join_audit,
            "visual_teacher_coverage": ds.coverage,
            "split": {
                "source": "frozen registration episode partition",
                "episode_ids": split_groups,
                "rows": {
                    name: int(np.sum(split == name))
                    for name in ("train", "calib", "test")
                },
                "confident_labels": {
                    name: int(ds.mask[torch.tensor(split == name)].sum())
                    for name in ("train", "calib", "test")
                },
            },
            "training": {name: fit_metadata(fits[name]) for name in ARM_NAMES},
            "evaluation_only": {
                "summary_path": str(summary_path),
                "summary_sha256": summary_sha,
                "stage_definition": (
                    "count of V080 place-event indices 1,3,5 achieved by row t"
                ),
                "all_milestone_stage_computed_but_not_used_for_gate": True,
                "arms": eval_rows,
                "visual_teacher_agreement": visual_teacher_agreement(
                    ds,
                    stages,
                    test,
                    args.bootstrap,
                    args.bootstrap_seed + 9001,
                ),
                "object_identity_permutation": object_identity_permutation_diagnostic(
                    object_probabilities["full"],
                    stages,
                    ds,
                    test,
                    args.bootstrap,
                    args.bootstrap_seed + 12001,
                ),
            },
            "gate": {
                "thresholds": {
                    "co_primary_point_min": PRIMARY_RHO_FLOOR,
                    "co_primary_episode_bootstrap_lower_strict_min": PRIMARY_CI_FLOOR,
                    "null_abs_max": NULL_ABS_CEILING,
                },
                "co_primary_signal_checks": signal_checks,
                "null_checks": null_checks,
                "passed": gate_passed,
                "decision": "ADVANCE" if gate_passed else "STOP",
            },
            "provenance": {
                "argv": list(sys.argv),
                "git": git_head(),
                "input_sha256": {
                    **input_hashes,
                    "eval_only_summary": summary_sha,
                },
                "registration_episode_ids": split_groups,
                "privileged_data_phase": (
                    "summary opened only after immutable phi.pt was saved and hashed"
                ),
                "success_read_or_used": False,
                "events_used_for": "post-seal Gate evaluation only",
            },
        }
        result_path = temporary / "summary.json"
        write_json(result_path, result)
        complete = {
            "schema": "v242_vlm_phi_gate_complete_v1",
            "status": "complete",
            "phi_sha256": checkpoint_sha,
            "phi_seal_sha256": phi_seal_sha,
            "result_sha256": sha256_file(result_path),
            "gate_passed": gate_passed,
            "decision": "ADVANCE" if gate_passed else "STOP",
        }
        write_json(temporary / "COMPLETE.json", complete)
        os.replace(temporary, out)
        try:
            print(
                "Gate 0 " + ("PASS / ADVANCE" if gate_passed else "FAIL / STOP"),
                flush=True,
            )
            print(f"-> {out}", flush=True)
        except (BrokenPipeError, OSError):
            # The atomic COMPLETE artifact is authoritative; a closed status
            # stream must not turn a completed run into a false process failure.
            pass
        return out, result
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def make_synthetic_fixture(
    root: Path,
    frames_per_episode: int = 7,
    zdim: int = 49,
) -> dict[str, Any]:
    """Create a CPU-small formal-contract fixture; never scientific evidence."""
    rng = np.random.default_rng(242)
    zs, us, zn, ep, times = [], [], [], [], []
    summary_records, replay_rows, label_rows = [], [], []
    latent_rows: list[tuple[int, int, int, np.ndarray, list[int]]] = []
    for episode in range(16, 96):
        env_seed = 8700 + episode
        place_times = [
            10 + 10 * (episode % 3),
            30 + 10 * ((episode // 3) % 3),
            40 + 10 * ((episode // 9) % 3),
        ]
        events: dict[str, int] = {}
        for obj, pt in enumerate(place_times):
            events[str(2 * obj)] = max(0, pt - 10)
            if pt <= (frames_per_episode - 1) * 10:
                events[str(2 * obj + 1)] = pt
        summary_records.append({
            "idx": episode, "seed": env_seed, "success": False,
            "success_step": None, "steps": frames_per_episode * 10,
            "events": events,
        })
        for chunk in range(frames_per_episode):
            tt = chunk * 10
            inside = [int(pt <= tt) for pt in place_times]
            z = rng.normal(0, 0.35, zdim).astype(np.float32)
            z[:3] += np.asarray(inside) * 5.0 - 2.5
            # Last 25 dimensions are independent proprioception by construction.
            z[-25:] = rng.normal(0, 1, 25)
            latent_rows.append((episode, env_seed, tt, z, inside))
            zs.append(torch.tensor(z))
            us.append(torch.zeros(10, 7))
            zn.append(torch.tensor(z + rng.normal(0, .01, zdim).astype(np.float32)))
            ep.append(episode)
            times.append(tt)
    tape = root / "tape.pt"
    torch.save(
        {
            "z": torch.stack(zs),
            "u": torch.stack(us),
            "z_next": torch.stack(zn),
            "episode": torch.tensor(ep),
            "t": torch.tensor(times),
            "sigma": torch.zeros(len(zs)),
            "success": torch.zeros(80),
            "task": "chain3_lr2",
            "c": 10,
            "latent_dim": zdim,
            "proprio_dim": 25,
            "proprio_keys": ["synthetic"] * 6,
        },
        tape,
    )
    source_summary = root / "source_summary.json"
    write_json(
        source_summary,
        {
            "task": "chain3_lr2",
            "episodes": 80,
            "successes": 0,
            "episode_records": summary_records,
            "note": "synthetic contract fixture",
        },
    )
    tape_sha = sha256_file(tape)
    source_summary_sha = sha256_file(source_summary)
    for row_index, (episode, env_seed, tt, _z, inside) in enumerate(latent_rows):
        fid = f"{tape_sha[:12]}:ep{episode}:t{tt}"
        image_sha = hashlib.sha256(f"synthetic-image|{fid}".encode()).hexdigest()
        replay_rows.append(
            {
                "schema": REPLAY_SCHEMA,
                "frame_id": fid,
                "task": "chain3_lr2",
                "tape_sha256": tape_sha,
                "summary_sha256": source_summary_sha,
                "episode_id": episode,
                "env_seed": env_seed,
                "t": tt,
                "row_index": row_index,
                "chunk_index": tt // 10,
                "images": {
                    "agentview": {
                        "path": f"synthetic/{fid}.jpg",
                        "sha256": image_sha,
                        "width": 360,
                        "height": 360,
                    }
                },
            }
        )
        objects = {}
        for obj_i, obj in enumerate(OBJECTS):
            if (row_index + obj_i * 17) % 101 == 0:
                objects[obj] = {"label": "uncertain"}
            else:
                objects[obj] = {"label": "inside" if inside[obj_i] else "outside"}
        label_rows.append(
            {
                "schema": LABEL_COMPAT_SCHEMA,
                "teacher_schema": TEACHER_SCHEMA,
                "producer": TEACHER_PRODUCER,
                "frame_id": fid,
                "task": "chain3_lr2",
                "parse_ok": True,
                "objects": objects,
                "episode_id": episode,
                "env_seed": env_seed,
                "t": tt,
                "row_index": row_index,
                "chunk_index": tt // 10,
                "camera": "agentview",
                "prompt_frame_index": 0,
                "post_t0_prompt_count": 0,
                "source_image_sha256": image_sha,
                "source_summary_sha256_declared_only": source_summary_sha,
            }
        )
    manifest_paths: list[Path] = []
    manifest_sha_by_frame: dict[str, str] = {}
    for name, rows in (
        ("fit", [row for row in replay_rows if int(row["episode_id"]) < 64]),
        ("calib_test", [row for row in replay_rows if int(row["episode_id"]) >= 64]),
    ):
        manifest = root / f"replay_{name}.jsonl"
        manifest.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        manifest_sha = sha256_file(manifest)
        manifest_paths.append(manifest)
        for row in rows:
            manifest_sha_by_frame[str(row["frame_id"])] = manifest_sha
    manifest_shas = sorted(manifest_sha_by_frame.values())
    manifest_shas = sorted(set(manifest_shas))

    checkpoint_sha = hashlib.sha256(b"synthetic-sam2-checkpoint").hexdigest()
    probe_sha = hashlib.sha256(b"synthetic-probe").hexdigest()
    partition = {
        "tracker_and_initializer_calibration_episode_ids": list(range(16)),
        "phi_fit_episode_ids": list(range(16, 64)),
        "phi_calibration_episode_ids": list(range(64, 80)),
        "phi_test_episode_ids": list(range(80, 96)),
        "environment_seed_rule": "env_seed = 8700 + episode_id",
        "disjoint": True,
    }
    registration_path = root / "registration.json"
    write_json(
        registration_path,
        {
            "schema": REGISTRATION_SCHEMA,
            "synthetic_contract_fixture": True,
            "status": "frozen_before_any_episode_16_to_95_tracking_output",
            "target": {
                "task": "chain3_lr2",
                "camera": "agentview",
                "source_tape_sha256": tape_sha,
                "chunk_steps": 10,
                "horizon_steps": 750,
            },
            "episode_partition": partition,
            "tracker": {
                "checkpoint_sha256": checkpoint_sha,
                "probe_code_sha256": probe_sha,
            },
        },
    )
    registration_sha = sha256_file(registration_path)
    for row in label_rows:
        row["replay_manifest_sha256"] = manifest_sha_by_frame[row["frame_id"]]
        row["registration_sha256"] = registration_sha
        row["checkpoint_sha256"] = checkpoint_sha

    teacher_output = root / "teacher_output"
    teacher_output.mkdir()
    labels = teacher_output / "labels.jsonl"
    labels.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in label_rows),
        encoding="utf-8",
    )
    teacher_code = REPO / "scripts" / "track_v243_sam2_teacher.py"
    teacher_code_sha = sha256_file(teacher_code)
    teacher_result = {
        "schema": TEACHER_RUN_SCHEMA,
        "status": "complete",
        "registration": {"path": str(registration_path), "sha256": registration_sha},
        "checkpoint": {"sha256": checkpoint_sha},
        "code": {"path": str(teacher_code), "sha256": teacher_code_sha},
        "inputs": {
            "manifests": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in manifest_paths
            ]
        },
        "outputs": {
            "labels_relative_path": "labels.jsonl",
            "labels_sha256": sha256_file(labels),
            "label_rows": len(label_rows),
        },
    }
    teacher_result_path = teacher_output / "result.json"
    write_json(teacher_result_path, teacher_result)
    write_json(
        teacher_output / "COMPLETE.json",
        {
            "schema": TEACHER_COMPLETE_SCHEMA,
            "status": "complete",
            "result_sha256": sha256_file(teacher_result_path),
            "labels_sha256": sha256_file(labels),
            "code_sha256": teacher_code_sha,
            "registration_sha256": registration_sha,
        },
    )

    teacher_artifact = {
        "run_path": str(teacher_output),
        "result_sha256": sha256_file(teacher_result_path),
        "labels_sha256": sha256_file(labels),
        "code_sha256": teacher_code_sha,
        "checkpoint_sha256": checkpoint_sha,
        "registration_sha256": registration_sha,
        "manifest_sha256s": manifest_shas,
    }
    gate_dir = root / "tracker_gate"
    gate_dir.mkdir()
    evaluator_code_sha = sha256_file(TRACKER_EVALUATOR_PATH)
    preseal_path = gate_dir / "pre_privileged_seal.json"
    write_json(
        preseal_path,
        {
            "schema": "synthetic_pre_privileged_seal_v1",
            "registration_sha256": registration_sha,
            "manifest_sha256s": manifest_shas,
        },
    )
    preseal_sha = sha256_file(preseal_path)
    gate_result_path = gate_dir / "result.json"
    write_json(
        gate_result_path,
        {
            "schema": "synthetic_tracker_gate_result_v1",
            "passed": True,
            "decision": "PASS",
            "pre_privileged_seal_sha256": preseal_sha,
        },
    )
    result_sha = sha256_file(gate_result_path)
    gate_path = gate_dir / "GATE.json"
    write_json(
        gate_path,
        {
            "schema": TRACKER_GATE_SCHEMA,
            "status": "complete",
            "passed": True,
            "decision": "PASS",
            "registration_sha256": registration_sha,
            "source_summary_sha256": source_summary_sha,
            "episode_ids": list(range(16, 96)),
            "manifest_sha256s": manifest_shas,
            "teacher_artifacts": [teacher_artifact],
            "pre_privileged_seal_sha256": preseal_sha,
            "result_sha256": result_sha,
            "evaluator_code_sha256": evaluator_code_sha,
        },
    )
    write_json(
        gate_dir / "COMPLETE.json",
        {
            "schema": TRACKER_GATE_COMPLETE_SCHEMA,
            "status": "complete",
            "passed": True,
            "decision": "PASS",
            "gate_sha256": sha256_file(gate_path),
            "result_sha256": result_sha,
            "pre_privileged_seal_sha256": preseal_sha,
            "evaluator_code_sha256": evaluator_code_sha,
            "registration_sha256": registration_sha,
            "source_summary_sha256": source_summary_sha,
        },
    )
    return {
        "tape": tape,
        "manifests": manifest_paths,
        "teacher_output": teacher_output,
        "registration": registration_path,
        "tracker_gate": gate_path,
        "eval_summary": source_summary,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tape", type=Path)
    ap.add_argument("--replay-manifest", type=Path, action="append")
    ap.add_argument("--teacher-output", type=Path, action="append")
    ap.add_argument(
        "--registration",
        type=Path,
        default=REPO / "plan_and_progress" / "v242_chain3_sam2_registration.json",
    )
    ap.add_argument("--tracker-gate", type=Path)
    ap.add_argument("--eval-summary", type=Path)
    ap.add_argument("--task", default="chain3_lr2")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--epochs", type=int, default=160)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--eval-batch", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--model-seed", type=int, default=242)
    ap.add_argument("--bootstrap-seed", type=int, default=24200)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--synthetic-smoke", action="store_true")
    a = ap.parse_args()
    if a.epochs < 1 or a.patience < 1 or a.batch < 1 or a.bootstrap < 20:
        ap.error("epochs/patience/batch must be positive and bootstrap >= 20")
    if a.device == "auto":
        a.device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        a.device_obj = torch.device(a.device)
    if a.device_obj.type == "cuda" and not torch.cuda.is_available():
        ap.error("CUDA requested but unavailable")
    formal_inputs = (
        a.tape,
        a.replay_manifest,
        a.teacher_output,
        a.registration,
        a.tracker_gate,
        a.eval_summary,
    )
    if not a.synthetic_smoke and not all(formal_inputs):
        ap.error(
            "--tape, one or more --replay-manifest/--teacher-output, "
            "--registration, --tracker-gate and --eval-summary are required"
        )
    return a


def main() -> int:
    args = parse_args()
    if not args.synthetic_smoke:
        run(args)
        return 0
    with tempfile.TemporaryDirectory(prefix="v242_phi_smoke_") as td:
        root = Path(td)
        fixture = make_synthetic_fixture(root)
        smoke = copy.copy(args)
        smoke.tape = fixture["tape"]
        smoke.replay_manifest = fixture["manifests"]
        smoke.teacher_output = [fixture["teacher_output"]]
        smoke.registration = fixture["registration"]
        smoke.tracker_gate = fixture["tracker_gate"]
        smoke.eval_summary = fixture["eval_summary"]
        smoke._synthetic_contract = True
        smoke.out = root / "result"
        smoke.device_obj = torch.device("cpu")
        smoke.epochs = min(args.epochs, 12)
        smoke.patience = min(args.patience, 5)
        smoke.hidden = min(args.hidden, 48)
        smoke.batch = min(args.batch, 64)
        smoke.eval_batch = min(args.eval_batch, 128)
        smoke.bootstrap = min(args.bootstrap, 100)
        out, result = run(smoke)
        ck = torch.load(out / "phi.pt", weights_only=False, map_location="cpu")
        assert ck["schema"] == PHI_SCHEMA
        assert ck["provenance"]["privileged_summary_opened_before_seal"] is False
        assert ck["provenance"]["registration_episode_ids"] == {
            "train": list(range(16, 64)),
            "calib": list(range(64, 80)),
            "test": list(range(80, 96)),
        }
        assert result["join_audit"]["joined_rows"] == 80 * 7
        assert result["visual_source_audit"]["exact_frame_bijection"] is True
        assert result["evaluation_only"]["summary_sha256"] == sha256_file(
            fixture["eval_summary"]
        )
        assert (out / "PHI_SEALED.json").is_file()
        gate_dir = fixture["tracker_gate"].parent
        for artifact_name, expected_error in (
            ("result.json", "tracker evaluator result SHA mismatch"),
            (
                "pre_privileged_seal.json",
                "tracker evaluator pre-privileged seal SHA mismatch",
            ),
        ):
            artifact_path = gate_dir / artifact_name
            original = artifact_path.read_bytes()
            artifact_path.write_bytes(original + b"\n")
            tampered = copy.copy(smoke)
            tampered.tape = root / "TAMPER_MUST_NOT_OPEN_TAPE.pt"
            tampered.out = root / f"tampered_{artifact_name}"
            try:
                run(tampered)
            except ValueError as error:
                assert expected_error in str(error)
            else:
                raise AssertionError(f"tampered {artifact_name} was accepted")
            finally:
                artifact_path.write_bytes(original)
            assert not tampered.out.exists()
        failed_gate = load_json_object(fixture["tracker_gate"])
        failed_gate["passed"] = False
        failed_gate["decision"] = "STOP"
        write_json(fixture["tracker_gate"], failed_gate)
        blocked = copy.copy(smoke)
        blocked.tape = root / "MUST_NOT_BE_OPENED.pt"
        blocked.out = root / "blocked_result"
        try:
            run(blocked)
        except ValueError as error:
            assert "STOP before tape access" in str(error)
        else:
            raise AssertionError("failed tracker gate did not stop before tape access")
        assert not blocked.out.exists()
        print(
            "SYNTHETIC SMOKE PASS: tracker PASS, multi-source visual join, "
            "registered split, full Gate hash chain, Phi seal, post-seal evaluation, "
            "tamper/fail-before-tape"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
