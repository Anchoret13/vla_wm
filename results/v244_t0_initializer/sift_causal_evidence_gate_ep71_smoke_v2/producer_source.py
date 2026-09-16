#!/usr/bin/env python3
"""RGB-only probe for alternative SAM2 frame-zero initializers.

This is deliberately independent of the frozen V243 registration and teacher.
It reads only replay/reset JSONL manifests and the JPEGs referenced by them.  It
does not open tape files, replay/source summaries, task definitions, simulator
state, events, milestones, or success labels.  Any privileged evaluation must
therefore run later, against the completed artifact written here.

Five deterministic candidates are supported:

``fixed_points``
    Four task-calibrated positive points.  This tests whether V243's broad boxes,
    rather than coordinate shift itself, caused the identity failures.

``dinov2_points``
    Local DINOv2-small patch prototypes are formed from reset RGB episodes 0--15
    at the same four calibration locations.  A constrained cosine search in the
    new frame selects each SAM2 positive point.  DINO receives no labels other
    than the fixed task-specific calibration regions and is loaded offline.

``dinov2_cluster_points``
    Uses the same local prototypes, but selects the weighted centroid of the
    largest connected component among the top-K dense-similarity patches over
    a wider development search region.

``sift_cluster_points``
    Matches tight RGB templates from development reset episode 0 against the
    full frame, then selects an inverse-distance weighted centroid from the
    largest spatially connected Lowe-ratio match cluster.

``sift_causal_reacquire``
    Before each streamed SAM2 frame, verifies current-frame SIFT matches with
    affine RANSAC and prompts only valid identities.  Invalid later-frame
    evidence falls back to SAM2 memory from past frames; no future RGB is
    loaded into the streaming session.

The default is a frame-zero segmentation probe.  ``--full-track`` propagates
the result through the RGB episode.  Only ``sift_causal_reacquire`` may add
later prompts, and it does so before each streamed frame from current RGB only.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
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
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import transformers
from PIL import Image, ImageDraw
from transformers import Sam2ImageProcessor, Sam2VideoProcessor, Sam2VideoVideoProcessor


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import probe_v242_sam2_tracking as sam2_support  # noqa: E402
import track_v243_sam2_teacher as v243_support  # noqa: E402


SCHEMA = "v244_t0_initializer_probe_v1"
LABEL_SCHEMA = "v244_t0_initializer_label_v1"
COMPLETE_SCHEMA = "v244_t0_initializer_probe_complete_v1"
TASK = "chain3_lr2"
CAMERA = "agentview"
TARGET_OBJECTS = ("alphabet_soup_1", "tomato_sauce_1", "cream_cheese_1")
ID_TO_NAME = {
    1: "basket_1",
    2: "tomato_sauce_1",
    3: "alphabet_soup_1",
    4: "cream_cheese_1",
}
PALETTE = {
    1: (255, 255, 0),
    2: (0, 255, 0),
    3: (0, 160, 255),
    4: (255, 0, 255),
}

# RGB-only development provenance: the initial centers came from reset episodes
# 0--15.  The first exploratory cream-cheese x=308 point produced whole-frame
# SAM masks on the already-open development episodes 37 and 74, so x=303 was
# selected using those failures (with support from the 0--15 reset images) and
# frozen before the nine-episode/full-80 tracking probes.  It is not an
# independent result; seed-8000/8100 panels remain reserved for that purpose.
FIXED_POINTS_XY = {
    "basket_1": (35.0, 230.0),
    "tomato_sauce_1": (158.0, 202.0),
    "alphabet_soup_1": (255.0, 202.0),
    "cream_cheese_1": (303.0, 282.0),
}

# Search regions are task-calibrated t0 image regions, not outcome information.
DINO_SEARCH_XYXY = {
    "basket_1": (0.0, 130.0, 110.0, 310.0),
    "tomato_sauce_1": (105.0, 145.0, 205.0, 250.0),
    "alphabet_soup_1": (200.0, 145.0, 300.0, 250.0),
    "cream_cheese_1": (250.0, 230.0, 355.0, 330.0),
}
# Development-v2 search intentionally covers the full tabletop, including the
# ep71 cream-cheese displacement to the lower left.  Top-K clustering was added
# after the single-patch DINO control produced edge prompts/whole-frame masks on
# development episodes 66, 74, 90, and 93.
DINO_CLUSTER_SEARCH_XYXY = {
    "basket_1": (0.0, 130.0, 110.0, 310.0),
    "tomato_sauce_1": (80.0, 140.0, 320.0, 255.0),
    "alphabet_soup_1": (80.0, 140.0, 320.0, 255.0),
    "cream_cheese_1": (80.0, 220.0, 355.0, 340.0),
}
DINO_CLUSTER_TOP_K = 16
DINO_CLUSTER_SOFTMAX_TEMPERATURE = 0.02
DINO_INPUT_SIZE = 364  # 26 x 26 patches for a 14-pixel patch model.
DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)
DEFAULT_DINO = Path(
    "/home/stargazer/.cache/huggingface/hub/"
    "models--facebook--dinov2-small/snapshots/"
    "ed25f3a31f01632728cabb09d1542f84ab7b0056"
)
DEFAULT_SAM2 = Path("/home/stargazer/.cache/sam2/checkpoints/sam2.1_hiera_large.pt")

# Tight template crops are the RGB-visible ep0 SAM masks from the already-open
# calibration artifact, with no event-time or outcome input.  SIFT clustering
# was explored after ep71 exposed the fixed-point assumption.  Every choice
# below is development-derived and requires a separate untouched-panel test.
SIFT_TEMPLATE_XYXY = {
    "basket_1": (0, 161, 81, 277),
    "tomato_sauce_1": (125, 179, 162, 233),
    "alphabet_soup_1": (228, 177, 265, 231),
    "cream_cheese_1": (282, 257, 319, 299),
}
SIFT_UPSCALE = 3
SIFT_CONTRAST_THRESHOLD = 0.01
SIFT_LOWE_RATIO = 0.70
SIFT_CLUSTER_EPS_PX = 22.0
SIFT_MIN_CLUSTER_MATCHES = 4
IDENTITY_COLLISION_MASK_IOU_GTE = 0.5
SIFT_RANSAC_REPROJECTION_THRESHOLD_PX = 3.0
SIFT_RANSAC_MAX_ITERS = 2000
SIFT_RANSAC_CONFIDENCE = 0.99
SIFT_RANSAC_REFINE_ITERS = 10
SIFT_RANSAC_MIN_INLIERS = 4
SIFT_RANSAC_MIN_INLIER_RATIO = 0.5
SIFT_RANSAC_SCALE_RANGE = (0.5, 2.0)
TEMPORAL_TARGET_PROMPT_COLLISION_DISTANCE_PX = 4.0
TEMPORAL_TARGET_MASK_COLLISION_IOU_GTE = 0.5
SIFT_TEMPLATE_DERIVATION = {
    "artifact": "results/v242_sam2_probe/chain3_ep0_agentview_dense_calibration_v6/result.json",
    "artifact_sha256": "cf0a8c4a0a6bd376e7d16a4ae52478dcb8e962b908162b0e6db967be65d110f7",
    "rule": "exact t0 RGB-visible SAM mask bounding boxes from development episode 0",
    "data_boundary": "RGB-only calibration; no event, success, simulator-state, or task-definition input",
}

# All adaptive choices above belong to the opened seed-8700 development panel.
# In particular, the wide locator requirement came from ep71; DINO clustering
# responded to failures in eps66/74/90/93; and the tight SIFT alternative was
# checked on eps28/37/40/63/66/71/74/87/90/93 before this full-development
# probe.  Seed-8000/8100 and chain2b remain outside this script's accepted seed
# rule and are reserved for independent evaluation.
DEVELOPMENT_LINEAGE = {
    "panel": "chain3_lr2 seed rule 8700 + episode_id",
    "adaptive_episode_ids": [0, 28, 37, 40, 63, 66, 71, 74, 87, 90, 93],
    "independent_panels_opened": False,
}

THRESHOLDS = {
    "object_score_logit_lte": 0.0,
    "target_mask_area_px_lt": 64,
    "target_mask_area_px_gt": 5000,
    "basket_mask_area_px_lt": 3000,
    "basket_mask_area_px_gt": 15000,
    "nonfinite_output": True,
}
FORBIDDEN_KEYS = {
    "success",
    "successes",
    "is_success",
    "event",
    "events",
    "milestone",
    "milestones",
    "bddl",
    "stage_reached",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def reject_semantic_fields(value: Any, where: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            require(
                normalized not in FORBIDDEN_KEYS,
                f"{where}.{key}: privileged semantic field is forbidden",
            )
            reject_semantic_fields(child, f"{where}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_semantic_fields(child, f"{where}[{index}]")


def git_head() -> str | None:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def load_manifest_rows(
    paths: Sequence[Path], requested: set[int], *, reset_only: bool = False
) -> tuple[dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
    by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    provenance: list[dict[str, Any]] = []
    seen_frames: set[str] = set()
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        require(path.is_file(), f"missing manifest: {path}")
        manifest_sha = sha256_file(path)
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                require(bool(line.strip()), f"{path}:{line_number}: blank row")
                row = json.loads(line)
                require(isinstance(row, dict), f"{path}:{line_number}: row is not object")
                reject_semantic_fields(row, f"{path}:{line_number}")
                schema = row.get("schema")
                allowed = {"v242_reset_rgb_v1"} if reset_only else {"v242_rgb_replay_v1"}
                require(schema in allowed, f"{path}:{line_number}: schema {schema!r}")
                episode_id = int(row["episode_id"])
                if episode_id not in requested:
                    continue
                require(row.get("task") == TASK, f"{path}:{line_number}: task mismatch")
                require(int(row["env_seed"]) == 8700 + episode_id, "env_seed rule mismatch")
                t_value = int(row["t"])
                if reset_only:
                    require(t_value == 0, "reset calibration manifest must contain t0 only")
                image_record = row["images"][CAMERA]
                image_path = (path.parent / str(image_record["path"])).resolve()
                require(image_path.is_relative_to(path.parent), "image escapes manifest dir")
                require(image_path.is_file(), f"missing JPEG: {image_path}")
                image_sha = sha256_file(image_path)
                require(image_sha == image_record["sha256"], "JPEG SHA mismatch")
                with Image.open(image_path) as opened:
                    require(opened.size == (360, 360), "expected 360x360 RGB")
                    opened.verify()
                frame_id = str(row["frame_id"])
                require(frame_id not in seen_frames, f"duplicate frame {frame_id}")
                seen_frames.add(frame_id)
                by_episode[episode_id].append(
                    {
                        "frame_id": frame_id,
                        "episode_id": episode_id,
                        "env_seed": int(row["env_seed"]),
                        "t": t_value,
                        "image_path": image_path,
                        "image_sha256": image_sha,
                        "manifest_path": path,
                        "manifest_sha256": manifest_sha,
                    }
                )
                count += 1
        provenance.append(
            {"path": str(path), "sha256": manifest_sha, "selected_rows": count}
        )
    require(set(by_episode) == requested, f"found episodes {set(by_episode)} != {requested}")
    for episode_id, rows in by_episode.items():
        rows.sort(key=lambda row: row["t"])
        require(rows[0]["t"] == 0, f"episode {episode_id} does not start at t0")
        require(
            len({row["t"] for row in rows}) == len(rows),
            f"episode {episode_id} has duplicate times",
        )
        if not reset_only:
            require(
                [row["t"] for row in rows] == [10 * i for i in range(len(rows))],
                f"episode {episode_id} is not a dense ten-step prefix",
            )
    return dict(by_episode), provenance


def dino_preprocess(images: Sequence[Image.Image]) -> torch.Tensor:
    mean = torch.tensor(DINO_MEAN, dtype=torch.float32)[:, None, None]
    std = torch.tensor(DINO_STD, dtype=torch.float32)[:, None, None]
    tensors = []
    for image in images:
        resized = image.resize(
            (DINO_INPUT_SIZE, DINO_INPUT_SIZE), Image.Resampling.BICUBIC
        )
        array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        tensors.append((tensor - mean) / std)
    return torch.stack(tensors)


def dino_features(model: Any, images: Sequence[Image.Image]) -> torch.Tensor:
    with torch.inference_mode():
        hidden = model(pixel_values=dino_preprocess(images)).last_hidden_state[:, 1:]
    patch_count = int(hidden.shape[1])
    grid = int(round(math.sqrt(patch_count)))
    require(grid * grid == patch_count, f"DINO patch count {patch_count} not square")
    return torch.nn.functional.normalize(
        hidden.reshape(len(images), grid, grid, hidden.shape[-1]), dim=-1
    )


def pixel_to_patch(value: float, grid: int) -> int:
    return max(0, min(grid - 1, int(value / 360.0 * grid)))


def connected_topk_centroid(
    similarity: torch.Tensor,
) -> tuple[float, float, dict[str, Any]]:
    """Largest 8-connected component among the deterministic top-K patches."""
    flat = similarity.flatten()
    values, indices = torch.topk(flat, min(DINO_CLUSTER_TOP_K, flat.numel()))
    width = int(similarity.shape[1])
    coordinates = [(int(index) // width, int(index) % width) for index in indices]
    seen: set[int] = set()
    groups: list[list[int]] = []
    for start in range(len(coordinates)):
        if start in seen:
            continue
        seen.add(start)
        queue = [start]
        group: list[int] = []
        while queue:
            current = queue.pop()
            group.append(current)
            cy, cx = coordinates[current]
            for candidate, (yy, xx) in enumerate(coordinates):
                if candidate not in seen and max(abs(cy - yy), abs(cx - xx)) <= 1:
                    seen.add(candidate)
                    queue.append(candidate)
        groups.append(group)
    require(bool(groups), "DINO top-K clustering returned no component")
    groups.sort(
        key=lambda group: (
            len(group),
            float(values[group].mean()),
            -min(group),
        ),
        reverse=True,
    )
    selected = groups[0]
    selected_values = values[selected]
    weights = torch.softmax(
        (selected_values - selected_values.max()) / DINO_CLUSTER_SOFTMAX_TEMPERATURE,
        dim=0,
    )
    y_patch = sum(
        coordinates[index][0] * float(weights[position])
        for position, index in enumerate(selected)
    )
    x_patch = sum(
        coordinates[index][1] * float(weights[position])
        for position, index in enumerate(selected)
    )
    diagnostics = {
        "top_k": int(len(values)),
        "connected_components": len(groups),
        "selected_component_patches": len(selected),
        "selected_component_mean_cosine": float(selected_values.mean()),
        "selected_component_max_cosine": float(selected_values.max()),
    }
    return x_patch, y_patch, diagnostics


def make_dino_points(
    dino_root: Path,
    calibration: Mapping[int, list[dict[str, Any]]],
    episodes: Mapping[int, list[dict[str, Any]]],
    *,
    clustered: bool,
) -> tuple[dict[int, dict[str, tuple[float, float]]], dict[int, dict[str, Any]], dict[str, Any]]:
    from transformers import AutoModel

    require((dino_root / "config.json").is_file(), "DINO config missing")
    require((dino_root / "model.safetensors").is_file(), "DINO weights missing")
    calibration_ids = sorted(calibration)
    episode_ids = sorted(episodes)
    calibration_images = [
        Image.open(calibration[episode_id][0]["image_path"]).convert("RGB")
        for episode_id in calibration_ids
    ]
    episode_images = [
        Image.open(episodes[episode_id][0]["image_path"]).convert("RGB")
        for episode_id in episode_ids
    ]
    model = AutoModel.from_pretrained(dino_root, local_files_only=True).eval()
    model.requires_grad_(False)
    features = dino_features(model, [*calibration_images, *episode_images])
    grid = features.shape[1]
    prototypes: dict[str, torch.Tensor] = {}
    for name, (x_value, y_value) in FIXED_POINTS_XY.items():
        x_patch = pixel_to_patch(x_value, grid)
        y_patch = pixel_to_patch(y_value, grid)
        crop = features[
            : len(calibration_ids),
            max(0, y_patch - 1) : min(grid, y_patch + 2),
            max(0, x_patch - 1) : min(grid, x_patch + 2),
        ]
        prototypes[name] = torch.nn.functional.normalize(
            crop.reshape(-1, crop.shape[-1]).mean(0), dim=0
        )

    points: dict[int, dict[str, tuple[float, float]]] = {}
    scores: dict[int, dict[str, Any]] = {}
    for feature_index, episode_id in enumerate(episode_ids, start=len(calibration_ids)):
        points[episode_id] = {}
        scores[episode_id] = {}
        for name, prototype in prototypes.items():
            search_regions = DINO_CLUSTER_SEARCH_XYXY if clustered else DINO_SEARCH_XYXY
            x0, y0, x1, y1 = search_regions[name]
            xa, xb = pixel_to_patch(x0, grid), pixel_to_patch(x1, grid) + 1
            ya, yb = pixel_to_patch(y0, grid), pixel_to_patch(y1, grid) + 1
            similarity = (features[feature_index] @ prototype)[ya:yb, xa:xb]
            if clustered:
                local_x, local_y, cluster_diagnostics = connected_topk_centroid(
                    similarity
                )
                gx, gy = xa + local_x, ya + local_y
                score_record: Any = cluster_diagnostics
            else:
                flat_index = int(similarity.argmax())
                dy, dx = divmod(flat_index, similarity.shape[1])
                gy, gx = ya + dy, xa + dx
                score_record = {"max_patch_cosine": float(similarity.flatten()[flat_index])}
            points[episode_id][name] = (
                (gx + 0.5) * 360.0 / grid,
                (gy + 0.5) * 360.0 / grid,
            )
            scores[episode_id][name] = score_record
    provenance = {
        "path": str(dino_root),
        "config_sha256": sha256_file(dino_root / "config.json"),
        "weights_sha256": sha256_file(dino_root / "model.safetensors"),
        "input_size": DINO_INPUT_SIZE,
        "normalization_mean": DINO_MEAN,
        "normalization_std": DINO_STD,
        "prototype_calibration_episode_ids": calibration_ids,
        "prototype_rule": "mean normalized patch embedding in 3x3 neighborhoods around frozen calibration points",
        "search_regions_xyxy": (
            DINO_CLUSTER_SEARCH_XYXY if clustered else DINO_SEARCH_XYXY
        ),
        "location_rule": (
            "softmax-weighted centroid of largest 8-connected component among top-16 patches"
            if clustered
            else "single maximum-similarity patch center"
        ),
        "cluster_top_k": DINO_CLUSTER_TOP_K if clustered else None,
        "cluster_softmax_temperature": (
            DINO_CLUSTER_SOFTMAX_TEMPERATURE if clustered else None
        ),
        "grid_size": int(grid),
        "offline_local_files_only": True,
    }
    del model, features
    return points, scores, provenance


def sift_connected_cluster(
    points_xy: np.ndarray, distances: np.ndarray
) -> tuple[np.ndarray, list[int], list[list[int]]]:
    require(len(points_xy) == len(distances) and len(points_xy) > 0, "empty SIFT matches")
    seen: set[int] = set()
    groups: list[list[int]] = []
    for start in range(len(points_xy)):
        if start in seen:
            continue
        seen.add(start)
        queue = [start]
        group: list[int] = []
        while queue:
            current = queue.pop()
            group.append(current)
            delta = np.linalg.norm(points_xy - points_xy[current], axis=1)
            for candidate in np.flatnonzero(delta <= SIFT_CLUSTER_EPS_PX):
                index = int(candidate)
                if index not in seen:
                    seen.add(index)
                    queue.append(index)
        groups.append(group)
    groups.sort(
        key=lambda group: (
            len(group),
            -float(np.mean(distances[group])),
            -min(group),
        ),
        reverse=True,
    )
    selected = groups[0]
    require(
        len(selected) >= SIFT_MIN_CLUSTER_MATCHES,
        f"largest SIFT cluster has only {len(selected)} matches",
    )
    weights = 1.0 / (distances[selected] + 1e-6)
    centroid = (points_xy[selected] * weights[:, None]).sum(0) / weights.sum()
    return centroid, selected, groups


def make_sift_points(
    calibration: Mapping[int, list[dict[str, Any]]],
    episodes: Mapping[int, list[dict[str, Any]]],
) -> tuple[dict[int, dict[str, tuple[float, float]]], dict[int, dict[str, Any]], dict[str, Any]]:
    import cv2

    require(0 in calibration, "SIFT calibration requires development episode 0")
    calibration_row = calibration[0][0]
    base_rgb = np.asarray(Image.open(calibration_row["image_path"]).convert("RGB"))
    base_bgr = base_rgb[:, :, ::-1]
    sift = cv2.SIFT_create(contrastThreshold=SIFT_CONTRAST_THRESHOLD)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    templates: dict[str, tuple[Any, np.ndarray]] = {}
    template_diagnostics: dict[str, Any] = {}
    for name, (x0, y0, x1, y1) in SIFT_TEMPLATE_XYXY.items():
        crop = base_bgr[y0:y1, x0:x1]
        enlarged = cv2.resize(
            crop,
            None,
            fx=SIFT_UPSCALE,
            fy=SIFT_UPSCALE,
            interpolation=cv2.INTER_CUBIC,
        )
        keypoints, descriptors = sift.detectAndCompute(enlarged, None)
        require(descriptors is not None and len(keypoints) >= 2, f"no SIFT template for {name}")
        templates[name] = (keypoints, descriptors)
        template_diagnostics[name] = {
            "keypoints": len(keypoints),
            "crop_pixel_sha256": hashlib.sha256(crop.tobytes()).hexdigest(),
        }

    points: dict[int, dict[str, tuple[float, float]]] = {}
    diagnostics: dict[int, dict[str, Any]] = {}
    for episode_id in sorted(episodes):
        rgb = np.asarray(Image.open(episodes[episode_id][0]["image_path"]).convert("RGB"))
        enlarged = cv2.resize(
            rgb[:, :, ::-1],
            None,
            fx=SIFT_UPSCALE,
            fy=SIFT_UPSCALE,
            interpolation=cv2.INTER_CUBIC,
        )
        test_keypoints, test_descriptors = sift.detectAndCompute(enlarged, None)
        require(test_descriptors is not None, f"ep{episode_id}: no SIFT descriptors")
        points[episode_id] = {}
        diagnostics[episode_id] = {}
        for name, (_, template_descriptors) in templates.items():
            pairs = matcher.knnMatch(template_descriptors, test_descriptors, k=2)
            ratio_matches = [
                pair[0]
                for pair in pairs
                if len(pair) == 2
                and pair[0].distance < SIFT_LOWE_RATIO * pair[1].distance
            ]
            # Template descriptors are not independent if several of them map
            # to one scene keypoint.  Keep only the lowest-distance match for
            # each target keypoint before applying the cluster support floor.
            best_by_train_idx: dict[int, Any] = {}
            for match in ratio_matches:
                previous = best_by_train_idx.get(int(match.trainIdx))
                if previous is None or (
                    float(match.distance), int(match.queryIdx)
                ) < (float(previous.distance), int(previous.queryIdx)):
                    best_by_train_idx[int(match.trainIdx)] = match
            good = sorted(
                best_by_train_idx.values(),
                key=lambda match: (
                    float(match.distance),
                    int(match.trainIdx),
                    int(match.queryIdx),
                ),
            )
            require(bool(good), f"ep{episode_id} {name}: no ratio-test matches")
            match_points = np.asarray(
                [test_keypoints[match.trainIdx].pt for match in good], dtype=np.float64
            ) / SIFT_UPSCALE
            distances = np.asarray([match.distance for match in good], dtype=np.float64)
            centroid, selected, groups = sift_connected_cluster(match_points, distances)
            require(
                bool(np.isfinite(centroid).all())
                and 0 <= centroid[0] < 360
                and 0 <= centroid[1] < 360,
                f"ep{episode_id} {name}: invalid centroid",
            )
            points[episode_id][name] = (float(centroid[0]), float(centroid[1]))
            diagnostics[episode_id][name] = {
                "ratio_test_matches_before_target_dedup": len(ratio_matches),
                "unique_target_ratio_test_matches": len(good),
                "connected_components": len(groups),
                "selected_cluster_matches": len(selected),
                "selected_cluster_mean_descriptor_distance": float(
                    distances[selected].mean()
                ),
                "selected_cluster_bbox_xyxy": [
                    float(match_points[selected, 0].min()),
                    float(match_points[selected, 1].min()),
                    float(match_points[selected, 0].max()),
                    float(match_points[selected, 1].max()),
                ],
            }
    provenance = {
        "family": "OpenCV SIFT + Lowe ratio + largest spatial connected cluster",
        "opencv_version": cv2.__version__,
        "offline": True,
        "calibration_episode_id": 0,
        "calibration_image_sha256": calibration_row["image_sha256"],
        "template_xyxy": SIFT_TEMPLATE_XYXY,
        "template_diagnostics": template_diagnostics,
        "upscale": SIFT_UPSCALE,
        "contrast_threshold": SIFT_CONTRAST_THRESHOLD,
        "lowe_ratio": SIFT_LOWE_RATIO,
        "cluster_eps_px": SIFT_CLUSTER_EPS_PX,
        "min_cluster_matches": SIFT_MIN_CLUSTER_MATCHES,
        "target_keypoint_deduplication": "lowest descriptor distance per trainIdx",
        "location_rule": "inverse-descriptor-distance weighted centroid of largest connected cluster",
        "template_derivation": SIFT_TEMPLATE_DERIVATION,
        "development_lineage": DEVELOPMENT_LINEAGE,
    }
    return points, diagnostics, provenance


class CausalSiftAffineLocator:
    """Current-frame-only SIFT evidence with affine/RANSAC verification.

    Later-frame failures are soft: an invalid object simply receives no new
    prompt and SAM2 can propagate from past memory.  Frame zero remains strict.
    """

    def __init__(self, calibration: Mapping[int, list[dict[str, Any]]]) -> None:
        import cv2

        require(0 in calibration, "causal SIFT calibration requires episode 0")
        self.cv2 = cv2
        self.sift = cv2.SIFT_create(contrastThreshold=SIFT_CONTRAST_THRESHOLD)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.calibration_row = calibration[0][0]
        base_rgb = np.asarray(
            Image.open(self.calibration_row["image_path"]).convert("RGB")
        )
        base_bgr = base_rgb[:, :, ::-1]
        self.templates: dict[str, dict[str, Any]] = {}
        template_diagnostics: dict[str, Any] = {}
        for name, (x0, y0, x1, y1) in SIFT_TEMPLATE_XYXY.items():
            crop = base_bgr[y0:y1, x0:x1]
            enlarged = cv2.resize(
                crop,
                None,
                fx=SIFT_UPSCALE,
                fy=SIFT_UPSCALE,
                interpolation=cv2.INTER_CUBIC,
            )
            keypoints, descriptors = self.sift.detectAndCompute(enlarged, None)
            require(
                descriptors is not None and len(keypoints) >= 2,
                f"no causal SIFT template for {name}",
            )
            self.templates[name] = {
                "keypoints": keypoints,
                "descriptors": descriptors,
                "center_xy": ((x1 - x0) / 2.0, (y1 - y0) / 2.0),
            }
            template_diagnostics[name] = {
                "keypoints": len(keypoints),
                "crop_pixel_sha256": hashlib.sha256(crop.tobytes()).hexdigest(),
            }
        self.provenance = {
            "family": "OpenCV SIFT cluster + affine-partial-2D RANSAC causal re-acquisition",
            "opencv_version": cv2.__version__,
            "offline": True,
            "causal_rule": "each frame uses only that frame RGB plus the fixed episode-0 template",
            "later_locator_failure": "soft: omit that object prompt and use past-only SAM2 propagation",
            "calibration_episode_id": 0,
            "calibration_image_sha256": self.calibration_row["image_sha256"],
            "template_xyxy": SIFT_TEMPLATE_XYXY,
            "template_diagnostics": template_diagnostics,
            "template_derivation": SIFT_TEMPLATE_DERIVATION,
            "development_lineage": DEVELOPMENT_LINEAGE,
            "upscale": SIFT_UPSCALE,
            "contrast_threshold": SIFT_CONTRAST_THRESHOLD,
            "lowe_ratio": SIFT_LOWE_RATIO,
            "cluster_eps_px": SIFT_CLUSTER_EPS_PX,
            "cluster_min_unique_target_matches": SIFT_MIN_CLUSTER_MATCHES,
            "target_keypoint_deduplication": "lowest descriptor distance per trainIdx",
            "ransac": {
                "model": "cv2.estimateAffinePartial2D",
                "reprojection_threshold_px": SIFT_RANSAC_REPROJECTION_THRESHOLD_PX,
                "max_iters": SIFT_RANSAC_MAX_ITERS,
                "confidence": SIFT_RANSAC_CONFIDENCE,
                "refine_iters": SIFT_RANSAC_REFINE_ITERS,
                "min_inliers": SIFT_RANSAC_MIN_INLIERS,
                "min_inlier_ratio": SIFT_RANSAC_MIN_INLIER_RATIO,
                "scale_range_inclusive": SIFT_RANSAC_SCALE_RANGE,
                "rng_seed_before_each_fit": 0,
            },
            "prompt_rule": "inverse-distance weighted centroid of RANSAC-inlier scene keypoints",
            "relation_evidence_rule": (
                "use affine-projected template center only when it lies in the current prompted target mask; "
                "otherwise retain the current SAM target-mask centroid"
            ),
        }

    @staticmethod
    def _connected_groups(points_xy: np.ndarray) -> list[list[int]]:
        seen: set[int] = set()
        groups: list[list[int]] = []
        for start in range(len(points_xy)):
            if start in seen:
                continue
            seen.add(start)
            queue = [start]
            group: list[int] = []
            while queue:
                current = queue.pop()
                group.append(current)
                delta = np.linalg.norm(points_xy - points_xy[current], axis=1)
                for candidate in np.flatnonzero(delta <= SIFT_CLUSTER_EPS_PX):
                    index = int(candidate)
                    if index not in seen:
                        seen.add(index)
                        queue.append(index)
            groups.append(group)
        return groups

    def locate(
        self, image: Image.Image
    ) -> tuple[dict[str, tuple[float, float]], dict[str, dict[str, Any]]]:
        cv2 = self.cv2
        rgb = np.asarray(image.convert("RGB"))
        enlarged = cv2.resize(
            rgb[:, :, ::-1],
            None,
            fx=SIFT_UPSCALE,
            fy=SIFT_UPSCALE,
            interpolation=cv2.INTER_CUBIC,
        )
        scene_keypoints, scene_descriptors = self.sift.detectAndCompute(
            enlarged, None
        )
        points: dict[str, tuple[float, float]] = {}
        diagnostics: dict[str, dict[str, Any]] = {}
        for name, template in self.templates.items():
            reasons: list[str] = []
            ratio_matches: list[Any] = []
            unique_matches: list[Any] = []
            groups: list[list[int]] = []
            selected: list[int] = []
            inlier_indices: list[int] = []
            affine = None
            scale = None
            projected_center = None
            prompt_point = None
            if scene_descriptors is None:
                reasons.append("no_scene_descriptors")
            else:
                pairs = self.matcher.knnMatch(
                    template["descriptors"], scene_descriptors, k=2
                )
                ratio_matches = [
                    pair[0]
                    for pair in pairs
                    if len(pair) == 2
                    and pair[0].distance < SIFT_LOWE_RATIO * pair[1].distance
                ]
                best_by_train_idx: dict[int, Any] = {}
                for match in ratio_matches:
                    previous = best_by_train_idx.get(int(match.trainIdx))
                    if previous is None or (
                        float(match.distance), int(match.queryIdx)
                    ) < (float(previous.distance), int(previous.queryIdx)):
                        best_by_train_idx[int(match.trainIdx)] = match
                unique_matches = sorted(
                    best_by_train_idx.values(),
                    key=lambda match: (
                        float(match.distance),
                        int(match.trainIdx),
                        int(match.queryIdx),
                    ),
                )
                scene_points = (
                    np.asarray(
                        [scene_keypoints[match.trainIdx].pt for match in unique_matches],
                        dtype=np.float64,
                    )
                    / SIFT_UPSCALE
                    if unique_matches
                    else np.empty((0, 2), dtype=np.float64)
                )
                groups = self._connected_groups(scene_points)
                groups.sort(
                    key=lambda group: (
                        len(group),
                        -float(
                            np.mean(
                                [unique_matches[index].distance for index in group]
                            )
                        ),
                        -min(group),
                    ),
                    reverse=True,
                )
                selected = groups[0] if groups else []
                if len(selected) < SIFT_MIN_CLUSTER_MATCHES:
                    reasons.append("cluster_support_lt_min")
                else:
                    template_points = np.asarray(
                        [
                            template["keypoints"][unique_matches[index].queryIdx].pt
                            for index in selected
                        ],
                        dtype=np.float32,
                    ) / SIFT_UPSCALE
                    selected_scene_points = scene_points[selected].astype(np.float32)
                    cv2.setRNGSeed(0)
                    affine, inlier_mask = cv2.estimateAffinePartial2D(
                        template_points,
                        selected_scene_points,
                        method=cv2.RANSAC,
                        ransacReprojThreshold=SIFT_RANSAC_REPROJECTION_THRESHOLD_PX,
                        maxIters=SIFT_RANSAC_MAX_ITERS,
                        confidence=SIFT_RANSAC_CONFIDENCE,
                        refineIters=SIFT_RANSAC_REFINE_ITERS,
                    )
                    if affine is None or inlier_mask is None:
                        reasons.append("affine_fit_failed")
                    else:
                        inlier_indices = [
                            selected[index]
                            for index, flag in enumerate(inlier_mask.ravel())
                            if bool(flag)
                        ]
                        inlier_ratio = len(inlier_indices) / len(selected)
                        if len(inlier_indices) < SIFT_RANSAC_MIN_INLIERS:
                            reasons.append("ransac_inliers_lt_min")
                        if inlier_ratio < SIFT_RANSAC_MIN_INLIER_RATIO:
                            reasons.append("ransac_inlier_ratio_lt_min")
                        scale = float(math.hypot(affine[0, 0], affine[1, 0]))
                        if not (
                            SIFT_RANSAC_SCALE_RANGE[0]
                            <= scale
                            <= SIFT_RANSAC_SCALE_RANGE[1]
                        ):
                            reasons.append("affine_scale_out_of_range")
                        center = np.asarray(
                            [*template["center_xy"], 1.0], dtype=np.float64
                        )
                        projected = affine @ center
                        projected_center = (float(projected[0]), float(projected[1]))
                        if not (
                            np.isfinite(projected).all()
                            and 0 <= projected[0] < 360
                            and 0 <= projected[1] < 360
                        ):
                            reasons.append("projected_center_out_of_frame")
                        if inlier_indices:
                            distances = np.asarray(
                                [
                                    unique_matches[index].distance
                                    for index in inlier_indices
                                ],
                                dtype=np.float64,
                            )
                            weights = 1.0 / (distances + 1e-6)
                            inlier_scene_points = scene_points[inlier_indices]
                            prompt = (
                                inlier_scene_points * weights[:, None]
                            ).sum(0) / weights.sum()
                            prompt_point = (float(prompt[0]), float(prompt[1]))
                            if not (
                                np.isfinite(prompt).all()
                                and 0 <= prompt[0] < 360
                                and 0 <= prompt[1] < 360
                            ):
                                reasons.append("prompt_point_out_of_frame")
            valid = not reasons
            if valid:
                require(prompt_point is not None, "valid SIFT evidence has no prompt")
                points[name] = prompt_point
            diagnostics[name] = {
                "valid": valid,
                "invalid_reasons": sorted(set(reasons)),
                "ratio_test_matches_before_target_dedup": len(ratio_matches),
                "unique_target_ratio_test_matches": len(unique_matches),
                "connected_components": len(groups),
                "selected_cluster_matches": len(selected),
                "ransac_inliers": len(inlier_indices),
                "ransac_inlier_ratio": (
                    len(inlier_indices) / len(selected) if selected else None
                ),
                "affine_scale": scale,
                "affine_matrix": affine.tolist() if affine is not None else None,
                "projected_template_center_xy": (
                    list(projected_center) if projected_center is not None else None
                ),
                "prompt_point_xy": list(prompt_point) if prompt_point is not None else None,
            }
        return points, diagnostics


def normalize_masks(processed: torch.Tensor, count: int) -> torch.Tensor:
    while processed.ndim > 3 and processed.shape[0] == 1:
        processed = processed[0]
    if processed.ndim == 4 and processed.shape[1] == 1:
        processed = processed[:, 0]
    require(
        processed.ndim == 3 and processed.shape[0] == count,
        f"unexpected mask shape {tuple(processed.shape)}",
    )
    return processed


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 0.0


def render_t0(
    image: Image.Image,
    masks: Mapping[int, np.ndarray],
    diagnostics: Mapping[str, Mapping[str, Any]],
    points: Mapping[str, tuple[float, float]],
) -> Image.Image:
    pixels = np.asarray(image).astype(np.float32)
    for obj_id, mask in masks.items():
        color = np.asarray(PALETTE[obj_id], dtype=np.float32)
        pixels[mask] = 0.55 * pixels[mask] + 0.45 * color
    output = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(output)
    for obj_id, name in ID_TO_NAME.items():
        x, y = points[name]
        color = PALETTE[obj_id]
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=color, width=2)
        bbox = diagnostics[name]["mask_bbox_xyxy"]
        if bbox is not None:
            draw.rectangle(bbox, outline=color, width=2)
        draw.text((x + 5, y - 8), name, fill=color)
    return output


def classify_output(processor, output) -> tuple[dict[str, Any], dict[int, np.ndarray]]:
    processed = processor.post_process_masks(
        [output.pred_masks], original_sizes=[[360, 360]], binarize=True
    )[0]
    processed = normalize_masks(processed, len(output.object_ids))
    return v243_support.classify_frame(output, processed, ID_TO_NAME, THRESHOLDS)


def track_episode(
    model: Any,
    processor: Any,
    rows: Sequence[dict[str, Any]],
    points: Mapping[str, tuple[float, float]],
    point_scores: Mapping[str, Any] | None,
    *,
    dtype: torch.dtype,
    full_track: bool,
    causal_locator: CausalSiftAffineLocator | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], Image.Image]:
    frames = [Image.open(row["image_path"]).convert("RGB") for row in rows]
    require(
        causal_locator is None or full_track,
        "causal re-acquisition requires --full-track",
    )
    session_frames = None if causal_locator else (frames if full_track else frames[:1])
    session = processor.init_video_session(
        video=session_frames,
        inference_device=model.device,
        inference_state_device=model.device,
        processing_device=model.device,
        video_storage_device="cpu",
        dtype=dtype,
    )
    object_ids = sorted(ID_TO_NAME)
    name_to_id = {name: obj_id for obj_id, name in ID_TO_NAME.items()}
    if causal_locator is None:
        input_points = [[[list(points[ID_TO_NAME[obj_id]])] for obj_id in object_ids]]
        input_labels = [[[1] for _ in object_ids]]
        processor.add_inputs_to_inference_session(
            inference_session=session,
            frame_idx=0,
            obj_ids=list(object_ids),
            input_points=input_points,
            input_labels=input_labels,
        )
    output_rows: list[dict[str, Any]] = []
    t0_overlay = None
    t0_mask_ious: dict[str, float] | None = None
    temporal_prompt_applications = 0
    temporal_locator_valid = {name: 0 for name in ID_TO_NAME.values()}
    temporal_locator_invalid = {name: 0 for name in ID_TO_NAME.values()}
    temporal_preprompt_collisions = 0
    temporal_postmask_collisions = 0
    temporal_relation_overrides = 0
    temporal_evidence_state = {name: "outside" for name in TARGET_OBJECTS}
    temporal_evidence_confirmed_flips = {name: 0 for name in TARGET_OBJECTS}
    temporal_evidence_blocked_flips = {name: 0 for name in TARGET_OBJECTS}

    def causal_outputs() -> Iterable[tuple[Any, dict[str, tuple[float, float]], dict[str, Any], int]]:
        nonlocal temporal_prompt_applications, temporal_preprompt_collisions
        for frame_index, frame in enumerate(frames):
            located_points, locator_diagnostics = causal_locator.locate(frame)
            for name in ID_TO_NAME.values():
                if name in located_points:
                    temporal_locator_valid[name] += 1
                else:
                    temporal_locator_invalid[name] += 1

            # Independently optimized target templates may still pick one
            # physical patch.  Reject both later-frame prompts when their
            # locations are essentially identical; frame zero fails closed.
            rejected: set[str] = set()
            for left_index, left_name in enumerate(TARGET_OBJECTS):
                for right_name in TARGET_OBJECTS[left_index + 1 :]:
                    if left_name not in located_points or right_name not in located_points:
                        continue
                    distance = math.dist(
                        located_points[left_name], located_points[right_name]
                    )
                    if distance <= TEMPORAL_TARGET_PROMPT_COLLISION_DISTANCE_PX:
                        rejected.update((left_name, right_name))
                        temporal_preprompt_collisions += 1
                        for name, other in (
                            (left_name, right_name),
                            (right_name, left_name),
                        ):
                            locator_diagnostics[name]["valid"] = False
                            locator_diagnostics[name]["invalid_reasons"] = sorted(
                                set(
                                    locator_diagnostics[name]["invalid_reasons"]
                                    + [f"target_prompt_collision_with:{other}"]
                                )
                            )
            for name in rejected:
                located_points.pop(name, None)
            if frame_index == 0:
                require(
                    set(located_points) == set(ID_TO_NAME.values()),
                    "causal locator frame zero did not initialize all identities",
                )
                require(
                    all(
                        math.dist(located_points[name], points[name]) <= 1e-6
                        for name in ID_TO_NAME.values()
                    ),
                    "causal frame-zero locator differs from preflight",
                )
            prompt_ids = [
                obj_id
                for obj_id in object_ids
                if ID_TO_NAME[obj_id] in located_points
            ]
            if prompt_ids:
                processor.add_inputs_to_inference_session(
                    inference_session=session,
                    frame_idx=frame_index,
                    obj_ids=prompt_ids,
                    input_points=[
                        [
                            [list(located_points[ID_TO_NAME[obj_id]])]
                            for obj_id in prompt_ids
                        ]
                    ],
                    input_labels=[[[1] for _ in prompt_ids]],
                    original_size=(360, 360),
                )
                if frame_index > 0:
                    temporal_prompt_applications += len(prompt_ids)
            processed_frame = processor.video_processor(
                videos=[frame], device=model.device, return_tensors="pt"
            ).pixel_values_videos[0, 0]
            output = model(
                inference_session=session,
                frame=processed_frame,
                frame_idx=frame_index,
            )
            yield output, located_points, locator_diagnostics, temporal_prompt_applications

    with torch.inference_mode():
        if causal_locator is not None:
            outputs = causal_outputs()
        else:
            if full_track:
                model(inference_session=session, frame_idx=0)
                raw_outputs = model.propagate_in_video_iterator(
                    session, show_progress_bar=False
                )
            else:
                raw_outputs = [model(inference_session=session, frame_idx=0)]
            outputs = (
                (output, dict(points), dict(point_scores or {}), 0)
                for output in raw_outputs
            )
        for output, current_points, current_locator_diagnostics, prompt_count in outputs:
            frame_index = int(output.frame_idx)
            diagnostics, masks = classify_output(processor, output)
            source = rows[frame_index]
            current_target_mask_ious: dict[str, float] = {}
            for left_index, left_name in enumerate(TARGET_OBJECTS):
                for right_name in TARGET_OBJECTS[left_index + 1 :]:
                    pair = f"{left_name}::{right_name}"
                    value = mask_iou(
                        masks[name_to_id[left_name]], masks[name_to_id[right_name]]
                    )
                    current_target_mask_ious[pair] = value
                    if (
                        causal_locator is not None
                        and value >= TEMPORAL_TARGET_MASK_COLLISION_IOU_GTE
                    ):
                        temporal_postmask_collisions += 1
                        for name, other in (
                            (left_name, right_name),
                            (right_name, left_name),
                        ):
                            reasons = diagnostics[name]["uncertain_reasons"] + [
                                f"target_mask_collision_with:{other}"
                            ]
                            diagnostics[name]["uncertain_reasons"] = sorted(set(reasons))
                            diagnostics[name]["label"] = "uncertain"

            # A verified appearance-space object center is a better relation
            # point than a clipped mask centroid, but is used only if the
            # current prompted SAM mask actually contains it.
            if causal_locator is not None:
                basket_bbox = diagnostics["basket_1"]["mask_bbox_xyxy"]
                for name in TARGET_OBJECTS:
                    evidence = current_locator_diagnostics[name]
                    projected = evidence.get("projected_template_center_xy")
                    contained = False
                    if name in current_points and projected is not None:
                        x_value, y_value = (float(value) for value in projected)
                        x_index = min(359, max(0, int(round(x_value))))
                        y_index = min(359, max(0, int(round(y_value))))
                        contained = bool(masks[name_to_id[name]][y_index, x_index])
                    evidence["projected_template_center_in_current_mask"] = contained
                    diagnostics[name]["relation_point_source"] = "sam_mask_centroid"
                    if contained:
                        relation = bool(
                            basket_bbox is not None
                            and basket_bbox[0] <= projected[0] < basket_bbox[2]
                            and basket_bbox[1] <= projected[1] < basket_bbox[3]
                        )
                        diagnostics[name]["centroid_in_tracked_basket_bbox"] = relation
                        diagnostics[name]["relation_point_source"] = (
                            "sift_affine_projected_template_center"
                        )
                        if not diagnostics[name]["uncertain_reasons"]:
                            diagnostics[name]["label"] = (
                                "inside" if relation else "outside"
                            )
                        temporal_relation_overrides += 1
                # A SAM-only mask can maintain the last evidence-backed state,
                # but cannot create a state change.  This prevents a tracker
                # drift from becoming a semantic transition while remaining
                # non-monotonic: later SIFT+SAM agreement can confirm either
                # direction.
                for name in TARGET_OBJECTS:
                    candidate_label = diagnostics[name]["label"]
                    previous_state = temporal_evidence_state[name]
                    evidence = current_locator_diagnostics[name]
                    sift_sam_agree = bool(
                        name in current_points
                        and evidence.get("valid") is True
                        and evidence.get(
                            "projected_template_center_in_current_mask"
                        )
                        and diagnostics[name].get("relation_point_source")
                        == "sift_affine_projected_template_center"
                    )
                    decision = "upstream_uncertain"
                    if candidate_label in {"inside", "outside"}:
                        if candidate_label == previous_state:
                            decision = (
                                "sift_sam_maintain"
                                if sift_sam_agree
                                else "sam_only_maintain"
                            )
                        elif sift_sam_agree:
                            temporal_evidence_state[name] = candidate_label
                            temporal_evidence_confirmed_flips[name] += 1
                            decision = "sift_sam_confirmed_flip"
                        else:
                            diagnostics[name]["pre_evidence_gate_label"] = (
                                candidate_label
                            )
                            diagnostics[name]["label"] = "uncertain"
                            diagnostics[name]["uncertain_reasons"] = sorted(
                                set(
                                    diagnostics[name]["uncertain_reasons"]
                                    + ["sam_only_state_change"]
                                )
                            )
                            temporal_evidence_blocked_flips[name] += 1
                            decision = "sam_only_flip_blocked"
                    diagnostics[name]["evidence_state_before"] = previous_state
                    diagnostics[name]["evidence_state_after"] = (
                        temporal_evidence_state[name]
                    )
                    diagnostics[name]["evidence_gate_decision"] = decision
            if frame_index == 0:
                t0_overlay = render_t0(frames[0], masks, diagnostics, current_points)
                t0_mask_ious = {}
                for left_index, left_id in enumerate(object_ids):
                    for right_id in object_ids[left_index + 1 :]:
                        pair = f"{ID_TO_NAME[left_id]}::{ID_TO_NAME[right_id]}"
                        t0_mask_ious[pair] = mask_iou(
                            masks[left_id], masks[right_id]
                        )
            scalar_similarities = None
            if current_locator_diagnostics and all(
                isinstance(record, Mapping) and "max_patch_cosine" in record
                for record in current_locator_diagnostics.values()
            ):
                scalar_similarities = {
                    name: float(record["max_patch_cosine"])
                    for name, record in current_locator_diagnostics.items()
                }
            output_rows.append(
                {
                    "schema": LABEL_SCHEMA,
                    "frame_id": source["frame_id"],
                    "task": TASK,
                    "episode_id": source["episode_id"],
                    "env_seed": source["env_seed"],
                    "t": source["t"],
                    "camera": CAMERA,
                    "source_image_sha256": source["image_sha256"],
                    "source_manifest_sha256": source["manifest_sha256"],
                    "prompt_frame_index": (
                        frame_index if causal_locator is not None and current_points else 0
                    ),
                    "post_t0_prompt_count": prompt_count,
                    "prompt_points_xy": {
                        name: list(value) for name, value in current_points.items()
                    },
                    "point_similarity": scalar_similarities,
                    "initializer_diagnostics": (
                        current_locator_diagnostics
                        if current_locator_diagnostics
                        else None
                    ),
                    "target_pairwise_mask_iou": current_target_mask_ious,
                    "objects": {
                        name: {"label": diagnostics[name]["label"]}
                        for name in TARGET_OBJECTS
                    },
                    "object_diagnostics": {
                        name: diagnostics[name] for name in TARGET_OBJECTS
                    },
                    "basket_diagnostics": diagnostics["basket_1"],
                }
            )
    output_rows.sort(key=lambda row: row["t"])
    require(t0_overlay is not None, "no frame-zero output")
    require(t0_mask_ious is not None, "no frame-zero mask-overlap diagnostics")
    t0 = output_rows[0]
    t0_masks = {}
    # Reconstruct overlap diagnostics from the already-computed t0 bounding
    # outputs is insufficient, so use mask bboxes/areas plus point containment as
    # the fail-closed automatic sanity.  Identity itself remains a visual audit.
    point_contained = {}
    for name in ID_TO_NAME.values():
        bbox = (
            t0["basket_diagnostics"]["mask_bbox_xyxy"]
            if name == "basket_1"
            else t0["object_diagnostics"][name]["mask_bbox_xyxy"]
        )
        x, y = points[name]
        point_contained[name] = bool(
            bbox is not None and bbox[0] <= x < bbox[2] and bbox[1] <= y < bbox[3]
        )
    summary = {
        "episode_id": rows[0]["episode_id"],
        "frame_count": len(output_rows),
        "full_track": full_track,
        "t0_point_contained_in_mask_bbox": point_contained,
        "t0_pairwise_mask_iou": t0_mask_ious,
        "t0_identity_collision_pairs": {
            pair: value
            for pair, value in t0_mask_ious.items()
            if value >= IDENTITY_COLLISION_MASK_IOU_GTE
        },
        "t0_objects": {
            "basket_1": t0["basket_diagnostics"],
            **t0["object_diagnostics"],
        },
        "uncertain_fraction": sum(
            row["objects"][name]["label"] == "uncertain"
            for row in output_rows
            for name in TARGET_OBJECTS
        )
        / (len(output_rows) * len(TARGET_OBJECTS)),
        "transitions": {
            name: v243_support.transition_summary(output_rows, name)
            for name in TARGET_OBJECTS
        },
        "causal_reacquisition": (
            {
                "strict_current_frame_rgb": True,
                "streaming_session": True,
                "post_t0_prompt_applications": temporal_prompt_applications,
                "locator_valid_frames": temporal_locator_valid,
                "locator_invalid_frames": temporal_locator_invalid,
                "preprompt_target_collisions": temporal_preprompt_collisions,
                "postmask_target_collisions": temporal_postmask_collisions,
                "relation_point_overrides": temporal_relation_overrides,
                "evidence_gate_rule": (
                    "SAM-only may maintain the last evidence-backed state; a change in either direction "
                    "requires current-frame SIFT+SAM agreement, otherwise raw label is uncertain"
                ),
                "evidence_confirmed_flips": temporal_evidence_confirmed_flips,
                "evidence_blocked_flips": temporal_evidence_blocked_flips,
            }
            if causal_locator is not None
            else None
        ),
    }
    del session, frames, t0_masks
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_rows, summary, t0_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--episode-id", type=int, action="append", required=True)
    parser.add_argument(
        "--mode",
        choices=(
            "fixed_points",
            "dinov2_points",
            "dinov2_cluster_points",
            "sift_cluster_points",
            "sift_causal_reacquire",
        ),
        default="fixed_points",
    )
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        default=REPO
        / "results/v242_reset_rgb/chain3_lr2_first16_v242/manifest.jsonl",
    )
    parser.add_argument("--sam2-checkpoint", type=Path, default=DEFAULT_SAM2)
    parser.add_argument("--dino-model", type=Path, default=DEFAULT_DINO)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--full-track", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    episode_ids = sorted(set(args.episode_id))
    require(len(episode_ids) == len(args.episode_id), "duplicate --episode-id")
    require(all(0 <= episode_id <= 95 for episode_id in episode_ids), "episode outside 0..95")
    require(
        args.mode != "sift_causal_reacquire" or args.full_track,
        "sift_causal_reacquire requires --full-track",
    )
    device = torch.device(args.device)
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA requested but unavailable")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    if device.type == "cpu":
        require(dtype == torch.float32, "CPU probe requires float32")

    episodes, manifest_provenance = load_manifest_rows(
        args.manifest, set(episode_ids), reset_only=False
    )
    dino_provenance = None
    sift_provenance = None
    calibration_provenance = None
    causal_locator = None
    if args.mode in {"dinov2_points", "dinov2_cluster_points"}:
        calibration_ids = set(range(16))
        calibration, calibration_provenance = load_manifest_rows(
            [args.calibration_manifest], calibration_ids, reset_only=True
        )
        points_by_episode, scores_by_episode, dino_provenance = make_dino_points(
            args.dino_model.expanduser().resolve(),
            calibration,
            episodes,
            clustered=args.mode == "dinov2_cluster_points",
        )
        dino_provenance["calibration_manifests"] = calibration_provenance
    elif args.mode in {"sift_cluster_points", "sift_causal_reacquire"}:
        calibration, calibration_provenance = load_manifest_rows(
            [args.calibration_manifest], {0}, reset_only=True
        )
        if args.mode == "sift_causal_reacquire":
            causal_locator = CausalSiftAffineLocator(calibration)
            points_by_episode = {}
            scores_by_episode = {}
            for episode_id in episode_ids:
                frame = Image.open(
                    episodes[episode_id][0]["image_path"]
                ).convert("RGB")
                located, diagnostics = causal_locator.locate(frame)
                require(
                    set(located) == set(ID_TO_NAME.values()),
                    f"ep{episode_id}: causal SIFT preflight did not initialize every identity",
                )
                points_by_episode[episode_id] = located
                scores_by_episode[episode_id] = diagnostics
            sift_provenance = dict(causal_locator.provenance)
        else:
            points_by_episode, scores_by_episode, sift_provenance = make_sift_points(
                calibration, episodes
            )
        sift_provenance["calibration_manifests"] = calibration_provenance
    else:
        points_by_episode = {
            episode_id: dict(FIXED_POINTS_XY) for episode_id in episode_ids
        }
        scores_by_episode = {episode_id: {} for episode_id in episode_ids}

    checkpoint = args.sam2_checkpoint.expanduser().resolve()
    require(checkpoint.is_file(), f"missing SAM2 checkpoint {checkpoint}")
    started = time.perf_counter()
    model, tensor_count = sam2_support.load_model(checkpoint, device, dtype)
    model.requires_grad_(False)
    processor = Sam2VideoProcessor(
        image_processor=Sam2ImageProcessor(),
        video_processor=Sam2VideoVideoProcessor(),
    )

    output_dir = args.output_dir.expanduser().resolve()
    require(not output_dir.exists(), f"output exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        overlay_dir = temp_dir / "t0_overlays"
        overlay_dir.mkdir()
        all_labels: list[dict[str, Any]] = []
        summaries: dict[str, Any] = {}
        for episode_id in episode_ids:
            rows, summary, overlay = track_episode(
                model,
                processor,
                episodes[episode_id],
                points_by_episode[episode_id],
                scores_by_episode[episode_id] or None,
                dtype=dtype,
                full_track=args.full_track,
                causal_locator=causal_locator,
            )
            overlay_path = overlay_dir / f"ep{episode_id:04d}_t0000.jpg"
            overlay.save(overlay_path, quality=95)
            summary["t0_overlay"] = str(overlay_path.relative_to(temp_dir))
            summaries[str(episode_id)] = summary
            all_labels.extend(rows)
            print(
                f"ep{episode_id}: t0 boxes="
                + ", ".join(
                    f"{name}:{summary['t0_objects'][name]['mask_bbox_xyxy']}"
                    for name in ID_TO_NAME.values()
                ),
                flush=True,
            )

        all_labels.sort(key=lambda row: (row["episode_id"], row["t"]))
        labels_path = temp_dir / "labels.jsonl"
        with labels_path.open("w", encoding="utf-8") as handle:
            for row in all_labels:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        code_path = Path(__file__).resolve()
        archived_code_path = temp_dir / "producer_source.py"
        shutil.copy2(code_path, archived_code_path)
        archived_code_sha256 = sha256_file(archived_code_path)
        initializer_config = {
            "mode": args.mode,
            "fixed_points_xy": FIXED_POINTS_XY if args.mode == "fixed_points" else None,
            "dino_prototype_anchor_points_xy": (
                FIXED_POINTS_XY if dino_provenance else None
            ),
            "dino_search_xyxy": (
                dino_provenance["search_regions_xyxy"] if dino_provenance else None
            ),
            "dino_input_size": DINO_INPUT_SIZE if dino_provenance else None,
            "dino_normalization_mean": DINO_MEAN if dino_provenance else None,
            "dino_normalization_std": DINO_STD if dino_provenance else None,
            "dino_prototype_calibration_episode_ids": (
                dino_provenance["prototype_calibration_episode_ids"]
                if dino_provenance
                else None
            ),
            "sift_template_xyxy": (
                SIFT_TEMPLATE_XYXY if sift_provenance else None
            ),
            "dino_cluster_top_k": (
                DINO_CLUSTER_TOP_K
                if args.mode == "dinov2_cluster_points"
                else None
            ),
            "dino_cluster_softmax_temperature": (
                DINO_CLUSTER_SOFTMAX_TEMPERATURE
                if args.mode == "dinov2_cluster_points"
                else None
            ),
            "sift_upscale": SIFT_UPSCALE if sift_provenance else None,
            "sift_contrast_threshold": (
                SIFT_CONTRAST_THRESHOLD if sift_provenance else None
            ),
            "sift_lowe_ratio": SIFT_LOWE_RATIO if sift_provenance else None,
            "sift_cluster_eps_px": SIFT_CLUSTER_EPS_PX if sift_provenance else None,
            "sift_min_cluster_matches": (
                SIFT_MIN_CLUSTER_MATCHES if sift_provenance else None
            ),
            "sift_target_keypoint_deduplication": (
                "lowest descriptor distance per trainIdx"
                if sift_provenance
                else None
            ),
            "identity_collision_mask_iou_gte": IDENTITY_COLLISION_MASK_IOU_GTE,
            "causal_reacquisition": (
                {
                    "streaming_current_frame_only": True,
                    "ransac_reprojection_threshold_px": SIFT_RANSAC_REPROJECTION_THRESHOLD_PX,
                    "ransac_max_iters": SIFT_RANSAC_MAX_ITERS,
                    "ransac_confidence": SIFT_RANSAC_CONFIDENCE,
                    "ransac_refine_iters": SIFT_RANSAC_REFINE_ITERS,
                    "ransac_min_inliers": SIFT_RANSAC_MIN_INLIERS,
                    "ransac_min_inlier_ratio": SIFT_RANSAC_MIN_INLIER_RATIO,
                    "ransac_scale_range": SIFT_RANSAC_SCALE_RANGE,
                    "target_prompt_collision_distance_px_lte": TEMPORAL_TARGET_PROMPT_COLLISION_DISTANCE_PX,
                    "target_mask_collision_iou_gte": TEMPORAL_TARGET_MASK_COLLISION_IOU_GTE,
                    "later_locator_failure": "omit object prompt; SAM2 uses past memory",
                    "evidence_state_gate": (
                        "state starts outside; SAM-only may maintain but cannot change state; "
                        "a bidirectional change requires current-frame SIFT+SAM agreement, "
                        "otherwise emit uncertain"
                    ),
                }
                if causal_locator is not None
                else None
            ),
            "thresholds": THRESHOLDS,
        }
        automatic_t0_sanity_pass = all(
            all(summary["t0_point_contained_in_mask_bbox"].values())
            and not summary["t0_identity_collision_pairs"]
            for summary in summaries.values()
        )
        result = {
            "schema": SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "rgb_only_candidate_complete_not_privileged_evaluated",
            "claim_scope": "development probe only; not a frozen V2 registration",
            "data_boundary": (
                "v242 replay/reset manifest JSONL and referenced agentview JPEGs only; "
                "no tape/summary/task-definition/simulator/event/milestone/success/BDDL access"
            ),
            "privileged_inputs_read": False,
            "automatic_t0_sanity_pass": automatic_t0_sanity_pass,
            "development_lineage": DEVELOPMENT_LINEAGE,
            "mode": args.mode,
            "full_track": args.full_track,
            "causal_current_and_past_only": causal_locator is not None,
            "episode_ids": episode_ids,
            "initializer": {
                "configuration": initializer_config,
                "config_sha256": canonical_sha256(initializer_config),
                "per_episode_points_xy": {
                    str(episode_id): points_by_episode[episode_id]
                    for episode_id in episode_ids
                },
                "per_episode_locator_diagnostics": {
                    str(episode_id): scores_by_episode[episode_id]
                    for episode_id in episode_ids
                }
                if dino_provenance or sift_provenance
                else None,
            },
            "dino": dino_provenance,
            "sift": sift_provenance,
            "sam2": {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "tensor_count": tensor_count,
                "dtype": str(dtype),
                "device": str(device),
                "support_code": {
                    "path": str(Path(sam2_support.__file__).resolve()),
                    "sha256": sha256_file(Path(sam2_support.__file__).resolve()),
                },
                "classification_support_code": {
                    "path": str(Path(v243_support.__file__).resolve()),
                    "sha256": sha256_file(Path(v243_support.__file__).resolve()),
                },
                "transformers_modeling_source": {
                    "path": str(Path(inspect.getsourcefile(type(model))).resolve()),
                    "sha256": sha256_file(
                        Path(inspect.getsourcefile(type(model))).resolve()
                    ),
                },
                "transformers_processing_source": {
                    "path": str(
                        Path(inspect.getsourcefile(Sam2VideoProcessor)).resolve()
                    ),
                    "sha256": sha256_file(
                        Path(inspect.getsourcefile(Sam2VideoProcessor)).resolve()
                    ),
                },
            },
            "thresholds": THRESHOLDS,
            "inputs": manifest_provenance,
            "episodes": summaries,
            "outputs": {
                "labels": "labels.jsonl",
                "labels_sha256": sha256_file(labels_path),
                "label_rows": len(all_labels),
                "t0_overlay_dir": "t0_overlays",
            },
            "runtime": {
                "seconds": time.perf_counter() - started,
                "argv": list(sys.argv),
                "git_head": git_head(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            "code": {
                "path": str(code_path),
                "sha256": archived_code_sha256,
                "archived_path": "producer_source.py",
                "archived_sha256": archived_code_sha256,
            },
        }
        result_path = temp_dir / "result.json"
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        complete = {
            "schema": COMPLETE_SCHEMA,
            "status": "complete",
            "result_sha256": sha256_file(result_path),
            "labels_sha256": sha256_file(labels_path),
            "code_sha256": archived_code_sha256,
            "privileged_inputs_read": False,
        }
        (temp_dir / "COMPLETE.json").write_text(
            json.dumps(complete, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.rename(temp_dir, output_dir)
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "result_sha256": complete["result_sha256"],
                "labels_sha256": complete["labels_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
