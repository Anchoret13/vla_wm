#!/usr/bin/env python3
"""RGB-only probe for alternative SAM2 frame-zero initializers.

This is deliberately independent of the frozen V243 registration and teacher.
It reads only replay/reset JSONL manifests and the JPEGs referenced by them.  It
does not open tape files, replay/source summaries, task definitions, simulator
state, events, milestones, or success labels.  Any privileged evaluation must
therefore run later, against the completed artifact written here.

Four deterministic candidates are supported:

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

The default is a frame-zero segmentation probe.  ``--full-track`` propagates the
result through the RGB episode, still with no prompt after frame zero.
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
            good = [
                pair[0]
                for pair in pairs
                if len(pair) == 2
                and pair[0].distance < SIFT_LOWE_RATIO * pair[1].distance
            ]
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
                "ratio_test_matches": len(good),
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
        "location_rule": "inverse-descriptor-distance weighted centroid of largest connected cluster",
    }
    return points, diagnostics, provenance


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
) -> tuple[list[dict[str, Any]], dict[str, Any], Image.Image]:
    frames = [Image.open(row["image_path"]).convert("RGB") for row in rows]
    session_frames = frames if full_track else frames[:1]
    session = processor.init_video_session(
        video=session_frames,
        inference_device=model.device,
        inference_state_device=model.device,
        processing_device=model.device,
        video_storage_device="cpu",
        dtype=dtype,
    )
    object_ids = sorted(ID_TO_NAME)
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
    with torch.inference_mode():
        if full_track:
            model(inference_session=session, frame_idx=0)
            outputs = model.propagate_in_video_iterator(
                session, show_progress_bar=False
            )
        else:
            outputs = [model(inference_session=session, frame_idx=0)]
        for output in outputs:
            frame_index = int(output.frame_idx)
            diagnostics, masks = classify_output(processor, output)
            source = rows[frame_index]
            if frame_index == 0:
                t0_overlay = render_t0(frames[0], masks, diagnostics, points)
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
                    "prompt_frame_index": 0,
                    "post_t0_prompt_count": 0,
                    "prompt_points_xy": {
                        name: list(points[name]) for name in ID_TO_NAME.values()
                    },
                    "point_similarity": dict(point_scores) if point_scores else None,
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
    elif args.mode == "sift_cluster_points":
        calibration, calibration_provenance = load_manifest_rows(
            [args.calibration_manifest], {0}, reset_only=True
        )
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
            "dino_search_xyxy": (
                dino_provenance["search_regions_xyxy"] if dino_provenance else None
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
            "thresholds": THRESHOLDS,
        }
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
            "mode": args.mode,
            "full_track": args.full_track,
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
