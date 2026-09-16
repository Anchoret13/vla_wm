#!/usr/bin/env python3
"""Fuse sealed agentview tracks with eye-in-hand SIFT cream evidence.

This producer is restricted to already disclosed seed-8700 development RGB.
It consumes sealed v245 SAM3 agentview frame records, but never their post-seal
event evaluation.  A fixed ep0 eye-in-hand cream crop supplies class identity.
Each later SIFT query uses only the current eye-in-hand frame.  The causal state
machine can emit a cream-placement increment only after an authenticated track
approaches the basket rim and then disappears with short-window occlusion
evidence.  Cross-class overlap with a generic can fails closed.

No rollout summary, event, success, latent tape, task definition, or simulator
state is opened.  The result is post-hoc development evidence, not Gate-0
validation and not authorization to open a new panel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lcwm.v247_cream_state import (
    CreamFrameEvidence,
    CreamStateConfig,
    run_causal_cream_state,
)


SCHEMA = "v247_multiview_cream_development_probe_v1"
FRAME_SCHEMA = "v247_multiview_cream_development_frame_v1"
COMPLETE_SCHEMA = "v247_multiview_cream_development_complete_v1"
SAM3_RESULT_SCHEMA = "v245_sam3_video_progress_probe_v1"
SAM3_FRAME_SCHEMA = "v245_sam3_video_progress_frame_v1"
SAM3_COMPLETE_SCHEMA = "v245_sam3_video_progress_complete_v1"
EXPECTED_SAM3_PRODUCER_SHA256 = (
    "fe783028bc59459ac5642e285fee17b6842c5a7f597b190d430936ec0d5042f1"
)
EXPECTED_TEMPLATE_RESULT_SHA256 = (
    "c85813f2cfd03e5aad839544d50849f3db8cdea2f0bbf2f93ca814b18014b881"
)
EXPECTED_TEMPLATE_IMAGE_SHA256 = (
    "84e211e05688b0d02c6e98d58c6e0037e9feb5820e78f4840e72b8551dacbfe1"
)
TASK = "chain3_lr2"
ALLOWED_EPISODES = frozenset({0, 12, 37, 71})
FORBIDDEN_KEYS = frozenset(
    {
        "success",
        "successes",
        "is_success",
        "event",
        "events",
        "milestone",
        "milestones",
        "stage_reached",
        "bddl",
        "outcome",
    }
)

SIFT_UPSCALE = 3
SIFT_CONTRAST_THRESHOLD = 0.01
SIFT_LOWE_RATIO = 0.70
SIFT_CLUSTER_EPS_PX = 22.0
SIFT_MIN_CLUSTER_MATCHES = 4
SIFT_RANSAC_REPROJECTION_THRESHOLD_PX = 3.0
SIFT_RANSAC_MAX_ITERS = 2000
SIFT_RANSAC_CONFIDENCE = 0.99
SIFT_RANSAC_REFINE_ITERS = 10
SIFT_RANSAC_MIN_INLIERS = 4
SIFT_RANSAC_MIN_INLIER_RATIO = 0.50
SIFT_SCALE_RANGE = (0.50, 4.00)

RIM_MARGIN_PX = 8.0
RIM_MAX_DEPTH_FRACTION = 0.30
CROSS_CLASS_BOX_IOS_GTE = 0.50


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def reject_semantic_fields(value: Any, where: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            require(normalized not in FORBIDDEN_KEYS, f"{where}.{key}: forbidden field")
            reject_semantic_fields(child, f"{where}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_semantic_fields(child, f"{where}[{index}]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--sam3-artifact", action="append", type=Path, required=True)
    parser.add_argument("--episode-id", action="append", type=int, required=True)
    parser.add_argument("--template-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix-check-frames", type=int, default=50)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"{path}: expected JSON object")
    return value


def load_manifest_rows(
    paths: Sequence[Path], episode_ids: Sequence[int]
) -> tuple[dict[tuple[int, int], dict[str, Any]], list[dict[str, Any]]]:
    wanted = set(episode_ids)
    require(len(wanted) == len(episode_ids), "duplicate episode ID")
    require(wanted <= ALLOWED_EPISODES, "only disclosed development episodes allowed")
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    sources = []
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        require(path.is_file(), f"missing manifest: {path}")
        selected = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                reject_semantic_fields(row, f"{path}:{line_number}")
                episode_id = int(row.get("episode_id", -1))
                if episode_id not in wanted:
                    continue
                require(row.get("schema") == "v242_rgb_replay_v1", "manifest schema")
                require(row.get("task") == TASK, "manifest task mismatch")
                require(
                    int(row.get("env_seed", -1)) == 8700 + episode_id,
                    "manifest is not seed-8700 development data",
                )
                key = (episode_id, int(row["t"]))
                require(key not in rows, f"duplicate manifest row {key}")
                camera_rows = {}
                for camera in ("agentview", "eye_in_hand"):
                    image = row.get("images", {}).get(camera)
                    require(isinstance(image, Mapping), f"{key}: missing {camera}")
                    image_path = (path.parent / str(image["path"])).resolve()
                    require(image_path.is_file(), f"missing image {image_path}")
                    expected = str(image["sha256"])
                    require(sha256_file(image_path) == expected, f"image drift {image_path}")
                    camera_rows[camera] = {
                        "path": str(image_path),
                        "sha256": expected,
                        "height": int(image["height"]),
                        "width": int(image["width"]),
                    }
                rows[key] = {
                    "episode_id": episode_id,
                    "env_seed": int(row["env_seed"]),
                    "t": int(row["t"]),
                    "frame_id": str(row["frame_id"]),
                    "images": camera_rows,
                }
                selected += 1
        sources.append(
            {"path": str(path), "sha256": sha256_file(path), "selected_rows": selected}
        )
    for episode_id in wanted:
        require(any(key[0] == episode_id for key in rows), f"ep{episode_id}: no manifest")
    return rows, sources


def load_sam3_artifacts(
    roots: Sequence[Path], episode_ids: Sequence[int]
) -> tuple[dict[tuple[int, int], dict[str, Any]], list[dict[str, Any]]]:
    wanted = set(episode_ids)
    frames: dict[tuple[int, int], dict[str, Any]] = {}
    sources = []
    for raw_root in roots:
        root = raw_root.expanduser().resolve()
        result_path = root / "result.json"
        frames_path = root / "frames.jsonl"
        producer_path = root / "producer_source.py"
        complete_path = root / "COMPLETE.json"
        for path in (result_path, frames_path, producer_path, complete_path):
            require(path.is_file(), f"missing sealed SAM3 file: {path}")
        complete = load_json(complete_path)
        require(complete.get("schema") == SAM3_COMPLETE_SCHEMA, "SAM3 COMPLETE schema")
        require(complete.get("status") == "complete", "SAM3 artifact incomplete")
        require(complete.get("result_sha256") == sha256_file(result_path), "result drift")
        require(complete.get("frames_sha256") == sha256_file(frames_path), "frames drift")
        require(
            complete.get("producer_source_sha256") == sha256_file(producer_path),
            "producer snapshot drift",
        )
        require(
            sha256_file(producer_path) == EXPECTED_SAM3_PRODUCER_SHA256,
            "unexpected SAM3 producer version",
        )
        result = load_json(result_path)
        reject_semantic_fields(result, f"{result_path}")
        require(result.get("schema") == SAM3_RESULT_SCHEMA, "SAM3 result schema")
        require(result.get("privileged_inputs_read") is False, "SAM3 privileged input")
        selected = 0
        with frames_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                frame = json.loads(line)
                reject_semantic_fields(frame, f"{frames_path}:{line_number}")
                require(frame.get("schema") == SAM3_FRAME_SCHEMA, "SAM3 frame schema")
                episode_id = int(frame["episode_id"])
                if episode_id not in wanted:
                    continue
                key = (episode_id, int(frame["t"]))
                require(key not in frames, f"duplicate SAM3 frame {key}")
                frames[key] = frame
                selected += 1
        sources.append(
            {
                "root": str(root),
                "result_sha256": sha256_file(result_path),
                "frames_sha256": sha256_file(frames_path),
                "complete_sha256": sha256_file(complete_path),
                "selected_frames": selected,
            }
        )
    for episode_id in wanted:
        require(any(key[0] == episode_id for key in frames), f"ep{episode_id}: no SAM3")
    return frames, sources


def connected_groups(points_xy: np.ndarray) -> list[list[int]]:
    seen: set[int] = set()
    groups = []
    for start in range(len(points_xy)):
        if start in seen:
            continue
        seen.add(start)
        queue = [start]
        group = []
        while queue:
            current = queue.pop()
            group.append(current)
            distances = np.linalg.norm(points_xy - points_xy[current], axis=1)
            for candidate in np.flatnonzero(distances <= SIFT_CLUSTER_EPS_PX):
                index = int(candidate)
                if index not in seen:
                    seen.add(index)
                    queue.append(index)
        groups.append(group)
    return groups


class EyeCreamSift:
    def __init__(
        self,
        template_result_path: Path | None,
        *,
        sealed_metadata_bytes: bytes | None = None,
        sealed_image_bytes: bytes | None = None,
        expected_metadata_sha256: str | None = None,
        expected_image_sha256: str | None = None,
    ) -> None:
        if sealed_metadata_bytes is not None or sealed_image_bytes is not None:
            require(
                template_result_path is None
                and isinstance(sealed_metadata_bytes, bytes)
                and isinstance(sealed_image_bytes, bytes),
                "formal template inputs must use only captured bytes",
            )
            require(
                expected_metadata_sha256 == EXPECTED_TEMPLATE_RESULT_SHA256
                and hashlib.sha256(sealed_metadata_bytes).hexdigest()
                == expected_metadata_sha256,
                "sealed template metadata drift",
            )
            require(
                expected_image_sha256 == EXPECTED_TEMPLATE_IMAGE_SHA256
                and hashlib.sha256(sealed_image_bytes).hexdigest()
                == expected_image_sha256,
                "sealed template image drift",
            )
            result = json.loads(sealed_metadata_bytes.decode("utf-8"))
            result_label = "sealed registered template metadata"
            image_label = "sealed registered template image"
            bgr = cv2.imdecode(
                np.frombuffer(sealed_image_bytes, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
        else:
            require(template_result_path is not None, "template result path missing")
            path = template_result_path.expanduser().resolve()
            require(path.is_file(), f"missing template result: {path}")
            require(
                sha256_file(path) == EXPECTED_TEMPLATE_RESULT_SHA256,
                "template result drift",
            )
            result = load_json(path)
            result_label = str(path)
            image = result.get("image")
            require(isinstance(image, Mapping), "template result has no image record")
            image_path = Path(str(image["path"])).expanduser().resolve()
            require(image_path.is_file(), "template image missing")
            require(
                str(image["sha256"]) == EXPECTED_TEMPLATE_IMAGE_SHA256,
                "template image record",
            )
            require(
                sha256_file(image_path) == EXPECTED_TEMPLATE_IMAGE_SHA256,
                "template image drift",
            )
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            image_label = str(image_path)
        reject_semantic_fields(result, result_label)
        image = result.get("image")
        require(isinstance(image, Mapping), "template result has no image record")
        require(str(image["sha256"]) == EXPECTED_TEMPLATE_IMAGE_SHA256, "template image record")
        instances = result.get("predictions", {}).get("cream_cheese_1", {}).get("instances")
        require(isinstance(instances, list) and instances, "template cream missing")
        require(
            all(
                float(left["score"]) >= float(right["score"])
                for left, right in zip(instances, instances[1:])
            ),
            "template instances are not score ordered",
        )
        box = instances[0]["box_xyxy"]
        require(isinstance(box, list) and len(box) == 4, "template box invalid")
        x0, y0, x1, y1 = [int(round(float(value))) for value in box]
        require(0 <= x0 < x1 <= 360 and 0 <= y0 < y1 <= 360, "template crop invalid")
        require(bgr is not None, f"template image decode failed: {image_label}")
        crop = bgr[y0:y1, x0:x1]
        enlarged = cv2.resize(
            crop,
            None,
            fx=SIFT_UPSCALE,
            fy=SIFT_UPSCALE,
            interpolation=cv2.INTER_CUBIC,
        )
        self.sift = cv2.SIFT_create(contrastThreshold=SIFT_CONTRAST_THRESHOLD)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.keypoints, self.descriptors = self.sift.detectAndCompute(enlarged, None)
        require(self.descriptors is not None and len(self.keypoints) >= 2, "no template SIFT")
        self.center = ((x1 - x0) / 2.0, (y1 - y0) / 2.0)
        self.provenance = {
            "result_path": result_label,
            "result_sha256": EXPECTED_TEMPLATE_RESULT_SHA256,
            "image_path": image_label,
            "image_sha256": EXPECTED_TEMPLATE_IMAGE_SHA256,
            "crop_xyxy_nearest_even_rounding": [x0, y0, x1, y1],
            "crop_sha256": hashlib.sha256(np.ascontiguousarray(crop).tobytes()).hexdigest(),
            "template_keypoints": len(self.keypoints),
        }

    def locate(
        self,
        image_path: Path | None,
        *,
        sealed_bytes: bytes | None = None,
        expected_sha256: str | None = None,
    ) -> dict[str, Any]:
        if sealed_bytes is not None:
            require(image_path is None, "sealed SIFT input must not reopen a path")
            require(
                expected_sha256 is not None
                and hashlib.sha256(sealed_bytes).hexdigest() == expected_sha256,
                "sealed SIFT buffer hash mismatch",
            )
            bgr = cv2.imdecode(
                np.frombuffer(sealed_bytes, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            label = "sealed eye-in-hand buffer"
        else:
            require(image_path is not None, "SIFT image path missing")
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            label = str(image_path)
        require(bgr is not None, f"image decode failed: {label}")
        enlarged = cv2.resize(
            bgr,
            None,
            fx=SIFT_UPSCALE,
            fy=SIFT_UPSCALE,
            interpolation=cv2.INTER_CUBIC,
        )
        scene_keypoints, scene_descriptors = self.sift.detectAndCompute(enlarged, None)
        reasons = []
        ratio_matches = []
        unique_matches = []
        groups: list[list[int]] = []
        selected: list[int] = []
        inliers: list[int] = []
        affine = None
        scale = None
        projected = None
        if scene_descriptors is None:
            reasons.append("no_scene_descriptors")
        else:
            pairs = self.matcher.knnMatch(self.descriptors, scene_descriptors, k=2)
            ratio_matches = [
                pair[0]
                for pair in pairs
                if len(pair) == 2
                and pair[0].distance < SIFT_LOWE_RATIO * pair[1].distance
            ]
            best_by_target: dict[int, Any] = {}
            for match in ratio_matches:
                previous = best_by_target.get(int(match.trainIdx))
                if previous is None or (float(match.distance), int(match.queryIdx)) < (
                    float(previous.distance),
                    int(previous.queryIdx),
                ):
                    best_by_target[int(match.trainIdx)] = match
            unique_matches = sorted(
                best_by_target.values(),
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
            groups = connected_groups(scene_points)
            groups.sort(
                key=lambda group: (
                    len(group),
                    -float(np.mean([unique_matches[index].distance for index in group])),
                    -min(group),
                ),
                reverse=True,
            )
            selected = groups[0] if groups else []
            if len(selected) < SIFT_MIN_CLUSTER_MATCHES:
                reasons.append("cluster_support_lt_min")
            else:
                template_points = (
                    np.asarray(
                        [self.keypoints[unique_matches[index].queryIdx].pt for index in selected],
                        dtype=np.float32,
                    )
                    / SIFT_UPSCALE
                )
                selected_scene = scene_points[selected].astype(np.float32)
                cv2.setRNGSeed(0)
                affine, inlier_mask = cv2.estimateAffinePartial2D(
                    template_points,
                    selected_scene,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=SIFT_RANSAC_REPROJECTION_THRESHOLD_PX,
                    maxIters=SIFT_RANSAC_MAX_ITERS,
                    confidence=SIFT_RANSAC_CONFIDENCE,
                    refineIters=SIFT_RANSAC_REFINE_ITERS,
                )
                if affine is None or inlier_mask is None:
                    reasons.append("affine_fit_failed")
                else:
                    inliers = [
                        selected[index]
                        for index, flag in enumerate(inlier_mask.ravel())
                        if bool(flag)
                    ]
                    ratio = len(inliers) / len(selected)
                    if len(inliers) < SIFT_RANSAC_MIN_INLIERS:
                        reasons.append("ransac_inliers_lt_min")
                    if ratio < SIFT_RANSAC_MIN_INLIER_RATIO:
                        reasons.append("ransac_inlier_ratio_lt_min")
                    scale = float(math.hypot(affine[0, 0], affine[1, 0]))
                    if not SIFT_SCALE_RANGE[0] <= scale <= SIFT_SCALE_RANGE[1]:
                        reasons.append("affine_scale_out_of_range")
                    center = np.asarray([*self.center, 1.0], dtype=np.float64)
                    projected_array = affine @ center
                    projected = [float(projected_array[0]), float(projected_array[1])]
                    height, width = bgr.shape[:2]
                    if not (
                        np.isfinite(projected_array).all()
                        and 0 <= projected_array[0] < width
                        and 0 <= projected_array[1] < height
                    ):
                        reasons.append("projected_center_out_of_frame")
        valid = not reasons
        return {
            "valid": valid,
            "invalid_reasons": sorted(set(reasons)),
            "ratio_matches": len(ratio_matches),
            "unique_target_matches": len(unique_matches),
            "connected_components": len(groups),
            "selected_cluster_matches": len(selected),
            "ransac_inliers": len(inliers),
            "ransac_inlier_ratio": len(inliers) / len(selected) if selected else None,
            "affine_scale": scale if valid else None,
            "projected_template_center_xy": projected if valid else None,
            "affine_matrix": affine.tolist() if valid and affine is not None else None,
        }


def box_intersection_over_smaller(left: Sequence[float], right: Sequence[float]) -> float:
    ix0 = max(float(left[0]), float(right[0]))
    iy0 = max(float(left[1]), float(right[1]))
    ix1 = min(float(left[2]), float(right[2]))
    iy1 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(
        0.0, float(left[3]) - float(left[1])
    )
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(
        0.0, float(right[3]) - float(right[1])
    )
    denominator = min(left_area, right_area)
    return intersection / denominator if denominator > 0 else 0.0


def derive_evidence(
    episode_frames: Sequence[dict[str, Any]],
    sift_rows: Sequence[dict[str, Any]],
) -> tuple[list[CreamFrameEvidence], int]:
    require(len(episode_frames) == len(sift_rows), "frame/SIFT length mismatch")
    initial = episode_frames[0].get("selection", {}).get("cream")
    require(isinstance(initial, list) and len(initial) == 1, "t0 cream track not unique")
    initial_track_id = int(initial[0]["object_id"])
    evidence = []
    for frame_index, (frame, sift) in enumerate(zip(episode_frames, sift_rows, strict=True)):
        selected_cream = frame.get("selection", {}).get("cream", [])
        initial_records = [
            item for item in selected_cream if int(item["object_id"]) == initial_track_id
        ]
        require(len(initial_records) <= 1, "duplicate initial cream track")
        track = initial_records[0] if initial_records else None
        baskets = frame.get("selection", {}).get("basket", [])
        basket = baskets[0] if len(baskets) == 1 else None
        near_rim = False
        cross_collision = False
        if track is not None:
            track_box = track.get("mask_bbox_xyxy")
            require(isinstance(track_box, list) and len(track_box) == 4, "cream box missing")
            overlaps = [
                box_intersection_over_smaller(track_box, item["mask_bbox_xyxy"])
                for item in frame.get("selection", {}).get("can", [])
                if isinstance(item.get("mask_bbox_xyxy"), list)
            ]
            cross_collision = max(overlaps, default=0.0) >= CROSS_CLASS_BOX_IOS_GTE
            if basket is not None:
                center = track.get("mask_centroid_xy")
                basket_box = basket.get("mask_bbox_xyxy")
                require(isinstance(center, list) and len(center) == 2, "cream center missing")
                require(
                    isinstance(basket_box, list) and len(basket_box) == 4,
                    "basket box missing",
                )
                top = float(basket_box[1])
                depth = float(basket_box[3]) - top
                near_rim = bool(
                    float(basket_box[0]) <= float(center[0]) <= float(basket_box[2])
                    and top - RIM_MARGIN_PX
                    <= float(center[1])
                    <= top + RIM_MAX_DEPTH_FRACTION * depth
                )
        evidence.append(
            CreamFrameEvidence(
                frame_index=frame_index,
                t=int(frame["t"]),
                initial_track_present=track is not None,
                near_basket_rim=near_rim,
                cross_class_collision=cross_collision,
                sift_valid=bool(sift["valid"]),
                sift_scale=(float(sift["affine_scale"]) if sift["valid"] else None),
            )
        )
    return evidence, initial_track_id


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    episode_ids = list(args.episode_id)
    require(args.prefix_check_frames >= 0, "prefix-check-frames must be nonnegative")
    output_dir = args.output_dir.expanduser().resolve()
    require(not output_dir.exists(), f"refusing to overwrite {output_dir}")
    manifests, manifest_sources = load_manifest_rows(args.manifest, episode_ids)
    sam3_frames, sam3_sources = load_sam3_artifacts(args.sam3_artifact, episode_ids)
    require(set(sam3_frames) <= set(manifests), "SAM3 frame has no manifest row")
    locator = EyeCreamSift(args.template_result)
    config = CreamStateConfig()
    output_frames = []
    diagnostics = {}
    prefix_checks = {}

    for episode_id in episode_ids:
        keys = sorted(
            (key for key in sam3_frames if key[0] == episode_id), key=lambda key: key[1]
        )
        episode_frames = [sam3_frames[key] for key in keys]
        require(keys[0][1] == 0, f"ep{episode_id}: first frame is not t0")
        for key, frame in zip(keys, episode_frames, strict=True):
            manifest = manifests[key]
            require(
                frame["image_sha256"]
                == manifest["images"]["agentview"]["sha256"],
                "agentview binding drift",
            )
            require(frame["frame_id"] == manifest["frame_id"], "frame ID binding drift")
        sift_rows = [
            locator.locate(Path(manifests[key]["images"]["eye_in_hand"]["path"]))
            for key in keys
        ]
        evidence, initial_track_id = derive_evidence(episode_frames, sift_rows)
        states = run_causal_cream_state(evidence, config)
        for frame, sift, item, state in zip(
            episode_frames, sift_rows, evidence, states, strict=True
        ):
            record = {
                "schema": FRAME_SCHEMA,
                "episode_id": episode_id,
                "env_seed": int(frame["env_seed"]),
                "frame_index": item.frame_index,
                "t": item.t,
                "frame_id": str(frame["frame_id"]),
                "agentview_image_sha256": str(frame["image_sha256"]),
                "eye_in_hand_image_sha256": manifests[(episode_id, item.t)][
                    "images"
                ]["eye_in_hand"]["sha256"],
                "initial_cream_track_id": initial_track_id,
                "evidence": {
                    "initial_track_present": item.initial_track_present,
                    "near_basket_rim": item.near_basket_rim,
                    "cross_class_collision": item.cross_class_collision,
                    "eye_sift": sift,
                },
                "causal_cream_state": state.to_dict(),
            }
            output_frames.append(record)
        transitions = [
            {"frame_index": state.frame_index, "t": state.t}
            for state in states
            if state.transition_now
        ]
        diagnostics[str(episode_id)] = {
            "frames": len(states),
            "initial_cream_track_id": initial_track_id,
            "sift_valid_frames": sum(int(row["valid"]) for row in sift_rows),
            "near_rim_frames": sum(int(item.near_basket_rim) for item in evidence),
            "cross_class_collision_frames": sum(
                int(item.cross_class_collision) for item in evidence
            ),
            "cream_transition_frames": transitions,
            "terminal_cream_achieved": states[-1].cream_achieved,
            "reliable_frames": sum(int(state.observation_reliable) for state in states),
        }
        if args.prefix_check_frames:
            count = min(args.prefix_check_frames, len(evidence))
            prefix_states = run_causal_cream_state(evidence[:count], config)
            full_view = [state.to_dict() for state in states[:count]]
            prefix_view = [state.to_dict() for state in prefix_states]
            prefix_checks[str(episode_id)] = {
                "frames": count,
                "full_prefix_sha256": canonical_sha256(full_view),
                "prefix_only_sha256": canonical_sha256(prefix_view),
                "exact": full_view == prefix_view,
            }

    source_path = Path(__file__).resolve()
    state_source = (source_path.parent.parent / "lcwm" / "v247_cream_state.py").resolve()
    result = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "development_probe_complete",
        "claim_scope": (
            "post-hoc disclosed seed-8700 multiview cream diagnostic; "
            "not independent validation and not a Gate-0 pass"
        ),
        "privileged_inputs_read": False,
        "episode_ids": episode_ids,
        "inputs": {"manifests": manifest_sources, "sam3_artifacts": sam3_sources},
        "template": locator.provenance,
        "frozen_rules": {
            "sift": {
                "upscale": SIFT_UPSCALE,
                "contrast_threshold": SIFT_CONTRAST_THRESHOLD,
                "lowe_ratio": SIFT_LOWE_RATIO,
                "cluster_eps_px": SIFT_CLUSTER_EPS_PX,
                "min_cluster_matches": SIFT_MIN_CLUSTER_MATCHES,
                "ransac_reprojection_px": SIFT_RANSAC_REPROJECTION_THRESHOLD_PX,
                "ransac_min_inliers": SIFT_RANSAC_MIN_INLIERS,
                "ransac_min_inlier_ratio": SIFT_RANSAC_MIN_INLIER_RATIO,
                "scale_range": list(SIFT_SCALE_RANGE),
            },
            "rim_margin_px": RIM_MARGIN_PX,
            "rim_max_depth_fraction": RIM_MAX_DEPTH_FRACTION,
            "cross_class_box_ios_gte": CROSS_CLASS_BOX_IOS_GTE,
            "state": config.to_dict(),
        },
        "episodes": diagnostics,
        "prefix_checks": prefix_checks,
        "frames_jsonl": "frames.jsonl",
        "producer_source_sha256": sha256_file(source_path),
        "state_source_sha256": sha256_file(state_source),
    }
    reject_semantic_fields(result)
    staging = Path(str(output_dir) + f".tmp.{os.getpid()}")
    require(not staging.exists(), f"staging path exists: {staging}")
    staging.mkdir(parents=True)
    try:
        frames_path = staging / "frames.jsonl"
        with frames_path.open("w", encoding="utf-8") as handle:
            for record in output_frames:
                handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        shutil.copy2(source_path, staging / "PRODUCER.py")
        shutil.copy2(state_source, staging / "CREAM_STATE.py")
        atomic_write_json(staging / "RESULT.json", result)
        complete = {
            "schema": COMPLETE_SCHEMA,
            "status": "complete",
            "result_sha256": sha256_file(staging / "RESULT.json"),
            "frames_sha256": sha256_file(frames_path),
            "producer_sha256": sha256_file(staging / "PRODUCER.py"),
            "state_source_sha256": sha256_file(staging / "CREAM_STATE.py"),
        }
        atomic_write_json(staging / "COMPLETE.json", complete)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.rename(staging, output_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    print(json.dumps({"output": str(output_dir), "episodes": diagnostics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
