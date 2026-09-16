#!/usr/bin/env python3
"""Run a sealed RGB-only, causal SAM3 video progress probe.

The primary output is deliberately identity-free for the two visually similar
cans: one open-vocabulary prompt proposes can instances, a fixed top-two rule
counts their centroids inside the tracked basket, and a sticky monotone state
survives later occlusion.  Cream cheese is tracked as a separate concept.  The
script never opens rollout summaries, latent tapes, simulator state, events, or
success labels.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import os
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from transformers import (
    CLIPTokenizer,
    Sam2VideoVideoProcessor,
    Sam3ImageProcessor,
    Sam3VideoConfig,
    Sam3VideoModel,
    Sam3VideoProcessor,
)


SCHEMA = "v245_sam3_video_progress_probe_v1"
FRAME_SCHEMA = "v245_sam3_video_progress_frame_v1"
COMPLETE_SCHEMA = "v245_sam3_video_progress_complete_v1"
PROMPTS = {
    "basket": "white woven basket",
    "can": "tomato sauce can",
    "cream": "cream cheese box",
}

# Frozen before opening a hard-case panel.  Detection/new-track thresholds match
# the already-open ep0 SAM3 image smoke; all remaining geometry bounds are the
# existing v244 RGB-only development bounds.
DETECTION_SCORE_THRESHOLD = 0.10
NEW_TRACK_SCORE_THRESHOLD = 0.10
DETECTOR_NMS_IOU_THRESHOLD = 0.10
DOWNSTREAM_DUPLICATE_MASK_IOU_GTE = 0.50
DOWNSTREAM_DUPLICATE_BOX_IOU_GTE = 0.70
MAX_CAN_INSTANCES = 2
MASK_AREA_BOUNDS = {
    "basket": (3000, 15000),
    "can": (64, 5000),
    "cream": (64, 5000),
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
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def tree_sha256(root: Path) -> str:
    records = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        records.append({"relative_path": str(path.relative_to(root)), "sha256": sha256_file(path)})
    return canonical_sha256(records)


def reject_semantic_fields(value: Any, where: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            require(normalized not in FORBIDDEN_KEYS, f"{where}.{key}: forbidden semantic field")
            reject_semantic_fields(child, f"{where}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_semantic_fields(child, f"{where}[{index}]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--episode-id", action="append", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--clip-tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--state-device", default="cpu")
    parser.add_argument(
        "--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument(
        "--prefix-check-frames",
        type=int,
        default=0,
        help="Re-run this many leading frames in a fresh session and compare outputs.",
    )
    return parser.parse_args()


def load_rows(
    manifest_paths: Sequence[Path], episode_ids: Sequence[int]
) -> tuple[dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
    wanted = set(episode_ids)
    require(len(wanted) == len(episode_ids), "duplicate --episode-id")
    rows_by_episode: dict[int, list[dict[str, Any]]] = {episode_id: [] for episode_id in wanted}
    manifest_records = []
    for raw_path in manifest_paths:
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
                require(row.get("schema") == "v242_rgb_replay_v1", "unexpected manifest schema")
                require(row.get("task") == "chain3_lr2", "only chain3_lr2 development is allowed")
                require(
                    int(row.get("env_seed", -1)) == 8700 + episode_id,
                    f"ep{episode_id}: only the seed-8700 development panel is allowed",
                )
                image_record = row.get("images", {}).get("agentview")
                require(isinstance(image_record, dict), f"ep{episode_id}: missing agentview")
                image_path = (path.parent / image_record["path"]).resolve()
                require(image_path.is_file(), f"missing image: {image_path}")
                rows_by_episode[episode_id].append(
                    {
                        "episode_id": episode_id,
                        "env_seed": int(row["env_seed"]),
                        "t": int(row["t"]),
                        "row_index": int(row["row_index"]),
                        "frame_id": str(row["frame_id"]),
                        "image_path": str(image_path),
                        "image_sha256": str(image_record["sha256"]),
                        "height": int(image_record["height"]),
                        "width": int(image_record["width"]),
                        "manifest_path": str(path),
                    }
                )
                selected += 1
        manifest_records.append(
            {"path": str(path), "sha256": sha256_file(path), "selected_rows": selected}
        )
    for episode_id, rows in rows_by_episode.items():
        require(rows, f"ep{episode_id}: no rows in supplied manifests")
        rows.sort(key=lambda row: (row["t"], row["row_index"]))
        require(len({row["t"] for row in rows}) == len(rows), f"ep{episode_id}: duplicate t")
        require(rows[0]["t"] == 0, f"ep{episode_id}: first RGB frame is not t0")
        require(
            all(left["t"] < right["t"] for left, right in zip(rows, rows[1:])),
            f"ep{episode_id}: non-monotone time",
        )
    return rows_by_episode, manifest_records


def load_model(checkpoint: Path, device: str, dtype: torch.dtype) -> Sam3VideoModel:
    require(checkpoint.is_file(), f"missing SAM3 checkpoint: {checkpoint}")
    config = Sam3VideoConfig()
    model = Sam3VideoModel(config)
    state = {}
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            state[key] = handle.get_tensor(key)
    expected = model.state_dict()
    require(set(state) == set(expected), "SAM3 video checkpoint keys do not exactly match")
    mismatched = [key for key in state if tuple(state[key].shape) != tuple(expected[key].shape)]
    require(not mismatched, f"SAM3 video checkpoint shape mismatch: {mismatched[:8]}")
    incompatible = model.load_state_dict(state, strict=True)
    require(not incompatible.missing_keys, "SAM3 video checkpoint missing keys")
    require(not incompatible.unexpected_keys, "SAM3 video checkpoint unexpected keys")
    del state
    model.score_threshold_detection = DETECTION_SCORE_THRESHOLD
    model.new_det_thresh = NEW_TRACK_SCORE_THRESHOLD
    model.det_nms_thresh = DETECTOR_NMS_IOU_THRESHOLD
    return model.to(device=device, dtype=dtype).eval()


def build_processor(tokenizer_path: Path) -> Sam3VideoProcessor:
    image_processor = Sam3ImageProcessor()
    # Streaming uses image_processor.  Keep the otherwise-unused video processor
    # numerically aligned so an accidental async call cannot silently preprocess
    # frames differently.
    video_processor = Sam2VideoVideoProcessor(
        size={"height": 1008, "width": 1008},
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
    )
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    return Sam3VideoProcessor(
        image_processor=image_processor,
        video_processor=video_processor,
        tokenizer=tokenizer,
    )


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 0.0


def box_iou(left: Sequence[float], right: Sequence[float]) -> float:
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
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def mask_geometry(mask: np.ndarray) -> tuple[int, list[float] | None, list[float] | None]:
    yy, xx = np.nonzero(mask)
    if not len(xx):
        return 0, None, None
    bbox = [float(xx.min()), float(yy.min()), float(xx.max() + 1), float(yy.max() + 1)]
    centroid = [float(xx.mean()), float(yy.mean())]
    return int(len(xx)), bbox, centroid


def inside_box(point_xy: Sequence[float], box_xyxy: Sequence[float]) -> bool:
    return bool(
        float(box_xyxy[0]) <= float(point_xy[0]) <= float(box_xyxy[2])
        and float(box_xyxy[1]) <= float(point_xy[1]) <= float(box_xyxy[3])
    )


def instance_record(
    object_id: int,
    prompt_name: str,
    score: float,
    tracker_score: float | None,
    mask: np.ndarray,
) -> dict[str, Any]:
    area, bbox, centroid = mask_geometry(mask)
    lower, upper = MASK_AREA_BOUNDS[prompt_name]
    return {
        "object_id": int(object_id),
        "prompt_name": prompt_name,
        "score": float(score),
        "tracker_score": float(tracker_score) if tracker_score is not None else None,
        "mask_area_px": area,
        "mask_bbox_xyxy": bbox,
        "mask_centroid_xy": centroid,
        "mask_sha256": hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest(),
        "area_valid": lower <= area <= upper,
    }


def duplicate_audit(instances: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for left, right in itertools.combinations(instances, 2):
        mask_overlap = mask_iou(left["_mask"], right["_mask"])
        bbox_overlap = (
            box_iou(left["mask_bbox_xyxy"], right["mask_bbox_xyxy"])
            if left["mask_bbox_xyxy"] is not None and right["mask_bbox_xyxy"] is not None
            else 0.0
        )
        flagged = (
            mask_overlap >= DOWNSTREAM_DUPLICATE_MASK_IOU_GTE
            or bbox_overlap >= DOWNSTREAM_DUPLICATE_BOX_IOU_GTE
        )
        records.append(
            {
                "object_ids": [left["object_id"], right["object_id"]],
                "mask_iou": mask_overlap,
                "bbox_iou": bbox_overlap,
                "flagged_duplicate": flagged,
            }
        )
    return records


def select_unique(
    instances: Sequence[dict[str, Any]], limit: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(instances, key=lambda item: (-item["score"], item["object_id"]))
    selected: list[dict[str, Any]] = []
    rejected = []
    for candidate in ordered:
        reasons = []
        if not candidate["area_valid"]:
            reasons.append("area_out_of_bounds")
        for kept in selected:
            mask_overlap = mask_iou(candidate["_mask"], kept["_mask"])
            bbox_overlap = (
                box_iou(candidate["mask_bbox_xyxy"], kept["mask_bbox_xyxy"])
                if candidate["mask_bbox_xyxy"] is not None
                and kept["mask_bbox_xyxy"] is not None
                else 0.0
            )
            if (
                mask_overlap >= DOWNSTREAM_DUPLICATE_MASK_IOU_GTE
                or bbox_overlap >= DOWNSTREAM_DUPLICATE_BOX_IOU_GTE
            ):
                reasons.append(f"duplicate_of_object_{kept['object_id']}")
                break
        if reasons:
            rejected.append({"object_id": candidate["object_id"], "reasons": reasons})
            continue
        if len(selected) < limit:
            selected.append(candidate)
        else:
            rejected.append({"object_id": candidate["object_id"], "reasons": ["below_top_k"]})
    return selected, rejected


def public_instance(instance: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in instance.items() if key != "_mask"}


def session_for_episode(
    processor: Sam3VideoProcessor,
    device: str,
    state_device: str,
    dtype: torch.dtype,
) -> Any:
    session = processor.init_video_session(
        video=None,
        inference_device=device,
        inference_state_device=state_device,
        processing_device="cpu",
        video_storage_device="cpu",
        max_vision_features_cache_size=1,
        dtype=dtype,
    )
    processor.add_text_prompt(session, list(PROMPTS.values()))
    return session


def process_rows(
    model: Sam3VideoModel,
    processor: Sam3VideoProcessor,
    rows: Sequence[dict[str, Any]],
    *,
    device: str,
    state_device: str,
    dtype: torch.dtype,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    session = session_for_episode(processor, device, state_device, dtype)
    prompt_to_name = {text: name for name, text in PROMPTS.items()}
    sticky_can_count = 0
    sticky_cream_inside = False
    frame_records = []
    all_seen_ids: dict[str, set[int]] = {name: set() for name in PROMPTS}
    selected_can_ids_at_t0: list[int] = []
    can_id_set_changes = 0
    previous_can_ids: tuple[int, ...] | None = None
    carry_frames = {"can": 0, "cream": 0}
    progress_increase_frames = []

    for frame_index, row in enumerate(rows):
        sealed_bytes = row.get("sealed_image_bytes")
        if row.get("formal_same_buffer") is True:
            require(isinstance(sealed_bytes, bytes), "formal image bytes missing")
            require(
                hashlib.sha256(sealed_bytes).hexdigest() == row["image_sha256"],
                "formal image buffer seal mismatch",
            )
            image = Image.open(io.BytesIO(sealed_bytes)).convert("RGB")
        else:
            image_path = Path(row["image_path"])
            require(
                sha256_file(image_path) == row["image_sha256"],
                f"image seal mismatch: {image_path}",
            )
            image = Image.open(image_path).convert("RGB")
        require(image.size == (row["width"], row["height"]), "manifest/image size mismatch")
        inputs = processor(images=image, return_tensors="pt")
        frame = inputs["pixel_values"]
        with torch.inference_mode():
            outputs = model(session, frame=frame, frame_idx=frame_index)
            processed = processor.postprocess_outputs(
                session,
                outputs,
                original_sizes=[[row["height"], row["width"]]],
            )

        object_ids = processed["object_ids"].tolist()
        scores = processed["scores"].detach().float().cpu().tolist()
        masks = processed["masks"].detach().to(device="cpu", dtype=torch.bool).numpy()
        tracker_scores = outputs.obj_id_to_tracker_score or {}
        instances_by_prompt: dict[str, list[dict[str, Any]]] = {name: [] for name in PROMPTS}
        for object_id, score, mask in zip(object_ids, scores, masks, strict=True):
            prompt_id = session.obj_id_to_prompt_id[int(object_id)]
            prompt_text = session.prompts[prompt_id]
            prompt_name = prompt_to_name[prompt_text]
            tracker_score = tracker_scores.get(int(object_id))
            record = instance_record(
                int(object_id), prompt_name, float(score), tracker_score, mask
            )
            record["_mask"] = mask
            instances_by_prompt[prompt_name].append(record)
            all_seen_ids[prompt_name].add(int(object_id))

        audits = {
            name: duplicate_audit(instances) for name, instances in instances_by_prompt.items()
        }
        basket_selected, basket_rejected = select_unique(instances_by_prompt["basket"], 1)
        can_selected, can_rejected = select_unique(
            instances_by_prompt["can"], MAX_CAN_INSTANCES
        )
        cream_selected, cream_rejected = select_unique(instances_by_prompt["cream"], 1)
        basket = basket_selected[0] if basket_selected else None

        raw_can_inside_count = None
        can_inside_flags = None
        if basket is not None and len(can_selected) == MAX_CAN_INSTANCES:
            can_inside_flags = [
                inside_box(candidate["mask_centroid_xy"], basket["mask_bbox_xyxy"])
                for candidate in can_selected
            ]
            raw_can_inside_count = int(sum(can_inside_flags))
        previous_progress = sticky_can_count + int(sticky_cream_inside)
        if raw_can_inside_count is None:
            carry_frames["can"] += int(sticky_can_count > 0)
        else:
            sticky_can_count = max(sticky_can_count, raw_can_inside_count)

        raw_cream_inside = None
        if basket is not None and cream_selected:
            raw_cream_inside = inside_box(
                cream_selected[0]["mask_centroid_xy"], basket["mask_bbox_xyxy"]
            )
        if raw_cream_inside is None:
            carry_frames["cream"] += int(sticky_cream_inside)
        elif raw_cream_inside:
            sticky_cream_inside = True
        progress_scalar = sticky_can_count + int(sticky_cream_inside)
        if progress_scalar > previous_progress:
            progress_increase_frames.append(
                {"frame_index": frame_index, "t": row["t"], "to": progress_scalar}
            )

        can_ids = tuple(sorted(candidate["object_id"] for candidate in can_selected))
        if frame_index == 0:
            selected_can_ids_at_t0 = list(can_ids)
        if previous_can_ids is not None and can_ids != previous_can_ids:
            can_id_set_changes += 1
        previous_can_ids = can_ids

        frame_record = {
            "schema": FRAME_SCHEMA,
            "episode_id": row["episode_id"],
            "env_seed": row["env_seed"],
            "frame_index": frame_index,
            "t": row["t"],
            "frame_id": row["frame_id"],
            "image_sha256": row["image_sha256"],
            "instances": {
                name: [public_instance(instance) for instance in instances]
                for name, instances in instances_by_prompt.items()
            },
            "duplicate_audit": audits,
            "selection": {
                "basket": [public_instance(instance) for instance in basket_selected],
                "can": [public_instance(instance) for instance in can_selected],
                "cream": [public_instance(instance) for instance in cream_selected],
                "rejected": {
                    "basket": basket_rejected,
                    "can": can_rejected,
                    "cream": cream_rejected,
                },
            },
            "raw_geometry": {
                "can_inside_flags": can_inside_flags,
                "can_inside_count": raw_can_inside_count,
                "cream_inside": raw_cream_inside,
            },
            "causal_progress": {
                "sticky_can_inside_count": sticky_can_count,
                "sticky_cream_inside": sticky_cream_inside,
                "scalar_0_to_3": progress_scalar,
                "used_can_carry": raw_can_inside_count is None and sticky_can_count > 0,
                "used_cream_carry": raw_cream_inside is None and sticky_cream_inside,
            },
        }
        frame_records.append(frame_record)

    diagnostics = {
        "episode_id": rows[0]["episode_id"],
        "frame_count": len(rows),
        "terminal_progress_scalar": (
            frame_records[-1]["causal_progress"]["scalar_0_to_3"]
        ),
        "terminal_sticky_can_count": sticky_can_count,
        "terminal_sticky_cream_inside": sticky_cream_inside,
        "progress_increase_frames": progress_increase_frames,
        "unique_track_ids_seen": {name: sorted(ids) for name, ids in all_seen_ids.items()},
        "selected_can_ids_at_t0": selected_can_ids_at_t0,
        "selected_can_id_set_changes": can_id_set_changes,
        "carry_frame_counts": carry_frames,
        "flagged_duplicate_pairs": {
            name: sum(
                int(pair["flagged_duplicate"])
                for frame in frame_records
                for pair in frame["duplicate_audit"][name]
            )
            for name in PROMPTS
        },
    }
    return frame_records, diagnostics


def prefix_comparison_view(frame: Mapping[str, Any]) -> dict[str, Any]:
    def rounded(value: Any) -> Any:
        if isinstance(value, float):
            return round(value, 6)
        if isinstance(value, dict):
            return {key: rounded(child) for key, child in value.items()}
        if isinstance(value, list):
            return [rounded(child) for child in value]
        return value

    return rounded(dict(frame))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    manifests = [path.expanduser().resolve() for path in args.manifest]
    checkpoint = args.checkpoint.expanduser().resolve()
    tokenizer_path = args.clip_tokenizer.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    require(not output_dir.exists(), f"refusing to overwrite {output_dir}")
    require(tokenizer_path.is_dir(), f"missing tokenizer directory: {tokenizer_path}")
    require(args.prefix_check_frames >= 0, "--prefix-check-frames must be nonnegative")
    require(args.device != "cuda" or torch.cuda.is_available(), "CUDA unavailable")
    dtype = getattr(torch, args.dtype)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32 = False
    rows_by_episode, manifest_records = load_rows(manifests, args.episode_id)

    model = load_model(checkpoint, args.device, dtype)
    processor = build_processor(tokenizer_path)
    all_frames = []
    episodes = {}
    prefix_checks = {}
    for episode_id in args.episode_id:
        rows = rows_by_episode[episode_id]
        frames, diagnostics = process_rows(
            model,
            processor,
            rows,
            device=args.device,
            state_device=args.state_device,
            dtype=dtype,
        )
        all_frames.extend(frames)
        episodes[str(episode_id)] = diagnostics
        if args.prefix_check_frames:
            count = min(args.prefix_check_frames, len(rows))
            repeated, _ = process_rows(
                model,
                processor,
                rows[:count],
                device=args.device,
                state_device=args.state_device,
                dtype=dtype,
            )
            reference_view = [prefix_comparison_view(frame) for frame in frames[:count]]
            repeated_view = [prefix_comparison_view(frame) for frame in repeated]
            prefix_checks[str(episode_id)] = {
                "frames": count,
                "reference_sha256": canonical_sha256(reference_view),
                "fresh_session_sha256": canonical_sha256(repeated_view),
                "exact_after_rounding_1e_6": reference_view == repeated_view,
                "interpretation": (
                    "A fresh streaming session consumed only the same RGB prefix; "
                    "no later image was decoded before its earlier output was recorded."
                ),
            }

    source_path = Path(__file__).resolve()
    source_sha = sha256_file(source_path)
    result = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "rgb_only_streaming_development_probe_complete",
        "claim_scope": (
            "identity-free task-progress scalar on seed-8700 development RGB; "
            "not a per-can identity gate and not independent validation"
        ),
        "privileged_inputs_read": False,
        "data_boundary": {
            "decoded_inputs": "agentview RGB only, one current frame at a time",
            "manifest_metadata": "episode id, seed, t, image path, image hash, dimensions",
            "explicitly_unopened": [
                "rollout summary contents",
                "latent tape contents",
                "simulator state",
                "task events",
                "success labels",
                "seed-8000/8100 and chain2b RGB",
            ],
        },
        "inputs": manifest_records,
        "episode_ids": list(args.episode_id),
        "prompts": PROMPTS,
        "frozen_rules": {
            "detector_score_threshold": DETECTION_SCORE_THRESHOLD,
            "new_track_score_threshold": NEW_TRACK_SCORE_THRESHOLD,
            "detector_mask_nms_iou_threshold": DETECTOR_NMS_IOU_THRESHOLD,
            "downstream_duplicate_mask_iou_gte": DOWNSTREAM_DUPLICATE_MASK_IOU_GTE,
            "downstream_duplicate_box_iou_gte": DOWNSTREAM_DUPLICATE_BOX_IOU_GTE,
            "mask_area_bounds_inclusive": MASK_AREA_BOUNDS,
            "can_selection": "top two valid nonduplicate instances by score then object id",
            "relation": "instance mask centroid inside selected basket mask bounding box",
            "state_update": (
                "can count is max(previous, current) only when basket and two cans are valid; "
                "cream is sticky after a current-frame inside observation"
            ),
            "scalar": "sticky can count (0..2) + sticky cream indicator (0..1)",
        },
        "model": {
            "family": "Transformers Sam3VideoModel",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "state_dict_exact_key_and_shape_match": True,
            "config": Sam3VideoConfig().to_dict(),
            "device": args.device,
            "state_device": args.state_device,
            "dtype": args.dtype,
            "clip_tokenizer": str(tokenizer_path),
            "clip_tokenizer_tree_sha256": tree_sha256(tokenizer_path),
        },
        "episodes": episodes,
        "prefix_checks": prefix_checks,
        "frames_jsonl": "frames.jsonl",
        "producer_source": "producer_source.py",
        "producer_source_sha256": source_sha,
    }

    staging = Path(str(output_dir) + f".tmp.{os.getpid()}")
    require(not staging.exists(), f"staging path exists: {staging}")
    staging.mkdir(parents=True)
    try:
        frames_path = staging / "frames.jsonl"
        with frames_path.open("w", encoding="utf-8") as handle:
            for frame in all_frames:
                handle.write(json.dumps(frame, sort_keys=True, allow_nan=False) + "\n")
        shutil.copy2(source_path, staging / "producer_source.py")
        write_json(staging / "result.json", result)
        complete = {
            "schema": COMPLETE_SCHEMA,
            "status": "complete",
            "result_sha256": sha256_file(staging / "result.json"),
            "frames_sha256": sha256_file(frames_path),
            "producer_source_sha256": sha256_file(staging / "producer_source.py"),
        }
        write_json(staging / "COMPLETE.json", complete)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.rename(staging, output_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "episodes": episodes,
                "prefix_checks": prefix_checks,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
