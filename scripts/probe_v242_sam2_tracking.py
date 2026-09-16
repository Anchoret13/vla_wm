#!/usr/bin/env python3
"""Calibration-only SAM2.1 tracking probe for a v242 RGB replay.

This probe intentionally consumes only the replay JSONL manifest and JPEGs.  It
does not open a latent tape, replay summary, simulator state, event log, success
label, or task definition.  The default first-frame boxes were manually chosen
from ep0 agentview and are therefore *calibration*, not a deployable detector.

The local checkpoint is in the original Meta SAM2 format.  ``get_config`` and
``replace_keys`` below follow Hugging Face's official SAM2 Video conversion
script, and loading is required to pass ``strict=True``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import transformers
from PIL import Image, ImageDraw
from transformers import (
    Sam2HieraDetConfig,
    Sam2ImageProcessor,
    Sam2VideoConfig,
    Sam2VideoMaskDecoderConfig,
    Sam2VideoModel,
    Sam2VideoProcessor,
    Sam2VideoPromptEncoderConfig,
    Sam2VideoVideoProcessor,
    Sam2VisionConfig,
)


SCHEMA = "v242_sam2_calibration_probe_v1"
OBJECTS = {
    # xyxy in the 360x360 ep0/t0 agentview JPEG.  These are human calibration
    # inputs for this one proof only, not outputs of an automatic detector.
    1: {"name": "basket", "box": [0.0, 156.0, 83.0, 284.0], "color": [255, 255, 0]},
    2: {"name": "left_cylindrical_can", "box": [123.0, 176.0, 165.0, 235.0], "color": [0, 255, 0]},
    3: {
        "name": "right_cylindrical_can",
        "box": [224.0, 176.0, 268.0, 235.0],
        "color": [0, 160, 255],
    },
    4: {"name": "cream_cheese_box", "box": [283.0, 254.0, 321.0, 302.0], "color": [255, 0, 255]},
}


KEYS_TO_MODIFY_MAPPING = {
    "iou_prediction_head.layers.0": "iou_prediction_head.proj_in",
    "iou_prediction_head.layers.1": "iou_prediction_head.layers.0",
    "iou_prediction_head.layers.2": "iou_prediction_head.proj_out",
    "mask_decoder.output_upscaling.0": "mask_decoder.upscale_conv1",
    "mask_decoder.output_upscaling.1": "mask_decoder.upscale_layer_norm",
    "mask_decoder.output_upscaling.3": "mask_decoder.upscale_conv2",
    "mask_downscaling.0": "mask_embed.conv1",
    "mask_downscaling.1": "mask_embed.layer_norm1",
    "mask_downscaling.3": "mask_embed.conv2",
    "mask_downscaling.4": "mask_embed.layer_norm2",
    "mask_downscaling.6": "mask_embed.conv3",
    "dwconv": "depthwise_conv",
    "pwconv": "pointwise_conv",
    "fuser": "memory_fuser",
    "point_embeddings": "point_embed",
    "pe_layer.positional_encoding_gaussian_matrix": "shared_embedding.positional_embedding",
    "obj_ptr_tpos_proj": "temporal_positional_encoding_projection_layer",
    "no_obj_embed_spatial": "occlusion_spatial_embedding_parameter",
    "sam_prompt_encoder": "prompt_encoder",
    "sam_mask_decoder": "mask_decoder",
    "maskmem_tpos_enc": "memory_temporal_positional_encoding",
    "gamma": "scale",
    "image_encoder.neck": "vision_encoder.neck",
    "image_encoder": "vision_encoder.backbone",
    "neck.0": "neck.conv1",
    "neck.1": "neck.layer_norm1",
    "neck.2": "neck.conv2",
    "neck.3": "neck.layer_norm2",
    "pix_feat_proj": "feature_projection",
    "patch_embed.proj": "patch_embed.projection",
    "no_mem_embed": "no_memory_embedding",
    "no_mem_pos_enc": "no_memory_positional_encoding",
    "obj_ptr": "object_pointer",
    ".norm": ".layer_norm",
    "trunk.": "",
    "out_proj": "o_proj",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sam21_hiera_large_config() -> Sam2VideoConfig:
    backbone = Sam2HieraDetConfig(
        hidden_size=144,
        embed_dim_per_stage=[144, 288, 576, 1152],
        num_attention_heads_per_stage=[2, 4, 8, 16],
        blocks_per_stage=[2, 6, 36, 4],
        global_attention_blocks=[23, 33, 43],
        # Transformers 5.5 uses strict dataclass validation and requires a
        # list here (the upstream conversion script currently shows a tuple).
        window_positional_embedding_background_size=[7, 7],
        window_size_per_stage=[8, 4, 16, 8],
    )
    vision = Sam2VisionConfig(
        backbone_config=backbone,
        backbone_channel_list=[1152, 576, 288, 144],
    )
    return Sam2VideoConfig(
        vision_config=vision,
        prompt_encoder_config=Sam2VideoPromptEncoderConfig(),
        mask_decoder_config=Sam2VideoMaskDecoderConfig(),
        enable_temporal_pos_encoding_for_object_pointers=True,
        enable_occlusion_spatial_embedding=True,
    )


def replace_original_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert original SAM2 keys exactly as the official HF converter does."""
    converted: dict[str, torch.Tensor] = {}
    hypernet = re.compile(r".*\.output_hypernetworks_mlps\.(\d+)\.layers\.(\d+).*")
    decoder_mlp = re.compile(r"mask_decoder\.transformer\.layers\.(\d+)\.mlp\.layers\.(\d+).*")
    score_head = re.compile(r"mask_decoder\.pred_obj_score_head\.layers\.(\d+).*")
    vision_mlp = re.compile(r"vision_encoder\.backbone\.blocks\.(\d+)\.mlp\.layers\.(\d+).*")
    vision_neck = re.compile(r"vision_encoder\.neck\.convs\.(\d+)\.conv")
    memory_projection = re.compile(r"memory_encoder\.o_proj.*")
    pointer_projection = re.compile(r"object_pointer_proj\.layers\.(\d+).*")
    mask_downsampler = re.compile(r"memory_encoder\.mask_downsampler\.encoder\.(\d+).*")

    for original_key, value in state_dict.items():
        key = original_key
        for old, new in KEYS_TO_MODIFY_MAPPING.items():
            if old in key:
                key = key.replace(old, new)
        match = vision_mlp.match(key)
        if match:
            layer = int(match.group(2))
            key = key.replace(f"layers.{layer}", "proj_in" if layer == 0 else "proj_out")
        match = decoder_mlp.match(key)
        if match:
            layer = int(match.group(2))
            key = key.replace(
                f"mlp.layers.{layer}", "mlp.proj_in" if layer == 0 else "mlp.proj_out"
            )
        match = score_head.match(key)
        if match:
            layer = int(match.group(1))
            target = {0: "proj_in", 1: "layers.0", 2: "proj_out"}[layer]
            key = key.replace(f"layers.{layer}", target)
        match = hypernet.match(key)
        if match:
            layer = int(match.group(2))
            target = {0: "proj_in", 1: "layers.0", 2: "proj_out"}[layer]
            key = key.replace(f"layers.{layer}", target)
        if vision_neck.match(key):
            key = key.replace(".conv.", ".")
        if memory_projection.match(key):
            key = key.replace(".o_proj.", ".projection.")
        match = pointer_projection.match(key)
        if match:
            layer = int(match.group(1))
            target = {0: "proj_in", 1: "layers.0", 2: "proj_out"}[layer]
            key = key.replace(f"layers.{layer}", target)
        match = mask_downsampler.match(key)
        if match:
            layer = int(match.group(1))
            if layer == 12:
                key = key.replace("encoder.12", "final_conv")
            elif layer % 3 == 0:
                key = key.replace(f"encoder.{layer}", f"layers.{layer // 3}.conv")
            elif layer % 3 == 1:
                key = key.replace(f"encoder.{layer}", f"layers.{layer // 3}.layer_norm")
        if key in converted:
            raise RuntimeError(f"key collision while converting {original_key!r} -> {key!r}")
        converted[key] = value

    converted["shared_image_embedding.positional_embedding"] = converted[
        "prompt_encoder.shared_embedding.positional_embedding"
    ]
    converted["prompt_encoder.point_embed.weight"] = torch.cat(
        [converted.pop(f"prompt_encoder.point_embed.{idx}.weight") for idx in range(4)], dim=0
    )
    return converted


def load_model(
    checkpoint: Path, device: torch.device, dtype: torch.dtype
) -> tuple[Sam2VideoModel, int]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or set(payload) != {"model"}
        or not isinstance(payload["model"], dict)
    ):
        raise ValueError(
            "expected original SAM2 checkpoint with exactly one top-level 'model' state dict"
        )
    original_count = len(payload["model"])
    converted = replace_original_keys(payload["model"])
    model = Sam2VideoModel(sam21_hiera_large_config()).eval()
    incompatible = model.load_state_dict(converted, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict load mismatch: {incompatible}")
    del payload, converted
    return model.to(device=device, dtype=dtype).eval(), original_count


def load_frames(
    manifest: Path, replay_dir: Path, episode_id: int, camera: str
) -> tuple[list[dict], list[Image.Image]]:
    rows: list[dict] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            if record.get("schema") != "v242_rgb_replay_v1":
                raise ValueError(f"line {line_number}: unsupported schema")
            if int(record["episode_id"]) != episode_id:
                continue
            if camera not in record["images"]:
                raise ValueError(f"line {line_number}: missing camera {camera!r}")
            image_record = record["images"][camera]
            image_path = (replay_dir / image_record["path"]).resolve()
            if not image_path.is_relative_to(replay_dir.resolve()):
                raise ValueError(f"line {line_number}: image escapes replay directory")
            if sha256_file(image_path) != image_record["sha256"]:
                raise ValueError(f"line {line_number}: JPEG hash mismatch")
            rows.append(
                {
                    "frame_id": str(record["frame_id"]),
                    "t": int(record["t"]),
                    "image_path": str(image_path),
                    "image_sha256": str(image_record["sha256"]),
                }
            )
    rows.sort(key=lambda row: row["t"])
    if not rows or rows[0]["t"] != 0 or len({row["t"] for row in rows}) != len(rows):
        raise ValueError("episode frames must be nonempty, uniquely timed, and start at t=0")
    frames = [Image.open(row["image_path"]).convert("RGB") for row in rows]
    if len({frame.size for frame in frames}) != 1:
        raise ValueError("all frames must have the same dimensions")
    return rows, frames


def mask_stats(mask: np.ndarray) -> dict:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return {"area_px": 0, "bbox_xyxy": None, "centroid_xy": None}
    return {
        "area_px": int(len(xs)),
        "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
        "centroid_xy": [float(xs.mean()), float(ys.mean())],
    }


def centroid_in_bbox(centroid: list[float] | None, bbox: list[int] | None) -> bool | None:
    if centroid is None or bbox is None:
        return None
    x, y = centroid
    x0, y0, x1, y1 = bbox
    return bool(x0 <= x < x1 and y0 <= y < y1)


def render_overlay(
    frame: Image.Image, masks: dict[int, np.ndarray], stats: dict[int, dict]
) -> Image.Image:
    base = np.asarray(frame).astype(np.float32)
    for obj_id, mask in masks.items():
        color = np.asarray(OBJECTS[obj_id]["color"], dtype=np.float32)
        base[mask] = 0.55 * base[mask] + 0.45 * color
    output = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(output)
    for obj_id, obj_stats in stats.items():
        bbox = obj_stats["bbox_xyxy"]
        if bbox is None:
            continue
        color = tuple(OBJECTS[obj_id]["color"])
        draw.rectangle(bbox, outline=color, width=2)
        draw.text((bbox[0] + 2, max(0, bbox[1] - 12)), OBJECTS[obj_id]["name"], fill=color)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/home/stargazer/.cache/sam2/checkpoints/sam2.1_hiera_large.pt"),
    )
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--camera", choices=["agentview", "eye_in_hand"], default="agentview")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    replay_dir = args.replay_dir.resolve()
    manifest = replay_dir / "manifest.jsonl"
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    rows, frames = load_frames(manifest, replay_dir, args.episode_id, args.camera)
    width, height = frames[0].size
    if (width, height) != (360, 360):
        raise ValueError("default calibration boxes are valid only for 360x360 frames")

    model, checkpoint_tensor_count = load_model(checkpoint, device, dtype)
    processor = Sam2VideoProcessor(
        image_processor=Sam2ImageProcessor(),
        video_processor=Sam2VideoVideoProcessor(),
    )
    session = processor.init_video_session(
        video=frames,
        inference_device=device,
        inference_state_device=device,
        processing_device=device,
        video_storage_device="cpu",
        dtype=dtype,
    )
    calibration_obj_ids = tuple(OBJECTS)
    input_boxes = [[OBJECTS[obj_id]["box"] for obj_id in calibration_obj_ids]]
    processor.add_inputs_to_inference_session(
        inference_session=session,
        frame_idx=0,
        # The processor currently mutates list inputs while registering them;
        # pass a disposable list and retain the immutable calibration IDs.
        obj_ids=list(calibration_obj_ids),
        input_boxes=input_boxes,
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    overlays = args.output_dir / "overlays"
    overlays.mkdir()
    frame_results: list[dict] = []
    with torch.inference_mode():
        # Materialize the prompted conditioning-frame output before asking the
        # iterator to infer its propagation start (required by the video API).
        model(inference_session=session, frame_idx=0)
        outputs = model.propagate_in_video_iterator(session, show_progress_bar=True)
        for output in outputs:
            frame_index = int(output.frame_idx)
            processed = processor.post_process_masks(
                [output.pred_masks],
                original_sizes=[[height, width]],
                binarize=True,
            )[0]
            while processed.ndim > 3 and processed.shape[0] == 1:
                processed = processed[0]
            if processed.ndim == 4 and processed.shape[1] == 1:
                processed = processed[:, 0]
            if processed.ndim != 3 or processed.shape[0] != len(output.object_ids):
                raise RuntimeError(
                    "unexpected processed mask shape "
                    f"{tuple(processed.shape)} for ids {output.object_ids}"
                )
            masks: dict[int, np.ndarray] = {}
            stats: dict[int, dict] = {}
            for output_index, obj_id in enumerate(output.object_ids):
                obj_id = int(obj_id)
                mask = processed[output_index].detach().cpu().numpy().astype(bool)
                masks[obj_id] = mask
                stats[obj_id] = mask_stats(mask)
            basket_bbox = stats[1]["bbox_xyxy"]
            for obj_id in (2, 3, 4):
                stats[obj_id]["centroid_in_tracked_basket_bbox"] = centroid_in_bbox(
                    stats[obj_id]["centroid_xy"], basket_bbox
                )
            overlay_path = overlays / (
                f"frame_{frame_index:04d}_t{rows[frame_index]['t']:04d}.jpg"
            )
            render_overlay(frames[frame_index], masks, stats).save(overlay_path, quality=95)
            frame_results.append(
                {
                    "frame_index": frame_index,
                    "frame_id": rows[frame_index]["frame_id"],
                    "t": rows[frame_index]["t"],
                    "object_score_logits": {
                        str(int(obj_id)): float(
                            output.object_score_logits[idx]
                            .detach()
                            .float()
                            .cpu()
                            .reshape(-1)[0]
                        )
                        for idx, obj_id in enumerate(output.object_ids)
                    },
                    "objects": {str(obj_id): stats[obj_id] for obj_id in calibration_obj_ids},
                    "overlay_path": str(overlay_path.resolve()),
                }
            )

    frame_results.sort(key=lambda row: row["frame_index"])
    if [row["frame_index"] for row in frame_results] != list(range(len(rows))):
        raise RuntimeError("SAM2 propagation did not return every input frame exactly once")
    final = frame_results[-1]
    requested_t700 = [row for row in frame_results if row["t"] == 700]
    if len(requested_t700) != 1:
        raise RuntimeError(f"expected exactly one t700 frame, found {len(requested_t700)}")
    requested_t700 = requested_t700[0]
    final_relation = {
        OBJECTS[obj_id]["name"]: final["objects"][str(obj_id)]["centroid_in_tracked_basket_bbox"]
        for obj_id in (2, 3, 4)
    }
    t700_relation = {
        OBJECTS[obj_id]["name"]: requested_t700["objects"][str(obj_id)][
            "centroid_in_tracked_basket_bbox"
        ]
        for obj_id in (2, 3, 4)
    }
    result = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "claim_scope": (
            "ep0 human-t0-box calibration proof only; not an automatic/deployable detector"
        ),
        "data_access": (
            "manifest JSONL image metadata plus JPEGs only; no replay summary/tape/"
            "simulator/event/success/task-definition access"
        ),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "original_tensor_count": checkpoint_tensor_count,
            "format": "original_meta_sam2_model_state",
            "strict_hf_load": True,
        },
        "runtime": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": str(device),
            "dtype": str(dtype),
            "argv": sys.argv,
        },
        "conversion_source": (
            "https://github.com/huggingface/transformers/blob/main/src/transformers/"
            "models/sam2_video/convert_sam2_video_to_hf.py"
        ),
        "inputs": {
            "manifest_path": str(manifest),
            "manifest_sha256": sha256_file(manifest),
            "episode_id": args.episode_id,
            "camera": args.camera,
            "frame_count": len(rows),
            "frame_times": [row["t"] for row in rows],
            "frames": rows,
            "human_t0_calibration": {
                str(obj_id): OBJECTS[obj_id] for obj_id in calibration_obj_ids
            },
        },
        "frames": frame_results,
        "final_t": final["t"],
        "final_centroid_in_tracked_basket_bbox": final_relation,
        "t700_centroid_in_tracked_basket_bbox": t700_relation,
        "t700_relation_matches_calibration_question": bool(
            t700_relation["left_cylindrical_can"]
            and t700_relation["right_cylindrical_can"]
            and not t700_relation["cream_cheese_box"]
        ),
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }
    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "result_path": str(result_path.resolve()),
                "t700_relation": t700_relation,
                "final_relation": final_relation,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
