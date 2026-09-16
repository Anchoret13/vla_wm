#!/usr/bin/env python3
"""Run a sealed, RGB-only SAM3 open-vocabulary development probe.

This probe deliberately accepts one image path and fixed text concepts only.  It
does not open a latent tape, rollout summary, simulator state, events, or success
labels.  The local mirror contains a combined detector/tracker checkpoint; the
Transformers image model consumes the strictly matched ``detector_model.``
subtree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from transformers import (
    CLIPTokenizer,
    Sam3Config,
    Sam3ImageProcessor,
    Sam3Model,
    Sam3Processor,
)


CONCEPTS = {
    "basket_1": "white woven basket",
    "alphabet_soup_1": "alphabet soup can",
    "tomato_sauce_1": "tomato sauce can",
    "cream_cheese_1": "cream cheese box",
}
SCHEMA = "v245_sam3_open_vocab_probe_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--clip-tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument("--score-threshold", type=float, default=0.10)
    parser.add_argument("--mask-threshold", type=float, default=0.50)
    return parser.parse_args()


def load_detector(checkpoint: Path) -> Sam3Model:
    require(checkpoint.is_file(), f"missing SAM3 checkpoint: {checkpoint}")
    model = Sam3Model(Sam3Config())
    detector_state = {}
    prefix = "detector_model."
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.startswith(prefix):
                detector_state[key[len(prefix) :]] = handle.get_tensor(key)
    expected = model.state_dict()
    require(
        set(detector_state) == set(expected),
        "SAM3 detector subtree does not exactly match Transformers Sam3Model",
    )
    mismatched = [
        key
        for key, value in detector_state.items()
        if tuple(value.shape) != tuple(expected[key].shape)
    ]
    require(not mismatched, f"SAM3 detector shape mismatch: {mismatched[:8]}")
    incompatible = model.load_state_dict(detector_state, strict=True)
    require(not incompatible.missing_keys, "SAM3 detector has missing keys")
    require(not incompatible.unexpected_keys, "SAM3 detector has unexpected keys")
    return model


def mask_record(mask: torch.Tensor, score: float, box: list[float]) -> dict[str, Any]:
    binary = mask.detach().to(device="cpu", dtype=torch.bool).numpy()
    yy, xx = np.nonzero(binary)
    centroid = None
    if len(xx):
        centroid = [float(xx.mean()), float(yy.mean())]
    return {
        "score": float(score),
        "box_xyxy": [float(value) for value in box],
        "mask_area_px": int(binary.sum()),
        "mask_centroid_xy": centroid,
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.rename(temporary, path)


def main() -> int:
    args = parse_args()
    image_path = args.image.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    tokenizer_path = args.clip_tokenizer.expanduser().resolve()
    output = args.output.expanduser().resolve()
    mask_dir = args.mask_dir.expanduser().resolve()
    require(image_path.is_file(), f"missing image: {image_path}")
    require(tokenizer_path.is_dir(), f"missing tokenizer: {tokenizer_path}")
    require(not output.exists(), f"output exists: {output}")
    require(not mask_dir.exists(), f"mask directory exists: {mask_dir}")
    require(args.device != "cuda" or torch.cuda.is_available(), "CUDA unavailable")

    dtype = getattr(torch, args.dtype)
    model = load_detector(checkpoint).to(device=args.device, dtype=dtype).eval()
    processor = Sam3Processor(
        image_processor=Sam3ImageProcessor(),
        tokenizer=CLIPTokenizer.from_pretrained(
            tokenizer_path, local_files_only=True
        ),
    )
    image = Image.open(image_path).convert("RGB")
    image_inputs = processor(images=image, return_tensors="pt")
    pixel_values = image_inputs["pixel_values"].to(device=args.device, dtype=dtype)

    predictions: dict[str, Any] = {}
    selected_masks: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        vision = model.get_vision_features(pixel_values)
        for object_name, prompt in CONCEPTS.items():
            text_inputs = processor(text=prompt, return_tensors="pt")
            model_inputs = {
                key: value.to(args.device)
                for key, value in text_inputs.items()
                if key in {"input_ids", "attention_mask"}
            }
            outputs = model(vision_embeds=vision, **model_inputs)
            processed = processor.post_process_instance_segmentation(
                outputs,
                threshold=args.score_threshold,
                mask_threshold=args.mask_threshold,
                target_sizes=[(image.height, image.width)],
            )[0]
            instances = [
                mask_record(mask, score, box)
                for score, box, mask in zip(
                    processed["scores"].detach().float().cpu().tolist(),
                    processed["boxes"].detach().float().cpu().tolist(),
                    processed["masks"],
                    strict=True,
                )
            ]
            order = sorted(
                range(len(instances)),
                key=lambda index: instances[index]["score"],
                reverse=True,
            )
            instances = [instances[index] for index in order]
            predictions[object_name] = {
                "text_prompt": prompt,
                "presence_logit": float(
                    outputs.presence_logits.detach().float().cpu().reshape(-1)[0]
                ),
                "instances": instances,
            }
            if order:
                selected_masks[object_name] = (
                    processed["masks"][order[0]]
                    .detach()
                    .to(device="cpu", dtype=torch.bool)
                    .numpy()
                )

    mask_dir.mkdir(parents=True, exist_ok=False)
    mask_files = {}
    for object_name, binary in selected_masks.items():
        mask_path = mask_dir / f"{object_name}.png"
        Image.fromarray(binary.astype(np.uint8) * 255, mode="L").save(mask_path)
        mask_files[object_name] = {
            "path": str(mask_path),
            "sha256": sha256_file(mask_path),
        }

    payload = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "rgb_only_development_probe_complete",
        "privileged_inputs_read": False,
        "image": {
            "path": str(image_path),
            "sha256": sha256_file(image_path),
            "size_wh": [image.width, image.height],
        },
        "model": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_subtree": "detector_model.",
            "config": Sam3Config().to_dict(),
            "clip_tokenizer": str(tokenizer_path),
            "device": args.device,
            "dtype": args.dtype,
        },
        "thresholds": {
            "score": args.score_threshold,
            "mask": args.mask_threshold,
        },
        "predictions": predictions,
        "selected_mask_files": mask_files,
        "producer_code_sha256": sha256_file(Path(__file__).resolve()),
    }
    atomic_json(output, payload)
    print(json.dumps({"output": str(output), "predictions": predictions}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
