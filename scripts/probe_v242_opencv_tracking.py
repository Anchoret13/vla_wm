#!/usr/bin/env python3
"""RGB-only OpenCV tracking probe for the v242 chain3 episode-0 replay.

This is deliberately a *proof probe*, not a deployable object detector.  It
starts from three manually calibrated boxes in the first agent-view frame and
asks whether an unweighted OpenCV TrackerMIL instance can preserve identity
through the two pick-and-place motions.  A final-frame DINOv2 global re-ID
diagnostic is optional; it uses an already-local checkpoint and never reads
task outcomes.

The program only opens ``manifest.jsonl`` and the JPEGs referenced by it.  It
does not open ``summary.json``, a tape, BDDL, simulator info, rewards, events,
milestones, stages, or success labels.  The t=700 RGB reference boxes are
evaluation annotations made from pixels only.  They are never given to either
tracker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SCHEMA = "v242_opencv_tracking_probe_v1"

# Pixel-only calibration on ep0/t0/agentview.  Boxes are x, y, width, height.
# Neutral target names intentionally avoid importing object identities from
# task metadata.
INITIAL_BOXES: dict[str, tuple[float, float, float, float]] = {
    "target_a": (122.0, 177.0, 44.0, 68.0),
    "target_b": (225.0, 175.0, 43.0, 68.0),
    "target_c": (280.0, 252.0, 42.0, 51.0),
}

# Pixel-only audit annotations on ep0/t700/agentview.  These are evaluation
# references and are not used during tracking or DINO re-identification.
FINAL_REFERENCE_BOXES: dict[str, tuple[float, float, float, float]] = {
    "target_a": (0.0, 160.0, 39.0, 59.0),
    "target_b": (31.0, 153.0, 29.0, 68.0),
    "target_c": (281.0, 224.0, 39.0, 51.0),
}

# A fixed pixel calibration of the basket interior/mouth in the static
# agent-view camera.  Membership is defined by bbox-centre containment.
BASKET_ROI_XYXY = (0.0, 145.0, 86.0, 234.0)

# Outcome-bearing fields are forbidden even if a future replay manifest adds
# them.  ``summary_sha256`` is harmless provenance and is simply ignored.
FORBIDDEN_FIELDS = {
    "bddl",
    "event",
    "events",
    "info",
    "milestone",
    "milestones",
    "outcome",
    "reward",
    "stage",
    "stages",
    "success",
    "terminated",
    "truncated",
}

COLORS = {
    "target_a": (55, 220, 55),
    "target_b": (255, 210, 40),
    "target_c": (230, 60, 230),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_forbidden_fields(value: Any, context: str = "row") -> None:
    if isinstance(value, dict):
        forbidden = sorted(FORBIDDEN_FIELDS.intersection(value))
        if forbidden:
            raise ValueError(f"{context} contains forbidden semantic fields: {forbidden}")
        for key, child in value.items():
            reject_forbidden_fields(child, f"{context}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_forbidden_fields(child, f"{context}[{index}]")


def load_rgb_rows(manifest: Path, camera: str) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(manifest.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError(f"manifest line {line_number} is not an object")
        reject_forbidden_fields(row, f"line {line_number}")
        rows.append(row)
    if len(rows) < 2:
        raise ValueError("probe requires at least two RGB frames")

    # Only these fields are consumed.  Environment seed, task, proprio, tape,
    # source summary hash, and any other metadata remain unread.
    rows.sort(key=lambda row: int(row["t"]))
    episode_ids = {int(row["episode_id"]) for row in rows}
    if len(episode_ids) != 1:
        raise ValueError(f"expected exactly one episode, got {sorted(episode_ids)}")
    times = [int(row["t"]) for row in rows]
    if len(set(times)) != len(times) or times != sorted(times):
        raise ValueError("frame times must be unique and increasing")
    if times[0] != 0 or 700 not in times:
        raise ValueError(f"expected t=0 and t=700 in the RGB replay, got [{times[0]}, {times[-1]}]")

    frames: list[np.ndarray] = []
    shape: tuple[int, int, int] | None = None
    for row in rows:
        image_meta = row["images"][camera]
        image_path = (manifest.parent / str(image_meta["path"])).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        observed_sha = sha256_file(image_path)
        if observed_sha != str(image_meta["sha256"]):
            raise ValueError(f"JPEG hash mismatch: {image_path}")
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"OpenCV failed to decode {image_path}")
        if shape is None:
            shape = frame.shape
        elif frame.shape != shape:
            raise ValueError(f"inconsistent image shape: {frame.shape} != {shape}")
        frames.append(frame)
    if shape != (360, 360, 3):
        raise ValueError(f"calibration is defined for 360x360 RGB, got {shape}")
    return rows, frames


def center(box: tuple[float, float, float, float] | list[float]) -> tuple[float, float]:
    x, y, width, height = box
    return float(x + width / 2.0), float(y + height / 2.0)


def iou(
    left: tuple[float, float, float, float] | list[float],
    right: tuple[float, float, float, float] | list[float],
) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    x0, y0 = max(lx, rx), max(ly, ry)
    x1, y1 = min(lx + lw, rx + rw), min(ly + lh, ry + rh)
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = lw * lh + rw * rh - intersection
    return float(intersection / union) if union > 0 else 0.0


def inside_basket(box: tuple[float, float, float, float] | list[float]) -> bool:
    x, y = center(box)
    x0, y0, x1, y1 = BASKET_ROI_XYXY
    return bool(x0 <= x <= x1 and y0 <= y <= y1)


def box_metrics(
    predicted: tuple[float, float, float, float] | list[float],
    reference: tuple[float, float, float, float] | list[float],
) -> dict[str, Any]:
    px, py = center(predicted)
    rx, ry = center(reference)
    return {
        "predicted_box_xywh": [round(float(value), 4) for value in predicted],
        "reference_box_xywh": [round(float(value), 4) for value in reference],
        "center_error_px": round(math.hypot(px - rx, py - ry), 6),
        "iou": round(iou(predicted, reference), 8),
        "predicted_inside_basket": inside_basket(predicted),
        "reference_inside_basket": inside_basket(reference),
        "basket_relation_correct": inside_basket(predicted) == inside_basket(reference),
    }


def run_mil(
    rows: list[dict[str, Any]], frames: list[np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cv2.setRNGSeed(0)
    trackers: dict[str, Any] = {}
    current = {name: tuple(box) for name, box in INITIAL_BOXES.items()}
    for name, box in INITIAL_BOXES.items():
        tracker = cv2.TrackerMIL_create()
        tracker.init(frames[0], tuple(int(value) for value in box))
        trackers[name] = tracker

    trajectory: list[dict[str, Any]] = [{
        "t": int(rows[0]["t"]),
        "targets": {
            name: {
                "api_update_ok": True,
                "box_xywh": list(box),
                "center_xy": list(center(box)),
                "inside_basket": inside_basket(box),
            }
            for name, box in current.items()
        },
    }]
    update_ok: dict[str, list[bool]] = {name: [] for name in trackers}
    for row, frame in zip(rows[1:], frames[1:], strict=True):
        targets: dict[str, Any] = {}
        for name, tracker in trackers.items():
            ok, raw_box = tracker.update(frame)
            box = tuple(float(value) for value in raw_box)
            current[name] = box
            update_ok[name].append(bool(ok))
            targets[name] = {
                "api_update_ok": bool(ok),
                "box_xywh": [round(value, 4) for value in box],
                "center_xy": [round(value, 4) for value in center(box)],
                "inside_basket": inside_basket(box),
            }
        trajectory.append({"t": int(row["t"]), "targets": targets})

    final_index = next(index for index, row in enumerate(rows) if int(row["t"]) == 700)
    metrics: dict[str, Any] = {}
    for name in INITIAL_BOXES:
        boxes = [entry["targets"][name]["box_xywh"] for entry in trajectory[: final_index + 1]]
        centres = [center(box) for box in boxes]
        step_motion = [
            math.hypot(right[0] - left[0], right[1] - left[1])
            for left, right in zip(centres, centres[1:])
        ]
        final_box = boxes[-1]
        target_metrics = box_metrics(final_box, FINAL_REFERENCE_BOXES[name])
        target_metrics.update({
            "api_updates_ok": int(sum(update_ok[name][:final_index])),
            "api_updates_total": int(final_index),
            "path_length_px": round(float(sum(step_motion)), 6),
            "net_displacement_px": round(
                math.hypot(centres[-1][0] - centres[0][0], centres[-1][1] - centres[0][1]), 6,
            ),
            "frozen_transition_fraction_lt_2px": round(
                float(sum(distance < 2.0 for distance in step_motion) / len(step_motion)), 8,
            ),
        })
        metrics[name] = target_metrics
    return trajectory, metrics


def dino_global_reid(
    first_frame: np.ndarray,
    final_frame: np.ndarray,
    snapshot: Path,
    input_size: int,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Final-frame appearance diagnostic; not a temporal tracker."""
    import torch
    from transformers import AutoModel

    if not snapshot.is_dir():
        raise FileNotFoundError(snapshot)
    config_path = snapshot / "config.json"
    weights_path = snapshot / "model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError("DINO snapshot must contain config.json and model.safetensors")
    model = AutoModel.from_pretrained(snapshot, local_files_only=True).eval()
    patch_size = int(model.config.patch_size)
    if input_size % patch_size:
        raise ValueError(f"--dino-input-size must be divisible by patch size {patch_size}")
    grid_size = input_size // patch_size
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def features(frame: np.ndarray) -> Any:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        tensor = (tensor - mean) / std
        with torch.inference_mode():
            tokens = model(pixel_values=tensor).last_hidden_state[0, 1:]
        if len(tokens) != grid_size * grid_size:
            raise ValueError(f"unexpected DINO token count: {len(tokens)}")
        return torch.nn.functional.normalize(tokens, dim=-1).reshape(grid_size, grid_size, -1)

    first_features, final_features = features(first_frame), features(final_frame)
    scale = input_size / float(first_frame.shape[1])
    patch_centres = (torch.arange(grid_size, dtype=torch.float32) + 0.5) * patch_size / scale
    yy, xx = torch.meshgrid(patch_centres, patch_centres, indexing="ij")
    output: dict[str, Any] = {}
    for name, (x, y, width, height) in INITIAL_BOXES.items():
        mask = (xx >= x) & (xx <= x + width) & (yy >= y) & (yy <= y + height)
        template_tokens = first_features[mask]
        if len(template_tokens) == 0:
            raise ValueError(f"no DINO patches fall inside {name} calibration box")
        template = torch.nn.functional.normalize(template_tokens.mean(dim=0), dim=0)
        similarity = (final_features * template).sum(dim=-1)
        flat_index = int(similarity.argmax())
        row, column = divmod(flat_index, grid_size)
        cx = float(patch_centres[column])
        cy = float(patch_centres[row])
        predicted = (
            min(max(0.0, cx - width / 2.0), first_frame.shape[1] - width),
            min(max(0.0, cy - height / 2.0), first_frame.shape[0] - height),
            width,
            height,
        )
        metrics = box_metrics(predicted, FINAL_REFERENCE_BOXES[name])
        metrics.update({
            "peak_cosine_similarity": round(float(similarity[row, column]), 8),
            "peak_patch_center_xy": [round(cx, 4), round(cy, 4)],
            "template_patch_count": int(mask.sum()),
        })
        output[name] = metrics
    provenance = {
        "snapshot": snapshot.resolve().as_posix(),
        "config_sha256": sha256_file(config_path),
        "weights_sha256": sha256_file(weights_path),
        "transform": (
            f"BGR->RGB; bicubic resize {input_size}x{input_size}; ImageNet mean/std; "
            "mean normalized t0 box patches; final-frame global cosine argmax"
        ),
    }
    return output, provenance


def relation_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    predicted = {name: bool(value["predicted_inside_basket"]) for name, value in metrics.items()}
    reference = {name: bool(value["reference_inside_basket"]) for name, value in metrics.items()}
    return {
        "predicted_inside_by_target": predicted,
        "reference_inside_by_target": reference,
        "predicted_inside_count": int(sum(predicted.values())),
        "reference_inside_count": int(sum(reference.values())),
        "per_target_accuracy": round(
            sum(predicted[name] == reference[name] for name in predicted) / len(predicted), 8,
        ),
        "exact_relation_match": predicted == reference,
    }


def aggregate(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "mean_final_iou": round(float(np.mean([value["iou"] for value in metrics.values()])), 8),
        "mean_final_center_error_px": round(
            float(np.mean([value["center_error_px"] for value in metrics.values()])), 8,
        ),
    }


def draw_box(frame: np.ndarray, box: list[float] | tuple[float, ...], color: tuple[int, int, int], label: str, width: int = 2) -> None:
    x, y, box_width, box_height = [int(round(value)) for value in box]
    cv2.rectangle(frame, (x, y), (x + box_width, y + box_height), color, width)
    cv2.putText(frame, label, (max(0, x + 2), max(14, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def write_visuals(
    rows: list[dict[str, Any]],
    frames: list[np.ndarray],
    trajectory: list[dict[str, Any]],
    out: Path,
) -> None:
    by_time = {int(entry["t"]): entry for entry in trajectory}
    frame_by_time = {int(row["t"]): frame for row, frame in zip(rows, frames, strict=True)}
    key_times = [time for time in (0, 50, 100, 150, 200, 250, 700) if time in by_time]
    tiles: list[np.ndarray] = []
    for time in key_times:
        tile = frame_by_time[time].copy()
        x0, y0, x1, y1 = [int(value) for value in BASKET_ROI_XYXY]
        cv2.rectangle(tile, (x0, y0), (x1, y1), (0, 145, 255), 2)
        for name, target in by_time[time]["targets"].items():
            draw_box(tile, target["box_xywh"], COLORS[name], f"MIL {name[-1].upper()}")
        if time == 700:
            for name, box in FINAL_REFERENCE_BOXES.items():
                draw_box(tile, box, COLORS[name], f"REF {name[-1].upper()}", width=1)
        cv2.putText(tile, f"t={time}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    columns = 4
    rows_count = math.ceil(len(tiles) / columns)
    canvas = np.zeros((rows_count * 360, columns * 360, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        canvas[row * 360:(row + 1) * 360, column * 360:(column + 1) * 360] = tile
    if not cv2.imwrite(str(out / "keyframes_mil.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise OSError("failed to write keyframes_mil.jpg")

    final = frame_by_time[700].copy()
    x0, y0, x1, y1 = [int(value) for value in BASKET_ROI_XYXY]
    cv2.rectangle(final, (x0, y0), (x1, y1), (0, 145, 255), 2)
    for name, target in by_time[700]["targets"].items():
        draw_box(final, target["box_xywh"], COLORS[name], f"MIL {name[-1].upper()}", width=2)
        draw_box(final, FINAL_REFERENCE_BOXES[name], COLORS[name], f"REF {name[-1].upper()}", width=1)
    if not cv2.imwrite(str(out / "final_overlay.jpg"), final, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise OSError("failed to write final_overlay.jpg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--camera", default="agentview", choices=("agentview",))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dino-snapshot", type=Path, default=None)
    parser.add_argument("--dino-input-size", type=int, default=364)
    args = parser.parse_args()
    if args.dino_input_size <= 0:
        parser.error("--dino-input-size must be positive")
    return args


def main() -> int:
    args = parse_args()
    manifest = args.manifest.resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)

    rows, frames = load_rgb_rows(manifest, args.camera)
    trajectory, mil_metrics = run_mil(rows, frames)
    t700_index = next(index for index, row in enumerate(rows) if int(row["t"]) == 700)
    dino_metrics: dict[str, Any] | None = None
    dino_provenance: dict[str, str] | None = None
    if args.dino_snapshot is not None:
        dino_metrics, dino_provenance = dino_global_reid(
            frames[0], frames[t700_index], args.dino_snapshot.resolve(), args.dino_input_size,
        )

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "verdict": (
            "REJECT: TrackerMIL does not preserve the two moved targets or recover the "
            "t=700 basket relation on this proof episode."
        ),
        "source": {
            "manifest": manifest.as_posix(),
            "manifest_sha256": sha256_file(manifest),
            "camera": args.camera,
            "frames": len(rows),
            "times": [int(row["t"]) for row in rows],
            "image_shape_hwc": list(frames[0].shape),
        },
        "protocol": {
            "tracker": f"OpenCV {cv2.__version__} TrackerMIL_create; RNG seed 0",
            "allowed_runtime_inputs": [
                "manifest t/episode_id/image path/image sha256",
                "agentview JPEG pixels",
                "three fixed t=0 RGB calibration boxes",
            ],
            "explicitly_not_opened": [
                "summary.json", "action/latent tape", "BDDL", "simulator info",
                "reward", "event", "milestone/stage", "success/outcome labels",
            ],
            "initial_boxes_xywh": {name: list(box) for name, box in INITIAL_BOXES.items()},
            "evaluation_only_t700_reference_boxes_xywh": {
                name: list(box) for name, box in FINAL_REFERENCE_BOXES.items()
            },
            "basket_roi_xyxy": list(BASKET_ROI_XYXY),
            "basket_rule": "bbox centre lies inside fixed agentview basket ROI",
            "evaluation_reference_note": (
                "The t=700 boxes and basket ROI were annotated from RGB pixels only and are "
                "used only after tracking to quantify the proof."
            ),
        },
        "tracker_mil": {
            "per_target": mil_metrics,
            "aggregate": aggregate(mil_metrics),
            "basket_relation": relation_summary(mil_metrics),
        },
        "dino_global_reid_diagnostic": None,
        "limitations": [
            "This is a single-episode, manually initialized proof, not a detector or Gate-0 result.",
            "TrackerMIL returning api_update_ok=true is not evidence that the object is localized.",
            "No intermediate-frame ground-truth boxes were used; drift is measured against the t=700 RGB audit boxes.",
            "The fixed basket ROI is valid only for this static agent-view camera geometry.",
        ],
    }
    if dino_metrics is not None:
        report["dino_global_reid_diagnostic"] = {
            "scope": (
                "appearance-only t=0 template to t=700 global re-identification; it is not "
                "used to alter the MIL trajectory"
            ),
            "provenance": dino_provenance,
            "per_target": dino_metrics,
            "aggregate": aggregate(dino_metrics),
            "basket_relation": relation_summary(dino_metrics),
        }

    with (out / "trajectory.jsonl").open("w") as handle:
        for entry in trajectory:
            handle.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
    write_visuals(rows, frames, trajectory, out)
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "out": out.as_posix(),
        "verdict": report["verdict"],
        "mil_relation": report["tracker_mil"]["basket_relation"],
        "dino_relation": (
            report["dino_global_reid_diagnostic"]["basket_relation"]
            if report["dino_global_reid_diagnostic"] is not None else None
        ),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
