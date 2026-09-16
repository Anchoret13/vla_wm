#!/usr/bin/env python3
"""Run a sealed SAM3 video probe on disclosed eye-in-hand development RGB.

This is a narrow diagnostic companion to ``probe_v245_sam3_video_progress``.
It reuses that frozen streaming implementation but selects the eye-in-hand
camera.  Only already disclosed seed-8700 episodes are accepted.  No rollout
summary, event, success label, latent tape, or simulator state is opened.

The output is development evidence only.  It cannot promote Gate 0 and it must
not be used to tune a candidate after an independent panel is opened.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

import probe_v245_sam3_video_progress as base


SCHEMA = "v247_sam3_eye_video_development_probe_v1"
COMPLETE_SCHEMA = "v247_sam3_eye_video_development_complete_v1"
CAMERA = "eye_in_hand"
ALLOWED_EPISODES = frozenset({0, 12, 37, 71})


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
    parser.add_argument("--prefix-check-frames", type=int, default=0)
    return parser.parse_args()


def load_rows(
    manifest_paths: Sequence[Path], episode_ids: Sequence[int]
) -> tuple[dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
    wanted = set(episode_ids)
    base.require(len(wanted) == len(episode_ids), "duplicate --episode-id")
    base.require(wanted <= ALLOWED_EPISODES, "episode is not disclosed development data")
    rows_by_episode: dict[int, list[dict[str, Any]]] = {
        episode_id: [] for episode_id in wanted
    }
    manifest_records = []
    for raw_path in manifest_paths:
        path = raw_path.expanduser().resolve()
        base.require(path.is_file(), f"missing manifest: {path}")
        selected = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                base.reject_semantic_fields(row, f"{path}:{line_number}")
                episode_id = int(row.get("episode_id", -1))
                if episode_id not in wanted:
                    continue
                base.require(
                    row.get("schema") == "v242_rgb_replay_v1",
                    "unexpected manifest schema",
                )
                base.require(
                    row.get("task") == "chain3_lr2",
                    "only chain3_lr2 development is allowed",
                )
                base.require(
                    int(row.get("env_seed", -1)) == 8700 + episode_id,
                    f"ep{episode_id}: only seed-8700 development is allowed",
                )
                image_record = row.get("images", {}).get(CAMERA)
                base.require(
                    isinstance(image_record, dict),
                    f"ep{episode_id}: missing {CAMERA}",
                )
                image_path = (path.parent / image_record["path"]).resolve()
                base.require(image_path.is_file(), f"missing image: {image_path}")
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
                        "frame_kind": str(row.get("frame_kind", "pre_action")),
                    }
                )
                selected += 1
        manifest_records.append(
            {"path": str(path), "sha256": base.sha256_file(path), "selected_rows": selected}
        )
    for episode_id, rows in rows_by_episode.items():
        base.require(rows, f"ep{episode_id}: no rows in supplied manifests")
        rows.sort(key=lambda row: (row["t"], row["row_index"]))
        base.require(len({row["t"] for row in rows}) == len(rows), "duplicate t")
        base.require(rows[0]["t"] == 0, f"ep{episode_id}: first frame is not t0")
        base.require(
            all(left["t"] < right["t"] for left, right in zip(rows, rows[1:])),
            f"ep{episode_id}: non-monotone time",
        )
    return rows_by_episode, manifest_records


def main() -> int:
    args = parse_args()
    manifests = [path.expanduser().resolve() for path in args.manifest]
    checkpoint = args.checkpoint.expanduser().resolve()
    tokenizer_path = args.clip_tokenizer.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    base.require(not output_dir.exists(), f"refusing to overwrite {output_dir}")
    base.require(tokenizer_path.is_dir(), f"missing tokenizer: {tokenizer_path}")
    base.require(args.prefix_check_frames >= 0, "prefix frames must be nonnegative")
    base.require(args.device != "cuda" or torch.cuda.is_available(), "CUDA unavailable")
    dtype = getattr(torch, args.dtype)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32 = False
    rows_by_episode, manifest_records = load_rows(manifests, args.episode_id)

    model = base.load_model(checkpoint, args.device, dtype)
    processor = base.build_processor(tokenizer_path)
    all_frames = []
    episodes = {}
    prefix_checks = {}
    for episode_id in args.episode_id:
        rows = rows_by_episode[episode_id]
        frames, diagnostics = base.process_rows(
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
            repeated, _ = base.process_rows(
                model,
                processor,
                rows[:count],
                device=args.device,
                state_device=args.state_device,
                dtype=dtype,
            )
            reference = [base.prefix_comparison_view(frame) for frame in frames[:count]]
            fresh = [base.prefix_comparison_view(frame) for frame in repeated]
            prefix_checks[str(episode_id)] = {
                "frames": count,
                "reference_sha256": base.canonical_sha256(reference),
                "fresh_session_sha256": base.canonical_sha256(fresh),
                "exact_after_rounding_1e_6": reference == fresh,
            }

    source_path = Path(__file__).resolve()
    dependency_path = Path(base.__file__).resolve()
    result = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "development_probe_complete",
        "claim_scope": "post-hoc seed-8700 eye-in-hand diagnostic; not Gate-0 validation",
        "camera": CAMERA,
        "privileged_inputs_read": False,
        "inputs": manifest_records,
        "episode_ids": list(args.episode_id),
        "prompts": base.PROMPTS,
        "episodes": episodes,
        "prefix_checks": prefix_checks,
        "model": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": base.sha256_file(checkpoint),
            "clip_tokenizer": str(tokenizer_path),
            "clip_tokenizer_tree_sha256": base.tree_sha256(tokenizer_path),
            "device": args.device,
            "state_device": args.state_device,
            "dtype": args.dtype,
        },
        "producer_source_sha256": base.sha256_file(source_path),
        "streaming_dependency_sha256": base.sha256_file(dependency_path),
        "frames_jsonl": "frames.jsonl",
    }
    staging = Path(str(output_dir) + f".tmp.{os.getpid()}")
    base.require(not staging.exists(), f"staging path exists: {staging}")
    staging.mkdir(parents=True)
    try:
        frames_path = staging / "frames.jsonl"
        with frames_path.open("w", encoding="utf-8") as handle:
            for frame in all_frames:
                handle.write(json.dumps(frame, sort_keys=True, allow_nan=False) + "\n")
        shutil.copy2(source_path, staging / "producer_source.py")
        shutil.copy2(dependency_path, staging / "streaming_dependency.py")
        base.write_json(staging / "result.json", result)
        complete = {
            "schema": COMPLETE_SCHEMA,
            "status": "complete",
            "result_sha256": base.sha256_file(staging / "result.json"),
            "frames_sha256": base.sha256_file(frames_path),
            "producer_source_sha256": base.sha256_file(staging / "producer_source.py"),
            "streaming_dependency_sha256": base.sha256_file(
                staging / "streaming_dependency.py"
            ),
        }
        base.write_json(staging / "COMPLETE.json", complete)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.rename(staging, output_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    print(
        json.dumps(
            {
                "output": str(output_dir),
                "episodes": episodes,
                "prefix_checks": prefix_checks,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
