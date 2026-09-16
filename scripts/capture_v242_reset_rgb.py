#!/usr/bin/env python
"""Capture deployment-camera RGB at reset for a validated v121 tape panel.

This is a zero-action initialization audit for the v242 temporal tracker.  It
uses the tape/summary only to bind episode IDs to deterministic environment
seeds and to verify the reset proprioception against the first stored latent.
Frame selection is exactly one t=0 frame per selected episode; outcome,
milestone, event, success and BDDL fields are neither inspected nor emitted.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from replay_v242_rgb_tape import (
    CAMERAS,
    PROPRIO_DIM,
    REPO,
    git_head,
    make_v080_env,
    np,
    policy_oriented_rgb,
    proprio,
    save_jpeg,
    sha256_file,
    torch,
    validate_source,
)


SCHEMA = "v242_reset_rgb_v1"
SUMMARY_SCHEMA = "v242_reset_rgb_summary_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tape", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--task", default="chain3_lr2")
    parser.add_argument("--expected-c", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=None,
                        help="capture the first N episode resets; default is the full tape")
    parser.add_argument("--episode-ids", type=int, nargs="+", default=None,
                        help="capture explicit episode IDs; mutually exclusive with --episodes")
    parser.add_argument("--max-proprio-error", type=float, default=1e-6)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.expected_c <= 0:
        parser.error("--expected-c must be positive")
    if args.episodes is not None and args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.episodes is not None and args.episode_ids is not None:
        parser.error("--episodes and --episode-ids are mutually exclusive")
    if args.episode_ids is not None:
        if any(episode_id < 0 for episode_id in args.episode_ids):
            parser.error("--episode-ids must be non-negative")
        if len(set(args.episode_ids)) != len(args.episode_ids):
            parser.error("--episode-ids must be unique")
    if not np.isfinite(args.max_proprio_error) or args.max_proprio_error < 0:
        parser.error("--max-proprio-error must be finite and non-negative")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in [1,100]")
    return args


def main() -> int:
    args = parse_args()
    tape_path = args.tape.resolve()
    summary_path = (args.summary or tape_path.with_name("summary.json")).resolve()
    if not tape_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"missing tape or summary: {tape_path}, {summary_path}")

    tape_sha = sha256_file(tape_path)
    source_summary_sha = sha256_file(summary_path)
    tape = torch.load(tape_path, map_location="cpu", weights_only=False)
    if not isinstance(tape, dict):
        raise TypeError("tape must contain a dictionary")
    source_summary = json.loads(summary_path.read_text())
    episode_rows, episode_records, c, zdim = validate_source(
        tape, source_summary, args.task, args.expected_c,
    )

    episode_ids = sorted(episode_rows)
    if args.episode_ids is not None:
        unknown = sorted(set(args.episode_ids) - set(episode_ids))
        if unknown:
            raise ValueError(f"requested episode IDs not present in tape: {unknown}")
        episode_ids = sorted(args.episode_ids)
    elif args.episodes is not None:
        if args.episodes > len(episode_ids):
            raise ValueError(f"requested {args.episodes} episodes, tape has {len(episode_ids)}")
        episode_ids = episode_ids[:args.episodes]

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    default_out = REPO / "results" / "v242_reset_rgb" / f"{args.task}_{stamp}"
    out = (args.out or default_out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    images_dir = out / "images"
    images_dir.mkdir()
    manifest_path = out / "manifest.jsonl"

    errors: list[float] = []
    env = make_v080_env(args.task)
    try:
        with manifest_path.open("w") as manifest:
            for episode_id in episode_ids:
                row_index = episode_rows[episode_id][0]
                if int(tape["t"][row_index]) != 0:
                    raise ValueError(f"episode {episode_id} has no t=0 tape row")
                seed = episode_records[episode_id]["seed"]
                torch.manual_seed(seed)
                np.random.seed(seed)
                obs, _ = env.reset(seed=seed)

                observed = proprio(obs)
                recorded = tape["z"][row_index, -PROPRIO_DIM:].detach().float().cpu()
                delta = (observed - recorded).abs()
                max_error = float(delta.max())
                mean_error = float(delta.mean())
                if max_error > args.max_proprio_error:
                    raise RuntimeError(
                        f"reset replay divergence at episode={episode_id}, seed={seed}: "
                        f"max_abs_error={max_error:.9g} exceeds {args.max_proprio_error:.9g}"
                    )

                images: dict[str, dict] = {}
                for camera, obs_key in CAMERAS.items():
                    image_path = images_dir / f"ep{episode_id:04d}_t0000_{camera}.jpg"
                    metadata = save_jpeg(
                        policy_oriented_rgb(obs, obs_key), image_path, args.jpeg_quality,
                    )
                    metadata["path"] = image_path.relative_to(out).as_posix()
                    images[camera] = metadata

                record = {
                    "schema": SCHEMA,
                    "frame_id": f"{tape_sha[:12]}:ep{episode_id}:t0",
                    "task": args.task,
                    "tape_path": tape_path.as_posix(),
                    "tape_sha256": tape_sha,
                    "summary_sha256": source_summary_sha,
                    "episode_id": episode_id,
                    "env_seed": seed,
                    "t": 0,
                    "row_index": row_index,
                    "images": images,
                    "proprio": {
                        "max_abs_error": max_error,
                        "mean_abs_error": mean_error,
                    },
                    "sampling": {"mode": "one_reset_frame_per_episode", "env_steps": 0},
                }
                manifest.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                manifest.flush()
                errors.append(max_error)
                print(
                    f"[{len(errors):03d}/{len(episode_ids):03d}] "
                    f"episode={episode_id} seed={seed} max_err={max_error:.3g}",
                    flush=True,
                )
    finally:
        env.close()

    script_path = Path(__file__).resolve()
    summary = {
        "schema": SUMMARY_SCHEMA,
        "utc": stamp,
        "task": args.task,
        "c": c,
        "latent_dim": zdim,
        "proprio_dim": PROPRIO_DIM,
        "tape_path": tape_path.as_posix(),
        "tape_sha256": tape_sha,
        "source_summary_path": summary_path.as_posix(),
        "source_summary_sha256": source_summary_sha,
        "manifest_path": manifest_path.relative_to(out).as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "episodes_available": len(episode_rows),
        "episodes_captured": len(episode_ids),
        "episode_ids": episode_ids,
        "env_seeds": [episode_records[e]["seed"] for e in episode_ids],
        "env_steps": 0,
        "frames_saved": len(errors),
        "sampling": {"mode": "one_reset_frame_per_episode", "t": 0},
        "semantic_fields_accessed": [],
        "proprio_validation": {
            "threshold_max_abs_error": args.max_proprio_error,
            "observed_max_abs_error": max(errors, default=0.0),
            "mean_of_frame_max_abs_error": float(np.mean(errors)) if errors else 0.0,
            "frames_checked": len(errors),
        },
        "images": {
            "cameras": CAMERAS,
            "format": "JPEG",
            "quality": args.jpeg_quality,
            "orientation": "rotated 180 degrees to match LeRobot LiberoProcessorStep",
        },
        "code": {"path": script_path.as_posix(), "sha256": sha256_file(script_path)},
        "git": {
            "head": git_head(),
            "status_short": subprocess.run(
                ["git", "status", "--short"], cwd=REPO, capture_output=True,
                text=True, check=False,
            ).stdout.splitlines(),
        },
        "cli": vars(args) | {
            "tape": tape_path.as_posix(),
            "summary": summary_path.as_posix(),
            "out": out.as_posix(),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"captured {len(errors)} reset frames, env_steps=0 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
