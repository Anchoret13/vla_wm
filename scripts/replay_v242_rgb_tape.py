#!/usr/bin/env python
"""Deterministically replay a v121 action tape and recover sparse RGB frames.

The v121 tape stores the *normalized actions that were actually executed*, but
not the corresponding RGB observations.  This script resets the chain3
environment with each recorded episode seed, replays those actions through the
same LeRobot policy/environment postprocessors used during collection, and
writes pre-action frames at a fixed chunk stride.

Sampling is deliberately outcome-blind: neither milestone/stage fields nor
success fields participate in frame selection.  A frame is retained iff its
zero-based chunk index is divisible by ``--chunk-stride``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# These must be set before importing LIBERO/robosuite.  Keep all caches in a
# task-specific writable location, and make Hugging Face loading hermetic: the
# action processor artifacts are already present locally.
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/v242_numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/v242_mpl_cache")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig  # noqa: E402
from lerobot.envs.factory import make_env_pre_post_processors  # noqa: E402
from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: E402,F401
from lerobot.processor import (  # noqa: E402
    PolicyProcessorPipeline,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME  # noqa: E402
from lcwm.v080_bench import V080_TASKS, make_v080_env  # noqa: E402


SCHEMA = "v242_rgb_replay_v1"
SUMMARY_SCHEMA = "v242_rgb_replay_summary_v1"
DEFAULT_MODEL = "lerobot/pi05_libero_finetuned"
PROPRIO_DIM = 25
PROPRIO_KEYS = (
    "robot_state.eef.pos",
    "robot_state.eef.quat",
    "robot_state.gripper.qpos",
    "robot_state.gripper.qvel",
    "robot_state.joints.pos",
    "robot_state.joints.vel",
)
CAMERAS = {"agentview": "image", "eye_in_hand": "image2"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
        text=True, check=False,
    ).stdout.strip()


def nested_value(tree: dict[str, Any], dotted_key: str) -> Any:
    value: Any = tree
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"observation is missing {dotted_key!r}")
        value = value[part]
    return value


def proprio(obs: dict[str, Any]) -> torch.Tensor:
    parts = [
        torch.as_tensor(np.asarray(nested_value(obs, key)), dtype=torch.float32).flatten()
        for key in PROPRIO_KEYS
    ]
    result = torch.cat(parts)
    if result.shape != (PROPRIO_DIM,):
        raise ValueError(f"expected {PROPRIO_DIM}-d proprio, got {tuple(result.shape)}")
    if not torch.isfinite(result).all():
        raise ValueError("observation proprio contains NaN/Inf")
    return result


class ReplayActionPostprocessor:
    """Decode normalized pi0.5 actions without loading VLA model weights.

    Loading only ``policy_postprocessor.json`` avoids constructing the policy,
    image processor, or tokenizer.  The decoded action is then passed through
    LeRobot's LIBERO environment postprocessor, exactly as in ``Pi05Runner``.
    """

    def __init__(self, model_id: str) -> None:
        cfg = PreTrainedConfig.from_pretrained(model_id)
        if not isinstance(cfg, PI05Config):
            raise TypeError(f"expected a PI05Config for {model_id!r}, got {type(cfg).__name__}")
        cfg.device = "cpu"
        self.policy_postprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=model_id,
            config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )
        _, self.env_postprocessor = make_env_pre_post_processors(
            env_cfg=LiberoEnvConfig(task="libero_10"), policy_cfg=cfg,
        )

    @torch.no_grad()
    def chunk_to_env(self, chunk: torch.Tensor) -> np.ndarray:
        chunk = torch.as_tensor(chunk, dtype=torch.float32, device="cpu")
        if chunk.ndim != 2 or chunk.shape[1] != 7:
            raise ValueError(f"expected normalized chunk [T,7], got {tuple(chunk.shape)}")
        actions: list[np.ndarray] = []
        for action in chunk:
            decoded = self.policy_postprocessor(action.unsqueeze(0))
            transition = self.env_postprocessor({"action": decoded})
            value = transition["action"]
            array = (
                value[0].detach().cpu().numpy()
                if torch.is_tensor(value)
                else np.asarray(value)[0]
            )
            array = np.asarray(array, dtype=np.float32)
            if array.shape != (7,) or not np.isfinite(array).all():
                raise ValueError(f"invalid decoded environment action: shape={array.shape}")
            actions.append(array)
        return np.stack(actions)


def _integer_vector(value: Any, name: str, n: int) -> torch.Tensor:
    if not torch.is_tensor(value) or value.ndim != 1 or len(value) != n:
        raise ValueError(f"{name} must be a tensor [{n}]")
    integer = value.to(dtype=torch.long, device="cpu")
    if not torch.equal(value.cpu(), integer.to(dtype=value.dtype)):
        raise ValueError(f"{name} contains non-integer values")
    return integer


def validate_source(
    tape: dict[str, Any], source_summary: dict[str, Any], expected_task: str,
    expected_c: int,
) -> tuple[dict[int, list[int]], dict[int, dict[str, int]], int, int]:
    """Validate source tensors, row continuity, and episode-to-seed binding."""
    required = {"z", "u", "z_next", "episode", "t", "task", "c", "latent_dim"}
    missing = sorted(required - tape.keys())
    if missing:
        raise ValueError(f"tape missing required fields: {missing}")

    task = str(tape["task"])
    if task != expected_task:
        raise ValueError(f"task mismatch: CLI={expected_task!r}, tape={task!r}")
    if task not in V080_TASKS or not task.startswith("chain3"):
        raise ValueError(f"RGB replay is restricted to registered chain3 tasks, got {task!r}")
    c = int(tape["c"])
    if c != expected_c:
        raise ValueError(f"commit mismatch: expected c={expected_c}, tape has c={c}")

    z, u, z_next = tape["z"], tape["u"], tape["z_next"]
    if not all(torch.is_tensor(x) for x in (z, u, z_next)):
        raise TypeError("z, u, and z_next must be tensors")
    if z.ndim != 2 or len(z) == 0:
        raise ValueError(f"z must be a non-empty [N,D] tensor, got {tuple(z.shape)}")
    n, zdim = z.shape
    if zdim < PROPRIO_DIM or z_next.shape != z.shape:
        raise ValueError(f"invalid latent shapes: z={tuple(z.shape)}, z_next={tuple(z_next.shape)}")
    if u.shape != (n, c, 7):
        raise ValueError(f"u must have shape {(n, c, 7)}, got {tuple(u.shape)}")
    if int(tape["latent_dim"]) != zdim:
        raise ValueError(f"latent_dim metadata {tape['latent_dim']} != tensor dim {zdim}")
    if int(tape.get("proprio_dim", -1)) != PROPRIO_DIM:
        raise ValueError(f"expected proprio_dim={PROPRIO_DIM}, got {tape.get('proprio_dim')!r}")
    if tuple(tape.get("proprio_keys", ())) != PROPRIO_KEYS:
        raise ValueError("tape proprio_keys do not match the v121 25-d layout")
    if not torch.isfinite(u).all() or not torch.isfinite(z[:, -PROPRIO_DIM:]).all():
        raise ValueError("action/proprio tensors contain NaN/Inf")

    episode = _integer_vector(tape["episode"], "episode", n)
    times = _integer_vector(tape["t"], "t", n)
    episode_rows: dict[int, list[int]] = {}
    previous_ep = -1
    for row, (episode_id_t, time_t) in enumerate(zip(episode, times, strict=True)):
        episode_id, t = int(episode_id_t), int(time_t)
        if episode_id < previous_ep:
            raise ValueError(f"episode rows are interleaved/out of order at row {row}")
        previous_ep = episode_id
        episode_rows.setdefault(episode_id, []).append(row)
        local = len(episode_rows[episode_id]) - 1
        if t != local * c:
            raise ValueError(
                f"non-contiguous tape: episode {episode_id}, row {row}, "
                f"t={t}, expected {local * c}"
            )

    records_raw = source_summary.get("episode_records")
    if not isinstance(records_raw, list) or not records_raw:
        raise ValueError("summary episode_records must be a non-empty list")
    episode_records: dict[int, dict[str, int]] = {}
    seen_seeds: set[int] = set()
    for record in records_raw:
        if not isinstance(record, dict) or "idx" not in record or "seed" not in record:
            raise ValueError("every episode record must contain idx and seed")
        episode_id, seed = int(record["idx"]), int(record["seed"])
        if episode_id in episode_records or seed in seen_seeds:
            raise ValueError(
                "duplicate episode id or seed in summary: "
                f"idx={episode_id}, seed={seed}"
            )
        steps = int(record.get("steps", len(episode_rows.get(episode_id, ())) * c))
        episode_records[episode_id] = {"seed": seed, "steps": steps}
        seen_seeds.add(seed)

    expected_ids = list(range(len(episode_records)))
    if sorted(episode_records) != expected_ids:
        raise ValueError("summary episode ids must be contiguous from zero")
    if sorted(episode_rows) != expected_ids:
        raise ValueError("tape episode ids and summary episode ids differ")
    if int(source_summary.get("episodes", -1)) != len(episode_records):
        raise ValueError("summary episodes count does not match episode_records")
    if int(source_summary.get("triples", -1)) != n:
        raise ValueError("summary triples count does not match tape rows")
    if str(source_summary.get("task")) != task or int(source_summary.get("c", -1)) != c:
        raise ValueError("summary task/c metadata does not match tape")
    if int(source_summary.get("latent_dim", -1)) != zdim:
        raise ValueError("summary latent_dim does not match tape")
    seed_range = source_summary.get("seed_range")
    if seed_range != [min(seen_seeds), max(seen_seeds)]:
        raise ValueError(f"summary seed_range {seed_range!r} does not match episode seeds")
    for episode_id, rows in episode_rows.items():
        recorded_steps = len(rows) * c
        source_steps = episode_records[episode_id]["steps"]
        if source_steps < recorded_steps or source_steps - recorded_steps > c:
            raise ValueError(
                f"episode {episode_id}: summary steps={source_steps} incompatible with "
                f"{len(rows)} stored chunks of c={c}"
            )
    return episode_rows, episode_records, c, zdim


def policy_oriented_rgb(obs: dict[str, Any], obs_key: str) -> np.ndarray:
    try:
        raw = np.asarray(obs["pixels"][obs_key])
    except (KeyError, TypeError) as exc:
        raise KeyError(f"observation is missing pixels/{obs_key}") from exc
    if raw.ndim != 3 or raw.shape[2] != 3 or raw.dtype != np.uint8:
        raise ValueError(f"pixels/{obs_key} must be uint8 [H,W,3], got {raw.dtype} {raw.shape}")
    # LeRobot's LiberoProcessorStep rotates both cameras by 180 degrees before
    # the policy sees them.  Save that same, human-readable orientation.
    return np.ascontiguousarray(raw[::-1, ::-1])


def save_jpeg(array: np.ndarray, path: Path, quality: int) -> dict[str, Any]:
    Image.fromarray(array, mode="RGB").save(
        path, format="JPEG", quality=quality, subsampling=0, optimize=False,
    )
    height, width = array.shape[:2]
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "width": int(width),
        "height": int(height),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tape", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None,
                        help="source v121 summary (default: sibling summary.json)")
    parser.add_argument("--task", default="chain3_lr2")
    parser.add_argument("--expected-c", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=None,
                        help="replay only the first N episodes (for smoke tests)")
    parser.add_argument("--episode-ids", type=int, nargs="+", default=None,
                        help="replay these explicit episode IDs; mutually exclusive "
                             "with --episodes")
    parser.add_argument("--chunk-stride", type=int, default=5)
    parser.add_argument("--max-proprio-error", type=float, default=1e-6)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.expected_c <= 0 or args.chunk_stride <= 0:
        parser.error("--expected-c and --chunk-stride must be positive")
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
    summary_path = (args.summary or (tape_path.parent / "summary.json")).resolve()
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

    all_episode_ids = sorted(episode_rows)
    if args.episode_ids is not None:
        unknown = sorted(set(args.episode_ids) - set(all_episode_ids))
        if unknown:
            raise ValueError(f"requested episode IDs not present in tape: {unknown}")
        selected_ids = sorted(args.episode_ids)
    elif args.episodes is not None:
        if args.episodes > len(all_episode_ids):
            raise ValueError(f"requested {args.episodes} episodes, tape has {len(all_episode_ids)}")
        selected_ids = all_episode_ids[:args.episodes]
    else:
        selected_ids = all_episode_ids

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = (args.out or (REPO / "results" / "v242_rgb_replay" / f"{args.task}_{stamp}")).resolve()
    out.mkdir(parents=True, exist_ok=False)
    images_dir = out / "images"
    images_dir.mkdir()
    manifest_path = out / "manifest.jsonl"

    decoder = ReplayActionPostprocessor(args.model_id)
    env = make_v080_env(args.task)
    env_steps_replayed = 0
    frames_saved = 0
    errors: list[float] = []
    missing_terminal_steps = 0
    episodes_with_missing_terminal_action = 0

    try:
        with manifest_path.open("w") as manifest:
            for episode_id in selected_ids:
                rows = episode_rows[episode_id]
                seed = episode_records[episode_id]["seed"]
                source_steps = episode_records[episode_id]["steps"]
                missing = source_steps - len(rows) * c
                if missing:
                    missing_terminal_steps += missing
                    episodes_with_missing_terminal_action += 1

                torch.manual_seed(seed)
                np.random.seed(seed)
                obs, _ = env.reset(seed=seed)
                for chunk_index, row_index in enumerate(rows):
                    t = int(tape["t"][row_index])
                    if chunk_index % args.chunk_stride == 0:
                        observed = proprio(obs)
                        recorded = tape["z"][row_index, -PROPRIO_DIM:].detach().float().cpu()
                        delta = (observed - recorded).abs()
                        max_error = float(delta.max())
                        mean_error = float(delta.mean())
                        if max_error > args.max_proprio_error:
                            raise RuntimeError(
                                f"proprio replay divergence at episode={episode_id}, t={t}, "
                                f"row={row_index}: max_abs_error={max_error:.9g} exceeds "
                                f"{args.max_proprio_error:.9g}"
                            )

                        image_records: dict[str, dict[str, Any]] = {}
                        for camera, obs_key in CAMERAS.items():
                            filename = f"ep{episode_id:04d}_t{t:04d}_{camera}.jpg"
                            image_path = images_dir / filename
                            metadata = save_jpeg(
                                policy_oriented_rgb(obs, obs_key), image_path, args.jpeg_quality,
                            )
                            metadata["path"] = image_path.relative_to(out).as_posix()
                            image_records[camera] = metadata

                        frame_id = f"{tape_sha[:12]}:ep{episode_id}:t{t}"
                        row = {
                            "schema": SCHEMA,
                            "frame_id": frame_id,
                            "task": args.task,
                            "tape_path": tape_path.as_posix(),
                            "tape_sha256": tape_sha,
                            "summary_sha256": source_summary_sha,
                            "episode_id": episode_id,
                            "env_seed": seed,
                            "t": t,
                            "row_index": row_index,
                            "chunk_index": chunk_index,
                            "images": image_records,
                            "proprio": {
                                "max_abs_error": max_error,
                                "mean_abs_error": mean_error,
                            },
                            "sampling": {
                                "chunk_stride": args.chunk_stride,
                                "pre_action": True,
                            },
                        }
                        manifest.write(
                            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                        )
                        manifest.flush()
                        frames_saved += 1
                        errors.append(max_error)

                    env_actions = decoder.chunk_to_env(tape["u"][row_index])
                    for action_offset, action in enumerate(env_actions):
                        obs, _reward, terminated, truncated, _info = env.step(action)
                        env_steps_replayed += 1
                        if terminated or truncated:
                            raise RuntimeError(
                                f"environment terminated while replaying a stored transition: "
                                f"episode={episode_id}, t={t}, action_offset={action_offset}"
                            )
                print(
                    f"[{len(errors):04d} frames] episode {episode_id} seed={seed}: "
                    f"{len(rows)} chunks, missing_terminal_steps={missing}",
                    flush=True,
                )
    finally:
        env.close()

    manifest_sha = sha256_file(manifest_path)
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
        "summary_sha256": source_summary_sha,
        "manifest_path": manifest_path.relative_to(out).as_posix(),
        "manifest_sha256": manifest_sha,
        "episodes_available": len(all_episode_ids),
        "episodes_replayed": len(selected_ids),
        "episode_ids": selected_ids,
        "env_seeds": [episode_records[e]["seed"] for e in selected_ids],
        "chunks_replayed": sum(len(episode_rows[e]) for e in selected_ids),
        "env_steps_replayed": env_steps_replayed,
        "frames_saved": frames_saved,
        "sampling": {"chunk_stride": args.chunk_stride, "pre_action": True},
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
        "terminal_action_recovery": {
            "recoverable": False,
            "episodes_with_unstored_terminal_steps": episodes_with_missing_terminal_action,
            "unstored_terminal_env_steps": missing_terminal_steps,
            "note": (
                "v121 stores an action chunk only when a subsequent z_next exists. "
                "A chunk that terminates mid-execution is therefore absent and cannot "
                "be recovered; this replay stops after the last stored chunk."
            ),
        },
        "action_decoder": {
            "model_id": args.model_id,
            "implementation": (
                "LeRobot policy_postprocessor + LIBERO env_postprocessor; "
                "no VLA weights, image processor, tokenizer, or new policy actions loaded"
            ),
        },
        "cli": sys.argv,
        "git": git_head(),
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote {frames_saved} pre-action frames from {len(selected_ids)} episodes; "
        f"replayed {env_steps_replayed} env steps -> {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
