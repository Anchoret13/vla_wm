#!/usr/bin/env python3
"""Compose sealed anonymous-can and multiview-cream development progress.

The inputs are observation-only artifacts.  This composer never opens an event
summary.  It replaces v245's strict centroid-in-box can relation with the frozen
v247 eight-pixel top-rim allowance, then sums anonymous can count and the sealed
causal cream state.  Output remains development-only until an independently
registered producer/evaluator passes an untouched panel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lcwm.v247_can_state import CanStateConfig, CausalCanCount  # noqa: E402


SCHEMA = "v247_multiview_progress_development_composition_v1"
FRAME_SCHEMA = "v247_multiview_progress_development_frame_v1"
COMPLETE_SCHEMA = "v247_multiview_progress_development_complete_v1"
CREAM_RESULT_SCHEMA = "v247_multiview_cream_development_probe_v1"
CREAM_FRAME_SCHEMA = "v247_multiview_cream_development_frame_v1"
CREAM_COMPLETE_SCHEMA = "v247_multiview_cream_development_complete_v1"
SAM3_RESULT_SCHEMA = "v245_sam3_video_progress_probe_v1"
SAM3_FRAME_SCHEMA = "v245_sam3_video_progress_frame_v1"
SAM3_COMPLETE_SCHEMA = "v245_sam3_video_progress_complete_v1"


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


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"{path}: expected object")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cream-run", type=Path, required=True)
    parser.add_argument("--sam3-run", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix-check-frames", type=int, default=50)
    return parser.parse_args()


def verify_cream_run(
    raw_root: Path,
) -> tuple[dict[str, Any], dict[tuple[int, int], dict[str, Any]], dict[str, Any]]:
    root = raw_root.expanduser().resolve()
    paths = {
        "result": root / "RESULT.json",
        "frames": root / "frames.jsonl",
        "complete": root / "COMPLETE.json",
        "producer": root / "PRODUCER.py",
        "state": root / "CREAM_STATE.py",
    }
    for path in paths.values():
        require(path.is_file(), f"missing cream artifact: {path}")
    complete = load_json(paths["complete"])
    require(complete.get("schema") == CREAM_COMPLETE_SCHEMA, "cream COMPLETE schema")
    require(complete.get("status") == "complete", "cream artifact incomplete")
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    bindings = {
        "result_sha256": hashes["result"],
        "frames_sha256": hashes["frames"],
        "producer_sha256": hashes["producer"],
        "state_source_sha256": hashes["state"],
    }
    for key, expected in bindings.items():
        require(complete.get(key) == expected, f"cream {key} drift")
    result = load_json(paths["result"])
    require(result.get("schema") == CREAM_RESULT_SCHEMA, "cream result schema")
    require(result.get("privileged_inputs_read") is False, "cream privileged input")
    require(result.get("producer_source_sha256") == hashes["producer"], "cream producer bind")
    require(result.get("state_source_sha256") == hashes["state"], "cream state bind")
    frames = {}
    with paths["frames"].open("r", encoding="utf-8") as handle:
        for line in handle:
            frame = json.loads(line)
            require(frame.get("schema") == CREAM_FRAME_SCHEMA, "cream frame schema")
            key = (int(frame["episode_id"]), int(frame["t"]))
            require(key not in frames, f"duplicate cream frame {key}")
            frames[key] = frame
    provenance = {"root": str(root), **{f"{key}_sha256": value for key, value in hashes.items()}}
    return result, frames, provenance


def verify_sam3_runs(
    roots: Sequence[Path], wanted: set[int]
) -> tuple[dict[tuple[int, int], dict[str, Any]], list[dict[str, Any]]]:
    frames = {}
    provenance = []
    for raw_root in roots:
        root = raw_root.expanduser().resolve()
        paths = {
            "result": root / "result.json",
            "frames": root / "frames.jsonl",
            "complete": root / "COMPLETE.json",
            "producer": root / "producer_source.py",
        }
        for path in paths.values():
            require(path.is_file(), f"missing SAM3 artifact: {path}")
        complete = load_json(paths["complete"])
        require(complete.get("schema") == SAM3_COMPLETE_SCHEMA, "SAM3 COMPLETE schema")
        require(complete.get("status") == "complete", "SAM3 artifact incomplete")
        hashes = {name: sha256_file(path) for name, path in paths.items()}
        require(complete.get("result_sha256") == hashes["result"], "SAM3 result drift")
        require(complete.get("frames_sha256") == hashes["frames"], "SAM3 frames drift")
        require(
            complete.get("producer_source_sha256") == hashes["producer"],
            "SAM3 producer drift",
        )
        result = load_json(paths["result"])
        require(result.get("schema") == SAM3_RESULT_SCHEMA, "SAM3 result schema")
        require(result.get("privileged_inputs_read") is False, "SAM3 privileged input")
        selected = 0
        with paths["frames"].open("r", encoding="utf-8") as handle:
            for line in handle:
                frame = json.loads(line)
                require(frame.get("schema") == SAM3_FRAME_SCHEMA, "SAM3 frame schema")
                episode_id = int(frame["episode_id"])
                if episode_id not in wanted:
                    continue
                key = (episode_id, int(frame["t"]))
                require(key not in frames, f"duplicate SAM3 frame {key}")
                frames[key] = frame
                selected += 1
        provenance.append(
            {
                "root": str(root),
                **{f"{key}_sha256": value for key, value in hashes.items()},
                "selected_frames": selected,
            }
        )
    return frames, provenance


def can_observation(frame: Mapping[str, Any]) -> tuple[list[float] | None, list[list[float]]]:
    baskets = frame.get("selection", {}).get("basket", [])
    cans = frame.get("selection", {}).get("can", [])
    basket = baskets[0].get("mask_bbox_xyxy") if len(baskets) == 1 else None
    centers = [item["mask_centroid_xy"] for item in cans] if len(cans) == 2 else []
    return basket, centers


def compose_episode(
    keys: Sequence[tuple[int, int]],
    cream_frames: Mapping[tuple[int, int], dict[str, Any]],
    sam3_frames: Mapping[tuple[int, int], dict[str, Any]],
    config: CanStateConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    counter = CausalCanCount(config)
    records = []
    can_transitions = []
    for frame_index, key in enumerate(keys):
        cream = cream_frames[key]
        sam3 = sam3_frames[key]
        require(cream["frame_id"] == sam3["frame_id"], "cream/SAM3 frame drift")
        basket, centers = can_observation(sam3)
        can_state = counter.step(basket, centers)
        cream_state = cream["causal_cream_state"]
        cream_value = int(bool(cream_state["cream_achieved"]))
        atoms = {
            "can_ge1": int(can_state.sticky_count >= 1),
            "can_ge2": int(can_state.sticky_count >= 2),
            "cream": cream_value,
        }
        if can_state.transition_size:
            can_transitions.extend([key[1]] * can_state.transition_size)
        can_reliable = can_state.raw_count is not None or can_state.sticky_count > 0
        record = {
            "schema": FRAME_SCHEMA,
            "episode_id": key[0],
            "env_seed": int(sam3["env_seed"]),
            "frame_index": frame_index,
            "t": key[1],
            "frame_id": str(sam3["frame_id"]),
            "image_sha256": str(sam3["image_sha256"]),
            "anonymous_can": {
                "raw_count": can_state.raw_count,
                "sticky_count": can_state.sticky_count,
                "used_carry": can_state.used_carry,
                "transition_size": can_state.transition_size,
                "observation_reliable": can_reliable,
            },
            "cream": cream_state,
            "ordinal_atoms": atoms,
            "scalar_0_to_3": sum(atoms.values()),
            "observation_reliable": bool(
                can_reliable and cream_state["observation_reliable"]
            ),
        }
        records.append(record)
    transitions = {
        "can": can_transitions,
        "cream": [
            int(record["t"])
            for record in records
            if record["cream"]["transition_now"]
        ],
    }
    diagnostics = {
        "frames": len(records),
        "can_transition_times": transitions["can"],
        "cream_transition_times": transitions["cream"],
        "terminal_scalar": records[-1]["scalar_0_to_3"],
        "reliable_frames": sum(int(record["observation_reliable"]) for record in records),
        "can_carry_frames": sum(int(record["anonymous_can"]["used_carry"]) for record in records),
    }
    return records, diagnostics


def main() -> int:
    args = parse_args()
    require(args.prefix_check_frames >= 0, "prefix-check-frames must be nonnegative")
    output_dir = args.output_dir.expanduser().resolve()
    require(not output_dir.exists(), f"refusing to overwrite {output_dir}")
    cream_result, cream_frames, cream_provenance = verify_cream_run(args.cream_run)
    episode_ids = [int(value) for value in cream_result["episode_ids"]]
    wanted = set(episode_ids)
    require(len(wanted) == len(episode_ids), "duplicate cream episode ID")
    sam3_frames, sam3_provenance = verify_sam3_runs(args.sam3_run, wanted)
    require(set(cream_frames) == set(sam3_frames), "cream/SAM3 frame surface mismatch")
    config = CanStateConfig()
    output_frames = []
    diagnostics = {}
    prefix_checks = {}
    for episode_id in episode_ids:
        keys = sorted(
            (key for key in cream_frames if key[0] == episode_id), key=lambda key: key[1]
        )
        records, episode_diagnostics = compose_episode(
            keys, cream_frames, sam3_frames, config
        )
        output_frames.extend(records)
        diagnostics[str(episode_id)] = episode_diagnostics
        if args.prefix_check_frames:
            count = min(args.prefix_check_frames, len(keys))
            prefix_records, _ = compose_episode(
                keys[:count], cream_frames, sam3_frames, config
            )
            full_view = records[:count]
            prefix_checks[str(episode_id)] = {
                "frames": count,
                "full_prefix_sha256": canonical_sha256(full_view),
                "prefix_only_sha256": canonical_sha256(prefix_records),
                "exact": full_view == prefix_records,
            }

    source_path = Path(__file__).resolve()
    can_state_path = (REPO / "lcwm" / "v247_can_state.py").resolve()
    result = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "development_composition_complete",
        "claim_scope": "opened seed-8700 development only; not Gate-0 validation",
        "privileged_inputs_read": False,
        "episode_ids": episode_ids,
        "inputs": {"cream": cream_provenance, "sam3": sam3_provenance},
        "can_state_config": config.to_dict(),
        "scalar": "P(can_count>=1)+P(can_count>=2)+P(cream_achieved)",
        "episodes": diagnostics,
        "prefix_checks": prefix_checks,
        "frames_jsonl": "frames.jsonl",
        "producer_source_sha256": sha256_file(source_path),
        "can_state_source_sha256": sha256_file(can_state_path),
    }
    staging = Path(str(output_dir) + f".tmp.{os.getpid()}")
    require(not staging.exists(), f"staging path exists: {staging}")
    staging.mkdir(parents=True)
    try:
        frames_path = staging / "frames.jsonl"
        with frames_path.open("w", encoding="utf-8") as handle:
            for record in output_frames:
                handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        shutil.copy2(source_path, staging / "PRODUCER.py")
        shutil.copy2(can_state_path, staging / "CAN_STATE.py")
        result_path = staging / "RESULT.json"
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        complete = {
            "schema": COMPLETE_SCHEMA,
            "status": "complete",
            "result_sha256": sha256_file(result_path),
            "frames_sha256": sha256_file(frames_path),
            "producer_sha256": sha256_file(staging / "PRODUCER.py"),
            "can_state_source_sha256": sha256_file(staging / "CAN_STATE.py"),
        }
        (staging / "COMPLETE.json").write_text(
            json.dumps(complete, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
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
