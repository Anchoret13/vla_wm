#!/usr/bin/env python3
"""Materialize every RGB boundary of one sealed seed-8700 development replay.

This producer is deliberately narrow.  It accepts only the outcome-sanitized
deployment tape, a passing v245 rich-reencode oracle, and a registration frozen
for episode 12.  It never accepts or opens a deployment summary.  The stored
normalized actions are replayed causally; all 75 pre-action boundaries and the
final N+1 observation at t=750 must match the rich oracle's raw-RGB and proprio
hashes exactly before an atomic output can be committed.

Episode 12 was selected post hoc after an already-open seed-8700 development
summary showed positive cream support.  Consequently this artifact can measure
development recall only and is prohibited from serving as a formal validation
panel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

# These variables must be fixed before importing LIBERO/robosuite.  All caches
# are task-specific, and action-processor resolution is offline-only.
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/v245_cream_rgb_numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/v245_cream_rgb_mpl_cache")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402


REGISTRATION_SCHEMA = "v245_cream_dev_rgb_registration_v1"
MANIFEST_SCHEMA = "v242_rgb_replay_v1"  # compatibility with sealed RGB-only consumers
RESULT_SCHEMA = "v245_cream_dev_rgb_replay_result_v1"
COMPLETE_SCHEMA = "v245_cream_dev_rgb_replay_complete_v1"
SANITIZED_RESULT_SCHEMA = "v245_sanitized_deploy_tape_result_v1"
SANITIZED_COMPLETE_SCHEMA = "v245_sanitized_deploy_tape_complete_v1"
RICH_RESULT_SCHEMA = "v245_rich_reencode_result_v1"
RICH_COMPLETE_SCHEMA = "v245_rich_reencode_complete_v1"
RICH_BOUNDARY_SCHEMA = "v245_rich_reencode_boundary_v1"

EPISODE_ID = 12
ENV_SEED = 8712
SEED_START = 8700
TASK = "chain3_lr2"
C = 10
EXPECTED_ROWS = 75
EXPECTED_STEPS = 750
EXPECTED_BOUNDARIES = 76
PROPRIO_DIM = 25
PROPRIO_KEYS = (
    "robot_state.eef.pos",
    "robot_state.eef.quat",
    "robot_state.gripper.qpos",
    "robot_state.gripper.qvel",
    "robot_state.joints.pos",
    "robot_state.joints.vel",
)
SANITIZED_FIELDS = frozenset(
    {
        "z",
        "u",
        "z_next",
        "episode",
        "t",
        "sigma",
        "task",
        "c",
        "latent_dim",
        "proprio_dim",
        "proprio_keys",
    }
)
CAMERAS = {"agentview": "image", "eye_in_hand": "image2"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path, chunk_size: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    header = canonical_bytes({"dtype": str(tensor.dtype), "shape": list(tensor.shape)})
    return hashlib.sha256(header + tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    header = canonical_bytes({"dtype": array.dtype.str, "shape": list(array.shape)})
    return hashlib.sha256(header + array.tobytes(order="C")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def write_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    write_bytes(path, canonical_bytes(value))


def nested_value(tree: Mapping[str, Any], dotted_key: str) -> Any:
    value: Any = tree
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(f"observation is missing {dotted_key!r}")
        value = value[part]
    return value


def proprio(obs: Mapping[str, Any]) -> torch.Tensor:
    pieces = [
        torch.as_tensor(np.asarray(nested_value(obs, key)), dtype=torch.float32).flatten()
        for key in PROPRIO_KEYS
    ]
    value = torch.cat(pieces).contiguous()
    require(value.shape == (PROPRIO_DIM,), f"invalid proprio shape: {tuple(value.shape)}")
    require(bool(torch.isfinite(value).all()), "proprio contains NaN/Inf")
    return value


def raw_rgb(obs: Mapping[str, Any], key: str) -> np.ndarray:
    pixels = obs.get("pixels")
    require(isinstance(pixels, Mapping), "observation lacks pixels mapping")
    require(key in pixels, f"observation lacks pixels/{key}")
    value = np.asarray(pixels[key])
    require(
        value.dtype == np.uint8 and value.ndim == 3 and value.shape[-1] == 3,
        f"pixels/{key} is not uint8 HWC RGB: {value.dtype} {value.shape}",
    )
    return np.ascontiguousarray(value)


def policy_oriented_rgb(value: np.ndarray) -> np.ndarray:
    # LeRobot's LiberoProcessorStep rotates each camera by 180 degrees.
    return np.ascontiguousarray(value[::-1, ::-1])


def save_jpeg(value: np.ndarray, path: Path, quality: int) -> dict[str, Any]:
    Image.fromarray(value, mode="RGB").save(
        path,
        format="JPEG",
        quality=quality,
        subsampling=0,
        optimize=False,
    )
    height, width = value.shape[:2]
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "width": int(width),
        "height": int(height),
    }


def validate_self_registration(path: Path) -> tuple[dict[str, Any], str]:
    registration = load_json(path)
    require(registration.get("schema") == REGISTRATION_SCHEMA, "registration schema mismatch")
    require(
        registration.get("status") == "frozen_before_rgb_replay",
        "registration was not frozen before RGB replay",
    )
    self_sha = registration.get("self_sha256")
    body = {key: value for key, value in registration.items() if key != "self_sha256"}
    require(is_sha256(self_sha), "registration self_sha256 is invalid")
    require(self_sha == canonical_sha256(body), "registration canonical self hash mismatch")
    require(
        registration.get("producer", {}).get("source_sha256")
        == sha256_file(Path(__file__).resolve()),
        "registered RGB producer source drifted",
    )
    selection = registration.get("development_selection", {})
    require(selection.get("independent_validation") is False, "ep12 must remain development")
    require(selection.get("selection_timing") == "posthoc_after_opened_8700_summary", "bad timing")
    require(
        selection.get("formal_validation_prohibited") is True,
        "formal-validation prohibition missing",
    )
    replay = registration.get("replay", {})
    expected_replay = {
        "episode_id": EPISODE_ID,
        "env_seed": ENV_SEED,
        "seed_start": SEED_START,
        "seed_rule": "seed_start_plus_episode_id",
        "task": TASK,
        "c": C,
        "stored_rows": EXPECTED_ROWS,
        "stored_steps": EXPECTED_STEPS,
        "boundaries": EXPECTED_BOUNDARIES,
        "sampling": "all_pre_action_plus_final_n_plus_1",
        "jpeg_quality": 95,
    }
    require(replay == expected_replay, "registered replay specification mismatch")
    return registration, sha256_file(path)


def validate_sanitized(
    tape_path: Path, registration: dict[str, Any]
) -> tuple[dict[str, Any], list[int], dict[str, str]]:
    root = tape_path.parent
    result_path = root / "RESULT.json"
    complete_path = root / "COMPLETE.json"
    producer_path = root / "PRODUCER.py"
    for path in (tape_path, result_path, complete_path, producer_path):
        require(path.is_file(), f"missing sanitized artifact file: {path}")
    result = load_json(result_path)
    complete = load_json(complete_path)
    hashes = {
        "tape_sha256": sha256_file(tape_path),
        "result_sha256": sha256_file(result_path),
        "complete_sha256": sha256_file(complete_path),
        "producer_sha256": sha256_file(producer_path),
    }
    registered = registration.get("sanitized_source", {})
    for key, actual in hashes.items():
        require(registered.get(key) == actual, f"registered sanitized {key} mismatch")
    require(
        result.get("schema") == SANITIZED_RESULT_SCHEMA and result.get("status") == "PASS",
        "sanitizer RESULT is not passing",
    )
    require(
        complete.get("schema") == SANITIZED_COMPLETE_SCHEMA
        and complete.get("status") == "atomic_success"
        and complete.get("atomic_commit") is True,
        "sanitizer COMPLETE is not an atomic success",
    )
    require(complete.get("sanitized_tape_sha256") == hashes["tape_sha256"], "tape seal")
    require(complete.get("result_sha256") == hashes["result_sha256"], "result seal")
    require(complete.get("producer_snapshot_sha256") == hashes["producer_sha256"], "producer seal")
    require(result.get("source", {}).get("summary_opened") is False, "summary invariant absent")

    tape = torch.load(tape_path, map_location="cpu", weights_only=False)
    require(isinstance(tape, dict) and set(tape) == set(SANITIZED_FIELDS), "tape allow-list")
    require(tape.get("task") == TASK and int(tape.get("c", -1)) == C, "tape task/c mismatch")
    require(int(tape.get("latent_dim", -1)) == 8217, "tape must be the rich8217 oracle")
    require(int(tape.get("proprio_dim", -1)) == PROPRIO_DIM, "proprio dimension mismatch")
    require(tuple(tape.get("proprio_keys", ())) == PROPRIO_KEYS, "proprio keys mismatch")
    for name in ("z", "u", "z_next", "episode", "t", "sigma"):
        require(torch.is_tensor(tape[name]), f"sanitized {name} is not a tensor")
        require(
            result.get("tensor_sha256", {}).get(name) == tensor_sha256(tape[name]),
            f"sanitized {name} tensor seal mismatch",
        )
    rows = [index for index, value in enumerate(tape["episode"]) if int(value) == EPISODE_ID]
    require(len(rows) == EXPECTED_ROWS, f"ep12 has {len(rows)} rows, expected 75")
    require(
        [int(tape["t"][row]) for row in rows] == list(range(0, EXPECTED_STEPS, C)),
        "ep12 time grid is not exactly t0..t740",
    )
    require(tape["u"].shape[1:] == (C, 7), "stored-u shape mismatch")
    require(bool(torch.isfinite(tape["u"][rows]).all()), "stored-u contains NaN/Inf")
    return tape, rows, hashes


def validate_rich_oracle(
    root: Path,
    registration: dict[str, Any],
    tape: dict[str, Any],
    selected_rows: list[int],
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    paths = {
        "result": root / "RESULT.json",
        "complete": root / "COMPLETE.json",
        "boundaries": root / "boundaries.jsonl",
        "rich_tape": root / "rich_tape.pt",
        "producer": root / "PRODUCER.py",
        "registration": root / "REGISTRATION.json",
    }
    for path in paths.values():
        require(path.is_file(), f"missing rich-oracle file: {path}")
    hashes = {f"{name}_sha256": sha256_file(path) for name, path in paths.items()}
    registered = registration.get("rich_oracle", {})
    for key, actual in hashes.items():
        require(registered.get(key) == actual, f"registered rich-oracle {key} mismatch")

    result = load_json(paths["result"])
    complete = load_json(paths["complete"])
    require(
        result.get("schema") == RICH_RESULT_SCHEMA and result.get("status") == "PASS",
        "rich RESULT is not passing",
    )
    require(
        complete.get("schema") == RICH_COMPLETE_SCHEMA
        and complete.get("status") == "atomic_success"
        and complete.get("atomic_commit") is True,
        "rich COMPLETE is not an atomic success",
    )
    seal_map = {
        "result_sha256": hashes["result_sha256"],
        "boundary_manifest_sha256": hashes["boundaries_sha256"],
        "rich_tape_sha256": hashes["rich_tape_sha256"],
        "producer_snapshot_sha256": hashes["producer_sha256"],
        "registration_sha256": hashes["registration_sha256"],
    }
    for key, expected in seal_map.items():
        require(complete.get(key) == expected, f"rich COMPLETE {key} mismatch")
    require(
        result.get("output", {}).get("boundary_manifest_sha256")
        == hashes["boundaries_sha256"],
        "boundary RESULT seal",
    )
    require(
        result.get("output", {}).get("rich_tape_sha256") == hashes["rich_tape_sha256"],
        "rich tape RESULT seal",
    )
    metrics = result.get("replay", {})
    expected_metrics = {
        "episodes": 1,
        "rows": EXPECTED_ROWS,
        "boundaries": EXPECTED_BOUNDARIES,
        "env_steps": EXPECTED_STEPS,
        "oracle": "rich8217_exact",
        "oracle_vectors_compared_exactly": 150,
        "sequence_fields_exact": True,
    }
    for key, expected in expected_metrics.items():
        require(metrics.get(key) == expected, f"rich replay metric {key} mismatch")

    rich_registration = load_json(paths["registration"])
    predecessor = registration.get("predecessor_rich_registration", {})
    require(
        predecessor.get("file_sha256") == hashes["registration_sha256"],
        "predecessor rich registration file SHA mismatch",
    )
    require(
        predecessor.get("canonical_self_sha256") == rich_registration.get("self_sha256"),
        "predecessor canonical self SHA mismatch",
    )
    rich_body = {
        key: value for key, value in rich_registration.items() if key != "self_sha256"
    }
    require(
        canonical_sha256(rich_body) == rich_registration.get("self_sha256"),
        "predecessor rich registration canonical self seal mismatch",
    )
    require(
        rich_registration.get("replay", {}).get("episode_ids") == [EPISODE_ID],
        "rich registration did not select only ep12",
    )
    require(
        rich_registration.get("development_selection", {}).get("independent_validation")
        is False,
        "rich registration must remain development-only",
    )

    rich_tape = torch.load(paths["rich_tape"], map_location="cpu", weights_only=False)
    require(isinstance(rich_tape, dict), "rich_tape.pt is not a dictionary")
    selected_index = torch.tensor(selected_rows, dtype=torch.long)
    for name in ("u", "episode", "t", "sigma"):
        expected = tape[name].index_select(0, selected_index)
        require(torch.equal(rich_tape[name], expected), f"rich oracle changed {name}")
        expected_sha = result.get("output", {}).get("sequence_tensor_sha256", {}).get(name)
        require(tensor_sha256(rich_tape[name]) == expected_sha, f"rich {name} hash mismatch")
    for rich_name, source_name in (("z", "z"), ("z_next", "z_next")):
        expected = tape[source_name].index_select(0, selected_index)
        require(torch.equal(rich_tape[rich_name], expected), f"rich exact oracle {rich_name}")

    boundary_rows = []
    with paths["boundaries"].open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            require(isinstance(value, dict), f"boundary line {line_number} is not an object")
            boundary_rows.append(value)
    require(len(boundary_rows) == EXPECTED_BOUNDARIES, "boundary count mismatch")
    for index, boundary in enumerate(boundary_rows):
        expected_before = selected_rows[index - 1] if index > 0 else None
        expected_after = selected_rows[index] if index < EXPECTED_ROWS else None
        require(boundary.get("schema") == RICH_BOUNDARY_SCHEMA, f"boundary {index} schema")
        require(boundary.get("boundary_index") == index, f"boundary {index} index")
        require(boundary.get("episode_id") == EPISODE_ID, f"boundary {index} episode")
        require(boundary.get("env_seed") == ENV_SEED, f"boundary {index} seed")
        require(boundary.get("t") == index * C, f"boundary {index} time")
        require(boundary.get("source_row_before") == expected_before, f"boundary {index} before")
        require(boundary.get("source_row_after") == expected_after, f"boundary {index} after")
        boundary_hashes = boundary.get("hashes", {})
        require(is_sha256(boundary_hashes.get("proprio_sha256")), f"boundary {index} proprio")
        rgb_hashes = boundary_hashes.get("raw_rgb_sha256")
        require(isinstance(rgb_hashes, dict) and set(rgb_hashes) == {"image", "image2"}, "RGB keys")
        require(all(is_sha256(value) for value in rgb_hashes.values()), "RGB hash invalid")
        require(
            boundary_hashes.get("raw_rgb_aggregate_sha256") == canonical_sha256(rgb_hashes),
            f"boundary {index} aggregate RGB hash mismatch",
        )
    return boundary_rows, hashes, result


def verify_policy_snapshot(registration: dict[str, Any]) -> str:
    from huggingface_hub import snapshot_download

    spec = registration.get("action_decoder", {}).get("policy_snapshot")
    require(isinstance(spec, dict), "registered policy snapshot missing")
    root = Path(
        snapshot_download(
            repo_id=spec["repo_id"],
            revision=spec["revision"],
            local_files_only=True,
        )
    ).resolve()
    require(root.name == spec["revision"], "resolved policy revision mismatch")
    for relative, expected_sha in spec.get("files_sha256", {}).items():
        path = root / relative
        require(path.is_file(), f"missing policy artifact: {path}")
        require(sha256_file(path) == expected_sha, f"policy artifact drift: {relative}")
    return root.as_posix()


class ReplayActionPostprocessor:
    """Decode stored normalized actions without generating any new action."""

    def __init__(self, model_path: str) -> None:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
        from lerobot.envs.factory import make_env_pre_post_processors
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.processor import (
            PolicyProcessorPipeline,
            policy_action_to_transition,
            transition_to_policy_action,
        )
        from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME

        config = PreTrainedConfig.from_pretrained(model_path)
        require(isinstance(config, PI05Config), "registered action decoder is not PI05")
        config.device = "cpu"
        self.policy_postprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=model_path,
            config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )
        _, self.env_postprocessor = make_env_pre_post_processors(
            env_cfg=LiberoEnvConfig(task="libero_10"), policy_cfg=config
        )

    @torch.no_grad()
    def chunk_to_env(self, chunk: torch.Tensor) -> np.ndarray:
        value = torch.as_tensor(chunk, dtype=torch.float32, device="cpu")
        require(value.shape == (C, 7), f"stored chunk shape drifted: {tuple(value.shape)}")
        decoded_actions = []
        for action in value:
            decoded = self.policy_postprocessor(action.unsqueeze(0))
            transition = self.env_postprocessor({"action": decoded})
            output = transition["action"]
            array = output[0].detach().cpu().numpy() if torch.is_tensor(output) else output[0]
            array = np.asarray(array, dtype=np.float32)
            require(array.shape == (7,) and np.isfinite(array).all(), "bad env action")
            decoded_actions.append(array)
        return np.stack(decoded_actions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanitized-tape", type=Path, required=True)
    parser.add_argument("--rich-oracle", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def capture_boundary(
    *,
    obs: Mapping[str, Any],
    boundary: dict[str, Any],
    tape: dict[str, Any],
    stage: Path,
    tape_path: Path,
    tape_sha256: str,
    rich_oracle_path: Path,
) -> tuple[dict[str, Any], float]:
    boundary_index = int(boundary["boundary_index"])
    t_value = int(boundary["t"])
    before = boundary["source_row_before"]
    after = boundary["source_row_after"]
    observed = proprio(obs)
    expected_vectors = []
    if after is not None:
        expected_vectors.append(("z", tape["z"][after, -PROPRIO_DIM:].float().cpu()))
    if before is not None:
        expected_vectors.append(("z_next", tape["z_next"][before, -PROPRIO_DIM:].float().cpu()))
    require(expected_vectors, f"boundary t{t_value} has no proprio oracle")
    max_error = 0.0
    for side, expected in expected_vectors:
        difference = (observed - expected).abs()
        side_error = float(difference.max())
        max_error = max(max_error, side_error)
        require(
            torch.equal(observed, expected),
            f"proprio exact mismatch at t={t_value} against {side}: max={side_error:.9g}",
        )
    observed_proprio_sha = tensor_sha256(observed)
    require(
        observed_proprio_sha == boundary["hashes"]["proprio_sha256"],
        f"proprio hash mismatch at t={t_value}",
    )

    image_records: dict[str, dict[str, Any]] = {}
    observed_raw_hashes: dict[str, str] = {}
    for camera, obs_key in CAMERAS.items():
        pixels = raw_rgb(obs, obs_key)
        raw_hash = array_sha256(pixels)
        expected_hash = boundary["hashes"]["raw_rgb_sha256"][obs_key]
        require(raw_hash == expected_hash, f"raw RGB exact mismatch at t={t_value} {camera}")
        observed_raw_hashes[obs_key] = raw_hash
        filename = f"ep{EPISODE_ID:04d}_t{t_value:04d}_{camera}.jpg"
        image_path = stage / "images" / filename
        metadata = save_jpeg(policy_oriented_rgb(pixels), image_path, quality=95)
        metadata["path"] = image_path.relative_to(stage).as_posix()
        metadata["raw_rgb_sha256"] = raw_hash
        image_records[camera] = metadata
    aggregate = canonical_sha256(observed_raw_hashes)
    require(
        aggregate == boundary["hashes"]["raw_rgb_aggregate_sha256"],
        f"raw RGB aggregate mismatch at t={t_value}",
    )
    source_row = after if after is not None else before
    frame_kind = "pre_action" if after is not None else "terminal_n_plus_1"
    manifest_row = {
        "schema": MANIFEST_SCHEMA,
        "producer_schema": RESULT_SCHEMA,
        "frame_id": boundary["boundary_id"],
        "frame_kind": frame_kind,
        "task": TASK,
        "tape_path": tape_path.as_posix(),
        "tape_sha256": tape_sha256,
        "rich_oracle_path": rich_oracle_path.as_posix(),
        "rich_boundary_id": boundary["boundary_id"],
        "episode_id": EPISODE_ID,
        "env_seed": ENV_SEED,
        "t": t_value,
        "row_index": int(source_row),
        "chunk_index": boundary_index,
        "source_row_before": before,
        "source_row_after": after,
        "images": image_records,
        "raw_rgb": {
            "sha256": observed_raw_hashes,
            "aggregate_sha256": aggregate,
        },
        "proprio": {
            "sha256": observed_proprio_sha,
            "exact": True,
            "max_abs_error": max_error,
            "oracles": [side for side, _expected in expected_vectors],
        },
        "sampling": {
            "chunk_stride": 1,
            "pre_action": after is not None,
            "terminal_n_plus_1": after is None,
            "outcome_blind_within_registered_episode": True,
        },
    }
    return manifest_row, max_error


def run_replay(
    *,
    registration: dict[str, Any],
    registration_path: Path,
    registration_sha256: str,
    tape: dict[str, Any],
    selected_rows: list[int],
    tape_path: Path,
    sanitized_hashes: dict[str, str],
    rich_root: Path,
    rich_boundaries: list[dict[str, Any]],
    rich_hashes: dict[str, str],
    rich_result: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    from lcwm.libero_paths import ensure_project_libero_config

    ensure_project_libero_config()
    from lcwm.v080_bench import make_v080_env

    model_path = verify_policy_snapshot(registration)
    decoder = ReplayActionPostprocessor(model_path)
    require(not output_path.exists(), f"refusing to overwrite {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.partial-", dir=output_path.parent))
    env = None
    try:
        (stage / "images").mkdir()
        torch.manual_seed(ENV_SEED)
        np.random.seed(ENV_SEED)
        env = make_v080_env(TASK)
        obs, _reset_info = env.reset(seed=ENV_SEED)
        manifest_rows = []
        max_errors = []
        env_steps = 0
        termination_records = []
        for boundary_index, boundary in enumerate(rich_boundaries):
            manifest_row, max_error = capture_boundary(
                obs=obs,
                boundary=boundary,
                tape=tape,
                stage=stage,
                tape_path=tape_path,
                tape_sha256=sanitized_hashes["tape_sha256"],
                rich_oracle_path=rich_root,
            )
            manifest_rows.append(manifest_row)
            max_errors.append(max_error)
            if boundary_index == EXPECTED_ROWS:
                break
            row_index = selected_rows[boundary_index]
            require(row_index == boundary["source_row_after"], "stored-u row binding mismatch")
            env_actions = decoder.chunk_to_env(tape["u"][row_index])
            for action_offset, action in enumerate(env_actions):
                obs, _reward, terminated, truncated, _info = env.step(action)
                env_steps += 1
                if terminated or truncated:
                    termination_records.append(
                        {
                            "t": boundary_index * C + action_offset + 1,
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                        }
                    )
                    raise RuntimeError(
                        "environment terminated inside the 750 stored-u steps at "
                        f"t={boundary_index * C + action_offset + 1}"
                    )
        require(len(manifest_rows) == EXPECTED_BOUNDARIES, "did not capture 76 boundaries")
        require(env_steps == EXPECTED_STEPS, "did not execute exactly 750 stored-u steps")
        require(manifest_rows[-1]["frame_kind"] == "terminal_n_plus_1", "missing N+1 frame")
        manifest_path = stage / "manifest.jsonl"
        write_bytes(manifest_path, b"".join(canonical_bytes(row) for row in manifest_rows))

        snapshots = {
            "PRODUCER.py": Path(__file__).resolve(),
            "REGISTRATION.json": registration_path,
            "SANITIZER_RESULT.json": tape_path.parent / "RESULT.json",
            "SANITIZER_COMPLETE.json": tape_path.parent / "COMPLETE.json",
            "RICH_ORACLE_RESULT.json": rich_root / "RESULT.json",
            "RICH_ORACLE_COMPLETE.json": rich_root / "COMPLETE.json",
            "RICH_ORACLE_BOUNDARIES.jsonl": rich_root / "boundaries.jsonl",
        }
        snapshot_hashes = {}
        for name, source in snapshots.items():
            destination = stage / name
            write_bytes(destination, source.read_bytes())
            snapshot_hashes[name] = sha256_file(destination)
            require(snapshot_hashes[name] == sha256_file(source), f"snapshot changed: {name}")

        created_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        result = {
            "schema": RESULT_SCHEMA,
            "status": "PASS",
            "created_utc": created_utc,
            "development_selection": {
                "independent_validation": False,
                "posthoc_after_opened_8700_summary": True,
                "formal_validation_prohibited": True,
                "claim_scope": "seed-8700 development cream recall support only",
            },
            "inputs": {
                "summary_opened_by_producer": False,
                "sanitized": sanitized_hashes,
                "rich_oracle": rich_hashes,
                "registration_sha256": registration_sha256,
                "registration_canonical_self_sha256": registration["self_sha256"],
                "predecessor_rich_registration": registration[
                    "predecessor_rich_registration"
                ],
            },
            "replay": {
                "episode_id": EPISODE_ID,
                "env_seed": ENV_SEED,
                "rows": EXPECTED_ROWS,
                "boundaries": len(manifest_rows),
                "env_steps": env_steps,
                "stored_u_only": True,
                "new_policy_actions_generated": False,
                "termination_records": termination_records,
                "frame_kinds": {
                    "pre_action": EXPECTED_ROWS,
                    "terminal_n_plus_1": 1,
                },
            },
            "exact_oracle": {
                "rich_reencode_oracle": rich_result["replay"]["oracle"],
                "rich_vectors_compared_previously": rich_result["replay"][
                    "oracle_vectors_compared_exactly"
                ],
                "proprio_boundaries_exact": len(manifest_rows),
                "raw_rgb_boundaries_exact": len(manifest_rows),
                "raw_rgb_camera_arrays_exact": len(manifest_rows) * len(CAMERAS),
                "max_proprio_abs_error": max(max_errors, default=0.0),
            },
            "output": {
                "manifest_sha256": sha256_file(manifest_path),
                "frames": len(manifest_rows),
                "jpeg_files": len(manifest_rows) * len(CAMERAS),
                "jpeg_quality": 95,
                "orientation": "rotated 180 degrees to match LeRobot LiberoProcessorStep",
                "snapshot_sha256": snapshot_hashes,
            },
            "action_decoder": {
                "policy_snapshot": registration["action_decoder"]["policy_snapshot"],
                "implementation": (
                    "LeRobot policy_postprocessor plus LIBERO env_postprocessor; "
                    "no VLA weights loaded for inference and no action generation"
                ),
            },
            "argv": list(sys.argv),
        }
        result_path = stage / "RESULT.json"
        write_json(result_path, result)
        complete = {
            "schema": COMPLETE_SCHEMA,
            "status": "atomic_success",
            "atomic_commit": True,
            "registration_sha256": registration_sha256,
            "sanitized_tape_sha256": sanitized_hashes["tape_sha256"],
            "rich_oracle_result_sha256": rich_hashes["result_sha256"],
            "rich_oracle_boundaries_sha256": rich_hashes["boundaries_sha256"],
            "manifest_sha256": result["output"]["manifest_sha256"],
            "producer_snapshot_sha256": snapshot_hashes["PRODUCER.py"],
            "result_sha256": sha256_file(result_path),
        }
        write_json(stage / "COMPLETE.json", complete)
        os.replace(stage, output_path)
        return {
            "output": output_path.as_posix(),
            "registration_sha256": registration_sha256,
            "manifest_sha256": result["output"]["manifest_sha256"],
            "frames": len(manifest_rows),
            "env_steps": env_steps,
            "raw_rgb_arrays_exact": len(manifest_rows) * len(CAMERAS),
            "max_proprio_abs_error": max(max_errors, default=0.0),
        }
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    finally:
        if env is not None:
            env.close()


def main() -> int:
    args = parse_args()
    tape_path = args.sanitized_tape.expanduser().resolve()
    rich_root = args.rich_oracle.expanduser().resolve()
    registration_path = args.registration.expanduser().resolve()
    output_path = args.out.expanduser().resolve()
    require(tape_path.is_file(), f"missing sanitized tape: {tape_path}")
    require(rich_root.is_dir(), f"missing rich oracle: {rich_root}")
    require(registration_path.is_file(), f"missing registration: {registration_path}")
    registration, registration_sha = validate_self_registration(registration_path)
    tape, selected_rows, sanitized_hashes = validate_sanitized(tape_path, registration)
    boundaries, rich_hashes, rich_result = validate_rich_oracle(
        rich_root, registration, tape, selected_rows
    )
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "VALID",
                    "registration_sha256": registration_sha,
                    "sanitized_tape_sha256": sanitized_hashes["tape_sha256"],
                    "rich_oracle_result_sha256": rich_hashes["result_sha256"],
                    "boundaries_validated": len(boundaries),
                    "simulator_imported": False,
                    "output_created": False,
                },
                indent=2,
            )
        )
        return 0
    output = run_replay(
        registration=registration,
        registration_path=registration_path,
        registration_sha256=registration_sha,
        tape=tape,
        selected_rows=selected_rows,
        tape_path=tape_path,
        sanitized_hashes=sanitized_hashes,
        rich_root=rich_root,
        rich_boundaries=boundaries,
        rich_hashes=rich_hashes,
        rich_result=rich_result,
        output_path=output_path,
    )
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
