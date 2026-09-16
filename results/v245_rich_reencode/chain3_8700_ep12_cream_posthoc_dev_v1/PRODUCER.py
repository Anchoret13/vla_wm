#!/usr/bin/env python
"""Deterministically replay a sanitized deployment tape into canonical 8217-d state.

This is stage 2 of the rich-latent replay.  The only data inputs are a sealed
stage-1 sanitized tape and a content-addressed registration frozen before the
first simulator reset.  Seeds come exclusively from the registered rule
``seed_start + episode_id``; no deployment summary or outcome artifact is
accepted by this command.

For every selected episode with N stored chunks, the tool recreates N+1 direct
observations, executes the stored normalized actions through the original
Pi0.5 postprocessors, and runs the original full-prompt preprocessing and
``prefix_forward``.  At each boundary it computes both:

* legacy 2073-d = masked mean of the full prefix + 25-d proprioception;
* rich 8217-d = per-camera mean/max over [0:256]/[256:512] + proprioception.

The development oracle requires exact rich equality for an existing 8217-d
tape.  A legacy-panel registration requires exact legacy equality before any
recomputed rich tape can be committed.  Any mismatch raises before the atomic
output directory (and therefore before ``COMPLETE.json``) exists.

Registration schema (``v245_rich_reencode_registration_v1``):

``source`` binds the sanitized tape/result/COMPLETE hashes and its task, c,
latent dimension, and episode count. ``replay`` freezes purpose, episode IDs,
seed start/rule, oracle, device, token blocks, policy and tokenizer snapshots,
and runtime versions. ``provenance.source_files_sha256`` binds this encoder and
the exact collector/chassis/feature/env/BDDL sources. ``self_sha256`` is the
canonical JSON SHA-256 of every other field.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

# These settings precede all LIBERO/robosuite/Hugging Face imports.  The
# registered snapshots must already be local; replay never contacts the Hub.
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/v245_rich_numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/v245_rich_mpl_cache")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import torch


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

SANITIZED_RESULT_SCHEMA = "v245_sanitized_deploy_tape_result_v1"
SANITIZED_COMPLETE_SCHEMA = "v245_sanitized_deploy_tape_complete_v1"
REGISTRATION_SCHEMA = "v245_rich_reencode_registration_v1"
RESULT_SCHEMA = "v245_rich_reencode_result_v1"
COMPLETE_SCHEMA = "v245_rich_reencode_complete_v1"
BOUNDARY_SCHEMA = "v245_rich_reencode_boundary_v1"

TENSOR_FIELDS = ("z", "u", "z_next", "episode", "t", "sigma")
METADATA_FIELDS = ("task", "c", "latent_dim", "proprio_dim", "proprio_keys")
SANITIZED_FIELDS = frozenset((*TENSOR_FIELDS, *METADATA_FIELDS))
SEQUENCE_FIELDS = ("u", "episode", "t", "sigma")
PROPRIO_DIM = 25
PROPRIO_KEYS = (
    "robot_state.eef.pos",
    "robot_state.eef.quat",
    "robot_state.gripper.qpos",
    "robot_state.gripper.qvel",
    "robot_state.joints.pos",
    "robot_state.joints.vel",
)
LEGACY_DIM = 2048 + PROPRIO_DIM
RICH_DIM = 4 * 2048 + PROPRIO_DIM
CAMERA_TOKEN_BLOCKS = ((0, 256), (256, 512))
RAW_RGB_KEYS = ("image", "image2")

REQUIRED_MODEL_FILES = frozenset(
    {
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_preprocessor_step_2_normalizer_processor.safetensors",
        "policy_postprocessor.json",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    }
)
REQUIRED_TOKENIZER_FILES = frozenset(
    {
        "added_tokens.json",
        "config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
)
REQUIRED_RUNTIME_KEYS = frozenset(
    {"python", "torch", "numpy", "lerobot", "libero", "robosuite", "mujoco", "transformers"}
)


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
    raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
    return hashlib.sha256(header + raw).hexdigest()


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    header = canonical_bytes({"dtype": array.dtype.str, "shape": list(array.shape)})
    return hashlib.sha256(header + array.tobytes(order="C")).hexdigest()


def _write_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    _write_bytes(path, canonical_bytes(value))


def _save_torch(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def nested_value(tree: Mapping[str, Any], dotted_key: str) -> Any:
    value: Any = tree
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(f"observation is missing {dotted_key!r}")
        value = value[part]
    return value


def proprio(obs: Mapping[str, Any]) -> torch.Tensor:
    parts = [
        torch.as_tensor(np.asarray(nested_value(obs, key)), dtype=torch.float32).flatten()
        for key in PROPRIO_KEYS
    ]
    value = torch.cat(parts)
    if value.shape != (PROPRIO_DIM,) or not bool(torch.isfinite(value).all()):
        raise ValueError(f"invalid direct-observation proprio: {tuple(value.shape)}")
    return value


def flatten_arrays(value: Any, prefix: str = "") -> list[tuple[str, np.ndarray]]:
    if isinstance(value, Mapping):
        arrays: list[tuple[str, np.ndarray]] = []
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            arrays.extend(flatten_arrays(value[key], child))
        return arrays
    if torch.is_tensor(value):
        return [(prefix, value.detach().cpu().contiguous().numpy())]
    if isinstance(value, (np.ndarray, int, float, bool, np.number)):
        return [(prefix, np.asarray(value))]
    return []


def named_arrays_sha256(items: Iterable[tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    count = 0
    for name, value in sorted(items, key=lambda item: item[0]):
        array = np.ascontiguousarray(np.asarray(value))
        descriptor = {
            "name": name,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
        }
        digest.update(canonical_bytes(descriptor))
        digest.update(array.tobytes(order="C"))
        count += 1
    if count == 0:
        raise ValueError("cannot hash an empty named-array collection")
    return digest.hexdigest()


def git_info(path: Path) -> dict[str, Any]:
    current = path.resolve()
    if current.is_file():
        current = current.parent
    root: Path | None = None
    for candidate in (current, *current.parents):
        probe = subprocess.run(
            ["git", "-C", str(candidate), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode == 0 and probe.stdout.strip():
            root = Path(probe.stdout.strip()).resolve()
            break
    if root is None:
        return {"repository": None, "commit": None, "dirty": None}
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return {"repository": root.as_posix(), "commit": head or None, "dirty": bool(status)}


@dataclass(frozen=True)
class SanitizedBundle:
    tape_path: Path
    tape: dict[str, Any]
    result: dict[str, Any]
    complete: dict[str, Any]
    tape_sha256: str
    result_sha256: str
    complete_sha256: str
    episode_rows: dict[int, list[int]]


def validate_sanitized(tape_path: Path) -> SanitizedBundle:
    root = tape_path.parent
    result_path = root / "RESULT.json"
    complete_path = root / "COMPLETE.json"
    producer_path = root / "PRODUCER.py"
    for path in (tape_path, result_path, complete_path, producer_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    result = load_json(result_path)
    complete = load_json(complete_path)
    tape_sha = sha256_file(tape_path)
    result_sha = sha256_file(result_path)
    complete_sha = sha256_file(complete_path)
    if result.get("schema") != SANITIZED_RESULT_SCHEMA or result.get("status") != "PASS":
        raise ValueError("sanitizer RESULT is not a passing v245 artifact")
    if complete.get("schema") != SANITIZED_COMPLETE_SCHEMA:
        raise ValueError("sanitizer COMPLETE schema mismatch")
    if complete.get("status") != "atomic_success" or complete.get("atomic_commit") is not True:
        raise ValueError("sanitizer artifact is not an atomic success")
    expected_seal = {
        "sanitized_tape_sha256": tape_sha,
        "result_sha256": result_sha,
        "producer_snapshot_sha256": sha256_file(producer_path),
    }
    for key, expected in expected_seal.items():
        if complete.get(key) != expected or result.get(key) not in (None, expected):
            raise ValueError(f"sanitizer seal mismatch for {key}")
    if result.get("source", {}).get("summary_opened") is not False:
        raise ValueError("sanitizer RESULT lacks the no-summary invariant")

    tape = torch.load(tape_path, map_location="cpu", weights_only=False)
    if not isinstance(tape, dict) or set(tape) != set(SANITIZED_FIELDS):
        raise ValueError("sanitized tape field allow-list mismatch")
    if result.get("field_policy", {}).get("kept") != sorted(SANITIZED_FIELDS):
        raise ValueError("sanitizer RESULT field policy mismatch")
    for name in TENSOR_FIELDS:
        if not torch.is_tensor(tape[name]):
            raise TypeError(f"sanitized {name} must be a tensor")
        if result.get("tensor_sha256", {}).get(name) != tensor_sha256(tape[name]):
            raise ValueError(f"sanitized tensor hash mismatch: {name}")

    z, z_next, u = tape["z"], tape["z_next"], tape["u"]
    if z.ndim != 2 or len(z) == 0 or z_next.shape != z.shape:
        raise ValueError("invalid sanitized z/z_next shapes")
    rows, latent_dim = z.shape
    c = int(tape["c"])
    if u.shape != (rows, c, 7) or int(tape["latent_dim"]) != latent_dim:
        raise ValueError("invalid sanitized u or latent metadata")
    if int(tape["proprio_dim"]) != PROPRIO_DIM:
        raise ValueError("sanitized proprio_dim mismatch")
    if tuple(tape["proprio_keys"]) != PROPRIO_KEYS:
        raise ValueError("sanitized proprio_keys mismatch")
    for name in ("episode", "t", "sigma"):
        if tape[name].ndim != 1 or len(tape[name]) != rows:
            raise ValueError(f"sanitized {name} shape mismatch")

    episode_rows: dict[int, list[int]] = {}
    previous_episode = -1
    for row in range(rows):
        episode_id = int(tape["episode"][row])
        t_value = int(tape["t"][row])
        if episode_id < previous_episode:
            raise ValueError("sanitized episode rows are interleaved")
        previous_episode = episode_id
        episode_rows.setdefault(episode_id, []).append(row)
        if t_value != (len(episode_rows[episode_id]) - 1) * c:
            raise ValueError(f"episode {episode_id} has non-contiguous t at row {row}")
    if sorted(episode_rows) != list(range(len(episode_rows))):
        raise ValueError("sanitized episode ids must be contiguous from zero")
    validation = result.get("validation", {})
    if validation.get("rows") != rows or validation.get("episodes") != len(episode_rows):
        raise ValueError("sanitizer validation counts do not match tape")
    return SanitizedBundle(
        tape_path=tape_path,
        tape=tape,
        result=result,
        complete=complete,
        tape_sha256=tape_sha,
        result_sha256=result_sha,
        complete_sha256=complete_sha,
        episode_rows=episode_rows,
    )


def _required_source_paths(task: str) -> frozenset[str]:
    return frozenset(
        {
            "scripts/reencode_v245_rich_tape.py",
            "scripts/collect_v121_deploy_latents.py",
            "lcwm/chassis.py",
            "lcwm/sampler.py",
            "lcwm/v082_m0.py",
            "lcwm/v080_bench.py",
            f"bddl/chains/{task}.bddl",
        }
    )


def validate_registration(
    path: Path, bundle: SanitizedBundle
) -> tuple[dict[str, Any], str]:
    registration = load_json(path)
    registration_sha = sha256_file(path)
    if registration.get("schema") != REGISTRATION_SCHEMA:
        raise ValueError("registration schema mismatch")
    if registration.get("status") != "frozen_before_replay":
        raise ValueError("registration was not frozen before replay")
    self_sha = registration.get("self_sha256")
    body = {key: value for key, value in registration.items() if key != "self_sha256"}
    if not is_sha256(self_sha) or self_sha != canonical_sha256(body):
        raise ValueError("registration self_sha256 mismatch")

    source = registration.get("source")
    if not isinstance(source, dict):
        raise ValueError("registration.source must be an object")
    source_expected = {
        "sanitized_tape_sha256": bundle.tape_sha256,
        "sanitizer_result_sha256": bundle.result_sha256,
        "sanitizer_complete_sha256": bundle.complete_sha256,
        "task": bundle.tape["task"],
        "c": int(bundle.tape["c"]),
        "source_latent_dim": int(bundle.tape["latent_dim"]),
        "source_episode_count": len(bundle.episode_rows),
    }
    for key, expected in source_expected.items():
        if source.get(key) != expected:
            raise ValueError(f"registration.source.{key} mismatch")

    replay = registration.get("replay")
    if not isinstance(replay, dict):
        raise ValueError("registration.replay must be an object")
    purpose = replay.get("purpose")
    if purpose not in {"development_calibration", "formal_panel_reencode"}:
        raise ValueError("registration replay purpose is invalid")
    if replay.get("seed_rule") != "seed_start_plus_episode_id":
        raise ValueError("registration seed rule mismatch")
    if not isinstance(replay.get("seed_start"), int) or replay["seed_start"] < 0:
        raise ValueError("registration seed_start must be a non-negative integer")
    selected = replay.get("episode_ids")
    if (
        not isinstance(selected, list)
        or not selected
        or any(not isinstance(value, int) for value in selected)
        or selected != sorted(set(selected))
    ):
        raise ValueError("registration episode_ids must be non-empty, sorted, and unique")
    unknown = sorted(set(selected) - set(bundle.episode_rows))
    if unknown:
        raise ValueError(f"registration selects unknown episodes: {unknown}")
    if purpose == "formal_panel_reencode" and selected != sorted(bundle.episode_rows):
        raise ValueError("formal reencode must include every sanitized episode")
    oracle = replay.get("oracle")
    if oracle not in {"rich8217_exact", "legacy2073_exact"}:
        raise ValueError("registration oracle must be rich8217_exact or legacy2073_exact")
    expected_source_dim = RICH_DIM if oracle == "rich8217_exact" else LEGACY_DIM
    if int(bundle.tape["latent_dim"]) != expected_source_dim:
        raise ValueError("source latent dimension is incompatible with registered oracle")
    if replay.get("legacy_dim") != LEGACY_DIM or replay.get("rich_dim") != RICH_DIM:
        raise ValueError("registration canonical latent dimensions mismatch")
    if replay.get("camera_token_blocks") != [list(value) for value in CAMERA_TOKEN_BLOCKS]:
        raise ValueError("registration camera token blocks mismatch")
    if replay.get("suite_name") != "libero_10":
        raise ValueError("registration suite_name must be libero_10")
    if replay.get("device") != "cuda":
        raise ValueError("canonical exact replay currently requires device=cuda")

    provenance = registration.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("registration.provenance must be an object")
    source_hashes = provenance.get("source_files_sha256")
    required_paths = _required_source_paths(str(bundle.tape["task"]))
    if not isinstance(source_hashes, dict) or set(source_hashes) != set(required_paths):
        raise ValueError("registration source-file closure mismatch")
    for relative, expected_sha in source_hashes.items():
        candidate = (REPO / relative).resolve()
        try:
            candidate.relative_to(REPO)
        except ValueError as exc:
            raise ValueError(f"registration source escapes repository: {relative}") from exc
        if not is_sha256(expected_sha) or not candidate.is_file():
            raise ValueError(f"registration source is invalid: {relative}")
        actual_sha = sha256_file(candidate)
        if actual_sha != expected_sha:
            raise ValueError(f"registered source drift: {relative}")

    runtime = replay.get("runtime_versions")
    if not isinstance(runtime, dict) or set(runtime) != set(REQUIRED_RUNTIME_KEYS):
        raise ValueError("registration runtime version closure mismatch")
    validate_artifact_registration(replay.get("policy_snapshot"), REQUIRED_MODEL_FILES)
    validate_artifact_registration(replay.get("tokenizer_snapshot"), REQUIRED_TOKENIZER_FILES)
    return registration, registration_sha


def validate_artifact_registration(value: Any, required_files: frozenset[str]) -> None:
    if not isinstance(value, dict):
        raise ValueError("registered Hugging Face artifact must be an object")
    if not isinstance(value.get("repo_id"), str) or not value["repo_id"]:
        raise ValueError("registered artifact repo_id is invalid")
    revision = value.get("revision")
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("registered artifact revision must be a 40-character commit")
    files = value.get("files_sha256")
    if not isinstance(files, dict) or set(files) != set(required_files):
        raise ValueError("registered artifact file closure mismatch")
    if any(not is_sha256(digest) for digest in files.values()):
        raise ValueError("registered artifact contains an invalid file SHA")


def resolve_and_verify_artifact(spec: dict[str, Any]) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    root = Path(
        snapshot_download(
            repo_id=spec["repo_id"],
            revision=spec["revision"],
            local_files_only=True,
        )
    ).resolve()
    if root.name != spec["revision"]:
        raise ValueError(f"resolved snapshot revision mismatch: {root}")
    actual: dict[str, str] = {}
    for name, expected in sorted(spec["files_sha256"].items()):
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual[name] = sha256_file(path.resolve())
        if actual[name] != expected:
            raise ValueError(f"registered artifact drift: {spec['repo_id']}:{name}")
    refs_main = root.parent.parent / "refs" / "main"
    main_value = refs_main.read_text().strip() if refs_main.is_file() else None
    if spec.get("require_main_ref", False) and main_value != spec["revision"]:
        raise ValueError(f"local main ref does not select registered snapshot: {spec['repo_id']}")
    return {
        "repo_id": spec["repo_id"],
        "revision": spec["revision"],
        "resolved_path": root.as_posix(),
        "files_sha256": actual,
        "main_ref": main_value,
    }


def module_runtime() -> tuple[dict[str, str], dict[str, Any]]:
    import lerobot
    import libero
    import mujoco
    import robosuite
    import transformers

    modules = {
        "lerobot": lerobot,
        "libero": libero,
        "robosuite": robosuite,
        "mujoco": mujoco,
        "transformers": transformers,
    }
    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        **{
            name: str(getattr(module, "__version__", "unknown"))
            for name, module in modules.items()
        },
    }
    provenance: dict[str, Any] = {}
    for name, module in modules.items():
        raw_path = getattr(module, "__file__", None)
        module_path = Path(raw_path).resolve() if raw_path else None
        provenance[name] = {
            "version": versions[name],
            "module_path": module_path.as_posix() if module_path else None,
            "module_file_sha256": sha256_file(module_path) if module_path else None,
            "git": git_info(module_path) if module_path else None,
        }
    return versions, provenance


def direct_features(
    runner: Any, obs: Mapping[str, Any], task_description: str
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    from lcwm.sampler import prefix_forward
    from lcwm.v082_m0 import masked_prefix_mean
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    policy_batch = runner._obs_to_policy_batch(obs, task_description)
    with torch.no_grad():
        prefix = prefix_forward(runner.policy, policy_batch)
        hidden = prefix.hidden[0].detach().float().cpu().contiguous()
        mask = prefix.pad_masks[0].detach().cpu().bool().contiguous()
        legacy_visual = masked_prefix_mean(hidden, mask.float())
        rich_parts: list[torch.Tensor] = []
        for lo, hi in CAMERA_TOKEN_BLOCKS:
            block = hidden[lo:hi][mask[lo:hi]]
            if len(block) == 0:
                block = hidden[lo:hi]
            rich_parts.extend((block.mean(0), block.max(0).values))
        physical = proprio(obs)
        legacy = torch.cat((legacy_visual, physical)).contiguous()
        rich = torch.cat((*rich_parts, physical)).contiguous()
    if legacy.shape != (LEGACY_DIM,) or rich.shape != (RICH_DIM,):
        raise ValueError(
            f"unexpected feature shapes: legacy={tuple(legacy.shape)}, rich={tuple(rich.shape)}"
        )
    if hidden.ndim != 2 or hidden.shape[1] != 2048 or len(mask) != len(hidden):
        raise ValueError(f"unexpected prefix layout: {tuple(hidden.shape)}, {tuple(mask.shape)}")
    if prefix.n_img_tokens < CAMERA_TOKEN_BLOCKS[-1][1]:
        raise ValueError(f"prefix has only {prefix.n_img_tokens} registered image tokens")

    pixels = obs.get("pixels")
    if not isinstance(pixels, Mapping):
        raise ValueError("direct observation lacks pixels mapping")
    rgb_hashes = {}
    for key in RAW_RGB_KEYS:
        if key not in pixels:
            raise ValueError(f"direct observation lacks pixels/{key}")
        array = np.asarray(pixels[key])
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(f"pixels/{key} is not uint8 HWC RGB: {array.dtype} {array.shape}")
        rgb_hashes[key] = array_sha256(array)
    robot_state = obs.get("robot_state")
    if not isinstance(robot_state, Mapping):
        raise ValueError("direct observation lacks robot_state mapping")
    state_hash = named_arrays_sha256(flatten_arrays(robot_state, "robot_state"))
    policy_state_items = [
        (name, array)
        for name, array in flatten_arrays(policy_batch)
        if "state" in name.lower()
    ]
    language_tokens = policy_batch[OBS_LANGUAGE_TOKENS].detach().cpu().contiguous()
    language_mask = policy_batch[OBS_LANGUAGE_ATTENTION_MASK].detach().cpu().contiguous()
    hashes = {
        "raw_rgb_sha256": rgb_hashes,
        "raw_rgb_aggregate_sha256": canonical_sha256(rgb_hashes),
        "state_sha256": state_hash,
        "policy_state_sha256": named_arrays_sha256(policy_state_items),
        "proprio_sha256": tensor_sha256(physical),
        "language_tokens_sha256": tensor_sha256(language_tokens),
        "language_attention_mask_sha256": tensor_sha256(language_mask),
        "prefix_hidden_sha256": tensor_sha256(hidden),
        "prefix_mask_sha256": tensor_sha256(mask),
        "legacy_sha256": tensor_sha256(legacy),
        "rich_sha256": tensor_sha256(rich),
        "prefix_shape": list(hidden.shape),
        "valid_prefix_tokens": int(mask.sum()),
        "n_img_tokens": int(prefix.n_img_tokens),
    }
    del prefix, policy_batch, hidden, mask
    return legacy, rich, hashes


def exact_oracle(
    computed: torch.Tensor,
    expected: torch.Tensor,
    *,
    episode_id: int,
    t_value: int,
    side: str,
    oracle: str,
) -> None:
    expected_cpu = expected.detach().float().cpu().contiguous()
    if computed.dtype != expected_cpu.dtype or computed.shape != expected_cpu.shape:
        raise RuntimeError(
            f"{oracle} oracle shape/dtype mismatch at ep={episode_id} t={t_value} {side}: "
            f"computed={computed.dtype}/{tuple(computed.shape)}, "
            f"expected={expected_cpu.dtype}/{tuple(expected_cpu.shape)}"
        )
    if torch.equal(computed, expected_cpu):
        return
    difference = (computed - expected_cpu).abs()
    flat_index = int(torch.argmax(difference))
    raise RuntimeError(
        f"{oracle} oracle mismatch at ep={episode_id} t={t_value} {side}: "
        f"first_nonzero={int(torch.nonzero(difference, as_tuple=False)[0])}, "
        f"max_index={flat_index}, max_abs={float(difference.flatten()[flat_index]):.9g}, "
        f"computed_sha={tensor_sha256(computed)}, expected_sha={tensor_sha256(expected_cpu)}"
    )


def replay(
    bundle: SanitizedBundle,
    registration: dict[str, Any],
    policy_snapshot: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    from lcwm.chassis import Pi05Runner
    from lcwm.v080_bench import make_v080_env

    tape = bundle.tape
    replay_spec = registration["replay"]
    c = int(tape["c"])
    oracle = replay_spec["oracle"]
    selected_episode_ids = replay_spec["episode_ids"]
    if replay_spec["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "registered exact replay requires CUDA, but torch cannot access a CUDA device; "
            "refusing LeRobot's silent CPU fallback"
        )
    runner = Pi05Runner(
        model_id=policy_snapshot["resolved_path"],
        suite_name=replay_spec["suite_name"],
        device=replay_spec["device"],
        n_action_steps=c,
    )
    if str(runner.policy_cfg.device) != replay_spec["device"]:
        raise RuntimeError(
            f"runner device drifted to {runner.policy_cfg.device!r}; "
            f"registered {replay_spec['device']!r}"
        )
    env = make_v080_env(str(tape["task"]))
    rich_z: list[torch.Tensor] = []
    rich_z_next: list[torch.Tensor] = []
    legacy_z: list[torch.Tensor] = []
    legacy_z_next: list[torch.Tensor] = []
    selected_rows: list[int] = []
    boundaries: list[dict[str, Any]] = []
    env_steps = 0
    oracle_vectors = 0
    try:
        for episode_id in selected_episode_ids:
            rows = bundle.episode_rows[episode_id]
            seed = int(replay_spec["seed_start"]) + episode_id
            torch.manual_seed(seed)
            np.random.seed(seed)
            runner.reset()
            obs, _ = env.reset(seed=seed)
            task_description = env.task_description
            episode_legacy: list[torch.Tensor] = []
            episode_rich: list[torch.Tensor] = []
            episode_hashes: list[dict[str, Any]] = []
            for boundary_index in range(len(rows) + 1):
                legacy, rich, hashes = direct_features(runner, obs, task_description)
                episode_legacy.append(legacy)
                episode_rich.append(rich)
                episode_hashes.append(hashes)
                if boundary_index == len(rows):
                    break
                row = rows[boundary_index]
                t_value = int(tape["t"][row])
                if t_value != boundary_index * c:
                    raise RuntimeError(
                        f"source t mismatch at episode={episode_id}, row={row}: {t_value}"
                    )
                env_actions = runner.chunk_to_env(tape["u"][row])
                if env_actions.shape != (c, 7):
                    raise RuntimeError(f"decoded action shape mismatch: {env_actions.shape}")
                for action_offset, action in enumerate(env_actions):
                    obs, _reward, terminated, truncated, _info = env.step(action)
                    env_steps += 1
                    if terminated or truncated:
                        raise RuntimeError(
                            "environment terminated inside a stored transition: "
                            f"episode={episode_id}, t={t_value}, offset={action_offset}"
                        )

            for boundary_index, (legacy, rich, hashes) in enumerate(
                zip(episode_legacy, episode_rich, episode_hashes, strict=True)
            ):
                t_value = boundary_index * c
                expected: list[tuple[str, torch.Tensor]] = []
                if boundary_index < len(rows):
                    expected.append(("z", tape["z"][rows[boundary_index]]))
                if boundary_index > 0:
                    expected.append(("z_next", tape["z_next"][rows[boundary_index - 1]]))
                computed = rich if oracle == "rich8217_exact" else legacy
                for side, expected_vector in expected:
                    exact_oracle(
                        computed,
                        expected_vector,
                        episode_id=episode_id,
                        t_value=t_value,
                        side=side,
                        oracle=oracle,
                    )
                    oracle_vectors += 1
                boundaries.append(
                    {
                        "schema": BOUNDARY_SCHEMA,
                        "boundary_id": (
                            f"{bundle.tape_sha256[:12]}:ep{episode_id}:t{t_value}"
                        ),
                        "episode_id": episode_id,
                        "env_seed": seed,
                        "boundary_index": boundary_index,
                        "t": t_value,
                        "source_row_before": (
                            rows[boundary_index - 1] if boundary_index > 0 else None
                        ),
                        "source_row_after": (
                            rows[boundary_index] if boundary_index < len(rows) else None
                        ),
                        "hashes": hashes,
                    }
                )

            selected_rows.extend(rows)
            legacy_z.extend(episode_legacy[:-1])
            legacy_z_next.extend(episode_legacy[1:])
            rich_z.extend(episode_rich[:-1])
            rich_z_next.extend(episode_rich[1:])
            print(
                f"episode {episode_id} seed={seed}: {len(rows)} rows, "
                f"{len(rows) + 1} exact boundaries",
                flush=True,
            )
    finally:
        env.close()

    row_index = torch.tensor(selected_rows, dtype=torch.long)
    output_tape: dict[str, Any] = {
        "z": torch.stack(rich_z),
        "u": tape["u"].index_select(0, row_index).clone(),
        "z_next": torch.stack(rich_z_next),
        "episode": tape["episode"].index_select(0, row_index).clone(),
        "t": tape["t"].index_select(0, row_index).clone(),
        "sigma": tape["sigma"].index_select(0, row_index).clone(),
        "task": tape["task"],
        "c": c,
        "latent_dim": RICH_DIM,
        "proprio_dim": PROPRIO_DIM,
        "proprio_keys": list(PROPRIO_KEYS),
        "legacy_z": torch.stack(legacy_z),
        "legacy_z_next": torch.stack(legacy_z_next),
        "legacy_latent_dim": LEGACY_DIM,
        "source_row_index": row_index,
        "source_sanitized_tape_sha256": bundle.tape_sha256,
    }
    for name in SEQUENCE_FIELDS:
        expected = tape[name].index_select(0, row_index)
        actual = output_tape[name]
        if tensor_sha256(expected) != tensor_sha256(actual) or not torch.equal(expected, actual):
            raise RuntimeError(f"sequence field changed during replay: {name}")
    if len(output_tape["z"]) != len(selected_rows):
        raise RuntimeError("reencoded row count mismatch")
    metrics = {
        "episodes": len(selected_episode_ids),
        "rows": len(selected_rows),
        "boundaries": len(boundaries),
        "env_steps": env_steps,
        "oracle": oracle,
        "oracle_vectors_compared_exactly": oracle_vectors,
        "sequence_fields_exact": True,
        "source_row_index_sha256": tensor_sha256(row_index),
    }
    return output_tape, boundaries, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanitized-tape", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate all seals, source hashes, and local model files without reset/replay",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tape_path = args.sanitized_tape.expanduser().resolve()
    registration_path = args.registration.expanduser().resolve()
    output_path = args.out.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    if not registration_path.is_file():
        raise FileNotFoundError(registration_path)

    # No simulator, VLA, or LIBERO import occurs before both input seals and the
    # frozen source closure have passed.
    bundle = validate_sanitized(tape_path)
    registration, registration_sha = validate_registration(registration_path, bundle)
    policy_snapshot = resolve_and_verify_artifact(registration["replay"]["policy_snapshot"])
    tokenizer_snapshot = resolve_and_verify_artifact(
        registration["replay"]["tokenizer_snapshot"]
    )
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "VALID",
                    "registration_sha256": registration_sha,
                    "sanitized_tape_sha256": bundle.tape_sha256,
                    "policy_revision": policy_snapshot["revision"],
                    "tokenizer_revision": tokenizer_snapshot["revision"],
                    "simulator_imported": False,
                    "output_created": False,
                },
                indent=2,
            )
        )
        return 0

    # LIBERO resolves its config path at import time.  Establish the repository
    # config only after all pre-replay seals pass, but before the first import.
    from lcwm.libero_paths import ensure_project_libero_config

    ensure_project_libero_config()
    runtime_versions, external_provenance = module_runtime()
    if runtime_versions != registration["replay"]["runtime_versions"]:
        raise ValueError(
            "registered runtime versions drifted: "
            f"expected={registration['replay']['runtime_versions']}, actual={runtime_versions}"
        )
    output_tape, boundaries, metrics = replay(bundle, registration, policy_snapshot)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.partial-", dir=output_path.parent)
    )
    try:
        tape_out = stage / "rich_tape.pt"
        _save_torch(tape_out, output_tape)
        reloaded = torch.load(tape_out, map_location="cpu", weights_only=False)
        if not isinstance(reloaded, dict):
            raise TypeError("round-tripped rich tape is not a dictionary")
        for name in (*SEQUENCE_FIELDS, "z", "z_next", "legacy_z", "legacy_z_next"):
            if tensor_sha256(reloaded[name]) != tensor_sha256(output_tape[name]):
                raise RuntimeError(f"round-trip tensor bytes changed: {name}")
            if not torch.equal(reloaded[name], output_tape[name]):
                raise RuntimeError(f"round-trip tensor values changed: {name}")

        boundary_path = stage / "boundaries.jsonl"
        boundary_payload = b"".join(canonical_bytes(row) for row in boundaries)
        _write_bytes(boundary_path, boundary_payload)
        producer_path = stage / "PRODUCER.py"
        _write_bytes(producer_path, Path(__file__).resolve().read_bytes())
        registration_snapshot_path = stage / "REGISTRATION.json"
        _write_bytes(registration_snapshot_path, registration_path.read_bytes())
        sequence_hashes = {
            name: tensor_sha256(reloaded[name]) for name in SEQUENCE_FIELDS
        }
        utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
        result = {
            "schema": RESULT_SCHEMA,
            "status": "PASS",
            "utc": utc,
            "registration_sha256": registration_sha,
            "registration_self_sha256": registration["self_sha256"],
            "sanitized": {
                "tape_sha256": bundle.tape_sha256,
                "result_sha256": bundle.result_sha256,
                "complete_sha256": bundle.complete_sha256,
            },
            "output": {
                "rich_tape_sha256": sha256_file(tape_out),
                "boundary_manifest_sha256": sha256_file(boundary_path),
                "producer_snapshot_sha256": sha256_file(producer_path),
                "registration_snapshot_sha256": sha256_file(registration_snapshot_path),
                "rich_z_sha256": tensor_sha256(reloaded["z"]),
                "rich_z_next_sha256": tensor_sha256(reloaded["z_next"]),
                "legacy_z_sha256": tensor_sha256(reloaded["legacy_z"]),
                "legacy_z_next_sha256": tensor_sha256(reloaded["legacy_z_next"]),
                "sequence_tensor_sha256": sequence_hashes,
            },
            "replay": metrics,
            "hash_ledger": {
                "raw_rgb": True,
                "prefix_hidden_tokens": True,
                "prefix_mask": True,
                "raw_robot_state": True,
                "policy_state": True,
                "proprio": True,
                "boundary_rows": len(boundaries),
            },
            "model_provenance": {
                "policy_snapshot": policy_snapshot,
                "tokenizer_snapshot": tokenizer_snapshot,
            },
            "environment_provenance": {
                "task": bundle.tape["task"],
                "suite_name": registration["replay"]["suite_name"],
                "seed_start": registration["replay"]["seed_start"],
                "seed_rule": registration["replay"]["seed_rule"],
                "runtime_versions": runtime_versions,
                "external_modules": external_provenance,
                "mujo_co_gl": os.environ.get("MUJOCO_GL"),
                "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
            },
            "argv": list(sys.argv),
        }
        result_path = stage / "RESULT.json"
        _write_json(result_path, result)
        complete = {
            "schema": COMPLETE_SCHEMA,
            "status": "atomic_success",
            "atomic_commit": True,
            "registration_sha256": registration_sha,
            "sanitized_tape_sha256": bundle.tape_sha256,
            "rich_tape_sha256": result["output"]["rich_tape_sha256"],
            "boundary_manifest_sha256": result["output"]["boundary_manifest_sha256"],
            "result_sha256": sha256_file(result_path),
            "producer_snapshot_sha256": result["output"]["producer_snapshot_sha256"],
        }
        _write_json(stage / "COMPLETE.json", complete)
        directory_fd = os.open(stage, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(stage, output_path)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "output": output_path.as_posix(),
                "registration_sha256": registration_sha,
                "rich_tape_sha256": result["output"]["rich_tape_sha256"],
                **metrics,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
