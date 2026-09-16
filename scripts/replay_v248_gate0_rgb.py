#!/usr/bin/env python3
"""Replay the pre-registered v248 Gate-0 panels into a sealed RGB manifest.

Only an explicitly named outcome-sanitized legacy2073 tape and a registration
selected through an externally anchored freeze COMPLETE -> CAMPAIGN -> slot
chain are accepted.  A caller cannot supply a standalone registration digest
or choose an output path: the output COMPLETE slot and absolute target root are
frozen in the campaign projection.  No rollout summary is a CLI argument,
discovered, opened, or copied.  Ordinary episodes replay only their
stored normalized action chunks.  Seed-8100 episode 77 has one separately
registered recovery exception: all 65 stored chunks must first be regenerated
bit-for-bit, then the missing sigma=0.6 chunk may be sampled and only its first
action executed.  That endpoint is published for Phi only and never becomes a
dynamics transition.

The recovery episode is replayed from two fresh resets.  Its recovered chunk,
decoded actions, raw camera arrays, and proprioception must agree exactly
between passes before any JPEG is encoded.  Every output is staged privately,
validated, fsynced, and committed with a no-replace rename.
"""
from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import io
import json
import os
import shutil
import stat
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence


os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/v248_gate0_rgb_numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/v248_gate0_rgb_mpl_cache")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from lcwm import v248_gate0_contract as gate0_contract  # noqa: E402

from lcwm.v248_gate0_contract import (  # noqa: E402
    CAMERAS,
    C,
    FRAME_ROLES,
    Gate0ContractError,
    LATENT_DIM,
    PROPRIO_DIM,
    PROPRIO_KEYS,
    RGB_COMPLETE_SCHEMA,
    RGB_MANIFEST_SCHEMA,
    RGB_RESULT_SCHEMA,
    TASK,
    assert_no_simulator_modules_imported,
    canonical_bytes,
    canonical_sha256,
    is_sha256,
    load_and_validate_campaign_bundle,
    loads_strict_json,
    require,
    require_exact_keys,
    sha256_file,
)


SANITIZED_TAPE_FIELDS = frozenset(
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
SANITIZER_RESULT_FIELDS = frozenset(
    {
        "schema",
        "status",
        "utc",
        "source",
        "field_policy",
        "validation",
        "metadata",
        "tensor_sha256",
        "metadata_sha256",
        "sanitized_tape_sha256",
        "producer_snapshot_sha256",
        "git_head",
        "torch_version",
    }
)
SANITIZER_COMPLETE_FIELDS = frozenset(
    {
        "schema",
        "status",
        "atomic_commit",
        "source_tape_sha256",
        "sanitized_tape_sha256",
        "result_sha256",
        "producer_snapshot_sha256",
    }
)
SANITIZER_SOURCE_FIELDS = frozenset({"path", "sha256", "bytes", "summary_opened"})
SANITIZER_FIELD_POLICY_FIELDS = frozenset(
    {"kept", "dropped_names_without_value_access", "declared_outcome_fields"}
)
SANITIZER_VALIDATION_FIELDS = frozenset(
    {"rows", "latent_dim", "episodes", "episode_ids", "episode_row_counts", "c"}
)
SANITIZER_METADATA_FIELDS = frozenset(
    {"task", "c", "latent_dim", "proprio_dim", "proprio_keys"}
)
SEQUENCE_TENSOR_FIELDS = frozenset({"z", "u", "z_next", "episode", "t", "sigma"})

MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "frame_id",
        "panel_id",
        "phase",
        "task",
        "episode_id",
        "env_seed",
        "frame_index",
        "t",
        "source",
        "images",
        "proprio",
    }
)
MANIFEST_SOURCE_FIELDS = frozenset(
    {"tape_sha256", "source_row_index", "boundary_role"}
)
MANIFEST_IMAGE_FIELDS = frozenset(
    {"path", "sha256", "raw_array_sha256", "width", "height"}
)
MANIFEST_PROPRIO_FIELDS = frozenset(
    {"reference", "max_abs_error", "mean_abs_error"}
)
RESULT_FIELDS = frozenset(
    {
        "schema",
        "status",
        "created_utc",
        "registration",
        "authority",
        "panel_id",
        "phase",
        "task",
        "episode_ids",
        "inputs",
        "replay",
        "terminal_recovery_audit",
        "output",
        "safety",
    }
)
RESULT_REGISTRATION_FIELDS = frozenset(
    {"registration_id", "file_sha256", "canonical_self_sha256"}
)
AUTHORITY_FIELDS = frozenset(
    {
        "freeze_complete_sha256",
        "campaign_manifest_sha256",
        "campaign_body_sha256",
        "campaign_projection_sha256",
        "campaign_id",
        "registration_slot",
        "output_complete_slot",
        "confirm_authorization_complete_sha256",
    }
)
RESULT_INPUT_FIELDS = frozenset({"sanitized_tape_sha256", "source_tape_sha256"})
RESULT_REPLAY_FIELDS = frozenset(
    {
        "episodes",
        "ordinary_episodes",
        "recovery_episodes",
        "published_environment_steps",
        "recovery_audit_environment_steps",
        "fresh_recovery_passes",
        "all_registered_legacy2073_oracles_bit_exact",
        "all_registered_proprio_oracles_bit_exact",
        "all_stored_recovery_chunks_bit_exact",
    }
)
RESULT_RECOVERY_FIELDS = frozenset(
    {
        "enabled",
        "episode_id",
        "stored_chunks",
        "stored_chunks_exact_bitwise",
        "recovered_chunk_tensor_sha256",
        "two_fresh_replays",
        "cross_run_boundary_hashes_exact",
        "cross_run_decoded_actions_exact",
        "halted_at_registered_t",
        "phi_only",
        "dynamics_transition_eligible",
        "recovered_dynamics_rows_written",
    }
)
RESULT_OUTPUT_FIELDS = frozenset(
    {
        "manifest_sha256",
        "frames",
        "jpeg_files",
        "raw_rgb_arrays",
        "image_tree_sha256",
        "producer_sha256",
    }
)
RESULT_SAFETY_FIELDS = frozenset(
    {
        "summary_cli_supported",
        "summary_discovered_or_opened",
        "manifest_semantic_field_scan_passed",
        "recovered_endpoint_phi_only",
        "recovered_endpoint_dynamics_transition_eligible",
        "dynamics_tape_written",
    }
)
COMPLETE_FIELDS = frozenset(
    {
        "schema",
        "status",
        "atomic_commit",
        "registration_sha256",
        "registration_self_sha256",
        "authority",
        "panel_id",
        "phase",
        "result_sha256",
        "manifest_sha256",
        "producer_sha256",
        "image_tree_sha256",
    }
)

SANITIZER_RESULT_SCHEMA = "v245_sanitized_deploy_tape_result_v1"
SANITIZER_COMPLETE_SCHEMA = "v245_sanitized_deploy_tape_complete_v1"
JPEG_QUALITY = 95
FORBIDDEN_MANIFEST_KEYS = frozenset(
    {
        "done",
        "is_success",
        "success",
        "successes",
        "terminated",
        "truncated",
        "terminal",
        "event",
        "events",
        "milestone",
        "milestones",
        "outcome",
        "reward",
    }
)


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    header = canonical_bytes({"dtype": str(tensor.dtype), "shape": list(tensor.shape)})
    return hashlib.sha256(header + tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    header = canonical_bytes({"dtype": array.dtype.str, "shape": list(array.shape)})
    return hashlib.sha256(header + array.tobytes(order="C")).hexdigest()


def tensors_bitwise_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Require torch.equal plus identical dtype, shape, and raw bytes.

    ``torch.equal`` alone treats +0.0 and -0.0 as equal.  The raw tensor digest
    closes that gap while preserving the explicitly registered comparator.
    """
    return bool(
        left.dtype == right.dtype
        and tuple(left.shape) == tuple(right.shape)
        and torch.equal(left, right)
        and tensor_sha256(left) == tensor_sha256(right)
    )


def assert_tensors_bitwise_equal(left: torch.Tensor, right: torch.Tensor, where: str) -> None:
    require(left.dtype == right.dtype, f"{where} dtype mismatch")
    require(tuple(left.shape) == tuple(right.shape), f"{where} shape mismatch")
    require(torch.equal(left, right), f"{where} torch.equal failed")
    require(tensor_sha256(left) == tensor_sha256(right), f"{where} raw-byte mismatch")


def _write_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    _write_bytes(path, canonical_bytes(value))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _path_lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _absolute_lexical(path: Path) -> Path:
    """Make a path absolute without resolving a possibly dangling symlink."""
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _assert_no_symlink_components(path: Path, where: str) -> None:
    """Reject every existing symlink component without resolving the input."""
    lexical = _absolute_lexical(path)
    current = Path(lexical.anchor)
    for component in lexical.parts[1:]:
        current /= component
        if not _path_lexists(current):
            continue
        mode = os.lstat(current).st_mode
        require(not stat.S_ISLNK(mode), f"{where} has symlink component: {current}")


def _mkdir_safe_beneath(target_root: Path, parent: Path) -> None:
    """Create only real directories below the frozen target root."""
    target_root = _absolute_lexical(target_root)
    parent = _absolute_lexical(parent)
    try:
        relative = parent.relative_to(target_root)
    except ValueError as error:
        raise Gate0ContractError("output parent escapes frozen target root") from error
    _assert_no_symlink_components(target_root.parent, "frozen target root parent")
    current = target_root
    for component in ((), *[(part,) for part in relative.parts]):
        if component:
            current /= component[0]
        if _path_lexists(current):
            mode = os.lstat(current).st_mode
            require(not stat.S_ISLNK(mode), f"output parent is a symlink: {current}")
            require(stat.S_ISDIR(mode), f"output parent is not a directory: {current}")
        else:
            current.mkdir()
            mode = os.lstat(current).st_mode
            require(
                stat.S_ISDIR(mode) and not stat.S_ISLNK(mode),
                f"unsafe output parent: {current}",
            )


def _complete_path_from_slot(target_root: Path, complete_slot: str) -> Path:
    require(isinstance(complete_slot, str), "output COMPLETE slot must be text")
    relative = PurePosixPath(complete_slot)
    require(
        not relative.is_absolute()
        and relative.parts
        and all(part not in {"", ".", ".."} for part in relative.parts),
        "output COMPLETE slot is not a safe relative path",
    )
    require(relative.name == "COMPLETE.json", "output slot must name COMPLETE.json")
    target_root = _absolute_lexical(target_root)
    require(target_root.is_absolute(), "frozen target root must be absolute")
    _assert_no_symlink_components(target_root, "frozen target root")
    complete_path = target_root.joinpath(*relative.parts)
    output = complete_path.parent
    require(output != target_root, "output slot cannot equal frozen target root")
    _assert_no_symlink_components(output, "output slot parent")
    return complete_path


def _output_path_from_slot(target_root: Path, complete_slot: str) -> Path:
    complete_path = _complete_path_from_slot(target_root, complete_slot)
    output = complete_path.parent
    require(not _path_lexists(output), f"output already exists: {output}")
    return output


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Linux no-clobber directory rename; fail closed if unavailable."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    require(renameat2 is not None, "renameat2 is unavailable; refusing non-atomic fallback")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(destination)
        raise OSError(error_number, os.strerror(error_number), destination)


def atomic_publish_directory(
    output_path: Path,
    builder: Callable[[Path], None],
    validator: Callable[[Path], None],
    *,
    trusted_target_root: Path | None = None,
) -> None:
    """Build, independently validate, and no-clobber publish one directory."""
    output_path = _absolute_lexical(output_path)
    require(not _path_lexists(output_path), f"refusing to overwrite output: {output_path}")
    if trusted_target_root is None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        _mkdir_safe_beneath(trusted_target_root, output_path.parent)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.partial-", dir=output_path.parent)
    )
    try:
        builder(stage)
        validator(stage)
        _fsync_directory(stage)
        if trusted_target_root is not None:
            _assert_no_symlink_components(output_path.parent, "output slot parent")
        require(not _path_lexists(output_path), f"output appeared during staging: {output_path}")
        _rename_noreplace(stage, output_path)
        _fsync_directory(output_path.parent)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        raise


def _strict_json_file(path: Path, where: str) -> dict[str, Any]:
    require(path.is_file(), f"missing {where}: {path}")
    value = loads_strict_json(path.read_text(encoding="utf-8"), where=where)
    require(isinstance(value, dict), f"{where} must contain an object")
    return value


def _integer_tensor(value: Any, rows: int, where: str) -> torch.Tensor:
    require(torch.is_tensor(value) and value.ndim == 1 and len(value) == rows, f"{where} shape")
    require(value.dtype == torch.int64, f"{where} must be torch.int64")
    return value.detach().cpu().contiguous()


@dataclass
class PreparedReplay:
    freeze_bundle_root: Path
    expected_freeze_complete_sha256: str
    registration_slot: str
    registration_path: Path
    registration: dict[str, Any]
    registration_sha256: str
    authority: dict[str, Any]
    target_root: Path
    output_complete_slot: str
    confirm_authorization_complete_sha256: str | None
    tape_path: Path
    tape: dict[str, Any]
    selected_rows: dict[int, list[int]]
    sanitizer_paths: dict[str, Path]
    sanitizer_hashes: dict[str, str]
    output_path: Path
    panel_id: str
    phase: str


def validate_sanitized_tape(
    tape_path: Path, registration: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[int, list[int]], dict[str, Path], dict[str, str]]:
    root = tape_path.parent
    paths = {
        "tape": tape_path,
        "result": root / "RESULT.json",
        "complete": root / "COMPLETE.json",
        "producer": root / "PRODUCER.py",
    }
    for role, path in paths.items():
        require(path.is_file(), f"sanitized {role} file missing: {path}")
    hashes = {f"{role}_sha256": sha256_file(path) for role, path in paths.items()}
    registered = registration["sanitized_source"]
    for key in ("tape_sha256", "result_sha256", "complete_sha256", "producer_sha256"):
        require(hashes[key] == registered[key], f"registered sanitized {key} drift")

    result = _strict_json_file(paths["result"], "sanitizer RESULT")
    complete = _strict_json_file(paths["complete"], "sanitizer COMPLETE")
    require_exact_keys(result, SANITIZER_RESULT_FIELDS, "sanitizer RESULT")
    require_exact_keys(complete, SANITIZER_COMPLETE_FIELDS, "sanitizer COMPLETE")
    require(
        result["schema"] == SANITIZER_RESULT_SCHEMA and result["status"] == "PASS",
        "sanitizer RESULT is not PASS",
    )
    require(complete["schema"] == SANITIZER_COMPLETE_SCHEMA, "sanitizer COMPLETE schema")
    require(
        complete["status"] == "atomic_success" and complete["atomic_commit"] is True,
        "sanitizer output was not atomically committed",
    )
    require(
        complete["source_tape_sha256"] == registered["source_tape_sha256"],
        "source tape registration drift",
    )
    require(complete["sanitized_tape_sha256"] == hashes["tape_sha256"], "sanitizer tape seal")
    require(complete["result_sha256"] == hashes["result_sha256"], "sanitizer result seal")
    require(
        complete["producer_snapshot_sha256"] == hashes["producer_sha256"],
        "sanitizer producer seal",
    )
    require(
        result["sanitized_tape_sha256"] == hashes["tape_sha256"],
        "sanitizer RESULT tape seal",
    )
    require(
        result["producer_snapshot_sha256"] == hashes["producer_sha256"],
        "sanitizer RESULT producer seal",
    )

    source = require_exact_keys(
        result["source"], SANITIZER_SOURCE_FIELDS, "sanitizer RESULT.source"
    )
    require(source["sha256"] == registered["source_tape_sha256"], "sanitizer source SHA drift")
    require(
        source["summary_opened"] is False,
        "sanitizer did not preserve the summary-blind invariant",
    )
    policy = require_exact_keys(
        result["field_policy"],
        SANITIZER_FIELD_POLICY_FIELDS,
        "sanitizer RESULT.field_policy",
    )
    require(
        policy["kept"] == sorted(SANITIZED_TAPE_FIELDS),
        "sanitizer kept-field allow-list drift",
    )
    require(
        policy["dropped_names_without_value_access"] == ["success"],
        "formal source must drop exactly the legacy success vector",
    )
    validation = require_exact_keys(
        result["validation"],
        SANITIZER_VALIDATION_FIELDS,
        "sanitizer RESULT.validation",
    )
    metadata = require_exact_keys(
        result["metadata"], SANITIZER_METADATA_FIELDS, "sanitizer RESULT.metadata"
    )
    tensor_hashes = require_exact_keys(
        result["tensor_sha256"],
        SEQUENCE_TENSOR_FIELDS,
        "sanitizer RESULT.tensor_sha256",
    )
    require(
        metadata
        == {
            "task": TASK,
            "c": C,
            "latent_dim": LATENT_DIM,
            "proprio_dim": PROPRIO_DIM,
            "proprio_keys": list(PROPRIO_KEYS),
        },
        "sanitizer metadata is not the chain3 legacy2073 contract",
    )
    require(
        validation["latent_dim"] == LATENT_DIM and validation["c"] == C,
        "sanitizer validation layout",
    )
    require(
        validation["episodes"] == 96
        and validation["episode_ids"] == list(range(96)),
        "sanitized source is not the complete 96-episode panel",
    )

    tape = torch.load(tape_path, map_location="cpu", weights_only=False)
    require(
        isinstance(tape, dict) and set(tape) == set(SANITIZED_TAPE_FIELDS),
        "sanitized tape field closure mismatch",
    )
    require(
        tape["task"] == TASK and type(tape["c"]) is int and tape["c"] == C,
        "tape task/c mismatch",
    )
    require(
        type(tape["latent_dim"]) is int and tape["latent_dim"] == LATENT_DIM,
        "tape is not legacy2073",
    )
    require(
        type(tape["proprio_dim"]) is int and tape["proprio_dim"] == PROPRIO_DIM,
        "tape proprio_dim",
    )
    require(tape["proprio_keys"] == list(PROPRIO_KEYS), "tape proprio keys")
    for name in SEQUENCE_TENSOR_FIELDS:
        require(torch.is_tensor(tape[name]), f"tape {name} is not a tensor")
        require(tensor_sha256(tape[name]) == tensor_hashes[name], f"tape {name} tensor seal drift")
    rows = len(tape["z"])
    require(rows == validation["rows"], "tape row count disagrees with sanitizer")
    require(
        tape["z"].dtype == torch.float32 and tape["z"].shape == (rows, LATENT_DIM),
        "tape z layout",
    )
    require(
        tape["z_next"].dtype == torch.float32
        and tape["z_next"].shape == (rows, LATENT_DIM),
        "tape z_next layout",
    )
    require(
        tape["u"].dtype == torch.float32 and tape["u"].shape == (rows, C, 7),
        "tape u layout",
    )
    require(
        tape["sigma"].dtype == torch.float32 and tape["sigma"].shape == (rows,),
        "tape sigma layout",
    )
    episode_tensor = _integer_tensor(tape["episode"], rows, "tape episode")
    time_tensor = _integer_tensor(tape["t"], rows, "tape t")
    for name in ("z", "z_next", "u", "sigma"):
        require(bool(torch.isfinite(tape[name]).all()), f"tape {name} contains NaN/Inf")
    episode_ids = torch.unique_consecutive(episode_tensor).tolist()
    require(episode_ids == list(range(96)), "tape episode rows are not grouped 0..95")
    observed_counts = []
    for episode_id in episode_ids:
        indices = torch.nonzero(
            episode_tensor == episode_id, as_tuple=False
        ).flatten()
        observed_counts.append(len(indices))
        expected_times = torch.arange(len(indices), dtype=torch.int64) * C
        require(
            torch.equal(time_tensor[indices], expected_times),
            f"source episode {episode_id} time grid drift",
        )
        if len(indices) > 1:
            assert_tensors_bitwise_equal(
                tape["z_next"].index_select(0, indices[:-1]),
                tape["z"].index_select(0, indices[1:]),
                f"source episode {episode_id} boundary continuity",
            )
    require(
        observed_counts == validation["episode_row_counts"],
        "tape episode row counts disagree with sanitizer",
    )

    selected_rows: dict[int, list[int]] = {}
    for episode_spec in registration["panel"]["episodes"]:
        episode_id = episode_spec["episode_id"]
        indices = torch.nonzero(episode_tensor == episode_id, as_tuple=False).flatten().tolist()
        indices = [int(index) for index in indices]
        require(
            len(indices) == episode_spec["stored_rows"],
            f"episode {episode_id} registered row count drift",
        )
        require(
            [int(time_tensor[index]) for index in indices]
            == list(range(0, len(indices) * C, C)),
            f"episode {episode_id} time grid drift",
        )
        for left, right in zip(indices, indices[1:]):
            assert_tensors_bitwise_equal(
                tape["z_next"][left],
                tape["z"][right],
                f"episode {episode_id} internal boundary",
            )
        selected_rows[episode_id] = indices

    if registration["terminal_recovery"]["enabled"]:
        episode_id = registration["terminal_recovery"]["episode_id"]
        rows_for_episode = selected_rows[episode_id]
        raw_sigmas = reconstruct_recovery_sigmas(registration)
        require(len(raw_sigmas) == 66, "recovery sigma sequence length")
        for offset, row_index in enumerate(rows_for_episode):
            observed = tape["sigma"][row_index].reshape(1).contiguous()
            expected = torch.tensor([np.float32(raw_sigmas[offset])], dtype=torch.float32)
            assert_tensors_bitwise_equal(observed, expected, f"ep77 sigma[{offset}]")
        require(
            raw_sigmas[65]
            == registration["terminal_recovery"]["missing_raw_sigma_choice"],
            "missing raw sigma registration drift",
        )
    return tape, selected_rows, paths, hashes


def reconstruct_recovery_sigmas(registration: Mapping[str, Any]) -> list[float]:
    recovery = registration["terminal_recovery"]
    require(recovery["enabled"] is True, "terminal recovery is disabled")
    spec = recovery["sigma_rng"]
    generator = np.random.default_rng(spec["seed"])
    choices = spec["choices"]
    selected: list[float] | None = None
    for episode_id, count in enumerate(spec["source_proposed_chunks_by_episode"]):
        draws = [choices[int(generator.integers(len(choices)))] for _ in range(count)]
        if episode_id == recovery["episode_id"]:
            selected = draws
    require(selected is not None, "recovery episode absent from sigma RNG metadata")
    return selected


def _contract_anchored_bytes(path: Path, expected_sha256: str, where: str) -> bytes:
    return gate0_contract.read_anchored_regular_bytes(path, expected_sha256, where)


def _json_from_bytes(payload: bytes, where: str) -> dict[str, Any]:
    try:
        value = loads_strict_json(payload.decode("utf-8"), where=where)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise Gate0ContractError(f"cannot decode {where}") from error
    require(isinstance(value, dict), f"{where} must be an object")
    return dict(value)


def _load_registered_source_module(
    registration: Mapping[str, Any], role: str, module_name: str
) -> Any:
    reference = registration["sources"][role]
    path = (REPO / reference["relative_path"]).resolve()
    require(path.is_relative_to(REPO), f"registered {role} escapes repository")
    payload = _contract_anchored_bytes(path, reference["sha256"], f"registered {role}")
    module = type(sys)(module_name)
    module.__file__ = path.as_posix()
    sys.modules[module_name] = module
    exec(compile(payload, path.as_posix(), "exec"), module.__dict__)  # noqa: S102
    return module


def _require_replay_phase_authority(
    registration: Mapping[str, Any],
    expected_confirm_authorization_complete_sha256: str | None,
) -> None:
    if registration["phase"] == "screen":
        require(
            expected_confirm_authorization_complete_sha256 is None,
            "screen replay must not consume a confirm-authorization anchor",
        )
        return
    require(
        is_sha256(expected_confirm_authorization_complete_sha256),
        "formal confirm replay requires an external confirm-authorization COMPLETE SHA",
    )


def _validate_confirm_authorization(
    bundle: Mapping[str, Any], expected_complete_sha256: str
) -> dict[str, Any]:
    require(is_sha256(expected_complete_sha256), "confirm authorization SHA invalid")
    registration = bundle["registrations"]["chain3_8000_screen.json"][
        "registration"
    ]
    authorizer = _load_registered_source_module(
        registration,
        "confirm_authorizer",
        "v248_registered_confirm_authorizer_for_replay",
    )
    validated = authorizer.validate_authorization_artifact(
        bundle,
        expected_complete_sha256,
    )
    assert_no_simulator_modules_imported()
    return {
        "root": validated["root"],
        "complete_sha256": validated["complete_sha256"],
        "screen_evaluation_complete_sha256": validated["screen"][
            "complete_sha256"
        ],
        "captured_authorization_file_sha256s": validated[
            "captured_authorization_file_sha256s"
        ],
    }


def _check_output_separation(output_path: Path, input_paths: Sequence[Path]) -> None:
    resolved_output = output_path.resolve(strict=False)
    require(not _path_lexists(output_path), f"output already exists: {output_path}")
    for path in input_paths:
        resolved_input = path.resolve()
        require(resolved_output != resolved_input, "output aliases an input")
        require(not resolved_input.is_relative_to(resolved_output), "output would contain an input")
        require(
            not resolved_output.is_relative_to(resolved_input),
            "output is nested inside an input file",
        )


def prepare_inputs(
    *,
    sanitized_tape: Path,
    freeze_bundle_root: Path,
    expected_freeze_complete_sha256: str,
    registration_slot: str,
    expected_confirm_authorization_complete_sha256: str | None = None,
) -> PreparedReplay:
    require(
        is_sha256(expected_freeze_complete_sha256),
        "expected freeze COMPLETE SHA must be 64 lowercase hex",
    )
    bundle = load_and_validate_campaign_bundle(
        freeze_bundle_root,
        expected_freeze_complete_sha256,
        expected_source_role="formal_replay",
    )
    require(
        registration_slot in bundle["registrations"],
        "registration slot is not a member of the anchored campaign",
    )
    registration_entry = bundle["registrations"][registration_slot]
    registration_path = Path(registration_entry["path"])
    registration = dict(registration_entry["registration"])
    registration_sha = registration_entry["file_sha256"]
    require(
        registration["campaign"]["registration_slot"] == registration_slot,
        "selected registration does not point to its campaign slot",
    )
    _require_replay_phase_authority(
        registration, expected_confirm_authorization_complete_sha256
    )
    panel_id = registration["panel"]["panel_id"]
    phase = registration["phase"]
    output_slots = bundle["target_output_slots"]
    output_key = f"{phase}_rgb_complete_by_panel"
    output_complete_slot = output_slots[output_key][panel_id]
    target_root = Path(bundle["target_root"])
    confirm_authorization = None
    if phase == "confirm":
        assert expected_confirm_authorization_complete_sha256 is not None
        confirm_authorization = _validate_confirm_authorization(
            bundle, expected_confirm_authorization_complete_sha256
        )
    # For confirm, the complete A->F/P/S/Q/E validator above must finish before
    # even probing the frozen RGB output slot or any of its path components.
    output = _output_path_from_slot(target_root, output_complete_slot)
    authority = {
        "freeze_complete_sha256": bundle["freeze_complete_sha256"],
        "campaign_manifest_sha256": bundle["campaign_manifest_sha256"],
        "campaign_body_sha256": bundle["campaign"]["body_sha256"],
        "campaign_projection_sha256": bundle["campaign"]["projection_sha256"],
        "campaign_id": bundle["campaign"]["campaign_id"],
        "registration_slot": registration_slot,
        "output_complete_slot": output_complete_slot,
        "confirm_authorization_complete_sha256": (
            None
            if confirm_authorization is None
            else confirm_authorization["complete_sha256"]
        ),
    }
    _validate_authority(authority, registration, expected=authority)
    tape_path = sanitized_tape.expanduser().resolve()
    require(tape_path.is_file(), f"sanitized tape missing: {tape_path}")
    tape, selected_rows, paths, hashes = validate_sanitized_tape(tape_path, registration)
    protected_inputs = [Path(bundle["root"]), registration_path, *paths.values()]
    if confirm_authorization is not None:
        protected_inputs.append(Path(confirm_authorization["root"]))
    _check_output_separation(
        output,
        protected_inputs,
    )
    require(
        not output.is_relative_to(tape_path.parent),
        "output must not be nested in the sanitized artifact",
    )
    return PreparedReplay(
        freeze_bundle_root=Path(bundle["root"]),
        expected_freeze_complete_sha256=expected_freeze_complete_sha256,
        registration_slot=registration_slot,
        registration_path=registration_path,
        registration=registration,
        registration_sha256=registration_sha,
        authority=authority,
        target_root=target_root,
        output_complete_slot=output_complete_slot,
        confirm_authorization_complete_sha256=(
            None
            if confirm_authorization is None
            else confirm_authorization["complete_sha256"]
        ),
        tape_path=tape_path,
        tape=tape,
        selected_rows=selected_rows,
        sanitizer_paths=paths,
        sanitizer_hashes=hashes,
        output_path=output,
        panel_id=panel_id,
        phase=phase,
    )


def revalidate_static_inputs(prepared: PreparedReplay) -> None:
    bundle = load_and_validate_campaign_bundle(
        prepared.freeze_bundle_root,
        prepared.expected_freeze_complete_sha256,
        expected_source_role="formal_replay",
    )
    require(
        prepared.registration_slot in bundle["registrations"],
        "registration slot disappeared from anchored campaign",
    )
    entry = bundle["registrations"][prepared.registration_slot]
    require(
        entry["file_sha256"] == prepared.registration_sha256,
        "registration anchor drift before reset/publish",
    )
    require(
        entry["registration"] == prepared.registration,
        "registration value drift before reset/publish",
    )
    require(
        Path(bundle["target_root"]) == prepared.target_root
        and bundle["target_output_slots"][
            f"{prepared.phase}_rgb_complete_by_panel"
        ][
            prepared.panel_id
        ]
        == prepared.output_complete_slot,
        "campaign output authority drift before reset/publish",
    )
    current_authority = {
        "freeze_complete_sha256": bundle["freeze_complete_sha256"],
        "campaign_manifest_sha256": bundle["campaign_manifest_sha256"],
        "campaign_body_sha256": bundle["campaign"]["body_sha256"],
        "campaign_projection_sha256": bundle["campaign"]["projection_sha256"],
        "campaign_id": bundle["campaign"]["campaign_id"],
        "registration_slot": prepared.registration_slot,
        "output_complete_slot": prepared.output_complete_slot,
        "confirm_authorization_complete_sha256": (
            prepared.confirm_authorization_complete_sha256
        ),
    }
    require(current_authority == prepared.authority, "campaign genesis authority drift")
    if prepared.phase == "confirm":
        assert prepared.confirm_authorization_complete_sha256 is not None
        _validate_confirm_authorization(
            bundle, prepared.confirm_authorization_complete_sha256
        )
    registered = prepared.registration["sanitized_source"]
    for role, path in prepared.sanitizer_paths.items():
        require(
            sha256_file(path) == registered[f"{role}_sha256"],
            f"sanitized {role} drift before reset/publish",
        )
    require(
        _output_path_from_slot(prepared.target_root, prepared.output_complete_slot)
        == prepared.output_path,
        "derived output slot drift before reset/publish",
    )


def _nested_value(tree: Mapping[str, Any], dotted_key: str) -> Any:
    value: Any = tree
    for component in dotted_key.split("."):
        require(
            isinstance(value, Mapping) and component in value,
            f"observation missing {dotted_key}",
        )
        value = value[component]
    return value


def observation_proprio(obs: Mapping[str, Any]) -> torch.Tensor:
    parts = [
        torch.as_tensor(np.asarray(_nested_value(obs, key)), dtype=torch.float32).flatten()
        for key in PROPRIO_KEYS
    ]
    value = torch.cat(parts).contiguous()
    require(
        value.dtype == torch.float32 and value.shape == (PROPRIO_DIM,),
        "observation proprio layout",
    )
    require(bool(torch.isfinite(value).all()), "observation proprio contains NaN/Inf")
    return value


def raw_rgb(obs: Mapping[str, Any], key: str) -> np.ndarray:
    pixels = obs.get("pixels")
    require(isinstance(pixels, Mapping) and key in pixels, f"observation missing pixels/{key}")
    value = np.asarray(pixels[key])
    require(
        value.dtype == np.uint8 and value.ndim == 3 and value.shape[-1] == 3,
        f"pixels/{key} is not uint8 HWC RGB",
    )
    return np.ascontiguousarray(value)


@dataclass
class CapturedFrame:
    episode_id: int
    env_seed: int
    t: int
    source_row_index: int | None
    boundary_role: str
    proprio_reference: str
    proprio_max_abs_error: float
    proprio_mean_abs_error: float
    proprio_sha256: str
    raw_arrays: dict[str, np.ndarray]
    raw_hashes: dict[str, str]


def capture_frame(
    *,
    obs: Mapping[str, Any],
    tape: Mapping[str, Any],
    episode_id: int,
    env_seed: int,
    t: int,
    source_row_index: int | None,
    boundary_role: str,
    reference_rows: Sequence[tuple[str, int]],
    legacy_encoder: Callable[[Mapping[str, Any], str], torch.Tensor] | None,
    task_description: str,
) -> CapturedFrame:
    require(boundary_role in FRAME_ROLES, "invalid boundary role")
    observed = observation_proprio(obs)
    if reference_rows:
        require(legacy_encoder is not None, "legacy2073 encoder missing")
        live_legacy = legacy_encoder(obs, task_description)
        require(
            live_legacy.dtype == torch.float32
            and live_legacy.shape == (LATENT_DIM,),
            "live legacy2073 layout mismatch",
        )
        for tensor_name, row_index in reference_rows:
            expected_legacy = (
                tape[tensor_name][row_index].detach().cpu().contiguous()
            )
            assert_tensors_bitwise_equal(
                live_legacy,
                expected_legacy,
                f"ep{episode_id} t{t} legacy2073/{tensor_name}",
            )
    differences = []
    for tensor_name, row_index in reference_rows:
        expected = (
            tape[tensor_name][row_index, -PROPRIO_DIM:]
            .detach()
            .cpu()
            .contiguous()
        )
        require(
            expected.dtype == torch.float32 and expected.shape == (PROPRIO_DIM,),
            "proprio oracle layout",
        )
        difference = (observed - expected).abs()
        differences.append(difference)
        assert_tensors_bitwise_equal(
            observed,
            expected,
            f"ep{episode_id} t{t} proprio/{tensor_name}",
        )
    if reference_rows:
        combined = torch.cat(differences)
        maximum = float(combined.max())
        mean = float(combined.mean())
        names = [name for name, _row in reference_rows]
        if names == ["z"]:
            reference = "z"
        elif names == ["z", "z_next"]:
            reference = "z_and_previous_z_next"
        elif names == ["z_next"]:
            reference = "last_z_next"
        else:
            raise Gate0ContractError(f"unexpected proprio reference set: {names}")
    else:
        maximum = 0.0
        mean = 0.0
        reference = "cross_run_exact"
    arrays = {camera: raw_rgb(obs, obs_key).copy() for camera, obs_key in CAMERAS.items()}
    hashes = {camera: array_sha256(value) for camera, value in arrays.items()}
    return CapturedFrame(
        episode_id=episode_id,
        env_seed=env_seed,
        t=t,
        source_row_index=source_row_index,
        boundary_role=boundary_role,
        proprio_reference=reference,
        proprio_max_abs_error=maximum,
        proprio_mean_abs_error=mean,
        proprio_sha256=tensor_sha256(observed),
        raw_arrays=arrays,
        raw_hashes=hashes,
    )


def _seed_replay(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalized_policy_chunk(
    runner: Any,
    sample_chunks_fn: Callable[..., torch.Tensor],
    obs: Mapping[str, Any],
    task_description: str,
    seed: int,
    t: int,
    sigma: float,
) -> torch.Tensor:
    with torch.no_grad():
        if sigma == 0.0:
            raw = runner.sample_chunk(obs, task_description)
        else:
            policy_observation = runner._obs_to_policy_batch(obs, task_description)
            raw = sample_chunks_fn(
                runner.policy,
                policy_observation,
                1,
                seed=seed * 7919 + t,
                sigma=sigma,
            )
    require(
        torch.is_tensor(raw)
        and raw.ndim == 3
        and raw.shape[0] >= 1
        and raw.shape[1] >= C,
        "sampled chunk layout",
    )
    chunk = raw[0, :C].detach().to(dtype=torch.float32, device="cpu").contiguous()
    require(
        chunk.shape == (C, 7) and bool(torch.isfinite(chunk).all()),
        "sampled normalized chunk invalid",
    )
    return chunk


def _decoded_chunk(runner: Any, chunk: torch.Tensor) -> np.ndarray:
    decoded = np.ascontiguousarray(np.asarray(runner.chunk_to_env(chunk), dtype=np.float32))
    require(
        decoded.shape == (C, 7) and np.isfinite(decoded).all(),
        "decoded environment chunk invalid",
    )
    return decoded


def _step_without_semantic_read(
    env: Any, action: np.ndarray
) -> tuple[Mapping[str, Any], bool, bool]:
    obs, _ignored_reward, stopped, capped, _ignored_metadata = env.step(action)
    require(isinstance(obs, Mapping), "environment observation must be a mapping")
    return obs, bool(stopped), bool(capped)


@dataclass
class RecoveryPass:
    frames: list[CapturedFrame]
    decoded_chunk_hashes: list[str]
    recovered_chunk: torch.Tensor
    recovered_decoded_action_sha256: str
    environment_steps: int


def run_recovery_pass(
    *,
    runner: Any,
    env_factory: Callable[[], Any],
    sample_chunks_fn: Callable[..., torch.Tensor],
    tape: Mapping[str, Any],
    row_indices: Sequence[int],
    registration: Mapping[str, Any],
    legacy_encoder: Callable[[Mapping[str, Any], str], torch.Tensor],
) -> RecoveryPass:
    recovery = registration["terminal_recovery"]
    require(recovery["enabled"] is True, "recovery pass requested for disabled panel")
    require(len(row_indices) == recovery["stored_chunks"] == 65, "recovery stored-row prerequisite")
    episode_id = recovery["episode_id"]
    seed = recovery["env_seed"]
    sigmas = reconstruct_recovery_sigmas(registration)
    _seed_replay(seed)
    runner.reset()
    env = env_factory()
    frames: list[CapturedFrame] = []
    decoded_hashes: list[str] = []
    environment_steps = 0
    try:
        obs, _ignored_reset_metadata = env.reset(seed=seed)
        require(isinstance(obs, Mapping), "reset observation must be a mapping")
        task_description = env.task_description
        require(isinstance(task_description, str) and task_description, "task description missing")
        for offset, row_index in enumerate(row_indices):
            t = offset * C
            references = [("z", row_index)]
            if offset:
                references.append(("z_next", row_indices[offset - 1]))
            frames.append(
                capture_frame(
                    obs=obs,
                    tape=tape,
                    episode_id=episode_id,
                    env_seed=seed,
                    t=t,
                    source_row_index=row_index,
                    boundary_role="pre_action",
                    reference_rows=references,
                    legacy_encoder=legacy_encoder,
                    task_description=task_description,
                )
            )
            proposed = normalized_policy_chunk(
                runner,
                sample_chunks_fn,
                obs,
                task_description,
                seed,
                t,
                sigmas[offset],
            )
            stored = tape["u"][row_index].detach().cpu().contiguous()
            assert_tensors_bitwise_equal(
                proposed, stored, f"ep77 stored normalized chunk {offset}"
            )
            decoded = _decoded_chunk(runner, stored)
            decoded_hashes.append(array_sha256(decoded))
            for action in decoded:
                obs, stopped, capped = _step_without_semantic_read(env, action)
                environment_steps += 1
                require(
                    not stopped and not capped,
                    f"ep77 stopped inside stored chunk at t={environment_steps}",
                )

        require(environment_steps == recovery["stored_steps"] == 650, "recovery stored-step count")
        frames.append(
            capture_frame(
                obs=obs,
                tape=tape,
                episode_id=episode_id,
                env_seed=seed,
                t=recovery["missing_chunk_t"],
                source_row_index=row_indices[-1],
                boundary_role="n_plus_1",
                reference_rows=[("z_next", row_indices[-1])],
                legacy_encoder=legacy_encoder,
                task_description=task_description,
            )
        )
        missing_chunk = normalized_policy_chunk(
            runner,
            sample_chunks_fn,
            obs,
            task_description,
            seed,
            recovery["missing_chunk_t"],
            recovery["missing_raw_sigma_choice"],
        )
        missing_decoded = _decoded_chunk(runner, missing_chunk)
        first_action_sha = array_sha256(missing_decoded[:1])
        obs, stopped, capped = _step_without_semantic_read(env, missing_decoded[0])
        environment_steps += 1
        require(
            stopped is True and capped is False,
            "registered missing chunk did not halt on its first action",
        )
        require(
            environment_steps == recovery["source_terminal_t"] == 651,
            "registered recovery halt time mismatch",
        )
        frames.append(
            capture_frame(
                obs=obs,
                tape=tape,
                episode_id=episode_id,
                env_seed=seed,
                t=environment_steps,
                source_row_index=None,
                boundary_role="recovered_endpoint",
                reference_rows=[],
                legacy_encoder=None,
                task_description=task_description,
            )
        )
        require(
            len(frames) == 67,
            "recovery pass must expose t0..t650 boundaries plus t651 endpoint",
        )
        return RecoveryPass(
            frames=frames,
            decoded_chunk_hashes=decoded_hashes,
            recovered_chunk=missing_chunk,
            recovered_decoded_action_sha256=first_action_sha,
            environment_steps=environment_steps,
        )
    finally:
        env.close()


def compare_recovery_passes(left: RecoveryPass, right: RecoveryPass) -> None:
    assert_tensors_bitwise_equal(
        left.recovered_chunk,
        right.recovered_chunk,
        "recovered normalized chunk across fresh replays",
    )
    require(
        left.decoded_chunk_hashes == right.decoded_chunk_hashes,
        "decoded stored actions differ across recovery replays",
    )
    require(
        left.recovered_decoded_action_sha256
        == right.recovered_decoded_action_sha256,
        "decoded recovered action differs across replays",
    )
    require(len(left.frames) == len(right.frames) == 67, "recovery frame-count mismatch")
    for index, (first, second) in enumerate(zip(left.frames, right.frames)):
        identity_first = (
            first.episode_id,
            first.env_seed,
            first.t,
            first.source_row_index,
            first.boundary_role,
        )
        identity_second = (
            second.episode_id,
            second.env_seed,
            second.t,
            second.source_row_index,
            second.boundary_role,
        )
        require(identity_first == identity_second, f"recovery boundary {index} identity drift")
        require(first.raw_hashes == second.raw_hashes, f"recovery boundary {index} raw RGB drift")
        require(
            first.proprio_sha256 == second.proprio_sha256,
            f"recovery boundary {index} proprio drift",
        )


def run_ordinary_episode(
    *,
    runner: Any,
    env_factory: Callable[[], Any],
    tape: Mapping[str, Any],
    row_indices: Sequence[int],
    episode_id: int,
    env_seed: int,
    legacy_encoder: Callable[[Mapping[str, Any], str], torch.Tensor],
) -> tuple[list[CapturedFrame], int]:
    require(len(row_indices) == 75, "ordinary episode must have 75 stored chunks")
    _seed_replay(env_seed)
    runner.reset()
    env = env_factory()
    frames: list[CapturedFrame] = []
    environment_steps = 0
    try:
        obs, _ignored_reset_metadata = env.reset(seed=env_seed)
        require(isinstance(obs, Mapping), "reset observation must be a mapping")
        task_description = env.task_description
        require(
            isinstance(task_description, str) and task_description,
            "task description missing",
        )
        for offset, row_index in enumerate(row_indices):
            t = offset * C
            references = [("z", row_index)]
            if offset:
                references.append(("z_next", row_indices[offset - 1]))
            frames.append(
                capture_frame(
                    obs=obs,
                    tape=tape,
                    episode_id=episode_id,
                    env_seed=env_seed,
                    t=t,
                    source_row_index=row_index,
                    boundary_role="pre_action",
                    reference_rows=references,
                    legacy_encoder=legacy_encoder,
                    task_description=task_description,
                )
            )
            decoded = _decoded_chunk(runner, tape["u"][row_index])
            for action in decoded:
                obs, stopped, capped = _step_without_semantic_read(env, action)
                environment_steps += 1
                require(
                    not stopped and not capped,
                    f"ordinary episode {episode_id} stopped inside stored replay "
                    f"at t={environment_steps}",
                )
        frames.append(
            capture_frame(
                obs=obs,
                tape=tape,
                episode_id=episode_id,
                env_seed=env_seed,
                t=environment_steps,
                source_row_index=row_indices[-1],
                boundary_role="n_plus_1",
                reference_rows=[("z_next", row_indices[-1])],
                legacy_encoder=legacy_encoder,
                task_description=task_description,
            )
        )
        require(environment_steps == 750 and len(frames) == 76, "ordinary replay length drift")
        return frames, environment_steps
    finally:
        env.close()


@dataclass
class PanelReplay:
    frames: list[CapturedFrame]
    published_environment_steps: int
    recovery_audit_environment_steps: int
    recovery_audit: dict[str, Any]


def run_panel_replay(
    *,
    prepared: PreparedReplay,
    runner: Any,
    env_factory: Callable[[], Any],
    sample_chunks_fn: Callable[..., torch.Tensor],
    legacy_encoder: Callable[[Mapping[str, Any], str], torch.Tensor],
) -> PanelReplay:
    all_frames: list[CapturedFrame] = []
    published_steps = 0
    audit_steps = 0
    recovery = prepared.registration["terminal_recovery"]
    recovery_frames: list[CapturedFrame] | None = None
    if recovery["enabled"]:
        rows = prepared.selected_rows[recovery["episode_id"]]
        first = run_recovery_pass(
            runner=runner,
            env_factory=env_factory,
            sample_chunks_fn=sample_chunks_fn,
            tape=prepared.tape,
            row_indices=rows,
            registration=prepared.registration,
            legacy_encoder=legacy_encoder,
        )
        second = run_recovery_pass(
            runner=runner,
            env_factory=env_factory,
            sample_chunks_fn=sample_chunks_fn,
            tape=prepared.tape,
            row_indices=rows,
            registration=prepared.registration,
            legacy_encoder=legacy_encoder,
        )
        compare_recovery_passes(first, second)
        recovery_frames = first.frames
        published_steps += first.environment_steps
        audit_steps += first.environment_steps + second.environment_steps
        recovery_audit = {
            "enabled": True,
            "episode_id": recovery["episode_id"],
            "stored_chunks": 65,
            "stored_chunks_exact_bitwise": True,
            "recovered_chunk_tensor_sha256": tensor_sha256(first.recovered_chunk),
            "two_fresh_replays": 2,
            "cross_run_boundary_hashes_exact": True,
            "cross_run_decoded_actions_exact": True,
            "halted_at_registered_t": True,
            "phi_only": True,
            "dynamics_transition_eligible": False,
            "recovered_dynamics_rows_written": 0,
        }
    else:
        recovery_audit = {
            "enabled": False,
            "episode_id": None,
            "stored_chunks": None,
            "stored_chunks_exact_bitwise": None,
            "recovered_chunk_tensor_sha256": None,
            "two_fresh_replays": 0,
            "cross_run_boundary_hashes_exact": None,
            "cross_run_decoded_actions_exact": None,
            "halted_at_registered_t": None,
            "phi_only": True,
            "dynamics_transition_eligible": False,
            "recovered_dynamics_rows_written": 0,
        }

    for episode_spec in prepared.registration["panel"]["episodes"]:
        episode_id = episode_spec["episode_id"]
        if episode_spec["mode"] == "recover_last_action":
            require(recovery_frames is not None, "registered recovery frames missing")
            all_frames.extend(recovery_frames)
            continue
        frames, steps = run_ordinary_episode(
            runner=runner,
            env_factory=env_factory,
            tape=prepared.tape,
            row_indices=prepared.selected_rows[episode_id],
            episode_id=episode_id,
            env_seed=episode_spec["env_seed"],
            legacy_encoder=legacy_encoder,
        )
        all_frames.extend(frames)
        published_steps += steps
    return PanelReplay(
        frames=all_frames,
        published_environment_steps=published_steps,
        recovery_audit_environment_steps=audit_steps,
        recovery_audit=recovery_audit,
    )


def _jpeg_bytes(array: np.ndarray) -> bytes:
    oriented = np.ascontiguousarray(array[::-1, ::-1])
    buffer = io.BytesIO()
    Image.fromarray(oriented, mode="RGB").save(
        buffer,
        format="JPEG",
        quality=JPEG_QUALITY,
        subsampling=0,
        optimize=False,
    )
    return buffer.getvalue()


def _reject_manifest_semantics(value: Any, where: str = "$manifest") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            require(
                normalized not in FORBIDDEN_MANIFEST_KEYS,
                f"{where}.{key}: forbidden semantic key",
            )
            _reject_manifest_semantics(child, f"{where}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_manifest_semantics(child, f"{where}[{index}]")


def make_manifest_row(
    *,
    frame: CapturedFrame,
    frame_index: int,
    prepared: PreparedReplay,
    image_records: Mapping[str, Any],
) -> dict[str, Any]:
    row = {
        "schema": RGB_MANIFEST_SCHEMA,
        "frame_id": f"{prepared.panel_id}:{prepared.phase}:ep{frame.episode_id:02d}:t{frame.t:04d}",
        "panel_id": prepared.panel_id,
        "phase": prepared.phase,
        "task": TASK,
        "episode_id": frame.episode_id,
        "env_seed": frame.env_seed,
        "frame_index": frame_index,
        "t": frame.t,
        "source": {
            "tape_sha256": prepared.sanitizer_hashes["tape_sha256"],
            "source_row_index": frame.source_row_index,
            "boundary_role": frame.boundary_role,
        },
        "images": dict(image_records),
        "proprio": {
            "reference": frame.proprio_reference,
            "max_abs_error": frame.proprio_max_abs_error,
            "mean_abs_error": frame.proprio_mean_abs_error,
        },
    }
    validate_manifest_row(row)
    return row


def validate_manifest_row(row: Any) -> dict[str, Any]:
    item = dict(require_exact_keys(row, MANIFEST_FIELDS, "manifest row"))
    require(item["schema"] == RGB_MANIFEST_SCHEMA, "manifest schema")
    require(isinstance(item["frame_id"], str) and item["frame_id"], "manifest frame_id")
    require(item["task"] == TASK, "manifest task")
    for name in ("episode_id", "env_seed", "frame_index", "t"):
        require(type(item[name]) is int and item[name] >= 0, f"manifest {name}")
    source = require_exact_keys(item["source"], MANIFEST_SOURCE_FIELDS, "manifest source")
    require(is_sha256(source["tape_sha256"]), "manifest source tape SHA")
    require(
        source["source_row_index"] is None
        or (
            type(source["source_row_index"]) is int
            and source["source_row_index"] >= 0
        ),
        "manifest source row",
    )
    require(source["boundary_role"] in FRAME_ROLES, "manifest boundary role")
    require(
        (source["source_row_index"] is None)
        == (source["boundary_role"] == "recovered_endpoint"),
        "only recovered endpoint may have null source row",
    )
    images = require_exact_keys(item["images"], frozenset(CAMERAS), "manifest images")
    for camera in CAMERAS:
        image = require_exact_keys(
            images[camera], MANIFEST_IMAGE_FIELDS, f"manifest images.{camera}"
        )
        path = Path(image["path"])
        require(
            not path.is_absolute()
            and ".." not in path.parts
            and path.as_posix() == image["path"],
            "manifest image path",
        )
        require(
            is_sha256(image["sha256"])
            and is_sha256(image["raw_array_sha256"]),
            "manifest image SHA",
        )
        require(type(image["width"]) is int and image["width"] > 0, "manifest image width")
        require(type(image["height"]) is int and image["height"] > 0, "manifest image height")
    proprio = require_exact_keys(
        item["proprio"], MANIFEST_PROPRIO_FIELDS, "manifest proprio"
    )
    require(
        proprio["reference"]
        in {"z", "z_and_previous_z_next", "last_z_next", "cross_run_exact"},
        "manifest proprio reference",
    )
    for name in ("max_abs_error", "mean_abs_error"):
        require(
            type(proprio[name]) is float
            and np.isfinite(proprio[name])
            and proprio[name] >= 0.0,
            f"manifest proprio {name}",
        )
    _reject_manifest_semantics(item)
    return item


def _image_tree_sha256(root: Path, manifest_rows: Sequence[Mapping[str, Any]]) -> str:
    records = []
    for row in manifest_rows:
        for camera in CAMERAS:
            image = row["images"][camera]
            records.append(
                {
                    "path": image["path"],
                    "sha256": image["sha256"],
                    "bytes": (root / image["path"]).stat().st_size,
                }
            )
    records.sort(key=lambda item: item["path"])
    return canonical_sha256(records)


def _validate_authority(
    value: Any,
    registration: Mapping[str, Any],
    *,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    authority = dict(require_exact_keys(value, AUTHORITY_FIELDS, "RGB authority"))
    for name in (
        "freeze_complete_sha256",
        "campaign_manifest_sha256",
        "campaign_body_sha256",
        "campaign_projection_sha256",
    ):
        require(is_sha256(authority[name]), f"RGB authority {name} invalid")
    campaign = registration["campaign"]
    require(
        authority["campaign_id"] == campaign["campaign_id"],
        "RGB authority campaign_id drift",
    )
    require(
        authority["campaign_projection_sha256"]
        == campaign["projection_sha256"],
        "RGB authority projection drift",
    )
    require(
        authority["registration_slot"] == campaign["registration_slot"],
        "RGB authority registration slot drift",
    )
    require(
        isinstance(authority["output_complete_slot"], str)
        and authority["output_complete_slot"].endswith("/COMPLETE.json")
        and not Path(authority["output_complete_slot"]).is_absolute()
        and ".." not in Path(authority["output_complete_slot"]).parts,
        "RGB authority output COMPLETE slot invalid",
    )
    if registration["phase"] == "screen":
        require(
            authority["confirm_authorization_complete_sha256"] is None,
            "screen RGB must not consume confirm authorization",
        )
    else:
        require(
            is_sha256(authority["confirm_authorization_complete_sha256"]),
            "confirm RGB lacks an external confirm-authorization anchor",
        )
    if expected is not None:
        require(authority == dict(expected), "RGB authority differs from anchored bundle")
    return authority


def _validate_result_and_complete(
    stage: Path,
    registration: Mapping[str, Any],
    expected_authority: Mapping[str, Any] | None = None,
) -> None:
    result_path = stage / "RESULT.json"
    manifest_path = stage / "manifest.jsonl"
    complete_path = stage / "COMPLETE.json"
    producer_path = stage / "PRODUCER.py"
    for path in (result_path, manifest_path, complete_path, producer_path):
        require(
            path.is_file() and not path.is_symlink(),
            f"published file missing/symlink: {path.name}",
        )
    require(
        {path.name for path in stage.iterdir()}
        == {"RESULT.json", "manifest.jsonl", "COMPLETE.json", "PRODUCER.py", "images"},
        "RGB artifact root file closure mismatch",
    )
    result = _strict_json_file(result_path, "RGB RESULT")
    complete = _strict_json_file(complete_path, "RGB COMPLETE")
    require_exact_keys(result, RESULT_FIELDS, "RGB RESULT")
    require_exact_keys(
        result["registration"], RESULT_REGISTRATION_FIELDS, "RGB RESULT.registration"
    )
    result_authority = _validate_authority(
        result["authority"], registration, expected=expected_authority
    )
    require_exact_keys(result["inputs"], RESULT_INPUT_FIELDS, "RGB RESULT.inputs")
    require_exact_keys(result["replay"], RESULT_REPLAY_FIELDS, "RGB RESULT.replay")
    require_exact_keys(
        result["terminal_recovery_audit"],
        RESULT_RECOVERY_FIELDS,
        "RGB RESULT.terminal_recovery_audit",
    )
    require_exact_keys(result["output"], RESULT_OUTPUT_FIELDS, "RGB RESULT.output")
    require_exact_keys(result["safety"], RESULT_SAFETY_FIELDS, "RGB RESULT.safety")
    require(
        result["schema"] == RGB_RESULT_SCHEMA and result["status"] == "PASS",
        "RGB RESULT status",
    )
    require(result["task"] == TASK, "RGB RESULT task")
    require(
        result["terminal_recovery_audit"]["phi_only"] is True
        and result["terminal_recovery_audit"]["dynamics_transition_eligible"] is False
        and result["terminal_recovery_audit"]["recovered_dynamics_rows_written"] == 0,
        "RGB recovery isolation contract",
    )
    require(
        result["safety"]
        == {
            "summary_cli_supported": False,
            "summary_discovered_or_opened": False,
            "manifest_semantic_field_scan_passed": True,
            "recovered_endpoint_phi_only": True,
            "recovered_endpoint_dynamics_transition_eligible": False,
            "dynamics_tape_written": False,
        },
        "RGB RESULT safety contract",
    )
    require_exact_keys(complete, COMPLETE_FIELDS, "RGB COMPLETE")
    complete_authority = _validate_authority(
        complete["authority"], registration, expected=expected_authority
    )
    require(
        complete_authority == result_authority,
        "RESULT/COMPLETE authority seal drift",
    )
    require(
        complete["schema"] == RGB_COMPLETE_SCHEMA
        and complete["status"] == "atomic_success"
        and complete["atomic_commit"] is True,
        "RGB COMPLETE status",
    )
    require(complete["result_sha256"] == sha256_file(result_path), "RGB RESULT seal")
    require(complete["manifest_sha256"] == sha256_file(manifest_path), "RGB manifest seal")
    require(complete["producer_sha256"] == sha256_file(producer_path), "RGB producer seal")
    require(
        complete["producer_sha256"]
        == registration["sources"]["formal_replay"]["sha256"],
        "RGB producer does not equal the pre-registered formal replay source",
    )
    require(
        result["output"]["manifest_sha256"] == complete["manifest_sha256"],
        "RESULT/COMPLETE manifest seal",
    )
    require(
        result["output"]["producer_sha256"] == complete["producer_sha256"],
        "RESULT/COMPLETE producer seal",
    )
    require(
        complete["registration_sha256"] == result["registration"]["file_sha256"]
        and complete["registration_self_sha256"]
        == result["registration"]["canonical_self_sha256"],
        "RESULT/COMPLETE registration seal",
    )
    require(
        complete["panel_id"] == result["panel_id"]
        and complete["phase"] == result["phase"],
        "RESULT/COMPLETE panel or phase drift",
    )
    rows = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = loads_strict_json(line, where=f"manifest line {line_number}")
            rows.append(validate_manifest_row(value))
    require(len(rows) == result["output"]["frames"], "manifest frame count")
    require(
        [row["frame_index"] for row in rows] == list(range(len(rows))),
        "manifest frame_index sequence",
    )
    require(len({row["frame_id"] for row in rows}) == len(rows), "duplicate manifest frame_id")
    require(
        all(
            row["panel_id"] == result["panel_id"]
            and row["phase"] == result["phase"]
            and row["episode_id"] in result["episode_ids"]
            and row["source"]["tape_sha256"]
            == result["inputs"]["sanitized_tape_sha256"]
            for row in rows
        ),
        "manifest/result binding drift",
    )
    for row in rows:
        for camera in CAMERAS:
            image = row["images"][camera]
            path = (stage / image["path"]).resolve()
            require(path.is_relative_to(stage.resolve()), "manifest image escapes output")
            require(path.is_file() and not path.is_symlink(), "manifest image missing/symlink")
            require(sha256_file(path) == image["sha256"], "manifest JPEG drift")
    tree_hash = _image_tree_sha256(stage, rows)
    require(
        tree_hash
        == result["output"]["image_tree_sha256"]
        == complete["image_tree_sha256"],
        "image tree seal",
    )
    require(
        result["output"]["jpeg_files"] == len(rows) * len(CAMERAS)
        and result["output"]["raw_rgb_arrays"] == len(rows) * len(CAMERAS),
        "RGB output image counts",
    )
    images = list((stage / "images").iterdir())
    require(
        len(images) == result["output"]["jpeg_files"]
        and all(path.is_file() and not path.is_symlink() for path in images),
        "RGB image directory closure mismatch",
    )


def publish_replay(prepared: PreparedReplay, replay: PanelReplay) -> None:
    source_path = Path(__file__).resolve()

    def build(stage: Path) -> None:
        images_root = stage / "images"
        images_root.mkdir()
        manifest_rows = []
        for frame_index, frame in enumerate(replay.frames):
            image_records = {}
            for camera, raw in frame.raw_arrays.items():
                filename = f"ep{frame.episode_id:02d}_t{frame.t:04d}_{camera}.jpg"
                relative = Path("images") / filename
                payload = _jpeg_bytes(raw)
                _write_bytes(stage / relative, payload)
                height, width = raw.shape[:2]
                image_records[camera] = {
                    "path": relative.as_posix(),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "raw_array_sha256": frame.raw_hashes[camera],
                    "width": int(width),
                    "height": int(height),
                }
            manifest_rows.append(
                make_manifest_row(
                    frame=frame,
                    frame_index=frame_index,
                    prepared=prepared,
                    image_records=image_records,
                )
            )
        _fsync_directory(images_root)
        manifest_path = stage / "manifest.jsonl"
        _write_bytes(manifest_path, b"".join(canonical_bytes(row) for row in manifest_rows))
        _write_bytes(stage / "PRODUCER.py", source_path.read_bytes())
        require(
            sha256_file(stage / "PRODUCER.py")
            == prepared.registration["sources"]["formal_replay"]["sha256"],
            "producer snapshot does not equal registered formal replay source",
        )
        recovery_count = int(prepared.registration["terminal_recovery"]["enabled"])
        ordinary_count = len(prepared.registration["panel"]["episodes"]) - recovery_count
        result = {
            "schema": RGB_RESULT_SCHEMA,
            "status": "PASS",
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "registration": {
                "registration_id": prepared.registration["registration_id"],
                "file_sha256": prepared.registration_sha256,
                "canonical_self_sha256": prepared.registration["self_sha256"],
            },
            "authority": dict(prepared.authority),
            "panel_id": prepared.panel_id,
            "phase": prepared.phase,
            "task": TASK,
            "episode_ids": [
                item["episode_id"]
                for item in prepared.registration["panel"]["episodes"]
            ],
            "inputs": {
                "sanitized_tape_sha256": prepared.sanitizer_hashes["tape_sha256"],
                "source_tape_sha256": prepared.registration["sanitized_source"][
                    "source_tape_sha256"
                ],
            },
            "replay": {
                "episodes": len(prepared.registration["panel"]["episodes"]),
                "ordinary_episodes": ordinary_count,
                "recovery_episodes": recovery_count,
                "published_environment_steps": replay.published_environment_steps,
                "recovery_audit_environment_steps": replay.recovery_audit_environment_steps,
                "fresh_recovery_passes": 2 if recovery_count else 0,
                "all_registered_legacy2073_oracles_bit_exact": True,
                "all_registered_proprio_oracles_bit_exact": True,
                "all_stored_recovery_chunks_bit_exact": True if recovery_count else None,
            },
            "terminal_recovery_audit": replay.recovery_audit,
            "output": {
                "manifest_sha256": sha256_file(manifest_path),
                "frames": len(manifest_rows),
                "jpeg_files": len(manifest_rows) * len(CAMERAS),
                "raw_rgb_arrays": len(manifest_rows) * len(CAMERAS),
                "image_tree_sha256": _image_tree_sha256(stage, manifest_rows),
                "producer_sha256": sha256_file(stage / "PRODUCER.py"),
            },
            "safety": {
                "summary_cli_supported": False,
                "summary_discovered_or_opened": False,
                "manifest_semantic_field_scan_passed": True,
                "recovered_endpoint_phi_only": True,
                "recovered_endpoint_dynamics_transition_eligible": False,
                "dynamics_tape_written": False,
            },
        }
        _write_json(stage / "RESULT.json", result)
        complete = {
            "schema": RGB_COMPLETE_SCHEMA,
            "status": "atomic_success",
            "atomic_commit": True,
            "registration_sha256": prepared.registration_sha256,
            "registration_self_sha256": prepared.registration["self_sha256"],
            "authority": dict(prepared.authority),
            "panel_id": prepared.panel_id,
            "phase": prepared.phase,
            "result_sha256": sha256_file(stage / "RESULT.json"),
            "manifest_sha256": result["output"]["manifest_sha256"],
            "producer_sha256": result["output"]["producer_sha256"],
            "image_tree_sha256": result["output"]["image_tree_sha256"],
        }
        _write_json(stage / "COMPLETE.json", complete)

    atomic_publish_directory(
        prepared.output_path,
        build,
        lambda stage: _validate_result_and_complete(
            stage, prepared.registration, prepared.authority
        ),
        trusted_target_root=prepared.target_root,
    )


def _resolve_registered_tokenizer_snapshot(
    tokenizer: Mapping[str, Any], snapshot_download_fn: Callable[..., str]
) -> Path:
    """Resolve exactly the frozen revision, never a mutable repo main ref."""
    live_tokenizer_root = Path(
        snapshot_download_fn(
            repo_id=tokenizer["repo_id"],
            revision=tokenizer["revision"],
            local_files_only=True,
        )
    ).resolve()
    require(
        live_tokenizer_root.name == tokenizer["revision"],
        "the tokenizer repo main ref is not the registered revision",
    )
    for member in tokenizer["files"]:
        registered_path = Path(tokenizer["local_root"]) / member["relative_path"]
        live_path = live_tokenizer_root / member["relative_path"]
        require(live_path.is_file(), f"live tokenizer member missing: {live_path}")
        require(
            live_path.stat().st_size == member["bytes"]
            and sha256_file(live_path) == member["sha256"]
            and sha256_file(registered_path) == member["sha256"],
            f"live tokenizer member drift: {member['relative_path']}",
        )
    return live_tokenizer_root


def load_runtime(
    prepared: PreparedReplay,
) -> tuple[
    Any,
    Callable[[], Any],
    Callable[..., torch.Tensor],
    Callable[[Mapping[str, Any], str], torch.Tensor],
]:
    """Import policy/simulator code only after all validation-only exits."""
    require(torch.cuda.is_available(), "registered CUDA runtime is unavailable")
    from huggingface_hub import snapshot_download

    tokenizer = prepared.registration["models"]["tokenizer"]
    _resolve_registered_tokenizer_snapshot(tokenizer, snapshot_download)
    from lcwm.chassis import Pi05Runner
    from lcwm.sampler import prefix_forward, sample_chunks
    from lcwm.v082_m0 import masked_prefix_mean
    from lcwm.v080_bench import make_v080_env

    model_root = prepared.registration["models"]["pi05"]["local_root"]
    runner = Pi05Runner(
        model_id=model_root,
        suite_name="libero_10",
        device="cuda",
        n_action_steps=C,
    )
    require(runner.model_id == model_root, "live runner model binding drift")
    require(runner.policy_cfg.n_action_steps == C, "live runner action commitment drift")

    @torch.no_grad()
    def legacy_encoder(
        obs: Mapping[str, Any], task_description: str
    ) -> torch.Tensor:
        policy_observation = runner._obs_to_policy_batch(obs, task_description)
        prefix = prefix_forward(runner.policy, policy_observation)
        hidden = prefix.hidden[0].detach().to(dtype=torch.float32, device="cpu")
        mask = prefix.pad_masks[0].detach().to(device="cpu").bool()
        pooled = masked_prefix_mean(hidden, mask.to(dtype=torch.float32))
        encoded = torch.cat([pooled, observation_proprio(obs)]).contiguous()
        require(
            encoded.dtype == torch.float32 and encoded.shape == (LATENT_DIM,),
            "live legacy2073 encoder layout drift",
        )
        require(bool(torch.isfinite(encoded).all()), "live legacy2073 is non-finite")
        return encoded

    return runner, lambda: make_v080_env(TASK), sample_chunks, legacy_encoder


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanitized-tape", type=Path, required=True)
    parser.add_argument("--freeze-bundle", type=Path, required=True)
    parser.add_argument("--expected-freeze-complete-sha256", required=True)
    parser.add_argument("--registration-slot", required=True)
    parser.add_argument("--expected-confirm-authorization-complete-sha256")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if not is_sha256(args.expected_freeze_complete_sha256):
        parser.error("--expected-freeze-complete-sha256 must be 64 lowercase hex")
    if (
        args.expected_confirm_authorization_complete_sha256 is not None
        and not is_sha256(args.expected_confirm_authorization_complete_sha256)
    ):
        parser.error(
            "--expected-confirm-authorization-complete-sha256 must be 64 lowercase hex"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    prepared = prepare_inputs(
        sanitized_tape=args.sanitized_tape,
        freeze_bundle_root=args.freeze_bundle,
        expected_freeze_complete_sha256=args.expected_freeze_complete_sha256,
        registration_slot=args.registration_slot,
        expected_confirm_authorization_complete_sha256=(
            args.expected_confirm_authorization_complete_sha256
        ),
    )
    if args.validate_only:
        assert_no_simulator_modules_imported()
        print(
            json.dumps(
                {
                    "status": "VALID",
                    "freeze_complete_sha256": prepared.authority[
                        "freeze_complete_sha256"
                    ],
                    "campaign_id": prepared.authority["campaign_id"],
                    "registration_slot": prepared.registration_slot,
                    "registration_sha256": prepared.registration_sha256,
                    "panel_id": prepared.panel_id,
                    "phase": prepared.phase,
                    "selected_episode_ids": sorted(prepared.selected_rows),
                    "simulator_imported": False,
                    "summary_opened": False,
                    "output_created": False,
                    "output_complete_slot": prepared.output_complete_slot,
                    "confirm_authorization_complete_sha256": (
                        prepared.confirm_authorization_complete_sha256
                    ),
                },
                indent=2,
            )
        )
        return 0

    # Validate once before loading runtime modules, then again after the model
    # loader but before make_env/reset.  This catches policy/config drift during
    # construction without allowing a single environment observation to open.
    revalidate_static_inputs(prepared)
    runner, env_factory, sample_chunks_fn, legacy_encoder = load_runtime(prepared)
    revalidate_static_inputs(prepared)
    replay = run_panel_replay(
        prepared=prepared,
        runner=runner,
        env_factory=env_factory,
        sample_chunks_fn=sample_chunks_fn,
        legacy_encoder=legacy_encoder,
    )
    revalidate_static_inputs(prepared)
    publish_replay(prepared, replay)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": prepared.output_path.as_posix(),
                "panel_id": prepared.panel_id,
                "phase": prepared.phase,
                "frames": len(replay.frames),
                "summary_opened": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
