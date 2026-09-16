#!/usr/bin/env python
"""Fail-closed contract core for the EVOLVE-1 rich-latent D1 collection.

This module deliberately does **not** yet integrate the real LIBERO / pi0.5
runner.  It makes the parts that can be established without an environment
interaction executable and testable:

* a content-addressed registration, D0 tape, and actor contract validated before
  any environment package is imported;
* one content-addressed, runtime-fingerprinted residual policy per episode;
* a 750-step loop that ignores task-success termination but halts on the raw
  environment ``info["done"]`` or truncation before the horizon;
* 75 action chunks and 76 N+1 rich-latent boundaries per valid episode;
* exact normalized and decoded base/executed action traces, with preregistered
  non-zero perturbation coverage, and an outcome-only sidecar;
* atomic artifact publication with a final ``COMPLETE.json`` seal.

``--validate-only`` is safe and performs no simulator import or reset.
``--collect`` currently raises ``RealCollectorNotImplemented`` *after* all
registration inputs have validated and still before any simulator import.  That
is intentional: a formal registration must not be created until Gate 0, M0, and
two rich8217 chain3 collector actors exist.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
import random
import shutil
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import torch


REPO = Path(__file__).resolve().parent.parent

REGISTRATION_SCHEMA = "v246_evolve1_d1_mechanical_registration_v5"
ACTOR_SCHEMA = "v246_chain3_rich_collector_actor_v3"
SEED_LEDGER_SCHEMA = "v246_evolve1_seed_ledger_v3"
D0_TAPE_SCHEMA = "v246_evolve1_d0_rich_tape_v1"
DATA_TAPE_SCHEMA = "v246_evolve1_d1_role_isolated_tape_v3"
BOUNDARY_SCHEMA = "v246_evolve1_d1_boundaries_v1"
TRACE_SCHEMA = "v246_evolve1_d1_action_trace_v3"
SIDECAR_SCHEMA = "v246_evolve1_d1_evaluation_sidecar_v1"
RESULT_SCHEMA = "v246_evolve1_d1_mechanical_result_v2"
COMPLETE_SCHEMA = "v246_evolve1_d1_mechanical_complete_v2"
M0_CHECKPOINT_SCHEMA = "v246_evolve1_m0_checkpoint_v1"
M0_CONFIG_SCHEMA = "v246_evolve1_m0_config_v1"
PHI_CHECKPOINT_SCHEMA = "v246_evolve1_count_phi_checkpoint_v1"
PHI_CONFIG_SCHEMA = "v246_count_phi_config_v1"
BASE_VLA_CHECKPOINT_SCHEMA = "v246_evolve1_base_vla_checkpoint_v1"
BASE_VLA_CONFIG_SCHEMA = "v246_evolve1_base_vla_config_v1"
TOKENIZER_SCHEMA = "v246_evolve1_tokenizer_manifest_v1"
ACTOR_CONFIG_SCHEMA = "v246_evolve1_residual_actor_config_v1"
RUNTIME_BINDING_SCHEMA = "v246_evolve1_runtime_binding_v1"
TRAINING_ACCESS_SCHEMA = "v246_evolve1_training_access_v1"
RICH_ENCODING_SCHEMA = "v246_evolve1_rich_encoding_evidence_v1"
ATTEMPT_LEDGER_SCHEMA = "v246_evolve1_d1_attempt_ledger_v2"
ATTEMPT_EVENT_SCHEMA = "v246_evolve1_d1_attempt_event_v2"
REAL_SEMANTIC_BINDING_IMPLEMENTED = False
CORE_OWNED_POLICY_SEPARATION_IMPLEMENTED = False
RAW_OBSERVATION_BINDING_IMPLEMENTED = False
FORMAL_DETERMINISM_ATTESTATION_IMPLEMENTED = False
ATTEMPT_LEDGER_EXTERNAL_ANCHOR_IMPLEMENTED = False
MECHANICAL_EVIDENCE_CLASS = "mechanical_contract_test_only_no_gate1"
MECHANICAL_RESULT_STATUS = "MECHANICAL_CONTRACT_TEST_ONLY"
MECHANICAL_COMPLETE_STATUS = "mechanical_test_only_atomic_success"

TASK = "chain3_lr2"
HORIZON_STEPS = 750
C = 10
ACTION_DIM = 7
CHUNKS_PER_EPISODE = HORIZON_STEPS // C
BOUNDARIES_PER_EPISODE = CHUNKS_PER_EPISODE + 1
CAMERA_BLOCK_DIM = 2048
RICH_DIM = 4 * CAMERA_BLOCK_DIM + 25
PROPRIO_DIM = 25
PROPRIO_KEYS = (
    "robot_state.eef.pos",
    "robot_state.eef.quat",
    "robot_state.gripper.qpos",
    "robot_state.gripper.qvel",
    "robot_state.joints.pos",
    "robot_state.joints.vel",
)
CAMERA_TOKEN_BLOCKS = ((0, 256), (256, 512))
COLLECTOR_IDS = ("A", "B")
SPLITS = ("fit", "calibration")
EXPECTED_EPISODES = 96
EXPECTED_PER_COLLECTOR = 48
EXPECTED_FIT_PER_COLLECTOR = 36
EXPECTED_CAL_PER_COLLECTOR = 12
EXPECTED_GATE3_ACTOR_TRAINING_SEEDS = 16
MIN_REGISTERED_CHANGED_ACTION_FRACTION = 0.50
MIN_REGISTERED_DECODED_ACTION_DELTA_L2 = 0.05
MEANINGFUL_DECODED_STEP_L2 = 0.01
MIN_MEANINGFUL_ACTION_FRACTION = 0.90
MIN_DECODED_STEP_L2_P10 = 0.01
MAX_DECODED_STEP_L2 = 0.25
REGISTERED_RESIDUAL_SCALE = 0.04
TRAINING_FEATURE_FIELDS = ("z", "u", "z_next")
SPLIT_BOOKKEEPING_FIELDS = ("episode", "t")
CONTROLLER_RUNTIME_COMPONENTS = (
    "actor",
    "base_vla",
    "action_decoder",
)
REGISTERED_RUNTIME_COMPONENTS = (
    "base_vla",
    "action_decoder",
    "feature_encoder",
    "environment",
)

ACTOR_FIELDS = frozenset(
    {
        "schema",
        "task",
        "input_schema",
        "input_dim",
        "c",
        "adim",
        "collector_id",
        "training_data_role",
        "env_steps",
        "d0_tape_sha256",
        "m0_sha256",
        "phi_sha256",
        "actor_seed",
        "scale",
        "fixed_hyperparameters_sha256",
        "architecture_config",
        "semantic_binding",
        "normalizer_mu",
        "normalizer_sd",
        "normalizer_mu_sha256",
        "normalizer_sd_sha256",
        "state_dict",
        "runtime_fingerprint_sha256",
    }
)

DATA_TAPE_FIELDS = frozenset(
    {
        "schema",
        "z",
        "u",
        "z_next",
        "episode",
        "t",
        "sigma",
        "data_role",
        "task",
        "c",
        "latent_dim",
        "proprio_dim",
        "proprio_keys",
        "training_feature_fields",
        "split_bookkeeping_fields",
        "legacy_outcome_fields",
    }
)
FORBIDDEN_DATA_FIELDS = frozenset(
    {
        "success",
        "success_step",
        "events",
        "outcome",
        "policy_id",
        "collector_id",
        "episode_length",
        "trajectory_length",
    }
)
D0_REQUIRED_FIELDS = frozenset(
    {
        "schema",
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
        "training_feature_fields",
        "split_bookkeeping_fields",
        "legacy_outcome_fields",
    }
)
D0_ALLOWED_OUTCOME_FIELDS = frozenset({"success"})

REGISTRATION_FIELDS = frozenset(
    {
        "schema",
        "status",
        "evidence_class",
        "collection",
        "inputs",
        "runtime_components",
        "provenance",
        "self_sha256",
    }
)
COLLECTION_FIELDS = frozenset(
    {
        "task",
        "horizon_steps",
        "c",
        "action_shape",
        "rich_latent_dim",
        "proprio_dim",
        "proprio_keys",
        "camera_token_blocks",
        "expected_episodes",
        "expected_per_collector",
        "fit_per_collector",
        "calibration_per_collector",
        "perturbation_scope",
        "chunk_noise",
        "sigma_value",
        "success_termination",
        "technical_done_source",
        "outcome_storage",
        "decoded_action_trace",
        "changed_action_step_definition",
        "decoded_action_delta_norm",
        "min_changed_action_fraction_per_episode",
        "min_decoded_action_delta_l2_per_episode",
        "meaningful_decoded_step_l2",
        "min_meaningful_action_fraction_per_episode",
        "min_decoded_step_l2_p10_per_episode",
        "max_decoded_step_l2_per_episode",
        "attempt_ledger_path",
        "attempt_ledger_policy",
        "episode_assignments",
    }
)
ASSIGNMENT_FIELDS = frozenset({"episode_id", "env_seed", "collector_id", "split"})
INPUT_FIELDS = frozenset(
    {
        "d0_tape",
        "m0_checkpoint",
        "phi_checkpoint",
        "seed_ledger",
        "base_vla_checkpoint",
        "tokenizer_manifest",
        "actors",
    }
)
PROVENANCE_FIELDS = frozenset({"source_files_sha256"})
ARTIFACT_REF_FIELDS = frozenset({"path", "sha256"})

REQUIRED_SOURCE_PATHS = frozenset(
    {
        "scripts/collect_v246_evolve1_d1.py",
        "lcwm/chassis.py",
        "lcwm/sampler.py",
        "lcwm/v082_m0.py",
        "lcwm/v080_bench.py",
        "lcwm/loho.py",
        "bddl/chains/chain3_lr2.bddl",
    }
)
ENVIRONMENT_MODULE_PREFIXES = ("lerobot", "libero", "robosuite", "mujoco")


class ContractError(RuntimeError):
    """A prospective EVOLVE-1 collection invariant was violated."""


class TechnicalTermination(ContractError):
    """The raw environment ended before the registered 750-step horizon."""


class PolicyAssignmentError(ContractError):
    """An episode's immutable collector policy changed or did not match."""


class RealCollectorNotImplemented(ContractError):
    """The formal real-runner integration remains deliberately disabled."""


@dataclass(frozen=True)
class ArtifactRef:
    path: Path
    sha256: str
    payload: bytes


@dataclass(frozen=True)
class EpisodeAssignment:
    episode_id: int
    env_seed: int
    collector_id: str
    split: str


@dataclass(frozen=True)
class ActorInfo:
    collector_id: str
    path: Path
    sha256: str
    actor_seed: int
    scale: float
    fixed_hyperparameters_sha256: str
    runtime_fingerprint_sha256: str


@dataclass(frozen=True)
class ActionChangeContract:
    min_changed_action_fraction_per_episode: float
    min_decoded_action_delta_l2_per_episode: float
    meaningful_decoded_step_l2: float
    min_meaningful_action_fraction_per_episode: float
    min_decoded_step_l2_p10_per_episode: float
    max_decoded_step_l2_per_episode: float


@dataclass(frozen=True)
class ActionChangeMetrics:
    changed_action_steps: int
    meaningful_action_steps: int
    decoded_action_delta_l2: float
    decoded_step_l2_p10: float
    decoded_step_l2_median: float
    decoded_step_l2_p90: float
    decoded_step_l2_max: float


@dataclass(frozen=True)
class D0Info:
    path: Path
    sha256: str
    rows: int
    episodes: int


@dataclass(frozen=True)
class ModelArtifactInfo:
    path: Path
    sha256: str
    schema: str
    runtime_state_sha256: str


@dataclass(frozen=True)
class TokenizerInfo:
    path: Path
    sha256: str
    model_id: str
    vocab_size: int


@dataclass(frozen=True)
class RuntimeComponentInfo:
    name: str
    artifact_sha256: str
    runtime_state_sha256: str
    runtime_fingerprint_sha256: str
    runtime_config: dict[str, Any]


@dataclass(frozen=True)
class ValidatedRegistration:
    path: Path
    file_sha256: str
    self_sha256: str
    raw_bytes: bytes
    payload: dict[str, Any]
    assignments: tuple[EpisodeAssignment, ...]
    actors: dict[str, ActorInfo]
    d0: D0Info
    action_change: ActionChangeContract
    m0: ModelArtifactInfo
    phi: ModelArtifactInfo
    base_vla: ModelArtifactInfo
    tokenizer: TokenizerInfo
    runtime_components: dict[str, RuntimeComponentInfo]
    attempt_ledger_path: Path


@dataclass(frozen=True)
class RichEncoding:
    value: torch.Tensor
    evidence: dict[str, Any]


@dataclass(frozen=True, slots=True)
class EpisodeCollection:
    assignment: EpisodeAssignment
    boundaries: torch.Tensor
    u_base: torch.Tensor
    delta: torch.Tensor
    u_exec: torch.Tensor
    env_action_base: torch.Tensor
    env_action_exec: torch.Tensor
    changed_action_steps: int
    meaningful_action_steps: int
    decoded_action_delta_l2: float
    decoded_step_l2_p10: float
    decoded_step_l2_median: float
    decoded_step_l2_p90: float
    decoded_step_l2_max: float
    actor_artifact_sha256: str
    actor_runtime_fingerprint_sha256: str
    runtime_component_fingerprints_sha256: dict[str, str]
    boundary_evidence: tuple[dict[str, Any], ...]
    evaluation_sidecar: dict[str, Any]


class EpisodeController(Protocol):
    """Adapter contract for a frozen pi0.5 + one frozen residual actor."""

    collector_id: str
    actor_artifact_sha256: str
    base_vla_artifact_sha256: str
    action_decoder_artifact_sha256: str

    def recompute_runtime_fingerprints_sha256(self) -> Mapping[str, str]:
        """Recompute actor/base-VLA/decoder fingerprints from live state/config."""

    def reset_episode(self, seed: int) -> None:
        """Reset only policy/controller state for one registered episode."""

    def propose(
        self,
        obs: Any,
        rich_z: torch.Tensor,
        chunk_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized ``(u_base, delta)``, each exactly ``[10, 7]``."""

    def decode(self, normalized_action: torch.Tensor) -> np.ndarray:
        """Purely map normalized input to executable float32 ``[10, 7]`` actions."""


class RichFeatureEncoder(Protocol):
    artifact_sha256: str

    def recompute_runtime_fingerprint_sha256(self) -> str:
        """Recompute the registered rich encoder fingerprint from live state."""

    def encode(self, obs: Any) -> RichEncoding:
        """Return rich8217 and hashes proving its four blocks plus proprio source."""


class RuntimeEnvironment(Protocol):
    artifact_sha256: str

    def recompute_runtime_fingerprint_sha256(self) -> str:
        """Recompute the environment fingerprint from live code/config."""

    def reset(self, seed: int) -> Any:
        """Reset the environment after all prospective checks pass."""

    def step(self, action: np.ndarray) -> Any:
        """Execute one exact decoded action."""


def _unresolved_absolute_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded
    return Path.cwd() / expanded


def _reject_final_symlink(path: Path, label: str) -> None:
    candidate = _unresolved_absolute_path(path)
    try:
        metadata = os.lstat(candidate)
    except FileNotFoundError as error:
        raise ContractError(f"{label} does not exist: {candidate}") from error
    if stat.S_ISLNK(metadata.st_mode):
        raise ContractError(f"{label} must not be a symbolic link: {candidate}")


def read_regular_file_snapshot(path: Path, label: str = "artifact") -> bytes:
    """Read, lock, and identify one regular file through one descriptor.

    ``O_NOFOLLOW`` and the pre-open ``lstat`` reject final-component symlinks.
    The two ``fstat`` calls ensure the descriptor did not change identity or
    size while its bytes were read.  Hashing/parsing callers consume only the
    returned snapshot, never a second path lookup.
    """
    candidate = _unresolved_absolute_path(path)
    _reject_final_symlink(candidate, label)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(candidate, flags)
    except OSError as error:
        raise ContractError(f"cannot safely open {label}: {candidate}") from error
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ContractError(f"{label} must be a regular file: {candidate}")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 16 << 20)
            if not block:
                break
            chunks.append(block)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after or len(payload) != after.st_size:
            raise ContractError(f"{label} changed while its snapshot was read")
        return payload
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def sha256_file(path: Path, chunk_size: int = 16 << 20) -> str:
    del chunk_size
    return hashlib.sha256(read_regular_file_snapshot(path, str(path))).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def exact_json_equal(actual: Any, expected: Any) -> bool:
    """JSON equality that does not collapse bool/int/float into one value."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            exact_json_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            exact_json_equal(left, right) for left, right in zip(actual, expected)
        )
    return bool(actual == expected)


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    header = canonical_bytes({"dtype": str(tensor.dtype), "shape": list(tensor.shape)})
    raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
    return hashlib.sha256(header + raw).hexdigest()


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def load_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"{label} must contain valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ContractError(f"{label} must contain one JSON object")
    return value


def load_json(path: Path) -> dict[str, Any]:
    return load_json_bytes(read_regular_file_snapshot(path, str(path)), str(path))


def load_torch_bytes(payload: bytes, label: str) -> Any:
    try:
        return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    except Exception as error:
        raise ContractError(f"{label} is not a safe torch artifact") from error


def load_torch(path: Path, label: str | None = None) -> Any:
    artifact_label = label or str(path)
    return load_torch_bytes(
        read_regular_file_snapshot(path, artifact_label),
        artifact_label,
    )


def _attempt_row_sha256(row: Mapping[str, Any]) -> str:
    return canonical_sha256({key: value for key, value in row.items() if key != "row_sha256"})


def parse_attempt_ledger_bytes(
    payload: bytes,
    registration_sha256: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(payload.splitlines(), start=1):
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ContractError(f"attempt ledger JSON error at line {line_number}") from error
        if not isinstance(row, dict):
            raise ContractError("attempt ledger row must be an object")
        rows.append(row)
    if not rows:
        raise ContractError("attempt ledger is empty")
    header_fields = {
        "schema",
        "record_type",
        "registration_sha256",
        "previous_row_sha256",
        "row_sha256",
    }
    header = rows[0]
    if set(header) != header_fields:
        raise ContractError("attempt ledger header field closure mismatch")
    if (
        header.get("schema") != ATTEMPT_LEDGER_SCHEMA
        or header.get("record_type") != "header"
        or header.get("registration_sha256") != registration_sha256
        or header.get("previous_row_sha256") != "0" * 64
        or header.get("row_sha256") != _attempt_row_sha256(header)
    ):
        raise ContractError("attempt ledger header mismatch")
    event_fields = {
        "schema",
        "record_type",
        "sequence",
        "registration_sha256",
        "episode_id",
        "env_seed",
        "collector_id",
        "event",
        "reason",
        "episode_payload_sha256",
        "previous_row_sha256",
        "row_sha256",
    }
    states: dict[int, str] = {}
    identities: dict[int, tuple[int, str]] = {}
    previous_sha = header["row_sha256"]
    for sequence, row in enumerate(rows[1:]):
        if set(row) != event_fields:
            raise ContractError("attempt event field closure mismatch")
        if (
            row.get("schema") != ATTEMPT_EVENT_SCHEMA
            or row.get("record_type") != "event"
            or type(row.get("sequence")) is not int
            or row.get("sequence") != sequence
            or row.get("registration_sha256") != registration_sha256
            or row.get("previous_row_sha256") != previous_sha
            or row.get("row_sha256") != _attempt_row_sha256(row)
        ):
            raise ContractError(f"attempt event {sequence} hash/identity mismatch")
        episode_id = row.get("episode_id")
        env_seed = row.get("env_seed")
        collector_id = row.get("collector_id")
        event = row.get("event")
        reason = row.get("reason")
        episode_payload_sha = row.get("episode_payload_sha256")
        if (
            type(episode_id) is not int
            or type(env_seed) is not int
            or collector_id not in COLLECTOR_IDS
            or event not in {"started", "completed", "technical_failure", "contract_failure"}
        ):
            raise ContractError(f"attempt event {sequence} value mismatch")
        prior = states.get(episode_id)
        if event == "started":
            if (
                prior is not None
                or reason is not None
                or episode_payload_sha is not None
            ):
                raise ContractError("attempt ledger retries/replacement are forbidden")
            states[episode_id] = "started"
            identities[episode_id] = (env_seed, collector_id)
        else:
            if prior != "started":
                raise ContractError("attempt completion lacks one unique start")
            if identities[episode_id] != (env_seed, collector_id):
                raise ContractError("attempt completion changed seed/controller identity")
            if event == "completed" and reason is not None:
                raise ContractError("completed attempt cannot carry a failure reason")
            if event == "completed" and not is_sha256(episode_payload_sha):
                raise ContractError("completed attempt requires an episode payload SHA")
            if event != "completed" and (not isinstance(reason, str) or not reason):
                raise ContractError("failed attempt requires a reason")
            if event != "completed" and episode_payload_sha is not None:
                raise ContractError("failed attempt cannot carry an episode payload SHA")
            states[episode_id] = event
        previous_sha = row["row_sha256"]
    return rows


def read_attempt_ledger(path: Path, registration_sha256: str) -> list[dict[str, Any]]:
    payload = read_regular_file_snapshot(path, "attempt ledger")
    return parse_attempt_ledger_bytes(payload, registration_sha256)


@dataclass(slots=True)
class AttemptLedger:
    path: Path
    registration_sha256: str
    formal_path_bound: bool
    device: int
    inode: int
    mtime_ns: int
    ctime_ns: int
    expected_payload: bytes

    @staticmethod
    def _read_descriptor(descriptor: int) -> bytes:
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                return b"".join(chunks)
            chunks.append(block)

    @staticmethod
    def _write_all(descriptor: int, payload: bytes) -> None:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise ContractError("attempt ledger append made no progress")
            view = view[written:]

    @classmethod
    def _create(
        cls,
        path: Path,
        registration: ValidatedRegistration,
        *,
        formal_path_bound: bool,
    ) -> "AttemptLedger":
        unresolved = _unresolved_absolute_path(path)
        path = unresolved.parent.resolve() / unresolved.name
        path.parent.mkdir(parents=True, exist_ok=True)
        header = {
            "schema": ATTEMPT_LEDGER_SCHEMA,
            "record_type": "header",
            "registration_sha256": registration.file_sha256,
            "previous_row_sha256": "0" * 64,
        }
        header["row_sha256"] = _attempt_row_sha256(header)
        payload = canonical_bytes(header)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            cls._write_all(descriptor, payload)
            os.fsync(descriptor)
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != len(payload):
                raise ContractError("attempt ledger creation did not seal a regular file")
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
        _fsync_directory(path.parent)
        return cls(
            path=path,
            registration_sha256=registration.file_sha256,
            formal_path_bound=formal_path_bound,
            device=metadata.st_dev,
            inode=metadata.st_ino,
            mtime_ns=metadata.st_mtime_ns,
            ctime_ns=metadata.st_ctime_ns,
            expected_payload=payload,
        )

    @classmethod
    def create_registered(
        cls,
        registration: ValidatedRegistration,
    ) -> "AttemptLedger":
        """Formal entrypoint: disabled before the live adapter is implemented."""
        require_formal_collection_ready("registered attempt-ledger creation")
        raise RealCollectorNotImplemented(
            "NO-GO: registered attempt-ledger writer is not wired"
        )

    @classmethod
    def create_preregistered_path_contract_test_only(
        cls,
        registration: ValidatedRegistration,
    ) -> "AttemptLedger":
        """Exercise path binding without creating a formal ledger."""
        return cls._create(
            registration.attempt_ledger_path,
            registration,
            formal_path_bound=True,
        )

    @classmethod
    def create_contract_test_only(
        cls,
        path: Path,
        registration: ValidatedRegistration,
    ) -> "AttemptLedger":
        """Create an unpublishable ledger for zero-environment CPU tests."""
        return cls._create(path, registration, formal_path_bound=False)

    @classmethod
    def open_snapshot_contract_test_only(
        cls,
        path: Path,
        registration_sha256: str,
        *,
        formal_path_bound: bool,
    ) -> "AttemptLedger":
        """Bind a read-only validation handle to one existing inode/prefix."""
        unresolved = _unresolved_absolute_path(path)
        canonical_path = unresolved.parent.resolve() / unresolved.name
        _reject_final_symlink(canonical_path, "attempt ledger")
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(canonical_path, flags)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH)
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise ContractError("attempt ledger snapshot is not a regular file")
            payload = cls._read_descriptor(descriptor)
            after = os.fstat(descriptor)
            if (
                (metadata.st_dev, metadata.st_ino, metadata.st_size,
                 metadata.st_mtime_ns, metadata.st_ctime_ns)
                != (after.st_dev, after.st_ino, after.st_size,
                    after.st_mtime_ns, after.st_ctime_ns)
                or len(payload) != after.st_size
            ):
                raise ContractError("attempt ledger changed during snapshot")
            parse_attempt_ledger_bytes(payload, registration_sha256)
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
        return cls(
            path=canonical_path,
            registration_sha256=registration_sha256,
            formal_path_bound=formal_path_bound,
            device=metadata.st_dev,
            inode=metadata.st_ino,
            mtime_ns=metadata.st_mtime_ns,
            ctime_ns=metadata.st_ctime_ns,
            expected_payload=payload,
        )

    def snapshot(self) -> bytes:
        payload = read_regular_file_snapshot(self.path, "attempt ledger")
        metadata = os.lstat(self.path)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_dev != self.device
            or metadata.st_ino != self.inode
            or metadata.st_mtime_ns != self.mtime_ns
            or metadata.st_ctime_ns != self.ctime_ns
            or payload != self.expected_payload
        ):
            raise ContractError(
                "attempt ledger inode/prefix changed outside append-only writer"
            )
        return payload

    def _append(
        self,
        assignment: EpisodeAssignment,
        event: str,
        reason: str | None,
        episode_payload_sha256: str | None,
    ) -> None:
        flags = os.O_RDWR | os.O_APPEND
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.path, flags)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_dev != self.device
                or metadata.st_ino != self.inode
                or metadata.st_mtime_ns != self.mtime_ns
                or metadata.st_ctime_ns != self.ctime_ns
            ):
                raise ContractError("attempt ledger inode changed before append")
            payload = self._read_descriptor(descriptor)
            if payload != self.expected_payload:
                raise ContractError(
                    "attempt ledger prefix changed outside append-only writer"
                )
            rows = parse_attempt_ledger_bytes(payload, self.registration_sha256)
            episode_rows = [
                row
                for row in rows[1:]
                if row.get("episode_id") == assignment.episode_id
            ]
            if event == "started":
                if episode_rows or reason is not None:
                    raise ContractError(
                        f"episode {assignment.episode_id} already attempted; "
                        "replacement forbidden"
                    )
            elif (
                len(episode_rows) != 1
                or episode_rows[0].get("event") != "started"
                or episode_rows[0].get("env_seed") != assignment.env_seed
                or episode_rows[0].get("collector_id") != assignment.collector_id
            ):
                raise ContractError("attempt terminal event lacks its exact unique start")
            if event == "completed" and (
                reason is not None or not is_sha256(episode_payload_sha256)
            ):
                raise ContractError(
                    "completed attempt requires one payload SHA and no failure reason"
                )
            if event != "completed" and event != "started" and (
                not isinstance(reason, str) or not reason
            ):
                raise ContractError("failed attempt requires a reason")
            if event != "completed" and episode_payload_sha256 is not None:
                raise ContractError("non-completed attempt cannot carry a payload SHA")
            row = {
                "schema": ATTEMPT_EVENT_SCHEMA,
                "record_type": "event",
                "sequence": len(rows) - 1,
                "registration_sha256": self.registration_sha256,
                "episode_id": assignment.episode_id,
                "env_seed": assignment.env_seed,
                "collector_id": assignment.collector_id,
                "event": event,
                "reason": reason,
                "episode_payload_sha256": episode_payload_sha256,
                "previous_row_sha256": rows[-1]["row_sha256"],
            }
            row["row_sha256"] = _attempt_row_sha256(row)
            addition = canonical_bytes(row)
            self._write_all(descriptor, addition)
            os.fsync(descriptor)
            after = os.fstat(descriptor)
            if (
                after.st_dev != self.device
                or after.st_ino != self.inode
                or after.st_size != len(payload) + len(addition)
            ):
                raise ContractError("attempt ledger append identity/size mismatch")
            self.expected_payload = payload + addition
            self.mtime_ns = after.st_mtime_ns
            self.ctime_ns = after.st_ctime_ns
            parse_attempt_ledger_bytes(
                self.expected_payload,
                self.registration_sha256,
            )
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
        self.snapshot()

    def start(self, assignment: EpisodeAssignment) -> None:
        self._append(assignment, "started", None, None)

    def finish(
        self,
        assignment: EpisodeAssignment,
        event: str,
        reason: str | None = None,
        episode_payload_sha256: str | None = None,
    ) -> None:
        if event not in {"completed", "technical_failure", "contract_failure"}:
            raise ContractError("invalid attempt terminal event")
        self._append(assignment, event, reason, episode_payload_sha256)


def validate_attempt_ledger_for_publication(
    attempt_ledger: AttemptLedger,
    registration: ValidatedRegistration,
    *,
    require_registered_path: bool = True,
    expected_episode_payload_sha256: Mapping[int, str] | None = None,
) -> bytes:
    if attempt_ledger.registration_sha256 != registration.file_sha256:
        raise ContractError("attempt ledger registration binding mismatch")
    if require_registered_path and (
        not attempt_ledger.formal_path_bound
        or attempt_ledger.path != registration.attempt_ledger_path
    ):
        raise ContractError("formal publication requires the one preregistered ledger")
    payload = attempt_ledger.snapshot()
    ledger_sha_before = hashlib.sha256(payload).hexdigest()
    rows = parse_attempt_ledger_bytes(payload, registration.file_sha256)
    events = rows[1:]
    if len(events) != 2 * EXPECTED_EPISODES:
        raise ContractError("formal D1 requires exactly two attempt events per episode")
    by_episode: dict[int, list[dict[str, Any]]] = {}
    for row in events:
        episode_id = row["episode_id"]
        if not 0 <= episode_id < EXPECTED_EPISODES:
            raise ContractError("attempt ledger contains an unregistered episode")
        assignment = registration.assignments[episode_id]
        if (
            row["env_seed"] != assignment.env_seed
            or row["collector_id"] != assignment.collector_id
        ):
            raise ContractError("attempt ledger assignment identity mismatch")
        by_episode.setdefault(episode_id, []).append(row)
    if set(by_episode) != set(range(EXPECTED_EPISODES)):
        raise ContractError("attempt ledger does not cover exactly episodes 0..95")
    for episode_id, episode_rows in by_episode.items():
        if [row["event"] for row in episode_rows] != ["started", "completed"]:
            raise ContractError(
                f"episode {episode_id} has failure/retry; formal replacement is forbidden"
            )
    if expected_episode_payload_sha256 is not None:
        completed_payloads = {
            episode_id: rows_for_episode[-1]["episode_payload_sha256"]
            for episode_id, rows_for_episode in by_episode.items()
        }
        if completed_payloads != dict(expected_episode_payload_sha256):
            raise ContractError("attempt ledger/episode payload SHA binding mismatch")
    final_payload = attempt_ledger.snapshot()
    if (
        final_payload != payload
        or hashlib.sha256(final_payload).hexdigest() != ledger_sha_before
    ):
        raise ContractError("attempt ledger drifted while it was being validated")
    if not payload.endswith(b"\n"):
        raise ContractError("attempt ledger is not newline sealed")
    return payload


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


def environment_modules_loaded() -> list[str]:
    """Report environment stacks already imported into this Python process."""
    return sorted(
        name
        for name in sys.modules
        if any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in ENVIRONMENT_MODULE_PREFIXES
        )
    )


def require_formal_collection_ready(operation: str) -> None:
    """Hard-stop every registered/formal entrypoint until all bindings exist."""
    blockers = []
    if not REAL_SEMANTIC_BINDING_IMPLEMENTED:
        blockers.append("live pi0.5/rich8217/ChainEnv semantic adapter")
    if not CORE_OWNED_POLICY_SEPARATION_IMPLEMENTED:
        blockers.append("core-owned base-policy/residual-actor separation")
    if not RAW_OBSERVATION_BINDING_IMPLEMENTED:
        blockers.append("core-owned raw-observation hashing")
    if not FORMAL_DETERMINISM_ATTESTATION_IMPLEMENTED:
        blockers.append("formal deterministic-actor attestation")
    if not ATTEMPT_LEDGER_EXTERNAL_ANCHOR_IMPLEMENTED:
        blockers.append("externally anchored append-only attempt ledger")
    if blockers:
        raise RealCollectorNotImplemented(
            f"NO-GO: {operation} blocked before reset/write: " + "; ".join(blockers)
        )


def _resolve_registered_path(raw: Any) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ContractError("registered artifact path must be a non-empty string")
    path = Path(raw).expanduser()
    candidate = path if path.is_absolute() else REPO / path
    # Resolve the parent for stable identity while deliberately retaining the
    # final component so a registered symlink cannot disappear via resolve().
    return candidate.parent.resolve() / candidate.name


def _artifact_ref(value: Any, label: str) -> ArtifactRef:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be an artifact reference")
    if set(value) != set(ARTIFACT_REF_FIELDS):
        raise ContractError(f"{label} artifact-reference field closure mismatch")
    path = _resolve_registered_path(value.get("path"))
    expected = value.get("sha256")
    if not is_sha256(expected):
        raise ContractError(f"{label}.sha256 is invalid")
    payload = read_regular_file_snapshot(path, label)
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ContractError(f"{label} SHA mismatch: expected {expected}, got {actual}")
    return ArtifactRef(path=path, sha256=actual, payload=payload)


def _require_tensor(
    value: Any,
    shape: tuple[int, ...],
    label: str,
    *,
    finite: bool = True,
) -> torch.Tensor:
    if not torch.is_tensor(value) or tuple(value.shape) != shape:
        got = tuple(value.shape) if torch.is_tensor(value) else type(value).__name__
        raise ContractError(f"{label} must have shape {shape}, got {got}")
    tensor = value.detach().cpu().contiguous()
    if finite and not bool(torch.isfinite(tensor).all()):
        raise ContractError(f"{label} contains NaN/Inf")
    return tensor


def actor_runtime_fingerprint_sha256(checkpoint: Mapping[str, Any]) -> str:
    """Hash the actor state an adapter must actually hold at collection time.

    The artifact SHA covers the serialized checkpoint.  This second digest is
    intentionally serialization-independent: it covers all fixed actor
    metadata, both normalizer tensors, and every named state-dict tensor.
    """
    metadata_keys = (
        "schema",
        "task",
        "input_schema",
        "input_dim",
        "c",
        "adim",
        "collector_id",
        "training_data_role",
        "env_steps",
        "d0_tape_sha256",
        "m0_sha256",
        "phi_sha256",
        "actor_seed",
        "scale",
        "fixed_hyperparameters_sha256",
        "architecture_config",
        "semantic_binding",
    )
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ContractError("actor runtime fingerprint requires a state_dict")
    state_hashes: dict[str, str] = {}
    for name in sorted(state_dict):
        value = state_dict[name]
        if not isinstance(name, str) or not torch.is_tensor(value):
            raise ContractError("actor runtime fingerprint requires tensor-only state")
        state_hashes[name] = tensor_sha256(value)
    mu = checkpoint.get("normalizer_mu")
    sd = checkpoint.get("normalizer_sd")
    if not torch.is_tensor(mu) or not torch.is_tensor(sd):
        raise ContractError("actor runtime fingerprint requires normalizer tensors")
    payload = {
        "metadata": {key: checkpoint.get(key) for key in metadata_keys},
        "normalizer_mu_sha256": tensor_sha256(mu),
        "normalizer_sd_sha256": tensor_sha256(sd),
        "state_dict_sha256": state_hashes,
    }
    return canonical_sha256(payload)


def _validate_tensor_state_dict(
    state_dict: Any,
    expected_shapes: Mapping[str, tuple[int, ...]],
    label: str,
) -> dict[str, torch.Tensor]:
    if not isinstance(state_dict, Mapping) or set(state_dict) != set(expected_shapes):
        actual = set(state_dict) if isinstance(state_dict, Mapping) else set()
        raise ContractError(
            f"{label} state_dict field mismatch: "
            f"missing={sorted(set(expected_shapes) - actual)}, "
            f"unknown={sorted(actual - set(expected_shapes))}"
        )
    checked: dict[str, torch.Tensor] = {}
    for name, shape in expected_shapes.items():
        tensor = _require_tensor(state_dict[name], shape, f"{label}.{name}")
        if tensor.dtype != torch.float32:
            raise ContractError(f"{label}.{name} must be float32")
        checked[name] = tensor
    return checked


def model_runtime_state_sha256(
    metadata: Mapping[str, Any],
    state_dict: Mapping[str, torch.Tensor],
) -> str:
    return canonical_sha256(
        {
            "metadata": dict(metadata),
            "state_dict_sha256": {
                name: tensor_sha256(state_dict[name]) for name in sorted(state_dict)
            },
        }
    )


def validate_tokenizer(reference: ArtifactRef) -> TokenizerInfo:
    manifest = load_json_bytes(reference.payload, "tokenizer manifest")
    fields = {
        "schema",
        "status",
        "model_id",
        "tokenizer_class",
        "vocab_size",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "vocabulary_sha256",
        "semantic_binding",
    }
    if set(manifest) != fields:
        raise ContractError("tokenizer manifest field closure mismatch")
    expected = {
        "schema": TOKENIZER_SCHEMA,
        "status": "frozen",
        "model_id": "pi0.5",
        "tokenizer_class": "PaliGemmaTokenizer",
        "semantic_binding": "mechanical_fixture_not_semantically_bound",
    }
    for key, value in expected.items():
        if not exact_json_equal(manifest.get(key), value):
            raise ContractError(f"tokenizer {key} mismatch")
    vocab_size = manifest.get("vocab_size")
    if type(vocab_size) is not int or vocab_size < 1024:
        raise ContractError("tokenizer vocab_size is invalid")
    special_ids = [
        manifest.get("bos_token_id"),
        manifest.get("eos_token_id"),
        manifest.get("pad_token_id"),
    ]
    if any(type(value) is not int or not 0 <= value < vocab_size for value in special_ids):
        raise ContractError("tokenizer special token IDs are invalid")
    if not is_sha256(manifest.get("vocabulary_sha256")):
        raise ContractError("tokenizer vocabulary SHA is invalid")
    return TokenizerInfo(
        path=reference.path,
        sha256=reference.sha256,
        model_id=manifest["model_id"],
        vocab_size=vocab_size,
    )


def validate_base_vla_checkpoint(
    reference: ArtifactRef,
    tokenizer_sha256: str,
) -> ModelArtifactInfo:
    checkpoint = load_torch_bytes(reference.payload, "base VLA checkpoint")
    fields = {
        "schema",
        "status",
        "model_id",
        "task_family",
        "c",
        "adim",
        "tokenizer_sha256",
        "model_config",
        "state_dict",
        "runtime_state_sha256",
        "semantic_binding",
    }
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != fields:
        raise ContractError("base VLA checkpoint field closure mismatch")
    expected_scalars = {
        "schema": BASE_VLA_CHECKPOINT_SCHEMA,
        "status": "frozen",
        "model_id": "pi0.5",
        "task_family": "libero",
        "c": C,
        "adim": ACTION_DIM,
        "tokenizer_sha256": tokenizer_sha256,
        "semantic_binding": "mechanical_fixture_not_semantically_bound",
    }
    for key, expected in expected_scalars.items():
        if not exact_json_equal(checkpoint.get(key), expected):
            raise ContractError(f"base VLA {key} mismatch")
    config = checkpoint.get("model_config")
    config_fields = {"schema", "hidden_dim", "action_chunk_shape", "dtype"}
    if not isinstance(config, Mapping) or set(config) != config_fields:
        raise ContractError("base VLA model_config field closure mismatch")
    hidden = config.get("hidden_dim")
    if (
        config.get("schema") != BASE_VLA_CONFIG_SCHEMA
        or type(hidden) is not int
        or hidden < 2
        or config.get("action_chunk_shape") != [C, ACTION_DIM]
        or config.get("dtype") != "float32"
    ):
        raise ContractError("base VLA model_config mismatch")
    state = _validate_tensor_state_dict(
        checkpoint.get("state_dict"),
        {
            "vision_projection.weight": (hidden, hidden),
            "vision_projection.bias": (hidden,),
            "action_head.weight": (C * ACTION_DIM, hidden),
            "action_head.bias": (C * ACTION_DIM,),
        },
        "base VLA",
    )
    metadata = {key: checkpoint[key] for key in expected_scalars}
    metadata["model_config"] = dict(config)
    runtime_state = model_runtime_state_sha256(metadata, state)
    if checkpoint.get("runtime_state_sha256") != runtime_state:
        raise ContractError("base VLA runtime-state fingerprint mismatch")
    return ModelArtifactInfo(
        path=reference.path,
        sha256=reference.sha256,
        schema=BASE_VLA_CHECKPOINT_SCHEMA,
        runtime_state_sha256=runtime_state,
    )


def validate_m0_checkpoint(
    reference: ArtifactRef,
    d0_sha256: str,
) -> ModelArtifactInfo:
    checkpoint = load_torch_bytes(reference.payload, "M0 checkpoint")
    fields = {
        "schema",
        "status",
        "task",
        "input_schema",
        "input_dim",
        "c",
        "adim",
        "training_data_role",
        "d0_tape_sha256",
        "model_config",
        "state_dict",
        "runtime_state_sha256",
        "semantic_binding",
    }
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != fields:
        raise ContractError("M0 checkpoint field closure mismatch")
    expected_scalars = {
        "schema": M0_CHECKPOINT_SCHEMA,
        "status": "frozen",
        "task": TASK,
        "input_schema": "rich8217_v1",
        "input_dim": RICH_DIM,
        "c": C,
        "adim": ACTION_DIM,
        "training_data_role": "D0_only",
        "d0_tape_sha256": d0_sha256,
        "semantic_binding": "mechanical_fixture_not_semantically_bound",
    }
    for key, expected in expected_scalars.items():
        if not exact_json_equal(checkpoint.get(key), expected):
            raise ContractError(f"M0 {key} mismatch")
    config = checkpoint.get("model_config")
    config_fields = {"schema", "hidden_dim", "transition_kind", "dtype"}
    if not isinstance(config, Mapping) or set(config) != config_fields:
        raise ContractError("M0 model_config field closure mismatch")
    hidden = config.get("hidden_dim")
    if (
        config.get("schema") != M0_CONFIG_SCHEMA
        or type(hidden) is not int
        or hidden < 2
        or config.get("transition_kind") != "residual_latent_mlp"
        or config.get("dtype") != "float32"
    ):
        raise ContractError("M0 model_config mismatch")
    state = _validate_tensor_state_dict(
        checkpoint.get("state_dict"),
        {
            "state_proj.weight": (hidden, RICH_DIM),
            "state_proj.bias": (hidden,),
            "action_proj.weight": (hidden, C * ACTION_DIM),
            "action_proj.bias": (hidden,),
            "next_delta_head.weight": (RICH_DIM, hidden),
            "next_delta_head.bias": (RICH_DIM,),
        },
        "M0",
    )
    metadata = {key: checkpoint[key] for key in expected_scalars}
    metadata["model_config"] = dict(config)
    runtime_state = model_runtime_state_sha256(metadata, state)
    if checkpoint.get("runtime_state_sha256") != runtime_state:
        raise ContractError("M0 runtime-state fingerprint mismatch")
    return ModelArtifactInfo(
        path=reference.path,
        sha256=reference.sha256,
        schema=M0_CHECKPOINT_SCHEMA,
        runtime_state_sha256=runtime_state,
    )


def validate_phi_checkpoint(
    reference: ArtifactRef,
    d0_sha256: str,
) -> ModelArtifactInfo:
    checkpoint = load_torch_bytes(reference.payload, "Phi checkpoint")
    fields = {
        "schema",
        "status",
        "task",
        "input_schema",
        "d0_tape_sha256",
        "gate0_status",
        "gate0_result_sha256",
        "model_config",
        "state_dict",
        "runtime_state_sha256",
        "semantic_binding",
    }
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != fields:
        raise ContractError("Phi checkpoint field closure mismatch")
    expected_scalars = {
        "schema": PHI_CHECKPOINT_SCHEMA,
        "status": "frozen",
        "task": TASK,
        "input_schema": "rich8217_v1",
        "d0_tape_sha256": d0_sha256,
        "gate0_status": "PASS",
        "semantic_binding": "mechanical_fixture_not_semantically_bound",
    }
    for key, expected in expected_scalars.items():
        if not exact_json_equal(checkpoint.get(key), expected):
            raise ContractError(f"Phi {key} mismatch")
    if not is_sha256(checkpoint.get("gate0_result_sha256")):
        raise ContractError("Phi Gate0 result SHA is invalid")
    config = checkpoint.get("model_config")
    config_fields = {"schema", "input_dim", "hidden_dim", "dropout"}
    if not isinstance(config, Mapping) or set(config) != config_fields:
        raise ContractError("Phi model_config field closure mismatch")
    hidden = config.get("hidden_dim")
    if (
        config.get("schema") != PHI_CONFIG_SCHEMA
        or config.get("input_dim") != RICH_DIM
        or type(hidden) is not int
        or hidden < 2
        or type(config.get("dropout")) is not float
        or config.get("dropout") != 0.0
    ):
        raise ContractError("Phi model_config mismatch")
    state = _validate_tensor_state_dict(
        checkpoint.get("state_dict"),
        {
            "network.0.weight": (RICH_DIM,),
            "network.0.bias": (RICH_DIM,),
            "network.1.weight": (hidden, RICH_DIM),
            "network.1.bias": (hidden,),
            "network.4.weight": (3, hidden),
            "network.4.bias": (3,),
        },
        "Phi",
    )
    metadata = {key: checkpoint[key] for key in expected_scalars}
    metadata["gate0_result_sha256"] = checkpoint["gate0_result_sha256"]
    metadata["model_config"] = dict(config)
    runtime_state = model_runtime_state_sha256(metadata, state)
    if checkpoint.get("runtime_state_sha256") != runtime_state:
        raise ContractError("Phi runtime-state fingerprint mismatch")
    return ModelArtifactInfo(
        path=reference.path,
        sha256=reference.sha256,
        schema=PHI_CHECKPOINT_SCHEMA,
        runtime_state_sha256=runtime_state,
    )


def runtime_component_fingerprint_sha256(
    artifact_sha256: str,
    runtime_state_sha256: str,
    runtime_config: Mapping[str, Any],
) -> str:
    return canonical_sha256(
        {
            "artifact_sha256": artifact_sha256,
            "runtime_state_sha256": runtime_state_sha256,
            "runtime_config": dict(runtime_config),
        }
    )


def validate_runtime_components(
    value: Any,
    base_vla: ModelArtifactInfo,
    source_hashes: Mapping[str, str],
) -> dict[str, RuntimeComponentInfo]:
    if not isinstance(value, Mapping) or set(value) != set(REGISTERED_RUNTIME_COMPONENTS):
        raise ContractError("runtime_components field closure mismatch")
    bddl_sha = source_hashes["bddl/chains/chain3_lr2.bddl"]
    expected: dict[str, tuple[str, str, dict[str, Any]]] = {
        "base_vla": (
            base_vla.sha256,
            base_vla.runtime_state_sha256,
            {
                "schema": "v246_base_vla_runtime_config_v1",
                "implementation": "pi0.5",
                "model_id": "pi0.5",
                "task": TASK,
                "c": C,
                "adim": ACTION_DIM,
            },
        ),
        "action_decoder": (
            base_vla.sha256,
            base_vla.runtime_state_sha256,
            {
                "schema": "v246_action_decoder_runtime_config_v1",
                "implementation": "pi0.5_action_decoder",
                "input_dtype": "float32",
                "output_dtype": "float32",
                "chunk_shape": [C, ACTION_DIM],
                "clipping": "registered_policy_decoder",
            },
        ),
        "feature_encoder": (
            base_vla.sha256,
            base_vla.runtime_state_sha256,
            {
                "schema": "v246_rich_encoder_runtime_config_v1",
                "implementation": "pi0.5_four_block_rich8217",
                "source": "two_camera_token_blocks_plus_proprio",
                "output_schema": "rich8217_v1",
                "output_dim": RICH_DIM,
                "camera_block_count": 4,
                "camera_block_dim": CAMERA_BLOCK_DIM,
                "proprio_dim": PROPRIO_DIM,
                "proprio_keys": list(PROPRIO_KEYS),
            },
        ),
        "environment": (
            bddl_sha,
            bddl_sha,
            {
                "schema": "v246_chain3_environment_runtime_config_v1",
                "implementation": "ChainEnv",
                "task": TASK,
                "bddl_path": "bddl/chains/chain3_lr2.bddl",
                "horizon_steps": HORIZON_STEPS,
                "technical_done_source": "info.done",
            },
        ),
    }
    result: dict[str, RuntimeComponentInfo] = {}
    fields = {
        "schema",
        "name",
        "artifact_sha256",
        "runtime_state_sha256",
        "runtime_config",
        "runtime_fingerprint_sha256",
    }
    for name in REGISTERED_RUNTIME_COMPONENTS:
        row = value[name]
        if not isinstance(row, Mapping) or set(row) != fields:
            raise ContractError(f"runtime component {name} field closure mismatch")
        artifact_sha, runtime_state_sha, config = expected[name]
        expected_scalars = {
            "schema": RUNTIME_BINDING_SCHEMA,
            "name": name,
            "artifact_sha256": artifact_sha,
            "runtime_state_sha256": runtime_state_sha,
            "runtime_config": config,
        }
        for key, expected_value in expected_scalars.items():
            if not exact_json_equal(row.get(key), expected_value):
                raise ContractError(f"runtime component {name} {key} mismatch")
        fingerprint = runtime_component_fingerprint_sha256(
            artifact_sha,
            runtime_state_sha,
            config,
        )
        if row.get("runtime_fingerprint_sha256") != fingerprint:
            raise ContractError(f"runtime component {name} fingerprint mismatch")
        result[name] = RuntimeComponentInfo(
            name=name,
            artifact_sha256=artifact_sha,
            runtime_state_sha256=runtime_state_sha,
            runtime_fingerprint_sha256=fingerprint,
            runtime_config=config,
        )
    return result


def validate_d0_tape(reference: ArtifactRef) -> D0Info:
    """Validate only the D0 dynamics schema; never inspect an outcome value."""
    tape = load_torch_bytes(reference.payload, "D0 tape")
    if not isinstance(tape, dict):
        raise ContractError("D0 tape must contain a dictionary")
    legacy_outcome_fields = tape.get("legacy_outcome_fields")
    if legacy_outcome_fields not in ([], ["success"]):
        raise ContractError("D0 legacy_outcome_fields must be [] or ['success']")
    keys = frozenset(tape)
    missing = sorted(D0_REQUIRED_FIELDS - keys)
    declared_legacy = frozenset(legacy_outcome_fields or [])
    if not declared_legacy.issubset(D0_ALLOWED_OUTCOME_FIELDS):
        raise ContractError("D0 declares an unsupported legacy outcome field")
    unknown = sorted(keys - D0_REQUIRED_FIELDS - declared_legacy)
    undeclared = sorted(declared_legacy - keys)
    if missing or unknown or undeclared:
        raise ContractError(
            "D0 tape field mismatch: "
            f"missing={missing}, unknown={unknown}, undeclared={undeclared}"
        )
    if tape["schema"] != D0_TAPE_SCHEMA:
        raise ContractError("D0 tape schema mismatch")
    if tuple(tape["training_feature_fields"]) != TRAINING_FEATURE_FIELDS:
        raise ContractError("D0 training_feature_fields mismatch")
    if tuple(tape["split_bookkeeping_fields"]) != SPLIT_BOOKKEEPING_FIELDS:
        raise ContractError("D0 split_bookkeeping_fields mismatch")
    if (
        tape["task"] != TASK
        or type(tape["c"]) is not int
        or tape["c"] != C
    ):
        raise ContractError("D0 task/c mismatch")
    if (
        type(tape["latent_dim"]) is not int
        or tape["latent_dim"] != RICH_DIM
        or type(tape["proprio_dim"]) is not int
        or tape["proprio_dim"] != PROPRIO_DIM
    ):
        raise ContractError("D0 rich/proprio dimension mismatch")
    if tuple(tape["proprio_keys"]) != PROPRIO_KEYS:
        raise ContractError("D0 proprio key ordering mismatch")
    z = tape["z"]
    z_next = tape["z_next"]
    u = tape["u"]
    if (
        not torch.is_tensor(z)
        or z.dtype != torch.float32
        or z.ndim != 2
        or tuple(z.shape[1:]) != (RICH_DIM,)
    ):
        raise ContractError("D0 z must be non-empty [N,8217]")
    rows = len(z)
    if rows == 0 or not torch.is_tensor(z_next) or z_next.shape != z.shape:
        raise ContractError("D0 z_next must match non-empty z")
    if not torch.is_tensor(z_next) or z_next.dtype != torch.float32:
        raise ContractError("D0 z_next must be float32")
    if (
        not torch.is_tensor(u)
        or u.dtype != torch.float32
        or tuple(u.shape) != (rows, C, ACTION_DIM)
    ):
        raise ContractError("D0 action shape mismatch")
    for name in ("z", "z_next", "u", "sigma"):
        value = tape[name]
        if not torch.is_tensor(value) or not bool(torch.isfinite(value).all()):
            raise ContractError(f"D0 {name} is not a finite tensor")
    episode = tape["episode"]
    times = tape["t"]
    if (
        not torch.is_tensor(episode)
        or episode.dtype != torch.int64
        or episode.ndim != 1
        or len(episode) != rows
    ):
        raise ContractError("D0 episode must be [N]")
    if (
        not torch.is_tensor(times)
        or times.dtype != torch.int64
        or times.ndim != 1
        or len(times) != rows
    ):
        raise ContractError("D0 t must be [N]")
    if (
        not torch.is_tensor(tape["sigma"])
        or tape["sigma"].dtype != torch.float32
        or tape["sigma"].shape != episode.shape
    ):
        raise ContractError("D0 sigma must be [N]")
    episode_long = episode.detach().cpu().long()
    times_long = times.detach().cpu().long()
    if not torch.equal(episode.detach().cpu(), episode_long.to(dtype=episode.dtype)):
        raise ContractError("D0 episode IDs are not integral")
    if not torch.equal(times.detach().cpu(), times_long.to(dtype=times.dtype)):
        raise ContractError("D0 times are not integral")
    unique = torch.unique_consecutive(episode_long).tolist()
    if unique != list(range(len(unique))):
        raise ContractError("D0 episode rows must be grouped and contiguous from zero")
    for episode_id in unique:
        indices = torch.nonzero(episode_long == episode_id, as_tuple=False).flatten()
        expected_t = torch.arange(len(indices), dtype=torch.long) * C
        if not torch.equal(times_long[indices], expected_t):
            raise ContractError(f"D0 episode {episode_id} has non-contiguous times")
        if len(indices) > 1 and not torch.equal(z_next[indices[:-1]], z[indices[1:]]):
            raise ContractError(f"D0 episode {episode_id} violates N+1 continuity")
    return D0Info(
        path=reference.path,
        sha256=reference.sha256,
        rows=rows,
        episodes=len(unique),
    )


def validate_actor(
    reference: ArtifactRef,
    collector_id: str,
    d0_sha256: str,
    m0_sha256: str,
    phi_sha256: str,
) -> ActorInfo:
    checkpoint = load_torch_bytes(
        reference.payload,
        f"collector {collector_id} actor checkpoint",
    )
    if not isinstance(checkpoint, dict):
        raise ContractError(f"collector {collector_id} actor must be a dictionary")
    if set(checkpoint) != set(ACTOR_FIELDS):
        missing = sorted(ACTOR_FIELDS - set(checkpoint))
        unknown = sorted(set(checkpoint) - ACTOR_FIELDS)
        raise ContractError(
            f"collector {collector_id} actor field mismatch: "
            f"missing={missing}, unknown={unknown}"
        )
    expected_scalars = {
        "schema": ACTOR_SCHEMA,
        "task": TASK,
        "input_schema": "rich8217_v1",
        "input_dim": RICH_DIM,
        "c": C,
        "adim": ACTION_DIM,
        "collector_id": collector_id,
        "training_data_role": "D0_only",
        "env_steps": 0,
        "d0_tape_sha256": d0_sha256,
        "m0_sha256": m0_sha256,
        "phi_sha256": phi_sha256,
        "semantic_binding": "mechanical_fixture_not_semantically_bound",
    }
    for key, expected in expected_scalars.items():
        if not exact_json_equal(checkpoint.get(key), expected):
            raise ContractError(
                f"collector {collector_id} actor {key} mismatch: "
                f"expected {expected!r}, got {checkpoint.get(key)!r}"
            )
    actor_seed = checkpoint.get("actor_seed")
    scale = checkpoint.get("scale")
    hyper_sha = checkpoint.get("fixed_hyperparameters_sha256")
    if type(actor_seed) is not int or actor_seed < 0:
        raise ContractError(f"collector {collector_id} actor_seed is invalid")
    if (
        type(scale) is not float
        or not exact_json_equal(scale, REGISTERED_RESIDUAL_SCALE)
    ):
        raise ContractError(
            f"collector {collector_id} residual scale must equal fixed "
            f"{REGISTERED_RESIDUAL_SCALE}"
        )
    if not is_sha256(hyper_sha):
        raise ContractError(f"collector {collector_id} hyperparameter SHA is invalid")
    mu = _require_tensor(checkpoint.get("normalizer_mu"), (RICH_DIM,), "actor mu")
    sd = _require_tensor(checkpoint.get("normalizer_sd"), (RICH_DIM,), "actor sd")
    if mu.dtype != torch.float32 or sd.dtype != torch.float32:
        raise ContractError(f"collector {collector_id} normalizer must be float32")
    if not bool((sd > 0).all()):
        raise ContractError(f"collector {collector_id} normalizer has non-positive sd")
    if checkpoint.get("normalizer_mu_sha256") != tensor_sha256(mu):
        raise ContractError(f"collector {collector_id} normalizer mu SHA mismatch")
    if checkpoint.get("normalizer_sd_sha256") != tensor_sha256(sd):
        raise ContractError(f"collector {collector_id} normalizer sd SHA mismatch")
    config = checkpoint.get("architecture_config")
    config_fields = {
        "schema",
        "hidden_dim",
        "activation",
        "output_shape",
        "dtype",
        "residual_scale",
    }
    if not isinstance(config, Mapping) or set(config) != config_fields:
        raise ContractError(f"collector {collector_id} actor config field mismatch")
    hidden = config.get("hidden_dim")
    if (
        config.get("schema") != ACTOR_CONFIG_SCHEMA
        or type(hidden) is not int
        or hidden < 2
        or config.get("activation") != "tanh"
        or config.get("output_shape") != [C, ACTION_DIM]
        or config.get("dtype") != "float32"
        or config.get("residual_scale") != float(scale)
    ):
        raise ContractError(f"collector {collector_id} actor config mismatch")
    if hyper_sha != canonical_sha256(dict(config)):
        raise ContractError(f"collector {collector_id} hyperparameter SHA mismatch")
    _validate_tensor_state_dict(
        checkpoint.get("state_dict"),
        {
            "network.0.weight": (hidden, RICH_DIM),
            "network.0.bias": (hidden,),
            "network.2.weight": (C * ACTION_DIM, hidden),
            "network.2.bias": (C * ACTION_DIM,),
        },
        f"collector {collector_id}",
    )
    runtime_fingerprint = actor_runtime_fingerprint_sha256(checkpoint)
    declared_runtime_fingerprint = checkpoint.get("runtime_fingerprint_sha256")
    if declared_runtime_fingerprint != runtime_fingerprint:
        raise ContractError(
            f"collector {collector_id} runtime fingerprint mismatch: "
            f"expected {runtime_fingerprint}, got {declared_runtime_fingerprint!r}"
        )
    return ActorInfo(
        collector_id=collector_id,
        path=reference.path,
        sha256=reference.sha256,
        actor_seed=actor_seed,
        scale=float(scale),
        fixed_hyperparameters_sha256=hyper_sha,
        runtime_fingerprint_sha256=runtime_fingerprint,
    )


def _validate_assignments(value: Any) -> tuple[EpisodeAssignment, ...]:
    if not isinstance(value, list) or len(value) != EXPECTED_EPISODES:
        raise ContractError(f"registration must contain {EXPECTED_EPISODES} assignments")
    assignments: list[EpisodeAssignment] = []
    for row in value:
        if not isinstance(row, Mapping):
            raise ContractError("episode assignment must be an object")
        if set(row) != set(ASSIGNMENT_FIELDS):
            raise ContractError("episode assignment field closure mismatch")
        assignment = EpisodeAssignment(
            episode_id=row.get("episode_id"),
            env_seed=row.get("env_seed"),
            collector_id=row.get("collector_id"),
            split=row.get("split"),
        )
        if type(assignment.episode_id) is not int or assignment.episode_id < 0:
            raise ContractError("episode_id must be a non-negative integer")
        if type(assignment.env_seed) is not int or assignment.env_seed < 0:
            raise ContractError("env_seed must be a non-negative integer")
        if assignment.collector_id not in COLLECTOR_IDS or assignment.split not in SPLITS:
            raise ContractError("assignment collector/split is invalid")
        assignments.append(assignment)
    assignments.sort(key=lambda item: item.episode_id)
    if [item.episode_id for item in assignments] != list(range(EXPECTED_EPISODES)):
        raise ContractError("episode IDs must be exactly 0..95")
    seeds = [item.env_seed for item in assignments]
    if len(seeds) != len(set(seeds)):
        raise ContractError("D1 environment seeds must be unique across policies/splits")
    for collector_id in COLLECTOR_IDS:
        selected = [item for item in assignments if item.collector_id == collector_id]
        fit = [item for item in selected if item.split == "fit"]
        calibration = [item for item in selected if item.split == "calibration"]
        if len(selected) != EXPECTED_PER_COLLECTOR:
            raise ContractError(f"collector {collector_id} must own exactly 48 episodes")
        if len(fit) != EXPECTED_FIT_PER_COLLECTOR or len(calibration) != EXPECTED_CAL_PER_COLLECTOR:
            raise ContractError(f"collector {collector_id} must split 36 fit / 12 calibration")
    return tuple(assignments)


def _validate_action_change_contract(
    collection: Mapping[str, Any],
) -> ActionChangeContract:
    fraction = collection.get("min_changed_action_fraction_per_episode")
    minimum_l2 = collection.get("min_decoded_action_delta_l2_per_episode")
    if (
        type(fraction) is not float
        or not np.isfinite(float(fraction))
        or not MIN_REGISTERED_CHANGED_ACTION_FRACTION <= float(fraction) <= 1.0
    ):
        raise ContractError(
            "registration.collection.min_changed_action_fraction_per_episode "
            f"must be finite in [{MIN_REGISTERED_CHANGED_ACTION_FRACTION},1]"
        )
    if (
        type(minimum_l2) is not float
        or not np.isfinite(float(minimum_l2))
        or not float(minimum_l2) >= MIN_REGISTERED_DECODED_ACTION_DELTA_L2
    ):
        raise ContractError(
            "registration.collection.min_decoded_action_delta_l2_per_episode "
            f"must be at least {MIN_REGISTERED_DECODED_ACTION_DELTA_L2}"
        )
    fixed_strength_fields = {
        "meaningful_decoded_step_l2": MEANINGFUL_DECODED_STEP_L2,
        "min_meaningful_action_fraction_per_episode": (
            MIN_MEANINGFUL_ACTION_FRACTION
        ),
        "min_decoded_step_l2_p10_per_episode": MIN_DECODED_STEP_L2_P10,
        "max_decoded_step_l2_per_episode": MAX_DECODED_STEP_L2,
    }
    for name, expected in fixed_strength_fields.items():
        if not exact_json_equal(collection.get(name), expected):
            raise ContractError(
                f"registration.collection.{name} must equal fixed {expected}"
            )
    return ActionChangeContract(
        min_changed_action_fraction_per_episode=float(fraction),
        min_decoded_action_delta_l2_per_episode=float(minimum_l2),
        meaningful_decoded_step_l2=MEANINGFUL_DECODED_STEP_L2,
        min_meaningful_action_fraction_per_episode=(
            MIN_MEANINGFUL_ACTION_FRACTION
        ),
        min_decoded_step_l2_p10_per_episode=MIN_DECODED_STEP_L2_P10,
        max_decoded_step_l2_per_episode=MAX_DECODED_STEP_L2,
    )


def _validate_seed_ledger(
    reference: ArtifactRef,
    assignments: Sequence[EpisodeAssignment],
    actors: Mapping[str, ActorInfo],
) -> None:
    ledger = load_json_bytes(reference.payload, "seed ledger")
    if ledger.get("schema") != SEED_LEDGER_SCHEMA or ledger.get("status") != "frozen":
        raise ContractError("seed ledger schema/status mismatch")
    role_names = (
        "prior_chain3_used",
        "gate3_evaluation_reserved",
        "gate3_actor_training_reserved",
        "other_forbidden_before_d1",
    )
    expected_ledger_fields = {"schema", "status", *role_names}
    if set(ledger) != expected_ledger_fields:
        raise ContractError("seed ledger field closure mismatch")
    role_sets: dict[str, set[int]] = {}
    for name in role_names:
        values = ledger.get(name)
        if (
            not isinstance(values, list)
            or any(type(seed) is not int or seed < 0 for seed in values)
            or len(values) != len(set(values))
        ):
            raise ContractError(f"seed ledger {name} must be a unique integer list")
        role_sets[name] = set(values)
    for required_nonempty in (
        "gate3_evaluation_reserved",
        "gate3_actor_training_reserved",
    ):
        if not role_sets[required_nonempty]:
            raise ContractError(f"seed ledger {required_nonempty} must not be empty")
    if (
        len(role_sets["gate3_actor_training_reserved"])
        != EXPECTED_GATE3_ACTOR_TRAINING_SEEDS
    ):
        raise ContractError(
            "seed ledger must reserve exactly 16 future Gate-3 actor-init seeds"
        )
    for index, left_name in enumerate(role_names):
        for right_name in role_names[index + 1 :]:
            overlap = sorted(role_sets[left_name] & role_sets[right_name])
            if overlap:
                raise ContractError(
                    f"seed roles {left_name}/{right_name} overlap: {overlap[:8]}"
                )
    d1 = {assignment.env_seed for assignment in assignments}
    for name, values in role_sets.items():
        overlap = sorted(d1 & values)
        if overlap:
            raise ContractError(f"D1 seeds overlap {name}: {overlap[:8]}")
    actor_seeds = {
        collector_id: actors[collector_id].actor_seed for collector_id in COLLECTOR_IDS
    }
    if actor_seeds["A"] == actor_seeds["B"]:
        raise ContractError("collector actor seeds must differ")
    for collector_id, actor_seed in actor_seeds.items():
        if actor_seed in d1:
            raise ContractError(f"collector {collector_id} actor_seed overlaps D1 env seeds")
        for role_name, role_values in role_sets.items():
            if actor_seed in role_values:
                raise ContractError(
                    f"collector {collector_id} actor_seed overlaps {role_name}"
                )
    known_required = set(range(8000, 8196)) | set(range(8700, 8796)) | set(range(8900, 8996))
    if not known_required.issubset(role_sets["prior_chain3_used"]):
        raise ContractError("seed ledger omits known chain3 panels 8000/8100/8700/8900")


def validate_registration(path: Path) -> ValidatedRegistration:
    """Validate all prospective inputs without importing an environment stack."""
    unresolved = _unresolved_absolute_path(path)
    path = unresolved.parent.resolve() / unresolved.name
    before_env_modules = set(environment_modules_loaded())
    raw_bytes = read_regular_file_snapshot(path, "registration")
    file_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    payload = load_json_bytes(raw_bytes, "registration")
    if set(payload) != set(REGISTRATION_FIELDS):
        raise ContractError("registration top-level field closure mismatch")
    if payload.get("schema") != REGISTRATION_SCHEMA:
        raise ContractError("registration schema mismatch")
    if payload.get("status") != "frozen_before_collection":
        raise ContractError("registration was not frozen before collection")
    if payload.get("evidence_class") != MECHANICAL_EVIDENCE_CLASS:
        raise ContractError("registration evidence_class is not mechanical-only")
    self_sha = payload.get("self_sha256")
    body = {key: value for key, value in payload.items() if key != "self_sha256"}
    if not is_sha256(self_sha) or canonical_sha256(body) != self_sha:
        raise ContractError("registration self_sha256 mismatch")

    collection = payload.get("collection")
    if not isinstance(collection, Mapping):
        raise ContractError("registration.collection must be an object")
    if set(collection) != set(COLLECTION_FIELDS):
        raise ContractError("registration.collection field closure mismatch")
    expected_collection = {
        "task": TASK,
        "horizon_steps": HORIZON_STEPS,
        "c": C,
        "action_shape": [C, ACTION_DIM],
        "rich_latent_dim": RICH_DIM,
        "proprio_dim": PROPRIO_DIM,
        "proprio_keys": list(PROPRIO_KEYS),
        "camera_token_blocks": [list(block) for block in CAMERA_TOKEN_BLOCKS],
        "expected_episodes": EXPECTED_EPISODES,
        "expected_per_collector": EXPECTED_PER_COLLECTOR,
        "fit_per_collector": EXPECTED_FIT_PER_COLLECTOR,
        "calibration_per_collector": EXPECTED_CAL_PER_COLLECTOR,
        "perturbation_scope": "episode_policy",
        "chunk_noise": "forbidden",
        "sigma_value": 0.0,
        "success_termination": "ignored",
        "technical_done_source": "info.done",
        "outcome_storage": "sealed_evaluation_sidecar_only",
        "decoded_action_trace": "base_and_executed_float32_exact",
        "changed_action_step_definition": "any_exact_component_difference",
        "decoded_action_delta_norm": "episode_frobenius_l2_float64",
        "attempt_ledger_policy": (
            "single_preregistered_append_only_no_retry_or_seed_replacement"
        ),
    }
    for key, expected in expected_collection.items():
        if not exact_json_equal(collection.get(key), expected):
            raise ContractError(f"registration.collection.{key} mismatch")
    action_change = _validate_action_change_contract(collection)
    assignments = _validate_assignments(collection.get("episode_assignments"))
    attempt_ledger_path = _resolve_registered_path(
        collection.get("attempt_ledger_path")
    )
    if attempt_ledger_path.suffix != ".jsonl":
        raise ContractError("registration attempt_ledger_path must end in .jsonl")

    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ContractError("registration.inputs must be an object")
    if set(inputs) != set(INPUT_FIELDS):
        raise ContractError("registration.inputs field closure mismatch")
    d0_ref = _artifact_ref(inputs.get("d0_tape"), "inputs.d0_tape")
    m0_ref = _artifact_ref(inputs.get("m0_checkpoint"), "inputs.m0_checkpoint")
    phi_ref = _artifact_ref(inputs.get("phi_checkpoint"), "inputs.phi_checkpoint")
    ledger_ref = _artifact_ref(inputs.get("seed_ledger"), "inputs.seed_ledger")
    tokenizer_ref = _artifact_ref(
        inputs.get("tokenizer_manifest"),
        "inputs.tokenizer_manifest",
    )
    base_vla_ref = _artifact_ref(
        inputs.get("base_vla_checkpoint"),
        "inputs.base_vla_checkpoint",
    )

    d0 = validate_d0_tape(d0_ref)
    tokenizer = validate_tokenizer(tokenizer_ref)
    base_vla = validate_base_vla_checkpoint(base_vla_ref, tokenizer.sha256)
    m0 = validate_m0_checkpoint(m0_ref, d0.sha256)
    phi = validate_phi_checkpoint(phi_ref, d0.sha256)
    actor_values = inputs.get("actors")
    if not isinstance(actor_values, Mapping) or set(actor_values) != set(COLLECTOR_IDS):
        raise ContractError("registration must bind exactly actors A and B")
    actors: dict[str, ActorInfo] = {}
    for collector_id in COLLECTOR_IDS:
        actor_ref = _artifact_ref(
            actor_values[collector_id], f"inputs.actors.{collector_id}"
        )
        actors[collector_id] = validate_actor(
            actor_ref,
            collector_id,
            d0.sha256,
            m0_ref.sha256,
            phi_ref.sha256,
        )
    _validate_seed_ledger(ledger_ref, assignments, actors)
    if actors["A"].scale != actors["B"].scale:
        raise ContractError("collector residual scales must be fixed and equal")
    if actors["A"].fixed_hyperparameters_sha256 != actors["B"].fixed_hyperparameters_sha256:
        raise ContractError("collector actor hyperparameters must match exactly")

    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ContractError("registration.provenance must be an object")
    if set(provenance) != set(PROVENANCE_FIELDS):
        raise ContractError("registration.provenance field closure mismatch")
    source_hashes = provenance.get("source_files_sha256")
    if not isinstance(source_hashes, Mapping) or set(source_hashes) != set(REQUIRED_SOURCE_PATHS):
        raise ContractError("registration source-file closure mismatch")
    for relative, expected in source_hashes.items():
        if not is_sha256(expected):
            raise ContractError(f"invalid source hash for {relative}")
        source = (REPO / relative).resolve()
        if not source.is_file() or sha256_file(source) != expected:
            raise ContractError(f"registered source drift: {relative}")
    runtime_components = validate_runtime_components(
        payload.get("runtime_components"),
        base_vla,
        source_hashes,
    )

    after_env_modules = set(environment_modules_loaded())
    introduced = sorted(after_env_modules - before_env_modules)
    if introduced:
        raise ContractError(f"validation imported environment modules: {introduced}")
    if read_regular_file_snapshot(path, "registration") != raw_bytes:
        raise ContractError("registration drifted while its inputs were validated")
    return ValidatedRegistration(
        path=path,
        file_sha256=file_sha256,
        self_sha256=self_sha,
        raw_bytes=raw_bytes,
        payload=payload,
        assignments=assignments,
        actors=actors,
        d0=d0,
        action_change=action_change,
        m0=m0,
        phi=phi,
        base_vla=base_vla,
        tokenizer=tokenizer,
        runtime_components=runtime_components,
        attempt_ledger_path=attempt_ledger_path,
    )


def _rich_boundary(value: Any, label: str) -> torch.Tensor:
    tensor = _require_tensor(value, (RICH_DIM,), label)
    if tensor.dtype != torch.float32:
        raise ContractError(f"{label} must be float32, got {tensor.dtype}")
    return tensor.clone()


def _action_chunk(value: Any, label: str) -> torch.Tensor:
    tensor = _require_tensor(value, (C, ACTION_DIM), label)
    if tensor.dtype != torch.float32:
        raise ContractError(f"{label} must be float32, got {tensor.dtype}")
    return tensor.clone()


def _controller_identity(
    controller: EpisodeController,
    registration: ValidatedRegistration,
    actor: ActorInfo,
    phase: str,
) -> tuple[tuple[str, str], ...]:
    try:
        fingerprints = controller.recompute_runtime_fingerprints_sha256()
        artifacts = {
            "actor": str(controller.actor_artifact_sha256),
            "base_vla": str(controller.base_vla_artifact_sha256),
            "action_decoder": str(controller.action_decoder_artifact_sha256),
        }
        collector_id = str(controller.collector_id)
    except (AttributeError, TypeError) as error:
        raise PolicyAssignmentError(
            f"controller runtime protocol missing {phase}: {error}"
        ) from error
    if not isinstance(fingerprints, Mapping) or set(fingerprints) != set(
        CONTROLLER_RUNTIME_COMPONENTS
    ):
        raise PolicyAssignmentError(
            f"controller runtime fingerprint closure mismatch {phase}"
        )
    expected_fingerprints = {
        "actor": actor.runtime_fingerprint_sha256,
        "base_vla": registration.runtime_components[
            "base_vla"
        ].runtime_fingerprint_sha256,
        "action_decoder": registration.runtime_components[
            "action_decoder"
        ].runtime_fingerprint_sha256,
    }
    expected_artifacts = {
        "actor": actor.sha256,
        "base_vla": registration.runtime_components["base_vla"].artifact_sha256,
        "action_decoder": registration.runtime_components[
            "action_decoder"
        ].artifact_sha256,
    }
    identity = tuple(
        sorted(
            [("collector_id", collector_id)]
            + [(f"artifact:{name}", value) for name, value in artifacts.items()]
            + [
                (f"fingerprint:{name}", str(fingerprints[name]))
                for name in CONTROLLER_RUNTIME_COMPONENTS
            ]
        )
    )
    expected = tuple(
        sorted(
            [("collector_id", actor.collector_id)]
            + [
                (f"artifact:{name}", value)
                for name, value in expected_artifacts.items()
            ]
            + [
                (f"fingerprint:{name}", value)
                for name, value in expected_fingerprints.items()
            ]
        )
    )
    if identity != expected:
        raise PolicyAssignmentError(
            f"controller actor identity mismatch {phase}: "
            f"expected {expected}, got {identity}"
        )
    return identity


def _external_runtime_identity(
    feature_encoder: RichFeatureEncoder,
    env: RuntimeEnvironment,
    registration: ValidatedRegistration,
    phase: str,
) -> tuple[tuple[str, str], ...]:
    try:
        actual = {
            "feature_encoder": (
                str(feature_encoder.artifact_sha256),
                str(feature_encoder.recompute_runtime_fingerprint_sha256()),
            ),
            "environment": (
                str(env.artifact_sha256),
                str(env.recompute_runtime_fingerprint_sha256()),
            ),
        }
    except (AttributeError, TypeError) as error:
        raise ContractError(
            f"feature/environment runtime protocol missing {phase}: {error}"
        ) from error
    expected = {
        name: (
            registration.runtime_components[name].artifact_sha256,
            registration.runtime_components[name].runtime_fingerprint_sha256,
        )
        for name in ("feature_encoder", "environment")
    }
    if actual != expected:
        raise ContractError(
            f"feature/environment runtime identity mismatch {phase}: "
            f"expected {expected}, got {actual}"
        )
    return tuple(
        sorted(
            (f"{name}:artifact", artifact)
            for name, (artifact, _fingerprint) in actual.items()
        )
        + sorted(
            (f"{name}:fingerprint", fingerprint)
            for name, (_artifact, fingerprint) in actual.items()
        )
    )


def _rich_encoding(
    value: Any,
    label: str,
    expected_encoder_fingerprint: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not isinstance(value, RichEncoding):
        raise ContractError(f"{label} must be RichEncoding, not a bare/clock vector")
    tensor = _rich_boundary(value.value, label)
    evidence = value.evidence
    evidence_fields = {
        "schema",
        "observation_sha256",
        "camera_block_sha256",
        "proprio_sha256",
        "feature_encoder_runtime_fingerprint_sha256",
    }
    if not isinstance(evidence, Mapping) or set(evidence) != evidence_fields:
        raise ContractError(f"{label} evidence field closure mismatch")
    camera_blocks = [
        tensor[
            index * CAMERA_BLOCK_DIM : (index + 1) * CAMERA_BLOCK_DIM
        ]
        for index in range(4)
    ]
    camera_hashes = [tensor_sha256(block) for block in camera_blocks]
    proprio_hash = tensor_sha256(tensor[-PROPRIO_DIM:])
    observation_sha = canonical_sha256(
        {
            "camera_block_sha256": camera_hashes,
            "proprio_sha256": proprio_hash,
        }
    )
    expected_evidence = {
        "schema": RICH_ENCODING_SCHEMA,
        "observation_sha256": observation_sha,
        "camera_block_sha256": camera_hashes,
        "proprio_sha256": proprio_hash,
        "feature_encoder_runtime_fingerprint_sha256": expected_encoder_fingerprint,
    }
    if dict(evidence) != expected_evidence:
        raise ContractError(f"{label} evidence does not match encoded rich blocks")
    if any(float(block.std(unbiased=False)) <= 1e-8 for block in camera_blocks):
        raise ContractError(f"{label} contains a constant/clock camera block")
    if len(set(camera_hashes)) < 2:
        raise ContractError(f"{label} camera blocks are all identical")
    return tensor, dict(evidence)


def _decoded_action_chunk(value: Any, label: str) -> torch.Tensor:
    array = np.asarray(value)
    if array.dtype != np.float32:
        raise ContractError(f"{label} must be exact float32, got {array.dtype}")
    if array.shape != (C, ACTION_DIM) or not np.isfinite(array).all():
        raise ContractError(f"{label} is not finite float32 [10,7]")
    return torch.from_numpy(np.ascontiguousarray(array)).clone()


def _decoded_action_change_metrics(
    env_action_base: torch.Tensor,
    env_action_exec: torch.Tensor,
    contract: ActionChangeContract,
) -> ActionChangeMetrics:
    if env_action_base.shape != env_action_exec.shape:
        raise ContractError("decoded base/executed action shapes differ")
    if env_action_base.shape[-1] != ACTION_DIM:
        raise ContractError("decoded actions have the wrong action dimension")
    difference = env_action_exec.double() - env_action_base.double()
    flat_difference = difference.reshape(-1, ACTION_DIM)
    step_l2 = torch.linalg.vector_norm(flat_difference, dim=-1)
    if not bool(torch.isfinite(step_l2).all()) or len(step_l2) == 0:
        raise ContractError("decoded step deltas are empty or non-finite")
    changed_steps = int(torch.any(flat_difference != 0, dim=-1).sum())
    meaningful_steps = int(
        (step_l2 >= contract.meaningful_decoded_step_l2).sum()
    )
    total_steps = len(step_l2)
    changed_fraction = changed_steps / total_steps
    meaningful_fraction = meaningful_steps / total_steps
    decoded_delta_l2 = float(torch.linalg.vector_norm(difference).item())
    p10, median, p90 = (
        float(value)
        for value in torch.quantile(
            step_l2,
            torch.tensor([0.10, 0.50, 0.90], dtype=torch.float64),
        ).tolist()
    )
    maximum = float(step_l2.max().item())
    if changed_fraction < contract.min_changed_action_fraction_per_episode:
        raise ContractError(
            "decoded changed-action coverage below preregistered minimum: "
            f"{changed_fraction} < "
            f"{contract.min_changed_action_fraction_per_episode}"
        )
    if decoded_delta_l2 < contract.min_decoded_action_delta_l2_per_episode:
        raise ContractError(
            "decoded action delta L2 below preregistered minimum: "
            f"{decoded_delta_l2} < "
            f"{contract.min_decoded_action_delta_l2_per_episode}"
        )
    if (
        meaningful_fraction
        < contract.min_meaningful_action_fraction_per_episode
    ):
        raise ContractError(
            "meaningful decoded-action coverage below fixed minimum: "
            f"{meaningful_fraction} < "
            f"{contract.min_meaningful_action_fraction_per_episode}"
        )
    if p10 < contract.min_decoded_step_l2_p10_per_episode:
        raise ContractError(
            "decoded step-L2 p10 below fixed minimum: "
            f"{p10} < {contract.min_decoded_step_l2_p10_per_episode}"
        )
    if maximum > contract.max_decoded_step_l2_per_episode:
        raise ContractError(
            "decoded per-step action delta exceeds fixed maximum: "
            f"{maximum} > {contract.max_decoded_step_l2_per_episode}"
        )
    return ActionChangeMetrics(
        changed_action_steps=changed_steps,
        meaningful_action_steps=meaningful_steps,
        decoded_action_delta_l2=decoded_delta_l2,
        decoded_step_l2_p10=p10,
        decoded_step_l2_median=median,
        decoded_step_l2_p90=p90,
        decoded_step_l2_max=maximum,
    )


def _validate_decoded_action_chunk_strength(
    env_action_base: torch.Tensor,
    env_action_exec: torch.Tensor,
    contract: ActionChangeContract,
    chunk_index: int,
) -> None:
    difference = (env_action_exec.double() - env_action_base.double()).reshape(
        -1,
        ACTION_DIM,
    )
    step_l2 = torch.linalg.vector_norm(difference, dim=-1)
    meaningful_fraction = float(
        (step_l2 >= contract.meaningful_decoded_step_l2).double().mean().item()
    )
    p10 = float(torch.quantile(step_l2, 0.10).item())
    maximum = float(step_l2.max().item())
    if (
        meaningful_fraction < contract.min_meaningful_action_fraction_per_episode
        or p10 < contract.min_decoded_step_l2_p10_per_episode
        or maximum > contract.max_decoded_step_l2_per_episode
    ):
        raise ContractError(
            f"decoded chunk {chunk_index} violates meaningful fixed action "
            "strength/quantile/maximum contract"
        )


def _global_rng_state_sha256() -> str:
    numpy_state = np.random.get_state()
    return canonical_sha256(
        {
            "python_random": hashlib.sha256(
                repr(random.getstate()).encode("utf-8")
            ).hexdigest(),
            "numpy_kind": numpy_state[0],
            "numpy_keys_sha256": hashlib.sha256(
                numpy_state[1].tobytes(order="C")
            ).hexdigest(),
            "numpy_position": int(numpy_state[2]),
            "numpy_has_gauss": int(numpy_state[3]),
            "numpy_cached_gaussian": float(numpy_state[4]),
            "torch_cpu": tensor_sha256(torch.random.get_rng_state()),
        }
    )


def _propose_without_entropy_contract_test_only(
    controller: EpisodeController,
    obs: Any,
    rich_z: torch.Tensor,
    chunk_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mechanical tripwire; formal determinism remains separately hard-gated."""
    before = _global_rng_state_sha256()
    original_os_urandom = os.urandom
    original_random_urandom = getattr(random, "_urandom", None)

    def blocked_urandom(_size: int) -> bytes:
        raise ContractError("entropy source use is forbidden during proposal")

    os.urandom = blocked_urandom
    if original_random_urandom is not None:
        random._urandom = blocked_urandom  # type: ignore[attr-defined]
    try:
        result = controller.propose(obs, rich_z, chunk_index)
    finally:
        os.urandom = original_os_urandom
        if original_random_urandom is not None:
            random._urandom = original_random_urandom  # type: ignore[attr-defined]
    after = _global_rng_state_sha256()
    if after != before:
        raise ContractError("global RNG state changed during deterministic proposal")
    return result


def _collect_fixed_horizon_episode_contract_test_only_core(
    registration: ValidatedRegistration,
    episode_id: int,
    controller: EpisodeController,
    env: RuntimeEnvironment,
    feature_encoder: RichFeatureEncoder,
) -> EpisodeCollection:
    """Collect one fixed-horizon episode through an injected, testable adapter.

    The returned wrapper ``terminated`` flag is deliberately not a loop
    condition because ``ChainEnv`` sets it for task success.  ``info['done']``
    is the underlying simulator termination bit and is the only technical-done
    source accepted by the formal contract.
    """
    registration = revalidate_registration_unchanged(registration)
    if type(episode_id) is not int or not 0 <= episode_id < EXPECTED_EPISODES:
        raise ContractError("episode_id is not a registered D1 episode")
    assignment = registration.assignments[episode_id]
    if assignment.episode_id != episode_id:
        raise ContractError("registration assignment lookup is inconsistent")
    actor = registration.actors[assignment.collector_id]
    action_change = registration.action_change
    frozen_actor_identity = _controller_identity(
        controller,
        registration,
        actor,
        "before reset",
    )
    frozen_external_identity = _external_runtime_identity(
        feature_encoder,
        env,
        registration,
        "before reset",
    )

    random.seed(assignment.env_seed)
    np.random.seed(assignment.env_seed)
    torch.manual_seed(assignment.env_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(assignment.env_seed)
    controller.reset_episode(assignment.env_seed)
    if (
        _controller_identity(controller, registration, actor, "after reset")
        != frozen_actor_identity
    ):
        raise PolicyAssignmentError("controller actor identity changed during reset")
    if (
        _external_runtime_identity(
            feature_encoder,
            env,
            registration,
            "after controller reset",
        )
        != frozen_external_identity
    ):
        raise ContractError("external runtime identity changed during controller reset")
    reset_value = env.reset(seed=assignment.env_seed)
    if (
        _controller_identity(
            controller,
            registration,
            actor,
            "after environment reset",
        )
        != frozen_actor_identity
    ):
        raise PolicyAssignmentError("controller actor identity changed during env reset")
    if (
        _external_runtime_identity(
            feature_encoder,
            env,
            registration,
            "after environment reset",
        )
        != frozen_external_identity
    ):
        raise ContractError("external runtime identity changed during env reset")
    obs = reset_value[0] if isinstance(reset_value, tuple) else reset_value

    def encode_boundary(observation: Any, label: str) -> tuple[torch.Tensor, dict[str, Any]]:
        before = _external_runtime_identity(
            feature_encoder,
            env,
            registration,
            f"before {label}",
        )
        encoded = feature_encoder.encode(observation)
        tensor, evidence = _rich_encoding(
            encoded,
            label,
            registration.runtime_components[
                "feature_encoder"
            ].runtime_fingerprint_sha256,
        )
        after = _external_runtime_identity(
            feature_encoder,
            env,
            registration,
            f"after {label}",
        )
        if before != frozen_external_identity or after != frozen_external_identity:
            raise ContractError(f"external runtime identity changed during {label}")
        return tensor, evidence

    first_boundary, first_evidence = encode_boundary(obs, "boundary t=0")
    boundaries = [first_boundary]
    boundary_evidence = [first_evidence]
    base_chunks: list[torch.Tensor] = []
    deltas: list[torch.Tensor] = []
    executed_chunks: list[torch.Tensor] = []
    decoded_base_chunks: list[torch.Tensor] = []
    decoded_exec_chunks: list[torch.Tensor] = []
    first_success_step: int | None = None
    raw_done_at_horizon = False
    truncated_at_horizon = False
    t = 0

    for chunk_index in range(CHUNKS_PER_EPISODE):
        if (
            _controller_identity(
                controller,
                registration,
                actor,
                f"before chunk {chunk_index}",
            )
            != frozen_actor_identity
        ):
            raise PolicyAssignmentError(
                f"controller actor identity changed before chunk {chunk_index}"
            )
        u_base_raw, delta_raw = _propose_without_entropy_contract_test_only(
            controller,
            obs,
            boundaries[-1],
            chunk_index,
        )
        if (
            _controller_identity(
                controller,
                registration,
                actor,
                f"after chunk {chunk_index} proposal",
            )
            != frozen_actor_identity
        ):
            raise PolicyAssignmentError(
                f"controller actor identity changed while proposing chunk {chunk_index}"
            )
        u_base = _action_chunk(u_base_raw, f"u_base chunk {chunk_index}")
        delta = _action_chunk(delta_raw, f"delta chunk {chunk_index}")
        if float(delta.abs().max().item()) > actor.scale:
            raise ContractError(
                f"normalized residual exceeds registered actor scale {actor.scale}"
            )
        replay_base_raw, replay_delta_raw = _propose_without_entropy_contract_test_only(
            controller,
            obs,
            boundaries[-1],
            chunk_index,
        )
        replay_base = _action_chunk(
            replay_base_raw,
            f"replayed u_base chunk {chunk_index}",
        )
        replay_delta = _action_chunk(
            replay_delta_raw,
            f"replayed delta chunk {chunk_index}",
        )
        if not torch.equal(u_base, replay_base) or not torch.equal(delta, replay_delta):
            raise ContractError(
                f"chunk-noise/deterministic replay violation at chunk {chunk_index}"
            )
        third_base_raw, third_delta_raw = _propose_without_entropy_contract_test_only(
            controller,
            obs,
            boundaries[-1],
            chunk_index,
        )
        third_base = _action_chunk(
            third_base_raw,
            f"third u_base replay chunk {chunk_index}",
        )
        third_delta = _action_chunk(
            third_delta_raw,
            f"third delta replay chunk {chunk_index}",
        )
        if not torch.equal(u_base, third_base) or not torch.equal(delta, third_delta):
            raise ContractError(
                f"pair-cache/non-adjacent replay violation at chunk {chunk_index}"
            )
        if (
            _controller_identity(
                controller,
                registration,
                actor,
                f"after chunk {chunk_index} replay",
            )
            != frozen_actor_identity
        ):
            raise PolicyAssignmentError(
                f"controller actor identity changed during replay {chunk_index}"
            )
        u_exec = (u_base + delta).contiguous()
        decoded_base = _decoded_action_chunk(
            controller.decode(u_base), f"decoded base chunk {chunk_index}"
        )
        if (
            _controller_identity(
                controller,
                registration,
                actor,
                f"after base decode {chunk_index}",
            )
            != frozen_actor_identity
        ):
            raise PolicyAssignmentError(
                f"controller actor identity changed during base decode {chunk_index}"
            )
        decoded_exec = _decoded_action_chunk(
            controller.decode(u_exec), f"decoded executed chunk {chunk_index}"
        )
        decoded_base_repeat = _decoded_action_chunk(
            controller.decode(u_base), f"repeated decoded base chunk {chunk_index}"
        )
        decoded_exec_repeat = _decoded_action_chunk(
            controller.decode(u_exec),
            f"repeated decoded executed chunk {chunk_index}",
        )
        if not torch.equal(decoded_base, decoded_base_repeat) or not torch.equal(
            decoded_exec, decoded_exec_repeat
        ):
            raise ContractError(
                f"controller decoder is not pure/exact at chunk {chunk_index}"
            )
        if (
            _controller_identity(
                controller,
                registration,
                actor,
                f"after executed decode {chunk_index}",
            )
            != frozen_actor_identity
        ):
            raise PolicyAssignmentError(
                f"controller actor identity changed during executed decode {chunk_index}"
            )
        _validate_decoded_action_chunk_strength(
            decoded_base,
            decoded_exec,
            action_change,
            chunk_index,
        )
        base_chunks.append(u_base)
        deltas.append(delta)
        executed_chunks.append(u_exec)
        decoded_base_chunks.append(decoded_base)
        decoded_exec_chunks.append(decoded_exec)

        for action_offset in range(C):
            recorded_action = decoded_exec[action_offset]
            step_action = recorded_action.numpy().copy()
            if not np.array_equal(step_action, recorded_action.numpy()):
                raise ContractError("executed env action changed before env.step")
            obs, _reward, terminated, truncated, info = env.step(step_action)
            if not np.array_equal(step_action, recorded_action.numpy()):
                raise ContractError("environment mutated the passed action buffer")
            t += 1
            if not isinstance(info, Mapping) or "done" not in info:
                raise ContractError("environment info must expose underlying info['done']")
            raw_done = bool(info["done"])
            is_success = bool(info.get("is_success", False))
            if is_success and first_success_step is None:
                first_success_step = t
            if bool(terminated) and not (raw_done or is_success or bool(truncated)):
                raise ContractError("terminated has no raw-done/success/truncation explanation")
            if (raw_done or bool(truncated)) and t < HORIZON_STEPS:
                reason = "raw_done" if raw_done else "truncated"
                raise TechnicalTermination(
                    f"episode {assignment.episode_id} {reason} at {t} < {HORIZON_STEPS}"
                )
            if t == HORIZON_STEPS:
                raw_done_at_horizon = raw_done
                truncated_at_horizon = bool(truncated)

        boundary, evidence = encode_boundary(obs, f"boundary t={t}")
        boundaries.append(boundary)
        boundary_evidence.append(evidence)

    if t != HORIZON_STEPS or len(boundaries) != BOUNDARIES_PER_EPISODE:
        raise ContractError("fixed-horizon collection did not produce 75/76")
    if (
        _controller_identity(controller, registration, actor, "at episode end")
        != frozen_actor_identity
    ):
        raise PolicyAssignmentError("controller actor identity changed by episode end")
    if (
        _external_runtime_identity(
            feature_encoder,
            env,
            registration,
            "at episode end",
        )
        != frozen_external_identity
    ):
        raise ContractError("external runtime identity changed by episode end")
    boundary_tensor = torch.stack(boundaries)
    u_base_tensor = torch.stack(base_chunks)
    delta_tensor = torch.stack(deltas)
    u_exec_tensor = torch.stack(executed_chunks)
    env_action_base_tensor = torch.stack(decoded_base_chunks)
    env_action_exec_tensor = torch.stack(decoded_exec_chunks)
    if not torch.equal(u_exec_tensor, u_base_tensor + delta_tensor):
        raise ContractError("action trace violates u_exec = u_base + delta")
    action_metrics = _decoded_action_change_metrics(
        env_action_base_tensor,
        env_action_exec_tensor,
        action_change,
    )
    sidecar = {
        "schema": SIDECAR_SCHEMA,
        "episode_id": assignment.episode_id,
        "env_seed": assignment.env_seed,
        "success_observed": first_success_step is not None,
        "first_success_step": first_success_step,
        "raw_done_at_horizon": raw_done_at_horizon,
        "truncated_at_horizon": truncated_at_horizon,
    }
    return EpisodeCollection(
        assignment=assignment,
        boundaries=boundary_tensor,
        u_base=u_base_tensor,
        delta=delta_tensor,
        u_exec=u_exec_tensor,
        env_action_base=env_action_base_tensor,
        env_action_exec=env_action_exec_tensor,
        changed_action_steps=action_metrics.changed_action_steps,
        meaningful_action_steps=action_metrics.meaningful_action_steps,
        decoded_action_delta_l2=action_metrics.decoded_action_delta_l2,
        decoded_step_l2_p10=action_metrics.decoded_step_l2_p10,
        decoded_step_l2_median=action_metrics.decoded_step_l2_median,
        decoded_step_l2_p90=action_metrics.decoded_step_l2_p90,
        decoded_step_l2_max=action_metrics.decoded_step_l2_max,
        actor_artifact_sha256=actor.sha256,
        actor_runtime_fingerprint_sha256=actor.runtime_fingerprint_sha256,
        runtime_component_fingerprints_sha256={
            "actor": actor.runtime_fingerprint_sha256,
            **{
                name: registration.runtime_components[name].runtime_fingerprint_sha256
                for name in REGISTERED_RUNTIME_COMPONENTS
            },
        },
        boundary_evidence=tuple(boundary_evidence),
        evaluation_sidecar=sidecar,
    )


def collect_fixed_horizon_episode_contract_test_only(
    registration: ValidatedRegistration,
    episode_id: int,
    controller: EpisodeController,
    env: RuntimeEnvironment,
    feature_encoder: RichFeatureEncoder,
    attempt_ledger: AttemptLedger,
) -> EpisodeCollection:
    """Collect exactly one registered attempt and append its terminal status."""
    registration = revalidate_registration_unchanged(registration)
    if type(episode_id) is not int or not 0 <= episode_id < EXPECTED_EPISODES:
        raise ContractError("episode_id is not a registered D1 episode")
    assignment = registration.assignments[episode_id]
    actor = registration.actors[assignment.collector_id]
    if attempt_ledger.registration_sha256 != registration.file_sha256:
        raise ContractError("attempt ledger registration binding mismatch")
    if REAL_SEMANTIC_BINDING_IMPLEMENTED and (
        not attempt_ledger.formal_path_bound
        or attempt_ledger.path.expanduser().resolve()
        != registration.attempt_ledger_path
    ):
        raise ContractError(
            "real collection requires the one preregistered append-only ledger"
        )
    read_attempt_ledger(attempt_ledger.path, registration.file_sha256)
    _controller_identity(controller, registration, actor, "before reset")
    _external_runtime_identity(
        feature_encoder,
        env,
        registration,
        "before reset",
    )
    attempt_ledger.start(assignment)
    try:
        episode = _collect_fixed_horizon_episode_contract_test_only_core(
            registration,
            episode_id,
            controller,
            env,
            feature_encoder,
        )
    except TechnicalTermination as error:
        attempt_ledger.finish(assignment, "technical_failure", str(error))
        raise
    except Exception as error:
        attempt_ledger.finish(assignment, "contract_failure", str(error))
        raise
    attempt_ledger.finish(
        assignment,
        "completed",
        episode_payload_sha256=episode_collection_payload_sha256(
            registration,
            episode,
        ),
    )
    return episode


def collect_fixed_horizon_episode(
    registration: ValidatedRegistration,
    episode_id: int,
    controller: EpisodeController,
    env: RuntimeEnvironment,
    feature_encoder: RichFeatureEncoder,
    attempt_ledger: AttemptLedger,
) -> EpisodeCollection:
    """Formal entrypoint; currently rejects before ledger access or env reset."""
    del registration, episode_id, controller, env, feature_encoder, attempt_ledger
    require_formal_collection_ready("formal fixed-horizon collection")
    raise RealCollectorNotImplemented(
        "NO-GO: formal core-owned collection executor is not implemented"
    )


def validate_episode_collection(
    registration: ValidatedRegistration,
    episode: EpisodeCollection,
) -> None:
    if type(episode) is not EpisodeCollection:
        raise ContractError("episode collection must use the exact frozen schema")
    assignment = episode.assignment
    if (
        type(assignment.episode_id) is not int
        or not 0 <= assignment.episode_id < EXPECTED_EPISODES
        or registration.assignments[assignment.episode_id] != assignment
    ):
        raise ContractError("episode assignment is not the exact registered assignment")
    actor = registration.actors[assignment.collector_id]
    if episode.actor_artifact_sha256 != actor.sha256:
        raise ContractError("episode actor artifact SHA mismatch")
    if episode.actor_runtime_fingerprint_sha256 != actor.runtime_fingerprint_sha256:
        raise ContractError("episode actor runtime fingerprint mismatch")
    expected_runtime = {
        "actor": actor.runtime_fingerprint_sha256,
        **{
            name: registration.runtime_components[name].runtime_fingerprint_sha256
            for name in REGISTERED_RUNTIME_COMPONENTS
        },
    }
    if episode.runtime_component_fingerprints_sha256 != expected_runtime:
        raise ContractError("episode runtime-component fingerprints mismatch")
    if len(episode.boundary_evidence) != BOUNDARIES_PER_EPISODE:
        raise ContractError("episode boundary evidence count mismatch")
    boundaries = _require_tensor(
        episode.boundaries,
        (BOUNDARIES_PER_EPISODE, RICH_DIM),
        "episode boundaries",
    )
    if boundaries.dtype != torch.float32:
        raise ContractError("episode boundaries must be float32")
    encoder_fingerprint = registration.runtime_components[
        "feature_encoder"
    ].runtime_fingerprint_sha256
    for index, evidence in enumerate(episode.boundary_evidence):
        checked, _ = _rich_encoding(
            RichEncoding(boundaries[index], evidence),
            f"stored boundary {index}",
            encoder_fingerprint,
        )
        if not torch.equal(checked, boundaries[index]):
            raise ContractError("stored rich boundary changed during validation")
    u_base = _require_tensor(
        episode.u_base,
        (CHUNKS_PER_EPISODE, C, ACTION_DIM),
        "episode u_base",
    )
    delta = _require_tensor(
        episode.delta,
        (CHUNKS_PER_EPISODE, C, ACTION_DIM),
        "episode delta",
    )
    u_exec = _require_tensor(
        episode.u_exec,
        (CHUNKS_PER_EPISODE, C, ACTION_DIM),
        "episode u_exec",
    )
    env_action_base = _require_tensor(
        episode.env_action_base,
        (CHUNKS_PER_EPISODE, C, ACTION_DIM),
        "episode env_action_base",
    )
    env_action_exec = _require_tensor(
        episode.env_action_exec,
        (CHUNKS_PER_EPISODE, C, ACTION_DIM),
        "episode env_action_exec",
    )
    for name, tensor in (
        ("u_base", u_base),
        ("delta", delta),
        ("u_exec", u_exec),
        ("env_action_base", env_action_base),
        ("env_action_exec", env_action_exec),
    ):
        if tensor.dtype != torch.float32:
            raise ContractError(f"episode {name} must be float32")
    if not torch.equal(u_exec, u_base + delta):
        raise ContractError("episode action trace decomposition changed")
    if float(delta.abs().max().item()) > actor.scale:
        raise ContractError("episode normalized residual exceeds actor scale")
    action_metrics = _decoded_action_change_metrics(
        env_action_base,
        env_action_exec,
        registration.action_change,
    )
    if (
        type(episode.changed_action_steps) is not int
        or episode.changed_action_steps != action_metrics.changed_action_steps
    ):
        raise ContractError("episode changed-action count changed")
    if (
        type(episode.meaningful_action_steps) is not int
        or episode.meaningful_action_steps
        != action_metrics.meaningful_action_steps
    ):
        raise ContractError("episode meaningful-action count changed")
    if (
        isinstance(episode.decoded_action_delta_l2, bool)
        or not isinstance(episode.decoded_action_delta_l2, (int, float))
        or not np.isfinite(float(episode.decoded_action_delta_l2))
        or float(episode.decoded_action_delta_l2)
        != action_metrics.decoded_action_delta_l2
    ):
        raise ContractError("episode decoded-action norm changed")
    metric_fields = (
        "decoded_step_l2_p10",
        "decoded_step_l2_median",
        "decoded_step_l2_p90",
        "decoded_step_l2_max",
    )
    for name in metric_fields:
        value = getattr(episode, name)
        if (
            type(value) is not float
            or not np.isfinite(value)
            or value != getattr(action_metrics, name)
        ):
            raise ContractError(f"episode {name} changed")
    _validate_episode_sidecar(assignment, episode.evaluation_sidecar)


def _validate_episode_sidecar(
    assignment: EpisodeAssignment,
    sidecar: Mapping[str, Any],
) -> None:
    sidecar_fields = {
        "schema",
        "episode_id",
        "env_seed",
        "success_observed",
        "first_success_step",
        "raw_done_at_horizon",
        "truncated_at_horizon",
    }
    if not isinstance(sidecar, Mapping) or set(sidecar) != sidecar_fields:
        raise ContractError("episode evaluation sidecar field closure mismatch")
    if (
        sidecar.get("schema") != SIDECAR_SCHEMA
        or type(sidecar.get("episode_id")) is not int
        or sidecar.get("episode_id") != assignment.episode_id
        or type(sidecar.get("env_seed")) is not int
        or sidecar.get("env_seed") != assignment.env_seed
        or type(sidecar.get("success_observed")) is not bool
        or type(sidecar.get("raw_done_at_horizon")) is not bool
        or type(sidecar.get("truncated_at_horizon")) is not bool
    ):
        raise ContractError("episode evaluation sidecar identity/type mismatch")
    success = sidecar["success_observed"]
    success_step = sidecar["first_success_step"]
    if success:
        if type(success_step) is not int or not 1 <= success_step <= HORIZON_STEPS:
            raise ContractError("episode first_success_step is invalid")
    elif success_step is not None:
        raise ContractError("failed episode cannot carry first_success_step")


def validate_evaluation_sidecar(
    registration: ValidatedRegistration,
    sidecar: Mapping[str, Any],
) -> None:
    """Validate the physically isolated 96-row outcome aggregate exactly."""
    if not isinstance(sidecar, Mapping) or set(sidecar) != {
        "schema",
        "registration_sha256",
        "episodes",
    }:
        raise ContractError("evaluation sidecar aggregate field closure mismatch")
    episodes = sidecar.get("episodes")
    if (
        sidecar.get("schema") != SIDECAR_SCHEMA
        or sidecar.get("registration_sha256") != registration.file_sha256
        or not isinstance(episodes, list)
        or len(episodes) != EXPECTED_EPISODES
    ):
        raise ContractError("evaluation sidecar aggregate identity/count mismatch")
    for assignment, episode_sidecar in zip(registration.assignments, episodes):
        _validate_episode_sidecar(assignment, episode_sidecar)


def _episode_payload_sha256(
    registration_sha256: str,
    assignment: EpisodeAssignment,
    boundaries: torch.Tensor,
    boundary_evidence: Sequence[Mapping[str, Any]],
    u_base: torch.Tensor,
    delta: torch.Tensor,
    u_exec: torch.Tensor,
    env_action_base: torch.Tensor,
    env_action_exec: torch.Tensor,
    changed_action_steps: int,
    meaningful_action_steps: int,
    decoded_action_delta_l2: float,
    decoded_step_l2_p10: float,
    decoded_step_l2_median: float,
    decoded_step_l2_p90: float,
    decoded_step_l2_max: float,
    actor_artifact_sha256: str,
    actor_runtime_fingerprint_sha256: str,
    runtime_component_fingerprints_sha256: Mapping[str, str],
    evaluation_sidecar: Mapping[str, Any],
) -> str:
    """Digest every per-episode datum that a completed attempt attests."""
    return canonical_sha256(
        {
            "schema": "v246_evolve1_d1_episode_payload_digest_v1",
            "registration_sha256": registration_sha256,
            "assignment": {
                "episode_id": assignment.episode_id,
                "env_seed": assignment.env_seed,
                "collector_id": assignment.collector_id,
                "split": assignment.split,
            },
            "tensor_sha256": {
                "boundaries": tensor_sha256(boundaries),
                "u_base": tensor_sha256(u_base),
                "delta": tensor_sha256(delta),
                "u_exec": tensor_sha256(u_exec),
                "env_action_base": tensor_sha256(env_action_base),
                "env_action_exec": tensor_sha256(env_action_exec),
            },
            "boundary_evidence_sha256": canonical_sha256(
                [dict(item) for item in boundary_evidence]
            ),
            "changed_action_steps": changed_action_steps,
            "meaningful_action_steps": meaningful_action_steps,
            "decoded_action_delta_l2": decoded_action_delta_l2,
            "decoded_step_l2_p10": decoded_step_l2_p10,
            "decoded_step_l2_median": decoded_step_l2_median,
            "decoded_step_l2_p90": decoded_step_l2_p90,
            "decoded_step_l2_max": decoded_step_l2_max,
            "actor_artifact_sha256": actor_artifact_sha256,
            "actor_runtime_fingerprint_sha256": (
                actor_runtime_fingerprint_sha256
            ),
            "runtime_component_fingerprints_sha256": dict(
                runtime_component_fingerprints_sha256
            ),
            "evaluation_sidecar_sha256": canonical_sha256(
                dict(evaluation_sidecar)
            ),
        }
    )


def episode_collection_payload_sha256(
    registration: ValidatedRegistration,
    episode: EpisodeCollection,
) -> str:
    validate_episode_collection(registration, episode)
    return _episode_payload_sha256(
        registration.file_sha256,
        episode.assignment,
        episode.boundaries,
        episode.boundary_evidence,
        episode.u_base,
        episode.delta,
        episode.u_exec,
        episode.env_action_base,
        episode.env_action_exec,
        episode.changed_action_steps,
        episode.meaningful_action_steps,
        episode.decoded_action_delta_l2,
        episode.decoded_step_l2_p10,
        episode.decoded_step_l2_median,
        episode.decoded_step_l2_p90,
        episode.decoded_step_l2_max,
        episode.actor_artifact_sha256,
        episode.actor_runtime_fingerprint_sha256,
        episode.runtime_component_fingerprints_sha256,
        episode.evaluation_sidecar,
    )


def episode_collection_payload_sha256_by_id(
    registration: ValidatedRegistration,
    episodes: Sequence[EpisodeCollection],
) -> dict[int, str]:
    if len(episodes) != EXPECTED_EPISODES:
        raise ContractError("payload SHA binding requires exactly 96 episodes")
    result: dict[int, str] = {}
    for episode in episodes:
        episode_id = episode.assignment.episode_id
        if episode_id in result:
            raise ContractError("payload SHA binding contains a duplicate episode")
        result[episode_id] = episode_collection_payload_sha256(
            registration,
            episode,
        )
    if set(result) != set(range(EXPECTED_EPISODES)):
        raise ContractError("payload SHA binding does not cover episodes 0..95")
    return result


def derive_episode_payloads(
    registration: ValidatedRegistration,
    episode: EpisodeCollection,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Mechanically derive one tape, boundary record, and action trace."""
    validate_episode_collection(registration, episode)
    action_change = registration.action_change
    boundaries = _require_tensor(
        episode.boundaries,
        (BOUNDARIES_PER_EPISODE, RICH_DIM),
        "episode boundaries",
    ).float()
    u_base = _require_tensor(
        episode.u_base, (CHUNKS_PER_EPISODE, C, ACTION_DIM), "episode u_base"
    ).float()
    delta = _require_tensor(
        episode.delta, (CHUNKS_PER_EPISODE, C, ACTION_DIM), "episode delta"
    ).float()
    u_exec = _require_tensor(
        episode.u_exec, (CHUNKS_PER_EPISODE, C, ACTION_DIM), "episode u_exec"
    ).float()
    env_action_base = _require_tensor(
        episode.env_action_base,
        (CHUNKS_PER_EPISODE, C, ACTION_DIM),
        "episode env_action_base",
    ).float()
    env_action_exec = _require_tensor(
        episode.env_action_exec,
        (CHUNKS_PER_EPISODE, C, ACTION_DIM),
        "episode env_action_exec",
    ).float()
    if not torch.equal(u_exec, u_base + delta):
        raise ContractError("episode action trace decomposition changed")
    action_metrics = _decoded_action_change_metrics(
        env_action_base,
        env_action_exec,
        action_change,
    )
    if action_metrics.changed_action_steps != episode.changed_action_steps:
        raise ContractError("episode changed-action count changed")
    if action_metrics.meaningful_action_steps != episode.meaningful_action_steps:
        raise ContractError("episode meaningful-action count changed")
    if action_metrics.decoded_action_delta_l2 != episode.decoded_action_delta_l2:
        raise ContractError("episode decoded-action norm changed")
    if not is_sha256(episode.actor_artifact_sha256):
        raise ContractError("episode actor artifact SHA is invalid")
    if not is_sha256(episode.actor_runtime_fingerprint_sha256):
        raise ContractError("episode actor runtime fingerprint is invalid")
    episode_ids = torch.full(
        (CHUNKS_PER_EPISODE,), episode.assignment.episode_id, dtype=torch.long
    )
    times = torch.arange(CHUNKS_PER_EPISODE, dtype=torch.long) * C
    tape = {
        "schema": DATA_TAPE_SCHEMA,
        "z": boundaries[:-1].clone(),
        "u": u_exec.clone(),
        "z_next": boundaries[1:].clone(),
        "episode": episode_ids,
        "t": times,
        "sigma": torch.zeros(CHUNKS_PER_EPISODE, dtype=torch.float32),
        "data_role": episode.assignment.split,
        "task": TASK,
        "c": C,
        "latent_dim": RICH_DIM,
        "proprio_dim": PROPRIO_DIM,
        "proprio_keys": list(PROPRIO_KEYS),
        "training_feature_fields": list(TRAINING_FEATURE_FIELDS),
        "split_bookkeeping_fields": list(SPLIT_BOOKKEEPING_FIELDS),
        "legacy_outcome_fields": [],
    }
    if set(tape) != set(DATA_TAPE_FIELDS) or set(tape) & set(FORBIDDEN_DATA_FIELDS):
        raise ContractError("derived tape violates the outcome-free allow-list")
    if not torch.equal(tape["z_next"][:-1], tape["z"][1:]):
        raise ContractError("derived tape is not N+1 continuous")
    boundary_record = {
        "schema": BOUNDARY_SCHEMA,
        "episode_id": episode.assignment.episode_id,
        "z": boundaries.clone(),
        "evidence": list(episode.boundary_evidence),
    }
    trace = {
        "schema": TRACE_SCHEMA,
        "episode_id": episode.assignment.episode_id,
        "collector_id": episode.assignment.collector_id,
        "t": times,
        "u_base": u_base.clone(),
        "delta": delta.clone(),
        "u_exec": u_exec.clone(),
        "env_action_base": env_action_base.clone(),
        "env_action_exec": env_action_exec.clone(),
        "changed_action_steps": action_metrics.changed_action_steps,
        "changed_action_fraction": (
            action_metrics.changed_action_steps / HORIZON_STEPS
        ),
        "meaningful_action_steps": action_metrics.meaningful_action_steps,
        "meaningful_action_fraction": (
            action_metrics.meaningful_action_steps / HORIZON_STEPS
        ),
        "decoded_action_delta_l2": action_metrics.decoded_action_delta_l2,
        "decoded_step_l2_p10": action_metrics.decoded_step_l2_p10,
        "decoded_step_l2_median": action_metrics.decoded_step_l2_median,
        "decoded_step_l2_p90": action_metrics.decoded_step_l2_p90,
        "decoded_step_l2_max": action_metrics.decoded_step_l2_max,
        "min_changed_action_fraction_per_episode": (
            action_change.min_changed_action_fraction_per_episode
        ),
        "min_decoded_action_delta_l2_per_episode": (
            action_change.min_decoded_action_delta_l2_per_episode
        ),
        "meaningful_decoded_step_l2": (
            action_change.meaningful_decoded_step_l2
        ),
        "min_meaningful_action_fraction_per_episode": (
            action_change.min_meaningful_action_fraction_per_episode
        ),
        "min_decoded_step_l2_p10_per_episode": (
            action_change.min_decoded_step_l2_p10_per_episode
        ),
        "max_decoded_step_l2_per_episode": (
            action_change.max_decoded_step_l2_per_episode
        ),
        "actor_artifact_sha256": episode.actor_artifact_sha256,
        "actor_runtime_fingerprint_sha256": (
            episode.actor_runtime_fingerprint_sha256
        ),
        "runtime_component_fingerprints_sha256": dict(
            episode.runtime_component_fingerprints_sha256
        ),
    }
    trace_fields = {
        "schema",
        "episode_id",
        "collector_id",
        "t",
        "u_base",
        "delta",
        "u_exec",
        "env_action_base",
        "env_action_exec",
        "changed_action_steps",
        "changed_action_fraction",
        "meaningful_action_steps",
        "meaningful_action_fraction",
        "decoded_action_delta_l2",
        "decoded_step_l2_p10",
        "decoded_step_l2_median",
        "decoded_step_l2_p90",
        "decoded_step_l2_max",
        "min_changed_action_fraction_per_episode",
        "min_decoded_action_delta_l2_per_episode",
        "meaningful_decoded_step_l2",
        "min_meaningful_action_fraction_per_episode",
        "min_decoded_step_l2_p10_per_episode",
        "max_decoded_step_l2_per_episode",
        "actor_artifact_sha256",
        "actor_runtime_fingerprint_sha256",
        "runtime_component_fingerprints_sha256",
    }
    if set(trace) != trace_fields:
        raise ContractError("action trace field closure mismatch")
    if any(key in trace for key in ("success", "success_step", "outcome", "events")):
        raise ContractError("action trace contains an outcome field")
    return tape, boundary_record, trace


def training_tensor_view(
    tape: Mapping[str, Any],
    requested_fields: Sequence[str] = TRAINING_FEATURE_FIELDS,
) -> dict[str, torch.Tensor]:
    """Return the only tensors a world-model trainer is authorized to consume."""
    if not isinstance(tape, Mapping) or set(tape) != set(DATA_TAPE_FIELDS):
        raise ContractError("training consumer requires the exact D1 tape schema")
    if tape.get("schema") != DATA_TAPE_SCHEMA:
        raise ContractError("training consumer tape schema mismatch")
    metadata = {
        "data_role": "fit",
        "task": TASK,
        "c": C,
        "latent_dim": RICH_DIM,
        "proprio_dim": PROPRIO_DIM,
        "proprio_keys": list(PROPRIO_KEYS),
        "training_feature_fields": list(TRAINING_FEATURE_FIELDS),
        "split_bookkeeping_fields": list(SPLIT_BOOKKEEPING_FIELDS),
        "legacy_outcome_fields": [],
    }
    if any(
        not exact_json_equal(tape.get(name), value)
        for name, value in metadata.items()
    ):
        raise ContractError(
            "training consumer requires canonical fit-only tape metadata"
        )
    if tuple(tape.get("training_feature_fields", ())) != TRAINING_FEATURE_FIELDS:
        raise ContractError("training consumer feature allow-list mismatch")
    if tuple(tape.get("split_bookkeeping_fields", ())) != SPLIT_BOOKKEEPING_FIELDS:
        raise ContractError("training consumer bookkeeping declaration mismatch")
    if tape.get("legacy_outcome_fields") != []:
        raise ContractError("D1 training tape declares an outcome field")
    rows = tape.get("z").shape[0] if torch.is_tensor(tape.get("z")) else -1
    tensor_contracts = {
        "z": ((rows, RICH_DIM), torch.float32),
        "u": ((rows, C, ACTION_DIM), torch.float32),
        "z_next": ((rows, RICH_DIM), torch.float32),
        "episode": ((rows,), torch.int64),
        "t": ((rows,), torch.int64),
        "sigma": ((rows,), torch.float32),
    }
    if rows <= 0:
        raise ContractError("training consumer fit tape must be non-empty")
    for name, (shape, dtype) in tensor_contracts.items():
        value = tape.get(name)
        if (
            not torch.is_tensor(value)
            or value.dtype != dtype
            or tuple(value.shape) != shape
            or not bool(torch.isfinite(value).all())
        ):
            raise ContractError(
                f"training consumer tensor {name} must be finite {dtype} {shape}"
            )
    if not torch.equal(tape["sigma"], torch.zeros_like(tape["sigma"])):
        raise ContractError("training consumer sigma must be exactly zero")
    requested = tuple(requested_fields)
    if len(requested) != len(set(requested)) or not set(requested).issubset(
        TRAINING_FEATURE_FIELDS
    ):
        raise ContractError(
            "world-model training may read only z/u/z_next; "
            "episode/t/sigma/trace/sidecar are forbidden"
        )
    result: dict[str, torch.Tensor] = {}
    for name in requested:
        value = tape[name]
        if not torch.is_tensor(value) or not bool(torch.isfinite(value).all()):
            raise ContractError(f"training tensor {name} is invalid")
        result[name] = value.detach().cpu().clone()
    return result


def validate_calibration_tape(tape: Mapping[str, Any]) -> None:
    """Validate held-out calibration storage without exposing a training view."""
    if not isinstance(tape, Mapping) or tape.get("data_role") != "calibration":
        raise ContractError("calibration tape must declare calibration data_role")
    fit_schema_proxy = dict(tape)
    fit_schema_proxy["data_role"] = "fit"
    training_tensor_view(fit_schema_proxy, ())


def d0_training_tensor_view(tape: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Isolate legacy D0 outcomes: this function never indexes their values."""
    if not isinstance(tape, Mapping):
        raise ContractError("D0 training consumer requires a mapping")
    legacy = tape.get("legacy_outcome_fields")
    if legacy not in ([], ["success"]):
        raise ContractError("D0 legacy outcome declaration mismatch")
    expected_fields = set(D0_REQUIRED_FIELDS) | set(legacy)
    if set(tape) != expected_fields or tape.get("schema") != D0_TAPE_SCHEMA:
        raise ContractError("D0 training consumer schema/field closure mismatch")
    if tuple(tape.get("training_feature_fields", ())) != TRAINING_FEATURE_FIELDS:
        raise ContractError("D0 training feature allow-list mismatch")
    if tuple(tape.get("split_bookkeeping_fields", ())) != SPLIT_BOOKKEEPING_FIELDS:
        raise ContractError("D0 bookkeeping declaration mismatch")
    result: dict[str, torch.Tensor] = {}
    for name in TRAINING_FEATURE_FIELDS:
        value = tape[name]
        if not torch.is_tensor(value) or not bool(torch.isfinite(value).all()):
            raise ContractError(f"D0 training tensor {name} is invalid")
        result[name] = value.detach().cpu().clone()
    return result


def training_access_policy() -> dict[str, Any]:
    return {
        "schema": TRAINING_ACCESS_SCHEMA,
        "consumer": "world_model_training",
        "allowed_files": ["training/fit_tape.pt"],
        "allowed_tensor_fields": list(TRAINING_FEATURE_FIELDS),
        "split_bookkeeping_fields": list(SPLIT_BOOKKEEPING_FIELDS),
        "requires_data_role": "fit",
        "calibration_is_training": False,
        "forbidden_paths": ["audit", "calibration", "evaluation"],
    }


def _concatenate_collection(
    registration: ValidatedRegistration,
    episodes: Sequence[EpisodeCollection],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Build a full mechanical payload; formal publication remains disabled."""
    ordered = sorted(episodes, key=lambda item: item.assignment.episode_id)
    if len(ordered) != EXPECTED_EPISODES:
        raise ContractError("formal publication requires exactly 96 episodes")
    if tuple(item.assignment for item in ordered) != registration.assignments:
        raise ContractError("collected episodes do not match registered assignments")
    for item in ordered:
        actor = registration.actors[item.assignment.collector_id]
        if item.actor_artifact_sha256 != actor.sha256:
            raise ContractError("collected episode actor artifact SHA mismatch")
        if (
            item.actor_runtime_fingerprint_sha256
            != actor.runtime_fingerprint_sha256
        ):
            raise ContractError("collected episode actor runtime fingerprint mismatch")
    episode_payloads = [derive_episode_payloads(registration, item) for item in ordered]
    tapes = [payload[0] for payload in episode_payloads]
    boundaries = [payload[1] for payload in episode_payloads]
    traces = [payload[2] for payload in episode_payloads]
    tensor_fields = ("z", "u", "z_next", "episode", "t", "sigma")

    def concatenate_role(role: str, expected_episodes: int) -> dict[str, Any]:
        selected = [item for item in tapes if item["data_role"] == role]
        if len(selected) != expected_episodes:
            raise ContractError(
                f"{role} tape requires exactly {expected_episodes} episodes"
            )
        combined = {
            name: torch.cat([item[name] for item in selected])
            for name in tensor_fields
        }
        combined.update(
            {
                "schema": DATA_TAPE_SCHEMA,
                "data_role": role,
                "task": TASK,
                "c": C,
                "latent_dim": RICH_DIM,
                "proprio_dim": PROPRIO_DIM,
                "proprio_keys": list(PROPRIO_KEYS),
                "training_feature_fields": list(TRAINING_FEATURE_FIELDS),
                "split_bookkeeping_fields": list(SPLIT_BOOKKEEPING_FIELDS),
                "legacy_outcome_fields": [],
            }
        )
        if set(combined) != set(DATA_TAPE_FIELDS):
            raise ContractError(f"{role} tape field allow-list mismatch")
        return combined

    fit_tape = concatenate_role(
        "fit",
        len(COLLECTOR_IDS) * EXPECTED_FIT_PER_COLLECTOR,
    )
    calibration_tape = concatenate_role(
        "calibration",
        len(COLLECTOR_IDS) * EXPECTED_CAL_PER_COLLECTOR,
    )
    training_tensor_view(fit_tape, ())
    validate_calibration_tape(calibration_tape)
    boundary_payload = {
        "schema": BOUNDARY_SCHEMA,
        "episode_id": torch.tensor([item["episode_id"] for item in boundaries]),
        "z": torch.stack([item["z"] for item in boundaries]),
        "evidence": [item["evidence"] for item in boundaries],
    }
    trace_payload = {
        "schema": TRACE_SCHEMA,
        "episode_id": torch.tensor([item.assignment.episode_id for item in ordered]),
        "collector_id": [item.assignment.collector_id for item in ordered],
        "t": torch.stack([item["t"] for item in traces]),
        "u_base": torch.stack([item["u_base"] for item in traces]),
        "delta": torch.stack([item["delta"] for item in traces]),
        "u_exec": torch.stack([item["u_exec"] for item in traces]),
        "env_action_base": torch.stack(
            [item["env_action_base"] for item in traces]
        ),
        "env_action_exec": torch.stack(
            [item["env_action_exec"] for item in traces]
        ),
        "changed_action_steps": torch.tensor(
            [item["changed_action_steps"] for item in traces]
        ),
        "changed_action_fraction": torch.tensor(
            [item["changed_action_fraction"] for item in traces],
            dtype=torch.float64,
        ),
        "meaningful_action_steps": torch.tensor(
            [item["meaningful_action_steps"] for item in traces]
        ),
        "meaningful_action_fraction": torch.tensor(
            [item["meaningful_action_fraction"] for item in traces],
            dtype=torch.float64,
        ),
        "decoded_action_delta_l2": torch.tensor(
            [item["decoded_action_delta_l2"] for item in traces],
            dtype=torch.float64,
        ),
        "decoded_step_l2_p10": torch.tensor(
            [item["decoded_step_l2_p10"] for item in traces],
            dtype=torch.float64,
        ),
        "decoded_step_l2_median": torch.tensor(
            [item["decoded_step_l2_median"] for item in traces],
            dtype=torch.float64,
        ),
        "decoded_step_l2_p90": torch.tensor(
            [item["decoded_step_l2_p90"] for item in traces],
            dtype=torch.float64,
        ),
        "decoded_step_l2_max": torch.tensor(
            [item["decoded_step_l2_max"] for item in traces],
            dtype=torch.float64,
        ),
        "actor_artifact_sha256": [
            item["actor_artifact_sha256"] for item in traces
        ],
        "actor_runtime_fingerprint_sha256": [
            item["actor_runtime_fingerprint_sha256"] for item in traces
        ],
        "runtime_component_fingerprints_sha256": [
            item["runtime_component_fingerprints_sha256"] for item in traces
        ],
        "min_changed_action_fraction_per_episode": (
            registration.action_change.min_changed_action_fraction_per_episode
        ),
        "min_decoded_action_delta_l2_per_episode": (
            registration.action_change.min_decoded_action_delta_l2_per_episode
        ),
        "meaningful_decoded_step_l2": (
            registration.action_change.meaningful_decoded_step_l2
        ),
        "min_meaningful_action_fraction_per_episode": (
            registration.action_change.min_meaningful_action_fraction_per_episode
        ),
        "min_decoded_step_l2_p10_per_episode": (
            registration.action_change.min_decoded_step_l2_p10_per_episode
        ),
        "max_decoded_step_l2_per_episode": (
            registration.action_change.max_decoded_step_l2_per_episode
        ),
    }
    if not torch.equal(
        trace_payload["u_exec"], trace_payload["u_base"] + trace_payload["delta"]
    ):
        raise ContractError("full action decomposition mismatch")
    _decoded_action_change_metrics(
        trace_payload["env_action_base"].flatten(0, 1),
        trace_payload["env_action_exec"].flatten(0, 1),
        registration.action_change,
    )
    sidecar = {
        "schema": SIDECAR_SCHEMA,
        "registration_sha256": registration.file_sha256,
        "episodes": [item.evaluation_sidecar for item in ordered],
    }
    validate_evaluation_sidecar(registration, sidecar)
    return fit_tape, calibration_tape, boundary_payload, trace_payload, sidecar


def revalidate_registration_unchanged(
    registration: ValidatedRegistration,
) -> ValidatedRegistration:
    """Re-hash the registration, every input, and every registered source."""
    current = validate_registration(registration.path)
    if current.file_sha256 != registration.file_sha256:
        raise ContractError("registration file drifted during collection")
    if current.self_sha256 != registration.self_sha256:
        raise ContractError("registration self digest drifted during collection")
    if current.payload != registration.payload:
        raise ContractError("registration payload drifted during collection")
    if current.raw_bytes != registration.raw_bytes:
        raise ContractError("registration byte snapshot drifted during collection")
    reconstructed_fields = (
        "assignments",
        "actors",
        "d0",
        "action_change",
        "m0",
        "phi",
        "base_vla",
        "tokenizer",
        "runtime_components",
        "attempt_ledger_path",
    )
    for field in reconstructed_fields:
        if getattr(current, field) != getattr(registration, field):
            raise ContractError(
                f"validated registration object field was forged/drifted: {field}"
            )
    return current


def registration_snapshot_bytes(registration: ValidatedRegistration) -> bytes:
    """Return exact revalidated registration bytes for the published snapshot."""
    current = revalidate_registration_unchanged(registration)
    if current.raw_bytes != registration.raw_bytes:
        raise ContractError(
            "registration changed between validation and snapshot capture"
        )
    return current.raw_bytes


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def atomic_commit_directory(output_path: Path, builder: Callable[[Path], None]) -> None:
    """Publish a completed directory by one rename; partial work is non-evidence."""
    unresolved = _unresolved_absolute_path(output_path)
    output_path = unresolved.parent.resolve() / unresolved.name
    if os.path.lexists(output_path):
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.partial-", dir=output_path.parent)
    )
    try:
        builder(stage)
        complete_path = stage / "COMPLETE.json"
        if not complete_path.is_file():
            raise ContractError("atomic builder did not create COMPLETE.json")
        complete = load_json(complete_path)
        if (
            complete.get("schema") != COMPLETE_SCHEMA
            or complete.get("status") != MECHANICAL_COMPLETE_STATUS
            or complete.get("atomic_commit") is not True
            or complete.get("evidence_class") != MECHANICAL_EVIDENCE_CLASS
        ):
            raise ContractError("COMPLETE.json is not a mechanical atomic seal")
        _fsync_directory(stage)
        os.replace(stage, output_path)
        _fsync_directory(output_path.parent)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _require_mechanical_output_namespace(output_path: Path) -> None:
    if not output_path.name.lstrip(".").startswith("mechanical-contract-test-"):
        raise ContractError(
            "mechanical publication path must start mechanical-contract-test-"
        )


def _directory_entry_names(path: Path, label: str) -> set[str]:
    """Return an exact directory closure while rejecting every direct symlink."""
    candidate = _unresolved_absolute_path(path)
    _reject_final_symlink(candidate, label)
    metadata = os.lstat(candidate)
    if not stat.S_ISDIR(metadata.st_mode):
        raise ContractError(f"{label} must be a directory")
    result: set[str] = set()
    for entry in candidate.iterdir():
        entry_metadata = os.lstat(entry)
        if stat.S_ISLNK(entry_metadata.st_mode):
            raise ContractError(f"{label} contains forbidden symlink: {entry.name}")
        result.add(entry.name)
    return result


def publish_collection_contract_test_only(
    registration: ValidatedRegistration,
    episodes: Sequence[EpisodeCollection],
    attempt_ledger: AttemptLedger,
    output_path: Path,
) -> None:
    """Exercise atomic mechanics in a namespace that cannot be formal evidence."""
    _require_mechanical_output_namespace(output_path)
    registration = revalidate_registration_unchanged(registration)
    episode_payload_sha256 = episode_collection_payload_sha256_by_id(
        registration,
        episodes,
    )
    attempt_ledger_bytes = validate_attempt_ledger_for_publication(
        attempt_ledger,
        registration,
        expected_episode_payload_sha256=episode_payload_sha256,
    )
    fit_tape, calibration_tape, boundaries, trace, sidecar = (
        _concatenate_collection(registration, episodes)
    )

    def build(stage: Path) -> None:
        training_dir = stage / "training"
        calibration_dir = stage / "calibration"
        audit_dir = stage / "audit"
        evaluation_dir = stage / "evaluation"
        training_dir.mkdir()
        calibration_dir.mkdir()
        audit_dir.mkdir()
        evaluation_dir.mkdir()
        fit_tape_path = training_dir / "fit_tape.pt"
        calibration_tape_path = calibration_dir / "calibration_tape.pt"
        boundary_path = audit_dir / "boundaries.pt"
        trace_path = audit_dir / "action_trace.pt"
        attempt_path = audit_dir / "attempt_ledger.jsonl"
        sidecar_path = evaluation_dir / "evaluation_sidecar.json"
        access_path = stage / "ACCESS_POLICY.json"
        producer_path = stage / "PRODUCER.py"
        registration_path = stage / "REGISTRATION.json"
        _save_torch(fit_tape_path, fit_tape)
        _save_torch(calibration_tape_path, calibration_tape)
        _save_torch(boundary_path, boundaries)
        _save_torch(trace_path, trace)
        _write_bytes(attempt_path, attempt_ledger_bytes)
        _write_json(sidecar_path, sidecar)
        _write_json(access_path, training_access_policy())
        _write_bytes(
            producer_path,
            read_regular_file_snapshot(Path(__file__), "collector source"),
        )
        registered_producer_sha = registration.payload["provenance"][
            "source_files_sha256"
        ]["scripts/collect_v246_evolve1_d1.py"]
        if sha256_file(producer_path) != registered_producer_sha:
            raise ContractError("PRODUCER.py differs from registered collector source")
        snapshot_bytes = registration_snapshot_bytes(registration)
        _write_bytes(registration_path, snapshot_bytes)
        if sha256_file(registration_path) != registration.file_sha256:
            raise ContractError(
                "REGISTRATION.json snapshot SHA differs from registration.file_sha256"
            )
        reloaded_fit_tape = load_torch(fit_tape_path, "staged fit tape")
        reloaded_calibration_tape = load_torch(
            calibration_tape_path,
            "staged calibration tape",
        )
        if (
            not isinstance(reloaded_fit_tape, dict)
            or set(reloaded_fit_tape) != set(DATA_TAPE_FIELDS)
            or not isinstance(reloaded_calibration_tape, dict)
            or set(reloaded_calibration_tape) != set(DATA_TAPE_FIELDS)
        ):
            raise ContractError("round-tripped role tape field allow-list mismatch")
        training_tensor_view(reloaded_fit_tape)
        validate_calibration_tape(reloaded_calibration_tape)
        for name in ("z", "u", "z_next", "episode", "t", "sigma"):
            if tensor_sha256(reloaded_fit_tape[name]) != tensor_sha256(fit_tape[name]):
                raise ContractError(f"round-tripped fit tensor changed: {name}")
            if tensor_sha256(reloaded_calibration_tape[name]) != tensor_sha256(
                calibration_tape[name]
            ):
                raise ContractError(f"round-tripped calibration tensor changed: {name}")
        artifact_paths = (
            fit_tape_path,
            calibration_tape_path,
            boundary_path,
            trace_path,
            attempt_path,
            sidecar_path,
            access_path,
            producer_path,
            registration_path,
        )
        file_hashes = {
            str(path.relative_to(stage)): sha256_file(path)
            for path in artifact_paths
        }
        result = {
            "schema": RESULT_SCHEMA,
            "status": MECHANICAL_RESULT_STATUS,
            "evidence_class": MECHANICAL_EVIDENCE_CLASS,
            "formal_gate1_evidence": False,
            "registration_sha256": registration.file_sha256,
            "episodes": EXPECTED_EPISODES,
            "episodes_per_collector": EXPECTED_PER_COLLECTOR,
            "transitions": EXPECTED_EPISODES * CHUNKS_PER_EPISODE,
            "boundaries": EXPECTED_EPISODES * BOUNDARIES_PER_EPISODE,
            "env_steps": EXPECTED_EPISODES * HORIZON_STEPS,
            "task": TASK,
            "c": C,
            "latent_dim": RICH_DIM,
            "outcome_values_in_result": False,
            "file_sha256": file_hashes,
        }
        result_path = stage / "RESULT.json"
        _write_json(result_path, result)
        revalidate_registration_unchanged(registration)
        fresh_attempt_bytes = validate_attempt_ledger_for_publication(
            attempt_ledger,
            registration,
            expected_episode_payload_sha256=episode_payload_sha256,
        )
        if fresh_attempt_bytes != attempt_ledger_bytes:
            raise ContractError("attempt ledger drifted during publication")
        if sha256_file(attempt_path) != hashlib.sha256(attempt_ledger_bytes).hexdigest():
            raise ContractError("staged attempt ledger snapshot changed")
        if sha256_file(registration_path) != registration.file_sha256:
            raise ContractError("REGISTRATION.json changed before final seal")
        complete = {
            "schema": COMPLETE_SCHEMA,
            "status": MECHANICAL_COMPLETE_STATUS,
            "evidence_class": MECHANICAL_EVIDENCE_CLASS,
            "formal_gate1_evidence": False,
            "atomic_commit": True,
            "registration_sha256": registration.file_sha256,
            "result_sha256": sha256_file(result_path),
            "file_sha256": file_hashes,
        }
        _write_json(stage / "COMPLETE.json", complete)
        for directory in (
            training_dir,
            calibration_dir,
            audit_dir,
            evaluation_dir,
        ):
            _fsync_directory(directory)
        _fsync_directory(stage)
        validate_published_collection_contract_test_only(stage, registration)

    atomic_commit_directory(output_path, build)
    validate_published_collection_contract_test_only(output_path, registration)


def validate_published_collection_contract_test_only(
    output_path: Path,
    expected_registration: ValidatedRegistration,
) -> dict[str, Any]:
    """Reload every mechanical artifact; this can never establish Gate-1."""
    _require_mechanical_output_namespace(output_path)
    expected_registration = revalidate_registration_unchanged(expected_registration)
    unresolved_root = _unresolved_absolute_path(output_path)
    _reject_final_symlink(unresolved_root, "published collection root")
    root = unresolved_root.parent.resolve() / unresolved_root.name
    expected_entries = {
        "training",
        "calibration",
        "audit",
        "evaluation",
        "ACCESS_POLICY.json",
        "PRODUCER.py",
        "REGISTRATION.json",
        "RESULT.json",
        "COMPLETE.json",
    }
    if _directory_entry_names(root, "published collection root") != expected_entries:
        raise ContractError("published D1 directory topology mismatch")
    expected_nested = {
        "training": {"fit_tape.pt"},
        "calibration": {"calibration_tape.pt"},
        "audit": {"boundaries.pt", "action_trace.pt", "attempt_ledger.jsonl"},
        "evaluation": {"evaluation_sidecar.json"},
    }
    for directory, names in expected_nested.items():
        path = root / directory
        if _directory_entry_names(path, f"published D1 {directory}") != names:
            raise ContractError(f"published D1 {directory} topology mismatch")

    complete_path = root / "COMPLETE.json"
    complete_snapshot = read_regular_file_snapshot(complete_path, "published COMPLETE")
    complete_snapshot_sha256 = hashlib.sha256(complete_snapshot).hexdigest()
    complete = load_json_bytes(complete_snapshot, "published COMPLETE")
    complete_fields = {
        "schema",
        "status",
        "evidence_class",
        "formal_gate1_evidence",
        "atomic_commit",
        "registration_sha256",
        "result_sha256",
        "file_sha256",
    }
    if set(complete) != complete_fields:
        raise ContractError("COMPLETE field closure mismatch; shallow seal rejected")
    if (
        complete.get("schema") != COMPLETE_SCHEMA
        or complete.get("status") != MECHANICAL_COMPLETE_STATUS
        or complete.get("evidence_class") != MECHANICAL_EVIDENCE_CLASS
        or complete.get("formal_gate1_evidence") is not False
        or complete.get("atomic_commit") is not True
        or complete.get("registration_sha256")
        != expected_registration.file_sha256
    ):
        raise ContractError("COMPLETE identity/status mismatch")
    result_path = root / "RESULT.json"
    result_snapshot = read_regular_file_snapshot(result_path, "published RESULT")
    if complete.get("result_sha256") != hashlib.sha256(result_snapshot).hexdigest():
        raise ContractError("COMPLETE result SHA mismatch")
    expected_file_names = {
        "training/fit_tape.pt",
        "calibration/calibration_tape.pt",
        "audit/boundaries.pt",
        "audit/action_trace.pt",
        "audit/attempt_ledger.jsonl",
        "evaluation/evaluation_sidecar.json",
        "ACCESS_POLICY.json",
        "PRODUCER.py",
        "REGISTRATION.json",
    }
    file_hashes = complete.get("file_sha256")
    if not isinstance(file_hashes, Mapping) or set(file_hashes) != expected_file_names:
        raise ContractError("COMPLETE artifact hash closure mismatch")
    artifact_snapshots: dict[str, bytes] = {}
    for relative, expected_sha in file_hashes.items():
        artifact = root / relative
        if not is_sha256(expected_sha):
            raise ContractError(f"published artifact reference invalid: {relative}")
        snapshot = read_regular_file_snapshot(artifact, f"published {relative}")
        artifact_snapshots[relative] = snapshot
        if hashlib.sha256(snapshot).hexdigest() != expected_sha:
            raise ContractError(f"published artifact SHA mismatch: {relative}")

    registration_snapshot = artifact_snapshots["REGISTRATION.json"]
    if (
        hashlib.sha256(registration_snapshot).hexdigest()
        != expected_registration.file_sha256
    ):
        raise ContractError("published registration snapshot SHA mismatch")
    revalidate_registration_unchanged(expected_registration)
    if (
        load_json_bytes(registration_snapshot, "published registration")
        != expected_registration.payload
    ):
        raise ContractError("published registration snapshot payload mismatch")
    producer_sha = expected_registration.payload["provenance"][
        "source_files_sha256"
    ]["scripts/collect_v246_evolve1_d1.py"]
    if hashlib.sha256(artifact_snapshots["PRODUCER.py"]).hexdigest() != producer_sha:
        raise ContractError("published producer source mismatch")
    if not exact_json_equal(
        load_json_bytes(
            artifact_snapshots["ACCESS_POLICY.json"],
            "published access policy",
        ),
        training_access_policy(),
    ):
        raise ContractError("published training access policy mismatch")

    result = load_json_bytes(result_snapshot, "published RESULT")
    result_fields = {
        "schema",
        "status",
        "evidence_class",
        "formal_gate1_evidence",
        "registration_sha256",
        "episodes",
        "episodes_per_collector",
        "transitions",
        "boundaries",
        "env_steps",
        "task",
        "c",
        "latent_dim",
        "outcome_values_in_result",
        "file_sha256",
    }
    expected_result = {
        "schema": RESULT_SCHEMA,
        "status": MECHANICAL_RESULT_STATUS,
        "evidence_class": MECHANICAL_EVIDENCE_CLASS,
        "formal_gate1_evidence": False,
        "registration_sha256": expected_registration.file_sha256,
        "episodes": EXPECTED_EPISODES,
        "episodes_per_collector": EXPECTED_PER_COLLECTOR,
        "transitions": EXPECTED_EPISODES * CHUNKS_PER_EPISODE,
        "boundaries": EXPECTED_EPISODES * BOUNDARIES_PER_EPISODE,
        "env_steps": EXPECTED_EPISODES * HORIZON_STEPS,
        "task": TASK,
        "c": C,
        "latent_dim": RICH_DIM,
        "outcome_values_in_result": False,
        "file_sha256": dict(file_hashes),
    }
    if set(result) != result_fields or not exact_json_equal(result, expected_result):
        raise ContractError("published RESULT field/value contract mismatch")

    fit_tape = load_torch_bytes(
        artifact_snapshots["training/fit_tape.pt"],
        "published fit tape",
    )
    calibration_tape = load_torch_bytes(
        artifact_snapshots["calibration/calibration_tape.pt"],
        "published calibration tape",
    )
    training_tensor_view(fit_tape, ())
    validate_calibration_tape(calibration_tape)
    role_tapes = {"fit": fit_tape, "calibration": calibration_tape}
    expected_role_episode_ids: dict[str, list[int]] = {
        role: [
            assignment.episode_id
            for assignment in expected_registration.assignments
            if assignment.split == role
        ]
        for role in SPLITS
    }
    for role, role_tape in role_tapes.items():
        expected_ids = torch.tensor(
            expected_role_episode_ids[role],
            dtype=torch.int64,
        ).repeat_interleave(CHUNKS_PER_EPISODE)
        expected_times = torch.arange(CHUNKS_PER_EPISODE).repeat(
            len(expected_role_episode_ids[role])
        ) * C
        if not torch.equal(role_tape["episode"], expected_ids):
            raise ContractError(f"published {role} episode bookkeeping mismatch")
        if not torch.equal(role_tape["t"], expected_times):
            raise ContractError(f"published {role} t bookkeeping mismatch")

    tape_rows_by_episode: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for role_tape in role_tapes.values():
        for episode_id in torch.unique_consecutive(role_tape["episode"]).tolist():
            selected = role_tape["episode"] == episode_id
            tape_rows_by_episode[int(episode_id)] = (
                role_tape["z"][selected],
                role_tape["z_next"][selected],
                role_tape["u"][selected],
            )
    if set(tape_rows_by_episode) != set(range(EXPECTED_EPISODES)):
        raise ContractError("fit/calibration tapes do not partition episodes 0..95")

    boundaries = load_torch_bytes(
        artifact_snapshots["audit/boundaries.pt"],
        "published boundaries",
    )
    boundary_fields = {"schema", "episode_id", "z", "evidence"}
    if not isinstance(boundaries, Mapping) or set(boundaries) != boundary_fields:
        raise ContractError("published boundary field closure mismatch")
    boundary_episode_id = boundaries.get("episode_id")
    boundary_z = boundaries.get("z")
    if (
        boundaries.get("schema") != BOUNDARY_SCHEMA
        or not torch.is_tensor(boundary_episode_id)
        or boundary_episode_id.dtype != torch.int64
        or tuple(boundary_episode_id.shape) != (EXPECTED_EPISODES,)
        or not torch.is_tensor(boundary_z)
        or boundary_z.dtype != torch.float32
        or tuple(boundary_z.shape)
        != (EXPECTED_EPISODES, BOUNDARIES_PER_EPISODE, RICH_DIM)
        or not bool(torch.isfinite(boundary_z).all())
        or not isinstance(boundaries["evidence"], list)
        or len(boundaries["evidence"]) != EXPECTED_EPISODES
    ):
        raise ContractError("published boundary shape/schema mismatch")
    if not torch.equal(boundary_episode_id.cpu(), torch.arange(96)):
        raise ContractError("published boundary episode IDs mismatch")
    for episode_id in range(EXPECTED_EPISODES):
        tape_z, tape_z_next, _tape_u = tape_rows_by_episode[episode_id]
        if not torch.equal(boundary_z[episode_id, :-1], tape_z):
            raise ContractError("published boundary/tape z mismatch")
        if not torch.equal(boundary_z[episode_id, 1:], tape_z_next):
            raise ContractError("published boundary/tape z_next mismatch")
    encoder_fp = expected_registration.runtime_components[
        "feature_encoder"
    ].runtime_fingerprint_sha256
    for episode_id, evidence_rows in enumerate(boundaries["evidence"]):
        if not isinstance(evidence_rows, list) or len(evidence_rows) != BOUNDARIES_PER_EPISODE:
            raise ContractError("published boundary evidence shape mismatch")
        for boundary_id, evidence in enumerate(evidence_rows):
            _rich_encoding(
                RichEncoding(boundary_z[episode_id, boundary_id], evidence),
                f"published boundary {episode_id}/{boundary_id}",
                encoder_fp,
            )

    trace = load_torch_bytes(
        artifact_snapshots["audit/action_trace.pt"],
        "published action trace",
    )
    trace_fields = {
        "schema",
        "episode_id",
        "collector_id",
        "t",
        "u_base",
        "delta",
        "u_exec",
        "env_action_base",
        "env_action_exec",
        "changed_action_steps",
        "changed_action_fraction",
        "meaningful_action_steps",
        "meaningful_action_fraction",
        "decoded_action_delta_l2",
        "decoded_step_l2_p10",
        "decoded_step_l2_median",
        "decoded_step_l2_p90",
        "decoded_step_l2_max",
        "actor_artifact_sha256",
        "actor_runtime_fingerprint_sha256",
        "runtime_component_fingerprints_sha256",
        "min_changed_action_fraction_per_episode",
        "min_decoded_action_delta_l2_per_episode",
        "meaningful_decoded_step_l2",
        "min_meaningful_action_fraction_per_episode",
        "min_decoded_step_l2_p10_per_episode",
        "max_decoded_step_l2_per_episode",
    }
    if not isinstance(trace, Mapping) or set(trace) != trace_fields:
        raise ContractError("published action trace field closure mismatch")
    expected_action_shape = (
        EXPECTED_EPISODES,
        CHUNKS_PER_EPISODE,
        C,
        ACTION_DIM,
    )
    for name in ("u_base", "delta", "u_exec", "env_action_base", "env_action_exec"):
        value = trace.get(name)
        if (
            not torch.is_tensor(value)
            or value.dtype != torch.float32
            or tuple(value.shape) != expected_action_shape
            or not bool(torch.isfinite(value).all())
        ):
            raise ContractError(
                f"published action trace {name} must be finite float32 "
                f"{expected_action_shape}"
            )
    expected_trace_t_shape = (EXPECTED_EPISODES, CHUNKS_PER_EPISODE)
    tensor_contracts = {
        "episode_id": ((EXPECTED_EPISODES,), torch.int64),
        "t": (expected_trace_t_shape, torch.int64),
        "changed_action_steps": ((EXPECTED_EPISODES,), torch.int64),
        "changed_action_fraction": ((EXPECTED_EPISODES,), torch.float64),
        "meaningful_action_steps": ((EXPECTED_EPISODES,), torch.int64),
        "meaningful_action_fraction": ((EXPECTED_EPISODES,), torch.float64),
        "decoded_action_delta_l2": ((EXPECTED_EPISODES,), torch.float64),
        "decoded_step_l2_p10": ((EXPECTED_EPISODES,), torch.float64),
        "decoded_step_l2_median": ((EXPECTED_EPISODES,), torch.float64),
        "decoded_step_l2_p90": ((EXPECTED_EPISODES,), torch.float64),
        "decoded_step_l2_max": ((EXPECTED_EPISODES,), torch.float64),
    }
    for name, (shape, dtype) in tensor_contracts.items():
        value = trace.get(name)
        if (
            not torch.is_tensor(value)
            or value.dtype != dtype
            or tuple(value.shape) != shape
            or not bool(torch.isfinite(value).all())
        ):
            raise ContractError(
                f"published action trace {name} must be finite {dtype} {shape}"
            )
    for name in (
        "collector_id",
        "actor_artifact_sha256",
        "actor_runtime_fingerprint_sha256",
        "runtime_component_fingerprints_sha256",
    ):
        if not isinstance(trace.get(name), list) or len(trace[name]) != EXPECTED_EPISODES:
            raise ContractError(f"published action trace {name} aggregate mismatch")
    if (
        trace.get("schema") != TRACE_SCHEMA
        or not torch.equal(trace["episode_id"].cpu().long(), torch.arange(96))
        or trace.get("collector_id")
        != [assignment.collector_id for assignment in expected_registration.assignments]
        or not torch.equal(
            trace["t"].cpu().long(),
            torch.arange(CHUNKS_PER_EPISODE).repeat(EXPECTED_EPISODES, 1) * C,
        )
        or type(trace.get("min_changed_action_fraction_per_episode")) is not float
        or trace.get("min_changed_action_fraction_per_episode")
        != expected_registration.action_change.min_changed_action_fraction_per_episode
        or type(trace.get("min_decoded_action_delta_l2_per_episode")) is not float
        or trace.get("min_decoded_action_delta_l2_per_episode")
        != expected_registration.action_change.min_decoded_action_delta_l2_per_episode
        or trace.get("meaningful_decoded_step_l2")
        != expected_registration.action_change.meaningful_decoded_step_l2
        or trace.get("min_meaningful_action_fraction_per_episode")
        != expected_registration.action_change.min_meaningful_action_fraction_per_episode
        or trace.get("min_decoded_step_l2_p10_per_episode")
        != expected_registration.action_change.min_decoded_step_l2_p10_per_episode
        or trace.get("max_decoded_step_l2_per_episode")
        != expected_registration.action_change.max_decoded_step_l2_per_episode
    ):
        raise ContractError("published action trace identity/threshold mismatch")
    if not torch.equal(trace["u_exec"], trace["u_base"] + trace["delta"]):
        raise ContractError("published action decomposition mismatch")
    for episode_id in range(EXPECTED_EPISODES):
        if not torch.equal(
            tape_rows_by_episode[episode_id][2],
            trace["u_exec"][episode_id],
        ):
            raise ContractError(
                "published role tape/action-trace executed actions mismatch"
            )
    for episode_id in range(EXPECTED_EPISODES):
        action_metrics = _decoded_action_change_metrics(
            trace["env_action_base"][episode_id],
            trace["env_action_exec"][episode_id],
            expected_registration.action_change,
        )
        if (
            int(trace["changed_action_steps"][episode_id])
            != action_metrics.changed_action_steps
        ):
            raise ContractError("published changed-action count mismatch")
        if float(trace["changed_action_fraction"][episode_id]) != (
            action_metrics.changed_action_steps / HORIZON_STEPS
        ):
            raise ContractError("published changed-action fraction mismatch")
        if (
            int(trace["meaningful_action_steps"][episode_id])
            != action_metrics.meaningful_action_steps
            or float(trace["meaningful_action_fraction"][episode_id])
            != action_metrics.meaningful_action_steps / HORIZON_STEPS
        ):
            raise ContractError("published meaningful-action coverage mismatch")
        if (
            float(trace["decoded_action_delta_l2"][episode_id])
            != action_metrics.decoded_action_delta_l2
        ):
            raise ContractError("published decoded-action norm mismatch")
        for name in (
            "decoded_step_l2_p10",
            "decoded_step_l2_median",
            "decoded_step_l2_p90",
            "decoded_step_l2_max",
        ):
            if float(trace[name][episode_id]) != getattr(action_metrics, name):
                raise ContractError(f"published {name} mismatch")
        assignment = expected_registration.assignments[episode_id]
        actor = expected_registration.actors[assignment.collector_id]
        if float(trace["delta"][episode_id].abs().max().item()) > actor.scale:
            raise ContractError("published normalized residual exceeds actor scale")
        if (
            trace["actor_artifact_sha256"][episode_id] != actor.sha256
            or trace["actor_runtime_fingerprint_sha256"][episode_id]
            != actor.runtime_fingerprint_sha256
        ):
            raise ContractError("published trace actor identity mismatch")
        expected_runtime = {
            "actor": actor.runtime_fingerprint_sha256,
            **{
                name: expected_registration.runtime_components[
                    name
                ].runtime_fingerprint_sha256
                for name in REGISTERED_RUNTIME_COMPONENTS
            },
        }
        if trace["runtime_component_fingerprints_sha256"][episode_id] != expected_runtime:
            raise ContractError("published trace runtime fingerprint mismatch")

    sidecar = load_json_bytes(
        artifact_snapshots["evaluation/evaluation_sidecar.json"],
        "published evaluation sidecar",
    )
    validate_evaluation_sidecar(expected_registration, sidecar)

    published_episode_payload_sha256 = {
        episode_id: _episode_payload_sha256(
            expected_registration.file_sha256,
            expected_registration.assignments[episode_id],
            boundary_z[episode_id],
            boundaries["evidence"][episode_id],
            trace["u_base"][episode_id],
            trace["delta"][episode_id],
            trace["u_exec"][episode_id],
            trace["env_action_base"][episode_id],
            trace["env_action_exec"][episode_id],
            int(trace["changed_action_steps"][episode_id]),
            int(trace["meaningful_action_steps"][episode_id]),
            float(trace["decoded_action_delta_l2"][episode_id]),
            float(trace["decoded_step_l2_p10"][episode_id]),
            float(trace["decoded_step_l2_median"][episode_id]),
            float(trace["decoded_step_l2_p90"][episode_id]),
            float(trace["decoded_step_l2_max"][episode_id]),
            trace["actor_artifact_sha256"][episode_id],
            trace["actor_runtime_fingerprint_sha256"][episode_id],
            trace["runtime_component_fingerprints_sha256"][episode_id],
            sidecar["episodes"][episode_id],
        )
        for episode_id in range(EXPECTED_EPISODES)
    }

    parse_attempt_ledger_bytes(
        artifact_snapshots["audit/attempt_ledger.jsonl"],
        expected_registration.file_sha256,
    )
    published_ledger = AttemptLedger.open_snapshot_contract_test_only(
        root / "audit" / "attempt_ledger.jsonl",
        expected_registration.file_sha256,
        formal_path_bound=True,
    )
    validate_attempt_ledger_for_publication(
        published_ledger,
        expected_registration,
        require_registered_path=False,
        expected_episode_payload_sha256=published_episode_payload_sha256,
    )
    if _directory_entry_names(root, "published collection root") != expected_entries:
        raise ContractError("published D1 topology drifted during validation")
    for relative, expected_sha in file_hashes.items():
        if sha256_file(root / relative) != expected_sha:
            raise ContractError(
                f"published artifact drifted during validation: {relative}"
            )
    if (
        sha256_file(result_path) != complete["result_sha256"]
        or sha256_file(complete_path) != complete_snapshot_sha256
    ):
        raise ContractError("published seal drifted during validation")
    revalidate_registration_unchanged(expected_registration)
    return result


def publish_collection(
    registration: ValidatedRegistration,
    episodes: Sequence[EpisodeCollection],
    attempt_ledger: AttemptLedger,
    output_path: Path,
) -> None:
    """Formal entrypoint; rejects before touching ledger or output path."""
    del registration, episodes, attempt_ledger, output_path
    require_formal_collection_ready("formal D1 publication")
    raise RealCollectorNotImplemented("NO-GO: formal D1 publisher is not implemented")


def validate_published_collection(
    output_path: Path,
    expected_registration: ValidatedRegistration,
) -> dict[str, Any]:
    """Formal entrypoint; no mechanical artifact can be promoted through it."""
    del output_path, expected_registration
    require_formal_collection_ready("formal D1 published-artifact validation")
    raise RealCollectorNotImplemented(
        "NO-GO: formal D1 published-artifact validator is not implemented"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--collect", action="store_true")
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registration = validate_registration(args.registration)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "MECHANICAL_CONTRACT_VALID_REAL_BINDING_NO_GO",
                    "registration_sha256": registration.file_sha256,
                    "d0_tape_sha256": registration.d0.sha256,
                    "actor_sha256": {
                        key: registration.actors[key].sha256 for key in COLLECTOR_IDS
                    },
                    "assignments": len(registration.assignments),
                    "environment_modules_imported": environment_modules_loaded(),
                    "environment_reset": False,
                    "output_created": False,
                    "real_semantic_binding_implemented": (
                        REAL_SEMANTIC_BINDING_IMPLEMENTED
                    ),
                },
                indent=2,
            )
        )
        return 0
    if args.out is None:
        raise ContractError("--collect requires --out")
    require_formal_collection_ready("real D1 collection")
    raise AssertionError("unreachable until the real D1 runner is implemented")


if __name__ == "__main__":
    raise SystemExit(main())
