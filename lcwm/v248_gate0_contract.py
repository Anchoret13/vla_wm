# RETIRED 2026-09-16. Nothing supersedes it: the gate it serves was removed as a
# prerequisite at plan_and_progress/2026-09-12.md:105, and the round used a supplied
# Phi' annotation instead. It has produced zero artifacts - none of
# results/v248_gate0_{registration_bundle,materialized_models,formal_campaign}_v1 exists.
# Blockers never cleared: WORM ledger, runtime fingerprint, registration bundle, and
# non-deterministic cuDNN/TF32 (plan_and_progress/2026-09-02.md:108-129).
# Do not extend, do not import, do not cite as capability. Recoverable at d35e2832.
"""Shared, simulator-free contract for the v248 Gate-0 formal RGB panel.

The registration is frozen before either formal panel is rendered.  This
module is intentionally limited to standard-library metadata inspection plus
lazy NumPy/Torch imports.  In particular it must never import LIBERO,
robosuite, LeRobot, the VLA chassis, or the environment factory.  Both the RGB
replay producer and the semantic producer validate the same exact JSON shape
through :func:`load_and_validate_registration`.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import stat
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parent.parent

REGISTRATION_SCHEMA = "v248_gate0_pre_registration_v2"
REGISTRATION_STATUS = "frozen_before_any_formal_rgb"
CAMPAIGN_SCHEMA = "v248_gate0_campaign_protocol_v1"
CAMPAIGN_MANIFEST_SCHEMA = "v248_gate0_campaign_manifest_v1"
CAMPAIGN_PROJECTION_SCHEMA = "v248_gate0_campaign_projection_v1"
CAMPAIGN_INTENT_SCHEMA = "v248_gate0_campaign_intent_v1"
FREEZE_COMPLETE_SCHEMA = "v248_gate0_registration_freeze_complete_v1"
RGB_RESULT_SCHEMA = "v248_gate0_rgb_replay_result_v1"
RGB_MANIFEST_SCHEMA = "v248_gate0_rgb_manifest_frame_v1"
RGB_COMPLETE_SCHEMA = "v248_gate0_rgb_replay_complete_v1"

TASK = "chain3_lr2"
C = 10
LATENT_LAYOUT = "legacy2073"
LATENT_DIM = 2073
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
FRAME_ROLES = ("pre_action", "n_plus_1", "recovered_endpoint")

TOP_FIELDS = frozenset(
    {
        "schema",
        "status",
        "registration_id",
        "created_utc",
        "campaign",
        "phase",
        "panel",
        "replay",
        "terminal_recovery",
        "sanitized_source",
        "oracle",
        "models",
        "templates",
        "sources",
        "runtime",
        "semantic_contract",
        "self_sha256",
    }
)
CAMPAIGN_FIELDS = frozenset(
    {
        "schema",
        "campaign_id",
        "protocol_sha256",
        "projection_sha256",
        "manifest_relative_path",
        "registration_slot",
    }
)
PANEL_FIELDS = frozenset(
    {"panel_id", "seed_start", "source_episode_count", "episodes"}
)
EPISODE_FIELDS = frozenset({"episode_id", "env_seed", "stored_rows", "mode"})
REPLAY_FIELDS = frozenset(
    {
        "task",
        "c",
        "latent_layout",
        "latent_dim",
        "proprio_dim",
        "proprio_keys",
        "cameras",
        "jpeg_quality",
        "orientation",
        "frame_roles",
        "two_fresh_recovery_replays",
    }
)
RECOVERY_FIELDS = frozenset(
    {
        "enabled",
        "episode_id",
        "env_seed",
        "stored_chunks",
        "stored_steps",
        "source_terminal_t",
        "missing_chunk_t",
        "missing_raw_sigma_choice",
        "sigma_rng",
        "sampler",
        "phi_only",
        "dynamics_transition_eligible",
    }
)
SIGMA_RNG_FIELDS = frozenset(
    {
        "algorithm",
        "seed",
        "choices",
        "source_proposed_chunks_by_episode",
        "missing_draw_offset_within_episode",
    }
)
SAMPLER_FIELDS = frozenset(
    {
        "zero_sigma_path",
        "positive_sigma_path",
        "seed_rule",
        "n",
        "normalized_dtype",
        "exact_comparator",
        "prerequisite_chunks",
        "missing_chunk_actions",
        "required_halt_action_offset",
        "required_halt_t",
    }
)
SANITIZED_FIELDS = frozenset(
    {
        "tape_sha256",
        "result_sha256",
        "complete_sha256",
        "producer_sha256",
        "source_tape_sha256",
    }
)
ORACLE_FIELDS = frozenset(
    {"summary", "access", "fields", "success_values_accessed"}
)
MODEL_FIELDS = frozenset({"pi05", "tokenizer", "sam3"})
PI05_FIELDS = frozenset(
    {
        "repo_id",
        "revision",
        "local_root",
        "checkpoint",
        "config",
        "preprocessor",
        "normalizer",
        "postprocessor",
        "unnormalizer",
    }
)
TOKENIZER_FIELDS = frozenset({"repo_id", "revision", "local_root", "files"})
SAM3_FIELDS = frozenset(
    {
        "model_id",
        "revision",
        "checkpoint",
        "clip_tokenizer",
        "config_sha256",
        "processor_sha256",
    }
)
TREE_FIELDS = frozenset({"root", "tree_sha256", "files"})
SNAPSHOT_MEMBER_FIELDS = frozenset({"relative_path", "sha256", "bytes"})
EXTERNAL_FILE_FIELDS = frozenset({"path", "sha256", "bytes"})
REPO_FILE_FIELDS = frozenset({"relative_path", "sha256", "bytes"})
TEMPLATE_FIELDS = frozenset(
    {"cream_template_image", "cream_template_metadata", "semantic_config"}
)
SOURCE_ROLES = frozenset(
    {
        "contract",
        "formal_replay",
        "formal_replay_test",
        "collector",
        "sanitizer",
        "chassis",
        "sampler",
        "latent_pooler",
        "env_factory",
        "chain_env",
        "bddl",
        "semantic_producer",
        "semantic_evaluator",
        "confirm_authorizer",
        "sam3_engine",
        "sift_engine",
        "cream_state",
        "can_state",
    }
)
KNOWN_SOURCE_PATHS = {
    "contract": "lcwm/v248_gate0_contract.py",
    "formal_replay": "scripts/replay_v248_gate0_rgb.py",
    "formal_replay_test": "scripts/test_v248_gate0_rgb.py",
    "collector": "scripts/collect_v121_deploy_latents.py",
    "sanitizer": "scripts/sanitize_v245_deploy_tape.py",
    "chassis": "lcwm/chassis.py",
    "sampler": "lcwm/sampler.py",
    "latent_pooler": "lcwm/v082_m0.py",
    "env_factory": "lcwm/v080_bench.py",
    "chain_env": "lcwm/loho.py",
    "bddl": "bddl/chains/chain3_lr2.bddl",
    "sam3_engine": "scripts/probe_v245_sam3_video_progress.py",
    "sift_engine": "scripts/probe_v247_multiview_cream.py",
    "cream_state": "lcwm/v247_cream_state.py",
    "can_state": "lcwm/v247_can_state.py",
    "confirm_authorizer": "scripts/authorize_v248_gate0_confirm.py",
}
RUNTIME_FIELDS = frozenset(
    {
        "python",
        "platform",
        "torch",
        "numpy",
        "pillow",
        "opencv",
        "transformers",
        "lerobot",
        "libero",
        "robosuite",
        "cuda",
        "device",
        "runtime_sources",
    }
)
RUNTIME_SOURCE_ROLES = frozenset(
    {
        "lerobot_policy_config",
        "lerobot_policy_factory",
        "lerobot_processor",
        "lerobot_env_factory",
        "lerobot_libero_env",
        "pi05_configuration",
        "pi05_modeling",
        "libero_bddl_env",
        "robosuite_environment",
        "transformers_sam3_video_configuration",
        "transformers_sam3_video_modeling",
        "transformers_sam3_video_processing",
        "transformers_sam3_image_processing",
        "transformers_sam2_video_processing",
        "transformers_clip_tokenization",
    }
)
SEMANTIC_FIELDS = frozenset(
    {
        "manifest_schema",
        "result_schema",
        "complete_schema",
        "allowed_frame_roles",
        "recovered_endpoint",
    }
)
RECOVERED_SEMANTIC_FIELDS = frozenset(
    {"phi_only", "dynamics_transition_eligible"}
)
FREEZE_COMPLETE_FIELDS = frozenset(
    {
        "schema",
        "status",
        "atomic_commit",
        "input_closure_sha256",
        "materialization_complete_sha256",
        "campaign_manifest_sha256",
        "campaign_body_sha256",
        "campaign_id",
        "projection_sha256",
        "result_sha256",
        "producer_sha256",
        "registration_sha256s",
    }
)
CAMPAIGN_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "status",
        "created_utc",
        "campaign_id",
        "protocol_sha256",
        "projection_sha256",
        "projection",
        "screen_episode_ids",
        "confirm_episode_ids",
        "automatic_expansion_rule",
        "confirm_authorization_required",
        "registrations",
        "body_sha256",
    }
)
CAMPAIGN_PROJECTION_FIELDS = frozenset({"schema", "intent", "campaign_id"})
CAMPAIGN_INTENT_FIELDS = frozenset(
    {
        "schema",
        "input_closure_sha256",
        "protocol_sha256",
        "registration_core_sha256s",
        "screen_episode_ids",
        "confirm_episode_ids",
        "automatic_expansion_rule",
        "confirm_authorization_required",
        "target_root",
        "target_output_slots",
        "dag",
    }
)
CAMPAIGN_REGISTRATION_FIELDS = frozenset(
    {
        "file_sha256",
        "self_sha256",
        "core_sha256",
        "panel_id",
        "seed_start",
        "phase",
    }
)

SCREEN_EPISODES = {
    8000: (6, 12, 21, 24, 28, 29, 35, 43, 57, 82, 85, 94),
    8100: (23, 24, 37, 43, 48, 62, 63, 68, 77, 80, 93, 94),
}
PANEL_IDS = {8000: "chain3_8000", 8100: "chain3_8100"}

AUTOMATIC_CONFIRM_EXPANSION_RULE = (
    "if_and_only_if_combined_two_panel_screen_evaluation_all_frozen_metric_checks_"
    "true_then_expand_each_unchanged_panel_schedule_to_episode_ids_0_through_95"
)

REGISTRATION_SLOTS = tuple(
    f"{PANEL_IDS[seed]}_{phase}.json"
    for phase in ("screen", "confirm")
    for seed in (8000, 8100)
)
TARGET_OUTPUT_SLOTS = {
    "screen_rgb_complete_by_panel": {
        "chain3_8000": "screen/rgb/chain3_8000/COMPLETE.json",
        "chain3_8100": "screen/rgb/chain3_8100/COMPLETE.json",
    },
    "screen_progress_complete": "screen/progress/COMPLETE.json",
    "screen_pre_oracle_complete": "screen/preoracle/COMPLETE.json",
    "screen_evaluation_complete": "screen/evaluation/COMPLETE.json",
    "confirm_authorization_complete": "confirm/authorization/COMPLETE.json",
    "confirm_rgb_complete_by_panel": {
        "chain3_8000": "confirm/rgb/chain3_8000/COMPLETE.json",
        "chain3_8100": "confirm/rgb/chain3_8100/COMPLETE.json",
    },
    "confirm_progress_complete": "confirm/progress/COMPLETE.json",
    "confirm_pre_oracle_complete": "confirm/preoracle/COMPLETE.json",
    "confirm_evaluation_complete": "confirm/evaluation/COMPLETE.json",
}
CAMPAIGN_DAG = [
    "freeze_complete->screen_rgb",
    "screen_rgb->screen_progress",
    "screen_progress->screen_preoracle",
    "screen_preoracle->screen_evaluation",
    "screen_evaluation->confirm_authorization",
    "confirm_authorization->confirm_rgb",
    "confirm_rgb->confirm_progress",
    "confirm_progress->confirm_preoracle",
    "confirm_preoracle->confirm_evaluation",
]

PI05_MEMBER_PATHS = {
    "checkpoint": "model.safetensors",
    "config": "config.json",
    "preprocessor": "policy_preprocessor.json",
    "normalizer": "policy_preprocessor_step_2_normalizer_processor.safetensors",
    "postprocessor": "policy_postprocessor.json",
    "unnormalizer": "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
}
TOKENIZER_MEMBER_PATHS = frozenset(
    {
        "added_tokens.json",
        "config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
)


class Gate0ContractError(ValueError):
    """A fail-closed v248 pre-registration or artifact validation error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Gate0ContractError(message)


def require_exact_keys(
    value: Any, expected: frozenset[str], where: str
) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"{where} must be an object")
    actual = frozenset(str(key) for key in value)
    require(
        actual == expected,
        f"{where} field closure mismatch: expected {sorted(expected)}, "
        f"got {sorted(actual)}",
    )
    return value


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _reject_json_constant(value: str) -> None:
    raise Gate0ContractError(f"non-finite JSON constant is forbidden: {value}")


def _unique_json_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def loads_strict_json(payload: str, where: str = "JSON") -> Any:
    try:
        return json.loads(
            payload,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise Gate0ContractError(f"cannot decode strict {where}") from error


def sha256_file(path: Path, chunk_size: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def read_anchored_regular_bytes(
    path: Path, expected_sha256: str, where: str
) -> bytes:
    """Read and hash from one O_NOFOLLOW descriptor and reject path replacement."""
    require(is_sha256(expected_sha256), f"{where} expected SHA-256 invalid")
    absolute = Path(os.path.abspath(os.fspath(path.expanduser())))
    require(not absolute.is_symlink(), f"{where} is a symlink")
    current_parent = Path(absolute.anchor)
    for component in absolute.parts[1:-1]:
        current_parent /= component
        if os.path.lexists(current_parent):
            require(
                not current_parent.is_symlink(),
                f"{where} has a symlink parent: {current_parent}",
            )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(absolute, flags)
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"{where} is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    require(
        (before.st_dev, before.st_ino, before.st_size)
        == (after.st_dev, after.st_ino, after.st_size),
        f"{where} changed while being read",
    )
    current = os.stat(absolute, follow_symlinks=False)
    require(
        (before.st_dev, before.st_ino, before.st_size)
        == (current.st_dev, current.st_ino, current.st_size),
        f"{where} path was replaced while being read",
    )
    require(
        hashlib.sha256(payload).hexdigest() == expected_sha256,
        f"{where} differs from external SHA-256 anchor",
    )
    return payload


# Backwards-compatible private spelling for in-tree callers.  New security
# boundaries should use the public name so every consumer shares the same
# O_NOFOLLOW/single-descriptor implementation.
_read_anchored_regular_bytes = read_anchored_regular_bytes


def _strict_json_bytes(payload: bytes, where: str) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise Gate0ContractError(f"{where} is not UTF-8") from error
    value = loads_strict_json(text, where=where)
    require(isinstance(value, dict), f"{where} must be an object")
    return dict(value)


def is_sha256(value: Any) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", value)) if isinstance(value, str) else False


def _nonempty_text(value: Any, where: str) -> str:
    require(
        isinstance(value, str) and value.strip() == value and bool(value),
        f"{where} must be non-empty canonical text",
    )
    return value


def _safe_relative(value: Any, where: str) -> str:
    text = _nonempty_text(value, where)
    path = Path(text)
    require(
        not path.is_absolute() and ".." not in path.parts,
        f"{where} must be a safe relative path",
    )
    require(path.as_posix() == text, f"{where} must use canonical POSIX separators")
    return text


def _verify_file(path: Path, expected_sha: Any, expected_bytes: Any, where: str) -> None:
    require(is_sha256(expected_sha), f"{where}.sha256 is invalid")
    require(
        type(expected_bytes) is int and expected_bytes >= 0,
        f"{where}.bytes is invalid",
    )
    payload = read_anchored_regular_bytes(path, expected_sha, where)
    require(len(payload) == expected_bytes, f"{where} byte-size drift")


def validate_external_file_ref(value: Any, where: str) -> Path:
    item = require_exact_keys(value, EXTERNAL_FILE_FIELDS, where)
    path_text = _nonempty_text(item["path"], f"{where}.path")
    path = Path(path_text)
    require(
        path.is_absolute() and path.resolve().as_posix() == path_text,
        f"{where}.path must be canonical absolute path",
    )
    _verify_file(path, item["sha256"], item["bytes"], where)
    return path


def validate_repo_file_ref(value: Any, where: str, expected_relative: str | None = None) -> Path:
    item = require_exact_keys(value, REPO_FILE_FIELDS, where)
    relative = _safe_relative(item["relative_path"], f"{where}.relative_path")
    if expected_relative is not None:
        require(relative == expected_relative, f"{where} must bind {expected_relative}")
    path = (REPO / relative).resolve()
    require(path.is_relative_to(REPO), f"{where} escapes repository")
    _verify_file(path, item["sha256"], item["bytes"], where)
    return path


def validate_snapshot_member(
    value: Any,
    root: Path,
    where: str,
    expected_relative: str | None = None,
    *,
    require_materialized: bool = False,
) -> Path:
    item = require_exact_keys(value, SNAPSHOT_MEMBER_FIELDS, where)
    relative = _safe_relative(item["relative_path"], f"{where}.relative_path")
    if expected_relative is not None:
        require(relative == expected_relative, f"{where} must bind {expected_relative}")
    unresolved = root / relative
    if require_materialized:
        require(
            not unresolved.is_symlink(),
            f"{where} must be a materialized regular file, not a symlink",
        )
        path = unresolved.resolve()
        require(path.is_relative_to(root.resolve()), f"{where} escapes snapshot root")
    else:
        path = unresolved
    _verify_file(path, item["sha256"], item["bytes"], where)
    return path


def file_tree_sha256(members: Sequence[Mapping[str, Any]]) -> str:
    """Canonical digest of a field-closed, complete file-tree inventory."""
    records = [
        {
            "relative_path": str(member["relative_path"]),
            "sha256": str(member["sha256"]),
            "bytes": int(member["bytes"]),
        }
        for member in members
    ]
    records.sort(key=lambda item: item["relative_path"])
    return canonical_sha256(records)


def sam3_config_sha256(config_dict: Mapping[str, Any]) -> str:
    """Hash exactly ``Sam3VideoConfig().to_dict()`` in canonical JSON form."""
    require(isinstance(config_dict, Mapping), "SAM3 config fingerprint needs a mapping")
    return canonical_sha256(dict(config_dict))


def sam3_processor_sha256(
    image_processor_dict: Mapping[str, Any],
    video_processor_dict: Mapping[str, Any],
    clip_tokenizer_tree_sha256: str,
) -> str:
    """Hash the live SAM3 processor components and registered CLIP tree.

    Callers pass ``Sam3ImageProcessor().to_dict()`` and the ``to_dict()`` of
    ``Sam2VideoVideoProcessor(size=1008, mean=.5, std=.5)`` after constructing
    the actual processor used for inference.
    """
    require(
        isinstance(image_processor_dict, Mapping),
        "SAM3 image processor fingerprint",
    )
    require(
        isinstance(video_processor_dict, Mapping),
        "SAM3 video processor fingerprint",
    )
    require(is_sha256(clip_tokenizer_tree_sha256), "SAM3 CLIP tokenizer tree hash")
    return canonical_sha256(
        {
            "image_processor": dict(image_processor_dict),
            "video_processor": dict(video_processor_dict),
            "tokenizer_class": "CLIPTokenizer",
            "clip_tokenizer_tree_sha256": clip_tokenizer_tree_sha256,
        }
    )


def validate_file_tree(value: Any, where: str) -> Path:
    tree = require_exact_keys(value, TREE_FIELDS, where)
    root_text = _nonempty_text(tree["root"], f"{where}.root")
    root = Path(root_text)
    require(
        root.is_absolute()
        and root.resolve().as_posix() == root_text
        and root.is_dir()
        and not root.is_symlink(),
        f"{where}.root must be an existing canonical absolute directory",
    )
    require(is_sha256(tree["tree_sha256"]), f"{where}.tree_sha256 invalid")
    files = tree["files"]
    require(
        isinstance(files, list) and files,
        f"{where}.files must be a non-empty list",
    )
    seen: set[str] = set()
    for index, member in enumerate(files):
        item = require_exact_keys(
            member, SNAPSHOT_MEMBER_FIELDS, f"{where}.files[{index}]"
        )
        relative = _safe_relative(
            item["relative_path"], f"{where}.files[{index}].relative_path"
        )
        require(relative not in seen, f"{where} has duplicate member {relative}")
        seen.add(relative)
        validate_snapshot_member(
            item,
            root,
            f"{where}.files[{index}]",
            require_materialized=True,
        )
    all_members = list(root.rglob("*"))
    require(
        all(not path.is_symlink() for path in all_members),
        f"{where} contains a symlink",
    )
    actual = sorted(
        path.relative_to(root).as_posix()
        for path in all_members
        if path.is_file()
    )
    require(sorted(seen) == actual, f"{where} is not a complete tree inventory")
    require(
        file_tree_sha256(files) == tree["tree_sha256"],
        f"{where}.tree_sha256 mismatch",
    )
    ref_file = root.parent.parent / "refs" / "main"
    if ref_file.is_file():
        require(
            ref_file.read_text(encoding="utf-8").strip() == root.name,
            f"{where} Hugging Face main ref does not bind the registered revision",
        )
    return root


def _distribution_version(*names: str) -> str:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "NOT_INSTALLED"


def current_runtime_fingerprint(device: str = "cuda") -> dict[str, Any]:
    """Return runtime fields without importing any simulator package."""
    import torch

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": _distribution_version("torch"),
        "numpy": _distribution_version("numpy"),
        "pillow": _distribution_version("Pillow"),
        "opencv": _distribution_version("opencv-python", "opencv-python-headless"),
        "transformers": _distribution_version("transformers"),
        "lerobot": _distribution_version("lerobot"),
        "libero": _distribution_version("libero"),
        "robosuite": _distribution_version("robosuite"),
        "cuda": torch.version.cuda,
        "device": device,
    }


def _validate_created_utc(value: Any) -> None:
    text = _nonempty_text(value, "registration.created_utc")
    require(text.endswith("Z"), "registration.created_utc must use UTC Z suffix")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as error:
        raise Gate0ContractError("registration.created_utc is not ISO-8601") from error
    require(
        parsed.utcoffset() is not None
        and parsed.utcoffset().total_seconds() == 0,
        "registration.created_utc is not UTC",
    )


def _validate_panel(
    value: Any, phase: str
) -> tuple[int, tuple[dict[str, Any], ...]]:
    panel = require_exact_keys(value, PANEL_FIELDS, "registration.panel")
    seed_start = panel["seed_start"]
    require(
        seed_start in {8000, 8100},
        "formal panel seed_start must be 8000 or 8100",
    )
    require(
        panel["panel_id"] == PANEL_IDS[seed_start],
        "formal panel_id must be chain3_8000 or chain3_8100",
    )
    require(
        panel["source_episode_count"] == 96,
        "formal source panel must have 96 episodes",
    )
    episodes_raw = panel["episodes"]
    require(
        isinstance(episodes_raw, list) and episodes_raw,
        "registration.panel.episodes must be a non-empty list",
    )
    episodes: list[dict[str, Any]] = []
    last_episode = -1
    recovery_count = 0
    for index, raw in enumerate(episodes_raw):
        episode = dict(
            require_exact_keys(
                raw, EPISODE_FIELDS, f"registration.panel.episodes[{index}]"
            )
        )
        episode_id = episode["episode_id"]
        require(type(episode_id) is int and 0 <= episode_id < 96, f"episode {index} ID")
        require(
            episode_id > last_episode,
            "panel episode IDs must be unique and strictly sorted",
        )
        require(
            episode["env_seed"] == seed_start + episode_id,
            f"episode {episode_id} env-seed rule",
        )
        require(
            type(episode["stored_rows"]) is int
            and 1 <= episode["stored_rows"] <= 75,
            f"episode {episode_id} stored_rows",
        )
        require(
            episode["mode"] in {"ordinary", "recover_last_action"},
            f"episode {episode_id} mode",
        )
        if episode["mode"] == "ordinary":
            require(
                episode["stored_rows"] == 75,
                f"ordinary episode {episode_id} must expose 75 complete rows",
            )
        else:
            recovery_count += 1
            require(
                seed_start == 8100
                and episode_id == 77
                and episode["stored_rows"] == 65,
                "only seed-8100 episode 77 may use recovery",
            )
        episodes.append(episode)
        last_episode = episode_id
    require(
        recovery_count == (1 if seed_start == 8100 else 0),
        "panel recovery episode count mismatch",
    )
    observed_ids = tuple(item["episode_id"] for item in episodes)
    expected_ids = SCREEN_EPISODES[seed_start] if phase == "screen" else tuple(range(96))
    require(
        observed_ids == expected_ids,
        f"{phase} episode set mismatch for {PANEL_IDS[seed_start]}",
    )
    return seed_start, tuple(episodes)


def _validate_replay(value: Any) -> None:
    replay = require_exact_keys(value, REPLAY_FIELDS, "registration.replay")
    expected = {
        "task": TASK,
        "c": C,
        "latent_layout": LATENT_LAYOUT,
        "latent_dim": LATENT_DIM,
        "proprio_dim": PROPRIO_DIM,
        "proprio_keys": list(PROPRIO_KEYS),
        "cameras": CAMERAS,
        "jpeg_quality": 95,
        "orientation": "rotate_180_to_lerobot_policy_view",
        "frame_roles": list(FRAME_ROLES),
        "two_fresh_recovery_replays": True,
    }
    require(
        dict(replay) == expected,
        "registration.replay does not equal the frozen v248 replay contract",
    )


def _validate_recovery(
    value: Any, seed_start: int, episodes: Sequence[dict[str, Any]]
) -> None:
    recovery = require_exact_keys(
        value, RECOVERY_FIELDS, "registration.terminal_recovery"
    )
    require(
        isinstance(recovery["enabled"], bool),
        "terminal_recovery.enabled must be boolean",
    )
    require(recovery["phi_only"] is True, "recovered endpoint must be phi_only=true")
    require(
        recovery["dynamics_transition_eligible"] is False,
        "recovered endpoint must be dynamics_transition_eligible=false",
    )
    enabled = seed_start == 8100
    require(
        recovery["enabled"] is enabled,
        "terminal recovery enablement disagrees with panel",
    )
    if not enabled:
        inactive_fields = RECOVERY_FIELDS - {
            "enabled",
            "phi_only",
            "dynamics_transition_eligible",
        }
        for key in inactive_fields:
            require(recovery[key] is None, f"disabled terminal_recovery.{key} must be null")
        return

    scalar_expected = {
        "episode_id": 77,
        "env_seed": 8177,
        "stored_chunks": 65,
        "stored_steps": 650,
        "source_terminal_t": 651,
        "missing_chunk_t": 650,
        "missing_raw_sigma_choice": 0.6,
    }
    for key, expected in scalar_expected.items():
        require(recovery[key] == expected, f"terminal_recovery.{key} mismatch")
    selected = [item for item in episodes if item["mode"] == "recover_last_action"]
    require(
        len(selected) == 1 and selected[0]["episode_id"] == 77,
        "ep77 recovery selection missing",
    )

    rng = require_exact_keys(
        recovery["sigma_rng"],
        SIGMA_RNG_FIELDS,
        "registration.terminal_recovery.sigma_rng",
    )
    require(
        rng["algorithm"] == "numpy.random.Generator(PCG64)",
        "sigma RNG algorithm mismatch",
    )
    require(rng["seed"] == 8100, "sigma RNG seed mismatch")
    require(rng["choices"] == [0.0, 0.6, 1.5, 3.0], "sigma RNG choices mismatch")
    counts = rng["source_proposed_chunks_by_episode"]
    require(
        isinstance(counts, list) and len(counts) == 78,
        "sigma RNG needs source proposed chunks for episodes 0..77",
    )
    require(
        counts == [75] * 77 + [66],
        "sigma RNG proposed chunk counts must equal the sealed source rows",
    )
    require(
        rng["missing_draw_offset_within_episode"] == 65,
        "missing sigma draw must be ep77 offset 65",
    )
    import numpy as np

    generator = np.random.default_rng(8100)
    choices = rng["choices"]
    missing_choice = None
    for episode_id, proposed_chunks in enumerate(counts):
        draws = [
            choices[int(generator.integers(len(choices)))]
            for _ in range(proposed_chunks)
        ]
        if episode_id == 77:
            missing_choice = draws[65]
    require(
        missing_choice == 0.6,
        "registered global sigma RNG metadata does not reproduce raw choice 0.6",
    )

    sampler = require_exact_keys(
        recovery["sampler"],
        SAMPLER_FIELDS,
        "registration.terminal_recovery.sampler",
    )
    expected_sampler = {
        "zero_sigma_path": "Pi05Runner.sample_chunk",
        "positive_sigma_path": "lcwm.sampler.sample_chunks",
        "seed_rule": "env_seed*7919+t",
        "n": 1,
        "normalized_dtype": "torch.float32",
        "exact_comparator": "torch.equal",
        "prerequisite_chunks": 65,
        "missing_chunk_actions": 10,
        "required_halt_action_offset": 0,
        "required_halt_t": 651,
    }
    require(
        dict(sampler) == expected_sampler,
        "terminal-recovery sampler contract mismatch",
    )


def _validate_sanitized(value: Any) -> None:
    source = require_exact_keys(value, SANITIZED_FIELDS, "registration.sanitized_source")
    for key, digest in source.items():
        require(is_sha256(digest), f"registration.sanitized_source.{key} invalid")


def _validate_oracle(value: Any, expected_source_role: str | None) -> None:
    """Validate the evaluator-only source oracle without producer-side I/O.

    Formal replay and semantic production may learn the registered path and
    digest strings from the pre-registration, but they must not stat, hash, or
    open the summary.  Only an explicitly role-tagged semantic evaluator may
    verify the file, after its independent progress artifact has been sealed.
    """
    oracle = require_exact_keys(value, ORACLE_FIELDS, "registration.oracle")
    summary = require_exact_keys(
        oracle["summary"], EXTERNAL_FILE_FIELDS, "registration.oracle.summary"
    )
    path_text = _nonempty_text(summary["path"], "registration.oracle.summary.path")
    path = Path(path_text)
    require(
        path.is_absolute()
        and "." not in path.parts
        and ".." not in path.parts
        and path.as_posix() == path_text,
        "registration.oracle.summary.path must be canonical absolute syntax",
    )
    require(is_sha256(summary["sha256"]), "registration.oracle.summary.sha256 invalid")
    require(
        type(summary["bytes"]) is int and summary["bytes"] >= 0,
        "registration.oracle.summary.bytes invalid",
    )
    require(
        oracle["access"] == "post_progress_seal_evaluator_only",
        "registration.oracle.access mismatch",
    )
    require(
        oracle["fields"]
        == ["episode_records.idx", "seed", "steps", "events"],
        "registration.oracle.fields mismatch",
    )
    require(
        oracle["success_values_accessed"] is False,
        "registration oracle must prohibit success-value access",
    )
    require(
        expected_source_role
        in {
            None,
            "formal_replay",
            "semantic_producer",
            "semantic_evaluator",
            "confirm_authorizer",
        },
        f"unsupported registration validation role: {expected_source_role!r}",
    )
    if expected_source_role == "semantic_evaluator":
        require(
            path.resolve().as_posix() == path_text,
            "oracle summary canonical path drift",
        )
        require(not path.is_symlink(), "oracle summary must be a regular file")
        _verify_file(
            path,
            summary["sha256"],
            summary["bytes"],
            "registration.oracle.summary",
        )


def _snapshot_root(spec: Mapping[str, Any], where: str) -> Path:
    root_text = _nonempty_text(spec["local_root"], f"{where}.local_root")
    root = Path(root_text)
    require(
        root.is_absolute()
        and root.resolve().as_posix() == root_text
        and root.is_dir()
        and not root.is_symlink(),
        f"{where}.local_root must be an existing canonical absolute directory",
    )
    revision = _nonempty_text(spec["revision"], f"{where}.revision")
    require(
        root.name == revision,
        f"{where}.local_root basename must equal frozen revision",
    )
    return root


def _validate_exact_snapshot_inventory(
    root: Path, expected_relative_paths: Sequence[str], where: str
) -> None:
    members = list(root.rglob("*"))
    require(
        all(not path.is_symlink() for path in members),
        f"{where} contains a symlink",
    )
    actual_files = sorted(
        path.relative_to(root).as_posix() for path in members if path.is_file()
    )
    require(
        actual_files == sorted(expected_relative_paths),
        f"{where} file-tree closure mismatch",
    )


def _validate_models(value: Any) -> None:
    models = require_exact_keys(value, MODEL_FIELDS, "registration.models")
    pi05 = require_exact_keys(
        models["pi05"], PI05_FIELDS, "registration.models.pi05"
    )
    require(pi05["repo_id"] == "lerobot/pi05_libero_finetuned", "pi05 repo_id mismatch")
    require(
        bool(re.fullmatch(r"[0-9a-f]{40}", str(pi05["revision"]))),
        "pi05 revision must be an exact 40-hex commit",
    )
    pi05_root = _snapshot_root(pi05, "registration.models.pi05")
    for role, relative in PI05_MEMBER_PATHS.items():
        validate_snapshot_member(
            pi05[role],
            pi05_root,
            f"registration.models.pi05.{role}",
            relative,
            require_materialized=True,
        )
    _validate_exact_snapshot_inventory(
        pi05_root,
        list(PI05_MEMBER_PATHS.values()),
        "registration.models.pi05",
    )

    tokenizer = require_exact_keys(
        models["tokenizer"], TOKENIZER_FIELDS, "registration.models.tokenizer"
    )
    require(
        tokenizer["repo_id"] == "google/paligemma-3b-pt-224",
        "tokenizer repo_id mismatch",
    )
    require(
        bool(re.fullmatch(r"[0-9a-f]{40}", str(tokenizer["revision"]))),
        "tokenizer revision must be an exact 40-hex commit",
    )
    tokenizer_root = _snapshot_root(tokenizer, "registration.models.tokenizer")
    tokenizer_ref = tokenizer_root.parent.parent / "refs" / "main"
    require(
        tokenizer_ref.is_file() and not tokenizer_ref.is_symlink(),
        "tokenizer cache refs/main is missing/symlink",
    )
    require(
        tokenizer_ref.read_text(encoding="utf-8").strip()
        == tokenizer["revision"],
        "tokenizer refs/main does not equal frozen revision",
    )
    files = tokenizer["files"]
    require(
        isinstance(files, list),
        "registration.models.tokenizer.files must be a list",
    )
    seen = set()
    for index, member in enumerate(files):
        relative = member.get("relative_path") if isinstance(member, Mapping) else None
        validate_snapshot_member(
            member,
            tokenizer_root,
            f"registration.models.tokenizer.files[{index}]",
            require_materialized=True,
        )
        require(relative not in seen, "duplicate tokenizer snapshot member")
        seen.add(relative)
    require(seen == TOKENIZER_MEMBER_PATHS, "tokenizer file allow-list mismatch")
    _validate_exact_snapshot_inventory(
        tokenizer_root,
        sorted(TOKENIZER_MEMBER_PATHS),
        "registration.models.tokenizer",
    )

    sam3 = require_exact_keys(
        models["sam3"], SAM3_FIELDS, "registration.models.sam3"
    )
    _nonempty_text(sam3["model_id"], "registration.models.sam3.model_id")
    _nonempty_text(sam3["revision"], "registration.models.sam3.revision")
    validate_external_file_ref(
        sam3["checkpoint"], "registration.models.sam3.checkpoint"
    )
    validate_file_tree(
        sam3["clip_tokenizer"], "registration.models.sam3.clip_tokenizer"
    )
    require(
        is_sha256(sam3["config_sha256"]),
        "registration.models.sam3.config_sha256 invalid",
    )
    require(
        is_sha256(sam3["processor_sha256"]),
        "registration.models.sam3.processor_sha256 invalid",
    )


def _validate_templates(value: Any) -> None:
    templates = require_exact_keys(value, TEMPLATE_FIELDS, "registration.templates")
    for role in sorted(TEMPLATE_FIELDS):
        validate_external_file_ref(templates[role], f"registration.templates.{role}")


def _validate_sources(value: Any, expected_source_role: str | None) -> None:
    sources = require_exact_keys(value, SOURCE_ROLES, "registration.sources")
    for role in sorted(SOURCE_ROLES):
        expected = KNOWN_SOURCE_PATHS.get(role)
        path = validate_repo_file_ref(
            sources[role], f"registration.sources.{role}", expected
        )
        if role in {"semantic_producer", "semantic_evaluator"}:
            require(
                path.parent == REPO / "scripts" and path.suffix == ".py",
                f"{role} must be a scripts/*.py source",
            )
    if expected_source_role is not None:
        require(
            expected_source_role in SOURCE_ROLES,
            f"unknown expected source role {expected_source_role!r}",
        )


def _validate_runtime(value: Any) -> None:
    runtime = require_exact_keys(value, RUNTIME_FIELDS, "registration.runtime")
    require(runtime["device"] == "cuda", "formal replay runtime device must be cuda")
    expected = current_runtime_fingerprint(device="cuda")
    for key, actual in expected.items():
        require(runtime[key] == actual, f"registration.runtime.{key} drift")
    sources = require_exact_keys(
        runtime["runtime_sources"],
        RUNTIME_SOURCE_ROLES,
        "registration.runtime.runtime_sources",
    )
    for role in sorted(RUNTIME_SOURCE_ROLES):
        validate_external_file_ref(
            sources[role], f"registration.runtime.runtime_sources.{role}"
        )


def _validate_semantic(value: Any) -> None:
    semantic = require_exact_keys(
        value, SEMANTIC_FIELDS, "registration.semantic_contract"
    )
    expected_scalars = {
        "manifest_schema": RGB_MANIFEST_SCHEMA,
        "result_schema": RGB_RESULT_SCHEMA,
        "complete_schema": RGB_COMPLETE_SCHEMA,
        "allowed_frame_roles": list(FRAME_ROLES),
    }
    for key, expected in expected_scalars.items():
        require(
            semantic[key] == expected,
            f"registration.semantic_contract.{key} mismatch",
        )
    recovered = require_exact_keys(
        semantic["recovered_endpoint"],
        RECOVERED_SEMANTIC_FIELDS,
        "registration.semantic_contract.recovered_endpoint",
    )
    require(recovered["phi_only"] is True, "semantic recovered endpoint must be phi-only")
    require(
        recovered["dynamics_transition_eligible"] is False,
        "semantic recovered endpoint must be dynamics-ineligible",
    )


def campaign_protocol_payload(registration: Mapping[str, Any]) -> dict[str, Any]:
    """Return the phase-invariant protocol committed by all four registrations.

    Registration file digests cannot be embedded here without creating a hash
    cycle.  The freeze-bundle COMPLETE file closes over the four registration
    file digests; this payload instead makes every registration independently
    commit to the exact shared implementation and the deterministic
    screen-to-confirm expansion rule.
    """
    return {
        "registration_schema": REGISTRATION_SCHEMA,
        "task": TASK,
        "replay": registration["replay"],
        "models": registration["models"],
        "templates": registration["templates"],
        "sources": registration["sources"],
        "runtime": registration["runtime"],
        "semantic_contract": registration["semantic_contract"],
        "screen_episode_ids": {
            PANEL_IDS[seed]: list(SCREEN_EPISODES[seed]) for seed in (8000, 8100)
        },
        "confirm_episode_ids": {
            PANEL_IDS[seed]: list(range(96)) for seed in (8000, 8100)
        },
        "automatic_expansion_rule": AUTOMATIC_CONFIRM_EXPANSION_RULE,
        "confirm_authorization_required": True,
    }


def _validate_campaign(value: Any, registration: Mapping[str, Any]) -> None:
    campaign = require_exact_keys(value, CAMPAIGN_FIELDS, "registration.campaign")
    require(campaign["schema"] == CAMPAIGN_SCHEMA, "campaign schema mismatch")
    expected_payload = campaign_protocol_payload(registration)
    expected_protocol = canonical_sha256(expected_payload)
    require(
        campaign["protocol_sha256"] == expected_protocol,
        "campaign protocol_sha256 mismatch",
    )
    require(
        isinstance(campaign["campaign_id"], str)
        and bool(re.fullmatch(r"v248-gate0-[0-9a-f]{16}", campaign["campaign_id"])),
        "campaign_id syntax mismatch",
    )
    require(
        is_sha256(campaign["projection_sha256"]),
        "campaign projection_sha256 invalid",
    )
    require(
        campaign["manifest_relative_path"] == "CAMPAIGN.json",
        "campaign manifest relative path mismatch",
    )
    seed_start = int(registration["panel"]["seed_start"])
    expected_slot = f"{PANEL_IDS[seed_start]}_{registration['phase']}.json"
    require(
        campaign["registration_slot"] == expected_slot,
        "campaign registration slot mismatch",
    )


def validate_registration(
    payload: Any, expected_source_role: str | None = None
) -> dict[str, Any]:
    """Validate exact closure, self seal, artifacts, runtime, and panel mechanics."""
    registration = dict(require_exact_keys(payload, TOP_FIELDS, "registration"))
    require(registration["schema"] == REGISTRATION_SCHEMA, "registration schema mismatch")
    require(
        registration["status"] == REGISTRATION_STATUS,
        "registration was not frozen before formal RGB",
    )
    _nonempty_text(registration["registration_id"], "registration.registration_id")
    _validate_created_utc(registration["created_utc"])
    require(
        registration["phase"] in {"screen", "confirm"},
        "registration.phase must be screen or confirm",
    )
    self_sha = registration["self_sha256"]
    require(is_sha256(self_sha), "registration.self_sha256 invalid")
    body = {key: value for key, value in registration.items() if key != "self_sha256"}
    require(canonical_sha256(body) == self_sha, "registration canonical self_sha256 mismatch")

    seed_start, episodes = _validate_panel(
        registration["panel"], registration["phase"]
    )
    _validate_replay(registration["replay"])
    _validate_recovery(registration["terminal_recovery"], seed_start, episodes)
    _validate_sanitized(registration["sanitized_source"])
    _validate_oracle(registration["oracle"], expected_source_role)
    _validate_models(registration["models"])
    _validate_templates(registration["templates"])
    _validate_sources(registration["sources"], expected_source_role)
    _validate_runtime(registration["runtime"])
    _validate_semantic(registration["semantic_contract"])
    _validate_campaign(registration["campaign"], registration)
    return registration


def load_and_validate_registration(
    path: Path | str,
    expected_source_role: str | None = None,
    expected_file_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Load one exact self-hashed pre-registration and return it plus file SHA."""
    resolved = Path(path).expanduser().resolve()
    require(resolved.is_file(), f"registration file missing: {resolved}")
    try:
        if expected_file_sha256 is None:
            raw = resolved.read_bytes()
            actual_sha256 = hashlib.sha256(raw).hexdigest()
        else:
            raw = read_anchored_regular_bytes(
                resolved,
                expected_file_sha256,
                "externally anchored registration",
            )
            actual_sha256 = expected_file_sha256
        payload = loads_strict_json(raw.decode("utf-8"), where="registration JSON")
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Gate0ContractError(f"cannot load registration: {resolved}") from error
    registration = validate_registration(
        payload, expected_source_role=expected_source_role
    )
    return registration, actual_sha256


def load_and_validate_campaign_bundle(
    root: Path | str,
    expected_freeze_complete_sha256: str,
    *,
    expected_source_role: str | None = None,
) -> dict[str, Any]:
    """Validate external freeze COMPLETE -> CAMPAIGN -> all four registrations.

    The returned mapping is the sole downstream authority view.  A consumer
    must not replace any registration or output slot with a free-standing CLI
    path after this function succeeds.
    """
    lexical_root = Path(os.path.abspath(os.fspath(Path(root).expanduser())))
    require(not lexical_root.is_symlink(), "campaign bundle root is a symlink")
    resolved_root = lexical_root.resolve()
    require(resolved_root.is_dir(), "campaign bundle root is missing")
    require(
        {path.name for path in resolved_root.iterdir()}
        == {
            "registrations",
            "PRODUCER.py",
            "CAMPAIGN.json",
            "FREEZE_RESULT.json",
            "COMPLETE.json",
        },
        "campaign bundle root closure mismatch",
    )
    registrations_root = resolved_root / "registrations"
    require(
        registrations_root.is_dir() and not registrations_root.is_symlink(),
        "campaign registrations directory missing/symlink",
    )
    require(
        {path.name for path in registrations_root.iterdir()} == set(REGISTRATION_SLOTS),
        "campaign registration slot closure mismatch",
    )

    complete_path = resolved_root / "COMPLETE.json"
    complete = _strict_json_bytes(
        _read_anchored_regular_bytes(
            complete_path,
            expected_freeze_complete_sha256,
            "externally anchored freeze COMPLETE",
        ),
        "freeze COMPLETE",
    )
    require_exact_keys(complete, FREEZE_COMPLETE_FIELDS, "freeze COMPLETE")
    require(
        complete["schema"] == FREEZE_COMPLETE_SCHEMA
        and complete["status"] == "atomic_success"
        and complete["atomic_commit"] is True,
        "freeze COMPLETE status",
    )
    for key in (
        "input_closure_sha256",
        "materialization_complete_sha256",
        "campaign_manifest_sha256",
        "campaign_body_sha256",
        "projection_sha256",
        "result_sha256",
        "producer_sha256",
    ):
        require(is_sha256(complete[key]), f"freeze COMPLETE.{key} invalid")
    registration_sha256s = require_exact_keys(
        complete["registration_sha256s"],
        frozenset(REGISTRATION_SLOTS),
        "freeze COMPLETE registration_sha256s",
    )
    require(
        all(is_sha256(value) for value in registration_sha256s.values()),
        "freeze COMPLETE registration digest invalid",
    )
    _read_anchored_regular_bytes(
        resolved_root / "FREEZE_RESULT.json",
        complete["result_sha256"],
        "freeze RESULT",
    )
    _read_anchored_regular_bytes(
        resolved_root / "PRODUCER.py",
        complete["producer_sha256"],
        "freeze producer",
    )

    campaign_path = resolved_root / "CAMPAIGN.json"
    campaign_bytes = _read_anchored_regular_bytes(
        campaign_path,
        complete["campaign_manifest_sha256"],
        "campaign manifest",
    )
    campaign = _strict_json_bytes(campaign_bytes, "campaign manifest")
    require_exact_keys(campaign, CAMPAIGN_MANIFEST_FIELDS, "campaign manifest")
    require(
        campaign["schema"] == CAMPAIGN_MANIFEST_SCHEMA
        and campaign["status"] == "frozen_before_any_formal_rgb",
        "campaign manifest status",
    )
    body = {key: value for key, value in campaign.items() if key != "body_sha256"}
    require(
        campaign["body_sha256"]
        == complete["campaign_body_sha256"]
        == canonical_sha256(body),
        "campaign manifest body seal mismatch",
    )
    projection = dict(
        require_exact_keys(
            campaign["projection"],
            CAMPAIGN_PROJECTION_FIELDS,
            "campaign projection",
        )
    )
    require(
        projection["schema"] == CAMPAIGN_PROJECTION_SCHEMA,
        "campaign projection schema",
    )
    require(
        campaign["projection_sha256"] == canonical_sha256(projection),
        "campaign projection seal mismatch",
    )
    intent = dict(
        require_exact_keys(
            projection["intent"], CAMPAIGN_INTENT_FIELDS, "campaign intent"
        )
    )
    require(intent["schema"] == CAMPAIGN_INTENT_SCHEMA, "campaign intent schema")
    for key in ("input_closure_sha256", "protocol_sha256"):
        require(is_sha256(intent[key]), f"campaign intent.{key} invalid")
    campaign_id = f"v248-gate0-{canonical_sha256(intent)[:16]}"
    require(
        projection["campaign_id"] == campaign["campaign_id"] == campaign_id,
        "campaign ID/intent anchor mismatch",
    )
    require(
        complete["campaign_id"] == campaign_id
        and complete["projection_sha256"] == campaign["projection_sha256"],
        "freeze COMPLETE campaign identity mismatch",
    )
    require(
        campaign["protocol_sha256"] == intent["protocol_sha256"],
        "campaign protocol mismatch",
    )
    require(
        complete["input_closure_sha256"] == intent["input_closure_sha256"],
        "campaign input closure mismatch",
    )
    expected_screen = {
        PANEL_IDS[seed]: list(SCREEN_EPISODES[seed]) for seed in (8000, 8100)
    }
    expected_confirm = {
        PANEL_IDS[seed]: list(range(96)) for seed in (8000, 8100)
    }
    require(
        intent["screen_episode_ids"]
        == campaign["screen_episode_ids"]
        == expected_screen,
        "campaign screen schedule mismatch",
    )
    require(
        intent["confirm_episode_ids"]
        == campaign["confirm_episode_ids"]
        == expected_confirm,
        "campaign confirm schedule mismatch",
    )
    require(
        intent["automatic_expansion_rule"]
        == campaign["automatic_expansion_rule"]
        == AUTOMATIC_CONFIRM_EXPANSION_RULE,
        "campaign automatic expansion rule mismatch",
    )
    require(
        intent["confirm_authorization_required"] is True
        and campaign["confirm_authorization_required"] is True,
        "campaign confirm authorization is not mandatory",
    )
    target_root_text = _nonempty_text(intent["target_root"], "campaign target_root")
    target_root = Path(target_root_text)
    require(
        target_root.is_absolute()
        and "." not in target_root.parts
        and ".." not in target_root.parts
        and target_root.as_posix() == target_root_text,
        "campaign target_root is not canonical absolute syntax",
    )
    require(
        intent["target_output_slots"] == TARGET_OUTPUT_SLOTS,
        "campaign target output slots mismatch",
    )
    require(intent["dag"] == CAMPAIGN_DAG, "campaign DAG mismatch")

    core_sha256s = require_exact_keys(
        intent["registration_core_sha256s"],
        frozenset(REGISTRATION_SLOTS),
        "campaign registration core SHA-256s",
    )
    campaign_registration_refs = require_exact_keys(
        campaign["registrations"],
        frozenset(REGISTRATION_SLOTS),
        "campaign registrations",
    )
    registrations: dict[str, dict[str, Any]] = {}
    for slot in REGISTRATION_SLOTS:
        reference = require_exact_keys(
            campaign_registration_refs[slot],
            CAMPAIGN_REGISTRATION_FIELDS,
            f"campaign registrations.{slot}",
        )
        file_sha = registration_sha256s[slot]
        require(
            reference["file_sha256"] == file_sha,
            f"campaign registration file seal mismatch: {slot}",
        )
        path = registrations_root / slot
        registration = validate_registration(
            _strict_json_bytes(
                _read_anchored_regular_bytes(path, file_sha, f"registration {slot}"),
                f"registration {slot}",
            ),
            expected_source_role=expected_source_role,
        )
        core = {
            key: value
            for key, value in registration.items()
            if key not in {"campaign", "self_sha256"}
        }
        require(
            is_sha256(core_sha256s[slot])
            and canonical_sha256(core)
            == core_sha256s[slot]
            == reference["core_sha256"],
            f"campaign registration core mismatch: {slot}",
        )
        require(
            registration["self_sha256"] == reference["self_sha256"],
            f"campaign registration self seal mismatch: {slot}",
        )
        require(
            reference["seed_start"] in {8000, 8100}
            and reference["phase"] in {"screen", "confirm"},
            f"campaign registration reference identity invalid: {slot}",
        )
        expected_panel = PANEL_IDS[reference["seed_start"]]
        require(
            slot == f"{expected_panel}_{reference['phase']}.json",
            f"campaign registration reference uses wrong slot: {slot}",
        )
        require(
            reference["panel_id"] == expected_panel == registration["panel"]["panel_id"]
            and reference["phase"] == registration["phase"],
            f"campaign registration identity mismatch: {slot}",
        )
        campaign_ref = registration["campaign"]
        require(
            campaign_ref
            == {
                "schema": CAMPAIGN_SCHEMA,
                "campaign_id": campaign_id,
                "protocol_sha256": intent["protocol_sha256"],
                "projection_sha256": campaign["projection_sha256"],
                "manifest_relative_path": "CAMPAIGN.json",
                "registration_slot": slot,
            },
            f"registration does not point back to campaign: {slot}",
        )
        require(
            canonical_sha256(campaign_protocol_payload(registration))
            == intent["protocol_sha256"],
            f"registration shared protocol mismatch: {slot}",
        )
        registrations[slot] = {
            "path": path,
            "registration": registration,
            "file_sha256": file_sha,
        }
    return {
        "root": resolved_root,
        "complete": complete,
        "freeze_complete_sha256": expected_freeze_complete_sha256,
        "campaign": campaign,
        "campaign_manifest_sha256": complete["campaign_manifest_sha256"],
        "registrations": registrations,
        "target_root": target_root,
        "target_output_slots": intent["target_output_slots"],
    }


def assert_no_simulator_modules_imported() -> None:
    """Fail if validation-only code accidentally imported a simulation stack."""
    forbidden_prefixes = ("libero", "robosuite", "mujoco", "lerobot")
    imported = sorted(
        name
        for name in sys.modules
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in forbidden_prefixes
        )
    )
    require(
        not imported,
        f"simulator/policy modules imported during validation-only: {imported[:8]}",
    )


__all__ = [
    "CAMERAS",
    "C",
    "CAMPAIGN_SCHEMA",
    "CAMPAIGN_DAG",
    "CAMPAIGN_MANIFEST_SCHEMA",
    "CAMPAIGN_PROJECTION_SCHEMA",
    "CAMPAIGN_INTENT_SCHEMA",
    "AUTOMATIC_CONFIRM_EXPANSION_RULE",
    "FRAME_ROLES",
    "Gate0ContractError",
    "LATENT_DIM",
    "LATENT_LAYOUT",
    "PANEL_IDS",
    "PROPRIO_DIM",
    "PROPRIO_KEYS",
    "REGISTRATION_SCHEMA",
    "REGISTRATION_STATUS",
    "REGISTRATION_SLOTS",
    "RGB_COMPLETE_SCHEMA",
    "RGB_MANIFEST_SCHEMA",
    "RGB_RESULT_SCHEMA",
    "SCREEN_EPISODES",
    "TASK",
    "TARGET_OUTPUT_SLOTS",
    "assert_no_simulator_modules_imported",
    "canonical_bytes",
    "canonical_sha256",
    "campaign_protocol_payload",
    "current_runtime_fingerprint",
    "file_tree_sha256",
    "is_sha256",
    "loads_strict_json",
    "read_anchored_regular_bytes",
    "load_and_validate_registration",
    "load_and_validate_campaign_bundle",
    "require",
    "require_exact_keys",
    "sam3_config_sha256",
    "sam3_processor_sha256",
    "sha256_file",
    "validate_external_file_ref",
    "validate_file_tree",
    "validate_registration",
    "validate_repo_file_ref",
    "validate_snapshot_member",
]
