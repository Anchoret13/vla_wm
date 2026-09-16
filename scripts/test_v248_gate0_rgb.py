#!/usr/bin/env python3
# RETIRED 2026-09-16. Nothing supersedes it: the gate it serves was removed as a
# prerequisite at plan_and_progress/2026-09-12.md:105, and the round used a supplied
# Phi' annotation instead. It has produced zero artifacts - none of
# results/v248_gate0_{registration_bundle,materialized_models,formal_campaign}_v1 exists.
# Blockers never cleared: WORM ledger, runtime fingerprint, registration bundle, and
# non-deterministic cuDNN/TF32 (plan_and_progress/2026-09-02.md:108-129).
# Do not extend, do not import, do not cite as capability. Recoverable at d35e2832.
"""CPU-only contract tests for the v248 formal Gate-0 RGB replay."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from unittest import mock

import numpy as np
import torch


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from lcwm import v248_gate0_contract as C  # noqa: E402
import eval_v248_gate0_progress as E  # noqa: E402
import produce_v248_gate0_progress as P  # noqa: E402


def _load_replay_module():
    path = REPO / "scripts" / "replay_v248_gate0_rgb.py"
    spec = importlib.util.spec_from_file_location("v248_gate0_rgb_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


R = _load_replay_module()


def _load_authorizer_module():
    path = REPO / "scripts" / "authorize_v248_gate0_confirm.py"
    spec = importlib.util.spec_from_file_location(
        "v248_confirm_authorizer_under_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


A = _load_authorizer_module()


def expect_error(function: Callable[[], Any], contains: str | None = None) -> Exception:
    try:
        function()
    except Exception as error:  # noqa: BLE001 - negative-path contract test
        if contains is not None:
            assert contains in str(error), (contains, str(error))
        return error
    raise AssertionError("expected failure")


def file_ref(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": path.as_posix(),
        "sha256": C.sha256_file(path),
        "bytes": path.stat().st_size,
    }


def repo_ref(relative: str) -> dict[str, Any]:
    path = REPO / relative
    return {
        "relative_path": relative,
        "sha256": C.sha256_file(path),
        "bytes": path.stat().st_size,
    }


def member_ref(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    return {
        "relative_path": relative,
        "sha256": C.sha256_file(path),
        "bytes": path.stat().st_size,
    }


def write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def materialized_snapshot(
    root: Path, revision: str, names: list[str]
) -> tuple[Path, list[dict[str, Any]]]:
    snapshot = root / "snapshots" / revision
    refs = []
    for index, name in enumerate(names):
        write(snapshot / name, f"fixture-{index}-{name}\n".encode())
        refs.append(member_ref(snapshot, name))
    write(root / "refs" / "main", revision.encode())
    return snapshot, refs


def build_external_contract_files(root: Path) -> dict[str, Path]:
    names = (
        "runtime.py",
        "sam3.safetensors",
        "cream.jpg",
        "cream.json",
        "semantic.json",
    )
    return {name: write(root / name, f"{name}\n".encode()) for name in names}


def build_registration(root: Path, sanitized: dict[str, str]) -> tuple[dict[str, Any], Path]:
    policy_revision = "a" * 40
    tokenizer_revision = "b" * 40
    pi_names = list(C.PI05_MEMBER_PATHS.values())
    policy_root, pi_refs = materialized_snapshot(root / "policy", policy_revision, pi_names)
    pi_by_path = {item["relative_path"]: item for item in pi_refs}
    tokenizer_names = sorted(C.TOKENIZER_MEMBER_PATHS)
    tokenizer_root, tokenizer_refs = materialized_snapshot(
        root / "tokenizer", tokenizer_revision, tokenizer_names
    )
    clip_root = root / "clip"
    write(clip_root / "vocab.json", b"{}\n")
    write(clip_root / "merges.txt", b"fixture\n")
    clip_files = [member_ref(clip_root, name) for name in ("merges.txt", "vocab.json")]
    external = build_external_contract_files(root / "external")
    external["semantic.json"].write_bytes(
        (REPO / "plan_and_progress" / "v248_gate0_semantic_config.json").read_bytes()
    )
    runtime_ref = file_ref(external["runtime.py"])
    runtime = C.current_runtime_fingerprint(device="cuda")
    runtime["runtime_sources"] = {
        role: copy.deepcopy(runtime_ref) for role in C.RUNTIME_SOURCE_ROLES
    }
    source_paths = {
        **C.KNOWN_SOURCE_PATHS,
        "semantic_producer": "scripts/produce_v248_gate0_progress.py",
        "semantic_evaluator": "scripts/eval_v248_gate0_progress.py",
    }
    sources = {role: repo_ref(source_paths[role]) for role in C.SOURCE_ROLES}
    episodes = [
        {
            "episode_id": episode_id,
            "env_seed": 8000 + episode_id,
            "stored_rows": 75,
            "mode": "ordinary",
        }
        for episode_id in C.SCREEN_EPISODES[8000]
    ]
    recovery = {
        "enabled": False,
        "episode_id": None,
        "env_seed": None,
        "stored_chunks": None,
        "stored_steps": None,
        "source_terminal_t": None,
        "missing_chunk_t": None,
        "missing_raw_sigma_choice": None,
        "sigma_rng": None,
        "sampler": None,
        "phi_only": True,
        "dynamics_transition_eligible": False,
    }
    registration = {
        "schema": C.REGISTRATION_SCHEMA,
        "status": C.REGISTRATION_STATUS,
        "registration_id": "cpu_fixture_chain3_8000_screen",
        "created_utc": "2026-09-01T00:00:00Z",
        "phase": "screen",
        "panel": {
            "panel_id": "chain3_8000",
            "seed_start": 8000,
            "source_episode_count": 96,
            "episodes": episodes,
        },
        "replay": {
            "task": C.TASK,
            "c": C.C,
            "latent_layout": C.LATENT_LAYOUT,
            "latent_dim": C.LATENT_DIM,
            "proprio_dim": C.PROPRIO_DIM,
            "proprio_keys": list(C.PROPRIO_KEYS),
            "cameras": C.CAMERAS,
            "jpeg_quality": 95,
            "orientation": "rotate_180_to_lerobot_policy_view",
            "frame_roles": list(C.FRAME_ROLES),
            "two_fresh_recovery_replays": True,
        },
        "terminal_recovery": recovery,
        "sanitized_source": sanitized,
        "oracle": {
            "summary": {
                "path": "/v248-evaluator-only/nonexistent-summary.json",
                "sha256": "c" * 64,
                "bytes": 123,
            },
            "access": "post_progress_seal_evaluator_only",
            "fields": ["episode_records.idx", "seed", "steps", "events"],
            "success_values_accessed": False,
        },
        "models": {
            "pi05": {
                "repo_id": "lerobot/pi05_libero_finetuned",
                "revision": policy_revision,
                "local_root": policy_root.resolve().as_posix(),
                **{
                    role: pi_by_path[path]
                    for role, path in C.PI05_MEMBER_PATHS.items()
                },
            },
            "tokenizer": {
                "repo_id": "google/paligemma-3b-pt-224",
                "revision": tokenizer_revision,
                "local_root": tokenizer_root.resolve().as_posix(),
                "files": tokenizer_refs,
            },
            "sam3": {
                "model_id": "facebook/sam3",
                "revision": "fixture-revision",
                "checkpoint": file_ref(external["sam3.safetensors"]),
                "clip_tokenizer": {
                    "root": clip_root.resolve().as_posix(),
                    "tree_sha256": C.file_tree_sha256(clip_files),
                    "files": clip_files,
                },
                "config_sha256": "d" * 64,
                "processor_sha256": "e" * 64,
            },
        },
        "templates": {
            "cream_template_image": file_ref(external["cream.jpg"]),
            "cream_template_metadata": file_ref(external["cream.json"]),
            "semantic_config": file_ref(external["semantic.json"]),
        },
        "sources": sources,
        "runtime": runtime,
        "semantic_contract": {
            "manifest_schema": C.RGB_MANIFEST_SCHEMA,
            "result_schema": C.RGB_RESULT_SCHEMA,
            "complete_schema": C.RGB_COMPLETE_SCHEMA,
            "allowed_frame_roles": list(C.FRAME_ROLES),
            "recovered_endpoint": {
                "phi_only": True,
                "dynamics_transition_eligible": False,
            },
        },
    }
    protocol_sha256 = C.canonical_sha256(C.campaign_protocol_payload(registration))
    registration["campaign"] = {
        "schema": C.CAMPAIGN_SCHEMA,
        "campaign_id": "v248-gate0-" + "1" * 16,
        "protocol_sha256": protocol_sha256,
        "projection_sha256": "2" * 64,
        "manifest_relative_path": "CAMPAIGN.json",
        "registration_slot": "chain3_8000_screen.json",
    }
    registration["self_sha256"] = C.canonical_sha256(registration)
    path = root / "REGISTRATION.json"
    path.write_bytes(C.canonical_bytes(registration))
    return registration, path


def _episode_specs(seed_start: int, phase: str) -> list[dict[str, Any]]:
    episode_ids = C.SCREEN_EPISODES[seed_start] if phase == "screen" else range(96)
    return [
        {
            "episode_id": episode_id,
            "env_seed": seed_start + episode_id,
            "stored_rows": 65 if seed_start == 8100 and episode_id == 77 else 75,
            "mode": (
                "recover_last_action"
                if seed_start == 8100 and episode_id == 77
                else "ordinary"
            ),
        }
        for episode_id in episode_ids
    ]


def _terminal_recovery(seed_start: int) -> dict[str, Any]:
    if seed_start == 8000:
        return {
            "enabled": False,
            "episode_id": None,
            "env_seed": None,
            "stored_chunks": None,
            "stored_steps": None,
            "source_terminal_t": None,
            "missing_chunk_t": None,
            "missing_raw_sigma_choice": None,
            "sigma_rng": None,
            "sampler": None,
            "phi_only": True,
            "dynamics_transition_eligible": False,
        }
    counts = [75] * 96
    counts[77] = 66
    return {
        "enabled": True,
        "episode_id": 77,
        "env_seed": 8177,
        "stored_chunks": 65,
        "stored_steps": 650,
        "source_terminal_t": 651,
        "missing_chunk_t": 650,
        "missing_raw_sigma_choice": 0.6,
        "sigma_rng": {
            "algorithm": "numpy.random.Generator(PCG64)",
            "seed": 8100,
            "choices": [0.0, 0.6, 1.5, 3.0],
            "source_proposed_chunks_by_episode": counts[:78],
            "missing_draw_offset_within_episode": 65,
        },
        "sampler": {
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
        },
        "phi_only": True,
        "dynamics_transition_eligible": False,
    }


def build_campaign_bundle(
    root: Path, sanitized: dict[str, str]
) -> tuple[Path, str, dict[str, dict[str, Any]]]:
    base, _standalone_path = build_registration(root, sanitized)
    base_core = {
        key: copy.deepcopy(value)
        for key, value in base.items()
        if key not in {"campaign", "self_sha256"}
    }
    cores: dict[str, dict[str, Any]] = {}
    for phase in ("screen", "confirm"):
        for seed_start in (8000, 8100):
            panel_id = C.PANEL_IDS[seed_start]
            slot = f"{panel_id}_{phase}.json"
            core = copy.deepcopy(base_core)
            core["registration_id"] = f"cpu_fixture_{panel_id}_{phase}"
            core["phase"] = phase
            core["panel"] = {
                "panel_id": panel_id,
                "seed_start": seed_start,
                "source_episode_count": 96,
                "episodes": _episode_specs(seed_start, phase),
            }
            core["terminal_recovery"] = _terminal_recovery(seed_start)
            cores[slot] = core
    protocols = [C.campaign_protocol_payload(core) for core in cores.values()]
    assert all(protocol == protocols[0] for protocol in protocols)
    protocol_sha256 = C.canonical_sha256(protocols[0])
    core_sha256s = {
        slot: C.canonical_sha256(core) for slot, core in cores.items()
    }
    target_root = (root / "formal_target").resolve()
    intent = {
        "schema": C.CAMPAIGN_INTENT_SCHEMA,
        "input_closure_sha256": "3" * 64,
        "protocol_sha256": protocol_sha256,
        "registration_core_sha256s": core_sha256s,
        "screen_episode_ids": {
            C.PANEL_IDS[seed]: list(C.SCREEN_EPISODES[seed])
            for seed in (8000, 8100)
        },
        "confirm_episode_ids": {
            C.PANEL_IDS[seed]: list(range(96)) for seed in (8000, 8100)
        },
        "automatic_expansion_rule": C.AUTOMATIC_CONFIRM_EXPANSION_RULE,
        "confirm_authorization_required": True,
        "target_root": target_root.as_posix(),
        "target_output_slots": C.TARGET_OUTPUT_SLOTS,
        "dag": C.CAMPAIGN_DAG,
    }
    campaign_id = f"v248-gate0-{C.canonical_sha256(intent)[:16]}"
    projection = {
        "schema": C.CAMPAIGN_PROJECTION_SCHEMA,
        "intent": intent,
        "campaign_id": campaign_id,
    }
    projection_sha256 = C.canonical_sha256(projection)
    registrations: dict[str, dict[str, Any]] = {}
    for slot, core in cores.items():
        registration = {
            **core,
            "campaign": {
                "schema": C.CAMPAIGN_SCHEMA,
                "campaign_id": campaign_id,
                "protocol_sha256": protocol_sha256,
                "projection_sha256": projection_sha256,
                "manifest_relative_path": "CAMPAIGN.json",
                "registration_slot": slot,
            },
        }
        registration["self_sha256"] = C.canonical_sha256(registration)
        registrations[slot] = registration

    bundle_root = root / "freeze"
    registration_root = bundle_root / "registrations"
    registration_root.mkdir(parents=True)
    registration_seals = {}
    for slot, registration in registrations.items():
        path = write(registration_root / slot, C.canonical_bytes(registration))
        registration_seals[slot] = C.sha256_file(path)
    campaign_body = {
        "schema": C.CAMPAIGN_MANIFEST_SCHEMA,
        "status": "frozen_before_any_formal_rgb",
        "created_utc": "2026-09-01T00:00:00Z",
        "campaign_id": campaign_id,
        "protocol_sha256": protocol_sha256,
        "projection_sha256": projection_sha256,
        "projection": projection,
        "screen_episode_ids": intent["screen_episode_ids"],
        "confirm_episode_ids": intent["confirm_episode_ids"],
        "automatic_expansion_rule": C.AUTOMATIC_CONFIRM_EXPANSION_RULE,
        "confirm_authorization_required": True,
        "registrations": {
            slot: {
                "file_sha256": registration_seals[slot],
                "self_sha256": registrations[slot]["self_sha256"],
                "core_sha256": core_sha256s[slot],
                "panel_id": registrations[slot]["panel"]["panel_id"],
                "seed_start": registrations[slot]["panel"]["seed_start"],
                "phase": registrations[slot]["phase"],
            }
            for slot in C.REGISTRATION_SLOTS
        },
    }
    campaign = {
        **campaign_body,
        "body_sha256": C.canonical_sha256(campaign_body),
    }
    campaign_path = write(bundle_root / "CAMPAIGN.json", C.canonical_bytes(campaign))
    result_path = write(bundle_root / "FREEZE_RESULT.json", b"fixture-result\n")
    producer_path = write(bundle_root / "PRODUCER.py", b"fixture-producer\n")
    complete = {
        "schema": C.FREEZE_COMPLETE_SCHEMA,
        "status": "atomic_success",
        "atomic_commit": True,
        "input_closure_sha256": intent["input_closure_sha256"],
        "materialization_complete_sha256": "4" * 64,
        "campaign_manifest_sha256": C.sha256_file(campaign_path),
        "campaign_body_sha256": campaign["body_sha256"],
        "campaign_id": campaign_id,
        "projection_sha256": projection_sha256,
        "result_sha256": C.sha256_file(result_path),
        "producer_sha256": C.sha256_file(producer_path),
        "registration_sha256s": registration_seals,
    }
    complete_path = write(bundle_root / "COMPLETE.json", C.canonical_bytes(complete))
    return bundle_root, C.sha256_file(complete_path), registrations


def build_sanitized(
    root: Path, *, complete_panel: bool = False
) -> tuple[Path, dict[str, str]]:
    counts = [
        75
        if complete_panel or episode in C.SCREEN_EPISODES[8000]
        else 1
        for episode in range(96)
    ]
    episodes = []
    times = []
    for episode_id, count in enumerate(counts):
        episodes.extend([episode_id] * count)
        times.extend(range(0, count * C.C, C.C))
    rows = len(episodes)
    tape = {
        "z": torch.zeros((rows, C.LATENT_DIM), dtype=torch.float32),
        "u": torch.zeros((rows, C.C, 7), dtype=torch.float32),
        "z_next": torch.zeros((rows, C.LATENT_DIM), dtype=torch.float32),
        "episode": torch.tensor(episodes, dtype=torch.int64),
        "t": torch.tensor(times, dtype=torch.int64),
        "sigma": torch.zeros(rows, dtype=torch.float32),
        "task": C.TASK,
        "c": C.C,
        "latent_dim": C.LATENT_DIM,
        "proprio_dim": C.PROPRIO_DIM,
        "proprio_keys": list(C.PROPRIO_KEYS),
    }
    artifact = root / "sanitized"
    artifact.mkdir(parents=True)
    tape_path = artifact / "sanitized_tape.pt"
    torch.save(tape, tape_path)
    producer = write(
        artifact / "PRODUCER.py",
        (REPO / "scripts" / "sanitize_v245_deploy_tape.py").read_bytes(),
    )
    source_sha = "f" * 64
    tensor_hashes = {
        name: R.tensor_sha256(tape[name]) for name in R.SEQUENCE_TENSOR_FIELDS
    }
    metadata = {
        "task": C.TASK,
        "c": C.C,
        "latent_dim": C.LATENT_DIM,
        "proprio_dim": C.PROPRIO_DIM,
        "proprio_keys": list(C.PROPRIO_KEYS),
    }
    result = {
        "schema": R.SANITIZER_RESULT_SCHEMA,
        "status": "PASS",
        "utc": "2026-09-01T000000Z",
        "source": {
            "path": "/fixture/private/tape.pt",
            "sha256": source_sha,
            "bytes": 1,
            "summary_opened": False,
        },
        "field_policy": {
            "kept": sorted(R.SANITIZED_TAPE_FIELDS),
            "dropped_names_without_value_access": ["success"],
            "declared_outcome_fields": ["success"],
        },
        "validation": {
            "rows": rows,
            "latent_dim": C.LATENT_DIM,
            "episodes": 96,
            "episode_ids": list(range(96)),
            "episode_row_counts": counts,
            "c": C.C,
        },
        "metadata": metadata,
        "tensor_sha256": tensor_hashes,
        "metadata_sha256": C.canonical_sha256(metadata),
        "sanitized_tape_sha256": C.sha256_file(tape_path),
        "producer_snapshot_sha256": C.sha256_file(producer),
        "git_head": None,
        "torch_version": torch.__version__,
    }
    result_path = artifact / "RESULT.json"
    result_path.write_bytes(C.canonical_bytes(result))
    complete = {
        "schema": R.SANITIZER_COMPLETE_SCHEMA,
        "status": "atomic_success",
        "atomic_commit": True,
        "source_tape_sha256": source_sha,
        "sanitized_tape_sha256": C.sha256_file(tape_path),
        "result_sha256": C.sha256_file(result_path),
        "producer_snapshot_sha256": C.sha256_file(producer),
    }
    complete_path = artifact / "COMPLETE.json"
    complete_path.write_bytes(C.canonical_bytes(complete))
    return tape_path, {
        "tape_sha256": C.sha256_file(tape_path),
        "result_sha256": C.sha256_file(result_path),
        "complete_sha256": C.sha256_file(complete_path),
        "producer_sha256": C.sha256_file(producer),
        "source_tape_sha256": source_sha,
    }


def fixture(root: Path) -> tuple[R.PreparedReplay, Path]:
    tape_path, sanitized = build_sanitized(root)
    bundle_root, freeze_complete_sha256, _registrations = build_campaign_bundle(
        root, sanitized
    )
    prepared = R.prepare_inputs(
        sanitized_tape=tape_path,
        freeze_bundle_root=bundle_root,
        expected_freeze_complete_sha256=freeze_complete_sha256,
        registration_slot="chain3_8000_screen.json",
    )
    return prepared, bundle_root


def test_registration_closure_anchor_oracle_and_pre_reset_drift() -> None:
    with tempfile.TemporaryDirectory(prefix="v248-reg-test-") as raw:
        root = Path(raw)
        prepared, bundle_root = fixture(root)
        assert not Path(
            prepared.registration["oracle"]["summary"]["path"]
        ).exists()
        C.load_and_validate_registration(
            prepared.registration_path, expected_source_role="formal_replay"
        )
        C.load_and_validate_registration(
            prepared.registration_path, expected_source_role="semantic_producer"
        )
        expect_error(
            lambda: C.load_and_validate_registration(
                prepared.registration_path, expected_source_role="semantic_evaluator"
            ),
            "No such file or directory",
        )
        extra = copy.deepcopy(prepared.registration)
        extra["unregistered"] = True
        extra["self_sha256"] = C.canonical_sha256(
            {key: value for key, value in extra.items() if key != "self_sha256"}
        )
        expect_error(lambda: C.validate_registration(extra), "field closure")
        wrong_panel = copy.deepcopy(prepared.registration)
        wrong_panel["panel"]["episodes"] = wrong_panel["panel"]["episodes"][:-1]
        wrong_panel["self_sha256"] = C.canonical_sha256(
            {key: value for key, value in wrong_panel.items() if key != "self_sha256"}
        )
        expect_error(lambda: C.validate_registration(wrong_panel), "episode set")

        semantic_config = Path(
            prepared.registration["templates"]["semantic_config"]["path"]
        )
        semantic_config.write_bytes(b"drift\n")
        reset_count = 0
        expect_error(
            lambda: R.revalidate_static_inputs(prepared), "semantic_config"
        )
        assert reset_count == 0
        assert not prepared.output_path.exists()
        assert bundle_root.is_dir()


def _robot_state() -> dict[str, Any]:
    return {
        "eef": {"pos": np.zeros(3), "quat": np.zeros(4)},
        "gripper": {"qpos": np.zeros(2), "qvel": np.zeros(2)},
        "joints": {"pos": np.zeros(7), "vel": np.zeros(7)},
    }


def fake_observation(t: int, perturb: int = 0) -> dict[str, Any]:
    value = np.uint8((t + perturb) % 251)
    return {
        "clock": t,
        "robot_state": _robot_state(),
        "pixels": {
            "image": np.full((3, 4, 3), value, dtype=np.uint8),
            "image2": np.full((3, 4, 3), value + np.uint8(1), dtype=np.uint8),
        },
    }


class FakeEnv:
    task_description = "fixture task"

    def __init__(self, perturb: int = 0) -> None:
        self.t = 0
        self.perturb = perturb

    def reset(self, seed: int):
        assert seed == 8177
        self.t = 0
        return fake_observation(0, self.perturb), {}

    def step(self, _action: np.ndarray):
        self.t += 1
        stopped = self.t == 651
        return fake_observation(self.t, self.perturb), 0.0, stopped, False, {}

    def close(self) -> None:
        return None


class FakeRunner:
    def __init__(self, signed_zero: bool = False) -> None:
        self.policy = object()
        self.signed_zero = signed_zero
        self.proposals = 0

    def reset(self) -> None:
        return None

    def _chunk(self) -> torch.Tensor:
        self.proposals += 1
        value = torch.zeros((1, C.C, 7), dtype=torch.float32)
        if self.signed_zero and self.proposals == 1:
            value[0, 0, 0] = -0.0
        return value

    def sample_chunk(self, _obs: Any, _task: str) -> torch.Tensor:
        return self._chunk()

    def _obs_to_policy_batch(self, obs: dict[str, Any], _task: str) -> dict[str, Any]:
        return {"clock": obs["clock"], "runner": self}

    def chunk_to_env(self, chunk: torch.Tensor) -> np.ndarray:
        return chunk.detach().cpu().numpy()


def fake_sample_chunks(
    _policy: Any, observation: dict[str, Any], _n: int, **_kwargs: Any
) -> torch.Tensor:
    return observation["runner"]._chunk()


def recovery_registration() -> dict[str, Any]:
    counts = [8] + [1] * 76 + [66]
    return {
        "terminal_recovery": {
            "enabled": True,
            "episode_id": 77,
            "env_seed": 8177,
            "stored_chunks": 65,
            "stored_steps": 650,
            "source_terminal_t": 651,
            "missing_chunk_t": 650,
            "missing_raw_sigma_choice": 0.6,
            "sigma_rng": {
                "algorithm": "numpy.random.Generator(PCG64)",
                "seed": 8100,
                "choices": [0.0, 0.6, 1.5, 3.0],
                "source_proposed_chunks_by_episode": counts,
                "missing_draw_offset_within_episode": 65,
            },
            "sampler": {},
            "phi_only": True,
            "dynamics_transition_eligible": False,
        }
    }


def recovery_tape() -> dict[str, torch.Tensor]:
    registration = recovery_registration()
    sigmas = R.reconstruct_recovery_sigmas(registration)[:65]
    return {
        "z": torch.zeros((65, C.LATENT_DIM), dtype=torch.float32),
        "z_next": torch.zeros((65, C.LATENT_DIM), dtype=torch.float32),
        "u": torch.zeros((65, C.C, 7), dtype=torch.float32),
        "sigma": torch.tensor(sigmas, dtype=torch.float32),
    }


def zero_legacy(_obs: Any, _task: str) -> torch.Tensor:
    return torch.zeros(C.LATENT_DIM, dtype=torch.float32)


def test_recovery_requires_all_65_bitwise_chunks_before_missing() -> None:
    legacy_drift = recovery_tape()
    legacy_drift["z"][0, 0] = 1.0
    legacy_runner = FakeRunner()
    legacy_env = FakeEnv()
    expect_error(
        lambda: R.run_recovery_pass(
            runner=legacy_runner,
            env_factory=lambda: legacy_env,
            sample_chunks_fn=fake_sample_chunks,
            tape=legacy_drift,
            row_indices=list(range(65)),
            registration=recovery_registration(),
            legacy_encoder=zero_legacy,
        ),
        "legacy2073/z",
    )
    assert legacy_runner.proposals == 0
    assert legacy_env.t == 0

    tape = recovery_tape()
    tape["u"][0, 0, 0] = -0.0
    runner = FakeRunner(signed_zero=False)
    error = expect_error(
        lambda: R.run_recovery_pass(
            runner=runner,
            env_factory=FakeEnv,
            sample_chunks_fn=fake_sample_chunks,
            tape=tape,
            row_indices=list(range(65)),
            registration=recovery_registration(),
            legacy_encoder=zero_legacy,
        ),
        "raw-byte mismatch",
    )
    assert isinstance(error, C.Gate0ContractError)
    assert runner.proposals == 1

    clean = recovery_tape()
    first_runner = FakeRunner()
    second_runner = FakeRunner()
    first = R.run_recovery_pass(
        runner=first_runner,
        env_factory=FakeEnv,
        sample_chunks_fn=fake_sample_chunks,
        tape=clean,
        row_indices=list(range(65)),
        registration=recovery_registration(),
        legacy_encoder=zero_legacy,
    )
    second = R.run_recovery_pass(
        runner=second_runner,
        env_factory=FakeEnv,
        sample_chunks_fn=fake_sample_chunks,
        tape=clean,
        row_indices=list(range(65)),
        registration=recovery_registration(),
        legacy_encoder=zero_legacy,
    )
    assert first_runner.proposals == second_runner.proposals == 66
    assert len(first.frames) == 67 and first.frames[-1].t == 651
    assert first.frames[-1].boundary_role == "recovered_endpoint"
    R.compare_recovery_passes(first, second)
    second.frames[-1].raw_hashes["agentview"] = "0" * 64
    expect_error(
        lambda: R.compare_recovery_passes(first, second), "raw RGB drift"
    )


def test_manifest_is_exact_and_semantic_free() -> None:
    row = {
        "schema": C.RGB_MANIFEST_SCHEMA,
        "frame_id": "chain3_8100:screen:ep77:t0651",
        "panel_id": "chain3_8100",
        "phase": "screen",
        "task": C.TASK,
        "episode_id": 77,
        "env_seed": 8177,
        "frame_index": 0,
        "t": 651,
        "source": {
            "tape_sha256": "a" * 64,
            "source_row_index": None,
            "boundary_role": "recovered_endpoint",
        },
        "images": {
            camera: {
                "path": f"images/{camera}.jpg",
                "sha256": "b" * 64,
                "raw_array_sha256": "c" * 64,
                "width": 4,
                "height": 3,
            }
            for camera in C.CAMERAS
        },
        "proprio": {
            "reference": "cross_run_exact",
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
        },
    }
    assert R.validate_manifest_row(row) == row
    for forbidden in ("done", "is_success", "terminal", "events"):
        injected = copy.deepcopy(row)
        injected[forbidden] = False
        expect_error(lambda value=injected: R.validate_manifest_row(value))
    assert not any(
        key in json.dumps(row, sort_keys=True)
        for key in ('"done"', '"is_success"', '"terminal"', '"events"')
    )


def test_atomic_publish_is_no_clobber_and_cleans_failures() -> None:
    with tempfile.TemporaryDirectory(prefix="v248-atomic-test-") as raw:
        root = Path(raw)
        failed = root / "failed"

        def failing_builder(stage: Path) -> None:
            write(stage / "partial", b"partial")
            raise RuntimeError("injected")

        expect_error(
            lambda: R.atomic_publish_directory(failed, failing_builder, lambda _p: None),
            "injected",
        )
        assert not failed.exists()
        assert not list(root.glob(".failed.partial-*"))

        output = root / "published"

        def good_builder(stage: Path) -> None:
            write(stage / "marker", b"original")

        R.atomic_publish_directory(output, good_builder, lambda p: (p / "marker").read_bytes())
        assert (output / "marker").read_bytes() == b"original"
        expect_error(
            lambda: R.atomic_publish_directory(
                output, lambda p: write(p / "marker", b"replacement"), lambda _p: None
            ),
            "overwrite",
        )
        assert (output / "marker").read_bytes() == b"original"

        raced = root / "raced"

        def race_validator(_stage: Path) -> None:
            raced.mkdir()
            write(raced / "owner", b"other")

        expect_error(
            lambda: R.atomic_publish_directory(raced, good_builder, race_validator),
            "appeared",
        )
        assert (raced / "owner").read_bytes() == b"other"


def test_producer_snapshot_must_match_registration() -> None:
    with tempfile.TemporaryDirectory(prefix="v248-producer-seal-test-") as raw:
        root = Path(raw)
        prepared, _registration_path = fixture(root)
        prepared.registration["sources"]["formal_replay"]["sha256"] = "0" * 64
        replay = R.PanelReplay(
            frames=[],
            published_environment_steps=0,
            recovery_audit_environment_steps=0,
            recovery_audit={},
        )
        expect_error(
            lambda: R.publish_replay(prepared, replay),
            "producer snapshot does not equal registered formal replay source",
        )
        assert not prepared.output_path.exists()
        assert not list(root.glob(".formal_rgb.partial-*"))


def test_validate_only_subprocess_blocks_simulator_imports() -> None:
    with tempfile.TemporaryDirectory(prefix="v248-validation-test-") as raw:
        root = Path(raw)
        prepared, bundle_root = fixture(root)
        blocker = root / "blocker"
        marker = root / "forbidden-import.txt"
        sitecustomize = f"""
import importlib.abc, pathlib, sys
MARKER = pathlib.Path({str(marker)!r})
PREFIXES = ('libero', 'robosuite', 'mujoco', 'lerobot')
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == p or fullname.startswith(p + '.') for p in PREFIXES):
            MARKER.write_text(fullname)
            raise ImportError('forbidden validation-only import: ' + fullname)
        return None
sys.meta_path.insert(0, Blocker())
"""
        write(blocker / "sitecustomize.py", sitecustomize.encode())
        python = Path(sys.executable)
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join([str(blocker), str(REPO)])
        command = [
            str(python),
            str(REPO / "scripts" / "replay_v248_gate0_rgb.py"),
            "--sanitized-tape",
            str(prepared.tape_path),
            "--freeze-bundle",
            str(bundle_root),
            "--expected-freeze-complete-sha256",
            prepared.expected_freeze_complete_sha256,
            "--registration-slot",
            prepared.registration_slot,
            "--validate-only",
        ]
        process = subprocess.run(
            command,
            cwd=REPO,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert process.returncode == 0, (process.stdout, process.stderr)
        assert '"simulator_imported": false' in process.stdout.lower()
        assert not marker.exists()
        assert not prepared.target_root.exists()


def _rewrite_json(path: Path, value: Any) -> None:
    path.write_bytes(C.canonical_bytes(value))


def _reseal_bundle(bundle_root: Path) -> str:
    campaign_path = bundle_root / "CAMPAIGN.json"
    campaign = json.loads(campaign_path.read_text())
    campaign_body = {
        key: value for key, value in campaign.items() if key != "body_sha256"
    }
    campaign["body_sha256"] = C.canonical_sha256(campaign_body)
    _rewrite_json(campaign_path, campaign)
    complete_path = bundle_root / "COMPLETE.json"
    complete = json.loads(complete_path.read_text())
    complete["campaign_manifest_sha256"] = C.sha256_file(campaign_path)
    complete["campaign_body_sha256"] = campaign["body_sha256"]
    complete["campaign_id"] = campaign["campaign_id"]
    complete["projection_sha256"] = campaign["projection_sha256"]
    for slot in C.REGISTRATION_SLOTS:
        complete["registration_sha256s"][slot] = C.sha256_file(
            bundle_root / "registrations" / slot
        )
    _rewrite_json(complete_path, complete)
    return C.sha256_file(complete_path)


def test_formal_authority_rejects_unanchored_wrong_phase_and_legacy_cli() -> None:
    with tempfile.TemporaryDirectory(prefix="v248-authority-test-") as raw:
        root = Path(raw)
        tape_path, sanitized = build_sanitized(root)
        bundle_root, anchor, _registrations = build_campaign_bundle(root, sanitized)
        expect_error(
            lambda: R.prepare_inputs(
                sanitized_tape=tape_path,
                freeze_bundle_root=bundle_root,
                expected_freeze_complete_sha256="0" * 64,
                registration_slot="chain3_8000_screen.json",
            ),
            "differs from external SHA-256 anchor",
        )
        expect_error(
            lambda: R.prepare_inputs(
                sanitized_tape=tape_path,
                freeze_bundle_root=bundle_root,
                expected_freeze_complete_sha256=anchor,
                registration_slot="chain3_9999_screen.json",
            ),
            "not a member",
        )
        expect_error(
            lambda: R.prepare_inputs(
                sanitized_tape=tape_path,
                freeze_bundle_root=bundle_root,
                expected_freeze_complete_sha256=anchor,
                registration_slot="chain3_8000_confirm.json",
            ),
            "confirm-authorization COMPLETE",
        )
        expect_error(
            lambda: R.prepare_inputs(
                sanitized_tape=tape_path,
                freeze_bundle_root=bundle_root,
                expected_freeze_complete_sha256=anchor,
                registration_slot="chain3_8000_screen.json",
                expected_confirm_authorization_complete_sha256="a" * 64,
            ),
            "screen replay must not consume",
        )
        process = subprocess.run(
            [
                sys.executable,
                str(REPO / "scripts" / "replay_v248_gate0_rgb.py"),
                "--sanitized-tape",
                str(tape_path),
                "--registration",
                str(bundle_root / "registrations/chain3_8000_screen.json"),
                "--expected-registration-sha256",
                "0" * 64,
                "--panel",
                "chain3_8000",
                "--phase",
                "screen",
                "--out",
                str(root / "attacker-output"),
                "--validate-only",
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        assert process.returncode != 0
        assert "--freeze-bundle" in process.stderr
        assert not (root / "attacker-output").exists()


def test_confirm_authorization_rejection_precedes_all_target_io() -> None:
    with tempfile.TemporaryDirectory(prefix="v248-confirm-before-target-io-") as raw:
        root = Path(raw)
        tape_path, sanitized = build_sanitized(root)
        bundle_root, anchor, _registrations = build_campaign_bundle(root, sanitized)
        target_root = (root / "formal_target").resolve()
        target_calls: list[str] = []

        def is_target(raw_path: Any) -> bool:
            try:
                candidate = Path(os.path.abspath(os.fspath(raw_path)))
            except TypeError:
                return False
            return candidate == target_root or candidate.is_relative_to(target_root)

        def trap(owner: Any, name: str) -> mock._patch:
            original = getattr(owner, name)

            def trapped(
                raw_path: Any,
                *args: Any,
                _name: str = name,
                _original: Any = original,
                **kwargs: Any,
            ) -> Any:
                if is_target(raw_path):
                    target_calls.append(f"{_name}:{raw_path}")
                    raise AssertionError(
                        f"confirm target I/O before authorization: {_name}:{raw_path}"
                    )
                return _original(raw_path, *args, **kwargs)

            return mock.patch.object(owner, name, side_effect=trapped)

        with (
            trap(os, "open"),
            trap(os, "stat"),
            trap(os, "lstat"),
            trap(os, "scandir"),
            trap(os.path, "lexists"),
        ):
            expect_error(
                lambda: R.prepare_inputs(
                    sanitized_tape=tape_path,
                    freeze_bundle_root=bundle_root,
                    expected_freeze_complete_sha256=anchor,
                    registration_slot="chain3_8000_confirm.json",
                ),
                "confirm-authorization COMPLETE",
            )
            assert target_calls == []

            rejected_sha = "f" * 64
            with mock.patch.object(
                R,
                "_validate_confirm_authorization",
                side_effect=C.Gate0ContractError("synthetic invalid authorization"),
            ) as validate_authorization:
                expect_error(
                    lambda: R.prepare_inputs(
                        sanitized_tape=tape_path,
                        freeze_bundle_root=bundle_root,
                        expected_freeze_complete_sha256=anchor,
                        registration_slot="chain3_8000_confirm.json",
                        expected_confirm_authorization_complete_sha256=rejected_sha,
                    ),
                    "synthetic invalid authorization",
                )
            validate_authorization.assert_called_once()
            assert validate_authorization.call_args.args[1] == rejected_sha
            assert target_calls == []


def test_campaign_mixing_and_post_screen_refreeze_are_rejected() -> None:
    with tempfile.TemporaryDirectory(prefix="v248-mixing-test-") as raw:
        root = Path(raw)
        tape_path, sanitized = build_sanitized(root)
        bundle_root, anchor, _registrations = build_campaign_bundle(root, sanitized)
        prepared = R.prepare_inputs(
            sanitized_tape=tape_path,
            freeze_bundle_root=bundle_root,
            expected_freeze_complete_sha256=anchor,
            registration_slot="chain3_8000_screen.json",
        )

        # A post-screen replacement can be internally resealed, but it cannot
        # replace the externally held genesis digest in PreparedReplay.
        campaign_path = bundle_root / "CAMPAIGN.json"
        campaign = json.loads(campaign_path.read_text())
        campaign["created_utc"] = "2026-09-01T00:00:01Z"
        _rewrite_json(campaign_path, campaign)
        _reseal_bundle(bundle_root)
        expect_error(
            lambda: R.revalidate_static_inputs(prepared),
            "differs from external SHA-256 anchor",
        )

    with tempfile.TemporaryDirectory(prefix="v248-cross-campaign-test-") as raw:
        root = Path(raw)
        tape_path, sanitized = build_sanitized(root)
        bundle_root, _anchor, _registrations = build_campaign_bundle(root, sanitized)
        slot = "chain3_8000_screen.json"
        registration_path = bundle_root / "registrations" / slot
        registration = json.loads(registration_path.read_text())
        original_protocol = registration["campaign"]["protocol_sha256"]
        registration["campaign"]["campaign_id"] = "v248-gate0-" + "f" * 16
        body = {
            key: value for key, value in registration.items() if key != "self_sha256"
        }
        registration["self_sha256"] = C.canonical_sha256(body)
        _rewrite_json(registration_path, registration)
        campaign_path = bundle_root / "CAMPAIGN.json"
        campaign = json.loads(campaign_path.read_text())
        campaign["registrations"][slot]["file_sha256"] = C.sha256_file(
            registration_path
        )
        campaign["registrations"][slot]["self_sha256"] = registration[
            "self_sha256"
        ]
        _rewrite_json(campaign_path, campaign)
        new_anchor = _reseal_bundle(bundle_root)
        assert registration["campaign"]["protocol_sha256"] == original_protocol
        expect_error(
            lambda: R.prepare_inputs(
                sanitized_tape=tape_path,
                freeze_bundle_root=bundle_root,
                expected_freeze_complete_sha256=new_anchor,
                registration_slot=slot,
            ),
            "does not point back to campaign",
        )


def test_resealed_manifest_output_attack_and_unsafe_parents_are_rejected() -> None:
    with tempfile.TemporaryDirectory(prefix="v248-manifest-attack-") as raw:
        root = Path(raw)
        tape_path, sanitized = build_sanitized(root)
        bundle_root, _anchor, _registrations = build_campaign_bundle(root, sanitized)
        campaign_path = bundle_root / "CAMPAIGN.json"
        campaign = json.loads(campaign_path.read_text())
        intent = campaign["projection"]["intent"]
        intent["target_output_slots"]["screen_rgb_complete_by_panel"][
            "chain3_8000"
        ] = "screen/rgb/chain3_8000/nested/COMPLETE.json"
        campaign_id = f"v248-gate0-{C.canonical_sha256(intent)[:16]}"
        campaign["projection"]["campaign_id"] = campaign_id
        campaign["campaign_id"] = campaign_id
        campaign["projection_sha256"] = C.canonical_sha256(campaign["projection"])
        _rewrite_json(campaign_path, campaign)
        new_anchor = _reseal_bundle(bundle_root)
        expect_error(
            lambda: R.prepare_inputs(
                sanitized_tape=tape_path,
                freeze_bundle_root=bundle_root,
                expected_freeze_complete_sha256=new_anchor,
                registration_slot="chain3_8000_screen.json",
            ),
            "target output slots mismatch",
        )

    with tempfile.TemporaryDirectory(prefix="v248-output-parent-test-") as raw:
        root = Path(raw)
        tape_path, sanitized = build_sanitized(root)
        bundle_root, anchor, _registrations = build_campaign_bundle(root, sanitized)
        target_root = root / "formal_target"
        target_root.mkdir()
        outside = root / "outside"
        outside.mkdir()
        (target_root / "screen").symlink_to(outside, target_is_directory=True)
        expect_error(
            lambda: R.prepare_inputs(
                sanitized_tape=tape_path,
                freeze_bundle_root=bundle_root,
                expected_freeze_complete_sha256=anchor,
                registration_slot="chain3_8000_screen.json",
            ),
            "symlink component",
        )
        assert not (outside / "rgb").exists()

    with tempfile.TemporaryDirectory(prefix="v248-existing-output-test-") as raw:
        root = Path(raw)
        prepared, _bundle_root = fixture(root)
        prepared.output_path.mkdir(parents=True)
        expect_error(lambda: R.revalidate_static_inputs(prepared), "output already exists")


def test_tokenizer_loader_requests_exact_frozen_revision() -> None:
    with tempfile.TemporaryDirectory(prefix="v248-tokenizer-revision-test-") as raw:
        prepared, _bundle_root = fixture(Path(raw))
        tokenizer = prepared.registration["models"]["tokenizer"]
        calls: list[dict[str, Any]] = []

        def fake_snapshot_download(**kwargs: Any) -> str:
            calls.append(kwargs)
            return tokenizer["local_root"]

        resolved = R._resolve_registered_tokenizer_snapshot(
            tokenizer, fake_snapshot_download
        )
        assert resolved == Path(tokenizer["local_root"]).resolve()
        assert calls == [
            {
                "repo_id": tokenizer["repo_id"],
                "revision": tokenizer["revision"],
                "local_files_only": True,
            }
        ]


SEMANTIC_RUNTIME_ROLES = (
    "transformers_sam3_video_configuration",
    "transformers_sam3_video_modeling",
    "transformers_sam3_video_processing",
    "transformers_sam3_image_processing",
    "transformers_sam2_video_processing",
    "transformers_clip_tokenization",
)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    write(path, b"".join(C.canonical_bytes(row) for row in rows))


def _synthetic_progress_frame(
    *,
    seed_start: int,
    phase: str,
    episode_id: int,
    rgb_index: int,
    local_index: int,
    t: int,
    role: str,
) -> dict[str, Any]:
    recovered = role == "recovered_endpoint"
    return {
        "schema": P.FRAME_SCHEMA,
        "panel_id": C.PANEL_IDS[seed_start],
        "phase": phase,
        "task": C.TASK,
        "episode_id": episode_id,
        "env_seed": seed_start + episode_id,
        "rgb_frame_index": rgb_index,
        "episode_frame_index": local_index,
        "t": t,
        "frame_id": (
            f"{C.PANEL_IDS[seed_start]}:{phase}:"
            f"ep{episode_id:02d}:t{t:04d}"
        ),
        "boundary_role": role,
        "observation_hashes": {
            "agentview_sha256": "1" * 64,
            "eye_in_hand_sha256": "2" * 64,
        },
        "atoms": {"can_ge1": False, "can_ge2": False, "cream": False},
        "scalar": 0,
        "reliability": {
            "can_ge1": True,
            "can_ge2": True,
            "cream": True,
            "scalar": True,
        },
        "transitions": {
            "can_ge1": False,
            "can_ge2": False,
            "cream": False,
        },
        "causal_state": {
            "sticky_can_count": 0,
            "cream_mode": "search",
            "can_used_carry": False,
            "cream_used_carry": False,
        },
        "evidence": {
            "basket_bbox_xyxy": [100.0, 100.0, 200.0, 200.0],
            "anonymous_can_centers_xy": [[10.0, 10.0], [20.0, 20.0]],
            "raw_can_inside_count": 0,
            "initial_cream_track_present": True,
            "cream_near_basket_rim": False,
            "cream_cross_class_collision": False,
            "sift_valid": True,
            "sift_affine_scale": 1.0,
            "sift_invalid_reasons": [],
            "cream_state_reasons": ["searching"],
        },
        "phi_only": recovered,
        "dynamics_transition_eligible": not recovered,
    }


def _synthetic_screen_frames(
    registrations: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    frames = []
    for seed_start in (8000, 8100):
        rgb_index = 0
        registration = registrations[seed_start]
        for spec in registration["panel"]["episodes"]:
            times, roles = P._expected_episode_schedule(spec)
            for local_index, (t, role) in enumerate(zip(times, roles, strict=True)):
                frames.append(
                    _synthetic_progress_frame(
                        seed_start=seed_start,
                        phase="screen",
                        episode_id=spec["episode_id"],
                        rgb_index=rgb_index,
                        local_index=local_index,
                        t=t,
                        role=role,
                    )
                )
                rgb_index += 1
    return frames


def _progress_engine_record(registration: dict[str, Any]) -> dict[str, Any]:
    return {
        "sam3_engine_sha256": registration["sources"]["sam3_engine"]["sha256"],
        "sift_engine_sha256": registration["sources"]["sift_engine"]["sha256"],
        "can_state_sha256": registration["sources"]["can_state"]["sha256"],
        "cream_state_sha256": registration["sources"]["cream_state"]["sha256"],
        "sam3_checkpoint_sha256": registration["models"]["sam3"]["checkpoint"][
            "sha256"
        ],
        "clip_tokenizer_tree_sha256": registration["models"]["sam3"]
        ["clip_tokenizer"]["tree_sha256"],
        "sam3_config_sha256": registration["models"]["sam3"]["config_sha256"],
        "sam3_processor_sha256": registration["models"]["sam3"][
            "processor_sha256"
        ],
        "cream_template_image_sha256": registration["templates"]
        ["cream_template_image"]["sha256"],
        "cream_template_metadata_sha256": registration["templates"]
        ["cream_template_metadata"]["sha256"],
        "semantic_config_sha256": registration["templates"]["semantic_config"][
            "sha256"
        ],
        "transformers_source_sha256s": {
            role: registration["runtime"]["runtime_sources"][role]["sha256"]
            for role in SEMANTIC_RUNTIME_ROLES
        },
    }


def _perfect_screen_metrics() -> dict[str, Any]:
    return {
        "coverage": {
            "frame_coverage": 1.0,
            "evidence_coverage": 1.0,
            "uncertainty_rate": 0.0,
        },
        "classification": {
            "exact_scalar_accuracy_certain_only": 1.0,
            "effective_scalar_accuracy_uncertain_incorrect": 1.0,
            "macro_f1": 1.0,
        },
        "transitions": {
            "missing_transitions": 0,
            "extra_transitions": 0,
            "median_absolute_error_steps": 0.0,
            "p95_absolute_error_steps": 0.0,
            "maximum_absolute_error_steps": 0.0,
            "chain3_cream_positive_transitions": 1,
        },
        "never_achieved_terminal_specificity": {
            "denominator": 1,
            "specificity": 1.0,
        },
        "held_cream_specificity": {"denominator": 1, "specificity": 1.0},
        "carry_audit": {"carry_created_transitions": 0},
    }


def _build_screen_ancestry(
    bundle: dict[str, Any],
) -> dict[str, Any]:
    target_root = Path(bundle["target_root"])
    registrations = {
        seed: bundle["registrations"][f"{C.PANEL_IDS[seed]}_screen.json"][
            "registration"
        ]
        for seed in (8000, 8100)
    }
    registration_hashes = {
        seed: bundle["registrations"][f"{C.PANEL_IDS[seed]}_screen.json"][
            "file_sha256"
        ]
        for seed in (8000, 8100)
    }
    frames = _synthetic_screen_frames(registrations)
    progress_root = (
        target_root / bundle["target_output_slots"]["screen_progress_complete"]
    ).parent
    progress_root.mkdir(parents=True)
    write(progress_root / P.PRODUCER_FILENAME, Path(P.__file__).read_bytes())
    write(
        progress_root / P.CONFIG_FILENAME,
        (REPO / "plan_and_progress" / "v248_gate0_semantic_config.json").read_bytes(),
    )
    _write_jsonl(progress_root / P.FRAMES_FILENAME, frames)
    progress_authority = A._combined_authority(
        bundle,
        phase="screen",
        output_complete_slot=bundle["target_output_slots"][
            "screen_progress_complete"
        ],
    )
    prefix_checks = {
        f"{C.PANEL_IDS[seed]}:ep{episode_id:02d}": {
            "frames": 8,
            "reference_sha256": "3" * 64,
            "fresh_session_sha256": "3" * 64,
            "exact_after_rounding_6_decimals": True,
        }
        for seed in (8000, 8100)
        for episode_id in C.SCREEN_EPISODES[seed]
    }
    progress_result = {
        "schema": P.RESULT_SCHEMA,
        "status": "observation_only_sealed",
        "created_utc": "2026-09-01T00:00:00Z",
        "claim_scope": "visual_progress_producer_only",
        "phase": "screen",
        "task": C.TASK,
        "authority": progress_authority,
        "registrations": {
            C.PANEL_IDS[seed]: {
                "registration_id": registrations[seed]["registration_id"],
                "file_sha256": registration_hashes[seed],
                "canonical_self_sha256": registrations[seed]["self_sha256"],
            }
            for seed in (8000, 8100)
        },
        "rgb_inputs": {
            C.PANEL_IDS[seed]: {
                "result_sha256": ("4" if seed == 8000 else "5") * 64,
                "manifest_sha256": ("6" if seed == 8000 else "7") * 64,
                "complete_sha256": ("8" if seed == 8000 else "9") * 64,
                "producer_sha256": registrations[seed]["sources"][
                    "formal_replay"
                ]["sha256"],
                "image_tree_sha256": ("a" if seed == 8000 else "b") * 64,
                "frames": E._expected_frames_for_registration(registrations[seed]),
            }
            for seed in (8000, 8100)
        },
        "episode_ids": {
            C.PANEL_IDS[seed]: list(C.SCREEN_EPISODES[seed])
            for seed in (8000, 8100)
        },
        "observation_contract": {
            "decoded_cameras": ["agentview", "eye_in_hand"],
            "semantic_inputs": ["current_agentview_rgb", "current_eye_in_hand_rgb"],
            "causal_past_state_only": True,
        },
        "engines": _progress_engine_record(registrations[8000]),
        "prefix_checks": prefix_checks,
        "recovered_endpoint": {
            "required": True,
            "present": True,
            "panel_id": "chain3_8100",
            "episode_id": 77,
            "t": 651,
            "phi_only": True,
            "dynamics_transition_eligible": False,
            "rgb_recovery_prerequisites_passed": True,
        },
        "output": {
            "frames": len(frames),
            "frames_sha256": C.sha256_file(progress_root / P.FRAMES_FILENAME),
            "producer_sha256": C.sha256_file(progress_root / P.PRODUCER_FILENAME),
            "semantic_config_sha256": C.sha256_file(
                progress_root / P.CONFIG_FILENAME
            ),
        },
        "safety": {
            "oracle_opened_or_hashed": False,
            "privileged_inputs_read": False,
            "simulator_imported_or_run": False,
            "model_input_fields": ["agentview_rgb", "eye_in_hand_rgb"],
            "forbidden_public_field_scan_passed": True,
            "source_images_mutated": False,
            "dynamics_rows_written": 0,
        },
    }
    _rewrite_json(progress_root / P.RESULT_FILENAME, progress_result)
    progress_complete = {
        "schema": P.COMPLETE_SCHEMA,
        "status": "atomic_success",
        "atomic_commit": True,
        "phase": "screen",
        "authority": progress_authority,
        "registration_sha256s": {
            C.PANEL_IDS[seed]: registration_hashes[seed]
            for seed in (8000, 8100)
        },
        "result_sha256": C.sha256_file(progress_root / P.RESULT_FILENAME),
        "frames_sha256": C.sha256_file(progress_root / P.FRAMES_FILENAME),
        "producer_sha256": C.sha256_file(progress_root / P.PRODUCER_FILENAME),
        "semantic_config_sha256": C.sha256_file(progress_root / P.CONFIG_FILENAME),
    }
    _rewrite_json(progress_root / P.COMPLETE_FILENAME, progress_complete)
    progress_complete_sha = C.sha256_file(progress_root / P.COMPLETE_FILENAME)
    (
        _loaded_bundle,
        _loaded_regs,
        _loaded_hashes,
        loaded_progress_result,
        _loaded_frames,
        config,
        progress_seals,
    ) = E.load_anchored_campaign_progress(
        freeze_bundle_root=Path(bundle["root"]),
        expected_freeze_complete_sha256=bundle["freeze_complete_sha256"],
        progress_root=progress_root,
        expected_progress_complete_sha256=progress_complete_sha,
    )

    pre_oracle_root = (
        target_root
        / bundle["target_output_slots"]["screen_pre_oracle_complete"]
    ).parent
    pre_oracle_root.mkdir(parents=True)
    evaluator_bytes = Path(E.__file__).read_bytes()
    evaluator_sha = C.sha256_file(Path(E.__file__))
    write(pre_oracle_root / E.EVALUATOR_FILENAME, evaluator_bytes)
    pre_oracle_authority = E.campaign_node_authority(
        bundle, "screen", "pre_oracle", None
    )
    _preseal_path, pre_oracle_seal_sha = E.make_pre_oracle_seal(
        stage=pre_oracle_root,
        phase="screen",
        registration_hashes=registration_hashes,
        progress_seals=progress_seals,
        evaluator_sha256=evaluator_sha,
        authority=pre_oracle_authority,
    )
    pre_oracle_complete = {
        "schema": E.PRE_ORACLE_COMPLETE_SCHEMA,
        "status": "atomic_pre_oracle_seal_complete",
        "atomic_commit": True,
        "pre_oracle_seal_sha256": pre_oracle_seal_sha,
        "evaluator_sha256": evaluator_sha,
        "authority": pre_oracle_authority,
    }
    _rewrite_json(pre_oracle_root / E.COMPLETE_FILENAME, pre_oracle_complete)
    pre_oracle_complete_sha = C.sha256_file(pre_oracle_root / E.COMPLETE_FILENAME)
    E.validate_published_pre_oracle(
        root=pre_oracle_root,
        expected_pre_oracle_sha256=pre_oracle_seal_sha,
        expected_complete_sha256=pre_oracle_complete_sha,
        phase="screen",
        registration_hashes=registration_hashes,
        progress_seals=progress_seals,
        evaluator_sha256=evaluator_sha,
        expected_authority=pre_oracle_authority,
    )

    evaluation_root = (
        target_root / bundle["target_output_slots"]["screen_evaluation_complete"]
    ).parent
    evaluation_root.mkdir(parents=True)
    write(evaluation_root / E.EVALUATOR_FILENAME, evaluator_bytes)
    write(
        evaluation_root / E.PRE_ORACLE_FILENAME,
        (pre_oracle_root / E.PRE_ORACLE_FILENAME).read_bytes(),
    )
    evaluation_authority = E.campaign_node_authority(
        bundle, "screen", "evaluation", None
    )
    metrics = _perfect_screen_metrics()
    subgate = E.visual_subgate_decision(
        metrics=metrics,
        config=config,
        phase="screen",
        primary_episode_count=sum(
            len(registrations[seed]["panel"]["episodes"])
            for seed in (8000, 8100)
        ),
        all_prefix_exact=True,
        recovery_passed=True,
    )
    assert subgate["metric_checks_passed"] is True
    evaluation_result = {
        "schema": E.EVALUATION_SCHEMA,
        "status": "postseal_visual_evaluation_complete",
        "created_utc": "2026-09-01T00:00:00Z",
        "claim_scope": "gate0_visual_subgate_only",
        "phase": "screen",
        "decision": "STOP",
        "authority": evaluation_authority,
        "pre_oracle_seal_sha256": pre_oracle_seal_sha,
        "pre_oracle_complete_sha256": pre_oracle_complete_sha,
        "registrations": progress_result_registrations(loaded_progress_result),
        "progress_artifact": progress_seals,
        "oracle_inputs": {
            C.PANEL_IDS[seed]: {
                "file_sha256": registrations[seed]["oracle"]["summary"]["sha256"],
                "bytes": registrations[seed]["oracle"]["summary"]["bytes"],
                "fields_used": [
                    "episode_records.idx",
                    "seed",
                    "steps",
                    "events",
                ],
                "success_values_accessed": False,
            }
            for seed in (8000, 8100)
        },
        "metrics": metrics,
        "visual_subgate": subgate,
        "provenance": {
            "evaluator_sha256": evaluator_sha,
            "registered_evaluator_sha256": evaluator_sha,
            "producer_sha256": progress_seals["producer_sha256"],
            "semantic_config_sha256": progress_seals["semantic_config_sha256"],
        },
    }
    _rewrite_json(evaluation_root / E.RESULT_FILENAME, evaluation_result)
    _rewrite_json(evaluation_root / E.SUBGATE_FILENAME, subgate)
    marker = {
        "schema": E.MARKER_SCHEMA,
        "decision": "STOP",
        "claim_scope": "gate0_visual_subgate_only",
        "result_sha256": C.sha256_file(evaluation_root / E.RESULT_FILENAME),
        "subgate_sha256": C.sha256_file(evaluation_root / E.SUBGATE_FILENAME),
    }
    _rewrite_json(evaluation_root / "STOP.json", marker)
    evaluation_complete = {
        "schema": E.COMPLETE_SCHEMA,
        "status": "atomic_evaluation_complete",
        "atomic_commit": True,
        "decision": "STOP",
        "authority": evaluation_authority,
        "result_sha256": C.sha256_file(evaluation_root / E.RESULT_FILENAME),
        "subgate_sha256": C.sha256_file(evaluation_root / E.SUBGATE_FILENAME),
        "marker_filename": "STOP.json",
        "marker_sha256": C.sha256_file(evaluation_root / "STOP.json"),
        "pre_oracle_seal_sha256": pre_oracle_seal_sha,
        "pre_oracle_complete_sha256": pre_oracle_complete_sha,
        "evaluator_sha256": evaluator_sha,
    }
    _rewrite_json(evaluation_root / E.COMPLETE_FILENAME, evaluation_complete)
    evaluation_complete_sha = C.sha256_file(evaluation_root / E.COMPLETE_FILENAME)
    E._validate_evaluation_stage(
        evaluation_root,
        registrations,
        registration_hashes,
        progress_seals,
        expected_authority=evaluation_authority,
    )
    return {
        "progress_root": progress_root,
        "progress_complete_sha256": progress_complete_sha,
        "pre_oracle_root": pre_oracle_root,
        "pre_oracle_seal_sha256": pre_oracle_seal_sha,
        "pre_oracle_complete_sha256": pre_oracle_complete_sha,
        "evaluation_root": evaluation_root,
        "evaluation_complete_sha256": evaluation_complete_sha,
        "progress_seals": progress_seals,
        "registrations": registrations,
        "registration_hashes": registration_hashes,
    }


def progress_result_registrations(
    result: dict[str, Any],
) -> dict[str, Any]:
    return copy.deepcopy(result["registrations"])


def _authorization_authority_for_test(
    bundle: dict[str, Any], screen_complete_sha256: str
) -> dict[str, Any]:
    campaign = bundle["campaign"]
    return {
        "freeze_complete_sha256": bundle["freeze_complete_sha256"],
        "campaign_manifest_sha256": bundle["campaign_manifest_sha256"],
        "campaign_body_sha256": campaign["body_sha256"],
        "campaign_projection_sha256": campaign["projection_sha256"],
        "campaign_id": campaign["campaign_id"],
        "screen_evaluation_complete_sha256": screen_complete_sha256,
        "screen_evaluation_complete_slot": bundle["target_output_slots"][
            "screen_evaluation_complete"
        ],
        "authorization_complete_slot": bundle["target_output_slots"][
            "confirm_authorization_complete"
        ],
    }


def _build_fake_authorization(
    root: Path, bundle: dict[str, Any], screen_complete_sha256: str
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    root.mkdir(parents=True)
    predicate = A.frozen_predicate()
    authority = _authorization_authority_for_test(
        bundle, screen_complete_sha256
    )
    producer_source = REPO / "scripts" / "authorize_v248_gate0_confirm.py"
    producer_path = write(root / "PRODUCER.py", producer_source.read_bytes())
    predicate_path = write(root / "PREDICATE.json", C.canonical_bytes(predicate))
    result = {
        "schema": A.AUTHORIZATION_RESULT_SCHEMA,
        "status": "screen_predicate_satisfied",
        "created_utc": "2026-09-01T00:00:00Z",
        "decision": "AUTHORIZE_CONFIRM",
        "claim_scope": "authorize_exact_frozen_confirm_schedule_only",
        "authority": authority,
        "predicate": predicate,
        "screen_registrations": A._registration_refs(bundle, "screen"),
        "confirm_registrations": A._registration_refs(bundle, "confirm"),
        "input": {
            "screen_rgb_complete_sha256s": {
                "chain3_8000": "d" * 64,
                "chain3_8100": "e" * 64,
            },
            "screen_progress_complete_sha256": "f" * 64,
            "screen_pre_oracle_seal_sha256": "1" * 64,
            "screen_pre_oracle_complete_sha256": "2" * 64,
            "screen_evaluation_complete_sha256": screen_complete_sha256,
            "screen_evaluation_result_sha256": "b" * 64,
            "screen_visual_subgate_sha256": "c" * 64,
        },
        "output": {
            "predicate_sha256": C.sha256_file(predicate_path),
            "producer_sha256": C.sha256_file(producer_path),
        },
        "safety": {
            "oracle_path_cli_supported": False,
            "oracle_opened_or_hashed": False,
            "simulator_imported": False,
            "environment_constructed_or_reset": False,
            "caller_selected_registration": False,
            "caller_selected_output": False,
        },
    }
    result_path = write(root / "RESULT.json", C.canonical_bytes(result))
    marker = {
        "schema": A.AUTHORIZATION_MARKER_SCHEMA,
        "decision": "AUTHORIZE_CONFIRM",
        "result_sha256": C.sha256_file(result_path),
        "predicate_sha256": C.sha256_file(predicate_path),
    }
    marker_path = write(root / "AUTHORIZED.json", C.canonical_bytes(marker))
    complete = {
        "schema": A.AUTHORIZATION_COMPLETE_SCHEMA,
        "status": "atomic_confirm_authorization",
        "atomic_commit": True,
        "decision": "AUTHORIZE_CONFIRM",
        "authority": authority,
        "result_sha256": C.sha256_file(result_path),
        "predicate_sha256": C.sha256_file(predicate_path),
        "producer_sha256": C.sha256_file(producer_path),
        "marker_sha256": C.sha256_file(marker_path),
    }
    complete_path = write(root / "COMPLETE.json", C.canonical_bytes(complete))
    registered_sha = bundle["registrations"]["chain3_8000_screen.json"][
        "registration"
    ]["sources"]["confirm_authorizer"]["sha256"]
    A._validate_authorization_stage(
        root,
        authority,
        registered_sha,
        expected_input=result["input"],
        expected_screen_registrations=result["screen_registrations"],
        expected_confirm_registrations=result["confirm_registrations"],
    )
    return C.sha256_file(complete_path), result, authority


def test_confirm_authorizer_predicate_and_forged_authorization_fail_closed() -> None:
    checks = {
        name: True for name in reversed(A.REQUIRED_METRIC_CHECK_NAMES)
    }
    visual = {
        "decision": "STOP",
        "promotion_authorized": False,
        "screen_metrics_passed": True,
        "metric_checks_passed": True,
        "checks": checks,
        "thresholds": {"phase_required_primary_chain3_episodes": 24},
    }
    screen_result = {
        "phase": "screen",
        "decision": "STOP",
        "visual_subgate": visual,
    }
    canonical_screen = json.loads(C.canonical_bytes(screen_result))
    A._assert_frozen_screen_predicate(
        canonical_screen,
        canonical_screen["visual_subgate"],
        canonical_screen["visual_subgate"],
    )
    failed = copy.deepcopy(screen_result)
    failed["visual_subgate"]["screen_metrics_passed"] = False
    expect_error(
        lambda: A._assert_frozen_screen_predicate(
            failed,
            failed["visual_subgate"],
            failed["visual_subgate"],
        ),
        "screen metrics predicate",
    )

    with tempfile.TemporaryDirectory(prefix="v248-forged-auth-test-") as raw:
        root = Path(raw)
        _tape_path, sanitized = build_sanitized(root)
        bundle_root, anchor, _registrations = build_campaign_bundle(root, sanitized)
        bundle = C.load_and_validate_campaign_bundle(
            bundle_root, anchor, expected_source_role="confirm_authorizer"
        )
        authorization_slot = bundle["target_output_slots"][
            "confirm_authorization_complete"
        ]
        authorization_root = (
            Path(bundle["target_root"]) / Path(authorization_slot)
        ).parent
        fake_complete_sha, _result, _authority = _build_fake_authorization(
            authorization_root, bundle, "a" * 64
        )
        # The forged artifact is internally complete and fully resealed, but
        # there is no valid combined-screen evaluation at the frozen slot.
        expect_error(
            lambda: R._validate_confirm_authorization(bundle, fake_complete_sha),
            "screen evaluation root missing",
        )


def _authorize_kwargs(
    bundle_root: Path,
    freeze_complete_sha256: str,
    ancestry: dict[str, Any],
) -> dict[str, Any]:
    return {
        "freeze_bundle_root": bundle_root,
        "expected_freeze_complete_sha256": freeze_complete_sha256,
        "expected_screen_progress_complete_sha256": ancestry[
            "progress_complete_sha256"
        ],
        "expected_screen_pre_oracle_seal_sha256": ancestry[
            "pre_oracle_seal_sha256"
        ],
        "expected_screen_pre_oracle_complete_sha256": ancestry[
            "pre_oracle_complete_sha256"
        ],
        "expected_screen_evaluation_complete_sha256": ancestry[
            "evaluation_complete_sha256"
        ],
    }


def _reseal_evaluation(root: Path) -> str:
    result_path = root / E.RESULT_FILENAME
    subgate_path = root / E.SUBGATE_FILENAME
    marker_path = root / "STOP.json"
    complete_path = root / E.COMPLETE_FILENAME
    marker = json.loads(marker_path.read_text())
    marker["result_sha256"] = C.sha256_file(result_path)
    marker["subgate_sha256"] = C.sha256_file(subgate_path)
    _rewrite_json(marker_path, marker)
    complete = json.loads(complete_path.read_text())
    complete["result_sha256"] = C.sha256_file(result_path)
    complete["subgate_sha256"] = C.sha256_file(subgate_path)
    complete["marker_sha256"] = C.sha256_file(marker_path)
    _rewrite_json(complete_path, complete)
    return C.sha256_file(complete_path)


def _reseal_authorization_stage(root: Path) -> str:
    result_path = root / "RESULT.json"
    marker_path = root / "AUTHORIZED.json"
    complete_path = root / "COMPLETE.json"
    marker = json.loads(marker_path.read_text())
    marker["result_sha256"] = C.sha256_file(result_path)
    _rewrite_json(marker_path, marker)
    complete = json.loads(complete_path.read_text())
    complete["result_sha256"] = C.sha256_file(result_path)
    complete["marker_sha256"] = C.sha256_file(marker_path)
    _rewrite_json(complete_path, complete)
    return C.sha256_file(complete_path)


def test_complete_synthetic_screen_to_authorization_to_confirm_validation() -> None:
    with tempfile.TemporaryDirectory(prefix="v248-full-authority-chain-") as raw:
        root = Path(raw)
        tape_path, sanitized = build_sanitized(root, complete_panel=True)
        bundle_root, freeze_sha, _registrations = build_campaign_bundle(
            root, sanitized
        )
        bundle = C.load_and_validate_campaign_bundle(
            bundle_root, freeze_sha, expected_source_role="confirm_authorizer"
        )
        ancestry = _build_screen_ancestry(bundle)
        kwargs = _authorize_kwargs(bundle_root, freeze_sha, ancestry)

        parsed = A.parse_args(
            [
                "--freeze-bundle",
                str(bundle_root),
                "--expected-freeze-complete-sha256",
                freeze_sha,
                "--expected-screen-progress-complete-sha256",
                ancestry["progress_complete_sha256"],
                "--expected-screen-pre-oracle-seal-sha256",
                ancestry["pre_oracle_seal_sha256"],
                "--expected-screen-pre-oracle-complete-sha256",
                ancestry["pre_oracle_complete_sha256"],
                "--expected-screen-evaluation-complete-sha256",
                ancestry["evaluation_complete_sha256"],
                "--validate-only",
            ]
        )
        assert parsed.validate_only is True
        assert not hasattr(parsed, "output_dir")
        authorization_root = (
            Path(bundle["target_root"])
            / bundle["target_output_slots"]["confirm_authorization_complete"]
        ).parent
        validated_only = A.authorize(**kwargs, validate_only=True)
        assert validated_only["status"] == "VALID"
        assert validated_only["output_created"] is False
        assert not authorization_root.exists()

        published = A.authorize(**kwargs)
        assert published["status"] == "PASS"
        authorization_sha = published["complete_sha256"]
        bundle = C.load_and_validate_campaign_bundle(
            bundle_root, freeze_sha, expected_source_role="formal_replay"
        )
        captured = A.validate_authorization_artifact(bundle, authorization_sha)
        assert captured["captured_authorization_file_sha256s"]["COMPLETE.json"] == (
            authorization_sha
        )
        replay_validation = R._validate_confirm_authorization(
            bundle, authorization_sha
        )
        assert replay_validation["complete_sha256"] == authorization_sha
        assert replay_validation["screen_evaluation_complete_sha256"] == ancestry[
            "evaluation_complete_sha256"
        ]
        prepared = R.prepare_inputs(
            sanitized_tape=tape_path,
            freeze_bundle_root=bundle_root,
            expected_freeze_complete_sha256=freeze_sha,
            registration_slot="chain3_8000_confirm.json",
            expected_confirm_authorization_complete_sha256=authorization_sha,
        )
        assert prepared.phase == "confirm"
        assert prepared.panel_id == "chain3_8000"
        assert prepared.confirm_authorization_complete_sha256 == authorization_sha
        assert not prepared.output_path.exists()

        expected_input = captured["result"]["input"]
        expected_screen_refs = captured["result"]["screen_registrations"]
        expected_confirm_refs = captured["result"]["confirm_registrations"]
        registered_sha = bundle["registrations"]["chain3_8000_screen.json"][
            "registration"
        ]["sources"]["confirm_authorizer"]["sha256"]
        for field, bad_value, message in (
            ("status", "attacker_resealed", "RESULT status"),
            ("claim_scope", "authorize_any_confirm", "claim scope"),
            ("created_utc", "2026-09-01T00:00:00+00:00", "created_utc"),
        ):
            attack_root = root / f"authorization-{field}-attack"
            shutil.copytree(authorization_root, attack_root)
            attacked = json.loads((attack_root / "RESULT.json").read_text())
            attacked[field] = bad_value
            _rewrite_json(attack_root / "RESULT.json", attacked)
            _reseal_authorization_stage(attack_root)
            expect_error(
                lambda path=attack_root: A._validate_authorization_stage(
                    path,
                    captured["result"]["authority"],
                    registered_sha,
                    expected_input=expected_input,
                    expected_screen_registrations=expected_screen_refs,
                    expected_confirm_registrations=expected_confirm_refs,
                ),
                message,
            )
        C.assert_no_simulator_modules_imported()


def test_screen_ancestry_attacks_and_snapshot_drift_fail_closed() -> None:
    with tempfile.TemporaryDirectory(prefix="v248-screen-ancestry-attacks-") as raw:
        root = Path(raw)
        _tape_path, sanitized = build_sanitized(root, complete_panel=True)
        bundle_root, freeze_sha, _registrations = build_campaign_bundle(
            root, sanitized
        )
        bundle = C.load_and_validate_campaign_bundle(
            bundle_root, freeze_sha, expected_source_role="confirm_authorizer"
        )
        ancestry = _build_screen_ancestry(bundle)
        kwargs = _authorize_kwargs(bundle_root, freeze_sha, ancestry)

        for key in (
            "expected_screen_progress_complete_sha256",
            "expected_screen_pre_oracle_seal_sha256",
            "expected_screen_pre_oracle_complete_sha256",
        ):
            attacked = dict(kwargs)
            attacked[key] = "f" * 64
            expect_error(lambda values=attacked: A.authorize(**values, validate_only=True))

        evaluation_root = ancestry["evaluation_root"]
        original_files = {
            path.name: path.read_bytes() for path in evaluation_root.iterdir()
        }

        result = json.loads((evaluation_root / E.RESULT_FILENAME).read_text())
        del result["visual_subgate"]["checks"][
            A.REQUIRED_METRIC_CHECK_NAMES[0]
        ]
        _rewrite_json(evaluation_root / E.RESULT_FILENAME, result)
        _rewrite_json(
            evaluation_root / E.SUBGATE_FILENAME, result["visual_subgate"]
        )
        check_attack_sha = _reseal_evaluation(evaluation_root)
        attacked = dict(kwargs)
        attacked["expected_screen_evaluation_complete_sha256"] = check_attack_sha
        expect_error(
            lambda: A.authorize(**attacked, validate_only=True),
            "check-name closure",
        )

        for name, payload in original_files.items():
            (evaluation_root / name).write_bytes(payload)
        result = json.loads((evaluation_root / E.RESULT_FILENAME).read_text())
        result["metrics"]["coverage"]["frame_coverage"] = 0.5
        _rewrite_json(evaluation_root / E.RESULT_FILENAME, result)
        metric_attack_sha = _reseal_evaluation(evaluation_root)
        attacked = dict(kwargs)
        attacked["expected_screen_evaluation_complete_sha256"] = metric_attack_sha
        expect_error(
            lambda: A.authorize(**attacked, validate_only=True),
            "frozen-config recomputation",
        )

        for name, payload in original_files.items():
            (evaluation_root / name).write_bytes(payload)
        result_path = evaluation_root / E.RESULT_FILENAME
        result_path.write_bytes(result_path.read_bytes() + b" ")
        expect_error(
            lambda: A.authorize(**kwargs, validate_only=True),
            "differs from external SHA-256 anchor",
        )

    with tempfile.TemporaryDirectory(prefix="v248-cross-campaign-auth-") as raw:
        root = Path(raw)
        _tape_path, sanitized = build_sanitized(root / "first", complete_panel=True)
        first_bundle_root, first_freeze, _ = build_campaign_bundle(
            root / "first", sanitized
        )
        first_bundle = C.load_and_validate_campaign_bundle(
            first_bundle_root,
            first_freeze,
            expected_source_role="confirm_authorizer",
        )
        ancestry = _build_screen_ancestry(first_bundle)

        _second_tape, second_sanitized = build_sanitized(
            root / "second", complete_panel=True
        )
        second_bundle_root, second_freeze, _ = build_campaign_bundle(
            root / "second", second_sanitized
        )
        second_bundle = C.load_and_validate_campaign_bundle(
            second_bundle_root,
            second_freeze,
            expected_source_role="confirm_authorizer",
        )
        shutil.copytree(
            Path(first_bundle["target_root"]), Path(second_bundle["target_root"])
        )
        cross_kwargs = _authorize_kwargs(
            second_bundle_root, second_freeze, ancestry
        )
        expect_error(
            lambda: A.authorize(**cross_kwargs, validate_only=True),
            "genesis drift",
        )


def test_authorizer_uses_derived_slots_single_capture_and_registered_modules() -> None:
    with tempfile.TemporaryDirectory(prefix="v248-authorizer-capture-") as raw:
        root = Path(raw)
        _tape_path, sanitized = build_sanitized(root, complete_panel=True)
        bundle_root, freeze_sha, _registrations = build_campaign_bundle(
            root, sanitized
        )
        bundle = C.load_and_validate_campaign_bundle(
            bundle_root, freeze_sha, expected_source_role="confirm_authorizer"
        )
        ancestry = _build_screen_ancestry(bundle)
        derived_slots: list[str] = []

        def derived(target_root: Path, slot: str) -> Path:
            derived_slots.append(slot)
            return R._complete_path_from_slot(target_root, slot)

        replay_support = SimpleNamespace(_complete_path_from_slot=derived)
        reads: dict[Path, int] = {}
        original_reader = A._read_regular_bytes

        def counted_reader(path: Path, digest: str, where: str) -> bytes:
            resolved = Path(path).resolve()
            reads[resolved] = reads.get(resolved, 0) + 1
            return original_reader(path, digest, where)

        with mock.patch.object(A, "_read_regular_bytes", side_effect=counted_reader):
            screen = A._validate_screen_evaluation(
                bundle,
                ancestry["evaluation_complete_sha256"],
                ancestry["progress_complete_sha256"],
                ancestry["pre_oracle_seal_sha256"],
                ancestry["pre_oracle_complete_sha256"],
                replay_support=replay_support,
            )
        evaluation_complete = ancestry["evaluation_root"] / "COMPLETE.json"
        assert reads[evaluation_complete.resolve()] == 1
        assert screen["captured_evaluation_file_sha256s"]["COMPLETE.json"] == ancestry[
            "evaluation_complete_sha256"
        ]
        assert set(derived_slots) == {
            bundle["target_output_slots"]["screen_evaluation_complete"],
            bundle["target_output_slots"]["screen_progress_complete"],
            bundle["target_output_slots"]["screen_pre_oracle_complete"],
        }

        registration = bundle["registrations"]["chain3_8000_screen.json"][
            "registration"
        ]
        ambient_name = "produce_v248_gate0_progress"
        ambient = sys.modules.get(ambient_name)
        sys.modules[ambient_name] = SimpleNamespace(__file__="/tmp/ambient-drift.py")
        try:
            expect_error(
                lambda: A._load_registered_evaluator(registration),
                "origin differs",
            )
        finally:
            if ambient is None:
                sys.modules.pop(ambient_name, None)
            else:
                sys.modules[ambient_name] = ambient


def test_single_fd_reader_rejects_symlink_and_path_identity_swap() -> None:
    with tempfile.TemporaryDirectory(prefix="v248-single-fd-test-") as raw:
        root = Path(raw)
        regular = write(root / "regular.json", b"{}\n")
        digest = C.sha256_file(regular)
        symlink = root / "symlink.json"
        symlink.symlink_to(regular)
        expect_error(
            lambda: A._read_regular_bytes(symlink, digest, "symlink fixture"),
            "symlink",
        )
        observed = regular.stat()

        def swapped_stat(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(
                st_dev=observed.st_dev,
                st_ino=observed.st_ino + 1,
                st_size=observed.st_size,
                st_mode=observed.st_mode,
            )

        with mock.patch.object(C.os, "stat", side_effect=swapped_stat):
            expect_error(
                lambda: A._read_regular_bytes(
                    regular, digest, "path replacement fixture"
                ),
                "path was replaced",
            )


def main() -> int:
    tests = (
        test_registration_closure_anchor_oracle_and_pre_reset_drift,
        test_recovery_requires_all_65_bitwise_chunks_before_missing,
        test_manifest_is_exact_and_semantic_free,
        test_atomic_publish_is_no_clobber_and_cleans_failures,
        test_producer_snapshot_must_match_registration,
        test_validate_only_subprocess_blocks_simulator_imports,
        test_formal_authority_rejects_unanchored_wrong_phase_and_legacy_cli,
        test_confirm_authorization_rejection_precedes_all_target_io,
        test_campaign_mixing_and_post_screen_refreeze_are_rejected,
        test_resealed_manifest_output_attack_and_unsafe_parents_are_rejected,
        test_tokenizer_loader_requests_exact_frozen_revision,
        test_confirm_authorizer_predicate_and_forged_authorization_fail_closed,
        test_complete_synthetic_screen_to_authorization_to_confirm_validation,
        test_screen_ancestry_attacks_and_snapshot_drift_fail_closed,
        test_authorizer_uses_derived_slots_single_capture_and_registered_modules,
        test_single_fd_reader_rejects_symlink_and_path_identity_swap,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS all {len(tests)} v248 Gate-0 RGB CPU contract tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
