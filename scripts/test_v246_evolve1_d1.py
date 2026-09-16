#!/usr/bin/env python
# RETIRED 2026-09-16. Superseded by scripts/collect_v250_chain3_round.py, which
# collected the round's D0 and D1. The --collect path here raises
# RealCollectorNotImplemented after all validation, and the test file asserts that it
# does; the contract reasoning is what is being kept.
# Do not extend, do not import, do not cite as capability. Recoverable at 609c29f8.
"""CPU-only mechanical tests for the fail-closed EVOLVE-1 D1 collector core."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable
from unittest import mock

import numpy as np
import torch


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import collect_v246_evolve1_d1 as C  # noqa: E402


@dataclass(frozen=True)
class Fixture:
    root: Path
    registration: Path
    actor_a: Path
    actor_b: Path
    d0_tape: Path
    m0: Path
    phi: Path
    policy: Path
    tokenizer: Path
    seed_ledger: Path


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(C.canonical_bytes(value))


def rewrite_registration(path: Path, payload: dict[str, Any]) -> None:
    payload.pop("self_sha256", None)
    payload["self_sha256"] = C.canonical_sha256(payload)
    write_json(path, payload)


def expect_raises(error_type: type[BaseException], fn: Callable[[], Any]) -> BaseException:
    try:
        fn()
    except error_type as error:
        return error
    raise AssertionError(f"expected {error_type.__name__}")


def build_fixture(
    root: Path,
    *,
    actor_task: str = C.TASK,
    d0_task: str = C.TASK,
    bad_source_hash: bool = False,
    overlap_d1_seed: bool = False,
    actor_seed_overlaps_gate3: bool = False,
    overlapping_seed_roles: bool = False,
    bad_actor_runtime_fingerprint: bool = False,
    extra_actor_field: bool = False,
    same_actor_seed: bool = False,
    actor_scale: float = C.REGISTERED_RESIDUAL_SCALE,
) -> Fixture:
    root.mkdir(parents=True, exist_ok=True)
    d0_path = root / "d0.pt"
    boundaries = torch.stack(
        [
            torch.zeros(C.RICH_DIM),
            torch.ones(C.RICH_DIM),
            torch.full((C.RICH_DIM,), 2.0),
        ]
    )
    d0 = {
        "schema": C.D0_TAPE_SCHEMA,
        "z": boundaries[:-1].clone(),
        "u": torch.zeros(2, C.C, C.ACTION_DIM),
        "z_next": boundaries[1:].clone(),
        "episode": torch.zeros(2, dtype=torch.long),
        "t": torch.tensor([0, C.C]),
        "sigma": torch.zeros(2),
        "success": torch.zeros(1),
        "task": d0_task,
        "c": C.C,
        "latent_dim": C.RICH_DIM,
        "proprio_dim": C.PROPRIO_DIM,
        "proprio_keys": list(C.PROPRIO_KEYS),
        "training_feature_fields": list(C.TRAINING_FEATURE_FIELDS),
        "split_bookkeeping_fields": list(C.SPLIT_BOOKKEEPING_FIELDS),
        "legacy_outcome_fields": ["success"],
    }
    torch.save(d0, d0_path)
    d0_sha = C.sha256_file(d0_path)

    m0_path = root / "m0.pt"
    phi_path = root / "phi.pt"
    policy_path = root / "base_vla.pt"
    tokenizer_path = root / "tokenizer.json"
    tokenizer = {
        "schema": C.TOKENIZER_SCHEMA,
        "status": "frozen",
        "model_id": "pi0.5",
        "tokenizer_class": "PaliGemmaTokenizer",
        "vocab_size": 4096,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "vocabulary_sha256": C.canonical_sha256({"fixture_vocab": 4096}),
        "semantic_binding": "mechanical_fixture_not_semantically_bound",
    }
    write_json(tokenizer_path, tokenizer)
    tokenizer_sha = C.sha256_file(tokenizer_path)

    hidden = 2
    base_config = {
        "schema": C.BASE_VLA_CONFIG_SCHEMA,
        "hidden_dim": hidden,
        "action_chunk_shape": [C.C, C.ACTION_DIM],
        "dtype": "float32",
    }
    base_state = {
        "vision_projection.weight": torch.eye(hidden),
        "vision_projection.bias": torch.zeros(hidden),
        "action_head.weight": torch.zeros(C.C * C.ACTION_DIM, hidden),
        "action_head.bias": torch.zeros(C.C * C.ACTION_DIM),
    }
    base_metadata = {
        "schema": C.BASE_VLA_CHECKPOINT_SCHEMA,
        "status": "frozen",
        "model_id": "pi0.5",
        "task_family": "libero",
        "c": C.C,
        "adim": C.ACTION_DIM,
        "tokenizer_sha256": tokenizer_sha,
        "semantic_binding": "mechanical_fixture_not_semantically_bound",
        "model_config": base_config,
    }
    base_vla = {
        **base_metadata,
        "state_dict": base_state,
        "runtime_state_sha256": C.model_runtime_state_sha256(
            base_metadata,
            base_state,
        ),
    }
    torch.save(base_vla, policy_path)

    m0_config = {
        "schema": C.M0_CONFIG_SCHEMA,
        "hidden_dim": hidden,
        "transition_kind": "residual_latent_mlp",
        "dtype": "float32",
    }
    m0_state = {
        "state_proj.weight": torch.zeros(hidden, C.RICH_DIM),
        "state_proj.bias": torch.zeros(hidden),
        "action_proj.weight": torch.zeros(hidden, C.C * C.ACTION_DIM),
        "action_proj.bias": torch.zeros(hidden),
        "next_delta_head.weight": torch.zeros(C.RICH_DIM, hidden),
        "next_delta_head.bias": torch.zeros(C.RICH_DIM),
    }
    m0_metadata = {
        "schema": C.M0_CHECKPOINT_SCHEMA,
        "status": "frozen",
        "task": C.TASK,
        "input_schema": "rich8217_v1",
        "input_dim": C.RICH_DIM,
        "c": C.C,
        "adim": C.ACTION_DIM,
        "training_data_role": "D0_only",
        "d0_tape_sha256": d0_sha,
        "semantic_binding": "mechanical_fixture_not_semantically_bound",
        "model_config": m0_config,
    }
    m0 = {
        **m0_metadata,
        "state_dict": m0_state,
        "runtime_state_sha256": C.model_runtime_state_sha256(
            m0_metadata,
            m0_state,
        ),
    }
    torch.save(m0, m0_path)

    phi_config = {
        "schema": C.PHI_CONFIG_SCHEMA,
        "input_dim": C.RICH_DIM,
        "hidden_dim": hidden,
        "dropout": 0.0,
    }
    phi_state = {
        "network.0.weight": torch.ones(C.RICH_DIM),
        "network.0.bias": torch.zeros(C.RICH_DIM),
        "network.1.weight": torch.zeros(hidden, C.RICH_DIM),
        "network.1.bias": torch.zeros(hidden),
        "network.4.weight": torch.zeros(3, hidden),
        "network.4.bias": torch.zeros(3),
    }
    phi_metadata = {
        "schema": C.PHI_CHECKPOINT_SCHEMA,
        "status": "frozen",
        "task": C.TASK,
        "input_schema": "rich8217_v1",
        "d0_tape_sha256": d0_sha,
        "gate0_status": "PASS",
        "gate0_result_sha256": C.canonical_sha256({"fixture_gate0": "PASS"}),
        "semantic_binding": "mechanical_fixture_not_semantically_bound",
        "model_config": phi_config,
    }
    phi = {
        **phi_metadata,
        "state_dict": phi_state,
        "runtime_state_sha256": C.model_runtime_state_sha256(
            phi_metadata,
            phi_state,
        ),
    }
    torch.save(phi, phi_path)
    m0_sha = C.sha256_file(m0_path)
    phi_sha = C.sha256_file(phi_path)
    actor_config = {
        "schema": C.ACTOR_CONFIG_SCHEMA,
        "hidden_dim": hidden,
        "activation": "tanh",
        "output_shape": [C.C, C.ACTION_DIM],
        "dtype": "float32",
        "residual_scale": actor_scale,
    }
    fixed_hyperparameters_sha = C.canonical_sha256(actor_config)

    actor_paths: dict[str, Path] = {}
    actor_seed_rows = (("A", 24601), ("B", 24601 if same_actor_seed else 24602))
    for collector_id, actor_seed in actor_seed_rows:
        mu = torch.zeros(C.RICH_DIM)
        sd = torch.ones(C.RICH_DIM)
        actor = {
            "schema": C.ACTOR_SCHEMA,
            "task": actor_task,
            "input_schema": "rich8217_v1",
            "input_dim": C.RICH_DIM,
            "c": C.C,
            "adim": C.ACTION_DIM,
            "collector_id": collector_id,
            "training_data_role": "D0_only",
            "env_steps": 0,
            "d0_tape_sha256": d0_sha,
            "m0_sha256": m0_sha,
            "phi_sha256": phi_sha,
            "actor_seed": actor_seed,
            "scale": actor_scale,
            "fixed_hyperparameters_sha256": fixed_hyperparameters_sha,
            "architecture_config": actor_config,
            "semantic_binding": "mechanical_fixture_not_semantically_bound",
            "normalizer_mu": mu,
            "normalizer_sd": sd,
            "normalizer_mu_sha256": C.tensor_sha256(mu),
            "normalizer_sd_sha256": C.tensor_sha256(sd),
            "state_dict": {
                "network.0.weight": torch.full(
                    (hidden, C.RICH_DIM),
                    float(actor_seed) / 100000.0,
                ),
                "network.0.bias": torch.zeros(hidden),
                "network.2.weight": torch.zeros(C.C * C.ACTION_DIM, hidden),
                "network.2.bias": torch.zeros(C.C * C.ACTION_DIM),
            },
        }
        actor["runtime_fingerprint_sha256"] = (
            C.actor_runtime_fingerprint_sha256(actor)
        )
        if bad_actor_runtime_fingerprint and collector_id == "A":
            actor["runtime_fingerprint_sha256"] = "0" * 64
        if extra_actor_field and collector_id == "A":
            actor["unregistered_runtime_switch"] = True
        actor_path = root / f"actor_{collector_id}.pt"
        torch.save(actor, actor_path)
        actor_paths[collector_id] = actor_path

    prior = sorted(
        set(range(8000, 8196))
        | set(range(8700, 8796))
        | set(range(8900, 8996))
    )
    ledger_path = root / "seed_ledger.json"
    ledger = {
        "schema": C.SEED_LEDGER_SCHEMA,
        "status": "frozen",
        "prior_chain3_used": prior,
        "gate3_evaluation_reserved": list(range(12000, 12096)),
        "gate3_actor_training_reserved": list(range(25000, 25016)),
        "other_forbidden_before_d1": list(range(13000, 13096)),
    }
    if overlap_d1_seed:
        ledger["other_forbidden_before_d1"].append(10000)
    if actor_seed_overlaps_gate3:
        ledger["gate3_actor_training_reserved"][0] = 24601
    if overlapping_seed_roles:
        ledger["other_forbidden_before_d1"].append(12000)
    write_json(ledger_path, ledger)

    assignments = []
    for episode_id in range(C.EXPECTED_EPISODES):
        if episode_id < C.EXPECTED_PER_COLLECTOR:
            collector_id = "A"
            local_id = episode_id
            env_seed = 10000 + local_id
        else:
            collector_id = "B"
            local_id = episode_id - C.EXPECTED_PER_COLLECTOR
            env_seed = 10100 + local_id
        assignments.append(
            {
                "episode_id": episode_id,
                "env_seed": env_seed,
                "collector_id": collector_id,
                "split": (
                    "fit" if local_id < C.EXPECTED_FIT_PER_COLLECTOR else "calibration"
                ),
            }
        )

    source_hashes = {
        relative: C.sha256_file(REPO / relative) for relative in C.REQUIRED_SOURCE_PATHS
    }
    if bad_source_hash:
        source_hashes["lcwm/chassis.py"] = "0" * 64
    policy_sha = C.sha256_file(policy_path)
    base_runtime_state = base_vla["runtime_state_sha256"]
    bddl_sha = source_hashes["bddl/chains/chain3_lr2.bddl"]
    runtime_specs = {
        "base_vla": (
            policy_sha,
            base_runtime_state,
            {
                "schema": "v246_base_vla_runtime_config_v1",
                "implementation": "pi0.5",
                "model_id": "pi0.5",
                "task": C.TASK,
                "c": C.C,
                "adim": C.ACTION_DIM,
            },
        ),
        "action_decoder": (
            policy_sha,
            base_runtime_state,
            {
                "schema": "v246_action_decoder_runtime_config_v1",
                "implementation": "pi0.5_action_decoder",
                "input_dtype": "float32",
                "output_dtype": "float32",
                "chunk_shape": [C.C, C.ACTION_DIM],
                "clipping": "registered_policy_decoder",
            },
        ),
        "feature_encoder": (
            policy_sha,
            base_runtime_state,
            {
                "schema": "v246_rich_encoder_runtime_config_v1",
                "implementation": "pi0.5_four_block_rich8217",
                "source": "two_camera_token_blocks_plus_proprio",
                "output_schema": "rich8217_v1",
                "output_dim": C.RICH_DIM,
                "camera_block_count": 4,
                "camera_block_dim": C.CAMERA_BLOCK_DIM,
                "proprio_dim": C.PROPRIO_DIM,
                "proprio_keys": list(C.PROPRIO_KEYS),
            },
        ),
        "environment": (
            bddl_sha,
            bddl_sha,
            {
                "schema": "v246_chain3_environment_runtime_config_v1",
                "implementation": "ChainEnv",
                "task": C.TASK,
                "bddl_path": "bddl/chains/chain3_lr2.bddl",
                "horizon_steps": C.HORIZON_STEPS,
                "technical_done_source": "info.done",
            },
        ),
    }
    runtime_components = {
        name: {
            "schema": C.RUNTIME_BINDING_SCHEMA,
            "name": name,
            "artifact_sha256": artifact_sha,
            "runtime_state_sha256": runtime_state_sha,
            "runtime_config": config,
            "runtime_fingerprint_sha256": (
                C.runtime_component_fingerprint_sha256(
                    artifact_sha,
                    runtime_state_sha,
                    config,
                )
            ),
        }
        for name, (artifact_sha, runtime_state_sha, config) in runtime_specs.items()
    }
    registration = {
        "schema": C.REGISTRATION_SCHEMA,
        "status": "frozen_before_collection",
        "evidence_class": "mechanical_contract_test_only_no_gate1",
        "collection": {
            "task": C.TASK,
            "horizon_steps": C.HORIZON_STEPS,
            "c": C.C,
            "action_shape": [C.C, C.ACTION_DIM],
            "rich_latent_dim": C.RICH_DIM,
            "proprio_dim": C.PROPRIO_DIM,
            "proprio_keys": list(C.PROPRIO_KEYS),
            "camera_token_blocks": [list(block) for block in C.CAMERA_TOKEN_BLOCKS],
            "expected_episodes": C.EXPECTED_EPISODES,
            "expected_per_collector": C.EXPECTED_PER_COLLECTOR,
            "fit_per_collector": C.EXPECTED_FIT_PER_COLLECTOR,
            "calibration_per_collector": C.EXPECTED_CAL_PER_COLLECTOR,
            "perturbation_scope": "episode_policy",
            "chunk_noise": "forbidden",
            "sigma_value": 0.0,
            "success_termination": "ignored",
            "technical_done_source": "info.done",
            "outcome_storage": "sealed_evaluation_sidecar_only",
            "decoded_action_trace": "base_and_executed_float32_exact",
            "changed_action_step_definition": "any_exact_component_difference",
            "decoded_action_delta_norm": "episode_frobenius_l2_float64",
            "min_changed_action_fraction_per_episode": 1.0,
            "min_decoded_action_delta_l2_per_episode": 0.1,
            "meaningful_decoded_step_l2": C.MEANINGFUL_DECODED_STEP_L2,
            "min_meaningful_action_fraction_per_episode": (
                C.MIN_MEANINGFUL_ACTION_FRACTION
            ),
            "min_decoded_step_l2_p10_per_episode": C.MIN_DECODED_STEP_L2_P10,
            "max_decoded_step_l2_per_episode": C.MAX_DECODED_STEP_L2,
            "attempt_ledger_path": str(root / "formal_attempt_ledger.jsonl"),
            "attempt_ledger_policy": (
                "single_preregistered_append_only_no_retry_or_seed_replacement"
            ),
            "episode_assignments": assignments,
        },
        "inputs": {
            "d0_tape": {"path": str(d0_path), "sha256": d0_sha},
            "m0_checkpoint": {"path": str(m0_path), "sha256": m0_sha},
            "phi_checkpoint": {"path": str(phi_path), "sha256": phi_sha},
            "seed_ledger": {
                "path": str(ledger_path),
                "sha256": C.sha256_file(ledger_path),
            },
            "base_vla_checkpoint": {
                "path": str(policy_path),
                "sha256": policy_sha,
            },
            "tokenizer_manifest": {
                "path": str(tokenizer_path),
                "sha256": tokenizer_sha,
            },
            "actors": {
                collector_id: {
                    "path": str(actor_paths[collector_id]),
                    "sha256": C.sha256_file(actor_paths[collector_id]),
                }
                for collector_id in C.COLLECTOR_IDS
            },
        },
        "runtime_components": runtime_components,
        "provenance": {"source_files_sha256": source_hashes},
    }
    registration["self_sha256"] = C.canonical_sha256(registration)
    registration_path = root / "registration.json"
    write_json(registration_path, registration)
    return Fixture(
        root=root,
        registration=registration_path,
        actor_a=actor_paths["A"],
        actor_b=actor_paths["B"],
        d0_tape=d0_path,
        m0=m0_path,
        phi=phi_path,
        policy=policy_path,
        tokenizer=tokenizer_path,
        seed_ledger=ledger_path,
    )


def new_attempt_ledger(
    root: Path,
    registration: C.ValidatedRegistration,
    name: str,
) -> C.AttemptLedger:
    return C.AttemptLedger.create_contract_test_only(
        root / f"{name}.jsonl",
        registration,
    )


def completed_attempt_ledger(
    root: Path,
    registration: C.ValidatedRegistration,
    name: str,
) -> C.AttemptLedger:
    del root, name
    ledger = C.AttemptLedger.create_preregistered_path_contract_test_only(
        registration
    )
    for assignment in registration.assignments:
        ledger.start(assignment)
        ledger.finish(
            assignment,
            "completed",
            episode_payload_sha256=mechanical_fixture_payload_sha256(
                registration,
                assignment.episode_id,
            ),
        )
    return ledger


def mechanical_fixture_payload_sha256(
    registration: C.ValidatedRegistration,
    episode_id: int,
) -> str:
    return C.canonical_sha256(
        {
            "mechanical_test_only": True,
            "registration_sha256": registration.file_sha256,
            "episode_id": episode_id,
        }
    )


def mechanical_fixture_payload_sha256_by_id(
    registration: C.ValidatedRegistration,
) -> dict[int, str]:
    return {
        assignment.episode_id: mechanical_fixture_payload_sha256(
            registration,
            assignment.episode_id,
        )
        for assignment in registration.assignments
    }


def base_vla_live_runtime_state_sha256(checkpoint: dict[str, Any]) -> str:
    metadata_keys = (
        "schema",
        "status",
        "model_id",
        "task_family",
        "c",
        "adim",
        "tokenizer_sha256",
        "semantic_binding",
    )
    metadata = {key: checkpoint[key] for key in metadata_keys}
    metadata["model_config"] = checkpoint["model_config"]
    return C.model_runtime_state_sha256(metadata, checkpoint["state_dict"])


class FakeEnv:
    def __init__(
        self,
        registration: C.ValidatedRegistration,
        *,
        success_at: int | None = None,
        raw_done_at: int | None = None,
        truncated_at: int | None = None,
    ):
        runtime = registration.runtime_components["environment"]
        self.artifact_sha256 = runtime.artifact_sha256
        self.bddl_path = REPO / "bddl/chains/chain3_lr2.bddl"
        self.runtime_config = dict(runtime.runtime_config)
        self.success_at = success_at
        self.raw_done_at = raw_done_at
        self.truncated_at = truncated_at
        self.step_count = 0
        self.reset_count = 0
        self.actions: list[np.ndarray] = []

    def recompute_runtime_fingerprint_sha256(self) -> str:
        return C.runtime_component_fingerprint_sha256(
            self.artifact_sha256,
            C.sha256_file(self.bddl_path),
            self.runtime_config,
        )

    @staticmethod
    def observation(step: int, seed: int) -> dict[str, Any]:
        base = torch.arange(C.CAMERA_BLOCK_DIM, dtype=torch.float32)
        base = base / float(C.CAMERA_BLOCK_DIM)
        return {
            "step": step,
            "seed": seed,
            "camera_blocks": tuple(base + float(index + step) for index in range(4)),
            "proprio": torch.arange(C.PROPRIO_DIM, dtype=torch.float32) + float(step),
        }

    def reset(self, seed: int):
        self.reset_count += 1
        self.step_count = 0
        self.actions = []
        self.seed = seed
        return self.observation(0, seed), {}

    def step(self, action: np.ndarray):
        self.actions.append(np.array(action, copy=True))
        self.step_count += 1
        success = self.step_count == self.success_at
        raw_done = self.step_count == self.raw_done_at
        truncated = self.step_count == self.truncated_at
        terminated = success or raw_done
        info = {"done": raw_done, "is_success": success}
        return (
            self.observation(self.step_count, self.seed),
            0.0,
            terminated,
            truncated,
            info,
        )


class FakeFeatureEncoder:
    def __init__(self, registration: C.ValidatedRegistration):
        runtime = registration.runtime_components["feature_encoder"]
        self.artifact_sha256 = runtime.artifact_sha256
        self.runtime_config = dict(runtime.runtime_config)
        self.base_vla_checkpoint = torch.load(
            registration.base_vla.path,
            map_location="cpu",
            weights_only=True,
        )

    def recompute_runtime_fingerprint_sha256(self) -> str:
        return C.runtime_component_fingerprint_sha256(
            self.artifact_sha256,
            base_vla_live_runtime_state_sha256(self.base_vla_checkpoint),
            self.runtime_config,
        )

    def encode(self, obs: dict[str, Any]) -> C.RichEncoding:
        blocks = tuple(block.detach().clone() for block in obs["camera_blocks"])
        proprio = obs["proprio"].detach().clone()
        value = torch.cat((*blocks, proprio)).float()
        camera_hashes = [C.tensor_sha256(block) for block in blocks]
        proprio_hash = C.tensor_sha256(proprio)
        evidence = {
            "schema": C.RICH_ENCODING_SCHEMA,
            "observation_sha256": C.canonical_sha256(
                {
                    "camera_block_sha256": camera_hashes,
                    "proprio_sha256": proprio_hash,
                }
            ),
            "camera_block_sha256": camera_hashes,
            "proprio_sha256": proprio_hash,
            "feature_encoder_runtime_fingerprint_sha256": (
                self.recompute_runtime_fingerprint_sha256()
            ),
        }
        return C.RichEncoding(value=value, evidence=evidence)


class ClockFeatureEncoder(FakeFeatureEncoder):
    def encode(self, obs: dict[str, Any]) -> C.RichEncoding:
        value = torch.full((C.RICH_DIM,), float(obs["step"]))
        blocks = [
            value[
                index * C.CAMERA_BLOCK_DIM : (index + 1) * C.CAMERA_BLOCK_DIM
            ]
            for index in range(4)
        ]
        camera_hashes = [C.tensor_sha256(block) for block in blocks]
        proprio_hash = C.tensor_sha256(value[-C.PROPRIO_DIM :])
        evidence = {
            "schema": C.RICH_ENCODING_SCHEMA,
            "observation_sha256": C.canonical_sha256(
                {
                    "camera_block_sha256": camera_hashes,
                    "proprio_sha256": proprio_hash,
                }
            ),
            "camera_block_sha256": camera_hashes,
            "proprio_sha256": proprio_hash,
            "feature_encoder_runtime_fingerprint_sha256": (
                self.recompute_runtime_fingerprint_sha256()
            ),
        }
        return C.RichEncoding(value=value, evidence=evidence)


class FakeController:
    def __init__(
        self,
        registration: C.ValidatedRegistration,
        collector_id: str = "A",
        *,
        mutate_runtime_at: int | None = None,
        artifact_sha256: str | None = None,
        delta_value: float = 0.01,
        decode_mode: str = "identity",
        chunk_noise: bool = False,
        pair_cached_entropy: bool = False,
        global_rng_noise: bool = False,
        delta_mode: str = "dense",
    ):
        actor = registration.actors[collector_id]
        self.collector_id = actor.collector_id
        self.actor_artifact_sha256 = artifact_sha256 or actor.sha256
        self.base_vla_artifact_sha256 = registration.base_vla.sha256
        self.action_decoder_artifact_sha256 = registration.base_vla.sha256
        self.actor_checkpoint = torch.load(
            actor.path,
            map_location="cpu",
            weights_only=True,
        )
        self.base_vla_checkpoint = torch.load(
            registration.base_vla.path,
            map_location="cpu",
            weights_only=True,
        )
        self.runtime_components = registration.runtime_components
        self.mutate_runtime_at = mutate_runtime_at
        self.delta_value = delta_value
        self.decode_mode = decode_mode
        self.chunk_noise = chunk_noise
        self.pair_cached_entropy = pair_cached_entropy
        self.global_rng_noise = global_rng_noise
        self.delta_mode = delta_mode
        self._entropy_pair_value: float | None = None
        self.propose_calls = 0
        self.reset_count = 0
        self.decode_calls = 0

    def _base_runtime_state_sha256(self) -> str:
        return base_vla_live_runtime_state_sha256(self.base_vla_checkpoint)

    def recompute_runtime_fingerprints_sha256(self) -> dict[str, str]:
        base_runtime_state = self._base_runtime_state_sha256()
        return {
            "actor": C.actor_runtime_fingerprint_sha256(self.actor_checkpoint),
            "base_vla": C.runtime_component_fingerprint_sha256(
                self.base_vla_artifact_sha256,
                base_runtime_state,
                self.runtime_components["base_vla"].runtime_config,
            ),
            "action_decoder": C.runtime_component_fingerprint_sha256(
                self.action_decoder_artifact_sha256,
                base_runtime_state,
                self.runtime_components["action_decoder"].runtime_config,
            ),
        }

    def reset_episode(self, _seed: int) -> None:
        self.reset_count += 1

    def propose(
        self,
        _obs: Any,
        _rich_z: torch.Tensor,
        chunk_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.propose_calls += 1
        base = torch.full((C.C, C.ACTION_DIM), float(chunk_index) / 1000.0)
        delta = torch.full((C.C, C.ACTION_DIM), self.delta_value)
        if self.delta_mode == "sparse_epsilon":
            delta.fill_(0.000001)
            if chunk_index == 0:
                delta[0, 0] = 0.04
        elif self.delta_mode != "dense":
            raise AssertionError(f"unknown delta mode {self.delta_mode}")
        if self.chunk_noise:
            delta = delta + float(self.propose_calls) / 10000.0
        if self.pair_cached_entropy:
            if self.propose_calls % 2 == 1:
                self._entropy_pair_value = int.from_bytes(os.urandom(8), "little") / 2**64
            assert self._entropy_pair_value is not None
            delta = delta + self._entropy_pair_value / 1000.0
        if self.global_rng_noise:
            delta = delta + float(torch.rand(())) / 1000.0
        if self.mutate_runtime_at == chunk_index:
            self.actor_checkpoint["state_dict"]["network.0.bias"][0] += 1.0
        return base, delta

    def decode(self, u_exec: torch.Tensor) -> np.ndarray:
        self.decode_calls += 1
        result = u_exec.detach().cpu().numpy().copy()
        if self.decode_mode == "constant":
            result.fill(0.0)
        elif self.decode_mode == "stateful":
            result += np.float32(self.decode_calls / 1000.0)
        elif self.decode_mode == "float64":
            result = result.astype(np.float64)
        elif self.decode_mode == "triple":
            result *= np.float32(3.0)
        elif self.decode_mode == "times_ten":
            result *= np.float32(10.0)
        elif self.decode_mode != "identity":
            raise AssertionError(f"unknown decode mode {self.decode_mode}")
        return result


def clock_feature(obs: dict[str, Any]) -> torch.Tensor:
    return torch.full((C.RICH_DIM,), float(obs["step"]))


def synthetic_mechanical_episode(
    registration: C.ValidatedRegistration,
    assignment: C.EpisodeAssignment,
    encoder: FakeFeatureEncoder,
) -> C.EpisodeCollection:
    """Build one tiny-dimensional 75/76 payload without invoking an env."""
    encoded = [
        encoder.encode(FakeEnv.observation(boundary_id * C.C, assignment.env_seed))
        for boundary_id in range(C.BOUNDARIES_PER_EPISODE)
    ]
    boundaries = torch.stack([item.value for item in encoded])
    u_base = torch.zeros(
        C.CHUNKS_PER_EPISODE,
        C.C,
        C.ACTION_DIM,
        dtype=torch.float32,
    )
    delta = torch.full_like(u_base, 0.01)
    u_exec = u_base + delta
    metrics = C._decoded_action_change_metrics(
        u_base,
        u_exec,
        registration.action_change,
    )
    actor = registration.actors[assignment.collector_id]
    runtime_fingerprints = {
        "actor": actor.runtime_fingerprint_sha256,
        **{
            name: registration.runtime_components[
                name
            ].runtime_fingerprint_sha256
            for name in C.REGISTERED_RUNTIME_COMPONENTS
        },
    }
    return C.EpisodeCollection(
        assignment=assignment,
        boundaries=boundaries,
        u_base=u_base,
        delta=delta,
        u_exec=u_exec,
        env_action_base=u_base.clone(),
        env_action_exec=u_exec.clone(),
        changed_action_steps=metrics.changed_action_steps,
        meaningful_action_steps=metrics.meaningful_action_steps,
        decoded_action_delta_l2=metrics.decoded_action_delta_l2,
        decoded_step_l2_p10=metrics.decoded_step_l2_p10,
        decoded_step_l2_median=metrics.decoded_step_l2_median,
        decoded_step_l2_p90=metrics.decoded_step_l2_p90,
        decoded_step_l2_max=metrics.decoded_step_l2_max,
        actor_artifact_sha256=actor.sha256,
        actor_runtime_fingerprint_sha256=actor.runtime_fingerprint_sha256,
        runtime_component_fingerprints_sha256=runtime_fingerprints,
        boundary_evidence=tuple(item.evidence for item in encoded),
        evaluation_sidecar={
            "schema": C.SIDECAR_SCHEMA,
            "episode_id": assignment.episode_id,
            "env_seed": assignment.env_seed,
            "success_observed": False,
            "first_success_step": None,
            "raw_done_at_horizon": False,
            "truncated_at_horizon": False,
        },
    )


def test_registration_validates_before_environment_import() -> None:
    assert C.environment_modules_loaded() == []
    with tempfile.TemporaryDirectory() as directory:
        fixture = build_fixture(Path(directory))
        validated = C.validate_registration(fixture.registration)
        assert len(validated.assignments) == 96
        assert validated.d0.rows == 2
        assert validated.actors["A"].scale == validated.actors["B"].scale == 0.04
        assert validated.action_change.min_changed_action_fraction_per_episode == 1.0
        assert all(
            C.is_sha256(actor.runtime_fingerprint_sha256)
            for actor in validated.actors.values()
        )
        d0 = torch.load(fixture.d0_tape, map_location="cpu", weights_only=True)

        class OutcomePoison(dict[str, Any]):
            def __getitem__(self, key: str) -> Any:
                if key == "success":
                    raise AssertionError("D0 consumer read legacy success")
                return super().__getitem__(key)

        d0_view = C.d0_training_tensor_view(OutcomePoison(d0))
        assert set(d0_view) == set(C.TRAINING_FEATURE_FIELDS)
        unbound_ledger = new_attempt_ledger(
            Path(directory),
            validated,
            "unpublishable-alternate",
        )
        error = expect_raises(
            C.ContractError,
            lambda: C.validate_attempt_ledger_for_publication(
                unbound_ledger,
                validated,
            ),
        )
        assert "one preregistered ledger" in str(error)
        assert C.environment_modules_loaded() == []


def test_hash_actor_tape_seed_and_source_fail_before_environment_import() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = build_fixture(Path(directory) / "hash")
        fixture.actor_a.write_bytes(fixture.actor_a.read_bytes() + b"drift")
        error = expect_raises(
            C.ContractError,
            lambda: C.validate_registration(fixture.registration),
        )
        assert "SHA mismatch" in str(error)
        assert C.environment_modules_loaded() == []


def test_exact_field_closures_and_structured_artifacts_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)

        def registration_case(
            name: str,
            mutate: Callable[[dict[str, Any]], None],
            expected: str,
        ) -> None:
            fixture = build_fixture(root / name)
            payload = json.loads(fixture.registration.read_text())
            mutate(payload)
            rewrite_registration(fixture.registration, payload)
            error = expect_raises(
                C.ContractError,
                lambda: C.validate_registration(fixture.registration),
            )
            assert expected in str(error), (expected, str(error))
            assert C.environment_modules_loaded() == []

        registration_case(
            "registration-extra",
            lambda payload: payload.__setitem__("success", True),
            "registration top-level field closure mismatch",
        )
        registration_case(
            "collection-extra",
            lambda payload: payload["collection"].__setitem__(
                "trajectory_length",
                750,
            ),
            "registration.collection field closure mismatch",
        )
        registration_case(
            "collection-c-float",
            lambda payload: payload["collection"].__setitem__("c", 10.0),
            "registration.collection.c mismatch",
        )
        registration_case(
            "collection-sigma-bool",
            lambda payload: payload["collection"].__setitem__(
                "sigma_value",
                False,
            ),
            "registration.collection.sigma_value mismatch",
        )
        registration_case(
            "assignment-extra",
            lambda payload: payload["collection"]["episode_assignments"][0].__setitem__(
                "policy_id_hint",
                "A",
            ),
            "episode assignment field closure mismatch",
        )
        registration_case(
            "inputs-extra",
            lambda payload: payload["inputs"].__setitem__("success", False),
            "registration.inputs field closure mismatch",
        )
        registration_case(
            "provenance-extra",
            lambda payload: payload["provenance"].__setitem__(
                "trajectory_length",
                750,
            ),
            "registration.provenance field closure mismatch",
        )
        registration_case(
            "artifact-ref-extra",
            lambda payload: payload["inputs"]["d0_tape"].__setitem__(
                "policy_id_hint",
                "none",
            ),
            "artifact-reference field closure mismatch",
        )
        registration_case(
            "runtime-extra",
            lambda payload: payload["runtime_components"]["base_vla"].__setitem__(
                "success",
                True,
            ),
            "runtime component base_vla field closure mismatch",
        )

        seed_fixture = build_fixture(root / "seed-ledger-extra")
        seed_ledger = json.loads(seed_fixture.seed_ledger.read_text())
        seed_ledger["policy_id_hint"] = "forbidden"
        write_json(seed_fixture.seed_ledger, seed_ledger)
        payload = json.loads(seed_fixture.registration.read_text())
        payload["inputs"]["seed_ledger"]["sha256"] = C.sha256_file(
            seed_fixture.seed_ledger
        )
        rewrite_registration(seed_fixture.registration, payload)
        error = expect_raises(
            C.ContractError,
            lambda: C.validate_registration(seed_fixture.registration),
        )
        assert "seed ledger field closure mismatch" in str(error)

        d0_fixture = build_fixture(root / "d0-extra")
        d0 = torch.load(d0_fixture.d0_tape, map_location="cpu", weights_only=True)
        d0["trajectory_length"] = torch.tensor([20])
        torch.save(d0, d0_fixture.d0_tape)
        payload = json.loads(d0_fixture.registration.read_text())
        payload["inputs"]["d0_tape"]["sha256"] = C.sha256_file(
            d0_fixture.d0_tape
        )
        rewrite_registration(d0_fixture.registration, payload)
        error = expect_raises(
            C.ContractError,
            lambda: C.validate_registration(d0_fixture.registration),
        )
        assert "D0 tape field mismatch" in str(error)

        undeclared_success = build_fixture(root / "d0-undeclared-success")
        d0 = torch.load(
            undeclared_success.d0_tape,
            map_location="cpu",
            weights_only=True,
        )
        d0["legacy_outcome_fields"] = []
        torch.save(d0, undeclared_success.d0_tape)
        payload = json.loads(undeclared_success.registration.read_text())
        payload["inputs"]["d0_tape"]["sha256"] = C.sha256_file(
            undeclared_success.d0_tape
        )
        rewrite_registration(undeclared_success.registration, payload)
        error = expect_raises(
            C.ContractError,
            lambda: C.validate_registration(undeclared_success.registration),
        )
        assert "D0 tape field mismatch" in str(error)

        d0_dtype = build_fixture(root / "d0-bookkeeping-dtype")
        d0 = torch.load(d0_dtype.d0_tape, map_location="cpu", weights_only=True)
        d0["episode"] = d0["episode"].float()
        torch.save(d0, d0_dtype.d0_tape)
        payload = json.loads(d0_dtype.registration.read_text())
        payload["inputs"]["d0_tape"]["sha256"] = C.sha256_file(
            d0_dtype.d0_tape
        )
        rewrite_registration(d0_dtype.registration, payload)
        error = expect_raises(
            C.ContractError,
            lambda: C.validate_registration(d0_dtype.registration),
        )
        assert "D0 episode must be [N]" in str(error)

        for name, attribute, invalid_payload, expected in (
            (
                "m0-arbitrary",
                "m0",
                {"state_dict": {"test.weight": torch.ones(1)}},
                "M0 checkpoint field closure mismatch",
            ),
            (
                "phi-arbitrary",
                "phi",
                {"state_dict": {"test.weight": torch.ones(1)}},
                "Phi checkpoint field closure mismatch",
            ),
            (
                "policy-arbitrary",
                "policy",
                {"state_dict": {"test.weight": torch.ones(1)}},
                "base VLA checkpoint field closure mismatch",
            ),
        ):
            fixture = build_fixture(root / name)
            artifact = getattr(fixture, attribute)
            torch.save(invalid_payload, artifact)
            payload = json.loads(fixture.registration.read_text())
            input_name = {
                "m0": "m0_checkpoint",
                "phi": "phi_checkpoint",
                "policy": "base_vla_checkpoint",
            }[attribute]
            payload["inputs"][input_name]["sha256"] = C.sha256_file(artifact)
            rewrite_registration(fixture.registration, payload)
            error = expect_raises(
                C.ContractError,
                lambda fixture=fixture: C.validate_registration(
                    fixture.registration
                ),
            )
            assert expected in str(error)

        tokenizer_fixture = build_fixture(root / "tokenizer-arbitrary")
        write_json(tokenizer_fixture.tokenizer, {"vocab": ["arbitrary"]})
        payload = json.loads(tokenizer_fixture.registration.read_text())
        payload["inputs"]["tokenizer_manifest"]["sha256"] = C.sha256_file(
            tokenizer_fixture.tokenizer
        )
        rewrite_registration(tokenizer_fixture.registration, payload)
        error = expect_raises(
            C.ContractError,
            lambda: C.validate_registration(tokenizer_fixture.registration),
        )
        assert "tokenizer manifest field closure mismatch" in str(error)

        actor_fixture = build_fixture(root / "actor-test-weight")
        actor = torch.load(actor_fixture.actor_a, map_location="cpu", weights_only=True)
        actor["state_dict"] = {"test.weight": torch.ones(1)}
        torch.save(actor, actor_fixture.actor_a)
        payload = json.loads(actor_fixture.registration.read_text())
        payload["inputs"]["actors"]["A"]["sha256"] = C.sha256_file(
            actor_fixture.actor_a
        )
        rewrite_registration(actor_fixture.registration, payload)
        error = expect_raises(
            C.ContractError,
            lambda: C.validate_registration(actor_fixture.registration),
        )
        assert "state_dict field mismatch" in str(error)

        actor_dtype = build_fixture(root / "actor-normalizer-dtype")
        actor = torch.load(actor_dtype.actor_a, map_location="cpu", weights_only=True)
        actor["normalizer_mu"] = actor["normalizer_mu"].double()
        torch.save(actor, actor_dtype.actor_a)
        payload = json.loads(actor_dtype.registration.read_text())
        payload["inputs"]["actors"]["A"]["sha256"] = C.sha256_file(
            actor_dtype.actor_a
        )
        rewrite_registration(actor_dtype.registration, payload)
        error = expect_raises(
            C.ContractError,
            lambda: C.validate_registration(actor_dtype.registration),
        )
        assert "normalizer must be float32" in str(error)

        bad_actor = build_fixture(Path(directory) / "actor", actor_task="chain1b_lr2")
        error = expect_raises(
            C.ContractError,
            lambda: C.validate_registration(bad_actor.registration),
        )
        assert "actor task mismatch" in str(error)
        assert C.environment_modules_loaded() == []

        bad_tape = build_fixture(Path(directory) / "tape", d0_task="chain1b_lr2")
        error = expect_raises(
            C.ContractError,
            lambda: C.validate_registration(bad_tape.registration),
        )
        assert "D0 task/c mismatch" in str(error)
        assert C.environment_modules_loaded() == []

        bad_seed = build_fixture(Path(directory) / "seed", overlap_d1_seed=True)
        error = expect_raises(
            C.ContractError,
            lambda: C.validate_registration(bad_seed.registration),
        )
        assert "D1 seeds overlap" in str(error)
        assert C.environment_modules_loaded() == []

        bad_source = build_fixture(Path(directory) / "source", bad_source_hash=True)
        error = expect_raises(
            C.ContractError,
            lambda: C.validate_registration(bad_source.registration),
        )
        assert "registered source drift" in str(error)
        assert C.environment_modules_loaded() == []

        bad_runtime = build_fixture(
            Path(directory) / "runtime",
            bad_actor_runtime_fingerprint=True,
        )
        error = expect_raises(
            C.ContractError,
            lambda: C.validate_registration(bad_runtime.registration),
        )
        assert "runtime fingerprint mismatch" in str(error)

        extra_actor = build_fixture(
            Path(directory) / "actor-field",
            extra_actor_field=True,
        )
        error = expect_raises(
            C.ContractError,
            lambda: C.validate_registration(extra_actor.registration),
        )
        assert "actor field mismatch" in str(error)

        actor_seed_overlap = build_fixture(
            Path(directory) / "actor-seed-overlap",
            actor_seed_overlaps_gate3=True,
        )
        error = expect_raises(
            C.ContractError,
            lambda: C.validate_registration(actor_seed_overlap.registration),
        )
        assert "actor_seed overlaps gate3_actor_training_reserved" in str(error)

        role_overlap = build_fixture(
            Path(directory) / "role-overlap",
            overlapping_seed_roles=True,
        )
        error = expect_raises(
            C.ContractError,
            lambda: C.validate_registration(role_overlap.registration),
        )
        assert "seed roles" in str(error) and "overlap" in str(error)

        same_seed = build_fixture(
            Path(directory) / "same-actor-seed",
            same_actor_seed=True,
        )
        error = expect_raises(
            C.ContractError,
            lambda: C.validate_registration(same_seed.registration),
        )
        assert "collector actor seeds must differ" in str(error)

        weak_scale = build_fixture(
            Path(directory) / "weak-actor-scale",
            actor_scale=0.001,
        )
        error = expect_raises(
            C.ContractError,
            lambda: C.validate_registration(weak_scale.registration),
        )
        assert "residual scale must equal fixed" in str(error)
        assert not C.exact_json_equal(96.0, 96)
        assert not C.exact_json_equal(False, 0)
        assert C.environment_modules_loaded() == []


def test_success_at_20_still_runs_750_and_derives_75_76() -> None:
    with tempfile.TemporaryDirectory() as directory:
        validated = C.validate_registration(
            build_fixture(Path(directory)).registration
        )
        assignment = validated.assignments[0]
        actor = validated.actors["A"]
        controller = FakeController(validated)
        env = FakeEnv(validated, success_at=20)
        success_ledger = new_attempt_ledger(
            Path(directory),
            validated,
            "success",
        )
        with mock.patch.object(C.torch.cuda, "is_available", return_value=False):
            episode = C.collect_fixed_horizon_episode_contract_test_only(
                validated,
                assignment.episode_id,
                controller,
                env,
                FakeFeatureEncoder(validated),
                success_ledger,
            )
        assert env.reset_count == 1
        assert env.step_count == C.HORIZON_STEPS
        assert controller.propose_calls == 3 * C.CHUNKS_PER_EPISODE
        assert episode.boundaries.shape == (
            C.BOUNDARIES_PER_EPISODE,
            C.RICH_DIM,
        )
        assert episode.u_exec.shape == (
            C.CHUNKS_PER_EPISODE,
            C.C,
            C.ACTION_DIM,
        )
        assert float(episode.boundaries[-1, 0]) == C.HORIZON_STEPS
        assert episode.evaluation_sidecar["success_observed"] is True
        assert episode.evaluation_sidecar["first_success_step"] == 20
        assert episode.changed_action_steps == C.HORIZON_STEPS
        ledger_rows = C.read_attempt_ledger(
            success_ledger.path,
            validated.file_sha256,
        )
        assert ledger_rows[-1]["episode_payload_sha256"] == (
            C.episode_collection_payload_sha256(validated, episode)
        )
        stepped_actions = np.stack(env.actions).reshape(
            C.CHUNKS_PER_EPISODE,
            C.C,
            C.ACTION_DIM,
        )
        assert np.array_equal(stepped_actions, episode.env_action_exec.numpy())
        tape, boundary, trace = C.derive_episode_payloads(validated, episode)
        assert set(tape) == set(C.DATA_TAPE_FIELDS)
        assert not (set(tape) & set(C.FORBIDDEN_DATA_FIELDS))
        assert tape["z"].shape == (75, C.RICH_DIM)
        assert boundary["z"].shape == (76, C.RICH_DIM)
        assert torch.equal(tape["z"], boundary["z"][:-1])
        assert torch.equal(tape["z_next"], boundary["z"][1:])
        assert torch.equal(trace["u_exec"], trace["u_base"] + trace["delta"])
        assert not torch.equal(trace["env_action_base"], trace["env_action_exec"])
        assert trace["actor_artifact_sha256"] == actor.sha256
        assert (
            trace["actor_runtime_fingerprint_sha256"]
            == actor.runtime_fingerprint_sha256
        )
        assert "success" not in trace and "success_step" not in trace
        training_view = C.training_tensor_view(tape)
        assert set(training_view) == set(C.TRAINING_FEATURE_FIELDS)
        assert "episode" not in training_view and "t" not in training_view
        error = expect_raises(
            C.ContractError,
            lambda: C.training_tensor_view(tape, ("z", "episode")),
        )
        assert "may read only z/u/z_next" in str(error)
        expect_raises(C.ContractError, lambda: C.training_tensor_view(trace))
        expect_raises(
            C.ContractError,
            lambda: C.training_tensor_view(episode.evaluation_sidecar),
        )

        tampered_sidecar = dict(episode.evaluation_sidecar)
        tampered_sidecar["policy_id_hint"] = "A"
        error = expect_raises(
            C.ContractError,
            lambda: C.derive_episode_payloads(
                validated,
                replace(episode, evaluation_sidecar=tampered_sidecar),
            ),
        )
        assert "sidecar field closure mismatch" in str(error)
        error = expect_raises(
            C.ContractError,
            lambda: C.derive_episode_payloads(
                validated,
                replace(episode, changed_action_steps=749),
            ),
        )
        assert "changed-action count changed" in str(error)
        tampered_runtime = dict(episode.runtime_component_fingerprints_sha256)
        tampered_runtime["base_vla"] = "0" * 64
        error = expect_raises(
            C.ContractError,
            lambda: C.derive_episode_payloads(
                validated,
                replace(
                    episode,
                    runtime_component_fingerprints_sha256=tampered_runtime,
                ),
            ),
        )
        assert "runtime-component fingerprints mismatch" in str(error)

        aggregate = {
            "schema": C.SIDECAR_SCHEMA,
            "registration_sha256": validated.file_sha256,
            "episodes": [
                {
                    "schema": C.SIDECAR_SCHEMA,
                    "episode_id": item.episode_id,
                    "env_seed": item.env_seed,
                    "success_observed": False,
                    "first_success_step": None,
                    "raw_done_at_horizon": False,
                    "truncated_at_horizon": False,
                }
                for item in validated.assignments
            ],
        }
        C.validate_evaluation_sidecar(validated, aggregate)
        tampered_aggregate = {
            **aggregate,
            "episodes": [dict(item) for item in aggregate["episodes"]],
        }
        tampered_aggregate["episodes"][0]["trajectory_length"] = 20
        error = expect_raises(
            C.ContractError,
            lambda: C.validate_evaluation_sidecar(validated, tampered_aggregate),
        )
        assert "sidecar field closure mismatch" in str(error)
        typed_aggregate = {
            **aggregate,
            "episodes": [dict(item) for item in aggregate["episodes"]],
        }
        typed_aggregate["episodes"][0]["episode_id"] = 0.0
        error = expect_raises(
            C.ContractError,
            lambda: C.validate_evaluation_sidecar(validated, typed_aggregate),
        )
        assert "sidecar identity/type mismatch" in str(error)
        bool_typed_aggregate = {
            **aggregate,
            "episodes": [dict(item) for item in aggregate["episodes"]],
        }
        bool_typed_aggregate["episodes"][0]["success_observed"] = 0
        error = expect_raises(
            C.ContractError,
            lambda: C.validate_evaluation_sidecar(
                validated,
                bool_typed_aggregate,
            ),
        )
        assert "sidecar identity/type mismatch" in str(error)


def test_raw_done_before_750_halts_and_policy_assignment_is_immutable() -> None:
    with tempfile.TemporaryDirectory() as directory:
        validated = C.validate_registration(
            build_fixture(Path(directory)).registration
        )
        assignment = validated.assignments[0]
        env = FakeEnv(validated, raw_done_at=23)
        failure_ledger = new_attempt_ledger(Path(directory), validated, "raw-done")
        with mock.patch.object(C.torch.cuda, "is_available", return_value=False):
            error = expect_raises(
                C.TechnicalTermination,
                lambda: C.collect_fixed_horizon_episode_contract_test_only(
                    validated,
                    assignment.episode_id,
                    FakeController(validated),
                    env,
                    FakeFeatureEncoder(validated),
                    failure_ledger,
                ),
            )
        assert "raw_done at 23" in str(error)
        assert env.step_count == 23
        rows = C.read_attempt_ledger(failure_ledger.path, validated.file_sha256)
        assert [row.get("event") for row in rows[1:]] == [
            "started",
            "technical_failure",
        ]
        retry_env = FakeEnv(validated)
        error = expect_raises(
            C.ContractError,
            lambda: C.collect_fixed_horizon_episode_contract_test_only(
                validated,
                assignment.episode_id,
                FakeController(validated),
                retry_env,
                FakeFeatureEncoder(validated),
                failure_ledger,
            ),
        )
        assert "already attempted; replacement forbidden" in str(error)
        assert retry_env.reset_count == 0

        tampered_ledger = new_attempt_ledger(
            Path(directory),
            validated,
            "tampered-ledger",
        )
        tampered_ledger.start(validated.assignments[1])
        tampered_ledger.finish(
            validated.assignments[1],
            "technical_failure",
            "raw_done at 23",
        )
        tampered_rows = [
            json.loads(line)
            for line in tampered_ledger.path.read_text().splitlines()
        ]
        tampered_rows[-1]["reason"] = "silently rewritten failure"
        tampered_ledger.path.write_bytes(
            b"".join(C.canonical_bytes(row) for row in tampered_rows)
        )
        error = expect_raises(
            C.ContractError,
            lambda: C.read_attempt_ledger(
                tampered_ledger.path,
                validated.file_sha256,
            ),
        )
        assert "hash/identity mismatch" in str(error)

        typed_ledger = new_attempt_ledger(
            Path(directory),
            validated,
            "typed-ledger",
        )
        typed_ledger.start(validated.assignments[2])
        typed_rows = [
            json.loads(line) for line in typed_ledger.path.read_text().splitlines()
        ]
        typed_rows[-1]["sequence"] = False
        typed_rows[-1]["row_sha256"] = C._attempt_row_sha256(typed_rows[-1])
        typed_ledger.path.write_bytes(
            b"".join(C.canonical_bytes(row) for row in typed_rows)
        )
        error = expect_raises(
            C.ContractError,
            lambda: C.read_attempt_ledger(
                typed_ledger.path,
                validated.file_sha256,
            ),
        )
        assert "hash/identity mismatch" in str(error)

        mutable = FakeController(validated, mutate_runtime_at=3)
        with mock.patch.object(C.torch.cuda, "is_available", return_value=False):
            error = expect_raises(
                C.PolicyAssignmentError,
                lambda: C.collect_fixed_horizon_episode_contract_test_only(
                    validated,
                    assignment.episode_id,
                    mutable,
                    FakeEnv(validated),
                    FakeFeatureEncoder(validated),
                    new_attempt_ledger(Path(directory), validated, "mutable"),
                ),
            )
        assert mutable.collector_id == "A"
        assert "actor identity mismatch after chunk 3 proposal" in str(error)

        wrong_artifact = FakeController(validated, artifact_sha256="e" * 64)
        error = expect_raises(
            C.PolicyAssignmentError,
            lambda: C.collect_fixed_horizon_episode_contract_test_only(
                validated,
                assignment.episode_id,
                wrong_artifact,
                FakeEnv(validated),
                FakeFeatureEncoder(validated),
                new_attempt_ledger(Path(directory), validated, "wrong-artifact"),
            ),
        )
        assert "identity mismatch before reset" in str(error)


def test_episode_contract_runtime_and_chunk_noise_fail_before_steps() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fixture = build_fixture(root / "valid")
        validated = C.validate_registration(fixture.registration)

        forged = replace(
            validated,
            assignments=(
                replace(validated.assignments[0], env_seed=999999),
                *validated.assignments[1:],
            ),
            action_change=replace(
                validated.action_change,
                min_changed_action_fraction_per_episode=1e-12,
                min_decoded_action_delta_l2_per_episode=1e-12,
            ),
        )
        env = FakeEnv(validated)
        episode = C.collect_fixed_horizon_episode_contract_test_only
        error = expect_raises(
            C.ContractError,
            lambda: episode(
                forged,
                0,
                FakeController(validated),
                env,
                FakeFeatureEncoder(validated),
                new_attempt_ledger(root, validated, "forged"),
            ),
        )
        assert "forged/drifted: assignments" in str(error)
        assert env.reset_count == 0

        weak_object = replace(
            validated,
            action_change=replace(
                validated.action_change,
                min_changed_action_fraction_per_episode=1e-12,
                min_decoded_action_delta_l2_per_episode=1e-12,
            ),
        )
        weak_object_env = FakeEnv(validated)
        error = expect_raises(
            C.ContractError,
            lambda: episode(
                weak_object,
                0,
                FakeController(validated),
                weak_object_env,
                FakeFeatureEncoder(validated),
                new_attempt_ledger(root, validated, "weak-object"),
            ),
        )
        assert "forged/drifted: action_change" in str(error)
        assert weak_object_env.reset_count == 0

        invalid_id_env = FakeEnv(validated)
        error = expect_raises(
            C.ContractError,
            lambda: episode(
                validated,
                -1,
                FakeController(validated),
                invalid_id_env,
                FakeFeatureEncoder(validated),
                new_attempt_ledger(root, validated, "invalid-id"),
            ),
        )
        assert "not a registered D1 episode" in str(error)
        assert invalid_id_env.reset_count == 0

        wrong_actor_env = FakeEnv(validated)
        wrong_actor = FakeController(validated, collector_id="B")
        error = expect_raises(
            C.PolicyAssignmentError,
            lambda: episode(
                validated,
                0,
                wrong_actor,
                wrong_actor_env,
                FakeFeatureEncoder(validated),
                new_attempt_ledger(root, validated, "wrong-actor"),
            ),
        )
        assert "identity mismatch before reset" in str(error)
        assert wrong_actor.reset_count == 0 and wrong_actor_env.reset_count == 0

        wrong_base_env = FakeEnv(validated)
        wrong_base = FakeController(validated)
        wrong_base.base_vla_checkpoint["state_dict"]["action_head.bias"][0] = 1.0
        error = expect_raises(
            C.PolicyAssignmentError,
            lambda: episode(
                validated,
                0,
                wrong_base,
                wrong_base_env,
                FakeFeatureEncoder(validated),
                new_attempt_ledger(root, validated, "wrong-base"),
            ),
        )
        assert "identity mismatch before reset" in str(error)
        assert wrong_base_env.reset_count == 0

        wrong_decoder_env = FakeEnv(validated)
        wrong_decoder = FakeController(validated)
        wrong_decoder.action_decoder_artifact_sha256 = "d" * 64
        error = expect_raises(
            C.PolicyAssignmentError,
            lambda: episode(
                validated,
                0,
                wrong_decoder,
                wrong_decoder_env,
                FakeFeatureEncoder(validated),
                new_attempt_ledger(root, validated, "wrong-decoder"),
            ),
        )
        assert "identity mismatch before reset" in str(error)
        assert wrong_decoder_env.reset_count == 0

        wrong_feature_env = FakeEnv(validated)
        wrong_feature = FakeFeatureEncoder(validated)
        wrong_feature.runtime_config["output_dim"] = 1
        error = expect_raises(
            C.ContractError,
            lambda: episode(
                validated,
                0,
                FakeController(validated),
                wrong_feature_env,
                wrong_feature,
                new_attempt_ledger(root, validated, "wrong-feature"),
            ),
        )
        assert "runtime identity mismatch before reset" in str(error)
        assert wrong_feature_env.reset_count == 0

        wrong_env = FakeEnv(validated)
        wrong_env.runtime_config["horizon_steps"] = 1
        error = expect_raises(
            C.ContractError,
            lambda: episode(
                validated,
                0,
                FakeController(validated),
                wrong_env,
                FakeFeatureEncoder(validated),
                new_attempt_ledger(root, validated, "wrong-env"),
            ),
        )
        assert "runtime identity mismatch before reset" in str(error)
        assert wrong_env.reset_count == 0

        bare_clock_env = FakeEnv(validated)
        error = expect_raises(
            C.ContractError,
            lambda: episode(
                validated,
                0,
                FakeController(validated),
                bare_clock_env,
                clock_feature,
                new_attempt_ledger(root, validated, "bare-clock"),
            ),
        )
        assert "runtime protocol missing before reset" in str(error)
        assert bare_clock_env.reset_count == 0

        structured_clock_env = FakeEnv(validated)
        error = expect_raises(
            C.ContractError,
            lambda: episode(
                validated,
                0,
                FakeController(validated),
                structured_clock_env,
                ClockFeatureEncoder(validated),
                new_attempt_ledger(root, validated, "structured-clock"),
            ),
        )
        assert "constant/clock camera block" in str(error)
        assert structured_clock_env.step_count == 0

        noisy_env = FakeEnv(validated)
        error = expect_raises(
            C.ContractError,
            lambda: episode(
                validated,
                0,
                FakeController(validated, chunk_noise=True),
                noisy_env,
                FakeFeatureEncoder(validated),
                new_attempt_ledger(root, validated, "chunk-noise"),
            ),
        )
        assert "chunk-noise/deterministic replay violation" in str(error)
        assert noisy_env.step_count == 0

        pair_cached_env = FakeEnv(validated)
        error = expect_raises(
            C.ContractError,
            lambda: episode(
                validated,
                0,
                FakeController(validated, pair_cached_entropy=True),
                pair_cached_env,
                FakeFeatureEncoder(validated),
                new_attempt_ledger(root, validated, "pair-cached-entropy"),
            ),
        )
        assert "entropy source use is forbidden" in str(error)
        assert pair_cached_env.step_count == 0

        global_rng_env = FakeEnv(validated)
        error = expect_raises(
            C.ContractError,
            lambda: episode(
                validated,
                0,
                FakeController(validated, global_rng_noise=True),
                global_rng_env,
                FakeFeatureEncoder(validated),
                new_attempt_ledger(root, validated, "global-rng-noise"),
            ),
        )
        assert "global RNG state changed" in str(error)
        assert global_rng_env.step_count == 0

        weak_fixture = build_fixture(root / "weak-threshold")
        payload = json.loads(weak_fixture.registration.read_text())
        payload["collection"]["min_changed_action_fraction_per_episode"] = 1e-12
        payload["collection"]["min_decoded_action_delta_l2_per_episode"] = 1e-12
        rewrite_registration(weak_fixture.registration, payload)
        error = expect_raises(
            C.ContractError,
            lambda: C.validate_registration(weak_fixture.registration),
        )
        assert "must be finite in" in str(error)

        weak_l2_fixture = build_fixture(root / "weak-l2-threshold")
        payload = json.loads(weak_l2_fixture.registration.read_text())
        payload["collection"]["min_changed_action_fraction_per_episode"] = 0.5
        payload["collection"]["min_decoded_action_delta_l2_per_episode"] = 1e-12
        rewrite_registration(weak_l2_fixture.registration, payload)
        error = expect_raises(
            C.ContractError,
            lambda: C.validate_registration(weak_l2_fixture.registration),
        )
        assert "must be at least" in str(error)

        weak_meaningful = build_fixture(root / "weak-meaningful-threshold")
        payload = json.loads(weak_meaningful.registration.read_text())
        payload["collection"]["meaningful_decoded_step_l2"] = 1e-12
        rewrite_registration(weak_meaningful.registration, payload)
        error = expect_raises(
            C.ContractError,
            lambda: C.validate_registration(weak_meaningful.registration),
        )
        assert "must equal fixed" in str(error)


def test_decoded_action_change_and_decoder_purity_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        validated = C.validate_registration(
            build_fixture(Path(directory)).registration
        )
        assignment = validated.assignments[0]

        cases = (
            (
                FakeController(validated, delta_value=0.0),
                "violates meaningful fixed action",
            ),
            (
                FakeController(validated, delta_value=0.000001),
                "violates meaningful fixed action",
            ),
            (
                FakeController(validated, decode_mode="constant"),
                "violates meaningful fixed action",
            ),
            (
                FakeController(
                    validated,
                    delta_mode="sparse_epsilon",
                    decode_mode="triple",
                ),
                "violates meaningful fixed action",
            ),
            (
                FakeController(validated, delta_value=0.041),
                "normalized residual exceeds registered actor scale",
            ),
            (
                FakeController(validated, delta_value=0.04, decode_mode="times_ten"),
                "violates meaningful fixed action",
            ),
        )
        for case_index, (controller, expected_message) in enumerate(cases):
            case_env = FakeEnv(validated)
            with mock.patch.object(C.torch.cuda, "is_available", return_value=False):
                error = expect_raises(
                    C.ContractError,
                    lambda controller=controller: (
                        C.collect_fixed_horizon_episode_contract_test_only(
                            validated,
                            assignment.episode_id,
                            controller,
                            case_env,
                            FakeFeatureEncoder(validated),
                            new_attempt_ledger(
                                Path(directory),
                                validated,
                                f"action-case-{case_index}",
                            ),
                        )
                    ),
                )
            assert expected_message in str(error)
            assert case_env.step_count == 0

        stateful_env = FakeEnv(validated)
        with mock.patch.object(C.torch.cuda, "is_available", return_value=False):
            error = expect_raises(
                C.ContractError,
                lambda: C.collect_fixed_horizon_episode_contract_test_only(
                    validated,
                    assignment.episode_id,
                    FakeController(validated, decode_mode="stateful"),
                    stateful_env,
                    FakeFeatureEncoder(validated),
                    new_attempt_ledger(Path(directory), validated, "stateful"),
                ),
            )
        assert "decoder is not pure/exact" in str(error)
        assert stateful_env.step_count == 0

        float64_env = FakeEnv(validated)
        with mock.patch.object(C.torch.cuda, "is_available", return_value=False):
            error = expect_raises(
                C.ContractError,
                lambda: C.collect_fixed_horizon_episode_contract_test_only(
                    validated,
                    assignment.episode_id,
                    FakeController(validated, decode_mode="float64"),
                    float64_env,
                    FakeFeatureEncoder(validated),
                    new_attempt_ledger(Path(directory), validated, "float64"),
                ),
            )
        assert "must be exact float32" in str(error)
        assert float64_env.step_count == 0


def test_publish_revalidates_inputs_sources_and_registration_snapshot() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fixture = build_fixture(root / "input-drift")
        validated = C.validate_registration(fixture.registration)
        snapshot = C.registration_snapshot_bytes(validated)
        assert C.hashlib.sha256(snapshot).hexdigest() == validated.file_sha256

        fixture.policy.write_bytes(b"collection-time-policy-drift")
        error = expect_raises(
            C.ContractError,
            lambda: C.publish_collection_contract_test_only(
                validated,
                [],
                completed_attempt_ledger(root, validated, "input-drift-attempts"),
                root / "mechanical-contract-test-must-not-publish",
            ),
        )
        assert "SHA mismatch" in str(error)
        assert not (root / "mechanical-contract-test-must-not-publish").exists()

        registration_drift = build_fixture(root / "registration-drift")
        validated_registration = C.validate_registration(
            registration_drift.registration
        )
        registration_drift.registration.write_bytes(
            registration_drift.registration.read_bytes() + b"\n"
        )
        error = expect_raises(
            C.ContractError,
            lambda: C.registration_snapshot_bytes(validated_registration),
        )
        assert "registration file drifted" in str(error)

        source_drift = build_fixture(root / "source-drift")
        validated_source = C.validate_registration(source_drift.registration)
        real_sha256_file = C.sha256_file
        drift_target = (REPO / "lcwm/chassis.py").resolve()

        def source_hash_drift(path: Path, chunk_size: int = 16 << 20) -> str:
            if Path(path).resolve() == drift_target:
                return "0" * 64
            return real_sha256_file(path, chunk_size)

        with mock.patch.object(C, "sha256_file", side_effect=source_hash_drift):
            error = expect_raises(
                C.ContractError,
                lambda: C.revalidate_registration_unchanged(validated_source),
            )
        assert "registered source drift" in str(error)
        assert C.environment_modules_loaded() == []

        late_drift = build_fixture(root / "late-drift")
        validated_late = C.validate_registration(late_drift.registration)
        minimal_tape = {
            "schema": C.DATA_TAPE_SCHEMA,
            "z": torch.zeros(1, C.RICH_DIM),
            "u": torch.zeros(1, C.C, C.ACTION_DIM),
            "z_next": torch.zeros(1, C.RICH_DIM),
            "episode": torch.zeros(1, dtype=torch.long),
            "t": torch.zeros(1, dtype=torch.long),
            "sigma": torch.zeros(1),
            "data_role": "fit",
            "task": C.TASK,
            "c": C.C,
            "latent_dim": C.RICH_DIM,
            "proprio_dim": C.PROPRIO_DIM,
            "proprio_keys": list(C.PROPRIO_KEYS),
            "training_feature_fields": list(C.TRAINING_FEATURE_FIELDS),
            "split_bookkeeping_fields": list(C.SPLIT_BOOKKEEPING_FIELDS),
            "legacy_outcome_fields": [],
        }
        minimal_payloads = (
            minimal_tape,
            {**minimal_tape, "data_role": "calibration"},
            {"schema": C.BOUNDARY_SCHEMA},
            {"schema": C.TRACE_SCHEMA},
            {"schema": C.SIDECAR_SCHEMA},
        )
        real_revalidate = C.revalidate_registration_unchanged
        revalidation_calls = 0

        def drift_before_final_seal(
            registration: C.ValidatedRegistration,
        ) -> C.ValidatedRegistration:
            nonlocal revalidation_calls
            revalidation_calls += 1
            if revalidation_calls == 3:
                late_drift.policy.write_bytes(b"drift-after-staging")
            return real_revalidate(registration)

        late_output = root / "mechanical-contract-test-late-drift-output"
        with mock.patch.object(
            C,
            "episode_collection_payload_sha256_by_id",
            return_value=mechanical_fixture_payload_sha256_by_id(
                validated_late
            ),
        ):
            with mock.patch.object(
                C,
                "_concatenate_collection",
                return_value=minimal_payloads,
            ):
                with mock.patch.object(
                    C,
                    "revalidate_registration_unchanged",
                    side_effect=drift_before_final_seal,
                ):
                    error = expect_raises(
                        C.ContractError,
                        lambda: C.publish_collection_contract_test_only(
                            validated_late,
                            [],
                            completed_attempt_ledger(
                                root,
                                validated_late,
                                "late-drift-attempts",
                            ),
                            late_output,
                        ),
                    )
        assert revalidation_calls == 3
        assert "SHA mismatch" in str(error)
        assert not late_output.exists()
        assert not list(
            root.glob(".mechanical-contract-test-late-drift-output.partial-*")
        )

        invalid_aggregate = build_fixture(root / "invalid-precommit-aggregate")
        validated_invalid = C.validate_registration(invalid_aggregate.registration)
        invalid_output = root / "mechanical-contract-test-invalid-precommit-output"
        with mock.patch.object(
            C,
            "episode_collection_payload_sha256_by_id",
            return_value=mechanical_fixture_payload_sha256_by_id(
                validated_invalid
            ),
        ):
            with mock.patch.object(
                C,
                "_concatenate_collection",
                return_value=minimal_payloads,
            ):
                error = expect_raises(
                    C.ContractError,
                    lambda: C.publish_collection_contract_test_only(
                        validated_invalid,
                        [],
                        completed_attempt_ledger(
                            root,
                            validated_invalid,
                            "invalid-precommit-attempts",
                        ),
                        invalid_output,
                    ),
                )
        assert any(
            marker in str(error)
            for marker in ("bookkeeping", "boundary", "field closure")
        )
        assert not invalid_output.exists()
        assert not list(
            root.glob(
                ".mechanical-contract-test-invalid-precommit-output.partial-*"
            )
        )


def test_atomic_staging_requires_valid_complete_and_cleans_failure() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        output = root / "published"

        def good_builder(stage: Path) -> None:
            (stage / "payload.bin").write_bytes(b"payload")
            write_json(
                stage / "COMPLETE.json",
                {
                    "schema": C.COMPLETE_SCHEMA,
                    "status": C.MECHANICAL_COMPLETE_STATUS,
                    "evidence_class": C.MECHANICAL_EVIDENCE_CLASS,
                    "atomic_commit": True,
                },
            )

        events: list[tuple[str, Path]] = []
        real_fsync_directory = C._fsync_directory
        real_replace = C.os.replace

        def recorded_fsync(path: Path) -> None:
            events.append(("fsync", Path(path)))
            real_fsync_directory(path)

        def recorded_replace(source: Path, destination: Path) -> None:
            events.append(("replace", Path(destination)))
            real_replace(source, destination)

        with mock.patch.object(C, "_fsync_directory", side_effect=recorded_fsync):
            with mock.patch.object(C.os, "replace", side_effect=recorded_replace):
                C.atomic_commit_directory(output, good_builder)
        assert output.is_dir()
        assert (output / "payload.bin").read_bytes() == b"payload"
        complete = json.loads((output / "COMPLETE.json").read_text())
        assert complete["atomic_commit"] is True
        final_replace_index = events.index(("replace", output))
        assert events[final_replace_index - 1][0] == "fsync"
        assert events[final_replace_index + 1] == ("fsync", output.parent)

        failed_output = root / "failed"

        def failing_builder(stage: Path) -> None:
            (stage / "partial.bin").write_bytes(b"partial")
            raise RuntimeError("injected failure")

        expect_raises(
            RuntimeError,
            lambda: C.atomic_commit_directory(failed_output, failing_builder),
        )
        assert not failed_output.exists()
        assert not list(root.glob(".failed.partial-*"))

        invalid_output = root / "invalid"

        def invalid_builder(stage: Path) -> None:
            write_json(stage / "COMPLETE.json", {"status": "not-complete"})

        expect_raises(
            C.ContractError,
            lambda: C.atomic_commit_directory(invalid_output, invalid_builder),
        )
        assert not invalid_output.exists()


def test_shallow_complete_is_not_a_published_gate1_artifact() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        validated = C.validate_registration(
            build_fixture(root / "fixture").registration
        )
        shallow = root / "mechanical-contract-test-shallow"
        (shallow / "training").mkdir(parents=True)
        (shallow / "calibration").mkdir()
        (shallow / "audit").mkdir()
        (shallow / "evaluation").mkdir()
        for relative in (
            "training/fit_tape.pt",
            "calibration/calibration_tape.pt",
            "audit/boundaries.pt",
            "audit/action_trace.pt",
            "audit/attempt_ledger.jsonl",
            "evaluation/evaluation_sidecar.json",
            "ACCESS_POLICY.json",
            "PRODUCER.py",
            "REGISTRATION.json",
            "RESULT.json",
        ):
            (shallow / relative).write_bytes(b"")
        write_json(
            shallow / "COMPLETE.json",
            {
                "schema": C.COMPLETE_SCHEMA,
                "status": C.MECHANICAL_COMPLETE_STATUS,
                "evidence_class": C.MECHANICAL_EVIDENCE_CLASS,
                "formal_gate1_evidence": False,
                "atomic_commit": True,
            },
        )
        error = expect_raises(
            C.ContractError,
            lambda: C.validate_published_collection_contract_test_only(
                shallow,
                validated,
            ),
        )
        assert "shallow seal rejected" in str(error)


def test_safe_snapshots_and_append_only_ledger_reject_rewrites_and_symlinks() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fixture = build_fixture(root / "fixture")
        C.validate_registration(fixture.registration)

        registration_link = root / "registration-link.json"
        registration_link.symlink_to(fixture.registration)
        error = expect_raises(
            C.ContractError,
            lambda: C.validate_registration(registration_link),
        )
        assert "symbolic link" in str(error)

        actor_target = root / "actor-a-target.pt"
        os.replace(fixture.actor_a, actor_target)
        fixture.actor_a.symlink_to(actor_target)
        error = expect_raises(
            C.ContractError,
            lambda: C.validate_registration(fixture.registration),
        )
        assert "symbolic link" in str(error)

        rewrite_fixture = build_fixture(root / "rewrite-fixture")
        rewrite_registration = C.validate_registration(
            rewrite_fixture.registration
        )
        ledger = new_attempt_ledger(root, rewrite_registration, "rewrite-ledger")
        ledger.start(rewrite_registration.assignments[0])
        metadata = ledger.path.stat()
        ledger.path.write_bytes(ledger.path.read_bytes())
        os.utime(
            ledger.path,
            ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 1_000_000),
        )
        error = expect_raises(C.ContractError, ledger.snapshot)
        assert "changed outside append-only writer" in str(error)

        replacement = new_attempt_ledger(
            root,
            rewrite_registration,
            "replacement-ledger",
        )
        replacement_bytes = replacement.path.read_bytes()
        replacement_path = root / "replacement-bytes.jsonl"
        replacement_path.write_bytes(replacement_bytes)
        os.replace(replacement_path, replacement.path)
        error = expect_raises(C.ContractError, replacement.snapshot)
        assert "changed outside append-only writer" in str(error)


def test_full96_mechanical_publish_is_role_isolated_sealed_and_no_overwrite() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with (
            mock.patch.object(C, "CAMERA_BLOCK_DIM", 2),
            mock.patch.object(C, "PROPRIO_DIM", 1),
            mock.patch.object(C, "PROPRIO_KEYS", ("mechanical.proprio",)),
            mock.patch.object(C, "RICH_DIM", 9),
        ):
            fixture = build_fixture(root / "fixture")
            registration = C.validate_registration(fixture.registration)
            encoder = FakeFeatureEncoder(registration)
            episodes = [
                synthetic_mechanical_episode(registration, assignment, encoder)
                for assignment in registration.assignments
            ]
            assert len(episodes) == C.EXPECTED_EPISODES
            assert all(
                tuple(episode.boundaries.shape)
                == (C.BOUNDARIES_PER_EPISODE, C.RICH_DIM)
                for episode in episodes
            )
            ledger = C.AttemptLedger.create_preregistered_path_contract_test_only(
                registration
            )
            for episode in episodes:
                ledger.start(episode.assignment)
                ledger.finish(
                    episode.assignment,
                    "completed",
                    episode_payload_sha256=C.episode_collection_payload_sha256(
                        registration,
                        episode,
                    ),
                )

            output = root / "mechanical-contract-test-full96"
            C.publish_collection_contract_test_only(
                registration,
                episodes,
                ledger,
                output,
            )
            result = C.validate_published_collection_contract_test_only(
                output,
                registration,
            )
            assert result["status"] == C.MECHANICAL_RESULT_STATUS
            assert result["formal_gate1_evidence"] is False
            assert (output / "training" / "fit_tape.pt").is_file()
            assert (
                output / "calibration" / "calibration_tape.pt"
            ).is_file()
            assert not (output / "training" / "rich_tape.pt").exists()

            fit_tape = C.load_torch(
                output / "training" / "fit_tape.pt",
                "test fit tape",
            )
            calibration_tape = C.load_torch(
                output / "calibration" / "calibration_tape.pt",
                "test calibration tape",
            )
            training_view = C.training_tensor_view(fit_tape)
            assert set(training_view) == set(C.TRAINING_FEATURE_FIELDS)
            assert len(training_view["z"]) == 72 * C.CHUNKS_PER_EPISODE
            assert len(calibration_tape["z"]) == 24 * C.CHUNKS_PER_EPISODE
            expect_raises(
                C.ContractError,
                lambda: C.training_tensor_view(calibration_tape),
            )
            expect_raises(
                C.ContractError,
                lambda: C.training_tensor_view(fit_tape, ("episode",)),
            )

            error = expect_raises(
                FileExistsError,
                lambda: C.publish_collection_contract_test_only(
                    registration,
                    episodes,
                    ledger,
                    output,
                ),
            )
            assert str(output) in str(error)

            calibration_path = output / "calibration" / "calibration_tape.pt"
            original_calibration = calibration_path.read_bytes()
            calibration_path.write_bytes(original_calibration + b"mutation")
            error = expect_raises(
                C.ContractError,
                lambda: C.validate_published_collection_contract_test_only(
                    output,
                    registration,
                ),
            )
            assert "SHA mismatch" in str(error)
            calibration_path.write_bytes(original_calibration)

            calibration_target = root / "calibration-target.pt"
            os.replace(calibration_path, calibration_target)
            calibration_path.symlink_to(calibration_target)
            error = expect_raises(
                C.ContractError,
                lambda: C.validate_published_collection_contract_test_only(
                    output,
                    registration,
                ),
            )
            assert "symlink" in str(error)
            assert C.environment_modules_loaded() == []


def test_real_collect_cli_remains_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = build_fixture(Path(directory) / "fixture")
        validated = C.validate_registration(fixture.registration)
        output = Path(directory) / "must-not-exist"
        assert not validated.attempt_ledger_path.exists()
        error = expect_raises(
            C.RealCollectorNotImplemented,
            lambda: C.AttemptLedger.create_registered(validated),
        )
        assert "blocked before reset/write" in str(error)
        assert not validated.attempt_ledger_path.exists()

        formal_env = FakeEnv(validated)
        untouched_ledger = new_attempt_ledger(
            Path(directory),
            validated,
            "formal-entry-untouched",
        )
        ledger_before = untouched_ledger.path.read_bytes()
        error = expect_raises(
            C.RealCollectorNotImplemented,
            lambda: C.collect_fixed_horizon_episode(
                validated,
                0,
                FakeController(validated),
                formal_env,
                FakeFeatureEncoder(validated),
                untouched_ledger,
            ),
        )
        assert "blocked before reset/write" in str(error)
        assert formal_env.reset_count == 0 and formal_env.step_count == 0
        assert untouched_ledger.path.read_bytes() == ledger_before

        error = expect_raises(
            C.RealCollectorNotImplemented,
            lambda: C.publish_collection(
                validated,
                [],
                new_attempt_ledger(
                    Path(directory),
                    validated,
                    "mechanical-cannot-publish",
                ),
                output,
            ),
        )
        assert "formal D1 publication" in str(error)
        error = expect_raises(
            C.RealCollectorNotImplemented,
            lambda: C.validate_published_collection(output, validated),
        )
        assert "published-artifact validation" in str(error)
        argv = [
            "collect_v246_evolve1_d1.py",
            "--registration",
            str(fixture.registration),
            "--collect",
            "--out",
            str(output),
        ]
        with mock.patch.object(sys, "argv", argv):
            error = expect_raises(C.RealCollectorNotImplemented, C.main)
        assert "NO-GO" in str(error)
        assert not output.exists()
        assert C.environment_modules_loaded() == []


def main() -> None:
    tests = (
        test_registration_validates_before_environment_import,
        test_hash_actor_tape_seed_and_source_fail_before_environment_import,
        test_exact_field_closures_and_structured_artifacts_fail_closed,
        test_success_at_20_still_runs_750_and_derives_75_76,
        test_raw_done_before_750_halts_and_policy_assignment_is_immutable,
        test_episode_contract_runtime_and_chunk_noise_fail_before_steps,
        test_decoded_action_change_and_decoder_purity_fail_closed,
        test_publish_revalidates_inputs_sources_and_registration_snapshot,
        test_atomic_staging_requires_valid_complete_and_cleans_failure,
        test_shallow_complete_is_not_a_published_gate1_artifact,
        test_safe_snapshots_and_append_only_ledger_reject_rewrites_and_symlinks,
        test_full96_mechanical_publish_is_role_isolated_sealed_and_no_overwrite,
        test_real_collect_cli_remains_fail_closed,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)} v246 D1 contract tests; real environment steps = 0")


if __name__ == "__main__":
    main()
