#!/usr/bin/env python3
# RETIRED 2026-09-16. Nothing supersedes it: the gate it serves was removed as a
# prerequisite at plan_and_progress/2026-09-12.md:105, and the round used a supplied
# Phi' annotation instead. It has produced zero artifacts - none of
# results/v248_gate0_{registration_bundle,materialized_models,formal_campaign}_v1 exists.
# Blockers never cleared: WORM ledger, runtime fingerprint, registration bundle, and
# non-deterministic cuDNN/TF32 (plan_and_progress/2026-09-02.md:108-129).
# Do not extend, do not import, do not cite as capability. Recoverable at d35e2832.
"""CPU-only synthetic tests for the v248 formal visual Gate-0 contract.

No simulator, model, formal registration, or 8000/8100/8500 image is opened.
All artifact bytes constructed here live under pytest temporary directories.
"""
from __future__ import annotations

import copy
import inspect
import json
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import pytest


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import eval_v248_gate0_progress as evaluator  # noqa: E402
import produce_v248_gate0_progress as producer  # noqa: E402
from lcwm.v248_gate0_contract import (  # noqa: E402
    Gate0ContractError,
    PANEL_IDS,
    SCREEN_EPISODES,
    TARGET_OUTPUT_SLOTS,
    canonical_bytes,
    sha256_file,
)


DIGEST = "1" * 64
DIGEST_2 = "2" * 64
TRANSFORMER_ROLES = (
    "transformers_sam3_video_configuration",
    "transformers_sam3_video_modeling",
    "transformers_sam3_video_processing",
    "transformers_sam3_image_processing",
    "transformers_sam2_video_processing",
    "transformers_clip_tokenization",
)


def synthetic_campaign_bundle(root: Path) -> dict[str, Any]:
    target_root = root / "formal-target"
    registrations = {
        f"{PANEL_IDS[seed]}_{phase}.json": {
            "path": root / "freeze" / "registrations" / f"{PANEL_IDS[seed]}_{phase}.json",
            "file_sha256": DIGEST,
            "registration": {"phase": phase},
        }
        for phase in ("screen", "confirm")
        for seed in (8000, 8100)
    }
    return {
        "root": root / "freeze",
        "freeze_complete_sha256": DIGEST,
        "campaign_manifest_sha256": DIGEST,
        "campaign": {
            "body_sha256": DIGEST,
            "projection_sha256": DIGEST,
            "campaign_id": "v248-gate0-1111111111111111",
        },
        "registrations": registrations,
        "target_root": target_root,
        "target_output_slots": copy.deepcopy(TARGET_OUTPUT_SLOTS),
    }


def install_target_io_trap(
    monkeypatch: pytest.MonkeyPatch, protected_roots: list[Path]
) -> list[str]:
    """Record and reject OS metadata/open calls below synthetic target roots."""
    protected = [Path(os.path.abspath(os.fspath(path))) for path in protected_roots]
    calls: list[str] = []

    def is_protected(raw: Any) -> bool:
        try:
            candidate = Path(os.path.abspath(os.fspath(raw)))
        except TypeError:
            return False
        return any(
            candidate == root or candidate.is_relative_to(root) for root in protected
        )

    for name in ("open", "stat", "lstat", "scandir"):
        original = getattr(os, name)

        def trapped(
            raw: Any,
            *args: Any,
            _name: str = name,
            _original: Any = original,
            **kwargs: Any,
        ) -> Any:
            if is_protected(raw):
                calls.append(f"{_name}:{raw}")
                raise AssertionError(f"target I/O before authorization: {_name}:{raw}")
            return _original(raw, *args, **kwargs)

        monkeypatch.setattr(os, name, trapped)
    return calls


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(canonical_bytes(dict(value)))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_bytes(b"".join(canonical_bytes(row) for row in rows))


def episode_specs(seed_start: int) -> list[dict[str, Any]]:
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
        for episode_id in SCREEN_EPISODES[seed_start]
    ]


def registrations() -> tuple[dict[int, dict[str, Any]], dict[int, str]]:
    producer_sha = sha256_file(REPO / "scripts" / "produce_v248_gate0_progress.py")
    evaluator_sha = sha256_file(REPO / "scripts" / "eval_v248_gate0_progress.py")
    output = {}
    hashes = {8000: "8" * 64, 8100: "9" * 64}
    for seed_start in (8000, 8100):
        output[seed_start] = {
            "registration_id": f"fixture-{seed_start}",
            "self_sha256": ("a" if seed_start == 8000 else "b") * 64,
            "phase": "screen",
            "campaign": {
                "campaign_id": "v248-gate0-1111111111111111",
                "projection_sha256": DIGEST,
            },
            "panel": {
                "panel_id": PANEL_IDS[seed_start],
                "seed_start": seed_start,
                "episodes": episode_specs(seed_start),
            },
            "sources": {
                "formal_replay": {"sha256": DIGEST},
                "semantic_producer": {"sha256": producer_sha},
                "semantic_evaluator": {"sha256": evaluator_sha},
                "sam3_engine": {"sha256": DIGEST},
                "sift_engine": {"sha256": DIGEST},
                "can_state": {"sha256": DIGEST},
                "cream_state": {"sha256": DIGEST},
            },
            "models": {
                "sam3": {
                    "checkpoint": {"sha256": DIGEST},
                    "clip_tokenizer": {"tree_sha256": DIGEST},
                    "config_sha256": DIGEST,
                    "processor_sha256": DIGEST,
                }
            },
            "templates": {
                "cream_template_image": {"sha256": DIGEST},
                "cream_template_metadata": {"sha256": DIGEST},
                "semantic_config": {
                    "sha256": producer.EXPECTED_CONFIG_SHA256
                },
            },
            "runtime": {
                "runtime_sources": {
                    role: {"sha256": DIGEST} for role in TRANSFORMER_ROLES
                }
            },
        }
    return output, hashes


def make_frame(
    *,
    seed_start: int,
    episode_id: int,
    rgb_index: int,
    local_index: int,
    t: int,
    role: str,
    phase: str = "screen",
) -> dict[str, Any]:
    recovered = role == "recovered_endpoint"
    return {
        "schema": producer.FRAME_SCHEMA,
        "panel_id": PANEL_IDS[seed_start],
        "phase": phase,
        "task": "chain3_lr2",
        "episode_id": episode_id,
        "env_seed": seed_start + episode_id,
        "rgb_frame_index": rgb_index,
        "episode_frame_index": local_index,
        "t": t,
        "frame_id": (
            f"{PANEL_IDS[seed_start]}:{phase}:ep{episode_id:02d}:t{t:04d}"
        ),
        "boundary_role": role,
        "observation_hashes": {
            "agentview_sha256": DIGEST,
            "eye_in_hand_sha256": DIGEST_2,
        },
        "atoms": {"can_ge1": False, "can_ge2": False, "cream": False},
        "scalar": 0,
        "reliability": {
            "can_ge1": True,
            "can_ge2": True,
            "cream": True,
            "scalar": True,
        },
        "transitions": {"can_ge1": False, "can_ge2": False, "cream": False},
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


def all_screen_frames(regs: Mapping[int, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for seed_start in (8000, 8100):
        rgb_index = 0
        for spec in regs[seed_start]["panel"]["episodes"]:
            times, roles = producer._expected_episode_schedule(spec)
            for local_index, (t, role) in enumerate(zip(times, roles, strict=True)):
                rows.append(
                    make_frame(
                        seed_start=seed_start,
                        episode_id=spec["episode_id"],
                        rgb_index=rgb_index,
                        local_index=local_index,
                        t=t,
                        role=role,
                    )
                )
                rgb_index += 1
    return rows


def engine_record(regs: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    registration = regs[8000]
    return {
        "sam3_engine_sha256": DIGEST,
        "sift_engine_sha256": DIGEST,
        "can_state_sha256": DIGEST,
        "cream_state_sha256": DIGEST,
        "sam3_checkpoint_sha256": DIGEST,
        "clip_tokenizer_tree_sha256": DIGEST,
        "sam3_config_sha256": DIGEST,
        "sam3_processor_sha256": DIGEST,
        "cream_template_image_sha256": DIGEST,
        "cream_template_metadata_sha256": DIGEST,
        "semantic_config_sha256": producer.EXPECTED_CONFIG_SHA256,
        "transformers_source_sha256s": {
            role: registration["runtime"]["runtime_sources"][role]["sha256"]
            for role in TRANSFORMER_ROLES
        },
    }


def create_progress_fixture(
    root: Path,
) -> tuple[dict[int, dict[str, Any]], dict[int, str], list[dict[str, Any]]]:
    root.mkdir()
    regs, hashes = registrations()
    frames = all_screen_frames(regs)
    shutil.copyfile(
        REPO / "scripts" / "produce_v248_gate0_progress.py",
        root / producer.PRODUCER_FILENAME,
    )
    shutil.copyfile(
        REPO / "plan_and_progress" / "v248_gate0_semantic_config.json",
        root / producer.CONFIG_FILENAME,
    )
    write_jsonl(root / producer.FRAMES_FILENAME, frames)
    prefix = {
        f"{PANEL_IDS[seed]}:ep{episode_id:02d}": {
            "frames": 8,
            "reference_sha256": DIGEST,
            "fresh_session_sha256": DIGEST,
            "exact_after_rounding_6_decimals": True,
        }
        for seed in (8000, 8100)
        for episode_id in SCREEN_EPISODES[seed]
    }
    result = {
        "schema": producer.RESULT_SCHEMA,
        "status": "observation_only_sealed",
        "created_utc": "2026-09-01T00:00:00Z",
        "claim_scope": "visual_progress_producer_only",
        "phase": "screen",
        "task": "chain3_lr2",
        "authority": {
            "freeze_complete_sha256": DIGEST,
            "campaign_manifest_sha256": DIGEST,
            "campaign_body_sha256": DIGEST,
            "campaign_projection_sha256": DIGEST,
            "campaign_id": "v248-gate0-1111111111111111",
            "registration_slots": [
                "chain3_8000_screen.json",
                "chain3_8100_screen.json",
            ],
            "output_complete_slot": "screen/progress/COMPLETE.json",
            "confirm_authorization_complete_sha256": None,
        },
        "registrations": {
            PANEL_IDS[seed]: {
                "registration_id": regs[seed]["registration_id"],
                "file_sha256": hashes[seed],
                "canonical_self_sha256": regs[seed]["self_sha256"],
            }
            for seed in (8000, 8100)
        },
        "rgb_inputs": {
            PANEL_IDS[seed]: {
                "result_sha256": DIGEST,
                "manifest_sha256": DIGEST,
                "complete_sha256": DIGEST,
                "producer_sha256": DIGEST,
                "image_tree_sha256": DIGEST,
                "frames": evaluator._expected_frames_for_registration(regs[seed]),
            }
            for seed in (8000, 8100)
        },
        "episode_ids": {
            PANEL_IDS[seed]: list(SCREEN_EPISODES[seed]) for seed in (8000, 8100)
        },
        "observation_contract": {
            "decoded_cameras": ["agentview", "eye_in_hand"],
            "semantic_inputs": ["current_agentview_rgb", "current_eye_in_hand_rgb"],
            "causal_past_state_only": True,
        },
        "engines": engine_record(regs),
        "prefix_checks": prefix,
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
            "frames_sha256": sha256_file(root / producer.FRAMES_FILENAME),
            "producer_sha256": sha256_file(root / producer.PRODUCER_FILENAME),
            "semantic_config_sha256": sha256_file(root / producer.CONFIG_FILENAME),
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
    write_json(root / producer.RESULT_FILENAME, result)
    complete = {
        "schema": producer.COMPLETE_SCHEMA,
        "status": "atomic_success",
        "atomic_commit": True,
        "phase": "screen",
        "authority": result["authority"],
        "registration_sha256s": {
            PANEL_IDS[seed]: hashes[seed] for seed in (8000, 8100)
        },
        "result_sha256": sha256_file(root / producer.RESULT_FILENAME),
        "frames_sha256": sha256_file(root / producer.FRAMES_FILENAME),
        "producer_sha256": sha256_file(root / producer.PRODUCER_FILENAME),
        "semantic_config_sha256": sha256_file(root / producer.CONFIG_FILENAME),
    }
    write_json(root / producer.COMPLETE_FILENAME, complete)
    return regs, hashes, frames


def reseal_frames(root: Path, frames: list[dict[str, Any]]) -> None:
    write_jsonl(root / producer.FRAMES_FILENAME, frames)
    result = json.loads((root / producer.RESULT_FILENAME).read_text())
    result["output"]["frames"] = len(frames)
    result["output"]["frames_sha256"] = sha256_file(
        root / producer.FRAMES_FILENAME
    )
    write_json(root / producer.RESULT_FILENAME, result)
    complete = json.loads((root / producer.COMPLETE_FILENAME).read_text())
    complete["frames_sha256"] = sha256_file(root / producer.FRAMES_FILENAME)
    complete["result_sha256"] = sha256_file(root / producer.RESULT_FILENAME)
    write_json(root / producer.COMPLETE_FILENAME, complete)


def validate_fixture(
    root: Path,
    regs: Mapping[int, Mapping[str, Any]],
    hashes: Mapping[int, str],
    expected_complete_sha256: str | None = None,
) -> Any:
    complete_path = root / producer.COMPLETE_FILENAME
    return evaluator.validate_progress_artifact_without_oracle(
        root,
        regs,
        hashes,
        expected_complete_sha256
        or (sha256_file(complete_path) if complete_path.is_file() else DIGEST),
    )


def test_complete_synthetic_progress_artifact_validates(tmp_path: Path) -> None:
    root = tmp_path / "progress"
    regs, hashes, frames = create_progress_fixture(root)
    result, loaded, _config, seals = validate_fixture(root, regs, hashes)
    assert len(loaded) == len(frames)
    assert result["phase"] == "screen"
    assert seals["producer_sha256"] == regs[8000]["sources"]["semantic_producer"][
        "sha256"
    ]


def test_incomplete_and_tampered_artifacts_fail_closed(tmp_path: Path) -> None:
    incomplete = tmp_path / "incomplete"
    regs, hashes, _ = create_progress_fixture(incomplete)
    (incomplete / producer.COMPLETE_FILENAME).unlink()
    with pytest.raises(Gate0ContractError):
        validate_fixture(incomplete, regs, hashes)

    tampered = tmp_path / "tampered"
    regs, hashes, _ = create_progress_fixture(tampered)
    with (tampered / producer.FRAMES_FILENAME).open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(Gate0ContractError):
        validate_fixture(tampered, regs, hashes)

    extra = tmp_path / "extra"
    regs, hashes, _ = create_progress_fixture(extra)
    (extra / "UNREGISTERED.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(Gate0ContractError, match="closure"):
        validate_fixture(extra, regs, hashes)


def test_resealed_leakage_field_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "leakage"
    regs, hashes, frames = create_progress_fixture(root)
    frames[0]["success"] = False
    reseal_frames(root, frames)
    with pytest.raises(Gate0ContractError):
        validate_fixture(root, regs, hashes)


def test_external_complete_anchor_rejects_fully_resealed_semantics(
    tmp_path: Path,
) -> None:
    root = tmp_path / "externally_anchored"
    regs, hashes, frames = create_progress_fixture(root)
    external_anchor = sha256_file(root / producer.COMPLETE_FILENAME)
    frames[0]["evidence"]["sift_affine_scale"] = 1.25
    reseal_frames(root, frames)
    with pytest.raises(Gate0ContractError, match="external.*COMPLETE|SHA-256"):
        validate_fixture(root, regs, hashes, external_anchor)


def test_atoms_must_equal_causal_state_and_rgb_hashes_must_match_manifest() -> None:
    frame = make_frame(
        seed_start=8000,
        episode_id=0,
        rgb_index=0,
        local_index=0,
        t=0,
        role="pre_action",
    )
    inconsistent = copy.deepcopy(frame)
    inconsistent["atoms"]["can_ge1"] = True
    inconsistent["scalar"] = 1
    inconsistent["transitions"]["can_ge1"] = True
    with pytest.raises(Gate0ContractError, match="causal sticky count"):
        producer.validate_progress_frame(inconsistent)

    manifest = {
        "frame_id": frame["frame_id"],
        "panel_id": frame["panel_id"],
        "phase": frame["phase"],
        "task": frame["task"],
        "episode_id": frame["episode_id"],
        "env_seed": frame["env_seed"],
        "frame_index": frame["rgb_frame_index"],
        "t": frame["t"],
        "source": {"boundary_role": frame["boundary_role"]},
        "images": {
            "agentview": {"sha256": DIGEST},
            "eye_in_hand": {"sha256": DIGEST_2},
        },
    }
    tampered_hash = copy.deepcopy(frame)
    tampered_hash["observation_hashes"]["agentview_sha256"] = "0" * 64
    with pytest.raises(Gate0ContractError, match="sealed RGB pair"):
        producer._validate_frame_against_rgb_manifest(
            tampered_hash,
            manifest,
            local_index=0,
            previous_atoms={"can_ge1": False, "can_ge2": False, "cream": False},
        )


def test_progress_and_evaluator_snapshots_need_prereg_root_links(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source_links"
    regs, hashes, _frames = create_progress_fixture(root)
    bad_progress = copy.deepcopy(regs)
    for seed in (8000, 8100):
        bad_progress[seed]["sources"]["semantic_producer"]["sha256"] = "0" * 64
    with pytest.raises(Gate0ContractError, match="producer prereg"):
        validate_fixture(root, bad_progress, hashes)

    bad_evaluator = copy.deepcopy(regs)
    for seed in (8000, 8100):
        bad_evaluator[seed]["sources"]["semantic_evaluator"]["sha256"] = "0" * 64
    with pytest.raises(
        Gate0ContractError, match="evaluator prereg|semantic evaluator SHA-256 drift"
    ):
        evaluator.validate_live_evaluator_source(
            bad_evaluator, REPO / "scripts" / "eval_v248_gate0_progress.py"
        )


def test_missing_recovered_ep77_is_rejected_even_when_resealed(tmp_path: Path) -> None:
    root = tmp_path / "missing_ep77"
    regs, hashes, frames = create_progress_fixture(root)
    frames = [
        frame
        for frame in frames
        if not (
            frame["panel_id"] == "chain3_8100"
            and frame["episode_id"] == 77
            and frame["t"] == 651
        )
    ]
    reseal_frames(root, frames)
    with pytest.raises(Gate0ContractError):
        validate_fixture(root, regs, hashes)


def metric_frame(
    *,
    t: int,
    can: bool,
    transition: bool,
    carry: bool,
) -> dict[str, Any]:
    return {
        "panel_id": "chain3_8000",
        "episode_id": 0,
        "t": t,
        "atoms": {"can_ge1": can, "can_ge2": False, "cream": False},
        "scalar": int(can),
        "reliability": {
            "can_ge1": True,
            "can_ge2": True,
            "cream": True,
            "scalar": True,
        },
        "transitions": {
            "can_ge1": transition,
            "can_ge2": False,
            "cream": False,
        },
        "causal_state": {
            "can_used_carry": carry,
            "cream_used_carry": False,
        },
    }


def test_extra_and_carry_created_transitions_are_counted() -> None:
    frames = [
        metric_frame(t=0, can=False, transition=False, carry=False),
        metric_frame(t=10, can=True, transition=True, carry=True),
    ]
    records = {
        ("chain3_8000", 0): {"env_seed": 8000, "steps": 20, "events": {}}
    }
    metrics = evaluator.compute_metrics(frames, records, expected_frames=2)
    assert metrics["transitions"]["extra_transitions"] == 1
    assert metrics["carry_audit"]["carry_created_transitions"] == 1


def perfect_metrics() -> dict[str, Any]:
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


def test_screen_can_never_promote_and_prefix_failure_stops() -> None:
    config = producer.validate_semantic_config(
        json.loads(
            (REPO / "plan_and_progress" / "v248_gate0_semantic_config.json").read_text()
        )
    )
    screen = evaluator.visual_subgate_decision(
        metrics=perfect_metrics(),
        config=config,
        phase="screen",
        primary_episode_count=24,
        all_prefix_exact=True,
        recovery_passed=True,
    )
    assert screen["metric_checks_passed"] is True
    assert screen["promotion_authorized"] is False
    assert screen["decision"] == "STOP"

    prefix_stop = evaluator.visual_subgate_decision(
        metrics=perfect_metrics(),
        config=config,
        phase="confirm",
        primary_episode_count=192,
        all_prefix_exact=False,
        recovery_passed=True,
    )
    assert prefix_stop["checks"]["all_prefix_checks_exact"] is False
    assert prefix_stop["decision"] == "STOP"

    abstaining = perfect_metrics()
    abstaining["coverage"]["evidence_coverage"] = 0.96
    abstaining["coverage"]["uncertainty_rate"] = 0.04
    abstaining["classification"]["exact_scalar_accuracy_certain_only"] = 1.0
    abstaining["classification"][
        "effective_scalar_accuracy_uncertain_incorrect"
    ] = 0.94
    abstention_stop = evaluator.visual_subgate_decision(
        metrics=abstaining,
        config=config,
        phase="confirm",
        primary_episode_count=192,
        all_prefix_exact=True,
        recovery_passed=True,
    )
    assert abstention_stop["checks"]["minimum_exact_scalar_accuracy"] is False
    assert abstention_stop["decision"] == "STOP"


def oracle_payload(order: list[int]) -> dict[str, Any]:
    return {
        "episode_records": [
            {
                "idx": episode_id,
                "seed": 8000 + episode_id,
                "steps": 750,
                "events": {},
                "success": bool(episode_id % 2),
            }
            for episode_id in order
        ]
    }


def oracle_registration(path: Path) -> dict[str, Any]:
    return {
        "panel": {"seed_start": 8000},
        "oracle": {
            "summary": {
                "path": path.resolve().as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        },
    }


def test_oracle_summary_order_is_exact_and_success_is_not_used(tmp_path: Path) -> None:
    good = tmp_path / "good.json"
    write_json(good, oracle_payload(list(range(96))))
    records, provenance = evaluator.load_oracle_records(oracle_registration(good))
    assert list(records) == list(range(96))
    assert provenance["success_values_accessed"] is False

    bad = tmp_path / "bad.json"
    order = list(range(96))
    order[0], order[1] = order[1], order[0]
    write_json(bad, oracle_payload(order))
    with pytest.raises(Gate0ContractError, match="order"):
        evaluator.load_oracle_records(oracle_registration(bad))

    missing = tmp_path / "missing.json"
    payload = oracle_payload(list(range(96)))
    del payload["episode_records"][0]["events"]
    write_json(missing, payload)
    with pytest.raises(Gate0ContractError, match="missing a required field"):
        evaluator.load_oracle_records(oracle_registration(missing))


def test_can_oracle_is_identity_free_under_reverse_order() -> None:
    assert evaluator.ordinal_truth_times({3: 10, 1: 30, 5: 50}) == {
        "can_ge1": 10,
        "can_ge2": 30,
        "cream": 50,
    }
    assert evaluator.ordinal_truth_times({3: 10}) == {
        "can_ge1": 10,
        "can_ge2": None,
        "cream": None,
    }


def test_evaluator_cli_exposes_only_campaign_anchored_two_stage_flow() -> None:
    args = evaluator.parse_args(
        [
            "seal",
            "--freeze-bundle",
            "/frozen/bundle",
            "--expected-freeze-complete-sha256",
            DIGEST,
            "--progress-run",
            "/campaign/screen/progress",
            "--expected-progress-complete-sha256",
            DIGEST_2,
        ]
    )
    assert args.command == "seal"
    assert args.expected_confirm_authorization_complete_sha256 is None
    assert not hasattr(args, "registration")
    assert not hasattr(args, "output_dir")
    producer_args = producer.parse_args(
        [
            "--freeze-bundle",
            "/frozen/bundle",
            "--expected-freeze-complete-sha256",
            DIGEST,
            "--rgb-run",
            "/campaign/screen/rgb/chain3_8000",
            "--rgb-run",
            "/campaign/screen/rgb/chain3_8100",
            "--expected-rgb-complete-sha256",
            DIGEST,
            "--expected-rgb-complete-sha256",
            DIGEST_2,
        ]
    )
    assert not hasattr(producer_args, "output_dir")
    with pytest.raises(SystemExit):
        evaluator.parse_args(
            [
                "--registration",
                "/attacker/registration.json",
                "--output-dir",
                "/attacker/output",
            ]
        )


def test_confirm_producer_validates_full_authorization_before_rgb_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert "rgb_support" not in inspect.signature(producer.prepare_inputs).parameters
    assert "rgb_support" not in producer.PreparedInputs.__dataclass_fields__
    bundle = synthetic_campaign_bundle(tmp_path)
    target_root = Path(bundle["target_root"])
    roots = [
        (
            target_root
            / bundle["target_output_slots"]["confirm_rgb_complete_by_panel"][
                PANEL_IDS[seed]
            ]
        ).parent
        for seed in (8000, 8100)
    ]
    events: list[str] = []

    class RejectingReplay:
        def _validate_confirm_authorization(
            self, _bundle: Mapping[str, Any], _sha256: str
        ) -> None:
            events.append("full_authorization")
            raise Gate0ContractError("authorization artifact root closure mismatch")

    def forbidden_rgb_validation(*_args: Any, **_kwargs: Any) -> None:
        events.append("rgb_io")
        raise AssertionError("confirm RGB was touched before authorization")

    def load_registered_replay(registration: Mapping[str, Any]) -> RejectingReplay:
        events.append("load_registered_replay")
        assert registration is bundle["registrations"][
            "chain3_8000_confirm.json"
        ]["registration"]
        return RejectingReplay()

    monkeypatch.setattr(
        producer, "load_and_validate_campaign_bundle", lambda *_args, **_kwargs: bundle
    )
    monkeypatch.setattr(producer, "_load_rgb_support", load_registered_replay)
    monkeypatch.setattr(producer, "_validate_rgb_panel", forbidden_rgb_validation)
    target_io_calls = install_target_io_trap(monkeypatch, roots)
    with pytest.raises(Gate0ContractError, match="authorization artifact root closure"):
        producer.prepare_inputs(
            freeze_bundle_root=tmp_path / "freeze",
            expected_freeze_complete_sha256=DIGEST,
            rgb_roots=roots,
            expected_rgb_complete_sha256s=[DIGEST, DIGEST_2],
            expected_confirm_authorization_complete_sha256=DIGEST,
        )
    assert events == ["load_registered_replay", "full_authorization"]
    assert target_io_calls == []

    events.clear()
    with pytest.raises(Gate0ContractError, match="requires external confirm authorization"):
        producer.prepare_inputs(
            freeze_bundle_root=tmp_path / "freeze",
            expected_freeze_complete_sha256=DIGEST,
            rgb_roots=roots,
            expected_rgb_complete_sha256s=[DIGEST, DIGEST_2],
        )
    assert events == []
    assert target_io_calls == []


def test_screen_producer_forbids_authorization_but_does_not_require_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = synthetic_campaign_bundle(tmp_path)
    target_root = Path(bundle["target_root"])
    roots = [
        (
            target_root
            / bundle["target_output_slots"]["screen_rgb_complete_by_panel"][
                PANEL_IDS[seed]
            ]
        ).parent
        for seed in (8000, 8100)
    ]
    authorization_calls = 0

    class ScreenReplay:
        def _validate_confirm_authorization(self, *_args: Any) -> None:
            nonlocal authorization_calls
            authorization_calls += 1
            raise AssertionError("screen must not validate confirm authorization")

    class ReachedScreenRgb(RuntimeError):
        pass

    monkeypatch.setattr(
        producer, "load_and_validate_campaign_bundle", lambda *_args, **_kwargs: bundle
    )
    monkeypatch.setattr(producer, "_load_rgb_support", lambda *_args: ScreenReplay())
    monkeypatch.setattr(
        producer,
        "_validate_rgb_panel",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ReachedScreenRgb()),
    )
    with pytest.raises(ReachedScreenRgb):
        producer.prepare_inputs(
            freeze_bundle_root=tmp_path / "freeze",
            expected_freeze_complete_sha256=DIGEST,
            rgb_roots=roots,
            expected_rgb_complete_sha256s=[DIGEST, DIGEST_2],
        )
    assert authorization_calls == 0

    with pytest.raises(Gate0ContractError, match="screen progress must not accept"):
        producer.prepare_inputs(
            freeze_bundle_root=tmp_path / "freeze",
            expected_freeze_complete_sha256=DIGEST,
            rgb_roots=roots,
            expected_rgb_complete_sha256s=[DIGEST, DIGEST_2],
            expected_confirm_authorization_complete_sha256=DIGEST,
        )
    assert authorization_calls == 0


def test_confirm_evaluator_validates_authorization_before_progress_io_and_binds_a(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = synthetic_campaign_bundle(tmp_path)
    progress_root = (
        Path(bundle["target_root"])
        / bundle["target_output_slots"]["confirm_progress_complete"]
    ).parent
    events: list[str] = []

    class Replay:
        def _validate_confirm_authorization(
            self, _bundle: Mapping[str, Any], sha256: str
        ) -> dict[str, Any]:
            events.append("full_authorization")
            return {
                "root": (
                    Path(bundle["target_root"])
                    / bundle["target_output_slots"][
                        "confirm_authorization_complete"
                    ]
                ).parent,
                "complete_sha256": sha256,
            }

    locator = canonical_bytes(
        {
            "phase": "confirm",
            "authority": {
                "confirm_authorization_complete_sha256": DIGEST_2,
            },
        }
    )

    def progress_read(*_args: Any, **_kwargs: Any) -> bytes:
        events.append("progress_io")
        return locator

    monkeypatch.setattr(
        producer, "load_and_validate_campaign_bundle", lambda *_args, **_kwargs: bundle
    )
    monkeypatch.setattr(producer, "_load_rgb_support", lambda *_args: Replay())
    monkeypatch.setattr(producer, "_read_sealed_regular_bytes", progress_read)
    with pytest.raises(Gate0ContractError, match="differs from the external anchor"):
        evaluator.load_anchored_campaign_progress(
            freeze_bundle_root=tmp_path / "freeze",
            expected_freeze_complete_sha256=DIGEST,
            progress_root=progress_root,
            expected_progress_complete_sha256=DIGEST,
            expected_confirm_authorization_complete_sha256=DIGEST,
        )
    assert events == ["full_authorization", "progress_io"]

    events.clear()
    with pytest.raises(Gate0ContractError, match="requires an external"):
        evaluator.load_anchored_campaign_progress(
            freeze_bundle_root=tmp_path / "freeze",
            expected_freeze_complete_sha256=DIGEST,
            progress_root=progress_root,
            expected_progress_complete_sha256=DIGEST,
        )
    assert events == []

    class RejectingReplay:
        def _validate_confirm_authorization(
            self, _bundle: Mapping[str, Any], _sha256: str
        ) -> None:
            events.append("full_authorization_rejected")
            raise Gate0ContractError("authorization child closure mismatch")

    monkeypatch.setattr(producer, "_load_rgb_support", lambda *_args: RejectingReplay())
    target_io_calls = install_target_io_trap(monkeypatch, [progress_root])
    with pytest.raises(Gate0ContractError, match="authorization child closure"):
        evaluator.load_anchored_campaign_progress(
            freeze_bundle_root=tmp_path / "freeze",
            expected_freeze_complete_sha256=DIGEST,
            progress_root=progress_root,
            expected_progress_complete_sha256=DIGEST,
            expected_confirm_authorization_complete_sha256=DIGEST,
        )
    assert events == ["full_authorization_rejected"]
    assert target_io_calls == []


def test_output_cannot_be_nested_in_or_contain_a_sealed_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    with pytest.raises(Gate0ContractError, match="nested in input"):
        producer._check_output_separation(sealed / "nested", [sealed])
    future_parent = tmp_path / "future_parent"
    with pytest.raises(Gate0ContractError, match="would contain input"):
        producer._check_output_separation(
            future_parent, [future_parent / "sealed_input"]
        )

    bundle = synthetic_campaign_bundle(tmp_path)
    phase = "screen"
    config_path = tmp_path / "registered-semantic-config.json"
    config = {"schema": producer.CONFIG_SCHEMA}
    config_bytes = b"registered semantic config\n"
    shared_campaign = {
        "schema": "v248_gate0_campaign_ref_v1",
        "campaign_id": bundle["campaign"]["campaign_id"],
        "protocol_sha256": DIGEST,
        "projection_sha256": bundle["campaign"]["projection_sha256"],
    }
    registrations_by_seed: dict[int, dict[str, Any]] = {}
    registration_paths: dict[int, Path] = {}
    panels: dict[int, producer.RgbPanel] = {}
    for seed in (8000, 8100):
        panel_id = PANEL_IDS[seed]
        slot = f"{panel_id}_{phase}.json"
        registration = {
            "phase": phase,
            "campaign": dict(shared_campaign),
            "models": {"sealed": True},
            "templates": {
                "semantic_config": {
                    "path": config_path.as_posix(),
                    "sha256": producer.EXPECTED_CONFIG_SHA256,
                }
            },
            "sources": {"semantic_producer": {"sha256": DIGEST}},
            "runtime": {"sealed": True},
            "semantic_contract": {"sealed": True},
            "oracle": {
                "summary": {"path": (tmp_path / f"oracle-{seed}.json").as_posix()}
            },
        }
        bundle["registrations"][slot]["registration"] = registration
        registrations_by_seed[seed] = registration
        registration_paths[seed] = Path(bundle["registrations"][slot]["path"])
        rgb_slot = bundle["target_output_slots"][
            f"{phase}_rgb_complete_by_panel"
        ][panel_id]
        panel_root = (Path(bundle["target_root"]) / rgb_slot).parent
        panels[seed] = producer.RgbPanel(
            root=panel_root,
            panel_id=panel_id,
            phase=phase,
            result={},
            complete={},
            rows=(),
            seals={},
        )

    progress_slot = bundle["target_output_slots"][f"{phase}_progress_complete"]
    frozen_output = (Path(bundle["target_root"]) / progress_slot).parent
    authority = {
        "freeze_complete_sha256": bundle["freeze_complete_sha256"],
        "campaign_manifest_sha256": bundle["campaign_manifest_sha256"],
        "campaign_body_sha256": bundle["campaign"]["body_sha256"],
        "campaign_projection_sha256": bundle["campaign"]["projection_sha256"],
        "campaign_id": bundle["campaign"]["campaign_id"],
        "registration_slots": [
            f"{PANEL_IDS[seed]}_{phase}.json" for seed in (8000, 8100)
        ],
        "output_complete_slot": progress_slot,
        "confirm_authorization_complete_sha256": None,
    }
    prepared = producer.PreparedInputs(
        campaign_bundle=bundle,
        authority=authority,
        registrations=registrations_by_seed,
        registration_hashes={8000: DIGEST, 8100: DIGEST},
        registration_paths=registration_paths,
        panels=panels,
        config=config,
        config_path=config_path,
        config_bytes=config_bytes,
        config_sha256=producer.EXPECTED_CONFIG_SHA256,
        expected_rgb_complete_sha256s={8000: DIGEST, 8100: DIGEST_2},
        expected_confirm_authorization_complete_sha256=None,
        output_dir=frozen_output,
    )

    loaded_registrations: list[Mapping[str, Any]] = []
    rgb_validation_supports: list[Any] = []
    publications: list[tuple[Path, Path]] = []

    class FreshReplay:
        def atomic_publish_directory(
            self,
            output: Path,
            _build: Any,
            _validate: Any,
            *,
            trusted_target_root: Path,
        ) -> None:
            publications.append((output, trusted_target_root))

    fresh_replay = FreshReplay()

    def load_fresh_replay(registration: Mapping[str, Any]) -> FreshReplay:
        loaded_registrations.append(registration)
        return fresh_replay

    panel_by_root = {panel.root: panel for panel in panels.values()}

    def validate_rgb(root: Path, *_args: Any) -> producer.RgbPanel:
        rgb_validation_supports.append(_args[-1])
        return panel_by_root[root]

    config_loads: list[Path] = []

    def load_config(path: Path) -> tuple[dict[str, Any], str, bytes]:
        config_loads.append(path)
        return config, producer.EXPECTED_CONFIG_SHA256, config_bytes

    monkeypatch.setattr(
        producer, "load_and_validate_campaign_bundle", lambda *_args, **_kwargs: bundle
    )
    monkeypatch.setattr(producer, "_load_rgb_support", load_fresh_replay)
    monkeypatch.setattr(producer, "_validate_rgb_panel", validate_rgb)
    monkeypatch.setattr(producer, "load_semantic_config", load_config)
    monkeypatch.setattr(producer, "_validate_frame_sequence", lambda *_args: None)
    monkeypatch.setattr(producer, "_rgb_recovery_passed", lambda *_args: True)
    monkeypatch.setattr(
        producer, "_read_sealed_regular_bytes", lambda *_args, **_kwargs: b"source\n"
    )

    with pytest.raises(TypeError):
        replace(prepared, rgb_support=object())

    producer.publish_progress(prepared, [], {})
    assert loaded_registrations == [registrations_by_seed[8000]]
    assert rgb_validation_supports == [fresh_replay, fresh_replay]
    assert config_loads == [config_path]
    assert publications == [(frozen_output, Path(bundle["target_root"]))]

    with monkeypatch.context() as final_reload_patch:
        target_calls = install_target_io_trap(
            final_reload_patch, [Path(bundle["target_root"])]
        )

        def reject_fresh_replay(_registration: Mapping[str, Any]) -> None:
            raise Gate0ContractError("registered replay source SHA drift")

        final_reload_patch.setattr(
            producer, "_load_rgb_support", reject_fresh_replay
        )
        with pytest.raises(Gate0ContractError, match="registered replay source SHA drift"):
            producer.publish_progress(prepared, [], {})
        assert target_calls == []

    attacks = (
        (
            replace(prepared, output_dir=tmp_path / "attacker-output"),
            "prepared output path differs from frozen progress slot",
        ),
        (
            replace(
                prepared,
                registration_paths={
                    **registration_paths,
                    8000: tmp_path / "attacker-registration.json",
                },
            ),
            "prepared registration paths differ from frozen bundle",
        ),
        (
            replace(prepared, config_path=tmp_path / "attacker-config.json"),
            "prepared semantic config path differs from frozen registration",
        ),
    )
    for attacked, message in attacks:
        with pytest.raises(Gate0ContractError, match=message):
            producer.publish_progress(attacked, [], {})
    assert loaded_registrations == [registrations_by_seed[8000]]
    assert rgb_validation_supports == [fresh_replay, fresh_replay]
    assert config_loads == [config_path]
    assert publications == [(frozen_output, Path(bundle["target_root"]))]


def test_pre_oracle_requires_both_external_file_and_complete_anchors(
    tmp_path: Path,
) -> None:
    root = tmp_path / "preoracle"
    root.mkdir()
    evaluator_path = root / evaluator.EVALUATOR_FILENAME
    evaluator_path.write_bytes(b"registered evaluator snapshot\n")
    evaluator_sha = sha256_file(evaluator_path)
    _regs, registration_hashes = registrations()
    authority = {
        "freeze_complete_sha256": DIGEST,
        "campaign_manifest_sha256": DIGEST,
        "campaign_body_sha256": DIGEST,
        "campaign_projection_sha256": DIGEST,
        "campaign_id": "v248-gate0-1111111111111111",
        "registration_slots": [
            "chain3_8000_screen.json",
            "chain3_8100_screen.json",
        ],
        "output_complete_slot": "screen/preoracle/COMPLETE.json",
        "confirm_authorization_complete_sha256": None,
    }
    progress_seals = {
        "complete_sha256": DIGEST,
        "result_sha256": DIGEST,
        "frames_sha256": DIGEST,
        "producer_sha256": DIGEST,
        "semantic_config_sha256": DIGEST,
        "authority": {
            **authority,
            "output_complete_slot": "screen/progress/COMPLETE.json",
        },
    }
    _path, preseal_sha = evaluator.make_pre_oracle_seal(
        stage=root,
        phase="screen",
        registration_hashes=registration_hashes,
        progress_seals=progress_seals,
        evaluator_sha256=evaluator_sha,
        authority=authority,
    )
    write_json(
        root / evaluator.COMPLETE_FILENAME,
        {
            "schema": evaluator.PRE_ORACLE_COMPLETE_SCHEMA,
            "status": "atomic_pre_oracle_seal_complete",
            "atomic_commit": True,
            "pre_oracle_seal_sha256": preseal_sha,
            "evaluator_sha256": evaluator_sha,
            "authority": authority,
        },
    )
    complete_sha = sha256_file(root / evaluator.COMPLETE_FILENAME)
    evaluator.validate_published_pre_oracle(
        root=root,
        expected_pre_oracle_sha256=preseal_sha,
        expected_complete_sha256=complete_sha,
        phase="screen",
        registration_hashes=registration_hashes,
        progress_seals=progress_seals,
        evaluator_sha256=evaluator_sha,
        expected_authority=authority,
    )
    with pytest.raises(Gate0ContractError, match="external SHA-256 anchor|SHA-256 drift"):
        evaluator.validate_published_pre_oracle(
            root=root,
            expected_pre_oracle_sha256="0" * 64,
            expected_complete_sha256=complete_sha,
            phase="screen",
            registration_hashes=registration_hashes,
            progress_seals=progress_seals,
            evaluator_sha256=evaluator_sha,
            expected_authority=authority,
        )


def test_evaluation_stage_rejects_preoracle_result_phase_drift(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "evaluation"
    stage.mkdir()
    regs, registration_hashes = registrations()
    progress_seals = {
        "complete_sha256": DIGEST,
        "result_sha256": DIGEST,
        "frames_sha256": DIGEST,
        "producer_sha256": DIGEST,
        "semantic_config_sha256": DIGEST,
        "authority": {
            "freeze_complete_sha256": DIGEST,
            "campaign_manifest_sha256": DIGEST,
            "campaign_body_sha256": DIGEST,
            "campaign_projection_sha256": DIGEST,
            "campaign_id": "v248-gate0-1111111111111111",
            "registration_slots": [
                "chain3_8000_screen.json",
                "chain3_8100_screen.json",
            ],
            "output_complete_slot": "screen/progress/COMPLETE.json",
            "confirm_authorization_complete_sha256": None,
        },
    }
    preseal = {
        "schema": evaluator.PRE_ORACLE_SCHEMA,
        "status": "sealed_before_oracle_access",
        "created_utc": "2026-09-02T00:00:00Z",
        "oracle_opened_or_hashed": False,
        "phase": "confirm",
        "registration_sha256s": {
            PANEL_IDS[seed]: registration_hashes[seed] for seed in (8000, 8100)
        },
        "progress_artifact": progress_seals,
        "evaluator_sha256": DIGEST,
        "authority": progress_seals["authority"],
    }
    result = {
        "schema": evaluator.EVALUATION_SCHEMA,
        "status": "postseal_visual_evaluation_complete",
        "created_utc": "2026-09-02T00:00:00Z",
        "claim_scope": "gate0_visual_subgate_only",
        "phase": "screen",
        "decision": "STOP",
        "authority": progress_seals["authority"],
        "pre_oracle_seal_sha256": DIGEST,
        "pre_oracle_complete_sha256": DIGEST,
        "registrations": {},
        "progress_artifact": progress_seals,
        "oracle_inputs": {},
        "metrics": {},
        "visual_subgate": {},
        "provenance": {},
    }
    write_json(stage / evaluator.PRE_ORACLE_FILENAME, preseal)
    write_json(stage / evaluator.RESULT_FILENAME, result)
    write_json(stage / evaluator.SUBGATE_FILENAME, {})
    write_json(stage / evaluator.COMPLETE_FILENAME, {})
    (stage / evaluator.EVALUATOR_FILENAME).write_bytes(b"placeholder\n")
    with pytest.raises(Gate0ContractError, match="phase drift"):
        evaluator._validate_evaluation_stage(
            stage,
            regs,
            registration_hashes,
            progress_seals,
        )
