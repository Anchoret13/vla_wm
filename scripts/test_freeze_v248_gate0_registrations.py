#!/usr/bin/env python3
# RETIRED 2026-09-16. Nothing supersedes it: the gate it serves was removed as a
# prerequisite at plan_and_progress/2026-09-12.md:105, and the round used a supplied
# Phi' annotation instead. It has produced zero artifacts - none of
# results/v248_gate0_{registration_bundle,materialized_models,formal_campaign}_v1 exists.
# Blockers never cleared: WORM ledger, runtime fingerprint, registration bundle, and
# non-deterministic cuDNN/TF32 (plan_and_progress/2026-09-02.md:108-129).
# Do not extend, do not import, do not cite as capability. Recoverable at d35e2832.
"""CPU-only fake-artifact tests for the v248 registration freezer."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lcwm import v248_gate0_contract as C  # noqa: E402


def _load_freezer():
    path = REPO / "scripts/freeze_v248_gate0_registrations.py"
    spec = importlib.util.spec_from_file_location("v248_freezer_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


F = _load_freezer()


def expect_error(function: Callable[[], Any], contains: str | None = None) -> Exception:
    try:
        function()
    except Exception as error:  # noqa: BLE001 - explicit negative-path test
        if contains is not None:
            assert contains in str(error), (contains, str(error))
        return error
    raise AssertionError("expected failure")


def write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def make_cache(
    root: Path, revision: str, members: list[str], *, symlink_snapshot: bool
) -> Path:
    write(root / "refs/main", revision.encode())
    snapshot = root / "snapshots" / revision
    for index, relative in enumerate(members):
        if symlink_snapshot:
            blob = write(root / "blobs" / f"blob-{index}", f"{relative}\n".encode())
            link = snapshot / relative
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(blob)
        else:
            write(snapshot / relative, f"{relative}\n".encode())
    return root


def make_clip_source(root: Path) -> Path:
    for relative in F.CLIP_TOKENIZER_MEMBER_PATHS:
        source = F.DEFAULT_CLIP_SOURCE / relative
        assert source.is_file(), source
        destination = root / relative
        write(destination, source.read_bytes())
    return root


def make_sanitized(root: Path, seed_start: int) -> tuple[Path, Path]:
    source_root = root / "opaque_source"
    source_tape = write(source_root / "tape.pt", f"source-{seed_start}".encode())
    summary = write(
        source_root / "summary.json",
        b'{"success": THIS_IS_INTENTIONALLY_NOT_VALID_JSON',
    )
    artifact = root / "sanitized"
    tape = write(artifact / "sanitized_tape.pt", f"sanitized-{seed_start}".encode())
    producer = write(
        artifact / "PRODUCER.py",
        (REPO / "scripts/sanitize_v245_deploy_tape.py").read_bytes(),
    )
    counts = [75] * 96
    if seed_start == 8100:
        counts[77] = 65
    metadata = {
        "task": C.TASK,
        "c": C.C,
        "latent_dim": C.LATENT_DIM,
        "proprio_dim": C.PROPRIO_DIM,
        "proprio_keys": list(C.PROPRIO_KEYS),
    }
    result = {
        "schema": F.SANITIZER_RESULT_SCHEMA,
        "status": "PASS",
        "utc": "2026-09-01T00:00:00Z",
        "source": {
            "path": source_tape.resolve().as_posix(),
            "sha256": C.sha256_file(source_tape),
            "bytes": source_tape.stat().st_size,
            "summary_opened": False,
        },
        "field_policy": {
            "kept": sorted(F.SANITIZED_TAPE_FIELDS),
            "dropped_names_without_value_access": ["success"],
            "declared_outcome_fields": ["success"],
        },
        "validation": {
            "rows": sum(counts),
            "latent_dim": C.LATENT_DIM,
            "episodes": 96,
            "episode_ids": list(range(96)),
            "episode_row_counts": counts,
            "c": C.C,
        },
        "metadata": metadata,
        "tensor_sha256": {
            name: chr(97 + index) * 64
            for index, name in enumerate(sorted(F.SEQUENCE_TENSOR_FIELDS))
        },
        "metadata_sha256": C.canonical_sha256(metadata),
        "sanitized_tape_sha256": C.sha256_file(tape),
        "producer_snapshot_sha256": C.sha256_file(producer),
        "git_head": None,
        "torch_version": "fixture",
    }
    result_path = write(artifact / "RESULT.json", C.canonical_bytes(result))
    complete = {
        "schema": F.SANITIZER_COMPLETE_SCHEMA,
        "status": "atomic_success",
        "atomic_commit": True,
        "source_tape_sha256": C.sha256_file(source_tape),
        "sanitized_tape_sha256": C.sha256_file(tape),
        "result_sha256": C.sha256_file(result_path),
        "producer_snapshot_sha256": C.sha256_file(producer),
    }
    write(artifact / "COMPLETE.json", C.canonical_bytes(complete))
    return artifact, summary


def fixture(root: Path) -> F.FreezeOptions:
    sanitized_8000, oracle_8000 = make_sanitized(root / "seed8000", 8000)
    sanitized_8100, oracle_8100 = make_sanitized(root / "seed8100", 8100)
    pi_cache = make_cache(
        root / "pi05_cache",
        "a" * 40,
        list(C.PI05_MEMBER_PATHS.values()),
        symlink_snapshot=True,
    )
    pali_cache = make_cache(
        root / "paligemma_cache",
        "b" * 40,
        sorted(C.TOKENIZER_MEMBER_PATHS),
        symlink_snapshot=True,
    )
    clip_source = make_clip_source(root / "clip_source")
    checkpoint = write(root / "sam3.safetensors", b"fake-sam3")
    cream_image = write(root / "cream.jpg", b"fake-template-image")
    cream_metadata = write(root / "cream.json", b"fake-template-metadata")
    semantic_config = write(
        root / "semantic_config.json",
        (REPO / "plan_and_progress/v248_gate0_semantic_config.json").read_bytes(),
    )
    return F.FreezeOptions(
        output_root=root / "freeze_bundle",
        materialized_root=root / "materialized_models",
        target_root=root / "formal_targets",
        sanitized_roots={8000: sanitized_8000, 8100: sanitized_8100},
        oracle_paths={8000: oracle_8000, 8100: oracle_8100},
        pi05_cache_root=pi_cache,
        paligemma_cache_root=pali_cache,
        clip_source_root=clip_source,
        clip_revision="c" * 40,
        sam3_checkpoint=checkpoint,
        sam3_model_id="fixture/sam3",
        sam3_revision="d" * 40,
        cream_template_image=cream_image,
        cream_template_metadata=cream_metadata,
        semantic_config=semantic_config,
        semantic_producer=REPO / "scripts/produce_v248_gate0_progress.py",
        semantic_evaluator=REPO / "scripts/eval_v248_gate0_progress.py",
        runtime_source_overrides={},
        created_utc="2026-09-01T00:00:00Z",
    )


def assert_hardlinks(plan: F.FreezePlan) -> None:
    roots = F._materialized_roots(plan.options.materialized_root, plan)
    pairs = (
        (plan.pi05.snapshot_root, roots["pi05"], plan.pi05.members),
        (plan.paligemma.snapshot_root, roots["paligemma"], plan.paligemma.members),
        (plan.options.clip_source_root, roots["clip"], F.CLIP_TOKENIZER_MEMBER_PATHS),
    )
    for source_root, destination_root, members in pairs:
        for relative in members:
            source = (source_root / relative).resolve()
            destination = destination_root / relative
            assert destination.is_file() and not destination.is_symlink()
            assert (source.stat().st_dev, source.stat().st_ino) == (
                destination.stat().st_dev,
                destination.stat().st_ino,
            )


def test_plan_and_two_phase_no_clobber_freeze() -> None:
    with tempfile.TemporaryDirectory(prefix="v248-freezer-test-") as raw:
        root = Path(raw)
        options = fixture(root)
        plan = F.collect_plan(options)
        assert not options.output_root.exists()
        assert not options.materialized_root.exists()
        assert plan.tokenization_proof["exact"] is True
        assert plan.tokenization_proof["prompts"] == list(F.SAM3_PROMPTS)
        expect_error(
            lambda: F.freeze_bundle(plan, "0" * 64),
            "reviewed validate-only digest",
        )
        assert not options.output_root.exists()
        assert not options.materialized_root.exists()

        output = F.freeze_bundle(plan, plan.input_closure_sha256)
        assert output == options.output_root
        assert_hardlinks(plan)
        materialization = json.loads(
            (options.materialized_root / "MATERIALIZATION_COMPLETE.json").read_text()
        )
        assert materialization["rgb_authorized"] is False
        assert materialization["status"] == "materialized_only_not_rgb_authorization"
        registrations = sorted((output / "registrations").glob("*.json"))
        assert len(registrations) == 4
        campaign = json.loads((output / "CAMPAIGN.json").read_text())
        campaign_body = {
            key: value for key, value in campaign.items() if key != "body_sha256"
        }
        assert campaign["body_sha256"] == C.canonical_sha256(campaign_body)
        assert set(campaign["registrations"]) == {path.name for path in registrations}
        projection = campaign["projection"]
        intent = projection["intent"]
        assert projection["campaign_id"] == campaign["campaign_id"]
        assert campaign["campaign_id"] == (
            f"v248-gate0-{C.canonical_sha256(intent)[:16]}"
        )
        assert intent["input_closure_sha256"] == plan.input_closure_sha256
        assert intent["target_root"] == options.target_root.resolve().as_posix()
        assert intent["target_output_slots"] == F.TARGET_OUTPUT_SLOTS
        assert intent["dag"] == F.CAMPAIGN_DAG
        loaded = {}
        for path in registrations:
            registration, _sha = C.load_and_validate_registration(
                path, expected_source_role="formal_replay"
            )
            loaded[(registration["panel"]["seed_start"], registration["phase"])] = registration
            assert registration["registration_id"].endswith("_v2")
        assert [
            row["episode_id"] for row in loaded[(8000, "screen")]["panel"]["episodes"]
        ] == list(C.SCREEN_EPISODES[8000])
        assert [
            row["episode_id"] for row in loaded[(8100, "screen")]["panel"]["episodes"]
        ] == list(C.SCREEN_EPISODES[8100])
        assert len(loaded[(8000, "confirm")]["panel"]["episodes"]) == 96
        for registration in loaded.values():
            authorizer = registration["sources"]["confirm_authorizer"]
            assert authorizer["relative_path"] == "scripts/authorize_v248_gate0_confirm.py"
            assert authorizer["sha256"] == C.sha256_file(
                REPO / authorizer["relative_path"]
            )
        recovery = loaded[(8100, "confirm")]["terminal_recovery"]
        assert recovery["env_seed"] == 8177
        assert recovery["stored_chunks"] == 65
        assert recovery["source_terminal_t"] == 651
        assert recovery["missing_raw_sigma_choice"] == 0.6
        freeze_complete_sha = C.sha256_file(output / "COMPLETE.json")
        authority = C.load_and_validate_campaign_bundle(
            output,
            freeze_complete_sha,
            expected_source_role="formal_replay",
        )
        assert authority["freeze_complete_sha256"] == freeze_complete_sha
        assert set(authority["registrations"]) == {path.name for path in registrations}
        assert authority["target_root"] == options.target_root.resolve()
        expect_error(
            lambda: C.load_and_validate_campaign_bundle(
                output,
                "0" * 64,
                expected_source_role="formal_replay",
            ),
            "externally anchored freeze COMPLETE",
        )
        expect_error(
            lambda: F.freeze_bundle(plan, plan.input_closure_sha256),
            "already exists",
        )
        pi05_snapshot = F._materialized_roots(options.materialized_root, plan)["pi05"]
        (pi05_snapshot / "unexpected-empty-directory").mkdir()
        expect_error(
            lambda: F._validate_materialized_stage(options.materialized_root, plan),
            "snapshot inventory drift",
        )


def test_stage_a_is_not_authorization_when_stage_b_input_drifts() -> None:
    with tempfile.TemporaryDirectory(prefix="v248-freezer-drift-test-") as raw:
        root = Path(raw)
        options = fixture(root)
        plan = F.collect_plan(options)
        original_config = options.semantic_config.read_bytes()
        options.semantic_config.write_bytes(b"drift-after-reviewed-plan\n")
        expect_error(
            lambda: F.freeze_bundle(plan, plan.input_closure_sha256),
            "semantic_config changed",
        )
        assert options.materialized_root.is_dir()
        assert not options.output_root.exists()
        complete = json.loads(
            (options.materialized_root / "MATERIALIZATION_COMPLETE.json").read_text()
        )
        assert complete["rgb_authorized"] is False
        materialization_anchor = C.sha256_file(
            options.materialized_root / "MATERIALIZATION_COMPLETE.json"
        )
        options.semantic_config.write_bytes(original_config)
        retry_plan = F.collect_plan(
            options,
            expected_materialization_complete_sha256=materialization_anchor,
        )
        assert retry_plan.input_closure_sha256 == plan.input_closure_sha256
        output = F.freeze_bundle(
            retry_plan,
            retry_plan.input_closure_sha256,
            expected_materialization_complete_sha256=materialization_anchor,
        )
        assert output == options.output_root


def _prospective_registrations(
    plan: F.FreezePlan,
) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
    roots = F._materialized_roots(plan.options.materialized_root, plan)
    models = F._models(plan, roots["pi05"], roots["paligemma"], roots["clip"])
    return F.make_registrations(plan, models, plan.options.created_utc)


def test_campaign_identity_binds_data_and_all_four_prescreen_slots() -> None:
    with tempfile.TemporaryDirectory(prefix="v248-freezer-campaign-test-") as raw:
        root = Path(raw)
        options = fixture(root)
        first_plan = F.collect_plan(options)
        first_registrations, first_projection = _prospective_registrations(first_plan)
        assert set(first_registrations) == {
            (8000, "screen"),
            (8100, "screen"),
            (8000, "confirm"),
            (8100, "confirm"),
        }

        Path(options.oracle_paths[8000]).write_bytes(b"different opaque oracle bytes")
        second_plan = F.collect_plan(options)
        second_registrations, second_projection = _prospective_registrations(second_plan)
        first_intent = first_projection["intent"]
        second_intent = second_projection["intent"]
        assert first_intent["protocol_sha256"] == second_intent["protocol_sha256"]
        assert (
            first_intent["registration_core_sha256s"]
            != second_intent["registration_core_sha256s"]
        )
        assert first_projection["campaign_id"] != second_projection["campaign_id"]
        assert first_plan.input_closure_sha256 != second_plan.input_closure_sha256
        assert set(second_registrations) == set(first_registrations)


def test_existing_target_or_overlapping_target_fails_before_materialization() -> None:
    with tempfile.TemporaryDirectory(prefix="v248-freezer-target-test-") as raw:
        root = Path(raw)
        options = fixture(root)
        screen_complete = (
            options.target_root / F.TARGET_OUTPUT_SLOTS["screen_progress_complete"]
        )
        write(screen_complete, b"fake screen output")
        expect_error(lambda: F.collect_plan(options), "post-screen freeze")
        assert not options.output_root.exists()
        assert not options.materialized_root.exists()

    with tempfile.TemporaryDirectory(prefix="v248-freezer-overlap-test-") as raw:
        root = Path(raw)
        options = fixture(root)
        overlapping = replace(options, target_root=options.output_root / "targets")
        expect_error(lambda: F.collect_plan(overlapping), "overlaps protected")
        assert not options.output_root.exists()
        assert not options.materialized_root.exists()


def _cli_args(options: F.FreezeOptions) -> list[str]:
    return [
        "--validate-only",
        "--output-root",
        str(options.output_root),
        "--materialized-root",
        str(options.materialized_root),
        "--target-root",
        str(options.target_root),
        "--sanitized-8000",
        str(options.sanitized_roots[8000]),
        "--sanitized-8100",
        str(options.sanitized_roots[8100]),
        "--oracle-8000",
        str(options.oracle_paths[8000]),
        "--oracle-8100",
        str(options.oracle_paths[8100]),
        "--pi05-cache-root",
        str(options.pi05_cache_root),
        "--paligemma-cache-root",
        str(options.paligemma_cache_root),
        "--clip-tokenizer-source",
        str(options.clip_source_root),
        "--clip-tokenizer-revision",
        str(options.clip_revision),
        "--sam3-checkpoint",
        str(options.sam3_checkpoint),
        "--sam3-model-id",
        options.sam3_model_id,
        "--sam3-revision",
        str(options.sam3_revision),
        "--cream-template-image",
        str(options.cream_template_image),
        "--cream-template-metadata",
        str(options.cream_template_metadata),
        "--semantic-config",
        str(options.semantic_config),
        "--semantic-producer",
        str(options.semantic_producer),
        "--semantic-evaluator",
        str(options.semantic_evaluator),
        "--created-utc",
        str(options.created_utc),
    ]


def test_validate_only_subprocess_has_no_simulator_import_or_output() -> None:
    with tempfile.TemporaryDirectory(prefix="v248-freezer-cli-test-") as raw:
        root = Path(raw)
        options = fixture(root)
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
            raise ImportError('forbidden freezer import: ' + fullname)
        return None
sys.meta_path.insert(0, Blocker())
"""
        write(blocker / "sitecustomize.py", sitecustomize.encode())
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join([str(blocker), str(REPO)])
        process = subprocess.run(
            [
                sys.executable,
                str(REPO / "scripts/freeze_v248_gate0_registrations.py"),
                *_cli_args(options),
            ],
            cwd=REPO,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert process.returncode == 0, (process.stdout, process.stderr)
        payload = json.loads(process.stdout)
        assert payload["status"] == "VALID"
        assert payload["output_created"] is False
        assert payload["simulator_modules_imported"] is False
        assert payload["oracle_json_decoded"] is False
        assert not marker.exists()
        assert not options.output_root.exists()
        assert not options.materialized_root.exists()


def main() -> int:
    tests = (
        test_plan_and_two_phase_no_clobber_freeze,
        test_stage_a_is_not_authorization_when_stage_b_input_drifts,
        test_campaign_identity_binds_data_and_all_four_prescreen_slots,
        test_existing_target_or_overlapping_target_fails_before_materialization,
        test_validate_only_subprocess_has_no_simulator_import_or_output,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS all {len(tests)} v248 registration freezer CPU tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
