#!/usr/bin/env python3
"""Authorize v248 Gate-0 confirm only from the frozen combined-screen result.

The command accepts one external genesis anchor (the registration-freeze
``COMPLETE.json`` digest) plus the externally recorded screen progress (P),
pre-oracle seal (S), pre-oracle COMPLETE (Q), and evaluation COMPLETE (E)
digests.  Every location, registration, predicate, and output slot is derived
from the frozen CAMPAIGN projection.  It never accepts registration paths,
oracle paths, or an output directory, and it never imports or resets a robot
environment.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lcwm import v248_gate0_contract as C  # noqa: E402


AUTHORIZATION_RESULT_SCHEMA = "v248_gate0_confirm_authorization_result_v1"
AUTHORIZATION_COMPLETE_SCHEMA = "v248_gate0_confirm_authorization_complete_v1"
AUTHORIZATION_PREDICATE_SCHEMA = "v248_gate0_confirm_predicate_v1"
AUTHORIZATION_MARKER_SCHEMA = "v248_gate0_confirm_authorized_marker_v1"

COMBINED_AUTHORITY_FIELDS = frozenset(
    {
        "freeze_complete_sha256",
        "campaign_manifest_sha256",
        "campaign_body_sha256",
        "campaign_projection_sha256",
        "campaign_id",
        "registration_slots",
        "output_complete_slot",
        "confirm_authorization_complete_sha256",
    }
)
AUTHORIZATION_AUTHORITY_FIELDS = frozenset(
    {
        "freeze_complete_sha256",
        "campaign_manifest_sha256",
        "campaign_body_sha256",
        "campaign_projection_sha256",
        "campaign_id",
        "screen_evaluation_complete_sha256",
        "screen_evaluation_complete_slot",
        "authorization_complete_slot",
    }
)
PREDICATE_FIELDS = frozenset(
    {
        "schema",
        "automatic_expansion_rule",
        "required_phase",
        "required_screen_decision",
        "required_screen_promotion_authorized",
        "required_screen_metrics_passed",
        "required_metric_checks_passed",
        "require_every_metric_check_true",
        "required_metric_check_names",
        "screen_episode_ids",
        "required_combined_primary_episodes",
        "authorized_confirm_episode_ids",
    }
)
REGISTRATION_REF_FIELDS = frozenset(
    {"registration_slot", "file_sha256", "canonical_self_sha256"}
)
INPUT_FIELDS = frozenset(
    {
        "screen_rgb_complete_sha256s",
        "screen_progress_complete_sha256",
        "screen_pre_oracle_seal_sha256",
        "screen_pre_oracle_complete_sha256",
        "screen_evaluation_complete_sha256",
        "screen_evaluation_result_sha256",
        "screen_visual_subgate_sha256",
    }
)
REQUIRED_METRIC_CHECK_NAMES = (
    "exact_primary_episode_count",
    "complete_progress_to_rgb_frame_bijection",
    "minimum_evidence_coverage",
    "maximum_uncertainty_rate",
    "minimum_exact_scalar_accuracy",
    "minimum_macro_f1",
    "maximum_missing_transitions",
    "maximum_extra_transitions",
    "maximum_median_timing_error_steps",
    "maximum_p95_timing_error_steps",
    "maximum_timing_error_steps",
    "minimum_never_achieved_terminal_specificity",
    "never_achieved_support_present",
    "minimum_held_cream_specificity",
    "held_cream_support_present",
    "minimum_chain3_cream_positive_transitions",
    "maximum_carry_created_transitions",
    "all_prefix_checks_exact",
    "terminal_recovery_passed",
)
OUTPUT_FIELDS = frozenset(
    {"predicate_sha256", "producer_sha256"}
)
SAFETY_FIELDS = frozenset(
    {
        "oracle_path_cli_supported",
        "oracle_opened_or_hashed",
        "simulator_imported",
        "environment_constructed_or_reset",
        "caller_selected_registration",
        "caller_selected_output",
    }
)
RESULT_FIELDS = frozenset(
    {
        "schema",
        "status",
        "created_utc",
        "decision",
        "claim_scope",
        "authority",
        "predicate",
        "screen_registrations",
        "confirm_registrations",
        "input",
        "output",
        "safety",
    }
)
COMPLETE_FIELDS = frozenset(
    {
        "schema",
        "status",
        "atomic_commit",
        "decision",
        "authority",
        "result_sha256",
        "predicate_sha256",
        "producer_sha256",
        "marker_sha256",
    }
)
MARKER_FIELDS = frozenset(
    {"schema", "decision", "result_sha256", "predicate_sha256"}
)


def _read_regular_bytes(path: Path, expected_sha256: str, where: str) -> bytes:
    return C.read_anchored_regular_bytes(path, expected_sha256, where)


def _strict_json_bytes(payload: bytes, where: str) -> dict[str, Any]:
    try:
        value = C.loads_strict_json(payload.decode("utf-8"), where=where)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise C.Gate0ContractError(f"cannot decode {where}") from error
    C.require(isinstance(value, dict), f"{where} must be an object")
    return dict(value)


def _strict_json_file(path: Path, where: str) -> dict[str, Any]:
    C.require(path.is_file() and not path.is_symlink(), f"{where} missing/symlink")
    return _strict_json_bytes(path.read_bytes(), where)


def _load_registered_module(
    registration: Mapping[str, Any], role: str, module_name: str
) -> ModuleType:
    reference = registration["sources"][role]
    path = (REPO / reference["relative_path"]).resolve()
    C.require(path.is_relative_to(REPO), f"registered {role} escapes repository")
    payload = _read_regular_bytes(path, reference["sha256"], f"registered {role}")
    module = type(sys)(module_name)
    module.__file__ = path.as_posix()
    sys.modules[module_name] = module
    exec(compile(payload, path.as_posix(), "exec"), module.__dict__)  # noqa: S102
    return module


def _fresh_registered_module(
    registration: Mapping[str, Any], role: str, canonical_name: str
) -> ModuleType:
    """Replace an ambient dependency with the exact pre-registered bytes."""
    reference = registration["sources"][role]
    path = (REPO / reference["relative_path"]).resolve()
    C.require(path.is_relative_to(REPO), f"registered {role} escapes repository")
    ambient = sys.modules.get(canonical_name)
    if ambient is not None:
        ambient_file = getattr(ambient, "__file__", None)
        C.require(
            isinstance(ambient_file, str)
            and Path(ambient_file).resolve() == path,
            f"ambient {canonical_name} origin differs from registered {role}",
        )
    payload = _read_regular_bytes(path, reference["sha256"], f"registered {role}")
    module = ModuleType(canonical_name)
    module.__file__ = path.as_posix()
    module.__package__ = canonical_name.rpartition(".")[0]
    sys.modules[canonical_name] = module
    exec(compile(payload, path.as_posix(), "exec"), module.__dict__)  # noqa: S102
    if canonical_name.startswith("lcwm."):
        package = __import__("lcwm")
        setattr(package, canonical_name.rsplit(".", 1)[1], module)
    return module


def _load_registered_evaluator(registration: Mapping[str, Any]) -> ModuleType:
    """Fresh-load evaluator and every local import that controls its decision."""
    contract = _fresh_registered_module(
        registration, "contract", "lcwm.v248_gate0_contract"
    )
    can_state = _fresh_registered_module(
        registration, "can_state", "lcwm.v247_can_state"
    )
    cream_state = _fresh_registered_module(
        registration, "cream_state", "lcwm.v247_cream_state"
    )
    producer = _fresh_registered_module(
        registration, "semantic_producer", "produce_v248_gate0_progress"
    )
    evaluator = _fresh_registered_module(
        registration,
        "semantic_evaluator",
        "v248_registered_screen_evaluator_for_authorization",
    )
    C.require(evaluator.producer is producer, "evaluator used ambient producer module")
    C.require(
        evaluator.require is contract.require and producer.require is contract.require,
        "evaluator/producer used ambient contract module",
    )
    C.require(
        producer.CausalCanCount is can_state.CausalCanCount
        and producer.run_causal_cream_state is cream_state.run_causal_cream_state,
        "semantic producer used ambient state engine module",
    )
    return evaluator


def _load_registered_replay(bundle: Mapping[str, Any]) -> ModuleType:
    return _load_registered_module(
        bundle["registrations"]["chain3_8000_screen.json"]["registration"],
        "formal_replay",
        "v248_registered_replay_for_confirm_authorization",
    )


def _derived_path(
    replay: ModuleType, target_root: Path, complete_slot: str
) -> tuple[Path, Path]:
    complete_path = replay._complete_path_from_slot(target_root, complete_slot)
    return complete_path.parent, complete_path


def _bundle_authority(bundle: Mapping[str, Any]) -> dict[str, Any]:
    campaign = bundle["campaign"]
    return {
        "freeze_complete_sha256": bundle["freeze_complete_sha256"],
        "campaign_manifest_sha256": bundle["campaign_manifest_sha256"],
        "campaign_body_sha256": campaign["body_sha256"],
        "campaign_projection_sha256": campaign["projection_sha256"],
        "campaign_id": campaign["campaign_id"],
    }


def _combined_authority(
    bundle: Mapping[str, Any], *, phase: str, output_complete_slot: str
) -> dict[str, Any]:
    return {
        **_bundle_authority(bundle),
        "registration_slots": [
            f"{C.PANEL_IDS[seed]}_{phase}.json" for seed in (8000, 8100)
        ],
        "output_complete_slot": output_complete_slot,
        "confirm_authorization_complete_sha256": None,
    }


def frozen_predicate() -> dict[str, Any]:
    return {
        "schema": AUTHORIZATION_PREDICATE_SCHEMA,
        "automatic_expansion_rule": C.AUTOMATIC_CONFIRM_EXPANSION_RULE,
        "required_phase": "screen",
        "required_screen_decision": "STOP",
        "required_screen_promotion_authorized": False,
        "required_screen_metrics_passed": True,
        "required_metric_checks_passed": True,
        "require_every_metric_check_true": True,
        "required_metric_check_names": sorted(REQUIRED_METRIC_CHECK_NAMES),
        "screen_episode_ids": {
            C.PANEL_IDS[seed]: list(C.SCREEN_EPISODES[seed])
            for seed in (8000, 8100)
        },
        "required_combined_primary_episodes": sum(
            len(C.SCREEN_EPISODES[seed]) for seed in (8000, 8100)
        ),
        "authorized_confirm_episode_ids": {
            C.PANEL_IDS[seed]: list(range(96)) for seed in (8000, 8100)
        },
    }


def _registration_refs(
    bundle: Mapping[str, Any], phase: str
) -> dict[str, dict[str, Any]]:
    result = {}
    for seed in (8000, 8100):
        panel_id = C.PANEL_IDS[seed]
        slot = f"{panel_id}_{phase}.json"
        entry = bundle["registrations"][slot]
        result[panel_id] = {
            "registration_slot": slot,
            "file_sha256": entry["file_sha256"],
            "canonical_self_sha256": entry["registration"]["self_sha256"],
        }
    return result


def _assert_frozen_screen_predicate(
    result: Mapping[str, Any],
    subgate: Mapping[str, Any],
    recomputed_subgate: Mapping[str, Any],
) -> None:
    predicate = frozen_predicate()
    C.require_exact_keys(predicate, PREDICATE_FIELDS, "confirm predicate")
    visual = result["visual_subgate"]
    C.require(result["phase"] == predicate["required_phase"], "screen phase predicate")
    C.require(
        result["decision"] == visual["decision"] == predicate["required_screen_decision"],
        "screen decision predicate",
    )
    C.require(
        visual["promotion_authorized"]
        is predicate["required_screen_promotion_authorized"],
        "screen promotion predicate",
    )
    C.require(
        visual["screen_metrics_passed"]
        is predicate["required_screen_metrics_passed"],
        "screen metrics predicate",
    )
    C.require(
        visual["metric_checks_passed"]
        is predicate["required_metric_checks_passed"],
        "screen combined predicate",
    )
    C.require(
        len(visual["checks"]) == len(predicate["required_metric_check_names"])
        and sorted(visual["checks"]) == predicate["required_metric_check_names"],
        "screen metric check-name closure drift",
    )
    C.require(
        all(value is True for value in visual["checks"].values()),
        "not every frozen screen metric check passed",
    )
    C.require(
        visual["thresholds"]["phase_required_primary_chain3_episodes"]
        == predicate["required_combined_primary_episodes"],
        "screen episode-count predicate",
    )
    C.require(
        result["visual_subgate"] == subgate == dict(recomputed_subgate),
        "screen subgate does not equal frozen-config recomputation",
    )


def _validate_screen_evaluation(
    bundle: Mapping[str, Any],
    expected_complete_sha256: str,
    expected_progress_complete_sha256: str,
    expected_pre_oracle_seal_sha256: str,
    expected_pre_oracle_complete_sha256: str,
    *,
    replay_support: ModuleType | None = None,
) -> dict[str, Any]:
    replay = replay_support or _load_registered_replay(bundle)
    output_slot = bundle["target_output_slots"]["screen_evaluation_complete"]
    root, complete_path = _derived_path(
        replay, Path(bundle["target_root"]), output_slot
    )
    C.require(root.is_dir() and not root.is_symlink(), "screen evaluation root missing/symlink")
    complete_bytes = _read_regular_bytes(
        complete_path,
        expected_complete_sha256,
        "externally anchored screen evaluation COMPLETE",
    )
    complete = _strict_json_bytes(
        complete_bytes,
        "screen evaluation COMPLETE",
    )
    screen_entries = {
        seed: bundle["registrations"][f"{C.PANEL_IDS[seed]}_screen.json"]
        for seed in (8000, 8100)
    }
    evaluator = _load_registered_evaluator(
        screen_entries[8000]["registration"]
    )
    C.require_exact_keys(complete, evaluator.COMPLETE_FIELDS, "screen evaluation COMPLETE")
    marker_filename = complete["marker_filename"]
    C.require(
        isinstance(marker_filename, str)
        and Path(marker_filename).name == marker_filename,
        "screen evaluation marker filename is unsafe",
    )
    sealed_files = {
        "RESULT.json": complete["result_sha256"],
        "VISUAL_SUBGATE.json": complete["subgate_sha256"],
        "PRE_ORACLE_SEAL.json": complete["pre_oracle_seal_sha256"],
        "EVALUATOR.py": complete["evaluator_sha256"],
        marker_filename: complete["marker_sha256"],
        "COMPLETE.json": expected_complete_sha256,
    }
    C.require(
        {path.name for path in root.iterdir()} == set(sealed_files),
        "screen evaluation root closure mismatch",
    )
    captured = {
        name: (
            complete_bytes
            if name == "COMPLETE.json"
            else _read_regular_bytes(
                root / name, digest, f"screen evaluation {name}"
            )
        )
        for name, digest in sealed_files.items()
    }
    result = _strict_json_bytes(captured["RESULT.json"], "screen evaluation RESULT")
    subgate = _strict_json_bytes(
        captured["VISUAL_SUBGATE.json"], "screen visual subgate"
    )
    C.require_exact_keys(result, evaluator.EVALUATION_FIELDS, "screen evaluation RESULT")
    expected_authority = _combined_authority(
        bundle, phase="screen", output_complete_slot=output_slot
    )
    C.require_exact_keys(
        result.get("authority"), COMBINED_AUTHORITY_FIELDS, "screen evaluation authority"
    )
    C.require_exact_keys(
        complete.get("authority"), COMBINED_AUTHORITY_FIELDS, "screen COMPLETE authority"
    )
    C.require(result["authority"] == expected_authority, "screen evaluation genesis drift")
    C.require(complete["authority"] == expected_authority, "screen COMPLETE genesis drift")

    progress_complete_sha = expected_progress_complete_sha256
    C.require(
        C.is_sha256(progress_complete_sha)
        and result["progress_artifact"]["complete_sha256"]
        == progress_complete_sha,
        "screen evaluation differs from external progress COMPLETE anchor",
    )
    progress_root, _progress_complete_path = _derived_path(
        replay,
        Path(bundle["target_root"]),
        bundle["target_output_slots"]["screen_progress_complete"],
    )
    (
        reloaded_bundle,
        progress_registrations,
        progress_registration_hashes,
        progress_result,
        _progress_frames,
        progress_config,
        progress_seals,
    ) = evaluator.load_anchored_campaign_progress(
        freeze_bundle_root=Path(bundle["root"]),
        expected_freeze_complete_sha256=bundle["freeze_complete_sha256"],
        progress_root=progress_root,
        expected_progress_complete_sha256=progress_complete_sha,
    )
    C.require(
        _bundle_authority(reloaded_bundle) == _bundle_authority(bundle),
        "screen progress belongs to a different campaign",
    )
    C.require(
        progress_seals == result["progress_artifact"],
        "screen evaluation does not seal the actual progress artifact",
    )
    pre_oracle_root, _pre_oracle_complete_path = _derived_path(
        replay,
        Path(bundle["target_root"]),
        bundle["target_output_slots"]["screen_pre_oracle_complete"],
    )
    pre_oracle_authority = evaluator.campaign_node_authority(
        bundle, "screen", "pre_oracle", None
    )
    evaluator.validate_published_pre_oracle(
        root=pre_oracle_root,
        expected_pre_oracle_sha256=expected_pre_oracle_seal_sha256,
        expected_complete_sha256=expected_pre_oracle_complete_sha256,
        phase="screen",
        registration_hashes=progress_registration_hashes,
        progress_seals=progress_seals,
        evaluator_sha256=complete["evaluator_sha256"],
        expected_authority=pre_oracle_authority,
    )
    C.require(
        result["pre_oracle_seal_sha256"] == expected_pre_oracle_seal_sha256
        and result["pre_oracle_complete_sha256"]
        == expected_pre_oracle_complete_sha256,
        "screen evaluation differs from external pre-oracle anchors",
    )
    with tempfile.TemporaryDirectory(prefix="v248-screen-eval-snapshot-") as raw:
        snapshot = Path(raw)
        for name, payload in captured.items():
            destination = snapshot / name
            with destination.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        evaluator._validate_evaluation_stage(
            snapshot,
            {seed: screen_entries[seed]["registration"] for seed in (8000, 8100)},
            {seed: screen_entries[seed]["file_sha256"] for seed in (8000, 8100)},
            progress_seals,
            expected_authority=expected_authority,
        )
    prefix_exact = all(
        bool(check["exact_after_rounding_6_decimals"])
        for check in progress_result["prefix_checks"].values()
    )
    recovery_passed = bool(
        progress_result["recovered_endpoint"]["present"]
        and progress_result["recovered_endpoint"][
            "rgb_recovery_prerequisites_passed"
        ]
    )
    recomputed_subgate = evaluator.visual_subgate_decision(
        metrics=result["metrics"],
        config=progress_config,
        phase="screen",
        primary_episode_count=sum(
            len(progress_registrations[seed]["panel"]["episodes"])
            for seed in (8000, 8100)
        ),
        all_prefix_exact=prefix_exact,
        recovery_passed=recovery_passed,
    )
    _assert_frozen_screen_predicate(result, subgate, recomputed_subgate)
    return {
        "root": root,
        "complete": complete,
        "result": result,
        "subgate": subgate,
        "complete_sha256": expected_complete_sha256,
        "result_sha256": complete["result_sha256"],
        "subgate_sha256": complete["subgate_sha256"],
        "progress_seals": progress_seals,
        "progress_result": progress_result,
        "pre_oracle_seal_sha256": expected_pre_oracle_seal_sha256,
        "pre_oracle_complete_sha256": expected_pre_oracle_complete_sha256,
        "captured_evaluation_file_sha256s": dict(sealed_files),
    }


def _authorization_authority(
    bundle: Mapping[str, Any], screen: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        **_bundle_authority(bundle),
        "screen_evaluation_complete_sha256": screen["complete_sha256"],
        "screen_evaluation_complete_slot": bundle["target_output_slots"][
            "screen_evaluation_complete"
        ],
        "authorization_complete_slot": bundle["target_output_slots"][
            "confirm_authorization_complete"
        ],
    }


def _authorization_input(screen: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "screen_rgb_complete_sha256s": {
            panel_id: screen["progress_result"]["rgb_inputs"][panel_id][
                "complete_sha256"
            ]
            for panel_id in C.PANEL_IDS.values()
        },
        "screen_progress_complete_sha256": screen["progress_seals"][
            "complete_sha256"
        ],
        "screen_pre_oracle_seal_sha256": screen[
            "pre_oracle_seal_sha256"
        ],
        "screen_pre_oracle_complete_sha256": screen[
            "pre_oracle_complete_sha256"
        ],
        "screen_evaluation_complete_sha256": screen["complete_sha256"],
        "screen_evaluation_result_sha256": screen["result_sha256"],
        "screen_visual_subgate_sha256": screen["subgate_sha256"],
    }


def _screen_closure(screen: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "input": _authorization_input(screen),
        "progress_seals": screen["progress_seals"],
        "captured_evaluation_file_sha256s": screen[
            "captured_evaluation_file_sha256s"
        ],
    }


def _validate_created_utc(value: Any) -> None:
    C.require(isinstance(value, str), "authorization created_utc must be text")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise C.Gate0ContractError(
            "authorization created_utc must be canonical second-resolution UTC"
        ) from error
    C.require(
        parsed.strftime("%Y-%m-%dT%H:%M:%SZ") == value,
        "authorization created_utc is not canonical",
    )


def _validate_authorization_stage(
    stage: Path,
    expected_authority: Mapping[str, Any],
    registered_producer_sha256: str,
    *,
    expected_input: Mapping[str, Any],
    expected_screen_registrations: Mapping[str, Any],
    expected_confirm_registrations: Mapping[str, Any],
) -> None:
    C.require(
        {path.name for path in stage.iterdir()}
        == {
            "RESULT.json",
            "PREDICATE.json",
            "AUTHORIZED.json",
            "PRODUCER.py",
            "COMPLETE.json",
        },
        "authorization artifact closure mismatch",
    )
    result = _strict_json_file(stage / "RESULT.json", "authorization RESULT")
    predicate = _strict_json_file(stage / "PREDICATE.json", "authorization predicate")
    marker = _strict_json_file(stage / "AUTHORIZED.json", "authorization marker")
    complete = _strict_json_file(stage / "COMPLETE.json", "authorization COMPLETE")
    C.require_exact_keys(result, RESULT_FIELDS, "authorization RESULT")
    C.require_exact_keys(predicate, PREDICATE_FIELDS, "authorization predicate")
    C.require_exact_keys(marker, MARKER_FIELDS, "authorization marker")
    C.require_exact_keys(complete, COMPLETE_FIELDS, "authorization COMPLETE")
    C.require_exact_keys(result["authority"], AUTHORIZATION_AUTHORITY_FIELDS, "authority")
    C.require_exact_keys(complete["authority"], AUTHORIZATION_AUTHORITY_FIELDS, "authority")
    C.require(
        predicate == result["predicate"] == frozen_predicate(),
        "authorization predicate drift",
    )
    C.require(
        result["authority"] == complete["authority"] == dict(expected_authority),
        "authority drift",
    )
    C.require(
        result["schema"] == AUTHORIZATION_RESULT_SCHEMA
        and result["status"] == "screen_predicate_satisfied"
        and result["decision"] == "AUTHORIZE_CONFIRM",
        "authorization RESULT status",
    )
    C.require(
        result["claim_scope"] == "authorize_exact_frozen_confirm_schedule_only",
        "authorization claim scope drift",
    )
    _validate_created_utc(result["created_utc"])
    for collection in (result["screen_registrations"], result["confirm_registrations"]):
        C.require_exact_keys(collection, frozenset(C.PANEL_IDS.values()), "registration pair")
        for panel_id, reference in collection.items():
            C.require_exact_keys(reference, REGISTRATION_REF_FIELDS, f"registration {panel_id}")
    C.require_exact_keys(result["input"], INPUT_FIELDS, "authorization input")
    C.require_exact_keys(result["output"], OUTPUT_FIELDS, "authorization output")
    C.require_exact_keys(result["safety"], SAFETY_FIELDS, "authorization safety")
    C.require(
        result["input"] == dict(expected_input),
        "authorization ancestor input drift",
    )
    C.require(
        result["screen_registrations"] == dict(expected_screen_registrations),
        "authorization screen registration drift",
    )
    C.require(
        result["confirm_registrations"] == dict(expected_confirm_registrations),
        "authorization confirm registration drift",
    )
    C.require(
        result["safety"]
        == {
            "oracle_path_cli_supported": False,
            "oracle_opened_or_hashed": False,
            "simulator_imported": False,
            "environment_constructed_or_reset": False,
            "caller_selected_registration": False,
            "caller_selected_output": False,
        },
        "authorization safety contract",
    )
    producer_sha = C.sha256_file(stage / "PRODUCER.py")
    result_sha = C.sha256_file(stage / "RESULT.json")
    predicate_sha = C.sha256_file(stage / "PREDICATE.json")
    marker_sha = C.sha256_file(stage / "AUTHORIZED.json")
    C.require(producer_sha == registered_producer_sha256, "authorization producer drift")
    C.require(result["output"]["producer_sha256"] == producer_sha, "producer result seal")
    C.require(result["output"]["predicate_sha256"] == predicate_sha, "predicate result seal")
    C.require(
        marker
        == {
            "schema": AUTHORIZATION_MARKER_SCHEMA,
            "decision": "AUTHORIZE_CONFIRM",
            "result_sha256": result_sha,
            "predicate_sha256": predicate_sha,
        },
        "authorization marker drift",
    )
    C.require(
        complete
        == {
            "schema": AUTHORIZATION_COMPLETE_SCHEMA,
            "status": "atomic_confirm_authorization",
            "atomic_commit": True,
            "decision": "AUTHORIZE_CONFIRM",
            "authority": dict(expected_authority),
            "result_sha256": result_sha,
            "predicate_sha256": predicate_sha,
            "producer_sha256": producer_sha,
            "marker_sha256": marker_sha,
        },
        "authorization COMPLETE drift",
    )


def validate_authorization_artifact(
    bundle: Mapping[str, Any],
    expected_complete_sha256: str,
    *,
    replay_support: ModuleType | None = None,
) -> dict[str, Any]:
    """Capture A once, revalidate its F/P/S/Q/E ancestry, then validate privately."""
    C.require(C.is_sha256(expected_complete_sha256), "authorization A anchor invalid")
    replay = replay_support or _load_registered_replay(bundle)
    complete_slot = bundle["target_output_slots"]["confirm_authorization_complete"]
    root, complete_path = _derived_path(
        replay, Path(bundle["target_root"]), complete_slot
    )
    C.require(
        root.is_dir() and not root.is_symlink(),
        "confirm authorization root missing/symlink",
    )
    complete_bytes = _read_regular_bytes(
        complete_path,
        expected_complete_sha256,
        "externally anchored confirm authorization COMPLETE",
    )
    complete = _strict_json_bytes(
        complete_bytes, "confirm authorization COMPLETE"
    )
    C.require_exact_keys(complete, COMPLETE_FIELDS, "confirm authorization COMPLETE")
    for field in (
        "result_sha256",
        "predicate_sha256",
        "producer_sha256",
        "marker_sha256",
    ):
        C.require(C.is_sha256(complete[field]), f"authorization {field} invalid")
    sealed_files = {
        "RESULT.json": complete["result_sha256"],
        "PREDICATE.json": complete["predicate_sha256"],
        "AUTHORIZED.json": complete["marker_sha256"],
        "PRODUCER.py": complete["producer_sha256"],
        "COMPLETE.json": expected_complete_sha256,
    }
    C.require(
        {path.name for path in root.iterdir()} == set(sealed_files),
        "authorization artifact closure mismatch",
    )
    captured = {
        name: (
            complete_bytes
            if name == "COMPLETE.json"
            else _read_regular_bytes(root / name, digest, f"authorization {name}")
        )
        for name, digest in sealed_files.items()
    }
    result = _strict_json_bytes(captured["RESULT.json"], "authorization RESULT")
    C.require_exact_keys(result, RESULT_FIELDS, "authorization RESULT")
    inputs = dict(C.require_exact_keys(result["input"], INPUT_FIELDS, "authorization input"))
    for field in (
        "screen_progress_complete_sha256",
        "screen_pre_oracle_seal_sha256",
        "screen_pre_oracle_complete_sha256",
        "screen_evaluation_complete_sha256",
        "screen_evaluation_result_sha256",
        "screen_visual_subgate_sha256",
    ):
        C.require(C.is_sha256(inputs[field]), f"authorization input {field} invalid")
    C.require_exact_keys(
        inputs["screen_rgb_complete_sha256s"],
        frozenset(C.PANEL_IDS.values()),
        "authorization screen RGB anchors",
    )
    C.require(
        all(C.is_sha256(value) for value in inputs["screen_rgb_complete_sha256s"].values()),
        "authorization screen RGB anchor invalid",
    )
    screen = _validate_screen_evaluation(
        bundle,
        inputs["screen_evaluation_complete_sha256"],
        inputs["screen_progress_complete_sha256"],
        inputs["screen_pre_oracle_seal_sha256"],
        inputs["screen_pre_oracle_complete_sha256"],
        replay_support=replay,
    )
    expected_input = _authorization_input(screen)
    C.require(inputs == expected_input, "authorization does not bind captured F/P/S/Q/E")
    expected_authority = _authorization_authority(bundle, screen)
    registered_sha = bundle["registrations"]["chain3_8000_screen.json"][
        "registration"
    ]["sources"]["confirm_authorizer"]["sha256"]
    with tempfile.TemporaryDirectory(prefix="v248-auth-snapshot-") as raw:
        snapshot = Path(raw)
        for name, payload in captured.items():
            destination = snapshot / name
            with destination.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        _validate_authorization_stage(
            snapshot,
            expected_authority,
            registered_sha,
            expected_input=expected_input,
            expected_screen_registrations=_registration_refs(bundle, "screen"),
            expected_confirm_registrations=_registration_refs(bundle, "confirm"),
        )
    C.assert_no_simulator_modules_imported()
    return {
        "root": root,
        "complete": complete,
        "result": result,
        "complete_sha256": expected_complete_sha256,
        "screen": screen,
        "captured_authorization_file_sha256s": dict(sealed_files),
    }


def authorize(
    *,
    freeze_bundle_root: Path,
    expected_freeze_complete_sha256: str,
    expected_screen_progress_complete_sha256: str,
    expected_screen_pre_oracle_seal_sha256: str,
    expected_screen_pre_oracle_complete_sha256: str,
    expected_screen_evaluation_complete_sha256: str,
    validate_only: bool = False,
) -> dict[str, Any]:
    bundle = C.load_and_validate_campaign_bundle(
        freeze_bundle_root,
        expected_freeze_complete_sha256,
        expected_source_role="confirm_authorizer",
    )
    C.assert_no_simulator_modules_imported()
    replay = _load_registered_replay(bundle)
    C.assert_no_simulator_modules_imported()
    screen = _validate_screen_evaluation(
        bundle,
        expected_screen_evaluation_complete_sha256,
        expected_screen_progress_complete_sha256,
        expected_screen_pre_oracle_seal_sha256,
        expected_screen_pre_oracle_complete_sha256,
        replay_support=replay,
    )
    C.assert_no_simulator_modules_imported()
    authority = _authorization_authority(bundle, screen)
    C.require_exact_keys(authority, AUTHORIZATION_AUTHORITY_FIELDS, "authorization authority")
    output_slot = authority["authorization_complete_slot"]
    output_root = replay._output_path_from_slot(bundle["target_root"], output_slot)
    replay._check_output_separation(
        output_root,
        [Path(bundle["root"]), Path(screen["root"])],
    )
    if validate_only:
        return {
            "status": "VALID",
            "decision": "AUTHORIZE_CONFIRM",
            "authority": authority,
            "output_created": False,
        }

    def revalidate() -> None:
        current = C.load_and_validate_campaign_bundle(
            bundle["root"],
            expected_freeze_complete_sha256,
            expected_source_role="confirm_authorizer",
        )
        C.require(_bundle_authority(current) == _bundle_authority(bundle), "campaign drift")
        current_screen = _validate_screen_evaluation(
            current,
            expected_screen_evaluation_complete_sha256,
            expected_screen_progress_complete_sha256,
            expected_screen_pre_oracle_seal_sha256,
            expected_screen_pre_oracle_complete_sha256,
        )
        C.require(
            _screen_closure(current_screen) == _screen_closure(screen),
            "screen ancestor seals changed before authorization publish",
        )
        C.require(not os.path.lexists(output_root), "authorization output appeared")
        C.assert_no_simulator_modules_imported()

    revalidate()
    predicate = frozen_predicate()
    registered_sha = bundle["registrations"]["chain3_8000_screen.json"][
        "registration"
    ]["sources"]["confirm_authorizer"]["sha256"]
    producer_reference = bundle["registrations"]["chain3_8000_screen.json"][
        "registration"
    ]["sources"]["confirm_authorizer"]
    producer_source = (REPO / producer_reference["relative_path"]).resolve()
    producer_bytes = _read_regular_bytes(
        producer_source, registered_sha, "registered confirm authorizer source"
    )
    published_complete_sha256: str | None = None

    def build(stage: Path) -> None:
        nonlocal published_complete_sha256
        replay._write_bytes(stage / "PRODUCER.py", producer_bytes)
        C.require(C.sha256_file(stage / "PRODUCER.py") == registered_sha, "producer source drift")
        replay._write_json(stage / "PREDICATE.json", predicate)
        payload = {
            "schema": AUTHORIZATION_RESULT_SCHEMA,
            "status": "screen_predicate_satisfied",
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "decision": "AUTHORIZE_CONFIRM",
            "claim_scope": "authorize_exact_frozen_confirm_schedule_only",
            "authority": authority,
            "predicate": predicate,
            "screen_registrations": _registration_refs(bundle, "screen"),
            "confirm_registrations": _registration_refs(bundle, "confirm"),
            "input": _authorization_input(screen),
            "output": {
                "predicate_sha256": C.sha256_file(stage / "PREDICATE.json"),
                "producer_sha256": C.sha256_file(stage / "PRODUCER.py"),
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
        replay._write_json(stage / "RESULT.json", payload)
        marker = {
            "schema": AUTHORIZATION_MARKER_SCHEMA,
            "decision": "AUTHORIZE_CONFIRM",
            "result_sha256": C.sha256_file(stage / "RESULT.json"),
            "predicate_sha256": payload["output"]["predicate_sha256"],
        }
        replay._write_json(stage / "AUTHORIZED.json", marker)
        complete = {
            "schema": AUTHORIZATION_COMPLETE_SCHEMA,
            "status": "atomic_confirm_authorization",
            "atomic_commit": True,
            "decision": "AUTHORIZE_CONFIRM",
            "authority": authority,
            "result_sha256": C.sha256_file(stage / "RESULT.json"),
            "predicate_sha256": C.sha256_file(stage / "PREDICATE.json"),
            "producer_sha256": C.sha256_file(stage / "PRODUCER.py"),
            "marker_sha256": C.sha256_file(stage / "AUTHORIZED.json"),
        }
        replay._write_json(stage / "COMPLETE.json", complete)
        published_complete_sha256 = C.sha256_file(stage / "COMPLETE.json")

    def validate(stage: Path) -> None:
        current = C.load_and_validate_campaign_bundle(
            bundle["root"],
            expected_freeze_complete_sha256,
            expected_source_role="confirm_authorizer",
        )
        current_screen = _validate_screen_evaluation(
            current,
            expected_screen_evaluation_complete_sha256,
            expected_screen_progress_complete_sha256,
            expected_screen_pre_oracle_seal_sha256,
            expected_screen_pre_oracle_complete_sha256,
        )
        C.require(
            _bundle_authority(current) == _bundle_authority(bundle)
            and _screen_closure(current_screen) == _screen_closure(screen),
            "F/P/S/Q/E closure changed during authorization staging",
        )
        _validate_authorization_stage(
            stage,
            authority,
            registered_sha,
            expected_input=_authorization_input(screen),
            expected_screen_registrations=_registration_refs(bundle, "screen"),
            expected_confirm_registrations=_registration_refs(bundle, "confirm"),
        )

    replay.atomic_publish_directory(
        output_root,
        build,
        validate,
        trusted_target_root=bundle["target_root"],
    )
    C.require(
        C.is_sha256(published_complete_sha256),
        "authorization COMPLETE digest was not captured before publish",
    )
    revalidate_bundle = C.load_and_validate_campaign_bundle(
        bundle["root"],
        expected_freeze_complete_sha256,
        expected_source_role="confirm_authorizer",
    )
    C.require(
        _bundle_authority(revalidate_bundle) == _bundle_authority(bundle),
        "campaign drift after publish",
    )
    published = validate_authorization_artifact(
        revalidate_bundle,
        published_complete_sha256,
        replay_support=replay,
    )
    final_screen = published["screen"]
    C.require(
        _screen_closure(final_screen) == _screen_closure(screen),
        "screen ancestor closure changed after authorization publish",
    )
    C.assert_no_simulator_modules_imported()
    return {
        "status": "PASS",
        "decision": "AUTHORIZE_CONFIRM",
        "output": output_root.as_posix(),
        "complete_sha256": published["complete_sha256"],
        "authority": authority,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-bundle", type=Path, required=True)
    parser.add_argument("--expected-freeze-complete-sha256", required=True)
    parser.add_argument("--expected-screen-progress-complete-sha256", required=True)
    parser.add_argument("--expected-screen-pre-oracle-seal-sha256", required=True)
    parser.add_argument("--expected-screen-pre-oracle-complete-sha256", required=True)
    parser.add_argument("--expected-screen-evaluation-complete-sha256", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    for name in (
        "expected_freeze_complete_sha256",
        "expected_screen_progress_complete_sha256",
        "expected_screen_pre_oracle_seal_sha256",
        "expected_screen_pre_oracle_complete_sha256",
        "expected_screen_evaluation_complete_sha256",
    ):
        if not C.is_sha256(getattr(args, name)):
            parser.error(f"--{name.replace('_', '-')} must be 64 lowercase hex")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = authorize(
        freeze_bundle_root=args.freeze_bundle,
        expected_freeze_complete_sha256=args.expected_freeze_complete_sha256,
        expected_screen_progress_complete_sha256=(
            args.expected_screen_progress_complete_sha256
        ),
        expected_screen_pre_oracle_seal_sha256=(
            args.expected_screen_pre_oracle_seal_sha256
        ),
        expected_screen_pre_oracle_complete_sha256=(
            args.expected_screen_pre_oracle_complete_sha256
        ),
        expected_screen_evaluation_complete_sha256=(
            args.expected_screen_evaluation_complete_sha256
        ),
        validate_only=args.validate_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
