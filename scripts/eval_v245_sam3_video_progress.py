#!/usr/bin/env python3
"""Post-seal privileged evaluation for V245 SAM3 video progress probes.

Every producer hash and every causal state transition is verified before this
program opens the deployment summary.  Ground truth uses only episode id, seed,
steps, and sticky first-achievement events 1/3/5; success fields are ignored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parent.parent
SCHEMA = "v245_sam3_video_progress_postseal_eval_v1"
PROBE_SCHEMA = "v245_sam3_video_progress_probe_v1"
FRAME_SCHEMA = "v245_sam3_video_progress_frame_v1"
COMPLETE_SCHEMA = "v245_sam3_video_progress_complete_v1"
EXPECTED_SUMMARY_SHA256 = "a4dc0d5f73328247d11592ad08d55031023a2bb00bbaa3c7436e0a4a9387ef58"
INCREMENTS = {
    "can_1": {"kind": "can", "level": 1, "event_index": 1},
    "can_2": {"kind": "can", "level": 2, "event_index": 3},
    "cream": {"kind": "cream", "level": 1, "event_index": 5},
}
GATE_THRESHOLDS = {
    "emitted_frame_coverage_gte": 0.95,
    "current_evidence_uncertainty_lte": 0.05,
    "stage_exact_accuracy_gte": 0.95,
    "true_increment_coverage_gte": 1.0,
    "median_absolute_transition_error_steps_lte": 20,
    "extra_increment_count_lte": 0,
    "never_true_terminal_specificity_gte": 0.95,
    "fresh_prefix_checks_required": True,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"{path}: expected JSON object")
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            require(bool(line.strip()), f"{path}:{line_number}: blank row")
            row = json.loads(line)
            require(isinstance(row, dict), f"{path}:{line_number}: expected object")
            rows.append(row)
    require(rows, f"{path}: empty JSONL")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-run", action="append", type=Path, required=True)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def verify_frame_dynamics(frames: Sequence[dict[str, Any]], episode_id: int) -> None:
    previous_t = -1
    previous_can = 0
    previous_cream = False
    previous_scalar = 0
    for index, frame in enumerate(frames):
        where = f"ep{episode_id}:frame{index}"
        require(frame.get("schema") == FRAME_SCHEMA, f"{where}: schema mismatch")
        require(int(frame.get("episode_id", -1)) == episode_id, f"{where}: episode mismatch")
        require(int(frame.get("frame_index", -1)) == index, f"{where}: frame index mismatch")
        t_value = int(frame["t"])
        require(t_value > previous_t, f"{where}: non-monotone t")
        previous_t = t_value
        raw = frame.get("raw_geometry")
        progress = frame.get("causal_progress")
        require(isinstance(raw, dict) and isinstance(progress, dict), f"{where}: missing state")
        raw_can = raw.get("can_inside_count")
        require(raw_can is None or raw_can in {0, 1, 2}, f"{where}: invalid raw can count")
        raw_cream = raw.get("cream_inside")
        require(raw_cream is None or isinstance(raw_cream, bool), f"{where}: invalid cream")
        expected_can = previous_can if raw_can is None else max(previous_can, int(raw_can))
        expected_cream = previous_cream or raw_cream is True
        expected_scalar = expected_can + int(expected_cream)
        require(
            int(progress["sticky_can_inside_count"]) == expected_can,
            f"{where}: can state is not causal sticky max",
        )
        require(
            bool(progress["sticky_cream_inside"]) is expected_cream,
            f"{where}: cream state is not causal sticky OR",
        )
        require(int(progress["scalar_0_to_3"]) == expected_scalar, f"{where}: scalar mismatch")
        require(expected_scalar >= previous_scalar, f"{where}: progress decreased")
        require(
            bool(progress["used_can_carry"])
            is (raw_can is None and previous_can > 0),
            f"{where}: can carry flag mismatch",
        )
        require(
            bool(progress["used_cream_carry"])
            is (raw_cream is None and previous_cream),
            f"{where}: cream carry flag mismatch",
        )
        previous_can = expected_can
        previous_cream = expected_cream
        previous_scalar = expected_scalar


def validate_probe_run(
    run_dir: Path,
) -> tuple[dict[str, Any], dict[int, list[dict[str, Any]]], dict[str, Any]]:
    run_dir = run_dir.resolve()
    complete_path = run_dir / "COMPLETE.json"
    result_path = run_dir / "result.json"
    frames_path = run_dir / "frames.jsonl"
    source_path = run_dir / "producer_source.py"
    require(
        all(path.is_file() for path in (complete_path, result_path, frames_path, source_path)),
        f"{run_dir}: incomplete producer artifact",
    )
    complete = load_json(complete_path)
    require(complete.get("schema") == COMPLETE_SCHEMA, f"{run_dir}: complete schema")
    require(complete.get("status") == "complete", f"{run_dir}: not complete")
    actual_hashes = {
        "result_sha256": sha256_file(result_path),
        "frames_sha256": sha256_file(frames_path),
        "producer_source_sha256": sha256_file(source_path),
    }
    for key, value in actual_hashes.items():
        require(complete.get(key) == value, f"{run_dir}: {key} seal mismatch")
    result = load_json(result_path)
    require(result.get("schema") == PROBE_SCHEMA, f"{run_dir}: result schema mismatch")
    require(result.get("privileged_inputs_read") is False, f"{run_dir}: producer boundary")
    require(
        result.get("status") == "rgb_only_streaming_development_probe_complete",
        f"{run_dir}: result status mismatch",
    )
    require(
        result.get("producer_source_sha256") == actual_hashes["producer_source_sha256"],
        f"{run_dir}: result/source mismatch",
    )
    frames = load_jsonl(frames_path)
    frames_by_episode: dict[int, list[dict[str, Any]]] = {}
    for frame in frames:
        episode_id = int(frame["episode_id"])
        frames_by_episode.setdefault(episode_id, []).append(frame)
    declared = [int(value) for value in result.get("episode_ids", [])]
    require(set(frames_by_episode) == set(declared), f"{run_dir}: episode set mismatch")
    for episode_id, episode_frames in frames_by_episode.items():
        episode_frames.sort(key=lambda frame: int(frame["t"]))
        verify_frame_dynamics(episode_frames, episode_id)
        diagnostics = result.get("episodes", {}).get(str(episode_id))
        require(isinstance(diagnostics, dict), f"{run_dir}: missing episode diagnostics")
        require(
            int(diagnostics["frame_count"]) == len(episode_frames),
            f"{run_dir}: frame count mismatch",
        )
    prefix = result.get("prefix_checks", {})
    prefix_records = {}
    for episode_id in declared:
        record = prefix.get(str(episode_id))
        prefix_records[episode_id] = {
            "present": isinstance(record, dict),
            "frames": int(record.get("frames", 0)) if isinstance(record, dict) else 0,
            "exact": bool(record.get("exact_after_rounding_1e_6"))
            if isinstance(record, dict)
            else False,
            "reference_sha256": record.get("reference_sha256")
            if isinstance(record, dict)
            else None,
            "fresh_session_sha256": record.get("fresh_session_sha256")
            if isinstance(record, dict)
            else None,
        }
        if isinstance(record, dict):
            require(
                record.get("reference_sha256") == record.get("fresh_session_sha256")
                if record.get("exact_after_rounding_1e_6")
                else True,
                f"{run_dir}: inconsistent prefix certificate",
            )
    seal = {
        "run_dir": str(run_dir),
        "complete_sha256": sha256_file(complete_path),
        **actual_hashes,
        "prefix_checks": prefix_records,
    }
    return result, frames_by_episode, seal


def load_truth_after_all_seals(
    summary_path: Path, episode_ids: Sequence[int]
) -> tuple[dict[int, dict[str, Any]], str]:
    summary_sha = sha256_file(summary_path)
    require(summary_sha == EXPECTED_SUMMARY_SHA256, "source summary SHA mismatch")
    payload = load_json(summary_path)
    require(payload.get("task") == "chain3_lr2", "source summary task mismatch")
    records = {}
    wanted = set(episode_ids)
    for raw in payload.get("episode_records", []):
        episode_id = int(raw["idx"])
        if episode_id not in wanted:
            continue
        require(int(raw["seed"]) == 8700 + episode_id, "source seed mismatch")
        records[episode_id] = {
            "seed": int(raw["seed"]),
            "steps": int(raw["steps"]),
            "events": {int(key): int(value) for key, value in raw["events"].items()},
        }
    require(set(records) == wanted, "source summary does not cover all episodes")
    return records, summary_sha


def truth_at(events: Mapping[int, int], t_value: int) -> tuple[int, bool, int]:
    can_count = sum(int(index in events and events[index] <= t_value) for index in (1, 3))
    cream = bool(5 in events and events[5] <= t_value)
    return can_count, cream, can_count + int(cream)


def bbox_overlap(left: Sequence[float], right: Sequence[float]) -> dict[str, float]:
    ix0 = max(float(left[0]), float(right[0]))
    iy0 = max(float(left[1]), float(right[1]))
    ix1 = min(float(left[2]), float(right[2]))
    iy1 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(
        0.0, float(left[3]) - float(left[1])
    )
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(
        0.0, float(right[3]) - float(right[1])
    )
    union = left_area + right_area - intersection
    smaller = min(left_area, right_area)
    return {
        "intersection_px2": intersection,
        "iou": intersection / union if union > 0 else 0.0,
        "intersection_over_smaller_box": intersection / smaller if smaller > 0 else 0.0,
    }


def cross_prompt_collision_audit(
    frames_by_episode: Mapping[int, Sequence[dict[str, Any]]]
) -> dict[str, Any]:
    rows = []
    transition_rows = []
    flagged_frames = 0
    any_overlap_frames = 0
    for episode_id in sorted(frames_by_episode):
        previous_cream = False
        for frame in frames_by_episode[episode_id]:
            cream_selected = frame["selection"]["cream"]
            can_selected = frame["selection"]["can"]
            pair_rows = []
            for cream in cream_selected:
                for can in can_selected:
                    cream_box = cream.get("mask_bbox_xyxy")
                    can_box = can.get("mask_bbox_xyxy")
                    if cream_box is None or can_box is None:
                        continue
                    overlap = bbox_overlap(cream_box, can_box)
                    if overlap["intersection_px2"] <= 0:
                        continue
                    pair_rows.append(
                        {
                            "cream_object_id": int(cream["object_id"]),
                            "can_object_id": int(can["object_id"]),
                            **overlap,
                        }
                    )
            flagged = any(
                pair["intersection_over_smaller_box"] >= 0.50 for pair in pair_rows
            )
            if pair_rows:
                any_overlap_frames += 1
            if flagged:
                flagged_frames += 1
            current_cream = bool(frame["causal_progress"]["sticky_cream_inside"])
            record = {
                "episode_id": episode_id,
                "t": int(frame["t"]),
                "cream_progress": current_cream,
                "any_bbox_overlap": bool(pair_rows),
                "collision_flag_intersection_over_smaller_box_gte_0_5": flagged,
                "pairs": pair_rows,
            }
            if pair_rows:
                rows.append(record)
            if current_cream and not previous_cream:
                transition_rows.append(record)
            previous_cream = current_cream
    return {
        "mask_overlap_quantifiable_from_sealed_v1": False,
        "mask_overlap_limitation": (
            "v1 sealed mask hashes but not mask pixels; evaluator therefore fails "
            "closed to bbox overlap and cannot claim cross-prompt mask separation"
        ),
        "bbox_collision_rule_for_diagnostic": (
            "intersection over the smaller selected cream/can bbox >= 0.50"
        ),
        "frames_with_any_bbox_overlap": any_overlap_frames,
        "frames_with_flagged_bbox_collision": flagged_frames,
        "cream_transition_rows": transition_rows,
        "overlap_rows": rows,
    }


def predicted_increment_time(frames: Sequence[dict[str, Any]], name: str) -> int | None:
    specification = INCREMENTS[name]
    for frame in frames:
        progress = frame["causal_progress"]
        if specification["kind"] == "can":
            reached = int(progress["sticky_can_inside_count"]) >= int(specification["level"])
        else:
            reached = bool(progress["sticky_cream_inside"])
        if reached:
            return int(frame["t"])
    return None


def compute_metrics(
    frames_by_episode: Mapping[int, Sequence[dict[str, Any]]],
    truth: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    total = 0
    exact = 0
    can_exact = 0
    cream_exact = 0
    current_certain = 0
    can_carry = 0
    cream_carry = 0
    confusion: Counter[str] = Counter()
    per_episode = {}
    increment_rows = []
    stage_error_rows = []

    for episode_id in sorted(frames_by_episode):
        frames = frames_by_episode[episode_id]
        events = truth[episode_id]["events"]
        episode_total = 0
        episode_exact = 0
        episode_error_ts = []
        for frame in frames:
            t_value = int(frame["t"])
            true_can, true_cream, true_stage = truth_at(events, t_value)
            progress = frame["causal_progress"]
            pred_can = int(progress["sticky_can_inside_count"])
            pred_cream = bool(progress["sticky_cream_inside"])
            pred_stage = int(progress["scalar_0_to_3"])
            total += 1
            episode_total += 1
            is_exact = pred_stage == true_stage
            exact += int(is_exact)
            episode_exact += int(is_exact)
            can_exact += int(pred_can == true_can)
            cream_exact += int(pred_cream == true_cream)
            confusion[f"true_{true_stage}_pred_{pred_stage}"] += 1
            raw = frame["raw_geometry"]
            current_certain += int(
                raw.get("can_inside_count") is not None
                and raw.get("cream_inside") is not None
            )
            can_carry += int(progress["used_can_carry"])
            cream_carry += int(progress["used_cream_carry"])
            if not is_exact:
                episode_error_ts.append(t_value)
                stage_error_rows.append(
                    {
                        "episode_id": episode_id,
                        "t": t_value,
                        "true_can_count": true_can,
                        "pred_can_count": pred_can,
                        "true_cream": true_cream,
                        "pred_cream": pred_cream,
                        "true_stage": true_stage,
                        "pred_stage": pred_stage,
                    }
                )
        per_episode[str(episode_id)] = {
            "frames": episode_total,
            "stage_exact_frames": episode_exact,
            "stage_exact_accuracy": episode_exact / episode_total,
            "stage_error_t": episode_error_ts,
        }
        last_t = int(frames[-1]["t"])
        for name, specification in INCREMENTS.items():
            event_index = int(specification["event_index"])
            true_time = events.get(event_index)
            true_observable = true_time is not None and true_time <= last_t
            pred_time = predicted_increment_time(frames, name)
            if true_observable and pred_time is not None:
                status = "matched"
            elif true_observable:
                status = "missing"
            elif pred_time is not None:
                status = "extra"
            else:
                status = "true_negative"
            increment_rows.append(
                {
                    "episode_id": episode_id,
                    "increment": name,
                    "event_index": event_index,
                    "true_time": true_time,
                    "true_observable": true_observable,
                    "predicted_time": pred_time,
                    "status": status,
                    "signed_error_steps": (
                        pred_time - true_time
                        if true_observable and pred_time is not None
                        else None
                    ),
                    "absolute_error_steps": (
                        abs(pred_time - true_time)
                        if true_observable and pred_time is not None
                        else None
                    ),
                }
            )

    matched = [row for row in increment_rows if row["status"] == "matched"]
    missing = [row for row in increment_rows if row["status"] == "missing"]
    extra = [row for row in increment_rows if row["status"] == "extra"]
    true_negative = [row for row in increment_rows if row["status"] == "true_negative"]
    observable = len(matched) + len(missing)
    terminal_negative = len(extra) + len(true_negative)
    absolute_errors = [int(row["absolute_error_steps"]) for row in matched]
    status_by_increment = {}
    for name in INCREMENTS:
        relevant = [row for row in increment_rows if row["increment"] == name]
        status_by_increment[name] = dict(Counter(row["status"] for row in relevant))

    return {
        "framewise": {
            "frames": total,
            "emitted_frame_coverage": 1.0,
            "current_evidence_certain_frames": current_certain,
            "current_evidence_coverage": current_certain / total,
            "current_evidence_uncertainty": 1.0 - current_certain / total,
            "can_carry_frames": can_carry,
            "cream_carry_frames": cream_carry,
            "stage_exact_frames": exact,
            "stage_exact_accuracy": exact / total,
            "can_count_exact_accuracy": can_exact / total,
            "cream_state_exact_accuracy": cream_exact / total,
            "stage_confusion": dict(sorted(confusion.items())),
            "per_episode": per_episode,
            "stage_error_rows": stage_error_rows,
        },
        "increments": {
            "ground_truth": "sticky first-achievement events 1/3/5",
            "identity_free_can_matching": (
                "predicted count levels 1/2 are matched in order to place events 1/3"
            ),
            "observable_true_increments": observable,
            "matched_true_increments": len(matched),
            "missing_true_increments": len(missing),
            "extra_increments": len(extra),
            "true_negative_increments": len(true_negative),
            "true_increment_coverage": len(matched) / observable if observable else None,
            "median_absolute_error_steps": (
                float(statistics.median(absolute_errors)) if absolute_errors else None
            ),
            "mean_absolute_error_steps": (
                sum(absolute_errors) / len(absolute_errors) if absolute_errors else None
            ),
            "max_absolute_error_steps": max(absolute_errors) if absolute_errors else None,
            "status_by_increment": status_by_increment,
            "rows": increment_rows,
        },
        "terminal_specificity": {
            "definition": "never-true increment remains negative at final observed frame",
            "denominator": terminal_negative,
            "true_negatives": len(true_negative),
            "false_positives": len(extra),
            "specificity": len(true_negative) / terminal_negative if terminal_negative else None,
            "rows": [row for row in increment_rows if row["status"] in {"extra", "true_negative"}],
        },
    }


def audit_ep71(frames: Sequence[dict[str, Any]]) -> dict[str, Any]:
    t0_ids = sorted(item["object_id"] for item in frames[0]["selection"]["can"])
    previous_ids = None
    change_rows = []
    carry_rows = []
    increment_rows = []
    previous_count = 0
    for frame in frames:
        ids = sorted(item["object_id"] for item in frame["selection"]["can"])
        if previous_ids is not None and ids != previous_ids:
            change_rows.append({"t": int(frame["t"]), "from": previous_ids, "to": ids})
        previous_ids = ids
        progress = frame["causal_progress"]
        count = int(progress["sticky_can_inside_count"])
        if progress["used_can_carry"]:
            carry_rows.append(
                {
                    "t": int(frame["t"]),
                    "selected_ids": ids,
                    "raw_can_inside_count": frame["raw_geometry"]["can_inside_count"],
                    "sticky_can_inside_count": count,
                }
            )
        if count > previous_count:
            increment_rows.append(
                {
                    "t": int(frame["t"]),
                    "from": previous_count,
                    "to": count,
                    "selected_ids": ids,
                    "same_as_t0_ids": ids == t0_ids,
                    "raw_can_inside_count": frame["raw_geometry"]["can_inside_count"],
                    "used_can_carry": bool(progress["used_can_carry"]),
                }
            )
        previous_count = count
    return {
        "t0_selected_can_ids": t0_ids,
        "selected_id_set_change_count": len(change_rows),
        "selected_id_set_change_rows": change_rows,
        "can_carry_frame_count": len(carry_rows),
        "can_carry_rows": carry_rows,
        "can_increment_rows": increment_rows,
        "carry_created_increment": any(row["used_can_carry"] for row in increment_rows),
        "id_churn_present_at_increment": any(
            not row["same_as_t0_ids"] for row in increment_rows
        ),
    }


def gate(metrics: Mapping[str, Any], prefix: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    framewise = metrics["framewise"]
    increments = metrics["increments"]
    terminal = metrics["terminal_specificity"]
    prefix_pass = bool(prefix) and all(
        record["present"]
        and record["frames"] > 0
        and record["exact"]
        and record["reference_sha256"] == record["fresh_session_sha256"]
        for record in prefix.values()
    )
    checks = {
        "emitted_frame_coverage_gte": (
            framewise["emitted_frame_coverage"]
            >= GATE_THRESHOLDS["emitted_frame_coverage_gte"]
        ),
        "current_evidence_uncertainty_lte": (
            framewise["current_evidence_uncertainty"]
            <= GATE_THRESHOLDS["current_evidence_uncertainty_lte"]
        ),
        "stage_exact_accuracy_gte": (
            framewise["stage_exact_accuracy"]
            >= GATE_THRESHOLDS["stage_exact_accuracy_gte"]
        ),
        "true_increment_coverage_gte": (
            increments["true_increment_coverage"] is not None
            and increments["true_increment_coverage"]
            >= GATE_THRESHOLDS["true_increment_coverage_gte"]
        ),
        "median_absolute_transition_error_steps_lte": (
            increments["median_absolute_error_steps"] is not None
            and increments["median_absolute_error_steps"]
            <= GATE_THRESHOLDS["median_absolute_transition_error_steps_lte"]
        ),
        "extra_increment_count_lte": (
            increments["extra_increments"]
            <= GATE_THRESHOLDS["extra_increment_count_lte"]
        ),
        "never_true_terminal_specificity_gte": (
            terminal["specificity"] is not None
            and terminal["specificity"]
            >= GATE_THRESHOLDS["never_true_terminal_specificity_gte"]
        ),
        "fresh_prefix_checks_required": prefix_pass,
    }
    return {
        "decision": "GO" if all(checks.values()) else "STOP",
        "thresholds": GATE_THRESHOLDS,
        "checks": checks,
        "failed_checks": sorted(name for name, passed in checks.items() if not passed),
    }


def git_head() -> str | None:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip() if process.returncode == 0 else None


def main() -> int:
    args = parse_args()
    all_frames: dict[int, list[dict[str, Any]]] = {}
    seals = []
    prefix: dict[int, dict[str, Any]] = {}
    producer_source_hashes = set()

    # Phase A: no summary access.  Verify every complete/result/frames/source seal
    # and replay the producer's causal state machine first.
    for raw_run in args.probe_run:
        result, frames_by_episode, seal = validate_probe_run(raw_run.expanduser().resolve())
        for episode_id, frames in frames_by_episode.items():
            require(episode_id not in all_frames, f"duplicate episode {episode_id} across runs")
            all_frames[episode_id] = frames
        for episode_id, record in seal["prefix_checks"].items():
            prefix[episode_id] = record
        producer_source_hashes.add(seal["producer_source_sha256"])
        seals.append(seal)
        require(result.get("privileged_inputs_read") is False, "producer read privileged input")
    require(len(producer_source_hashes) == 1, "probe runs do not share one frozen producer")

    # Phase B starts only after all Phase-A validation has returned successfully.
    summary_path = args.source_summary.expanduser().resolve()
    truth, summary_sha = load_truth_after_all_seals(summary_path, sorted(all_frames))
    metrics = compute_metrics(all_frames, truth)
    ep71 = audit_ep71(all_frames[71]) if 71 in all_frames else None
    cross_prompt = cross_prompt_collision_audit(all_frames)
    gate_result = gate(metrics, prefix)
    payload = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "postseal_privileged_evaluation_complete",
        "decision": gate_result["decision"],
        "producer_seals_verified_before_summary_access": seals,
        "source_summary": {
            "path": str(summary_path),
            "sha256": summary_sha,
            "fields_used": [
                "episode_records.idx",
                "episode_records.seed",
                "episode_records.steps",
                "episode_records.events",
            ],
            "success_fields_used": False,
        },
        "episode_ids": sorted(all_frames),
        "metrics": metrics,
        "fresh_prefix_checks": {str(key): value for key, value in sorted(prefix.items())},
        "ep71_can_identity_and_carry_audit": ep71,
        "cross_prompt_cream_can_collision_audit": cross_prompt,
        "identity_scope": {
            "per_can_identity_emitted": False,
            "interpretation": (
                "This evaluator can validate two-can count/stage but cannot satisfy a "
                "per-alphabet-versus-tomato identity gate."
            ),
        },
        "gate": gate_result,
        "provenance": {
            "evaluator_source_sha256": sha256_file(Path(__file__).resolve()),
            "producer_source_sha256": next(iter(producer_source_hashes)),
            "git_head": git_head(),
            "argv": list(sys.argv),
        },
    }
    output = args.output.expanduser().resolve()
    require(not output.exists(), f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.rename(temporary, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": sha256_file(output),
                "decision": gate_result["decision"],
                "failed_checks": gate_result["failed_checks"],
                "framewise": metrics["framewise"],
                "increments": metrics["increments"],
                "terminal_specificity": metrics["terminal_specificity"],
                "ep71_audit": ep71,
                "cross_prompt_collision_audit": cross_prompt,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
