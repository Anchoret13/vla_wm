#!/usr/bin/env python3
"""Post-seal development evaluation for SAM3-can + multiview-cream Phi.

All observation-only producer seals are verified before the privileged event
summary is opened.  The evaluator uses only placement event times 1, 3, and 5;
it never reads a success field.  Because episode 12 and the negative cases were
already used to design the cream state machine, this certificate is strictly a
development diagnostic and can never constitute Gate-0 promotion evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "v247_multiview_progress_development_evaluation_v1"
CREAM_RESULT_SCHEMA = "v247_multiview_cream_development_probe_v1"
CREAM_FRAME_SCHEMA = "v247_multiview_cream_development_frame_v1"
CREAM_COMPLETE_SCHEMA = "v247_multiview_cream_development_complete_v1"
SAM3_RESULT_SCHEMA = "v245_sam3_video_progress_probe_v1"
SAM3_FRAME_SCHEMA = "v245_sam3_video_progress_frame_v1"
SAM3_COMPLETE_SCHEMA = "v245_sam3_video_progress_complete_v1"
TASK = "chain3_lr2"
PLACE_EVENT_IDS = {"can": (1, 3), "cream": (5,)}


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
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"{path}: expected object")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cream-run", type=Path, required=True)
    parser.add_argument("--sam3-run", action="append", type=Path, required=True)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def verify_cream_run(root: Path) -> tuple[dict[str, Any], dict[tuple[int, int], dict[str, Any]]]:
    root = root.expanduser().resolve()
    result_path = root / "RESULT.json"
    frames_path = root / "frames.jsonl"
    complete_path = root / "COMPLETE.json"
    producer_path = root / "PRODUCER.py"
    state_path = root / "CREAM_STATE.py"
    for path in (result_path, frames_path, complete_path, producer_path, state_path):
        require(path.is_file(), f"missing cream artifact file: {path}")
    complete = load_json(complete_path)
    require(complete.get("schema") == CREAM_COMPLETE_SCHEMA, "cream COMPLETE schema")
    require(complete.get("status") == "complete", "cream run incomplete")
    expected = {
        "result_sha256": sha256_file(result_path),
        "frames_sha256": sha256_file(frames_path),
        "producer_sha256": sha256_file(producer_path),
        "state_source_sha256": sha256_file(state_path),
    }
    for key, value in expected.items():
        require(complete.get(key) == value, f"cream {key} drift")
    result = load_json(result_path)
    require(result.get("schema") == CREAM_RESULT_SCHEMA, "cream result schema")
    require(result.get("privileged_inputs_read") is False, "cream used privileged input")
    require(result.get("producer_source_sha256") == expected["producer_sha256"], "producer bind")
    require(result.get("state_source_sha256") == expected["state_source_sha256"], "state bind")
    frames = {}
    with frames_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            frame = json.loads(line)
            require(frame.get("schema") == CREAM_FRAME_SCHEMA, "cream frame schema")
            key = (int(frame["episode_id"]), int(frame["t"]))
            require(key not in frames, f"duplicate cream frame {key}")
            frames[key] = frame
    return result, frames


def verify_sam3_runs(
    roots: Sequence[Path], wanted: set[int]
) -> tuple[dict[tuple[int, int], dict[str, Any]], list[dict[str, Any]]]:
    frames = {}
    provenance = []
    for raw_root in roots:
        root = raw_root.expanduser().resolve()
        result_path = root / "result.json"
        frames_path = root / "frames.jsonl"
        complete_path = root / "COMPLETE.json"
        producer_path = root / "producer_source.py"
        for path in (result_path, frames_path, complete_path, producer_path):
            require(path.is_file(), f"missing SAM3 artifact file: {path}")
        complete = load_json(complete_path)
        require(complete.get("schema") == SAM3_COMPLETE_SCHEMA, "SAM3 COMPLETE schema")
        require(complete.get("status") == "complete", "SAM3 run incomplete")
        require(complete.get("result_sha256") == sha256_file(result_path), "SAM3 result drift")
        require(complete.get("frames_sha256") == sha256_file(frames_path), "SAM3 frames drift")
        require(
            complete.get("producer_source_sha256") == sha256_file(producer_path),
            "SAM3 producer drift",
        )
        result = load_json(result_path)
        require(result.get("schema") == SAM3_RESULT_SCHEMA, "SAM3 result schema")
        require(result.get("privileged_inputs_read") is False, "SAM3 privileged input")
        selected = 0
        with frames_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                frame = json.loads(line)
                require(frame.get("schema") == SAM3_FRAME_SCHEMA, "SAM3 frame schema")
                episode_id = int(frame["episode_id"])
                if episode_id not in wanted:
                    continue
                key = (episode_id, int(frame["t"]))
                require(key not in frames, f"duplicate SAM3 frame {key}")
                frames[key] = frame
                selected += 1
        provenance.append(
            {
                "root": str(root),
                "result_sha256": sha256_file(result_path),
                "frames_sha256": sha256_file(frames_path),
                "complete_sha256": sha256_file(complete_path),
                "selected_frames": selected,
            }
        )
    return frames, provenance


def summary_records(path: Path, wanted: set[int]) -> dict[int, dict[str, Any]]:
    summary = load_json(path)
    records = summary.get("episode_records")
    require(isinstance(records, list), "summary has no episode_records")
    selected = {}
    for row in records:
        require(isinstance(row, Mapping), "summary episode row is not an object")
        episode_id = int(row.get("idx", -1))
        if episode_id not in wanted:
            continue
        require(set(row) >= {"idx", "seed", "steps", "events"}, "summary row incomplete")
        events = row["events"]
        require(isinstance(events, Mapping), "summary events missing")
        selected[episode_id] = {
            "idx": episode_id,
            "seed": int(row["seed"]),
            "steps": int(row["steps"]),
            "events": {int(key): int(value) for key, value in events.items()},
        }
    require(set(selected) == wanted, "summary missing requested episodes")
    return selected


def f1_score(targets: Sequence[int], predictions: Sequence[int]) -> float:
    pairs = list(zip(targets, predictions, strict=True))
    tp = sum(int(target == 1 and prediction == 1) for target, prediction in pairs)
    fp = sum(int(target == 0 and prediction == 1) for target, prediction in pairs)
    fn = sum(int(target == 1 and prediction == 0) for target, prediction in pairs)
    if tp == fp == fn == 0:
        return 1.0
    return 2.0 * tp / (2.0 * tp + fp + fn)


def increments(values: Sequence[int], times: Sequence[int]) -> list[int]:
    output = []
    previous = values[0]
    for value, time in zip(values[1:], times[1:], strict=True):
        require(value >= previous, "predicted atom/count is non-monotone")
        output.extend([time] * (value - previous))
        previous = value
    return output


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    require(not output.exists(), f"refusing to overwrite {output}")
    cream_result, cream_frames = verify_cream_run(args.cream_run)
    wanted = {int(value) for value in cream_result["episode_ids"]}
    sam3_frames, sam3_provenance = verify_sam3_runs(args.sam3_run, wanted)
    require(set(sam3_frames) == set(cream_frames), "cream/SAM3 frame surface mismatch")

    # Privileged data is deliberately opened only after every producer seal.
    source_summary = args.source_summary.expanduser().resolve()
    require(source_summary.is_file(), "source summary missing")
    summary_sha = sha256_file(source_summary)
    oracle = summary_records(source_summary, wanted)

    atom_targets = {"can_ge1": [], "can_ge2": [], "cream": []}
    atom_predictions = {name: [] for name in atom_targets}
    scalar_exact = []
    reliable = []
    episode_metrics = {}
    all_timing_errors = []
    total_missing = 0
    total_extra = 0
    terminal_negative_correct = []
    for episode_id in sorted(wanted):
        keys = sorted((key for key in cream_frames if key[0] == episode_id), key=lambda key: key[1])
        times = [key[1] for key in keys]
        can_values = []
        cream_values = []
        oracle_can = []
        oracle_cream = []
        events = oracle[episode_id]["events"]
        for key in keys:
            sam3 = sam3_frames[key]
            cream = cream_frames[key]
            can_count = int(sam3["causal_progress"]["sticky_can_inside_count"])
            cream_value = int(cream["causal_cream_state"]["cream_achieved"])
            target_can = sum(
                int(event_id in events and events[event_id] <= key[1])
                for event_id in PLACE_EVENT_IDS["can"]
            )
            target_cream = int(5 in events and events[5] <= key[1])
            predicted_atoms = {
                "can_ge1": int(can_count >= 1),
                "can_ge2": int(can_count >= 2),
                "cream": cream_value,
            }
            target_atoms = {
                "can_ge1": int(target_can >= 1),
                "can_ge2": int(target_can >= 2),
                "cream": target_cream,
            }
            for name in atom_targets:
                atom_targets[name].append(target_atoms[name])
                atom_predictions[name].append(predicted_atoms[name])
            scalar_exact.append(int(can_count + cream_value == target_can + target_cream))
            reliable.append(int(cream["causal_cream_state"]["observation_reliable"]))
            can_values.append(can_count)
            cream_values.append(cream_value)
            oracle_can.append(target_can)
            oracle_cream.append(target_cream)

        predicted_transitions = {
            "can": increments(can_values, times),
            "cream": increments(cream_values, times),
        }
        oracle_transitions = {
            "can": sorted(
                events[event_id]
                for event_id in PLACE_EVENT_IDS["can"]
                if event_id in events
            ),
            "cream": [events[5]] if 5 in events else [],
        }
        per_episode_errors = []
        missing = 0
        extra = 0
        for channel in ("can", "cream"):
            predicted = predicted_transitions[channel]
            target = oracle_transitions[channel]
            matched = min(len(predicted), len(target))
            errors = [abs(predicted[index] - target[index]) for index in range(matched)]
            per_episode_errors.extend(errors)
            missing += max(0, len(target) - len(predicted))
            extra += max(0, len(predicted) - len(target))
            if not target:
                terminal_negative_correct.append(int(not predicted))
        all_timing_errors.extend(per_episode_errors)
        total_missing += missing
        total_extra += extra
        episode_metrics[str(episode_id)] = {
            "frames": len(keys),
            "oracle_transitions": oracle_transitions,
            "predicted_transitions": predicted_transitions,
            "absolute_timing_errors_steps": per_episode_errors,
            "missing_transitions": missing,
            "extra_transitions": extra,
            "exact_scalar_frames": sum(
                int(pred_can + pred_cream == true_can + true_cream)
                for pred_can, pred_cream, true_can, true_cream in zip(
                    can_values, cream_values, oracle_can, oracle_cream, strict=True
                )
            ),
        }

    atom_accuracy = {
        name: sum(
            int(left == right)
            for left, right in zip(
                atom_targets[name], atom_predictions[name], strict=True
            )
        )
        / len(atom_targets[name])
        for name in atom_targets
    }
    atom_f1 = {
        name: f1_score(atom_targets[name], atom_predictions[name]) for name in atom_targets
    }
    metrics = {
        "frames": len(scalar_exact),
        "exact_scalar_accuracy": sum(scalar_exact) / len(scalar_exact),
        "atom_accuracy": atom_accuracy,
        "atom_f1": atom_f1,
        "macro_f1": sum(atom_f1.values()) / len(atom_f1),
        "evidence_coverage": sum(reliable) / len(reliable),
        "uncertainty_rate": 1.0 - sum(reliable) / len(reliable),
        "missing_transitions": total_missing,
        "extra_transitions": total_extra,
        "absolute_timing_errors_steps": all_timing_errors,
        "median_timing_error_steps": (
            sorted(all_timing_errors)[len(all_timing_errors) // 2]
            if all_timing_errors
            else None
        ),
        "max_timing_error_steps": max(all_timing_errors, default=None),
        "never_achieved_terminal_specificity": (
            sum(terminal_negative_correct) / len(terminal_negative_correct)
            if terminal_negative_correct
            else None
        ),
    }
    payload = {
        "schema": SCHEMA,
        "status": "development_evaluation_complete",
        "decision": "STOP_FORMAL_PROMOTION_DEVELOPMENT_ONLY",
        "reason": "all four episodes, including ep12, informed development",
        "episode_ids": sorted(wanted),
        "producer_seals_verified_before_summary_access": True,
        "success_fields_used": False,
        "source_summary": {
            "path": str(source_summary),
            "sha256": summary_sha,
            "fields_used": ["episode_records.idx", "seed", "steps", "events"],
        },
        "cream_run": {
            "path": str(args.cream_run.expanduser().resolve()),
            "result_sha256": sha256_file(args.cream_run.expanduser().resolve() / "RESULT.json"),
        },
        "sam3_runs": sam3_provenance,
        "episodes": episode_metrics,
        "metrics": metrics,
        "evaluator_source_sha256": sha256_file(Path(__file__).resolve()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(json.dumps({"output": str(output), "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
