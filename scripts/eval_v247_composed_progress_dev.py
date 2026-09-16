#!/usr/bin/env python3
"""Post-seal evaluator for the composed v247 development progress stream."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "v247_composed_progress_development_evaluation_v1"
RUN_RESULT_SCHEMA = "v247_multiview_progress_development_composition_v1"
RUN_FRAME_SCHEMA = "v247_multiview_progress_development_frame_v1"
RUN_COMPLETE_SCHEMA = "v247_multiview_progress_development_complete_v1"
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
    parser.add_argument("--progress-run", type=Path, required=True)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def verify_run(
    raw_root: Path,
) -> tuple[dict[str, Any], dict[tuple[int, int], dict[str, Any]], dict[str, Any]]:
    root = raw_root.expanduser().resolve()
    paths = {
        "result": root / "RESULT.json",
        "frames": root / "frames.jsonl",
        "complete": root / "COMPLETE.json",
        "producer": root / "PRODUCER.py",
        "can_state": root / "CAN_STATE.py",
    }
    for path in paths.values():
        require(path.is_file(), f"missing progress artifact: {path}")
    complete = load_json(paths["complete"])
    require(complete.get("schema") == RUN_COMPLETE_SCHEMA, "COMPLETE schema")
    require(complete.get("status") == "complete", "progress artifact incomplete")
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    bindings = {
        "result_sha256": hashes["result"],
        "frames_sha256": hashes["frames"],
        "producer_sha256": hashes["producer"],
        "can_state_source_sha256": hashes["can_state"],
    }
    for key, expected in bindings.items():
        require(complete.get(key) == expected, f"{key} drift")
    result = load_json(paths["result"])
    require(result.get("schema") == RUN_RESULT_SCHEMA, "result schema")
    require(result.get("privileged_inputs_read") is False, "producer used privileged input")
    require(result.get("producer_source_sha256") == hashes["producer"], "producer bind")
    require(
        result.get("can_state_source_sha256") == hashes["can_state"],
        "can-state bind",
    )
    frames = {}
    with paths["frames"].open("r", encoding="utf-8") as handle:
        for line in handle:
            frame = json.loads(line)
            require(frame.get("schema") == RUN_FRAME_SCHEMA, "frame schema")
            key = (int(frame["episode_id"]), int(frame["t"]))
            require(key not in frames, f"duplicate frame {key}")
            frames[key] = frame
    provenance = {
        "root": str(root),
        **{f"{name}_sha256": value for name, value in hashes.items()},
    }
    return result, frames, provenance


def summary_records(path: Path, wanted: set[int]) -> dict[int, dict[str, Any]]:
    rows = load_json(path).get("episode_records")
    require(isinstance(rows, list), "summary has no episode_records")
    selected = {}
    for row in rows:
        require(isinstance(row, Mapping), "summary row is not an object")
        episode_id = int(row.get("idx", -1))
        if episode_id not in wanted:
            continue
        events = row.get("events")
        require(isinstance(events, Mapping), "summary events missing")
        selected[episode_id] = {
            "seed": int(row["seed"]),
            "steps": int(row["steps"]),
            "events": {int(key): int(value) for key, value in events.items()},
        }
    require(set(selected) == wanted, "summary missing episode")
    return selected


def f1_score(targets: Sequence[int], predictions: Sequence[int]) -> float:
    pairs = list(zip(targets, predictions, strict=True))
    tp = sum(int(target == prediction == 1) for target, prediction in pairs)
    fp = sum(int(target == 0 and prediction == 1) for target, prediction in pairs)
    fn = sum(int(target == 1 and prediction == 0) for target, prediction in pairs)
    if tp == fp == fn == 0:
        return 1.0
    return 2.0 * tp / (2.0 * tp + fp + fn)


def median(values: Sequence[int]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    require(not output.exists(), f"refusing to overwrite {output}")
    result, frames, run_provenance = verify_run(args.progress_run)
    episode_ids = [int(value) for value in result["episode_ids"]]
    wanted = set(episode_ids)
    require(len(wanted) == len(episode_ids), "duplicate episode ID")

    # The privileged summary is opened only after the complete producer chain.
    summary_path = args.source_summary.expanduser().resolve()
    require(summary_path.is_file(), "source summary missing")
    oracle = summary_records(summary_path, wanted)

    targets = {"can_ge1": [], "can_ge2": [], "cream": []}
    predictions = {name: [] for name in targets}
    scalar_exact = []
    reliability = []
    timing_errors = []
    missing_total = 0
    extra_total = 0
    negative_terminals = []
    episodes = {}
    for episode_id in episode_ids:
        keys = sorted((key for key in frames if key[0] == episode_id), key=lambda key: key[1])
        events = oracle[episode_id]["events"]
        predicted_transitions = {"can": [], "cream": []}
        previous_can = 0
        previous_cream = 0
        exact_frames = 0
        for key in keys:
            frame = frames[key]
            atoms = frame["ordinal_atoms"]
            predicted_can = int(atoms["can_ge1"]) + int(atoms["can_ge2"])
            predicted_cream = int(atoms["cream"])
            require(predicted_can >= previous_can, "can count is non-monotone")
            require(predicted_cream >= previous_cream, "cream state is non-monotone")
            predicted_transitions["can"].extend(
                [key[1]] * (predicted_can - previous_can)
            )
            predicted_transitions["cream"].extend(
                [key[1]] * (predicted_cream - previous_cream)
            )
            previous_can = predicted_can
            previous_cream = predicted_cream
            true_can = sum(
                int(event_id in events and events[event_id] <= key[1])
                for event_id in PLACE_EVENT_IDS["can"]
            )
            true_cream = int(5 in events and events[5] <= key[1])
            target_atoms = {
                "can_ge1": int(true_can >= 1),
                "can_ge2": int(true_can >= 2),
                "cream": true_cream,
            }
            for name in targets:
                targets[name].append(target_atoms[name])
                predictions[name].append(int(atoms[name]))
            exact = int(predicted_can + predicted_cream == true_can + true_cream)
            scalar_exact.append(exact)
            exact_frames += exact
            reliability.append(int(frame["observation_reliable"]))
        oracle_transitions = {
            "can": sorted(
                events[event_id]
                for event_id in PLACE_EVENT_IDS["can"]
                if event_id in events
            ),
            "cream": [events[5]] if 5 in events else [],
        }
        episode_errors = []
        missing = 0
        extra = 0
        for channel in ("can", "cream"):
            predicted = predicted_transitions[channel]
            expected = oracle_transitions[channel]
            matched = min(len(predicted), len(expected))
            episode_errors.extend(
                abs(predicted[index] - expected[index]) for index in range(matched)
            )
            missing += max(0, len(expected) - len(predicted))
            extra += max(0, len(predicted) - len(expected))
            if not expected:
                negative_terminals.append(int(not predicted))
        timing_errors.extend(episode_errors)
        missing_total += missing
        extra_total += extra
        episodes[str(episode_id)] = {
            "frames": len(keys),
            "oracle_transitions": oracle_transitions,
            "predicted_transitions": predicted_transitions,
            "absolute_timing_errors_steps": episode_errors,
            "missing_transitions": missing,
            "extra_transitions": extra,
            "exact_scalar_frames": exact_frames,
        }

    atom_accuracy = {
        name: sum(
            int(target == prediction)
            for target, prediction in zip(targets[name], predictions[name], strict=True)
        )
        / len(targets[name])
        for name in targets
    }
    atom_f1 = {
        name: f1_score(targets[name], predictions[name]) for name in targets
    }
    coverage = sum(reliability) / len(reliability)
    metrics = {
        "frames": len(scalar_exact),
        "exact_scalar_accuracy": sum(scalar_exact) / len(scalar_exact),
        "atom_accuracy": atom_accuracy,
        "atom_f1": atom_f1,
        "macro_f1": sum(atom_f1.values()) / len(atom_f1),
        "evidence_coverage": coverage,
        "uncertainty_rate": 1.0 - coverage,
        "missing_transitions": missing_total,
        "extra_transitions": extra_total,
        "absolute_timing_errors_steps": timing_errors,
        "median_timing_error_steps": median(timing_errors),
        "max_timing_error_steps": max(timing_errors, default=None),
        "never_achieved_terminal_specificity": (
            sum(negative_terminals) / len(negative_terminals)
            if negative_terminals
            else None
        ),
    }
    payload = {
        "schema": SCHEMA,
        "status": "development_evaluation_complete",
        "decision": "STOP_FORMAL_PROMOTION_DEVELOPMENT_ONLY",
        "reason": "the same episodes informed both rules and evaluation",
        "episode_ids": episode_ids,
        "producer_seals_verified_before_summary_access": True,
        "success_values_accessed": False,
        "progress_run": run_provenance,
        "source_summary": {
            "path": str(summary_path),
            "sha256": sha256_file(summary_path),
            "fields_used": ["episode_records.idx", "seed", "steps", "events"],
        },
        "episodes": episodes,
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
