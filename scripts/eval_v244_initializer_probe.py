#!/usr/bin/env python3
"""Post-seal privileged evaluation for an RGB-only V244 initializer probe.

The producer artifact must already be complete and hash-sealed.  Only after
verifying that boundary does this evaluator open the source summary, and it uses
only ``idx``, ``seed``, ``steps``, and sticky first-achievement ``events``.
Episode success fields are ignored.  Two policies are evaluated:

* ``raw``: uncertain labels are excluded from certain-only classification and
  skipped by transition detection, matching the V243 evaluator.
* ``causal_last_reliable_carry``: each object starts outside; a certain label
  immediately updates state, while uncertain holds the previous state.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(REPO / "scripts"))
import eval_v243_sam2_teacher_gate as gate_support  # noqa: E402


SCHEMA = "v244_t0_initializer_postseal_evaluation_v1"
PROBE_SCHEMA = "v244_t0_initializer_probe_v1"
COMPLETE_SCHEMA = "v244_t0_initializer_probe_complete_v1"
LABEL_SCHEMA = "v244_t0_initializer_label_v1"
EXPECTED_SUMMARY_SHA256 = "a4dc0d5f73328247d11592ad08d55031023a2bb00bbaa3c7436e0a4a9387ef58"
OBJECTS = ("alphabet_soup_1", "tomato_sauce_1", "cream_cheese_1")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"{path}: expected object")
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            require(bool(line.strip()), f"{path}:{line_number}: blank row")
            row = json.loads(line)
            require(isinstance(row, dict), f"{path}:{line_number}: expected object")
            rows.append(row)
    return rows


def git_head() -> str | None:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def validate_sealed_probe(
    run_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    complete_path = run_dir / "COMPLETE.json"
    result_path = run_dir / "result.json"
    labels_path = run_dir / "labels.jsonl"
    require(all(path.is_file() for path in (complete_path, result_path, labels_path)),
            "probe artifact is incomplete")
    complete = load_json(complete_path)
    require(complete.get("schema") == COMPLETE_SCHEMA, "complete schema mismatch")
    require(complete.get("status") == "complete", "probe is not complete")
    require(complete.get("result_sha256") == sha256_file(result_path),
            "sealed result SHA mismatch")
    require(complete.get("labels_sha256") == sha256_file(labels_path),
            "sealed labels SHA mismatch")
    require(complete.get("privileged_inputs_read") is False,
            "producer did not certify RGB-only generation")
    result = load_json(result_path)
    require(result.get("schema") == PROBE_SCHEMA, "probe result schema mismatch")
    require(result.get("privileged_inputs_read") is False,
            "probe result crossed privileged boundary")
    require(result.get("status") == "rgb_only_candidate_complete_not_privileged_evaluated",
            "probe status mismatch")
    require(result.get("full_track") is True, "posthoc metrics require --full-track")
    labels = load_jsonl(labels_path)
    require(len(labels) == result["outputs"]["label_rows"], "label row count mismatch")
    require(result["outputs"]["labels_sha256"] == sha256_file(labels_path),
            "result labels SHA mismatch")
    seen: set[str] = set()
    for row in labels:
        require(row.get("schema") == LABEL_SCHEMA, "label schema mismatch")
        frame_id = str(row.get("frame_id"))
        require(frame_id not in seen, f"duplicate label {frame_id}")
        seen.add(frame_id)
        require(set(row.get("objects", {})) == set(OBJECTS), "object set mismatch")
        for name in OBJECTS:
            require(row["objects"][name]["label"] in {"inside", "outside", "uncertain"},
                    "invalid label")
    seal = {
        "complete_sha256": sha256_file(complete_path),
        "result_sha256": sha256_file(result_path),
        "labels_sha256": sha256_file(labels_path),
        "producer_code_sha256": str(complete["code_sha256"]),
    }
    return result, labels, seal


def load_records_after_seal(
    summary_path: Path, episode_ids: list[int]
) -> dict[int, dict[str, Any]]:
    # This function is called only after validate_sealed_probe has returned.
    require(sha256_file(summary_path) == EXPECTED_SUMMARY_SHA256,
            "source summary SHA mismatch")
    payload = load_json(summary_path)
    require(payload.get("task") == "chain3_lr2", "summary task mismatch")
    require(int(payload.get("c", -1)) == 10, "summary chunk mismatch")
    records = {}
    for raw in payload.get("episode_records", []):
        episode_id = int(raw["idx"])
        if episode_id not in episode_ids:
            continue
        require(int(raw["seed"]) == 8700 + episode_id, "summary seed mismatch")
        records[episode_id] = {
            "seed": int(raw["seed"]),
            "steps": int(raw["steps"]),
            "events": {int(key): int(value) for key, value in raw["events"].items()},
        }
    require(set(records) == set(episode_ids), "summary does not cover probe episodes")
    return records


def frames_for_metrics(labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frames = []
    for row in labels:
        frames.append(
            {
                "frame_id": row["frame_id"],
                "episode_id": int(row["episode_id"]),
                "env_seed": int(row["env_seed"]),
                "t": int(row["t"]),
                "label_row": row,
            }
        )
    frames.sort(key=lambda frame: (frame["episode_id"], frame["t"]))
    return frames


def causal_carry(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = copy.deepcopy(frames)
    state: dict[tuple[int, str], str] = {}
    for frame in output:
        episode_id = frame["episode_id"]
        for name in OBJECTS:
            key = (episode_id, name)
            previous = state.get(key, "outside")
            raw = frame["label_row"]["objects"][name]["label"]
            current = previous if raw == "uncertain" else raw
            state[key] = current
            frame["label_row"]["objects"][name]["label"] = current
    return output


def summarize(metrics: dict[str, Any]) -> dict[str, Any]:
    classification = metrics["classification"]
    transitions = metrics["transitions"]
    terminal = metrics["terminal_specificity"]
    return {
        "uncertain_fraction": classification["uncertain_fraction"],
        "certain_only_accuracy": classification["framewise_place_accuracy_certain_only"],
        "effective_accuracy_uncertain_counted_incorrect": classification[
            "effective_accuracy_uncertain_counted_incorrect"
        ],
        "per_object_inside_positive_f1": classification[
            "per_object_inside_positive_f1"
        ],
        "macro_inside_positive_f1": classification["macro_inside_positive_f1"],
        "per_object_confusion": classification["per_object_confusion"],
        "observable_positive_transitions": transitions["observable_positive_events"],
        "matched_observable_positive_transitions": transitions[
            "matched_observable_positive_events"
        ],
        "missing_observable_transitions": transitions["missing_observable_transitions"],
        "transition_coverage": (
            transitions["matched_observable_positive_events"]
            / transitions["observable_positive_events"]
            if transitions["observable_positive_events"]
            else None
        ),
        "transition_median_absolute_error_steps": transitions[
            "median_absolute_error_steps"
        ],
        "right_censored_positive_events": transitions[
            "right_censored_positive_events"
        ],
        "terminal_true_negatives": terminal["true_negatives"],
        "terminal_denominator": terminal["denominator"],
        "terminal_specificity": terminal["specificity"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-run", type=Path, required=True)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.probe_run.expanduser().resolve()
    require(run_dir.is_dir(), f"missing probe run {run_dir}")
    result, labels, seal = validate_sealed_probe(run_dir)

    # Privileged access starts only here, after all producer hashes were sealed.
    episode_ids = [int(value) for value in result["episode_ids"]]
    summary_path = args.source_summary.expanduser().resolve()
    records = load_records_after_seal(summary_path, episode_ids)
    raw_frames = frames_for_metrics(labels)
    carry_frames = causal_carry(raw_frames)
    raw_metrics = gate_support.compute_metrics(raw_frames, records, episode_ids)
    carry_metrics = gate_support.compute_metrics(carry_frames, records, episode_ids)

    payload = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "postseal_privileged_evaluation_complete",
        "probe_run": str(run_dir),
        "probe_seal_verified_before_summary_access": seal,
        "source_summary": {
            "path": str(summary_path),
            "sha256": sha256_file(summary_path),
            "fields_used": ["episode_records.idx", "seed", "steps", "events"],
            "episode_success_fields_used": False,
        },
        "episode_ids": episode_ids,
        "raw": {"summary": summarize(raw_metrics), "full_metrics": raw_metrics},
        "causal_last_reliable_carry": {
            "rule": (
                "state starts outside independently per episode/object; each certain "
                "inside/outside updates immediately; uncertain holds previous state"
            ),
            "summary": summarize(carry_metrics),
            "full_metrics": carry_metrics,
        },
        "provenance": {
            "evaluator_code_sha256": sha256_file(Path(__file__).resolve()),
            "gate_metric_support_code_sha256": sha256_file(
                Path(gate_support.__file__).resolve()
            ),
            "git_head": git_head(),
            "argv": list(sys.argv),
        },
    }
    output = args.output.expanduser().resolve()
    require(not output.exists(), f"output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.rename(temporary, output)
    print(json.dumps({
        "output": str(output),
        "sha256": sha256_file(output),
        "raw": payload["raw"]["summary"],
        "carry": payload["causal_last_reliable_carry"]["summary"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
