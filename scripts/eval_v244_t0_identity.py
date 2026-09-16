#!/usr/bin/env python3
"""Post-seal privileged identity audit for a V244 RGB-only initializer.

The producer must be complete and hash-sealed before this program imports the
simulator.  For each episode, the program recreates only the reset state,
projects object body centers into the policy-oriented agentview image, and
compares those evaluation-only centers with the already-sealed prompt and SAM2
mask diagnostics.  It never reads rollout events, milestones, success, or a
source summary, and it never writes coordinates back into the producer.

Projected body centers are an identity/QC diagnostic, not pixel-perfect
segmentation truth.  This evaluator therefore reports distances, nearest-object
margins, and bounding-box containment without inventing a post-hoc pass
threshold.  A later formal registration must freeze any threshold before an
unseen panel is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO = Path(__file__).resolve().parent.parent
# Keep imports/replay hermetic and all caches in task-specific writable paths.
# These settings do not expose simulator state and are established before any
# LIBERO/robosuite module is imported inside ``main``.
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/v244_identity_numba")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/v244_identity_mpl")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

SCHEMA = "v244_t0_identity_postseal_evaluation_v1"
PROBE_SCHEMA = "v244_t0_initializer_probe_v1"
COMPLETE_SCHEMA = "v244_t0_initializer_probe_complete_v1"
LABEL_SCHEMA = "v244_t0_initializer_label_v1"
TASK = "chain3_lr2"
CAMERA = "agentview"
IMAGE_SIZE = 360
OBJECTS = (
    "basket_1",
    "tomato_sauce_1",
    "alphabet_soup_1",
    "cream_cheese_1",
)


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
    require(isinstance(payload, dict), f"{path}: expected a JSON object")
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            require(bool(line.strip()), f"{path}:{line_number}: blank row")
            row = json.loads(line)
            require(isinstance(row, dict), f"{path}:{line_number}: expected object")
            rows.append(row)
    return rows


def finite_xy(value: Any, where: str) -> np.ndarray:
    require(
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, (int, float)) and math.isfinite(item) for item in value),
        f"{where}: expected finite [x,y]",
    )
    result = np.asarray(value, dtype=np.float64)
    require(
        bool(((0 <= result) & (result < IMAGE_SIZE)).all()),
        f"{where}: point outside image",
    )
    return result


def mask_diag(row: dict[str, Any], name: str) -> dict[str, Any]:
    if name == "basket_1":
        diag = row.get("basket_diagnostics")
    else:
        diagnostics = row.get("object_diagnostics")
        diag = diagnostics.get(name) if isinstance(diagnostics, dict) else None
    require(isinstance(diag, dict), f"missing t0 diagnostic for {name}")
    return diag


def validate_sealed_probe(
    run_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    complete_path = run_dir / "COMPLETE.json"
    result_path = run_dir / "result.json"
    labels_path = run_dir / "labels.jsonl"
    require(
        all(path.is_file() for path in (complete_path, result_path, labels_path)),
        "probe artifact is incomplete",
    )
    complete = load_json(complete_path)
    require(complete.get("schema") == COMPLETE_SCHEMA, "complete schema mismatch")
    require(complete.get("status") == "complete", "probe is not complete")
    require(complete.get("privileged_inputs_read") is False, "producer crossed boundary")
    require(
        complete.get("result_sha256") == sha256_file(result_path),
        "sealed result SHA mismatch",
    )
    require(
        complete.get("labels_sha256") == sha256_file(labels_path),
        "sealed labels SHA mismatch",
    )

    result = load_json(result_path)
    require(result.get("schema") == PROBE_SCHEMA, "probe result schema mismatch")
    require(result.get("privileged_inputs_read") is False, "probe crossed boundary")
    require(
        result.get("status") == "rgb_only_candidate_complete_not_privileged_evaluated",
        "probe status mismatch",
    )
    require(
        result.get("mode")
        in {"fixed_points", "dinov2_points", "sift_points", "sift_cluster_points"},
        "unsupported initializer mode",
    )

    labels = load_jsonl(labels_path)
    outputs = result.get("outputs")
    require(isinstance(outputs, dict), "probe outputs missing")
    require(outputs.get("label_rows") == len(labels), "label count mismatch")
    require(outputs.get("labels_sha256") == sha256_file(labels_path),
            "result labels SHA mismatch")

    expected_episodes = result.get("episode_ids")
    require(
        isinstance(expected_episodes, list)
        and expected_episodes
        and all(isinstance(value, int) for value in expected_episodes)
        and expected_episodes == sorted(set(expected_episodes)),
        "invalid episode_ids",
    )
    t0_rows: dict[int, dict[str, Any]] = {}
    for index, row in enumerate(labels):
        where = f"labels row {index}"
        require(row.get("schema") == LABEL_SCHEMA, f"{where}: schema mismatch")
        require(row.get("task") == TASK, f"{where}: task mismatch")
        require(row.get("camera") == CAMERA, f"{where}: camera mismatch")
        episode_id = row.get("episode_id")
        env_seed = row.get("env_seed")
        t_value = row.get("t")
        require(isinstance(episode_id, int), f"{where}: invalid episode_id")
        require(isinstance(env_seed, int), f"{where}: invalid env_seed")
        require(isinstance(t_value, int) and t_value >= 0, f"{where}: invalid t")
        require(row.get("prompt_frame_index") == 0, f"{where}: prompt frame mismatch")
        require(row.get("post_t0_prompt_count") == 0, f"{where}: post-t0 prompt")
        prompts = row.get("prompt_points_xy")
        require(isinstance(prompts, dict) and set(prompts) == set(OBJECTS),
                f"{where}: prompt object set mismatch")
        if t_value == 0:
            require(episode_id not in t0_rows, f"duplicate t0 row for ep{episode_id}")
            t0_rows[episode_id] = row
    require(set(t0_rows) == set(expected_episodes), "t0 rows do not cover episode_ids")

    return result, [t0_rows[value] for value in expected_episodes], {
        "complete_sha256": sha256_file(complete_path),
        "result_sha256": sha256_file(result_path),
        "labels_sha256": sha256_file(labels_path),
        "producer_code_sha256": str(complete.get("code_sha256")),
    }


def percentile(values: Iterable[float], q: float) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.percentile(array, q)) if len(array) else None


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def distribution(field: str) -> dict[str, float | None]:
        values = [float(row[field]) for row in rows]
        return {
            "mean": float(np.mean(values)) if values else None,
            "p50": percentile(values, 50),
            "p90": percentile(values, 90),
            "p95": percentile(values, 95),
            "max": max(values) if values else None,
        }

    count = len(rows)
    return {
        "instances": count,
        "prompt_error_px": distribution("prompt_error_px"),
        "mask_centroid_error_px": distribution("mask_centroid_error_px"),
        "prompt_identity_margin_px": distribution("prompt_identity_margin_px"),
        "mask_centroid_identity_margin_px": distribution(
            "mask_centroid_identity_margin_px"
        ),
        "projected_center_inside_mask_bbox_fraction": (
            sum(bool(row["projected_center_inside_mask_bbox"]) for row in rows) / count
            if count
            else None
        ),
        "prompt_nearest_identity_fraction": (
            sum(bool(row["prompt_nearest_identity_correct"]) for row in rows) / count
            if count
            else None
        ),
        "mask_centroid_nearest_identity_fraction": (
            sum(bool(row["mask_centroid_nearest_identity_correct"]) for row in rows)
            / count
            if count
            else None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.probe_run.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    require(run_dir.is_dir(), f"missing probe run {run_dir}")
    require(not output_path.exists(), f"output exists: {output_path}")

    # The complete producer hash chain is checked before these privileged
    # simulator imports and before any environment is constructed.
    result, t0_rows, producer_seal = validate_sealed_probe(run_dir)
    sys.path.insert(0, str(REPO))
    from lcwm.probe_data import body_positions, discover_object_bodies
    from lcwm.seg_masks import project_points
    from lcwm.v080_bench import make_v080_env

    env = make_v080_env(TASK)
    evaluations: list[dict[str, Any]] = []
    try:
        for row in t0_rows:
            episode_id = int(row["episode_id"])
            env_seed = int(row["env_seed"])
            env.reset(seed=env_seed)
            bodies = discover_object_bodies(env)
            require(set(OBJECTS) <= set(bodies), f"ep{episode_id}: object body missing")
            positions = body_positions(env, [bodies[name] for name in OBJECTS])
            row_col = project_points(
                env, positions, camera=CAMERA, h=IMAGE_SIZE, w=IMAGE_SIZE
            )
            true_xy = {
                name: np.asarray([row_col[index, 1], row_col[index, 0]])
                for index, name in enumerate(OBJECTS)
            }
            require(
                all(bool(np.isfinite(point).all()) for point in true_xy.values()),
                f"ep{episode_id}: nonfinite projected center",
            )
            for name in OBJECTS:
                diag = mask_diag(row, name)
                prompt = finite_xy(row["prompt_points_xy"][name], f"ep{episode_id}:{name}:prompt")
                centroid = finite_xy(diag.get("mask_centroid_xy"),
                                     f"ep{episode_id}:{name}:centroid")
                bbox = diag.get("mask_bbox_xyxy")
                require(
                    isinstance(bbox, list)
                    and len(bbox) == 4
                    and all(isinstance(value, int) for value in bbox),
                    f"ep{episode_id}:{name}: invalid bbox",
                )
                x0, y0, x1, y1 = bbox
                require(0 <= x0 < x1 <= IMAGE_SIZE and 0 <= y0 < y1 <= IMAGE_SIZE,
                        f"ep{episode_id}:{name}: bbox outside image")

                prompt_distances = {
                    other: float(np.linalg.norm(prompt - point))
                    for other, point in true_xy.items()
                }
                centroid_distances = {
                    other: float(np.linalg.norm(centroid - point))
                    for other, point in true_xy.items()
                }
                prompt_wrong = min(
                    value for other, value in prompt_distances.items() if other != name
                )
                centroid_wrong = min(
                    value for other, value in centroid_distances.items() if other != name
                )
                projected = true_xy[name]
                evaluations.append(
                    {
                        "episode_id": episode_id,
                        "env_seed": env_seed,
                        "object": name,
                        "projected_center_xy": projected.tolist(),
                        "sealed_prompt_xy": prompt.tolist(),
                        "sealed_mask_centroid_xy": centroid.tolist(),
                        "sealed_mask_bbox_xyxy": bbox,
                        "prompt_error_px": prompt_distances[name],
                        "mask_centroid_error_px": centroid_distances[name],
                        "prompt_identity_margin_px": prompt_wrong - prompt_distances[name],
                        "mask_centroid_identity_margin_px": (
                            centroid_wrong - centroid_distances[name]
                        ),
                        "prompt_nearest_identity_correct": (
                            min(prompt_distances, key=prompt_distances.get) == name
                        ),
                        "mask_centroid_nearest_identity_correct": (
                            min(centroid_distances, key=centroid_distances.get) == name
                        ),
                        "projected_center_inside_mask_bbox": bool(
                            x0 <= projected[0] < x1 and y0 <= projected[1] < y1
                        ),
                    }
                )
    finally:
        env.close()

    by_object = {
        name: aggregate([row for row in evaluations if row["object"] == name])
        for name in OBJECTS
    }
    code_path = Path(__file__).resolve()
    support_paths = [
        REPO / "lcwm/probe_data.py",
        REPO / "lcwm/seg_masks.py",
        REPO / "lcwm/v080_bench.py",
    ]
    payload = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "postseal_privileged_identity_evaluation_complete",
        "claim_scope": (
            "evaluation-only reset-state object identity audit; projected body centers "
            "are not segmentation ground truth and no pass threshold is inferred"
        ),
        "privileged_inputs_read": True,
        "privileged_fields_used": [
            "reset simulator object body positions",
            "agentview camera calibration",
        ],
        "forbidden_rollout_fields_read": [],
        "probe_run": str(run_dir),
        "probe_mode": result["mode"],
        "probe_seal_verified_before_simulator_import": producer_seal,
        "task": TASK,
        "camera": CAMERA,
        "image_coordinate_convention": "policy-oriented agentview (x,y)",
        "episode_ids": result["episode_ids"],
        "thresholds_applied": None,
        "aggregate": aggregate(evaluations),
        "per_object": by_object,
        "rows": evaluations,
        "provenance": {
            "argv": sys.argv,
            "git_head": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO,
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip(),
            "evaluator": {"path": str(code_path), "sha256": sha256_file(code_path)},
            "support": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in support_paths
            ],
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, output_path)
    print(json.dumps({"output": str(output_path), "aggregate": payload["aggregate"]},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
