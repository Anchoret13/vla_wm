#!/usr/bin/env python
"""Audit an LC-Flow adapter against trivial held-out effect baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.branch_data import (  # noqa: E402
    MANIFEST,
    BranchGroupDataset,
)
from lcwm.lc_flow import load_lc_adapter, outcome_loss  # noqa: E402
from lcwm.lc_flow_train import (  # noqa: E402
    branch_outcome_predictions,
    outcome_effect_metrics,
    to_device_tensors,
    unroll_lc_history,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluator_source_sha256() -> str:
    digest = hashlib.sha256()
    for relative in (
        "scripts/eval_lc_flow_outcomes.py",
        "lcwm/lc_flow_train.py",
        "lcwm/lc_flow.py",
        "lcwm/branch_data.py",
    ):
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update((REPO_ROOT / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def safe_ratio(
    numerator: float,
    denominator: float,
    *,
    resolved: bool,
) -> float | None:
    return numerator / denominator if resolved and denominator > 0 else None


def average_metrics(records: list[dict]) -> dict[str, float]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for record in records:
        for key, value in record.items():
            if (
                value is None
                or isinstance(value, bool)
                or key.endswith(
                    (
                        "_over_zero",
                        "_over_group_mean",
                        "_matched_over_permutation",
                        "_cosine_valid",
                        "_permutation_valid",
                        "_resolved_effect",
                    )
                )
            ):
                continue
            totals[key] = totals.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return {key: value / counts[key] for key, value in totals.items()}


def add_ratios(
    metrics: dict[str, float],
    resolved_counts: dict[str, int],
    *,
    channels: tuple[str, ...] = ("d_q", "d_obj", "d_prog"),
) -> dict[str, float | None]:
    output: dict[str, float | None] = dict(metrics)
    for key in channels:
        resolved = resolved_counts.get(key, 0) > 0
        output[f"{key}_raw_model_over_zero"] = safe_ratio(
            metrics[f"{key}_raw_model_mse"],
            metrics[f"{key}_raw_zero_mse"],
            resolved=resolved,
        )
        output[f"{key}_centered_model_over_group_mean"] = safe_ratio(
            metrics[f"{key}_centered_model_mse"],
            metrics[f"{key}_group_mean_mse"],
            resolved=resolved,
        )
        output[f"{key}_matched_over_permutation"] = safe_ratio(
            metrics[f"{key}_centered_model_mse"],
            metrics[f"{key}_centered_permutation_mse"],
            resolved=resolved,
        )
    return output


@torch.no_grad()
def main() -> None:
    args = parse_args()
    dataset = BranchGroupDataset(args.data, args.split)
    if not len(dataset):
        raise RuntimeError(f"{args.split!r} split has no admitted groups")
    model, metadata = load_lc_adapter(
        args.adapter, device=args.device
    )
    model.eval()

    per_group = []
    source_episodes: set[str] = set()
    active_predicate_targets: set[tuple[bool, ...]] = set()
    coverage = {
        "groups": 0,
        "branches": 0,
        "predicate_flip_branches": 0,
        "labelled_continuations": 0,
        "successful_continuations": 0,
        "positive_flow_targets": 0,
    }
    for index, path in enumerate(dataset.paths):
        group = dataset[index]
        source_episodes.add(group["source_trajectory_id"])
        z_t = unroll_lc_history(
            model,
            group,
            device=args.device,
        )
        predictions = branch_outcome_predictions(
            model,
            z_t,
            group,
            device=args.device,
        )
        targets = to_device_tensors(group["targets"], args.device)
        loss, parts = outcome_loss(predictions, targets)
        effect = outcome_effect_metrics(predictions, targets)
        effect_values = {
            key: float(value) for key, value in effect.items()
        }
        for key in ("d_q", "d_obj", "d_prog"):
            if not bool(effect_values[f"{key}_centered_cosine_valid"]):
                effect_values[f"{key}_centered_cosine"] = None
            if not bool(effect_values[f"{key}_permutation_valid"]):
                effect_values[f"{key}_matched_over_permutation"] = None
        metrics = {
            "outcome_loss": float(loss),
            **{
                f"outcome_{key}": float(value)
                for key, value in parts.items()
            },
            **effect_values,
        }

        restore = group["restore_qc"]
        min_ratio = float(restore.get("min_effect_noise_ratio", 10.0))
        q_resolution = max(
            float(restore.get("q_tolerance", 0.0)),
            min_ratio * float(restore.get("replay_q_dispersion", 0.0)),
        )
        object_resolution = max(
            float(restore.get("object_tolerance", 0.0)),
            min_ratio
            * float(restore.get("replay_object_dispersion", 0.0)),
        )
        channel_resolved = {
            "d_q": bool(restore.get("q_resolved", False))
            and float(restore.get("action_q_dispersion", 0.0))
            > q_resolution,
            "d_obj": bool(restore.get("object_resolved", False))
            and float(restore.get("action_object_dispersion", 0.0))
            > object_resolution,
            "d_prog": float(
                group["dense_goal_progress"].max()
                - group["dense_goal_progress"].min()
            )
            > float(group["dense_progress_margin"]),
        }
        for key in ("d_q", "d_obj", "d_prog"):
            metrics[f"{key}_resolved_effect"] = channel_resolved[key]
            metrics[f"{key}_raw_model_over_zero"] = safe_ratio(
                metrics[f"{key}_raw_model_mse"],
                metrics[f"{key}_raw_zero_mse"],
                resolved=channel_resolved[key],
            )
            metrics[f"{key}_centered_model_over_group_mean"] = safe_ratio(
                metrics[f"{key}_centered_model_mse"],
                metrics[f"{key}_group_mean_mse"],
                resolved=channel_resolved[key],
            )
            metrics[f"{key}_matched_over_permutation"] = safe_ratio(
                metrics[f"{key}_centered_model_mse"],
                metrics[f"{key}_centered_permutation_mse"],
                resolved=channel_resolved[key],
            )

        atom_mask = group["atom_mask"].bool()
        for bits in group["bits_after"][:, atom_mask]:
            active_predicate_targets.add(
                tuple(bool(value) for value in bits.tolist())
            )
        predicate_flips = (
            group["bits_after"][:, atom_mask]
            != group["bits_before"][atom_mask][None]
        ).any(dim=1)
        labelled = group["continuation_mask"].bool()
        group_coverage = {
            "branches": int(group["actions_norm"].shape[0]),
            "predicate_flip_branches": int(predicate_flips.sum()),
            "labelled_continuations": int(labelled.sum()),
            "successful_continuations": int(
                (group["continuation_success"].bool() & labelled).sum()
            ),
            "positive_flow_targets": int(
                (group["branch_weights"] > 0).sum()
            ),
            "resolved_effect_channels": channel_resolved,
        }
        coverage["groups"] += 1
        for key, value in group_coverage.items():
            if key == "resolved_effect_channels":
                continue
            coverage[key] += value
        per_group.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "snapshot_id": group["snapshot_id"],
                "source_trajectory_id": group["source_trajectory_id"],
                "coverage": group_coverage,
                "metrics": metrics,
            }
        )

    resolved_counts = {
        key: sum(
            int(group["metrics"][f"{key}_resolved_effect"])
            for group in per_group
        )
        for key in ("d_q", "d_obj", "d_prog")
    }
    macro_metrics = add_ratios(
        average_metrics([group["metrics"] for group in per_group]),
        resolved_counts,
    )

    by_source: dict[str, list[dict]] = {}
    for group in per_group:
        by_source.setdefault(group["source_trajectory_id"], []).append(
            group["metrics"]
        )
    per_source_metrics = [
        average_metrics(records) for records in by_source.values()
    ]
    source_macro_metrics = add_ratios(
        average_metrics(per_source_metrics),
        resolved_counts,
    )

    effectful_macro_metrics = {}
    for key in ("d_q", "d_obj", "d_prog"):
        records = [
            group["metrics"]
            for group in per_group
            if group["metrics"][f"{key}_resolved_effect"]
        ]
        if not records:
            effectful_macro_metrics[key] = None
            continue
        channel_metrics = {
            name: value
            for name, value in average_metrics(records).items()
            if name.startswith(f"{key}_")
        }
        effectful_macro_metrics[key] = add_ratios(
            channel_metrics,
            {key: len(records)},
            channels=(key,),
        )

    # Resolved-only per-source reporting (2026-07-24 audit correction 3):
    # unresolved groups are excluded from these averages entirely, every
    # source gets its own resolved-channel value, and the promotion rule is
    # applied per source so a single strong source cannot present as broad
    # pairing through an aggregate ratio.
    by_source_groups: dict[str, list[dict]] = {}
    for group in per_group:
        by_source_groups.setdefault(
            group["source_trajectory_id"], []
        ).append(group)
    promotion_thresholds = {
        "centered_model_over_group_mean_max": 0.8,
        "centered_cosine_min": 0.2,
        "matched_over_permutation_max": 0.9,
    }
    resolved_source_metrics: dict[str, dict] = {}
    for key in ("d_q", "d_obj", "d_prog"):
        per_source: dict[str, dict | None] = {}
        for source_id, source_groups in sorted(by_source_groups.items()):
            resolved_records = [
                group["metrics"]
                for group in source_groups
                if group["metrics"][f"{key}_resolved_effect"]
            ]
            if not resolved_records:
                per_source[source_id] = None
                continue
            channel_average = {
                name: value
                for name, value in average_metrics(
                    resolved_records
                ).items()
                if name.startswith(f"{key}_")
            }
            with_ratios = add_ratios(
                channel_average,
                {key: len(resolved_records)},
                channels=(key,),
            )
            ratio = with_ratios.get(
                f"{key}_centered_model_over_group_mean"
            )
            cosine = with_ratios.get(f"{key}_centered_cosine")
            matched = with_ratios.get(f"{key}_matched_over_permutation")
            passes = (
                ratio is not None
                and ratio
                <= promotion_thresholds["centered_model_over_group_mean_max"]
                and cosine is not None
                and cosine > promotion_thresholds["centered_cosine_min"]
                and matched is not None
                and matched
                <= promotion_thresholds["matched_over_permutation_max"]
            )
            per_source[source_id] = {
                "resolved_groups": len(resolved_records),
                "metrics": with_ratios,
                "passes_promotion_rule": bool(passes),
            }
        resolved_sources = [
            value for value in per_source.values() if value is not None
        ]
        passing = sum(
            int(value["passes_promotion_rule"])
            for value in resolved_sources
        )
        resolved_source_metrics[key] = {
            "per_source": per_source,
            "resolved_source_count": len(resolved_sources),
            "sources_passing_promotion_rule": passing,
            "pass_fraction": (
                passing / len(resolved_sources)
                if resolved_sources
                else None
            ),
        }

    output = {
        "schema": "lc_flow_outcome_effect_audit_v2",
        "claim_boundary": (
            "held-out outcome/effect audit; does not measure policy performance"
        ),
        "split": args.split,
        "data": str(args.data.resolve()),
        "dataset_manifest_sha256": sha256(args.data / MANIFEST),
        "evaluator_source_sha256": evaluator_source_sha256(),
        "adapter": {
            "directory": str(args.adapter.resolve()),
            "weights_sha256": sha256(args.adapter / "lc_flow.safetensors"),
            "config_sha256": sha256(args.adapter / "lc_flow_config.json"),
            "metadata": metadata,
        },
        "coverage": {
            **coverage,
            "independent_source_episodes": len(source_episodes),
            "source_trajectory_ids": sorted(source_episodes),
            "unique_active_predicate_targets": [
                list(values) for values in sorted(active_predicate_targets)
            ],
            "resolved_effect_groups": resolved_counts,
        },
        "metric_definition": {
            "raw_zero": "predict an exactly zero continuous effect",
            "centered": (
                "subtract each snapshot group's branch mean from prediction "
                "and target"
            ),
            "group_mean": (
                "oracle snapshot-specific constant fitted from the same "
                "siblings; centered error equals empirical target variance "
                "and is diagnostic, not deployable"
            ),
            "resolved_effect": (
                "q/object action dispersion exceeds max(absolute tolerance, "
                "10x replay dispersion); progress range exceeds the stored "
                "dense-progress margin"
            ),
            "permutation": (
                "mean centered MSE over all non-identity cyclic sibling "
                "shifts; matched/permutation < 1 favors the observed "
                "action-outcome pairing"
            ),
            "undefined_cosine": "serialized as null and excluded from averages",
        },
        "macro_average_metrics": macro_metrics,
        "source_episode_macro_metrics": source_macro_metrics,
        "effectful_group_macro_metrics": effectful_macro_metrics,
        "promotion_thresholds": promotion_thresholds,
        "resolved_source_metrics": resolved_source_metrics,
        "groups": per_group,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps(output["coverage"], indent=2))
    print(json.dumps(output["macro_average_metrics"], indent=2))
    for key, value in resolved_source_metrics.items():
        print(
            f"resolved sources [{key}]: "
            f"{value['sources_passing_promotion_rule']}/"
            f"{value['resolved_source_count']} pass "
            f"(fraction={value['pass_fraction']})"
        )
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
