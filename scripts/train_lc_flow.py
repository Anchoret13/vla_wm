#!/usr/bin/env python
"""Train the first outcome-grounded LC-Flow adapter on grouped Chain-3 data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            os.environ.get(
                "LCWM_BRANCH_DIR",
                REPO_ROOT.parent / "datasets" / "chain_branches_v1_1",
            )
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "lc_flow_chain3",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--outcome-scale", type=float, default=1.0)
    parser.add_argument(
        "--outcome-objective",
        choices=("raw_v1", "effect_centered_v1"),
        required=True,
        help=(
            "explicit choice required (2026-07-24 audit correction 2): "
            "raw_v1 is rejected for training runs after its "
            "predicate/continuation shortcut; effect_centered_v1 is the "
            "active objective"
        ),
    )
    parser.add_argument("--flow-scale", type=float, default=0.1)
    parser.add_argument("--flow-microbatch", type=int, default=1)
    parser.add_argument("--truncate-bptt", type=int, default=16)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--probe-seed",
        type=int,
        default=12345,
        help="fixed noise/time seed for the deterministic pre/post probe",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and summarize groups without loading PI05",
    )
    return parser.parse_args()


def select_indices(length: int, maximum: int | None) -> list[int]:
    indices = list(range(length))
    return indices if maximum is None else indices[:maximum]


def source_balanced_epoch_order(
    indices_by_source: dict[str, list[int]],
    *,
    epoch: int,
    seed: int,
) -> list[int]:
    """Choose one rotating group per independent source episode."""
    selected = [
        indices[(epoch + seed) % len(indices)]
        for _, indices in sorted(indices_by_source.items())
    ]
    random.Random(seed + epoch).shuffle(selected)
    return selected


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_fingerprint() -> str:
    digest = hashlib.sha256()
    for relative in (
        "scripts/train_lc_flow.py",
        "lcwm/lc_flow_train.py",
        "lcwm/lc_flow.py",
        "lcwm/branch_data.py",
    ):
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update((REPO_ROOT / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def dry_run(args: argparse.Namespace) -> None:
    from lcwm.branch_data import BranchGroupDataset

    summaries = {}
    for split in ("train", "dev", "test"):
        dataset = BranchGroupDataset(args.data, split)
        groups = branches = positive = partial = 0
        for index in select_indices(len(dataset), args.max_groups):
            group = dataset[index]
            groups += 1
            branches += int(group["actions_norm"].shape[0])
            positive += int((group["branch_weights"] > 0).sum())
            partial += int((group["executed_lengths"] < 10).sum())
        summaries[split] = {
            "groups": groups,
            "branches": branches,
            "positive_flow_targets": positive,
            "partial_terminal_branches": partial,
        }
    print(json.dumps(summaries, indent=2))


@torch.no_grad()
def evaluate_outcomes(
    lc_state,
    dataset,
    indices,
    device,
    *,
    effect_scales=None,
) -> dict[str, float]:
    from lcwm.lc_flow_train import (
        branch_outcome_predictions,
        outcome_effect_metrics,
        to_device_tensors,
        unroll_lc_history,
    )
    from lcwm.lc_flow import outcome_loss
    from lcwm.lc_flow import centered_effect_loss

    if not indices:
        return {}
    was_training = lc_state.training
    lc_state.eval()
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for index in indices:
        group = dataset[index]
        z_t = unroll_lc_history(lc_state, group, device=device)
        predictions = branch_outcome_predictions(
            lc_state, z_t, group, device=device
        )
        targets = to_device_tensors(group["targets"], device)
        loss, parts = outcome_loss(predictions, targets)
        effect_metrics = outcome_effect_metrics(
            predictions,
            targets,
        )
        values: dict[str, float] = {"loss": float(loss), **{
            name: float(value) for name, value in parts.items()
        }}
        if effect_scales is not None:
            effect_objective, effect_parts = centered_effect_loss(
                predictions,
                targets,
                effect_scales,
            )
            values["effect_objective"] = float(effect_objective)
            values.update(
                {
                    name: float(value)
                    for name, value in effect_parts.items()
                }
            )
        for name, value in effect_metrics.items():
            if name.endswith("_centered_cosine_valid"):
                values[name] = float(value)
                continue
            if name.endswith("_centered_cosine"):
                valid_name = f"{name}_valid"
                if not bool(effect_metrics[valid_name]):
                    continue
            values[name] = float(value)
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + value
            counts[key] = counts.get(key, 0) + 1
    lc_state.train(was_training)
    return {
        f"dev_{key}": value / counts[key]
        for key, value in totals.items()
    }


def main() -> None:
    args = parse_args()
    if args.dry_run:
        dry_run(args)
        return
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.flow_scale < 0 or args.outcome_scale < 0:
        raise ValueError("loss scales must be nonnegative")
    if args.flow_scale == 0 and args.outcome_scale == 0:
        raise ValueError("at least one training loss must be active")

    from lcwm.branch_data import MANIFEST, SCHEMA, BranchGroupDataset
    from lcwm.chassis import Pi05Runner
    from lcwm.lc_flow import LCState, freeze_pi05_base, save_lc_adapter
    from lcwm.lc_flow_train import (
        config_from_group,
        fit_effect_scales,
        fixed_group_probe,
        initialize_effect_v1_heads,
        train_group_step,
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_data = BranchGroupDataset(args.data, "train")
    dev_data = BranchGroupDataset(args.data, "dev")
    train_indices = select_indices(len(train_data), args.max_groups)
    dev_indices = select_indices(len(dev_data), args.max_groups)
    if not train_indices:
        raise RuntimeError("training split contains no admitted branch groups")
    positive_counts = {
        index: int((train_data[index]["branch_weights"] > 0).sum())
        for index in train_indices
    }
    total_positive = sum(positive_counts.values())
    if args.flow_scale > 0 and total_positive == 0:
        raise RuntimeError(
            "flow loss is active but selected training groups contain no "
            "positive branch targets"
        )
    probe_index = (
        next(
            index
            for index in train_indices
            if positive_counts[index] > 0
        )
        if args.flow_scale > 0
        else train_indices[0]
    )

    runner = Pi05Runner(suite_name="libero_10", device=args.device)
    freeze_pi05_base(runner.policy)
    sample_group = train_data[probe_index]
    lc_config = config_from_group(
        sample_group,
        expert_width=runner.policy.model.action_in_proj.out_features,
    )
    lc_state = LCState(lc_config).to(device).train()
    effect_scales = None
    effect_scale_report = None
    indices_by_source: dict[str, list[int]] = {}
    if args.outcome_objective == "effect_centered_v1":
        selected_groups = [train_data[index] for index in train_indices]
        effect_scales, effect_scale_report = fit_effect_scales(
            selected_groups
        )
        initialize_effect_v1_heads(lc_state)
        for index, group in zip(
            train_indices, selected_groups, strict=True
        ):
            indices_by_source.setdefault(
                str(group["source_trajectory_id"]), []
            ).append(index)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in lc_state.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected_train_artifacts = [
        {
            "path": str(train_data.paths[index].resolve()),
            "sha256": sha256_file(train_data.paths[index]),
        }
        for index in train_indices
    ]
    selected_dev_artifacts = [
        {
            "path": str(dev_data.paths[index].resolve()),
            "sha256": sha256_file(dev_data.paths[index]),
        }
        for index in dev_indices
    ]
    config = {
        **vars(args),
        "data": str(args.data.resolve()),
        "output": str(output),
        "dataset_schema": SCHEMA,
        "base_model_id": runner.model_id,
        "lc_flow_config": asdict(lc_config),
        "group_weighting": (
            "one rotating group per source episode per epoch"
            if effect_scales is not None
            else "one optimizer step per snapshot group"
        ),
        "effect_scale_report": effect_scale_report,
        "unidentified_heads": (
            ["next_bits", "continuation_success"]
            if effect_scales is not None
            else []
        ),
        "proposal_language_usage": "provenance only; full_instruction conditions training",
        "selected_train_groups": len(train_indices),
        "selected_positive_flow_targets": total_positive,
        "fixed_probe_snapshot_id": sample_group["snapshot_id"],
        "dataset_manifest_sha256": sha256_file(train_data.root / MANIFEST),
        "selected_train_artifacts": selected_train_artifacts,
        "selected_dev_artifacts": selected_dev_artifacts,
        "training_source_sha256": source_fingerprint(),
    }
    (output / "training_config.json").write_text(
        json.dumps(config, indent=2, default=str)
    )
    metrics_path = output / "metrics.jsonl"
    global_step = 0
    with metrics_path.open("w") as metrics_file:
        dev_before = evaluate_outcomes(
            lc_state,
            dev_data,
            dev_indices,
            device,
            effect_scales=effect_scales,
        )
        dev_before_record = {
            "event": "dev_before",
            "global_step": global_step,
            **dev_before,
        }
        metrics_file.write(json.dumps(dev_before_record) + "\n")
        metrics_file.flush()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        probe_before = fixed_group_probe(
            runner,
            lc_state,
            sample_group,
            device=device,
            seed=args.probe_seed,
            effect_scales=effect_scales,
        )
        probe_before_record = {
            "event": "fixed_probe_before",
            "global_step": global_step,
            "snapshot_id": sample_group["snapshot_id"],
            "probe_seed": args.probe_seed,
            **probe_before,
        }
        if device.type == "cuda":
            probe_before_record["peak_gpu_gb"] = (
                torch.cuda.max_memory_allocated(device) / 2**30
            )
        metrics_file.write(json.dumps(probe_before_record) + "\n")
        metrics_file.flush()
        print(
            "[probe-before] "
            f"raw_out={probe_before['outcome_loss']:.4f} "
            f"train_out={probe_before['training_outcome_loss']:.4f} "
            f"flow={probe_before['flow_loss']:.4f} "
            f"exec_shift={probe_before['action_shift_exec_mean_abs']:.6f}",
            flush=True,
        )

        for epoch in range(args.epochs):
            if effect_scales is not None:
                order = source_balanced_epoch_order(
                    indices_by_source,
                    epoch=epoch,
                    seed=args.seed,
                )
            else:
                order = train_indices.copy()
                random.Random(args.seed + epoch).shuffle(order)
            for index in order:
                group = train_data[index]
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                metrics = train_group_step(
                    runner,
                    lc_state,
                    optimizer,
                    group,
                    device=device,
                    outcome_scale=args.outcome_scale,
                    flow_scale=args.flow_scale,
                    flow_microbatch=args.flow_microbatch,
                    flow_seed=args.seed + 1_000_000 * epoch + global_step,
                    truncate_bptt=args.truncate_bptt,
                    max_grad_norm=args.max_grad_norm,
                    effect_scales=effect_scales,
                )
                record = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "snapshot_id": group["snapshot_id"],
                    **metrics,
                }
                if device.type == "cuda":
                    record["peak_gpu_gb"] = (
                        torch.cuda.max_memory_allocated(device) / 2**30
                    )
                metrics_file.write(json.dumps(record) + "\n")
                metrics_file.flush()
                print(
                    f"[train] e{epoch} s{global_step} "
                    f"out={metrics['outcome_loss']:.4f} "
                    f"flow={metrics['flow_loss']:.4f} "
                    f"positive={int(metrics['positive_branches'])}",
                    flush=True,
                )
                global_step += 1

            dev_metrics = evaluate_outcomes(
                lc_state,
                dev_data,
                dev_indices,
                device,
                effect_scales=effect_scales,
            )
            epoch_record = {
                "epoch": epoch,
                "global_step": global_step,
                "event": "epoch_end",
                **dev_metrics,
            }
            metrics_file.write(json.dumps(epoch_record) + "\n")
            metrics_file.flush()
            save_lc_adapter(
                lc_state,
                output / f"epoch_{epoch:03d}",
                metadata={
                    "base_model_id": runner.model_id,
                    "dataset_schema": SCHEMA,
                    "optimizer_steps": global_step,
                    "outcome_scale": args.outcome_scale,
                    "flow_scale": args.flow_scale,
                    "outcome_objective": args.outcome_objective,
                    "effect_scales": (
                        asdict(effect_scales)
                        if effect_scales is not None
                        else None
                    ),
                },
            )
            torch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                },
                output / "optimizer_latest.pt",
            )

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        probe_after = fixed_group_probe(
            runner,
            lc_state,
            sample_group,
            device=device,
            seed=args.probe_seed,
            effect_scales=effect_scales,
        )
        probe_after_record = {
            "event": "fixed_probe_after",
            "global_step": global_step,
            "snapshot_id": sample_group["snapshot_id"],
            "probe_seed": args.probe_seed,
            **probe_after,
        }
        if device.type == "cuda":
            probe_after_record["peak_gpu_gb"] = (
                torch.cuda.max_memory_allocated(device) / 2**30
            )
        metrics_file.write(json.dumps(probe_after_record) + "\n")
        metrics_file.flush()
        print(
            "[probe-after] "
            f"raw_out={probe_after['outcome_loss']:.4f} "
            f"train_out={probe_after['training_outcome_loss']:.4f} "
            f"flow={probe_after['flow_loss']:.4f} "
            f"exec_shift={probe_after['action_shift_exec_mean_abs']:.6f}",
            flush=True,
        )

        dev_after = evaluate_outcomes(
            lc_state,
            dev_data,
            dev_indices,
            device,
            effect_scales=effect_scales,
        )
        dev_after_record = {
            "event": "dev_after",
            "global_step": global_step,
            **dev_after,
        }
        metrics_file.write(json.dumps(dev_after_record) + "\n")
        metrics_file.flush()

    summary = {
        "snapshot_id": sample_group["snapshot_id"],
        "probe_seed": args.probe_seed,
        "optimizer_steps": global_step,
        "before": probe_before,
        "after": probe_after,
        "delta": {
            key: probe_after[key] - probe_before[key]
            for key in probe_before
            if key in probe_after
        },
        "dev_before": dev_before,
        "dev_after": dev_after,
        "dev_delta": {
            key: dev_after[key] - dev_before[key]
            for key in dev_before
            if key in dev_after
        },
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    save_lc_adapter(
        lc_state,
        output / "final",
        metadata={
            "base_model_id": runner.model_id,
            "dataset_schema": SCHEMA,
            "optimizer_steps": global_step,
            "outcome_scale": args.outcome_scale,
            "flow_scale": args.flow_scale,
            "outcome_objective": args.outcome_objective,
            "effect_scales": (
                asdict(effect_scales)
                if effect_scales is not None
                else None
            ),
        },
    )
    print(f"adapter -> {output / 'final'}")


if __name__ == "__main__":
    main()
