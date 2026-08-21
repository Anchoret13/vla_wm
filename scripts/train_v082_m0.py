#!/usr/bin/env python3
"""Train the minimal V8.2 M0 predictor from a collector ``groups.pt``.

Normal run::

    python scripts/train_v082_m0.py \
      --groups results/v082_bootstrap/<run>/groups.pt \
      --output results/v082_m0/<run>

CPU-only mechanics smoke (uses generated source-disjoint groups)::

    python scripts/train_v082_m0.py --smoke --output /tmp/v082_m0_smoke

The hyperparameter schedule is intentionally fixed.  There is no sweep and
the test split is evaluated only after validation-selected checkpoints have
been frozen.  ``best.pt`` bundles both the action-conditioned model and its
capacity-matched state-only baseline.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lcwm.v082_m0 import (  # noqa: E402
    LossWeights,
    M0Predictor,
    evaluate_model,
    fit_target_stats,
    group_loss,
    infer_config,
    load_groups,
    make_smoke_groups,
    serializable_config,
    sha256_file,
    split_groups,
)


# One fixed training schedule, inherited from the existing tensor-only LCWM
# path where possible.  Changing it creates a new run/config, never an in-run
# sweep.
SEED = 0
EPOCHS = 30
SMOKE_EPOCHS = 3
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0
HIDDEN_DIM = 256
SMOKE_HIDDEN_DIM = 64
LOSS_WEIGHTS = LossWeights(physical=1.0, outcome=1.0, rank=0.25)


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()}


def _finite_metrics(metrics: dict[str, float | int]) -> bool:
    return all(math.isfinite(float(value)) for value in metrics.values())


def train_epoch(
    models: dict[str, M0Predictor],
    optimizers: dict[str, torch.optim.Optimizer],
    groups,
    stats,
    device: torch.device,
    epoch: int,
    *,
    balance_nontie: bool = False,
) -> None:
    # Both models see exactly the same source-group order.
    order = list(range(len(groups)))
    random.Random(SEED + epoch).shuffle(order)
    for model in models.values():
        model.train()
    for index in order:
        group = groups[index].to(device)
        for name in ("action_conditioned", "state_only"):
            model, optimizer = models[name], optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = group_loss(model, group, stats, LOSS_WEIGHTS,
                                    balance_nontie=balance_nontie)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"non-finite {name} loss at epoch={epoch}, "
                    f"anchor={group.anchor_id}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=Path,
                        help="collector groups.pt (required unless --smoke)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default=None,
                        help="default: cuda for full run, cpu for --smoke")
    parser.add_argument("--balance-nontie", action="store_true",
                        help=("group-level non-tie-balanced ranking batches "
                              "(Action 2M.1). Off by default so the v082 M0 "
                              "result stays reproducible."))
    parser.add_argument("--smoke", action="store_true",
                        help="run generated source-disjoint CPU data for 3 epochs")
    args = parser.parse_args()
    if not args.smoke and args.groups is None:
        parser.error("--groups is required unless --smoke")

    device = torch.device(args.device or ("cpu" if args.smoke else "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device cpu")
    torch.manual_seed(SEED)
    random.seed(SEED)

    groups = make_smoke_groups(SEED) if args.smoke else load_groups(args.groups)
    splits = split_groups(groups)
    stats_cpu = fit_target_stats(splits["train"])
    stats = stats_cpu.to(device)
    config = infer_config(
        groups, hidden_dim=SMOKE_HIDDEN_DIM if args.smoke else HIDDEN_DIM
    )

    # Identical initialization and parameter count.  The baseline differs only
    # by replacing actions with zeros in forward().
    action_model = M0Predictor(config, use_action=True).to(device)
    state_model = M0Predictor(config, use_action=False).to(device)
    state_model.load_state_dict(action_model.state_dict(), strict=True)
    models = {"action_conditioned": action_model, "state_only": state_model}
    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                weight_decay=WEIGHT_DECAY)
        for name, model in models.items()
    }

    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "metrics.jsonl"
    metrics_path.write_text("")
    epochs = SMOKE_EPOCHS if args.smoke else EPOCHS
    best = {
        name: {"loss": float("inf"), "epoch": -1, "state": None,
               "val": None}
        for name in models
    }

    with metrics_path.open("a") as metrics_file:
        for epoch in range(epochs):
            train_epoch(models, optimizers, splits["train"], stats, device,
                        epoch, balance_nontie=args.balance_nontie)
            for name, model in models.items():
                train_metrics = evaluate_model(
                    model, splits["train"], stats, LOSS_WEIGHTS
                )
                val_metrics = evaluate_model(
                    model, splits["val"], stats, LOSS_WEIGHTS
                )
                if not (_finite_metrics(train_metrics)
                        and _finite_metrics(val_metrics)):
                    raise FloatingPointError(f"non-finite metrics for {name}")
                for split, record in (("train", train_metrics),
                                      ("val", val_metrics)):
                    metrics_file.write(json.dumps({
                        "epoch": epoch,
                        "split": split,
                        "model": name,
                        **record,
                    }, sort_keys=True) + "\n")
                metrics_file.flush()
                if float(val_metrics["loss_total"]) < best[name]["loss"]:
                    best[name] = {
                        "loss": float(val_metrics["loss_total"]),
                        "epoch": epoch,
                        "state": _cpu_state_dict(model),
                        "val": copy.deepcopy(val_metrics),
                    }
            print(
                f"epoch={epoch:02d} "
                f"action_val={best['action_conditioned']['loss']:.6f} "
                f"state_val={best['state_only']['loss']:.6f}",
                flush=True,
            )

    # Freeze each model at its own validation-selected epoch.  Test is touched
    # exactly here, after both states are fixed.
    final_metrics: dict[str, dict[str, Any]] = {}
    for name, model in models.items():
        if best[name]["state"] is None:
            raise RuntimeError(f"no validation checkpoint selected for {name}")
        model.load_state_dict(best[name]["state"], strict=True)
        val_metrics = evaluate_model(model, splits["val"], stats, LOSS_WEIGHTS)
        test_metrics = evaluate_model(model, splits["test"], stats, LOSS_WEIGHTS)
        final_metrics[name] = {
            "best_epoch": best[name]["epoch"],
            "val": val_metrics,
            "test": test_metrics,
        }
        with metrics_path.open("a") as metrics_file:
            metrics_file.write(json.dumps({
                "epoch": best[name]["epoch"],
                "split": "test",
                "model": name,
                "checkpoint": "best",
                **test_metrics,
            }, sort_keys=True) + "\n")

    split_summary = {
        split: {
            "anchors": len(values),
            "sources": len({group.source_id for group in values}),
            "source_ids": sorted({group.source_id for group in values}),
            "branch_repeats": sum(int(group.repeat_mask.sum())
                                  for group in values),
        }
        for split, values in splits.items()
    }
    action_test = final_metrics["action_conditioned"]["test"]
    state_test = final_metrics["state_only"]["test"]
    comparison = {
        # Positive means the action-conditioned model is better.
        "prediction_loss_gain_vs_state_only": (
            state_test["matched_prediction_loss"]
            - action_test["matched_prediction_loss"]
        ),
        "physical_mse_gain_vs_state_only": (
            state_test["physical_mse"] - action_test["physical_mse"]
        ),
        "binary_brier_gain_vs_state_only": (
            state_test["outcome_binary_brier"]
            - action_test["outcome_binary_brier"]
        ),
        "rank_macro_accuracy_gain_vs_state_only": (
            action_test["rank_macro_accuracy"]
            - state_test["rank_macro_accuracy"]
        ),
        "action_model_shuffle_gap": action_test["action_shuffle_gap"],
        "state_only_shuffle_gap": state_test["action_shuffle_gap"],
    }

    groups_hash = None if args.smoke else sha256_file(args.groups)
    checkpoint = {
        "schema": "v082_m0_checkpoint_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "smoke": bool(args.smoke),
        "groups_path": None if args.smoke else str(args.groups.resolve()),
        "groups_sha256": groups_hash,
        "config": serializable_config(config, LOSS_WEIGHTS),
        "hyperparameters": {
            "epochs": epochs,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
            "checkpoint_selection": "minimum source-disjoint val loss_total",
        },
        "target_stats": stats_cpu.state_dict(),
        "action_conditioned": {
            "state_dict": best["action_conditioned"]["state"],
            "best_epoch": best["action_conditioned"]["epoch"],
        },
        "state_only": {
            "state_dict": best["state_only"]["state"],
            "best_epoch": best["state_only"]["epoch"],
        },
        "source_files": {
            "lcwm/v082_m0.py": sha256_file(REPO / "lcwm/v082_m0.py"),
            "scripts/train_v082_m0.py": sha256_file(Path(__file__)),
        },
    }
    torch.save(checkpoint, args.output / "best.pt")

    summary = {
        "schema": "v082_m0_summary_v1",
        "created_utc": checkpoint["created_utc"],
        "smoke": bool(args.smoke),
        "device": str(device),
        "groups_path": checkpoint["groups_path"],
        "groups_sha256": groups_hash,
        "config": checkpoint["config"],
        "hyperparameters": checkpoint["hyperparameters"],
        "splits": split_summary,
        "models": final_metrics,
        "comparison": comparison,
        "interpretation": {
            "primary": [
                "prediction_loss_gain_vs_state_only > 0",
                "rank_macro_accuracy_gain_vs_state_only > 0",
                "action_model_shuffle_gap > 0",
            ],
            "scope": ("single-collection M0 trainability readout; no selector, "
                      "policy-improvement, or benchmark-generalization claim"),
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n"
    )

    # Strict load is part of the CPU smoke and remains cheap in a full run.
    loaded = torch.load(args.output / "best.pt", map_location="cpu",
                        weights_only=False)
    reloaded = M0Predictor(config, use_action=True)
    reloaded.load_state_dict(loaded["action_conditioned"]["state_dict"],
                             strict=True)
    if args.smoke:
        smoke_stats = stats_cpu.to("cpu")
        smoke_metrics = evaluate_model(
            reloaded, splits["test"], smoke_stats, LOSS_WEIGHTS
        )
        if not _finite_metrics(smoke_metrics):
            raise AssertionError("CPU smoke reload produced non-finite metrics")
        print("CPU SMOKE PASS: split guard, finite train/eval, best checkpoint "
              "strict reload", flush=True)
    print(f"wrote {metrics_path}, {args.output / 'summary.json'}, "
          f"{args.output / 'best.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

