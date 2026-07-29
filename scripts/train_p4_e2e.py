#!/usr/bin/env python
"""p4_e2e_v1_wm — the single P4 smoke/full-run entry point (M1/M2 spec).

Simulator-free: imports go through lcwm.seq_windows / lcwm.p4_train only;
the 3B π0.5 model is never loaded here. --smoke runs one sequential episode
+ one branch group for two optimizer steps, asserts finite losses, strict
save/reload output reproduction, and frozen-adapter bitwise stability.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.p4_targets import sequential_targets  # noqa: E402
from lcwm.p4_train import (  # noqa: E402
    LR,
    MAX_GRAD_NORM,
    VIC_ACCUM_SOURCES,
    VIC_COV_WEIGHT,
    VIC_VAR_WEIGHT,
    WEIGHT_DECAY,
    branch_source_loss,
    build_models,
    checkpoint_bundle,
    load_branch_groups,
    sequential_source_loss,
    strict_load,
)
from lcwm.self_predict import variance_covariance_penalty  # noqa: E402
from lcwm.seq_windows import SequentialEpisodeDataset  # noqa: E402
from lcwm.value_heads import ValueHeads  # noqa: E402  (reload check)

CACHE_DIR = Path(
    "/home/stargazer/Desktop/vla_wm/datasets/seq_prefix_cache_v1"
)
BRANCH_DIR = Path(
    "/home/stargazer/Desktop/vla_wm/datasets/chain_branches_v1_1"
)
LABELS_DIR = Path(
    "/home/stargazer/Desktop/vla_wm/datasets/chain_source_labels_v1"
)
FAMILY_KEYS = (
    "self", "residual", "physical", "next_bits", "reward", "dphi", "v_next",
)


def adapter_snapshot(lc) -> list[torch.Tensor]:
    return [
        lc.wz_hidden.weight.detach().clone(),
        lc.wz_out.weight.detach().clone(),
    ]


def assert_adapter_unchanged(lc, snapshot) -> None:
    current = adapter_snapshot(lc)
    for a, b in zip(snapshot, current):
        assert torch.equal(a, b), "policy adapter changed during WM stage"


def combine(families: dict, device) -> torch.Tensor:
    total = torch.zeros((), device=device)
    for key in FAMILY_KEYS:
        value = families.get(key)
        if value is not None:
            total = total + value
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "lc_flow_v04_e2e_v1" / "world_model",
    )
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    train_set = SequentialEpisodeDataset(CACHE_DIR, splits=("train",))
    episodes = train_set.episodes
    targets = [sequential_targets(e) for e in episodes]
    branch_by_source = load_branch_groups(BRANCH_DIR, LABELS_DIR)
    chain_sources = sorted(branch_by_source)
    print(
        f"sequential sources: {len(episodes)} "
        f"({sum(1 for e in episodes if e['source_kind']=='demo')} demo + "
        f"{sum(1 for e in episodes if e['source_kind']=='chain')} chain); "
        f"branch sources: {len(chain_sources)} "
        f"({sum(len(v) for v in branch_by_source.values())} groups)",
        flush=True,
    )

    lc, predictor, ema, heads = build_models(device)
    trainable = [
        p
        for p in [*lc.parameters(), *predictor.parameters(),
                  *heads.parameters()]
        if p.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable, lr=LR, weight_decay=WEIGHT_DECAY
    )
    frozen = adapter_snapshot(lc)
    config = {
        "run": "p4_e2e_v1_wm",
        "seed": args.seed,
        "epochs": args.epochs,
        "spec": "2026-07-28 M1/M2 + 2026-07-29 A0/A1",
        "sources": {
            "sequential": [e["source_id"] for e in episodes],
            "branch": chain_sources,
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "training_config.json").write_text(
        json.dumps(config, indent=2)
    )

    if args.smoke:
        episode_ids = [0]
        n_epochs = 1
    else:
        episode_ids = list(range(len(episodes)))
        n_epochs = args.epochs

    logs = []
    global_step = 0
    smoke_outputs = None
    for epoch in range(n_epochs):
        order = episode_ids[:]
        random.shuffle(order)
        epoch_log = {k: [] for k in FAMILY_KEYS}
        # Source-balanced: chunked VIC accumulation over sequential sources.
        for start in range(0, len(order), VIC_ACCUM_SOURCES):
            chunk = order[start : start + VIC_ACCUM_SOURCES]
            optimizer.zero_grad(set_to_none=True)
            chunk_states, chunk_loss = [], torch.zeros((), device=device)
            for index in chunk:
                families, states = sequential_source_loss(
                    lc, predictor, ema, heads,
                    episodes[index], targets[index], device,
                )
                chunk_states.append(states)
                chunk_loss = chunk_loss + combine(families, device) / len(
                    chunk
                )
                for k, v in families.items():
                    if v is not None:
                        epoch_log[k].append(float(v))
            vic = variance_covariance_penalty(
                torch.cat(chunk_states),
                var_weight=VIC_VAR_WEIGHT,
                cov_weight=VIC_COV_WEIGHT,
            )
            (chunk_loss + vic).backward()
            torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM)
            optimizer.step()
            ema.ema_update(lc)
            global_step += 1
            if args.smoke and global_step >= 1:
                break
        # One admitted sibling group per chain source per epoch
        # (seed-1001's groups rotate deterministically).
        branch_iter = (
            chain_sources[:1] if args.smoke else chain_sources
        )
        for source in branch_iter:
            groups = branch_by_source[source]
            group = groups[epoch % len(groups)]
            optimizer.zero_grad(set_to_none=True)
            families = branch_source_loss(lc, heads, group, device)
            loss = combine(families, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM)
            optimizer.step()
            ema.ema_update(lc)
            global_step += 1
            for k, v in families.items():
                if v is not None:
                    epoch_log[k].append(float(v))
            if args.smoke and global_step >= 2:
                break
        means = {
            k: (sum(v) / len(v) if v else None)
            for k, v in epoch_log.items()
        }
        logs.append({"epoch": epoch, **means})
        print(
            f"[epoch {epoch}] "
            + " ".join(
                f"{k}={means[k]:.4f}"
                for k in FAMILY_KEYS
                if means[k] is not None
            ),
            flush=True,
        )
        if not args.smoke and (epoch + 1) % 5 == 0:
            torch.save(
                checkpoint_bundle(
                    lc, predictor, ema, heads, optimizer, global_step,
                    config,
                ),
                args.output / f"checkpoint_epoch{epoch:03d}.pt",
            )
        if args.smoke:
            with torch.no_grad():
                episode, target = episodes[0], targets[0]
                fam, _ = sequential_source_loss(
                    lc, predictor, ema, heads, episode, target, device,
                    train=False,
                )
                smoke_outputs = {
                    k: float(v) for k, v in fam.items() if v is not None
                }
            break

    assert_adapter_unchanged(lc, frozen)
    bundle = checkpoint_bundle(
        lc, predictor, ema, heads, optimizer, global_step, config
    )
    torch.save(bundle, args.output / "checkpoint_final.pt")
    (args.output / "per_loss_logs.json").write_text(json.dumps(logs))

    if args.smoke:
        for record in logs:
            for k in FAMILY_KEYS:
                if record.get(k) is not None:
                    assert record[k] == record[k], f"{k} is NaN"
        lc2, predictor2, ema2, heads2 = build_models(device)
        strict_load(
            torch.load(args.output / "checkpoint_final.pt",
                       weights_only=False),
            lc2, predictor2, ema2, heads2,
        )
        with torch.no_grad():
            fam2, _ = sequential_source_loss(
                lc2, predictor2, ema2, heads2,
                episodes[0], targets[0], device, train=False,
            )
            reloaded = {
                k: float(v) for k, v in fam2.items() if v is not None
            }
        for k, v in smoke_outputs.items():
            assert abs(reloaded[k] - v) < 1e-6, (
                f"reload mismatch on {k}: {v} vs {reloaded[k]}"
            )
        print("SMOKE: finite losses, strict reload reproduces outputs, "
              "adapter frozen — PASSED", flush=True)
    print(f"-> {args.output} (steps={global_step})", flush=True)


if __name__ == "__main__":
    main()
