#!/usr/bin/env python
"""P2 — one-step latent self-prediction training (Gate A candidate run).

Trains, on P1 sequential episodes (prefix-cache demos + chain histories):
- L_self^(1): p_θ(T_θ(z_i, a_i)) vs sg[EMA-target posterior z⁺_{i+1}]
  (per-token cosine distance, the recorded default);
- absolute physical heads on labeled windows (Δq, Δobj vs stored labels,
  per-dim scale-normalized) so the state cannot satisfy training through a
  collapsed target (v0.4 P2 requirement). Predicate/reward/value heads wait
  for P3 semantics.

Gate A evaluation on the dev split, per source episode: model distance vs
copy-state / ignore-action / mean-next baselines + collapse monitors.
The policy adapter is untouched (no flow loss here); π0.5 is not loaded.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.lc_flow import LCState  # noqa: E402
from lcwm.self_predict import (  # noqa: E402
    EMATarget,
    LatentPredictor,
    collapse_metrics,
    latent_distance,
)
from lcwm.seq_prefix_cache import CACHE_DIR  # noqa: E402
from lcwm.seq_windows import SequentialEpisodeDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ema-tau", type=float, default=0.996)
    parser.add_argument("--truncate-bptt", type=int, default=16)
    parser.add_argument("--self-scale", type=float, default=1.0)
    parser.add_argument("--outcome-scale", type=float, default=1.0)
    parser.add_argument("--distance", default="cosine")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "self_predict_v1",
    )
    return parser.parse_args()


def episode_to_device_slices(episode, device):
    """Yield per-record tensors lazily; full episodes are ~150MB fp16."""
    hidden = episode["prefix_hidden"]
    mask = episode["prefix_mask"]
    return hidden, mask


def outcome_targets(episode, index, device):
    d_q = (episode["q"][index + 1] - episode["q"][index]).to(device)
    d_obj = (
        episode["obj_pos"][index + 1] - episode["obj_pos"][index]
    ).to(device)
    return d_q[None], d_obj[None]


def fit_outcome_scales(episodes, device) -> dict[str, torch.Tensor]:
    d_qs, d_objs = [], []
    for episode in episodes:
        if not episode["has_labels"]:
            continue
        d_qs.append(episode["q"][1:] - episode["q"][:-1])
        d_objs.append(
            (episode["obj_pos"][1:] - episode["obj_pos"][:-1]).flatten(1)
        )
    d_q = torch.cat(d_qs)
    d_obj = torch.cat(d_objs)
    return {
        "q": d_q.std(dim=0).clamp_min(1e-4).to(device),
        "obj": d_obj.std(dim=0).clamp_min(1e-4).to(device),
        "obj_shape": torch.tensor(episodes[0]["obj_pos"].shape[1:]),
    }


def run_episode(
    lc,
    predictor,
    ema,
    episode,
    device,
    *,
    distance: str,
    truncate_bptt: int,
    scales,
    outcome_scale: float,
    train: bool,
):
    hidden, mask = episode["prefix_hidden"], episode["prefix_mask"]
    actions = episode["actions_norm"]
    exec_mask = episode["action_exec_mask"]
    T = hidden.shape[0]
    transitions = T - 1
    detach_before = (
        T - truncate_bptt if train and transitions > truncate_bptt else None
    )

    h0 = hidden[0:1].to(device)
    m0 = mask[0:1].to(device)
    z = lc.posterior(h0, m0)
    with torch.no_grad():
        z_target = ema.posterior(h0, m0)

    self_losses, out_losses = [], []
    predictions, targets, priors_no_action = [], [], []
    n_obj = int(scales["obj_shape"][0]) if scales else 0
    for index in range(transitions):
        if detach_before is not None and index == detach_before - 1:
            z = z.detach()
        a = actions[index : index + 1].to(device)
        a_mask = exec_mask[index : index + 1].to(device)
        h_next = hidden[index + 1 : index + 2].to(device)
        m_next = mask[index + 1 : index + 2].to(device)

        prior = lc.transition(z, a, a_mask)
        with torch.no_grad():
            z_target_next = ema.step(z_target, a, h_next, m_next, a_mask)
            prior_no_action = lc.transition(
                z, torch.zeros_like(a), a_mask
            )
        predicted = predictor(prior)
        self_losses.append(
            latent_distance(predicted, z_target_next, distance)
        )
        predictions.append(predicted.detach())
        targets.append(z_target_next)
        priors_no_action.append(predictor(prior_no_action).detach())

        if episode["has_labels"] and scales is not None:
            out = lc.outcome(prior)
            d_q_t, d_obj_t = outcome_targets(episode, index, device)
            q_loss = (
                ((out["d_q"] - d_q_t) / scales["q"]) ** 2
            ).mean()
            obj_pred = out["d_obj"][:, :n_obj].flatten(1)
            obj_loss = (
                ((obj_pred - d_obj_t.flatten(1)) / scales["obj"]) ** 2
            ).mean()
            out_losses.append(q_loss + obj_loss)

        z = lc.step(z, a, h_next, m_next, a_mask)
        with torch.no_grad():
            z_target = z_target_next

    self_loss = torch.stack(self_losses).mean()
    out_loss = (
        torch.stack(out_losses).mean()
        if out_losses
        else torch.zeros((), device=device)
    )
    return {
        "self_loss": self_loss,
        "out_loss": out_loss,
        "predictions": torch.cat(predictions),
        "targets": torch.cat(targets),
        "priors_no_action": torch.cat(priors_no_action),
    }


@torch.no_grad()
def evaluate(lc, predictor, ema, episodes, device, *, distance, scales):
    lc.eval()
    predictor.eval()
    per_source = {}
    all_targets, all_predictions = [], []
    for episode in episodes:
        result = run_episode(
            lc,
            predictor,
            ema,
            episode,
            device,
            distance=distance,
            truncate_bptt=10**9,
            scales=scales if episode["has_labels"] else None,
            outcome_scale=0.0,
            train=False,
        )
        targets = result["targets"]
        previous = torch.cat(
            [targets[0:1], targets[:-1]]
        )  # z⁺_i as copy-state prediction for z⁺_{i+1}
        model = float(
            latent_distance(result["predictions"], targets, distance)
        )
        copy_state = float(latent_distance(previous, targets, distance))
        ignore_action = float(
            latent_distance(result["priors_no_action"], targets, distance)
        )
        mean_next = float(
            latent_distance(
                targets.mean(dim=0, keepdim=True).expand_as(targets),
                targets,
                distance,
            )
        )
        per_source[episode["source_id"]] = {
            "kind": episode["source_kind"],
            "transitions": int(targets.shape[0]),
            "model": model,
            "copy_state": copy_state,
            "ignore_action": ignore_action,
            "mean_next": mean_next,
            "beats_all": bool(
                model < min(copy_state, ignore_action, mean_next)
            ),
        }
        all_targets.append(targets)
        all_predictions.append(result["predictions"])
    lc.train()
    predictor.train()
    targets = torch.cat(all_targets)
    predictions = torch.cat(all_predictions)
    passes = sum(v["beats_all"] for v in per_source.values())
    return {
        "per_source": per_source,
        "sources_beating_all_baselines": passes,
        "source_count": len(per_source),
        "collapse_targets": collapse_metrics(targets),
        "collapse_predictions": collapse_metrics(predictions),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    train_set = SequentialEpisodeDataset(
        args.cache_dir, splits=("train",)
    )
    dev_set = SequentialEpisodeDataset(args.cache_dir, splits=("dev",))
    print(
        f"train {len(train_set)} episodes "
        f"({train_set.transition_count()} transitions), "
        f"dev {len(dev_set)} episodes "
        f"({dev_set.transition_count()} transitions)",
        flush=True,
    )

    lc = LCState().to(device)
    predictor = LatentPredictor().to(device)
    ema = EMATarget(lc, tau=args.ema_tau).to(device)
    scales = fit_outcome_scales(train_set.episodes, device)

    optimizer = torch.optim.AdamW(
        [*lc.parameters(), *predictor.parameters()],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    history = []
    order = list(range(len(train_set)))
    for epoch in range(args.epochs):
        random.shuffle(order)
        epoch_self, epoch_out = [], []
        for index in order:
            episode = train_set.episodes[index]
            optimizer.zero_grad(set_to_none=True)
            result = run_episode(
                lc,
                predictor,
                ema,
                episode,
                device,
                distance=args.distance,
                truncate_bptt=args.truncate_bptt,
                scales=scales,
                outcome_scale=args.outcome_scale,
                train=True,
            )
            loss = (
                args.self_scale * result["self_loss"]
                + args.outcome_scale * result["out_loss"]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [*lc.parameters(), *predictor.parameters()], 1.0
            )
            optimizer.step()
            ema.ema_update(lc)
            epoch_self.append(float(result["self_loss"]))
            epoch_out.append(float(result["out_loss"]))
        record = {
            "epoch": epoch,
            "train_self": sum(epoch_self) / len(epoch_self),
            "train_out": sum(epoch_out) / len(epoch_out),
        }
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            record["dev"] = evaluate(
                lc,
                predictor,
                ema,
                dev_set.episodes,
                device,
                distance=args.distance,
                scales=scales,
            )
            dev = record["dev"]
            print(
                f"[epoch {epoch}] self={record['train_self']:.4f} "
                f"out={record['train_out']:.4f} | dev sources beating all "
                f"baselines: {dev['sources_beating_all_baselines']}/"
                f"{dev['source_count']} | target rank "
                f"{dev['collapse_targets']['effective_rank']:.1f} "
                f"min_std {dev['collapse_targets']['min_per_dim_std']:.2e}",
                flush=True,
            )
        else:
            print(
                f"[epoch {epoch}] self={record['train_self']:.4f} "
                f"out={record['train_out']:.4f}",
                flush=True,
            )
        history.append(record)

    final_eval = history[-1].get("dev") or evaluate(
        lc, predictor, ema, dev_set.episodes, device,
        distance=args.distance, scales=scales,
    )
    torch.save(
        {
            "lc_state": lc.state_dict(),
            "predictor": predictor.state_dict(),
            "ema": ema.state_dict(),
            "config": vars(args) | {"output": str(args.output)},
        },
        args.output / "self_predict.pt",
    )
    (args.output / "history.json").write_text(
        json.dumps(
            {
                "history": [
                    {
                        k: v
                        for k, v in record.items()
                    }
                    for record in history
                ],
                "gate_a": {
                    "rule": (
                        "held-out one-step prediction beats copy-state, "
                        "ignore-action and mean-next on dev sources; target "
                        "non-collapsed (structural gate, v0.4 §9 Gate A)"
                    ),
                    "final": final_eval,
                },
            },
            indent=2,
            default=float,
        )
    )
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
