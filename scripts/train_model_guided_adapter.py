#!/usr/bin/env python
"""M4/A3 — model-guided AdaRMS adapter finetuning (fast-pilot budget).

Frozen: π0.5 base, the whole P4 world model (transition/update/outcome/
value/predictor/EMA). Trainable: wz_hidden/wz_out only.

Per epoch: 8 decision-time-stratified late teacher states per train source
(cyclic epoch offset; zero-teacher states become rehearsal-only), each
paired 1:1 with one source-balanced early-state (pre-2/3) full-50 stock
rehearsal item. Model-guided credit is first-ten only via
credit_bounded_actions inside cached_branch_flow_loss; teacher weight
follows the positive_advantage_weights convention 1+clip(Â,0,1).

Spec deviation recorded: `paired_flow_noise_time` does not exist in the
codebase; deterministic per-item seeded noise/time is used instead.
Startup regression: suffix positions contribute gradient in rehearsal but
cannot change the model-guided first-ten loss/gradient.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

CACHE_DIR = Path(
    "/home/stargazer/Desktop/vla_wm/datasets/chain_episode_cache_v1"
)
EPOCHS = 5
LR = 1e-4
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0
TEACHERS_PER_SOURCE = 8
NOISE_BASE = 999_000


def item_noise_time(policy, index: int, device):
    cfg = policy.config
    g = torch.Generator().manual_seed(NOISE_BASE + index)
    noise = torch.randn(
        1, cfg.chunk_size, cfg.max_action_dim, generator=g
    ).to(device)
    time = torch.rand(1, generator=g).to(device)
    return noise, time


@torch.no_grad()
def early_pool(episode) -> list[int]:
    bits = episode["predicate_bits"]
    n_actions = episode["action_block_norm"].shape[0]
    return [
        d for d in range(n_actions)
        if int(bits[d].float().sum()) < 2
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--teacher",
        type=Path,
        default=REPO_ROOT / "results" / "lc_flow_v04_e2e_v1"
        / "candidate_teacher.pt",
    )
    parser.add_argument(
        "--wm",
        type=Path,
        default=REPO_ROOT / "results" / "lc_flow_v04_e2e_v1"
        / "world_model" / "checkpoint_final.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "lc_flow_v04_e2e_v1"
        / "policy_adapter",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    from build_candidate_teacher import unroll_states  # noqa: E402
    from lcwm.chassis import Pi05Runner
    from lcwm.lc_flow import (
        cached_branch_flow_loss,
        freeze_pi05_base,
        raw_flow_losses_from_prefix,
    )
    from lcwm.p4_train import build_models, strict_load
    from lcwm.policy_interface import ReplayPolicyInterface

    lc, predictor, ema, heads = build_models(device)
    strict_load(
        torch.load(args.wm, weights_only=False), lc, predictor, ema, heads
    )
    for p in lc.parameters():
        p.requires_grad_(False)
    lc.wz_hidden.requires_grad_(True)
    lc.wz_out.requires_grad_(True)
    lc.eval()

    runner = Pi05Runner(suite_name="libero_10")
    freeze_pi05_base(runner.policy)
    interface = ReplayPolicyInterface(runner)

    teacher = torch.load(args.teacher, weights_only=False)
    rows = teacher["rows"]
    by_source: dict[str, list[dict]] = {}
    for row in rows:
        by_source.setdefault(row["source_id"], []).append(row)
    for source in by_source:
        by_source[source].sort(key=lambda r: r["decision"])
    episodes = {}
    for path in sorted(CACHE_DIR.glob("*.pt")):
        data = torch.load(path, weights_only=False)
        if data["split"] == "train":
            episodes[data["source_trajectory_id"]] = data
    sources = sorted(by_source)
    early = {s: early_pool(episodes[s]) for s in sources}
    z_cache = {
        s: [z.detach() for z in unroll_states(lc, episodes[s], device)]
        for s in sources
    }

    trainable = [lc.wz_hidden.weight, lc.wz_out.weight] + [
        p
        for p in [lc.wz_hidden.bias, lc.wz_out.bias]
        if p is not None
    ]
    optimizer = torch.optim.AdamW(
        trainable, lr=LR, weight_decay=WEIGHT_DECAY
    )

    def teacher_loss(row, noise, time, suffix_perturb=None):
        episode = episodes[row["source_id"]]
        prefix = interface.prefix_at(episode, row["decision"])
        z = row["z_state"][None].to(device)
        bias = lc.adarms_bias(z)
        chunk = row["selected_chunk"][None].to(device).float()
        if suffix_perturb is not None:
            chunk = chunk.clone()
            chunk[:, 10:] += suffix_perturb
        weight = 1.0 + min(max(row["advantage"], 0.0), 1.0)
        loss, _ = cached_branch_flow_loss(
            runner.policy, prefix, chunk, bias,
            torch.tensor([weight], device=device),
            max_executed=10, noise=noise, time=time,
        )
        return loss

    def rehearsal_loss(source, decision, noise, time,
                       suffix_perturb=None):
        episode = episodes[source]
        prefix = interface.prefix_at(episode, decision)
        z = z_cache[source][decision]
        bias = lc.adarms_bias(z)
        chunk = episode["chunks_norm"][decision][None].to(device).float()
        if suffix_perturb is not None:
            chunk = chunk.clone()
            chunk[:, 10:] += suffix_perturb
        return raw_flow_losses_from_prefix(
            runner.policy, chunk, bias, prefix, noise=noise, time=time
        ).mean()

    # -- startup regression: suffix credit ------------------------------------
    row0 = next(r for r in rows if r["selected"] != 0)
    noise0, time0 = item_noise_time(runner.policy, 10**6, device)
    perturb = 0.5 * torch.randn(1, 40, 7, device=device)

    def wz_grad(fn):
        optimizer.zero_grad(set_to_none=True)
        fn().backward()
        return lc.wz_out.weight.grad.detach().clone()

    g_base = wz_grad(lambda: teacher_loss(row0, noise0, time0))
    g_pert = wz_grad(
        lambda: teacher_loss(row0, noise0, time0, suffix_perturb=perturb)
    )
    assert torch.equal(g_base, g_pert), (
        "model-guided first-ten gradient depends on unexecuted suffix"
    )
    s0 = sources[0]
    r_base = wz_grad(lambda: rehearsal_loss(s0, early[s0][0], noise0, time0))
    r_pert = wz_grad(
        lambda: rehearsal_loss(
            s0, early[s0][0], noise0, time0, suffix_perturb=perturb
        )
    )
    assert not torch.equal(r_base, r_pert), (
        "rehearsal must receive gradient from positions 10-49"
    )
    print("suffix-credit regression PASSED", flush=True)
    optimizer.zero_grad(set_to_none=True)

    logs = []
    item_index = 0
    for epoch in range(EPOCHS):
        epoch_teacher, epoch_rehearsal = [], []
        for si, source in enumerate(sources):
            items = by_source[source]
            bins = TEACHERS_PER_SOURCE
            for k in range(bins):
                lo = (k * len(items)) // bins
                hi = max(((k + 1) * len(items)) // bins, lo + 1)
                row = items[lo + (epoch % (hi - lo))]
                noise, time = item_noise_time(
                    runner.policy, item_index, device
                )
                optimizer.zero_grad(set_to_none=True)
                losses = []
                if row["selected"] != 0:
                    t_loss = teacher_loss(row, noise, time)
                    losses.append(t_loss)
                    epoch_teacher.append(float(t_loss))
                rehearsal_source = sources[
                    (si + item_index) % len(sources)
                ]
                pool = early[rehearsal_source]
                decision = pool[item_index % len(pool)]
                noise2, time2 = item_noise_time(
                    runner.policy, 500_000 + item_index, device
                )
                r_loss = rehearsal_loss(
                    rehearsal_source, decision, noise2, time2
                )
                losses.append(r_loss)
                epoch_rehearsal.append(float(r_loss))
                total = torch.stack(losses).mean()
                total.backward()
                torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM)
                optimizer.step()
                item_index += 1
        logs.append(
            {
                "epoch": epoch,
                "teacher": sum(epoch_teacher)
                / max(len(epoch_teacher), 1),
                "rehearsal": sum(epoch_rehearsal)
                / max(len(epoch_rehearsal), 1),
                "n_teacher": len(epoch_teacher),
            }
        )
        print(
            f"[adapter epoch {epoch}] teacher={logs[-1]['teacher']:.4f} "
            f"({logs[-1]['n_teacher']} items) "
            f"rehearsal={logs[-1]['rehearsal']:.4f} "
            f"|wz_out|={float(lc.wz_out.weight.norm()):.4f}",
            flush=True,
        )

    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "wz_hidden": lc.wz_hidden.state_dict(),
            "wz_out": lc.wz_out.state_dict(),
            "wm_checkpoint_sha256": teacher["wm_checkpoint_sha256"],
            "teacher_sha256": hashlib.sha256(
                args.teacher.read_bytes()
            ).hexdigest(),
            "config": {
                "epochs": EPOCHS, "lr": LR, "seed": args.seed,
                "spec": "2026-07-28 M4 fast-pilot",
            },
        },
        args.output / "adapter.pt",
    )
    (args.output / "adapter_logs.json").write_text(json.dumps(logs))

    # -- fixed-state first-ten action shift (sanity, not a gate) --------------
    from lcwm.lc_flow import sample_chunks_lc
    from lcwm.sampler import sample_chunks

    shifts = []
    with torch.no_grad():
        for source in sources:
            items = by_source[source]
            row = items[len(items) // 2]
            episode = episodes[source]
            prefix = interface.prefix_at(episode, row["decision"])
            batch = interface.batch_at(episode, row["decision"])
            g = torch.Generator().manual_seed(4242)
            noise = torch.randn(
                1, runner.policy.config.chunk_size,
                runner.policy.config.max_action_dim, generator=g,
            ).to(device)
            stock = sample_chunks(
                runner.policy, batch, n=1, noise=noise, prefix=prefix
            )
            bias = lc.adarms_bias(row["z_state"][None].to(device))
            adapted = sample_chunks_lc(
                runner.policy, batch, bias, n=1, noise=noise,
                prefix=prefix,
            )
            delta = (adapted[0, :10] - stock[0, :10])
            shifts.append(
                {
                    "source_id": source,
                    "decision": row["decision"],
                    "l2": float(delta.norm()),
                    "mean_abs": float(delta.abs().mean()),
                    "max_abs": float(delta.abs().max()),
                }
            )
    (args.output / "action_shift.json").write_text(
        json.dumps(shifts, indent=2)
    )
    print(json.dumps(shifts, indent=2))
    print(f"-> {args.output}", flush=True)


if __name__ == "__main__":
    main()
