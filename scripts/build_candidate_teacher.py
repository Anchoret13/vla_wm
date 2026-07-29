#!/usr/bin/env python
"""A2 — model-generated candidate teacher (M3 / 2026-07-29 A2).

For every actionable late-chain train decision (2/3-complete, non-terminal,
train chain sources; expected 207): sample the fixed N=4 candidate set from
frozen π0.5 (candidate 0 = the exact cached stock chunk; 3 fresh seeded
samples), score each candidate's executed first-ten block with the canonical
q_score under the trained P4 world model, and select the highest-scoring
non-stock candidate ONLY when it beats stock.

Saves candidate_teacher.pt (seeds, chunks, recurrent states, checkpoint
hash, score components) + teacher_diagnostics.json (positive-teacher
fraction by source/decision time, advantage distribution, candidate action
diversity vs stock, per-component advantage contributions).
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

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

CACHE_DIR = Path(
    "/home/stargazer/Desktop/vla_wm/datasets/chain_episode_cache_v1"
)
N_CANDIDATES = 4
CANDIDATE_SEED_BASE = 888_000


def unroll_states(lc, episode, device):
    """z_t at every decision boundary of a full chain episode (no grad)."""
    hidden, mask = episode["prefix_hidden"], episode["prefix_mask"]
    actions = episode["action_block_norm"]
    lengths = episode["executed_lengths"]
    stride = int(episode["stride"])
    z = lc.posterior(hidden[0:1].to(device), mask[0:1].to(device))
    states = [z]
    for i in range(actions.shape[0]):
        am = (
            torch.arange(stride, device=device)[None]
            < int(lengths[i])
        )
        z = lc.step(
            z,
            actions[i : i + 1].to(device).float(),
            hidden[i + 1 : i + 2].to(device),
            mask[i + 1 : i + 2].to(device),
            am,
        )
        states.append(z)
    return states


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wm",
        type=Path,
        default=REPO_ROOT
        / "results" / "lc_flow_v04_e2e_v1" / "world_model"
        / "checkpoint_final.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "lc_flow_v04_e2e_v1",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)

    from lcwm.chassis import Pi05Runner
    from lcwm.p4_train import build_models, strict_load
    from lcwm.policy_interface import ReplayPolicyInterface
    from lcwm.value_heads import GAMMA_DECISION, q_score

    lc, predictor, ema, heads = build_models(device)
    bundle = torch.load(args.wm, weights_only=False)
    strict_load(bundle, lc, predictor, ema, heads)
    lc.eval()
    heads.eval()
    wm_hash = hashlib.sha256(args.wm.read_bytes()).hexdigest()

    runner = Pi05Runner(suite_name="libero_10")
    interface = ReplayPolicyInterface(runner)

    rows = []
    global_index = 0
    for path in sorted(CACHE_DIR.glob("*.pt")):
        episode = torch.load(path, weights_only=False)
        if episode["split"] != "train":
            continue
        states = unroll_states(lc, episode, device)
        bits = episode["predicate_bits"]
        n_actions = episode["action_block_norm"].shape[0]
        for decision in range(n_actions):
            if int(bits[decision].float().sum()) != 2:
                continue
            seed = CANDIDATE_SEED_BASE + global_index
            candidates = interface.candidates_at(
                episode, decision, n=N_CANDIDATES - 1, seed=seed
            )
            stock = episode["chunks_norm"][decision][None].to(
                candidates.device
            )
            pool = torch.cat([stock, candidates])  # candidate 0 = stock
            first10 = pool[:, :10].float().to(device)
            z = states[decision].expand(pool.shape[0], -1, -1)
            prior = lc.transition(z, first10)
            vh = heads(prior)
            scores = q_score(vh["r_hat"], vh["dphi_hat"], vh["v_hat"])
            stock_score = float(scores[0])
            best = int(scores[1:].argmax()) + 1
            advantage = float(scores[best]) - stock_score
            selected = best if advantage > 0 else 0
            diversity = float(
                (pool[1:, :10].to(device) - first10[0:1])
                .norm(dim=-1).mean()
            )
            rows.append(
                {
                    "source_id": episode["source_trajectory_id"],
                    "decision": decision,
                    "candidate_seed": seed,
                    "scores": scores.cpu(),
                    "r_hat": vh["r_hat"].cpu(),
                    "dphi_hat": vh["dphi_hat"].cpu(),
                    "v_next_hat": (GAMMA_DECISION * vh["v_hat"]).cpu(),
                    "selected": selected,
                    "advantage": advantage if selected else 0.0,
                    "selected_chunk": pool[selected].cpu(),
                    "stock_chunk": pool[0].cpu(),
                    "z_state": states[decision][0].cpu(),
                    "diversity_l2": diversity,
                }
            )
            global_index += 1
        print(
            f"[teacher] {episode['source_trajectory_id'][-14:]}: "
            f"{sum(1 for r in rows if r['source_id']==episode['source_trajectory_id'])} decisions",
            flush=True,
        )
    positive = [r for r in rows if r["selected"] != 0]
    diagnostics = {
        "n_decisions": len(rows),
        "wm_checkpoint_sha256": wm_hash,
        "positive_teacher_fraction": len(positive) / max(len(rows), 1),
        "positive_by_source": {
            source: sum(
                1 for r in positive if r["source_id"] == source
            )
            / max(sum(1 for r in rows if r["source_id"] == source), 1)
            for source in {r["source_id"] for r in rows}
        },
        "advantage_quantiles": (
            torch.tensor([r["advantage"] for r in positive])
            .quantile(torch.tensor([0.1, 0.5, 0.9]))
            .tolist()
            if positive
            else None
        ),
        "mean_candidate_diversity_l2": float(
            torch.tensor([r["diversity_l2"] for r in rows]).mean()
        ),
        "advantage_components_mean_on_positive": (
            {
                "r": float(torch.tensor([
                    float(r["r_hat"][r["selected"]] - r["r_hat"][0])
                    for r in positive
                ]).mean()),
                "dphi": float(torch.tensor([
                    float(r["dphi_hat"][r["selected"]] - r["dphi_hat"][0])
                    for r in positive
                ]).mean()),
                "gamma_v_next": float(torch.tensor([
                    float(
                        r["v_next_hat"][r["selected"]]
                        - r["v_next_hat"][0]
                    )
                    for r in positive
                ]).mean()),
            }
            if positive
            else None
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"schema": "candidate_teacher_v1", "rows": rows,
         "wm_checkpoint_sha256": wm_hash},
        args.output / "candidate_teacher.pt",
    )
    (args.output / "teacher_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2)
    )
    print(json.dumps(diagnostics, indent=2))
    print(f"-> {args.output / 'candidate_teacher.pt'}")


if __name__ == "__main__":
    main()
