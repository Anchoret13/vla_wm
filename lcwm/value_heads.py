"""P3 — separated reward/progress/terminal/value heads and targets (v0.4 §4).

The legacy `ret` output must not stand for reward, return, and success at
once (audit correction; framework §4.2). Five heads, one documented target
each, all consuming the predicted prior z̄ = T(z, a):

- `r_hat`    one-block task reward. Target: goal-atom completion delta
             sum(bits_{t+1}) − sum(bits_t), from live predicate labels.
- `p_hat`    predicate fraction (NOT phase-aware — Codex mainline review
             2026-07-27). Target: fraction of goal atoms true at t+1. Kept
             as a coarse task-progress signal.
- `dphi_hat` phase-aware progress change ΔΦ under the candidate block —
             the A1 lexicographic potential (lcwm/phase_potential.py);
             these are exactly the ΔΦ semantics the P5 score consumes.
- `term`     terminal/success probability at t+1. Target: the stored
             terminal/success flag (real label variation exists in demo
             episodes and in π0 chain episodes via atom flips).
- `v_hat`    V^{π0}: discounted atom-completion return-to-go along
             π0-GENERATED episodes only (registered 2026-07-25; generating
             policy must be `pi05_full_prompt_frozen`; expert-demo returns
             are a separately-labeled diagnostic, never V^{π0} targets).

The ONLY model score (A1 registration; stale r̂+γ^c·V̂ superseded):

    Q̂_score = r̂ + λ_Φ·ΔΦ̂ + γ_decision·V̂,  γ_decision = 0.99, λ_Φ = 1.0

implemented once in `q_score` — every consumer must call it.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

GAMMA_DECISION = 0.99
LAMBDA_PHI = 1.0
PI0_POLICY_ID = "pi05_full_prompt_frozen"


def q_score(
    r_hat: Tensor,
    dphi_hat: Tensor,
    v_hat_next: Tensor,
    lambda_phi: float = LAMBDA_PHI,
    gamma_decision: float = GAMMA_DECISION,
) -> Tensor:
    """Canonical P5 model score — the single implementation all consumers
    must use: Q̂ = r̂ + λ_Φ·ΔΦ̂ + γ_decision·V̂(T(z,u))."""
    return r_hat + lambda_phi * dphi_hat + gamma_decision * v_hat_next


class ValueHeads(nn.Module):
    """Five scalar heads over the pooled predicted prior [B, M, d_z]."""

    def __init__(self, d_z: int = 384, hidden: int = 256):
        super().__init__()

        def head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(d_z, hidden), nn.GELU(), nn.Linear(hidden, 1)
            )

        self.reward = head()
        self.predicate_fraction = head()
        self.phase_delta = head()
        self.terminal = head()
        self.value = head()

    def forward(self, z_bar: Tensor) -> dict[str, Tensor]:
        pooled = z_bar.mean(dim=1)
        return {
            "r_hat": self.reward(pooled).squeeze(-1),
            "p_hat": self.predicate_fraction(pooled).squeeze(-1),
            "dphi_hat": self.phase_delta(pooled).squeeze(-1),
            "term_logit": self.terminal(pooled).squeeze(-1),
            "v_hat": self.value(pooled).squeeze(-1),
        }


def reward_target(bits_t: Tensor, bits_next: Tensor) -> Tensor:
    """One-block atom-completion delta (can be negative if an atom un-sets)."""
    return bits_next.float().sum(-1) - bits_t.float().sum(-1)


def predicate_fraction_target(bits_next: Tensor) -> Tensor:
    """Coarse task progress: fraction of goal atoms true. This is NOT the
    phase-aware progress — ΔΦ targets come from
    `lcwm.phase_potential.delta_phi_targets` (A1)."""
    return bits_next.float().mean(-1)


def value_target_mc(
    sidecar_bits: Tensor,
    decision_index: int,
    gamma: float = GAMMA_DECISION,
) -> float:
    """Discounted atom-completion return-to-go at a decision boundary of a
    π0 episode: Σ_k γ^k · (Δ atoms completed at decision t+k)."""
    counts = sidecar_bits.float().sum(-1)
    deltas = counts[1:] - counts[:-1]
    value = 0.0
    for offset in range(decision_index, deltas.shape[0]):
        value += (gamma ** (offset - decision_index)) * float(deltas[offset])
    return value


def episode_value_targets(episode: dict, gamma: float = GAMMA_DECISION):
    """[R] V^{π0} targets for a labeled chain episode, NaN where invalid.

    Refuses non-π0 generating policies: expert-demo returns must never be
    presented as V^{π0} targets.
    """
    R = episode["prefix_hidden"].shape[0]
    targets = torch.full((R,), float("nan"))
    if (
        not episode.get("has_labels")
        or episode.get("generating_policy") != PI0_POLICY_ID
        or "sidecar_bits" not in episode
    ):
        return targets
    mask = episode["label_mask"]
    for index in range(R):
        if bool(mask[index]):
            targets[index] = value_target_mc(
                episode["sidecar_bits"], index, gamma
            )
    return targets
