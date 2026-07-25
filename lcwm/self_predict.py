"""P2 — recursive one-step latent self-prediction (framework v0.4 §3).

Machinery only; the training loop lives in scripts/train_self_predict.py.

- `EMATarget`: a no-grad EMA copy of the LCState *state path* (posterior /
  update / transition, including E_a). The target recurrence is carried along
  the observed episode: z⁺_0 = U_ema(z0, h_0); z⁺_{i+1} = U_ema(T_ema(z⁺_i,
  a_i), h_{i+1}) — the stored next-prefix feature is an INPUT to the EMA
  posterior target, not itself the target state (v0.4 §3.1).
- `LatentPredictor`: small per-token projection p_θ applied to the online
  predicted prior T_θ(z_i, a_i).
- `latent_distance`: default distance is per-token cosine distance
  (chosen default for the open [DEC] "exact EMA target projection and latent
  distance"; recorded, revisitable).
- `collapse_metrics`: per-dim std and effective rank (participation ratio)
  of both target and prediction batches.
- Trivial baselines (v0.4 Gate A): copy-state, ignore-action, mean-next.
"""

from __future__ import annotations

import copy

import torch
from torch import Tensor, nn

from lcwm.lc_flow import LCState


class EMATarget(nn.Module):
    """EMA copy of the LCState state path; never receives gradients."""

    def __init__(self, lc_state: LCState, tau: float = 0.996):
        super().__init__()
        self.tau = float(tau)
        self.z0 = nn.Parameter(
            lc_state.z0.detach().clone(), requires_grad=False
        )
        self.update_module = copy.deepcopy(lc_state.update)
        self.transition_module = copy.deepcopy(lc_state.transition)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def ema_update(self, lc_state: LCState) -> None:
        pairs = [
            (self.z0, lc_state.z0),
            *zip(
                self.update_module.parameters(),
                lc_state.update.parameters(),
            ),
            *zip(
                self.transition_module.parameters(),
                lc_state.transition.parameters(),
            ),
        ]
        for target, online in pairs:
            target.mul_(self.tau).add_(
                online.detach().to(target.dtype), alpha=1.0 - self.tau
            )
        buffer_pairs = [
            *zip(self.update_module.buffers(), lc_state.update.buffers()),
            *zip(
                self.transition_module.buffers(),
                lc_state.transition.buffers(),
            ),
        ]
        for target, online in buffer_pairs:
            target.copy_(online.detach())

    @torch.no_grad()
    def initial(self, batch: int) -> Tensor:
        return self.z0[None].expand(batch, -1, -1).contiguous()

    @torch.no_grad()
    def posterior(self, h: Tensor, h_mask: Tensor | None = None) -> Tensor:
        return self.update_module(self.initial(h.shape[0]), h, h_mask)

    @torch.no_grad()
    def step(
        self,
        z_prev: Tensor,
        action_block: Tensor,
        h: Tensor,
        h_mask: Tensor | None = None,
        action_mask: Tensor | None = None,
    ) -> Tensor:
        z_bar = self.transition_module(z_prev, action_block, action_mask)
        return self.update_module(z_bar, h, h_mask)


class LatentPredictor(nn.Module):
    """p_θ — per-token MLP over the predicted prior."""

    def __init__(self, d_z: int = 384, hidden: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_z, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_z),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


def latent_distance(
    prediction: Tensor, target: Tensor, kind: str = "cosine"
) -> Tensor:
    """Mean per-token distance between [B,M,d] prediction and target."""
    if kind == "cosine":
        return (
            1.0
            - torch.nn.functional.cosine_similarity(
                prediction, target, dim=-1
            )
        ).mean()
    if kind == "mse":
        return torch.nn.functional.mse_loss(prediction, target)
    raise ValueError(f"unknown latent distance {kind!r}")


@torch.no_grad()
def collapse_metrics(states: Tensor) -> dict[str, float]:
    """states: [N,M,d] batch of latent states (targets or predictions)."""
    flat = states.reshape(states.shape[0], -1).float()
    centered = flat - flat.mean(dim=0, keepdim=True)
    per_dim_std = centered.std(dim=0)
    if flat.shape[0] > 1:
        covariance_spectrum = torch.linalg.svdvals(centered) ** 2
        spectrum = covariance_spectrum / covariance_spectrum.sum().clamp_min(
            1e-12
        )
        effective_rank = float(
            torch.exp(-(spectrum * spectrum.clamp_min(1e-12).log()).sum())
        )
    else:
        effective_rank = 0.0
    return {
        "mean_per_dim_std": float(per_dim_std.mean()),
        "min_per_dim_std": float(per_dim_std.min()),
        "effective_rank": effective_rank,
    }


@torch.no_grad()
def trivial_baseline_distances(
    target_previous: Tensor,
    target_next: Tensor,
    prior_no_action: Tensor,
    kind: str = "cosine",
) -> dict[str, float]:
    """Gate-A baselines evaluated in the same distance as the model.

    - copy-state: predict z⁺_{i+1} = z⁺_i;
    - ignore-action: the online prior computed with a zeroed action block
      (caller supplies it, passed through the same predictor when scoring the
      model — here it is compared raw, a conservative choice);
    - mean-next: batch mean of all targets.
    """
    mean_next = target_next.mean(dim=0, keepdim=True).expand_as(target_next)
    return {
        "copy_state": float(
            latent_distance(target_previous, target_next, kind)
        ),
        "ignore_action": float(
            latent_distance(prior_no_action, target_next, kind)
        ),
        "mean_next": float(latent_distance(mean_next, target_next, kind)),
    }
