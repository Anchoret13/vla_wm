"""V7.5 Stage 1 — the repaired recurrent state (plan_and_progress/archive/daily/2026-08-09.md V7.5).

Stage 0 (`scripts/diag_v075_recurrence.py`) localized the V7.3C
"functionally inert recurrence" to TWO independent sites in
`V06State.update`:

    c     = Anchor(h)
    z_bar = T(z_prev, E_a(a))
    state = LayerNorm(c + beta * tanh(R(c, z_bar))),   beta = 0.25

  upstream    `attended` is 5.19x larger than `c` -- history is not
              quantitatively small, it is nearly CONSTANT. Swapping in
              another anchor's history moves it by 4.8e-06 relative,
              because raw `z` states are near-degenerate (the same
              common-mode pathology V7.4A centering repaired on the
              policy side) so `proj(z_bar)` hands the cross-attention
              near-identical keys and values.

  downstream  `abs(R_pre)` exceeds 1 at 100% of units at every anchor
              (median max 28.77): tanh is pinned at +/-1, annihilating
              the 4.8e-06 that survived, AND killing the gradient
              (dtanh ~ 0), so the path is frozen rather than unused.

Measured consequence: observation/history output-gain ratio 1.24e+07.
Sweeping beta over {0.25..256} or rescaling `R.out` over {1..256} does
NOT recover usable gain -- both scale a CONSTANT saturated vector. The
registered sub-choice is therefore
`desaturate_and_reinject__scalar_fixes_insufficient`.

This module implements the two architecture parts of that correction
and nothing else. The objective part (framework_design.md 3.2 short
latent rollout) lives in the V7.5C trainer.

  part 1  `r_norm`: LayerNorm on R's output before the bound, so the
          path operates in its responsive regime and passes gradient.
          It also strips the constant component that made the history
          term a fixed 24%-of-||c|| offset.
  part 2  `hist_center`: running centering of the history channel, so
          near-degenerate raw states become separable attention keys.
          Mirrors the V4.7A policy-side repair exactly -- a mean VECTOR
          over d_z and a SCALAR population RMS -- and is frozen into
          the checkpoint and used at eval, never recomputed online.

Everything else -- Anchor, Transition, ActionEncoder, the heads, beta,
eta, the policy_bias path -- is inherited from V06State unchanged, so
the flow boundary and every downstream consumer are untouched.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from lcwm.v06_model import V06Config, V06State


class HistoryCentering(nn.Module):
    """Centering of the history channel, frozen at save.

    Statistic shape follows the registered V7.4A policy-side rule: a
    mean VECTOR over the feature dim and a SCALAR RMS of the centered
    values. A per-sample LayerNorm cannot substitute -- it removes each
    sample's own mean, not the mean SHARED across samples, which is
    exactly the common mode that destroys contrast here.

    Updated from detached inputs while training and unfrozen; once
    `freeze()` is called (at checkpoint save) the buffers are constant,
    so inference is a pure function of the frozen statistics.
    """

    def __init__(self, d_z: int, momentum: float = 0.01) -> None:
        super().__init__()
        self.momentum = momentum
        self.register_buffer("mu", torch.zeros(d_z))
        self.register_buffer("sigma", torch.ones(()))
        self.register_buffer("n_updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("frozen", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def observe(self, z: Tensor) -> None:
        """One EMA step from a detached (batch, tokens, d_z) block."""
        if bool(self.frozen):
            return
        flat = z.detach().reshape(-1, z.shape[-1]).float()
        mu = flat.mean(dim=0)
        sigma = (flat - mu).pow(2).mean().sqrt().clamp_min(1e-6)
        if int(self.n_updates) == 0:      # first batch seeds the estimate
            self.mu.copy_(mu)
            self.sigma.copy_(sigma)
        else:
            m = self.momentum
            self.mu.mul_(1 - m).add_(mu, alpha=m)
            self.sigma.mul_(1 - m).add_(sigma, alpha=m)
        self.n_updates += 1

    @torch.no_grad()
    def fit(self, z: Tensor) -> None:
        """Set the statistics directly from a state block (used by the
        pre-training validation probe and by an explicit refit)."""
        flat = z.detach().reshape(-1, z.shape[-1]).float()
        self.mu.copy_(flat.mean(dim=0))
        self.sigma.copy_(
            (flat - self.mu).pow(2).mean().sqrt().clamp_min(1e-6))
        self.n_updates.fill_(1)

    @torch.no_grad()
    def freeze(self) -> None:
        self.frozen.fill_(True)

    def _snapshot(self) -> tuple[Tensor, Tensor]:
        """Detached CLONES of the running statistics.

        The clone is load-bearing, not defensive style. `observe()`
        updates `mu`/`sigma` with in-place ops, while `(z - mu) / sigma`
        saves the divisor for backward. A recurrent unroll observes once
        per decision, so by the time backward runs the saved buffer has
        been mutated many times and autograd raises "variable needed for
        gradient computation has been modified by an inplace operation".
        A fresh clone carries its own version counter and is immune.
        (Same failure class as the V7.3D in-place stock-head swap.)
        """
        return self.mu.detach().clone(), self.sigma.detach().clone()

    def apply_frozen(self, z: Tensor) -> Tensor:
        """Center WITHOUT observing. Used by losses and readouts that
        must not perturb the statistics they normalize by, and that
        must stay differentiable in `z`."""
        mu, sigma = self._snapshot()
        return (z - mu) / sigma

    def forward(self, z: Tensor) -> Tensor:
        # Snapshot BEFORE observing: a step is normalized by statistics
        # that do not already include that step's own contribution.
        mu, sigma = self._snapshot()
        if self.training:
            self.observe(z)
        return (z - mu) / sigma


class V075State(V06State):
    """V06State with the Stage 0 defects repaired. Same interface."""

    def __init__(self, cfg: V06Config | None = None) -> None:
        super().__init__(cfg)
        d_z = self.cfg.d_z
        self.hist_center = HistoryCentering(d_z)
        self.r_norm = nn.LayerNorm(d_z)

    def update(self, z_bar: Tensor, h: Tensor,
               h_mask: Tensor | None = None) -> Tensor:
        c = self.anchor(h, h_mask)
        z_c = self.hist_center(z_bar)
        return self.norm(
            c + self.cfg.beta * torch.tanh(self.r_norm(self.r(c, z_c))))

    # ---- lineage --------------------------------------------------------
    def centering_state(self) -> dict:
        """The frozen history statistics, for the run lineage."""
        return {"mu": self.hist_center.mu.clone(),
                "sigma": self.hist_center.sigma.clone(),
                "n_updates": int(self.hist_center.n_updates),
                "frozen": bool(self.hist_center.frozen)}


def load_v06_weights(model: V075State, sd: dict) -> tuple[list, list]:
    """Load a V06State state dict into a V075State.

    The two new modules have no V06 counterpart, so the load is
    non-strict BY CONSTRUCTION and the missing keys are asserted to be
    exactly those two. Any other missing/unexpected key is a real
    mismatch and raises.
    """
    missing, unexpected = model.load_state_dict(sd, strict=False)
    allowed = {"hist_center.mu", "hist_center.sigma",
               "hist_center.n_updates", "hist_center.frozen",
               "r_norm.weight", "r_norm.bias"}
    extra = set(missing) - allowed
    assert not extra, f"unexpected missing keys beyond the V7.5 additions: {sorted(extra)}"
    assert not unexpected, f"unexpected keys in source state dict: {sorted(unexpected)}"
    return list(missing), list(unexpected)
