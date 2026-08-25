"""Reference-centered residual world model (Action 2M.5, M0.3).

M0.2 predicted absolute outcomes from absolute actions and came out
anti-correlated on held-out sibling ranking (effect corr -0.2275), overfitting
the action input by epoch 4.  In a local candidate pool most absolute-action
variation identifies the ANCHOR, while the useful signal is the small
within-anchor difference, so an absolute model can fit the former and never
learn the latter.

M0.3 removes that degree of freedom by construction:

    Y_hat(i)  = b(s) + Delta_hat(i)
    Delta_hat = h(z_s, u0, u_i - u0) - h(z_s, u0, 0)

`b(s)` and `z_s` come from M0.2's selected STATE-ONLY checkpoint and are frozen,
so the residual head sees only the reference-centered action difference.
`Delta_hat(reference) = 0` exactly, because `u_ref - u_ref = 0` makes the two
terms identical - the model cannot lose to the stock-reference predictor on the
reference itself.  The independent rank head is retired: the sibling score IS
the predicted consequence delta.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

RESIDUAL_HIDDEN = 64          # fixed by the action item; not a swept quantity


class ResidualWM(nn.Module):
    def __init__(self, frozen_state_only, z_dim: int, action_steps: int,
                 action_dim: int, physical_dim: int, continuous_dim: int,
                 hidden: int = RESIDUAL_HIDDEN):
        super().__init__()
        self.base = frozen_state_only
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.base.eval()
        self.physical_dim, self.continuous_dim = physical_dim, continuous_dim
        flat = action_steps * action_dim
        self.head = nn.Sequential(
            nn.LayerNorm(z_dim + 2 * flat),
            nn.Linear(z_dim + 2 * flat, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, physical_dim + continuous_dim))

    @torch.no_grad()
    def _z_and_base(self, state: Tensor, actions: Tensor) -> tuple[Tensor, dict]:
        """z_s and b(s) from the frozen state-only model.  Its action input is
        zeroed by construction (`use_action=False`), so both are functions of
        the state alone."""
        b = self.base
        s = b.state_mlp(state)
        a = b.action_mlp(torch.zeros_like(actions).flatten(1))
        z = b.transition_mlp(torch.cat([s, a], dim=-1))
        return z, {"physical_norm": b.physical_head(z),
                   "outcome_raw": b.outcome_head(z)}

    def _h(self, z: Tensor, u0: Tensor, du: Tensor) -> Tensor:
        return self.head(torch.cat([z, u0.flatten(1), du.flatten(1)], dim=-1))

    def forward(self, state: Tensor, actions: Tensor, ref_index: int) -> dict:
        z, base = self._z_and_base(state, actions)
        u0 = actions[ref_index].unsqueeze(0).expand_as(actions)
        du = actions - u0
        delta = self._h(z, u0, du) - self._h(z, u0, torch.zeros_like(du))
        return {"z": z, "base": base,
                "delta_physical": delta[:, :self.physical_dim],
                "delta_continuous": delta[:, self.physical_dim:]}


def residual_loss(pred: dict, tgt: dict, norm: dict, qi: int,
                  w_q: float = 1.0, w_phys: float = 0.5, w_aux: float = 0.25):
    """L = 1.0*SmoothL1(delta_QPhi) + 0.5*SmoothL1(delta_physical)
           + 0.25*SmoothL1(delta_auxiliary_consequences)

    Every target is a shared-seed candidate mean minus the matched reference
    mean.  There is no per-repeat classification label anywhere in this path.
    """
    f = nn.functional.smooth_l1_loss
    pc, tc = pred["delta_continuous"], tgt["delta_continuous"]
    aux = [i for i in range(tc.shape[-1]) if i != qi]
    l_q = f(pc[:, qi] / norm["cont"][qi], tc[:, qi] / norm["cont"][qi])
    l_p = f(pred["delta_physical"] / norm["phys"], tgt["delta_physical"] / norm["phys"])
    l_a = (f(pc[:, aux] / norm["cont"][aux], tc[:, aux] / norm["cont"][aux])
           if aux else pc.sum() * 0.0)
    total = w_q * l_q + w_phys * l_p + w_aux * l_a
    return total, {"q": l_q, "physical": l_p, "aux": l_a, "total": total}
