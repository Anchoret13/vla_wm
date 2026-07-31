"""LC-Flow v0.6 — the faithful language-conditioned predictive state
(2026-07-30 non-negotiable implementation graph).

    z̄_t = T(z_{t-c}, E_a(u_{t-c:t}))          action-conditioned prior
    z_t  = U(z̄_t, h_t) = LN(A(h_t) + β·tanh R(A(h_t), z̄_t))   β=0.25 fixed
    z̃_i = T(z_t, E_a(u_i))                    candidate prediction

One state. h_t is the full real-prompt π0.5 prefix feature (visual +
language tokens, one stream). Candidate outputs decode ONLY from z̃ via
D_next; D_current(z_t) grounds current automaton/history state and is
never an input to D_next. The policy interface is
b_t = η·tanh(W_z·Pool(z_t)) with W_z = 0 at init and η fixed — exact
stock parity at start, direct gradient, no learnable gate.

Forbidden by construction (asserted in scripts/test_v06_mechanical.py):
D_next sees no h, no z_t, no next observation/posterior, no D_current
features, no action bypass around E_a→T.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class V06Config:
    d_h: int = 2048
    d_z: int = 384
    tokens: int = 4
    heads: int = 6
    action_horizon: int = 10
    action_dim: int = 7
    max_objects: int = 8
    max_subgoals: int = 7
    horizons: tuple[int, ...] = (10, 30, 60, 100)
    beta: float = 0.25
    eta: float = 1.0
    expert_width: int = 1024
    ema_decay: float = 0.995


class ActionEncoder(nn.Module):
    """E_a: masked executed-action block → one token."""

    def __init__(self, cfg: V06Config):
        super().__init__()
        d_in = cfg.action_horizon * (cfg.action_dim + 1)
        self.net = nn.Sequential(
            nn.Linear(d_in, cfg.d_z), nn.GELU(),
            nn.Linear(cfg.d_z, cfg.d_z))
        self.cfg = cfg

    def forward(self, action: Tensor, mask: Tensor | None = None) -> Tensor:
        cfg = self.cfg
        b = action.shape[0]
        if mask is None:
            mask = torch.ones(b, cfg.action_horizon, dtype=torch.bool,
                              device=action.device)
        x = torch.cat([action * mask[..., None],
                       mask[..., None].float()], dim=-1)
        return self.net(x.flatten(1))[:, None]


class Transition(nn.Module):
    """T: transformer over [z tokens ‖ action token] → next z tokens."""

    def __init__(self, cfg: V06Config):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            cfg.d_z, cfg.heads, 4 * cfg.d_z, dropout=0.0,
            batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.type_emb = nn.Parameter(torch.zeros(2, cfg.d_z))
        self.out_norm = nn.LayerNorm(cfg.d_z)
        self.cfg = cfg

    def forward(self, z: Tensor, a_token: Tensor) -> Tensor:
        n = z.shape[1]
        seq = torch.cat([z + self.type_emb[0], a_token + self.type_emb[1]],
                        dim=1)
        return self.out_norm(self.enc(seq)[:, :n])


class CrossBlock(nn.Module):
    def __init__(self, cfg: V06Config, d_kv: int):
        super().__init__()
        self.proj = nn.Linear(d_kv, cfg.d_z)
        self.norm_q = nn.LayerNorm(cfg.d_z)
        self.norm_kv = nn.LayerNorm(cfg.d_z)
        self.attn = nn.MultiheadAttention(cfg.d_z, cfg.heads, dropout=0.0,
                                          batch_first=True)
        self.norm_mlp = nn.LayerNorm(cfg.d_z)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_z, 4 * cfg.d_z), nn.GELU(),
                                 nn.Linear(4 * cfg.d_z, cfg.d_z))

    def forward(self, q: Tensor, kv: Tensor,
                mask: Tensor | None = None) -> Tensor:
        kv = self.proj(kv.to(self.proj.weight.dtype))
        key_padding = None if mask is None else ~mask.bool()
        attended, _ = self.attn(self.norm_q(q), self.norm_kv(kv),
                                self.norm_kv(kv),
                                key_padding_mask=key_padding,
                                need_weights=False)
        q = q + attended
        return q + self.mlp(self.norm_mlp(q))


class Anchor(nn.Module):
    """A(h): learned queries cross-attend into the prompt-conditioned
    feature sequence → current-observation anchor tokens."""

    def __init__(self, cfg: V06Config):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(cfg.tokens, cfg.d_z) * 0.02)
        self.blocks = nn.ModuleList(
            [CrossBlock(cfg, cfg.d_h) for _ in range(2)])
        self.out_norm = nn.LayerNorm(cfg.d_z)

    def forward(self, h: Tensor, mask: Tensor | None = None) -> Tensor:
        q = self.queries[None].expand(h.shape[0], -1, -1)
        for block in self.blocks:
            q = block(q, h, mask)
        return self.out_norm(q)


class Residual(nn.Module):
    """R(c, z̄): bounded history correction; zero-initialized output so the
    state is exactly the anchored current observation at init."""

    def __init__(self, cfg: V06Config):
        super().__init__()
        self.block = CrossBlock(cfg, cfg.d_z)
        self.out = nn.Linear(cfg.d_z, cfg.d_z)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, c: Tensor, z_bar: Tensor) -> Tensor:
        return self.out(self.block(c, z_bar))


def head(d_in: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, 256), nn.GELU(),
                         nn.Linear(256, out_dim))


class DCurrent(nn.Module):
    """Current automaton/history grounding ONLY."""

    def __init__(self, cfg: V06Config):
        super().__init__()
        d = cfg.d_z
        self.valid_bits = head(d, cfg.max_subgoals)
        self.event_bits = head(d, cfg.max_subgoals)
        self.ordered_prefix = head(d, 1)

    def forward(self, z: Tensor) -> dict:
        p = z.mean(dim=1)
        return {
            "valid_bits": self.valid_bits(p),
            "event_bits": self.event_bits(p),
            "ordered_prefix": self.ordered_prefix(p).squeeze(-1),
        }


class DNext(nn.Module):
    """Every candidate-dependent output; consumes pooled z̃ ONLY."""

    def __init__(self, cfg: V06Config):
        super().__init__()
        d = cfg.d_z
        # NOTE: physical heads use standard init — the registered launch
        # assertion requires their gradients to reach E_a/T at init, which
        # a zero-initialized output layer would block. Only W_z (policy)
        # and R's output (state residual) are zero-initialized in v0.6.
        self.d_q = head(d, 9)
        self.d_obj = head(d, cfg.max_objects * 3)
        self.valid_bits = head(d, cfg.max_subgoals)
        self.event_bits = head(d, cfg.max_subgoals)
        self.flips_01 = head(d, cfg.max_subgoals)
        self.flips_10 = head(d, cfg.max_subgoals)
        self.damage = head(d, 1)
        self.reward = head(d, 1)
        self.success = head(d, 1)
        self.p_valid = head(d, 1)
        self.q_valid = head(d, len(cfg.horizons))
        self.tau_next = head(d, 1)
        self.score = head(d, 1)
        self.cfg = cfg

    def forward(self, z_tilde: Tensor) -> dict:
        p = z_tilde.mean(dim=1)
        return {
            "d_q": self.d_q(p),
            "d_obj": self.d_obj(p).view(-1, self.cfg.max_objects, 3),
            "valid_bits": self.valid_bits(p),
            "event_bits": self.event_bits(p),
            "flips_01": self.flips_01(p),
            "flips_10": self.flips_10(p),
            "damage": self.damage(p).squeeze(-1),
            "reward": self.reward(p).squeeze(-1),
            "success_logit": self.success(p).squeeze(-1),
            "p_valid": self.p_valid(p).squeeze(-1),
            "q_valid": self.q_valid(p),
            "tau_next": self.tau_next(p).squeeze(-1),
            "s": self.score(p).squeeze(-1),
        }


class V06State(nn.Module):
    def __init__(self, cfg: V06Config | None = None):
        super().__init__()
        cfg = cfg or V06Config()
        self.cfg = cfg
        self.z0 = nn.Parameter(torch.randn(cfg.tokens, cfg.d_z) * 0.02)
        self.e_a = ActionEncoder(cfg)
        self.t = Transition(cfg)
        self.anchor = Anchor(cfg)
        self.r = Residual(cfg)
        self.norm = nn.LayerNorm(cfg.d_z)
        self.d_current = DCurrent(cfg)
        self.d_next = DNext(cfg)
        self.w_z = nn.Linear(cfg.d_z, cfg.expert_width)
        nn.init.zeros_(self.w_z.weight)
        nn.init.zeros_(self.w_z.bias)

    # ---- state ----------------------------------------------------------
    def update(self, z_bar: Tensor, h: Tensor,
               h_mask: Tensor | None = None) -> Tensor:
        c = self.anchor(h, h_mask)
        return self.norm(c + self.cfg.beta * torch.tanh(self.r(c, z_bar)))

    def null_action(self, batch: int, device) -> tuple[Tensor, Tensor]:
        cfg = self.cfg
        return (torch.zeros(batch, cfg.action_horizon, cfg.action_dim,
                            device=device),
                torch.zeros(batch, cfg.action_horizon, dtype=torch.bool,
                            device=device))

    def initial_state(self, h: Tensor,
                      h_mask: Tensor | None = None) -> Tensor:
        """U(T(z_∅, a_∅), h) — reset mode; equals recurrent at episode
        start by construction."""
        b = h.shape[0]
        a0, m0 = self.null_action(b, h.device)
        z0 = self.z0[None].expand(b, -1, -1)
        return self.update(self.t(z0, self.e_a(a0, m0)), h, h_mask)

    def step(self, z_prev: Tensor, action: Tensor, h: Tensor,
             h_mask: Tensor | None = None,
             action_mask: Tensor | None = None) -> Tensor:
        return self.update(
            self.t(z_prev, self.e_a(action, action_mask)), h, h_mask)

    def predict(self, z: Tensor, action: Tensor,
                action_mask: Tensor | None = None) -> Tensor:
        """z̃_i = T(z, E_a(u_i)) — the ONLY path from a candidate action to
        any candidate-dependent output."""
        return self.t(z, self.e_a(action, action_mask))

    # ---- policy interface ----------------------------------------------
    def policy_bias(self, z: Tensor) -> Tensor:
        return self.cfg.eta * torch.tanh(self.w_z(z.mean(dim=1)))


def make_ema(model: V06State) -> V06State:
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


@torch.no_grad()
def ema_update(ema: V06State, model: V06State,
               decay: float | None = None) -> None:
    decay = decay if decay is not None else model.cfg.ema_decay
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.lerp_(pm.detach(), 1.0 - decay)
    for be, bm in zip(ema.buffers(), model.buffers()):
        be.copy_(bm)
