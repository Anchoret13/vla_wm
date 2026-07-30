"""LC-Flow v0.5-gated — factorized state per the H7.3 spec (2026-07-29).

State (equations from the registered spec):
    c_t      = U_c(H_early, H_late, q_t)          non-recurrent task path
    w̄_t     = T_w(w_{t-1}, a_{t-1})              physical memory prior
    w_t      = U_w(w̄_t, H_early, q_t)            language-independent
    ḡ_t     = T_g(g_{t-1}, w_t, a_{t-1})         task memory prior
    g_t      = U_g(ḡ_t, w_t, H_late)             language-conditioned
Language reaches c/g only through the real-prompt H_late; w never sees it.

Safe policy interface:
    b_t = W_c·Pool(c_t) + tanh(α_w)·W_w·Pool(w_t) + tanh(α_g)·W_g·Pool(g_t)
with α_w = α_g = 0 at init — the memory paths are an exact no-op and must
earn influence relative to the H4-supported current-only path.

Heads: physical effects (Δq, Δobj) from the w-path prior; public subgoal
logits, reward, ΔQ_public and V from (c,g); paraphrase consistency is a
training-time loss on g (same-goal paraphrases agree; compatible distinct
goals may differ).

Token budget preserved from v0.4: 2 physical + 2 task tokens, d_z = 384.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class V05Config:
    d_early: int = 2048
    d_late: int = 2048
    d_q: int = 9
    d_z: int = 384
    physical_tokens: int = 2
    task_tokens: int = 2
    heads: int = 6
    update_blocks: int = 2
    action_horizon: int = 10
    action_dim: int = 7
    max_objects: int = 8
    max_subgoals: int = 7
    expert_width: int = 1024


class CrossUpdate(nn.Module):
    """Queries update by cross-attending into a keyed context (pre-LN)."""

    def __init__(self, cfg: V05Config, d_context: int):
        super().__init__()
        self.proj = nn.Linear(d_context, cfg.d_z)
        self.blocks = nn.ModuleList()
        for _ in range(cfg.update_blocks):
            self.blocks.append(nn.ModuleDict({
                "norm_q": nn.LayerNorm(cfg.d_z),
                "norm_kv": nn.LayerNorm(cfg.d_z),
                "attn": nn.MultiheadAttention(
                    cfg.d_z, cfg.heads, batch_first=True),
                "norm_mlp": nn.LayerNorm(cfg.d_z),
                "mlp": nn.Sequential(
                    nn.Linear(cfg.d_z, 4 * cfg.d_z), nn.GELU(),
                    nn.Linear(4 * cfg.d_z, cfg.d_z)),
            }))
        self.out_norm = nn.LayerNorm(cfg.d_z)

    def forward(self, z: Tensor, context: Tensor,
                mask: Tensor | None = None) -> Tensor:
        kv = self.proj(context.to(self.proj.weight.dtype))
        key_padding = None if mask is None else ~mask.bool()
        for block in self.blocks:
            q = block["norm_q"](z)
            attended, _ = block["attn"](
                q, block["norm_kv"](kv), block["norm_kv"](kv),
                key_padding_mask=key_padding, need_weights=False)
            z = z + attended
            z = z + block["mlp"](block["norm_mlp"](z))
        return self.out_norm(z)


class ActionTransition(nn.Module):
    """T(z, a): tokens + one encoded-action token through a small encoder."""

    def __init__(self, cfg: V05Config, extra_tokens: int = 0):
        super().__init__()
        a_in = cfg.action_horizon * (cfg.action_dim + 1)
        self.a_enc = nn.Sequential(
            nn.Linear(a_in, cfg.d_z), nn.GELU(),
            nn.Linear(cfg.d_z, cfg.d_z))
        layer = nn.TransformerEncoderLayer(
            cfg.d_z, cfg.heads, 4 * cfg.d_z, batch_first=True,
            norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.type_emb = nn.Parameter(
            torch.zeros(2 + extra_tokens, cfg.d_z))
        self.out_norm = nn.LayerNorm(cfg.d_z)
        self.cfg = cfg

    def forward(self, z: Tensor, action: Tensor,
                extra: Tensor | None = None,
                action_mask: Tensor | None = None) -> Tensor:
        cfg = self.cfg
        b, n, _ = z.shape
        if action_mask is None:
            action_mask = torch.ones(
                b, cfg.action_horizon, dtype=torch.bool,
                device=action.device)
        a = torch.cat(
            [action * action_mask[..., None],
             action_mask[..., None].float()], dim=-1)
        a_token = self.a_enc(a.flatten(1))[:, None]
        seq = [z + self.type_emb[0], a_token + self.type_emb[1]]
        if extra is not None:
            seq.append(extra + self.type_emb[2])
        out = self.enc(torch.cat(seq, dim=1))
        return self.out_norm(out[:, :n])


class V05State(nn.Module):
    def __init__(self, cfg: V05Config | None = None):
        super().__init__()
        cfg = cfg or V05Config()
        self.cfg = cfg
        self.c0 = nn.Parameter(torch.randn(cfg.task_tokens, cfg.d_z) * 0.02)
        self.w0 = nn.Parameter(
            torch.randn(cfg.physical_tokens, cfg.d_z) * 0.02)
        self.g0 = nn.Parameter(torch.randn(cfg.task_tokens, cfg.d_z) * 0.02)
        self.q_proj = nn.Linear(cfg.d_q, cfg.d_z)
        # current path: H_early ++ H_late ++ q
        self.u_c_early = CrossUpdate(cfg, cfg.d_early)
        self.u_c_late = CrossUpdate(cfg, cfg.d_late)
        # physical memory
        self.t_w = ActionTransition(cfg)
        self.u_w = CrossUpdate(cfg, cfg.d_early)
        # task memory (w enters as extra token context)
        self.t_g = ActionTransition(cfg, extra_tokens=1)
        self.u_g_w = CrossUpdate(cfg, cfg.d_z)
        self.u_g_late = CrossUpdate(cfg, cfg.d_late)

        # Safe gated policy interface.
        self.w_c = nn.Linear(cfg.d_z, cfg.expert_width)
        self.w_w = nn.Linear(cfg.d_z, cfg.expert_width)
        self.w_g = nn.Linear(cfg.d_z, cfg.expert_width)
        nn.init.zeros_(self.w_c.weight)
        nn.init.zeros_(self.w_c.bias)
        self.alpha_w = nn.Parameter(torch.zeros(()))
        self.alpha_g = nn.Parameter(torch.zeros(()))

        d = cfg.d_z

        def head(out_dim):
            return nn.Sequential(
                nn.Linear(d, 256), nn.GELU(), nn.Linear(256, out_dim))

        self.h_dq = head(cfg.d_q)
        self.h_dobj = head(cfg.max_objects * 3)
        nn.init.zeros_(self.h_dq[-1].weight)
        nn.init.zeros_(self.h_dq[-1].bias)
        nn.init.zeros_(self.h_dobj[-1].weight)
        nn.init.zeros_(self.h_dobj[-1].bias)
        self.h_subgoal = head(cfg.max_subgoals)
        self.h_reward = head(1)
        self.h_dq_public = head(1)
        self.h_value = head(1)

    def initial(self, batch: int, device) -> tuple[Tensor, Tensor]:
        return (self.w0[None].expand(batch, -1, -1).contiguous().to(device),
                self.g0[None].expand(batch, -1, -1).contiguous().to(device))

    def current(self, h_early: Tensor, h_late: Tensor,
                late_mask: Tensor | None, q: Tensor) -> Tensor:
        b = h_late.shape[0]
        c = self.c0[None].expand(b, -1, -1)
        c = c + self.q_proj(q)[:, None]
        c = self.u_c_early(c, h_early)
        return self.u_c_late(c, h_late, late_mask)

    def step_physical(self, w_prev: Tensor, action: Tensor,
                      h_early: Tensor, q: Tensor,
                      action_mask: Tensor | None = None) -> Tensor:
        w_bar = self.t_w(w_prev, action, action_mask=action_mask)
        w = w_bar + self.q_proj(q)[:, None]
        return self.u_w(w, h_early)

    def step_task(self, g_prev: Tensor, w: Tensor, action: Tensor,
                  h_late: Tensor, late_mask: Tensor | None,
                  action_mask: Tensor | None = None) -> Tensor:
        w_summary = w.mean(dim=1, keepdim=True)
        g_bar = self.t_g(g_prev, action, extra=w_summary,
                         action_mask=action_mask)
        g = self.u_g_w(g_bar, w)
        return self.u_g_late(g, h_late, late_mask)

    def policy_bias(self, c: Tensor, w: Tensor, g: Tensor) -> Tensor:
        return (
            self.w_c(c.mean(dim=1))
            + torch.tanh(self.alpha_w) * self.w_w(w.mean(dim=1))
            + torch.tanh(self.alpha_g) * self.w_g(g.mean(dim=1))
        )

    def physical_prior(self, w: Tensor, action: Tensor,
                       action_mask: Tensor | None = None) -> Tensor:
        return self.t_w(w, action, action_mask=action_mask)

    def outcomes(self, w_prior: Tensor, c: Tensor, g: Tensor) -> dict:
        w_pool = w_prior.mean(dim=1)
        task_pool = torch.cat([c, g], dim=1).mean(dim=1)
        return {
            "d_q": self.h_dq(w_pool),
            "d_obj": self.h_dobj(w_pool).view(
                -1, self.cfg.max_objects, 3),
            "subgoal_logits": self.h_subgoal(task_pool),
            "r_hat": self.h_reward(task_pool).squeeze(-1),
            "dq_public_hat": self.h_dq_public(task_pool).squeeze(-1),
            "v_hat": self.h_value(task_pool).squeeze(-1),
        }
