"""Minimal Candidate A — monolithic task-conditioned carry (framework v0.2 §4).

Fixed instrument for the §11.2 input-arm sweep. Oracle TASKID conditioning via
FiLM on carry queries; heads receive Z only (no e_task bypass, contract §1).

    Z_t   = Carry(Z_{t-1}, tokens_t, q_t, a_prev, e_task)   # (B, M, d)
    Ẑ_t+1 = Transition(Z_t, action_block_t, e_task)
    heads : predicate bits | progress | proprio anchor      # from Z only
"""

from __future__ import annotations

import copy

import torch
from torch import nn


class CrossAttnBlock(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.ln_q, self.ln_kv = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d))

    def forward(self, x, kv):
        a, _ = self.attn(self.ln_q(x), self.ln_kv(kv), self.ln_kv(kv))
        x = x + a
        return x + self.mlp(self.ln2(x))


class Encoder(nn.Module):
    """Token/proprio -> carry update (the EMA-target subject).

    v2 (review 2026-07-22 item 4): previous-action input REMOVED. The carry
    encodes observation/proprio history only; actions enter solely through the
    Transition, so action-shuffle sensitivity cannot come from the transition
    reproducing action-history content in the target.
    """

    def __init__(self, d_in=2048, d=384, m=4, d_e=256, blocks=2, heads=6):
        super().__init__()
        self.tok = nn.Linear(d_in, d)
        self.qp = nn.Linear(9, d)
        self.film = nn.Linear(d_e, 2 * d)
        self.blocks = nn.ModuleList(CrossAttnBlock(d, heads) for _ in range(blocks))
        self.init_carry = nn.Parameter(0.02 * torch.randn(m, d))

    def step(self, z_prev, tokens, q, e):
        kv = torch.cat([self.tok(tokens), self.qp(q)[:, None]], dim=1)
        g, b = self.film(e).chunk(2, -1)
        z = z_prev * (1 + g[:, None]) + b[:, None]
        for blk in self.blocks:
            z = blk(z, kv)
        return z

    def forward(self, tokens, q, e):
        """Teacher-forced chain over L steps -> (B, L, M, d)."""
        B, L = tokens.shape[:2]
        z = self.init_carry.expand(B, -1, -1)
        out = []
        for t in range(L):
            z = self.step(z, tokens[:, t], q[:, t], e)
            out.append(z)
        return torch.stack(out, dim=1)


class Transition(nn.Module):
    def __init__(self, d=384, d_e=256, layers=4, heads=6):
        super().__init__()
        self.ap = nn.Sequential(nn.Linear(70, d), nn.GELU(), nn.Linear(d, d))
        self.ep = nn.Linear(d_e, d)
        self.type_emb = nn.Parameter(0.02 * torch.randn(3, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers)

    def forward(self, z, action_block, e):
        m = z.shape[1]
        seq = torch.cat([z + self.type_emb[0],
                         self.ap(action_block.flatten(1))[:, None] + self.type_emb[1],
                         self.ep(e)[:, None] + self.type_emb[2]], dim=1)
        return self.enc(seq)[:, :m]


class CandidateA(nn.Module):
    def __init__(self, d_in=2048, d=384, m=4, d_e=256, n_tasks=10,
                 max_atoms=3, ema_tau=0.996):
        super().__init__()
        self.e_task = nn.Embedding(n_tasks, d_e)
        self.encoder = Encoder(d_in, d, m)
        self.transition = Transition(d, d_e)
        self.head_pred = nn.Sequential(nn.Linear(m * d, 128), nn.GELU(),
                                       nn.Linear(128, max_atoms))
        self.head_prog = nn.Sequential(nn.Linear(m * d, 64), nn.GELU(),
                                       nn.Linear(64, 1))
        self.head_q = nn.Linear(m * d, 9)
        self.ema_encoder = copy.deepcopy(self.encoder)
        for p in self.ema_encoder.parameters():
            p.requires_grad_(False)
        self.ema_tau = ema_tau

    @torch.no_grad()
    def ema_update(self):
        for p, pe in zip(self.encoder.parameters(),
                         self.ema_encoder.parameters()):
            pe.mul_(self.ema_tau).add_(p, alpha=1 - self.ema_tau)

    def encode(self, batch, e, ema=False, zero_tokens=False):
        enc = self.ema_encoder if ema else self.encoder
        tokens = torch.zeros_like(batch["tokens"]) if zero_tokens \
            else batch["tokens"]
        return enc(tokens, batch["q"], e)

    def rollout(self, z, actions, e, k: int):
        """(B,M,d) + (B,>=k,10,7) -> list of k predicted carries."""
        out = []
        for j in range(k):
            z = self.transition(z, actions[:, j], e)
            out.append(z)
        return out

    def heads(self, z):
        f = z.flatten(1)
        return self.head_pred(f), self.head_prog(f).squeeze(-1), self.head_q(f)


def cosdist(a, b):
    a = nn.functional.layer_norm(a, a.shape[-1:])
    b = nn.functional.layer_norm(b, b.shape[-1:])
    return 1 - nn.functional.cosine_similarity(a, b, dim=-1).mean()
