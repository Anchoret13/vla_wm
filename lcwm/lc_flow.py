"""LC-Flow π0.5 core (framework v0.3 §3.1) — active primary implementation.

Persistent language-conditioned predictive state, directly conditioning the
π0.5 flow-matching action expert:

    z̄_t = T(z_{t-1}, E_a(a_exec_{t-1,0:10}))     action-conditioned prior
    z_t  = U(z̄_t, H_t^ℓ)                          observation/posterior update
    adarms_cond = time_emb + W_z · Pool(z_t)      zero-init AdaRMS injection
    outcome = D(T(z_t, E_a(a_candidate_0:10)))    ΔW / ΔY / ΔProg / R̂

Design constraints honored (v0.3 §1, §3.1, ledger):
- T receives NO task embedding: language enters only through H (real prompt),
  i.e. through the state it has already selected.
- The prefix forward that updates z REUSES the same trunk pass/KV cache used
  for action sampling (lcwm.sampler.prefix_forward) — no second 3B forward.
- W_z's output layer is zero-initialized: at init the injection is bitwise
  neutral on the chosen custom sampler path under fixed flow noise. Native
  PI05 parity is tested separately.
- Trainable boundary v0: E_a, T, U, D, W_z. PrefixVLM (2B) and action expert
  (300M) frozen; expert unfreezing is a later explicit escalation.
"""

from __future__ import annotations

import json
import types
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.policies.pi05.modeling_pi05 import (
    clone_past_key_values,
    make_att_2d_masks,
)
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)


@dataclass(frozen=True)
class LCFlowConfig:
    """Serializable dimensions for the frozen-base LC-Flow adapter."""

    d_h: int = 2048
    d_z: int = 384
    state_tokens: int = 4
    heads: int = 6
    transition_layers: int = 2
    update_blocks: int = 2
    expert_width: int = 1024
    action_dim: int = 7
    execution_horizon: int = 10
    proprio_dim: int = 9
    max_objects: int = 8
    max_atoms: int = 5


class CrossAttnBlock(nn.Module):
    """Cross-attention with an explicit prefix validity mask."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.ln_q, self.ln_kv = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)
        )

    def forward(
        self,
        x: Tensor,
        kv: Tensor,
        kv_mask: Tensor | None = None,
    ) -> Tensor:
        key_padding_mask = None
        if kv_mask is not None:
            if kv_mask.shape != kv.shape[:2]:
                raise ValueError(
                    f"prefix mask {tuple(kv_mask.shape)} != tokens {tuple(kv.shape[:2])}"
                )
            key_padding_mask = ~kv_mask.bool()
        kv_norm = self.ln_kv(kv)
        a, _ = self.attn(
            self.ln_q(x),
            kv_norm,
            kv_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + a
        return x + self.mlp(self.ln2(x))


class LCTransition(nn.Module):
    """T: (z, encoded action block) -> next z. No task-embedding input."""

    def __init__(self, d_z: int = 384, m: int = 4, heads: int = 6,
                 layers: int = 2):
        super().__init__()
        self.a_enc = nn.Sequential(nn.Linear(70, d_z), nn.GELU(),
                                   nn.Linear(d_z, d_z))
        self.type_emb = nn.Parameter(0.02 * torch.randn(2, d_z))
        layer = nn.TransformerEncoderLayer(
            d_z, heads, 4 * d_z, batch_first=True, norm_first=True,
            activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers)

    def forward(self, z: torch.Tensor, action_block: torch.Tensor):
        a = self.a_enc(action_block.flatten(1))[:, None] + self.type_emb[1]
        seq = torch.cat([z + self.type_emb[0], a], dim=1)
        return self.enc(seq)[:, : z.shape[1]]


class LCObservationUpdate(nn.Module):
    """U: posterior correction of the prior state from prefix hidden states."""

    def __init__(self, d_h: int = 2048, d_z: int = 384, heads: int = 6,
                 blocks: int = 2):
        super().__init__()
        self.h_proj = nn.Linear(d_h, d_z)
        self.blocks = nn.ModuleList(
            CrossAttnBlock(d_z, heads) for _ in range(blocks))

    def forward(self, z_bar: torch.Tensor, h: torch.Tensor):
        kv = self.h_proj(h)
        z = z_bar
        for blk in self.blocks:
            z = blk(z, kv)
        return z


class OutcomeHeads(nn.Module):
    """D: predicted one-block controlled effects from a (transitioned) state.

    ΔW  physical, instruction-independent: Δproprio (9) + Δobject positions
        (MAX_OBJ×3, masked per scene)
    ΔY  semantic: next predicate bits logits (MAX_ATOMS, masked per task)
    ΔProg, R̂: progress delta and short-horizon return/success logit
    """

    def __init__(self, d_z: int = 384, m: int = 4):
        super().__init__()
        d = m * d_z
        self.dq = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, 9))
        self.dobj = nn.Sequential(nn.Linear(d, 256), nn.GELU(),
                                  nn.Linear(256, MAX_OBJ * 3))
        self.dy = nn.Sequential(nn.Linear(d, 128), nn.GELU(),
                                nn.Linear(128, MAX_ATOMS))
        self.dprog = nn.Sequential(nn.Linear(d, 64), nn.GELU(),
                                   nn.Linear(64, 1))
        self.ret = nn.Sequential(nn.Linear(d, 64), nn.GELU(),
                                 nn.Linear(64, 1))

    def forward(self, z: torch.Tensor):
        f = z.flatten(1)
        return {"d_q": self.dq(f),
                "d_obj": self.dobj(f).reshape(-1, MAX_OBJ, 3),
                "next_bits_logits": self.dy(f),
                "d_prog": self.dprog(f).squeeze(-1),
                "ret": self.ret(f).squeeze(-1)}


class LCState(nn.Module):
    """Persistent state + all trainable LC-Flow components."""

    def __init__(self, d_h: int = 2048, d_z: int = 384, m: int = 4,
                 expert_width: int = 1024):
        super().__init__()
        self.m, self.d_z = m, d_z
        self.z0 = nn.Parameter(0.02 * torch.randn(m, d_z))
        self.transition = LCTransition(d_z, m)
        self.update = LCObservationUpdate(d_h, d_z)
        self.outcome = OutcomeHeads(d_z, m)
        # W_z: pooled state -> AdaRMS bias. Output layer ZERO-INIT => exact
        # stock no-op at initialization (framework §3.1).
        self.wz_hidden = nn.Linear(d_z, expert_width)
        self.wz_out = nn.Linear(expert_width, expert_width)
        nn.init.zeros_(self.wz_out.weight)
        nn.init.zeros_(self.wz_out.bias)

    def initial(self, batch: int, device, dtype=torch.float32):
        return self.z0[None].expand(batch, -1, -1).to(device, dtype)

    def step(self, z_prev, a_exec_block, h):
        """One decision-boundary update: prior T then posterior U."""
        z_bar = self.transition(z_prev, a_exec_block)
        return self.update(z_bar, h)

    def adarms_bias(self, z):
        pooled = z.mean(dim=1)                       # Pool over M tokens
        return self.wz_out(torch.nn.functional.gelu(self.wz_hidden(pooled)))


def attach_lc_bias(model) -> None:
    """Wrap PI05Pytorch.embed_suffix so adarms_cond += model._lc_bias.

    Instance-level wrap (no lerobot fork): token positions, prefix length and
    KV cache behavior are untouched — only the AdaRMS conditioning vector
    changes, exactly as specified in v0.3 §3.1. Set model._lc_bias = None to
    recover stock behavior identically.
    """
    if getattr(model, "_lc_wrapped", False):
        return
    original = model.embed_suffix

    def embed_suffix_lc(self, noisy_actions, timestep):
        embs, pad_masks, att_masks, adarms_cond = original(
            noisy_actions, timestep)
        bias = getattr(self, "_lc_bias", None)
        if bias is not None:
            adarms_cond = adarms_cond + bias.to(adarms_cond.dtype)
        return embs, pad_masks, att_masks, adarms_cond

    model.embed_suffix = types.MethodType(embed_suffix_lc, model)
    model._lc_bias = None
    model._lc_wrapped = True
