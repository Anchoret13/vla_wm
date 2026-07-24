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
    transition_dropout: float = 0.0
    update_blocks: int = 2
    expert_width: int = 1024
    action_dim: int = 7
    execution_horizon: int = 10
    proprio_dim: int = 9
    max_objects: int = 8
    max_atoms: int = 5


@dataclass(frozen=True)
class OutcomeLossWeights:
    d_q: float = 1.0
    d_obj: float = 1.0
    next_bits: float = 1.0
    d_prog: float = 1.0
    continuation_success: float = 1.0


@dataclass(frozen=True)
class EffectScales:
    """Frozen train-split scales for centered physical-effect supervision."""

    q_position: float
    q_quaternion: float
    q_gripper: float
    d_obj: float
    d_prog: float


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
            key_padding_mask = ~kv_mask.to(device=kv.device).bool()
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
                 layers: int = 2, action_dim: int = 7,
                 execution_horizon: int = 10, dropout: float = 0.0):
        super().__init__()
        self.action_dim = action_dim
        self.execution_horizon = execution_horizon
        self.a_enc = nn.Sequential(
            # Each step carries both the executed action and an explicit
            # executed/not-executed bit.  This prevents an early terminal
            # branch from treating the unexecuted proposal tail as dynamics.
            nn.Linear(execution_horizon * (action_dim + 1), d_z),
            nn.GELU(),
            nn.Linear(d_z, d_z),
        )
        self.type_emb = nn.Parameter(0.02 * torch.randn(2, d_z))
        layer = nn.TransformerEncoderLayer(
            d_z, heads, 4 * d_z, batch_first=True, norm_first=True,
            activation="gelu", dropout=dropout)
        self.enc = nn.TransformerEncoder(layer, layers)
        # This module is applied recurrently for hundreds of decisions.  A
        # final normalization is part of the state contract, not cosmetic:
        # otherwise residual magnitude compounds at every T step.
        self.out_norm = nn.LayerNorm(d_z)

    def forward(
        self,
        z: torch.Tensor,
        action_block: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ):
        expected = (self.execution_horizon, self.action_dim)
        if tuple(action_block.shape[-2:]) != expected:
            raise ValueError(
                f"action block tail {tuple(action_block.shape[-2:])} != {expected}"
            )
        if action_mask is None:
            action_mask = torch.ones(
                action_block.shape[:2],
                device=action_block.device,
                dtype=torch.bool,
            )
        if tuple(action_mask.shape) != tuple(action_block.shape[:2]):
            raise ValueError(
                f"action mask {tuple(action_mask.shape)} != "
                f"{tuple(action_block.shape[:2])}"
            )
        mask = action_mask.to(
            device=action_block.device, dtype=action_block.dtype
        )
        action_with_mask = torch.cat(
            [action_block * mask[..., None], mask[..., None]], dim=-1
        )
        a = (
            self.a_enc(action_with_mask.flatten(1))[:, None]
            + self.type_emb[1]
        )
        seq = torch.cat([z + self.type_emb[0], a], dim=1)
        return self.out_norm(self.enc(seq)[:, : z.shape[1]])


class LCObservationUpdate(nn.Module):
    """U: posterior correction of the prior state from prefix hidden states."""

    def __init__(self, d_h: int = 2048, d_z: int = 384, heads: int = 6,
                 blocks: int = 2):
        super().__init__()
        self.h_proj = nn.Linear(d_h, d_z)
        self.blocks = nn.ModuleList(
            CrossAttnBlock(d_z, heads) for _ in range(blocks))
        self.out_norm = nn.LayerNorm(d_z)

    def forward(
        self,
        z_bar: torch.Tensor,
        h: torch.Tensor,
        h_mask: torch.Tensor | None = None,
    ):
        h = h.to(device=self.h_proj.weight.device, dtype=self.h_proj.weight.dtype)
        kv = self.h_proj(h)
        z = z_bar
        for blk in self.blocks:
            z = blk(z, kv, h_mask)
        return self.out_norm(z)


class OutcomeHeads(nn.Module):
    """D: predicted one-block controlled effects from a (transitioned) state.

    ΔW  physical, instruction-independent: Δproprio + Δobject positions
        (masked per scene)
    ΔY  semantic: next predicate bits logits (masked per task)
    ΔProg, R̂: progress delta and short-horizon return/success logit
    """

    def __init__(
        self,
        d_z: int = 384,
        m: int = 4,
        proprio_dim: int = 9,
        max_objects: int = 8,
        max_atoms: int = 5,
    ):
        super().__init__()
        self.proprio_dim = proprio_dim
        self.max_objects = max_objects
        self.max_atoms = max_atoms
        d = m * d_z
        self.dq = nn.Sequential(
            nn.Linear(d, 256), nn.GELU(), nn.Linear(256, proprio_dim)
        )
        self.dobj = nn.Sequential(
            nn.Linear(d, 256), nn.GELU(), nn.Linear(256, max_objects * 3)
        )
        self.dy = nn.Sequential(
            nn.Linear(d, 128), nn.GELU(), nn.Linear(128, max_atoms)
        )
        self.dprog = nn.Sequential(nn.Linear(d, 64), nn.GELU(),
                                   nn.Linear(64, 1))
        self.ret = nn.Sequential(nn.Linear(d, 64), nn.GELU(),
                                 nn.Linear(64, 1))

    def forward(self, z: torch.Tensor):
        f = z.flatten(1)
        return {"d_q": self.dq(f),
                "d_obj": self.dobj(f).reshape(-1, self.max_objects, 3),
                "next_bits_logits": self.dy(f),
                "d_prog": self.dprog(f).squeeze(-1),
                "ret": self.ret(f).squeeze(-1)}


class LCState(nn.Module):
    """Persistent state + all trainable LC-Flow components."""

    def __init__(
        self,
        config: LCFlowConfig | None = None,
        **overrides,
    ):
        super().__init__()
        if config is not None and overrides:
            raise ValueError("pass either config or keyword overrides, not both")
        self.config = config or LCFlowConfig(**overrides)
        cfg = self.config
        self.m, self.d_z = cfg.state_tokens, cfg.d_z
        self.z0 = nn.Parameter(0.02 * torch.randn(cfg.state_tokens, cfg.d_z))
        self.transition = LCTransition(
            cfg.d_z,
            cfg.state_tokens,
            cfg.heads,
            cfg.transition_layers,
            cfg.action_dim,
            cfg.execution_horizon,
            cfg.transition_dropout,
        )
        self.update = LCObservationUpdate(
            cfg.d_h, cfg.d_z, cfg.heads, cfg.update_blocks
        )
        self.outcome = OutcomeHeads(
            cfg.d_z,
            cfg.state_tokens,
            cfg.proprio_dim,
            cfg.max_objects,
            cfg.max_atoms,
        )
        # W_z: pooled state -> AdaRMS bias. Output layer ZERO-INIT => exact
        # stock no-op at initialization (framework §3.1).
        self.wz_hidden = nn.Linear(cfg.d_z, cfg.expert_width)
        self.wz_out = nn.Linear(cfg.expert_width, cfg.expert_width)
        nn.init.zeros_(self.wz_out.weight)
        nn.init.zeros_(self.wz_out.bias)

    def initial(self, batch: int, device, dtype=torch.float32):
        return self.z0[None].expand(batch, -1, -1).to(
            device=device, dtype=dtype
        )

    def posterior(self, h, h_mask=None, z_prior=None):
        """First-decision update U(z0,H), with no fictitious zero-action step."""
        if z_prior is None:
            z_prior = self.initial(
                h.shape[0], self.z0.device, self.z0.dtype
            )
        return self.update(z_prior, h, h_mask)

    def step(
        self,
        z_prev,
        a_exec_block,
        h,
        h_mask=None,
        action_mask=None,
    ):
        """One decision-boundary update: prior T then posterior U."""
        z_bar = self.transition(z_prev, a_exec_block, action_mask)
        return self.update(z_bar, h, h_mask)

    def adarms_bias(self, z):
        pooled = z.mean(dim=1)                       # Pool over M tokens
        return self.wz_out(F.gelu(self.wz_hidden(pooled)))


def freeze_pi05_base(policy: nn.Module) -> None:
    """Freeze and put the complete stock PI05 policy in inference mode."""
    policy.requires_grad_(False)
    policy.eval()
    if any(p.requires_grad for p in policy.parameters()):
        raise AssertionError("PI05 base still has trainable parameters")


def _batch_bias(lc_bias: Tensor, batch: int, width: int, *, device, dtype) -> Tensor:
    if lc_bias.ndim != 2 or lc_bias.shape[1] != width:
        raise ValueError(
            f"LC bias must be [B,{width}], got {tuple(lc_bias.shape)}"
        )
    if lc_bias.shape[0] == 1 and batch != 1:
        lc_bias = lc_bias.expand(batch, -1)
    elif lc_bias.shape[0] != batch:
        raise ValueError(f"LC bias batch {lc_bias.shape[0]} != {batch}")
    return lc_bias.to(device=device, dtype=dtype)


def denoise_step_with_lc_bias(
    model,
    prefix_pad_masks: Tensor,
    past_key_values,
    x_t: Tensor,
    timestep: Tensor,
    lc_bias: Tensor,
) -> Tensor:
    """PI05 denoise step with an explicit LC AdaRMS condition.

    This mirrors the editable LeRobot 0.6.1 implementation while keeping the
    adapter input explicit and local. It avoids mutable ``model._lc_bias`` state,
    so one episode/batch cannot leak its state into another.
    """
    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
        model.embed_suffix(x_t, timestep)
    )
    adarms_cond = adarms_cond + _batch_bias(
        lc_bias,
        x_t.shape[0],
        adarms_cond.shape[-1],
        device=adarms_cond.device,
        dtype=adarms_cond.dtype,
    )

    suffix_len = suffix_pad_masks.shape[1]
    batch_size = prefix_pad_masks.shape[0]
    prefix_len = prefix_pad_masks.shape[1]
    prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
        batch_size, suffix_len, prefix_len
    )
    suffix_att_2d_masks = make_att_2d_masks(
        suffix_pad_masks, suffix_att_masks
    )
    full_att_2d_masks = torch.cat(
        [prefix_pad_2d_masks, suffix_att_2d_masks], dim=2
    )
    prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
    position_ids = (
        prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
    )
    full_att_4d = model._prepare_attention_masks_4d(full_att_2d_masks)
    model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = (
        "eager"
    )

    outputs_embeds, _ = model.paligemma_with_expert.forward(
        attention_mask=full_att_4d,
        position_ids=position_ids,
        past_key_values=clone_past_key_values(past_key_values),
        inputs_embeds=[None, suffix_embs],
        use_cache=False,
        adarms_cond=[None, adarms_cond],
    )
    suffix_out = outputs_embeds[1][:, -model.config.chunk_size :].float()
    return model.action_out_proj(suffix_out)


@torch.no_grad()
def sample_chunks_lc(
    policy,
    observation: dict,
    lc_bias: Tensor,
    n: int = 1,
    noise: Tensor | None = None,
    num_steps: int | None = None,
    seed: int | None = None,
    prefix=None,
) -> Tensor:
    """Sample PI05 chunks with explicit LC state conditioning.

    The prefix hidden state/cache can be supplied from the same forward used by
    the LC observation update. The LC bias is expanded explicitly for N-sampling
    and remains fixed over all Euler denoising steps.
    """
    from lcwm.sampler import _expand_cache, prefix_forward

    model = policy.model
    cfg = policy.config
    num_steps = num_steps or cfg.num_inference_steps
    prefix = prefix or prefix_forward(policy, observation)
    device = prefix.pad_masks.device
    if noise is None:
        shape = (n, cfg.chunk_size, cfg.max_action_dim)
        if seed is None:
            noise = model.sample_noise(shape, device)
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            noise = torch.randn(
                shape, generator=generator, dtype=torch.float32
            ).to(device)
    elif noise.shape[0] != n:
        raise ValueError(f"noise batch {noise.shape[0]} != n {n}")

    pad_masks_n = prefix.pad_masks.expand(n, -1)
    cache_n = _expand_cache(prefix.past_key_values, n)
    bias_n = _batch_bias(
        lc_bias,
        n,
        model.action_in_proj.out_features,
        device=device,
        dtype=torch.float32,
    )
    dt = -1.0 / num_steps
    x_t = noise
    for step in range(num_steps):
        time = 1.0 + step * dt
        timestep = torch.full(
            (n,), time, dtype=torch.float32, device=device
        )
        v_t = denoise_step_with_lc_bias(
            model, pad_masks_n, cache_n, x_t, timestep, bias_n
        )
        x_t = x_t + dt * v_t

    original_dim = cfg.output_features[ACTION].shape[0]
    return x_t[:, :, :original_dim]


def raw_flow_losses_with_lc_bias(
    policy,
    batch: dict[str, Tensor],
    lc_bias: Tensor,
    *,
    noise: Tensor | None = None,
    time: Tensor | None = None,
) -> Tensor:
    """Elementwise PI05 flow loss ``[B,50,real_action_dim]``.

    Unlike ``PI05Policy.forward(reduction="none")``, this preserves the time
    axis so branch supervision can mask unexecuted actions.
    """
    model = policy.model
    images, img_masks = policy._preprocess_images(batch)
    tokens = batch[OBS_LANGUAGE_TOKENS]
    masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    actions = policy.prepare_action(batch)
    noise = noise if noise is not None else model.sample_noise(
        actions.shape, actions.device
    )
    time = time if time is not None else model.sample_time(
        actions.shape[0], actions.device
    )
    if noise.shape != actions.shape:
        raise ValueError(f"noise {tuple(noise.shape)} != actions {tuple(actions.shape)}")
    if time.shape != (actions.shape[0],):
        raise ValueError(f"time {tuple(time.shape)} != {(actions.shape[0],)}")

    time_expanded = time[:, None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    target_velocity = noise - actions
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, tokens, masks
    )
    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
        model.embed_suffix(x_t, time)
    )
    adarms_cond = adarms_cond + _batch_bias(
        lc_bias,
        actions.shape[0],
        adarms_cond.shape[-1],
        device=adarms_cond.device,
        dtype=adarms_cond.dtype,
    )

    first_weight = (
        model.paligemma_with_expert.paligemma.model.language_model.layers[
            0
        ].self_attn.q_proj.weight
    )
    if first_weight.dtype == torch.bfloat16:
        prefix_embs = prefix_embs.bfloat16()
        suffix_embs = suffix_embs.bfloat16()

    pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
    att_2d = make_att_2d_masks(pad_masks, att_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    att_4d = model._prepare_attention_masks_4d(att_2d)
    (_, suffix_out), _ = model.paligemma_with_expert.forward(
        attention_mask=att_4d,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, suffix_embs],
        use_cache=False,
        adarms_cond=[None, adarms_cond],
    )
    velocity = model.action_out_proj(
        suffix_out[:, -model.config.chunk_size :].float()
    )
    losses = F.mse_loss(target_velocity, velocity, reduction="none")
    original_dim = policy.config.output_features[ACTION].shape[0]
    return losses[:, :, :original_dim]


def raw_flow_losses_from_prefix(
    policy,
    actions_norm: Tensor,
    lc_bias: Tensor,
    prefix,
    *,
    noise: Tensor | None = None,
    time: Tensor | None = None,
) -> Tensor:
    """Elementwise branch FM loss with one frozen PrefixVLM forward.

    ``actions_norm`` is already in PI05's normalized LIBERO action convention
    and is padded here from 7 to the model's internal 32 dimensions.  All
    sibling branches share the same full-instruction prefix cache.  By default
    they also share one flow time/noise draw, which removes an avoidable source
    of variance from within-snapshot weighting while preserving a valid FM
    objective.
    """
    if actions_norm.ndim != 3:
        raise ValueError(
            f"actions_norm must be [B,T,A], got {tuple(actions_norm.shape)}"
        )
    batch, steps, action_dim = actions_norm.shape
    cfg = policy.config
    if steps != cfg.chunk_size:
        raise ValueError(f"action steps {steps} != PI05 chunk {cfg.chunk_size}")
    real_dim = cfg.output_features[ACTION].shape[0]
    if action_dim != real_dim:
        raise ValueError(f"action dim {action_dim} != policy dim {real_dim}")

    model = policy.model
    device = prefix.pad_masks.device
    actions = policy.prepare_action(
        {ACTION: actions_norm.to(device=device, dtype=torch.float32)}
    )
    if noise is None:
        shared_noise = model.sample_noise(
            (1, cfg.chunk_size, cfg.max_action_dim), device
        )
        noise = shared_noise.expand(batch, -1, -1)
    elif noise.shape[0] == 1 and batch != 1:
        noise = noise.expand(batch, -1, -1)
    if tuple(noise.shape) != tuple(actions.shape):
        raise ValueError(
            f"noise {tuple(noise.shape)} != padded actions {tuple(actions.shape)}"
        )
    noise = noise.to(device=device, dtype=actions.dtype)

    if time is None:
        time = model.sample_time(1, device).expand(batch)
    elif time.shape == (1,) and batch != 1:
        time = time.expand(batch)
    if time.shape != (batch,):
        raise ValueError(f"time {tuple(time.shape)} != {(batch,)}")
    time = time.to(device=device, dtype=torch.float32)

    time_expanded = time[:, None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    target_velocity = noise - actions

    from lcwm.sampler import _expand_cache

    velocity = denoise_step_with_lc_bias(
        model,
        prefix.pad_masks.expand(batch, -1),
        _expand_cache(prefix.past_key_values, batch),
        x_t,
        time,
        lc_bias,
    )
    losses = F.mse_loss(target_velocity, velocity, reduction="none")
    return losses[:, :, :real_dim]


def masked_branch_flow_loss(
    raw_losses: Tensor,
    weights: Tensor,
    *,
    executed_lengths: Tensor | None = None,
    max_executed: int = 10,
) -> tuple[Tensor, Tensor]:
    """Reduce elementwise flow loss using only physically executed actions.

    Returns ``(weighted_loss, per_branch_loss)``. Tail *loss elements* never
    contribute. The full proposed chunk may still be used as PI05's denoising
    context because its action tokens attend within the suffix block.
    """
    if raw_losses.ndim != 3:
        raise ValueError(f"raw losses must be [B,T,A], got {raw_losses.shape}")
    batch, steps, action_dim = raw_losses.shape
    if weights.shape != (batch,):
        raise ValueError(f"weights {tuple(weights.shape)} != {(batch,)}")
    if (weights < 0).any():
        raise ValueError("branch flow weights must be nonnegative")
    if executed_lengths is None:
        executed_lengths = torch.full(
            (batch,), min(max_executed, steps),
            device=raw_losses.device, dtype=torch.long
        )
    else:
        executed_lengths = executed_lengths.to(
            device=raw_losses.device, dtype=torch.long
        ).clamp(min=0, max=min(max_executed, steps))
    mask = (
        torch.arange(steps, device=raw_losses.device)[None]
        < executed_lengths[:, None]
    )
    denom = (mask.sum(1) * action_dim).clamp_min(1)
    per_branch = (raw_losses * mask[:, :, None]).sum((1, 2)) / denom
    weights = weights.to(device=raw_losses.device, dtype=raw_losses.dtype)
    weight_sum = weights.sum()
    if float(weight_sum.detach()) == 0.0:
        return raw_losses.sum() * 0.0, per_branch
    return (weights * per_branch).sum() / weight_sum, per_branch


def credit_bounded_actions(
    actions_norm: Tensor,
    executed_lengths: Tensor | None = None,
    max_executed: int = 10,
) -> Tensor:
    """Suffix credit contract (2026-07-24 audit correction 4).

    The action expert attends bidirectionally over all chunk positions, so an
    unexecuted target suffix can leak into the first-executed predictions even
    when the loss is masked to the executed prefix.  Replacing every position
    at or beyond a branch's executed length with zeros — a branch-independent
    constant — before the actions enter the flow graph makes the executed
    prefix's losses and gradients exactly invariant to unexecuted target
    content.  The replacement mask is identical to the credit mask in
    ``masked_branch_flow_loss``.
    """
    if actions_norm.ndim != 3:
        raise ValueError(
            f"actions_norm must be [B,T,A], got {tuple(actions_norm.shape)}"
        )
    batch, steps, _ = actions_norm.shape
    if executed_lengths is None:
        executed_lengths = torch.full(
            (batch,), min(max_executed, steps),
            device=actions_norm.device, dtype=torch.long
        )
    else:
        executed_lengths = executed_lengths.to(
            device=actions_norm.device, dtype=torch.long
        ).clamp(min=0, max=min(max_executed, steps))
    keep = (
        torch.arange(steps, device=actions_norm.device)[None]
        < executed_lengths[:, None]
    )
    return actions_norm * keep[:, :, None].to(actions_norm.dtype)


def cached_branch_flow_loss(
    policy,
    prefix,
    actions_norm: Tensor,
    lc_bias: Tensor,
    weights: Tensor,
    *,
    executed_lengths: Tensor | None = None,
    max_executed: int = 10,
    noise: Tensor | None = None,
    time: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Weighted first-executed-actions loss without rerunning the VLM prefix."""
    raw = raw_flow_losses_from_prefix(
        policy,
        credit_bounded_actions(
            actions_norm,
            executed_lengths=executed_lengths,
            max_executed=max_executed,
        ),
        lc_bias,
        prefix,
        noise=noise,
        time=time,
    )
    loss, per_branch = masked_branch_flow_loss(
        raw,
        weights,
        executed_lengths=executed_lengths,
        max_executed=max_executed,
    )
    return loss, {"raw": raw, "per_branch": per_branch}


def branch_flow_loss(
    policy,
    batch: dict[str, Tensor],
    lc_bias: Tensor,
    weights: Tensor,
    *,
    executed_lengths: Tensor | None = None,
    max_executed: int = 10,
    noise: Tensor | None = None,
    time: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    batch = {
        **batch,
        ACTION: credit_bounded_actions(
            batch[ACTION],
            executed_lengths=executed_lengths,
            max_executed=max_executed,
        ),
    }
    raw = raw_flow_losses_with_lc_bias(
        policy, batch, lc_bias, noise=noise, time=time
    )
    loss, per_branch = masked_branch_flow_loss(
        raw,
        weights,
        executed_lengths=executed_lengths,
        max_executed=max_executed,
    )
    return loss, {"raw": raw, "per_branch": per_branch}


def _masked_mean(values: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return values.mean()
    mask = mask.to(device=values.device, dtype=values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(values)
    return (values * mask).sum() / mask.sum().clamp_min(1)


def centered_effect_loss(
    predictions: dict[str, Tensor],
    targets: dict[str, Tensor],
    scales: EffectScales,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Source-normalized sibling-effect loss with common mode removed.

    For each snapshot group, subtracting the sibling mean is equivalent to
    matching all pairwise action-outcome differences. Action-independent phase
    constants therefore contribute exactly zero signal. Binary predicate and
    return heads are intentionally absent; they require within-group label
    variation before they become identifiable.
    """

    required = ("d_q", "d_obj", "d_prog")
    missing = [
        key
        for key in required
        if key not in predictions or key not in targets
    ]
    if missing:
        raise KeyError(f"missing centered-effect values: {missing}")
    d_q = targets["d_q"]
    if d_q.ndim != 2 or d_q.shape[1] != 9:
        raise ValueError(
            "effect_centered_v1 requires q=[eef xyz, quaternion, gripper] "
            f"with width 9, got {tuple(d_q.shape)}"
        )
    if d_q.shape[0] < 2:
        raise ValueError("centered effect loss requires sibling branches")
    for name, value in asdict(scales).items():
        if value <= 0:
            raise ValueError(f"effect scale {name} must be positive")

    def centered(value: Tensor) -> Tensor:
        return value - value.mean(dim=0, keepdim=True)

    def normalized_mse(
        prediction: Tensor,
        target: Tensor,
        scale: float,
        mask: Tensor | None = None,
    ) -> Tensor:
        target = target.to(
            device=prediction.device, dtype=prediction.dtype
        )
        error = (centered(prediction) - centered(target)) / scale
        return _masked_mean(error.square(), mask)

    object_mask = targets.get("object_mask")
    if object_mask is not None:
        object_mask = object_mask.to(
            device=predictions["d_obj"].device,
            dtype=torch.bool,
        )
        if object_mask.shape[0] == predictions["d_obj"].shape[0] and not (
            object_mask
            == object_mask[0:1].expand_as(object_mask)
        ).all():
            raise ValueError(
                "centered object loss requires a branch-constant mask"
            )

    parts = {
        "effect_q_position": normalized_mse(
            predictions["d_q"][:, :3],
            targets["d_q"][:, :3],
            scales.q_position,
            targets.get("q_mask_position"),
        ),
        "effect_q_quaternion": normalized_mse(
            predictions["d_q"][:, 3:7],
            targets["d_q"][:, 3:7],
            scales.q_quaternion,
            targets.get("q_mask_quaternion"),
        ),
        "effect_q_gripper": normalized_mse(
            predictions["d_q"][:, 7:9],
            targets["d_q"][:, 7:9],
            scales.q_gripper,
            targets.get("q_mask_gripper"),
        ),
        "effect_d_obj": normalized_mse(
            predictions["d_obj"],
            targets["d_obj"],
            scales.d_obj,
            object_mask,
        ),
        "effect_d_prog": normalized_mse(
            predictions["d_prog"],
            targets["d_prog"],
            scales.d_prog,
            targets.get("progress_mask"),
        ),
    }
    return torch.stack(tuple(parts.values())).mean(), parts


def outcome_loss(
    predictions: dict[str, Tensor],
    targets: dict[str, Tensor],
    *,
    weights: OutcomeLossWeights | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Outcome-grounded one-block loss for every branch, including failures."""
    weights = weights or OutcomeLossWeights()
    required = ("d_q", "d_obj", "next_bits", "d_prog")
    missing = [key for key in required if key not in targets]
    if missing:
        raise KeyError(f"missing outcome targets: {missing}")

    parts = {
        "d_q": _masked_mean(
            (predictions["d_q"] - targets["d_q"]) ** 2,
            targets.get("q_mask"),
        ),
        "d_obj": _masked_mean(
            (predictions["d_obj"] - targets["d_obj"]) ** 2,
            targets.get("object_mask"),
        ),
        "next_bits": _masked_mean(
            F.binary_cross_entropy_with_logits(
                predictions["next_bits_logits"],
                targets["next_bits"].to(predictions["next_bits_logits"].dtype),
                reduction="none",
            ),
            targets.get("atom_mask"),
        ),
        "d_prog": _masked_mean(
            (predictions["d_prog"] - targets["d_prog"]) ** 2,
            targets.get("progress_mask"),
        ),
    }
    if "continuation_success" in targets:
        parts["continuation_success"] = _masked_mean(
            F.binary_cross_entropy_with_logits(
                predictions["ret"],
                targets["continuation_success"].to(predictions["ret"].dtype),
                reduction="none",
            ),
            targets.get("continuation_mask"),
        )
    else:
        parts["continuation_success"] = predictions["ret"].sum() * 0.0

    total = (
        weights.d_q * parts["d_q"]
        + weights.d_obj * parts["d_obj"]
        + weights.next_bits * parts["next_bits"]
        + weights.d_prog * parts["d_prog"]
        + weights.continuation_success * parts["continuation_success"]
    )
    return total, parts


def save_lc_adapter(
    lc_state: LCState,
    directory: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save only the LC adapter; the immutable stock PI05 stays external."""
    from safetensors.torch import save_file

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tensors = {
        key: value.detach().cpu().contiguous()
        for key, value in lc_state.state_dict().items()
    }
    save_file(tensors, str(directory / "lc_flow.safetensors"))
    manifest = {
        "schema": "lc_flow_adapter_v1",
        "config": asdict(lc_state.config),
        "metadata": metadata or {},
    }
    (directory / "lc_flow_config.json").write_text(
        json.dumps(manifest, indent=2)
    )


def load_lc_adapter(
    directory: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[LCState, dict[str, Any]]:
    """Strict-load an LC adapter independently from the stock PI05 checkpoint."""
    from safetensors.torch import load_file

    directory = Path(directory)
    manifest = json.loads((directory / "lc_flow_config.json").read_text())
    if manifest.get("schema") != "lc_flow_adapter_v1":
        raise ValueError(f"unsupported LC adapter schema: {manifest.get('schema')}")
    lc_state = LCState(LCFlowConfig(**manifest["config"]))
    state = load_file(str(directory / "lc_flow.safetensors"), device=str(device))
    lc_state.load_state_dict(state, strict=True)
    return lc_state.to(device), manifest.get("metadata", {})


class LCFlowRunner:
    """Decision-rate persistent LC state around the existing PI05 chassis."""

    def __init__(self, runner, lc_state: LCState):
        self.runner = runner
        freeze_pi05_base(self.runner.policy)
        base_device = next(self.runner.policy.parameters()).device
        # This wrapper is an inference executor.  Training uses LCState
        # directly; leaving TransformerEncoder dropout active here would make
        # fixed-noise policy evaluation stochastic for the wrong reason.
        self.lc_state = lc_state.to(base_device).eval()
        self._queue: deque[Tensor] = deque()
        self.z: Tensor | None = None
        self.previous_action_norm: Tensor | None = None
        self.decision_index = 0

    def reset(self) -> None:
        self.runner.reset()
        self._queue.clear()
        self.z = None
        self.previous_action_norm = None
        self.decision_index = 0

    @torch.no_grad()
    def _generate_chunk(
        self,
        obs: dict,
        task_description: str,
        *,
        noise: Tensor | None = None,
        seed: int | None = None,
    ) -> Tensor:
        from lcwm.sampler import prefix_forward

        batch = self.runner._obs_to_policy_batch(obs, task_description)
        prefix = prefix_forward(self.runner.policy, batch)
        h_mask = prefix.pad_masks
        if self.z is None:
            self.z = self.lc_state.posterior(
                prefix.hidden, h_mask
            )
        else:
            if self.previous_action_norm is None:
                raise RuntimeError("missing previous executed normalized action block")
            self.z = self.lc_state.step(
                self.z,
                self.previous_action_norm,
                prefix.hidden,
                h_mask,
            )
        bias = self.lc_state.adarms_bias(self.z)
        chunk = sample_chunks_lc(
            self.runner.policy,
            batch,
            bias,
            n=1,
            noise=noise,
            seed=seed,
            prefix=prefix,
        )
        horizon = self.lc_state.config.execution_horizon
        self.previous_action_norm = chunk[:, :horizon].detach()
        self._queue.extend(chunk[:, :horizon].transpose(0, 1))
        self.decision_index += 1
        return chunk

    @torch.no_grad()
    def select_action(
        self,
        obs: dict,
        task_description: str,
        *,
        noise: Tensor | None = None,
        seed: int | None = None,
    ):
        if not self._queue:
            self._generate_chunk(
                obs, task_description, noise=noise, seed=seed
            )
        action_norm = self._queue.popleft()
        return self.runner.action_to_env(action_norm)


def attach_lc_bias(model) -> None:
    """Deprecated single-process smoke hook.

    New training/runtime code must use the explicit-bias functions above. This
    mutable wrapper remains only for reproducing the original wiring smoke.
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
