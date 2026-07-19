"""Stage-4: batched N-chunk sampling from a frozen PI05 policy (plan §5).

The expensive part of pi0.5 inference is the VLM prefix (SigLIP images + language
through the 3B trunk). `sample_actions` already splits cleanly into
[prefix forward -> KV cache] + [10 denoise steps of the small action expert], so
N candidates need only: prefix ONCE at batch 1, expand the KV cache and pad masks
to N, then denoise N noise draws in parallel -> (N, chunk_size, action_dim).

Bonus (plan §6): the same prefix forward returns the last-layer hidden states —
the WM feature tap — so online candidate generation and feature extraction share
one trunk pass. See `lcwm.taps` for the feature side.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from transformers.cache_utils import DynamicCache


@dataclass
class PrefixCache:
    """Everything the denoise loop (and the feature tap) needs from one trunk pass."""
    hidden: Tensor            # (1, L, 2048) last-layer prefix hidden states
    pad_masks: Tensor         # (1, L)
    past_key_values: object   # DynamicCache, batch 1
    n_img_tokens: int         # image tokens occupy [:n_img_tokens], language the rest
    lang_masks: Tensor        # (1, n_lang) language pad mask


@torch.no_grad()
def prefix_forward(policy, observation: dict) -> PrefixCache:
    """One batch-1 trunk pass over the (already preprocessed) observation."""
    model = policy.model
    images, img_masks = policy._preprocess_images(observation)
    tokens = observation[OBS_LANGUAGE_TOKENS]
    masks = observation[OBS_LANGUAGE_ATTENTION_MASK]
    if tokens.shape[0] != 1:
        raise ValueError(f"prefix_forward expects batch size 1, got {tokens.shape[0]}")

    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, tokens, masks
    )
    att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    att_4d = model._prepare_attention_masks_4d(att_2d)
    model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

    (hidden, _), past_key_values = model.paligemma_with_expert.forward(
        attention_mask=att_4d,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )
    return PrefixCache(
        hidden=hidden,
        pad_masks=prefix_pad_masks,
        past_key_values=past_key_values,
        n_img_tokens=prefix_embs.shape[1] - tokens.shape[1],
        lang_masks=masks,
    )


def _expand_cache(past_key_values, n: int) -> DynamicCache:
    """Materialize an N-batched copy of a batch-1 prefix cache.

    Mirrors modeling_pi05.clone_past_key_values' (keys, values, sliding_window)
    layout; contiguous copies so the per-step clone inside denoise_step behaves
    exactly as in the batch-N case.
    """
    return DynamicCache(
        tuple(
            (
                keys.expand(n, *keys.shape[1:]).contiguous(),
                values.expand(n, *values.shape[1:]).contiguous(),
                sliding_window,
            )
            for keys, values, sliding_window in past_key_values
        )
    )


@torch.no_grad()
def sample_chunks(
    policy,
    observation: dict,
    n: int,
    noise: Tensor | None = None,
    num_steps: int | None = None,
    seed: int | None = None,
    prefix: PrefixCache | None = None,
) -> Tensor:
    """N candidate action chunks for ONE observation -> (n, chunk_size, action_dim).

    Output is in the policy's normalized action space, cropped to the real action
    dim — identical post-processing to `predict_action_chunk`. Run through the
    policy's postprocessor + env_postprocessor before executing in the env.
    `prefix` may be passed in to reuse a trunk pass (e.g. shared with the tap).
    """
    model = policy.model
    cfg = policy.config
    num_steps = num_steps or cfg.num_inference_steps

    if prefix is None:
        prefix = prefix_forward(policy, observation)
    device = prefix.pad_masks.device

    if noise is None:
        shape = (n, cfg.chunk_size, cfg.max_action_dim)
        if seed is not None:
            g = torch.Generator(device="cpu").manual_seed(seed)
            noise = torch.randn(shape, generator=g, dtype=torch.float32).to(device)
        else:
            noise = model.sample_noise(shape, device)
    elif noise.shape[0] != n:
        raise ValueError(f"noise batch {noise.shape[0]} != n {n}")

    pad_masks_n = prefix.pad_masks.expand(n, -1)
    cache_n = _expand_cache(prefix.past_key_values, n)

    dt = -1.0 / num_steps
    x_t = noise
    for step in range(num_steps):
        time = 1.0 + step * dt
        time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(n)
        v_t = model.denoise_step(
            prefix_pad_masks=pad_masks_n,
            past_key_values=cache_n,
            x_t=x_t,
            timestep=time_tensor,
        )
        x_t = x_t + dt * v_t

    from lerobot.utils.constants import ACTION
    original_action_dim = cfg.output_features[ACTION].shape[0]
    return x_t[:, :, :original_action_dim]
