"""Stage-5: feature tap on the frozen pi0.5 trunk (plan §6).

Per timestep the WM consumes:
- h_img: last-layer image-token hidden states, (n_img_tokens≈512, 2048) for 2 cams
- e_lang: pooled instruction-token hidden state, (2048,), from the REAL-prompt pass

Design constraint (plan §1): the conditioning-arm comparison requires language to
enter ONLY through the dynamics channel, so h_img is tapped under a CONSTANT
prompt. `feature_shift` quantifies what that substitution does to the image
features — run it before trusting any tapped dataset (plan §8 checklist).

Optional 2x2 average pooling over the token grid (per camera, 16x16 -> 8x8)
cuts storage 4x for segment datasets (plan §6 storage budget).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from lcwm.sampler import PrefixCache, prefix_forward

CONSTANT_PROMPT = "do the task"  # fixed instruction used for the dynamics-channel tap


@dataclass
class StepFeatures:
    h_img: Tensor    # (n_img_tokens, 2048)  fp32 (cast by caller for storage)
    e_lang: Tensor   # (2048,) pooled real-prompt instruction embedding
    n_img_tokens: int


def pool_image_tokens(h_img: Tensor, tokens_per_cam: int = 256, grid: int = 16) -> Tensor:
    """2x2 average-pool each camera's (grid x grid) token map -> 4x fewer tokens."""
    n_cams = h_img.shape[0] // tokens_per_cam
    d = h_img.shape[-1]
    out = []
    for c in range(n_cams):
        t = h_img[c * tokens_per_cam:(c + 1) * tokens_per_cam]
        t = t.reshape(grid, grid, d).permute(2, 0, 1)          # (d, g, g)
        t = torch.nn.functional.avg_pool2d(t.unsqueeze(0), 2).squeeze(0)
        out.append(t.permute(1, 2, 0).reshape(-1, d))          # (g/2*g/2, d)
    return torch.cat(out, dim=0)


def pooled_lang(prefix: PrefixCache) -> Tensor:
    """Masked mean over instruction-token hidden states -> (2048,)."""
    h_lang = prefix.hidden[0, prefix.n_img_tokens:]            # (n_lang, d)
    mask = prefix.lang_masks[0].to(h_lang.dtype)               # (n_lang,)
    denom = mask.sum().clamp(min=1.0)
    return (h_lang * mask[:, None]).sum(dim=0) / denom


@torch.no_grad()
def tap_step(
    runner,
    obs: dict,
    task_description: str,
    constant_prompt: str = CONSTANT_PROMPT,
    pool: bool = True,
) -> StepFeatures:
    """Extract WM features for one env step.

    Two trunk passes: constant-prompt (image features for dynamics) and
    real-prompt (pooled e_lang). `runner` is an lcwm.chassis.Pi05Runner —
    its preprocessing pipeline handles tokenization of the prompt string.
    """
    const_batch = runner._obs_to_policy_batch(obs, constant_prompt)
    const_prefix = prefix_forward(runner.policy, const_batch)
    # pi05 has 3 image slots; LIBERO fills 2 — drop masked (empty-slot) tokens,
    # they carry garbage values (pad_mask=0 keeps them out of attention only).
    img_valid = const_prefix.pad_masks[0, : const_prefix.n_img_tokens].bool()
    h_img = const_prefix.hidden[0, : const_prefix.n_img_tokens][img_valid].float()

    real_batch = runner._obs_to_policy_batch(obs, task_description)
    real_prefix = prefix_forward(runner.policy, real_batch)
    e_lang = pooled_lang(real_prefix).float()

    if pool:
        h_img = pool_image_tokens(h_img)
    return StepFeatures(h_img=h_img, e_lang=e_lang, n_img_tokens=h_img.shape[0])


@torch.no_grad()
def feature_shift(runner, obs: dict, task_description: str,
                  constant_prompt: str = CONSTANT_PROMPT) -> dict:
    """Quantify constant-vs-real prompt shift of the IMAGE tokens on one frame.

    Returns per-frame stats: mean/max cosine distance over image tokens and the
    relative L2 shift. Language tokens attend into image tokens (full prefix
    attention), so the shift is nonzero by construction — the question is size.
    """
    const_prefix = prefix_forward(runner.policy, runner._obs_to_policy_batch(obs, constant_prompt))
    real_prefix = prefix_forward(runner.policy, runner._obs_to_policy_batch(obs, task_description))
    n = min(const_prefix.n_img_tokens, real_prefix.n_img_tokens)
    a = const_prefix.hidden[0, :n].float()
    b = real_prefix.hidden[0, :n].float()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    rel_l2 = (a - b).norm(dim=-1) / b.norm(dim=-1).clamp(min=1e-6)
    return {
        "mean_cos_dist": float((1 - cos).mean()),
        "max_cos_dist": float((1 - cos).max()),
        "mean_rel_l2": float(rel_l2.mean()),
        "n_img_tokens": int(n),
    }
