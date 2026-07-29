#!/usr/bin/env python
"""LC-Flow wiring tests (ledger: fixed-noise pre-training equivalence).

1. NATIVE PARITY: custom N=1 sampling matches native PI05 within LeRobot's
   published OpenPI action tolerance.
2. NO-OP: with zero-init W_z, explicit-bias LC-Flow sampling under fixed flow
   noise is bitwise identical to the same custom runtime path.
3. CONTROL: a nonzero AdaRMS bias must change the sampled chunk (the adapter
   has causal control over the flow field).
4. STATE PIPELINE: prefix padding is ignored and state/outcome shapes are valid.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

from lcwm.chassis import Pi05Runner, make_task_env  # noqa: E402
from lcwm.lc_flow import (  # noqa: E402
    LCState,
    freeze_pi05_base,
    raw_flow_losses_from_prefix,
    raw_flow_losses_with_lc_bias,
    sample_chunks_lc,
)
from lcwm.sampler import prefix_forward, sample_chunks  # noqa: E402
from lerobot.utils.constants import (  # noqa: E402
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)


def main() -> None:
    dev = "cuda"
    runner = Pi05Runner(suite_name="libero_10")
    env = make_task_env("libero_10", 0)
    obs, _ = env.reset(seed=1000)
    task = env.task_description
    env.close()
    batch = runner._obs_to_policy_batch(obs, task)
    model = runner.policy.model
    freeze_pi05_base(runner.policy)

    g = torch.Generator().manual_seed(123)
    cfg = runner.policy.config
    noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim,
                        generator=g).to(dev)

    pre = prefix_forward(runner.policy, batch)
    stock = sample_chunks(runner.policy, batch, n=1, noise=noise, prefix=pre)

    # -- 1. native N=1 parity --------------------------------------------------
    images, img_masks = runner.policy._preprocess_images(batch)
    native = model.sample_actions(
        images,
        img_masks,
        batch[OBS_LANGUAGE_TOKENS],
        batch[OBS_LANGUAGE_ATTENTION_MASK],
        noise=noise,
    )[:, :, : stock.shape[-1]]
    native_diff = float((native - stock).abs().max())
    print(f"NATIVE: max|native - custom_N1| = {native_diff:.3e}")
    torch.testing.assert_close(native, stock, rtol=1e-2, atol=5e-3)

    # -- 2. explicit zero-bias equivalence ------------------------------------
    lc = LCState().to(dev)
    a_prev = torch.zeros(1, 10, 7, device=dev)
    z = lc.posterior(pre.hidden, pre.pad_masks)
    bias = lc.adarms_bias(z)
    assert float(bias.detach().abs().max()) == 0.0, (
        "zero-init W_z must emit zero bias"
    )
    lc_out = sample_chunks_lc(
        runner.policy, batch, bias, n=1, noise=noise, prefix=pre
    )
    diff = float((stock - lc_out).abs().max())
    print(f"NO-OP: max|stock - lcflow_zeroinit| = {diff:.2e}")
    assert diff == 0.0, "zero-init LC-Flow must be bitwise stock"

    # -- 3. nonzero bias has control ------------------------------------------
    torch.manual_seed(0)
    nonzero_bias = 0.5 * torch.randn_like(bias)
    perturbed = sample_chunks_lc(
        runner.policy, batch, nonzero_bias, n=1, noise=noise, prefix=pre
    )
    pdiff = float((stock - perturbed).abs().max())
    print(f"CONTROL: max|stock - biased| = {pdiff:.4f} "
          f"(action scale ~{float(stock.abs().mean()):.3f})")
    assert pdiff > 1e-3, "nonzero AdaRMS bias must change the sampled chunk"

    # -- 4. padding invariance + state pipeline shapes ------------------------
    invalid = ~pre.pad_masks.bool()
    h_changed = pre.hidden.clone()
    h_changed[invalid] = 1e4 * torch.randn_like(h_changed[invalid])
    z_changed = lc.posterior(h_changed, pre.pad_masks)
    torch.testing.assert_close(z, z_changed, rtol=0, atol=0)
    print(f"MASK: ignored {int(invalid.sum())} invalid prefix positions")

    z = lc.step(z, a_prev, pre.hidden, pre.pad_masks)
    out = lc.outcome(lc.transition(z, a_prev))
    shapes = {k: tuple(v.shape) for k, v in out.items()}
    print(f"STATE: z {tuple(z.shape)} | outcome {shapes}")
    n_train = sum(p.numel() for p in lc.parameters() if p.requires_grad)
    print(f"trainable LC params: {n_train/1e6:.2f}M "
          f"(PrefixVLM + expert frozen)")

    # -- 5. prefix-once training path matches the full PI05 flow graph --------
    train_batch = dict(batch)
    train_batch[ACTION] = stock
    flow_noise = torch.randn(
        1,
        cfg.chunk_size,
        cfg.max_action_dim,
        generator=torch.Generator().manual_seed(321),
    ).to(dev)
    flow_time = torch.tensor([0.5], device=dev)
    with torch.no_grad():
        full_raw = raw_flow_losses_with_lc_bias(
            runner.policy,
            train_batch,
            bias,
            noise=flow_noise,
            time=flow_time,
        )
        cached_raw = raw_flow_losses_from_prefix(
            runner.policy,
            stock,
            bias,
            pre,
            noise=flow_noise,
            time=flow_time,
        )
    cache_diff = float((full_raw - cached_raw).abs().max())
    print(f"FLOW-CACHE: max|full - prefix_once| = {cache_diff:.3e}")
    torch.testing.assert_close(
        full_raw, cached_raw, rtol=2e-2, atol=2e-2
    )

    # -- 6. adapter gradient through the prefix-once frozen action expert ------
    lc.zero_grad(set_to_none=True)
    with torch.enable_grad():
        train_bias = lc.adarms_bias(z.detach())
        sibling_actions = stock.expand(2, -1, -1).clone()
        sibling_actions[1, 0, 0] += 0.1
        raw = raw_flow_losses_from_prefix(
            runner.policy,
            sibling_actions,
            train_bias.expand(2, -1),
            pre,
            noise=flow_noise,
            time=flow_time,
        )
        grad_loss = raw[:, :10].mean()
        grad_loss.backward()
    grad = lc.wz_out.weight.grad
    assert grad is not None and float(grad.abs().sum()) > 0
    assert not any(p.requires_grad for p in runner.policy.parameters())
    assert all(p.grad is None for p in runner.policy.parameters())
    print(f"GRAD: |dL/dW_z_out|_1 = {float(grad.abs().sum()):.3e}")
    print("ALL LC-FLOW WIRING TESTS PASSED")


if __name__ == "__main__":
    main()
