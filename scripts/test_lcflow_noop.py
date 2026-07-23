#!/usr/bin/env python
"""LC-Flow wiring tests (ledger: fixed-noise pre-training equivalence).

1. NO-OP: with zero-init W_z, LC-Flow sampling under fixed flow noise must be
   EXACTLY the stock policy output (max |Δ| == 0).
2. CONTROL: a nonzero AdaRMS bias must change the sampled chunk (the adapter
   has causal control over the flow field).
3. STATE PIPELINE: LCState.step runs on a real prefix forward (shapes, dtype,
   shared trunk pass — no second 3B forward).
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
from lcwm.lc_flow import LCState, attach_lc_bias  # noqa: E402
from lcwm.sampler import prefix_forward, sample_chunks  # noqa: E402


def main() -> None:
    dev = "cuda"
    runner = Pi05Runner(suite_name="libero_10")
    env = make_task_env("libero_10", 0)
    obs, _ = env.reset(seed=1000)
    task = env.task_description
    env.close()
    batch = runner._obs_to_policy_batch(obs, task)
    model = runner.policy.model

    g = torch.Generator().manual_seed(123)
    cfg = runner.policy.config
    noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim,
                        generator=g).to(dev)

    pre = prefix_forward(runner.policy, batch)
    stock = sample_chunks(runner.policy, batch, n=1, noise=noise, prefix=pre)

    # -- 1. no-op equivalence -------------------------------------------------
    lc = LCState().to(dev)
    attach_lc_bias(model)
    z = lc.initial(1, dev)
    a_prev = torch.zeros(1, 10, 7, device=dev)
    h = pre.hidden.float()
    z = lc.step(z, a_prev, h)
    bias = lc.adarms_bias(z)
    assert float(bias.abs().max()) == 0.0, "zero-init W_z must emit zero bias"
    model._lc_bias = bias
    lc_out = sample_chunks(runner.policy, batch, n=1, noise=noise, prefix=pre)
    diff = float((stock - lc_out).abs().max())
    print(f"NO-OP: max|stock - lcflow_zeroinit| = {diff:.2e}")
    assert diff == 0.0, "zero-init LC-Flow must be bitwise stock"

    # -- 2. nonzero bias has control -----------------------------------------
    torch.manual_seed(0)
    model._lc_bias = 0.5 * torch.randn_like(bias)
    perturbed = sample_chunks(runner.policy, batch, n=1, noise=noise, prefix=pre)
    pdiff = float((stock - perturbed).abs().max())
    print(f"CONTROL: max|stock - biased| = {pdiff:.4f} "
          f"(action scale ~{float(stock.abs().mean()):.3f})")
    assert pdiff > 1e-3, "nonzero AdaRMS bias must change the sampled chunk"

    # -- 3. state pipeline shapes --------------------------------------------
    out = lc.outcome(lc.transition(z, a_prev))
    shapes = {k: tuple(v.shape) for k, v in out.items()}
    print(f"STATE: z {tuple(z.shape)} | outcome {shapes}")
    n_train = sum(p.numel() for p in lc.parameters() if p.requires_grad)
    print(f"trainable LC params: {n_train/1e6:.2f}M "
          f"(PrefixVLM + expert frozen)")
    model._lc_bias = None
    restored = sample_chunks(runner.policy, batch, n=1, noise=noise, prefix=pre)
    assert float((stock - restored).abs().max()) == 0.0, "bias=None must restore stock"
    print("ALL LC-FLOW WIRING TESTS PASSED")


if __name__ == "__main__":
    main()
