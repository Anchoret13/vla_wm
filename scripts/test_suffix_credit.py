#!/usr/bin/env python
"""Suffix credit contract regression test (2026-07-24 audit correction 4).

The action expert attends bidirectionally over all 50 chunk positions, so
before this contract the unexecuted target suffix could leak into first-ten
predictions even with the loss masked to positions 0:10. The contract
(`credit_bounded_actions`, applied inside cached_branch_flow_loss /
branch_flow_loss) replaces every position at or beyond the executed length
with zeros before the target enters the flow graph.

Assertions:
1. INVARIANCE: resampling the unexecuted suffix while holding the executed
   prefix, noise, time, and bias fixed leaves per-branch losses AND the
   adapter gradient bitwise unchanged.
2. SENSITIVITY (negative control): changing the executed prefix changes the
   per-branch losses — the invariance is not vacuous.
3. Per-branch executed lengths shorter than ten are honored: content at or
   beyond a branch's executed length is also invariant.
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
    cached_branch_flow_loss,
    freeze_pi05_base,
)
from lcwm.sampler import prefix_forward  # noqa: E402


def branch_losses_and_grad(runner, pre, lc, actions, weights, noise, time,
                           executed_lengths=None):
    lc.zero_grad(set_to_none=True)
    with torch.enable_grad():
        z = lc.posterior(pre.hidden, pre.pad_masks)
        bias = lc.adarms_bias(z).expand(actions.shape[0], -1)
        loss, parts = cached_branch_flow_loss(
            runner.policy,
            pre,
            actions,
            bias,
            weights,
            executed_lengths=executed_lengths,
            noise=noise,
            time=time,
        )
        loss.backward()
    grad = lc.wz_out.weight.grad.detach().clone()
    return parts["per_branch"].detach().clone(), grad


def main() -> None:
    dev = "cuda"
    runner = Pi05Runner(suite_name="libero_10")
    env = make_task_env("libero_10", 0)
    obs, _ = env.reset(seed=1000)
    task = env.task_description
    env.close()
    batch = runner._obs_to_policy_batch(obs, task)
    freeze_pi05_base(runner.policy)
    pre = prefix_forward(runner.policy, batch)
    lc = LCState().to(dev)

    cfg = runner.policy.config
    g = torch.Generator().manual_seed(7)
    base = torch.randn(2, cfg.chunk_size, 7, generator=g).to(dev) * 0.3
    noise = torch.randn(
        1, cfg.chunk_size, cfg.max_action_dim,
        generator=torch.Generator().manual_seed(11),
    ).to(dev)
    time = torch.tensor([0.5], device=dev)
    weights = torch.tensor([1.0, 1.0], device=dev)

    # -- 1. suffix invariance -------------------------------------------------
    variant = base.clone()
    variant[:, 10:] = torch.randn(
        2, cfg.chunk_size - 10, 7,
        generator=torch.Generator().manual_seed(23),
    ).to(dev) * 5.0
    ref_losses, ref_grad = branch_losses_and_grad(
        runner, pre, lc, base, weights, noise, time
    )
    var_losses, var_grad = branch_losses_and_grad(
        runner, pre, lc, variant, weights, noise, time
    )
    loss_diff = float((ref_losses - var_losses).abs().max())
    grad_diff = float((ref_grad - var_grad).abs().max())
    print(f"SUFFIX-INVARIANCE: max|Δloss|={loss_diff:.3e} "
          f"max|Δgrad|={grad_diff:.3e}")
    assert loss_diff == 0.0, "first-ten losses depend on unexecuted suffix"
    assert grad_diff == 0.0, "adapter gradient depends on unexecuted suffix"

    # -- 2. executed-prefix sensitivity (negative control) --------------------
    changed = base.clone()
    changed[:, :10] += 0.25
    chg_losses, _ = branch_losses_and_grad(
        runner, pre, lc, changed, weights, noise, time
    )
    sens = float((ref_losses - chg_losses).abs().max())
    print(f"PREFIX-SENSITIVITY: max|Δloss|={sens:.3e}")
    assert sens > 0.0, "losses must respond to the executed prefix"

    # -- 3. per-branch executed lengths honored -------------------------------
    lengths = torch.tensor([10, 6], device=dev)
    short_ref, short_ref_grad = branch_losses_and_grad(
        runner, pre, lc, base, weights, noise, time,
        executed_lengths=lengths,
    )
    tail_variant = base.clone()
    tail_variant[1, 6:10] += 3.0  # beyond branch 1's executed length
    short_var, short_var_grad = branch_losses_and_grad(
        runner, pre, lc, tail_variant, weights, noise, time,
        executed_lengths=lengths,
    )
    short_diff = float((short_ref - short_var).abs().max())
    short_grad_diff = float((short_ref_grad - short_var_grad).abs().max())
    print(f"SHORT-BRANCH: max|Δloss|={short_diff:.3e} "
          f"max|Δgrad|={short_grad_diff:.3e}")
    assert short_diff == 0.0 and short_grad_diff == 0.0, (
        "content beyond a branch's executed length leaked into its credit"
    )
    print("SUFFIX CREDIT CONTRACT TEST PASSED")


if __name__ == "__main__":
    main()
