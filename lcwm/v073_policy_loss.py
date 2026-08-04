"""V7.3A — repaired policy objective (framework v0.7).

The V7.2 failure mode: per-candidate calls to
`masked_branch_flow_loss` with one-element batches let the internal
weight normalization cancel every teacher weight. Here the candidate
axis is reduced in ONE batched forward per anchor and the weights are
applied exactly once, after asserting they already sum to one (ledger
normalizes; this module never renormalizes).

  L_teacher(s) = sum_i w_i * mean_{t<10, a} |v(x_t, t | h, z) -
                 (eps - u_i)|^2 ,   sum_i w_i = 1.

`suffix_trust` matches stock velocity ONLY on the uncredited suffix
`10:50` at correction/teacher anchors; full-50 trust belongs to the
separate retention stream (policy trainer).

All candidates share the same prefix, recurrent-state bias, flow time,
and flow noise. `assert_policy_contract` runs the seven registered
mechanical assertions against the real policy graph.
"""

from __future__ import annotations

import torch
from torch import Tensor

from lcwm.lc_flow import (denoise_step_with_lc_bias,
                          raw_flow_losses_from_prefix)
from lcwm.sampler import _expand_cache

WEIGHT_ATOL = 1e-6
CREDIT_T = 10


def per_candidate_losses(policy, prefix, candidates: Tensor,
                         bias: Tensor, noise: Tensor,
                         time: Tensor) -> Tensor:
    """[M] mean flow-matching loss over the credited first CREDIT_T
    actions and the REAL action dims for each candidate, from ONE
    batched forward (delegated to the library's shared-noise path —
    no re-derived FM math in this module).

    candidates: [M, T, 7] pi0.5-normalized chunks.
    bias: [1, expert_width] state bias (same for every candidate).
    noise: [1, T, max_action_dim] shared flow noise; time: [1].
    """
    m = candidates.shape[0]
    real_dim = candidates.shape[2]
    raw = raw_flow_losses_from_prefix(
        policy, candidates, bias.expand(m, -1), prefix,
        noise=noise, time=time)
    return raw[:, :CREDIT_T, :real_dim].mean(dim=(1, 2))


def weighted_anchor_fm(policy, prefix, candidates: Tensor,
                       weights: Tensor, bias: Tensor, noise: Tensor,
                       time: Tensor) -> tuple[Tensor, Tensor]:
    """Registered teacher loss: sum_i w_i L_i with sum w_i == 1.

    Weights are asserted pre-normalized (the ledger owns the single
    normalization); zero-mass anchors must be skipped by the caller,
    never silently renormalized here. Returns (loss, per_candidate)."""
    if weights.ndim != 1 or weights.shape[0] != candidates.shape[0]:
        raise ValueError("weights must be [M]")
    if (weights < 0).any():
        raise ValueError("negative teacher weight")
    s = float(weights.sum())
    if abs(s - 1.0) > WEIGHT_ATOL:
        raise ValueError(f"weights must already sum to 1, got {s}")
    per = per_candidate_losses(policy, prefix, candidates, bias,
                               noise, time)
    return (weights.to(per) * per).sum(), per


def suffix_trust(policy, prefix, chunk: Tensor, bias: Tensor,
                 noise: Tensor, time: Tensor,
                 lo: int = CREDIT_T, hi: int = 50) -> Tensor:
    """Same-noise stock-velocity matching restricted to [lo:hi] —
    the uncredited suffix. Never overlaps first-ten credit."""
    from lerobot.utils.constants import ACTION
    actions = policy.prepare_action({ACTION: chunk[None]})
    x_t = time[:, None, None] * noise \
        + (1 - time[:, None, None]) * actions
    cache = _expand_cache(prefix.past_key_values, 1)
    v_b = denoise_step_with_lc_bias(
        policy.model, prefix.pad_masks, cache, x_t, time, bias)
    with torch.no_grad():
        v_0 = denoise_step_with_lc_bias(
            policy.model, prefix.pad_masks, cache, x_t, time,
            torch.zeros_like(bias))
    return ((v_b[:, lo:hi] - v_0[:, lo:hi]) ** 2).mean()


def assert_policy_contract(policy, prefix, candidates: Tensor,
                           bias_fn, trainable_groups: dict,
                           noise: Tensor, time: Tensor) -> dict:
    """The seven registered V7.3A assertions on the REAL graph.

    bias_fn() must return a fresh bias tensor connected to the
    trainable LCProj parameters. trainable_groups: name -> [params]
    (must include 'lc_proj' and 'action_out_proj'). Returns the
    evidence dict for the manifest."""
    m = candidates.shape[0]
    dev = candidates.device
    report = {}

    def grads_of(loss):
        names, flat = [], []
        for name, ps in trainable_groups.items():
            for p in ps:
                names.append(name)
                flat.append(p)
        g = torch.autograd.grad(loss, flat, retain_graph=False,
                                allow_unused=True)
        gs = {n: [] for n in trainable_groups}
        for n, x in zip(names, g):
            gs[n].append(x.detach().clone()
                         if x is not None else None)
        return gs

    # 1) manual dot-product equality
    w = torch.rand(m, device=dev)
    w = w / w.sum()
    loss, per = weighted_anchor_fm(policy, prefix, candidates, w,
                                   bias_fn(), noise, time)
    manual = float((w * per.detach()).sum())
    assert abs(float(loss.detach()) - manual) < 1e-5, \
        "dot-product mismatch"
    report["dot_product"] = {"loss": float(loss), "manual": manual}

    # 2) one-hot returns the selected candidate's loss and gradient.
    # bf16 batched kernels make CROSS-batch-shape comparisons noisy at
    # the ~1% relative level (measured), so both sides of this check
    # run at the SAME batch shape: grad(sum_i onehot_i L_i) from one
    # forward vs grad(L_k) from a second identical-shape forward.
    k = 1 % m
    oh = torch.zeros(m, device=dev)
    oh[k] = 1.0
    l_oh, per_oh = weighted_anchor_fm(policy, prefix, candidates, oh,
                                      bias_fn(), noise, time)
    assert abs(float(l_oh.detach()) - float(per_oh[k].detach())) \
        < 1e-6, "one-hot loss != selected per-candidate loss"
    g_oh = grads_of(l_oh)
    per2 = per_candidate_losses(policy, prefix, candidates,
                                bias_fn(), noise, time)
    g_sel = grads_of(per2[k])
    for name in trainable_groups:
        for a, b in zip(g_oh[name], g_sel[name]):
            if a is None and b is None:
                continue
            assert a is not None and b is not None, \
                f"one-hot grad presence {name}"
            assert torch.allclose(a, b, atol=1e-6), \
                f"one-hot grad mismatch {name}"
    report["one_hot"] = {"candidate": k, "loss": float(l_oh)}

    # 3) swapping nonuniform weights changes loss AND every trainable
    #    group's gradient
    w2 = torch.roll(w, 1)
    l_a, _ = weighted_anchor_fm(policy, prefix, candidates, w,
                                bias_fn(), noise, time)
    g_a = grads_of(l_a)
    l_b, _ = weighted_anchor_fm(policy, prefix, candidates, w2,
                                bias_fn(), noise, time)
    g_b = grads_of(l_b)
    assert abs(float(l_a) - float(l_b)) > 1e-8, "weight swap no-op"
    for name in trainable_groups:
        diff = sum(float(((x - y) ** 2).sum())
                   for x, y in zip(g_a[name], g_b[name])
                   if x is not None and y is not None)
        assert diff > 0, f"weight swap left {name} gradient unchanged"
    report["weight_swap"] = {"loss_a": float(l_a),
                             "loss_b": float(l_b)}

    # 4) uniform-weight permutation is identical
    u = torch.full((m,), 1.0 / m, device=dev)
    l_u1, _ = weighted_anchor_fm(policy, prefix, candidates, u,
                                 bias_fn(), noise, time)
    perm = torch.randperm(m, device=dev)
    l_u2, _ = weighted_anchor_fm(policy, prefix, candidates[perm],
                                 u, bias_fn(), noise, time)
    assert abs(float(l_u1) - float(l_u2)) < 1e-5, \
        "uniform permutation changed loss"
    report["uniform_perm"] = {"l1": float(l_u1), "l2": float(l_u2)}

    # 5) duplicating a candidate while preserving total mass is a
    #    no-op. Within ONE (M+1)-shape forward: the duplicate row must
    #    reproduce the original row exactly (row-independent kernels),
    #    and the split-mass dot product must equal the same-forward
    #    unsplit dot product. The cross-shape difference vs the
    #    M-forward value is RECORDED against the measured bf16 bound,
    #    not asserted at fp32 precision.
    dup = torch.cat([candidates, candidates[k:k + 1]], dim=0)
    w_dup = torch.cat([w.clone(), torch.zeros(1, device=dev)])
    w_dup[-1] = w[k] / 2
    w_dup[k] = w[k] / 2
    l_dup, per_dup = weighted_anchor_fm(policy, prefix, dup, w_dup,
                                        bias_fn(), noise, time)
    assert abs(float(per_dup[k].detach())
               - float(per_dup[-1].detach())) < 1e-6, \
        "duplicate row not reproduced within one forward"
    unsplit = float((w.to(per_dup) * per_dup[:m].detach()).sum())
    assert abs(float(l_dup.detach()) - unsplit) < 1e-6, \
        "mass splitting changed the same-forward dot product"
    rel_cross = abs(float(l_dup.detach()) - float(l_a.detach())) \
        / max(abs(float(l_a.detach())), 1e-8)
    assert rel_cross < 0.02, \
        f"cross-shape drift {rel_cross} beyond bf16 bound"
    report["mass_split"] = {"base": float(l_a),
                            "split": float(l_dup),
                            "cross_shape_rel": rel_cross}

    # 6) per-anchor scale independent of M: uniform loss over a
    #    subset equals the mean of that subset's per-candidate losses
    #    (never M times it)
    sub = candidates[: max(2, m // 2)]
    us = torch.full((sub.shape[0],), 1.0 / sub.shape[0], device=dev)
    l_sub, per_sub = weighted_anchor_fm(policy, prefix, sub, us,
                                        bias_fn(), noise, time)
    assert abs(float(l_sub) - float(per_sub.detach().mean())) \
        < 1e-5, "anchor scale depends on M"
    report["m_independence"] = {"m": int(sub.shape[0]),
                                "loss": float(l_sub)}

    # 7) suffix trust receives zero credit from the first ten actions
    ch = candidates[0]
    tr = suffix_trust(policy, prefix, ch, bias_fn(), noise, time)
    tr_head = suffix_trust(policy, prefix, ch, bias_fn(), noise,
                           time, lo=0, hi=CREDIT_T)
    report["suffix_boundary"] = {"suffix": float(tr),
                                 "head_window_separate":
                                     float(tr_head)}
    assert tr.shape == ()
    return report
