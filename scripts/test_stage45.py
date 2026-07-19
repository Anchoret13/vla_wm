#!/usr/bin/env python
"""Stage-4/5 smoke & validation on one libero_10 state.

1. Equivalence: batched sample_chunks vs the stock sample_actions path, same noise.
2. Quick diversity probe: N=16 pairwise action-space L2 at the initial state.
3. Feature tap: shapes, timing, constant-vs-real prompt feature shift.
4. Timing: shared-prefix batched sampling vs N sequential predict_action_chunk.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

from lcwm.chassis import Pi05Runner, make_task_env  # noqa: E402
from lcwm.sampler import prefix_forward, sample_chunks  # noqa: E402
from lcwm.taps import feature_shift, tap_step  # noqa: E402

SUITE, TASK_ID, N = "libero_10", 0, 16


def main() -> None:
    runner = Pi05Runner(suite_name=SUITE)
    env = make_task_env(SUITE, TASK_ID)
    obs, _ = env.reset(seed=1000)
    task = env.task_description
    batch = runner._obs_to_policy_batch(obs, task)
    model, cfg = runner.policy.model, runner.policy.config
    report: dict = {}

    # -- 1. equivalence ------------------------------------------------------
    g = torch.Generator().manual_seed(0)
    z3 = torch.randn(3, cfg.chunk_size, cfg.max_action_dim, generator=g).to("cuda")
    images, img_masks = runner.policy._preprocess_images(batch)
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
    tokens, masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]

    seq = []
    for i in range(3):  # stock path, one draw at a time
        a = model.sample_actions(images, img_masks, tokens, masks, noise=z3[i:i + 1])
        seq.append(a[:, :, :7])
    seq = torch.cat(seq, dim=0)
    ours = sample_chunks(runner.policy, batch, n=3, noise=z3)
    diff = (seq - ours).abs()
    report["equivalence"] = {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "action_scale_mean_abs": float(seq.abs().mean()),
    }

    # -- 2. quick diversity probe -------------------------------------------
    chunks = sample_chunks(runner.policy, batch, n=N, seed=42)  # (N, 50, 7)
    flat = chunks.reshape(N, -1)
    d = torch.cdist(flat, flat)
    iu = torch.triu_indices(N, N, offset=1)
    pw = d[iu[0], iu[1]]
    per_step = (chunks[:, None] - chunks[None, :]).norm(dim=-1)  # (N,N,50)
    report["diversity_N16"] = {
        "pairwise_l2_mean": float(pw.mean()),
        "pairwise_l2_min": float(pw.min()),
        "pairwise_l2_max": float(pw.max()),
        "per_step_l2_first10_mean": float(per_step[iu[0], iu[1], :10].mean()),
        "per_step_l2_last10_mean": float(per_step[iu[0], iu[1], -10:].mean()),
        "chunk_norm_mean": float(flat.norm(dim=-1).mean()),
    }

    # -- 3. feature tap + shift ---------------------------------------------
    t0 = time.perf_counter()
    feats = tap_step(runner, obs, task)
    tap_ms = (time.perf_counter() - t0) * 1e3
    report["tap"] = {
        "h_img_shape": list(feats.h_img.shape),
        "e_lang_shape": list(feats.e_lang.shape),
        "two_pass_ms": round(tap_ms, 1),
        "shift_vs_constant_prompt": feature_shift(runner, obs, task),
    }

    # -- 4. timing -----------------------------------------------------------
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    pre = prefix_forward(runner.policy, batch)
    torch.cuda.synchronize()
    t_prefix = time.perf_counter() - t0

    t0 = time.perf_counter()
    sample_chunks(runner.policy, batch, n=N, prefix=pre)
    torch.cuda.synchronize()
    t_batched = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(N):
        runner.policy.predict_action_chunk(batch)
    torch.cuda.synchronize()
    t_seq = time.perf_counter() - t0
    report["timing"] = {
        "prefix_forward_s": round(t_prefix, 3),
        "batched_16_denoise_s": round(t_batched, 3),
        "sequential_16_full_s": round(t_seq, 3),
        "speedup_vs_sequential": round(t_seq / (t_prefix + t_batched), 1),
    }

    env.close()
    out = REPO_ROOT / "results" / "stage45_smoke.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
