#!/usr/bin/env python
"""V7.3A contract test — the seven registered assertions on the REAL
policy graph.

Fixture: one real V7.2 teacher anchor (re-used observation + its M
stored candidate chunks). Trainables exactly the v0.7 boundary:
bias-free zero-init LCProj and stock-initialized action_out_proj
(weight+bias). Also asserts stock flow parity at LCProj=0 with stock
action_out_proj. Writes
2026-08-03_v073_policy_contract/contract_report.json. Assertion-only:
no parameter updates, no training.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

from lcwm.v073_policy_loss import (assert_policy_contract,  # noqa: E402
                                   per_candidate_losses, suffix_trust)

RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
TEACH = RESULTS / "2026-08-03_v072_teacher_r1"
OUT = RESULTS / "2026-08-03_v073_policy_contract"


def main() -> None:
    from lcwm.chassis import Pi05Runner
    from lcwm.lc_flow import freeze_pi05_base, sample_chunks_lc
    from lcwm.sampler import prefix_forward, sample_chunks
    from lcwm.v067_lineage import flow_noise

    device = torch.device("cuda")
    torch.manual_seed(0)
    OUT.mkdir(exist_ok=True)
    runner = Pi05Runner(suite_name="libero_10")
    cfg = runner.policy.config
    freeze_pi05_base(runner.policy)
    for p in runner.policy.parameters():
        p.requires_grad_(False)
    aop = runner.policy.model.action_out_proj
    for p in aop.parameters():
        p.requires_grad_(True)

    # real anchor fixture
    led = [json.loads(x) for x in
           (TEACH / "teacher_ledger_pre_outcome.jsonl").open()]
    a = led[0]
    sid, d = a["source_id"], a["decision"]
    src = torch.load(TEACH / "teacher_sources" / f"{sid}.pt",
                     weights_only=False)
    chunks_store = torch.load(TEACH / "candidate_chunks.pt",
                              weights_only=False)
    cids = sorted(a["weights"])
    cands = torch.stack([
        torch.as_tensor(np.asarray(
            chunks_store[f"{sid}_d{d}_{c}"])).float()
        for c in cids]).to(device)
    with torch.no_grad():
        b = runner._obs_to_policy_batch(
            src["rows"][d]["obs"], src["language_canonical"])
        prefix = prefix_forward(runner.policy, b)

    lc_proj = nn.Linear(384, 1024, bias=False).to(device)
    nn.init.zeros_(lc_proj.weight)
    # a fixed nonzero weight so bias gradients are informative for
    # the swap assertions (zero-init would zero some grad paths)
    with torch.no_grad():
        lc_proj.weight.add_(0.01 * torch.randn_like(lc_proj.weight))
    pool = torch.randn(1, 384, device=device)

    def bias_fn():
        return lc_proj(pool)

    groups = {"lc_proj": list(lc_proj.parameters()),
              "action_out_proj": list(aop.parameters())}
    noise = flow_noise(1234, cfg.chunk_size,
                       cfg.max_action_dim).to(device)
    g = torch.Generator().manual_seed(1234)
    time = torch.rand(1, generator=g).to(device)

    report = assert_policy_contract(runner.policy, prefix, cands,
                                    bias_fn, groups, noise, time)

    # stock flow parity: LCProj=0 + stock action_out_proj
    with torch.no_grad():
        zbias = torch.zeros(1, 1024, device=device)
        nz = flow_noise(777, cfg.chunk_size,
                        cfg.max_action_dim).to(device)
        ch_lc = sample_chunks_lc(runner.policy, b, zbias, n=1,
                                 noise=nz, prefix=prefix)
        ch_st = sample_chunks(runner.policy, b, n=1, noise=nz,
                              prefix=prefix)
        parity = float((ch_lc - ch_st).abs().max())
    report["stock_parity_lcproj0"] = parity
    assert parity < 1e-5, f"stock parity violated: {parity}"

    # smoke: per-candidate losses finite; suffix trust finite
    with torch.no_grad():
        per = per_candidate_losses(runner.policy, prefix, cands,
                                   bias_fn(), noise, time)
        tr = suffix_trust(runner.policy, prefix, cands[0],
                          bias_fn(), noise, time)
    assert torch.isfinite(per).all() and torch.isfinite(tr)
    report["fixture"] = {"anchor": f"{sid}_d{d}", "M": len(cids)}
    report["git_sha"] = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True,
        text=True, cwd=REPO_ROOT).stdout.strip()
    (OUT / "contract_report.json").write_text(
        json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items()
                      if k != "git_sha"}, indent=1), flush=True)
    print("CONTRACT PASS", flush=True)


if __name__ == "__main__":
    main()
