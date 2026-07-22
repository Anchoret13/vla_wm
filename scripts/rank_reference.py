#!/usr/bin/env python
"""Q1 of the rank investigation: the data's own rank budget (pre-registered)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.candidate_a import CandidateA  # noqa: E402
from lcwm.wm_data import WMSeqDataset  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def eff_rank(x: torch.Tensor) -> float:
    xc = x.float() - x.float().mean(0, keepdim=True)
    s = torch.linalg.svdvals(xc)
    p = (s ** 2) / (s ** 2).sum()
    return float(torch.exp(-(p * (p + 1e-12).log()).sum()))


def main() -> None:
    dev = "cuda"
    te = WMSeqDataset("test", "siglip")
    loader = DataLoader(te, batch_size=64, num_workers=2)
    model = CandidateA().to(dev)
    model.load_state_dict(torch.load(
        "/home/stargazer/Desktop/vla_wm/checkpoints/wm_sweep/siglip_seed0.pt",
        weights_only=True))
    model.eval()

    g = torch.Generator().manual_seed(0)
    proj = (torch.randn(2048, 1536, generator=g) / 2048**0.5).to(dev)
    carries, tok_proj = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(dev) for k, v in batch.items()}
            e = model.e_task(batch["task_id"])
            zt = model.encode(batch, e, ema=True)          # (B,L,M,d)
            carries.append(zt[:, 2:].reshape(-1, zt.shape[-2] * zt.shape[-1]).cpu())
            tp = batch["tokens"][:, 2:].mean(dim=2) @ proj  # (B,L-2,1536)
            tok_proj.append(tp.reshape(-1, 1536).cpu())
    out = {
        "ema_carry_eff_rank": eff_rank(torch.cat(carries)),
        "raw_token_randproj_eff_rank": eff_rank(torch.cat(tok_proj)),
        "n_samples": int(sum(c.shape[0] for c in carries)),
    }
    (REPO_ROOT / "results" / "wm_sweep" / "rank_reference.json").write_text(
        json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
