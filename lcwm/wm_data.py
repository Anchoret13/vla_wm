"""Sequence dataset over the decision-rate shards (framework v0.2 §9 source 1).

Windows of L consecutive decision steps from one demo. Splits come EXCLUSIVELY
from the explicit split manifest (train/dev/audit; contract §1 siblings rule).
Arms: 'siglip' | 'real_ll' | 'fused' (token concat). Predicate bits padded to
MAX_ATOMS=3 with a mask. Progress = fraction of the demo's decision steps done.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

SEQ_DIR = Path(os.environ.get(
    "LCWM_SEQ_DIR", "/home/stargazer/Desktop/vla_wm/datasets/seq_libero_10_v2"))
MAX_ATOMS = 3
ARMS = ("siglip", "real_ll", "fused")


def load_split(split: str) -> list[dict]:
    """split in {train, dev, audit} — EXPLICIT ids from the split manifest
    (review 2026-07-22 items 1-2: no silent success-filtering, no on-the-fly
    percentage splits; the audit split stays sealed until the frozen
    verification)."""
    man = json.loads((SEQ_DIR / "split_manifest.json").read_text())
    out = []
    for tid_s, m in man["tasks"].items():
        for d in m[split]:
            s = torch.load(SEQ_DIR / f"task{tid_s}_demo{d}.pt",
                           weights_only=False)
            assert s["success"], f"manifest lists failed shard task{tid_s}/{d}"
            out.append(s)
    return out


class WMSeqDataset(Dataset):
    def __init__(self, split: str, arm: str, seq_len: int = 8):
        assert arm in ARMS
        self.arm, self.L = arm, seq_len
        self.windows = []          # (shard_idx, start)
        self.shards = load_split(split)
        for si, s in enumerate(self.shards):
            n = len(s["steps"])
            for t0 in range(0, max(1, n - seq_len + 1)):
                if t0 + seq_len <= n:
                    self.windows.append((si, t0))

    def __len__(self):
        return len(self.windows)

    def _tokens(self, step) -> torch.Tensor:
        if self.arm == "fused":
            return torch.cat([step["siglip"], step["real_ll"]], dim=0).float()
        return step[self.arm].float()

    def __getitem__(self, i):
        si, t0 = self.windows[i]
        s = self.shards[si]
        steps = s["steps"][t0:t0 + self.L]
        n_steps = len(s["steps"])
        n_atoms = len(s["goal_atoms"])
        bits = torch.zeros(self.L, MAX_ATOMS)
        mask = torch.zeros(self.L, MAX_ATOMS)
        mask[:, :n_atoms] = 1.0
        for j, st in enumerate(steps):
            bits[j, :n_atoms] = st["predicate_bits"].float()
        actions = torch.stack([st["action_block"] for st in steps])       # (L,10,7)
        a_prev = torch.cat([torch.zeros_like(actions[:1]), actions[:-1]])  # shifted
        return {
            "tokens": torch.stack([self._tokens(st) for st in steps]),    # (L,N,2048)
            "q": torch.stack([st["q"] for st in steps]),                  # (L,9)
            "actions": actions,
            "a_prev": a_prev,
            "task_id": torch.tensor(s["task_id"]),
            "bits": bits, "bits_mask": mask,
            "progress": torch.tensor(
                [(t0 + j + 1) / n_steps for j in range(self.L)]).float(),
        }
