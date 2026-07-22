"""Sequence dataset over the decision-rate shards (framework v0.2 §9 source 1).

Windows of L consecutive decision steps from one demo. Splits by demo per task
(7/1/2 train/val/test, numeric demo order — contract §1 siblings rule).
Arms: 'siglip' | 'real_ll' | 'fused' (token concat). Predicate bits padded to
MAX_ATOMS=3 with a mask. Progress = fraction of the demo's decision steps done.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset

SEQ_DIR = Path("/home/stargazer/Desktop/vla_wm/datasets/seq_libero_10")
MAX_ATOMS = 3
ARMS = ("siglip", "real_ll", "fused")


def load_split(split: str) -> list[dict]:
    """split in {train, val, test} -> list of demo shards."""
    by_task = defaultdict(list)
    for p in SEQ_DIR.glob("task*_demo*.pt"):
        s = torch.load(p, weights_only=False)
        if s["success"]:
            by_task[s["task_id"]].append(s)
    out = []
    for tid in sorted(by_task):
        demos = sorted(by_task[tid], key=lambda s: s["demo"])
        n = len(demos)
        cut1, cut2 = int(n * 0.7), int(n * 0.8)
        sel = {"train": demos[:cut1], "val": demos[cut1:cut2],
               "test": demos[cut2:]}[split]
        out.extend(sel)
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
