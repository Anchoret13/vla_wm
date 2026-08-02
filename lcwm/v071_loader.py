"""V7.1.1F — deterministic source-split replay loader.

Sampling universe = the 724 unique transitions in the frozen union
manifest (audit-only rows excluded). Ordering and sampling are
task→source→anchor→unique-action hierarchical and fully deterministic:
the same manifest yields byte-identical orderings on every run.
Language relabels and candidate multiplicity never increase a physical
branch's sampling mass (they are joined per transition at train time,
not enumerated as extra rows here).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch

TASKS = ["loho_t1_drawer", "loho_t2_basket3", "loho_t3_tray",
         "loho_t4_tray", "loho_t5_drawer_cabinet"]


class V071Loader:
    def __init__(self, union_dir: str | Path):
        self.dir = Path(union_dir)
        union = json.loads(
            (self.dir / "union_manifest.json").read_text())
        self.rows = [r for r in union["rows"]
                     if not r["audit_only"]]
        self.by_id = {r["transition_id"]: r for r in self.rows}
        self.roots = {k: Path(v)
                      for k, v in union["replay_roots"].items()}
        self._shard_cache: dict[str, dict] = {}

    def _hierarchy(self, split: str):
        h = defaultdict(lambda: defaultdict(
            lambda: defaultdict(list)))
        for r in self.rows:
            if r["split"] != split:
                continue
            h[r["task"]][r["source_id"]][r["anchor"]].append(r)
        return h

    def ordered_ids(self, split: str) -> list[str]:
        """task→source→anchor→branch_key round-robin, deterministic."""
        h = self._hierarchy(split)
        queues = []
        for task in TASKS:
            if task not in h:
                continue
            for sid in sorted(h[task]):
                for akey in sorted(h[task][sid]):
                    items = sorted(
                        h[task][sid][akey],
                        key=lambda r: str(r["branch_key"]))
                    queues.append([r["transition_id"]
                                   for r in items])
        out, i = [], 0
        while any(queues):
            for q in queues:
                if i < len(q):
                    out.append(q[i])
            i += 1
            if all(i >= len(q) for q in queues):
                break
        return out

    def exposure_report(self, split: str) -> dict:
        h = self._hierarchy(split)
        return {
            "tasks": {t: sum(len(v2) for s in h[t].values()
                             for v2 in s.values())
                      for t in h},
            "sources": {t: len(h[t]) for t in h},
            "anchors": {t: sum(len(s) for s in h[t].values())
                        for t in h},
        }

    def load_transition(self, transition_id: str) -> dict:
        r = self.by_id[transition_id]
        shard_path = (self.roots[r["run"]] / "shards" / r["shard"])
        key = str(shard_path)
        if key not in self._shard_cache:
            if len(self._shard_cache) > 3:
                self._shard_cache.clear()
            self._shard_cache[key] = torch.load(
                shard_path, weights_only=False)
        shard = self._shard_cache[key]
        for tr in shard["transitions"]:
            if tr["transition_id"] == transition_id:
                return {**tr, "meta": r}
        raise KeyError(transition_id)
