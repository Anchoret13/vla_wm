#!/usr/bin/env python
"""v2 split manifest: train/dev continue v1's ids (already consumed roles);
audit = v1's never-loaded val demos + newly collected demos. Sealed until the
frozen verification (review 2026-07-22 item 2)."""
import glob
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
SEQ = Path(sys.argv[1] if len(sys.argv) > 1 else
           "/home/stargazer/Desktop/vla_wm/datasets/seq_libero_10_v2")
v1 = json.loads((REPO / "results/wm_sweep/split_manifest_v1.json").read_text())

present = {}
for p in glob.glob(str(SEQ / "task*_demo*.pt")):
    s = torch.load(p, weights_only=False)
    present.setdefault(s["task_id"], {})[s["demo"]] = s["success"]

man = {"note": "v2: terminal records included; explicit ids; train/dev inherit "
               "v1 consumed roles; audit sealed (v1 unused val + new demos).",
       "tasks": {}}
for tid, dem in sorted(present.items()):
    m1 = v1["tasks"][str(tid)]
    ok = {d for d, s in dem.items() if s}
    fail = sorted(d for d, s in dem.items() if not s)
    train = [d for d in m1["train"] if d in ok]
    dev = [d for d in m1["dev_consumed"] if d in ok]
    known = set(train) | set(dev)
    audit = sorted(ok - known)
    man["tasks"][tid] = {"train": train, "dev": dev, "audit": audit,
                         "excluded_failures": fail}
(SEQ / "split_manifest.json").write_text(json.dumps(man, indent=1))
print(json.dumps({t: {k: len(v) if isinstance(v, list) else v
                      for k, v in m.items()} for t, m in man["tasks"].items()},
                 indent=1))
