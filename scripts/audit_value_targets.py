#!/usr/bin/env python
"""P3 value-target audit (read-only; basis for the MC/TD(λ)/FQE [DEC]).

Inventories every continuation-like signal in the existing data, with the
generating policy of each — the framework requires every value target to
record its generating policy (v0.4 §4.3):

1. cache demo episodes: length to success/terminal (policy: expert demo);
2. chain source histories: decisions to snapshot; episode terminal Q from the
   chain exam protocol is NOT stored per history — noted, not invented;
3. branch-group continuations: per-branch atomic-prompt continuations
   (policy: frozen pi0.5 under the recorded prompt + noise), lengths and
   success rates.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.seq_prefix_cache import CACHE_DIR  # noqa: E402
from lcwm.seq_windows import BRANCH_DIR_DEFAULT  # noqa: E402


def main() -> None:
    print("== cache demo episodes (policy: expert demonstration) ==")
    lengths, success_steps = [], []
    for path in sorted(Path(CACHE_DIR).glob("task*_demo*.pt")):
        data = torch.load(path, weights_only=False)
        lengths.append(int(data["T_actions"]))
        if data["first_success_t"] is not None:
            success_steps.append(int(data["first_success_t"]))
    print(
        f"episodes {len(lengths)}, action-lengths "
        f"min/median/max {min(lengths)}/{sorted(lengths)[len(lengths)//2]}/"
        f"{max(lengths)}, all reach success: "
        f"{len(success_steps)}/{len(lengths)}"
    )

    print("== branch continuations (policy: frozen pi0.5, recorded prompt) ==")
    manifest = json.loads(
        (BRANCH_DIR_DEFAULT / "branch_manifest.json").read_text()
    )
    total = succeeded = 0
    steps_when_success = []
    lengths_all = []
    for split, paths in manifest["splits"].items():
        for relative in paths:
            group = torch.load(
                BRANCH_DIR_DEFAULT / relative, weights_only=False
            )
            mask = group["continuation_mask"].bool()
            total += int(mask.sum())
            ok = group["continuation_success"].bool() & mask
            succeeded += int(ok.sum())
            lengths_all += group["continuation_steps"][mask].tolist()
            steps_when_success += group["continuation_steps"][ok].tolist()
    print(
        f"continuations {total}, successes {succeeded} "
        f"({100*succeeded/max(total,1):.1f}%), lengths "
        f"min/max {min(lengths_all)}/{max(lengths_all)}"
        + (
            f", success steps {sorted(steps_when_success)}"
            if steps_when_success
            else ""
        )
    )
    print("== chain source histories ==")
    print(
        "terminal chain outcomes are recorded in results/chains protocol "
        "files, not in the history tensors; usable for episode-level V "
        "targets only via the exam records"
    )


if __name__ == "__main__":
    main()
