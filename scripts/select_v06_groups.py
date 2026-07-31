#!/usr/bin/env python
"""V6.2 — freeze tranche-A replay tolerances, classify effect_resolved,
retain the chronologically first accepted group per slot per source.

Tolerances (registered): per-component 95th percentile of |cand0 − replay|
over ALL tranche-A train-source exact repeats:
  eef_pos, eef_quat, gripper (from q), obj_pos (max over objects),
  valid_bits (Hamming; if its 95th pct is 0, any bit difference is real).
effect_resolved: any candidate PAIR (0..3) differs beyond tolerance in
obj_pos or valid_bits. Action-space L2 never qualifies a group.
Rejected groups stay recorded. Same frozen tolerances apply to tranche B.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = Path("/home/stargazer/Desktop/vla_wm/datasets/libero_loho_public_v1"
            "/v06_effect_crossed")
OUT = REPO_ROOT / "results" / "libero_loho_public_v1"

COMPONENTS = ("eef_pos", "eef_quat", "gripper", "obj_pos", "valid_bits")


def branch_components(branch, reference) -> dict:
    dq = (branch["q_after"] - reference["q_after"]).abs()
    obj = float(np.abs(branch["obj_after"]
                       - reference["obj_after"]).max())
    bits = sum(a != b for a, b in zip(branch["valid_after"],
                                      reference["valid_after"]))
    return {"eef_pos": float(dq[:3].max()),
            "eef_quat": float(dq[3:7].max()),
            "gripper": float(dq[7:].max()),
            "obj_pos": obj, "valid_bits": float(bits)}


def main() -> None:
    sources = [torch.load(p, weights_only=False)
               for p in sorted((DATA / "sources").glob("*.pt"))]
    tranche_a = [s for s in sources if s["tranche"] == "A"]
    assert tranche_a, "no tranche-A sources found"

    # ---- freeze tolerances from exact repeats ---------------------------
    repeats = {k: [] for k in COMPONENTS}
    for s in tranche_a:
        if s["split"] != "train":
            continue
        for audit in s["audits"]:
            cand0 = next(b for b in audit["branches"]
                         if b["kind"] == "candidate"
                         and b["candidate"] == 0)
            replay = next(b for b in audit["branches"]
                          if b["kind"] == "replay")
            comp = branch_components(replay, cand0)
            for k in COMPONENTS:
                repeats[k].append(comp[k])
    tolerances = {k: float(np.percentile(v, 95)) if v else 0.0
                  for k, v in repeats.items()}
    n_rep = len(repeats["obj_pos"])

    # ---- classify and select -------------------------------------------
    accepted, rejected = [], []
    counts: dict[str, dict[str, int]] = {}
    for s in sources:
        chosen_slots = set()
        task_counts = counts.setdefault(
            s["task"], {"audited": 0, "effect_resolved": 0, "accepted": 0})
        for audit in s["audits"]:
            task_counts["audited"] += 1
            cands = [b for b in audit["branches"]
                     if b["kind"] == "candidate"]
            resolved = False
            for i in range(len(cands)):
                for j in range(i + 1, len(cands)):
                    comp = branch_components(cands[i], cands[j])
                    if (comp["obj_pos"] > tolerances["obj_pos"]
                            or comp["valid_bits"]
                            > tolerances["valid_bits"]):
                        resolved = True
            if resolved:
                task_counts["effect_resolved"] += 1
            entry = {"source_id": s["source_id"], "task": s["task"],
                     "seed": s["seed"], "provenance": s["provenance"],
                     "split": s["split"], "slot": audit["slot"],
                     "decision": audit["decision"],
                     "effect_resolved": resolved}
            if resolved and audit["slot"] not in chosen_slots:
                chosen_slots.add(audit["slot"])
                accepted.append(entry)
                task_counts["accepted"] += 1
            else:
                rejected.append(entry)

    payload = {
        "schema": "v06_group_selection_v1",
        "n_exact_repeats": n_rep,
        "tolerances_95pct": tolerances,
        "accepted": accepted,
        "rejected": rejected,
        "per_task_counts": counts,
        "rule": ("chronologically first effect_resolved per slot per "
                 "source; rejected groups keep first-ten/replay records, "
                 "no continuations"),
    }
    (OUT / "v06_group_selection.json").write_text(
        json.dumps(payload, indent=2))
    print(json.dumps({"tolerances": tolerances,
                      "n_repeats": n_rep,
                      "accepted": len(accepted),
                      "rejected": len(rejected),
                      "per_task": counts}, indent=2))
    print(f"-> {OUT / 'v06_group_selection.json'}")


if __name__ == "__main__":
    main()
