#!/usr/bin/env python
"""V7.0.2 — anchor-hard teacher + matched hard-random control manifests
(expanded pool, MECHANICAL-gate label; canonical manifest would be kept
separate if canonical ever passes).

- positive anchor: eligibility = candidates with both-repeat preference
  vs u_0 equal to +1 (expanded pool: canonical+atomic+distinct; servo
  and replay excluded); target = maximum registered Borda-mix score;
  tied maxima resolved by the frozen SHA256 of
  (run_id, source_id, decision);
- hard-random: identical positive-anchor mask and hierarchical mass;
  frozen hash-chosen NONSTOCK candidate from the oracle candidate's
  family where possible (else the same registered pool), excluding the
  oracle candidate;
- null/tied anchors carry hierarchical state mass through stock-flow
  trust only — one sampled u_0 is not the stock distribution and never
  becomes a w_0=1 correction target;
- hierarchical mass task→source→anchor over the 14 strict-clean train
  anchors.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.v070_replay import GATE_VERSION  # noqa: E402

DATA = Path("/home/stargazer/Desktop/vla_wm/datasets/libero_loho_public_v1"
            "/v06_effect_crossed")
RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
ORACLE = RESULTS / "v070_policy" / "oracle"
RUN_ID = "v070_hard1"
FAMILY_OF = {**{str(i): "canonical" for i in range(1, 16)},
             "sub0": "atomic", "sub1": "atomic",
             "sub2": "distinct", "sub3": "distinct"}


def tie_hash(source_id: str, decision: int) -> int:
    return int.from_bytes(hashlib.sha256(
        f"{RUN_ID}|{source_id}|{decision}".encode()).digest()[:8],
        "big")


def main() -> None:
    headroom = json.loads(
        (ORACLE / "oracle_headroom_train.json").read_text())
    mech = json.loads(
        (ORACLE / "frozen_mechanical_gate.json").read_text())
    assert mech["expanded"]["pass"], "expanded mechanical gate not met"
    rows = headroom["rows"]

    # hierarchical mass task -> source -> anchor over clean anchors
    by_task = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_task[r["task"]][r["source_id"]].append(r)
    mass = {}
    n_tasks = len(by_task)
    for task, srcs in by_task.items():
        for sid, anchors in srcs.items():
            for r in anchors:
                mass[r["anchor"]] = (1 / n_tasks) * (1 / len(srcs)) \
                    * (1 / len(anchors))

    teacher_rows, random_rows = [], []
    for r in rows:
        akey = r["anchor"]
        eligible = [k for k, v in r["pref_vs_u0"].items()
                    if v == 1 and k in FAMILY_OF]
        base = {"anchor": akey, "task": r["task"],
                "source_id": r["source_id"],
                "decision": r["decision"], "mass": mass[akey],
                "gate_label": "expanded_mechanical",
                "gate_version": GATE_VERSION}
        if not eligible:
            teacher_rows.append({**base, "positive": False,
                                 "target": None})
            random_rows.append({**base, "positive": False,
                                "target": None})
            continue
        smax = max(r["borda_mix"][k] for k in eligible)
        tied = sorted(k for k in eligible
                      if r["borda_mix"][k] == smax)
        h = tie_hash(r["source_id"], r["decision"])
        oracle_key = tied[h % len(tied)]
        fam = FAMILY_OF[oracle_key]
        fam_pool = [k for k in r["pref_vs_u0"]
                    if k in FAMILY_OF and FAMILY_OF[k] == fam
                    and k != oracle_key]
        rand_pool = fam_pool or [k for k in r["pref_vs_u0"]
                                 if k in FAMILY_OF
                                 and k != oracle_key]
        rand_key = sorted(rand_pool)[h % len(rand_pool)]
        grp = torch.load(
            DATA / "corrections_v069"
            / f"{r['source_id']}_d{r['decision']}.pt",
            weights_only=False)
        by_key = {str(b["key"]): b for b in grp["branch_summaries"]}

        def payload(key):
            b = by_key[key]
            return {"key": key, "family": FAMILY_OF[key],
                    "provenance": b["provenance"],
                    "behavior_goal_id": b["behavior_goal_id"],
                    "chunk_norm": b["chunk_norm"],
                    "borda_mix": r["borda_mix"][key],
                    "pref_vs_u0": r["pref_vs_u0"][key],
                    "executed_outcomes": [
                        rec["outcome"] for rec in sorted(
                            (x for x in grp["records"]
                             if str(x["branch_key"]) == key),
                            key=lambda x: x["repeat"])]}
        teacher_rows.append({
            **base, "positive": True, "tie_break_hash": h,
            "tied_max": tied, "target": payload(oracle_key)})
        random_rows.append({
            **base, "positive": True, "tie_break_hash": h,
            "random_pool": sorted(rand_pool),
            "target": payload(rand_key)})
        print(f"[hard] {akey}: oracle={oracle_key} "
              f"(borda {smax:+.3f}, tied {tied}) "
              f"random={rand_key} (pool {sorted(rand_pool)})",
              flush=True)

    n_pos = sum(1 for t in teacher_rows if t["positive"])
    assert n_pos == 2
    assert abs(sum(mass.values()) - 1.0) < 1e-9
    for name, rws in (("hard_teacher_manifest.pt", teacher_rows),
                      ("hard_random_manifest.pt", random_rows)):
        torch.save({"schema": "v070_hard_manifest_v1",
                    "run_schema": "v070", "run_id": RUN_ID,
                    "pool": "expanded",
                    "gate_label": "expanded_mechanical",
                    "gate_version": GATE_VERSION,
                    "n_anchors": len(rws), "n_positive": n_pos,
                    "rows": rws}, ORACLE / name)
    print(f"-> {ORACLE} (teacher + matched random; "
          f"{n_pos} positive / {len(teacher_rows)} anchors)",
          flush=True)


if __name__ == "__main__":
    main()
