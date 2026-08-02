#!/usr/bin/env python
"""V7.0.1 — oracle headroom by proposal family on strict-clean anchors.

Ordinal Borda preference scores from stored sibling-CRN outcomes only
(never legacy judged fields):

  canonical pool C = {u_0} ∪ {canonical candidates}:
      s_i = (1/(|C|-1)) Σ_{j≠i} pref_CRN(i,j)
  expanded pool adds atomic/distinct families with frozen mass
      α_canonical = α_atomic = α_distinct = 1/3,  m_j = α_f / n_f,
      s_i^mix = Σ_{j≠i} m_j pref_CRN(i,j) / Σ_{j≠i} m_j.

pref_CRN is the registered both-repeats-agree paired preference under
the frozen outcome tolerances. Servo is a reachability upper bound
only. Exact replay branches never enter any pool.

Two train-only gates:
  claim gate      — positive best-vs-u_0 aggregate margin across ≥2
                    independent train sources AND positive 95%
                    hierarchical task→source→anchor bootstrap lower
                    bound (10,000 replicates, seed 70001; tasks fixed
                    equal-weight strata over strata present);
  mechanical gate — ≥2 strict-positive train anchors from ≥2
                    independent sources, no integrity failure.

Dev outcomes were already read during V6.9: they are a source-held-out
development/calibration table, never blind, never used for training or
selection.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.task_automaton import paired_preference  # noqa: E402
from lcwm.v070_replay import GATE_VERSION  # noqa: E402

DATA = Path("/home/stargazer/Desktop/vla_wm/datasets/libero_loho_public_v1"
            "/v06_effect_crossed")
RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
AUDIT = RESULTS / "v070_policy" / "audit"
OUT = RESULTS / "v070_policy" / "oracle"
TASKS = ["loho_t1_drawer", "loho_t2_basket3", "loho_t3_tray",
         "loho_t4_tray", "loho_t5_drawer_cabinet"]
ALPHA = {"canonical": 1 / 3, "atomic": 1 / 3, "distinct": 1 / 3}
BOOT_N, BOOT_SEED = 10_000, 70_001


def family(key) -> str:
    k = str(key)
    if k == "0" or k in {str(i) for i in range(1, 16)}:
        return "canonical"
    if k in ("sub0", "sub1"):
        return "atomic"
    if k in ("sub2", "sub3"):
        return "distinct"
    if k == "support":
        return "servo"
    return "other"


def outs(grp, key):
    return [r["outcome"] for r in sorted(
        (r for r in grp["records"] if r["branch_key"] == key),
        key=lambda r: r["repeat"])]


def borda(grp, pool, tol, mix: bool):
    n_f = defaultdict(int)
    for k in pool:
        n_f[family(k)] += 1
    scores = {}
    for i in pool:
        num = den = 0.0
        for j in pool:
            if j == i:
                continue
            w = (ALPHA[family(j)] / n_f[family(j)]) if mix else 1.0
            num += w * paired_preference(outs(grp, i), outs(grp, j),
                                         tol)
            den += w
        scores[str(i)] = num / den if den else 0.0
    return scores


def hier_mean(rows, value):
    """task -> source -> anchor equal-weight mean over strata present."""
    per_task = []
    by_task = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_task[r["task"]][r["source_id"]].append(value(r))
    for task in sorted(by_task):
        src_means = [float(np.mean(v))
                     for v in by_task[task].values()]
        per_task.append(float(np.mean(src_means)))
    return float(np.mean(per_task)) if per_task else None


def hier_bootstrap_lb(rows, value, n=BOOT_N, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    by_task = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_task[r["task"]][r["source_id"]].append(value(r))
    tasks = sorted(by_task)
    stats = np.empty(n)
    for b in range(n):
        per_task = []
        for task in tasks:
            sources = list(by_task[task].values())
            pick = rng.integers(0, len(sources), len(sources))
            src_means = []
            for si in pick:
                vals = sources[si]
                idx = rng.integers(0, len(vals), len(vals))
                src_means.append(np.mean([vals[i] for i in idx]))
            per_task.append(np.mean(src_means))
        stats[b] = np.mean(per_task)
    return float(np.percentile(stats, 5)), float(np.percentile(
        stats, 2.5)), float(np.mean(stats))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    strict = json.loads(
        (AUDIT / "strict_replay_report.json").read_text())
    assert strict["teacher_construction_allowed"], \
        "strict audit refused teacher construction"
    clean = {s: set(strict["clean_anchor_lists"][s])
             for s in ("train", "dev")}

    anchor_rows = {"train": [], "dev": []}
    manifest = {}
    for p in sorted((DATA / "corrections_v069").glob("*.pt")):
        g = torch.load(p, weights_only=False)
        akey = f"{g['source_id']}_d{g['decision']}"
        split = g["split"]
        if akey not in clean.get(split, set()):
            continue
        tol = g["frozen_outcome_tolerances"]
        keys = [b["key"] for b in g["branch_summaries"]
                if b["kind"] != "replay"]
        canon_pool = [k for k in keys if family(k) == "canonical"]
        expanded_pool = [k for k in keys
                         if family(k) in ("canonical", "atomic",
                                          "distinct")]
        s_canon = borda(g, canon_pool, tol, mix=False)
        s_mix = borda(g, expanded_pool, tol, mix=True)
        pref_vs_u0 = {str(k): paired_preference(
            outs(g, k), outs(g, 0), tol) for k in keys if k != 0}
        best = {
            "canonical": max((pref_vs_u0[str(k)] for k in canon_pool
                              if k != 0), default=-1),
            "expanded": max((pref_vs_u0[str(k)]
                             for k in expanded_pool if k != 0),
                            default=-1),
            "servo": pref_vs_u0.get("support", -1),
        }
        row = {"anchor": akey, "task": g["task"],
               "source_id": g["source_id"],
               "decision": g["decision"],
               "anchor_form": g["anchor"]["form"],
               "best_vs_u0": best,
               "pref_vs_u0": pref_vs_u0,
               "borda_canonical": s_canon,
               "borda_mix": s_mix,
               "n_candidates": len(keys)}
        anchor_rows[split].append(row)
        manifest[akey] = {
            "file": p.name, "split": split,
            "pools": {"canonical": [str(k) for k in canon_pool],
                      "expanded": [str(k) for k in expanded_pool],
                      "servo": ["support"]},
            "alpha": ALPHA, "gate_version": GATE_VERSION}

    assert len(anchor_rows["train"]) == 14
    assert len(anchor_rows["dev"]) == 8

    def headroom_table(rows):
        table = {}
        for fam in ("canonical", "expanded", "servo"):
            pos = [r for r in rows if r["best_vs_u0"][fam] == 1]
            table[fam] = {
                "positive_anchors": sorted(r["anchor"] for r in pos),
                "n_positive_anchors": len(pos),
                "n_positive_sources": len(
                    {r["source_id"] for r in pos}),
                "win_tie_loss": [
                    sum(1 for r in rows if r["best_vs_u0"][fam] == v)
                    for v in (1, 0, -1)],
                "hier_margin": hier_mean(
                    rows, lambda r, f=fam: r["best_vs_u0"][f]),
            }
        return table

    train_table = headroom_table(anchor_rows["train"])
    dev_table = headroom_table(anchor_rows["dev"])

    gates = {}
    for fam in ("canonical", "expanded"):
        rows = anchor_rows["train"]
        pos_sources = train_table[fam]["n_positive_sources"]
        margin = train_table[fam]["hier_margin"]
        lb95, lb975, bmean = hier_bootstrap_lb(
            rows, lambda r, f=fam: r["best_vs_u0"][f])
        gates[fam] = {
            "claim_gate": {
                "aggregate_margin": margin,
                "n_positive_sources": pos_sources,
                "bootstrap_mean": bmean,
                "bootstrap_lb95": lb95,
                "n_replicates": BOOT_N, "seed": BOOT_SEED,
                "pass": bool(margin is not None and margin > 0
                             and pos_sources >= 2 and lb95 > 0)},
            "mechanical_gate": {
                "n_positive_anchors":
                    train_table[fam]["n_positive_anchors"],
                "n_positive_sources": pos_sources,
                "integrity": "strict audit passed",
                "pass": bool(
                    train_table[fam]["n_positive_anchors"] >= 2
                    and pos_sources >= 2)},
        }
    gates["servo"] = {"note": ("reachability upper bound only; never "
                               "satisfies the policy-teacher gate"),
                      "train_headroom":
                          train_table["servo"]["n_positive_anchors"]}

    (OUT / "anchor_manifest.json").write_text(
        json.dumps(manifest, indent=2))
    (OUT / "oracle_headroom_train.json").write_text(json.dumps(
        {"rows": anchor_rows["train"], "table": train_table},
        indent=2))
    (OUT / "oracle_headroom_dev_calibration.json").write_text(
        json.dumps({"note": ("source-held-out development/"
                             "calibration set; outcomes were read "
                             "during V6.9; NOT blind; never enters "
                             "training/sampling/checkpointing/"
                             "teacher construction"),
                    "rows": anchor_rows["dev"],
                    "table": dev_table}, indent=2))
    (OUT / "frozen_claim_gate.json").write_text(json.dumps(
        {f: gates[f]["claim_gate"] for f in ("canonical", "expanded")},
        indent=2))
    (OUT / "frozen_mechanical_gate.json").write_text(json.dumps(
        {f: gates[f]["mechanical_gate"]
         for f in ("canonical", "expanded")}, indent=2))
    print(json.dumps({"train": {f: {
        "pos_anchors": train_table[f]["n_positive_anchors"],
        "pos_sources": train_table[f]["n_positive_sources"],
        "margin": train_table[f]["hier_margin"]}
        for f in ("canonical", "expanded", "servo")},
        "gates": {f: {"claim": gates[f]["claim_gate"]["pass"],
                      "mechanical":
                          gates[f]["mechanical_gate"]["pass"]}
                  for f in ("canonical", "expanded")}},
        indent=1), flush=True)
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
