#!/usr/bin/env python
"""V6.7.6 analysis — registered contrasts on the repaired public matrix.

Per task and pooled, paired by common-random-number seeds (means of
per-seed differences, raw per-seed diffs listed; no inferential
statistics at n=5 seeds/cell):
    wm_vs_stock       v067_recurrent_wm − stock
    wm_vs_random      v067_recurrent_wm − v067_recurrent_random
    wm_vs_gt_rec      v067_recurrent_wm − v067_gt_recurrent
    gt_rec_vs_stock   v067_gt_recurrent − stock
    gt_cur_vs_stock   v067_gt_current − stock
    history_gt        v067_gt_recurrent − v067_gt_current
Metrics: terminal SR (primary), valid Q, ordered prefix P, damage,
plus first-unresolved subgoal counts and success time.

Registered promotion rule (unchanged): WM must improve pooled SR over
stock, improve paired Q over stock AND matched random, and be
nonnegative in paired Q versus recurrent GT. A development win freezes
the method and opens sealed confirmation.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT = REPO_ROOT / "results" / "libero_loho_public_v1" / "v067_eval"

EFFECTS = {
    "wm_vs_stock": ("v067_recurrent_wm", "stock"),
    "wm_vs_random": ("v067_recurrent_wm", "v067_recurrent_random"),
    "wm_vs_gt_rec": ("v067_recurrent_wm", "v067_gt_recurrent"),
    "gt_rec_vs_stock": ("v067_gt_recurrent", "stock"),
    "gt_cur_vs_stock": ("v067_gt_current", "stock"),
    "history_gt": ("v067_gt_recurrent", "v067_gt_current"),
}
METRICS = ("success", "q_valid", "p_valid", "damage_unrecovered")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-ids", nargs="+", required=True)
    parser.add_argument("--panel", default="development")
    args = parser.parse_args()

    records = []
    for line in (ROOT / args.panel
                 / "records.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["run_id"] in args.run_ids:
                records.append(r)
    cell = {}
    for r in records:
        key = (r["task"], r["seed"], r["arm"])
        assert key not in cell, f"duplicate record {key}"
        cell[key] = r
    tasks = sorted({r["task"] for r in records})
    seeds = sorted({r["seed"] for r in records})
    arms = sorted({r["arm"] for r in records})

    print(f"panel={args.panel} tasks={len(tasks)} seeds={len(seeds)} "
          f"arms={arms}")
    print("\n== per-arm means (pooled over tasks x seeds) ==")
    for arm in arms:
        rs = [cell[(t, s, arm)] for t in tasks for s in seeds
              if (t, s, arm) in cell]
        n = len(rs)
        line = f"{arm:24s} n={n:3d} "
        for m in METRICS:
            line += f"{m}={sum(r[m] for r in rs) / n:.3f} "
        print(line)
        fu = Counter(r["first_unresolved"] for r in rs
                     if r["first_unresolved"])
        print(f"{'':24s} first_unresolved: "
              f"{dict(fu.most_common(3))}")

    print("\n== paired effects (mean of per-seed differences) ==")
    for name, (a, b) in EFFECTS.items():
        if a not in arms or b not in arms:
            continue
        print(f"\n-- {name}: {a} - {b} --")
        pooled = defaultdict(list)
        for t in tasks:
            diffs = defaultdict(list)
            for s in seeds:
                if (t, s, a) in cell and (t, s, b) in cell:
                    for m in METRICS:
                        d = cell[(t, s, a)][m] - cell[(t, s, b)][m]
                        diffs[m].append(d)
                        pooled[m].append(d)
            if diffs["success"]:
                print(f"  {t:26s} n={len(diffs['success'])} "
                      + " ".join(
                          f"d{m}={sum(v) / len(v):+.3f}"
                          for m, v in diffs.items()))
        if pooled["success"]:
            n = len(pooled["success"])
            print(f"  {'POOLED':26s} n={n} " + " ".join(
                f"d{m}={sum(v) / n:+.3f}" for m, v in pooled.items()))
            print(f"  per-seed dSR: "
                  f"{[round(v, 3) for v in pooled['success']]}")
            print(f"  per-seed dQ:  "
                  f"{[round(v, 3) for v in pooled['q_valid']]}")

    # registered promotion rule
    need = [("v067_recurrent_wm", "stock"),
            ("v067_recurrent_wm", "v067_recurrent_random"),
            ("v067_recurrent_wm", "v067_gt_recurrent")]
    if all(a in arms and b in arms for a, b in need):
        def pooled_diff(a, b, m):
            vals = [cell[(t, s, a)][m] - cell[(t, s, b)][m]
                    for t in tasks for s in seeds
                    if (t, s, a) in cell and (t, s, b) in cell]
            return sum(vals) / len(vals)
        promo = {
            "wm_sr_gt_stock": pooled_diff(
                "v067_recurrent_wm", "stock", "success") > 0,
            "wm_q_gt_stock": pooled_diff(
                "v067_recurrent_wm", "stock", "q_valid") > 0,
            "wm_q_gt_random": pooled_diff(
                "v067_recurrent_wm", "v067_recurrent_random",
                "q_valid") > 0,
            "wm_q_ge_gt_rec": pooled_diff(
                "v067_recurrent_wm", "v067_gt_recurrent",
                "q_valid") >= 0,
        }
        promo["PROMOTION"] = all(promo.values())
        print(f"\n== registered promotion rule == {promo}")


if __name__ == "__main__":
    main()
