#!/usr/bin/env python
"""H7.5 analysis — registered matched effects on the public matrix.

Per registered H7.5 formulas, per task and pooled:
    selection_reset = reset_wm − reset_random
    selection_rec   = rec_wm − rec_random
    history_wm      = rec_wm − reset_wm
    history_random  = rec_random − reset_random
plus rec_wm − stock and the selection×history interaction
(selection_rec − selection_reset). Metrics: terminal SR and Q_public.
Seeds are paired (common random numbers): effects are means of per-seed
differences. n is always reported alongside. No inferential statistics
are attached at n=5 seeds/cell; per-seed differences are listed raw.
The v0.4 legacy reference is reported separately, outside the factorial.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT = REPO_ROOT / "results" / "libero_loho_public_v1" / "eval_matrix"

ARMS = ("stock", "v05_reset_wm", "v05_reset_random",
        "v05_recurrent_wm", "v05_recurrent_random")
EFFECTS = {
    "selection_reset": ("v05_reset_wm", "v05_reset_random"),
    "selection_rec": ("v05_recurrent_wm", "v05_recurrent_random"),
    "history_wm": ("v05_recurrent_wm", "v05_reset_wm"),
    "history_random": ("v05_recurrent_random", "v05_reset_random"),
    "rec_wm_vs_stock": ("v05_recurrent_wm", "stock"),
    "reset_wm_vs_stock": ("v05_reset_wm", "stock"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-ids", nargs="+", required=True)
    parser.add_argument("--panel", default="development")
    args = parser.parse_args()

    records = []
    for line in (ROOT / args.panel / "records.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["run_id"] in args.run_ids:
            records.append(r)

    cell = {}  # (task, seed, arm) -> record
    for r in records:
        key = (r["task"], r["seed"], r["arm"])
        assert key not in cell, f"duplicate record {key}"
        cell[key] = r

    tasks = sorted({r["task"] for r in records})
    seeds = sorted({r["seed"] for r in records})
    arms_present = sorted({r["arm"] for r in records})

    out = {"run_ids": args.run_ids, "tasks": tasks, "seeds": seeds,
           "arms": arms_present, "per_arm": {}, "effects": {},
           "per_subgoal": {}}

    for arm in arms_present:
        rows = [cell[(t, s, arm)] for t in tasks for s in seeds
                if (t, s, arm) in cell]
        per_task = {}
        for t in tasks:
            trows = [cell[(t, s, arm)] for s in seeds
                     if (t, s, arm) in cell]
            if trows:
                per_task[t] = {
                    "n": len(trows),
                    "sr": sum(r["success"] for r in trows) / len(trows),
                    "q_mean": sum(r["q_public"] for r in trows)
                    / len(trows),
                    "q_values": [r["q_public"] for r in trows],
                    "successes": [s for s in seeds
                                  if (t, s, arm) in cell
                                  and cell[(t, s, arm)]["success"]],
                    "first_unresolved": [
                        r["first_unresolved_subgoal"] for r in trows],
                    "n_invalidated": sum(
                        len(r["invalidated_subgoals"]) for r in trows),
                }
        out["per_arm"][arm] = {
            "n": len(rows),
            "sr": (sum(r["success"] for r in rows) / len(rows)
                   if rows else None),
            "q_mean": (sum(r["q_public"] for r in rows) / len(rows)
                       if rows else None),
            "per_task": per_task,
        }

    for name, (a, b) in EFFECTS.items():
        if a not in arms_present or b not in arms_present:
            continue
        per_task = {}
        pooled = []
        for t in tasks:
            diffs = []
            for s in seeds:
                if (t, s, a) in cell and (t, s, b) in cell:
                    diffs.append({
                        "seed": s,
                        "d_q": cell[(t, s, a)]["q_public"]
                        - cell[(t, s, b)]["q_public"],
                        "d_sr": int(cell[(t, s, a)]["success"])
                        - int(cell[(t, s, b)]["success"]),
                    })
            if diffs:
                per_task[t] = {
                    "n_pairs": len(diffs),
                    "mean_d_q": sum(d["d_q"] for d in diffs) / len(diffs),
                    "mean_d_sr": sum(d["d_sr"] for d in diffs)
                    / len(diffs),
                    "per_seed": diffs,
                }
                pooled.extend(diffs)
        out["effects"][name] = {
            "arms": [a, b],
            "per_task": per_task,
            "pooled_n_pairs": len(pooled),
            "pooled_mean_d_q": (sum(d["d_q"] for d in pooled)
                                / len(pooled) if pooled else None),
            "pooled_mean_d_sr": (sum(d["d_sr"] for d in pooled)
                                 / len(pooled) if pooled else None),
        }
    if ("selection_rec" in out["effects"]
            and "selection_reset" in out["effects"]):
        sr_ = out["effects"]["selection_rec"]["pooled_mean_d_q"]
        ss = out["effects"]["selection_reset"]["pooled_mean_d_q"]
        if sr_ is not None and ss is not None:
            out["effects"]["selection_x_history_interaction_d_q"] = sr_ - ss

    # per-subgoal first-completion counts per arm (pooled over seeds)
    sub = defaultdict(lambda: defaultdict(int))
    for r in records:
        for sg in r["subgoal_completion_steps"]:
            sub[r["arm"]][f"{r['task']}::{sg}"] += 1
    out["per_subgoal"] = {arm: dict(v) for arm, v in sub.items()}

    dest = ROOT / args.panel / ("analysis_" + "_".join(args.run_ids)
                                + ".json")
    dest.write_text(json.dumps(out, indent=2))

    print(f"arms (n episodes): " + ", ".join(
        f"{a}={out['per_arm'][a]['n']}" for a in arms_present))
    for a in arms_present:
        pa = out["per_arm"][a]
        print(f"  {a}: SR {pa['sr']:.3f}  Q {pa['q_mean']:.3f}")
    for name, e in out["effects"].items():
        if isinstance(e, dict):
            print(f"  {name}: dQ {e['pooled_mean_d_q']:+.4f} "
                  f"dSR {e['pooled_mean_d_sr']:+.3f} "
                  f"(n={e['pooled_n_pairs']} pairs)")
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
