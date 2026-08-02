#!/usr/bin/env python
"""V7.0.4 analysis — paired anchor-replay margins and the registered
routing.

M_A(x,y) = Mean_{task→source→anchor} pref_fresh(x,y) over strata
present in A; repeats form one paired verdict (both-repeats rule,
frozen tolerances), never extra rows.

Hard promotion: an oracle arm beats fresh stock AND its
coupling-matched hard-random control on train-positive anchors
(M>0, no negative source-level margin); with adequate dev coverage
(≥2 fresh-clean positive-headroom anchors from ≥2 sources) the same
on dev; on null anchors M(oracle,stock)≥0, no success loss, no damage
increase. Capacity-only: train passes but dev coverage insufficient.
Tie-break: larger equal-task dev margin → lower null regression →
lower matched-noise action-shift RMS → centered.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.task_automaton import paired_preference  # noqa: E402

RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
V070 = RESULTS / "v070_policy"
ORACLE_ARMS = {"hard_affine": "hard_random_affine",
               "hard_centered": "hard_random_centered"}


def hier_mean(rows, value):
    by_task = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_task[r["task"]][r["source_id"]].append(value(r))
    per_task = []
    for task in sorted(by_task):
        per_task.append(float(np.mean(
            [float(np.mean(v)) for v in by_task[task].values()])))
    return float(np.mean(per_task)) if per_task else None


def source_margins(rows, value):
    by_src = defaultdict(list)
    for r in rows:
        by_src[r["source_id"]].append(value(r))
    return {s: float(np.mean(v)) for s, v in by_src.items()}


def main() -> None:
    records = []
    for line in (V070 / "anchor_replay"
                 / "records.jsonl").read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    headroom_dev = json.loads(
        (V070 / "oracle"
         / "oracle_headroom_dev_calibration.json").read_text())
    dev_pos_anchors = set(
        headroom_dev["table"]["expanded"]["positive_anchors"])

    fresh_clean = [r for r in records if r["fresh_replay_clean"]]
    excluded = [r["anchor"] for r in records
                if not r["fresh_replay_clean"]]
    sets = {
        "train_positive": [r for r in fresh_clean
                           if r["split"] == "train" and r["positive"]],
        "train_null": [r for r in fresh_clean
                       if r["split"] == "train" and not r["positive"]],
        "dev_positive": [r for r in fresh_clean
                         if r["split"] == "dev"
                         and r["anchor"] in dev_pos_anchors],
        "dev_all": [r for r in fresh_clean if r["split"] == "dev"],
    }

    def pref_of(r, x, y):
        return paired_preference(r["outcomes"][x], r["outcomes"][y],
                                 r["tolerances"])

    matrix = {}
    for arm in list(ORACLE_ARMS) + list(ORACLE_ARMS.values()) + [
            "stored_oracle"]:
        matrix[arm] = {}
        for sname, rows in sets.items():
            rows_a = [r for r in rows if arm in r["outcomes"]]
            if not rows_a:
                matrix[arm][sname] = None
                continue
            entry = {
                "n_anchors": len(rows_a),
                "n_sources": len({r["source_id"] for r in rows_a}),
                "M_vs_stock": hier_mean(
                    rows_a, lambda r: pref_of(r, arm, "fresh_u0")),
                "source_margins_vs_stock": source_margins(
                    rows_a, lambda r: pref_of(r, arm, "fresh_u0")),
            }
            if arm in ORACLE_ARMS:
                ctrl = ORACLE_ARMS[arm]
                entry["M_vs_control"] = hier_mean(
                    rows_a, lambda r: pref_of(r, arm, ctrl))
                entry["source_margins_vs_control"] = source_margins(
                    rows_a, lambda r: pref_of(r, arm, ctrl))
            if "stored_oracle" in rows_a[0]["outcomes"] and \
                    arm != "stored_oracle":
                rows_o = [r for r in rows_a
                          if "stored_oracle" in r["outcomes"]]
                if rows_o:
                    entry["M_vs_stored_oracle"] = hier_mean(
                        rows_o,
                        lambda r: pref_of(r, arm, "stored_oracle"))
            # null contract components
            entry["success_delta"] = hier_mean(
                rows_a, lambda r:
                float(any(o["success_by_100"]
                          for o in r["outcomes"][arm]))
                - float(any(o["success_by_100"]
                            for o in r["outcomes"]["fresh_u0"])))
            entry["damage_delta"] = hier_mean(
                rows_a, lambda r:
                float(np.mean([-o["neg_damage"]
                               for o in r["outcomes"][arm]]))
                - float(np.mean([-o["neg_damage"]
                                 for o in r["outcomes"]["fresh_u0"]])))
            matrix[arm][sname] = entry

    dev_cov_ok = (len(sets["dev_positive"]) >= 2
                  and len({r["source_id"]
                           for r in sets["dev_positive"]}) >= 2)

    def passes_train(arm):
        e = matrix[arm]["train_positive"]
        if not e or e["M_vs_stock"] is None:
            return False
        return (e["M_vs_stock"] > 0 and e["M_vs_control"] > 0
                and all(v >= 0 for v in
                        e["source_margins_vs_stock"].values())
                and all(v >= 0 for v in
                        e["source_margins_vs_control"].values()))

    def null_ok(arm):
        e = matrix[arm]["train_null"]
        if not e:
            return True
        return (e["M_vs_stock"] >= 0 and e["success_delta"] >= 0
                and e["damage_delta"] <= 0)

    def passes_dev(arm):
        e = matrix[arm]["dev_positive"]
        if not e or e["M_vs_stock"] is None:
            return False
        # V7.1.0A fix: the dev gate must also require nonnegative
        # source-level margins versus the matched CONTROL (previously
        # only the vs-stock source margins were checked)
        return (e["M_vs_stock"] > 0
                and e.get("M_vs_control", -1) > 0
                and all(v >= 0 for v in
                        e["source_margins_vs_stock"].values())
                and all(v >= 0 for v in
                        e.get("source_margins_vs_control",
                              {}).values()))

    verdicts = {}
    for arm in ORACLE_ARMS:
        verdicts[arm] = {
            "train_pass": passes_train(arm),
            "null_ok": null_ok(arm),
            "dev_pass": (passes_dev(arm) if dev_cov_ok else None),
        }
    full_pass = [a for a, v in verdicts.items()
                 if v["train_pass"] and v["null_ok"]
                 and v["dev_pass"] is True]
    capacity_only = [a for a, v in verdicts.items()
                     if v["train_pass"] and v["null_ok"]
                     and not dev_cov_ok]
    if full_pass:
        routing = {"outcome": "hard_promotion",
                   "arms": full_pass,
                   "next": ("soft-oracle vs soft-permuted on the "
                            "winning coupling, then one selected "
                            "LoHo dev run")}
    elif capacity_only:
        routing = {"outcome": "capacity_only",
                   "arms": capacity_only,
                   "next": ("exploratory LoHo coverage probe with the "
                            "train-selected arm; no generalization/"
                            "promotion claim; soft skipped")}
    elif any(v["train_pass"] for v in verdicts.values()):
        routing = {"outcome": "train_pass_null_violation",
                   "next": "interface selectivity failure recorded"}
    else:
        routing = {"outcome": "interface_failure_matrix",
                   "next": ("no arm beats stock+control on train "
                            "positive anchors; localized interface "
                            "failure; centered-vs-affine and "
                            "stored-oracle rows localize it")}
    out = {
        "n_records": len(records),
        "fresh_replay_excluded_anchors": excluded,
        "dev_coverage_adequate": dev_cov_ok,
        "n_dev_positive_fresh_clean": len(sets["dev_positive"]),
        "matrix": matrix,
        "verdicts": verdicts,
        "routing": routing,
    }
    (V070 / "anchor_replay" / "anchor_replay_matrix.json").write_text(
        json.dumps(out, indent=2))
    show = {a: {s: (None if matrix[a][s] is None else {
        k: matrix[a][s][k] for k in ("n_anchors", "M_vs_stock",
                                     "M_vs_control")
        if k in matrix[a][s]})
        for s in sets} for a in matrix}
    print(json.dumps({"matrix": show, "verdicts": verdicts,
                      "routing": routing}, indent=1), flush=True)


if __name__ == "__main__":
    main()
