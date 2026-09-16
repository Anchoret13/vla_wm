#!/usr/bin/env python
"""Can a bounded sustained residual re-engage chain3's third object at all?

    python scripts/eval_v250_basin_reach.py results/v250_chain3_round/chain3_lr2_cal_*

GOAL ANCHOR. This is the go/no-go readout for the approved round's intervention
interface, and it is deliberately read BEFORE any world model is trained. The
project's recorded failure mode is running five reward candidates and seven
isolations before asking whether the thing being ranked can move the task at all.

WHAT IT MEASURES, and what each outcome means.

chain3's frozen policy places two cans by step 260 in 92-94 of 96 episodes and then
parks the gripper ~29 cm above the basket for a median of 490 further steps: measured
per-chunk EEF motion falls from 66.95 mm to 5.95 mm (8.9%), and the distance to
`cream_cheese_1` RISES from 0.347 m at reset to ~0.553 m and stays there. Everything
the round wants to improve lives in that stall.

So the question that has to be answered before a predictor of intervention
consequences is worth building is whether the intervention interface can reach the
third object at all. The readouts are:

  d_min      the closest the EEF gets to the FINAL goal object, counted only over
             boundaries at which that object is the active atom. Measured on 24
             sigma=0 base episodes the median is 0.5015 m and 0/24 come within 5 cm,
             while the cream cheese is displaced by more than 2 mm in 2/24. Under
             NEAR_DISTANCE = 0.05 m the object would be graspable.
  reach_rate the fraction of episodes whose basin d_min drops below a stated
             threshold - the event a residual would have to make more likely.
  spread     the across-episode standard deviation of basin d_min. A residual that
             moves the arm to a DIFFERENT wrong place still creates the outcome
             variation an action-conditioned model needs in order to be identifiable
             at all; v122 measured the action gain on pure on-policy data at +0.0035
             with a CI crossing zero.
  damage     milestones 0-3, which the base policy already achieves reliably. A scale
             that destroys them is outside the usable trust region whatever it does
             to d_min.

What each outcome means for the round:

* no scale changes d_min or its spread -> the residual interface cannot reach the
  frontier. No model over it can help, and the round's finding is then about the
  ACTION SUPPORT, not about prediction.
* d_min moves, but every scale that moves it destroys milestones 0-3 -> the usable
  trust region is empty at this residual parameterisation.
* some scale increases spread while keeping milestones 0-3 -> that scale is the D1
  trust region, and there is a consequence for the model to predict.

PRE-REGISTERED: every arm collected is reported, with its scale and its measured
|delta|. Nothing here selects an arm by outcome; it reports the whole ladder so that
the D1 scale is chosen on the damage/spread trade-off and recorded before D1 runs.
No p-value is computed: at n=24 per arm this is a calibration, not a test, and
calling it one would be the error CLAUDE.md rule 2 records.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lcwm.phase_potential import NEAR_DISTANCE  # noqa: E402
from lcwm.v250_progress import tape_progress  # noqa: E402


def stage_reached(events: dict, n: int) -> int:
    ks = {int(k) for k in events}
    m = 0
    while m < n and m in ks:
        m += 1
    return m


def arm_row(run: Path, basin_from_chunk: int, thresholds: list[float]) -> dict:
    man = json.loads((run / "manifest.json").read_text())
    out = json.loads((run / "outcome.json").read_text())
    labels = torch.load(run / "labels.pt", weights_only=False)
    prog = tape_progress(labels)
    d = prog["d_active"].numpy()
    phi = prog["phi"].numpy()
    n_ms = len(labels["milestones"])
    basin_t = basin_from_chunk * int(man["c"])

    # RESTRICT to boundaries at which the FINAL goal atom is the active one - i.e. the
    # phase in which the third object really is the task.  Taking the basin minimum of
    # d_active without this restriction silently measures the gripper standing next to a
    # CAN during the tail of that can's placement: on the sigma=0 base arm the
    # unrestricted readout reports 9/24 episodes within 5 cm of "the active object",
    # while restricted to cream-cheese-active boundaries it is 0/24 with a median of
    # 0.50 m.  The two answers point opposite ways, so the restriction is not a detail.
    bits = labels["bits"].numpy().astype(bool)
    lab_ep, lab_t = labels["episode"].numpy(), labels["t"].numpy()
    n_atoms = bits.shape[1]
    final_active = bits[:, :n_atoms - 1].all(axis=1) & ~bits[:, n_atoms - 1]

    dmins, phimax, stages, nact = [], [], [], []
    for rec in out["episodes"]:
        m = (lab_ep == rec["idx"]) & (lab_t >= basin_t) & final_active
        if not m.any():
            continue
        dmins.append(float(d[m].min()))
        phimax.append(float(phi[m].max()))
        stages.append(stage_reached(rec["events"], n_ms))
        nact.append(int(m.sum()))
    dmins = np.asarray(dmins)
    stages = np.asarray(stages)
    row = {
        "tag": man["tag"], "scale": man["actor_scale"],
        "start_chunk": man["residual_start_chunk"],
        "n": len(dmins), "abs_delta": man["mean_abs_delta_applied"],
        "final_active_boundaries_median": float(np.median(nact)) if nact else 0.0,
        "d_min_median": float(np.median(dmins)),
        "d_min_best": float(dmins.min()),
        "d_min_sd": float(dmins.std()),
        "phi_max_median": float(np.median(phimax)),
        "stage_ge3": int((stages >= 4).sum()),
        "stage_mean": float(stages.mean()),
        "success": out["successes"],
        "per_seed": {str(r["seed"]): v for r, v in zip(
            [e for e in out["episodes"]
             if ((lab_ep == e["idx"]) & (lab_t >= basin_t) & final_active).any()],
            dmins.tolist())},
    }
    for th in thresholds:
        row[f"reach_{th}"] = int((dmins <= th).sum())
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--basin-from-chunk", type=int, default=26,
                    help="first chunk of the stall basin; 26 = step 260 = the measured "
                         "median last-progress step of the frozen policy")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[NEAR_DISTANCE, 0.15, 0.30])
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    rows = [arm_row(r, a.basin_from_chunk, a.thresholds)
            for r in a.runs if (r / "manifest.json").exists()]
    rows.sort(key=lambda r: (r["scale"] is not None, r["scale"] or 0.0))

    ths = a.thresholds
    head = (f"{'arm':16s} {'scale':>6s} {'|d|':>7s} {'n':>3s} "
            f"{'d_min med':>9s} {'best':>6s} {'sd':>6s} "
            + " ".join(f"{'<'+str(t):>7s}" for t in ths)
            + f" {'stg>=4':>6s} {'stg mean':>8s} {'succ':>4s}")
    print(head)
    print("-" * len(head))
    for r in rows:
        sc = "base" if r["scale"] is None else f"{r['scale']:.2f}"
        print(f"{r['tag'][:16]:16s} {sc:>6s} {r['abs_delta']:7.4f} {r['n']:3d} "
              f"{r['d_min_median']:9.4f} {r['d_min_best']:6.4f} {r['d_min_sd']:6.4f} "
              + " ".join(f"{r['reach_'+str(t)]:7d}" for t in ths)
              + f" {r['stage_ge3']:6d} {r['stage_mean']:8.3f} {r['success']:4d}")
    # PAIRED readout. Every arm runs the SAME seed family, so the informative
    # comparison is within-seed: a marginal count of 4/24 against 1/24 is four
    # episodes, while the paired difference uses all 24. No p-value is reported -
    # at n=24 this is a calibration and calling it a test is the error CLAUDE.md
    # rule 2 records.
    base = next((r for r in rows if r["scale"] is None), None)
    if base is not None and len(rows) > 1:
        print(f"\nPAIRED vs base, on the {len(base['per_seed'])} shared seeds "
              f"(negative = the residual gets CLOSER to the final object):")
        print(f"{'arm':16s} {'n':>3s} {'median d':>9s} {'mean d':>8s} "
              f"{'closer':>7s} {'farther':>7s} {'best gain':>10s}")
        for r in rows:
            if r["scale"] is None:
                continue
            shared = sorted(set(base["per_seed"]) & set(r["per_seed"]))
            diffs = np.array([r["per_seed"][k] - base["per_seed"][k] for k in shared])
            if not len(diffs):
                continue
            print(f"{r['tag'][:16]:16s} {len(diffs):3d} {np.median(diffs):9.4f} "
                  f"{diffs.mean():8.4f} {int((diffs < 0).sum()):7d} "
                  f"{int((diffs > 0).sum()):7d} {diffs.min():10.4f}")
            r["paired_median_delta"] = float(np.median(diffs))
            r["paired_closer"] = int((diffs < 0).sum())
            r["paired_n"] = len(diffs)

    if a.out:
        a.out.write_text(json.dumps({"basin_from_chunk": a.basin_from_chunk,
                                     "thresholds": ths, "arms": rows}, indent=2))
        print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
