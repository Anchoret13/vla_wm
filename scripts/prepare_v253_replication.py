#!/usr/bin/env python
"""Prepare the cross-initialisation replication of the 2026-09-13 chain3 round.

    # measure only; builds nothing, touches no GPU
    python scripts/prepare_v253_replication.py --restart 1

    # after reading the measurement, build the two untrained control arms
    python scripts/prepare_v253_replication.py --restart 1 --build

WHAT THIS ANSWERS.  The 2026-09-13 round deployed restart 0 of 3 for every arm and
reported `M1 vs M0` at p = 1.0000.  CLAUDE.md rule 9 and the round's own
pre-registration name the same refutation: the configuration retrained from a second
initialisation.  Actor variance on this task is of the same order as the effect being
chased - the calibration measured milestone-4 at 4/24, 3/24 and 1/24 across three
independent untrained actors at one scale - so a single restart cannot separate a null
from init noise.  `actor_1.pt` and `actor_2.pt` already exist for all five trained arms;
what is missing is the untrained control, which was built at one init only, and a
measurement that says how different the restart-1 arms actually are BEFORE they are
deployed.

WHY THE UNTRAINED ARM NEEDS A DECISION RATHER THAN A REBUILD.  The 09-13 run-level
check fired on `rand`: it executed mean |delta| = 0.3129 against a trained-arm panel
mean of 0.2262, +38%, outside the pre-registered 25% band, so `M1 vs rand` is
magnitude-confounded and the driver was right to refuse to auto-report it.  An
untrained residual cannot match both the trained arms' BOUND and their REALISED
magnitude: training shrinks the output, randomisation does not.  This script therefore
prepares both arms and declares the difference rather than silently picking one:

    rand_bound  scale 0.80, a second init - the pure replication of the 09-13 control,
                matched on the bound, still magnitude-confounded by construction
    rand_mag    the scale whose predicted on-basin magnitude is closest to the
                restart-1 trained-arm mean - matched on the realised magnitude, at a
                deliberately different bound

WHY THE PREDICTOR IS CALIBRATED BEFORE IT IS USED.  Choosing `rand_mag`'s scale means
predicting a deployment magnitude from an offline measurement.  That predictor is fitted
on nothing and verified on cases that can refute it: the six restart-0 arms whose
realised panel magnitudes are already on disk.  The script reports, for each of them,
the on-basin prediction against the realised `mean_abs_delta_panel`, and the chosen
scale is only meaningful to the extent that table is tight.  A fix verified on a case
that cannot trigger it is not verified (CLAUDE.md, 2026-09-12).

WHAT IT DOES NOT DO.  It runs no environment steps, deploys nothing, and reads no
outcome of the panel it is preparing.  The basin latents come from the D1 collection
tapes, which are training-side data; seeds 9100-9387 are never opened here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from build_v252_rand_actor import build_rand_actor  # noqa: E402

OUT = REPO / "results" / "v253_replication"
MODEL_PAIR = REPO / "results/v241_model_pair/evolve1_2026-09-12T235711Z/model_pair.pt"
INTERFACE = REPO / "results/v251_interface"
DEPLOY_ROOT = REPO / "results/v206_belief_residual"
ROUND_TAG = "v251_2026-09-13T000155Z"

TRAINED_ARMS = {
    "M1": "chain3_lr2_m1_2026-09-12T235743Z",
    "M0": "chain3_lr2_m0_2026-09-12T235751Z",
    "M0_cont": "chain3_lr2_m0cont_2026-09-12T235800Z",
    "Q": "chain3_lr2_q_2026-09-12T235808Z",
    "M1_shuf": "chain3_lr2_m1shuf_2026-09-12T235817Z",
}
RAND_RESTART0 = REPO / "results/v252_rand_actor/chain3_lr2_rand_2026-09-12T235823Z/actor_0.pt"

# The D1 collection tapes: deployment rollouts under an untrained sustained residual.
D1_TAPES = [
    "chain3_lr2_d1_i0_2026-09-12T231912Z",
    "chain3_lr2_d1_i1_2026-09-12T232801Z",
    "chain3_lr2_d1_i2_2026-09-12T233655Z",
    "chain3_lr2_d1_i3_2026-09-12T234542Z",
]

BASIN_SEED = 253
N_BASIN = 600


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_basin_latents(start_chunk: int, n: int) -> tuple[torch.Tensor, dict]:
    """Deterministically sample real latents from the decision region of D1.

    The residual is gated to chunks >= ``start_chunk``, so magnitudes measured on
    earlier boundaries describe a region no arm ever acts in.
    """
    zs, provenance = [], []
    for tape_dir in D1_TAPES:
        p = REPO / "results/v250_chain3_round" / tape_dir / "tape.pt"
        tape = torch.load(p, map_location="cpu", weights_only=False)
        keep = tape["t"] >= start_chunk
        zs.append(tape["z"][keep].float())
        provenance.append({
            "tape": str(p.relative_to(REPO)),
            "rows_total": int(tape["z"].shape[0]),
            "rows_in_region": int(keep.sum()),
        })
        del tape
    pool = torch.cat(zs, 0)
    g = torch.Generator().manual_seed(BASIN_SEED)
    idx = torch.randperm(pool.shape[0], generator=g)[:n]
    return pool[idx].contiguous(), {
        "seed": BASIN_SEED, "requested": n, "pool_rows": int(pool.shape[0]),
        "sampled": int(idx.shape[0]), "start_chunk": start_chunk,
        "tapes": provenance,
    }


def mean_abs_delta(actor, z: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> float:
    """The executor's own input transform: zn = (o - mu)/sd, then actor(zn)."""
    with torch.no_grad():
        zn = (z - mu) / sd
        d = actor(zn)
    return float(d.abs().mean())


def load_actor(ckpt: Path):
    sys.path.insert(0, str(REPO / "scripts"))
    from train_v157_residual_actor import ResidualActor  # noqa: E402
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    act = ResidualActor(int(ck["zdim"]), int(ck["c"]), int(ck["adim"]),
                        scale=float(ck["scale"]))
    act.load_state_dict(ck["state_dict"])
    act.eval()
    for p in act.parameters():
        p.requires_grad_(False)
    return act, ck


def realised_panel_magnitudes() -> dict:
    out = {}
    for d in sorted(DEPLOY_ROOT.glob(f"chain3_lr2_{ROUND_TAG}_*")):
        s = json.loads((d / "summary.json").read_text())
        arm = d.name.split(f"{ROUND_TAG}_", 1)[1].rsplit("_2026", 1)[0]
        out[arm] = {
            "mean_abs_delta_panel": s.get("mean_abs_delta_panel"),
            "actor": s.get("actor"), "zero_residual": s.get("zero_residual"),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--restart", type=int, default=1,
                    help="actor restart index to prepare (0 was deployed on 09-13)")
    ap.add_argument("--residual-start-chunk", type=int, default=26,
                    help="the deployed schedule gate; defines the decision region")
    ap.add_argument("--n-basin", type=int, default=N_BASIN)
    ap.add_argument("--rand-init", type=int, default=8,
                    help="init for the replication's untrained arms; 7 was restart 0")
    ap.add_argument("--scale-lo", type=float, default=0.20)
    ap.add_argument("--scale-hi", type=float, default=0.90)
    ap.add_argument("--scale-step", type=float, default=0.01)
    ap.add_argument("--build", action="store_true",
                    help="build the two untrained arms at the chosen settings")
    a = ap.parse_args()

    if not MODEL_PAIR.exists():
        raise SystemExit(f"model pair missing: {MODEL_PAIR}")
    pair = torch.load(MODEL_PAIR, map_location="cpu", weights_only=False)
    mu, sd = pair["mu"].detach().cpu().float(), pair["sd"].detach().cpu().float()

    z, basin_prov = load_basin_latents(a.residual_start_chunk, a.n_basin)
    print(f"basin latents: {tuple(z.shape)} from {basin_prov['pool_rows']} rows "
          f"in the region t >= {a.residual_start_chunk}")

    # ---- 1. the predictor, verified on the restart-0 arms that can refute it ----
    realised = realised_panel_magnitudes()
    calibration = []
    for arm, rel in TRAINED_ARMS.items():
        ck_path = INTERFACE / rel / "actor_0.pt"
        act, _ = load_actor(ck_path)
        pred = mean_abs_delta(act, z, mu, sd)
        got = realised.get(arm, {}).get("mean_abs_delta_panel")
        calibration.append({
            "arm": arm, "restart": 0, "predicted_on_basin": pred,
            "realised_panel": got,
            "ratio": (pred / got) if got else None,
        })
    act, _ = load_actor(RAND_RESTART0)
    pred = mean_abs_delta(act, z, mu, sd)
    got = realised.get("rand", {}).get("mean_abs_delta_panel")
    calibration.append({"arm": "rand", "restart": 0, "predicted_on_basin": pred,
                        "realised_panel": got,
                        "ratio": (pred / got) if got else None})

    ratios = [c["ratio"] for c in calibration if c["ratio"]]
    cal_summary = {
        "n": len(ratios),
        "ratio_min": min(ratios), "ratio_max": max(ratios),
        "ratio_mean": sum(ratios) / len(ratios),
        "max_abs_pct_error": max(abs(r - 1.0) for r in ratios) * 100.0,
    }
    print("\npredictor calibration on the six deployed restart-0 arms")
    print(f"{'arm':<10}{'predicted':>12}{'realised':>12}{'ratio':>9}")
    for c in calibration:
        r = f"{c['ratio']:.3f}" if c["ratio"] else "n/a"
        g = f"{c['realised_panel']:.4f}" if c["realised_panel"] else "n/a"
        print(f"{c['arm']:<10}{c['predicted_on_basin']:>12.4f}{g:>12}{r:>9}")
    print(f"ratio range {cal_summary['ratio_min']:.3f}-{cal_summary['ratio_max']:.3f}, "
          f"worst error {cal_summary['max_abs_pct_error']:.1f}%")

    # ---- 2. the restart being prepared ----
    restart_arms = []
    for arm, rel in TRAINED_ARMS.items():
        ck_path = INTERFACE / rel / f"actor_{a.restart}.pt"
        if not ck_path.exists():
            raise SystemExit(f"missing {ck_path}")
        act, ck = load_actor(ck_path)
        restart_arms.append({
            "arm": arm, "checkpoint": str(ck_path.relative_to(REPO)),
            "sha256": sha256(ck_path), "scale": float(ck["scale"]),
            "predicted_on_basin": mean_abs_delta(act, z, mu, sd),
        })
    target = sum(r["predicted_on_basin"] for r in restart_arms) / len(restart_arms)
    spread = (min(r["predicted_on_basin"] for r in restart_arms),
              max(r["predicted_on_basin"] for r in restart_arms))
    print(f"\nrestart {a.restart} trained arms: predicted |delta| "
          f"{spread[0]:.4f}-{spread[1]:.4f}, mean {target:.4f}")
    for r in restart_arms:
        print(f"  {r['arm']:<10}{r['predicted_on_basin']:.4f}")

    # ---- 3. the scale that matches the untrained arm to that magnitude ----
    zdim, c, adim = (int(x) for x in pair["dims"])
    grid, s = [], a.scale_lo
    while s <= a.scale_hi + 1e-9:
        act = build_rand_actor(zdim, c, adim, round(s, 4), a.rand_init)
        grid.append({"scale": round(s, 4),
                     "predicted_on_basin": mean_abs_delta(act, z, mu, sd)})
        s += a.scale_step
    # The calibration above refutes the naive reading: on-basin magnitude
    # overestimates the realised panel magnitude by a factor that differs between
    # the trained arms (1.63-1.71, tight) and the untrained one (1.50). Matching
    # rand to the trained arms in PREDICTED units would therefore leave it
    # systematically heavy in the units the round actually checks. Each class is
    # converted with the ratio measured on its own deployed restart-0 arms.
    trained_ratios = [c["ratio"] for c in calibration
                      if c["arm"] in TRAINED_ARMS and c["ratio"]]
    trained_ratio = sum(trained_ratios) / len(trained_ratios)
    rand_ratio = next(c["ratio"] for c in calibration if c["arm"] == "rand")
    target_realised = target / trained_ratio
    for g in grid:
        g["realised_estimate"] = g["predicted_on_basin"] / rand_ratio
    best = min(grid, key=lambda g: abs(g["realised_estimate"] - target_realised))
    naive = min(grid, key=lambda g: abs(g["predicted_on_basin"] - target))
    at_bound = min(grid, key=lambda g: abs(g["scale"] - 0.80))
    band = 0.25
    in_band = abs(best["realised_estimate"] - target_realised) / target_realised <= band
    bound_dev = at_bound["realised_estimate"] / target_realised - 1
    print(f"\nconversion: trained arms {trained_ratio:.3f}x (n={len(trained_ratios)}, "
          f"spread {min(trained_ratios):.3f}-{max(trained_ratios):.3f}), "
          f"untrained {rand_ratio:.3f}x (n=1)")
    print(f"target realised |delta| {target_realised:.4f}")
    print(f"rand_mag:   scale {best['scale']} -> realised ~{best['realised_estimate']:.4f} "
          f"({100 * (best['realised_estimate'] / target_realised - 1):+.1f}%, "
          f"{'inside' if in_band else 'OUTSIDE'} the registered {int(band * 100)}% band)")
    print(f"rand_bound: scale 0.80 -> realised ~{at_bound['realised_estimate']:.4f} "
          f"({100 * bound_dev:+.1f}%) - restart 0 measured +38.3% at this bound, "
          f"which is the check on this conversion")
    print(f"(uncalibrated match would have picked scale {naive['scale']}, "
          f"realised ~{naive['realised_estimate']:.4f}, "
          f"{100 * (naive['realised_estimate'] / target_realised - 1):+.1f}%)")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / f"chain3_lr2_restart{a.restart}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    record = {
        "utc": stamp, "task": "chain3_lr2", "restart": a.restart,
        "purpose": "cross-initialisation replication of the 2026-09-13 round",
        "env_steps": 0, "panel_opened": False,
        "model_pair": str(MODEL_PAIR.relative_to(REPO)),
        "model_pair_sha256": sha256(MODEL_PAIR),
        "basin_sample": basin_prov,
        "predictor_calibration": calibration,
        "predictor_calibration_summary": cal_summary,
        "restart_arms": restart_arms,
        "target_mean_abs_delta_on_basin": target,
        "target_spread_on_basin": list(spread),
        "conversion_trained": trained_ratio,
        "conversion_trained_spread": [min(trained_ratios), max(trained_ratios)],
        "conversion_untrained": rand_ratio,
        "conversion_untrained_n": 1,
        "target_mean_abs_delta_realised": target_realised,
        "rand_init": a.rand_init,
        "rand_mag_scale": best["scale"],
        "rand_mag_predicted_on_basin": best["predicted_on_basin"],
        "rand_mag_realised_estimate": best["realised_estimate"],
        "rand_mag_within_25pct": in_band,
        "rand_mag_uncalibrated_scale": naive["scale"],
        "rand_bound_scale": 0.80,
        "rand_bound_predicted_on_basin": at_bound["predicted_on_basin"],
        "rand_bound_realised_estimate": at_bound["realised_estimate"],
        "rand_bound_deviation_from_target": bound_dev,
        "scale_grid": grid,
        "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                              capture_output=True, text=True).stdout.strip(),
    }
    (out / "prepare.json").write_text(json.dumps(record, indent=1))
    print(f"\nwrote {out.relative_to(REPO)}/prepare.json")

    if a.build:
        for tag, scale in (("randbound", 0.80), ("randmag", best["scale"])):
            cmd = [sys.executable, str(REPO / "scripts/build_v252_rand_actor.py"),
                   "--model-pair", str(MODEL_PAIR), "--scale", str(scale),
                   "--init", str(a.rand_init), "--out-tag", f"{tag}_r{a.restart}"]
            print("+", " ".join(cmd))
            r = subprocess.run(cmd, cwd=REPO)
            if r.returncode != 0:
                raise SystemExit(f"builder failed for {tag}")
    else:
        print("measurement only; rerun with --build to construct the untrained arms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
