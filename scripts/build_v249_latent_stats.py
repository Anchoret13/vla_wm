#!/usr/bin/env python
"""Producer for the chain3 rich-latent deployment normaliser.

    python scripts/build_v249_latent_stats.py \
        --tape results/v121_deploy_latents/chain3_lr2_2026-08-30T083910Z/tape.pt

WHY THIS FILE EXISTS. `results/v249_chain3_residual_stats/d0_latent_stats.pt` is a hard
required input of both `scripts/probe_v249_chain3_residual_scale.py` and
`scripts/collect_v250_chain3_round.py` - every residual they apply is conditioned on
`(z - mu) / sd`. It was first produced by an ad-hoc command, which left the round's
normaliser unreproducible. An adversarial audit on 2026-09-12 flagged exactly that: a
required input that no script in the repository writes. This script is that writer.

WHAT IT COMPUTES, and the one property that matters. `mu` and `sd` are the per-dimension
mean and standard deviation of the CURRENT-state rich latents `z` of a base-role
deployment tape - never of `z_next`, and never of a residual-perturbed tape. A collector
actor conditioned on statistics fitted to its own perturbed distribution would have its
input distribution entangled with its own behaviour; fitting them on base deployment data
once and freezing them keeps every arm of the round on one shared, declared coordinate
system.

It reads `z`, `u`, `c`, `latent_dim` and `task` from the tape and writes nothing derived
from any outcome field, so it is safe to run on a tape that has not been unsealed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

OUT = REPO / "results" / "v249_chain3_residual_scale"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tape", type=Path, required=True,
                    help="a BASE-role deployment tape; residual-perturbed tapes are "
                         "rejected so the normaliser cannot absorb a collector's bias")
    ap.add_argument("--task", default="chain3_lr2")
    ap.add_argument("--expect-latent-dim", type=int, default=8217)
    ap.add_argument("--out", type=Path, default=OUT / "d0_latent_stats.pt")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing normaliser. Off by default: silently "
                         "replacing it would change the coordinates of every actor "
                         "already trained against it.")
    a = ap.parse_args()

    if a.out.exists() and not a.force:
        raise SystemExit(f"{a.out} exists; pass --force to replace it (this changes "
                         f"the coordinate system of every actor trained against it)")

    d = torch.load(a.tape, weights_only=False)
    if d.get("task") != a.task:
        raise SystemExit(f"tape task {d.get('task')!r} != {a.task!r}")
    if int(d["latent_dim"]) != a.expect_latent_dim:
        raise SystemExit(f"latent_dim {d['latent_dim']} != {a.expect_latent_dim}")
    sig = d.get("sigma")
    z = d["z"].float()
    u = d["u"].float()
    if int(z.shape[-1]) != int(d["latent_dim"]):
        raise SystemExit(f"z width {z.shape[-1]} != latent_dim {d['latent_dim']}")

    mu, sd = z.mean(0), z.std(0) + 1e-6
    uf = u.reshape(-1, u.shape[-1])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mu": mu, "sd": sd,
        "c": int(d["c"]), "adim": int(u.shape[-1]),
        "latent_dim": int(d["latent_dim"]), "task": d["task"],
        "u_std": uf.std(0), "u_mean": uf.mean(0),
        "source": str(a.tape), "n_rows": int(z.shape[0]),
        "sigmas_present": sorted({float(x) for x in sig.tolist()}) if sig is not None else None,
    }
    torch.save(payload, a.out)
    manifest = {
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "source": str(a.tape),
        "source_sha256": hashlib.sha256(a.tape.read_bytes()).hexdigest(),
        "out": str(a.out),
        "out_sha256": hashlib.sha256(a.out.read_bytes()).hexdigest(),
        "task": d["task"], "c": int(d["c"]), "adim": int(u.shape[-1]),
        "latent_dim": int(d["latent_dim"]), "n_rows": int(z.shape[0]),
        "sd_min": float(sd.min()), "sd_max": float(sd.max()),
        "u_abs_mean": float(u.abs().mean()),
        "sigmas_present": payload["sigmas_present"],
        "source_sha256_of_producer": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                              capture_output=True, text=True).stdout.strip(),
    }
    (a.out.parent / "d0_latent_stats_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
