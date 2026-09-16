#!/usr/bin/env python
"""Build the round's UNTRAINED random-residual control arm.

    python scripts/build_v252_rand_actor.py --model-pair <model_pair.pt> \
        --scale 0.80 --init 7 --out-tag rand

WHY THIS ARM EXISTS, and why it may be the load-bearing control of the round.

The 2026-09-12 basin calibration deployed *random, untrained* sustained residuals on
chain3_lr2@750 and measured, at scale 0.80 on 24 seeds: 7/24 episodes inside the 5 cm
grasp radius, 4/24 picking the third object up, 1/24 placing it, 1/24 terminal success
— against 0/24, 0/24, 0/24 and 0 for frozen pi0.5 on the identical seeds.

So at this trust region the ACTION MAGNITUDE alone already moves every endpoint the
round reports. Without a matched untrained arm, an interface that merely equals a
random residual would show up as a large, significant improvement over `base`, and the
trust region's own effect would be attributed to the world model. That is the shape of
the 2026-08-27 failure recorded in CLAUDE.md: a real, replicated gain that was
off-mission because nothing established the mechanism it was credited to.

WHAT MAKES IT A MATCHED CONTROL, rather than merely a random baseline.

* Same architecture: `ResidualActor(zdim, c, adim, scale)`, the class every trained arm
  uses, bounded identically by `scale * tanh(.)`.
* Same coordinates: `mu`/`sd` are taken from the SAME model pair the trained arms use,
  not from the collector's `d0_latent_stats.pt`. A random actor conditioned on
  different normalisation would be a different function of the observation, and the
  comparison would confound the interface with its input space — the v235 error.
* Same init gain: `GAIN = 10.0`, matching `collect_v250_chain3_round.make_collector_actor`
  and `probe_v249`, so the measured `|delta| ~ 0.536 * scale` relationship carries over.
* Same deployment path: it is written in the `run_v206_belief_residual_deploy.py`
  checkpoint schema and deployed by that executor on the same panel, with the same
  `--residual-start-chunk` schedule, as every other arm.

The `rawwm` entry is required by the executor to construct its belief module, and the
belief is stepped every chunk, but with `condition="raw"` the actor reads only the
normalised latent (`run_v206...:364`) and the belief output is discarded. The transition
stored here is therefore never read for a decision; it is recorded in the provenance so
that an auditor does not mistake it for a model this arm consults.

Nothing is trained and no outcome is read.
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
sys.path.insert(0, str(REPO / "scripts"))

import torch  # noqa: E402

from train_v157_residual_actor import ResidualActor  # noqa: E402

OUT = REPO / "results" / "v252_rand_actor"
GAIN = 10.0


def build_rand_actor(zdim: int, c: int, adim: int, scale: float,
                     init: int) -> ResidualActor:
    """A bounded state-conditioned residual whose output layer is randomised.

    ``ResidualActor`` zero-initialises its output layer, so an untouched instance is
    identically a no-op and would silently reproduce the `base` arm.  Randomising that
    layer at a calibrated gain is what makes this a control rather than a duplicate.
    """
    torch.manual_seed(25200 + 17 * init + int(round(scale * 1000)))
    act = ResidualActor(zdim, c, adim, scale=scale)
    last = act.net[-1]
    torch.nn.init.normal_(last.weight, std=GAIN / (last.in_features ** 0.5))
    torch.nn.init.zeros_(last.bias)
    act.eval()
    for p in act.parameters():
        p.requires_grad_(False)
    return act


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-pair", type=Path, required=True,
                    help="supplies dims, task and the SHARED mu/sd coordinate system")
    ap.add_argument("--wm-arm", default="stale",
                    help="which transition to embed for the executor to construct; "
                         "never read for a decision under condition='raw'")
    ap.add_argument("--scale", type=float, required=True)
    ap.add_argument("--init", type=int, default=7)
    ap.add_argument("--out-tag", default="rand")
    a = ap.parse_args()

    pair = torch.load(a.model_pair, weights_only=False)
    zdim, c, adim = (int(x) for x in pair["dims"])
    if a.wm_arm not in pair["models"]:
        raise SystemExit(f"--wm-arm {a.wm_arm!r} not in {sorted(pair['models'])}")
    actor = build_rand_actor(zdim, c, adim, a.scale, a.init)

    mu, sd = pair["mu"].detach().cpu(), pair["sd"].detach().cpu()
    with torch.no_grad():
        probe = torch.zeros(4, zdim)
        probe[1] = 1.0
        probe[2] = -1.0
        probe[3] = torch.linspace(-1, 1, zdim)
        delta = actor(probe)
    if float(delta.abs().max()) > a.scale + 1e-6:
        raise SystemExit("residual exceeded its own bound")
    if float(delta.abs().mean()) == 0.0:
        raise SystemExit("the randomised actor is identically zero; it would "
                         "silently duplicate the base arm")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / f"{pair['task']}_{a.out_tag}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    ck = {
        "state_dict": actor.state_dict(), "zdim": zdim, "condition": "raw",
        "c": c, "adim": adim, "scale": float(a.scale),
        "rawwm": pair["models"][a.wm_arm],
        "dims": (zdim, c, adim), "mu": mu, "sd": sd,
        "bmu": torch.zeros(256), "bsd": torch.ones(256),
        "task": pair["task"], "arm": f"rand_s{a.scale}_i{a.init}",
        "trained": False, "init_gain": GAIN,
        "provenance": "untrained random residual; matched control for the action "
                      "magnitude. The embedded transition is never read for a "
                      "decision under condition='raw'.",
    }
    torch.save(ck, out / "actor_0.pt")
    summary = {
        "utc": stamp, "task": pair["task"], "scale": a.scale, "init": a.init,
        "init_gain": GAIN, "dims": [zdim, c, adim], "trained": False,
        "wm_arm_embedded": a.wm_arm,
        "model_pair": str(a.model_pair),
        "model_pair_sha256": hashlib.sha256(a.model_pair.read_bytes()).hexdigest(),
        "coordinates": "mu/sd taken from the model pair, identical to every trained arm",
        "probe_abs_mean_delta": float(delta.abs().mean()),
        "probe_abs_max_delta": float(delta.abs().max()),
        "actor_sha256": hashlib.sha256((out / "actor_0.pt").read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "env_steps": 0, "deployed_outcomes_read": False,
        "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                              capture_output=True, text=True).stdout.strip(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
