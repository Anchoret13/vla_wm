#!/usr/bin/env python
"""Automated go/no-go between interface training and the 6-arm evaluation.

    python scripts/gate_v251_prelaunch.py --model-pair <pair> --interface-dirs <dirs...>

WHY A GATE AT ALL. The evaluation is 6 arms x 288 episodes ~ 10.6 h of simulator. Three
failures upstream would make every one of those episodes uninterpretable, and all three
are visible for free in artifacts that already exist:

  1. THE ACTOR COLLAPSED TO A NO-OP. `ResidualActor` zero-initialises its output layer,
     so an interface that learned nothing is EXACTLY the `base` arm. Four arms would
     then be four copies of the control and every contrast would be a null by
     construction, reported as a null about world models.
  2. THE Phi' HEAD HAS NO SIGNAL WHERE IT IS USED. Measured on 2026-09-12, Phi' is 97.2%
     predictable from the timestep alone and a globally-fitted head ranks decision-region
     states worse than a region-fitted one in 5/5 splits. If the head's DECISION-REGION
     held-out correlation is near zero, the advantage every arm is trained on is noise,
     and M1-vs-M0 measures nothing.
  3. THE ARMS ARE NOT MATCHED. The round's claim is that M1 and M0 differ only in the
     learned transition. If they carry different Phi' heads, different normalisers, or
     different training budgets, the contrast is uninterpretable whatever it shows.

WHAT THIS GATE IS NOT. It does not predict or require a positive result, and it reads no
outcome. Passing it means the comparison is well posed, not that the world model helps.
A round that passes this gate and then shows nothing is a completed experiment.

Exit 0 = GO, exit 1 = NO-GO with the reasons printed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

#: A residual whose mean magnitude is below this fraction of its own bound has not
#: learned a policy; at 1% of the bound it is indistinguishable from the zero residual
#: the class initialises to.
MIN_DELTA_FRACTION_OF_BOUND = 0.01
#: The decision-region held-out correlation the shared head must clear. Deliberately
#: low: the 24-episode measurement put a region-fitted head anywhere in [-0.21, +0.89]
#: across five splits, so a high bar here would be fitted to noise. This only excludes
#: a head that carries no signal at all where it is used.
MIN_PHI_REGION_CORR = 0.10


def fail(msgs: list[str], why: str) -> None:
    msgs.append(why)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-pair", type=Path, required=True)
    ap.add_argument("--interface-dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--rand-dir", type=Path, default=None)
    ap.add_argument("--min-phi-corr", type=float, default=MIN_PHI_REGION_CORR)
    a = ap.parse_args()

    problems: list[str] = []
    notes: list[str] = []

    pair = torch.load(a.model_pair, weights_only=False)
    for arm in ("stale", "base_continue", "updated", "updated_shuffled"):
        if arm not in pair.get("models", {}):
            fail(problems, f"model pair is missing the {arm!r} arm")
    notes.append(f"model pair dims={tuple(pair['dims'])} task={pair['task']}")

    digests, norms, budgets = {}, {}, {}
    for d in a.interface_dirs:
        s = json.loads((d / "summary.json").read_text())
        tag = d.name
        ph = s.get("phi_head", {})
        digests[tag] = ph.get("digest")
        corr = ph.get("holdout_corr")
        region = ph.get("region_restricted")
        notes.append(f"{tag}: phi region_restricted={region} "
                     f"holdout_corr(region)={corr} "
                     f"holdout_corr(all)={ph.get('holdout_corr_all_boundaries')}")
        if region is not True:
            fail(problems, f"{tag}: the Phi' head was NOT fitted on the decision "
                           f"region; a globally fitted head ranked decision-region "
                           f"states worse in 5/5 measured splits")
        if corr is None or float(corr) < a.min_phi_corr:
            fail(problems, f"{tag}: Phi' decision-region holdout corr {corr} < "
                           f"{a.min_phi_corr}; the advantage would be noise")
        # the actor must not be a no-op
        rows = s.get("actors") or []
        if not rows:
            fail(problems, f"{tag}: summary.json has no 'actors' records")
        fracs = []
        for r in rows:
            frac = r.get("fraction_of_bound")
            if frac is None and r.get("mean_abs_delta") is not None and s.get("scale"):
                frac = float(r["mean_abs_delta"]) / float(s["scale"])
            if frac is None:
                fail(problems, f"{tag} restart {r.get('restart')}: no delta magnitude")
                continue
            fracs.append(float(frac))
            if float(frac) < MIN_DELTA_FRACTION_OF_BOUND:
                fail(problems, f"{tag} restart {r.get('restart')}: mean|delta| is "
                               f"{float(frac):.4f} of its bound - the interface is a "
                               f"no-op and this arm IS the base arm")
        notes.append(f"{tag}: |delta|/bound per restart = "
                     f"{[round(x, 4) for x in fracs]}")
        norms[tag] = s.get("phi_head", {}).get("normaliser_digest")
        budgets[tag] = (s.get("actor_epochs"), s.get("batch"), s.get("restarts"),
                        s.get("horizon"), s.get("scale"))

    uniq = {v for v in digests.values() if v is not None}
    if len(uniq) > 1:
        fail(problems, f"the arms do NOT share one Phi' head: {digests}")
    un = {v for v in norms.values() if v is not None}
    if len(un) > 1:
        fail(problems, f"the arms do NOT share one normaliser: {norms}")
    ub = set(budgets.values())
    if len(ub) > 1:
        fail(problems, f"training budgets are not matched across arms: {budgets}")

    if a.rand_dir is not None:
        rs = json.loads((a.rand_dir / "summary.json").read_text())
        if rs.get("trained") is not False:
            fail(problems, "the rand arm is marked trained; it must be untrained")
        if float(rs.get("probe_abs_mean_delta", 0.0)) <= 0.0:
            fail(problems, "the rand arm is identically zero and duplicates base")
        notes.append(f"rand: |delta| probe {rs.get('probe_abs_mean_delta')}")

    print("\n".join(f"  {n}" for n in notes))
    if problems:
        print("\nNO-GO:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nGO: the comparison is well posed. This says nothing about whether the "
          "world model helps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
