"""Progress annotation for the 2026-09-12 chain3 round.

WHY THIS EXISTS RATHER THAN `lcwm/phase_potential.py`.

The registered A1 potential is correct for what it was built for and is left
untouched.  It cannot supply a training signal on chain3 for a structural reason
measured on 2026-09-12: its approach term is

    phi = 0.2 * (1 - clip(d_eef / start_eef_d))

normalised by the EEF-object distance at the episode's *own first labelled
record*.  chain3's frozen policy places two cans by step 260 and then retreats to
a hover ~29 cm above the basket for a median of 490 further steps.  Its distance
to the remaining object therefore rises from 0.347 m at reset to a basin mean of
0.553 m, `clip(.)` saturates at 1, and Phi is flat at exactly 2.0 for every
boundary in the stall - the precise region the round needs to resolve.

The fix is one constant: normalise the approach term by a FIXED task scale instead
of the episode's own start distance.  Everything else - the lexicographic
C + phi(active atom) structure, the attachment-based grasp test, the lift/transport/
placement thresholds - is the registered A1 definition and is reproduced unchanged.

WHAT THIS SIGNAL IS, stated so no later claim can overstate it.  It is a SCRIPTED
PRIVILEGED STAND-IN for a per-boundary human video annotation, computed from
simulator object poses.  Every component is something an annotator reads off the
agentview frame - how many objects are already in the basket, whether the gripper
is closing on the next one, whether it is holding it, whether it is over the
basket - so it is a faithful proxy for the fixed-budget human stage feedback the
approved plan authorises, and it is finer than a human would actually supply.  It
is NOT an observation-only deployable signal: it may not be used at deployment
time, and any result obtained with it is scoped to "under this supplied progress
annotation".  Its cost is recorded as (episodes x boundaries) annotations.

It is identical across every arm of the round, so it cannot explain a difference
between arms - which is the only property the causal comparison requires of it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from lcwm.phase_potential import (
    ATTACH_DISPLACEMENT,
    BASKET_RADIUS_XY,
    LIFT_REFERENCE,
    LIFT_THRESHOLD,
    NEAR_DISTANCE,
    PLACEMENT_DESCENT_REF,
    _clip01,
    resolve_entity_row,
)

#: Fixed task scale for the approach term, in metres.  A constant of the task, not of
#: an episode, so two episodes at the same geometry receive the same label.
#:
#: It was first set to 0.60 m from the base policy's own range, and that was wrong in
#: the regime this round actually collects in.  Measured across all 192 calibration
#: episodes, the largest EEF-to-active-object distance is **1.186 m**, and at residual
#: scale 0.80 - the D1 setting - **23.6% of boundaries exceed 0.60 m** and would clip
#: to exactly zero.  That is the same saturation that makes the registered A1
#: potential flat across chain3's entire stall, reintroduced through a different
#: constant, and concentrated in precisely the episodes where the residual is doing
#: something.  Per-arm clipped fractions at 0.60 m were
#: base .027 / 0.10 .027 / 0.20 .064 / 0.40 .038 / 0.80 .236 / 0.80i1 .067 /
#: 0.80i2 .056 / 1.60 .095.
#:
#: 1.25 m covers the observed maximum with headroom, so the term stays strictly inside
#: (0, 0.2) everywhere and is monotone in distance throughout.  It costs dynamic range
#: - the useful 0.50 m -> 0.05 m span maps to 0.072 rather than 0.150 - which is the
#: right trade, because advantage-weighted regression standardises its advantage and
#: therefore needs monotonicity, which clipping destroys, far more than it needs scale.
APPROACH_REFERENCE = 1.25

#: Ordinal levels a human annotator could actually produce from video, for the
#: declared-budget framing and as a coarse cross-check on the continuous signal.
ORDINAL_LEVELS = ("far", "approaching", "at_object", "grasped", "lifted",
                  "over_basket", "placed")


@dataclass
class ProgressRecord:
    phi: float            # C + within-atom potential, in [0, n_atoms]
    phase: str
    active_atom: int      # -1 when every atom is complete
    violation: bool       # a previously-completed atom regressed
    d_active: float       # EEF -> active object distance, metres (raw, unclipped)
    ordinal: int          # index into ORDINAL_LEVELS, the human-labelable level


def _ordinal(phase: str, d_eef: float, completed: int, n_atoms: int) -> int:
    if completed >= n_atoms:
        return len(ORDINAL_LEVELS) - 1
    if phase == "placement":
        return 5
    if phase == "transport":
        return 4
    if phase == "lift":
        return 4
    if phase == "grasp":
        return 3 if d_eef <= NEAR_DISTANCE else 2
    return 1 if d_eef <= 0.25 else 0


def episode_progress(
    q: Tensor,
    obj_pos: Tensor,
    bits: Tensor,
    object_names: list[str],
    goal_atoms: list[list[str]],
) -> list[ProgressRecord]:
    """Per-boundary progress over one episode.

    ``q`` is the 25-d proprioception block whose first three entries are
    ``robot_state.eef.pos``; ``obj_pos`` is ``[T, n_obj, 3]``; ``bits`` is
    ``[T, n_atoms]`` goal-predicate truth.  CPU only, no environment.
    """
    T = int(q.shape[0])
    object_rows = [resolve_entity_row(object_names, atom[1]) for atom in goal_atoms]
    receptacle_rows = [resolve_entity_row(object_names, atom[2]) for atom in goal_atoms]
    n_atoms = len(goal_atoms)

    start_z = {row: float(obj_pos[0, row, 2]) for row in object_rows}
    start_pos = {row: obj_pos[0, row].clone() for row in object_rows}
    ever_true = [False] * n_atoms
    was_lifted = {row: False for row in object_rows}
    dxy_at_lift: dict[int, float | None] = {row: None for row in object_rows}

    out: list[ProgressRecord] = []
    for t in range(T):
        current = [bool(b) for b in bits[t]]
        violation = any(ever and not now for ever, now in zip(ever_true, current))
        for i, now in enumerate(current):
            ever_true[i] = ever_true[i] or now
        completed = sum(current)
        active = next((i for i, now in enumerate(current) if not now), -1)
        if active < 0:
            out.append(ProgressRecord(float(n_atoms), "done", -1, violation,
                                      0.0, len(ORDINAL_LEVELS) - 1))
            continue

        row, basket_row = object_rows[active], receptacle_rows[active]
        eef, obj = q[t, 0:3], obj_pos[t, row]
        d_eef = float(torch.linalg.vector_norm(eef - obj))
        dxy = float(torch.linalg.vector_norm(obj[:2] - obj_pos[t, basket_row, :2]))
        delta_z = float(obj[2]) - start_z[row]
        displacement = float(torch.linalg.vector_norm(obj - start_pos[row]))

        attached = d_eef <= NEAR_DISTANCE and displacement >= ATTACH_DISPLACEMENT
        if attached and delta_z >= LIFT_THRESHOLD and not was_lifted[row]:
            was_lifted[row] = True
            dxy_at_lift[row] = max(dxy, BASKET_RADIUS_XY + 1e-3)
        in_zone = dxy <= BASKET_RADIUS_XY and was_lifted[row]

        if in_zone:
            basket_z = float(obj_pos[t, basket_row, 2])
            descent = _clip01(1.0 - (float(obj[2]) - basket_z) / PLACEMENT_DESCENT_REF)
            phi, phase = 0.9 + 0.1 * descent, "placement"
        elif was_lifted[row] and attached:
            reference = dxy_at_lift[row] or dxy
            phi, phase = 0.6 + 0.3 * (1.0 - _clip01(dxy / reference)), "transport"
        elif attached and delta_z > 0:
            phi, phase = 0.4 + 0.2 * _clip01(delta_z / LIFT_REFERENCE), "lift"
        elif d_eef <= NEAR_DISTANCE:
            phi, phase = 0.2 + 0.2 * _clip01(displacement / ATTACH_DISPLACEMENT), "grasp"
        else:
            # THE ONE CHANGE vs A1: a fixed task scale, so the term does not
            # saturate when the arm parks further away than it started.
            phi, phase = 0.2 * (1.0 - _clip01(d_eef / APPROACH_REFERENCE)), "approach"

        out.append(ProgressRecord(completed + phi, phase, active, violation,
                                  d_eef, _ordinal(phase, d_eef, completed, n_atoms)))
    return out


def tape_progress(labels: dict) -> dict[str, Tensor]:
    """Φ', ordinal level and raw active-object distance for every tape row.

    ``labels`` is the ``labels.pt`` written by
    ``scripts/collect_v250_chain3_round.py``.  Rows come back in the same
    (episode, t) order the label artifact stores them.
    """
    ep = labels["episode"]
    phi = torch.zeros(len(ep))
    ordi = torch.zeros(len(ep), dtype=torch.long)
    dact = torch.zeros(len(ep))
    viol = torch.zeros(len(ep), dtype=torch.bool)
    for e in torch.unique(ep):
        m = torch.nonzero(ep == e).flatten()
        order = m[torch.argsort(labels["t"][m])]
        recs = episode_progress(labels["eef_proprio"][order], labels["obj_pos"][order],
                                labels["bits"][order], labels["object_names"],
                                labels["goal_atoms"])
        for i, r in zip(order.tolist(), recs):
            phi[i], ordi[i], dact[i], viol[i] = r.phi, r.ordinal, r.d_active, r.violation
    return {"phi": phi, "ordinal": ordi, "d_active": dact, "violation": viol,
            "episode": ep, "t": labels["t"]}
