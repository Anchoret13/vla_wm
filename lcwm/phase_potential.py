"""A1 — phase-aware state potential Φ (registered 2026-07-27 in
plan_and_progress/2026-07-25.md, review-to-action item A1).

Lexicographic pick-place potential for the ACTIVE goal atom (first false
atom in canonical goal order). Raw object-to-receptacle distance is used
ONLY inside the transport phase; approach/grasp/lift use EEF-object
distance, gripper closure, and object height, so correct manipulation is
never labeled negative merely because basket distance temporarily grows
(the defect that produced 8/104 flow-positive branches on 2026-07-24).

Φ = C + φ(active):
  C = number of goal atoms currently true that were ever true (no-damage:
      a previously-true atom turning false raises a violation flag that
      travels with the record — it is never silently re-scored);
  φ ∈ [0,1) per the registered phase table (approach [0,.2), grasp [.2,.4),
      lift [.4,.6), transport [.6,.9), placement [.9,1)).

All inputs are the stored per-decision labels (q, obj_pos, bits) — CPU-only,
no environment.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

# Registered constants (2026-07-25.md A1 registration).
NEAR_DISTANCE = 0.05          # m, matches the contact-collection trigger
LIFT_THRESHOLD = 0.03         # m above the object's episode-start height
LIFT_REFERENCE = 0.10         # m, full-credit lift height
BASKET_RADIUS_XY = 0.08       # m, placement-zone entry
PLACEMENT_DESCENT_REF = 0.10  # m, full-credit descent inside the zone
# A1 Amendment 1 (2026-07-27): grasp is detected by ATTACHMENT — the object
# has moved with the hand — not by gripper closure. Calibration showed the
# summed gripper qpos at grasp is object-dependent (soup −0.012, butter
# −0.002, cream-cheese box +0.003), so no closure threshold generalizes.
ATTACH_DISPLACEMENT = 0.02    # m the object must move from its start pose

PHASES = ("approach", "grasp", "lift", "transport", "placement", "done")


@dataclass
class PhaseRecord:
    phi: float          # C + within-atom potential
    phase: str
    active_atom: int    # -1 when all atoms complete
    violation: bool     # a previously-completed atom regressed


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


def resolve_entity_row(object_names: list[str], entity: str) -> int:
    """BDDL atoms may name a region of an object (e.g.
    "basket_1_contain_region"); positions are stored per object."""
    if entity in object_names:
        return object_names.index(entity)
    matches = [name for name in object_names if entity.startswith(name)]
    if not matches:
        raise KeyError(f"no object owns atom entity {entity!r}")
    return object_names.index(max(matches, key=len))


def episode_phase_potentials(
    q: Tensor,
    obj_pos: Tensor,
    bits: Tensor,
    object_names: list[str],
    goal_atoms: list[list[str]],
    label_mask: Tensor | None = None,
) -> list[PhaseRecord | None]:
    """Per-record Φ over one episode. Records outside label_mask -> None.

    Per-atom memos (was_lifted, transport reference distance) are carried
    across records, which is why this is an episode-level function.
    """
    records: list[PhaseRecord | None] = []
    T = q.shape[0]

    def resolve(entity: str) -> int:
        return resolve_entity_row(object_names, entity)

    object_rows = [resolve(atom[1]) for atom in goal_atoms]
    receptacle_rows = [resolve(atom[2]) for atom in goal_atoms]

    first = 0
    if label_mask is not None:
        valid = torch.nonzero(label_mask).flatten()
        if len(valid) == 0:
            return [None] * T
        first = int(valid[0])
    start_z = {row: float(obj_pos[first, row, 2]) for row in object_rows}
    start_pos = {row: obj_pos[first, row].clone() for row in object_rows}
    start_eef_d = {
        row: float(
            torch.linalg.vector_norm(q[first, 0:3] - obj_pos[first, row])
        )
        for row in object_rows
    }
    ever_true = [False] * len(goal_atoms)
    was_lifted = {row: False for row in object_rows}
    dxy_at_lift = {row: None for row in object_rows}

    for t in range(T):
        if label_mask is not None and not bool(label_mask[t]):
            records.append(None)
            continue
        current_bits = [bool(b) for b in bits[t]]
        violation = any(
            ever and not now for ever, now in zip(ever_true, current_bits)
        )
        for i, now in enumerate(current_bits):
            ever_true[i] = ever_true[i] or now
        completed = sum(current_bits)
        active = next(
            (i for i, now in enumerate(current_bits) if not now), -1
        )
        if active < 0:
            records.append(
                PhaseRecord(float(len(goal_atoms)), "done", -1, violation)
            )
            continue

        row = object_rows[active]
        basket_row = receptacle_rows[active]
        eef = q[t, 0:3]
        obj = obj_pos[t, row]
        d_eef = float(torch.linalg.vector_norm(eef - obj))
        dxy = float(
            torch.linalg.vector_norm(obj[:2] - obj_pos[t, basket_row, :2])
        )
        delta_z = float(obj[2]) - start_z[row]
        displacement = float(
            torch.linalg.vector_norm(obj - start_pos[row])
        )

        attached = (
            d_eef <= NEAR_DISTANCE
            and displacement >= ATTACH_DISPLACEMENT
        )
        lifted_now = attached and delta_z >= LIFT_THRESHOLD
        if lifted_now and not was_lifted[row]:
            was_lifted[row] = True
            dxy_at_lift[row] = max(dxy, BASKET_RADIUS_XY + 1e-3)
        in_zone = dxy <= BASKET_RADIUS_XY and was_lifted[row]

        if in_zone:
            basket_z = float(obj_pos[t, basket_row, 2])
            descent = _clip01(
                1.0
                - (float(obj[2]) - basket_z) / PLACEMENT_DESCENT_REF
            )
            phi, phase = 0.9 + 0.1 * descent, "placement"
        elif was_lifted[row] and attached:
            reference = dxy_at_lift[row] or dxy
            phi, phase = (
                0.6 + 0.3 * (1.0 - _clip01(dxy / reference)),
                "transport",
            )
        elif attached and delta_z > 0:
            phi, phase = (
                0.4 + 0.2 * _clip01(delta_z / LIFT_REFERENCE),
                "lift",
            )
        elif d_eef <= NEAR_DISTANCE:
            phi, phase = (
                0.2 + 0.2 * _clip01(displacement / ATTACH_DISPLACEMENT),
                "grasp",
            )
        else:
            reference = max(start_eef_d[row], NEAR_DISTANCE + 1e-3)
            phi, phase = (
                0.2 * (1.0 - _clip01(d_eef / reference)),
                "approach",
            )
        records.append(
            PhaseRecord(completed + phi, phase, active, violation)
        )
    return records


def delta_phi_targets(
    records: list[PhaseRecord | None],
) -> tuple[Tensor, Tensor, Tensor]:
    """Window targets: ΔΦ_i = Φ_{i+1} − Φ_i.

    Returns (delta_phi [T-1], valid [T-1] bool, violation [T-1] bool).
    A window is valid when both endpoint records are labeled.
    """
    T = len(records)
    delta = torch.zeros(max(T - 1, 0))
    valid = torch.zeros(max(T - 1, 0), dtype=torch.bool)
    violation = torch.zeros(max(T - 1, 0), dtype=torch.bool)
    for i in range(T - 1):
        a, b = records[i], records[i + 1]
        if a is None or b is None:
            continue
        delta[i] = b.phi - a.phi
        valid[i] = True
        violation[i] = a.violation or b.violation
    return delta, valid, violation
