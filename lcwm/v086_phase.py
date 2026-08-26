"""Forkable phase potential for Action 2M.4.

`lcwm.phase_potential.episode_phase_potentials` is an episode-level loop whose
per-atom memos - `ever_true`, `was_lifted`, `dxy_at_lift` - and per-episode
baselines - `start_z`, `start_pos`, `start_eef_d` - live in local variables.
Action 2M.4 requires those memos to be built on the SOURCE history through
`tau` and then FORKED with the environment snapshot for every candidate and
repeat: "Reinitializing approach, attachment, lift, or transport state at tau
is prohibited."

This module extracts exactly that state into an object.  The step body is a
transcription of the original loop body, and `scripts/test_v086_phase.py`
asserts the two produce identical `PhaseRecord`s on the same input.  Behaviour
is not changed here; only its ownership is.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

import torch
from torch import Tensor

from lcwm.phase_potential import (ATTACH_DISPLACEMENT, BASKET_RADIUS_XY,
                                  LIFT_REFERENCE, LIFT_THRESHOLD,
                                  NEAR_DISTANCE, PLACEMENT_DESCENT_REF,
                                  PhaseRecord, _clip01, resolve_entity_row)


@dataclass
class PhasePotential:
    """Stateful Φ whose memos can be forked per branch."""

    object_rows: list[int]
    receptacle_rows: list[int]
    n_atoms: int
    # baselines, fixed at the episode's first labeled record
    start_z: dict[int, float]
    start_pos: dict[int, Tensor]
    start_eef_d: dict[int, float]
    # evolving memos
    ever_true: list[bool]
    was_lifted: dict[int, bool]
    dxy_at_lift: dict[int, float | None]

    # ---- construction ---------------------------------------------------
    @classmethod
    def at_start(cls, q0: Tensor, obj_pos0: Tensor, object_names: Sequence[str],
                 goal_atoms: Sequence[Sequence[str]]) -> "PhasePotential":
        rows = [resolve_entity_row(list(object_names), a[1]) for a in goal_atoms]
        recs = [resolve_entity_row(list(object_names), a[2]) for a in goal_atoms]
        return cls(
            object_rows=rows, receptacle_rows=recs, n_atoms=len(goal_atoms),
            start_z={r: float(obj_pos0[r, 2]) for r in rows},
            start_pos={r: obj_pos0[r].clone() for r in rows},
            start_eef_d={r: float(torch.linalg.vector_norm(q0[0:3] - obj_pos0[r]))
                         for r in rows},
            ever_true=[False] * len(goal_atoms),
            was_lifted={r: False for r in rows},
            dxy_at_lift={r: None for r in rows})

    @classmethod
    def from_source_history(cls, q: Tensor, obj_pos: Tensor, bits: Tensor,
                            object_names: Sequence[str],
                            goal_atoms: Sequence[Sequence[str]]
                            ) -> tuple["PhasePotential", list[PhaseRecord]]:
        """Replay a source episode's per-step history, leaving the memos at its
        final step.  This is the object that gets forked at `tau`."""
        st = cls.at_start(q[0], obj_pos[0], object_names, goal_atoms)
        return st, [st.step(q[t], obj_pos[t], bits[t]) for t in range(q.shape[0])]

    def fork(self) -> "PhasePotential":
        """Independent evolving memos; baselines are shared read-only values."""
        return replace(self, ever_true=list(self.ever_true),
                       was_lifted=dict(self.was_lifted),
                       dxy_at_lift=dict(self.dxy_at_lift))

    # ---- one step -------------------------------------------------------
    def step(self, q_t: Tensor, obj_pos_t: Tensor, bits_t) -> PhaseRecord:
        current = [bool(b) for b in bits_t]
        violation = any(e and not n for e, n in zip(self.ever_true, current))
        for i, now in enumerate(current):
            self.ever_true[i] = self.ever_true[i] or now
        completed = sum(current)
        active = next((i for i, now in enumerate(current) if not now), -1)
        if active < 0:
            return PhaseRecord(float(self.n_atoms), "done", -1, violation)

        row = self.object_rows[active]
        basket_row = self.receptacle_rows[active]
        eef = q_t[0:3]
        obj = obj_pos_t[row]
        d_eef = float(torch.linalg.vector_norm(eef - obj))
        dxy = float(torch.linalg.vector_norm(obj[:2] - obj_pos_t[basket_row, :2]))
        delta_z = float(obj[2]) - self.start_z[row]
        displacement = float(torch.linalg.vector_norm(obj - self.start_pos[row]))

        attached = d_eef <= NEAR_DISTANCE and displacement >= ATTACH_DISPLACEMENT
        lifted_now = attached and delta_z >= LIFT_THRESHOLD
        if lifted_now and not self.was_lifted[row]:
            self.was_lifted[row] = True
            self.dxy_at_lift[row] = max(dxy, BASKET_RADIUS_XY + 1e-3)
        in_zone = dxy <= BASKET_RADIUS_XY and self.was_lifted[row]

        if in_zone:
            basket_z = float(obj_pos_t[basket_row, 2])
            descent = _clip01(1.0 - (float(obj[2]) - basket_z) / PLACEMENT_DESCENT_REF)
            phi, phase = 0.9 + 0.1 * descent, "placement"
        elif self.was_lifted[row] and attached:
            reference = self.dxy_at_lift[row] or dxy
            phi, phase = 0.6 + 0.3 * (1.0 - _clip01(dxy / reference)), "transport"
        elif attached and delta_z > 0:
            phi, phase = 0.4 + 0.2 * _clip01(delta_z / LIFT_REFERENCE), "lift"
        elif d_eef <= NEAR_DISTANCE:
            phi, phase = 0.2 + 0.2 * _clip01(displacement / ATTACH_DISPLACEMENT), "grasp"
        else:
            reference = max(self.start_eef_d[row], NEAR_DISTANCE + 1e-3)
            phi, phase = 0.2 * (1.0 - _clip01(d_eef / reference)), "approach"
        return PhaseRecord(completed + phi, phase, active, violation)


# --------------------------------------------------------------------------
# Action 2M.4 continuous action-consequence score
# --------------------------------------------------------------------------

GAMMA = 0.99


def discounted_delta(phis: Sequence[float], start: int, n: int,
                     gamma: float = GAMMA) -> float:
    """sum_{j=1..n} gamma^(j-1) * (Phi_{start+j} - Phi_{start+j-1}).

    `phis` must already hold Phi constant after an early terminal transition,
    so a branch that ends early contributes zero increments thereafter rather
    than an undefined value.
    """
    total = 0.0
    for j in range(1, n + 1):
        a, b = start + j - 1, start + j
        if b >= len(phis):
            break
        total += (gamma ** (j - 1)) * (phis[b] - phis[a])
    return float(total)


def q_phi(phis: Sequence[float], c: int = 10, h: int = 80,
          gamma: float = GAMMA) -> dict:
    """`phis[0] = Phi(tau)`, then one entry per executed environment step.

        GPhi_exec = sum_{j=1..c}   gamma^(j-1) (Phi_j - Phi_{j-1})
        GPhi_cont = sum_{j=1..H}   gamma^(j-1) (Phi_{c+j} - Phi_{c+j-1})
        QPhi      = GPhi_exec + gamma^c * GPhi_cont(80)
    """
    g_exec = discounted_delta(phis, 0, c, gamma)
    out = {"GPhi_exec": g_exec,
           "dPhi_exec": float(phis[min(c, len(phis) - 1)] - phis[0])}
    for hh in (20, 40, 80):
        out[f"GPhi_cont_{hh}"] = discounted_delta(phis, c, hh, gamma)
        out[f"dPhi_cont_{hh}"] = float(
            phis[min(c + hh, len(phis) - 1)] - phis[min(c, len(phis) - 1)])
    # `h` is the anchor's actual continuation length, which is FIXED at 80 for
    # the tau=160 banks but VARIABLE for a phase-aligned anchor whose tau moves
    # with the pick event.  Compute it explicitly rather than assuming it is one
    # of the precomputed reporting horizons.
    out[f"GPhi_cont_{h}"] = discounted_delta(phis, c, h, gamma)
    out[f"dPhi_cont_{h}"] = float(
        phis[min(c + h, len(phis) - 1)] - phis[min(c, len(phis) - 1)])
    out["QPhi"] = g_exec + (gamma ** c) * out[f"GPhi_cont_{h}"]
    return out
