"""V7.6 state reacher — the scripted servo with the missing lift phase.

Diagnosed 2026-08-15 by `scripts/precheck_v076_reacher.py` (0/12) and the
step trace recorded in `2026-08-15_v076_precheck_r1/servo_diagnosis.json`.

`scripts.collect_v069_corrections.scripted_servo_action` in `pick_up` mode
ends at

    else:                       # close enough to grasp
        target = obj + [0, 0, 0.01]
        grip   = +1

where `obj` is the object's CURRENT position. Once the gripper closes, the
object moves with the gripper, so the target is defined relative to the
thing being held: the error signal collapses to a constant ~1.5 cm offset
and the controller converges to a fixed point. Measured on t1 seed 2600 --
after step 32 the trace is flat at lateral 0.0039, `eef.z - obj.z` 0.0155,
object displacement 0.0044 m, and object dz **-0.0013 m**, held to step 80.

The `pick_up` milestone needs displacement >= 0.02 m AND dz >= +0.03 m
(`task_automaton.PICK_DISPLACEMENT/PICK_LIFT`). A servo with no lift phase
cannot ever satisfy it. That is the mechanical explanation of V7.4B's
`privileged_servo` scoring 1 positive in 63 attempts.

**The v069 function is left untouched on purpose.** V6.9, V7.3 and V7.4
records were produced against it; editing it in place would silently
change the meaning of those runs. This is a new instrument with its own
name, and any run using it says so.

What is repaired, and nothing else:

  pick_up   an explicit phase machine -- approach, descend, grasp (hold
            the close command while the fingers actually close), then
            LIFT toward `start_pos.z + lift_height`. The lift target is
            anchored to the object's EPISODE-START height, which is the
            same reference the milestone criterion uses, so the
            controller is driving the quantity it is scored on.
  place     transport above the region, lower, release. Unchanged in
            spirit from v069; it only ever worked when something was
            already held, which now happens.
  retreat   ROUND 2, diagnosed the same way. With the lift repaired, all
            twelve attempts stalled at depth 2->3: after placing into the
            cabinet drawer the end effector sat frozen at
            `[0.024, -0.112, 1.091]` with x already aligned to the next
            object and only +0.161 m of y to cover, for 70 straight
            steps. It was inside the drawer and commanding a diagonal
            move straight through the cabinet. Both phase machines now
            RETREAT to a clearance height first and TRAVERSE at that
            height, descending only over the target. Standard
            pick-and-place staging; the diagonal shortcut was the defect.

Gains, clip, and the 5 cm unit-action convention are inherited unchanged.

Every action this module produces is an ACQUISITION INSTRUMENT: it
creates states, and is never a policy target.
"""

from __future__ import annotations

import numpy as np

# inherited from scripts.collect_v069_corrections, unchanged
ACTION_UNIT = 0.05          # one unit of position command ~ 5 cm
LATERAL_NEAR = 0.015        # xy tolerance to start descending
DESCEND_CLEAR = 0.02        # eef-above-object height that still descends
PLACE_LATERAL = 0.03        # xy tolerance above the target region

# registered here
LIFT_HEIGHT = 0.12          # above episode-start z; milestone needs +0.03
GRASP_HOLD = 8              # steps to hold the close command before lifting
LIFT_DONE = 0.05            # dz above start that ends the lift phase
PLACE_DROP = 0.04           # height above the region site before release
RELEASE_HOLD = 5
CLEARANCE = 0.25            # traverse height above the object's start z
RETREAT_MAX = 25            # bounded, so retreat cannot eat the budget


def site_pos(env, name: str) -> np.ndarray:
    return env._env.env.sim.data.get_site_xpos(name).copy()


def eef_pos(env) -> np.ndarray:
    inner = env._env.env
    robot = inner.robots[0]
    if hasattr(robot, "eef_site_id"):
        return np.asarray(inner.sim.data.site_xpos[robot.eef_site_id]).copy()
    obs = env._format_raw_obs(inner._get_observations())
    return np.asarray(obs["robot_state"]["eef"]["pos"])


class ScriptedReacher:
    """Stateful phase machine for one (mode, object, region) phase.

    Stateful only where the physics demands it: closing a gripper takes
    several steps, and no instantaneous observation distinguishes "just
    commanded closed" from "holding". Everything else is recomputed from
    the current state each call, so a restored snapshot resumes correctly
    as long as the reacher is rebuilt with it.
    """

    def __init__(self, bodies: dict, lift_height: float = LIFT_HEIGHT,
                 grasp_hold: int = GRASP_HOLD):
        self.bodies = bodies
        self.lift_height = float(lift_height)
        self.grasp_hold = int(grasp_hold)
        self.phase = "retreat"
        self._held = 0
        self._released = 0
        self._retreated = 0
        self.trace: list[str] = []

    def reset_phase(self, phase: str = "retreat") -> None:
        self.phase = phase
        self._held = 0
        self._released = 0
        self._retreated = 0

    def _retreat(self, eef, clear_z, grip):
        """Straight up to clearance before any lateral motion.

        Bounded by RETREAT_MAX so a geometrically impossible clearance
        degrades into the next phase instead of consuming the budget.
        """
        self._retreated += 1
        if eef[2] >= clear_z or self._retreated > RETREAT_MAX:
            return None
        return self._cmd(eef + np.array([0.0, 0.0, 0.10]), eef, grip)

    def _obj(self, env, name: str) -> np.ndarray:
        from lcwm.probe_data import body_positions
        return body_positions(env, [self.bodies[name]])[0]

    @staticmethod
    def _cmd(target: np.ndarray, eef: np.ndarray, grip: float) -> np.ndarray:
        a = np.zeros(7, dtype=np.float64)
        a[:3] = np.clip((target - eef) / ACTION_UNIT, -1.0, 1.0)
        a[6] = grip
        return a

    def act(self, env, mode: str, obj_name: str, region: str | None,
            start_pos: np.ndarray | None = None) -> np.ndarray:
        eef = eef_pos(env)
        obj = self._obj(env, obj_name)
        if mode == "pick_up":
            a = self._pick(eef, obj, start_pos)
        elif mode == "place":
            a = self._place(env, eef, obj, region)
        else:
            raise ValueError(f"unsupported reacher mode {mode!r}")
        self.trace.append(self.phase)
        return a

    # ---- pick_up -----------------------------------------------------
    def _pick(self, eef, obj, start_pos) -> np.ndarray:
        lateral = float(np.linalg.norm((obj - eef)[:2]))
        # the lift target is anchored to the object's episode-start z --
        # the same reference PICK_LIFT is measured against
        base_z = float(start_pos[2]) if start_pos is not None \
            else float(obj[2])
        clear_z = base_z + CLEARANCE

        if self.phase == "retreat":
            a = self._retreat(eef, clear_z, -1.0)
            if a is not None:
                return a
            self.phase = "approach"

        if self.phase in ("approach", "descend") and lateral > LATERAL_NEAR:
            self.phase = "approach"
            # traverse AT clearance height, never diagonally through
            # whatever the previous phase left the arm inside
            return self._cmd(
                np.array([obj[0], obj[1], max(eef[2], clear_z)]),
                eef, -1.0)

        if self.phase in ("approach", "descend"):
            if eef[2] - obj[2] > DESCEND_CLEAR:
                self.phase = "descend"
                return self._cmd(obj + np.array([0.0, 0.0, 0.005]),
                                 eef, -1.0)
            self.phase = "grasp"
            self._held = 0

        if self.phase == "grasp":
            self._held += 1
            if self._held >= self.grasp_hold:
                self.phase = "lift"
            # hold position while the fingers close
            return self._cmd(obj + np.array([0.0, 0.0, 0.005]), eef, 1.0)

        # lift: drive the OBJECT up from where it started
        if obj[2] - base_z >= LIFT_DONE:
            self.phase = "lifted"
        target = np.array([obj[0], obj[1], base_z + self.lift_height]) \
            + (eef - obj)
        return self._cmd(target, eef, 1.0)

    # ---- place -------------------------------------------------------
    def _place(self, env, eef, obj, region) -> np.ndarray:
        if region is None:
            raise ValueError("place mode requires a region site")
        tgt = site_pos(env, region)
        lateral = float(np.linalg.norm((tgt - obj)[:2]))
        clear_z = max(float(tgt[2]), float(obj[2])) + CLEARANCE

        # NO retreat in place mode. Scoped by mechanism, not by score:
        # a place phase always follows a pick, whose lift phase has
        # already left the arm high and clear. Blanket-retreating here
        # raised the held object 0.25 m above the drawer before lowering
        # and regressed depth 2 from 2/3 to 0/3 (round-2 measurement,
        # kept in the record). Retreat is needed only where the arm ends
        # a phase INSIDE a confined region -- i.e. a pick that follows a
        # place.
        if self.phase == "retreat":
            self.phase = "transport"

        if self.phase not in ("lower", "release"):
            self.phase = "transport"
        if self.phase == "transport":
            if lateral > PLACE_LATERAL:
                # carry the OBJECT over the region at clearance height
                return self._cmd(
                    np.array([tgt[0], tgt[1], max(eef[2], clear_z)]),
                    eef, 1.0)
            self.phase = "lower"

        if self.phase == "lower":
            if obj[2] - tgt[2] > PLACE_DROP:
                return self._cmd(tgt + np.array([0.0, 0.0, PLACE_DROP])
                                 + (eef - obj), eef, 1.0)
            self.phase = "release"
            self._released = 0

        self._released += 1
        if self._released > RELEASE_HOLD:
            # retreat upward once the object is free
            return self._cmd(eef + np.array([0.0, 0.0, 0.05]), eef, -1.0)
        return self._cmd(eef, eef, -1.0)
