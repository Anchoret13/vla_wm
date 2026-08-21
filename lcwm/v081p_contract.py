"""Action 2P / V8.1P executable contract (daily 2026-08-20 §4).

Pure definitions and pure functions: no torch, no LIBERO, no environment.

Scope, stated once so no downstream artifact can drift from it: this pilot asks
whether VLA-supported first-10-action siblings at stratum-matched pre-failure
histories produce **repeatable local outcome variation** under a finite
deadline.  It tests no world model, no policy update, no generalization, and no
terminal-failure recovery, and its rows are permanently excluded from later
training and behavior evaluation.

Terminology is enforced here rather than left to prose (2026-08-20 §2):
`failure@L`, `right_censored@H`, `retrospective_deadline_backoff`.  The words
`terminal`, `irreducible`, `stall_onset`, and `earliest_unrecoverable` are not
available in this module and may not be introduced without the corresponding
evidence.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum

SCHEMA_VERSION = "v081p.1"

# --------------------------------------------------------------------------
# Frozen source, anchor, and horizon (2026-08-20 §4 "Frozen source and anchor")
# --------------------------------------------------------------------------

TASK = "chain1b_lr2"
DEADLINE = 250                  # L: a post-calibration EXPERIMENTAL horizon,
                                # frozen prospectively for this pilot.  It is
                                # not an intrinsic-capability horizon and not an
                                # external deployment requirement; every allowed
                                # claim is conditional on it.
C_PREFIX = 10                   # executed candidate prefix
TAU = 160                       # fixed anchor step: 160 + 10 + 80 = 250
H_READOUTS = (20, 40, 80)       # continuation readouts, steps after the prefix
H_PRIMARY = 80                  # the ADVANCE decision horizon
assert TAU + C_PREFIX + H_PRIMARY == DEADLINE

SOURCE_SEEDS = tuple(range(3480, 3500))     # all 20; never early-stop at quota
RESERVED_SEEDS = frozenset(range(3200, 3400))
RETIRED_SEEDS = frozenset(range(3440, 3480))   # old-registered 1R.2, retired
CALIBRATION_ONLY_SEEDS = frozenset(range(3400, 3440))

N_ANCHORS = 8
N_ALT_DRAWS = 32                # raw stochastic pi0 alternatives per anchor
MIN_UNIQUE_ALTS = 16            # after first-c dedup
N_DIVERSITY = 3
N_RANDOM = 3
N_CRN = 3                       # presealed continuation RNG keys
ANCHOR_KIND = "retrospective_deadline_backoff"


class Outcome(str, Enum):
    FAILURE_AT_L = "failure@250"
    SUCCESS_AT_L = "success@250"
    RIGHT_CENSORED = "right_censored@H"


class Subrole(str, Enum):
    SOURCE = "source"
    REF_PREFIX = "ref_prefix"
    REF_CONT = "ref_cont"
    DIV_PREFIX = "div_prefix"
    DIV_CONT = "div_cont"
    RAND_PREFIX = "rand_prefix"
    RAND_CONT = "rand_cont"


#: Distinct cap lines, per the daily's ledger requirement.  Every started or
#: aborted segment is charged; there are no retries, substitute anchors, extra
#: assessment branches, post-hoc quota increases, or reallocations.
_PREFIX_PLUS_CONT = C_PREFIX + H_PRIMARY        # 90 official steps per branch
INTERACTION_CAP = {
    Subrole.SOURCE.value: len(SOURCE_SEEDS) * DEADLINE,                  # 5,000
    Subrole.REF_PREFIX.value: N_ANCHORS * N_CRN * C_PREFIX,              #   240
    Subrole.REF_CONT.value: N_ANCHORS * N_CRN * H_PRIMARY,               # 1,920
    Subrole.DIV_PREFIX.value: N_ANCHORS * N_DIVERSITY * N_CRN * C_PREFIX,
    Subrole.DIV_CONT.value: N_ANCHORS * N_DIVERSITY * N_CRN * H_PRIMARY,
    Subrole.RAND_PREFIX.value: N_ANCHORS * N_RANDOM * N_CRN * C_PREFIX,
    Subrole.RAND_CONT.value: N_ANCHORS * N_RANDOM * N_CRN * H_PRIMARY,
}
INTERACTION_CAP_TOTAL = sum(INTERACTION_CAP.values())
BRANCH_SEGMENTS = N_ANCHORS * (N_CRN + (N_DIVERSITY + N_RANDOM) * N_CRN)   # 168
WORST_CASE_SEGMENTS = len(SOURCE_SEEDS) + 2 * BRANCH_SEGMENTS              # 356
assert INTERACTION_CAP_TOTAL == 5000 + N_ANCHORS * (
    N_CRN + (N_DIVERSITY + N_RANDOM) * N_CRN) * _PREFIX_PLUS_CONT == 20120


# --------------------------------------------------------------------------
# Eligible mask (2026-08-20 §4) — exact, not "approximately empty"
# --------------------------------------------------------------------------

#: chain1b subgoals are [pick_up tomato_sauce_1, place tomato_sauce_1 <region>].
ELIGIBLE = {"ever_achieved": frozenset(), "current_valid": frozenset(),
            "damaged": frozenset(), "actionable": frozenset({0}),
            "unresolved": frozenset({0, 1})}


def mask_conformant(ever_achieved, current_valid, damaged, actionable,
                    unresolved) -> tuple[bool, list[str]]:
    """Exact-mask conformance at `tau`.  Returns (ok, reasons_it_failed)."""
    got = {"ever_achieved": frozenset(map(int, ever_achieved)),
           "current_valid": frozenset(map(int, current_valid)),
           "damaged": frozenset(map(int, damaged)),
           "actionable": frozenset(map(int, actionable)),
           "unresolved": frozenset(map(int, unresolved))}
    bad = [f"{k}: want {sorted(ELIGIBLE[k])}, got {sorted(got[k])}"
           for k in ELIGIBLE if got[k] != ELIGIBLE[k]]
    return (not bad), bad


# --------------------------------------------------------------------------
# Presealed RNG keys — derived, never ad hoc
# --------------------------------------------------------------------------

def _key(root: str, *parts) -> int:
    h = hashlib.sha256("|".join([root, *map(str, parts)]).encode()).digest()
    return int.from_bytes(h[:8], "big") % (2 ** 31 - 1)


def reference_key(root: str, anchor_id: str) -> int:
    """Reference `u0` is drawn from its OWN key, disjoint from the alternative
    keys, with no action-geometry or outcome selection.  The source's cached
    post-`tau` action is never the reference: that suffix is conditioned to
    fail and cannot serve as a reference outcome."""
    return _key(root, "reference", anchor_id)


def alternative_keys(root: str, anchor_id: str) -> list[int]:
    return [_key(root, "alt", anchor_id, i) for i in range(N_ALT_DRAWS)]


def crn_keys(root: str, anchor_id: str) -> list[int]:
    """The same three continuation keys are used for the reference and for every
    alternative, so each alternative has three PAIRED comparisons."""
    return [_key(root, "crn", anchor_id, j) for j in range(N_CRN)]


# --------------------------------------------------------------------------
# Paired dominance (2026-08-20 §4) — same-key pairing, not best-of-three
# --------------------------------------------------------------------------

#: Priority order inherited from framework §5.5.  `dmg` is non-inferiority;
#: the rest are ordered gains.  `ttm` is lower-better.
COMPONENTS = ("dmg", "succ", "dp", "ttm", "G")
LOWER_IS_BETTER = frozenset({"dmg", "ttm"})


#: Continuation return: +1 at the step a milestone is first achieved,
#: discounted from the anchor.  Sealed here before execution so no reward can be
#: chosen after outcomes are visible.
GAMMA = 0.99


def outcome_vector(achieved_steps: dict, tau: int, horizon_end: int,
                   damage: int, success_step: int | None) -> dict:
    """The registered outcome components at one readout.

    `achieved_steps` maps milestone index -> absolute step, restricted by the
    caller to events at or before `horizon_end`.  `ttm` is censored at the
    readout when no milestone was gained, so a censored value can never appear
    to beat an attained one.
    """
    gains = {i: s for i, s in achieved_steps.items() if tau < s <= horizon_end}
    g = sum(GAMMA ** (s - tau) for s in gains.values())
    ttm = (min(gains.values()) - tau) if gains else (horizon_end - tau + 1)
    return {"dmg": float(damage),
            "succ": float(success_step is not None and success_step <= horizon_end),
            "dp": float(len(gains)), "ttm": float(ttm), "G": float(g)}


@dataclass(frozen=True)
class NoiseFloor:
    """Per-component floors, sealed from reference replay before execution."""
    dmg: float = 0.0
    succ: float = 0.0
    dp: float = 0.0
    ttm: float = 0.0
    G: float = 0.0

    def of(self, comp: str) -> float:
        return float(getattr(self, comp))


#: Automaton evaluation stride: milestone steps are only observed every 10
#: environment steps, so `ttm` is quantized to 10 and `G` inherits that timing
#: quantization.  Floors are DERIVED from that instrument property and sealed
#: before execution - they are not fitted to reference replay, because fitting a
#: floor to the very reference variability it must exclude would let the floor
#: be chosen after the outcomes exist.
EVAL_STRIDE = 10

#: The largest change in G that one stride of timing quantization can produce,
#: attained at the earliest possible milestone: GAMMA**0 - GAMMA**EVAL_STRIDE.
#: Conservative in the safe direction - it makes a difference HARDER to declare.
G_FLOOR = 1.0 - GAMMA ** EVAL_STRIDE

#: dmg, succ and dp are exact integer counts with no quantization error, so
#: their floors are zero: any difference in them is real.
SEALED_FLOORS_KWARGS = {"dmg": 0.0, "succ": 0.0, "dp": 0.0,
                        "ttm": float(EVAL_STRIDE), "G": G_FLOOR}


def compare_paired(alt: dict, ref: dict, floors: NoiseFloor) -> int:
    """One paired comparison at a single CRN key.  +1 alt better, -1 worse, 0 tie.

    Lexicographic over COMPONENTS with per-component noise floors; a difference
    inside its floor is a tie at that component and the comparison falls through.
    `dmg` is a hard non-inferiority gate: worse damage loses outright regardless
    of anything below it.
    """
    for comp in COMPONENTS:
        a, r = float(alt[comp]), float(ref[comp])
        if comp in LOWER_IS_BETTER:
            a, r = -a, -r
        d = a - r
        if abs(d) <= floors.of(comp):
            continue
        return 1 if d > 0 else -1
    return 0


@dataclass
class AlternativeVerdict:
    candidate_id: str
    per_key: list[int] = field(default_factory=list)
    paired_positive: bool = False
    paired_non_improving: bool = False
    label: str = "mixed"


def classify_alternative(cand_id: str, alt_by_key: list[dict],
                         ref_by_key: list[dict], floors: NoiseFloor
                         ) -> AlternativeVerdict:
    """`paired-positive`: beats its matched reference on ALL N_CRN keys.
    `paired-non-improving`: never beats reference AND is strictly worse on at
    least one paired key beyond the replay-noise floor.

    Reference variability and prefix physical effect by themselves are not
    candidate-induced outcome variation and cannot produce either label.
    """
    if len(alt_by_key) != N_CRN or len(ref_by_key) != N_CRN:
        raise ValueError(f"expected {N_CRN} paired keys, got "
                         f"{len(alt_by_key)}/{len(ref_by_key)}")
    per = [compare_paired(a, r, floors) for a, r in zip(alt_by_key, ref_by_key)]
    pos = all(c == 1 for c in per)
    non = (not any(c == 1 for c in per)) and any(c == -1 for c in per)
    label = "paired_positive" if pos else "paired_non_improving" if non else "mixed"
    return AlternativeVerdict(cand_id, per, pos, non, label)


@dataclass
class AnchorVerdict:
    anchor_id: str
    alternatives: list[AlternativeVerdict] = field(default_factory=list)

    @property
    def has_variation(self) -> bool:
        """Candidate-induced registered deadline-outcome variation: at least one
        alternative is not a pure tie against its matched reference."""
        return any(any(c != 0 for c in a.per_key) for a in self.alternatives)

    @property
    def n_positive(self) -> int:
        return sum(a.paired_positive for a in self.alternatives)

    @property
    def n_non_improving(self) -> int:
        return sum(a.paired_non_improving for a in self.alternatives)

    @property
    def has_both(self) -> bool:
        return self.n_positive >= 1 and self.n_non_improving >= 1


# --------------------------------------------------------------------------
# ADVANCE gate (2026-08-20 §4) — six conditions, all required
# --------------------------------------------------------------------------

GATE_MIN_VARIATION_ANCHORS = 4      # of 8, at H_PRIMARY
GATE_MIN_POSITIVE_ANCHORS = 2       # of 8, at H_PRIMARY
GATE_MIN_BOTH_ANCHORS = 2           # of 8, at H_PRIMARY


def advance_gate(sources_closed: bool, selected_in_order: bool,
                 reserved_used: int, replay_ok: int, replay_total: int,
                 restore_ok: int, restore_total: int,
                 pools_ok: bool, within_cap: bool,
                 anchors: list[AnchorVerdict]) -> dict:
    """All six conditions must hold.  Any unmet condition is a HALT, and a HALT
    does not justify training or sweeping a world model: the next change is one
    bounded anchor or proposal/setting revision, never a lower bar or a larger
    post-outcome budget."""
    n_var = sum(a.has_variation for a in anchors)
    n_pos = sum(a.n_positive >= 1 for a in anchors)
    n_both = sum(a.has_both for a in anchors)
    conds = {
        "1_sources_and_selection": bool(sources_closed and selected_in_order
                                        and reserved_used == 0),
        "2_restore_and_replay": bool(restore_ok == restore_total == BRANCH_SEGMENTS
                                     and replay_ok == replay_total
                                     == N_ANCHORS * N_CRN),
        "3_pools_and_cap": bool(pools_ok and within_cap),
        "4_variation": n_var >= GATE_MIN_VARIATION_ANCHORS,
        "5_positive": n_pos >= GATE_MIN_POSITIVE_ANCHORS,
        "6_both": n_both >= GATE_MIN_BOTH_ANCHORS,
    }
    return {"verdict": "ADVANCE" if all(conds.values()) else "HALT",
            "conditions": conds,
            "counts": {"anchors": len(anchors), "with_variation": n_var,
                       "with_positive": n_pos, "with_both": n_both},
            "horizon": H_PRIMARY,
            "note": ("H=20/40 are secondary diagnostics reported alongside; "
                     "diversity-vs-random yield may be described but the pilot "
                     "is not powered as a selector-efficiency comparison")}


def sealed_floors() -> "NoiseFloor":
    return NoiseFloor(**SEALED_FLOORS_KWARGS)


ALLOWED_ADVANCE_STATEMENT = (
    "On eight distinct chain1b@250, tau=160 anchors selected from one "
    "prospectively registered event-mask stratum, the six outcome-blind "
    "stock-VLA siblings per anchor produced repeatable candidate-induced "
    "finite-horizon outcome variation on at least 4/8 anchors, a paired-positive "
    "on at least 2/8, and both a positive and non-improving sibling on at least "
    "two anchors. It is not a WM, policy, benchmark-generalization, or "
    "terminal-recovery result.")
