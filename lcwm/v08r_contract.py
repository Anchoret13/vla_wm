"""Action 1R / V8.0R executable contract (daily 2026-08-19 §6, Stage 1R.0).

Pure definitions and pure functions: no torch, no LIBERO, no environment.  The
1R.1 / 1R.2 runners and every later V8 stage import this module so that the
anchor identity, stratum key, deadline arithmetic, stall rule, and decision
rules exist in exactly one place and are sealed by one hash.

Why each piece exists - all four are defects the 2026-08-19 audit found in the
V8.0 instrument, not hypotheticals:

* `stratum_key`         - V8.0 aggregated by a scalar phase count.  On
                          `chain2b` the same `phase = 2` covered five
                          tomato-done/cream-unresolved episodes and two
                          cream-done/tomato-unresolved ones: mutually exclusive
                          remaining problems pooled into one number.
* `c_eff` / `D`         - V8.0 never modelled the deadline.  16 of 18
                          `chain1b` failures had 20-60 steps left at the pick
                          event, so a 10-step candidate prefix leaves 10-50
                          steps of continuation against a frozen place-class
                          bound of 71.
* `observable_stall`    - the frozen `W = 345` exceeds `chain1b`'s entire
                          250-step episode, so a 345-step no-progress suffix
                          cannot be observed there at all.
* `clopper_pearson_*`   - the 1R.1 decision rule is exact one-sided, because
                          0/20 must be reported as "no effect detected above
                          14%", never as "no effect".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

SCHEMA_VERSION = "v08r.1"

# --------------------------------------------------------------------------
# 1R.0 item 4 - the deployment deadline is a hard boundary on OFFICIAL data
# --------------------------------------------------------------------------


class Role(str, Enum):
    """Ledger role.  `subrole` refines it; both are written on every step."""

    CALIBRATION = "calibration"          # 1R.1, 1R.2; never trains anything
    BOOTSTRAP = "bootstrap"              # V8.1 B_boot
    MODEL_TRAIN = "model_train"
    MODEL_CALIB = "model_calib"
    SELECTOR_ASSESS = "selector_assess"
    VERIFIED_CORRECTION = "verified_correction"
    MODEL_TARGET_SEAL = "model_target_seal"
    BEHAVIOR_EVAL = "behavior_eval"


class Subrole(str, Enum):
    ACQ_ROLLOUT = "acq_rollout"
    ANCHOR_SEARCH = "anchor_search"
    REF_REPEAT = "ref_repeat"
    BRANCH_EXEC = "branch_exec"
    BRANCH_CONT = "branch_cont"
    ASSESS = "assess"
    DEADLINE_PROBE = "deadline_probe"    # 1R.1 extended-horizon rollouts
    DUPLICATE_CHECK = "duplicate_check"  # 1R.1 prefix-identity subset


class Officiality(str, Enum):
    """Whether an observation may enter a success/correction/value claim."""

    OFFICIAL = "official"                # step <= L
    QUARANTINED_POST_DEADLINE = "quarantined_post_deadline"   # step > L


def officiality(step: int, deadline: int) -> Officiality:
    """1R.0 item 4.  Beyond the frozen deadline an outcome is diagnostic only.

    It is never deleted - the 1R.1 extended horizons exist precisely to observe
    it - but it may not enter official success, correction, value, or
    interaction-efficiency numbers at the frozen deadline.
    """
    return (Officiality.OFFICIAL if step <= deadline
            else Officiality.QUARANTINED_POST_DEADLINE)


# --------------------------------------------------------------------------
# 1R.0 item 3 - deadline-aware execution arithmetic
# --------------------------------------------------------------------------


def c_eff(c: int, deadline: int, tau: int) -> int:
    """Executable prefix length at anchor step `tau` under deadline `L`.

        c_eff = min(c, max(0, L - tau))

    Zero means the anchor admits no executable candidate at all: it is not a
    usable anchor, and the 1R.2 feasibility panel must reject it rather than
    silently execute a truncated branch.
    """
    if c < 0 or deadline < 0 or tau < 0:
        raise ValueError(f"negative input: {c=} {deadline=} {tau=}")
    return min(c, max(0, deadline - tau))


def continuation_budget(deadline: int, tau: int, c_eff_: int) -> int:
    """Official continuation steps remaining after the candidate prefix.

        D = max(0, L - tau - c_eff)
    """
    return max(0, deadline - tau - c_eff_)


def anchor_admissible(deadline: int, tau: int, c: int, min_continuation: int
                      ) -> tuple[bool, str]:
    """1R.0 item 5 deadline-backoff rule, applied prospectively.

    An anchor is admissible only if it leaves a full candidate prefix AND at
    least `min_continuation` official continuation steps.  `min_continuation`
    is frozen per milestone class from the achieved-state-conditioned q90; it
    is NOT the global `H_max`, which the audit reopened.
    """
    ce = c_eff(c, deadline, tau)
    if ce < c:
        return False, f"prefix truncated by deadline: c_eff={ce} < c={c}"
    d = continuation_budget(deadline, tau, ce)
    if d < min_continuation:
        return False, (f"continuation {d} < required {min_continuation} "
                       f"(tau={tau}, L={deadline})")
    return True, "admissible"


# --------------------------------------------------------------------------
# 1R.0 item 5 - a stall rule that is observable inside the deadline
# --------------------------------------------------------------------------


def observable_stall_window(w_derived: int, deadline: int, tau: int
                            ) -> tuple[int, bool]:
    """Return `(W_used, is_observable)`.

    A stall of `W` steps can only be *observed* from `tau` if `L - tau >= W`.
    V8.0 froze `W = 345` on a task whose episodes end at 250; no state there
    can literally be called a 345-step `stall_onset`.  When the window is not
    observable the caller must either move the anchor earlier, shorten `W` by a
    registered rule, or record the state under a different name - never assert
    the unobserved window.
    """
    available = max(0, deadline - tau)
    return (min(w_derived, available), available >= w_derived)


# --------------------------------------------------------------------------
# 1R.0 item 2 - stratum identity over achieved SETS, never a scalar phase
# --------------------------------------------------------------------------


#: Registered actionability preconditions.  For the Chain family the subgoal
#: list is interleaved `pick_up X` / `place X region`, and a placement is
#: actionable only while the object is held - i.e. its `pick_up` is CURRENTLY
#: valid.  A `pick_up` is actionable whenever it is unresolved.  This is the
#: only precondition model registered for V8; a task family needing another
#: must register it before use rather than inferring one at run time.
def actionable_indices(subgoals: tuple[str, ...], current_valid: frozenset[int],
                       unresolved: frozenset[int]) -> frozenset[int]:
    out = set()
    for i in sorted(unresolved):
        kind = subgoals[i].split()[0]
        if kind == "pick_up":
            out.add(i)
        elif kind == "place":
            pick = i - 1
            if pick >= 0 and subgoals[pick].split()[0] == "pick_up" \
                    and pick in current_valid:
                out.add(i)
        else:                       # open / close: no registered precondition
            out.add(i)
    return frozenset(out)


@dataclass(frozen=True)
class StratumKey:
    """Aggregation identity for anchors, corrections, value, and frontiers.

    Identity is the task plus three SETS of milestone indices, and `unresolved`
    and `actionable` are stored as real fields rather than recomputed by
    callers - 1R.0 item 2 requires them explicit.  `n_milestones` and
    `subgoals` are carried so the key is self-contained: an earlier version
    exposed `unresolved_set(n_milestones)`, which made the same key yield
    different unresolved sets depending on what the caller passed.

    `step` and `time_to_go` are deliberately absent: the audit requires them as
    covariates/bins, not identity, so two episodes at the same physical state
    with different clocks still aggregate together.
    """

    task: str
    n_milestones: int
    subgoals: tuple[str, ...]
    ever_achieved: frozenset[int]
    current_valid: frozenset[int]
    damaged: frozenset[int]
    unresolved: frozenset[int] = field(init=False)
    actionable: frozenset[int] = field(init=False)

    def __post_init__(self) -> None:
        if len(self.subgoals) != self.n_milestones:
            raise ValueError(
                f"{len(self.subgoals)} subgoals vs n_milestones={self.n_milestones}")
        for name, s_ in (("ever_achieved", self.ever_achieved),
                         ("current_valid", self.current_valid),
                         ("damaged", self.damaged)):
            bad = {i for i in s_ if not 0 <= i < self.n_milestones}
            if bad:
                raise ValueError(f"{name} has out-of-range indices {sorted(bad)}")
        if not self.current_valid <= self.ever_achieved:
            raise ValueError("current_valid must be a subset of ever_achieved")
        unresolved = frozenset(range(self.n_milestones)) - self.current_valid
        object.__setattr__(self, "unresolved", unresolved)
        object.__setattr__(self, "actionable",
                           actionable_indices(self.subgoals, self.current_valid,
                                              unresolved))

    def key(self) -> str:
        f = lambda s_: ",".join(str(i) for i in sorted(s_)) or "-"
        return (f"{self.task}|ever:{f(self.ever_achieved)}"
                f"|valid:{f(self.current_valid)}|dmg:{f(self.damaged)}")

    def to_dict(self) -> dict:
        f = lambda s_: sorted(s_)
        return {"key": self.key(), "task": self.task,
                "n_milestones": self.n_milestones, "subgoals": list(self.subgoals),
                "ever_achieved": f(self.ever_achieved),
                "current_valid": f(self.current_valid),
                "damaged": f(self.damaged), "unresolved": f(self.unresolved),
                "actionable": f(self.actionable)}


def stratum_key(task: str, subgoals, ever_achieved, current_valid,
                damaged=()) -> StratumKey:
    subgoals = tuple(subgoals)
    return StratumKey(task, len(subgoals), subgoals,
                      frozenset(map(int, ever_achieved)),
                      frozenset(map(int, current_valid)),
                      frozenset(map(int, damaged)))


def time_to_go_bin(deadline: int, tau: int, edges=(0, 25, 50, 100, 200)) -> str:
    """`time_to_go` as a reported covariate bin.  Never part of identity."""
    ttg = max(0, deadline - tau)
    for lo, hi in zip(edges, edges[1:]):
        if lo <= ttg < hi:
            return f"[{lo},{hi})"
    return f"[{edges[-1]},inf)"


def frontier_moved(before: StratumKey, after: StratumKey,
                   after_success: bool) -> bool:
    """Frontier movement = contraction of the unresolved SET, or success.

    Explicitly not an increase in an ordered-prefix index: on `chain2b` all 20
    successes ran cream->tomato, the reverse of the listed order, so an
    ordered-prefix frontier would report no movement on every success there.
    """
    if after_success:
        return True
    return after.unresolved < before.unresolved


# --------------------------------------------------------------------------
# 1R.0 item 7 - exact one-sided decision rule
# --------------------------------------------------------------------------


def _betacf(a: float, b: float, x: float) -> float:
    TINY, EPS, ITMAX = 1e-300, 3e-16, 500
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < TINY:
        d = TINY
    d, h = 1.0 / d, 1.0 / d
    for m in range(1, ITMAX + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < TINY:
            d = TINY
        if abs(c) < TINY:
            c = TINY
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < TINY:
            d = TINY
        if abs(c) < TINY:
            c = TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b); dependency-free."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbeta) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta) * _betacf(b, a, 1.0 - x) / b


def _beta_ppf(p: float, a: float, b: float) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if betainc(a, b, mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """Exact one-sided upper bound.  0/20 -> 0.1391."""
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    return _beta_ppf(1.0 - alpha, k + 1, n - k)


def clopper_pearson_lower(k: int, n: int, alpha: float = 0.05) -> float:
    """Exact one-sided lower bound."""
    if n <= 0 or k <= 0:
        return 0.0
    return _beta_ppf(alpha, k, n - k + 1)


# --------------------------------------------------------------------------
# 1R.1 registered decision rule (daily 2026-08-19 §6, Stage 1R.1)
# --------------------------------------------------------------------------

#: (frozen deployment deadline L, mid checkpoint, extended horizon), daily §6.
#: Held here rather than in the runner so the interaction cap is DERIVED from
#: the horizons it funds and the two cannot drift apart.
CHECKPOINTS: dict[str, tuple[int, int, int]] = {
    "chain1b_lr2": (250, 350, 500),
    "chain2b_lr2": (500, 750, 990),
}


def deadline(task: str) -> int:
    return CHECKPOINTS[task][0]


def extended_horizon(task: str) -> int:
    return CHECKPOINTS[task][2]


MATERIALITY = 0.15          # late-conversion effect size that counts
PANEL_1 = 20                # seeds 3400-3419
PANEL_2 = 20                # seeds 3420-3439, opened only on 1-3/20
MAX_PANELS = 2              # "No third panel is allowed."
DUPLICATE_N = 5             # seeds/task re-run at the original horizon


@dataclass
class DeadlineVerdict:
    label: str
    k: int
    n: int
    upper: float
    lower: float
    reason: str
    opens_second_panel: bool = False


def deadline_sensitivity_verdict(k: int, n: int) -> DeadlineVerdict:
    """`k` = late conversions (succeed after the frozen deadline, by the
    extended horizon) out of `n` calibration episodes.

    Panel 1 (n == 20 exactly):  0 -> no material effect detected;
                                >= 4 -> materially sensitive; 1-3 -> open panel 2.
    Pooled  (n == 40 exactly):  upper < 0.15 -> no material effect detected;
                                lower > 0.15 -> materially sensitive;
                                else INDETERMINATE, treated as sensitive.

    Every label is derived from the computed bound, never from `k == 0`.  The
    earlier version took the panel-1 branch for any `n <= 20` and shortcut on
    `k == 0`, so `deadline_sensitivity_verdict(0, 18)` returned "no material
    effect" carrying the literal text "upper bound 0.1533 < 0.15" - a sealed
    decision function asserting a claim its own interval refutes.  A short
    panel is now an underfill outcome, not a verdict.
    """
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"invalid counts: {k=} {n=}")
    up = clopper_pearson_upper(k, n)
    lo = clopper_pearson_lower(k, n)

    if n not in (PANEL_1, PANEL_1 + PANEL_2):
        if n > PANEL_1 + PANEL_2:
            raise ValueError(
                f"n={n} exceeds the registered maximum {PANEL_1 + PANEL_2}; "
                f"MAX_PANELS={MAX_PANELS} and no third panel is allowed")
        return DeadlineVerdict(
            "PANEL_UNDERFILLED", k, n, up, lo,
            f"n={n} is not a registered panel size ({PANEL_1} or "
            f"{PANEL_1 + PANEL_2}); the missing seeds must be re-run within the "
            f"stage cap, or the stage halts. No verdict is issued: at n={n} the "
            f"one-sided upper bound is {up:.4f}.")

    if n == PANEL_1:
        if k == 0:
            # assert rather than assume: upper(0, 20) = 0.1391 < 0.15
            assert up < MATERIALITY, f"panel size {n} underpowered: {up:.4f}"
            return DeadlineVerdict(
                "NO_MATERIAL_EFFECT_DETECTED", k, n, up, lo,
                f"0/{n}; one-sided 95% upper bound {up:.4f} < {MATERIALITY}: no "
                f"effect above {MATERIALITY:.0%} detected (not 'no effect')")
        if k >= 4:
            return DeadlineVerdict("MATERIALLY_DEADLINE_SENSITIVE", k, n, up, lo,
                                   f"{k}/{n} late conversions; lower bound {lo:.4f}")
        return DeadlineVerdict("OPEN_SECOND_PANEL", k, n, up, lo,
                               f"{k}/{n} is in the 1-3 band; bounds "
                               f"[{lo:.4f}, {up:.4f}] straddle {MATERIALITY}", True)

    if up < MATERIALITY:
        return DeadlineVerdict("NO_MATERIAL_EFFECT_DETECTED", k, n, up, lo,
                               f"pooled {k}/{n}; upper {up:.4f} < {MATERIALITY}")
    if lo > MATERIALITY:
        return DeadlineVerdict("MATERIALLY_DEADLINE_SENSITIVE", k, n, up, lo,
                               f"pooled {k}/{n}; lower {lo:.4f} > {MATERIALITY}")
    return DeadlineVerdict("INDETERMINATE_TREAT_AS_SENSITIVE", k, n, up, lo,
                           f"pooled {k}/{n}; [{lo:.4f},{up:.4f}] straddles "
                           f"{MATERIALITY}; conservatively deadline-sensitive")


# --------------------------------------------------------------------------
# 1R.2 stratum feasibility (daily 2026-08-19 §6, Stage 1R.2)
# --------------------------------------------------------------------------

#: Diagnostic readout horizons, measured in steps FROM THE ANCHOR, registered
#: by the audit.  On both tasks the later readouts land past the deployment
#: deadline by construction - a chain1b anchor at tau~190 reaches h=400 only at
#: absolute step 590 against L=250 - which is exactly why the audit pairs them
#: with "post-deadline observations quarantined".  They are diagnostic; only
#: `officiality(step, L) == OFFICIAL` observations may enter an official number.
DIAGNOSTIC_READOUTS = (60, 120, 240, 400)
CONTINUATION_LEN = max(DIAGNOSTIC_READOUTS)   # every continuation runs this far

TARGET_GROUPS = 8           # independent natural anchor groups per task
REPEATS_PER_GROUP = 3       # R: repeats 2-3 measure within-anchor stochasticity
SOURCE_SEED_CAP = 40        # per task; underfill is a registered outcome


def stratum_feasibility_verdict(groups_filled: int, seeds_consumed: int
                                ) -> tuple[str, str]:
    """Underfill is `STRATUM_NOT_IDENTIFIED`, never permission to sample on.

    The `chain2b` modal stratum had prevalence 5/30 in the V8.0 confirmation
    panel, so underfill is a live possibility rather than a formality.
    """
    if groups_filled >= TARGET_GROUPS:
        return "STRATUM_IDENTIFIED", (
            f"{groups_filled}/{TARGET_GROUPS} groups in {seeds_consumed} seeds")
    if seeds_consumed >= SOURCE_SEED_CAP:
        return "STRATUM_NOT_IDENTIFIED", (
            f"only {groups_filled}/{TARGET_GROUPS} groups at the frozen "
            f"{SOURCE_SEED_CAP}-seed cap; substituting an easier state or "
            f"raising the cap is prohibited")
    return "IN_PROGRESS", f"{groups_filled}/{TARGET_GROUPS}, {seeds_consumed} seeds"


# --------------------------------------------------------------------------
# Interaction budget (1R.0 item 6) - a hard cap, not an estimate
# --------------------------------------------------------------------------

INTERACTION_CAP = {
    "1R.1_panel_1": {t: PANEL_1 * extended_horizon(t) for t in CHECKPOINTS},
    "1R.1_panel_2_conditional": {t: PANEL_2 * extended_horizon(t) for t in CHECKPOINTS},
    "1R.1_duplicate_subset": {t: DUPLICATE_N * deadline(t) for t in CHECKPOINTS},
    "1R.2_source_rollouts": {t: SOURCE_SEED_CAP * deadline(t) for t in CHECKPOINTS},
    # Sized FROM the registered readout schedule, not guessed: every one of the
    # 8 x 3 reference continuations runs the full CONTINUATION_LEN so the
    # [60, 120, 240, 400] columns are all populated.  A 50/250-step allotment
    # would have truncated chain1b below the 71-step place floor that certified
    # its own anchors admissible - the configuration framework §3.1.1 prohibits
    # after V7.7, where the budget was the instrument.
    "1R.2_reference_continuations": {
        t: TARGET_GROUPS * REPEATS_PER_GROUP * (10 + CONTINUATION_LEN)
        for t in CHECKPOINTS},
}

#: Which cap line each (stage, subrole) charges.  A single pooled counter would
#: let the duplicate subset eat the panel budget - and the panel lines have
#: ZERO headroom by construction (PANEL_1 x extended_horizon exactly), so the
#: abort would correlate with the result being measured: more deadline
#: failures -> longer episodes -> more likely halt.
CAP_LINE = {
    ("1R.1", 1, Subrole.DEADLINE_PROBE.value): "1R.1_panel_1",
    ("1R.1", 2, Subrole.DEADLINE_PROBE.value): "1R.1_panel_2_conditional",
    ("1R.1", 1, Subrole.DUPLICATE_CHECK.value): "1R.1_duplicate_subset",
    # Panel 2 has NO duplicate line: the sealed cap funds one subset
    # (DUPLICATE_N x deadline), and the 2026-08-19 panel-2 run overran it by
    # 349 steps by scheduling a second.  Caps are not raised after execution,
    # so the unfunded activity is removed instead.  Prefix identity was
    # established in panel 1 (10/10) and reconfirmed 3/3 before the halt.
}


def cap_line(stage: str, panel: int, subrole: str) -> str:
    try:
        return CAP_LINE[(stage, panel, subrole)]
    except KeyError:
        raise KeyError(f"no registered cap line for {stage=} {panel=} {subrole=}")


INTERACTION_CAP_TOTAL = sum(v for stage in INTERACTION_CAP.values()
                            for v in stage.values())


def cap_for(stage: str, task: str) -> int:
    return INTERACTION_CAP[stage][task]
