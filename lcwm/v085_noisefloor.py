"""Action 2M.3 contract — separate candidate effect from continuation noise.

Pure definitions and pure statistics: no torch, no LIBERO, no environment.

The question is narrow and prior to everything downstream: at
`chain1b_lr2@250, tau=160`, does *which candidate you execute* move the
registered H80 outcome more than *which continuation seed you draw*?  Action
2M.1's bank could not answer it - 40 of its 48 anchors had a single repeat - and
when the 8 three-repeat anchors were decomposed the candidate effect came out
BELOW its own noise floor (12.50% matched-seed against 19.12% same-chunk
cross-seed).  This experiment is designed so the two rates are measured on the
same anchors with the same seeds.

Nothing here trains, selects, or evaluates a policy.  Every row is permanently
`calibration/noise_floor`.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

SCHEMA_VERSION = "v085_noisefloor_v1"

# --------------------------------------------------------------------------
# Locked experiment (daily 2026-08-23, "Locked experiment")
# --------------------------------------------------------------------------

TASK = "chain1b_lr2"
DEADLINE = 250
TAU = 160
C_PREFIX = 10
H = 80
assert TAU + C_PREFIX + H == DEADLINE

N_ANCHORS = 15                  # independent unit
N_RAW_DRAWS = 64                # outcome-blind pool per anchor
N_ALTERNATIVES = 4              # + 1 reference = 5 executed candidates
N_CANDIDATES = 1 + N_ALTERNATIVES
N_REPEATS = 12                  # shared continuation seeds, same 12 for all
MAX_SOURCES = 30                # hard ceiling; no extension on underfill

SOURCE_SEEDS = tuple(range(3900, 3930))          # exactly MAX_SOURCES
assert len(SOURCE_SEEDS) == MAX_SOURCES
RESERVED = frozenset(range(3200, 3400))
SPENT_BEFORE = frozenset(range(3400, 3440)) | frozenset(range(3480, 3500)) \
    | frozenset(range(3500, 3576)) | frozenset(range(3600, 3706)) \
    | frozenset(range(3800, 3816))
assert not (set(SOURCE_SEEDS) & (RESERVED | SPENT_BEFORE))

ROLE = "calibration"
SUBROLE = "noise_floor"
DATA_USE = ("permanently calibration/noise_floor: no row may enter WM or policy "
            "training, calibration of a later model, selector assessment, or "
            "behavior evaluation")

BRANCHES = N_ANCHORS * N_CANDIDATES * N_REPEATS                   # 900
SOURCE_CAP = MAX_SOURCES * DEADLINE                               # 7,500
BRANCH_CAP = BRANCHES * (C_PREFIX + H)                            # 81,000
INTERACTION_CAP = {"source": SOURCE_CAP, "branch": BRANCH_CAP}
INTERACTION_CAP_TOTAL = SOURCE_CAP + BRANCH_CAP                   # 88,500
assert BRANCHES == 900 and INTERACTION_CAP_TOTAL == 88_500

# --------------------------------------------------------------------------
# Outcome components and the sealed tie floors (unchanged from v081p/v082)
# --------------------------------------------------------------------------

#: Fixed BEFORE the run: D and ICC are computed separately for each, never
#: pooled and never chosen after the outcomes exist.
PRIMARY_CONTINUOUS = ("dp", "G", "ttm")
SECONDARY = ("dmg", "succ")               # mandatory safety/task readouts
ALL_COMPONENTS = ("dmg", "succ", "dp", "ttm", "G")

EVAL_STRIDE = 10
GAMMA = 0.99
#: Same derivation as Action 2P: floors come from the automaton's evaluation
#: stride, not fitted to the variability they must exclude.
FLOORS = {"dmg": 0.0, "succ": 0.0, "dp": 0.0,
          "ttm": float(EVAL_STRIDE), "G": 1.0 - GAMMA ** EVAL_STRIDE}


def is_nontie(a: float, b: float, component: str) -> bool:
    """Per-component tie test.  D is defined per component, so the label here is
    a single-component comparison, not the lexicographic one."""
    return abs(float(a) - float(b)) > FLOORS[component]


# --------------------------------------------------------------------------
# Candidate selection — outcome-blind, maximizing action-only spread
# --------------------------------------------------------------------------

def select_max_spread(alt_env, reference_env, n: int = N_ALTERNATIVES) -> list[int]:
    """Pick `n` of the raw draws to maximize action-only spread.

    Deterministic and outcome-blind: standardize the summed executed action over
    the pool, seed the greedy set with the draw FARTHEST FROM THE REFERENCE, then
    add farthest-point candidates, breaking ties by index.  Seeding at the
    reference-farthest draw (rather than a hash-chosen start, as in v081p/v083)
    is deliberate: this experiment wants the largest candidate/reference contrast
    the stock pool admits, because it is testing whether ANY contrast registers.
    """
    import numpy as np

    X = np.stack([np.asarray(a).reshape(-1) for a in alt_env])
    ref = np.asarray(reference_env).reshape(-1)
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    Z, zref = (X - mu) / sd, (ref - mu) / sd
    chosen = [int(np.argmax(np.linalg.norm(Z - zref, axis=1)))]
    while len(chosen) < n:
        d = np.min(np.linalg.norm(Z[:, None] - Z[chosen][None], axis=2), axis=1)
        d[chosen] = -1.0
        chosen.append(int(np.lexsort((np.arange(len(Z)), -d))[0]))
    return chosen


# --------------------------------------------------------------------------
# Primary statistic: D, anchor-clustered
# --------------------------------------------------------------------------

@dataclass
class AnchorRates:
    anchor_id: str
    candidate_nontie: int
    candidate_total: int
    noise_nontie: int
    noise_total: int

    @property
    def d(self) -> float:
        c = self.candidate_nontie / self.candidate_total if self.candidate_total else 0.0
        n = self.noise_nontie / self.noise_total if self.noise_total else 0.0
        return c - n


def anchor_rates(outcomes, component: str, reference_index: int = 0) -> AnchorRates:
    """`outcomes[c][r]` is the component value for candidate c, repeat r.

    candidate effect : candidate vs reference at the SAME repeat (shared seed)
    noise floor      : the SAME candidate across DIFFERENT repeats
    """
    C, R = len(outcomes), len(outcomes[0])
    cn = ct = nn = nt = 0
    for c in range(C):
        if c != reference_index:
            for r in range(R):
                ct += 1
                cn += is_nontie(outcomes[c][r], outcomes[reference_index][r], component)
        for i in range(R):
            for j in range(i + 1, R):
                nt += 1
                nn += is_nontie(outcomes[c][i], outcomes[c][j], component)
    return AnchorRates("", cn, ct, nn, nt)


def cluster_bootstrap_lower(values: list[float], *, key: int, b: int = 10000,
                            alpha: float = 0.05) -> dict:
    """One-sided lower bound, resampling ANCHORS with replacement.

    Anchors are the independent unit; repeats and candidate pairs are nested
    inside them and are never resampled as if independent.
    """
    import numpy as np

    if not values:
        return {"point": None, "lower": None, "b": b}
    rng = np.random.default_rng(key)
    v = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(v), size=(b, len(v)))
    means = v[idx].mean(axis=1)
    return {"point": float(v.mean()), "lower": float(np.quantile(means, alpha)),
            "b": b, "n_anchors": len(v), "alpha": alpha}


# --------------------------------------------------------------------------
# Candidate ICC
# --------------------------------------------------------------------------

def icc_one_way(outcomes) -> float | None:
    """ICC(1) with candidate as the grouping factor, within one anchor.

    Proportion of variance in the component attributable to candidate identity
    rather than to the continuation seed.  Returns None when undefined.
    """
    import numpy as np

    y = np.asarray(outcomes, dtype=float)
    k_groups, n = y.shape
    if k_groups < 2 or n < 2:
        return None
    grand = y.mean()
    msb = n * ((y.mean(axis=1) - grand) ** 2).sum() / (k_groups - 1)
    msw = ((y - y.mean(axis=1, keepdims=True)) ** 2).sum() / (k_groups * (n - 1))
    denom = msb + (n - 1) * msw
    if denom <= 0 or not math.isfinite(denom):
        return None
    return float((msb - msw) / denom)


def sign_consistency(outcomes, component: str, reference_index: int = 0) -> dict:
    """Per-candidate sign consistency across ALL repeats.

    A candidate effect that is real should hold its direction; Action 2M.1 had
    0 of 128 candidates do so across three repeats.
    """
    C, R = len(outcomes), len(outcomes[0])
    ever = consistent = 0
    for c in range(C):
        if c == reference_index:
            continue
        signs = []
        for r in range(R):
            d = float(outcomes[c][r]) - float(outcomes[reference_index][r])
            signs.append(0 if abs(d) <= FLOORS[component] else (1 if d > 0 else -1))
        nz = [s for s in signs if s != 0]
        if nz:
            ever += 1
            if len(set(nz)) == 1 and len(nz) == R:
                consistent += 1
    return {"candidates": C - 1, "ever_nontie": ever, "consistent_all_repeats": consistent}


# --------------------------------------------------------------------------
# Decision rule (daily 2026-08-23, "Decision and handoff")
# --------------------------------------------------------------------------

BOOTSTRAP_B = 10000
BOOTSTRAP_KEY_ROOT = "v085_noisefloor_bootstrap"


def bootstrap_key(root: str, component: str, statistic: str) -> int:
    h = hashlib.sha256(f"{BOOTSTRAP_KEY_ROOT}|{root}|{component}|{statistic}".encode())
    return int.from_bytes(h.digest()[:8], "big") % (2 ** 31 - 1)


def decide(per_component: dict) -> dict:
    """CANDIDATE_EFFECT_IDENTIFIED only if, for the SAME primary continuous
    component, the one-sided 95% anchor-cluster lower bound on D is above zero
    AND the candidate-ICC lower bound is positive."""
    hits = []
    for comp in PRIMARY_CONTINUOUS:
        r = per_component.get(comp, {})
        d_lo = (r.get("D") or {}).get("lower")
        i_lo = (r.get("ICC") or {}).get("lower")
        if d_lo is not None and i_lo is not None and d_lo > 0 and i_lo > 0:
            hits.append(comp)
    return {"verdict": "CANDIDATE_EFFECT_IDENTIFIED" if hits else "NOT_IDENTIFIED",
            "components_meeting_both": hits,
            "rule": ("same component must satisfy D_lower > 0 AND ICC_lower > 0; "
                     "components are never pooled and never chosen after the run"),
            "handoff": ("register the bounded-perturbation proposal family using the "
                        "measured effect/noise scale, then collect and train M0.2"
                        if hits else
                        "do NOT change the proposal family or train M0.2 at this "
                        "setting; the next decision is whether to change the outcome "
                        "readout or the anchor/setting - a formulation decision, not "
                        "another model or proposal sweep")}
