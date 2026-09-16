#!/usr/bin/env python
"""Matched four-arm evaluation driver for the 2026-09-12 chain3 round.

    # the default mode while the round is being set up: no simulator, no GPU
    python scripts/eval_v251_round.py --panel 96 --panel-start 9100 \
        --m1 <updated.pt> --m0 <stale.pt> --q <modelfree.pt>

    # only after the preregistration below has been read and the panel accepted
    python scripts/eval_v251_round.py --panel 96 --panel-start 9100 \
        --m1 ... --m0 ... --q ... --execute

WHAT THIS MEASURES.  Four arms on one shared environment panel of chain3_lr2 at
horizon 750, frozen pi0.5 backbone in every arm:

    M1    residual interface trained with advantages read off latents PREDICTED by
          the transition updated on D0 + D1
    M0    the same interface, the same data, the same budget, advantages read off a
          transition trained on D0 ONLY - the stale-WM control
    Q     the same interface, the same data, the same budget, no rollout - the
          same-data model-free control
    base  frozen pi0.5 through the identical code path with Delta := 0

The only quantity that may differ between M1 and M0 is the learned transition.
The checkpoint comparison is on unless `--no-check-matched` is passed: it loads
all three checkpoints on CPU, records every field that differs, and aborts when
the structural fields (dims, residual scale, conditioning space) differ or when
M1 and M0 carry the SAME transition weights.  It runs before the first
environment reset, not after 7 h of rollouts.  A field, or a transition, that no
checkpoint records is reported as NOT VERIFIABLE rather than as a passed check.
`check_run_matched` then repeats the comparison after the run on what the arms
actually ran with, from each arm's own summary.json.

WHAT WOULD OVERTURN ANY EFFECT THIS DRIVER REPORTS, registered here rather than
written after the numbers are read:

  * M1 vs M0 - the same configuration retrained from a second initialisation
    closing the gap.  Cross-panel replication is not cross-seed replication: on
    2026-09-01 one actor replicated at 0.750/0.760/0.719/0.771 across four panels
    while the same configuration at a different init gave 0.510, and seed variance
    at fixed hyperparameters was ~0.24 - larger than most between-arm differences
    chased that session (CLAUDE.md rule 9).
  * M1 vs Q - Q matching M1 once Q is given the same trained encoder.  Deleting the
    whole predictive module cannot separate a representation benefit from a rollout
    benefit; only a shared-encoder, rollout-free arm can.
  * M1 vs base - the same gain appearing for Q vs base, which places it in the
    deployment data rather than in the roll.
  * any milestone-4 or Phi' effect - not carrying into terminal success at a panel
    size that could detect it.  At the measured base rate it cannot, which is why
    the binary endpoint is registered as an ESTIMATION endpoint (see POWER).

PRE-REGISTERED, written to `preregistration.json` before the first env.reset():
the four arms and their checkpoint SHA-256, the panel (start and count), the
endpoint hierarchy, the exact comparison set, Holm over the registered families,
the seed-family guard, and the power arithmetic.  `--panel` and `--panel-start`
are required with no default.  CLAUDE.md rule 8: the seed count is committed
before anything is read; peeking at n = 96 and then deciding whether to extend is
how a null becomes a claim in this project (the same attribution went p = 0.1338
at 96 seeds, 0.1038 at 192, 0.2678 at 288).

POWER, printed and written into the preregistration BEFORE the run.  chain3's
frozen-policy terminal success is 2/96 = 0.021 (D0,
results/v121_deploy_latents/chain3_lr2_2026-08-30T083910Z).  Under the paired
exact McNemar a treated arm needs >= 5 discordant pairs all in its own direction
for one-sided p < 0.05 (7/96 against a 2/96 base, a 3.5x rate increase) and >= 6
for two-sided (8/96, 4.0x - the figure `power_block` computes, prints and writes
into the preregistration).  The binary endpoint at this n therefore cannot test
anything short of a very large effect and is reported as an interval only.  The
powered endpoints are the ones that vary inside the stall basin, where 63% of
every episode's budget is spent:

    E2  milestone-4 attainment  (pick_up cream_cheese_1; 5/96 under the frozen
        policy, so there is discordance to price)           binary, paired McNemar
    E3  contiguous stage reached  ({0:3, 3:2, 4:87, 5:2, 6:2} under the frozen
        policy)                                             ordinal, paired
    E4  last-progress step  (median 260 of 750)             continuous, paired
    E5  mean Phi' over the stall basin  (lcwm/v250_progress.py; the registered A1
        potential is flat at exactly 2.0 across the whole basin, which is why a
        fixed task-scale reference was needed)              continuous, paired

All endpoints are PAIRED by environment seed across arms, which is where the
power comes from - each arm sees the identical reset.

STATISTICS.  Every p-value and interval printed by this file is computed by this
file's own dependency-free implementation, so the reported number does not depend
on whether scipy is installed.  scipy 1.18.0 is present in `vf0s` and is used only
as an independent cross-check, recorded as `scipy_p` beside each own-computed p
and flagged when the two disagree.  Reused rather than re-implemented:
`clopper_pearson_lower/upper` from lcwm/v08r_contract.py:329-346 (exact, via a
dependency-free regularized incomplete beta) and the fixed-seed paired bootstrap
`cluster_bootstrap` from scripts/analyze_behavior_factorial.py:28.  No McNemar and
no Holm helper exists anywhere in the repo - the only prior Holm is four inline
lines at scripts/analyze_behavior_factorial.py:123-126 - so both are implemented
here.

WHAT THIS DRIVER REFUSES TO DO, each a defect measured in the existing loop
driver scripts/run_v208_deploy_loop.py:
  * it never resolves an arm's output directory by globbing for the newest match.
    v208 discards every subprocess return code (:101/:113/:122) and then takes
    `newest()`, so a crashed arm silently reports the previous run's directory.
    Here each arm's directory comes from that subprocess's own printed path, the
    return code aborts the driver, the directory name must carry this run's unique
    tag, and its summary.json must be newer than the subprocess start.
  * it never runs on a panel overlapping a collection family.  The guard is
    data-driven from the collection manifests passed with --collection-manifest
    plus a task-scoped registry of the families already spent on chain3.
  * it never runs an arm without the evaluation instrumentation, and it verifies
    afterwards that the arm ran with it.  Every arm is invoked with --eval-labels
    and --fixed-horizon.  Without --eval-labels the executor builds no automaton
    and writes events={}, stage_reached=0, last_progress_step=0 for every episode
    (run_v206_belief_residual_deploy.py:396-404); those are real numbers, not
    nulls, so this driver would have read them as measurements and reported the
    primary endpoint as 0/96 vs 0/96, p = 1.0 - a null manufactured by a missing
    flag.  Without --fixed-horizon ChainEnv returns terminated = done or
    is_success (lcwm/loho.py:70), so the surviving boundary count is a function of
    the outcome and every per-boundary average is biased by the arm's own success
    rate.
  * it never reports arms it has not checked are matched AFTER the run as well as
    before it.  The executor writes norm_digest (mu, sd, bmu, bsd), scale,
    condition, latent_dim, rich_latent, actor_params, zero_residual, eval_labels,
    fixed_horizon and the panel into each arm's summary.json and labels the first
    of those "arm-matching evidence: M1 and M0 must share these"
    (run_v206_belief_residual_deploy.py:601-613); `check_run_matched` reads it and
    aborts rather than reporting a contrast between arms that differ in the
    latent normaliser, the interface capacity or the executed panel.

SCOPE OF ANY RESULT.  Phi' is a declared scripted privileged stand-in for
per-boundary human video annotation (lcwm/v250_progress.py); it is identical
across arms, so it cannot explain a between-arm difference, but every effect
stated in terms of it is scoped to "under this supplied progress annotation".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from lcwm.v08r_contract import clopper_pearson_lower, clopper_pearson_upper
from analyze_behavior_factorial import BOOT_N, BOOT_SEED, cluster_bootstrap

try:                                             # cross-check only, never the source
    from scipy import stats as _scipy_stats
    HAVE_SCIPY = True
except Exception:                                # pragma: no cover - env-dependent
    _scipy_stats = None
    HAVE_SCIPY = False

PY_ = "/home/stargazer/miniconda3/envs/vf0s/bin/python"
DEPLOY = "scripts/run_v206_belief_residual_deploy.py"
OUT_ROOT = REPO / "results" / "v251_round"
SCHEMA_PREREG = "v251_round_prereg.1"
SCHEMA_SUMMARY = "v251_round_eval.1"

TASK = "chain3_lr2"
#: lcwm/v080_bench.py episode_length(chain3_lr2); held as a constant so this module
#: never imports the benchmark (and therefore never imports LIBERO).
HORIZON = 750
CHUNK = 10
#: lcwm/v080_bench.py ordered_subgoals for chain3_lr2, interleaved pick/place.
MILESTONES = ("pick_up alphabet_soup_1", "place alphabet_soup_1",
              "pick_up tomato_sauce_1", "place tomato_sauce_1",
              "pick_up cream_cheese_1", "place cream_cheese_1")
PRIMARY_MILESTONE = 4                            # pick_up cream_cheese_1
#: frozen-policy D0 record, results/v121_deploy_latents/chain3_lr2_2026-08-30T083910Z
BASE_PRIOR = (2, 96)
BASE_MILESTONE_PRIOR = (5, 96)
ALPHA = 0.05

# Fixed before the run, not adaptive.
#
# `rand` was added on 2026-09-12 after the basin calibration, BEFORE any evaluation
# panel was opened, and it is arguably the load-bearing control of the whole round:
# an UNTRAINED random residual at the round's scale already reached 7/24 episodes
# inside the 5 cm grasp radius, picked the third object up in 4/24 and produced 1/24
# terminal success, against 0/24 for frozen pi0.5 on the same seeds. Without this arm
# a trained interface that merely matches an untrained one at the same magnitude
# would be reported as an improvement over base, and the training would be credited
# with an effect that the trust region alone produces.
#
# `M1_shuf` is the action-conditioning control: same data, same budget, same head,
# but the updated model's D1 action pairing is broken within matched strata.
# `M0_cont` was added on 2026-09-12 after a pre-launch audit, BEFORE any evaluation
# panel was opened, and it fixes a confound in what had been the headline contrast.
# build_v241_model_pair trains `stale` for --base-steps (4000) and then builds
# `updated` as deepcopy(stale) plus a further --update-steps (3000). So M1 has had
# 7000 optimiser steps against M0's 4000 - 75% more - ON TOP OF seeing D1. "M1 beats
# M0" therefore confounds the updated dynamics with extra gradient, and the
# pre-registration's "they differ only in the learned transition" was false as built.
# `base_continue` is the arm that fixes it: same total optimiser budget as M1, still
# D0-only data. It already existed in the pair and was simply never deployed.
#
# So the PRIMARY dynamics contrast is M1 vs M0_cont (matched compute, differs in the
# new deployment data), and M1 vs M0 drops to context.
ARM_ORDER = ("base", "rand", "M0", "M0_cont", "M1", "M1_shuf", "Q")
PRIMARY_CONTRASTS = (("M1", "M0_cont"), ("M1", "Q"), ("M1", "rand"),
                     ("M1", "M1_shuf"), ("M1", "base"))
CONTEXT_CONTRASTS = (("M1", "M0"), ("M0_cont", "M0"), ("M0", "base"),
                     ("Q", "base"), ("rand", "base"), ("M0", "Q"))


# --------------------------------------------------------------------------
# endpoints
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Endpoint:
    key: str
    label: str
    kind: str          # "binary" | "ordinal" | "continuous"
    tier: str          # "estimation" | "primary" | "secondary"
    direction: int     # +1 if larger is the improvement direction
    prior: str
    caveat: str


ENDPOINTS = (
    Endpoint("success", "terminal success", "binary", "estimation", +1,
             "2/96 frozen policy (D0)",
             "an ESTIMATION endpoint at this n, not a test: see the power block. "
             "Reported as a Clopper-Pearson interval per arm and a discordant-pair "
             "table, never as a verdict."),
    Endpoint("milestone4", f"milestone-{PRIMARY_MILESTONE} attainment "
             f"({MILESTONES[PRIMARY_MILESTONE]})", "binary", "primary", +1,
             "5/96 frozen policy (D0)",
             "attaining the pick does not imply the place: milestone 5 was reached "
             "in 3/96. A milestone-4 gain that does not propagate is a gain in the "
             "approach, not in the task."),
    Endpoint("stage", "contiguous stage reached", "ordinal", "secondary", +1,
             "{0:3, 3:2, 4:87, 5:2, 6:2} frozen policy (D0)",
             "87/96 episodes sit on one value, so this endpoint is nearly constant "
             "and its Wilcoxon runs on few non-zero pairs."),
    Endpoint("last_progress_step", "last-progress step", "continuous", "secondary", +1,
             "median 260 of 750 frozen policy (D0)",
             "confounded with pace: an arm that reaches the SAME milestones more "
             "slowly scores higher. Read it only together with `stage`."),
    Endpoint("phi_basin_mean", "mean Phi' over the stall basin", "continuous",
             "secondary", +1, "flat 2.0 under the registered A1 potential",
             "Phi' is a scripted privileged stand-in for human video annotation "
             "(lcwm/v250_progress.py). It is identical across arms so it cannot "
             "explain a difference, but any effect stated in it is scoped to "
             "'under this supplied progress annotation'."),
    Endpoint("d_min_final_atom", "closest approach to the final goal object",
             "continuous", "secondary", -1, "median 0.502 m frozen policy (24 seeds)",
             "SIGN IS NEGATIVE: smaller is better. Counted only over boundaries where "
             "the final atom is the ACTIVE one - unrestricted, the same 24 base "
             "episodes read 9/24 within 5 cm instead of 0/24, because the minimum was "
             "the gripper beside a CAN during that can's placement. It is deliberately "
             "SECONDARY and must never be read as the outcome: on the calibration "
             "ladder it moved OPPOSITE to the outcome, with scale 1.60 giving the best "
             "approach of any arm (median 0.243 m, 20/24 seeds closer than base) while "
             "picking the object up half as often as scale 0.80 and never placing it. "
             "An arm can win this endpoint by driving the gripper at the object "
             "without ever closing on it."),
)
ENDPOINT_BY_KEY = {e.key: e for e in ENDPOINTS}

#: The basin rule, registered: boundaries strictly after the episode's last
#: milestone event. An episode whose progress ran to the horizon has no basin and
#: contributes its final boundary alone.
BASIN_RULE = ("boundaries with t > last_progress_step; if that set is empty the "
              "episode contributes its final boundary only")

OVERTURN = {
    ("M1", "M0"): "the same configuration retrained from a second initialisation "
                  "closing the gap (seed variance at fixed hyperparameters was ~0.24 "
                  "on 2026-09-01, larger than most between-arm differences), or the "
                  "difference not reproducing on a second environment panel",
    ("M1", "Q"): "Q matching M1 once Q shares M1's trained encoder; a rollout-free "
                 "arm that also lacks the representation cannot separate the two",
    ("M1", "base"): "the same gain appearing for Q vs base, which places it in the "
                    "deployment data rather than in the roll",
    ("M0", "base"): "M1 vs M0 being null, which would make this a data effect that "
                    "the WM update does not carry",
    ("Q", "base"): "nothing about the world model: this contrast is the model-free "
                   "reference and is reported as context only",
    ("M0", "Q"): "nothing about the WM update; it prices the stale model against no "
                 "model at all and is reported as context only",
}


# --------------------------------------------------------------------------
# seed families - the guard is task-scoped and data-driven
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SeedFamily:
    name: str
    lo: int                  # inclusive
    hi: int                  # exclusive
    role: str                # "collection" | "prior_eval" | "probe"
    provenance: str

    def overlaps(self, lo: int, hi: int) -> list[int]:
        return sorted(set(range(self.lo, self.hi)) & set(range(lo, hi)))


#: Families already spent on chain3_lr2. "collection" families can never be
#: evaluated on - a panel that overlaps one is refused with no override, because a
#: WM arm trained on those rows would be scored on its own training seeds.
KNOWN_FAMILIES = (
    SeedFamily("v121_chain3_tape_a", 8000, 8096, "collection",
               "results/v121_deploy_latents/chain3_lr2_2026-08-29T224421Z"),
    SeedFamily("v121_chain3_tape_b", 8100, 8196, "collection",
               "results/v121_deploy_latents/chain3_lr2_2026-08-29T232256Z"),
    SeedFamily("d0_rich_tape", 8700, 8796, "collection",
               "results/v121_deploy_latents/chain3_lr2_2026-08-30T083910Z; the "
               "round's D0"),
    SeedFamily("d1_residual_default", 8800, 8848, "collection",
               "scripts/collect_v250_chain3_round.py usage example; pass the real "
               "D1 manifest with --collection-manifest to register the actual range"),
    SeedFamily("historical_eval_panel", 8900, 8996, "prior_eval",
               "chain3 evaluation panel used before 2026-09-12"),
    SeedFamily("v249_residual_scale_probe", 9000, 9048, "probe",
               "scripts/probe_v249_chain3_residual_scale.py, 7 arms x 48 seeds"),
)


def families_from_manifest(path: Path) -> list[SeedFamily]:
    """Read seed ranges out of a collector manifest/summary/outcome JSON.

    `scripts/collect_v250_chain3_round.py` writes `seed_range: [first, last]` and
    `seed_start`/`episodes` into manifest.json; v121 summaries carry `seed_range`.
    Both forms are accepted; anything else raises rather than silently registering
    no family, because a guard that quietly protects nothing is worse than none.
    """
    doc = json.loads(Path(path).read_text())
    if "seed_range" in doc:
        lo, last = int(doc["seed_range"][0]), int(doc["seed_range"][1])
        hi = last + 1
    elif "seed_start" in doc and "episodes" in doc:
        lo = int(doc["seed_start"])
        hi = lo + int(doc["episodes"])
    else:
        raise SystemExit(f"{path}: no seed_range and no seed_start/episodes; this "
                         f"manifest cannot register a seed family")
    role = doc.get("role", "collection")
    return [SeedFamily(f"manifest:{Path(path).parent.name}:{role}", lo, hi,
                       "collection", str(path))]


def check_panel(panel_lo: int, panel_n: int, families,
                acknowledge_reuse: bool = False) -> dict:
    """Refuse an evaluation panel that touches a spent seed family."""
    panel_hi = panel_lo + panel_n
    hits = []
    for fam in families:
        overlap = fam.overlaps(panel_lo, panel_hi)
        if overlap:
            hits.append({"family": fam.name, "role": fam.role, "range": [fam.lo, fam.hi],
                         "provenance": fam.provenance, "n_overlap": len(overlap),
                         "first": overlap[0], "last": overlap[-1]})
    blocking = [h for h in hits if h["role"] == "collection" or not acknowledge_reuse]
    if blocking:
        lines = [f"panel {panel_lo}-{panel_hi - 1} (n={panel_n}) overlaps spent seeds:"]
        for h in blocking:
            lines.append(f"  {h['family']} [{h['role']}] {h['range'][0]}-"
                         f"{h['range'][1] - 1}: {h['n_overlap']} seeds "
                         f"({h['first']}..{h['last']})  {h['provenance']}")
        if all(h["role"] != "collection" for h in blocking):
            lines.append("  a non-collection overlap can be declared with "
                         "--acknowledge-panel-reuse; a collection overlap cannot")
        else:
            lines.append("  a collection overlap has no override: the arms were "
                         "trained on those seeds")
        raise SystemExit("\n".join(lines))
    return {"panel": [panel_lo, panel_hi - 1, panel_n],
            "checked_families": [{"name": f.name, "role": f.role,
                                  "range": [f.lo, f.hi - 1],
                                  "provenance": f.provenance} for f in families],
            "declared_reuse": hits, "clean": not hits}


# --------------------------------------------------------------------------
# statistics - own implementations; scipy only cross-checks
# --------------------------------------------------------------------------
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF by bisection on `norm_cdf`; ~1e-12 absolute."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"norm_ppf domain: {p}")
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def clopper_pearson(k: int, n: int, alpha_per_side: float = 0.025) -> tuple[float, float]:
    """Exact two-sided interval. alpha is PER SIDE: 0.025 gives a 95% interval.

    CLAUDE.md records that `clopper_pearson_lower` defaulted to 0.05, so every
    `cp95` field written before 2026-08-29 is really a 90% interval. This wrapper
    passes 0.025 explicitly on both sides so the label matches the arithmetic.
    """
    return (clopper_pearson_lower(k, n, alpha_per_side),
            clopper_pearson_upper(k, n, alpha_per_side))


def binom_tail_ge(k: int, n: int) -> float:
    """P(X >= k) for X ~ Binomial(n, 1/2), exact in integer arithmetic."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return sum(math.comb(n, i) for i in range(k, n + 1)) / (2.0 ** n)


def exact_mcnemar(b: int, c: int) -> dict:
    """Exact paired binary test on the discordant pairs alone.

    `b` = seeds where arm A succeeded and arm B did not, `c` = the reverse. Under
    H0 each discordant pair is a fair coin, so the two-sided p is
    `min(1, 2 * P(X <= min(b, c)))` and the one-sided p in A's direction is
    `P(X >= b)`. The discordant table is returned with the p-value: at these base
    rates the table is the result and the p-value is a summary of it.
    """
    if b < 0 or c < 0:
        raise ValueError(f"discordant counts must be >= 0: {b=} {c=}")
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "n_discordant": 0, "p_two_sided": 1.0,
                "p_one_sided_a_greater": 1.0, "method": "exact binomial",
                "note": "no discordant pairs: the arms differ on no seed"}
    p_one = binom_tail_ge(b, n)
    p_two = min(1.0, 2.0 * (1.0 - binom_tail_ge(min(b, c) + 1, n)))
    out = {"b": b, "c": c, "n_discordant": n, "p_two_sided": p_two,
           "p_one_sided_a_greater": p_one, "method": "exact binomial"}
    if HAVE_SCIPY:
        out["scipy_p_two_sided"] = float(
            _scipy_stats.binomtest(min(b, c), n, 0.5).pvalue)
        out["scipy_agrees"] = abs(out["scipy_p_two_sided"] - p_two) < 1e-9
    return out


def fisher_one_sided_greater(k1: int, n1: int, k2: int, n2: int) -> float:
    """Unpaired Fisher exact, P(arm-1 count >= k1 | margins). Context only."""
    total, succ = n1 + n2, k1 + k2
    denom = math.comb(total, n1)
    hi = min(n1, succ)
    return sum(math.comb(succ, i) * math.comb(total - succ, n1 - i)
               for i in range(k1, hi + 1)) / denom


def _signed_rank_stat(diffs) -> tuple[float, int, list[float], float]:
    """Sum of ranks of positive differences, with average ranks over ties."""
    nz = [d for d in diffs if d != 0.0]
    order = sorted(range(len(nz)), key=lambda i: abs(nz[i]))
    ranks = [0.0] * len(nz)
    i = 0
    tie_term = 0.0
    while i < len(order):
        j = i
        while j + 1 < len(order) and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        t = j - i + 1
        tie_term += t ** 3 - t
        i = j + 1
    t_plus = sum(r for r, d in zip(ranks, nz) if d > 0)
    return t_plus, len(nz), ranks, tie_term


def _exact_signed_rank_p(t_plus: float, n: int) -> float:
    """Two-sided exact p from the null distribution of T+ over 2^n sign patterns.

    Counted by DP over subset sums of {1..n} rather than enumeration, so n = 50
    costs ~64k integer additions.
    """
    total = n * (n + 1) // 2
    counts = [0] * (total + 1)
    counts[0] = 1
    for r in range(1, n + 1):
        for s in range(total, r - 1, -1):
            counts[s] += counts[s - r]
    denom = float(2 ** n)
    t_int = int(round(t_plus))
    p_le = sum(counts[: t_int + 1]) / denom
    p_ge = sum(counts[t_int:]) / denom
    return min(1.0, 2.0 * min(p_le, p_ge))


def wilcoxon_signed_rank(diffs, exact_max_n: int = 50) -> dict:
    """Two-sided Wilcoxon signed-rank on paired differences.

    Zeros are discarded (scipy's `zero_method="wilcox"`). Exact when there are no
    ties and no zeros and n <= `exact_max_n`; otherwise the normal approximation
    with the standard tie correction and no continuity correction, matching
    scipy's defaults so the recorded cross-check is a real check.
    """
    t_plus, n, ranks, tie_term = _signed_rank_stat(diffs)
    if n == 0:
        return {"n_pairs": len(diffs), "n_nonzero": 0, "t_plus": 0.0,
                "p_two_sided": 1.0, "method": "degenerate: every pair is a tie"}
    ties = tie_term > 0
    if not ties and n <= exact_max_n:
        p = _exact_signed_rank_p(t_plus, n)
        method = f"exact, {n} non-zero pairs"
    else:
        mu = n * (n + 1) / 4.0
        var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
        z = 0.0 if var <= 0 else (t_plus - mu) / math.sqrt(var)
        # erfc, not 1 - Phi: at n = 96 an all-positive difference vector gives
        # z ~ 8.5, where 1 - Phi underflows to exactly 0.0 and the report would
        # print an impossible "p = 0.0000". erfc carries it to ~1e-17.
        p = min(1.0, math.erfc(abs(z) / math.sqrt(2.0)))
        method = ("normal approximation with tie correction"
                  + (f", {n} non-zero pairs" if n > exact_max_n else ""))
    out = {"n_pairs": len(diffs), "n_nonzero": n, "t_plus": float(t_plus),
           "p_two_sided": float(p), "method": method, "n_zero_pairs": len(diffs) - n}
    if HAVE_SCIPY and n > 0:
        try:
            r = _scipy_stats.wilcoxon([d for d in diffs if d != 0.0],
                                      zero_method="wilcox", alternative="two-sided")
            out["scipy_p_two_sided"] = float(r.pvalue)
            out["scipy_agrees"] = abs(float(r.pvalue) - p) < 1e-6
        except Exception as exc:                 # pragma: no cover - scipy edge cases
            out["scipy_error"] = repr(exc)
    return out


def paired_bootstrap(diffs) -> dict:
    """Fixed-seed percentile bootstrap CI of the mean paired difference.

    `cluster_bootstrap` is the repo's existing implementation
    (scripts/analyze_behavior_factorial.py:28, seed 20260729, 10,000 resamples);
    it is reused rather than duplicated so two analyses in this repo cannot report
    two different bootstraps under one name.
    """
    if not diffs:
        return {"mean": 0.0, "ci95": [0.0, 0.0], "n": 0, "resamples": 0}
    lo, hi = cluster_bootstrap(list(diffs))
    return {"mean": sum(diffs) / len(diffs), "ci95": [lo, hi], "n": len(diffs),
            "resamples": BOOT_N, "seed": BOOT_SEED,
            "source": "scripts/analyze_behavior_factorial.py:cluster_bootstrap"}


def holm(pvals: dict, alpha: float = ALPHA, family_size: int | None = None) -> dict:
    """Holm step-down over a registered family.

    Returns, per test, its rank, the Holm-adjusted alpha it is compared against,
    the monotone Holm-adjusted p-value, and the reject flag. Ties in p are ordered
    by name so the result is deterministic.

    `family_size` is the size of the REGISTERED family and defaults to the number
    of tests supplied. It may not be smaller. The driver passes the registered
    size: when an endpoint turns out to be unavailable in the sidecar, correcting
    over only the tests that happened to run would make the correction weaker
    than the one written into the preregistration.
    """
    items = sorted(pvals.items(), key=lambda kv: (kv[1], kv[0]))
    m = len(items) if family_size is None else int(family_size)
    if m < len(items):
        raise ValueError(f"family_size {m} is smaller than the {len(items)} tests "
                         f"supplied")
    out, running, stopped = {}, 0.0, False
    for i, (name, p) in enumerate(items):
        adj_alpha = alpha / (m - i)
        running = max(running, min(1.0, (m - i) * p))
        if not stopped and p > adj_alpha:
            stopped = True
        out[name] = {"p": p, "rank": i + 1, "family_size": m,
                     "n_tests_run": len(items),
                     "alpha_adjusted": adj_alpha, "p_adjusted": running,
                     "reject": (not stopped) and p <= adj_alpha}
    return out


# --------------------------------------------------------------------------
# power, computed from the pre-registered n before anything is read
# --------------------------------------------------------------------------
def mcnemar_min_discordant(alpha: float = ALPHA, two_sided: bool = True) -> int:
    """Smallest all-in-one-direction discordant count reaching p < alpha."""
    for d in range(1, 200):
        p = 2.0 * 0.5 ** d if two_sided else 0.5 ** d
        if p < alpha:
            return d
    raise RuntimeError("unreachable for any sane alpha")


def min_detectable_paired_effect(n: int, alpha: float = ALPHA,
                                 power: float = 0.80) -> dict:
    """Standardised paired difference detectable at n, normal approximation."""
    z_a = norm_ppf(1.0 - alpha / 2.0)
    z_b = norm_ppf(power)
    d = (z_a + z_b) / math.sqrt(n)
    return {"n": n, "alpha_two_sided": alpha, "power": power,
            "cohen_dz": d, "wilcoxon_dz": d / math.sqrt(0.955),
            "note": "d_z is in units of the SD of the PAIRED difference; the "
                    "Wilcoxon figure applies its 0.955 asymptotic relative "
                    "efficiency versus the paired t under normal differences"}


def power_block(n: int, base_k: int = BASE_PRIOR[0],
                base_n: int = BASE_PRIOR[1]) -> dict:
    """Everything the report needs to not be read later as a test it was not."""
    rate = base_k / base_n
    d1 = mcnemar_min_discordant(ALPHA, two_sided=False)
    d2 = mcnemar_min_discordant(ALPHA, two_sided=True)
    expected_base_k = int(round(rate * n))
    k_one = expected_base_k + d1
    k_two = expected_base_k + d2
    k_fisher = None
    for k in range(expected_base_k, n + 1):
        if fisher_one_sided_greater(k, n, expected_base_k, n) < ALPHA:
            k_fisher = k
            break
    lo, hi = clopper_pearson(expected_base_k, n)
    estimation_only = k_two / max(n, 1) >= 3.0 * rate
    return {
        "binary_endpoint": {
            "base_rate_prior": {"k": base_k, "n": base_n, "rate": rate,
                                "source": "results/v121_deploy_latents/"
                                          "chain3_lr2_2026-08-30T083910Z"},
            "expected_base_successes_at_n": expected_base_k,
            "base_cp95_at_n": [lo, hi],
            "min_discordant_one_sided": d1,
            "min_discordant_two_sided": d2,
            "min_treated_count_one_sided": k_one,
            "min_treated_count_two_sided": k_two,
            "min_treated_rate_two_sided": k_two / n,
            "rate_multiple_two_sided": (k_two / n) / rate if rate else None,
            "min_treated_count_unpaired_fisher_one_sided": k_fisher,
            "is_estimation_endpoint_only": estimation_only,
            "statement": (
                f"at n={n} and a base rate of {base_k}/{base_n}={rate:.3f}, the "
                f"smallest treated-arm result reaching two-sided exact McNemar "
                f"p<{ALPHA} is {k_two}/{n}={k_two / n:.3f}, a "
                f"{(k_two / n) / rate:.1f}x rate increase (one-sided: {k_one}/{n}). "
                f"The binary terminal-success endpoint at this n CANNOT DETECT "
                f"anything short of a very large effect and is registered as an "
                f"ESTIMATION endpoint: it is reported with a Clopper-Pearson "
                f"interval and a discordant-pair table, and no reading of this "
                f"report may treat it as a test."),
        },
        "ordinal_and_continuous_endpoints": min_detectable_paired_effect(n),
        "ordinal_and_continuous_at_holm_first_step": min_detectable_paired_effect(
            n, alpha=ALPHA / len(PRIMARY_CONTRASTS)),
        "milestone4_prior": {"k": BASE_MILESTONE_PRIOR[0], "n": BASE_MILESTONE_PRIOR[1],
                             "rate": BASE_MILESTONE_PRIOR[0] / BASE_MILESTONE_PRIOR[1],
                             "note": "5/96 under the frozen policy, so there is "
                                     "discordance available for the paired test that "
                                     "terminal success does not offer"},
    }


# --------------------------------------------------------------------------
# checkpoints
# --------------------------------------------------------------------------
def sha256_file(path: Path) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


#: `epochs` and `beta` were in this tuple and are written by nothing, so they were
#: silently unverifiable - the check passed by comparing None to None. The fields that
#: actually carry the round's "one shared head, one shared pair" guarantee are added.
MATCHED_FIELDS = ("dims", "zdim", "scale", "condition", "use_action", "task",
                  "horizon", "actor_epochs", "batch", "advantage",
                  "phi_head_digest", "model_pair_sha256", "base_action",
                  "phi_readout", "include_zero_delta", "phi_region_from_chunk")
#: A difference in these is a different function, not a different training run.
STRUCTURAL_FIELDS = ("dims", "zdim", "scale", "condition", "phi_head_digest",
                     "model_pair_sha256")


def checkpoint_fingerprint(path: Path) -> dict:
    """CPU-only structural read of an actor checkpoint. No GPU, no simulator."""
    import torch                                  # local: keep the test import light
    ck = torch.load(path, map_location="cpu", weights_only=False)
    fields = {}
    for k in MATCHED_FIELDS:
        if k in ck:
            v = ck[k]
            fields[k] = list(v) if isinstance(v, (list, tuple)) else v
    sd = ck.get("state_dict", {})
    shapes = {k: list(v.shape) for k, v in sd.items()}
    n_params = int(sum(int(v.numel()) for v in sd.values()))
    wm_key = next((k for k in ("rawwm", "model", "wm") if k in ck), None)
    wm_sha = None
    if wm_key is not None and isinstance(ck[wm_key], dict):
        h = hashlib.sha256()
        for k in sorted(ck[wm_key]):
            t = ck[wm_key][k]
            h.update(k.encode())
            if torch.is_tensor(t):
                h.update(str(tuple(t.shape)).encode())
                h.update(t.detach().cpu().float().contiguous().numpy().tobytes())
            else:                                # a bookkeeping entry, not a weight
                h.update(repr(t).encode())
        wm_sha = h.hexdigest()
    del ck
    return {"fields": fields, "actor_param_shapes": shapes,
            "actor_n_params": n_params, "transition_key": wm_key,
            "transition_sha256": wm_sha}


#: `rand` is UNTRAINED by construction, so it cannot carry a training provenance -
#: no phi head, no model pair, no advantage, no optimiser budget. Comparing those
#: fields against the trained arms is not a matching check, it is a category error,
#: and it would abort the round for the arm's defining property. It IS compared on
#: everything that makes it the same kind of function in the same space: dims, zdim,
#: scale, condition, task and its parameter shapes. The exemption is recorded in the
#: result rather than applied silently.
TRAINING_PROVENANCE_FIELDS = ("horizon", "actor_epochs", "batch", "advantage",
                              "phi_head_digest", "model_pair_sha256", "base_action",
                              "phi_readout", "include_zero_delta",
                              "phi_region_from_chunk")
UNTRAINED_ARMS = ("rand",)


def check_matched(fps: dict) -> dict:
    """Every treated arm must differ only in what its contrast claims to isolate.

    A field absent from EVERY checkpoint compares equal to itself and so passes
    silently; it is listed in `not_verifiable` instead, because "no registered
    field differs" and "the checkpoints do not record the field" are different
    statements and only the first is evidence.

    `rand` is exempted from the training-provenance fields (see
    TRAINING_PROVENANCE_FIELDS) because it has no training to record; it is still
    compared on every structural field. The exemption is reported.
    """
    names = [n for n in ARM_ORDER if n != "base" and n in fps]
    trained = [n for n in names if n not in UNTRAINED_ARMS]
    diffs, hard, absent = [], [], []
    for fld in MATCHED_FIELDS:
        scope = trained if fld in TRAINING_PROVENANCE_FIELDS else names
        vals = {n: fps[n]["fields"].get(fld, "<absent>") for n in scope}
        if not vals or all(v == "<absent>" for v in vals.values()):
            absent.append(fld)
            continue
        if len({json.dumps(v, sort_keys=True, default=str) for v in vals.values()}) > 1:
            diffs.append({"field": fld, "values": vals,
                          "structural": fld in STRUCTURAL_FIELDS})
            if fld in STRUCTURAL_FIELDS:
                hard.append(fld)
    shapes = {n: json.dumps(fps[n]["actor_param_shapes"], sort_keys=True)
              for n in names}          # structural: applies to rand too
    if len(set(shapes.values())) > 1:
        diffs.append({"field": "actor_param_shapes",
                      "values": {n: fps[n]["actor_n_params"] for n in names},
                      "structural": True})
        hard.append("actor_param_shapes")
    same_transition = None
    comparison = "M1 or M0 absent"
    if "M1" in fps and "M0" in fps:
        s1, s0 = fps["M1"]["transition_sha256"], fps["M0"]["transition_sha256"]
        if s1 is None and s0 is None:
            comparison = ("neither checkpoint carries a transition tensor block, "
                          "so the round's only registered difference is NOT "
                          "verifiable from the checkpoints")
            absent.append("transition_sha256")
        elif s1 is None or s0 is None:
            comparison = "one checkpoint carries a transition and the other does not"
            hard.append("transition_sha256")
            diffs.append({"field": "transition_sha256",
                          "values": {"M1": s1, "M0": s0}, "structural": True,
                          "note": "the two arms would take different executor code "
                                  "paths (run_v206 picks rawwm / model / raw from "
                                  "the checkpoint's own keys)"})
        elif s1 == s0:
            same_transition = True
            comparison = "identical"
            hard.append("transition_sha256")
            diffs.append({"field": "transition_sha256",
                          "values": {"M1": s1, "M0": s0}, "structural": True,
                          "note": "M1 and M0 carry the SAME transition weights: the "
                                  "round's only registered difference is absent"})
        else:
            same_transition = False
            comparison = "differ"
    return {"compared": names, "trained_compared": trained,
            "untrained_exempt": {a: list(TRAINING_PROVENANCE_FIELDS)
                                 for a in UNTRAINED_ARMS if a in fps},
            "differences": diffs, "hard_failures": sorted(set(hard)),
            "m1_m0_transition_identical": same_transition,
            "m1_m0_transition_comparison": comparison,
            "not_verifiable": sorted(set(absent)),
            "rule": "nothing may differ between M1 and M0 except the learned "
                    "transition"}


# --------------------------------------------------------------------------
# arm execution
# --------------------------------------------------------------------------
class ArmFailed(RuntimeError):
    """A deployment subprocess failed, or its output directory is unresolvable."""


def run_logged(cmd, log_path: Path, env: dict) -> tuple[int, str]:
    """Run `cmd`, tee stdout+stderr to `log_path`, return (returncode, output)."""
    chunks = []
    with open(log_path, "w") as fh:
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            fh.write(line)
            fh.flush()
            sys.stdout.write(line)
            chunks.append(line)
        rc = proc.wait()
    return rc, "".join(chunks)


def resolve_out_dir(output: str, expect_tag: str, started_at: float) -> Path:
    """Take the arm's directory from the subprocess's OWN reported path.

    Never globs for the newest directory. run_v208_deploy_loop.py resolves arms
    with `newest()` after discarding the return code (:101/:113/:122), so a
    crashed arm silently reports the previous run's numbers. Here the path must be
    printed by this subprocess, must contain summary.json, must carry this run's
    unique tag in its name, and must have been written after the subprocess
    started.
    """
    cands = []
    for raw in output.splitlines():
        line = raw.strip()
        if line.startswith("OUT_DIR="):
            cands.append(line.split("=", 1)[1].strip())
        elif "->" in line:
            cands.append(line.rsplit("->", 1)[1].strip())
    rejected = []
    for cand in reversed(cands):
        p = Path(cand)
        if not p.is_absolute():
            p = REPO / p
        if not p.is_dir():
            rejected.append(f"{p}: not a directory")
            continue
        summary = p / "summary.json"
        if not summary.exists():
            rejected.append(f"{p}: no summary.json")
            continue
        if expect_tag not in p.name:
            rejected.append(f"{p}: name does not carry this run's tag {expect_tag!r}")
            continue
        if summary.stat().st_mtime < started_at - 1.0:
            rejected.append(f"{p}: summary.json predates the subprocess start")
            continue
        return p
    raise ArmFailed("could not resolve the arm's output directory from its own "
                    "stdout (no globbing is performed). candidates: "
                    + json.dumps(cands) + " rejected: " + json.dumps(rejected))


def arm_command(arm: str, actor: Path, panel: int, panel_start: int, tag: str,
                task: str = TASK, residual_start_chunk: int = 0) -> list[str]:
    """The executor invocation for one arm. Identical except `--zero-residual`.

    `--eval-labels` and `--fixed-horizon` go to EVERY arm and are not optional:

      * without --eval-labels the executor builds no GoalAutomaton, and every
        episode record carries events={}, stage_reached=0, last_progress_step=0
        (run_v206_belief_residual_deploy.py:396-404). Those are values, not
        nulls: this driver would read them as measurements and report the
        registered primary endpoint as 0/96 vs 0/96, p = 1.0, on every contrast.
      * without --fixed-horizon ChainEnv returns terminated = done or is_success
        (lcwm/loho.py:70), so an episode stops at success and the surviving
        boundary count is a function of the outcome. Phi' over the stall basin
        and the boundary counts it averages would then differ between arms
        BECAUSE the arms differ in success, which is the one thing a matched
        comparison may not let happen. The executor's own docstring says to set
        it on all arms or none.
    """
    cmd = [PY_, "-u", DEPLOY, "--actor", str(actor), "--task", task,
           "--panel", str(panel), "--panel-start", str(panel_start), "--tag", tag,
           "--eval-labels", "--fixed-horizon",
           # The SAME schedule every arm, including base. The interface is trained
           # only on boundaries at or after the collector's --residual-start-chunk;
           # deploying it earlier runs it at states it never saw, and the basin
           # calibration measured that an ungated sustained residual is what damages
           # the first two sub-tasks (24/24 intact when gated). Applied identically
           # to all arms, so it cannot separate them.
           "--residual-start-chunk", str(residual_start_chunk)]
    if arm == "base":
        # CODE-PATH control: Delta := 0 with everything else identical. CLAUDE.md
        # rule 5 - the reference number must come from THIS pipeline. The 2/96
        # figure comes from the v121 collector, not from this executor, so it is a
        # prior, never the baseline this round is judged against.
        cmd.append("--zero-residual")
    return cmd


def execute_arm(arm: str, cmd: list[str], log_path: Path, env: dict,
                expect_tag: str) -> Path:
    started = time.time()
    rc, output = run_logged(cmd, log_path, env)
    if rc != 0:
        raise ArmFailed(f"arm {arm!r} exited {rc}; the driver aborts rather than "
                        f"reporting three of four arms. log: {log_path}")
    return resolve_out_dir(output, expect_tag, started)


# --------------------------------------------------------------------------
# endpoint extraction from the deploy sidecar
# --------------------------------------------------------------------------
#: Files searched, in order, for per-episode endpoint records. The deploy script
#: writes the progress sidecar; summary.json is the fallback and carries only the
#: binary endpoint.
SIDECAR_FILES = ("progress.json", "sidecar.json", "endpoints.json",
                 "episode_progress.json", "outcome.json", "summary.json")
RECORD_LISTS = ("episodes", "episode_records", "records", "rows")
KEY_ALIASES = {
    "seed": ("seed", "env_seed"),
    "success": ("success", "terminal_success", "is_success"),
    "milestone4": ("milestone4", "milestone_4", "m4", "reached_milestone_4",
                   "pick_cheese"),
    "stage": ("stage", "contiguous_stage", "stage_reached"),
    "last_progress_step": ("last_progress_step", "last_progress",
                           "last_milestone_step"),
    "phi_basin_mean": ("phi_basin_mean", "mean_phi_basin", "basin_phi_mean",
                       "phi_basin"),
    "d_min_final_atom": ("d_min_final_atom", "d_min", "closest_approach_final_atom"),
    "events": ("events", "milestones_achieved", "events_achieved"),
    "phi": ("phi", "phi_series", "phi_prime"),
    "phi_t": ("phi_t", "t", "boundary_t"),
}


def _pick(rec: dict, key: str):
    for alias in KEY_ALIASES[key]:
        if alias in rec:
            return rec[alias]
    return None


def derive_from_events(events) -> dict:
    """milestone4 / contiguous stage / last-progress step from an events map.

    `events` is the GoalAutomaton's `events_achieved`: milestone index -> the env
    step at which it first became true (lcwm/task_automaton.py:88). The contiguous
    stage is the largest S with milestones 0..S-1 all achieved - the quantity whose
    frozen-policy distribution is {0:3, 3:2, 4:87, 5:2, 6:2}, NOT the count of
    achieved milestones.
    """
    try:
        idx = {int(k): int(v) for k, v in dict(events).items()}
    except (TypeError, ValueError) as exc:
        raise ArmFailed(f"events={events!r} is not a milestone-index -> step map "
                        f"({exc}); the stage and the last-progress step would be "
                        f"read off something else") from exc
    stage = 0
    while stage in idx:
        stage += 1
    return {"milestone4": float(PRIMARY_MILESTONE in idx),
            "stage": float(stage),
            "last_progress_step": float(max(idx.values())) if idx else 0.0}


#: Boundaries in one full-horizon episode: t in {0, CHUNK, ..., HORIZON}, i.e.
#: 75 chunk boundaries plus the terminal one. The round's collector and the
#: executor under --fixed-horizon both write exactly this many per episode.
N_BOUNDARIES = HORIZON // CHUNK + 1


def derive_phi_basin(rec: dict, last_progress_step):
    """Mean Phi' over the stall basin, from a scalar or a per-boundary series.

    BASIN_RULE is a predicate on the boundary time t. A phi series whose t is
    neither carried nor reconstructable is therefore NOT reduced to some other
    statistic and returned under this endpoint's name - it returns None and the
    endpoint is reported unavailable, which is the file's stated policy. The one
    reconstruction permitted is the fixed-horizon grid t = i * CHUNK, and only
    when the series is exactly N_BOUNDARIES long, the single case in which that
    grid is the series' own time base.
    """
    scalar = _pick(rec, "phi_basin_mean")
    if scalar is not None:
        return float(scalar)
    phi = _pick(rec, "phi")
    if phi is None:
        return None
    phi = [float(x) for x in phi]
    if not phi:
        return None
    ts = _pick(rec, "phi_t")
    if not isinstance(ts, (list, tuple)):        # a scalar "t" is an episode length
        ts = None
    if ts is None:
        if len(phi) != N_BOUNDARIES:
            return None
        ts = [float(i * CHUNK) for i in range(len(phi))]
    else:
        ts = [float(x) for x in ts]
    if len(ts) != len(phi):
        raise ArmFailed(f"phi has {len(phi)} boundaries and its t has {len(ts)}; "
                        f"zip would silently drop the tail of the longer one")
    if last_progress_step is None:
        return None
    basin = [q for q, t in zip(phi, ts) if t > float(last_progress_step)]
    return sum(basin) / len(basin) if basin else phi[-1]


ROW_KEYS = ("success", "milestone4", "stage", "last_progress_step",
            "phi_basin_mean", "d_min_final_atom")
#: every raw key any alias names. The merge copies these and nothing else, so a
#: sidecar episode carrying per-boundary label tensors is not pulled into memory.
WANTED_KEYS = tuple(sorted({a for al in KEY_ALIASES.values() for a in al}))


def _records_in(doc):
    """The per-episode list inside a sidecar document, whatever it is called."""
    if isinstance(doc, list):
        return doc if doc and isinstance(doc[0], dict) else []
    for key in RECORD_LISTS:
        val = doc.get(key)
        if isinstance(val, list) and val and isinstance(val[0], dict):
            return val
    return []


def _as_binary(val, field: str) -> float:
    """Strict 0/1 coercion for a binary endpoint.

    `float(bool(x))` reads the JSON string "False" as 1.0 and every non-empty
    string as a success. A binary endpoint is either a bool or 0/1 here, or the
    arm is not read at all.
    """
    if isinstance(val, bool):
        return float(val)
    if isinstance(val, (int, float)):
        f = float(val)
        if f in (0.0, 1.0):
            return f
        raise ArmFailed(f"{field}={val!r} is neither 0 nor 1")
    raise ArmFailed(f"{field}={val!r} is a {type(val).__name__}, not a binary "
                    f"endpoint value; it is not coerced silently")


def _row_from(rec: dict) -> dict:
    events = _pick(rec, "events")
    derived = derive_from_events(events) if events is not None else {}
    row = {}
    succ = _pick(rec, "success")
    row["success"] = _as_binary(succ, "success") if succ is not None else None
    for key in ("milestone4", "stage", "last_progress_step"):
        val = _pick(rec, key)
        if val is None:
            val = derived.get(key)
        if val is None:
            row[key] = None
        elif key == "milestone4":
            row[key] = _as_binary(val, key)
        else:
            row[key] = float(val)
    row["phi_basin_mean"] = derive_phi_basin(rec, row["last_progress_step"])
    # Read straight through: the deploy executor computes it, restricted to
    # boundaries where the FINAL goal atom is the active one
    # (run_v206_belief_residual_deploy.final_atom_reach). It is deliberately NOT
    # reconstructed here from a raw distance series - an unrestricted minimum is a
    # different quantity that reads 9/24 where the restricted one reads 0/24.
    d_min = _pick(rec, "d_min_final_atom")
    row["d_min_final_atom"] = None if d_min is None else float(d_min)
    return row


def read_arm_endpoints(arm_dir: Path, sidecar: str | None = None) -> dict:
    """Per-seed endpoint vectors for one arm, plus what could not be found.

    The files in `SIDECAR_FILES` are merged by seed, first file winning per raw
    key, because the progress sidecar and summary.json carry different halves:
    the deploy script's summary.json has the terminal outcome per seed, the
    progress sidecar has the per-boundary quantities. Taking only one of them
    would silently drop an endpoint the run actually measured.

    The merge happens on the RAW records, BEFORE any endpoint is derived. Merging
    derived rows instead computes each file's endpoints from that file alone, and
    the basin rule is a predicate on another field: a progress.json carrying
    phi/t and a summary.json carrying events would put every boundary with t > 0
    in the basin instead of the boundaries after the last milestone, and report
    the whole-episode mean of Phi' under the basin endpoint's name. Only the keys
    the aliases name are copied, so a sidecar that also stores per-boundary label
    tensors per episode does not get pulled in wholesale.

    The minimum a sidecar must carry is, per episode, `seed` and `events` - which
    is exactly what `scripts/collect_v250_chain3_round.py` already stores per
    episode. `phi_basin_mean` (or a per-boundary `phi` series with its `t`) is
    additionally required for E5. A missing endpoint is REPORTED as unavailable,
    never silently dropped and never imputed.
    """
    arm_dir = Path(arm_dir)
    names = (sidecar, "summary.json") if sidecar else SIDECAR_FILES
    raw: dict[int, dict] = {}
    sources = []
    for name in names:
        path = arm_dir / name
        if not path.exists():
            continue
        records = _records_in(json.loads(path.read_text()))
        used, seen = False, set()
        for rec in records:
            seed = _pick(rec, "seed")
            if seed is None:
                continue
            seed = int(seed)
            if seed in seen:
                raise ArmFailed(f"{path}: seed {seed} appears in more than one "
                                f"record; the per-seed join would keep one and "
                                f"drop the other, and the arm would be paired on "
                                f"fewer seeds than it ran")
            seen.add(seed)
            target = raw.setdefault(seed, {})
            for key in WANTED_KEYS:
                if key in rec and rec[key] is not None and key not in target:
                    target[key] = rec[key]
                    used = True
        if used:
            sources.append(str(path))
    if not raw:
        raise ArmFailed(f"{arm_dir}: no per-episode records with a seed found in "
                        f"any of {list(names)}; the driver will not report an arm "
                        f"it cannot read")
    merged = {seed: _row_from(rec) for seed, rec in raw.items()}
    missing = [k for k in ROW_KEYS
               if any(row[k] is None for row in merged.values())]
    return {"dir": str(arm_dir), "source": sources, "n": len(merged),
            "per_seed": merged, "unavailable": missing}


# --------------------------------------------------------------------------
# what the arms actually RAN with, read back from the executor's own record
# --------------------------------------------------------------------------
#: Fields run_v206_belief_residual_deploy.py:601-613 writes into every arm's
#: summary.json. They must agree across the treated arms: the interface capacity,
#: the space it reads and the instrumentation the endpoints are read out of are
#: all things the round holds fixed.
RUN_MATCHED_FIELDS = ("scale", "condition", "latent_dim", "rich_latent",
                      "use_action", "actor_params", "fixed_horizon", "eval_labels",
                      "residual_start_chunk")
#: The REALISED residual magnitude, which `scale` does NOT pin. `scale` is the bound;
#: what the arm actually executed is `mean_abs_delta`. The 2026-09-12 calibration
#: measured |delta| = 0.051/0.105/0.206 producing zero milestone-4 events and 0.440
#: producing 4/24 - magnitude, not policy content, moved every endpoint on that ladder.
#: Two arms can therefore share a bound, share every flag, and still differ in the one
#: quantity known to drive the result. It is reported per arm and checked against a
#: pre-registered tolerance rather than assumed.
REALISED_DELTA_TOLERANCE = 0.25   # relative, treated arms vs their mean
#: mu/sd normalise the LATENT and are a property of the deployment data, so they
#: must agree - a difference means the arms read different spaces.
RUN_MATCHED_NORMS = ("mu", "sd")
#: bmu/bsd normalise the BELIEF stream, which the transition produces. They are
#: expected to differ between M1 and M0 whenever condition == "belief", so a
#: difference is recorded and is not a matching failure.
RUN_FREE_NORMS = ("bmu", "bsd")


def read_run_meta(arm_dir) -> dict:
    """The executor's own record of how one arm ran."""
    doc = json.loads((Path(arm_dir) / "summary.json").read_text())
    meta = {k: doc.get(k) for k in RUN_MATCHED_FIELDS}
    meta["panel"] = doc.get("panel")
    meta["zero_residual"] = doc.get("zero_residual")
    meta["wm_source"] = doc.get("wm_source")
    meta["norm_digest"] = doc.get("norm_digest") or {}
    # the REALISED intervention magnitude: reported and tolerance-checked, but not a
    # RUN_MATCHED_FIELD, because it is a continuous measurement rather than a flag and
    # exact equality across arms is neither expected nor required
    meta["mean_abs_delta_panel"] = doc.get("mean_abs_delta_panel")
    meta["max_abs_delta_panel"] = doc.get("max_abs_delta_panel")
    return meta


def check_run_matched(metas: dict, panel_lo: int, panel_n: int) -> dict:
    """Compare what the arms RAN with, not only what their checkpoints declared.

    `check_matched` runs on the checkpoints before the first reset; this runs on
    the executor's own summary.json after the last one, and catches the things a
    checkpoint cannot show: the panel actually stepped, whether --zero-residual
    was really on for base and off for the rest, whether the evaluation
    instrumentation was on at all, and whether the latent normaliser the arms
    deployed under was the same one.
    """
    panel = [panel_lo, panel_lo + panel_n - 1, panel_n]
    hard, expected, differing, unverifiable = [], [], [], []
    for name, meta in sorted(metas.items()):
        if meta.get("panel") is not None and list(meta["panel"]) != panel:
            hard.append(f"{name}: ran panel {meta['panel']}, the preregistration "
                        f"says {panel}")
        want_zero = (name == "base")
        if meta.get("zero_residual") is not None \
                and bool(meta["zero_residual"]) != want_zero:
            hard.append(f"{name}: zero_residual={meta['zero_residual']}, the round "
                        f"registers {want_zero} for this arm")
        if meta.get("eval_labels") is False:
            hard.append(f"{name}: ran without --eval-labels, so events, stage and "
                        f"last-progress are 0 for every episode by construction "
                        f"(run_v206_belief_residual_deploy.py:396-404) and are not "
                        f"measurements")
        if meta.get("fixed_horizon") is False:
            hard.append(f"{name}: ran without --fixed-horizon, so the surviving "
                        f"boundary count is a function of the outcome "
                        f"(lcwm/loho.py:70) and every per-boundary average is "
                        f"biased by the arm's own success rate")
    treated = [n for n in ARM_ORDER if n != "base" if n in metas]
    for fld in RUN_MATCHED_FIELDS:
        vals = {n: metas[n].get(fld) for n in treated}
        if all(v is None for v in vals.values()):
            unverifiable.append(fld)
            continue
        if len({json.dumps(v, sort_keys=True, default=str) for v in vals.values()}) > 1:
            differing.append({"field": fld, "values": vals})
            hard.append(f"{fld} differs across the treated arms: {vals}")
    for key in RUN_MATCHED_NORMS:
        vals = {n: (metas[n].get("norm_digest") or {}).get(key)
                for n in treated}
        if all(v is None for v in vals.values()):
            unverifiable.append(f"norm_digest.{key}")
            continue
        if len({str(v) for v in vals.values()}) > 1:
            differing.append({"field": f"norm_digest.{key}", "values": vals})
            hard.append(f"the latent normaliser {key} differs across the treated "
                        f"arms: {vals}; they did not deploy in the same space")
    for key in RUN_FREE_NORMS:
        vals = {n: (metas[n].get("norm_digest") or {}).get(key)
                for n in treated}
        if len({str(v) for v in vals.values()}) > 1:
            expected.append({"field": f"norm_digest.{key}", "values": vals,
                             "note": "the belief normaliser is produced BY the "
                                     "transition; a difference is expected when "
                                     "condition == 'belief' and is not a matching "
                                     "failure"})
    # THE REALISED RESIDUAL MAGNITUDE. `scale` pins the bound; this is what the arm
    # actually executed. On the 2026-09-12 calibration ladder magnitude - not policy
    # content - moved every endpoint (|delta| 0.051/0.105/0.206 -> zero milestone-4
    # events, 0.440 -> 4/24), so two arms sharing a bound and every flag can still
    # differ in the one quantity known to drive the result. Reported always; flagged
    # when it exceeds the pre-registered tolerance.
    realised = {n: metas[n].get("mean_abs_delta_panel") for n in treated}
    have = {n: float(v) for n, v in realised.items() if v is not None}
    delta_block = {"per_arm": realised, "tolerance": REALISED_DELTA_TOLERANCE,
                   "checked": bool(have)}
    if not have:
        unavailable = ("mean_abs_delta_panel is absent from every arm's summary.json, "
                       "so the realised intervention magnitude - the quantity the "
                       "calibration showed drives every endpoint - is UNVERIFIED. "
                       "`scale` matches the bound, not the intervention.")
        delta_block["note"] = unavailable
        unverifiable.append("mean_abs_delta_panel")
    else:
        mean = sum(have.values()) / len(have)
        spread = {n: (abs(v - mean) / mean if mean > 0 else 0.0)
                  for n, v in have.items()}
        delta_block.update({"panel_mean": mean, "relative_deviation": spread})
        far = {n: round(r, 4) for n, r in spread.items()
               if r > REALISED_DELTA_TOLERANCE}
        if far:
            differing.append({"field": "mean_abs_delta_panel", "values": have})
            hard.append(
                f"the REALISED residual magnitude differs across treated arms by more "
                f"than {REALISED_DELTA_TOLERANCE:.0%}: {far} against a panel mean of "
                f"{mean:.4f}. The arms share a bound but did not execute comparable "
                f"interventions, and the calibration ladder attributes every endpoint "
                f"to magnitude, so a between-arm difference cannot be read as the "
                f"transition")
    return {"panel_registered": panel, "per_arm": metas, "compared": treated,
            "differences": differing, "expected_differences": expected,
            "realised_delta": delta_block,
            "not_verifiable": sorted(set(unverifiable)), "hard_failures": hard,
            "rule": "nothing may differ between M1 and M0_cont except the learned "
                    "transition; M1 vs M0 additionally differs in optimiser budget "
                    "and is reported as context, not as the dynamics contrast"}


#: Endpoints whose whole-panel constancy is the fingerprint of instrumentation
#: that was never switched on. chain3's frozen-policy stage distribution is
#: {0:3, 3:2, 4:87, 5:2, 6:2} and its last-progress median is 260 of 750.
DEGENERATE_KEYS = ("milestone4", "stage", "last_progress_step")


def degenerate_arms(arms, run_matched: dict | None) -> dict:
    """Arms reporting zero progress on every seed, and whether that is a reading.

    An arm at milestone4 = stage = last_progress_step = 0 on every seed is either
    an arm that reached no milestone in the whole panel or an arm whose sidecar
    was written without --eval-labels. If the executor's record says --eval-labels
    was on, it is a reading and is flagged; if that cannot be established it is a
    hard failure, because the contrast it would otherwise produce is p = 1.0 by
    construction rather than by measurement.
    """
    flagged, hard = [], []
    for arm in arms:
        vals = {k: [row[k] for row in arm.per_seed.values() if row[k] is not None]
                for k in DEGENERATE_KEYS}
        if not all(vals[k] for k in DEGENERATE_KEYS):
            continue
        if not all(v == 0.0 for k in DEGENERATE_KEYS for v in vals[k]):
            continue
        meta = (run_matched or {}).get("per_arm", {}).get(arm.name, {})
        entry = {"arm": arm.name, "n": len(arm.per_seed),
                 "eval_labels_recorded": meta.get("eval_labels"),
                 "keys": list(DEGENERATE_KEYS)}
        if meta.get("eval_labels") is True:
            entry["reading"] = ("the executor records --eval-labels as on, so this "
                                "is a measured whole-panel zero, not missing "
                                "instrumentation")
            flagged.append(entry)
        else:
            hard.append(f"{arm.name}: milestone4, stage and last_progress_step are "
                        f"0 on all {len(arm.per_seed)} seeds and the executor's "
                        f"record does not establish that --eval-labels was on; a "
                        f"contrast on these would be p = 1.0 by construction")
    return {"flagged": flagged, "hard_failures": hard}


# --------------------------------------------------------------------------
# contrasts
# --------------------------------------------------------------------------
@dataclass
class ArmResult:
    name: str
    role: str
    actor: str
    sha256: str | None
    out_dir: str | None = None
    tag: str | None = None
    per_seed: dict = field(default_factory=dict)
    unavailable: list = field(default_factory=list)
    source: list = field(default_factory=list)


def paired_vectors(a: ArmResult, b: ArmResult, key: str):
    seeds = sorted(set(a.per_seed) & set(b.per_seed))
    pairs = [(s, a.per_seed[s][key], b.per_seed[s][key]) for s in seeds]
    pairs = [(s, x, y) for s, x, y in pairs if x is not None and y is not None]
    return [s for s, _, _ in pairs], [x for _, x, _ in pairs], [y for _, _, y in pairs]


def contrast(a: ArmResult, b: ArmResult, ep: Endpoint, panel_text: str) -> dict:
    seeds, xa, xb = paired_vectors(a, b, ep.key)
    name = f"{ep.key}:{a.name}-vs-{b.name}"
    if not seeds:
        return {"contrast": name, "endpoint": ep.key, "available": False,
                "reason": f"no paired seeds carrying {ep.key}"}
    n = len(seeds)
    out = {"contrast": name, "endpoint": ep.key, "endpoint_tier": ep.tier,
           "available": True, "n_paired": n, "arm_a": a.name, "arm_b": b.name,
           "panel": panel_text, "endpoint_caveat": ep.caveat,
           "overturned_by": OVERTURN.get((a.name, b.name), "not registered")}
    if ep.kind == "binary":
        ka, kb = int(sum(xa)), int(sum(xb))
        b_cnt = sum(1 for i in range(n) if xa[i] > xb[i])
        c_cnt = sum(1 for i in range(n) if xb[i] > xa[i])
        mc = exact_mcnemar(b_cnt, c_cnt)
        out.update({"test": "exact McNemar (paired, discordant pairs only)",
                    "a_count": ka, "b_count": kb, "a_rate": ka / n, "b_rate": kb / n,
                    "a_cp95": list(clopper_pearson(ka, n)),
                    "b_cp95": list(clopper_pearson(kb, n)),
                    "discordant": mc, "p": mc["p_two_sided"],
                    "p_one_sided_a_greater": mc["p_one_sided_a_greater"],
                    "effect_rate_difference": (ka - kb) / n})
    else:
        # `direction` is +1 when larger is the improvement and -1 when smaller is
        # (d_min_final_atom: a CLOSER approach is better). It was registered per
        # endpoint and never applied, so a -1 endpoint would have printed its
        # better/worse counts and its bootstrap interval with the sign inverted, and
        # an arm that approached the object more closely would have been reported as
        # worse. Orienting the differences here makes "better" mean better for every
        # endpoint, and `raw_mean_difference` keeps the unoriented value visible.
        raw = [xa[i] - xb[i] for i in range(n)]
        diffs = raw if ep.direction >= 0 else [-d for d in raw]
        wil = wilcoxon_signed_rank(diffs)
        boot = paired_bootstrap(diffs)
        out.update({"test": "Wilcoxon signed-rank (paired) + fixed-seed paired "
                            "bootstrap CI of the mean difference",
                    "a_mean": sum(xa) / n, "b_mean": sum(xb) / n,
                    "a_median": _median(xa), "b_median": _median(xb),
                    "mean_difference": boot["mean"], "ci95": boot["ci95"],
                    "bootstrap": boot, "wilcoxon": wil, "p": wil["p_two_sided"],
                    "n_better": sum(1 for d in diffs if d > 0),
                    "n_worse": sum(1 for d in diffs if d < 0),
                    "n_tied": sum(1 for d in diffs if d == 0),
                    "direction": ep.direction,
                    "oriented": ep.direction < 0,
                    "raw_mean_difference": (sum(raw) / n) if n else None})
    return out


def _median(xs):
    ys = sorted(xs)
    n = len(ys)
    if n == 0:
        return None
    return ys[n // 2] if n % 2 else 0.5 * (ys[n // 2 - 1] + ys[n // 2])


# --------------------------------------------------------------------------
# preregistration + report
# --------------------------------------------------------------------------
def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                          capture_output=True, text=True).stdout.strip()


def build_prereg(args, arms: list[ArmResult], guard: dict, power: dict,
                 matched: dict | None, stamp: str) -> dict:
    return {
        "schema": SCHEMA_PREREG,
        "utc": stamp,
        "git": git_head(),
        "source_sha256": sha256_file(Path(__file__)),
        "task": args.task,
        "horizon": HORIZON,
        "chunk": CHUNK,
        "milestones": list(MILESTONES),
        "panel": {"start": args.panel_start, "count": args.panel,
                  "seeds": [args.panel_start, args.panel_start + args.panel - 1],
                  "committed_before_any_episode": True,
                  "rule": "CLAUDE.md rule 8 - the seed count is fixed here and is "
                          "not revisited after reading any endpoint; no extension "
                          "of this panel may be reported as the same test"},
        "arms": [{"name": a.name, "role": a.role, "actor": a.actor,
                  "sha256": a.sha256,
                  "status": "present" if a.sha256 else "MISSING at preregistration",
                  "zero_residual": a.name == "base"} for a in arms],
        "arm_run_order": list(ARM_ORDER),
        "executor": DEPLOY,
        "endpoints": [{"key": e.key, "label": e.label, "kind": e.kind, "tier": e.tier,
                       "direction": e.direction, "frozen_policy_prior": e.prior,
                       "caveat": e.caveat} for e in ENDPOINTS],
        "endpoint_hierarchy": {
            "estimation": ["success"],
            "primary": ["milestone4"],
            "secondary": ["stage", "last_progress_step", "phi_basin_mean",
                          "d_min_final_atom"],
            "rule": "terminal success is reported for every arm with a "
                    "Clopper-Pearson interval and is NOT a test at this n; the "
                    "primary test is milestone-4 attainment; the secondary "
                    "endpoints are supportive and are not confirmatory on their own",
        },
        "basin_rule": BASIN_RULE,
        "comparisons": {
            "primary": [list(c) for c in PRIMARY_CONTRASTS],
            "context": [list(c) for c in CONTEXT_CONTRASTS],
            "rule": "CLAUDE.md rule 7 - 'A beats base, B does not' is not 'A beats "
                    "B'. M1-vs-M0 and M1-vs-Q are run directly and are not inferred "
                    "from each arm's contrast against base",
        },
        "multiplicity": {
            "method": "Holm step-down",
            "alpha": ALPHA,
            "family_primary": [f"milestone4:{a}-vs-{b}" for a, b in PRIMARY_CONTRASTS],
            "family_secondary": [f"{e}:{a}-vs-{b}"
                                 for e in ("stage", "last_progress_step",
                                           "phi_basin_mean")
                                 for a, b in PRIMARY_CONTRASTS],
            "uncorrected": [f"*:{a}-vs-{b}" for a, b in CONTEXT_CONTRASTS],
            "note": "the context contrasts are descriptive and are reported with "
                    "intervals and uncorrected p-values, labelled as such",
        },
        "tests": {
            "binary": "exact McNemar on the discordant pairs via the binomial; the "
                      "discordant table (b, c) is reported with every p-value",
            "ordinal_continuous": "Wilcoxon signed-rank (exact when there are no "
                                  "ties and n <= 50, else the normal approximation "
                                  "with tie correction) plus a fixed-seed paired "
                                  "bootstrap 95% CI of the mean difference",
            "intervals": "Clopper-Pearson at alpha=0.025 per side (a true 95% "
                         "interval; CLAUDE.md records that earlier cp95 fields "
                         "written before 2026-08-29 are 90%)",
            "implementation": "computed in scripts/eval_v251_round.py with no "
                              "dependency on scipy; scipy is recorded alongside as "
                              "an independent cross-check when present",
            "scipy_available": HAVE_SCIPY,
        },
        "power": power,
        "seed_guard": guard,
        "matched_arms_check": matched,
        "declared_limits": [
            "Phi' is a scripted privileged stand-in for per-boundary human video "
            "annotation; it is identical across arms and may not be used at "
            "deployment time. Every Phi' effect is scoped to 'under this supplied "
            "progress annotation'.",
            "one environment panel and one training initialisation per arm: this "
            "round measures cross-seed environment variation, not cross-init "
            "training variation (CLAUDE.md rule 9). Seed variance at fixed "
            "hyperparameters was ~0.24 on 2026-09-01.",
            "the 2/96 frozen-policy figure comes from the v121 collector, not from "
            "this executor; it is a prior for the power arithmetic, and the base "
            "arm run here through --zero-residual is the only baseline any contrast "
            "is taken against (CLAUDE.md rule 5).",
        ],
    }


def format_report(summary: dict) -> str:
    lines = []
    pw = summary["preregistration"]["power"]["binary_endpoint"]
    lines.append(f"panel {summary['panel'][0]}-{summary['panel'][1]} "
                 f"(n={summary['panel'][2]}), task {summary['task']}, "
                 f"horizon {HORIZON}, mode {summary['mode']}")
    lines.append("")
    lines.append("terminal success (ESTIMATION endpoint at this n - not a test)")
    for name in ARM_ORDER:
        arm = summary["arms"].get(name)
        if not arm or arm.get("successes") is None:
            continue
        lo, hi = arm["cp95"]
        lines.append(f"  {name:5s} {arm['successes']:3d}/{arm['n']:<3d} = "
                     f"{arm['rate']:.3f}  CP95 [{lo:.3f}, {hi:.3f}]")
    lines.append(f"  {pw['statement']}")
    lines.append("")
    for tier, title in (("primary", "PRIMARY endpoint"),
                        ("secondary", "SECONDARY endpoints (supportive, not "
                                      "confirmatory)"),
                        ("estimation", "ESTIMATION endpoint, paired view")):
        rows = [c for c in summary["contrasts"]
                if c.get("available") and c.get("endpoint_tier") == tier
                and c.get("family") != "context"]
        if not rows:
            continue
        lines.append(title)
        for c in rows:
            suffix = ("  [ESTIMATION endpoint: this p is not a registered test at "
                      "this n]" if tier == "estimation" else "")
            lines.append("  " + _contrast_line(c) + suffix)
            lines.append(f"      overturned by: {c['overturned_by']}")
        lines.append("")
    ctx = [c for c in summary["contrasts"]
           if c.get("available") and c.get("family") == "context"]
    if ctx:
        lines.append("CONTEXT contrasts (descriptive, uncorrected)")
        for c in ctx:
            lines.append("  " + _contrast_line(c))
        lines.append("")
    if summary.get("unavailable_endpoints"):
        lines.append("NOT EVALUATED (registered but absent from the deploy sidecar): "
                     + ", ".join(summary["unavailable_endpoints"])
                     + ". Holm still divides by the REGISTERED family size, so a "
                       "test that could not be run does not loosen the correction")
        lines.append("")
    rm = summary.get("run_matched") or {}
    degen = (summary.get("degenerate_arms") or {}).get("flagged", [])
    if rm.get("not_verifiable") or rm.get("expected_differences") or degen:
        lines.append("run-level matching (read back from each arm's own summary.json)")
        if rm.get("not_verifiable"):
            lines.append("  NOT VERIFIABLE, the executor does not record it: "
                         + ", ".join(rm["not_verifiable"]))
        for e in rm.get("expected_differences", []):
            lines.append(f"  {e['field']} differs across arms: {e['note']}")
        for e in degen:
            lines.append(f"  arm {e['arm']}: zero progress on all {e['n']} seeds - "
                         f"{e['reading']}")
        lines.append("")
    shown = {c["endpoint"] for c in summary["contrasts"] if c.get("available")}
    if shown:
        lines.append("endpoint notes")
        for ep in ENDPOINTS:
            if ep.key in shown:
                lines.append(f"  {ep.key}: {ep.caveat}")
        lines.append("")
    lines.append("conditions: " + summary["conditions"])
    return "\n".join(lines)


def fmt_p(p: float) -> str:
    """Never print a p-value as 0.0000: a tail is small, not absent."""
    if p is None:
        return "n/a"
    return f"{p:.4f}" if p >= 1e-4 else f"{p:.2e}"


def _contrast_line(c: dict) -> str:
    ep = ENDPOINT_BY_KEY[c["endpoint"]]
    holm_txt = ""
    if "holm" in c:
        holm_txt = (f", Holm alpha {c['holm']['alpha_adjusted']:.4f} "
                    f"p_adj {fmt_p(c['holm']['p_adjusted'])} -> "
                    f"{'rejected' if c['holm']['reject'] else 'not rejected'}")
    if ep.kind == "binary":
        d = c["discordant"]
        return (f"{ep.key:18s} {c['arm_a']:4s} vs {c['arm_b']:4s}: "
                f"{c['a_count']}/{c['n_paired']} = {c['a_rate']:.3f} "
                f"[{c['a_cp95'][0]:.3f},{c['a_cp95'][1]:.3f}] vs "
                f"{c['b_count']}/{c['n_paired']} = {c['b_rate']:.3f} "
                f"[{c['b_cp95'][0]:.3f},{c['b_cp95'][1]:.3f}], "
                f"discordant +{d['b']} -{d['c']}, exact McNemar two-sided "
                f"p = {fmt_p(c['p'])}{holm_txt}")
    return (f"{ep.key:18s} {c['arm_a']:4s} vs {c['arm_b']:4s}: "
            f"{c['a_mean']:.3f} vs {c['b_mean']:.3f}, "
            f"mean paired diff {c['mean_difference']:+.3f} "
            f"CI95 [{c['ci95'][0]:+.3f},{c['ci95'][1]:+.3f}], "
            f"+{c['n_better']} -{c['n_worse']} ={c['n_tied']}, "
            f"Wilcoxon p = {fmt_p(c['p'])}{holm_txt}")


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="matched four-arm evaluation of the 2026-09-12 chain3 round")
    ap.add_argument("--panel", type=int, required=True,
                    help="REQUIRED, no default: the seed count, committed in the "
                         "preregistration before anything is read (CLAUDE.md rule 8)")
    ap.add_argument("--residual-start-chunk", type=int, default=0,
                    help="deployment schedule gate, passed identically to EVERY arm. "
                         "It must equal the --residual-start-chunk the D1 collectors "
                         "and the interface were trained under, or the interface runs "
                         "at states it never saw. Recorded in the preregistration.")
    ap.add_argument("--panel-start", type=int, required=True,
                    help="REQUIRED, no default: first evaluation seed. It must not "
                         "touch any collection family; see KNOWN_FAMILIES")
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--m1", type=Path, help="updated-WM actor checkpoint")
    ap.add_argument("--m0", type=Path, help="stale-WM actor checkpoint")
    ap.add_argument("--q", type=Path, help="same-data no-rollout actor checkpoint")
    ap.add_argument("--m0-cont", type=Path,
                    help="matched-compute stale-WM actor (build_v241 `base_continue`). "
                         "Without it, M1 vs M0 confounds the updated dynamics with "
                         "75%% more optimiser steps.")
    ap.add_argument("--m1-shuf", type=Path,
                    help="action-shuffled updated-WM actor checkpoint")
    ap.add_argument("--rand", type=Path,
                    help="UNTRAINED random residual at the round's scale. The "
                         "calibration measured an untrained residual reaching 4/24 "
                         "milestone-4 and 1/24 success against 0/24 for base, so "
                         "without this arm the trust region's own effect would be "
                         "credited to the interface training.")
    ap.add_argument("--base-actor", type=Path, default=None,
                    help="checkpoint used for the frozen-pi0.5 arm, run with "
                         "--zero-residual so the code path is identical. Defaults "
                         "to --m1: with Delta := 0 the actor weights cannot enter "
                         "the executed action")
    ap.add_argument("--collection-manifest", type=Path, action="append", default=None,
                    help="manifest.json / summary.json of a collection run; its "
                         "seed range is registered as a forbidden family. Repeatable")
    ap.add_argument("--acknowledge-panel-reuse", action="store_true",
                    help="declare an overlap with a non-collection family (a prior "
                         "evaluation panel or a probe). A collection overlap can "
                         "never be declared away")
    ap.add_argument("--sidecar", default=None,
                    help=f"sidecar filename to read per-episode endpoints from; "
                         f"default searches {list(SIDECAR_FILES)}")
    ap.add_argument("--execute", action="store_true",
                    help="actually run the four arms. Without it the driver does "
                         "the preregistration, the seed guard and the power "
                         "arithmetic and stops: that is the default")
    ap.add_argument("--dry-run", action="store_true",
                    help="explicit form of the default; incompatible with --execute")
    ap.add_argument("--no-check-matched", action="store_true",
                    help="skip the CPU-only checkpoint comparison. The check is the "
                         "only thing verifying that M1 and M0 differ ONLY in the "
                         "learned transition; skipping it is recorded in the "
                         "preregistration")
    ap.add_argument("--mujoco-gl", default="egl")
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run and args.execute:
        raise SystemExit("--dry-run and --execute are mutually exclusive")
    execute = bool(args.execute)
    if args.panel <= 0:
        raise SystemExit("--panel must be positive")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")

    base_actor = args.base_actor or args.m1
    spec = [("base", "frozen pi0.5, Delta := 0, identical code path", base_actor),
            ("rand", "UNTRAINED random residual at the round's scale", args.rand),
            ("M0", "stale WM: transition trained on D0 only", args.m0),
            ("M0_cont", "stale WM + M1's extra optimiser budget, still D0 only",
             args.m0_cont),
            ("M1", "updated WM: transition trained on D0 + D1", args.m1),
            ("M1_shuf", "updated WM with D1 action pairing broken", args.m1_shuf),
            ("Q", "same-data model-free / no-rollout control", args.q)]
    arms = [ArmResult(name=n, role=r, actor=str(p) if p else "<unset>",
                      sha256=sha256_file(p) if p else None) for n, r, p in spec]
    by_name = {a.name: a for a in arms}
    if execute:
        missing = [a.name for a in arms if a.sha256 is None]
        if missing:
            raise SystemExit(f"--execute needs every checkpoint to exist and hash; "
                             f"missing: {missing}")

    families = list(KNOWN_FAMILIES)
    for man in (args.collection_manifest or []):
        families.extend(families_from_manifest(man))
    guard = check_panel(args.panel_start, args.panel, families,
                        args.acknowledge_panel_reuse)
    power = power_block(args.panel)

    matched = {"skipped": True, "reason": "--no-check-matched"}
    if not args.no_check_matched:
        present = {n: by_name[n] for n in ARM_ORDER if n != "base"
                   if by_name[n].sha256 is not None}
        if len(present) < 2:
            matched = {"skipped": True,
                       "reason": "fewer than two arm checkpoints exist yet; rerun "
                                 "the dry run once they do"}
        else:
            fps = {n: checkpoint_fingerprint(Path(a.actor))
                   for n, a in present.items()}
            matched = check_matched(fps)
            if matched["hard_failures"] and execute:
                raise SystemExit(
                    "the arms are not matched on "
                    f"{matched['hard_failures']}; nothing may differ between M1 and "
                    "M0 except the learned transition. Details:\n"
                    + json.dumps(matched["differences"], indent=2, default=str))

    suffix = "" if execute else "_dryrun"
    out = Path(args.out_root) / f"{args.task}_{stamp}{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    prereg = build_prereg(args, arms, guard, power, matched, stamp)
    prereg_path = out / "preregistration.json"
    prereg_path.write_text(json.dumps(prereg, indent=2, default=str))
    prereg_sha = sha256_file(prereg_path)

    print(f"preregistration -> {prereg_path}")
    print(f"  panel {args.panel_start}-{args.panel_start + args.panel - 1} "
          f"(n={args.panel}), task {args.task}, horizon {HORIZON}")
    print(f"  arms (fixed run order {list(ARM_ORDER)}):")
    for a in arms:
        print(f"    {a.name:5s} {a.role:52s} sha256="
              f"{(a.sha256 or 'MISSING')[:16]}")
    print(f"  seed guard: {len(families)} families checked, "
          f"{'clean' if guard['clean'] else 'declared reuse'}")
    if matched.get("differences"):
        print(f"  matched-arms check: {len(matched['differences'])} field(s) differ "
              f"{[d['field'] for d in matched['differences']]}")
    elif not matched.get("skipped"):
        print("  matched-arms check: no registered field differs; M1 vs M0 "
              f"transition weights {matched.get('m1_m0_transition_comparison')}"
              + (f"; NOT VERIFIABLE from the checkpoints: "
                 f"{matched['not_verifiable']}" if matched.get("not_verifiable")
                 else ""))
    print("  power:")
    print(f"    {power['binary_endpoint']['statement']}")
    mde = power["ordinal_and_continuous_endpoints"]
    print(f"    ordinal/continuous, paired, alpha={ALPHA} two-sided, power "
          f"{mde['power']}: detectable d_z = {mde['cohen_dz']:.3f} SD of the paired "
          f"difference (Wilcoxon {mde['wilcoxon_dz']:.3f})")

    if not execute:
        print("\ndry run: no environment step was taken, no arm was run. "
              "Re-run with --execute once the preregistration above is accepted.")
        return 0

    env = dict(os.environ)
    env["MUJOCO_GL"] = args.mujoco_gl
    run_tag = f"v251_{stamp}"
    for name in ARM_ORDER:
        arm = by_name[name]
        tag = f"{run_tag}_{name}"
        cmd = arm_command(name, Path(arm.actor), args.panel, args.panel_start, tag,
                          args.task, args.residual_start_chunk)
        print(f"\n=== arm {name}: {' '.join(cmd)}", flush=True)
        started = time.time()
        arm_dir = execute_arm(name, cmd, out / f"{name}.log", env, tag)
        arm.out_dir, arm.tag = str(arm_dir), tag
        got = read_arm_endpoints(arm_dir, args.sidecar)
        arm.per_seed, arm.unavailable, arm.source = (got["per_seed"],
                                                     got["unavailable"],
                                                     got["source"])
        print(f"  {name}: {got['n']} episodes from {got['source']} "
              f"({time.time() - started:.0f}s)"
              + (f"; unavailable: {got['unavailable']}" if got["unavailable"] else ""))

    seed_sets = {a.name: set(a.per_seed) for a in arms}
    ref = seed_sets[ARM_ORDER[0]]
    bad = {n: sorted(s ^ ref) for n, s in seed_sets.items() if s != ref}
    if bad:
        shown = {n: v[:8] for n, v in bad.items()}
        raise SystemExit("arms did not run the same seeds; pairing is impossible: "
                         + json.dumps(shown))
    # the arms agreeing with each other is not the arms running the REGISTERED
    # panel: four arms that all fell back to the executor's own default seeds
    # would agree perfectly and would be scored on a spent family.
    registered_seeds = set(range(args.panel_start, args.panel_start + args.panel))
    if ref != registered_seeds:
        raise SystemExit(json.dumps(
            {"error": "the arms did not run the pre-registered panel",
             "registered": [args.panel_start,
                            args.panel_start + args.panel - 1, args.panel],
             "n_ran": len(ref),
             "registered_seeds_not_run": sorted(registered_seeds - ref)[:8],
             "seeds_run_but_not_registered": sorted(ref - registered_seeds)[:8]},
            indent=2))

    run_matched = check_run_matched({a.name: read_run_meta(a.out_dir) for a in arms},
                                    args.panel_start, args.panel)
    degen = degenerate_arms(arms, run_matched)
    blocking = run_matched["hard_failures"] + degen["hard_failures"]
    if blocking:
        raise SystemExit("the arms are not a matched set as they RAN; refusing to "
                         "report contrasts between them:\n  "
                         + "\n  ".join(blocking))
    if run_matched["not_verifiable"]:
        print(f"  run-level matching NOT VERIFIABLE for "
              f"{run_matched['not_verifiable']}: the executor's summary.json does "
              f"not record those fields")
    for entry in run_matched["expected_differences"]:
        print(f"  run-level: {entry['field']} differs across arms - {entry['note']}")
    for entry in degen["flagged"]:
        print(f"  run-level: arm {entry['arm']} reports zero progress on all "
              f"{entry['n']} seeds; {entry['reading']}")

    panel_text = (f"panel {args.panel_start}-{args.panel_start + args.panel - 1}, "
                  f"n={len(ref)} paired seeds")
    contrasts = []
    for ep in ENDPOINTS:
        for a_name, b_name in PRIMARY_CONTRASTS:
            c = contrast(by_name[a_name], by_name[b_name], ep, panel_text)
            c["family"] = "primary" if ep.tier == "primary" else (
                "secondary" if ep.tier == "secondary" else "estimation")
            contrasts.append(c)
        for a_name, b_name in CONTEXT_CONTRASTS:
            c = contrast(by_name[a_name], by_name[b_name], ep, panel_text)
            c["family"] = "context"
            contrasts.append(c)

    # the REGISTERED family sizes, the same expression build_prereg writes: an
    # endpoint missing from the sidecar must not shrink the family and weaken the
    # correction that was committed to before the run
    registered_family = {
        fam: len([e for e in ENDPOINTS if e.tier == fam]) * len(PRIMARY_CONTRASTS)
        for fam in ("primary", "secondary")}
    holm_out = {}
    for fam in ("primary", "secondary"):
        pv = {c["contrast"]: c["p"] for c in contrasts
              if c.get("available") and c.get("family") == fam}
        if pv:
            holm_out[fam] = holm(pv, ALPHA, registered_family[fam])
            for c in contrasts:
                if c.get("contrast") in holm_out[fam]:
                    c["holm"] = holm_out[fam][c["contrast"]]

    unavailable = sorted({k for a in arms for k in a.unavailable})
    summary = {
        "schema": SCHEMA_SUMMARY,
        "utc": stamp,
        "git": git_head(),
        "mode": "executed",
        "task": args.task,
        "panel": [args.panel_start, args.panel_start + args.panel - 1, args.panel],
        "preregistration_path": str(prereg_path),
        "preregistration_sha256": prereg_sha,
        "preregistration": prereg,
        "arms": {},
        "contrasts": contrasts,
        "holm": holm_out,
        "unavailable_endpoints": unavailable,
        "run_matched": run_matched,
        "degenerate_arms": degen,
        "conditions": (
            f"frozen pi0.5 backbone; chain3_lr2 at horizon {HORIZON}; one "
            f"environment panel; one training initialisation per arm; progress "
            f"annotation = lcwm/v250_progress.py Phi', a scripted privileged "
            f"stand-in for human video annotation, identical across arms. Every "
            f"effect in this report is an effect UNDER THESE CONDITIONS, and each carries "
            f"the arm that would overturn it."),
    }
    for a in arms:
        succ = [v["success"] for v in a.per_seed.values() if v["success"] is not None]
        k, n = int(sum(succ)), len(succ)
        summary["arms"][a.name] = {
            "role": a.role, "actor": a.actor, "sha256": a.sha256,
            "dir": a.out_dir, "tag": a.tag, "endpoint_source": a.source,
            "successes": k if n else None, "n": n,
            "rate": (k / n) if n else None,
            "cp95": list(clopper_pearson(k, n)) if n else None,
            "unavailable": a.unavailable,
            "per_seed": {str(s): v for s, v in sorted(a.per_seed.items())},
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    report = format_report(summary)
    (out / "report.txt").write_text(report + "\n")
    print("\n" + report)
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
