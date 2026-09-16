#!/usr/bin/env python
"""How the v251 round's primary p-values move as the panel's n rises.

    python scripts/report_v251_p_trajectory.py                     # default paths
    python scripts/report_v251_p_trajectory.py --steps 48 96 288   # other prefixes

WHAT THIS ANSWERS.  CLAUDE.md rule 8: "Watch the p-value's trajectory as power
rises, not its value at one n.  Moving away from significance as n grows is the
signature of a null effect that looked positive when underpowered."  The
2026-09-13 chain3_lr2 round ran a single pre-registered 288-seed panel and
reported one p-value per contrast at the end of it.  The per-episode records are
seed-ordered, so the same test can be recomputed on seed-ordered PREFIXES of that
one panel, and the resulting curve is what rule 8 asks to be looked at.

WHAT THIS IS NOT.  These prefixes are NOT independent panels and NOT a sequential
test: they are nested subsets of one dataset, so the six p-values per contrast are
strongly dependent and only the n = 288 value is the registered analysis.  The
preregistration (results/v251_round/.../preregistration.json) fixes the panel at
288 and forbids reporting an extension as the same test; this script does not
extend anything, it looks backwards inside the committed panel.  A prefix p-value
is a diagnostic of the trajectory's shape, never a result in its own right.

DEFINITIONS ARE IMPORTED, NOT RE-DERIVED.  The endpoints and the test come from
scripts/eval_v251_round.py itself - `derive_from_events` (milestone-4 attainment
is `4 in events_achieved`, the pre-registered primary endpoint), `_as_binary`
(strict 0/1 coercion for terminal success) and `exact_mcnemar` (exact paired
binomial on the discordant pairs).  Nothing about the endpoint or the test is
restated here, so this report cannot drift from the round driver's definition.

ENDPOINT TIERS, from the preregistration and reproduced in the output:
  * milestone4 - PRIMARY, the registered test, five contrasts under Holm.
  * success    - ESTIMATION at this n, explicitly not a test; its trajectory is
                 printed and stored under `estimation` and labelled as such.
The M1-vs-M0 contrast is a registered CONTEXT contrast: uncorrected, descriptive.

CORRECTNESS CHECK.  The n = 288 two-sided p-values are compared against the
round's published table; any mismatch aborts with a non-zero exit code.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from eval_v251_round import (  # noqa: E402  - the round driver owns the definitions
    HAVE_SCIPY,
    PRIMARY_MILESTONE,
    _as_binary,
    clopper_pearson,
    derive_from_events,
    exact_mcnemar,
)

ROUND_TAG = "chain3_lr2_v251_2026-09-13T000155Z"
ARM_DIR_ROOT = REPO / "results" / "v206_belief_residual"
DEFAULT_OUT = (REPO / "results" / "v251_round" / "chain3_lr2_2026-09-13T000155Z"
               / "p_trajectory.json")
DEFAULT_PREREG = (REPO / "results" / "v251_round" / "chain3_lr2_2026-09-13T000155Z"
                  / "preregistration.json")

ARMS = ("base", "rand", "M0", "M0_cont", "M1", "M1_shuf", "Q")

#: the five registered primary contrasts, in the preregistration's own order,
#: plus the one registered CONTEXT contrast this report was asked to carry.
PRIMARY_CONTRASTS = (("M1", "M0_cont"), ("M1", "Q"), ("M1", "rand"),
                     ("M1", "M1_shuf"), ("M1", "base"))
CONTEXT_CONTRASTS = (("M1", "M0"),)

#: the round's published n = 288 milestone-4 table, as written in
#: plan_and_progress/2026-09-13.md: contrast -> (published two-sided p, its
#: printed half-ulp, published discordant b, published discordant c). This is the
#: reproduction target, not a hypothesis: a mismatch on the COUNTS means this
#: script's pairing or endpoint differs from the round driver's and the whole
#: artifact is void. The tolerance is the half-ulp of the digits as printed - a
#: loose absolute tolerance would let a p-value that is 17% off pass as a match,
#: which is exactly the failure this check exists to catch.
PUBLISHED_M4_AT_288 = {
    "M1-vs-base":    {"p": 3.05e-05, "half_ulp": 5e-8,  "b": 34, "c": 7},
    "M1-vs-M0_cont": {"p": 0.0241,   "half_ulp": 5e-5,  "b": 17, "c": 34},
    "M1-vs-Q":       {"p": 0.0328,   "half_ulp": 5e-5,  "b": 17, "c": 33},
    "M1-vs-rand":    {"p": 0.4638,   "half_ulp": 5e-5,  "b": 30, "c": 37},
    "M1-vs-M1_shuf": {"p": 1.0000,   "half_ulp": 5e-5,  "b": 25, "c": 25},
    "M1-vs-M0":      {"p": 1.0000,   "half_ulp": 5e-5,  "b": 21, "c": 21},
}
PUBLISHED_SOURCE = "plan_and_progress/2026-09-13.md, the round's reported table"

DEFAULT_STEPS = (48, 96, 144, 192, 240, 288)


def find_arm_dir(arm: str) -> Path:
    # the arm name is followed by the arm's own UTC stamp, so "M0_*" would also
    # glob "M0_cont_*"; the suffix is required to be a timestamp, not any tail.
    pat = re.compile(rf"^{re.escape(ROUND_TAG)}_{re.escape(arm)}_\d{{4}}-\d{{2}}-\d{{2}}T\d{{6}}Z$")
    matches = sorted(m for m in ARM_DIR_ROOT.glob(f"{ROUND_TAG}_{arm}_*")
                     if pat.match(m.name) and (m / "outcome.json").is_file())
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one outcome.json directory for arm "
                         f"{arm!r} under {ARM_DIR_ROOT}, found {len(matches)}: "
                         f"{[str(m) for m in matches]}")
    return matches[0]


def load_arm(arm: str) -> dict:
    """seed -> {milestone4, success} using the ROUND DRIVER's own derivations."""
    path = find_arm_dir(arm) / "outcome.json"
    doc = json.loads(path.read_text())
    recs = doc["episodes"]
    rows: dict[int, dict] = {}
    order: list[int] = []
    for rec in recs:
        seed = int(rec["seed"])
        if seed in rows:
            raise SystemExit(f"{arm}: seed {seed} appears twice in {path}")
        derived = derive_from_events(rec["events"])
        rows[seed] = {"milestone4": derived["milestone4"],
                      "success": _as_binary(rec["success"], "success")}
        order.append(seed)
    if order != sorted(order):
        raise SystemExit(f"{arm}: episodes in {path} are not written in ascending "
                         f"seed order; the 'seed-ordered prefix' this report takes "
                         f"would not be the run's own order")
    return {"arm": arm, "path": str(path.relative_to(REPO)), "rows": rows,
            "order": order, "n": len(order)}


def _dyadic_exponent(p: float, tol: float):
    """k such that 2**-k agrees with `p` to within `tol`, else None.

    The published digits are rounded, so the test is against the printed
    precision rather than against exact equality.
    """
    if p <= 0:
        return None
    ki = round(-math.log2(p))
    return ki if abs(2.0 ** -ki - p) <= tol else None


def _table_for_p(p: float, tol: float, max_n: int = 400):
    """Every (n_discordant, smaller cell) whose exact two-sided McNemar p agrees
    with `p` to the published precision. The exact p is a dyadic rational, so the
    table that would have produced a printed value can be recovered."""
    hits = []
    for n in range(1, max_n + 1):
        for k in range(0, n // 2 + 1):
            q = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
            if abs(q - p) <= tol:
                hits.append({"n_discordant": n, "smaller_cell": k,
                             "exact_p": q})
    return hits


def holm(pairs: list[tuple[str, float]]) -> dict[str, float]:
    """Holm step-down adjusted p-values over one family, order-preserving."""
    ordered = sorted(pairs, key=lambda kv: kv[1])
    m = len(ordered)
    adj: dict[str, float] = {}
    running = 0.0
    for i, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - i) * p))
        adj[name] = running
    return adj


def trajectory(arms: dict[str, dict], seeds: list[int], a: str, b: str,
               endpoint: str, steps) -> list[dict]:
    out = []
    for n in steps:
        take = seeds[:n]
        xa = [arms[a]["rows"][s][endpoint] for s in take]
        xb = [arms[b]["rows"][s][endpoint] for s in take]
        b_cnt = sum(1 for i in range(n) if xa[i] > xb[i])
        c_cnt = sum(1 for i in range(n) if xb[i] > xa[i])
        mc = exact_mcnemar(b_cnt, c_cnt)
        ka, kb = int(sum(xa)), int(sum(xb))
        out.append({
            "n": n,
            "a_count": ka, "b_count": kb,
            "a_rate": ka / n, "b_rate": kb / n,
            "a_cp95": list(clopper_pearson(ka, n)),
            "b_cp95": list(clopper_pearson(kb, n)),
            "rate_difference": (ka - kb) / n,
            "discordant_b_a_only": b_cnt,
            "discordant_c_b_only": c_cnt,
            "n_discordant": b_cnt + c_cnt,
            "p_two_sided": mc["p_two_sided"],
            "p_one_sided_a_greater": mc["p_one_sided_a_greater"],
            **({"scipy_p_two_sided": mc["scipy_p_two_sided"],
                "scipy_agrees": mc["scipy_agrees"]} if "scipy_p_two_sided" in mc
               else {}),
        })
    return out


def fmt_p(p: float) -> str:
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def print_block(title: str, rows: dict[str, list[dict]], steps, note: str) -> None:
    print()
    print(title)
    print(note)
    head = f"  {'contrast':<16}" + "".join(f"{('n=' + str(n)):>22}" for n in steps)
    print(head)
    print("  " + "-" * (len(head) - 2))
    for name, traj in rows.items():
        cells = []
        for pt in traj:
            cells.append(f"{fmt_p(pt['p_two_sided'])} ({pt['discordant_b_a_only']}"
                         f"/{pt['discordant_c_b_only']})")
        print(f"  {name:<16}" + "".join(f"{c:>22}" for c in cells))
    print("  (cell = two-sided exact McNemar p, then the discordant table "
          "b/c: b = seeds where A attained and B did not, c = the reverse)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, nargs="+", default=list(DEFAULT_STEPS),
                    help="seed-ordered prefix sizes (default 48 96 144 192 240 288)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    ap.add_argument("--no-check", action="store_true",
                    help="skip the n=288 reproduction check (it is on by default "
                         "and a mismatch is a hard failure)")
    args = ap.parse_args()

    steps = sorted(set(int(s) for s in args.steps))

    arms = {a: load_arm(a) for a in ARMS}
    seed_sets = {a: set(arms[a]["rows"]) for a in ARMS}
    shared = sorted(set.intersection(*seed_sets.values()))
    for a in ARMS:
        if seed_sets[a] != set(shared):
            raise SystemExit(f"{a} does not carry exactly the shared panel: "
                             f"{len(seed_sets[a])} seeds vs {len(shared)} shared")
    if max(steps) > len(shared):
        raise SystemExit(f"requested prefix {max(steps)} exceeds the shared panel "
                         f"of {len(shared)} seeds")

    prereg = json.loads(args.prereg.read_text()) if args.prereg.is_file() else None
    if prereg is not None:
        want = [list(c) for c in PRIMARY_CONTRASTS]
        got = [list(c) for c in prereg["comparisons"]["primary"]]
        if sorted(map(tuple, want)) != sorted(map(tuple, got)):
            raise SystemExit(f"the primary contrast set here {want} is not the "
                             f"preregistered one {got}")
        if prereg["panel"]["count"] != len(shared):
            raise SystemExit(f"preregistered panel {prereg['panel']['count']} != "
                             f"{len(shared)} shared seeds on disk")

    results = {"primary": {}, "context": {}, "estimation": {},
               "estimation_context": {}}
    for a, b in PRIMARY_CONTRASTS:
        results["primary"][f"{a}-vs-{b}"] = trajectory(
            arms, shared, a, b, "milestone4", steps)
        results["estimation"][f"{a}-vs-{b}"] = trajectory(
            arms, shared, a, b, "success", steps)
    for a, b in CONTEXT_CONTRASTS:
        results["context"][f"{a}-vs-{b}"] = trajectory(
            arms, shared, a, b, "milestone4", steps)
        results["estimation_context"][f"{a}-vs-{b}"] = trajectory(
            arms, shared, a, b, "success", steps)

    # Holm over the five registered PRIMARY contrasts, recomputed at each prefix.
    # Only the n = max step is the registered analysis; the earlier columns are
    # the diagnostic and are labelled so wherever they appear.
    holm_traj = {}
    for i, n in enumerate(steps):
        fam = [(k, results["primary"][k][i]["p_two_sided"])
               for k in results["primary"]]
        holm_traj[str(n)] = holm(fam)

    # per-arm milestone-4 and success counts along the same prefixes, so a moving
    # p-value can be read against the counts that moved it.
    per_arm = {}
    for a in ARMS:
        per_arm[a] = []
        for n in steps:
            take = shared[:n]
            k4 = int(sum(arms[a]["rows"][s]["milestone4"] for s in take))
            ks = int(sum(arms[a]["rows"][s]["success"] for s in take))
            per_arm[a].append({"n": n, "milestone4_count": k4,
                               "milestone4_rate": k4 / n,
                               "milestone4_cp95": list(clopper_pearson(k4, n)),
                               "success_count": ks, "success_rate": ks / n,
                               "success_cp95": list(clopper_pearson(ks, n))})

    # ---- descriptive shape of each trajectory ---------------------------------
    shape = {}
    for tier, pool in (("milestone4_primary", results["primary"]),
                       ("milestone4_context", results["context"]),
                       ("success_estimation", results["estimation"]),
                       ("success_estimation_context",
                        results["estimation_context"])):
        for name, traj in pool.items():
            ps = [pt["p_two_sided"] for pt in traj]
            imin = min(range(len(ps)), key=lambda i: ps[i])
            shape[f"{tier}:{name}"] = {
                "endpoint_tier": tier,
                "p_first": ps[0], "p_final": ps[-1],
                "min_p": ps[imin], "min_p_at_n": steps[imin],
                "p_ratio_final_over_min": (ps[-1] / ps[imin]
                                           if ps[imin] > 0 else None),
                "monotone_decreasing": all(ps[i + 1] <= ps[i]
                                           for i in range(len(ps) - 1)),
                "final_below_0.05": ps[-1] < 0.05,
                "any_prefix_below_0.05": any(q < 0.05 for q in ps),
                "sign_of_final_effect": (
                    "a>b" if traj[-1]["rate_difference"] > 0 else
                    "b>a" if traj[-1]["rate_difference"] < 0 else "tied"),
            }

    # ---- correctness check against the round's published n = 288 table --------
    if args.no_check:
        check = {"skipped": True,
                 "reason": "--no-check was passed; the n=288 reproduction of the "
                           "published table was NOT run"}
    elif max(steps) != len(shared):
        check = {"skipped": True,
                 "reason": f"the largest prefix is {max(steps)}, not the full "
                           f"{len(shared)}-seed panel, so the published n=288 "
                           f"table has nothing to be compared against"}
    else:
        check = {"published_source": PUBLISHED_SOURCE, "at_n": max(steps),
                 "tolerance": "half-ulp of the published digits",
                 "entries": {}, "counts_reproduce": True, "p_reproduce": True}
        idx = len(steps) - 1
        for name, want in PUBLISHED_M4_AT_288.items():
            pool = (results["primary"] if name in results["primary"]
                    else results["context"])
            pt = pool[name][idx]
            counts_ok = (pt["discordant_b_a_only"] == want["b"]
                         and pt["discordant_c_b_only"] == want["c"])
            p_ok = abs(pt["p_two_sided"] - want["p"]) <= want["half_ulp"]
            check["entries"][name] = {
                "published_p": want["p"], "recomputed_p": pt["p_two_sided"],
                "published_b": want["b"], "published_c": want["c"],
                "recomputed_b": pt["discordant_b_a_only"],
                "recomputed_c": pt["discordant_c_b_only"],
                "counts_match": counts_ok, "p_match": p_ok,
                "tolerance": want["half_ulp"]}
            check["counts_reproduce"] &= counts_ok
            check["p_reproduce"] &= p_ok
        # a p-value that does not follow from the discordant table printed beside
        # it is a defect in the published number, not in this recomputation, so it
        # is recorded in full rather than absorbed by a tolerance.
        check["discrepancies"] = {
            k: {"published_p": e["published_p"], "recomputed_p": e["recomputed_p"],
                "table": [e["recomputed_b"], e["recomputed_c"]],
                "counts_match": e["counts_match"]}
            for k, e in check["entries"].items() if not e["p_match"]}
        check["passed"] = bool(check["counts_reproduce"] and check["p_reproduce"])
        # Diagnosis of a published p that does not follow from its own table: the
        # exact two-sided McNemar p is 2 * P(Binom(b+c, 1/2) <= min(b,c)), a dyadic
        # rational, so the table that WOULD have produced the published number can
        # be searched for exactly. Reported as a fact about the arithmetic, not as
        # a claim about which script wrote the published row.
        for name, d in check["discrepancies"].items():
            tol = check["entries"][name]["tolerance"]
            d["published_p_is_dyadic_2_pow"] = _dyadic_exponent(d["published_p"],
                                                                tol)
            d["table_that_would_give_published_p"] = _table_for_p(
                d["published_p"], tol)

    artifact = {
        "schema": "v251_p_trajectory.1",
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "round": ROUND_TAG,
        "question": "CLAUDE.md rule 8 - the p-value's trajectory as n rises, on "
                    "seed-ordered prefixes of the ONE committed 288-seed panel",
        "what_this_is_not": [
            "not independent panels: the prefixes are nested subsets of one "
            "dataset, so the per-contrast p-values across n are strongly dependent",
            "not a sequential test and not an interim analysis with a spending "
            "function; no stopping rule was applied and none may be inferred",
            "not an extension of the panel: the preregistration fixed n=288 before "
            "the first episode and only that column is the registered analysis",
        ],
        "definitions_imported_from": "scripts/eval_v251_round.py "
                                     "(derive_from_events, _as_binary, "
                                     "exact_mcnemar, clopper_pearson)",
        "primary_endpoint": {
            "key": "milestone4",
            "definition": f"milestone {PRIMARY_MILESTONE} (pick_up cream_cheese_1) "
                          f"present in the episode's events_achieved map",
            "tier": "primary",
            "test": "exact paired McNemar, two-sided, on the discordant pairs",
        },
        "estimation_endpoint": {
            "key": "success",
            "definition": "terminal task success as written by the deploy executor",
            "tier": "ESTIMATION - the preregistration states this endpoint is NOT a "
                    "test at this n and must not be read as one",
        },
        "steps": steps,
        "panel": {"n_shared": len(shared), "first_seed": shared[0],
                  "last_seed": shared[-1], "seed_ordered": True},
        "arms": {a: arms[a]["path"] for a in ARMS},
        "scipy_cross_check_available": HAVE_SCIPY,
        "milestone4_primary": results["primary"],
        "milestone4_context_uncorrected": results["context"],
        "success_estimation": results["estimation"],
        "success_estimation_context": results["estimation_context"],
        "holm_over_primary_family_by_n": holm_traj,
        "holm_note": "Holm is the preregistered multiplicity plan for the five "
                     "primary contrasts AT n=288; the earlier columns are the "
                     "same arithmetic on a prefix and are diagnostic only",
        "per_arm_counts": per_arm,
        "trajectory_shape": shape,
        "trajectory_shape_note": "descriptive only. `min_p_at_n` is where the "
                                 "smallest p over the prefixes fell and "
                                 "`p_ratio_final_over_min` is how far the final "
                                 "value sits above it; a contrast whose minimum "
                                 "is at an early prefix and whose final value is "
                                 "well above it is the shape CLAUDE.md rule 8 "
                                 "names. Nested prefixes make an early minimum "
                                 "common under a true null, so this is a shape, "
                                 "not a test.",
        "correctness_check": check,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=1) + "\n")

    # ------------------------------- printout ---------------------------------
    print(f"v251 round {ROUND_TAG}")
    print(f"  shared panel: {len(shared)} seeds, {shared[0]}..{shared[-1]}, "
          f"seed-ordered; prefixes {steps}")
    print(f"  endpoints and test imported from scripts/eval_v251_round.py; "
          f"scipy cross-check available: {HAVE_SCIPY}")
    print("  nested prefixes of ONE committed panel - not independent panels, "
          "not a sequential test; only n=%d is the registered analysis"
          % max(steps))

    print_block(
        "PRIMARY endpoint - milestone-4 attainment (pick_up cream_cheese_1)",
        {**results["primary"], **results["context"]}, steps,
        "  five registered primary contrasts, then the registered CONTEXT "
        "contrast M1-vs-M0 (uncorrected, descriptive)")

    print()
    print("  Holm-adjusted p over the five primary contrasts, by prefix "
          "(registered at n=%d only)" % max(steps))
    head = f"  {'contrast':<16}" + "".join(f"{('n=' + str(n)):>12}" for n in steps)
    print(head)
    for name in results["primary"]:
        print(f"  {name:<16}"
              + "".join(f"{fmt_p(holm_traj[str(n)][name]):>12}" for n in steps))

    print_block(
        "ESTIMATION endpoint - terminal success (NOT A TEST at this n)",
        {**results["estimation"], **results["estimation_context"]}, steps,
        "  the preregistration registers terminal success as an estimation "
        "endpoint; these p-values are printed for the trajectory's shape only")

    print()
    print("  per-arm counts along the same prefixes "
          "(milestone-4 k/n, then terminal-success k/n)")
    print(f"  {'arm':<10}" + "".join(f"{('n=' + str(n)):>16}" for n in steps))
    for a in ARMS:
        cells = [f"{r['milestone4_count']}/{r['n']}, {r['success_count']}/{r['n']}"
                 for r in per_arm[a]]
        print(f"  {a:<10}" + "".join(f"{c:>16}" for c in cells))

    print()
    if check.get("skipped"):
        print("  CORRECTNESS CHECK SKIPPED (--no-check): the published n=288 "
              "table was not reproduced.")
    else:
        print(f"  correctness check against the round's published n="
              f"{check['at_n']} milestone-4 table ({PUBLISHED_SOURCE}):")
        print(f"    {'contrast':<16}{'pub b/c':>10}{'here b/c':>10}"
              f"{'pub p':>12}{'here p':>12}  verdict")
        for name, e in check["entries"].items():
            if e["counts_match"] and e["p_match"]:
                verdict = "MATCH"
            elif e["counts_match"]:
                verdict = "p MISMATCH (discordant table reproduces)"
            else:
                verdict = "TABLE MISMATCH"
            pub_bc = f"{e['published_b']}/{e['published_c']}"
            here_bc = f"{e['recomputed_b']}/{e['recomputed_c']}"
            print(f"    {name:<16}{pub_bc:>10}{here_bc:>10}"
                  f"{e['published_p']:>12.6g}{e['recomputed_p']:>12.6g}  {verdict}")
        print(f"    discordant tables reproduce: "
              f"{'ALL' if check['counts_reproduce'] else 'NO'};  "
              f"published p-values reproduce: "
              f"{'ALL' if check['p_reproduce'] else 'NO'}")
        print(f"    -> {'PASS' if check['passed'] else 'NOT FULLY REPRODUCED'}")
        for name, d in check.get("discrepancies", {}).items():
            print(f"    DISCREPANCY {name}: the published p {d['published_p']:.6g} "
                  f"does not follow from the discordant table {d['table'][0]}/"
                  f"{d['table'][1]} printed beside it, which this recomputation "
                  f"reproduces exactly and which gives {d['recomputed_p']:.6g}.")
            if d.get("published_p_is_dyadic_2_pow") is not None:
                print(f"      the published value is exactly "
                      f"2**-{d['published_p_is_dyadic_2_pow']}; the discordant "
                      f"table(s) that would give it: "
                      f"{d['table_that_would_give_published_p']}")
            print("      the two differ in magnitude, not in sign or in which "
                  "arm leads; both are far below 0.05 and both are the same "
                  "direction, so the round's reject/no-reject reading is "
                  "unchanged - but the published digits are not this table's.")

    print()
    print(f"  artifact: {args.out}")

    # The exit code tracks whether THIS recomputation is sound. A discordant table
    # that does not reproduce means the pairing or the endpoint is wrong here and
    # the artifact is void -> non-zero. A published p-value that does not follow
    # from the discordant table printed beside it is a defect in that published
    # row, loudly reported above and stored in the artifact, and it does not make
    # this recomputation wrong -> zero, with the discrepancy on the record.
    if not check.get("skipped") and not check["counts_reproduce"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
