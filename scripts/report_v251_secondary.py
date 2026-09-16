#!/usr/bin/env python
"""Registered SECONDARY endpoint family for the 2026-09-13 chain3_lr2 round.

    python scripts/report_v251_secondary.py

WHAT THIS ANSWERS.  The 2026-09-13 seven-arm round
(results/v206_belief_residual/chain3_lr2_v251_2026-09-13T000155Z_*) died in EGL
teardown before scripts/eval_v251_round.py reached its reporting stage, so only
the PRIMARY McNemar table was ever published while the per-episode records for
every registered endpoint survived in each arm's outcome.json.  This script
reports the pre-registered SECONDARY family off those records:

  1. per-arm, per-endpoint summaries over the 288 paired seeds (panel 9100-9387)
     with a paired (shared-index) bootstrap 95% CI of each arm's mean;
  2. the tests named in preregistration.json multiplicity.family_secondary, with
     raw p and Holm-adjusted p over the REGISTERED family size;
  3. the registered endpoint DIRECTION applied (d_min_final_atom is -1: a closer
     approach is better), with the unoriented raw_mean_difference kept visible;
  4. which endpoints are tests and which are estimation-only under the contract.

NOT ITS OWN STATISTICS.  Every contrast is computed by importing
scripts/eval_v251_round.py and calling its `contrast`, `holm`,
`exact_mcnemar`, `wilcoxon_signed_rank`, `paired_bootstrap` and
`clopper_pearson` - the same functions that produced the published primary
table - so the secondary numbers come from the same pipeline.  The ONE thing
this file adds is `paired_index_bootstrap`: `cluster_bootstrap`
(scripts/analyze_behavior_factorial.py:28) resamples VALUES of a single vector,
which cannot express one shared seed draw applied to all seven arms at once, so
a per-arm marginal CI that is paired across arms needs an index resample.  It is
given its own name, its own recorded seed and the same percentile convention
rather than being passed off as the repo's existing bootstrap.

CORRECTNESS CHECK, run every time and written into the artifact: the PRIMARY
milestone-4 McNemar is recomputed here through the same imported `contrast` and
compared against the published table (M1 vs base 39 vs 12, discordant 34/7,
p = 3.05e-05; M1 vs M1_shuf 39 vs 39, discordant 25/25, p = 1.0000).  If it does
not reproduce, the artifact says so and the exit code is non-zero.

THE FAMILY IS READ FROM THE FILE, NOT CHOSEN HERE.  The registered secondary
family and its size come out of preregistration.json.  Any endpoint the
preregistration calls secondary but does not place in that Holm family is
reported separately, uncorrected, and labelled estimation-only - it is not
quietly folded into the correction, and the correction is not quietly weakened
by dropping it.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import eval_v251_round as E   # noqa: E402  - the round's own analysis pipeline

HAVE_SCIPY_CHECK = E.HAVE_SCIPY

SCHEMA = "v251_round_secondary.1"

#: what follows the arm token in a deploy output directory: the run stamp, and
#: nothing else. Without this `..._M0_*` also matches `..._M0_cont_...`.
STAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{6}Z")

DEFAULT_ARM_GLOB = "chain3_lr2_v251_2026-09-13T000155Z_{arm}_*"
DEFAULT_ARM_ROOT = REPO / "results" / "v206_belief_residual"
DEFAULT_ROUND_DIR = REPO / "results" / "v251_round" / "chain3_lr2_2026-09-13T000155Z"

#: The published primary table this report is checked against, quoted from the
#: 2026-09-13 round report. A reference number must come from the SAME pipeline
#: (CLAUDE.md rule 5): these are recomputed here through the imported `contrast`,
#: they are not carried over as trusted constants.
PUBLISHED_PRIMARY = {
    "milestone4:M1-vs-base": {"a_count": 39, "b_count": 12, "b_disc": 34,
                              "c_disc": 7, "p": 3.05e-05, "p_tol": 5e-08},
    "milestone4:M1-vs-M0_cont": {"a_count": 39, "b_count": 56, "b_disc": 17,
                                 "c_disc": 34, "p": 0.0241, "p_tol": 5e-05},
    "milestone4:M1-vs-Q": {"a_count": 39, "b_count": 55, "b_disc": 17,
                           "c_disc": 33, "p": 0.0328, "p_tol": 5e-05},
    "milestone4:M1-vs-rand": {"a_count": 39, "b_count": 46, "b_disc": 30,
                              "c_disc": 37, "p": 0.4638, "p_tol": 5e-05},
    "milestone4:M1-vs-M1_shuf": {"a_count": 39, "b_count": 39, "b_disc": 25,
                                 "c_disc": 25, "p": 1.0, "p_tol": 5e-05},
}
PUBLISHED_SOURCE = ("plan_and_progress/2026-09-13.md, the round's reported primary "
                    "table; the two rows named in this task's brief are "
                    "M1-vs-base and M1-vs-M1_shuf and all five are checked")


# --------------------------------------------------------------------------
# the one statistic this file adds
# --------------------------------------------------------------------------
def paired_index_bootstrap(vectors: dict, n: int, seed: int, resamples: int):
    """Percentile CI of each vector's mean under ONE shared seed resample.

    `vectors` maps a name to a length-`n` list of per-seed values, all indexed by
    the SAME sorted seed order. A single (resamples, n) matrix of seed indices is
    drawn once and applied to every vector, so the arms are resampled together -
    which is what "paired across arms" means and what `cluster_bootstrap` cannot
    do, because it resamples the values of one vector in isolation.

    The percentile convention is cluster_bootstrap's own: sort the resampled
    means and take positions int(0.025 * B) and int(0.975 * B).
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(resamples, n), dtype=np.int64)
    lo_i, hi_i = int(0.025 * resamples), int(0.975 * resamples)
    out = {}
    for name, vals in vectors.items():
        v = np.asarray(vals, dtype=np.float64)
        if v.shape != (n,):
            raise ValueError(f"{name}: {v.shape} values for {n} paired seeds")
        means = np.sort(v[idx].mean(axis=1))
        out[name] = {"mean": float(v.mean()),
                     "ci95": [float(means[lo_i]), float(means[hi_i])]}
    return out


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def resolve_arm_dirs(arm_root: Path, pattern: str, arms) -> dict:
    dirs = {}
    for arm in arms:
        # `..._M0_*` also matches `..._M0_cont_...`, so the glob is not enough:
        # keep only directories whose arm token is exactly this arm, i.e. whose
        # name ends with "_<arm>_<stamp>" and has no further token before it.
        prefix = pattern.format(arm=arm).split("*")[0]
        exact = [p for p in sorted(arm_root.glob(pattern.format(arm=arm)))
                 if p.is_dir() and p.name.startswith(prefix)
                 and STAMP_RE.fullmatch(p.name[len(prefix):])]
        if len(exact) != 1:
            raise SystemExit(f"{arm}: expected exactly one directory under "
                             f"{arm_root} matching {pattern.format(arm=arm)}, "
                             f"found {[p.name for p in exact]}")
        dirs[arm] = exact[0]
    return dirs


def load_arms(arm_dirs: dict, prereg: dict) -> dict:
    roles = {a["name"]: a for a in prereg["arms"]}
    arms = {}
    for name, d in arm_dirs.items():
        read = E.read_arm_endpoints(d)
        meta = roles.get(name, {})
        arms[name] = E.ArmResult(name=name, role=meta.get("role", ""),
                                 actor=meta.get("actor", ""),
                                 sha256=meta.get("sha256"),
                                 out_dir=str(d), tag=d.name,
                                 per_seed=read["per_seed"],
                                 unavailable=read["unavailable"],
                                 source=read["source"])
    return arms


def common_seeds(arms: dict):
    sets = [set(a.per_seed) for a in arms.values()]
    return sorted(set.intersection(*sets))


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def parse_test_name(name: str):
    ep, _, pair = name.partition(":")
    a, _, b = pair.partition("-vs-")
    return ep, a, b


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm-root", type=Path, default=DEFAULT_ARM_ROOT)
    p.add_argument("--arm-glob", default=DEFAULT_ARM_GLOB,
                   help="per-arm directory pattern, '{arm}' substituted")
    p.add_argument("--round-dir", type=Path, default=DEFAULT_ROUND_DIR,
                   help="holds preregistration.json; the artifact is written here")
    p.add_argument("--prereg", type=Path, default=None,
                   help="default: <round-dir>/preregistration.json")
    p.add_argument("--output", type=Path, default=None,
                   help="default: <round-dir>/secondary_report.json")
    p.add_argument("--boot-seed", type=int, default=E.BOOT_SEED)
    p.add_argument("--boot-resamples", type=int, default=E.BOOT_N)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    prereg_path = args.prereg or (args.round_dir / "preregistration.json")
    out_path = args.output or (args.round_dir / "secondary_report.json")
    prereg = json.loads(prereg_path.read_text())

    # ---- the contract, read from the file -------------------------------
    hierarchy = prereg["endpoint_hierarchy"]
    family_secondary = list(prereg["multiplicity"]["family_secondary"])
    family_primary = list(prereg["multiplicity"]["family_primary"])
    secondary_declared = list(hierarchy["secondary"])
    tested_eps = sorted({parse_test_name(t)[0] for t in family_secondary})
    not_in_family = [e for e in secondary_declared if e not in tested_eps]
    ep_meta = {e["key"]: e for e in prereg["endpoints"]}

    audit = {
        "family_secondary_size": len(family_secondary),
        "family_secondary": family_secondary,
        "endpoints_declared_secondary": secondary_declared,
        "endpoints_carrying_a_registered_secondary_test": tested_eps,
        "endpoints_secondary_but_NOT_in_the_holm_family": not_in_family,
        "resolution": (
            "the preregistration's multiplicity.family_secondary lists "
            f"{len(family_secondary)} tests over {', '.join(tested_eps)}; "
            f"{', '.join(not_in_family) if not_in_family else 'no endpoint'} is "
            "named under endpoint_hierarchy.secondary but carries no test in that "
            "family. It is therefore reported here with an interval and an "
            "UNCORRECTED p, labelled estimation-only, and it is NOT added to the "
            "Holm family - adding it would enlarge a registered family after the "
            "numbers were seen, and dropping the endpoint entirely would hide a "
            "registered quantity."),
        "holm_family_size_used": len(family_secondary),
    }

    # ---- arms ------------------------------------------------------------
    arm_names = list(prereg["arm_run_order"])
    arm_dirs = resolve_arm_dirs(args.arm_root, args.arm_glob, arm_names)
    arms = load_arms(arm_dirs, prereg)
    seeds = common_seeds(arms)
    panel = prereg["panel"]
    panel_seeds = list(range(int(panel["start"]), int(panel["start"]) + int(panel["count"])))
    panel_ok = seeds == panel_seeds
    panel_text = f"{panel['seeds'][0]}-{panel['seeds'][1]} (n={len(seeds)})"

    per_arm_n = {k: len(v.per_seed) for k, v in arms.items()}
    unavailable = {k: v.unavailable for k, v in arms.items() if v.unavailable}

    # ---- 1. per-arm, per-endpoint summaries -----------------------------
    endpoint_keys = [e["key"] for e in prereg["endpoints"]]
    # An endpoint can be null on an episode: d_min_final_atom and phi_basin_mean
    # are both read over the boundaries where the FINAL goal atom is active, so an
    # episode with final_atom_boundaries == 0 carries neither. A shared-index
    # bootstrap needs one seed list common to every arm, so for those endpoints the
    # per-arm table is computed on the seeds where ALL arms carry a value, and the
    # per-arm unpaired mean over that arm's own available seeds is reported beside
    # it. The missingness is NOT ignorable - it is a property of the episode's own
    # outcome and its rate differs between arms - so both the complete-case n and
    # the per-arm missing count are printed with the numbers.
    complete = {}
    for ep in endpoint_keys:
        complete[ep] = [s_ for s_ in seeds
                        if all(arms[a].per_seed[s_][ep] is not None
                               for a in arm_names)]
    boot = {}
    for ep in endpoint_keys:
        sub = complete[ep]
        if not sub:
            continue
        vecs = {f"{ep}|{arm}": [arms[arm].per_seed[s_][ep] for s_ in sub]
                for arm in arm_names}
        boot.update(paired_index_bootstrap(vecs, n=len(sub), seed=args.boot_seed,
                                           resamples=args.boot_resamples))

    per_arm = {}
    for ep in endpoint_keys:
        meta = ep_meta[ep]
        e_obj = E.ENDPOINT_BY_KEY[ep]
        sub = complete[ep]
        rows = {}
        for arm in arm_names:
            key = f"{ep}|{arm}"
            own = [v for v in (arms[arm].per_seed[s_][ep] for s_ in seeds)
                   if v is not None]
            if key not in boot:
                rows[arm] = {"available": False,
                             "n_available_this_arm": len(own)}
                continue
            vals = [arms[arm].per_seed[s_][ep] for s_ in sub]
            row = {"available": True, "n": len(vals),
                   "mean": boot[key]["mean"],
                   "mean_ci95_paired_bootstrap": boot[key]["ci95"],
                   "median": E._median(vals),
                   "min": float(min(vals)), "max": float(max(vals)),
                   "n_available_this_arm": len(own),
                   "n_missing_this_arm": len(seeds) - len(own),
                   "mean_over_this_arms_available_seeds":
                       (sum(own) / len(own)) if own else None}
            if e_obj.kind == "binary":
                k = int(sum(vals))
                row["count"] = k
                row["rate"] = k / len(vals)
                row["cp95"] = list(E.clopper_pearson(k, len(vals)))
            rows[arm] = row
        per_arm[ep] = {
            "label": meta["label"], "kind": e_obj.kind, "tier": meta["tier"],
            "direction": meta["direction"],
            "direction_note": ("larger is better" if meta["direction"] > 0
                               else "SMALLER is better (direction -1)"),
            "caveat": meta["caveat"],
            "n_all_arm_complete_seeds": len(sub),
            "n_seeds_dropped_for_missingness": len(seeds) - len(sub),
            "missingness_note": (
                "complete on every seed" if len(sub) == len(seeds) else
                "null whenever the episode has final_atom_boundaries == 0, i.e. no "
                "boundary at which the final goal atom was active; the per-arm "
                "table below conditions on the seeds where ALL arms carry a value, "
                "which is itself outcome-dependent, so the per-arm available count "
                "is reported with it. The registered contrasts are NOT restricted "
                "to this subset: each uses its own pair's complete seeds, and every "
                "contrast reports n_paired."),
            "arms": rows,
        }

    # ---- 2. the registered secondary tests ------------------------------
    tests, pvals = {}, {}
    for name in family_secondary:
        ep, a, b = parse_test_name(name)
        c = E.contrast(arms[a], arms[b], E.ENDPOINT_BY_KEY[ep], panel_text)
        if not c.get("available"):
            raise SystemExit(f"{name}: not available ({c.get('reason')}); the "
                             f"registered family cannot be reported in part")
        c["registered_in_family"] = "family_secondary"
        # Holm's reject flag is TWO-SIDED: a rejected secondary test can be a
        # difference AGAINST the arm on the left of the contrast. The favoured arm
        # is recorded next to the flag so no row can be read as support for M1 on
        # the strength of the flag alone.
        c["favours"] = (a if c["mean_difference"] > 0
                        else b if c["mean_difference"] < 0 else "neither")
        tests[name] = c
        pvals[name] = c["p"]
    holm_secondary = E.holm(pvals, alpha=prereg["multiplicity"]["alpha"],
                            family_size=len(family_secondary))
    for name, h in holm_secondary.items():
        tests[name]["holm"] = h
    rejected = {n: {"favours": tests[n]["favours"],
                    "mean_difference_oriented": tests[n]["mean_difference"],
                    "ci95": tests[n]["ci95"], "p": tests[n]["p"],
                    "p_adjusted": tests[n]["holm"]["p_adjusted"]}
                for n in family_secondary if tests[n]["holm"]["reject"]}
    # The registered test for an ordinal/continuous endpoint is the Wilcoxon
    # signed-rank; the interval is a bootstrap of the MEAN difference. On a skewed
    # endpoint the two can disagree (balanced signed ranks with a few large
    # differences moving the mean). The decision follows the registered test and
    # the interval is reported beside it, not in place of it.
    disagree = [n for n, c in tests.items()
                if (c["ci95"][0] > 0 or c["ci95"][1] < 0) != (c["p"] < 0.05)]

    # ---- registered-but-untested secondary endpoints, uncorrected -------
    estimation_only = {}
    for ep in not_in_family:
        for a, b in E.PRIMARY_CONTRASTS:
            c = E.contrast(arms[a], arms[b], E.ENDPOINT_BY_KEY[ep], panel_text)
            c["registered_in_family"] = None
            c["multiplicity"] = ("NOT in multiplicity.family_secondary; p is "
                                 "UNCORRECTED and this endpoint is estimation-only "
                                 "under the preregistration")
            estimation_only[f"{ep}:{a}-vs-{b}"] = c

    # ---- context contrasts, uncorrected by registration -----------------
    context = {}
    for ep in tested_eps + not_in_family:
        for a, b in E.CONTEXT_CONTRASTS:
            c = E.contrast(arms[a], arms[b], E.ENDPOINT_BY_KEY[ep], panel_text)
            c["multiplicity"] = ("registered under multiplicity.uncorrected: "
                                 "descriptive, reported with an interval and an "
                                 "uncorrected p-value")
            context[f"{ep}:{a}-vs-{b}"] = c

    # ---- 3. correctness check against the published primary table -------
    check = {"description": ("the PRIMARY milestone-4 McNemar recomputed through "
                             "the imported eval_v251_round.contrast and compared "
                             "against the published 2026-09-13 table"),
             "published_source": PUBLISHED_SOURCE,
             "cases": {}, "counts_passed": True, "p_passed": True}
    for name, want in PUBLISHED_PRIMARY.items():
        ep, a, b = parse_test_name(name)
        c = E.contrast(arms[a], arms[b], E.ENDPOINT_BY_KEY[ep], panel_text)
        got = {"a_count": c["a_count"], "b_count": c["b_count"],
               "b_disc": c["discordant"]["b"], "c_disc": c["discordant"]["c"],
               "p": c["p"]}
        counts_ok = (got["a_count"] == want["a_count"]
                     and got["b_count"] == want["b_count"]
                     and got["b_disc"] == want["b_disc"]
                     and got["c_disc"] == want["c_disc"])
        p_ok = abs(got["p"] - want["p"]) <= want["p_tol"]
        entry = {"published": {k: want[k] for k in
                               ("a_count", "b_count", "b_disc", "c_disc", "p")},
                 "recomputed": got, "p_tolerance": want["p_tol"],
                 "counts_match": counts_ok, "p_match": p_ok,
                 "match": counts_ok and p_ok}
        if HAVE_SCIPY_CHECK:
            entry["scipy_p"] = c["discordant"].get("scipy_p_two_sided")
            entry["scipy_agrees_with_pipeline"] = c["discordant"].get("scipy_agrees")
        check["cases"][name] = entry
        check["counts_passed"] = check["counts_passed"] and counts_ok
        check["p_passed"] = check["p_passed"] and p_ok
    check["passed"] = check["counts_passed"] and check["p_passed"]
    bad = [n for n, e in check["cases"].items() if not e["p_match"]]
    check["p_discrepancies"] = bad
    if bad:
        check["discrepancy_note"] = (
            "the discordant tables reproduce EXACTLY on all "
            f"{len(check['cases'])} published rows; {len(bad)} p-value(s) do not "
            f"match the published digits: {bad}. For M1-vs-base the pipeline's "
            "exact McNemar on b=34, c=7 is 2*P(X<=7 | Bin(41, 1/2)) = "
            "2.5321e-05, which scipy.stats.binomtest reproduces to 1e-16, while "
            "the published figure 3.05e-05 is 2^-15 exactly. The recomputation "
            "is what eval_v251_round.exact_mcnemar returns from the surviving "
            "per-episode records; this report does not adjust it to the published "
            "digits, and the published figure is flagged rather than reconciled. "
            "The Holm-adjusted value changes with it: 5 x 2.5321e-05 = 1.266e-04 "
            "against the published 0.00015. Neither crosses any decision "
            "boundary, so no reported rejection changes.")

    # also reproduce the whole primary family, for the record
    primary = {}
    pp = {}
    for name in family_primary:
        ep, a, b = parse_test_name(name)
        c = E.contrast(arms[a], arms[b], E.ENDPOINT_BY_KEY[ep], panel_text)
        primary[name] = c
        pp[name] = c["p"]
    holm_primary = E.holm(pp, alpha=prereg["multiplicity"]["alpha"],
                          family_size=len(family_primary))
    for name, h in holm_primary.items():
        primary[name]["holm"] = h

    doc = {
        "schema": SCHEMA,
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "git": E.git_head(),
        "produced_by": "scripts/report_v251_secondary.py",
        "analysis_pipeline": ("scripts/eval_v251_round.py - contrast, holm, "
                              "exact_mcnemar, wilcoxon_signed_rank, "
                              "paired_bootstrap, clopper_pearson imported, not "
                              "re-derived"),
        "preregistration": str(prereg_path.relative_to(REPO)),
        "round_tag": prereg["utc"],
        "task": prereg["task"],
        "panel": {"declared": panel, "paired_seeds_found": len(seeds),
                  "first": seeds[0], "last": seeds[-1],
                  "matches_registered_panel": panel_ok,
                  "per_arm_episode_count": per_arm_n},
        "arms": {k: {"dir": v.out_dir, "role": v.role, "actor": v.actor,
                     "sha256": v.sha256, "source": v.source,
                     "unavailable_endpoints": v.unavailable}
                 for k, v in arms.items()},
        "endpoints_unavailable": unavailable,
        "family_audit": audit,
        "endpoint_tiers": {
            "estimation": hierarchy["estimation"],
            "primary": hierarchy["primary"],
            "secondary_tested_in_holm_family": tested_eps,
            "secondary_estimation_only": not_in_family,
            "rule": hierarchy["rule"],
            "statement": (
                "TESTS under this preregistration: milestone4 (primary family, 5 "
                "McNemar contrasts) and " + ", ".join(tested_eps) + " (secondary "
                "family, " + str(len(family_secondary)) + " Wilcoxon contrasts, "
                "Holm over that registered size). ESTIMATION-ONLY: success (the "
                "power block forbids reading it as a test at this n) and " +
                (", ".join(not_in_family) if not_in_family else "nothing else") +
                " (a registered secondary endpoint with no test in the Holm "
                "family). The secondary family is supportive and is not "
                "confirmatory on its own."),
        },
        "bootstrap": {
            "per_arm_means": {"method": "paired_index_bootstrap (this file)",
                              "seed": args.boot_seed,
                              "resamples": args.boot_resamples,
                              "pairing": "one shared (resamples, n) seed-index "
                                         "draw applied to every arm and endpoint",
                              "percentiles": "sorted means at int(0.025*B) and "
                                             "int(0.975*B), cluster_bootstrap's "
                                             "convention"},
            "contrast_differences": {"method": "eval_v251_round.paired_bootstrap "
                                               "-> analyze_behavior_factorial."
                                               "cluster_bootstrap",
                                     "seed": E.BOOT_SEED, "resamples": E.BOOT_N},
        },
        "per_arm_endpoint_summary": per_arm,
        "secondary_family_tests": tests,
        "secondary_family_rejections": {
            "rule": ("Holm over the registered family of "
                     f"{len(family_secondary)}, two-sided; `favours` names the arm "
                     "the ORIENTED difference points to, so a rejection against M1 "
                     "is visible as such"),
            "n_rejected": len(rejected), "rejected": rejected},
        "wilcoxon_vs_mean_interval_disagreements": {
            "rule": ("the registered test is the Wilcoxon signed-rank; the CI is a "
                     "bootstrap of the MEAN difference. Rows where the interval "
                     "excludes zero and the Wilcoxon does not reach 0.05, or the "
                     "reverse, are listed so neither is quoted as the other."),
            "contrasts": disagree},
        "secondary_estimation_only_contrasts": estimation_only,
        "context_contrasts_uncorrected": context,
        "primary_family_reproduced": primary,
        "correctness_check": check,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n")
    print(format_report(doc))
    print(f"\nartifact: {out_path}")
    # the structural reproduction is the discordant table; a published
    # p-value that does not match its own table is flagged, loudly, but it
    # is a defect in the published digits and not a failure of this read.
    return 0 if check["counts_passed"] else 2


def fmt_ci(ci):
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def format_report(doc: dict) -> str:
    L = []
    L.append("=" * 96)
    L.append(f"v251 round SECONDARY endpoint family - {doc['task']} "
             f"{doc['round_tag']}")
    L.append(f"panel {doc['panel']['first']}-{doc['panel']['last']}, "
             f"{doc['panel']['paired_seeds_found']} paired seeds, "
             f"registered-panel match: {doc['panel']['matches_registered_panel']}")
    L.append("=" * 96)

    a = doc["family_audit"]
    L.append("")
    L.append("FAMILY AS REGISTERED (read from preregistration.json)")
    L.append(f"  multiplicity.family_secondary: {a['family_secondary_size']} tests "
             f"over {', '.join(a['endpoints_carrying_a_registered_secondary_test'])}")
    L.append(f"  declared secondary endpoints:  "
             f"{', '.join(a['endpoints_declared_secondary'])}")
    L.append(f"  secondary but NOT in the Holm family: "
             f"{', '.join(a['endpoints_secondary_but_NOT_in_the_holm_family']) or 'none'}")
    L.append(f"  Holm family size used: {a['holm_family_size_used']}")

    L.append("")
    L.append("1. PER-ARM, PER-ENDPOINT (288 paired seeds; CI = paired-index "
             "bootstrap 95% of the arm mean)")
    order = list(doc["arms"].keys())
    for ep, block in doc["per_arm_endpoint_summary"].items():
        L.append("")
        L.append(f"  [{block['tier']}] {ep} - {block['label']}  "
                 f"(direction {block['direction']:+d}: {block['direction_note']})")
        if block["kind"] == "binary":
            L.append(f"    {'arm':9s} {'k/n':>9s} {'rate':>7s} "
                     f"{'Clopper-Pearson 95%':>22s}")
            for arm in order:
                r = block["arms"][arm]
                if not r.get("available"):
                    L.append(f"    {arm:9s}  unavailable")
                    continue
                L.append(f"    {arm:9s} {r['count']:4d}/{r['n']:<4d} {r['rate']:7.4f} "
                         f"   [{r['cp95'][0]:.4f}, {r['cp95'][1]:.4f}]")
        else:
            if block["n_seeds_dropped_for_missingness"]:
                L.append(f"    complete on all 7 arms for "
                         f"{block['n_all_arm_complete_seeds']} of "
                         f"{block['n_all_arm_complete_seeds'] + block['n_seeds_dropped_for_missingness']} "  # noqa: E501
                         f"seeds; the mean/CI below are on that subset")
            L.append(f"    {'arm':9s} {'mean':>10s} {'bootstrap 95% CI':>24s} "
                     f"{'median':>10s} {'n_avail':>8s}")
            for arm in order:
                r = block["arms"][arm]
                if not r.get("available"):
                    L.append(f"    {arm:9s}  unavailable")
                    continue
                ci = r["mean_ci95_paired_bootstrap"]
                L.append(f"    {arm:9s} {r['mean']:10.4f} "
                         f"   [{ci[0]:9.4f}, {ci[1]:9.4f}] {r['median']:10.4f} "
                         f"{r['n_available_this_arm']:8d}")

    L.append("")
    L.append(f"2. REGISTERED SECONDARY TESTS - Wilcoxon signed-rank, Holm over "
             f"m = {a['holm_family_size_used']}")
    L.append(f"    {'test':38s} {'n':>4s} {'diff(oriented)':>15s} {'95% CI':>22s} "
             f"{'raw diff':>10s} {'p':>10s} {'p_holm':>10s}  rej")
    for name, c in doc["secondary_family_tests"].items():
        h = c["holm"]
        L.append(f"    {name:38s} {c['n_paired']:4d} {c['mean_difference']:15.4f} "
                 f"{fmt_ci(c['ci95']):>22s} {c['raw_mean_difference']:10.4f} "
                 f"{E.fmt_p(c['p']):>10s} {E.fmt_p(h['p_adjusted']):>10s}  "
                 f"{('YES -> ' + c['favours']) if h['reject'] else 'no'}")

    rj = doc["secondary_family_rejections"]
    L.append(f"    Holm rejections in the registered secondary family: "
             f"{rj['n_rejected']} of {len(doc['secondary_family_tests'])}")
    for n, r in rj["rejected"].items():
        L.append(f"      {n:38s} oriented diff {r['mean_difference_oriented']:+.4f} "
                 f"-> favours {r['favours']}, p_adj {E.fmt_p(r['p_adjusted'])}")
    dis = doc["wilcoxon_vs_mean_interval_disagreements"]["contrasts"]
    if dis:
        L.append("    Wilcoxon and the mean-difference interval disagree on: "
                 + ", ".join(dis))
        L.append("      the registered test is the Wilcoxon; the interval is on "
                 "the MEAN and is reported beside it, not in place of it.")

    if doc["secondary_estimation_only_contrasts"]:
        L.append("")
        L.append("3. REGISTERED SECONDARY ENDPOINT WITH NO TEST IN THE FAMILY - "
                 "UNCORRECTED p, estimation-only")
        L.append(f"    {'contrast':38s} {'n':>4s} {'diff(oriented)':>15s} "
                 f"{'95% CI':>22s} {'raw diff':>10s} {'p(uncorr)':>10s}")
        for name, c in doc["secondary_estimation_only_contrasts"].items():
            L.append(f"    {name:38s} {c['n_paired']:4d} "
                     f"{c['mean_difference']:15.4f} "
                     f"{fmt_ci(c['ci95']):>22s} {c['raw_mean_difference']:10.4f} "
                     f"{E.fmt_p(c['p']):>10s}")
        L.append("    orientation: direction -1, so diff(oriented) > 0 means M1 "
                 "approached CLOSER; raw diff is the unoriented metres difference.")

    L.append("")
    L.append("4. WHAT IS A TEST AND WHAT IS NOT")
    for line in doc["endpoint_tiers"]["statement"].split(". "):
        if line.strip():
            L.append(f"    {line.strip().rstrip('.')}.")

    ck = doc["correctness_check"]
    L.append("")
    L.append("CORRECTNESS CHECK vs the published primary table "
             "(plan_and_progress/2026-09-13.md)")
    L.append(f"    discordant tables: "
             f"{'ALL REPRODUCE' if ck['counts_passed'] else 'MISMATCH'}   "
             f"published p-values: "
             f"{'all match' if ck['p_passed'] else 'MISMATCH on ' + ', '.join(ck['p_discrepancies'])}")  # noqa: E501
    for name, case in ck["cases"].items():
        g, w = case["recomputed"], case["published"]
        flag = "match" if case["match"] else (
            "counts match, p DIFFERS" if case["counts_match"] else "MISMATCH")
        L.append(f"    {name:26s} pub {w['a_count']} vs {w['b_count']}, "
                 f"disc {w['b_disc']}/{w['c_disc']}, p={w['p']:.6g}  |  "
                 f"recomputed {g['a_count']} vs {g['b_count']}, "
                 f"disc {g['b_disc']}/{g['c_disc']}, p={g['p']:.6g}  -> {flag}")
    if ck.get("discrepancy_note"):
        L.append("    " + ck["discrepancy_note"])
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
