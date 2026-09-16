#!/usr/bin/env python3
"""v253 - restore the project interaction ledger and draw the framework 7.4 main plot.

Two stages, both offline (no environment steps, no GPU, no network):

  Stage A  walk results/ and recover every recorded environment-step count into
           results/v253_ledger/ledger.json (per-run rows, per-date, per-line,
           cumulative series, grand total, reconciliation against the V8 figure).

  Stage B  using that ledger plus the deployment outcomes on disk, draw
           "success against cumulative real interactions" to
           results/v253_ledger/main_plot.{pdf,png} and write every plotted
           number to results/v253_ledger/main_plot_data.json.

Nothing here is hand-entered.  Every number in the outputs is read off a file
under results/ or computed from one.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUT = RESULTS / "v253_ledger"

# The last figure the project's own ledger published: plan_and_progress/2026-08.md
# / framework_design.md 19.3, "Project total across all V8 ledgers: 637,145",
# stated as of Action 2M.5 on 2026-08-25.
V8_REFERENCE_TOTAL = 637_145
V8_REFERENCE_CUTOFF = "2026-08-25T235959Z"
V8_REFERENCE_SOURCE = "plan_and_progress/framework_design.md:1400"

MAX_JSON_BYTES = 3_000_000  # two labels.jsonl files exceed this; neither carries a step count
RUN_STATUS_PREFIXES = ("ABORTED_", "DISCARDED_", "SMOKE_", "HALTED_", "SUPERSEDED_", "VOID_")

STAMP_RE = re.compile(r"(\d{4}-\d{2}-\d{2})T(\d{6})Z")
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
TASK_RE = re.compile(r"(chain\d+b?_lr\d+|loho_t\d+_\w+)")
FAMILY_RE = re.compile(r"^v(\d{3})")


# --------------------------------------------------------------------------- #
# Stage A helpers
# --------------------------------------------------------------------------- #
def parse_stamp(text):
    """'2026-09-13T063530Z' -> datetime, tolerating the dashed variant."""
    if not isinstance(text, str):
        return None
    m = STAMP_RE.search(text)
    if m:
        try:
            return dt.datetime.strptime(m.group(1) + m.group(2), "%Y-%m-%d%H%M%S")
        except ValueError:
            return None
    m = DATE_RE.search(text)
    if m:
        try:
            return dt.datetime.strptime(m.group(1), "%Y-%m-%d")
        except ValueError:
            return None
    return None


def load_json(path: Path):
    try:
        if path.stat().st_size > MAX_JSON_BYTES:
            return None
        with path.open() as fh:
            return json.load(fh)
    except Exception:
        return None


def family_of(rel: str) -> str:
    return rel.split("/")[0]


def line_of(family: str) -> str:
    m = FAMILY_RE.match(family)
    if not m:
        return "pre-v0xx (July/August exploratory, unnumbered)"
    n = int(m.group(1))
    return f"v{n // 100}xx"


def task_of(blob, rel):
    if isinstance(blob, dict):
        t = blob.get("task")
        if isinstance(t, str):
            return t
        pt = blob.get("per_task")
        if isinstance(pt, dict) and pt:
            return "+".join(sorted(pt))
    m = TASK_RE.search(rel)
    return m.group(1) if m else None


def run_status(rel: str) -> str:
    for part in rel.split("/"):
        for pref in RUN_STATUS_PREFIXES:
            if part.startswith(pref):
                return pref.rstrip("_").lower()
        if part.endswith("_dryrun"):
            return "dryrun"
    return "official"


def extract_counts(blob):
    """Return [(spelling, value, kind)] for a parsed JSON document.

    kind is 'interaction' for steps the run actually spent stepping the
    simulator forward under a policy, and 'replay' for steps spent replaying
    an already-collected trajectory (re-rendering / re-encoding), which is
    real simulator work but not new interaction.
    """
    out = []
    if not isinstance(blob, dict):
        return out
    # The three interaction spellings below are three ways of writing the SAME
    # run total; v080_screen/summary.json carries all of totals.env_steps and
    # per_task.*.total_env_steps for one run.  Take the highest-priority one
    # present, never their sum.
    interaction = []
    v = blob.get("env_steps")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        interaction.append(("env_steps", int(v)))
    tot = blob.get("totals")
    if isinstance(tot, dict) and isinstance(tot.get("env_steps"), (int, float)):
        interaction.append(("totals.env_steps", int(tot["env_steps"])))
    pt = blob.get("per_task")
    if isinstance(pt, dict):
        s = sum(x.get("total_env_steps", 0) for x in pt.values() if isinstance(x, dict))
        if s:
            interaction.append(("per_task.*.total_env_steps", int(s)))
    if interaction:
        spelling, value = interaction[0]
        out.append((spelling, value, "interaction"))
        for sp, val in interaction[1:]:
            out.append((sp + " [duplicate spelling of the same run total, not counted]",
                        val, "duplicate_spelling"))
    rp = blob.get("replay")
    if isinstance(rp, dict) and isinstance(rp.get("env_steps"), (int, float)):
        out.append(("replay.env_steps", int(rp["env_steps"]), "replay"))
    v = blob.get("env_steps_replayed")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        out.append(("env_steps_replayed", int(v), "replay"))
    return out


def ledger_jsonl_sum(path: Path):
    """Sum n_steps over closing rows of a *ledger*.jsonl.  Returns (total, rows)."""
    total = 0
    rows = 0
    try:
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if not isinstance(r, dict):
                    continue
                if r.get("phase") == "open":
                    continue  # reservation, not spend
                n = r.get("n_steps")
                if isinstance(n, (int, float)) and not isinstance(n, bool):
                    total += int(n)
                    rows += 1
    except Exception:
        return None, 0
    return total, rows


def episode_steps_sum(blob):
    """Sum per-episode 'steps' fields, for runs that recorded no run-level total."""
    total = 0
    lists = 0

    def walk(o, depth=0):
        nonlocal total, lists
        if depth > 6:
            return
        if isinstance(o, list):
            if o and isinstance(o[0], dict) and isinstance(o[0].get("steps"), int):
                total += sum(e.get("steps", 0) for e in o
                             if isinstance(e, dict) and isinstance(e.get("steps"), int))
                lists += 1
                return
            for x in o[:4]:
                walk(x, depth + 1)
        elif isinstance(o, dict):
            for x in o.values():
                walk(x, depth + 1)

    walk(blob)
    return total, lists


# --------------------------------------------------------------------------- #
# Stage A
# --------------------------------------------------------------------------- #
def build_ledger(results: Path):
    json_rows = []          # (rel, dirrel, spelling, value, kind, blob-or-None)
    spellings = collections.Counter()
    dirs_with_json_count = set()
    all_dirs = set()
    unreadable = []

    for dp, _dn, fns in os.walk(results):
        dpp = Path(dp)
        rel_dir = str(dpp.relative_to(results))
        if rel_dir.startswith("v253_ledger"):
            continue  # never count our own outputs
        all_dirs.add(rel_dir)
        for fn in fns:
            if not fn.endswith(".json"):
                continue
            p = dpp / fn
            rel = str(p.relative_to(results))
            blob = load_json(p)
            if blob is None:
                if p.stat().st_size <= MAX_JSON_BYTES:
                    unreadable.append(rel)
                continue
            for spelling, value, kind in extract_counts(blob):
                spellings[spelling] += 1
                json_rows.append((rel, rel_dir, spelling, value, kind, blob))
                if kind == "interaction":
                    dirs_with_json_count.add(rel_dir)

    # *ledger*.jsonl, used only where the directory has no JSON interaction count
    ledger_rows = []
    ledger_crosschecks = []
    for dp, _dn, fns in os.walk(results):
        dpp = Path(dp)
        rel_dir = str(dpp.relative_to(results))
        if rel_dir.startswith("v253_ledger"):
            continue
        for fn in fns:
            if not (fn.endswith(".jsonl") and "ledger" in fn):
                continue
            p = dpp / fn
            rel = str(p.relative_to(results))
            total, nrows = ledger_jsonl_sum(p)
            if not total:
                continue
            spellings[f"{fn}:n_steps"] += 1
            if rel_dir in dirs_with_json_count:
                # cross-check instead of counting again
                js = sum(v for (_r, d, _s, v, k, _b) in json_rows
                         if d == rel_dir and k == "interaction")
                ledger_crosschecks.append(
                    {"dir": rel_dir, "ledger_file": rel, "ledger_sum": total,
                     "json_sum": js, "agrees": bool(js == total)})
            else:
                ledger_rows.append((rel, rel_dir, f"{fn}:n_steps", total, "interaction", nrows))

    # -------- assemble rows --------
    rows = []
    for rel, rel_dir, spelling, value, kind, blob in json_rows:
        ts = (parse_stamp(blob.get("utc") if isinstance(blob, dict) else None)
              or parse_stamp((blob.get("manifest") or {}).get("utc")
                             if isinstance(blob, dict) and isinstance(blob.get("manifest"), dict) else None)
              or parse_stamp((blob.get("created_utc") if isinstance(blob, dict) else None))
              or parse_stamp(rel))
        date_source = "utc_field"
        if ts is None:
            ts = dt.datetime.utcfromtimestamp((results / rel).stat().st_mtime)
            date_source = "file_mtime"
        elif not (isinstance(blob, dict) and (blob.get("utc") or blob.get("created_utc")
                                              or (isinstance(blob.get("manifest"), dict)
                                                  and blob["manifest"].get("utc")))):
            date_source = "directory_stamp"
        fam = family_of(rel)
        rows.append({
            "artifact": "results/" + rel,
            "run_dir": "results/" + rel_dir,
            "spelling": spelling,
            "env_steps": value,
            "kind": kind,
            "family": fam,
            "line": line_of(fam),
            "task": task_of(blob, rel),
            "utc": ts.strftime("%Y-%m-%dT%H%M%SZ"),
            "date": ts.strftime("%Y-%m-%d"),
            "date_source": date_source,
            "status": run_status(rel),
            "tier": "recorded",
        })
    for rel, rel_dir, spelling, value, kind, nrows in ledger_rows:
        ts = parse_stamp(rel) or dt.datetime.utcfromtimestamp((results / rel).stat().st_mtime)
        date_source = "directory_stamp" if parse_stamp(rel) else "file_mtime"
        fam = family_of(rel)
        rows.append({
            "artifact": "results/" + rel,
            "run_dir": "results/" + rel_dir,
            "spelling": spelling,
            "env_steps": value,
            "kind": kind,
            "family": fam,
            "line": line_of(fam),
            "task": task_of(None, rel),
            "utc": ts.strftime("%Y-%m-%dT%H%M%SZ"),
            "date": ts.strftime("%Y-%m-%d"),
            "date_source": date_source,
            "status": run_status(rel),
            "tier": "recorded",
            "ledger_records": nrows,
        })

    # -------- tier 2: recoverable from per-episode 'steps' where nothing else exists --------
    counted_dirs = {r["run_dir"] for r in rows}
    recovered = []
    for dp, _dn, fns in os.walk(results):
        dpp = Path(dp)
        rel_dir = str(dpp.relative_to(results))
        if rel_dir.startswith("v253_ledger") or ("results/" + rel_dir) in counted_dirs:
            continue
        for fn in fns:
            if not fn.endswith(".json"):
                continue
            p = dpp / fn
            blob = load_json(p)
            if blob is None:
                continue
            tot, nl = episode_steps_sum(blob)
            if tot <= 0:
                continue
            rel = str(p.relative_to(results))
            ts = parse_stamp(blob.get("utc") if isinstance(blob, dict) else None) or parse_stamp(rel)
            date_source = "utc_field" if ts else "file_mtime"
            if ts is None:
                ts = dt.datetime.utcfromtimestamp(p.stat().st_mtime)
            fam = family_of(rel)
            recovered.append({
                "artifact": "results/" + rel,
                "run_dir": "results/" + rel_dir,
                "spelling": "episodes[].steps (summed; no run-level total recorded)",
                "env_steps": int(tot),
                "kind": "interaction",
                "family": fam,
                "line": line_of(fam),
                "task": task_of(blob, rel),
                "utc": ts.strftime("%Y-%m-%dT%H%M%SZ"),
                "date": ts.strftime("%Y-%m-%d"),
                "date_source": date_source,
                "status": run_status(rel),
                "tier": "recovered",
                "episode_lists": nl,
            })

    rows.sort(key=lambda r: (r["utc"], r["artifact"]))
    recovered.sort(key=lambda r: (r["utc"], r["artifact"]))

    # -------- aggregates over the recorded interaction tier --------
    prim = [r for r in rows if r["kind"] == "interaction" and r["tier"] == "recorded"]
    replay = [r for r in rows if r["kind"] == "replay"]

    per_date = collections.Counter()
    per_line = collections.Counter()
    per_family = collections.Counter()
    per_task = collections.Counter()
    per_status = collections.Counter()
    for r in prim:
        per_date[r["date"]] += r["env_steps"]
        per_line[r["line"]] += r["env_steps"]
        per_family[r["family"]] += r["env_steps"]
        per_task[r["task"] or "(unattributed)"] += r["env_steps"]
        per_status[r["status"]] += r["env_steps"]

    cum = 0
    cumulative = []
    for d in sorted(per_date):
        cum += per_date[d]
        cumulative.append({"date": d, "env_steps": per_date[d], "cumulative": cum})
    grand_total = cum

    # per-run cumulative, used as the x-axis of the main plot
    c = 0
    for r in prim:
        c += r["env_steps"]
        r["cumulative_env_steps"] = c

    # -------- reconciliation against the V8 figure --------
    cutoff = parse_stamp(V8_REFERENCE_CUTOFF)
    upto = [r for r in prim if parse_stamp(r["utc"]) <= cutoff]
    upto_total = sum(r["env_steps"] for r in upto)
    upto_official = sum(r["env_steps"] for r in upto if r["status"] == "official")
    upto_nonofficial = upto_total - upto_official
    rec_upto = sum(r["env_steps"] for r in recovered if parse_stamp(r["utc"]) <= cutoff)

    # Search, rather than assume, for a family-level exclusion that closes the gap.
    fams_upto = sorted({r["family"] for r in upto})
    fam_tot = {f: sum(r["env_steps"] for r in upto if r["family"] == f) for f in fams_upto}
    exact = None
    import itertools as _it
    for k in range(1, min(4, len(fams_upto)) + 1):
        for combo in _it.combinations(fams_upto, k):
            if upto_total - sum(fam_tot[f] for f in combo) == V8_REFERENCE_TOTAL:
                exact = {"excluded_families": list(combo),
                         "excluded_env_steps": {f: fam_tot[f] for f in combo},
                         "remainder": V8_REFERENCE_TOTAL}
                break
        if exact:
            break

    # families that record no step count at all
    fam_with = {r["family"] for r in rows} | {r["family"] for r in recovered}
    fam_all = {p.name for p in results.iterdir() if p.is_dir()} - {"v253_ledger"}
    fam_without = sorted(f for f in fam_all if f not in fam_with)

    # countable rollouts that carry NO step count anywhere (the ledger's blind spot)
    blind = blind_spot_rollouts(results, sorted(fam_all))

    ledger = {
        "generated_utc": dt.datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ"),
        "generator": "scripts/build_v253_interaction_ledger.py",
        "scope": {
            "root": "results/",
            "json_files_over_size_limit_skipped": MAX_JSON_BYTES,
            "unreadable_json": unreadable,
        },
        "spellings_found": dict(spellings.most_common()),
        "grand_total_env_steps": grand_total,
        "totals": {
            "recorded_interaction": grand_total,
            "recorded_interaction_official_only":
                sum(r["env_steps"] for r in prim if r["status"] == "official"),
            "recorded_interaction_aborted_smoke_discarded_superseded":
                sum(r["env_steps"] for r in prim if r["status"] != "official"),
            "replay_re_execution_not_counted_in_grand_total":
                sum(r["env_steps"] for r in replay),
            "recovered_from_episode_steps_not_counted_in_grand_total":
                sum(r["env_steps"] for r in recovered),
            "n_runs_recorded": len(prim),
            "n_runs_replay": len(replay),
            "n_runs_recovered": len(recovered),
        },
        "reconciliation_v8": {
            "reference_total": V8_REFERENCE_TOTAL,
            "reference_source": V8_REFERENCE_SOURCE,
            "reference_as_of": "2026-08-25 (Action 2M.5)",
            "ledger_total_up_to_cutoff": upto_total,
            "ledger_total_up_to_cutoff_official_only": upto_official,
            "ledger_total_up_to_cutoff_non_official": upto_nonofficial,
            "recovered_tier_up_to_cutoff": rec_upto,
            "difference_ledger_minus_reference": upto_total - V8_REFERENCE_TOTAL,
            "agrees": bool(upto_total == V8_REFERENCE_TOTAL),
            "per_family_up_to_cutoff": fam_tot,
            "exact_match_when_these_families_are_excluded": exact,
            "note": "Nothing here is adjusted to make the numbers meet. The "
                    "exclusion above was found by searching family subsets for an "
                    "exact match; if it is null, the gap is unexplained.",
        },
        "per_date": [{"date": d, "env_steps": per_date[d]} for d in sorted(per_date)],
        "per_line": {k: per_line[k] for k in sorted(per_line)},
        "per_family": {k: per_family[k] for k in sorted(per_family, key=lambda x: -per_family[x])},
        "per_task": {k: per_task[k] for k in sorted(per_task, key=lambda x: -per_task[x])},
        "per_status": dict(per_status),
        "cumulative_series": cumulative,
        "families_with_no_step_count_recorded": fam_without,
        "cannot_see": blind,
        "ledger_jsonl_crosschecks": ledger_crosschecks,
        "rows": prim,
        "rows_replay": replay,
        "rows_recovered": recovered,
    }
    return ledger


def blind_spot_rollouts(results: Path, fam_without):
    """Count completed rollouts that exist as records but carry no step count.

    These are the runs the ledger genuinely cannot price: the artifact proves the
    rollout happened and does not say how long it was.
    """
    out = []
    for fam in fam_without:
        base = results / fam
        if not base.is_dir():
            continue
        completes = 0
        for dp, _dn, fns in os.walk(base):
            for fn in fns:
                if not fn.endswith(".jsonl") or "ledger" not in fn:
                    continue
                p = Path(dp) / fn
                try:
                    with p.open() as fh:
                        for line in fh:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                r = json.loads(line)
                            except Exception:
                                continue
                            if (isinstance(r, dict) and r.get("status") == "complete"
                                    and not isinstance(r.get("n_steps"), (int, float))):
                                completes += 1
                except Exception:
                    pass
        if completes:
            out.append({
                "family": fam,
                "completed_rollout_records_with_no_step_count": completes,
                "why": "these *ledger.jsonl rows record status/success only, with no "
                       "n_steps field: the artifact proves the rollout ran and does not "
                       "say how many environment steps it cost",
            })
    return out


# --------------------------------------------------------------------------- #
# Stage B - deployment outcomes and the main plot
# --------------------------------------------------------------------------- #
def clopper_pearson(k, n, alpha=0.05):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


V251_ARMS = {
    "base":    ("base (frozen pi0.5, no residual)", "no_phi", "frozen"),
    "rand":    ("rand (untrained residual)", "no_phi", "untrained_residual"),
    "M0":      ("M0 (stale WM)", "phi", "wm"),
    "M0_cont": ("M0_cont (stale WM, matched compute)", "phi", "wm"),
    "M1":      ("M1 (updated WM)", "phi", "wm"),
    "M1_shuf": ("M1_shuf (action-shuffled)", "phi", "wm_control"),
    "Q":       ("Q (no-rollout)", "phi", "no_rollout_control"),
}


def episode_mean_abs_delta(eps):
    """Mean |delta| over episode records that report one; None if none do."""
    if not isinstance(eps, list):
        return None
    vals = [e.get("mean_abs_delta") for e in eps
            if isinstance(e, dict) and isinstance(e.get("mean_abs_delta"), (int, float))]
    return (sum(vals) / len(vals)) if vals else None


def collect_outcomes(results: Path):
    """Every run dir on disk that reports (successes, n) for a deployed actor."""
    out = []
    for dp, _dn, fns in os.walk(results):
        dpp = Path(dp)
        rel_dir = str(dpp.relative_to(results))
        if rel_dir.startswith("v253_ledger"):
            continue
        for fn in ("summary.json", "outcome.json"):
            if fn not in fns:
                continue
            blob = load_json(dpp / fn)
            if not isinstance(blob, dict):
                continue
            k, n = blob.get("successes"), blob.get("n")
            if not (isinstance(k, int) and isinstance(n, int) and n > 0):
                continue
            man = load_json(dpp / "manifest.json") if "manifest.json" in fns else None
            man = man if isinstance(man, dict) else {}
            eps = blob.get("episodes") or blob.get("episode_records") or []
            m4 = None
            if isinstance(eps, list) and eps and isinstance(eps[0], dict) and "events" in eps[0]:
                m4 = sum(1 for e in eps if isinstance(e.get("events"), dict) and "4" in e["events"])
            out.append({
                "run_dir": "results/" + rel_dir,
                "artifact": "results/" + rel_dir + "/" + fn,
                "task": blob.get("task") or task_of(blob, rel_dir),
                "tag": blob.get("tag") or blob.get("role"),
                "utc": (parse_stamp(blob.get("utc")) or parse_stamp(rel_dir)
                        or dt.datetime.utcfromtimestamp((dpp / fn).stat().st_mtime)
                        ).strftime("%Y-%m-%dT%H%M%SZ"),
                "successes": k,
                "n": n,
                "milestone4": m4,
                "n_episodes_with_events": (len(eps) if m4 is not None else 0),
                "env_steps_this_run": blob.get("env_steps"),
                "zero_residual": blob.get("zero_residual"),
                "scale": blob.get("scale", man.get("actor_scale")),
                "actor": blob.get("actor"),
                "manifest_role": man.get("role"),
                "mean_abs_delta": (blob.get("mean_abs_delta_panel")
                                   if blob.get("mean_abs_delta_panel") is not None
                                   else man.get("mean_abs_delta_applied")
                                   if man.get("mean_abs_delta_applied") is not None
                                   else episode_mean_abs_delta(eps)),
                "stage": ("data_collection" if man.get("role") in ("base", "resid")
                          else "deployment_eval"),
            })
            break  # one outcome per run dir; summary.json wins
    out.sort(key=lambda r: r["utc"])
    return out


def cumulative_at(ledger_rows, utc):
    """Cumulative recorded interaction steps up to and including time `utc`."""
    t = parse_stamp(utc)
    return sum(r["env_steps"] for r in ledger_rows if parse_stamp(r["utc"]) <= t)


def classify_v251(tag):
    if not tag:
        return None
    for key, val in V251_ARMS.items():
        if tag.endswith("_" + key) or tag == key:
            return key, val
    return None


def build_plot_data(ledger, results: Path):
    rows = ledger["rows"]
    outcomes = collect_outcomes(results)

    points = []
    for o in outcomes:
        cls = classify_v251(o["tag"]) if o["task"] == "chain3_lr2" else None
        arm_key, arm_meta = (cls if cls else (None, None))
        if arm_meta:
            label, annot, role = arm_meta
        else:
            label = o["tag"] or Path(o["run_dir"]).name
            zr = o.get("zero_residual")
            sc = o.get("scale")
            mad = o.get("mean_abs_delta")
            # "frozen" = the run applied no residual at all: explicit zero_residual,
            # zero scale, a measured mean |delta| of exactly 0, a manifest role of
            # 'base', or no residual actor recorded anywhere.
            if zr is True or sc in (0, 0.0) or mad == 0 or o.get("manifest_role") == "base":
                role = "frozen"                     # positive evidence of no residual
            elif o.get("actor") or (isinstance(sc, (int, float)) and sc > 0) or (mad or 0) > 0:
                role = "residual_other"             # positive evidence of a residual
            else:
                # The summary schema predates the residual arms: it records neither a
                # residual actor nor a delta.  Do not guess which it was.
                role = "unclassified"
            annot = "phi" if role == "residual_other" else "no_phi"
        k, n = o["successes"], o["n"]
        lo, hi = clopper_pearson(k, n)
        pt = {
            "run_dir": o["run_dir"],
            "artifact": o["artifact"],
            "task": o["task"],
            "tag": o["tag"],
            "arm": arm_key,
            "label": label,
            "role": role,
            "phi_annotation": (annot == "phi"),
            "utc": o["utc"],
            "stage": o["stage"],
            "mean_abs_delta": o.get("mean_abs_delta"),
            "cumulative_env_steps": cumulative_at(rows, o["utc"]),
            "env_steps_this_run": o["env_steps_this_run"],
            "terminal_successes": k,
            "n": n,
            "terminal_rate": k / n,
            "terminal_cp95": [lo, hi],
        }
        if o["milestone4"] is not None:
            m = o["milestone4"]
            mlo, mhi = clopper_pearson(m, n)
            pt["milestone4_successes"] = m
            pt["milestone4_rate"] = m / n
            pt["milestone4_cp95"] = [mlo, mhi]
            pt["milestone4_definition"] = "'4' in episode events (non-contiguous, as registered)"
        else:
            pt["milestone4_successes"] = None
            pt["milestone4_note"] = "episode records carry no 'events' field; milestone-4 not computable"
        points.append(pt)

    claims = {
        "interface_comparison": {
            "what": "any residual-carrying arm vs frozen pi0.5 base",
            "arms": ["base", "rand", "M0", "M0_cont", "M1", "M1_shuf", "Q"],
            "caveat": "base and rand never receive the privileged Phi' annotation; "
                      "this contrast measures a bounded sustained residual trained on a "
                      "supplied progress annotation, not the world model.",
        },
        "world_model_comparison": {
            "what": "M1 vs its matched controls, all of which share Phi' and the data",
            "pairs": [["M1", "M0"], ["M1", "M0_cont"], ["M1", "Q"], ["M1", "M1_shuf"]],
            "caveat": "these are the only contrasts in this figure that are about the "
                      "rolled-forward latent model.",
        },
    }
    return {
        "generated_utc": ledger["generated_utc"],
        "x_axis": "cumulative recorded real environment steps over the whole project, "
                  "up to and including the run that produced the point "
                  "(ledger tier 'recorded', interaction kind only)",
        "y_axis": "terminal success rate and milestone-4 rate, Clopper-Pearson 95%",
        "grand_total_env_steps": ledger["grand_total_env_steps"],
        "claim_classes": claims,
        "points": points,
    }


def draw(plot_data, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    PALETTE = {
        "blue_main": "#0F4D92", "blue_secondary": "#3775BA",
        "green_3": "#8BCF8B", "red_strong": "#B64342",
        "neutral": "#CFCECE", "teal": "#42949E", "violet": "#9A4D8E",
        "grey_text": "#3A3A3A",
    }
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
        "font.size": 13,
        "axes.linewidth": 2.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    STYLE = {
        "frozen":             (PALETTE["neutral"],        "s", "frozen pi0.5, no residual"),
        "untrained_residual": (PALETTE["red_strong"],     "^", "untrained residual (rand)"),
        "wm":                 (PALETTE["blue_main"],      "o", "world-model arm (M0, M0_cont, M1)"),
        "wm_control":         (PALETTE["blue_secondary"], "D", "action-shuffled control (M1_shuf)"),
        "no_rollout_control": (PALETTE["teal"],           "v", "no-rollout control (Q)"),
        "residual_other":     (PALETTE["violet"],         "o", "other residual arm"),
        "unclassified":       ("#8A8A8A",                 "x", "residual state not recorded"),
    }

    pts = plot_data["points"]
    c3 = [p for p in pts if p["task"] == "chain3_lr2"]
    c12 = [p for p in pts if p["task"] in ("chain1b_lr2", "chain2b_lr2")]
    MM = 1e6

    def draw_points(ax, rows, ykey, ekey, big_only=False):
        for p in rows:
            y = p.get(ykey)
            if y is None:
                continue
            col, mk, _ = STYLE.get(p["role"], STYLE["unclassified"])
            deploy = (p["stage"] == "deployment_eval")
            if big_only and not deploy:
                continue
            lo, hi = p[ekey]
            ax.errorbar(p["cumulative_env_steps"] / MM, y,
                        yerr=[[max(0.0, y - lo)], [max(0.0, hi - y)]],
                        fmt=mk, color=col,
                        ms=10 if deploy else 5,
                        mew=1.4 if deploy else 0.8,
                        mec="black" if deploy else col,
                        elinewidth=1.6 if deploy else 0.8,
                        capsize=3 if deploy else 0,
                        alpha=0.95 if deploy else 0.45,
                        zorder=4 if deploy else 2)

    fig, axes = plt.subplots(1, 3, figsize=(22.0, 6.8))

    v251 = {p["arm"]: p for p in c3 if p["arm"]}
    wm_keys = [k for k in ("M0", "M0_cont", "M1", "M1_shuf", "Q") if k in v251]

    def mark_claims(ax, ykey):
        """Red arrow = the interface comparison.  Blue band = the world-model one."""
        if "base" in v251 and "M1" in v251:
            b, m = v251["base"], v251["M1"]
            ax.annotate("", xy=(m["cumulative_env_steps"] / MM, m[ykey]),
                        xytext=(b["cumulative_env_steps"] / MM, b[ykey]),
                        arrowprops=dict(arrowstyle="-|>", color=PALETTE["red_strong"],
                                        lw=2.0, alpha=0.85, shrinkA=8, shrinkB=8),
                        zorder=5)
        if wm_keys:
            xs = [v251[k]["cumulative_env_steps"] / MM for k in wm_keys]
            ax.axvspan(min(xs) - 0.08, max(xs) + 0.08,
                       color=PALETTE["blue_main"], alpha=0.07, zorder=0)
        if "base" in v251:
            ax.axhline(v251["base"][ykey], ls="--", lw=1.2,
                       color=PALETTE["neutral"], zorder=1)

    # ---- A: chain3 terminal success ----
    ax = axes[0]
    draw_points(ax, c3, "terminal_rate", "terminal_cp95")
    mark_claims(ax, "terminal_rate")
    ax.set_title("A.  chain3_lr2, terminal success", loc="left", fontweight="bold", fontsize=14)
    ax.set_ylabel("terminal success rate  (Clopper-Pearson 95%)")
    ax.set_ylim(-0.02, 0.45)
    for k in ("base", "rand", "M1", "Q"):
        if k in v251:
            p = v251[k]
            ax.annotate(k, (p["cumulative_env_steps"] / MM, p["terminal_rate"]),
                        textcoords="offset points", xytext=(8, 6),
                        fontsize=11, color=PALETTE["grey_text"])

    # ---- B: chain3 milestone-4 ----
    ax = axes[1]
    draw_points(ax, c3, "milestone4_rate", "milestone4_cp95")
    mark_claims(ax, "milestone4_rate")
    ax.set_title("B.  chain3_lr2, milestone-4 (primary endpoint)",
                 loc="left", fontweight="bold", fontsize=14)
    ax.set_ylabel("milestone-4 rate  (Clopper-Pearson 95%)")
    ax.set_ylim(-0.02, 0.45)
    offs = {"base": (-2, -20), "rand": (-16, 6), "M0": (-16, -8),
            "M0_cont": (-6, 12), "M1": (-4, -22), "M1_shuf": (30, -14), "Q": (6, 11)}
    for k in ("base", "rand") + tuple(wm_keys):
        if k in v251 and v251[k].get("milestone4_rate") is not None:
            p = v251[k]
            ax.annotate(k, (p["cumulative_env_steps"] / MM, p["milestone4_rate"]),
                        textcoords="offset points", xytext=offs.get(k, (7, 6)),
                        fontsize=10, color=PALETTE["grey_text"], ha="center")
    ax.text(0.02, 0.98,
            "red arrow = INTERFACE comparison\n"
            "base has no $\\Phi\'$ annotation:\n"
            "a bounded sustained residual,\n"
            "not the world model",
            transform=ax.transAxes, fontsize=10.5, va="top",
            color=PALETTE["red_strong"], linespacing=1.35)
    ax.text(0.02, 0.63,
            "blue band = WORLD-MODEL comparison\n"
            "M1 vs M0 / M0_cont / Q / M1_shuf,\n"
            "sharing $\\Phi\'$, data and budget",
            transform=ax.transAxes, fontsize=10.5, va="top",
            color=PALETTE["blue_main"], linespacing=1.35)

    # ---- C: chain1b / chain2b ----
    ax = axes[2]
    if c12:
        for p in c12:
            col, mk, _ = STYLE.get(p["role"], STYLE["unclassified"])
            y = p["terminal_rate"]
            lo, hi = p["terminal_cp95"]
            ax.errorbar(p["cumulative_env_steps"] / MM, y,
                        yerr=[[max(0.0, y - lo)], [max(0.0, hi - y)]],
                        fmt=mk, color=col,
                        mfc=(col if p["task"] == "chain1b_lr2" else "white"),
                        ms=6, mew=1.1, mec=col, elinewidth=0.9, alpha=0.55, zorder=3)
        ax.set_title("C.  chain1b_lr2 (filled), chain2b_lr2 (open)",
                     loc="left", fontweight="bold", fontsize=14)
    else:
        ax.text(0.5, 0.5, "no chain1b / chain2b deployed arms on disk",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("C.  chain1b / chain2b", loc="left", fontweight="bold")
    ax.set_ylabel("terminal success rate  (Clopper-Pearson 95%)")
    ax.set_ylim(-0.02, 1.02)

    for a in axes:
        a.set_xlabel("cumulative real environment steps  (millions)")
        a.grid(axis="y", ls=":", lw=0.7, color="#DDDDDD", zorder=0)

    handles = []
    for role, (col, mk, lab) in STYLE.items():
        if not any(p["role"] == role for p in pts):
            continue
        handles.append(Line2D([], [], color=col, marker=mk, ls="none", ms=9,
                              mec="black", mew=1.0, label=lab))
    handles += [
        Line2D([], [], color="#666666", marker="o", ls="none", ms=10, mec="black",
               label="deployment evaluation arm (large)"),
        Line2D([], [], color="#666666", marker="o", ls="none", ms=5, alpha=0.45,
               label="data-collection run (small)"),
        Patch(facecolor=PALETTE["blue_main"], alpha=0.10, label="world-model comparison"),
        Line2D([], [], color=PALETTE["red_strong"], lw=2.0, label="interface comparison"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.13), fontsize=11)

    fig.suptitle(
        "Success against cumulative real interactions "
        f"(project ledger: {plot_data['grand_total_env_steps']:,} recorded environment steps)",
        fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        q = out_dir / f"main_plot.{ext}"
        fig.savefig(q, dpi=300, bbox_inches="tight")
        paths.append(str(q))
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--skip-figure", action="store_true")
    args = ap.parse_args()

    results = Path(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ledger = build_ledger(results)
    (out / "ledger.json").write_text(json.dumps(ledger, indent=1))
    print(f"ledger: {len(ledger['rows'])} recorded runs, "
          f"grand total {ledger['grand_total_env_steps']:,} env steps")
    rec = ledger["reconciliation_v8"]
    print(f"V8 reconciliation: ledger<=2026-08-25 = {rec['ledger_total_up_to_cutoff']:,} "
          f"vs reference {rec['reference_total']:,} "
          f"(difference {rec['difference_ledger_minus_reference']:+,})")

    plot_data = build_plot_data(ledger, results)
    (out / "main_plot_data.json").write_text(json.dumps(plot_data, indent=1))
    print(f"plot data: {len(plot_data['points'])} deployed arms")
    if not args.skip_figure:
        for p in draw(plot_data, out):
            print("figure:", p)


if __name__ == "__main__":
    sys.exit(main())
