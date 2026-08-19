#!/usr/bin/env python
"""Stage 1R.1 - paired deadline-sensitivity calibration (daily 2026-08-19 §6).

Calibration only.  Stock pi0.5, full-prompt `N=1`, no candidates, no branches,
no training.  Its data may validate schemas and feasibility but may never train
a later FQE or model.

Measures ONE quantity: the *late conversion* rate - episodes that fail at the
frozen deadline `L` but succeed by the extended horizon.  V8.0 measured
`chain1b` at 12/30 with 19/30 episodes terminating exactly at the 250-step cap
and six of twelve successes landing in the last ten steps, so whether that rate
is a capability measurement or a deadline measurement is undetermined, and
every anchor decision downstream depends on the answer.

Verdicts come from the sealed `lcwm.v08r_contract`; this script chooses nothing.

    python scripts/run_v080r_1r1.py --plan            # no environment interaction
    python scripts/run_v080r_1r1.py --panel 1
    python scripts/run_v080r_1r1.py --panel 2         # gated on panel 1 opening it

`--plan` spends nothing: it validates provenance, resolves seeds, caps and
checkpoints, and returns before any policy or environment is constructed.  An
earlier `--dry-run` gated only the file writes while still stepping MuJoCo, so
it produced real environment interaction with no ledger row - the one thing
framework §7.6 forbids outright.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

from lcwm import v08r_contract as C  # noqa: E402
from lcwm.v080r_panel import (CHECKPOINTS, deadline, extended_horizon,  # noqa: E402
                              make_env_at, prefix_identity, run_deadline_episode)
from register_v080r import verify_sealed  # noqa: E402

DUPLICATE_N = C.DUPLICATE_N

TASKS = ("chain1b_lr2", "chain2b_lr2")
PANEL_SEEDS = {1: list(range(3400, 3420)), 2: list(range(3420, 3440))}
REGISTRATION = REPO / "results" / "v080r_registration" / "V080R_REGISTRATION.json"
OUT_ROOT = REPO / "results" / "v080r_1r1"


def guard_seeds(seeds: list[int]) -> None:
    reserved = set(range(3200, 3260)) | set(range(3300, 3400))
    hit = reserved & set(seeds)
    if hit:
        raise SystemExit(f"HALT: reserved seeds requested: {sorted(hit)}")


def load_panel1(tasks: list[str], reg_sha: str) -> tuple[Path, dict, list[dict]]:
    """Locate and validate the panel-1 artifact a panel-2 run must extend."""
    cands = sorted(OUT_ROOT.glob("*_panel1"))
    for d in reversed(cands):
        sm, mf = d / "summary.json", d / "manifest.json"
        if not (sm.exists() and mf.exists()):
            continue                      # no summary => incomplete run
        m = json.loads(mf.read_text())
        if m.get("registration_sha256") != reg_sha or m.get("panel") != 1:
            continue
        summary = json.loads(sm.read_text())
        if any(summary["per_task"].get(t, {}).get("n") != C.PANEL_1 for t in tasks):
            continue
        eps = [json.loads(l) for l in (d / "episodes.jsonl").read_text().splitlines() if l]
        return d, summary, eps
    raise SystemExit(
        "HALT: panel 2 requires a complete panel-1 artifact under "
        f"{OUT_ROOT} with matching registration_sha256 and n={C.PANEL_1} per task")


def cumulative_spend() -> dict:
    """Derived by summing every ledger on disk, never separately maintained.

    A run killed mid-panel writes its ledger rows (they are written before the
    charge) but cannot write a summary, so a cumulative file updated only on
    clean exit silently loses those steps.  One aborted panel-2 run spent 2,687
    ledgered steps that way.  Deriving from the ledgers cannot miss them.
    """
    spend: dict = {}
    for led in sorted(REPO.glob("results/v080r_*/**/interaction_ledger.jsonl")):
        for line in led.read_text().splitlines():
            if not line:
                continue
            r = json.loads(line)
            spend.setdefault(r["cap_line"], {}).setdefault(r["task"], 0)
            spend[r["cap_line"]][r["task"]] += r["n_steps"]
    spend["_total"] = sum(v for ln, d in spend.items() if not ln.startswith("_")
                          for v in d.values())
    spend["_cap_total"] = C.INTERACTION_CAP_TOTAL
    return spend


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=int, choices=(1, 2))
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--plan", action="store_true",
                    help="validate and print the plan; spend nothing")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="run on a dirty tree; stamps a provenance downgrade")
    args = ap.parse_args()
    if not args.plan and args.panel is None:
        raise SystemExit("--panel is required unless --plan")

    # ---- provenance preflight, before any GPU --------------------------
    pre = verify_sealed(strict_git=not args.allow_dirty)
    print(json.dumps({k: pre[k] for k in
                      ("ok", "self_sha256_match", "sealed_source_count")}, indent=2))
    if pre["source_drift"]:
        raise SystemExit(f"HALT: sealed source drift: {sorted(pre['source_drift'])}")
    if not pre["self_sha256_match"]:
        raise SystemExit("HALT: registration self_sha256 does not recompute")
    if pre["git"]["dirty"] and not args.allow_dirty:
        raise SystemExit("HALT: dirty tree; commit, or pass --allow-dirty and "
                         "accept the recorded provenance downgrade")
    reg = json.loads(REGISTRATION.read_text())

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    panel = args.panel or 1
    seeds = PANEL_SEEDS[panel]
    guard_seeds(seeds)

    # ---- panel-2 gating -------------------------------------------------
    p1_dir = p1_summary = None
    p1_eps: list[dict] = []
    if panel == 2:
        p1_dir, p1_summary, p1_eps = load_panel1(tasks, reg["self_sha256"])
        not_opened = [t for t in tasks
                      if not p1_summary["per_task"][t]["opens_second_panel"]]
        if not_opened:
            raise SystemExit(
                f"HALT: panel 1 did not open a second panel for {not_opened}; "
                f"their verdicts were "
                f"{[p1_summary['per_task'][t]['verdict'] for t in not_opened]}. "
                "Re-run with --tasks restricted to the tasks that opened.")

    # ---- cap feasibility, before spending -------------------------------
    plan_caps = {}
    for t in tasks:
        probe_line = C.cap_line("1R.1", panel, C.Subrole.DEADLINE_PROBE.value)
        dup_line = C.cap_line("1R.1", panel, C.Subrole.DUPLICATE_CHECK.value)
        worst_probe = len(seeds) * extended_horizon(t)
        worst_dup = DUPLICATE_N * deadline(t)
        plan_caps[t] = {
            probe_line: {"worst_case": worst_probe, "cap": C.cap_for(probe_line, t)},
            dup_line: {"worst_case": worst_dup, "cap": C.cap_for(dup_line, t)}}
        for line, v in plan_caps[t].items():
            if v["worst_case"] > v["cap"]:
                raise SystemExit(f"HALT: infeasible plan, {t} {line} needs "
                                 f"{v['worst_case']} > cap {v['cap']}")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT_ROOT / f"{stamp}_panel{panel}"
    head = pre["git"]["head"]
    manifest = {
        "stage": "1R.1", "panel": panel, "utc": stamp, "status": "RUNNING",
        "registration_sha256": pre["self_sha256_recomputed"],
        "registration_verified": True,
        "git_head": head, "git_dirty": pre["git"]["dirty"],
        "provenance_downgraded": bool(args.allow_dirty and pre["git"]["dirty"]),
        "runner_hashes": {f: hashlib.sha256((REPO / f).read_bytes()).hexdigest()
                          for f in ("scripts/run_v080r_1r1.py",
                                    "lcwm/v080r_panel.py", "lcwm/v08r_contract.py")},
        "tasks": tasks, "seeds": seeds,
        "checkpoints": {t: list(CHECKPOINTS[t]) for t in tasks},
        "duplicate_subset_n": DUPLICATE_N,
        "duplicate_tolerance": reg["rng_procedure"]["duplicate_run_tolerance"],
        "role": C.Role.CALIBRATION.value,
        "cap_plan": plan_caps,
        "extends_panel1": str(p1_dir.relative_to(REPO)) if p1_dir else None,
        "data_use_restriction": ("calibration-only; may validate schemas and "
                                 "feasibility, may never train an FQE or model"),
    }

    if args.plan:
        print(json.dumps({"PLAN_ONLY": True, "manifest": manifest,
                          "cumulative_spend": cumulative_spend()}, indent=2))
        return 0

    out.mkdir(parents=True, exist_ok=True)
    jl, led = out / "episodes.jsonl", out / "interaction_ledger.jsonl"
    jl.write_text(""); led.write_text("")

    from lcwm.chassis import DEFAULT_MODEL, Pi05Runner
    manifest["policy"] = reg["provenance"]["external"].get("lerobot", {})
    manifest["model_id"] = DEFAULT_MODEL
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in
                      ("stage", "panel", "checkpoints", "git_head", "git_dirty",
                       "registration_sha256", "extends_panel1")}, indent=2), flush=True)

    runner = Pi05Runner(model_id=DEFAULT_MODEL, suite_name="libero_10",
                        n_action_steps=10)

    prior = cumulative_spend()
    spent: dict[tuple[str, str], int] = {}
    episodes: list = []
    print(f"prior ledgered spend {prior['_total']}/{prior['_cap_total']}", flush=True)
    halted: dict | None = None

    def record(ep):
        """Ledger FIRST, then charge, then test the cap: the ledger must record
        steps actually spent, not steps the budget permitted."""
        nonlocal halted
        episodes.append(ep)
        with jl.open("a") as fh:
            fh.write(json.dumps(asdict(ep)) + "\n")
        line = C.cap_line("1R.1", panel, ep.subrole)
        with led.open("a") as fh:
            fh.write(json.dumps({
                "stage": "1R.1", "role": C.Role.CALIBRATION.value,
                "subrole": ep.subrole, "cap_line": line, "task": ep.task,
                "seed": ep.seed, "n_steps": ep.steps,
                "officiality": ep.officiality, "anchor_id": None,
                "policy_hash": DEFAULT_MODEL}) + "\n")
        key = (line, ep.task)
        spent[key] = spent.get(key, 0) + ep.steps
        cap = C.cap_for(line, ep.task)
        charged = spent[key] + prior.get(line, {}).get(ep.task, 0)
        if charged > cap:
            halted = {"cap_line": line, "task": ep.task, "this_run": spent[key],
                      "prior_ledgered": prior.get(line, {}).get(ep.task, 0),
                      "charged": charged, "cap": cap}
            raise SystemExit(f"HALT: {ep.task} {line} cap {cap} exceeded "
                             f"(charged {charged} = {spent[key]} this run + "
                             f"{prior.get(line, {}).get(ep.task, 0)} prior)")

    try:
        for task in tasks:
            env = make_env_at(task, extended_horizon(task))
            try:
                L = deadline(task)
                for seed in seeds:
                    ep = run_deadline_episode(runner, env, task, seed,
                                              C.Subrole.DEADLINE_PROBE.value)
                    record(ep)
                    tag = ("LATE" if ep.late_conversion else
                           "ok" if ep.success_official else "fail")
                    print(f"{task} seed={seed} L={L} H={ep.horizon_run} "
                          f"succ@{ep.success_step} {tag:4s} steps={ep.steps} "
                          f"events={ep.events_achieved} ({ep.wall_s:.0f}s)", flush=True)
            finally:
                try:
                    env.close()
                except Exception:
                    pass

        dup_report = {}
        for task in tasks:
            env = make_env_at(task, deadline(task))
            try:
                for seed in seeds[:DUPLICATE_N]:
                    dup = run_deadline_episode(runner, env, task, seed,
                                               C.Subrole.DUPLICATE_CHECK.value)
                    record(dup)
                    ext = next(e for e in episodes if e.task == task
                               and e.seed == seed
                               and e.subrole == C.Subrole.DEADLINE_PROBE.value)
                    ok, detail = prefix_identity(ext, dup)
                    dup_report[f"{task}|{seed}"] = detail
                    print(f"  dup {task} seed={seed}: "
                          f"{'MATCH' if ok else 'MISMATCH'}", flush=True)
            finally:
                try:
                    env.close()
                except Exception:
                    pass
    except SystemExit:
        manifest["status"] = "HALTED"
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        (out / "HALT.json").write_text(json.dumps(
            {"reason": "interaction cap exceeded", **(halted or {}),
             "episodes_completed": len(episodes),
             "note": "no summary.json is written; this directory is incomplete"},
            indent=2))
        raise

    # ---- verdicts --------------------------------------------------------
    per_task, pooled = {}, {}
    for task in tasks:
        probes = [e for e in episodes if e.task == task
                  and e.subrole == C.Subrole.DEADLINE_PROBE.value]
        n, k = len(probes), sum(e.late_conversion for e in probes)
        official = sum(e.success_official for e in probes)
        if panel == 1:
            v = C.deadline_sensitivity_verdict(k, n)
        else:
            # §6: the decision at panel 2 is the POOLED n=40 rule, never the
            # panel-1 rule applied a second time.
            p1 = [e for e in p1_eps if e["task"] == task
                  and e["subrole"] == C.Subrole.DEADLINE_PROBE.value]
            kp, np_ = k + sum(e["late_conversion"] for e in p1), n + len(p1)
            v = C.deadline_sensitivity_verdict(kp, np_)
            pooled[task] = {"k": kp, "n": np_, "verdict": v.label,
                            "reason": v.reason, "cp_upper": v.upper,
                            "cp_lower": v.lower,
                            "panel1_late": kp - k, "panel2_late": k}
            assert not v.opens_second_panel, "no third panel is allowed"
        per_task[task] = {
            "n": n, "official_success_at_L": official, "late_conversions": k,
            "verdict": v.label, "reason": v.reason, "cp_upper": v.upper,
            "cp_lower": v.lower, "opens_second_panel": v.opens_second_panel,
            "late_success_steps": sorted(e.success_step for e in probes
                                         if e.late_conversion),
            "env_steps_by_cap_line": {ln: sp for (ln, t), sp in spent.items()
                                      if t == task},
        }

    dup_all_match = all(d["match"] for d in dup_report.values())
    summary = {
        "manifest": manifest, "per_task": per_task,
        "pooled_n40": pooled or None,
        "duplicate_prefix": {"all_match": dup_all_match, "detail": dup_report},
        "provenance_verdict": "PASS" if dup_all_match else "HALT",
        "totals": {"episodes": len(episodes),
                   "env_steps": sum(spent.values()),
                   "by_cap_line": {f"{ln}|{t}": sp for (ln, t), sp in spent.items()},
                   "wall_s": round(sum(e.wall_s for e in episodes), 1)},
    }
    if not dup_all_match:
        # framework §14.4 lists replay/provenance failure as a HALT condition:
        # a deadline label may not be issued on a run whose prefix identity failed
        for t in per_task:
            per_task[t]["verdict"] = "PROVENANCE_HALT"
            per_task[t]["reason"] = ("duplicate-prefix mismatch; the "
                                     "deadline-sensitivity label is withheld")
        manifest["status"] = "HALTED"
    else:
        manifest["status"] = "COMPLETE"
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    cum = cumulative_spend()          # re-derived, now including this run
    (OUT_ROOT / "SPEND.json").write_text(json.dumps(
        {**cum, "_note": "DERIVED from all interaction_ledger.jsonl; do not "
                         "hand-edit"}, indent=2))

    print("\n=== Stage 1R.1 ===")
    for t, sm in per_task.items():
        print(f"{t:13s} official {sm['official_success_at_L']}/{sm['n']} @L | "
              f"late {sm['late_conversions']}/{sm['n']} | "
              f"CP[{sm['cp_lower']:.4f},{sm['cp_upper']:.4f}] | {sm['verdict']}")
        print(f"               {sm['reason']}")
        if sm["late_success_steps"]:
            print(f"               late success steps: {sm['late_success_steps']}")
    if pooled:
        print("pooled n=40:", json.dumps(pooled, indent=2))
    print(f"duplicate-prefix: {'ALL MATCH' if dup_all_match else 'MISMATCH'}")
    print(json.dumps(summary["totals"], indent=2))
    print(f"cumulative spend {cum['_total']}/{cum['_cap_total']}")
    print(f"artifacts: {out}")
    return 0 if dup_all_match else 2


if __name__ == "__main__":
    raise SystemExit(main())
