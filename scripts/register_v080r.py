#!/usr/bin/env python
"""Stage 1R.0 - executable registration and provenance closure (2026-08-19 §6).

No GPU, no environment, no rollout.  Emits ONE content-addressed registration
artifact that every later V8 stage cites.  It does not mutate
`BENCHMARK_FROZEN.json`: the 2026-08-19 audit reopened the derived branch
instrument, and an immutable artifact is superseded by a new one, never edited.

Closes the seven 1R.0 items and the provenance gaps the audit listed:
transitive repo-module hashes (not just the entry points), external
lerobot/LIBERO/robosuite/mujoco revisions, a full environment lock, the RNG
procedure, the duplicate-run tolerance, ledger role/subrole, and a hard
Action-1R interaction cap.

    python scripts/register_v080r.py            # write the registration
    python scripts/register_v080r.py --check    # verify a sealed one still matches
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lcwm import v08r_contract as C  # noqa: E402

OUT = REPO / "results" / "v080r_registration"

# Entry points whose transitive `lcwm.*` imports get sealed.
ROOTS = ["lcwm/v08r_contract.py", "lcwm/v080_bench.py",
         "scripts/screen_v080_benchmark.py", "scripts/register_v080r.py"]

EXTERNAL = ["lerobot", "libero", "robosuite", "mujoco", "torch",
            "transformers", "numpy", "gymnasium", "huggingface_hub"]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _lcwm_imports(path: Path) -> set[str]:
    """Module names imported from the `lcwm` package by one source file."""
    out: set[str] = set()
    try:
        tree = ast.parse(path.read_text())
    except Exception:
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("lcwm"):
            mod = node.module
            if mod == "lcwm":
                out |= {f"lcwm.{a.name}" for a in node.names}
            else:
                out.add(mod)
        elif isinstance(node, ast.Import):
            out |= {a.name for a in node.names if a.name.startswith("lcwm")}
    return out


def seal_sources() -> dict:
    """Transitive closure over repo modules - the audit's specific complaint
    was that `task_automaton.py`, `seq_data.py` and `probe_data.py` were
    imported but unsealed."""
    seen: dict[str, str] = {}
    queue = [REPO / r for r in ROOTS]
    while queue:
        p = queue.pop()
        rel = str(p.relative_to(REPO))
        if rel in seen or not p.exists():
            continue
        seen[rel] = sha256(p)
        for mod in _lcwm_imports(p):
            cand = REPO / (mod.replace(".", "/") + ".py")
            if cand.exists():
                queue.append(cand)
    for b in sorted((REPO / "bddl" / "chains").glob("*.bddl")):
        seen[str(b.relative_to(REPO))] = sha256(b)
    return dict(sorted(seen.items()))


def seal_external() -> dict:
    out = {}
    for name in EXTERNAL:
        entry: dict = {}
        try:
            mod = __import__(name)
            entry["version"] = getattr(mod, "__version__", "unknown")
            f = getattr(mod, "__file__", None)
            if f:
                entry["path"] = str(Path(f).parent)
                git = Path(f).parent.parent / ".git"
                if git.exists():
                    r = subprocess.run(["git", "-C", str(git.parent), "rev-parse", "HEAD"],
                                       capture_output=True, text=True)
                    if r.returncode == 0:
                        entry["git_sha"] = r.stdout.strip()
                        d = subprocess.run(["git", "-C", str(git.parent), "status",
                                            "--porcelain"], capture_output=True, text=True)
                        entry["git_dirty"] = bool(d.stdout.strip())
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {e}"
        out[name] = entry
    return out


def seal_environment() -> dict:
    env = {"python": sys.version, "executable": sys.executable}
    try:
        r = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                           capture_output=True, text=True, timeout=180)
        env["pip_freeze_sha256"] = hashlib.sha256(r.stdout.encode()).hexdigest()
        env["pip_freeze_lines"] = len(r.stdout.splitlines())
        (OUT).mkdir(parents=True, exist_ok=True)
        (OUT / "pip_freeze.txt").write_text(r.stdout)
        env["pip_freeze_file"] = "pip_freeze.txt"
    except Exception as e:
        env["pip_freeze_sha256"] = f"unavailable: {type(e).__name__}"
    return env


def git_state() -> dict:
    def g(*a):
        r = subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    porcelain = g("status", "--porcelain")
    return {"head": g("rev-parse", "HEAD"), "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(porcelain),
            "untracked_or_modified": [l for l in porcelain.splitlines()]}


def registration() -> dict:
    return {
        "schema_version": C.SCHEMA_VERSION,
        "stage": "1R.0",
        "action": "Action 1R / V8.0R - executable registration and provenance closure",
        "framework": "v1.0 §14",
        "daily": "plan_and_progress/2026-08-19.md §6",
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "supersedes": {
            "artifact": "results/v080_screen/BENCHMARK_FROZEN.json",
            "not_mutated": True,
            "what_survives": ("the V8.0 behavioral gate: chain1b 12/30 and "
                              "chain2b 20/30 at the frozen 250/500-step "
                              "deadlines, hash-bound"),
            "what_is_reopened": ("W, H_max, H, the per-class bounds, the "
                                 "'MC fallback does not fire' conclusion, and "
                                 "the scalar-phase aggregation - all were "
                                 "complete-case and context-aliased"),
        },

        # ---- 1R.0 item 1: artifact identity -----------------------------
        "artifact_identity": {
            "rule": "anchor_id + snapshot content hash",
            "anchor_id": "sha256(task|source_episode_id|step)[:16]",
            "content_hash": ("sha256 over the concatenated sim_state, "
                             "controller state, obs window, action history and "
                             "task-monitor bits, in that fixed order"),
            "note": ("two anchors with equal content hash are the same anchor "
                     "even if reached by different episodes; two with equal id "
                     "and different content hash are a provenance failure"),
        },

        # ---- 1R.0 item 2: strata ----------------------------------------
        "stratum": {
            "identity": "task + ever_achieved + current_valid + damaged (SETS)",
            "key_format": C.stratum_key(
                "<task>", ("pick_up X", "place X R"), [0], [0], []).key(),
            "covariates_not_identity": ["step", "time_to_go"],
            "time_to_go_bins": "[0,25) [25,50) [50,100) [100,200) [200,inf)",
            "explicit_fields": [f.name for f in __import__("dataclasses")
                                .fields(C.StratumKey)],
            "actionability_rule": ("place-k is actionable iff pick-k is "
                                   "CURRENTLY valid; pick-k is actionable "
                                   "whenever unresolved"),
            "scalar_phase": "DISPLAY ONLY - never an aggregation or anchor key",
            "why": ("V8.0 chain2b phase=2 pooled 5 tomato-done/cream-unresolved "
                    "with 2 cream-done/tomato-unresolved episodes: mutually "
                    "exclusive remaining problems under one number"),
            "frontier_rule": ("contraction of the unresolved SET, or success; "
                              "never an ordered-prefix increase - all 20 "
                              "chain2b successes ran cream->tomato, the "
                              "reverse of the listed order"),
        },

        # ---- 1R.0 item 3: deadline arithmetic ---------------------------
        "deadline_arithmetic": {
            "c": 10,
            "c_eff": "min(c, max(0, L - tau))",
            "D": "max(0, L - tau - c_eff)",
            "anchor_admissible": ("c_eff == c AND D >= min_continuation(class); "
                                  "min_continuation is the achieved-state-"
                                  "conditioned class bound, NOT global H_max"),
            "diagnostic_readouts": list(C.DIAGNOSTIC_READOUTS),
            "continuation_len": C.CONTINUATION_LEN,
            "readout_note": ("readouts are steps FROM THE ANCHOR and the later "
                             "ones land past L by construction; they are "
                             "diagnostic and quarantined, and the 1R.2 cap is "
                             "sized from CONTINUATION_LEN so the schedule and "
                             "the budget cannot drift apart"),
        },

        # ---- 1R.0 item 4: officiality ------------------------------------
        "officiality": {
            "rule": "official iff step <= L; beyond L is quarantined diagnostic",
            "prohibited_after_L": ["success", "correction", "value label",
                                   "official environment step"],
            "quarantine_label": C.Officiality.QUARANTINED_POST_DEADLINE.value,
        },

        # ---- 1R.0 item 5: stall + backoff --------------------------------
        "stall_rule": {
            "observable": "W_used = min(W_derived, L - tau); observable iff L - tau >= W",
            "prohibition": ("a state may not be called a W-step stall_onset "
                            "unless the window was observable; W=345 exceeds "
                            "chain1b's entire 250-step episode"),
            "deadline_backoff": ("anchors must satisfy anchor_admissible; "
                                 "applied to the V8.0 confirmation data this "
                                 "already rejects 16/16 chain1b last-progress "
                                 "anchors and 3/8 chain2b ones"),
            "fqe_caveat": ("FQE may replace an unidentified value target, but "
                           "does not make an unobservable stall window or an "
                           "inadmissible late anchor valid"),
        },

        # ---- 1R.0 item 6: seals ------------------------------------------
        "rng_procedure": {
            "rule": ("per episode: torch.manual_seed(seed), "
                     "numpy.random.seed(seed), torch.cuda.manual_seed_all(seed) "
                     "before runner.reset() and env.reset(seed=seed)"),
            "duplicate_run_tolerance": {
                "milestone_step": 0,
                "terminal_step": 0,
                "success_flag": 0,
                "note": ("exact equality required on the duplicate-prefix "
                         "subset; any mismatch is a provenance HALT, not a "
                         "tolerance to widen"),
            },
        },
        "ledger": {"roles": [r.value for r in C.Role],
                   "subroles": [s.value for s in C.Subrole],
                   "every_step_carries": ["role", "subrole", "task", "seed",
                                          "anchor_id", "policy_hash", "n_steps",
                                          "officiality"]},
        "interaction_cap": {"by_stage_task": C.INTERACTION_CAP,
                            "total_env_steps": C.INTERACTION_CAP_TOTAL,
                            "rule": ("hard cap; an overrun halts the stage. "
                                     "Unused budget is never reallocated.")},

        # ---- 1R.0 item 7: decision rules ---------------------------------
        "decision_rules": {
            "materiality": C.MATERIALITY,
            "panels": [C.PANEL_1, C.PANEL_2],
            "max_panels": C.MAX_PANELS,
            "interval": "exact one-sided Clopper-Pearson at alpha = 0.05",
            "worked_bounds": {
                "0/20_upper": round(C.clopper_pearson_upper(0, 20), 4),
                "3/20_upper": round(C.clopper_pearson_upper(3, 20), 4),
                "4/20_lower": round(C.clopper_pearson_lower(4, 20), 4),
                "6/40_upper": round(C.clopper_pearson_upper(6, 40), 4),
                "6/40_lower": round(C.clopper_pearson_lower(6, 40), 4),
            },
            "underfill": {
                "target_groups": C.TARGET_GROUPS,
                "repeats_per_group": C.REPEATS_PER_GROUP,
                "source_seed_cap": C.SOURCE_SEED_CAP,
                "rule": ("underfill at the cap is STRATUM_NOT_IDENTIFIED; "
                         "substituting an easier state or raising the cap is "
                         "prohibited"),
            },
            "band_notation": ("success band is CLOSED [0.20, 0.70]; the "
                              "'(0.2, 0.7)' rendering in V8.0 generated text "
                              "is a display bug, not a rule change - neither "
                              "confirmation rate lies on a boundary"),
        },

        "seeds": {
            "1R.1_panel_1": [3400, 3419], "1R.1_panel_2_conditional": [3420, 3439],
            "1R.2_source": [3440, 3479],
            "RESERVED_never_touched": {"behavior": [3200, 3259],
                                       "acquisition": [3300, 3399]},
            "already_spent_calibration": {"screen": [3000, 3009],
                                          "confirm": [3100, 3129]},
        },

        # ---- 1R.3 handoff schema (produced later, defined now) -----------
        "handoff_schema_1R3": {
            "file": "V081_HANDOFF.json",
            "immutable": True,
            "required_fields": ["retained_tasks", "admissible_strata_per_task",
                                "anchor_rule", "deadline_rule",
                                "estimator_choice", "v81_quotas",
                                "source_roles", "seed_map",
                                "interaction_budget", "pass_halt_denominators",
                                "queue_underfill_behavior",
                                "registration_sha256"],
        },

        "provenance": {"git": git_state(), "sources": seal_sources(),
                       "external": seal_external(), "environment": seal_environment()},
    }


def verify_sealed(strict_git: bool = True) -> dict:
    """Preflight every stage must pass before spending an environment step.

    Recomputes the registration's own hash rather than trusting the value it
    reports about itself, re-hashes all sealed sources, and reports tree
    cleanliness.  The registration's own identity rule - "two artifacts with
    equal id and different content hash are a provenance failure" - is
    unenforceable if the consumer copies the claimed hash instead of checking it.
    """
    path = OUT / "V080R_REGISTRATION.json"
    if not path.exists():
        return {"ok": False, "why": "no sealed registration"}
    reg = json.loads(path.read_text())
    claimed = reg.get("self_sha256")
    body = {k: v for k, v in reg.items() if k != "self_sha256"}
    recomputed = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    drift = {k: {"sealed": v, "now": sha256(REPO / k) if (REPO / k).exists() else None}
             for k, v in reg["provenance"]["sources"].items()
             if not (REPO / k).exists() or sha256(REPO / k) != v}
    g = git_state()
    ok = (claimed == recomputed) and not drift and (not strict_git or not g["dirty"])
    return {"ok": ok, "registration_path": str(path.relative_to(REPO)),
            "self_sha256_claimed": claimed, "self_sha256_recomputed": recomputed,
            "self_sha256_match": claimed == recomputed,
            "source_drift": drift, "git": g,
            "sealed_source_count": len(reg["provenance"]["sources"])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "V080R_REGISTRATION.json"

    reg = registration()
    body = {k: v for k, v in reg.items() if k != "self_sha256"}
    reg["self_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True).encode()).hexdigest()

    if args.check:
        if not path.exists():
            print("no sealed registration"); return 1
        old = json.loads(path.read_text())
        drift = {k for k in body
                 if k not in ("utc", "provenance") and old.get(k) != body[k]}
        src_drift = {k for k, v in body["provenance"]["sources"].items()
                     if old["provenance"]["sources"].get(k) != v}
        print(json.dumps({"rule_drift": sorted(drift),
                          "source_drift": sorted(src_drift)}, indent=2))
        return 1 if (drift or src_drift) else 0

    path.write_text(json.dumps(reg, indent=2))
    p = reg["provenance"]
    print(json.dumps({
        "sealed": str(path.relative_to(REPO)),
        "self_sha256": reg["self_sha256"][:16],
        "repo_sources_sealed": len(p["sources"]),
        "external_sealed": {k: v.get("version", v.get("error"))
                            for k, v in p["external"].items()},
        "git_head": p["git"]["head"][:12], "git_dirty": p["git"]["dirty"],
        "interaction_cap_total": reg["interaction_cap"]["total_env_steps"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
