#!/usr/bin/env python
"""Action 2P registration seal (daily 2026-08-20 §6 item 5).

No GPU, no environment.  Emits one content-addressed registration that the
2P runner verifies before spending a step, sealing exactly what the daily's
provenance clause requires: the actual runner and dependency closure, external
repository commit/dirty state, exact HF snapshot/weight hashes, BDDL/automaton,
segment-level official-versus-diagnostic accounting, and abort-safe cumulative
accounting.

    python scripts/register_v081p.py           # seal
    python scripts/register_v081p.py --check   # verify a sealed one
"""
from __future__ import annotations

import argparse, ast, hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from lcwm import v081p_contract as P  # noqa: E402

OUT = REPO / "results" / "v081p_registration"
ROOTS = ["lcwm/v081p_contract.py", "lcwm/v081p_exec.py", "lcwm/v08r_contract.py",
         "scripts/run_v081p.py", "scripts/register_v081p.py",
         "scripts/test_v081p_masks.py"]
EXTERNAL = ["lerobot", "libero", "robosuite", "mujoco", "torch", "transformers",
            "numpy", "huggingface_hub"]

sha256 = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _imports(path: Path) -> set[str]:
    out = set()
    try:
        tree = ast.parse(path.read_text())
    except Exception:
        return out
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and (n.module or "").startswith("lcwm"):
            out.add(n.module) if n.module != "lcwm" else out.update(
                f"lcwm.{a.name}" for a in n.names)
        elif isinstance(n, ast.Import):
            out |= {a.name for a in n.names if a.name.startswith("lcwm")}
    return out


def seal_sources() -> dict:
    seen, q = {}, [REPO / r for r in ROOTS]
    while q:
        p = q.pop()
        rel = str(p.relative_to(REPO))
        if rel in seen or not p.exists():
            continue
        seen[rel] = sha256(p)
        for m in _imports(p):
            c = REPO / (m.replace(".", "/") + ".py")
            if c.exists():
                q.append(c)
    for b in sorted((REPO / "bddl" / "chains").glob("chain1b*.bddl")):
        seen[str(b.relative_to(REPO))] = sha256(b)
    return dict(sorted(seen.items()))


def seal_model() -> dict:
    from lcwm.chassis import DEFAULT_MODEL
    info = {"model_id": DEFAULT_MODEL,
            "description": "LIBERO-finetuned pi0.5 as published; not tuned here"}
    try:
        from huggingface_hub import HfApi, snapshot_download
        info["hub_revision_sha"] = HfApi().model_info(DEFAULT_MODEL).sha
        snap = Path(snapshot_download(DEFAULT_MODEL))
        info["weight_files"] = {f.name: sha256(f)
                                for f in sorted(snap.glob("*.safetensors"))}
        if (snap / "config.json").exists():
            info["config_sha256"] = sha256(snap / "config.json")
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
    return info


def seal_external() -> dict:
    out = {}
    for n in EXTERNAL:
        e = {}
        try:
            m = __import__(n)
            e["version"] = getattr(m, "__version__", "unknown")
            f = getattr(m, "__file__", None)
            # Walk upward for a .git rather than assuming one fixed depth: an
            # editable install can sit at src/<pkg>/, so the old parent.parent
            # probe silently recorded NO external commit state at all.
            repo = None
            if f:
                cur = Path(f).parent
                for _ in range(5):
                    if (cur / ".git").exists():
                        repo = cur
                        break
                    if cur.parent == cur:
                        break
                    cur = cur.parent
            if repo is not None:
                e["git_repo"] = str(repo)
                e["git_sha"] = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                              capture_output=True, text=True).stdout.strip()
                e["git_dirty"] = bool(subprocess.run(
                    ["git", "-C", str(repo), "status", "--porcelain"],
                    capture_output=True, text=True).stdout.strip())
            else:
                e["git"] = "not_a_checkout"
        except Exception as ex:
            e["error"] = f"{type(ex).__name__}"
        out[n] = e
    return out


def git_state() -> dict:
    g = lambda *a: subprocess.run(["git", *a], cwd=REPO, capture_output=True,
                                  text=True).stdout.strip()
    porc = g("status", "--porcelain")
    return {"head": g("rev-parse", "HEAD"), "dirty": bool(porc),
            "porcelain": porc.splitlines()}


def registration() -> dict:
    return {
        "schema_version": P.SCHEMA_VERSION, "action": "Action 2P / V8.1P",
        "daily": "plan_and_progress/archive/daily/2026-08-20.md §4",
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "question": ("At distinct, stratum-matched pre-failure histories from stock "
                     "rollouts that miss L=250, do VLA-supported first-10-action "
                     "siblings produce repeatable local action-effect/outcome "
                     "variation, including both improvements and non-improving "
                     "outcomes under the same continuation noise?"),
        "non_claims": ["world-model prediction or model-guided acquisition",
                       "policy improvement", "two-task or LoHo generalization",
                       "continual learning",
                       "unlimited-horizon competence or terminal-failure recovery"],
        "data_use": ("rows are permanently excluded from later training and "
                     "behavior evaluation; no model or policy is updated"),
        "frozen": {
            "task": P.TASK, "policy": "stock pi0.5, full prompt, deployment N=1",
            "deadline_L": P.DEADLINE,
            "deadline_status": ("post-calibration EXPERIMENTAL finite-horizon "
                                "objective frozen before this pilot; not an "
                                "intrinsic-capability horizon and not an external "
                                "deployment requirement. Every allowed claim is "
                                "conditional on it."),
            "source_seeds": list(P.SOURCE_SEEDS),
            "source_rule": "run all 20; never early-stop at quota",
            "tau": P.TAU, "anchor_kind": P.ANCHOR_KIND,
            "tau_arithmetic": f"{P.TAU} + {P.C_PREFIX} + {P.H_PRIMARY} = {P.DEADLINE}",
            "eligible_mask": {k: sorted(v) for k, v in P.ELIGIBLE.items()},
            "selection": ("after the complete source panel closes, seal the first "
                          "eight eligible failures in seed order; underfill HALTs; "
                          "a failed replay is not replaced with the ninth source"),
            "source_suffix": ("establishes failure@250 only; conditioned to fail, "
                              "so it is never a reference outcome"),
        },
        "candidates": {
            "reference": ("first draw from a registered hash-derived stock-pi0 RNG "
                          "key, disjoint from the N=32 alternative keys, with no "
                          "action-geometry or outcome selection"),
            "n_alt_draws": P.N_ALT_DRAWS, "dedup": "executed first c=10 actions",
            "min_unique": P.MIN_UNIQUE_ALTS,
            "n_diversity": P.N_DIVERSITY, "n_random": P.N_RANDOM,
            "crn_keys": P.N_CRN,
            "blindness": ("no proposal feature, pool selection, or tie-break may use "
                          "the source trajectory after tau, its terminal mask/failure "
                          "phase, or any branch outcome"),
        },
        "outcomes": {
            "components": list(P.COMPONENTS), "lower_is_better": sorted(P.LOWER_IS_BETTER),
            "gamma": P.GAMMA, "readouts": list(P.H_READOUTS), "primary": P.H_PRIMARY,
            "pairing": ("same-key pairing, NOT best-of-three: a paired-positive beats "
                        "its matched reference on all three RNG keys at H=80; a "
                        "paired-non-improving never beats reference and is strictly "
                        "worse on at least one paired key beyond replay noise"),
            "excluded": ("reference variability and prefix physical effect by "
                         "themselves are not candidate-induced outcome variation"),
            "denominator": ("eight anchors, not 168 branches; repeats are nested "
                            "within an anchor"),
        },
        "budget": {"by_cap_line": P.INTERACTION_CAP,
                   "total_env_steps": P.INTERACTION_CAP_TOTAL,
                   "worst_case_segments": P.WORST_CASE_SEGMENTS,
                   "rule": ("every started or aborted segment is charged; no retries, "
                            "substitute anchors, extra assessment branches, post-hoc "
                            "quota increases, or reallocations"),
                   "pre_segment_headroom_check": True},
        "advance_gate": {
            "min_variation_anchors": P.GATE_MIN_VARIATION_ANCHORS,
            "min_positive_anchors": P.GATE_MIN_POSITIVE_ANCHORS,
            "min_both_anchors": P.GATE_MIN_BOTH_ANCHORS,
            "all_six_required": True,
            "halt_meaning": ("this anchor/proposal/setting did not expose sufficient "
                             "counterfactual support. It does not justify training or "
                             "sweeping a world model. The next change is one bounded "
                             "anchor or proposal/setting revision, not a lower bar, a "
                             "larger post-outcome budget, or benchmark shopping."),
            "allowed_statement": P.ALLOWED_ADVANCE_STATEMENT,
        },
        "terminology": {"allowed": ["failure@L", "right_censored@H",
                                    P.ANCHOR_KIND],
                        "forbidden_without_evidence": ["terminal", "irreducible",
                                                       "stall_onset",
                                                       "earliest_unrecoverable"]},
        "seeds": {"sources": [P.SOURCE_SEEDS[0], P.SOURCE_SEEDS[-1]],
                  "reserved_never_touched": [3200, 3399],
                  "calibration_only_spent": [3400, 3439],
                  "retired_not_reassigned": [3440, 3479]},
        "predecessor": {
            "action_1R_successor_note": "results/v080r_1r1/SUCCESSOR_NOTE_1R.json",
            "authoritative_1R_spend": 29800,
            "note": ("Action 1R's unused budget and unexecuted seeds are NOT reused; "
                     "Action 2P has its own independent cap.")},
        "provenance": {"git": git_state(), "sources": seal_sources(),
                       "model": seal_model(), "external": seal_external()},
    }


def verify_sealed(strict_git: bool = True) -> dict:
    path = OUT / "V081P_REGISTRATION.json"
    if not path.exists():
        return {"ok": False, "why": "no sealed registration"}
    reg = json.loads(path.read_text())
    body = {k: v for k, v in reg.items() if k != "self_sha256"}
    rec = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    drift = {k: v for k, v in reg["provenance"]["sources"].items()
             if not (REPO / k).exists() or sha256(REPO / k) != v}
    g = git_state()
    return {"ok": reg.get("self_sha256") == rec and not drift
                  and (not strict_git or not g["dirty"]),
            "self_sha256_match": reg.get("self_sha256") == rec,
            "self_sha256_recomputed": rec, "source_drift": drift, "git": g,
            "sealed_source_count": len(reg["provenance"]["sources"])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.check:
        print(json.dumps(verify_sealed(), indent=2)[:1500])
        return 0 if verify_sealed()["ok"] else 1
    reg = registration()
    body = {k: v for k, v in reg.items() if k != "self_sha256"}
    reg["self_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    (OUT / "V081P_REGISTRATION.json").write_text(json.dumps(reg, indent=2))
    print(json.dumps({"sealed": "results/v081p_registration/V081P_REGISTRATION.json",
                      "self_sha256": reg["self_sha256"][:16],
                      "sources_sealed": len(reg["provenance"]["sources"]),
                      "model": reg["provenance"]["model"].get("hub_revision_sha", "?")[:12],
                      "git_head": reg["provenance"]["git"]["head"][:12],
                      "git_dirty": reg["provenance"]["git"]["dirty"],
                      "cap_total": reg["budget"]["total_env_steps"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
