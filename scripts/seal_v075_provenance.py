#!/usr/bin/env python
"""V7.5 provenance seal — bind the record to CONTENT, not to a Git SHA.

Registered as action item 1 of `plan_and_progress/archive/daily/2026-08-09.md`
("Next mainline action items") and `framework_design.md` v0.9 §11.1:

    Land and freeze the evidence before new execution. Commit the actual
    V7.5 architecture, trainer, gate, daily amendment, and lightweight
    manifests; bind source-file hashes directly because the recorded Git
    SHA does not contain them.

Why a seal and not just a commit. Every V7.5 manifest names
`git_sha = 717a9da`, but that commit contains V7.4 code and no `v075_*`
implementation — the run's own lineage pointer is wrong, and committing
the files now cannot retroactively fix a manifest that was written
before the commit existed. So this script binds the run by the only
thing that is still true: the sha256 of the exact bytes on disk.

What it does, in four parts.

  code        Static import closure from every V7.5 entry point over
              repo-local modules, sha256 each. The closure is derived
              rather than hand-listed, so an inherited dependency
              (`v06_model`, `v073_schedule`, ...) cannot silently drop
              out of the record.
  artifacts   sha256 of every checkpoint, gate tensor, metric, and
              manifest under the V7.5 result roots. The 5825-key h-cache
              TENSORS are excluded by policy — they are regenerable and
              the exact training-time index is already lost (below) — but
              the index file itself is hashed.
  cross_checks
              Every hash a V7.5 manifest already claims is recomputed and
              compared. Checks marked `must_hold` are the run's internal
              consistency; a failure exits nonzero. Checks marked
              `known_broken` are the audited lineage defects, asserted to
              STILL be broken so that a later "fix" cannot go unnoticed.
  universe    The source/shard tensors needed to unroll the claimed
              training universe are verified present and hashed, per the
              action item's "do not delete source/shard tensors required
              to unroll a claimed training universe". `load_universe()`
              already asserts existence; this records identity too, so a
              future silent replacement is detectable.

The two audited defects are recorded, never repaired retroactively:

  1. `git_sha = 717a9da` predates the code it claims to describe.
  2. The 3978-key training h-cache index was overwritten in place by the
     5825-key policy-stage index. It is not recoverable — the stage
     counts do not even subtract cleanly (5825 - 1685 demo = 4140), so
     the training key set cannot be reconstructed by filtering.
     `scripts/build_v075_hcache.py` is patched so this cannot recur.

Output: `<date>_v075_seal_r1/provenance_seal.json`. Read-only apart from
that file.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
RID = "v075_seal_r1"

LCWM_ROOT = RESULTS / "2026-08-10_v075_lcwm_r1"
GATE_ROOT = RESULTS / "2026-08-10_v075_gate_r1"
DIAG_ROOT = RESULTS / "2026-08-09_v075_recurrence_diag_r1"
V074_ROOT = RESULTS / "2026-08-09_v074_data_r1"
SEMWIN = RESULTS / "2026-08-02_v071_semwin_r1"

# The six executables that produced every V7.5 artifact. Their transitive
# repo-local imports are derived, not listed.
ENTRYPOINTS = (
    "scripts/build_v075_hcache.py",
    "scripts/train_v075_lcwm.py",
    "scripts/gate_v075_binding.py",
    "scripts/diag_v075_recurrence.py",
    "scripts/diag_v075_repair_check.py",
    "scripts/test_v075_contract.py",
)

# Result roots whose non-tensor files are hashed wholesale. `hcache` is
# excluded by name (23 GB of regenerable prefix tensors).
ARTIFACT_ROOTS = (LCWM_ROOT, GATE_ROOT, DIAG_ROOT)
EXCLUDE_DIRS = {"hcache", "index_archive", "__pycache__"}

# The v074 inputs the V7.5 run read. Lightweight only; the tensors are
# covered by the universe section.
INPUT_FILES = (
    V074_ROOT / "anchor_manifest.json",
    V074_ROOT / "run_manifest.json",
    V074_ROOT / "source_split.json",
    V074_ROOT / "outcome_support.json",
    V074_ROOT / "physical_transition_index.jsonl",
    V074_ROOT / "semantic_query_index.jsonl",
    SEMWIN / "text_variants.json",
)

RECORDED_GIT_SHA = "717a9da"

# Code edited AFTER the run it produced. Its sha256 below is the current
# content, not the content the 2026-08-10 artifacts came from, and the
# seal says so rather than letting the hash imply run-exactness.
MODIFIED_AFTER_RUN = {
    "scripts/build_v075_hcache.py":
        "Patched 2026-08-15 while executing action item 1: "
        "`preserve_and_write` now archives any differing prior index "
        "before moving the pointer, so the in-place overwrite that "
        "destroyed the 3978-key training index cannot recur. The "
        "run-exact bytes were never committed and are NOT recoverable. "
        "What the builder produced is bound instead by "
        "hcache_manifest.json (enumeration rule, counts, input hashes) "
        "and by the surviving 5825-key index hashed here.",
}


def run_date() -> str:
    return subprocess.run(["date", "+%F"], capture_output=True, text=True,
                          env={"TZ": "America/Chicago"}).stdout.strip()


def _digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    return _digest(path, "sha256")


def md5_file(path: Path) -> str:
    return _digest(path, "md5")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _module_path(mod: str) -> Path | None:
    """Repo-local file for a dotted module name, or None if external."""
    parts = mod.split(".")
    flat = REPO_ROOT.joinpath(*parts).with_suffix(".py")
    if flat.is_file():
        return flat
    pkg = REPO_ROOT.joinpath(*parts) / "__init__.py"
    if pkg.is_file():
        return pkg
    return None


def _imports_of(path: Path) -> set[str]:
    """Dotted module names imported by `path`, absolute form.

    `from pkg import name` is ambiguous between a submodule and an
    attribute, so both `pkg.name` and `pkg` are emitted; `_module_path`
    discards whichever does not resolve to a file.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:            # relative — resolve against pkg
                base = path.resolve().parent
                for _ in range(node.level - 1):
                    base = base.parent
                try:
                    prefix = ".".join(base.relative_to(REPO_ROOT).parts)
                except ValueError:
                    continue
                mod = f"{prefix}.{node.module}" if node.module else prefix
            else:
                mod = node.module or ""
            if not mod:
                continue
            out.add(mod)
            for a in node.names:
                out.add(f"{mod}.{a.name}")
    return out


def import_closure(entrypoints: list[Path]) -> dict[str, Path]:
    """{dotted-or-script name -> file} over repo-local modules only."""
    found: dict[str, Path] = {}
    seen_files: set[Path] = set()
    queue = list(entrypoints)
    for p in entrypoints:
        found[rel(p)] = p
        seen_files.add(p.resolve())
    while queue:
        cur = queue.pop()
        for mod in sorted(_imports_of(cur)):
            mp = _module_path(mod)
            if mp is None or mp.resolve() in seen_files:
                continue
            seen_files.add(mp.resolve())
            found[mod] = mp
            queue.append(mp)
    return found


def git(*args: str) -> tuple[int, str]:
    r = subprocess.run(["git", *args], capture_output=True, text=True,
                       cwd=REPO_ROOT)
    return r.returncode, r.stdout.strip()


def walk_artifacts(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    out = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if EXCLUDE_DIRS & set(p.relative_to(root).parts):
            continue
        out.append(p)
    return out


def check(rows: list, name: str, recorded, actual, *, must_hold: bool,
          note: str = "") -> None:
    if actual is None:
        status = "missing"
    elif recorded is None:
        status = "not_recorded"
    elif recorded == actual:
        status = "match"
    else:
        status = "mismatch"
    ok = (status == "match") if must_hold else (status != "match")
    rows.append({"check": name, "recorded": recorded, "actual": actual,
                 "status": status,
                 "kind": "must_hold" if must_hold else "known_broken",
                 "ok": ok, "note": note})


def verify_init_sha(recorded: str) -> tuple[str | None, str]:
    """Recompute the trainer's fresh-init state hash.

    `scripts/train_v075_lcwm.py` sets `torch.manual_seed(0)` and then
    builds `V075State()` before anything else consumes the RNG, so the
    initial weights are reproducible on CPU. The recorded hash was taken
    after `.to(cuda)`, which is a bit-exact float32 round trip.
    """
    try:
        import torch
        from lcwm.v075_state import V075State
    except Exception as exc:                       # torch absent / import
        return None, f"not recomputed: {type(exc).__name__}: {exc}"
    torch.manual_seed(0)
    model = V075State()
    got = hashlib.sha256(b"".join(
        v.detach().cpu().numpy().tobytes()
        for _, v in sorted(model.state_dict().items())
        if v.dtype.is_floating_point)).hexdigest()
    same = "reproduced" if got == recorded else "DIVERGED"
    return got, f"torch.manual_seed(0) + V075State() on CPU — {same}"


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-universe-tensors", action="store_true",
                    help="record presence/size of the source and shard "
                         "tensors but do not hash 5.7 GB")
    ap.add_argument("--out", default=None,
                    help="override the <date>_v075_seal_r1 output root")
    args = ap.parse_args()

    date = run_date()
    out = Path(args.out) if args.out else RESULTS / f"{date}_{RID}"
    out.mkdir(parents=True, exist_ok=True)

    # ---- code -----------------------------------------------------
    entry_paths = [REPO_ROOT / e for e in ENTRYPOINTS]
    missing_entry = [rel(p) for p in entry_paths if not p.is_file()]
    assert not missing_entry, f"missing V7.5 entry points: {missing_entry}"
    closure = import_closure(entry_paths)
    code = {}
    for name in sorted(closure):
        p = closure[name]
        r = rel(p)
        entry = {"path": r, "bytes": p.stat().st_size,
                 "sha256": sha256_file(p),
                 "role": "entrypoint" if r in ENTRYPOINTS else "imported",
                 "run_exact": r not in MODIFIED_AFTER_RUN}
        if r in MODIFIED_AFTER_RUN:
            entry["modified_after_run"] = MODIFIED_AFTER_RUN[r]
        code[name] = entry
    # this file describes itself
    me = Path(__file__).resolve()
    code[rel(me)] = {"path": rel(me), "bytes": me.stat().st_size,
                     "sha256": sha256_file(me), "role": "seal",
                     "run_exact": None}
    n_dirty = sum(1 for v in code.values() if v["run_exact"] is False)
    print(f"[seal] code: {len(code)} files "
          f"({n_dirty} modified after their run)", flush=True)

    # ---- git lineage ----------------------------------------------
    rc_head, head = git("rev-parse", "--short", "HEAD")
    _, dirty = git("status", "--porcelain")
    in_recorded, in_head = {}, {}
    for name in sorted(code):
        path = code[name]["path"]
        in_recorded[path] = git("cat-file", "-e",
                                f"{RECORDED_GIT_SHA}:{path}")[0] == 0
        in_head[path] = git("cat-file", "-e", f"{head}:{path}")[0] == 0
    v075_files = [p for p in in_recorded if "v075" in Path(p).name]
    git_block = {
        "recorded_git_sha": RECORDED_GIT_SHA,
        "head_at_seal_time": head if rc_head == 0 else None,
        "worktree_dirty_at_seal_time": bool(dirty),
        "v075_files_present_in_recorded_sha":
            sorted(p for p in v075_files if in_recorded[p]),
        "v075_files_absent_from_recorded_sha":
            sorted(p for p in v075_files if not in_recorded[p]),
        "code_files_absent_from_head":
            sorted(p for p in in_head if not in_head[p]),
        "binding_rule": "This seal binds the V7.5 run by sha256 of file "
                        "CONTENT. The git_sha recorded in the run "
                        "manifests is known-wrong and is preserved "
                        "verbatim rather than rewritten.",
    }

    # ---- artifacts -------------------------------------------------
    artifacts = {}
    for root in ARTIFACT_ROOTS:
        for p in walk_artifacts(root):
            artifacts[rel(p)] = {"bytes": p.stat().st_size,
                                 "sha256": sha256_file(p)}
        print(f"[seal] artifacts: {rel(root)} done", flush=True)
    inputs = {}
    for p in INPUT_FILES:
        if p.is_file():
            inputs[rel(p)] = {"bytes": p.stat().st_size,
                              "sha256": sha256_file(p)}
        else:
            inputs[rel(p)] = {"bytes": None, "sha256": None,
                              "status": "MISSING"}

    # ---- cross-checks ----------------------------------------------
    rows: list = []
    man = json.loads((LCWM_ROOT / "run_manifest.json").read_text())
    hman = json.loads((LCWM_ROOT / "hcache_manifest.json").read_text())
    gate = json.loads((GATE_ROOT / "gate_report.json").read_text())

    final_pt = LCWM_ROOT / "checkpoints" / "final.pt"
    ep24 = LCWM_ROOT / "checkpoints" / "lcwm_epoch024.pt"
    check(rows, "gate.lcwm_sha256 == sha256(checkpoints/final.pt)",
          gate.get("lcwm_sha256"),
          sha256_file(final_pt) if final_pt.is_file() else None,
          must_hold=True,
          note="the gate ran on the checkpoint it names")
    check(rows, "md5(final.pt) == md5(lcwm_epoch024.pt)",
          md5_file(final_pt) if final_pt.is_file() else None,
          md5_file(ep24) if ep24.is_file() else None, must_hold=True,
          note="daily V7.5 Stage 2 claim: selection was final step only")

    for key, sub in (("anchor_states", gate.get("anchor_states", {})),
                     ("centering", gate.get("centering", {}))):
        rec_sha, rec_path = sub.get("sha256"), sub.get("path")
        act = None
        if rec_path:
            # the gate writes bare filenames relative to its own run root
            for cand in (Path(rec_path), REPO_ROOT / rec_path,
                         GATE_ROOT / rec_path):
                if cand.is_file():
                    act = sha256_file(cand)
                    break
        check(rows, f"gate.{key}.sha256 == sha256({rec_path})",
              rec_sha, act, must_hold=True,
              note="Stage 3 policy-state set is the one on disk")

    v074_am = V074_ROOT / "anchor_manifest.json"
    am_sha = sha256_file(v074_am) if v074_am.is_file() else None
    check(rows, "run_manifest.input_sha256.v074b_anchor_manifest.json",
          man.get("input_sha256", {}).get("v074b_anchor_manifest.json"),
          am_sha, must_hold=True,
          note="the trained universe is the collected universe")
    check(rows, "hcache_manifest.input_sha256.v074_anchor_manifest.json",
          hman.get("input_sha256", {}).get("v074_anchor_manifest.json"),
          am_sha, must_hold=True)
    tv = SEMWIN / "text_variants.json"
    check(rows, "hcache_manifest.input_sha256.text_variants.json",
          hman.get("input_sha256", {}).get("text_variants.json"),
          sha256_file(tv) if tv.is_file() else None, must_hold=True,
          note="cached prompts are the prompts the data was collected "
               "under")

    idx = LCWM_ROOT / "hcache_index.json"
    idx_sha = sha256_file(idx) if idx.is_file() else None
    n_keys = len(json.loads(idx.read_text())) if idx.is_file() else None
    check(rows, "run_manifest.input_sha256.hcache_index.json",
          man.get("input_sha256", {}).get("hcache_index.json"), idx_sha,
          must_hold=False,
          note=f"KNOWN BROKEN. The 3978-key training index was "
               f"overwritten in place by the {n_keys}-key policy-stage "
               f"index. Recorded, not repaired; the exact training "
               f"cache is unrecoverable.")

    init_recorded = man.get("init", {}).get("state_sha256")
    init_actual, init_note = verify_init_sha(init_recorded)
    check(rows, "run_manifest.init.state_sha256 (recomputed)",
          init_recorded, init_actual, must_hold=True, note=init_note)

    # ---- universe integrity ----------------------------------------
    universe = {"status": None, "anchors": {}}
    try:
        from lcwm import v075_schedule as S
        anchors, _tq = S.load_universe()
        src_dir = S.sources_for(S.UNIVERSE)
        shard_dir = S.v074_root() / "shards"
        for ak in sorted(anchors):
            a = anchors[ak]
            sid, d = a["source_id"], a["decision"]
            entry = {"task": a["task"], "split": a["split"],
                     "decision": d}
            for label, p in (("source", src_dir / f"{sid}.pt"),
                             ("shard", shard_dir / f"{sid}_d{d}.pt")):
                if not p.is_file():
                    entry[label] = {"path": rel(p), "status": "MISSING"}
                    continue
                rec = {"path": rel(p), "bytes": p.stat().st_size}
                if not args.skip_universe_tensors:
                    rec["sha256"] = sha256_file(p)
                entry[label] = rec
            universe["anchors"][ak] = entry
        universe["status"] = "loadable"
        universe["n_anchors"] = len(anchors)
        universe["hashed"] = not args.skip_universe_tensors
        print(f"[seal] universe: {len(anchors)} anchors verified",
              flush=True)
    except Exception as exc:
        universe["status"] = f"FAILED: {type(exc).__name__}: {exc}"

    # ---- assemble ---------------------------------------------------
    failed = [r for r in rows if not r["ok"]]
    seal = {
        "schema": "v075_provenance_seal_v1",
        "run_schema": "v075",
        "run_id": RID,
        "sealed_on": date,
        "registered_by": "plan_and_progress/archive/daily/2026-08-09.md — Next "
                         "mainline action items, item 1; "
                         "framework_design.md v0.9 §11.1",
        "purpose": "Bind the V7.5 evidence by content hash. The run "
                   "manifests name a Git SHA that predates their own "
                   "code, so file content is the only sound binding.",
        "git": git_block,
        "lineage_defects": [
            {"defect": "manifest git_sha predates the code",
             "detail": f"V7.5 manifests record git_sha "
                       f"{RECORDED_GIT_SHA}, which contains V7.4 code "
                       f"and no v075_* implementation.",
             "repaired": False,
             "mitigation": "this seal binds sha256 of the exact source "
                           "bytes; the V7.5 code is committed alongside "
                           "it"},
            {"defect": "training h-cache index overwritten",
             "detail": f"`--stage policy` rewrote hcache_index.json in "
                       f"place. The 3978-key training index "
                       f"(2394 unroll + 1584 next-obs) is gone; the "
                       f"present index has {n_keys} keys and cannot be "
                       f"filtered back (5825 - 1685 demo = 4140 != "
                       f"3978).",
             "repaired": False,
             "mitigation": "scripts/build_v075_hcache.py now archives "
                           "any differing prior index under "
                           "index_archive/ and appends to "
                           "index_lineage.jsonl before moving the "
                           "pointer. That patch postdates the run, so "
                           "the builder is marked run_exact: false."},
        ],
        "code": code,
        "code_fidelity": {
            "n_files": len(code),
            "run_exact": sorted(k for k, v in code.items()
                                if v["run_exact"] is True),
            "not_run_exact": sorted(k for k, v in code.items()
                                    if v["run_exact"] is False),
            "rule": "`run_exact: false` means the bytes hashed here are "
                    "NOT the bytes that produced the 2026-08-10 "
                    "artifacts. Such a file's hash binds the current "
                    "repository state only.",
        },
        "inputs": inputs,
        "artifacts": artifacts,
        "artifact_exclusions": {
            "hcache/": "23 GB of regenerable prompt-prefix tensors; the "
                       "index that names them is hashed instead",
        },
        "cross_checks": rows,
        "cross_check_summary": {
            "n": len(rows),
            "n_failed": len(failed),
            "failed": [r["check"] for r in failed],
        },
        "universe_integrity": universe,
    }
    dest = out / "provenance_seal.json"
    dest.write_text(json.dumps(seal, indent=2, sort_keys=False))
    print(f"[seal] wrote {rel(dest)}", flush=True)
    for r in rows:
        mark = "ok " if r["ok"] else "FAIL"
        print(f"  [{mark}] {r['status']:<12} {r['check']}")
    if failed:
        print(f"[seal] {len(failed)} check(s) failed", flush=True)
        sys.exit(1)
    print("[seal] all cross-checks behaved as recorded", flush=True)


if __name__ == "__main__":
    main()
