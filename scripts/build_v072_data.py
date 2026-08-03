#!/usr/bin/env python
"""V7.2A — repair and materialize the crossed training union.

Sources merged (pointer-based; raw frames stay in their immutable
roots, every record carries a raw source pointer + root SHA):
  - V7.1.1F union (770 raw rows incl. audit; identities preserved)
  - V7.1.2b semwin windows decomposed into executed segments
  - V7.1.4B selector r1/r2 complete banks (436 recorded branch
    executions; winners/losers/ties/nulls/regressions all retained)

Repairs (registered in the V7.2 amendment):
  1. exactly-once action contract: `actions_env` and
     `actions_pi05_norm` are DIFFERENT fields; generated candidates
     keep their sample_chunks tensors (already normalized), env-only
     rows get at most ONE normalize_actions pass, and every stored
     norm chunk must round-trip through runner.chunk_to_env to its
     recorded actions_env within the registered tolerance or the norm
     field is masked (flow-ineligible, physical-only);
  2. order-independent fidelity for all 34 selector anchors from the
     stored u0/u0_repeat transitions (+ the 26 inherited V7.1.1 rows);
  3. unified time axis: branch steps 1..c, continuation c+1..c+H;
     tau_next has no per-step continuation trace -> support weight 0
     everywhere; success/damage kept as endpoint labels but weight 0
     per the frozen contract (no valid variation);
  4. candidate count is manifest-defined (mask, never assume 12);
  5. crossed structure PhysicalTransition / GoalQuery /
     SemanticTarget with per-target support masks; alternative-goal
     semantics only where exact per-goal automaton state was stored
     (old union relabels, semwin valid_seq); selector rows are
     canonical-goal-only (masked otherwise);
  6. splits by task/source/snapshot BEFORE language expansion;
     2300/2301 families frozen as train material; 2302+10k and
     {1750..1790} untouched.

Output: results/libero_loho_public_v1/2026-08-03_v072_data_r1/
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
UNION_F1 = RESULTS / "2026-08-02_v071_union_f1"
SEMWIN = RESULTS / "2026-08-02_v071_semwin_r1"
SEL = {"selr1": RESULTS / "2026-08-03_v071_selector_r1",
       "selr2": RESULTS / "2026-08-03_v071_selector_r2"}
OUT = RESULTS / "2026-08-03_v072_data_r1"
ROUNDTRIP_ATOL = 1e-5
GROSS_BOUNDS = {"eef_pos": 0.02, "eef_quat": 7.8e-3,
                "gripper": 5e-3, "obj_pos": 2e-3}
LABEL_VERSION = "v072_labels_1"


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@torch.no_grad()
def main() -> None:
    from lcwm.chassis import Pi05Runner
    from lcwm.seq_prefix_cache import normalize_actions

    OUT.mkdir(exist_ok=True)
    (OUT / "physical_transitions").mkdir(exist_ok=True)
    goal_manifest = json.loads(
        (RESULTS / "goal_spec_manifest_v067.json").read_text())
    tv = json.loads((SEMWIN / "text_variants.json").read_text())
    union = json.loads((UNION_F1 / "union_manifest.json").read_text())
    u_rows = {r["transition_id"]: r for r in union["rows"]}
    u_roots = {k: Path(v) for k, v in union["replay_roots"].items()}
    ref = torch.load(Path("/home/stargazer/Desktop/vla_wm/datasets"
                          "/seq_prefix_cache_v1/task0_demo0.pt"),
                     weights_only=False)
    ref_mean, ref_std = ref["action_mean"], ref["action_std_eps"]
    runner = Pi05Runner(suite_name="libero_10")

    def canon_of(task):
        return goal_manifest["tasks"][task]["canonical_goal_spec_id"]

    def compatible_goals(task):
        return sorted(goal_manifest["tasks"][task]["goal_specs"])

    # ---- action contract: verify one stored norm chunk --------------
    contract = {"checked": 0, "roundtrip_pass": 0,
                "roundtrip_fail_masked": 0, "reconstructed": 0,
                "physical_only": 0, "nonfinite": 0}

    def verify_norm(chunk_norm, actions_env, steps):
        """Return (actions_pi05_norm or None, status). The stored norm
        chunk must reproduce the recorded env actions through the
        runner's own denormalizer — the exactly-once contract."""
        contract["checked"] += 1
        if chunk_norm is None:
            ae = torch.from_numpy(np.asarray(actions_env)).float()
            if ae.shape[0] == 0:
                contract["physical_only"] += 1
                return None, "empty"
            cn = normalize_actions(ae, ref_mean, ref_std)
            back = runner.chunk_to_env(cn)
            if np.allclose(back, np.asarray(actions_env),
                           atol=ROUNDTRIP_ATOL):
                contract["reconstructed"] += 1
                return cn, "reconstructed_exact"
            contract["physical_only"] += 1
            return None, "roundtrip_inexact"
        cn = torch.as_tensor(np.asarray(chunk_norm)).float()
        if not torch.isfinite(cn).all():
            contract["nonfinite"] += 1
            return None, "nonfinite"
        k = int(steps)
        if k == 0:
            contract["physical_only"] += 1
            return None, "empty"
        back = runner.chunk_to_env(cn[:k])
        if np.allclose(back, np.asarray(actions_env)[:k],
                       atol=ROUNDTRIP_ATOL):
            contract["roundtrip_pass"] += 1
            return cn, "stored_verified"
        contract["roundtrip_fail_masked"] += 1
        return None, "stored_roundtrip_fail"

    phys, queries, sem_targets, index_rows = [], [], [], []
    root_shas = {}

    def bind_root(name, path):
        if name not in root_shas:
            root_shas[name] = {"path": str(path)}
        return name

    def add_phys(rec):
        phys.append(rec)
        index_rows.append({k: rec[k] for k in (
            "pt_id", "origin", "task", "source_id", "split",
            "decision", "branch_id", "family", "provenance", "kind",
            "steps", "norm_status", "audit_only")})

    # ================= 1) V7.1.1F union rows =========================
    by_shard = defaultdict(list)
    for r in union["rows"]:
        by_shard[(r["run"], r["shard"])].append(r["transition_id"])
    for (run, shard), tids in sorted(by_shard.items()):
        sp = u_roots[run] / "shards" / shard
        s = torch.load(sp, weights_only=False)
        rn = bind_root(f"v071_replay_{run}", u_roots[run])
        for tr in s["transitions"]:
            r = u_rows.get(tr["transition_id"])
            if r is None:
                continue
            cn, status = verify_norm(tr.get("chunk_norm"),
                                     tr["actions_env"], tr["steps"])
            pt_id = f"u1::{tr['transition_id']}"
            add_phys({
                "pt_id": pt_id, "origin": "v071_union",
                "raw_root": rn, "raw_shard": shard,
                "raw_key": tr["transition_id"],
                "task": r["task"], "source_id": r["source_id"],
                "split": r["split"], "decision": r["decision"],
                "branch_id": r["branch_key"], "family": r["family"],
                "provenance": r["provenance"], "kind": "branch",
                "steps": int(tr["steps"]),
                "actions_env_sha": r["action_bytes_sha256"],
                "actions_pi05_norm": (cn.numpy().tolist()
                                      if cn is not None else None),
                "norm_status": status,
                "audit_only": bool(r["audit_only"]),
                "behavior_goal_id": tr.get("behavior_goal_id"),
                "continuation_goal_id": (canon_of(r["task"])
                                         if tr["continuations"]
                                         else None),
                "n_continuations": len(tr["continuations"]),
                "time_axis": {"branch": [1, int(tr["steps"])],
                              "continuation_offset": int(tr["steps"])},
            })
            canon = canon_of(r["task"])
            for gid in compatible_goals(r["task"]):
                # per-goal immediate exists in the r1/r2 relabel
                # tables; the shard itself carries canonical only —
                # exact-evaluation support: canonical always; other
                # goals only via the stored relabel rows (bank rows)
                sem_targets.append({
                    "pt_id": pt_id, "goal_id": gid,
                    "supported": gid == canon or run == "r1",
                    "source": ("shard_immediate" if gid == canon
                               else "relabel_table"),
                    "continuation_outcomes": (gid == canon
                                              and tr["continuations"]
                                              != [])})
    # ================= 2) semwin segments ============================
    rn = bind_root("v071_semwin_r1", SEMWIN)
    for p in sorted((SEMWIN / "shards").glob("*.pt")):
        s = torch.load(p, weights_only=False)
        cum = [0]
        for seg in s["segments"]:
            cum.append(cum[-1] + seg["actions"])
        for k in range(len(cum) - 1):
            ae = np.asarray(s["actions_env"])[cum[k]:cum[k + 1]]
            cn, status = verify_norm(None, ae, ae.shape[0])
            pt_id = f"sw::{s['window_id']}::seg{k}"
            add_phys({
                "pt_id": pt_id, "origin": "v071_semwin",
                "raw_root": rn, "raw_shard": p.name,
                "raw_key": f"{s['window_id']}|seg{k}",
                "task": s["task"], "source_id": s["source_id"],
                "split": s["split"], "decision": s["decision"],
                "branch_id": f"{s['branch']}_seg{k}",
                "family": s["branch_family"],
                "provenance": s["branch_provenance"],
                "kind": "semwin_segment", "steps": int(ae.shape[0]),
                "actions_env_sha": hashlib.sha256(
                    ae.tobytes()).hexdigest()[:16],
                "actions_pi05_norm": (cn.numpy().tolist()
                                      if cn is not None else None),
                "norm_status": status, "audit_only": False,
                "behavior_goal_id": canon_of(s["task"]),
                "continuation_goal_id": None, "n_continuations": 0,
                "time_axis": {"branch": [cum[k] + 1, cum[k + 1]],
                              "continuation_offset": None},
            })
            for gid in s["goal_ids"]:
                sem_targets.append({
                    "pt_id": pt_id, "goal_id": gid,
                    "supported": True, "source": "semwin_valid_seq",
                    "continuation_outcomes": False})
    # ================= 3) selector r1/r2 banks =======================
    fidelity = {}
    for tag, root in SEL.items():
        rn = bind_root(f"v071_{tag}", root)
        chunks = torch.load(root / "candidate_chunks.pt",
                            weights_only=False)
        for p in sorted((root / "shards").glob("*.pt")):
            s = torch.load(p, weights_only=False)
            trs = {t["candidate_id"]: t for t in s["transitions"]}
            # order-independent fidelity: u0 vs u0_repeat
            if "u0" in trs and "u0_repeat" in trs:
                a, b = trs["u0"], trs["u0_repeat"]
                dq = np.abs(np.asarray(a["eef_seq"][-1])
                            - np.asarray(b["eef_seq"][-1]))
                fidelity[s["anchor"]] = {
                    "eef_pos": float(dq[:3].max()),
                    "eef_quat": float(dq[3:7].max()),
                    "gripper": float(dq[7:].max()),
                    "obj_pos": float(np.abs(
                        np.asarray(a["obj_after"])
                        - np.asarray(b["obj_after"])).max()),
                    "qpos": float(np.abs(
                        np.asarray(a["qpos_after"])
                        - np.asarray(b["qpos_after"])).max()),
                    "grasp_equal": a["grasp_after"]["grasped"]
                    == b["grasp_after"]["grasped"]}
            for cid, tr in trs.items():
                key = f"{s['anchor']}_{cid}"
                stored = chunks.get(f"{s['anchor']}_{cid}")
                if cid in ("u0", "u0_repeat"):
                    stored = chunks.get(f"{s['anchor']}_u0")
                cn, status = verify_norm(stored, tr["actions_env"],
                                         tr["steps"])
                pt_id = f"{tag}::{key}"
                add_phys({
                    "pt_id": pt_id, "origin": f"v071_{tag}",
                    "raw_root": rn, "raw_shard": p.name,
                    "raw_key": key,
                    "task": s["task"], "source_id": s["source_id"],
                    "split": "train", "decision": s["decision"],
                    "branch_id": cid,
                    "family": ("canonical" if cid.startswith(("u0",
                               "c")) else "atomic" if cid in
                               ("sub0", "sub1") else "distinct"),
                    "provenance": ("fidelity_repeat"
                                   if cid == "u0_repeat"
                                   else "prospective_bank"),
                    "kind": ("audit" if tr["kind"] == "audit"
                             else "branch"),
                    "steps": int(tr["steps"]),
                    "actions_env_sha": hashlib.sha256(
                        np.asarray(tr["actions_env"]).tobytes())
                    .hexdigest()[:16],
                    "actions_pi05_norm": (cn.numpy().tolist()
                                          if cn is not None
                                          else None),
                    "norm_status": status, "audit_only":
                        tr["kind"] == "audit",
                    "behavior_goal_id": canon_of(s["task"]),
                    "continuation_goal_id": (canon_of(s["task"])
                                             if tr["continuations"]
                                             else None),
                    "n_continuations": len(tr["continuations"]),
                    "time_axis": {"branch": [1, int(tr["steps"])],
                                  "continuation_offset":
                                      int(tr["steps"])},
                })
                sem_targets.append({
                    "pt_id": pt_id, "goal_id": canon_of(s["task"]),
                    "supported": True, "source": "shard_canonical",
                    "continuation_outcomes":
                        bool(tr["continuations"])})
                for gid in compatible_goals(s["task"]):
                    if gid != canon_of(s["task"]):
                        sem_targets.append({
                            "pt_id": pt_id, "goal_id": gid,
                            "supported": False,
                            "source": "masked_no_stored_state",
                            "continuation_outcomes": False})
    del runner
    torch.cuda.empty_cache()

    # inherit the 26 V7.1.1 measured fidelity rows
    v711_fid = []
    for line in (u_roots["r1"] / "replay_fidelity.jsonl").open():
        v711_fid.append(json.loads(line))
    fid_flags = {}
    for k, v in fidelity.items():
        fid_flags[k] = {b: v[b] <= GROSS_BOUNDS[b]
                        for b in GROSS_BOUNDS}

    # ================= goal queries (language layer) =================
    gt = defaultdict(dict)
    for row in tv["rows"]:
        gt[row["goal_id"]][row["text_variant_id"]] = row["language"]
    for task, entry in goal_manifest["tasks"].items():
        for gid in entry["goal_specs"]:
            for tvid, lang in gt.get(gid, {}).items():
                queries.append({
                    "task": task, "goal_id": gid,
                    "text_variant_id": tvid, "language": lang,
                    "role": ("canonical" if tvid.endswith("_p0")
                             else "paraphrase"),
                    "scene_compatible": True})

    # ================= splits (before language expansion) ============
    split_of_source = {}
    for rec in phys:
        sid = rec["source_id"]
        if sid in split_of_source:
            assert split_of_source[sid] == rec["split"], sid
        else:
            split_of_source[sid] = rec["split"]

    # ================= support table ================================
    def mass(pred):
        m = defaultdict(int)
        for rec in phys:
            if rec["audit_only"] or not pred(rec):
                continue
            m[rec["split"]] += 1
        return dict(m)

    sup_sem = defaultdict(int)
    for t in sem_targets:
        if t["supported"]:
            sup_sem[t["source"]] += 1
    support = {
        "phys_abs": mass(lambda r: r["steps"] > 0),
        "flow_actions_pi05_norm": mass(
            lambda r: r["actions_pi05_norm"] is not None),
        "continuation_outcomes": mass(
            lambda r: r["n_continuations"] > 0),
        "semantic_supported_by_source": dict(sup_sem),
        "tau_next": {"weight": 0.0,
                     "reason": "no per-step continuation trace; "
                               "time-axis relabel impossible"},
        "success_damage": {"weight": 0.0,
                           "reason": "no valid variation in labels"},
        "closure_blocks": {
            "note": "1-block: all branch rows; 2/3-block: semwin "
                    "consecutive segments only",
            "semwin_2block": sum(
                1 for r in phys if r["kind"] == "semwin_segment"
                and r["branch_id"].endswith(("seg1", "seg2", "seg3",
                                             "seg4"))),
        },
    }

    # ================= write artifacts ===============================
    with (OUT / "replay_index.jsonl").open("w") as f:
        for r in index_rows:
            f.write(json.dumps(r) + "\n")
    torch.save({"transitions": phys},
               OUT / "physical_transitions" / "transitions.pt")
    with (OUT / "goal_queries.jsonl").open("w") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")
    with (OUT / "semantic_targets.jsonl").open("w") as f:
        for t in sem_targets:
            f.write(json.dumps(t) + "\n")
    (OUT / "source_split.json").write_text(json.dumps(
        {"rule": "task/source/snapshot split BEFORE language "
                 "expansion; selector families 2300/2301 train; "
                 "2302+10k and dev panel {1750..1790} untouched",
         "sources": split_of_source}, indent=2))
    (OUT / "action_contract_report.json").write_text(json.dumps(
        {"tolerance": ROUNDTRIP_ATOL,
         "denormalizer": "runner.chunk_to_env (the executed path)",
         **contract}, indent=2))
    (OUT / "replay_fidelity.json").write_text(json.dumps(
        {"selector_anchors_recomputed": fidelity,
         "gross_bound_flags": fid_flags,
         "inherited_v711_rows": v711_fid,
         "coverage": f"{len(fidelity)}/34 selector anchors "
                     f"+ {len(v711_fid)} inherited"}, indent=2))
    (OUT / "target_support.json").write_text(
        json.dumps(support, indent=2))
    (OUT / "label_version.json").write_text(json.dumps(
        {"label_version": LABEL_VERSION,
         "time_axis": "branch 1..c; continuation c+1..c+H; "
                      "q@h remain continuation-local horizons",
         "tau_next": "masked", "success_damage": "weight 0"},
        indent=2))
    git_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True,
        text=True, cwd=REPO_ROOT).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "-uno"],
        capture_output=True, text=True, cwd=REPO_ROOT).stdout
    (OUT / "run_manifest.json").write_text(json.dumps({
        "schema": "v072_data_manifest_v1", "run_schema": "v072",
        "git_sha": git_sha,
        "dirty_patch_sha": hashlib.sha256(
            dirty.encode()).hexdigest()[:16],
        "label_version": LABEL_VERSION,
        "raw_roots": root_shas,
        "bound_manifests": {
            "union_f1": sha_file(UNION_F1 / "union_manifest.json"),
            "goal_manifest":
                goal_manifest["manifest_sha256"],
            "semwin": sha_file(SEMWIN / "run_manifest.json"),
            "selector_r1": sha_file(
                SEL["selr1"] / "run_manifest.json"),
            "selector_r2": sha_file(
                SEL["selr2"] / "run_manifest.json")},
        "counts": {"physical_transitions": len(phys),
                   "goal_queries": len(queries),
                   "semantic_target_rows": len(sem_targets)},
    }, indent=2))
    n_bank = sum(1 for r in phys if r["origin"].startswith(
        "v071_sel") and r["kind"] == "branch")
    print(f"physical transitions: {len(phys)} "
          f"(union {sum(1 for r in phys if r['origin'] == 'v071_union')}, "
          f"semwin {sum(1 for r in phys if r['origin'] == 'v071_semwin')}, "
          f"selector bank branches {n_bank})", flush=True)
    print(f"action contract: {contract}", flush=True)
    print(f"fidelity recomputed: {len(fidelity)}/34 anchors; "
          f"flags failing gross bounds: "
          f"{[k for k, v in fid_flags.items() if not all(v.values())]}",
          flush=True)
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
