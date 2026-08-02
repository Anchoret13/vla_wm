#!/usr/bin/env python
"""V7.1.1F — freeze the trainable union and loader (bounded; no
recollection). The only blocking item before V7.1.2a.

- one immutable union manifest + split_manifest binding both replay
  roots, all 40 shard hashes, source-history paths/hashes, GoalSpecs,
  code/Git lineage, pi0.5 checkpoint, action normalization, and
  authoritative train/dev membership;
- 724 unique (anchor, actions_env bytes) rows = physical sampling
  universe; 40 fidelity copies + 6 stored-dev duplicates audit-only;
- 87 chunk_norm=None rows resolved deterministically: normalized chunk
  reconstructed from actions_env via the exact runner-inverse affine
  ONLY when the env->norm->env roundtrip is bit-exact; otherwise the
  row stays physical-only (flow-ineligible). Scripted servo is always
  flow-ineligible;
- H_t joins bound (shard -> immutable source history path+hash+row
  index); before-state body ordering bound per anchor from the shard's
  own grasp-key order;
- errata recorded (r2 manifest CRN text; two r1 quaternion gross-bound
  anchors marked not-strict-clean; smoke checkpoint/video hashes;
  generated-seed lineage gaps);
- exit: deterministic loader dry-run reproduces 724 = 533 train + 191
  dev, emits identical ordered transition IDs twice, zero source
  overlap.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.v067_lineage import sha256_file  # noqa: E402

RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
R1 = RESULTS / "2026-08-02_v071_replay_r1"
R2 = RESULTS / "2026-08-02_v071_replay_r2"
DATA = Path("/home/stargazer/Desktop/vla_wm/datasets/libero_loho_public_v1"
            "/v06_effect_crossed")
SMOKE = RESULTS / "2026-08-02_v071_0b_smoke_r1"
OUT = RESULTS / "2026-08-02_v071_union_f1"
QUAT_GROSS_ANCHORS = ["loho_t3_tray_correction_s2120_d19",
                      "loho_t5_drawer_cabinet_correction_s2142_d2"]


def main() -> None:
    from lcwm.seq_prefix_cache import normalize_actions
    OUT.mkdir(exist_ok=True)

    ref = torch.load(Path("/home/stargazer/Desktop/vla_wm/datasets"
                          "/seq_prefix_cache_v1/task0_demo0.pt"),
                     weights_only=False)
    mean, std_eps = ref["action_mean"], ref["action_std_eps"]

    rows, shard_hashes, source_bind = [], {}, {}
    n_flow_recon, n_flow_inelig = 0, 0
    for root, run_id in ((R1, "r1"), (R2, "r2")):
        for p in sorted((root / "shards").glob("*.pt")):
            shard_hashes[f"{run_id}/{p.name}"] = sha256_file(p)
            s = torch.load(p, weights_only=False)
            anchor = s["anchor"]
            task = s["task"]
            split = s["split"]
            # H_t join: bind the immutable source history
            if anchor.startswith(tuple(
                    f"loho_t{i}" for i in range(1, 6))):
                if "_correction_" in anchor:
                    sid = anchor.rsplit("_d", 1)[0]
                    spath = (DATA / "corrections_sources_v069"
                             / f"{sid}.pt")
                else:
                    sid = anchor.rsplit("_d", 1)[0]
                    spath = (R2 / "acquire_sources" / f"{sid}.pt")
            if sid not in source_bind:
                source_bind[sid] = {"path": str(spath),
                                    "sha256": sha256_file(spath)}
            body_order = list(
                s["transitions"][0]["grasp_after"]["grasped"])
            for tr in s["transitions"]:
                key = str(tr["branch_key"])
                a_bytes = tr["actions_env"].tobytes()
                chunk = tr.get("chunk_norm")
                flow_ok, chunk_src = False, None
                if tr["provenance"] == "scripted_servo":
                    flow_ok = False
                    chunk_src = "servo_flow_ineligible"
                elif chunk is not None:
                    flow_ok = True
                    chunk_src = "stored"
                else:
                    # deterministic reconstruction with bit-exact
                    # roundtrip requirement
                    ae = torch.from_numpy(
                        tr["actions_env"]).float()
                    if ae.shape[0] > 0:
                        cn = normalize_actions(ae, mean, std_eps)
                        back = (cn * std_eps + mean)
                        if torch.equal(back, ae):
                            flow_ok = True
                            chunk_src = "reconstructed_exact"
                            n_flow_recon += 1
                        else:
                            chunk_src = "roundtrip_inexact"
                            n_flow_inelig += 1
                    else:
                        chunk_src = "empty_actions"
                        n_flow_inelig += 1
                rows.append({
                    "transition_id": tr["transition_id"],
                    "run": run_id, "shard": p.name,
                    "anchor": anchor, "task": task, "split": split,
                    "source_id": sid,
                    "decision": s["decision"],
                    "branch_key": key,
                    "family": tr["family"],
                    "provenance": tr["provenance"],
                    "steps": int(tr["steps"]),
                    "action_bytes_sha256": hashlib.sha256(
                        a_bytes).hexdigest()[:16],
                    "flow_eligible": bool(flow_ok),
                    "chunk_source": chunk_src,
                    "n_continuations": len(tr["continuations"]),
                    "body_order": body_order,
                })

    # ---- dedup: unique (anchor, action bytes) --------------------------
    seen = {}
    audit_only = []
    for r in rows:
        k = (r["anchor"], r["action_bytes_sha256"])
        if k in seen:
            r["audit_only"] = True
            r["duplicate_of"] = seen[k]
            audit_only.append(r["transition_id"])
        else:
            r["audit_only"] = False
            seen[k] = r["transition_id"]
    unique = [r for r in rows if not r["audit_only"]]
    tr_train = [r for r in unique if r["split"] == "train"]
    tr_dev = [r for r in unique if r["split"] == "dev"]
    src_train = {r["source_id"] for r in tr_train}
    src_dev = {r["source_id"] for r in tr_dev}
    assert not (src_train & src_dev), "source overlap!"
    print(f"raw {len(rows)} -> unique {len(unique)} "
          f"({len(tr_train)} train + {len(tr_dev)} dev); "
          f"audit-only {len(audit_only)}; flow-reconstructed "
          f"{n_flow_recon}, flow-ineligible {n_flow_inelig}",
          flush=True)
    assert len(unique) == 724 and len(tr_train) == 533 \
        and len(tr_dev) == 191, "expected 724=533+191"
    assert len(audit_only) == 46

    git_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    hub = (Path.home() / ".cache" / "huggingface" / "hub"
           / "models--lerobot--pi05_libero_finetuned" / "snapshots")
    pi_rev = sorted(hub.iterdir())[-1].name if hub.exists() else None

    union = {
        "schema": "v071_union_manifest_v1", "run_schema": "v071",
        "replay_roots": {"r1": str(R1), "r2": str(R2)},
        "shard_sha256": shard_hashes,
        "source_histories": source_bind,
        "goal_specs_sha256": {
            "r1": sha256_file(R1 / "goal_specs.jsonl"),
            "r2": sha256_file(R2 / "goal_specs.jsonl")},
        "git_sha": git_sha,
        "pi05_revision": pi_rev,
        "action_normalization": {
            "kind": "affine mean/std_eps (runner exact inverse)",
            "mean_sha256": hashlib.sha256(
                mean.numpy().tobytes()).hexdigest()[:16],
            "std_eps_sha256": hashlib.sha256(
                std_eps.numpy().tobytes()).hexdigest()[:16]},
        "counts": {"raw": len(rows), "unique": len(unique),
                   "train": len(tr_train), "dev": len(tr_dev),
                   "audit_only": len(audit_only),
                   "flow_reconstructed": n_flow_recon,
                   "flow_ineligible": n_flow_inelig},
        "errata": {
            "r2_manifest_crn_text": (
                "r2 run_manifest crn_contract text hard-codes "
                "'v071_replay_r1'; the streams actually used the r2 "
                "namespace f'v071_replay_r2|...' via RUN_ID — text "
                "error only, recorded here, artifact not rewritten"),
            "r1_quaternion_gross_anchors": QUAT_GROSS_ANCHORS,
            "r1_quaternion_note": (
                "24/26 r1 anchors pass the 7.8e-3 quat gross bound; "
                "these two exceed it and are physical training rows "
                "but NOT strict-clean fidelity evidence"),
            "generated_action_seed_lineage": (
                "r2 canonical/atomic/distinct chunks were generated "
                "from registered integer seeds recorded in the specs; "
                "per-chunk noise-tensor SHAs were not persisted — "
                "recorded as a gap, derivable from seeds"),
            "smoke_artifacts": {
                p.name: sha256_file(p) for p in sorted(
                    (SMOKE / "checkpoints").glob("*.pt"))},
            "acquisition_last_d_bug": (
                "diversity second pass inherited last_d from pass 1 "
                "(gap enforced against wrong neighbor); FIXED in code "
                "for future rounds; completed acquisition NOT rerun "
                "per the queue"),
        },
        "rows": rows,
    }
    tmp = OUT / "union_manifest.json.tmp"
    tmp.write_text(json.dumps(union, indent=2))
    tmp.replace(OUT / "union_manifest.json")

    split_manifest = {
        "schema": "v071_split_manifest_v1", "run_schema": "v071",
        "rule": "source-disjoint; inherited bank splits; acquire "
                "sources train",
        "train_sources": sorted(src_train),
        "dev_sources": sorted(src_dev),
        "train_transitions": sorted(
            r["transition_id"] for r in tr_train),
        "dev_transitions": sorted(
            r["transition_id"] for r in tr_dev),
        "audit_only_transitions": sorted(audit_only),
        "quat_gross_anchors": QUAT_GROSS_ANCHORS,
        "union_manifest_sha256": sha256_file(
            OUT / "union_manifest.json"),
    }
    (OUT / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2))

    # ---- deterministic loader dry-run (exit artifact) ------------------
    from lcwm.v071_loader import V071Loader
    ids1 = V071Loader(OUT).ordered_ids("train") \
        + V071Loader(OUT).ordered_ids("dev")
    ids2 = V071Loader(OUT).ordered_ids("train") \
        + V071Loader(OUT).ordered_ids("dev")
    assert ids1 == ids2, "loader ordering not deterministic"
    assert len(ids1) == 724
    ldr = V071Loader(OUT)
    exposure = ldr.exposure_report("train")
    (OUT / "loader_dryrun.json").write_text(json.dumps({
        "n_ids": len(ids1),
        "train": len(ldr.ordered_ids("train")),
        "dev": len(ldr.ordered_ids("dev")),
        "identical_across_runs": True,
        "source_overlap": 0,
        "exposure_levels": exposure}, indent=2))
    print(f"loader dry-run OK: {len(ids1)} ids "
          f"({len(ldr.ordered_ids('train'))}/"
          f"{len(ldr.ordered_ids('dev'))}), deterministic",
          flush=True)
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
