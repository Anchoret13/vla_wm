#!/usr/bin/env python
"""V7.0.0 — exact lineage + correction-bank index + strict replay
re-derivation + selected-checkpoint audit.

Asserts the queue's expected audit result and REFUSES downstream teacher
construction on any disagreement:
- 26 unique anchors, task counts 6/4/7/3/6 (s2120_d5 marked as the
  preceding-candidate retry of the gate-rejected s2120_d19);
- strict-clean train anchors 14 / 8 sources; dev 8 / 5 sources;
- train winning chunks 10 across 6 positive anchors;
- train family headroom canonical 0 / expanded VLA 2 / servo 6;
  dev expanded 3 / servo 3.

Raw legacy fields are preserved beside the re-derived strict fields
with gate_version; old artifacts are never rewritten.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.v067_lineage import sha256_file  # noqa: E402
from lcwm.v070_replay import (GATE_VERSION,  # noqa: E402
                              strict_replay_clean)

DATA = Path("/home/stargazer/Desktop/vla_wm/datasets/libero_loho_public_v1"
            "/v06_effect_crossed")
RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
OUT = RESULTS / "v070_policy" / "audit"
TASKS = ["loho_t1_drawer", "loho_t2_basket3", "loho_t3_tray",
         "loho_t4_tray", "loho_t5_drawer_cabinet"]
CANONICAL_KEYS = {str(i) for i in range(1, 16)}
EXPANDED_KEYS = {"sub0", "sub1", "sub2", "sub3"}
RETRY_MARKS = {("loho_t3_tray_correction_s2120", 5): {
    "retry_of": "loho_t3_tray_correction_s2120_d19",
    "reason": ("preceding-candidate retry generated after d19 was "
               "rejected by the OLD endpoint gate; both files retained "
               "as primary data, neither supersedes the other under "
               "the strict gate")}}
EXPECTED = {
    "n_anchors": 26, "task_counts": [6, 4, 7, 3, 6],
    "train_clean_anchors": 14, "train_clean_sources": 8,
    "dev_clean_anchors": 8, "dev_clean_sources": 5,
    "train_winning_chunks": 10, "train_positive_anchors": 6,
    "train_family_headroom": {"canonical": 0, "expanded": 2,
                              "servo": 6},
    "dev_family_headroom_expanded": 3, "dev_family_headroom_servo": 3,
}


def outs(grp, key):
    return [r["outcome"] for r in sorted(
        (r for r in grp["records"] if r["branch_key"] == key),
        key=lambda r: r["repeat"])]


def main() -> None:
    from lcwm.task_automaton import paired_preference

    OUT.mkdir(parents=True, exist_ok=True)

    index, report_rows = {}, []
    for p in sorted((DATA / "corrections_sources_v069").glob("*.pt")):
        s = torch.load(p, weights_only=False)
        index[p.name] = {
            "sha256": sha256_file(p), "kind": "source",
            "schema": s["schema"], "run_schema": s["run_schema"],
            "task": s["task"], "split": s["split"], "seed": s["seed"],
            "n_decisions": s["n_decisions"],
        }
    for p in sorted((DATA / "corrections_v069").glob("*.pt")):
        g = torch.load(p, weights_only=False)
        tol = g["frozen_outcome_tolerances"]
        rep = next(b for b in g["branch_summaries"]
                   if b["kind"] == "replay")
        deltas = rep["replay_endpoint"]["deltas"]
        clean, info = strict_replay_clean(
            outs(g, "replay"), outs(g, 0), tol, deltas)
        # both-repeat candidate preferences vs u_0 (re-derived, never
        # from legacy judged fields)
        prefs = {}
        for b in g["branch_summaries"]:
            if b["kind"] in ("replay",) or b["key"] == 0:
                continue
            prefs[str(b["key"])] = paired_preference(
                outs(g, b["key"]), outs(g, 0), tol)
        anchor_key = (g["source_id"], g["decision"])
        entry = {
            "sha256": sha256_file(p), "kind": "correction_group",
            "schema": g["schema"], "run_schema": g["run_schema"],
            "task": g["task"], "split": g["split"],
            "source_id": g["source_id"], "decision": g["decision"],
            "anchor_form": g["anchor"]["form"],
            "proposal_inventory": sorted(
                str(b["key"]) for b in g["branch_summaries"]),
            "legacy_fields": {
                "judged_vs_u0": g["judged_vs_u0"],
                "replay_unstable": g.get("replay_unstable"),
                "gross_restore_failure":
                    g.get("gross_restore_failure"),
                "grounded_corrections": g["grounded_corrections"]},
            "strict": {**info, "clean": clean,
                       "endpoint_deltas": deltas,
                       "both_repeat_pref_vs_u0": prefs},
            "status": RETRY_MARKS.get(anchor_key,
                                      {"retry_of": None}),
        }
        index[p.name] = entry
        report_rows.append(entry)

    # ---- strict replay report + expected-count asserts -----------------
    anchors = {(r["source_id"], r["decision"]): r for r in report_rows}
    assert len(anchors) == EXPECTED["n_anchors"], len(anchors)
    tc = Counter(r["task"] for r in report_rows)
    assert [tc[t] for t in TASKS] == EXPECTED["task_counts"], tc

    def family(key):
        if key in CANONICAL_KEYS:
            return "canonical"
        if key in EXPANDED_KEYS:
            return "expanded"
        if key == "support":
            return "servo"
        return "other"

    stats = {"train": defaultdict(set), "dev": defaultdict(set)}
    n_win_chunks_train = 0
    pos_anchors_train = set()
    for r in report_rows:
        split = r["split"]
        akey = f"{r['source_id']}_d{r['decision']}"
        if not r["strict"]["clean"]:
            continue
        stats[split]["clean_anchors"].add(akey)
        stats[split]["clean_sources"].add(r["source_id"])
        for key, v in r["strict"]["both_repeat_pref_vs_u0"].items():
            if v != 1:
                continue
            fam = family(key)
            stats[split][f"headroom_{fam}"].add(akey)
            if split == "train" and fam in ("canonical", "expanded",
                                            "servo"):
                n_win_chunks_train += 1
                pos_anchors_train.add(akey)

    got = {
        "train_clean_anchors": len(stats["train"]["clean_anchors"]),
        "train_clean_sources": len(stats["train"]["clean_sources"]),
        "dev_clean_anchors": len(stats["dev"]["clean_anchors"]),
        "dev_clean_sources": len(stats["dev"]["clean_sources"]),
        "train_winning_chunks": n_win_chunks_train,
        "train_positive_anchors": len(pos_anchors_train),
        "train_family_headroom": {
            "canonical": len(stats["train"]["headroom_canonical"]),
            "expanded": len(stats["train"]["headroom_expanded"]),
            "servo": len(stats["train"]["headroom_servo"])},
        "dev_family_headroom_expanded":
            len(stats["dev"]["headroom_expanded"]),
        "dev_family_headroom_servo":
            len(stats["dev"]["headroom_servo"]),
    }
    mismatches = {k: (v, EXPECTED[k]) for k, v in got.items()
                  if EXPECTED[k] != v}
    report = {
        "gate_version": GATE_VERSION,
        "expected": EXPECTED, "derived": got,
        "mismatches": mismatches,
        "teacher_construction_allowed": not mismatches,
        "clean_anchor_lists": {
            s: sorted(stats[s]["clean_anchors"]) for s in stats},
        "train_positive_anchor_list": sorted(pos_anchors_train),
        "headroom_anchor_lists": {
            s: {f: sorted(stats[s][f"headroom_{f}"])
                for f in ("canonical", "expanded", "servo")}
            for s in stats},
    }
    (OUT / "strict_replay_report.json").write_text(
        json.dumps(report, indent=2))
    (OUT / "correction_bank_index.json").write_text(
        json.dumps(index, indent=2, default=str))
    print(json.dumps({"derived": got, "mismatches": mismatches},
                     indent=1), flush=True)
    assert not mismatches, (
        f"strict audit disagrees with the registered expectation: "
        f"{mismatches} — teacher construction refused")

    # ---- selected-checkpoint audit -------------------------------------
    wz_path = RESULTS / "v069_policy" / "grounded_wz" / "wz_selected.pt"
    man = json.loads((RESULTS / "v069_eval"
                      / "run_manifest_v69_dev.json").read_text())
    wz_sha = sha256_file(wz_path)
    assert wz_sha == man["wz_sha256"], \
        "evaluated W_z hash does not reproduce"
    wz = torch.load(wz_path, weights_only=False)
    ckpt_audit = {
        "wz_selected_sha256": wz_sha,
        "reproduces_run_manifest": True,
        "selected_step": wz["step"],
        "historical_defects_recorded_not_rewritten": {
            "bias_stats_computed_at_step": 401,
            "behavior_used_selected_step": wz["step"],
            "dev_metrics_run_schema_field": "v067 (label bug; artifact "
                                            "is the v069 checkpoint)",
            "v069_eval_q_resolution": "per-10-action chunk (superseded "
                                      "by corrected_v069_behavior)",
            "v069_policy_state_mode": "correction loss averaged "
                                      "reset/recurrent; rehearsal/"
                                      "trust/selection recurrent-only",
        },
    }
    (OUT / "selected_checkpoint_audit.json").write_text(
        json.dumps(ckpt_audit, indent=2))

    # ---- lineage --------------------------------------------------------
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    pi_dir = hub / "models--lerobot--pi05_libero_finetuned"
    revs = sorted((pi_dir / "snapshots").iterdir()) \
        if (pi_dir / "snapshots").exists() else []
    pi05 = {"repo": "lerobot/pi05_libero_finetuned",
            "revision": revs[-1].name if revs else None,
            "model_sha256": (sha256_file(
                revs[-1] / "model.safetensors")
                if revs and (revs[-1]
                             / "model.safetensors").exists()
                else None)}
    required = {
        "predictive_checkpoint": RESULTS / "v069_predictive"
        / "checkpoint_selected.pt",
        "wz_selected": wz_path,
        "behavior_records": RESULTS / "v069_eval" / "development"
        / "records.jsonl",
        "behavior_manifest": RESULTS / "v069_eval"
        / "run_manifest_v69_dev.json",
        "rehearsal_manifest": Path(
            "/home/stargazer/Desktop/vla_wm/datasets"
            "/libero_loho_public_v1/demo_rehearsal_v067"
            "/manifest.json"),
        "evaluator": REPO_ROOT / "scripts" / "eval_loho_v069.py",
        "strict_rule": REPO_ROOT / "lcwm" / "v070_replay.py",
    }
    lineage = {"git_base_commit": "6d6573e",
               "gate_version": GATE_VERSION,
               "pi05": pi05, "artifacts": {}}
    for name, path in required.items():
        assert path.exists(), f"required lineage artifact missing: " \
            f"{path} — audit fails rather than reconstructing"
        lineage["artifacts"][name] = {"path": str(path),
                                      "sha256": sha256_file(path)}
    lineage["correction_bank_index_sha256"] = hashlib.sha256(
        (OUT / "correction_bank_index.json").read_bytes()).hexdigest()
    (OUT / "lineage.json").write_text(json.dumps(lineage, indent=2))
    print(f"lineage bound ({len(lineage['artifacts'])} artifacts, "
          f"pi05 rev {pi05['revision']})", flush=True)
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
