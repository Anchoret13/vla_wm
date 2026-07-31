#!/usr/bin/env python
"""V6.0 — freeze H7 as an immutable negative baseline.

Hashes every governing H7 artifact and stores the implementation audit
beside them. Produces results/libero_loho_public_v1/h7_freeze/
{freeze_manifest.json, implementation_audit.json}. Read-only elsewhere.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = Path("/home/stargazer/Desktop/vla_wm/datasets/libero_loho_public_v1")
RES = REPO_ROOT / "results" / "libero_loho_public_v1"
OUT = RES / "h7_freeze"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    groups: dict[str, dict[str, str]] = {}

    groups["collection"] = {
        str(p.relative_to(DATA)): sha(p)
        for p in sorted(DATA.glob("loho_*.pt"))
    } | {"collection_manifest.json": sha(DATA / "collection_manifest.json")}
    groups["wm_checkpoints"] = {
        "v05_gated_wm_r1/checkpoint_final.pt":
            sha(RES / "v05_gated_wm_r1" / "checkpoint_final.pt"),
        "v05_gated_wm/checkpoint_final.pt":
            sha(RES / "v05_gated_wm" / "checkpoint_final.pt"),
    }
    groups["teachers"] = {
        "v05_teachers/teachers.pt": sha(RES / "v05_teachers" / "teachers.pt"),
        "v05_teachers/teacher_diagnostics.json":
            sha(RES / "v05_teachers" / "teacher_diagnostics.json"),
    }
    groups["adapters"] = {
        f"adapters/{arm}/adapter.pt": sha(RES / "adapters" / arm / "adapter.pt")
        for arm in ("v05_reset_wm", "v05_reset_random",
                    "v05_recurrent_wm", "v05_recurrent_random")
    }
    records = [
        line for line in
        (RES / "eval_matrix" / "development" / "records.jsonl")
        .read_text().splitlines()
        if line.strip() and json.loads(line)["run_id"] in
        ("h75a_tranche1", "h75b_tranche2")
    ]
    assert len(records) == 150, f"expected 150 dev records, got {len(records)}"
    groups["evaluation"] = {
        "records_count": "150",
        "records_sha256": hashlib.sha256(
            "\n".join(records).encode()).hexdigest(),
        "run_manifest_h75a_tranche1.json":
            sha(RES / "eval_matrix" / "run_manifest_h75a_tranche1.json"),
        "run_manifest_h75b_tranche2.json":
            sha(RES / "eval_matrix" / "run_manifest_h75b_tranche2.json"),
    }

    audit = {
        "status": ("valid negative result for the implemented v0.5 "
                   "teacher/readout/adapter contract; NOT a clean negative "
                   "test of language-conditioned predictive dynamics"),
        "defects": [
            "candidate-path readout bypass: v0 grounded heads decoded from "
            "pool(c,g) only; r1 added pool(w_prior) but task features "
            "remained a current-state bypass alongside the transition",
            "next-observation training mismatch: branch supervision used "
            "one null-action update from init, not the deployed unroll",
            "omitted within-sibling ranking loss (registered, not "
            "implemented in the fixed H7.3 run)",
            "unpaired continuation noise across siblings; 100-action "
            "endpoint washed out first-ten effects (3/20 train, 4/20 dev "
            "groups with sibling continuation-Q variation)",
            "policy targets generated from reset state for all arms",
            "recurrent policy gate effectively shut "
            "(|tanh alpha| <= 0.0011; rec-vs-reset bias delta 0.05%)",
        ],
        "task_provenance": ("five public tasks are a specification-based "
                            "LIBERO-LoHo reimplementation; no author-"
                            "protocol parity claim"),
        "h7_data_reuse_rule": ("ordinary/null-effect calibration or "
                               "rehearsal only; sibling-ranking use requires "
                               "re-evaluation under the v0.6 common-noise, "
                               "current-valid label contract"),
        "no_init_reuse": "v0.6 must not initialize from H7 model/adapters",
        "headline_result": ("150 dev episodes, seeds 1500-1540: stock SR "
                            "0.040 Q 0.516; no v0.5 arm above stock"),
    }

    manifest = {"schema": "h7_freeze_v1", "groups": groups}
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    (OUT / "freeze_manifest.json").write_text(json.dumps(manifest, indent=2))
    (OUT / "implementation_audit.json").write_text(json.dumps(audit, indent=2))
    n = sum(len(g) for g in groups.values())
    print(f"frozen {n} entries -> {OUT}")


if __name__ == "__main__":
    main()
