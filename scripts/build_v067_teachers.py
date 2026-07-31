#!/usr/bin/env python
"""V6.7.4 — corrected WM + matched-random + grounded-GT teacher manifests
on the v067 sibling-CRN outcome bank.

Unchanged from the registered iteration-1 rules: scalar Bradley–Terry
score; δ = Q0.90(|p_ij − 1[A_ij=+1]|) on source-disjoint non-tied DEV
pairs (δ=0.5 if none — emits nothing); emit iff i*≠0 and p_{i*0}−δ>0.5;
matched random i_random=(i*+k_s) mod 4; grounded GT independent of the
learned score, policy_gt/support_gt separated; support never enters the
WM/random pool; the margin is never relaxed because few teachers emit.

Corrected here:
- outcomes come ONLY from v067 sibling-paired continuations (run_schema
  guard refuses iteration-1 manifests);
- paired preferences use the FROZEN outcome tolerances from the v067
  support report (iteration 1 used {});
- `replay_unstable` groups are excluded from calibration, ranking, and
  GT support (registered exclusion);
- calibration is reported at independent source/snapshot-group level:
  n_sources, n_groups, n_pairs separately — many pairs from one group
  are not many calibration samples;
- task-balance floor reported: every public task needs ≥1 clean
  reference-improving train group before a task-balanced GT policy job.

Output: results/libero_loho_public_v1/v067_teachers/
"""

from __future__ import annotations

import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.task_automaton import paired_preference  # noqa: E402
from lcwm.v06_model import V06State  # noqa: E402
from lcwm.v067_lineage import RUN_SCHEMA, load_v067  # noqa: E402

DATA = Path("/home/stargazer/Desktop/vla_wm/datasets/libero_loho_public_v1"
            "/v06_effect_crossed")
RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
WM_CKPT = RESULTS / "v067_wm" / "checkpoint_final.pt"
OUT = RESULTS / "v067_teachers"
TARGET_PER_SOURCE = 14
NOISE_BASE = 40_000_000
TASKS = ["loho_t1_drawer", "loho_t2_basket3", "loho_t3_tray",
         "loho_t4_tray", "loho_t5_drawer_cabinet"]


def chunk_hash(chunk: torch.Tensor) -> str:
    return hashlib.sha256(
        chunk.float().numpy().tobytes()).hexdigest()[:16]


@torch.no_grad()
def unroll_z(model, feats, rows, device):
    zs, z = [], None
    for i in range(len(rows)):
        h = feats["h"][i][None].float().to(device)
        m = feats["mask"][i][None].to(device)
        if z is None:
            z = model.initial_state(h, m)
        else:
            a = rows[i - 1]["chunk_norm"][None, :10].float().to(device)
            am = (torch.arange(10, device=device)[None]
                  < rows[i - 1]["executed_len"])
            z = model.step(z, a, h, m, action_mask=am)
        zs.append(z)
    return zs


@torch.no_grad()
def main() -> None:
    device = torch.device("cuda")
    model = V06State().to(device)
    bundle = torch.load(WM_CKPT, weights_only=False)
    assert bundle.get("run_schema") == RUN_SCHEMA, \
        "WM checkpoint is not a v067 artifact"
    model.load_state_dict(bundle["model"])
    model.eval()
    wm_hash = hashlib.sha256(WM_CKPT.read_bytes()).hexdigest()

    support_report = json.loads(
        (RESULTS / "v067_support_report.json").read_text())
    assert support_report["run_schema"] == RUN_SCHEMA
    tolerances = support_report["frozen_outcome_tolerances"]
    unstable = {r["group"] for r in support_report["groups"]
                if r["replay_unstable"]}
    selection = json.loads(
        (RESULTS / "v06_group_selection.json").read_text())
    accepted: dict[str, list[int]] = {}
    for g in selection["accepted"]:
        accepted.setdefault(g["source_id"], []).append(g["decision"])

    # ---- δ calibration on source-disjoint non-tied dev pairs -----------
    def group_pairs(split_want: str):
        for cp in sorted((DATA / "continuations_v067").glob("*.pt")):
            cont = load_v067(cp, "continuations")
            if cont["split"] != split_want:
                continue
            gid_key = f"{cont['source_id']}_d{cont['decision']}"
            if gid_key in unstable:
                continue
            sid = cont["source_id"]
            source = torch.load(DATA / "sources" / f"{sid}.pt",
                                weights_only=False)
            feats = torch.load(
                DATA / "features" / f"{sid}__canonical.pt",
                weights_only=False)
            d = cont["decision"]
            zs = unroll_z(model, feats, source["rows"][:d + 1], device)
            z = zs[d]
            audit = next(a for a in source["audits"]
                         if a["decision"] == d)
            cands = {b["candidate"]: b["chunk_norm"]
                     for b in audit["branches"]
                     if b["kind"] == "candidate"}
            canon_gid = cont["canonical_goal_spec_id"]
            by_branch: dict[int, list] = {}
            for r in cont["records"]:
                if (r["goal_spec_id"] == canon_gid
                        and r["provenance"] == "policy"):
                    by_branch.setdefault(r["branch_key"], []).append(r)
            scores = {}
            for bi, chunk in cands.items():
                s = model.d_next(model.predict(
                    z, chunk[None, :10].float().to(device)))["s"]
                scores[bi] = float(s)
            for i, j in combinations(sorted(by_branch), 2):
                yi = [r["outcome"] for r in sorted(
                    by_branch[i], key=lambda r: r["repeat"])]
                yj = [r["outcome"] for r in sorted(
                    by_branch[j], key=lambda r: r["repeat"])]
                a_ij = paired_preference(yi, yj, tolerances)
                if a_ij == 0:
                    continue
                p_ij = torch.sigmoid(torch.tensor(
                    scores[i] - scores[j]))
                yield sid, gid_key, float(p_ij), a_ij

    dev_pairs = list(group_pairs("dev"))
    if dev_pairs:
        errs = torch.tensor([abs(p - (1.0 if a == 1 else 0.0))
                             for _s, _g, p, a in dev_pairs])
        delta = float(errs.quantile(0.90))
    else:
        delta = 0.5
    train_pairs = list(group_pairs("train"))

    def acc(pairs):
        return (float(torch.tensor(
            [1.0 if (p > 0.5) == (a == 1) else 0.0
             for _s, _g, p, a in pairs]).mean()) if pairs else None)

    calibration = {
        "delta_q90_dev": delta,
        "n_dev_nontied_pairs": len(dev_pairs),
        "n_dev_groups": len({g for _s, g, _p, _a in dev_pairs}),
        "n_dev_sources": len({s for s, _g, _p, _a in dev_pairs}),
        "dev_pair_accuracy": acc(dev_pairs),
        "n_train_nontied_pairs": len(train_pairs),
        "n_train_groups": len({g for _s, g, _p, _a in train_pairs}),
        "n_train_sources": len({s for s, _g, _p, _a in train_pairs}),
        "train_pair_accuracy": acc(train_pairs),
        "frozen_outcome_tolerances": tolerances,
        "excluded_replay_unstable_groups": sorted(unstable),
    }

    # ---- teacher manifest over train decisions -------------------------
    from lcwm.chassis import Pi05Runner
    from lcwm.sampler import prefix_forward, sample_chunks
    runner = Pi05Runner(suite_name="libero_10")

    rows_out, gt_rows = [], []
    for sp in sorted((DATA / "sources").glob("*.pt")):
        source = torch.load(sp, weights_only=False)
        if source["split"] != "train":
            continue
        sid = source["source_id"]
        task_index, seed = source["task_index"], source["seed"]
        feats = torch.load(DATA / "features" / f"{sp.stem}__canonical.pt",
                           weights_only=False)
        rows = source["rows"]
        zs = unroll_z(model, feats, rows, device)
        grounded = sorted(accepted.get(sid, []))
        pool_ds = [d for d in
                   (source["slot1_eligible"] + source["slot2_eligible"])
                   if d not in grounded and d < len(rows)]
        fill = pool_ds[::max(1, len(pool_ds)
                             // max(1, TARGET_PER_SOURCE
                                    - len(grounded)))][
            :TARGET_PER_SOURCE - len(grounded)]
        audit_by_d = {a["decision"]: a for a in source["audits"]}

        for d in grounded + sorted(fill):
            z = zs[d]
            if d in audit_by_d:
                cands = {b["candidate"]: b["chunk_norm"]
                         for b in audit_by_d[d]["branches"]
                         if b["kind"] == "candidate"}
                cand_src = "grounded_audit"
            else:
                cands = {0: rows[d]["chunk_norm"]}
                batch = runner._obs_to_policy_batch(
                    rows[d]["obs"], source["language_canonical"])
                prefix = prefix_forward(runner.policy, batch)
                for c in range(1, 4):
                    s_seed = (NOISE_BASE + task_index * 2_000_000
                              + seed * 1_000 + d + 100_000 * c)
                    cands[c] = sample_chunks(
                        runner.policy, batch, n=1, seed=s_seed,
                        prefix=prefix)[0].float().cpu()
                cand_src = "fresh_pool"
            s_vals, hashes = {}, {}
            for c in sorted(cands):
                s_vals[c] = float(model.d_next(model.predict(
                    z, cands[c][None, :10].float().to(device)))["s"])
                hashes[c] = chunk_hash(cands[c])
            i_star = max(sorted(s_vals),
                         key=lambda c: (s_vals[c], hashes[c]))
            p_i0 = float(torch.sigmoid(torch.tensor(
                s_vals[i_star] - s_vals[0])))
            emit = i_star != 0 and (p_i0 - delta) > 0.5
            k_s = 1 + int(hashlib.sha256(
                f"{sid}|{d}".encode()).hexdigest(), 16) % 3
            i_random = (i_star + k_s) % 4
            rows_out.append({
                "source_id": sid, "task": source["task"],
                "decision": d, "candidate_source": cand_src,
                "scores": s_vals, "candidate_sha256": hashes,
                "i_star": i_star, "p_i_star_0": p_i0,
                "margin": p_i0 - delta,
                "emit_model_teacher": bool(emit),
                "i_random": i_random, "k_s": k_s,
                "candidates": {c: cands[c] for c in sorted(cands)},
            })

        # grounded GT from clean v067 groups only
        for d in grounded:
            gid_key = f"{sid}_d{d}"
            cp = DATA / "continuations_v067" / f"{sid}_d{d}.pt"
            if not cp.exists() or gid_key in unstable:
                continue
            cont = load_v067(cp, "continuations")
            canon_gid = cont["canonical_goal_spec_id"]
            audit = audit_by_d[d]
            outcome_of: dict[int, list] = {}
            for r in cont["records"]:
                if (r["goal_spec_id"] == canon_gid
                        and r["provenance"] in ("policy", "support")):
                    outcome_of.setdefault(r["branch_key"], []).append(r)
            if 0 not in outcome_of:
                continue
            ref = [r["outcome"] for r in sorted(
                outcome_of[0], key=lambda r: r["repeat"])]
            positives = []
            for bi in sorted(outcome_of):
                if bi == 0:
                    continue
                alt = [r["outcome"] for r in sorted(
                    outcome_of[bi], key=lambda r: r["repeat"])]
                if paired_preference(alt, ref, tolerances) == 1:
                    positives.append((bi, alt))
            if not positives:
                continue
            best_bi, _ = positives[0]
            for bi, alt in positives[1:]:
                cur = [r["outcome"] for r in sorted(
                    outcome_of[best_bi], key=lambda r: r["repeat"])]
                if paired_preference(alt, cur, tolerances) == 1:
                    best_bi = bi
            branches = {(b["kind"], b.get("candidate")): b
                        for b in audit["branches"]}
            if best_bi == 4:
                chunk = branches[("support", None)]["chunk_norm"]
                gt_kind = "support_gt"
            else:
                chunk = branches[("candidate", best_bi)]["chunk_norm"]
                gt_kind = "policy_gt"
            gt_rows.append({
                "source_id": sid, "task": source["task"],
                "decision": d, "gt_kind": gt_kind,
                "branch_key": best_bi,
                "chunk_sha256": chunk_hash(chunk),
                "chunk_norm": chunk, "weight": 1.0,
            })
        print(f"[teachers] {sid}: "
              f"{len(grounded)} grounded + {len(fill)} filled",
              flush=True)

    n_emit = sum(1 for r in rows_out if r["emit_model_teacher"])
    gt_tasks = {t: sum(1 for g_ in gt_rows if g_["task"] == t
                       and g_["gt_kind"] == "policy_gt")
                for t in TASKS}
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": "v067_teachers_v1", "run_schema": RUN_SCHEMA,
                "rows": rows_out, "delta": delta,
                "wm_checkpoint_sha256": wm_hash},
               OUT / "teacher_manifest.pt")
    torch.save({"schema": "v067_gt_manifest_v1",
                "run_schema": RUN_SCHEMA, "rows": gt_rows,
                "wm_checkpoint_sha256": wm_hash},
               OUT / "gt_manifest.pt")
    summary = {
        **calibration,
        "n_teacher_states": len(rows_out),
        "n_model_teachers_emitted": n_emit,
        "gt_counts": {
            "policy_gt": sum(1 for g_ in gt_rows
                             if g_["gt_kind"] == "policy_gt"),
            "support_gt": sum(1 for g_ in gt_rows
                              if g_["gt_kind"] == "support_gt")},
        "gt_policy_by_task": gt_tasks,
        "task_balance_floor_met": all(v > 0 for v in gt_tasks.values()),
        "generated_channel_nonempty": n_emit > 0,
        "note": ("non-stock fraction is not evidence of learned "
                 "selection (random argmax is non-stock w.p. 3/4); "
                 "zero emitted teachers = generated channel unsupported "
                 "-> V6.8, never a margin relaxation"),
    }
    (OUT / "calibration.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
