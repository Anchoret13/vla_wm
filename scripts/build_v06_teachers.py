#!/usr/bin/env python
"""V6.4 — frozen WM + matched-random + grounded-GT teacher manifests.

Registered rules (2026-07-30 queue):
- teacher states: every source-train accepted (grounded) snapshot exactly,
  plus evenly spaced slot-eligible decisions to ~14/source (~200 total);
- z unrolled from episode reset along recorded observations/actions under
  the canonical prompt (cached h sidecars) — never reset/cached-at-snapshot;
- candidate pools: grounded states reuse audited policy_candidates[0:4];
  new states use [executed chunk (reference), 3 fresh-seeded draws] under
  the collection seed contract (all exchangeable stock draws);
- scoring ONLY through z → T(z,u_i) → D_next; scalar s_i; p_ij=σ(s_i−s_j);
- δ = Q0.90(|p_ij − 1[A_ij=+1]|) on source-disjoint non-tied DEV pairs
  (δ=0.5 if none exists — emits no model teacher);
- emit non-stock teacher iff i*≠0 and p_{i*0} − δ > 0.5 (ties broken by
  saved candidate hash); no forced top-1;
- matched random: hash-seeded k_s ∈ {1,2,3}, i_random = (i*+k_s) mod 4 —
  score-blind, never equal to the WM choice, may pick the reference;
- grounded GT: reference = candidate 0; alternative positive only when it
  beats the reference in BOTH paired repeats (lexicographic outcome
  order); policy_gt (cands 1-3) and support_gt (u_support) reported
  separately; support never enters the WM/random pool;
- non-stock fraction is NOT a metric (P(random argmax non-stock)=3/4).
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

DATA = Path("/home/stargazer/Desktop/vla_wm/datasets/libero_loho_public_v1"
            "/v06_effect_crossed")
WM_CKPT = (REPO_ROOT / "results" / "libero_loho_public_v1" / "v06_wm"
           / "checkpoint_final.pt")
OUT = REPO_ROOT / "results" / "libero_loho_public_v1" / "v06_teachers"
TARGET_PER_SOURCE = 14
NOISE_BASE = 40_000_000


def chunk_hash(chunk: torch.Tensor) -> str:
    return hashlib.sha256(
        chunk.float().numpy().tobytes()).hexdigest()[:16]


@torch.no_grad()
def unroll_z(model, feats, rows, device):
    zs = []
    z = None
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
    model.load_state_dict(bundle["model"])
    model.eval()
    wm_hash = hashlib.sha256(WM_CKPT.read_bytes()).hexdigest()
    selection = json.loads(
        (REPO_ROOT / "results" / "libero_loho_public_v1"
         / "v06_group_selection.json").read_text())
    accepted = {}
    for g in selection["accepted"]:
        accepted.setdefault(g["source_id"], []).append(g["decision"])

    # ---- δ calibration on source-disjoint non-tied dev pairs ------------
    def group_pairs(split_want: str):
        for cp in sorted((DATA / "continuations").glob("*.pt")):
            cont = torch.load(cp, weights_only=False)
            if cont["split"] != split_want:
                continue
            sid = cont["source_id"]
            source = torch.load(DATA / "sources" / f"{sid}.pt",
                                weights_only=False)
            feats = torch.load(
                DATA / "features"
                / f"{sid}__canonical.pt", weights_only=False)
            d = cont["decision"]
            rows = source["rows"]
            zs = unroll_z(model, feats, rows[:d + 1], device)
            z = zs[d]
            audit = next(a for a in source["audits"]
                         if a["decision"] == d)
            cands = {b["candidate"]: b["chunk_norm"]
                     for b in audit["branches"]
                     if b["kind"] == "candidate"}
            canon_gid = cont["goals"][0]
            by_branch: dict[int, list] = {}
            for r in cont["records"]:
                if (r["goal_spec_id"] == canon_gid
                        and r["branch_index"] != 4):
                    by_branch.setdefault(r["branch_index"], []).append(r)
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
                a_ij = paired_preference(yi, yj, {})
                if a_ij == 0:
                    continue
                p_ij = torch.sigmoid(torch.tensor(
                    scores[i] - scores[j]))
                yield split_want, float(p_ij), a_ij

    dev_pairs = list(group_pairs("dev"))
    if dev_pairs:
        errs = torch.tensor([abs(p - (1.0 if a == 1 else 0.0))
                             for _s, p, a in dev_pairs])
        delta = float(errs.quantile(0.90))
    else:
        delta = 0.5
    train_pairs = list(group_pairs("train"))
    calibration = {
        "delta_q90_dev": delta,
        "n_dev_nontied_pairs": len(dev_pairs),
        "dev_pair_accuracy": (float(torch.tensor(
            [1.0 if (p > 0.5) == (a == 1) else 0.0
             for _s, p, a in dev_pairs]).mean()) if dev_pairs else None),
        "n_train_nontied_pairs": len(train_pairs),
        "train_pair_accuracy": (float(torch.tensor(
            [1.0 if (p > 0.5) == (a == 1) else 0.0
             for _s, p, a in train_pairs]).mean())
            if train_pairs else None),
    }

    # ---- teacher manifest over ~200 train decisions ---------------------
    from lcwm.chassis import Pi05Runner
    from lcwm.sampler import prefix_forward, sample_chunks
    runner = Pi05Runner(suite_name="libero_10")

    rows_out, gt_rows = [], []
    for sp in sorted((DATA / "sources").glob("*.pt")):
        source = torch.load(sp, weights_only=False)
        if source["split"] != "train":
            continue
        sid = source["source_id"]
        task_index = source["task_index"]
        seed = source["seed"]
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

        # grounded GT for this source's accepted groups
        for d in grounded:
            cp = DATA / "continuations" / f"{sid}_d{d}.pt"
            if not cp.exists():
                continue
            cont = torch.load(cp, weights_only=False)
            canon_gid = cont["goals"][0]
            audit = audit_by_d[d]
            outcome_of: dict[int, list] = {}
            for r in cont["records"]:
                if r["goal_spec_id"] == canon_gid:
                    outcome_of.setdefault(
                        r["branch_index"], []).append(r)
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
                if paired_preference(alt, ref, {}) == 1:
                    positives.append((bi, alt))
            if not positives:
                continue
            best_bi, _ = positives[0]
            for bi, alt in positives[1:]:
                cur = [r["outcome"] for r in sorted(
                    outcome_of[best_bi], key=lambda r: r["repeat"])]
                if paired_preference(alt, cur, {}) == 1:
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
                "branch_index": best_bi,
                "chunk_sha256": chunk_hash(chunk),
                "chunk_norm": chunk, "weight": 1.0,
            })
        print(f"[teachers] {sid}: "
              f"{len(grounded)} grounded + {len(fill)} filled", flush=True)

    n_emit = sum(1 for r in rows_out if r["emit_model_teacher"])
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": "v06_teachers_v1", "rows": rows_out,
                "delta": delta, "wm_checkpoint_sha256": wm_hash},
               OUT / "teacher_manifest.pt")
    torch.save({"schema": "v06_gt_manifest_v1", "rows": gt_rows,
                "wm_checkpoint_sha256": wm_hash},
               OUT / "gt_manifest.pt")
    summary = {
        **calibration,
        "n_teacher_states": len(rows_out),
        "n_model_teachers_emitted": n_emit,
        "gt_counts": {
            "policy_gt": sum(1 for g in gt_rows
                             if g["gt_kind"] == "policy_gt"),
            "support_gt": sum(1 for g in gt_rows
                              if g["gt_kind"] == "support_gt")},
        "note": ("non-stock fraction is not evidence of learned "
                 "selection (random argmax is non-stock w.p. 3/4)"),
    }
    (OUT / "calibration.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
