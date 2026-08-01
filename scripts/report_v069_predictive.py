#!/usr/bin/env python
"""V6.7.3 — required held-out report for the trained v067 world model.
Runs BEFORE teacher construction; writes v067_wm/dev_metrics.json.

By task and by independent source group (dev split only):
- absolute + centered branch physical error vs copy/zero baselines;
- next-valid and flip precision/recall/AUPRC (immediate crossed labels);
- current grounding (valid-bit accuracy, ordered-prefix error);
- one/two/three-block latent closure;
- paraphrase state + transitioned-prediction agreement;
- crossed-goal physical agreement and semantic disagreement;
- pairwise ranking accuracy/BCE/regret + confidence calibration
  (frozen outcome tolerances; replay_unstable groups excluded);
- recurrent-vs-reset history contrast (or `unsupported`);
- latent variation across task/state (policy bias is zero-W_z here).

Diagnostic only: does NOT authorize any sweep before the policy readout.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.task_automaton import paired_preference  # noqa: E402
from lcwm.v06_model import V06State  # noqa: E402
from lcwm.v067_lineage import RUN_SCHEMA, load_v067  # noqa: E402

DATA = Path("/home/stargazer/Desktop/vla_wm/datasets/libero_loho_public_v1"
            "/v06_effect_crossed")
RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
WM_CKPT = RESULTS / "v069_predictive" / "checkpoint_selected.pt"


def auprc(scores, labels) -> float | None:
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    n_pos = sum(labels)
    if n_pos == 0 or n_pos == len(labels):
        return None
    tp = 0
    precisions = []
    for i, (_s, lab) in enumerate(pairs, 1):
        if lab:
            tp += 1
            precisions.append(tp / i)
    return float(np.mean(precisions))


@torch.no_grad()
def main() -> None:
    device = torch.device("cuda")
    model = V06State().to(device)
    bundle = torch.load(WM_CKPT, weights_only=False)
    assert bundle.get("run_schema") == "v069"
    model.load_state_dict(bundle["model"])
    model.eval()
    ema = V06State().to(device)
    ema.load_state_dict(bundle["ema"])
    ema.eval()

    support = json.loads(
        (RESULTS / "v067_support_report.json").read_text())
    tolerances = support["frozen_outcome_tolerances"]
    unstable = {r["group"] for r in support["groups"]
                if r["replay_unstable"]}
    hc_pairs = support["history_contrasts"]

    from scripts.build_v067_teachers import unroll_z
    from scripts.train_v067_wm import variant_of

    dev_sources = []
    for sp in sorted((DATA / "sources").glob("*.pt")):
        s = torch.load(sp, weights_only=False)
        if s["split"] == "dev":
            dev_sources.append((sp, s))

    metrics: dict = {"run_schema": RUN_SCHEMA, "per_source": {},
                     "pooled": {}}
    pooled = {"branch_abs_model": [], "branch_abs_zero": [],
              "branch_abs_copy": [], "branch_cent_model": [],
              "branch_cent_zero": [], "valid_scores": [],
              "valid_labels": [], "flip10_scores": [],
              "flip10_labels": [], "flip01_scores": [],
              "flip01_labels": [], "cur_acc": [], "prefix_err": [],
              "closure": {1: [], 2: [], 3: []}, "para_state": [],
              "para_pred": [], "cross_phys_agree": [],
              "cross_sem_disagree_detect": [], "rank": [],
              "z_by_task": {}}

    for sp, s in dev_sources:
        sid = s["source_id"]
        rows = s["rows"]
        labels = load_v067(DATA / "semantic_relabels_v067" / sp.name,
                           "semantic_labels")
        feats = {}
        for fp in sorted(DATA.glob(f"features/{sp.stem}__*.pt")):
            f = torch.load(fp, weights_only=False)
            feats[f["variant"]["variant_id"]] = f
        canon_gid = labels["canonical_goal_spec_id"]
        zs = unroll_z(model, feats["canonical"], rows, device)
        zs_ema = unroll_z(ema, feats["canonical"], rows, device)
        pooled["z_by_task"].setdefault(s["task"], []).extend(
            [z.mean(dim=1).cpu() for z in zs[::8]])

        # current grounding + closure
        for i in range(len(rows)):
            lab = labels["per_decision"][i]["goals"][canon_gid]
            cur = model.d_current(zs[i])
            n_sub = len(lab["valid_before"])
            pred_bits = (cur["valid_bits"][0, :n_sub] > 0).cpu()
            pooled["cur_acc"].append(float(
                (pred_bits == torch.tensor(
                    lab["valid_before"])).float().mean()))
            pooled["prefix_err"].append(abs(
                float(cur["ordered_prefix"][0])
                - lab["ordered_prefix_before"] / n_sub))
            a_i = rows[i]["chunk_norm"][None, :10].float().to(device)
            zo = zs[i]
            for k in range(3):
                if i + k + 1 >= len(rows):
                    break
                a_k = rows[i + k]["chunk_norm"][None, :10].float().to(
                    device)
                zo = model.predict(zo, a_k)
                pooled["closure"][k + 1].append(float(
                    torch.nn.functional.mse_loss(
                        zo, zs_ema[i + k + 1])))
                if k == 0:
                    del a_i

        # paraphrase agreement
        for vid, f in feats.items():
            if not vid.startswith("paraphrase"):
                continue
            zp = unroll_z(model, f, rows, device)
            for i in range(0, len(rows), 8):
                pooled["para_state"].append(float(
                    torch.nn.functional.mse_loss(zp[i], zs[i])))
                a = rows[i]["chunk_norm"][None, :10].float().to(device)
                pooled["para_pred"].append(float(
                    torch.nn.functional.mse_loss(
                        model.predict(zp[i], a),
                        model.predict(zs[i], a))))

        # branch physical + immediate semantic + ranking on dev groups
        per_src = {"groups": []}
        for cp in sorted(DATA.glob(f"continuations_v067/{sid}_d*.pt")):
            cont = load_v067(cp, "continuations")
            d = cont["decision"]
            gid_key = f"{sid}_d{d}"
            audit = next(a for a in s["audits"] if a["decision"] == d)
            cands = {b["candidate"]: b for b in audit["branches"]
                     if b["kind"] == "candidate"}
            z = zs[d]
            preds, targets = [], []
            for c, b in sorted(cands.items()):
                zt = model.predict(
                    z, b["chunk_norm"][None, :10].float().to(device))
                out = model.d_next(zt)
                d_obj = (b["obj_after"]
                         - rows[d]["obj_before"]).flatten()
                n_obj = d_obj.shape[0]
                preds.append(
                    out["d_obj"][0].cpu().numpy().flatten()[:n_obj])
                targets.append(np.asarray(d_obj, dtype=np.float64))
            P, T = np.stack(preds), np.stack(targets)
            pooled["branch_abs_model"].append(
                float(np.sqrt(((P - T) ** 2).mean())))
            pooled["branch_abs_zero"].append(
                float(np.sqrt((T ** 2).mean())))
            copy_base = np.zeros_like(T)  # copy = no object motion
            pooled["branch_abs_copy"].append(
                float(np.sqrt(((copy_base - T) ** 2).mean())))
            Pc, Tc = P - P.mean(0), T - T.mean(0)
            pooled["branch_cent_model"].append(
                float(np.sqrt(((Pc - Tc) ** 2).mean())))
            pooled["branch_cent_zero"].append(
                float(np.sqrt((Tc ** 2).mean())))

            # immediate next-semantic PR over all goals x candidates
            cross_sem_rows = 0
            for b in cont["branch_summaries"]:
                if b["kind"] == "replay":
                    continue
                chunk = cands.get(b["candidate"], {}).get("chunk_norm") \
                    if b["kind"] == "candidate" else None
                if chunk is None:
                    continue
                goal_valid_preds = {}
                for gid in cont["goals"]:
                    vid = variant_of(gid, canon_gid)
                    if vid == "canonical":
                        zg = z
                    elif vid in feats:
                        zg = unroll_z(model, feats[vid], rows[:d + 1],
                                      device)[-1]
                    else:
                        continue
                    out = model.d_next(model.predict(
                        zg, chunk[None, :10].float().to(device)))
                    imm = b["immediate"][gid]
                    n_sub = len(imm["valid_after"])
                    sc = torch.sigmoid(
                        out["valid_bits"][0, :n_sub]).cpu().tolist()
                    pooled["valid_scores"].extend(sc)
                    pooled["valid_labels"].extend(
                        [int(v) for v in imm["valid_after"]])
                    goal_valid_preds[gid] = [x > 0.5 for x in sc]
                    f10 = torch.sigmoid(
                        out["flips_10"][0, :n_sub]).cpu().tolist()
                    f10_lab = [0] * n_sub
                    for fl in imm["flips_10"]:
                        f10_lab[fl[1]] = 1
                    pooled["flip10_scores"].extend(f10)
                    pooled["flip10_labels"].extend(f10_lab)
                    f01 = torch.sigmoid(
                        out["flips_01"][0, :n_sub]).cpu().tolist()
                    f01_lab = [0] * n_sub
                    for fl in imm["flips_01"]:
                        f01_lab[fl[1]] = 1
                    pooled["flip01_scores"].extend(f01)
                    pooled["flip01_labels"].extend(f01_lab)
                # crossed rows where labels disagree between goals:
                # does the model's prediction disagree the same way?
                for g1, g2 in itertools.combinations(
                        sorted(goal_valid_preds), 2):
                    l1 = b["immediate"][g1]["valid_after"]
                    l2 = b["immediate"][g2]["valid_after"]
                    if l1 != l2:
                        cross_sem_rows += 1
                        pooled["cross_sem_disagree_detect"].append(
                            float(goal_valid_preds[g1]
                                  != goal_valid_preds[g2]))

            # ranking (clean groups only)
            if gid_key not in unstable:
                by_branch: dict[int, list] = {}
                for r in cont["records"]:
                    if (r["goal_spec_id"] == canon_gid
                            and r["provenance"] == "policy"):
                        by_branch.setdefault(
                            r["branch_key"], []).append(r)
                svals = {}
                for c, b in sorted(cands.items()):
                    svals[c] = float(model.d_next(model.predict(
                        z, b["chunk_norm"][None, :10].float().to(
                            device)))["s"])
                for i, j in itertools.combinations(
                        sorted(by_branch), 2):
                    yi = [r["outcome"] for r in sorted(
                        by_branch[i], key=lambda r: r["repeat"])]
                    yj = [r["outcome"] for r in sorted(
                        by_branch[j], key=lambda r: r["repeat"])]
                    a_ij = paired_preference(yi, yj, tolerances)
                    if a_ij == 0:
                        continue
                    p = 1 / (1 + np.exp(-(svals[i] - svals[j])))
                    pooled["rank"].append({
                        "group": gid_key, "p": float(p),
                        "label": int(a_ij == 1),
                        "task": s["task"]})
            per_src["groups"].append(gid_key)
        metrics["per_source"][sid] = per_src

    # crossed-goal physical agreement: same branch, two goal states
    # (computed from shared_phys structure): d_obj prediction distance
    # between canonical and distinct z on dev decisions
    for sp, s in dev_sources:
        feats = {}
        for fp in sorted(DATA.glob(f"features/{sp.stem}__*.pt")):
            f = torch.load(fp, weights_only=False)
            feats[f["variant"]["variant_id"]] = f
        rows = s["rows"]
        zs = unroll_z(model, feats["canonical"], rows, device)
        for vid, f in feats.items():
            if not vid.startswith("distinct"):
                continue
            zd = unroll_z(model, f, rows, device)
            for i in range(0, len(rows), 8):
                a = rows[i]["chunk_norm"][None, :10].float().to(device)
                o1 = model.d_next(model.predict(zs[i], a))["d_obj"]
                o2 = model.d_next(model.predict(zd[i], a))["d_obj"]
                pooled["cross_phys_agree"].append(float(
                    (o1 - o2).abs().mean()))

    rank = pooled["rank"]
    rank_groups = {r["group"] for r in rank}
    ece_bins = np.linspace(0, 1, 6)
    ece = None
    if rank:
        ps = np.array([r["p"] for r in rank])
        ls = np.array([r["label"] for r in rank])
        ece_terms = []
        for lo, hi in zip(ece_bins[:-1], ece_bins[1:]):
            m = (ps >= lo) & (ps < hi)
            if m.sum():
                ece_terms.append(m.mean()
                                 * abs(ps[m].mean() - ls[m].mean()))
        ece = float(sum(ece_terms))
    metrics["pooled"] = {
        "branch_abs_rmse_model": float(np.mean(
            pooled["branch_abs_model"])) if pooled[
            "branch_abs_model"] else None,
        "branch_abs_rmse_zero": float(np.mean(
            pooled["branch_abs_zero"])) if pooled[
            "branch_abs_zero"] else None,
        "branch_centered_rmse_model": float(np.mean(
            pooled["branch_cent_model"])) if pooled[
            "branch_cent_model"] else None,
        "branch_centered_rmse_zero": float(np.mean(
            pooled["branch_cent_zero"])) if pooled[
            "branch_cent_zero"] else None,
        "next_valid_auprc": auprc(pooled["valid_scores"],
                                  pooled["valid_labels"]),
        "flip10_auprc": auprc(pooled["flip10_scores"],
                              pooled["flip10_labels"]),
        "flip01_auprc": auprc(pooled["flip01_scores"],
                              pooled["flip01_labels"]),
        "current_valid_acc": float(np.mean(pooled["cur_acc"])),
        "ordered_prefix_mae": float(np.mean(pooled["prefix_err"])),
        "closure_mse": {k: (float(np.mean(v)) if v else None)
                        for k, v in pooled["closure"].items()},
        "paraphrase_state_mse": float(np.mean(pooled["para_state"])),
        "paraphrase_pred_mse": float(np.mean(pooled["para_pred"])),
        "cross_goal_phys_pred_dist": float(np.mean(
            pooled["cross_phys_agree"])) if pooled[
            "cross_phys_agree"] else None,
        "cross_sem_disagree_detect_rate": float(np.mean(
            pooled["cross_sem_disagree_detect"])) if pooled[
            "cross_sem_disagree_detect"] else None,
        "n_cross_sem_disagree_rows": len(
            pooled["cross_sem_disagree_detect"]),
        "ranking": {
            "n_pairs": len(rank),
            "n_groups": len(rank_groups),
            "accuracy": float(np.mean(
                [(r["p"] > 0.5) == (r["label"] == 1)
                 for r in rank])) if rank else None,
            "bce": float(np.mean(
                [-(r["label"] * np.log(max(r["p"], 1e-6))
                   + (1 - r["label"])
                   * np.log(max(1 - r["p"], 1e-6)))
                 for r in rank])) if rank else None,
            "ece_5bin": ece,
            "pairs": rank,
        },
        "history_contrast": ("unsupported (0 matched pairs)"
                             if not hc_pairs else
                             {"n_pairs": len(hc_pairs)}),
        "latent_task_variation": {
            t: float(torch.stack(v).std(dim=0).mean())
            for t, v in pooled["z_by_task"].items() if len(v) > 1},
    }
    out = RESULTS / "v069_predictive" / "dev_metrics.json"
    out.write_text(json.dumps(metrics, indent=2, default=str))
    show = {k: v for k, v in metrics["pooled"].items()
            if k != "ranking"}
    show["ranking"] = {k: v for k, v in
                       metrics["pooled"]["ranking"].items()
                       if k != "pairs"}
    print(json.dumps(show, indent=2, default=str), flush=True)
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
