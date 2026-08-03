#!/usr/bin/env python
"""V7.1.3 — direct task-conditioned outcome ensemble (first scorer).

Bindings registered in plan_and_progress/2026-08-02.md before launch.
Predictive state FROZEN (lc_full_selected); only scorer heads train.
166 train / 3 dev continuation-bearing transitions; the 6 audit-only
stored-dev-winner rows are EVAL-ONLY ranking candidates. GT preference
= paired_preference under the frozen outcome tolerances. Ensemble K=5,
no checkpoint selection. Matched comparison scorers: lc_full /
readout_baseline / shuffled_lang states, proprio-only, family-prior
floor. Tensor-only run. Output:
results/libero_loho_public_v1/2026-08-02_v071_outcome_r1/
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

from lcwm.v06_model import V06State  # noqa: E402
from lcwm.v071_loader import V071Loader  # noqa: E402
from lcwm.task_automaton import paired_preference, pref  # noqa: E402

RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
UNION = RESULTS / "2026-08-02_v071_union_f1"
LCWM2C = RESULTS / "2026-08-02_v071_lcwm2c_r1"
CACHE_2A = RESULTS / "2026-08-02_v071_lcwm_r1" / "train"
STAGE, RUN_ID = "outcome", "r1"
K_ENS, EPOCHS, LR, WD = 5, 300, 1e-3, 1e-4
DMG_S, TAU_S = 10.0, 100.0
STATE_VARIANTS = {
    "lc_full": [CACHE_2A / "h_canonical",
                LCWM2C / "train" / "ext_lc_full"],
    "readout_baseline": [CACHE_2A / "h_constant",
                         LCWM2C / "train" / "ext_readout_baseline"],
    "shuffled_lang": [LCWM2C / "train" / "ext_shuffled_lang",
                      LCWM2C / "train" / "query_shuffled_lang"],
}
FAMS = ("canonical", "atomic", "distinct", "servo")


def hget(dirs, lru, key, device):
    if key not in lru:
        if len(lru) > 300:
            lru.clear()
        for d in dirs:
            p = Path(d) / f"{key}.pt"
            if p.exists():
                lru[key] = torch.load(p, weights_only=False)
                break
        else:
            raise KeyError(key)
    d = lru[key]
    return (d["h"][None].float().to(device),
            d["mask"][None].to(device))


def target_vec(c):
    q = c["q_at_horizons"]
    return np.array([
        1.0 if c["success_by_100"] else 0.0,
        q[10], q[30], q[60], q[100], c["q_valid_mean"],
        c["p_valid_100"], c["neg_damage"] / DMG_S,
        c["neg_tau_next"] / TAU_S], dtype=np.float32)


def assemble_outcome(v):
    return {"success_by_100": float(v[0] > 0.5),
            "q_valid_mean": float(v[5]), "p_valid_100": float(v[6]),
            "neg_damage": float(v[7]) * DMG_S,
            "neg_tau_next": float(v[8]) * TAU_S}


class Head(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 256), nn.GELU(),
                                 nn.Linear(256, 256), nn.GELU(),
                                 nn.Linear(256, 9))

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def main() -> None:
    from lcwm.seq_prefix_cache import normalize_actions
    from lcwm.v067_lineage import sha256_file

    device = torch.device("cuda")
    date = subprocess.run(["date", "+%F"], capture_output=True,
                          text=True, env={"TZ": "America/Chicago"}
                          ).stdout.strip()
    root = None
    for p in sorted(RESULTS.glob(f"*_v071_{STAGE}_{RUN_ID}")):
        root = p
    root = root or RESULTS / f"{date}_v071_{STAGE}_{RUN_ID}"
    root.mkdir(exist_ok=True)

    ref = torch.load(Path("/home/stargazer/Desktop/vla_wm/datasets"
                          "/seq_prefix_cache_v1/task0_demo0.pt"),
                     weights_only=False)
    ref_mean, ref_std = ref["action_mean"], ref["action_std_eps"]
    loader = V071Loader(UNION)
    union = json.loads((UNION / "union_manifest.json").read_text())
    tolerances = json.loads(
        (RESULTS / "v067_support_report.json").read_text()
    )["frozen_outcome_tolerances"]
    sources_rows = {sid: torch.load(b["path"], weights_only=False)
                    ["rows"] for sid, b in
                    union["source_histories"].items()}
    roots = {k: Path(v) for k, v in union["replay_roots"].items()}
    all_rows = {r["transition_id"]: r for r in union["rows"]}

    git_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True,
        text=True, cwd=REPO_ROOT).stdout.strip()
    manifest = {
        "schema": "v071_outcome_manifest_v1", "run_schema": "v071",
        "run_id": RUN_ID, "stage": STAGE, "run_date": date,
        "git_sha": git_sha,
        "predictive_frozen": {
            v: sha256_file(LCWM2C / "checkpoints"
                           / f"{v}_selected.pt")
            for v in STATE_VARIANTS},
        "union_manifest_sha256": sha256_file(
            UNION / "union_manifest.json"),
        "tolerances": tolerances,
        "config": {"k_ens": K_ENS, "epochs": EPOCHS, "lr": LR,
                   "wd": WD, "dmg_scale": DMG_S, "tau_scale": TAU_S,
                   "selection": "NONE (final ensemble)"},
        "audit_rows_eval_only": True,
        "video": "tensor-only run",
    }
    mp = root / "run_manifest.json"
    if not mp.exists():
        mp.write_text(json.dumps(manifest, indent=2))

    # ---- collect labeled transitions (incl. audit-only for eval) ------
    by_shard = defaultdict(list)
    for r in union["rows"]:
        by_shard[(r["run"], r["shard"])].append(r["transition_id"])
    labeled = {}
    for (run, shard), tids in sorted(by_shard.items()):
        s = torch.load(roots[run] / "shards" / shard,
                       weights_only=False)
        for tr in s["transitions"]:
            if tr["transition_id"] in set(tids) \
                    and tr["continuations"]:
                r = all_rows[tr["transition_id"]]
                labeled[tr["transition_id"]] = {
                    "meta": r, "steps": int(tr["steps"]),
                    "actions_env": np.asarray(tr["actions_env"]),
                    "eef0": np.asarray(tr["eef_seq"][0]),
                    "conts": [dict(c) for c in tr["continuations"]],
                }
    train_ids = sorted(t for t, v in labeled.items()
                       if not v["meta"]["audit_only"]
                       and v["meta"]["split"] == "train")
    dev_ids = sorted(t for t, v in labeled.items()
                     if not v["meta"]["audit_only"]
                     and v["meta"]["split"] == "dev")
    audit_ids = sorted(t for t, v in labeled.items()
                       if v["meta"]["audit_only"])
    print(f"labeled: {len(train_ids)} train / {len(dev_ids)} dev / "
          f"{len(audit_ids)} audit-eval", flush=True)
    assert len(train_ids) == 166 and len(dev_ids) == 3

    # ---- state features per variant ------------------------------------
    def norm_chunk(a_env):
        ae = torch.from_numpy(np.asarray(a_env)).float()
        cn = normalize_actions(ae, ref_mean, ref_std)[None].to(device)
        am = (torch.arange(10, device=device)[None] < ae.shape[0])
        if cn.shape[1] < 10:
            cn = torch.cat([cn, torch.zeros(
                1, 10 - cn.shape[1], 7, device=device)], dim=1)
        return cn, am

    feats = {}
    for vname, dirs in STATE_VARIANTS.items():
        model = V06State().to(device)
        st = torch.load(LCWM2C / "checkpoints"
                        / f"{vname}_selected.pt", weights_only=False)
        model.load_state_dict(st["model"])
        model.eval()
        lru = {}
        z_anchor = {}
        fv = {}
        for tid in sorted(labeled):
            r = labeled[tid]["meta"]
            akey = (r["source_id"], r["decision"])
            if akey not in z_anchor:
                sid, d = akey
                z = None
                for dd in range(d + 1):
                    h, m = hget(dirs, lru, f"{sid}__dec{dd}", device)
                    if z is None:
                        z = model.initial_state(h, m)
                    else:
                        prev = sources_rows[sid][dd - 1]
                        a = prev["chunk_norm"][None, :10].float() \
                            .to(device)
                        am = (torch.arange(10, device=device)[None]
                              < prev["executed_len"])
                        z = model.step(z, a, h, m, action_mask=am)
                z_anchor[akey] = z
            z = z_anchor[akey]
            cn, am = norm_chunk(labeled[tid]["actions_env"])
            zt = model.predict(z, cn, action_mask=am)
            fv[tid] = torch.cat([z.mean(dim=1)[0],
                                 zt.mean(dim=1)[0]]).cpu().numpy()
        feats[vname] = fv
        del model
        print(f"[feats:{vname}] {len(fv)} rows", flush=True)
    fv = {}
    for tid in sorted(labeled):
        a = np.zeros((10, 7), dtype=np.float32)
        ae = labeled[tid]["actions_env"]
        a[:ae.shape[0]] = ae
        fv[tid] = np.concatenate(
            [labeled[tid]["eef0"], a.flatten()]).astype(np.float32)
    feats["proprio"] = fv
    feats["family_prior"] = {
        tid: np.eye(len(FAMS), dtype=np.float32)[
            FAMS.index(labeled[tid]["meta"]["family"])
            if labeled[tid]["meta"]["family"] in FAMS else 0]
        for tid in sorted(labeled)}

    # ---- train ensembles (grad enabled locally) ------------------------
    report = {}
    scorers = {}
    for vname, fv in feats.items():
        X = torch.tensor(np.stack(
            [fv[t] for t in train_ids for _ in
             range(len(labeled[t]["conts"]))])).to(device)
        Y = torch.tensor(np.stack(
            [target_vec(c) for t in train_ids
             for c in labeled[t]["conts"]])).to(device)
        ens = []
        with torch.enable_grad():
            for k in range(K_ENS):
                torch.manual_seed(k)
                head = Head(X.shape[1]).to(device)
                opt = torch.optim.AdamW(head.parameters(), lr=LR,
                                        weight_decay=WD)
                for ep in range(EPOCHS):
                    opt.zero_grad(set_to_none=True)
                    out = head(X)
                    loss = (nn.functional
                            .binary_cross_entropy_with_logits(
                                out[:, 0], Y[:, 0])
                            + nn.functional.mse_loss(out[:, 1:],
                                                     Y[:, 1:]))
                    loss.backward()
                    opt.step()
                head.eval()
                ens.append(head)
        scorers[vname] = ens

        def predict(tid):
            x = torch.tensor(fv[tid])[None].to(device)
            outs = []
            for head in ens:
                o = head(x)[0]
                o = torch.cat([torch.sigmoid(o[:1]), o[1:]])
                outs.append(o.cpu().numpy())
            outs = np.stack(outs)
            return outs.mean(0), outs.std(0)

        # ranking eval: decided GT pairs within anchors
        def rank_eval(ids):
            by_anchor = defaultdict(list)
            for t in ids:
                by_anchor[labeled[t]["meta"]["anchor"]].append(t)
            n_pairs, n_correct = 0, 0
            for anchor, ts in by_anchor.items():
                for i in range(len(ts)):
                    for j in range(i + 1, len(ts)):
                        gt = paired_preference(
                            labeled[ts[i]]["conts"],
                            labeled[ts[j]]["conts"], tolerances)
                        if gt == 0:
                            continue
                        mi, _ = predict(ts[i])
                        mj, _ = predict(ts[j])
                        pd = pref(assemble_outcome(mi),
                                  assemble_outcome(mj), tolerances)
                        n_pairs += 1
                        n_correct += int(pd == gt)
            return n_pairs, n_correct

        dev_pool = dev_ids + audit_ids
        ndp, ndc = rank_eval(dev_pool)
        ntp, ntc = rank_eval(train_ids)
        errs, stds = [], []
        for t in train_ids + dev_pool:
            m, s = predict(t)
            y = np.mean([target_vec(c)
                         for c in labeled[t]["conts"]], axis=0)
            errs.append(np.abs(m - y).mean())
            stds.append(s.mean())
        calib = float(np.corrcoef(stds, errs)[0, 1]) \
            if len(errs) > 2 else None
        dev_mse = float(np.mean([
            ((predict(t)[0] - np.mean([target_vec(c) for c in
                                       labeled[t]["conts"]],
                                      axis=0)) ** 2).mean()
            for t in dev_ids]))
        report[vname] = {
            "dev_rank_pairs_decided": ndp,
            "dev_rank_correct": ndc,
            "dev_rank_acc": (ndc / ndp) if ndp else None,
            "train_rank_pairs_decided": ntp,
            "train_rank_acc": (ntc / ntp) if ntp else None,
            "dev_mse_3rows_thin": dev_mse,
            "calibration_std_err_corr": calib,
        }
        print(f"[{vname}] dev rank {ndc}/{ndp} "
              f"train rank {ntc}/{ntp} calib={calib}", flush=True)

    torch.save({v: [h.state_dict() for h in ens]
                for v, ens in scorers.items()},
               root / "scorer_ensembles.pt")
    (root / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1), flush=True)
    print(f"-> {root}", flush=True)


if __name__ == "__main__":
    main()
