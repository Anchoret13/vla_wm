#!/usr/bin/env python
"""V7.3C — train ONE balanced LC predictive state.

Data: merged universe (lcwm/v073_schedule.py): the V7.2 repaired
union's 96 anchors + 18 RELEASED V7.2 teacher anchors + 33 V7.3B
recovery-crossed anchors. Init: the V7.2 byte-identical shared
initialization (never a V7.2 final arm).

Objective (NO scalar rank head anywhere):
  L = l_cl*closure + l_phys*physical + l_task*outcome + l_inv*language
with the coefficients frozen BEFORE the first update by the registered
gradient-balance rule: 20 source-balanced calibration microbatches at
the shared init (4/task; every bundle active in >=10 and >=1/task),
g_k = median isolated state-grad norm per bundle,
lambda_k = g_geo / g_k. The non-vacuous 3x check is on MEAN scaled
norms (median-scaled is an identity by construction; registered).

Support masks (2026-08-04 outcome_support + released teacher):
semantic valid/flips only where per-goal traces exist; terminal
success ONLY on recovery-terminal rows of tasks with label variance
(t2); damage + tau_next weight 0 (no variance / no trace).

Frozen optimizer: AdamW 3e-4/1e-4, 25 epochs, clip 1.0, TBPTT 16,
EMA 0.995, seed 0, final-step checkpoint ONLY. One (goal, text) query
per visit, rotation (epoch+index). Tensor-only.
Output: results/libero_loho_public_v1/2026-08-06_v073_lcwm_r1/
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

from lcwm import v072_schedule as S72  # noqa: E402
from lcwm import v073_schedule as S  # noqa: E402
from lcwm.self_predict import LatentPredictor, latent_distance  # noqa: E402
from lcwm.v06_model import V06State, ema_update, make_ema  # noqa: E402

RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
V72DATA = RESULTS / "2026-08-03_v072_data_r1"
V72LOSS = RESULTS / "2026-08-03_v072_lcwm_loss_r1"
V73DATA = RESULTS / "2026-08-04_v073_data_r1"
OUT = RESULTS / "2026-08-06_v073_lcwm_r1"
LR, WD, EPOCHS, GRAD_NORM, TBPTT = 3e-4, 1e-4, 25, 1.0, 16
EMA_DECAY = 0.995
Q_SCALE = torch.tensor([7.344e-3] * 3 + [5.451e-3] * 4
                       + [2.598e-4] * 2)
OBJ_SCALE = 1.175e-3
HUB_PHYS, HUB_OUT = 4.0, 0.1
CKPT_EVERY = 5
STATE_MODULES = ("e_a", "t", "anchor", "r", "norm", "z0")
SUCCESS_VARIANCE_TASKS = {"loho_t2_basket3"}


def hub(p, t, scale, delta):
    return torch.nn.functional.huber_loss(p / scale, t / scale,
                                          delta=delta)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda")
    torch.manual_seed(0)
    for sub in ("checkpoints", "metrics"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    bce = torch.nn.functional.binary_cross_entropy_with_logits

    anchors, tq = S.load_universe()
    hidx = json.loads((OUT / "hcache_index.json").read_text())
    goal_manifest = json.loads(
        (RESULTS / "goal_spec_manifest_v067.json").read_text())

    # v072B plumbing (delegated to the V7.2B machinery's data layout)
    transitions72 = S72.load_union()
    by_pt = {r["pt_id"]: r for r in transitions72}
    union72 = json.loads((RESULTS / "2026-08-02_v071_union_f1"
                          / "union_manifest.json").read_text())
    src_paths = {sid: b["path"] for sid, b in
                 union72["source_histories"].items()}
    for tag in ("selector_r1", "selector_r2"):
        for p in (RESULTS / f"2026-08-03_v071_{tag}"
                  / "prospective_sources").glob("*.pt"):
            src_paths[p.stem] = str(p)
    raw_roots = {k: Path(v["path"]) for k, v in json.loads(
        (V72DATA / "run_manifest.json").read_text())
        ["raw_roots"].items()}
    sem_sup = defaultdict(dict)
    for line in (V72DATA / "semantic_targets.jsonl").open():
        row = json.loads(line)
        sem_sup[row["pt_id"]][row["goal_id"]] = row
    relab = {}
    for line in (raw_roots["v071_replay_r1"]
                 / "semantic_relabels.jsonl").open():
        row = json.loads(line)
        relab[(row["transition_id"], row["goal_id"])] = \
            row["immediate"]

    src_lru, shard_lru, h_lru = {}, {}, {}

    def src72(sid):
        key = ("72", sid)
        if key not in src_lru:
            if len(src_lru) > 6:
                src_lru.clear()
            src_lru[key] = torch.load(src_paths[sid],
                                      weights_only=False)
        return src_lru[key]

    def srcU(universe, sid):
        if universe == "v072B":
            return src72(sid)
        key = (universe, sid)
        if key not in src_lru:
            if len(src_lru) > 6:
                src_lru.clear()
            src_lru[key] = torch.load(
                S.sources_for(universe) / f"{sid}.pt",
                weights_only=False)
        return src_lru[key]

    def shard72(root, name):
        key = (root, name)
        if key not in shard_lru:
            if len(shard_lru) > 3:
                shard_lru.clear()
            shard_lru[key] = torch.load(
                raw_roots[root] / "shards" / name,
                weights_only=False)
        return shard_lru[key]

    def shardU(universe, akey):
        key = (universe, akey)
        if key not in shard_lru:
            if len(shard_lru) > 3:
                shard_lru.clear()
            shard_lru[key] = torch.load(
                S.shards_for(universe) / f"{akey}.pt",
                weights_only=False)
        return shard_lru[key]

    def hget(key):
        if key not in h_lru:
            if len(h_lru) > 250:
                h_lru.clear()
            h_lru[key] = torch.load(hidx[key], weights_only=False)
        d = h_lru[key]
        return (d["h"][None].float().to(device),
                d["mask"][None].to(device))

    def n_sub(gid):
        for t in goal_manifest["tasks"].values():
            if gid in t["goal_specs"]:
                return len(t["goal_specs"][gid]["ordered_subgoals"])
        raise KeyError(gid)

    # ---- per-anchor payload adapters --------------------------------
    payload_lru = {}

    def payload(ak):
        """rows: pt -> {a_norm(list or None), a_env, steps, d_eef,
        d_obj, sem{gid:(v0,v1,f01,f10)}, cont{gid: gbar(5)},
        recterm_success(float|None), next_key_suffix}."""
        if ak in payload_lru:
            return payload_lru[ak]
        if len(payload_lru) > 30:
            payload_lru.clear()
        a = anchors[ak]
        rows = {}
        if a["universe"] == "v072B":
            if a["origin"] == "v071_semwin":
                rec0 = by_pt[a["rows"][0]]
                s = shard72(rec0["raw_root"], rec0["raw_shard"])
                cum = [0]
                for sg_ in s["segments"]:
                    cum.append(cum[-1] + sg_["actions"])
                for pt in a["rows"]:
                    rec = by_pt[pt]
                    k = int(rec["raw_key"].split("|")[1][3:])
                    sem = {}
                    for gid, vs in s["valid_seq"].items():
                        v0 = np.asarray(vs[cum[k]], dtype=bool)
                        v1 = np.asarray(vs[cum[k + 1]], dtype=bool)
                        sem[gid] = (v0, v1,
                                    (v1 & ~v0).astype(np.float32),
                                    (~v1 & v0).astype(np.float32))
                    rows[pt] = {
                        "a_norm": rec["actions_pi05_norm"],
                        "steps": rec["steps"],
                        "d_eef": (np.asarray(
                            s["eef_seq"][cum[k + 1]])
                            - np.asarray(s["eef_seq"][cum[k]])),
                        "d_obj": ((np.asarray(s["obj_after"])
                                   - np.asarray(s["obj_before"]))
                                  if k == len(cum) - 2 else None),
                        "sem": sem, "cont": {},
                        "recterm_success": None,
                        "next_key": f"::swb::{rec['raw_key'].split('|')[0]}::b{cum[k + 1]}"}
            else:
                obj_before = np.asarray(
                    src72(a["source_id"])["rows"][a["decision"]]
                    ["obj_before"])
                for pt in a["rows"]:
                    rec = by_pt[pt]
                    s = shard72(rec["raw_root"], rec["raw_shard"])
                    if rec["origin"].startswith("v071_sel"):
                        tr = next(
                            x for x in s["transitions"]
                            if f"{s['anchor']}_{x['candidate_id']}"
                            == rec["raw_key"])
                    else:
                        tr = next(x for x in s["transitions"]
                                  if x["transition_id"]
                                  == rec["raw_key"])
                    sem = {}
                    if rec["origin"] == "v071_union":
                        for gid, t_ in sem_sup.get(pt, {}).items():
                            if not t_["supported"]:
                                continue
                            imm = (relab.get((rec["raw_key"], gid))
                                   if t_["source"]
                                   == "relabel_table"
                                   else tr.get(
                                       "immediate_canonical"))
                            if imm is None:
                                continue
                            va = np.asarray(imm["valid_after"],
                                            dtype=bool)
                            f01 = np.zeros(len(va),
                                           dtype=np.float32)
                            f10 = np.zeros(len(va),
                                           dtype=np.float32)
                            for f in imm.get("flips_01", []):
                                f01[f[1]] = 1.0
                            for f in imm.get("flips_10", []):
                                f10[f[1]] = 1.0
                            sem[gid] = (None, va, f01, f10)
                    cont = {}
                    if tr["continuations"]:
                        cg = S.canon_goal(rec["task"])
                        gs = []
                        for c in tr["continuations"]:
                            q = {int(kk): vv for kk, vv in
                                 c["q_at_horizons"].items()}
                            gs.append([c["p_valid_100"], q[10],
                                       q[30], q[60], q[100]])
                        cont[cg] = np.mean(gs, axis=0)
                    rows[pt] = {
                        "a_norm": rec["actions_pi05_norm"],
                        "steps": rec["steps"],
                        "d_eef": (np.asarray(tr["eef_seq"][-1])
                                  - np.asarray(tr["eef_seq"][0])),
                        "d_obj": (np.asarray(tr["obj_after"])
                                  - obj_before),
                        "sem": sem, "cont": cont,
                        "recterm_success": None,
                        "next_key": f"::next::{pt}"}
        elif a["universe"] == "v072T":
            sid, d = a["source_id"], a["decision"]
            s = shardU("v072T", f"{sid}_d{d}")
            obj_before = np.asarray(
                srcU("v072T", sid)["rows"][d]["obj_before"])
            cg = S.canon_goal(a["task"])
            for tr in s["transitions"]:
                if tr["kind"] == "audit":
                    continue
                pt = f"v072T::{tr['transition_id']}"
                vs = tr["valid_seq_canon"]
                v0 = np.asarray(vs[0], dtype=bool)
                v1 = np.asarray(vs[-1], dtype=bool)
                cont = {}
                if tr["continuations"]:
                    gs = []
                    for c in tr["continuations"]:
                        q = {int(kk): vv for kk, vv in
                             c["q_at_horizons"].items()}
                        gs.append([c["p_valid_100"], q[10], q[30],
                                   q[60], q[100]])
                    cont[cg] = np.mean(gs, axis=0)
                rows[pt] = {
                    "a_norm": None,
                    "a_env": np.asarray(tr["actions_env"]),
                    "steps": tr["steps"],
                    "d_eef": (np.asarray(tr["eef_seq"][-1])
                              - np.asarray(tr["eef_seq"][0])),
                    "d_obj": (np.asarray(tr["obj_after"])
                              - obj_before),
                    "sem": {cg: (v0, v1,
                                 (v1 & ~v0).astype(np.float32),
                                 (~v1 & v0).astype(np.float32))},
                    "cont": cont, "recterm_success": None,
                    "next_key": f"::next::{pt}"}
        else:   # v073
            sid, d = a["source_id"], a["decision"]
            s = shardU("v073", f"{sid}_d{d}")
            obj_before = np.asarray(s["obj_before"])
            for tr in s["transitions"]:
                if tr["branch_key"] == "u0_repeat":
                    continue
                pt = f"v073::{tr['transition_id']}"
                sem = {}
                for gid, vs in tr["valid_seq"].items():
                    v0 = np.asarray(vs[0], dtype=bool)
                    v1 = np.asarray(vs[-1], dtype=bool)
                    sem[gid] = (v0, v1,
                                (v1 & ~v0).astype(np.float32),
                                (~v1 & v0).astype(np.float32))
                cont = {}
                for gid in s["goal_ids"]:
                    if tr["continuations"]:
                        gs = []
                        for ys in tr["continuations"]:
                            c = ys[gid]
                            q = {int(kk): vv for kk, vv in
                                 c["q_at_horizons"].items()}
                            gs.append([c["p_valid_100"], q[10],
                                       q[30], q[60], q[100]])
                        cont[gid] = np.mean(gs, axis=0)
                rts = None
                if tr["recovery_terminal"] is not None and \
                        a["task"] in SUCCESS_VARIANCE_TASKS:
                    cg = S.canon_goal(a["task"])
                    rts = float(tr["recovery_terminal"]
                                ["outcomes"][cg]["success_by_100"])
                rows[pt] = {
                    "a_norm": (tr["chunk_norm"]
                               if tr["chunk_norm"] is not None
                               else None),
                    "a_env": np.asarray(tr["actions_env"]),
                    "steps": tr["steps"],
                    "d_eef": (np.asarray(tr["eef_seq"][-1])
                              - np.asarray(tr["eef_seq"][0])),
                    "d_obj": (np.asarray(tr["obj_after"])
                              - obj_before),
                    "sem": sem, "cont": cont,
                    "recterm_success": rts,
                    "next_key": f"::next::{pt}"}
        payload_lru[ak] = rows
        return rows

    ref = torch.load(Path("/home/stargazer/Desktop/vla_wm/datasets"
                          "/seq_prefix_cache_v1/task0_demo0.pt"),
                     weights_only=False)
    ref_mean, ref_std = ref["action_mean"], ref["action_std_eps"]
    from lcwm.seq_prefix_cache import normalize_actions

    def a_norm_of(pl):
        if pl["a_norm"] is not None:
            cn = torch.as_tensor(np.asarray(pl["a_norm"]),
                                 dtype=torch.float32,
                                 device=device)[None, :10]
        else:
            ae = torch.from_numpy(
                np.asarray(pl["a_env"])).float()
            cn = normalize_actions(ae, ref_mean,
                                   ref_std)[None].to(device)[:, :10]
        if cn.shape[1] < 10:
            cn = torch.cat([cn, torch.zeros(
                1, 10 - cn.shape[1], 7, device=device)], dim=1)
        am = (torch.arange(10, device=device)[None]
              < min(int(pl["steps"]), 10))
        return cn, am

    model = V06State().to(device)
    init = torch.load(V72LOSS / "shared_initialization.pt",
                      weights_only=False)
    model.load_state_dict(init["model"])
    pred = LatentPredictor().to(device)
    pred.load_state_dict(init["predictor"])
    ema = make_ema(model)
    ema.load_state_dict(init["ema"])
    params = list(model.parameters()) + list(pred.parameters())
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    state_params = [p for n, p in model.named_parameters()
                    if n.split(".")[0] in STATE_MODULES]
    q_scale = Q_SCALE.to(device)

    def unroll(model_, ak, tvv, train):
        a = anchors[ak]
        sid, d = a["source_id"], a["decision"]
        z = None
        rows_ = srcU(a["universe"] if a["universe"] != "v072B"
                     else "v072B", sid)["rows"] \
            if a["universe"] != "v072B" else src72(sid)["rows"]
        for dd in range(d):
            h, m = hget(f"{tvv}::src::{sid}::d{dd}")
            if z is None:
                z = model_.initial_state(h, m)
            else:
                prev = rows_[dd - 1]
                aa = prev["chunk_norm"][None, :10].float().to(device)
                am = (torch.arange(10, device=device)[None]
                      < prev["executed_len"])
                z = model_.step(z, aa, h, m, action_mask=am)
            if train and dd > 0 and dd % TBPTT == 0:
                z = z.detach()
        if a["universe"] == "v072B" \
                and a["origin"] == "v071_semwin":
            wid = by_pt[a["rows"][0]]["raw_key"].split("|")[0]
            h, m = hget(f"{tvv}::swb::{wid}::b0")
        else:
            h, m = hget(f"{tvv}::src::{sid}::d{d}")
        if z is None:
            z = model_.initial_state(h, m)
        else:
            prev = rows_[d - 1]
            aa = prev["chunk_norm"][None, :10].float().to(device)
            am = (torch.arange(10, device=device)[None]
                  < prev["executed_len"])
            z = model_.step(z, aa, h, m, action_mask=am)
        return z

    def visit_losses(ak, g, tvv, train=True, second_stream=True):
        a = anchors[ak]
        pl = payload(ak)
        z = unroll(model, ak, tvv, train)
        with torch.no_grad():
            zbar = unroll(ema, ak, tvv, False)
        canon = S.canon_goal(a["task"])
        L = defaultdict(list)
        zt_by = {}
        cont_pairs = []
        for pt, row in pl.items():
            cn, am = a_norm_of(row)
            zt = model.predict(z, cn, action_mask=am)
            zt_by[pt] = zt
            hn, mn = hget(f"{tvv}{row['next_key']}")
            with torch.no_grad():
                tgt = ema.update(
                    ema.t(zbar, ema.e_a(cn, am)), hn, mn)
            L["cl"].append(latent_distance(pred(zt), tgt.detach(),
                                           "cosine").mean())
            out = model.d_next(zt)
            d_eef = torch.from_numpy(
                np.asarray(row["d_eef"])).float().to(device)
            L["phys"].append(hub(out["d_q"][0], d_eef, q_scale,
                                 HUB_PHYS))
            if row["d_obj"] is not None:
                d_obj = torch.from_numpy(
                    np.asarray(row["d_obj"])).float().to(device)
                L["phys"].append(hub(
                    out["d_obj"][0, :d_obj.shape[0]].flatten(),
                    d_obj.flatten(), OBJ_SCALE, HUB_PHYS))
            sm = row["sem"].get(g)
            if sm is not None:
                _v0, v1, f01, f10 = sm
                n = n_sub(g)
                L["task"].append(
                    bce(out["valid_bits"][0, :n],
                        torch.from_numpy(
                            v1[:n].astype(np.float32)).to(device))
                    + bce(out["flips_01"][0, :n],
                          torch.from_numpy(f01[:n]).to(device))
                    + bce(out["flips_10"][0, :n],
                          torch.from_numpy(f10[:n]).to(device)))
            # outcome targets ONLY under the query goal itself —
            # a canonical fallback would supervise the g-conditioned
            # heads with wrong-goal targets (review finding)
            cg_t = row["cont"].get(g)
            if cg_t is not None:
                gbar = torch.from_numpy(
                    np.asarray(cg_t, dtype=np.float32)).to(device)
                ghat = torch.sigmoid(torch.cat(
                    [out["p_valid"].reshape(1),
                     out["q_valid"][0]]))
                L["task"].append(
                    torch.nn.functional.huber_loss(
                        ghat, gbar, delta=HUB_OUT))
                cont_pairs.append((ghat, gbar))
            if row["recterm_success"] is not None:
                L["task"].append(bce(
                    out["success_logit"].reshape(1),
                    torch.tensor([row["recterm_success"]],
                                 device=device)))
        # within-anchor CENTERED outcome targets (contract:
        # absolute AND centered; review finding)
        if len(cont_pairs) >= 2:
            Ph = torch.stack([a_ for a_, _b in cont_pairs])
            Tb = torch.stack([b_ for _a, b_ in cont_pairs])
            L["task"].append(torch.nn.functional.huber_loss(
                Ph - Ph.mean(0, keepdim=True),
                Tb - Tb.mean(0, keepdim=True), delta=HUB_OUT))
        # centered phys within anchor
        if len(zt_by) >= 2:
            preds_, tgts_ = [], []
            for pt, row in pl.items():
                preds_.append(model.d_next(zt_by[pt])["d_q"][0])
                tgts_.append(torch.from_numpy(np.asarray(
                    row["d_eef"])).float().to(device))
            P = torch.stack(preds_)
            T_ = torch.stack(tgts_)
            L["phys"].append(hub(P - P.mean(0, keepdim=True),
                                 T_ - T_.mean(0, keepdim=True),
                                 q_scale, HUB_PHYS))
        # language contract second stream
        if second_stream:
            base_tv = canon + "_p0"
            if tvv != base_tv:
                z2 = unroll(model, ak, base_tv, train)
                if g == canon:
                    L["inv"].append(0.5 * (
                        latent_distance(z, z2.detach(),
                                        "cosine").mean()
                        + latent_distance(z2, z.detach(),
                                          "cosine").mean()))
                else:
                    pt0 = next(iter(zt_by), None)
                    if pt0 is not None:
                        cn, am = a_norm_of(pl[pt0])
                        zt2 = model.predict(z2, cn,
                                            action_mask=am)
                        o1 = model.d_next(zt_by[pt0])["d_q"][0]
                        o2 = model.d_next(zt2)["d_q"][0]
                        L["inv"].append(0.5 * (
                            hub(o1, o2.detach(), q_scale,
                                HUB_PHYS)
                            + hub(o2, o1.detach(), q_scale,
                                  HUB_PHYS)))
        return {k: torch.stack(v).mean() for k, v in L.items() if v}

    # ---- gradient-balance calibration (frozen before update 1) -----
    lam_path = OUT / "gradient_balance.json"
    if lam_path.exists():
        lam = json.loads(lam_path.read_text())["lambda"]
    else:
        cal = []
        visit_set = S.needed_visits(anchors, tq)
        by_task_v = defaultdict(list)
        for ak, g, tvv in visit_set:
            if anchors[ak]["split"] == "train":
                by_task_v[anchors[ak]["task"]].append((ak, g, tvv))
        for task in S.TASKS:
            vs = by_task_v[task]
            cp0 = S.canon_goal(task) + "_p0"
            inv_active = [v for v in vs if v[2] != cp0]
            plain = [v for v in vs if v[2] == cp0]
            picks, seen_ak = [], set()
            for pool, want in ((inv_active, 2), (plain, 2)):
                for v in pool:
                    if len([p_ for p_ in picks
                            if p_ in pool]) >= want:
                        break
                    if v[0] in seen_ak:
                        continue
                    picks.append(v)
                    seen_ak.add(v[0])
            while len(picks) < 4 and len(picks) < len(vs):
                for v in vs:
                    if v[0] not in seen_ak:
                        picks.append(v)
                        seen_ak.add(v[0])
                        break
                else:
                    break
            cal.extend(picks[:4])
        assert len(cal) == 20, f"calibration set {len(cal)} != 20"
        norms = defaultdict(list)
        detail = []
        for ak, g, tvv in cal:
            row = {"anchor": ak, "goal": g, "text": tvv}
            for k in ("cl", "phys", "task", "inv"):
                for p in params:
                    p.grad = None
                L = visit_losses(ak, g, tvv, train=True)
                if k not in L:
                    row[k] = None
                    continue
                L[k].backward()
                gn = float(sum((p.grad ** 2).sum()
                               for p in state_params
                               if p.grad is not None).sqrt())
                norms[k].append(gn)
                row[k] = gn
            detail.append(row)
        for p in params:
            p.grad = None
        for k in ("cl", "phys", "task", "inv"):
            assert len(norms[k]) >= 10, \
                f"bundle {k} active in only {len(norms[k])} mbs"
            assert all(np.isfinite(norms[k])) and \
                min(norms[k]) > 0, f"bundle {k} zero/nonfinite"
        med = {k: float(np.median(v)) for k, v in norms.items()}
        g_geo = float(np.exp(np.mean([np.log(v)
                                      for v in med.values()])))
        lam = {k: g_geo / (med[k] + 1e-8) for k in med}
        scaled_means = {k: lam[k] * float(np.mean(norms[k]))
                        for k in med}
        mx, mn = max(scaled_means.values()), min(
            scaled_means.values())
        assert mx / max(mn, 1e-12) < 3.0, \
            f"scaled MEAN norms exceed 3x: {scaled_means}"
        lam_path.write_text(json.dumps({
            "lambda": lam, "median": med, "g_geo": g_geo,
            "scaled_means": scaled_means,
            "microbatches": detail,
            "note": "3x check on MEAN scaled norms (median-scaled "
                    "is an identity by construction; registered)"},
            indent=2))
        print(f"[lam] {lam} scaled_means={scaled_means}",
              flush=True)

    git_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True,
        text=True, cwd=REPO_ROOT).stdout.strip()
    mp = OUT / "run_manifest.json"
    if not mp.exists():
        mp.write_text(json.dumps({
            "schema": "v073_lcwm_manifest_v1", "run_schema": "v073",
            "git_sha": git_sha,
            "init": str(V72LOSS / "shared_initialization.pt"),
            "universe": {"v072B": 96, "v072T": 18, "v073": 33},
            "optimizer": {"lr": LR, "wd": WD, "epochs": EPOCHS,
                          "clip": GRAD_NORM, "tbptt": TBPTT,
                          "ema": EMA_DECAY, "seed": 0,
                          "selection": "final step ONLY"},
            "masks": {"tau_next": 0, "damage": 0,
                      "terminal_success":
                          sorted(SUCCESS_VARIANCE_TASKS),
                      "rank_family": "REMOVED (no scalar rank head)"},
            "lambda_sha": hashlib.sha256(
                lam_path.read_bytes()).hexdigest()[:16],
        }, indent=2))

    exposure = defaultdict(int)
    logs = []
    start = 0
    if args.resume:
        cks = sorted((OUT / "checkpoints").glob("lcwm_epoch*.pt"))
        if cks:
            st = torch.load(cks[-1], weights_only=False)
            model.load_state_dict(st["model"])
            pred.load_state_dict(st["predictor"])
            ema.load_state_dict(st["ema"])
            optimizer.load_state_dict(st["optimizer"])
            start = st["epoch"] + 1
            print(f"[lcwm] resumed at {start}", flush=True)
    for epoch in range(start, EPOCHS):
        order = S.rr_schedule(anchors, "train", epoch)
        agg = defaultdict(list)
        for i, ak in enumerate(order):
            g, tvv = S.visit_query(anchors, tq, ak, epoch, i)
            exposure[anchors[ak]["task"]] += 1
            optimizer.zero_grad(set_to_none=True)
            L = visit_losses(ak, g, tvv, train=True)
            if not L:
                continue
            total = sum(lam[k] * v for k, v in L.items())
            assert torch.isfinite(total)
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, GRAD_NORM)
            optimizer.step()
            ema_update(ema, model, EMA_DECAY)
            for k, v in L.items():
                agg[k].append(float(v))
        means = {k: float(np.mean(v)) for k, v in agg.items()}
        logs.append({"epoch": epoch, **means})
        print(f"[lcwm ep{epoch}] " + " ".join(
            f"{k}={v:.4f}" for k, v in sorted(means.items())),
            flush=True)
        if (epoch + 1) % CKPT_EVERY == 0 or epoch + 1 == EPOCHS:
            with torch.no_grad():
                dv = defaultdict(list)
                for ak in sorted(anchors):
                    if anchors[ak]["split"] != "dev":
                        continue
                    for g, tvv in S.queries_for(anchors[ak], tq):
                        L = visit_losses(ak, g, tvv, train=False,
                                         second_stream=False)
                        for k, v in L.items():
                            dv[k].append(float(v))
            dev = {k: float(np.mean(v)) for k, v in dv.items()}
            torch.save({"model": model.state_dict(),
                        "predictor": pred.state_dict(),
                        "ema": ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch, "dev_readout": dev},
                       OUT / "checkpoints"
                       / f"lcwm_epoch{epoch:03d}.pt")
            print(f"  [lcwm dev ep{epoch}] " + " ".join(
                f"{k}={v:.4f}" for k, v in sorted(dev.items())),
                flush=True)
    final = OUT / "checkpoints" / "final.pt"
    last = sorted((OUT / "checkpoints").glob("lcwm_epoch*.pt"))[-1]
    final.write_bytes(last.read_bytes())
    (OUT / "metrics" / "loss_by_epoch.json").write_text(
        json.dumps(logs, indent=1))
    (OUT / "exposure_log.json").write_text(
        json.dumps(dict(exposure), indent=2))
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
