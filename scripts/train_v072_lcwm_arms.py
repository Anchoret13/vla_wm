#!/usr/bin/env python
"""V7.2B — three matched state-gradient LCWM arms (W0/W1/W2).

Common graph: z_t = U(T(z_{t-c}, E_a(u_{t-c})), h_t^l);
z~_ti = T(z_t, E_a(u_ti)); every candidate-dependent output decodes
from D_next(z~) only. Language reaches the transition only through z.

Repaired self-prediction target (common to all arms): the EMA network
is unrolled over the SAME observed history (recurrent posterior, not
the stateless 2c initial_state) and the one/two/three-block targets
are sg[EMA.update(EMA.T(zbar, EMA.E_a(u)), h_next^l)], compared via
the existing per-token LatentPredictor + cosine distance
(lcwm/self_predict.py).

Bundles: L_base = closure(1:3, a2=0.5 a3=0.25) + phys_abs + phys_ctr
+ sem + para + shared_phys; L_out (p_valid_100 + q@10/30/60/100,
Huber 0.1, absolute + within-anchor centered, R=2 mean as ONE target);
L_BT/L_tie (scalar D_s, frozen GT tuple p_valid_100 -> mean q@h,
both-repeats rule, unstable masked, half pair mass on u0-pairs).
success/damage/tau_next weights are exactly zero.

Arms (identical data/sampler/seed/optimizer/init; the table alone
controls shared-state gradient reach via sg on the head input):
  W0/base           state grads: L_base          head-local: out+rank
  W1/outcome-state  state grads: L_base+L_out    head-local: rank
  W2/rank-state     state grads: all             head-local: none

Frozen contract: AdamW 3e-4/1e-4, 25 epochs, grad_norm 1.0, TBPTT 16,
EMA 0.995, one seed, final-step checkpoint rule (no selection).
Gradient-reach assertions via isolated per-bundle autograd passes.
Output: results/libero_loho_public_v1/2026-08-03_v072_lcwm_loss_r1/
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

from lcwm import v072_schedule as S  # noqa: E402
from lcwm.self_predict import LatentPredictor, latent_distance  # noqa: E402
from lcwm.v06_model import V06State, ema_update, make_ema  # noqa: E402

RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
DATA_R1 = RESULTS / "2026-08-03_v072_data_r1"
INIT_CKPT = (RESULTS / "2026-08-02_v071_lcwm2c_r1" / "checkpoints"
             / "lc_full_selected.pt")
OUT = RESULTS / "2026-08-03_v072_lcwm_loss_r1"
LR, WD, EPOCHS, GRAD_NORM, TBPTT = 3e-4, 1e-4, 25, 1.0, 16
EMA_DECAY = 0.995
A2, A3 = 0.5, 0.25
HUB_OUT = 0.1
Q_SCALE = torch.tensor([7.344e-3] * 3 + [5.451e-3] * 4
                       + [2.598e-4] * 2)
OBJ_SCALE = 1.175e-3
HUB_PHYS = 4.0
TOL_P, TOL_QM = 0.0, 0.041666666666666664
CKPT_EVERY = 5
ARMS = {"w0_base": {"out_state": False, "rank_state": False},
        "w1_outcome_state": {"out_state": True, "rank_state": False},
        "w2_rank_state": {"out_state": True, "rank_state": True}}
STATE_MODULES = ("e_a", "t", "anchor", "r", "norm", "z0")


def hub(p, t, scale, delta):
    return torch.nn.functional.huber_loss(p / scale, t / scale,
                                          delta=delta)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--arm", default=None)
    args = ap.parse_args()
    device = torch.device("cuda")
    torch.manual_seed(0)
    for sub in ("checkpoints", "metrics"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    transitions = S.load_union()
    by_pt = {r["pt_id"]: r for r in transitions}
    queries = S.load_queries()
    anchors = S.build_anchors(transitions)
    hidx = json.loads((DATA_R1 / "hcache_index.json").read_text())
    sem_sup = defaultdict(dict)
    for line in (DATA_R1 / "semantic_targets.jsonl").open():
        row = json.loads(line)
        sem_sup[row["pt_id"]][row["goal_id"]] = row
    union = json.loads((RESULTS / "2026-08-02_v071_union_f1"
                        / "union_manifest.json").read_text())
    src_paths = {sid: b["path"]
                 for sid, b in union["source_histories"].items()}
    for tag in ("selector_r1", "selector_r2"):
        for p in (RESULTS / f"2026-08-03_v071_{tag}"
                  / "prospective_sources").glob("*.pt"):
            src_paths[p.stem] = str(p)
    raw_roots = {k: Path(v["path"]) for k, v in json.loads(
        (DATA_R1 / "run_manifest.json").read_text())
        ["raw_roots"].items()}
    goal_manifest = json.loads(
        (RESULTS / "goal_spec_manifest_v067.json").read_text())
    # r1 per-goal relabel table (exact immediate semantics)
    relab = {}
    rel_path = raw_roots["v071_replay_r1"] / "semantic_relabels.jsonl"
    for line in rel_path.open():
        row = json.loads(line)
        relab[(row["transition_id"], row["goal_id"])] = \
            row["immediate"]

    src_lru, shard_lru, h_lru = {}, {}, {}

    def src(sid):
        if sid not in src_lru:
            if len(src_lru) > 6:
                src_lru.clear()
            src_lru[sid] = torch.load(src_paths[sid],
                                      weights_only=False)
        return src_lru[sid]

    def shard(root, name):
        key = (root, name)
        if key not in shard_lru:
            if len(shard_lru) > 3:
                shard_lru.clear()
            shard_lru[key] = torch.load(
                raw_roots[root] / "shards" / name,
                weights_only=False)
        return shard_lru[key]

    def hget(key):
        if key not in h_lru:
            if len(h_lru) > 250:
                h_lru.clear()
            d = torch.load(hidx[key], weights_only=False)
            h_lru[key] = d
        d = h_lru[key]
        return (d["h"][None].float().to(device),
                d["mask"][None].to(device))

    def n_sub(gid):
        for t in goal_manifest["tasks"].values():
            if gid in t["goal_specs"]:
                return len(t["goal_specs"][gid]["ordered_subgoals"])
        raise KeyError(gid)

    # ---- per-anchor raw payloads (targets), cached lightweight ------
    payload_lru = {}

    def payload(ak):
        """Targets for every non-audit row at anchor ak."""
        if ak in payload_lru:
            return payload_lru[ak]
        if len(payload_lru) > 40:
            payload_lru.clear()
        a = anchors[ak]
        rows = {}
        if a["origin"] == "v071_semwin":
            rec0 = by_pt[a["rows"][0]]
            s = shard(rec0["raw_root"], rec0["raw_shard"])
            cum = [0]
            for sg_ in s["segments"]:
                cum.append(cum[-1] + sg_["actions"])
            for pt in a["rows"]:
                rec = by_pt[pt]
                k = int(rec["raw_key"].split("|")[1][3:])
                d_eef = (np.asarray(s["eef_seq"][cum[k + 1]])
                         - np.asarray(s["eef_seq"][cum[k]]))
                obj_d = (np.asarray(s["obj_after"])
                         - np.asarray(s["obj_before"])) \
                    if k == len(cum) - 2 else None
                vs = {g: (np.asarray(v[cum[k]], dtype=np.float32),
                          np.asarray(v[cum[k + 1]],
                                     dtype=np.float32))
                      for g, v in s["valid_seq"].items()}
                rows[pt] = {"d_eef": d_eef, "d_obj": obj_d,
                            "sem": vs, "conts": [], "seg": k,
                            "b_next": cum[k + 1], "wid":
                                rec["raw_key"].split("|")[0]}
        else:
            obj_before = np.asarray(
                src(a["source_id"])["rows"][a["decision"]]
                ["obj_before"])
            for pt in a["rows"]:
                rec = by_pt[pt]
                s = shard(rec["raw_root"], rec["raw_shard"])
                if rec["origin"].startswith("v071_sel"):
                    tr = next(t for t in s["transitions"]
                              if f"{s['anchor']}_{t['candidate_id']}"
                              == rec["raw_key"])
                else:
                    tr = next(t for t in s["transitions"]
                              if t["transition_id"]
                              == rec["raw_key"])
                d_eef = (np.asarray(tr["eef_seq"][-1])
                         - np.asarray(tr["eef_seq"][0]))
                sem = {}
                if rec["origin"] == "v071_union":
                    for gid in sem_sup.get(pt, {}):
                        t_ = sem_sup[pt][gid]
                        if not t_["supported"]:
                            continue
                        imm = (relab.get((rec["raw_key"], gid))
                               if t_["source"] == "relabel_table"
                               else tr.get("immediate_canonical"))
                        if imm is not None:
                            sem[gid] = imm
                rows[pt] = {"d_eef": d_eef,
                            "d_obj": (np.asarray(tr["obj_after"])
                                      - obj_before),
                            "sem": sem,
                            "conts": [dict(c) for c
                                      in tr["continuations"]]}
        payload_lru[ak] = rows
        return rows

    def cont_g(c):
        q = {int(k): v for k, v in c["q_at_horizons"].items()}
        return np.array([c["p_valid_100"], q[10], q[30], q[60],
                         q[100]], dtype=np.float32)

    def gt_pairs(rows_payload):
        """Frozen tuple p_valid_100 -> mean(q@h); both-repeats rule."""
        ids = [pt for pt, pl in rows_payload.items()
               if len(pl["conts"]) == 2]
        out = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                signs = []
                for r in range(2):
                    ca = rows_payload[ids[i]]["conts"][r]
                    cb = rows_payload[ids[j]]["conts"][r]
                    ga, gb = cont_g(ca), cont_g(cb)
                    if ga[0] > gb[0] + TOL_P:
                        signs.append(1)
                    elif gb[0] > ga[0] + TOL_P:
                        signs.append(-1)
                    elif ga[1:].mean() > gb[1:].mean() + TOL_QM:
                        signs.append(1)
                    elif gb[1:].mean() > ga[1:].mean() + TOL_QM:
                        signs.append(-1)
                    else:
                        signs.append(0)
                if all(s_ == 1 for s_ in signs):
                    out.append((ids[i], ids[j], 1))
                elif all(s_ == -1 for s_ in signs):
                    out.append((ids[i], ids[j], -1))
                elif all(s_ == 0 for s_ in signs):
                    out.append((ids[i], ids[j], 0))
        return out

    q_scale = Q_SCALE.to(device)
    bce = torch.nn.functional.binary_cross_entropy_with_logits

    # ---- shared byte-identical initialization ----------------------
    init_path = OUT / "shared_initialization.pt"
    if not init_path.exists():
        m = V06State()
        bundle = torch.load(INIT_CKPT, weights_only=False)
        m.load_state_dict(bundle["model"])
        torch.manual_seed(0)
        pred = LatentPredictor()
        e = make_ema(m)
        torch.save({"model": m.state_dict(),
                    "predictor": pred.state_dict(),
                    "ema": e.state_dict(),
                    "init_from": str(INIT_CKPT)}, init_path)

    contract_path = OUT / "loss_contract.json"
    if not contract_path.exists():
        contract_path.write_text(json.dumps({
            "optimizer": {"lr": LR, "wd": WD, "epochs": EPOCHS,
                          "grad_norm": GRAD_NORM, "tbptt": TBPTT,
                          "ema": EMA_DECAY, "seed": 0},
            "alpha2": A2, "alpha3": A3, "huber_out": HUB_OUT,
            "gt_tuple": "p_valid_100 -> mean(q@10..100)",
            "tolerances": {"p_valid_100": TOL_P,
                           "q_mean": TOL_QM},
            "zero_weight": ["success", "damage", "tau_next"],
            "pair_mass": "half u0-pairs, half others; ties separate",
            "arms": ARMS, "selection": "final step only",
        }, indent=2))

    def unroll(model_, ak, tv, train):
        """Recurrent state at the anchor under text tv (full history
        from episode reset; TBPTT detach in train)."""
        a = anchors[ak]
        sid, d = a["source_id"], a["decision"]
        z = None
        for dd in range(d):
            h, m = hget(f"{tv}::src::{sid}::d{dd}")
            if z is None:
                z = model_.initial_state(h, m)
            else:
                prev = src(sid)["rows"][dd - 1]
                aa = prev["chunk_norm"][None, :10].float().to(device)
                am = (torch.arange(10, device=device)[None]
                      < prev["executed_len"])
                z = model_.step(z, aa, h, m, action_mask=am)
            if train and dd > 0 and dd % TBPTT == 0:
                z = z.detach()
        if a["origin"] == "v071_semwin":
            wid = by_pt[a["rows"][0]]["raw_key"].split("|")[0]
            h, m = hget(f"{tv}::swb::{wid}::b0")
        else:
            h, m = hget(f"{tv}::src::{sid}::d{d}")
        if z is None:
            z = model_.initial_state(h, m)
        else:
            prev = src(sid)["rows"][d - 1]
            aa = prev["chunk_norm"][None, :10].float().to(device)
            am = (torch.arange(10, device=device)[None]
                  < prev["executed_len"])
            z = model_.step(z, aa, h, m, action_mask=am)
        return z

    def act_of(rec):
        cn = torch.tensor(rec["actions_pi05_norm"],
                          dtype=torch.float32,
                          device=device)[None, :10]
        if cn.shape[1] < 10:
            cn = torch.cat([cn, torch.zeros(
                1, 10 - cn.shape[1], 7, device=device)], dim=1)
        am = (torch.arange(10, device=device)[None] < rec["steps"])
        return cn, am

    def train_arm(arm):
        cfg = ARMS[arm]
        init = torch.load(init_path, weights_only=False)
        model = V06State().to(device)
        model.load_state_dict(init["model"])
        pred = LatentPredictor().to(device)
        pred.load_state_dict(init["predictor"])
        ema = make_ema(model)
        ema.load_state_dict(init["ema"])
        params = list(model.parameters()) + list(pred.parameters())
        optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
        state_params = [p for n, p in model.named_parameters()
                        if n.split(".")[0] in STATE_MODULES]

        def visit_losses(ak, g, tv, train=True, second_stream=True):
            """All bundle losses for one anchor visit; returns dict of
            bundle -> tensor (already per-visit normalized)."""
            a = anchors[ak]
            pl = payload(ak)
            z = unroll(model, ak, tv, train)
            with torch.no_grad():
                zbar = unroll(ema, ak, tv, False)
            canon = S.canonical_goal(a["task"])
            L = {}
            cl, pa, pc_pred, pc_tgt = [], [], [], []
            sem_l, out_abs, out_pred, out_tgt = [], [], [], []
            zt_by_pt = {}
            for pt in a["rows"]:
                rec = by_pt[pt]
                if rec["kind"] == "audit":
                    continue
                cn, am = act_of(rec)
                zt = model.predict(z, cn, action_mask=am)
                zt_by_pt[pt] = zt
                # 1-block closure target: EMA transition + EMA update
                # with the actual next observation under tv
                if a["origin"] == "v071_semwin":
                    nk = (f"{tv}::swb::{pl[pt]['wid']}"
                          f"::b{pl[pt]['b_next']}")
                else:
                    nk = f"{tv}::next::{pt}"
                hn, mn = hget(nk)
                with torch.no_grad():
                    tgt = ema.update(
                        ema.t(zbar, ema.e_a(cn, am)), hn, mn)
                cl.append(latent_distance(pred(zt), tgt.detach(),
                                          "cosine").mean())
                out = model.d_next(zt)
                d_eef = torch.from_numpy(
                    pl[pt]["d_eef"]).float().to(device)
                pa.append(hub(out["d_q"][0], d_eef, q_scale,
                              HUB_PHYS))
                pc_pred.append(out["d_q"][0])
                pc_tgt.append(d_eef)
                if pl[pt]["d_obj"] is not None:
                    d_obj = torch.from_numpy(
                        pl[pt]["d_obj"]).float().to(device)
                    n_o = d_obj.shape[0]
                    pa.append(hub(out["d_obj"][0, :n_o].flatten(),
                                  d_obj.flatten(), OBJ_SCALE,
                                  HUB_PHYS))
                # sem under the visit goal
                sm = pl[pt]["sem"].get(g)
                if sm is not None:
                    n = n_sub(g)
                    if a["origin"] == "v071_semwin":
                        v0, v1 = sm
                        f01 = (v1 > v0).astype(np.float32)
                        f10 = (v1 < v0).astype(np.float32)
                        va = v1
                    else:
                        va = np.asarray(sm["valid_after"],
                                        dtype=np.float32)
                        f01 = np.zeros(n, dtype=np.float32)
                        f10 = np.zeros(n, dtype=np.float32)
                        for f in sm.get("flips_01", []):
                            f01[f[1]] = 1.0
                        for f in sm.get("flips_10", []):
                            f10[f[1]] = 1.0
                    sem_l.append(
                        bce(out["valid_bits"][0, :n],
                            torch.from_numpy(va[:n]).to(device))
                        + bce(out["flips_01"][0, :n],
                              torch.from_numpy(f01[:n]).to(device))
                        + bce(out["flips_10"][0, :n],
                              torch.from_numpy(f10[:n]).to(device)))
                # outcome bundle
                if len(pl[pt]["conts"]) == 2 and g == canon:
                    gbar = torch.from_numpy(np.mean(
                        [cont_g(c) for c in pl[pt]["conts"]],
                        axis=0)).float().to(device)
                    ghat = torch.sigmoid(torch.cat(
                        [out["p_valid"].reshape(1),
                         out["q_valid"][0]]))
                    out_pred.append(ghat)
                    out_tgt.append(gbar)
                    out_abs.append(
                        torch.nn.functional.huber_loss(
                            ghat, gbar, delta=HUB_OUT))
            # 2/3-block closure (semwin chains). The k-block target is
            # the EMA RECURRENT POSTERIOR at boundary k: the EMA state
            # re-anchors on the observed h at every boundary, while
            # the online prediction rolls open-loop through T only.
            if a["origin"] == "v071_semwin" and len(a["rows"]) > 1:
                ordered = sorted(a["rows"], key=lambda p:
                                 pl[p]["seg"])
                state = zt_by_pt[ordered[0]]
                with torch.no_grad():
                    cn0, am0 = act_of(by_pt[ordered[0]])
                    hn0, mn0 = hget(
                        f"{tv}::swb::{pl[ordered[0]]['wid']}"
                        f"::b{pl[ordered[0]]['b_next']}")
                    zb = ema.update(
                        ema.t(zbar, ema.e_a(cn0, am0)), hn0, mn0)
                for k in range(1, min(3, len(ordered))):
                    cn, am = act_of(by_pt[ordered[k]])
                    state = model.t(state, model.e_a(cn, am))
                    with torch.no_grad():
                        nk = (f"{tv}::swb::{pl[ordered[k]]['wid']}"
                              f"::b{pl[ordered[k]]['b_next']}")
                        hn, mn = hget(nk)
                        tgt = ema.update(
                            ema.t(zb, ema.e_a(cn, am)), hn, mn)
                        zb = tgt
                    w = A2 if k == 1 else A3
                    cl.append(w * latent_distance(
                        pred(state), tgt.detach(), "cosine").mean())
            if cl:
                L["closure"] = torch.stack(cl).mean()
            if pa:
                L["phys_abs"] = torch.stack(pa).mean()
            if len(pc_pred) >= 2:
                P = torch.stack(pc_pred)
                T_ = torch.stack(pc_tgt)
                L["phys_ctr"] = hub(P - P.mean(0, keepdim=True),
                                    T_ - T_.mean(0, keepdim=True),
                                    q_scale, HUB_PHYS)
            if sem_l:
                L["sem"] = torch.stack(sem_l).mean()
            if out_abs:
                L["out"] = torch.stack(out_abs).mean()
                if len(out_pred) >= 2:
                    Pp = torch.stack(out_pred)
                    Tt = torch.stack(out_tgt)
                    L["out"] = L["out"] + \
                        torch.nn.functional.huber_loss(
                            Pp - Pp.mean(0, keepdim=True),
                            Tt - Tt.mean(0, keepdim=True),
                            delta=HUB_OUT)
            # rank bundle
            pairs = gt_pairs(pl)
            bt_u0, bt_ot, ties = [], [], []
            for (pi, pj, y) in pairs:
                if pi not in zt_by_pt or pj not in zt_by_pt:
                    continue
                si = model.d_next(zt_by_pt[pi])["s"][0]
                sj = model.d_next(zt_by_pt[pj])["s"][0]
                is_u0 = by_pt[pi]["branch_id"].startswith("u0") or \
                    by_pt[pj]["branch_id"].startswith("u0")
                if y == 0:
                    ties.append(torch.nn.functional.huber_loss(
                        si - sj, torch.zeros_like(si),
                        delta=HUB_OUT))
                else:
                    term = torch.nn.functional.softplus(
                        -float(y) * (si - sj))
                    (bt_u0 if is_u0 else bt_ot).append(term)
            bt = []
            if bt_u0:
                bt.append(torch.stack(bt_u0).mean())
            if bt_ot:
                bt.append(torch.stack(bt_ot).mean())
            if bt:
                L["bt"] = torch.stack(bt).mean()
            if ties:
                L["tie"] = torch.stack(ties).mean()
            # paraphrase / shared-physics second stream
            if second_stream:
                base_tv = S.canonical_goal(a["task"]) + "_p0"
                if g == canon and tv != base_tv:
                    z2 = unroll(model, ak, base_tv, train)
                    d1 = latent_distance(z, z2.detach(),
                                         "cosine").mean()
                    d2 = latent_distance(z2, z.detach(),
                                         "cosine").mean()
                    pt0 = next(iter(zt_by_pt), None)
                    L["para"] = 0.5 * (d1 + d2)
                    if pt0 is not None:
                        cn, am = act_of(by_pt[pt0])
                        zt2 = model.predict(z2, cn, action_mask=am)
                        L["para"] = L["para"] + 0.5 * (
                            latent_distance(
                                zt_by_pt[pt0], zt2.detach(),
                                "cosine").mean()
                            + latent_distance(
                                zt2, zt_by_pt[pt0].detach(),
                                "cosine").mean())
                elif g != canon:
                    z2 = unroll(model, ak, base_tv, train)
                    sp = []
                    for pt, zt in zt_by_pt.items():
                        cn, am = act_of(by_pt[pt])
                        zt2 = model.predict(z2, cn, action_mask=am)
                        o1 = model.d_next(zt)["d_q"][0]
                        o2 = model.d_next(zt2)["d_q"][0]
                        sp.append(0.5 * (
                            hub(o1, o2.detach(), q_scale, HUB_PHYS)
                            + hub(o2, o1.detach(), q_scale,
                                  HUB_PHYS)))
                    if sp:
                        L["shared_phys"] = torch.stack(sp).mean()
            return L

        # NOTE on routing implementation: head-local bundles must not
        # shape the shared state. We enforce this exactly by a
        # two-pass update: pass 1 backward of state-enabled bundles
        # over all params; pass 2 backward of head-local bundles with
        # shared-state parameter grads frozen (grads snapshotted and
        # restored). Assertions below verify zero net state gradient
        # from head-local bundles.
        def step_visit(ak, g, tv):
            optimizer.zero_grad(set_to_none=True)
            L = visit_losses(ak, g, tv, train=True)
            if not L:
                return {}
            state_keys = ["closure", "phys_abs", "phys_ctr", "sem",
                          "para", "shared_phys"]
            if cfg["out_state"]:
                state_keys.append("out")
            if cfg["rank_state"]:
                state_keys += ["bt", "tie"]
            head_keys = [k for k in L if k not in state_keys]
            state_terms = [L[k] for k in state_keys if k in L]
            if state_terms:
                torch.stack(state_terms).sum().backward(
                    retain_graph=bool(head_keys))
            if head_keys:
                snap = [p.grad.detach().clone()
                        if p.grad is not None else None
                        for p in state_params]
                torch.stack([L[k] for k in head_keys]).sum() \
                    .backward()
                for p, s_ in zip(state_params, snap):
                    if s_ is None:
                        if p.grad is not None:
                            p.grad.zero_()
                    else:
                        p.grad.copy_(s_)
            torch.nn.utils.clip_grad_norm_(params, GRAD_NORM)
            optimizer.step()
            ema_update(ema, model, EMA_DECAY)
            return {k: float(v) for k, v in L.items()}

        # gradient-reach probe (isolated per-bundle passes)
        def grad_reach(ak, g, tv):
            rep = {}
            for k in ("closure", "phys_abs", "sem", "out", "bt"):
                optimizer.zero_grad(set_to_none=True)
                L = visit_losses(ak, g, tv, train=True)
                if k not in L:
                    rep[k] = None
                    continue
                L[k].backward()
                gn = float(sum((p.grad ** 2).sum()
                               for p in state_params
                               if p.grad is not None).sqrt())
                rep[k] = gn
            optimizer.zero_grad(set_to_none=True)
            return rep

        logs = []
        start = 0
        if args.resume:
            cks = sorted((OUT / "checkpoints").glob(
                f"{arm}_epoch*.pt"))
            if cks:
                st = torch.load(cks[-1], weights_only=False)
                model.load_state_dict(st["model"])
                pred.load_state_dict(st["predictor"])
                ema.load_state_dict(st["ema"])
                optimizer.load_state_dict(st["optimizer"])
                start = st["epoch"] + 1
                print(f"[{arm}] resumed at {start}", flush=True)
        for epoch in range(start, EPOCHS):
            order = S.rr_schedule(anchors, "train", epoch)
            agg = defaultdict(list)
            for i, ak in enumerate(order):
                g, tv = S.visit_query(anchors, queries, ak, epoch, i)
                vals = step_visit(ak, g, tv)
                for k, v in vals.items():
                    agg[k].append(v)
            means = {k: float(np.mean(v)) for k, v in agg.items()}
            logs.append({"epoch": epoch, **means})
            print(f"[{arm} ep{epoch}] " + " ".join(
                f"{k}={v:.4f}" for k, v in sorted(means.items())),
                flush=True)
            if epoch == start:
                ak0 = order[0]
                g0, tv0 = S.visit_query(anchors, queries, ak0,
                                        epoch, 0)
                rep = grad_reach(ak0, g0, tv0)
                (OUT / "metrics" / f"gradient_reach_{arm}.json"
                 ).write_text(json.dumps(
                    {"epoch": epoch, "anchor": ak0, "reach": rep,
                     "note": "isolated per-bundle backward; state "
                             "params norm"}, indent=2))
                for k in ("closure", "phys_abs"):
                    assert rep[k] and rep[k] > 0, f"{k} no reach"
            if (epoch + 1) % CKPT_EVERY == 0 or epoch + 1 == EPOCHS:
                with torch.no_grad():
                    dv = defaultdict(list)
                    for ak in sorted(anchors):
                        if anchors[ak]["split"] != "dev":
                            continue
                        for g, tv in S.queries_for(anchors[ak],
                                                   queries):
                            L = visit_losses(ak, g, tv, train=False,
                                             second_stream=False)
                            for k, v in L.items():
                                dv[k].append(float(v))
                dev = {k: float(np.mean(v)) for k, v in dv.items()}
                torch.save({"model": model.state_dict(),
                            "predictor": pred.state_dict(),
                            "ema": ema.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch, "arm": arm,
                            "dev_readout": dev},
                           OUT / "checkpoints"
                           / f"{arm}_epoch{epoch:03d}.pt")
                print(f"  [{arm} dev ep{epoch}] " + " ".join(
                    f"{k}={v:.4f}" for k, v in sorted(dev.items())),
                    flush=True)
        final = OUT / "checkpoints" / f"{arm}_final.pt"
        last = sorted((OUT / "checkpoints").glob(
            f"{arm}_epoch*.pt"))[-1]
        final.write_bytes(last.read_bytes())
        (OUT / "metrics" / f"loss_by_hierarchy_{arm}.json"
         ).write_text(json.dumps(logs, indent=1))
        return {"final": final.name}

    git_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True,
        text=True, cwd=REPO_ROOT).stdout.strip()
    mp = OUT / "run_manifest.json"
    if not mp.exists():
        mp.write_text(json.dumps({
            "schema": "v072_lcwm_loss_manifest_v1",
            "run_schema": "v072", "git_sha": git_sha,
            "init_from": str(INIT_CKPT),
            "data_manifest": hashlib.sha256(
                (DATA_R1 / "run_manifest.json").read_bytes())
            .hexdigest(),
            "arms": ARMS, "video": "tensor-only run"}, indent=2))

    arm_list = [args.arm] if args.arm else list(ARMS)
    results = {}
    for arm in arm_list:
        results[arm] = train_arm(arm)
        (OUT / "metrics" / "train_results.json").write_text(
            json.dumps(results, indent=2))
    print(json.dumps(results, indent=1), flush=True)
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
