#!/usr/bin/env python
"""V7.4C — train ONE balanced LC predictive state (2026-08-09.md).

Data: merged five-universe schedule (lcwm/v074_schedule.py): the
V7.3 merged universe (v072B 96 + v072T 18 + v073 33) + the V7.4B
acquisition tranche (v074; SHORTFALL / family_shortfall rows are
masked cells, never anchors) + the 9 released V7.3 teacher-assessment
anchors (v073T_released, canonical-p0 only). Init: the V7.2
byte-identical shared initialization (never a V7.2 final arm).

Registered changes vs scripts/train_v073_lcwm.py — exactly TWO,
and NOTHING else:
  1. task-balanced exposure: rr_schedule_balanced wraps short task
     queues to equal per-task visit mass per epoch (V7.3C's
     700/725/550/425/625 imbalance repaired); query rotation kept;
  2. per-epoch ONLINE VALIDATION READOUTS on the dev split, logged
     to metrics/online_readouts.jsonl, REPORT-ONLY (final-step
     checkpoint rule stands): action-effect discrimination vs
     copy/no-action/stock-action, same-goal paraphrase invariance,
     different-goal separation, action x goal reversal accuracy
     where unmasked, sibling preference where eligible pairs exist,
     and the pre-C-amendment history_dependence readout (dev-split
     real-vs-reset token and pool distances). Every readout names
     its support; empty support prints `unsupported`, never a
     silent zero.

Terminal-success mask: the parent frozen SUCCESS_VARIANCE_TASKS
constant, unchanged. The v074 outcome_support registered_task_checks
are read only as a loud consistency assert against that constant —
a contradiction requires a registration amendment, not a code path.

Objective, gradient-balance calibration (split-half medians), AdamW
3e-4/1e-4, 25 epochs, clip 1.0, TBPTT 16, EMA 0.995, seed 0,
final-step checkpoint ONLY: inherited unchanged. One (goal, text)
query per visit, rotation (epoch + anchor hash). Tensor-only.
Output: results/libero_loho_public_v1/<date>_v074_lcwm_r1/
(hcache_index.json from scripts/build_v074_hcache.py).
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
from lcwm import v074_schedule as S  # noqa: E402
from lcwm.self_predict import LatentPredictor, latent_distance  # noqa: E402
from lcwm.v06_model import V06State, ema_update, make_ema  # noqa: E402
from lcwm.v067_lineage import sha256_file  # noqa: E402

RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
V72DATA = RESULTS / "2026-08-03_v072_data_r1"
V72LOSS = RESULTS / "2026-08-03_v072_lcwm_loss_r1"
RID = "v074_lcwm_r1"
LR, WD, EPOCHS, GRAD_NORM, TBPTT = 3e-4, 1e-4, 25, 1.0, 16
EMA_DECAY = 0.995
Q_SCALE = torch.tensor([7.344e-3] * 3 + [5.451e-3] * 4
                       + [2.598e-4] * 2)
OBJ_SCALE = 1.175e-3
HUB_PHYS, HUB_OUT = 4.0, 0.1
CKPT_EVERY = 5
STATE_MODULES = ("e_a", "t", "anchor", "r", "norm", "z0")
# parent frozen constant (train_v073_lcwm.py); v074 outcome_support
# is read below only as a consistency assert, never as a derivation
SUCCESS_VARIANCE_TASKS = {"loho_t2_basket3"}
# frozen replay-noise q tolerance: the acquisition builders'
# both-repeats constant (build_v073_data / build_v074_data)
TOL_QM = 0.0417


def hub(p, t, scale, delta):
    return torch.nn.functional.huber_loss(p / scale, t / scale,
                                          delta=delta)


def run_date() -> str:
    return subprocess.run(["date", "+%F"], capture_output=True,
                          text=True,
                          env={"TZ": "America/Chicago"}
                          ).stdout.strip()


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda")
    torch.manual_seed(0)
    OUT = None
    for p in sorted(RESULTS.glob(f"*_{RID}")):
        OUT = p
    OUT = OUT or RESULTS / f"{run_date()}_{RID}"
    for sub in ("checkpoints", "metrics"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    bce = torch.nn.functional.binary_cross_entropy_with_logits

    anchors, tq = S.load_universe()
    V74DATA = S.v074_root()
    hcache_path = OUT / "hcache_index.json"
    assert hcache_path.exists(), \
        f"{hcache_path} missing — run build_v074_hcache.py first"
    hidx = json.loads(hcache_path.read_text())
    goal_manifest = json.loads(
        (RESULTS / "goal_spec_manifest_v067.json").read_text())

    # ---- terminal-success mask: parent frozen constant --------------
    osup = json.loads(
        (V74DATA / "outcome_support.json").read_text())
    measured = {t for t, c in
                osup["registered_task_checks"].items()
                if c["milestone_pos_recovery_terminal_train"]}
    if measured != SUCCESS_VARIANCE_TASKS:
        raise RuntimeError(
            f"v074 measured terminal-success support "
            f"{sorted(measured)} contradicts the frozen "
            f"SUCCESS_VARIANCE_TASKS "
            f"{sorted(SUCCESS_VARIANCE_TASKS)}; changing the mask "
            f"requires a registration amendment in "
            f"plan_and_progress/2026-08-09.md, not a code edit")
    success_tasks = SUCCESS_VARIANCE_TASKS

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

    # released assessment candidate chunks (one small keyed store)
    chunks73T = {}

    def chunk73T(key):
        if not chunks73T:
            chunks73T.update(torch.load(
                S.V73TEACH / "candidate_chunks.pt",
                weights_only=False))
        return chunks73T[key]

    # ---- per-anchor payload adapters --------------------------------
    payload_lru = {}

    def payload(ak):
        """rows: pt -> {a_norm(list or None), a_env, steps, d_eef,
        d_obj, sem{gid:(v0,v1,f01,f10)}, cont{gid: gbar(5)},
        cont_reps{gid: [rep rows]} (v073/v074 only),
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
        elif a["universe"] == "v073T_released":
            # released V7.3 assessment shards: candidate_id-keyed
            # transitions, valid_seq under BOTH goals, canonical
            # continuations, NO obj_after (d_obj unsupported),
            # exact chunks from candidate_chunks.pt
            sid, d = a["source_id"], a["decision"]
            s = shardU("v073T_released", f"{sid}_d{d}")
            cg = S.canon_goal(a["task"])
            for tr in s["transitions"]:
                pt = f"v073T_released::{tr['transition_id']}"
                vs = tr["valid_seq_canon"]
                v0 = np.asarray(vs[0], dtype=bool)
                v1 = np.asarray(vs[-1], dtype=bool)
                sem = {cg: (v0, v1,
                            (v1 & ~v0).astype(np.float32),
                            (~v1 & v0).astype(np.float32))}
                for gid, avs in tr["valid_seq_alt"].items():
                    av0 = np.asarray(avs[0], dtype=bool)
                    av1 = np.asarray(avs[-1], dtype=bool)
                    sem[gid] = (av0, av1,
                                (av1 & ~av0).astype(np.float32),
                                (~av1 & av0).astype(np.float32))
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
                    "a_norm": chunk73T(tr["transition_id"]),
                    "steps": tr["steps"],
                    "d_eef": (np.asarray(tr["eef_seq"][-1])
                              - np.asarray(tr["eef_seq"][0])),
                    "d_obj": None,
                    "sem": sem, "cont": cont,
                    "recterm_success": None,
                    "next_key": f"::next::{pt}"}
        else:   # v073 / v074 — identical shard layout, parametrized
            u = a["universe"]
            sid, d = a["source_id"], a["decision"]
            s = shardU(u, f"{sid}_d{d}")
            obj_before = np.asarray(s["obj_before"])
            for tr in s["transitions"]:
                if tr["branch_key"] == "u0_repeat":
                    continue
                pt = f"{u}::{tr['transition_id']}"
                sem = {}
                for gid, vs in tr["valid_seq"].items():
                    v0 = np.asarray(vs[0], dtype=bool)
                    v1 = np.asarray(vs[-1], dtype=bool)
                    sem[gid] = (v0, v1,
                                (v1 & ~v0).astype(np.float32),
                                (~v1 & v0).astype(np.float32))
                cont, cont_reps = {}, {}
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
                        cont_reps[gid] = gs
                rts = None
                if tr["recovery_terminal"] is not None and \
                        a["task"] in success_tasks:
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
                    "cont_reps": cont_reps,
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
        rows_ = (src72(sid)["rows"] if a["universe"] == "v072B"
                 else srcU(a["universe"], sid)["rows"])
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

    # ---- online validation readouts (report-only; registered) ------
    dev_aks = [ak for ak in sorted(anchors)
               if anchors[ak]["split"] == "dev"]

    def rep_pref(ga, gb):
        """Both-repeats-agree preference on per-rep [p, q@h..] rows
        (the registered V7.4B rule); 0 = tie/unstable."""
        signs = []
        for ra, rb in zip(ga, gb):
            qa, qb = float(np.mean(ra[1:])), float(np.mean(rb[1:]))
            if ra[0] > rb[0]:
                signs.append(1)
            elif rb[0] > ra[0]:
                signs.append(-1)
            elif qa > qb + TOL_QM:
                signs.append(1)
            elif qb > qa + TOL_QM:
                signs.append(-1)
            else:
                signs.append(0)
        if signs and all(s_ == 1 for s_ in signs):
            return 1
        if signs and all(s_ == -1 for s_ in signs):
            return -1
        return 0

    def pred_sign(va, vb):
        """Predicted preference: mean of the sigmoid [p, q@h..]
        vector (monotone scalarization; exact tie -> 0)."""
        da = float(va.mean() - vb.mean())
        return 0 if da == 0 else (1 if da > 0 else -1)

    def stat(vals):
        return ({"n": len(vals), "value": float(np.mean(vals))}
                if vals else {"n": 0, "value": "unsupported"})

    def med(vals):
        return ({"n": len(vals), "median": float(np.median(vals))}
                if vals else {"n": 0, "median": "unsupported"})

    def quart(vals):
        if not vals:
            return {"n": 0, "median": "unsupported"}
        q25, q50, q75 = (float(q) for q in
                         np.percentile(vals, (25, 50, 75)))
        return {"n": len(vals), "q25": q25, "median": q50,
                "q75": q75}

    @torch.no_grad()
    def online_readouts(epoch):
        disc = {"copy": [], "no_action": [], "stock": []}
        errs = {"true": [], "copy": [], "no_action": [],
                "stock": []}
        para_d, sep_d, rev_rows, sib_rows = [], [], [], []
        hd_tok, hd_pool = [], []
        ae_semwin_excl = 0
        for ak in dev_aks:
            a = anchors[ak]
            cg = S.canon_goal(a["task"])
            cp0 = f"{cg}_p0"
            pl = payload(ak)
            z = unroll(model, ak, cp0, False)
            semwin = (a["universe"] == "v072B"
                      and a["origin"] == "v071_semwin")
            # history_dependence (pre-C amendment): real recurrent
            # state vs reset state at the anchor observation
            sid, d = a["source_id"], a["decision"]
            if semwin:
                wid = by_pt[a["rows"][0]]["raw_key"].split("|")[0]
                hr, mr = hget(f"{cp0}::swb::{wid}::b0")
            else:
                hr, mr = hget(f"{cp0}::src::{sid}::d{d}")
            z_reset = model.initial_state(hr, mr)
            hd_tok.append(float((z - z_reset).norm()))
            hd_pool.append(float((z.mean(dim=1)
                                  - z_reset.mean(dim=1)).norm()))
            # (a) action-effect discrimination vs baselines.
            # semwin anchors excluded: their k>0 segment rows are
            # predicted from a state that has not advanced past the
            # earlier segments (count recorded in the support census)
            if semwin:
                ae_semwin_excl += 1
                u0row = None
                ae_rows = []
            else:
                u0row = next((r for pt, r in pl.items()
                              if pt.endswith("_u0")), None)
                ae_rows = list(pl.items())
            for pt, row in ae_rows:
                d_eef = torch.from_numpy(np.asarray(
                    row["d_eef"])).float().to(device)

                def dq_err(cn, am):
                    o = model.d_next(model.predict(
                        z, cn, action_mask=am))
                    return float(((o["d_q"][0] - d_eef)
                                  / q_scale).abs().mean())

                cn, am = a_norm_of(row)
                e_true = dq_err(cn, am)
                e_copy = float((d_eef / q_scale).abs().mean())
                a0, m0 = model.null_action(1, device)
                e_na = dq_err(a0, m0)
                errs["true"].append(e_true)
                errs["copy"].append(e_copy)
                errs["no_action"].append(e_na)
                disc["copy"].append(e_true < e_copy)
                disc["no_action"].append(e_true < e_na)
                if u0row is not None and not pt.endswith("_u0"):
                    cn0, am0 = a_norm_of(u0row)
                    e_st = dq_err(cn0, am0)
                    errs["stock"].append(e_st)
                    disc["stock"].append(e_true < e_st)
            # (b)/(c) paraphrase invariance + goal separation on
            # pooled states, canonical-p0 base. Separation uses the
            # task's registered COMPATIBLE alt goal only — v072B
            # cross-task texts (archived-vacuous t1<->t5 crossing)
            # must not enter the metric
            pairs = S.queries_for(a, tq)
            alt_g = S.ALT_GOAL[a["task"]]
            if (cg, cp0) in pairs:
                p0 = z.mean(dim=1)
                for g, tvv in pairs:
                    if (g, tvv) == (cg, cp0):
                        continue
                    same = g == cg
                    if not same and (g != alt_g
                                     or tvv != f"{g}_p0"):
                        continue
                    pz = unroll(model, ak, tvv,
                                False).mean(dim=1)
                    if same:
                        para_d.append(float((pz - p0).norm()))
                    else:
                        sep_d.append(float((pz - p0).norm()))
            # (d)/(e) reversal + sibling preference (per-rep GT
            # exists only for the v073/v074 shard layout)
            if a["universe"] not in ("v073", "v074"):
                continue
            alt = S.ALT_GOAL[a["task"]]
            rows_l = [(pt, r) for pt, r in pl.items()
                      if r.get("cont_reps", {}).get(cg)]
            if not rows_l:
                continue
            z_a = unroll(model, ak, f"{alt}_p0", False)

            def ghat_of(zg, row):
                cn, am = a_norm_of(row)
                o = model.d_next(model.predict(
                    zg, cn, action_mask=am))
                return torch.sigmoid(torch.cat(
                    [o["p_valid"].reshape(1), o["q_valid"][0]]))

            gh_c = {pt: ghat_of(z, r) for pt, r in rows_l}
            gh_a = {pt: ghat_of(z_a, r) for pt, r in rows_l}
            for i in range(len(rows_l)):
                for j in range(i + 1, len(rows_l)):
                    (pi, ri), (pj, rj) = rows_l[i], rows_l[j]
                    sc = rep_pref(ri["cont_reps"][cg],
                                  rj["cont_reps"][cg])
                    pos_i = bool(ri["sem"][cg][2].any())
                    pos_j = bool(rj["sem"][cg][2].any())
                    if sc != 0 and pos_i != pos_j:
                        sib_rows.append(
                            pred_sign(gh_c[pi], gh_c[pj]) == sc)
                    ra_ = ri["cont_reps"].get(alt)
                    rb_ = rj["cont_reps"].get(alt)
                    if ra_ and rb_:
                        sa = rep_pref(ra_, rb_)
                        if sc != 0 and sa != 0 and sc != sa:
                            rev_rows.append(
                                pred_sign(gh_c[pi],
                                          gh_c[pj]) == sc
                                and pred_sign(gh_a[pi],
                                              gh_a[pj]) == sa)
        para_m, sep_m = med(para_d), med(sep_d)
        ratio = (sep_m["median"] / max(para_m["median"], 1e-12)
                 if para_d and sep_d else "unsupported")
        row = {"epoch": epoch,
               "action_effect_disc": {
                   "vs_copy": stat(disc["copy"]),
                   "vs_no_action": stat(disc["no_action"]),
                   "vs_stock_action": stat(disc["stock"]),
                   "err_medians": {k: med(v) for k, v in
                                   errs.items()}},
               "paraphrase_invariance_pooled_dist": para_m,
               "goal_separation_pooled_dist": {
                   **sep_m, "ratio_to_paraphrase": ratio},
               "reversal_accuracy": stat(rev_rows),
               "sibling_preference": stat(sib_rows),
               "history_dependence": {
                   "token_dist": quart(hd_tok),
                   "pool_dist": quart(hd_pool)},
               "rules": {"gt": "both-repeats p->q with TOL_QM",
                         "pred": "mean sigmoid [p,q@h] vector"}}
        support = {
            "action_effect_rows": len(disc["copy"]),
            "action_effect_excluded_semwin_multiseg_anchors":
                ae_semwin_excl,
            "stock_action_rows": len(disc["stock"]),
            "paraphrase_pairs": len(para_d),
            "goal_separation_pairs": len(sep_d),
            "reversal_pairs": len(rev_rows),
            "sibling_pairs": len(sib_rows),
            "history_dependence_anchors": len(hd_tok),
            "dev_anchors": len(dev_aks)}

        def show(s):
            v = s.get("value", s.get("median"))
            return (f"{v:.3f}(n={s['n']})" if s["n"]
                    else "unsupported")

        print(f"  [v074 ol ep{epoch}] "
              f"ae_copy={show(stat(disc['copy']))} "
              f"ae_noact={show(stat(disc['no_action']))} "
              f"ae_stock={show(stat(disc['stock']))} "
              f"para={show(para_m)} sep={show(sep_m)} "
              f"rev={show(stat(rev_rows))} "
              f"sib={show(stat(sib_rows))} "
              f"hist_tok={show(quart(hd_tok))} "
              f"hist_pool={show(quart(hd_pool))}", flush=True)
        return row, support

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
        # SPLIT-HALF validation (amended binding, inherited from
        # V7.3C verbatim): lambdas from the even-indexed half's
        # medians; the 3x check runs on the odd-indexed half's
        # SCALED MEDIANS (held-out, median-consistent)
        half_a = {k: v[0::2] for k, v in norms.items()}
        half_b = {k: v[1::2] for k, v in norms.items()}
        med_n = {k: float(np.median(v)) for k, v in half_a.items()}
        g_geo = float(np.exp(np.mean([np.log(v)
                                      for v in med_n.values()])))
        lam = {k: g_geo / (med_n[k] + 1e-8) for k in med_n}
        scaled_holdout = {k: lam[k] * float(np.median(half_b[k]))
                          for k in med_n}
        mx, mn = max(scaled_holdout.values()), min(
            scaled_holdout.values())
        assert mx / max(mn, 1e-12) < 3.0, \
            f"held-out scaled MEDIANS exceed 3x: {scaled_holdout}"
        lam_path.write_text(json.dumps({
            "lambda": lam, "median": med_n, "g_geo": g_geo,
            "scaled_holdout_medians": scaled_holdout,
            "microbatches": detail,
            "note": "split-half: lambda from even-half medians; 3x "
                    "check on odd-half scaled medians (held-out, "
                    "non-vacuous; V7.3C amended binding inherited)"},
            indent=2))
        print(f"[lam] {lam} scaled_holdout={scaled_holdout}",
              flush=True)

    git_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True,
        text=True, cwd=REPO_ROOT).stdout.strip()
    # registered manifest extras: universe census per task and the
    # epoch-0 balanced per-task visit mass (asserted equal)
    census = defaultdict(lambda: defaultdict(int))
    for ak in sorted(anchors):
        a = anchors[ak]
        census[a["universe"]][a["task"]] += 1
    order0 = S.rr_schedule_balanced(anchors, "train", 0)
    visits0 = defaultdict(int)
    for ak in order0:
        visits0[anchors[ak]["task"]] += 1
    assert len(set(visits0.values())) == 1, \
        f"epoch-0 schedule unbalanced: {dict(visits0)}"
    mp = OUT / "run_manifest.json"
    if not mp.exists():
        mp.write_text(json.dumps({
            "schema": "v074_lcwm_manifest_v1", "run_schema": "v074",
            "run_id": RID, "git_sha": git_sha,
            "init": str(V72LOSS / "shared_initialization.pt"),
            "data_root": str(V74DATA),
            "input_sha256": {
                "hcache_index.json": sha256_file(hcache_path),
                "shared_initialization.pt": sha256_file(
                    V72LOSS / "shared_initialization.pt"),
                "v074b_anchor_manifest.json": sha256_file(
                    V74DATA / "anchor_manifest.json")},
            "universe_census": {u: dict(v) for u, v
                                in sorted(census.items())},
            "balanced_epoch0_task_visits": dict(visits0),
            "optimizer": {"lr": LR, "wd": WD, "epochs": EPOCHS,
                          "clip": GRAD_NORM, "tbptt": TBPTT,
                          "ema": EMA_DECAY, "seed": 0,
                          "selection": "final step ONLY"},
            "masks": {"tau_next": 0, "damage": 0,
                      "terminal_success": sorted(success_tasks),
                      "terminal_success_rule":
                          "parent frozen SUCCESS_VARIANCE_TASKS; "
                          "v074 registered_task_checks asserted "
                          "consistent at startup",
                      "rank_family": "REMOVED (no scalar rank head)"},
            "online_readouts": "report-only, never checkpoint "
                               "selection; support census recorded "
                               "after first epoch",
            "lambda_sha": hashlib.sha256(
                lam_path.read_bytes()).hexdigest()[:16],
        }, indent=2))

    def record_support(support):
        m = json.loads(mp.read_text())
        if "online_readout_support" not in m:
            m["online_readout_support"] = support
            mp.write_text(json.dumps(m, indent=2))

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
    # restart-safe readout stream: epochs >= start will be re-run,
    # so their rows are dropped once here before any append
    ol_path = OUT / "metrics" / "online_readouts.jsonl"
    if ol_path.exists():
        kept = [ln for ln in ol_path.read_text().splitlines()
                if ln.strip() and json.loads(ln)["epoch"] < start]
        ol_path.write_text("".join(ln + "\n" for ln in kept))
    for epoch in range(start, EPOCHS):
        order = S.rr_schedule_balanced(anchors, "train", epoch)
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
        ol_row, ol_support = online_readouts(epoch)
        with ol_path.open("a") as f:
            f.write(json.dumps(ol_row) + "\n")
        record_support(ol_support)
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
