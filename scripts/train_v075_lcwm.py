#!/usr/bin/env python
"""V7.5C — train ONE LC predictive state with a LIVE recurrence.

Supersedes scripts/train_v074_lcwm.py before execution (plan_and_progress/archive/daily/2026-08-09.md,
"V7.5 registration"). V7.4C's two registered changes did not touch the
mechanism that made the V7.3C state a (current observation, language)
function, so running it would have spent a slice reproducing a known
result.

Universe (registered amendment): the v074 acquisition tranche ALONE --
24 train + 9 dev anchors over 22 sources, decision index 2..65. The
v072B/v072T/v073 tensors were deliberately deleted; v073T_released is
retired too because the recurrent unroll needs v073 `sources/`, which
do not survive (its shards carry transitions but no rows). The
physical head's support is materially thinner than V7.3C's and the
v071 semwin/replay transition diversity is gone -- carried verbatim
into every V7.5 result.

Init (registered amendment): the V7.2 byte-identical shared
initialization is unavailable (`shared_initialization.pt` deleted), and
the architecture changed anyway, so V7.5C initializes FRESH at seed 0
and records the resulting state hash in the run lineage. V7.5C trains
one model, so there are no arms to match at this stage; the matched
arms are V7.5D's four policies, which share stock pi0.5 and this
checkpoint.

Registered changes vs train_v074_lcwm.py -- exactly THREE:
  1. `lcwm/v075_state.py::V075State` -- Stage 0's two architecture
     repairs: `r_norm` desaturates the history bound (V7.3C ran tanh at
     100% saturation, max abs 28.77) and `hist_center` re-injects the
     history channel centered (V7.3C's attention saw near-identical
     keys, 4.8e-06 contrast). Plumbing-verified in Stage 1b.
  2. SHORT LATENT ROLLOUT (framework_design.md 3.2): from a state
     inside the final TBPTT window, predict ROLL_K steps ahead with NO
     intervening observation, against the observation-anchored states.
     The inherited `cl` term is 3.1 one-step closure on a detached
     cosine target, which any observation-anchored state satisfies
     trivially (V7.3C drove it to 1e-4). Rollout targets are compared
     in CENTERED units so the same degeneracy cannot make the new term
     vacuous. Starts are restricted to the final TBPTT window because
     that is where gradient can still reach the recurrent state.
  3. schedule + readouts: `lcwm/v075_schedule.py` rotates the query on
     wrapped repeats (t4 has ONE train anchor and would otherwise
     contribute six identical samples per epoch); per-epoch `sigma` is
     logged and asserted not to collapse; history-dependence is
     reported raw, centered, and RELATIVE, the last being the Stage 3
     binding quantity.

Per Stage 1b finding 1, the Stage 3 gate keys on the RESET contrast:
an open channel carrying nothing would pass a different-anchor
contrast. Per framework 3.3, rollout pressure is necessary but not
sufficient for history integration -- if the reset contrast does not
move, that is a clean registered negative about the objective, not a
silent limitation.

Terminal-success mask: the parent frozen SUCCESS_VARIANCE_TASKS
constant, unchanged; v074 outcome_support is read only as a loud
consistency assert.

Objective weights (split-half median calibration), AdamW 3e-4/1e-4,
25 epochs, clip 1.0, TBPTT 16, EMA 0.995, seed 0, final-step
checkpoint ONLY: inherited unchanged. Tensor-only.
Output: results/libero_loho_public_v1/<date>_v075_lcwm_r1/
(hcache_index.json from scripts/build_v075_hcache.py).
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

from lcwm import v075_schedule as S  # noqa: E402
from lcwm.self_predict import LatentPredictor, latent_distance  # noqa: E402
from lcwm.v06_model import ema_update, make_ema  # noqa: E402
from lcwm.v075_state import V075State  # noqa: E402
from lcwm.v067_lineage import sha256_file  # noqa: E402

RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
RID = "v075_lcwm_r1"
UNIVERSE = S.UNIVERSE
LR, WD, EPOCHS, GRAD_NORM, TBPTT = 3e-4, 1e-4, 25, 1.0, 16
EMA_DECAY = 0.995
Q_SCALE = torch.tensor([7.344e-3] * 3 + [5.451e-3] * 4
                       + [2.598e-4] * 2)
OBJ_SCALE = 1.175e-3
HUB_PHYS, HUB_OUT = 4.0, 0.1
CKPT_EVERY = 5
STATE_MODULES = ("e_a", "t", "anchor", "r", "norm", "z0",
                 "r_norm")
# registered short-rollout constants, frozen BEFORE training
ROLL_K = 3          # steps predicted with no new observation
ROLL_STARTS = 4     # start points per visit, evenly spaced
# Stage 3 binding threshold on the RELATIVE reset contrast
HIST_GATE = 1e-4
# sigma must not collapse below the Stage 1b measured value/10
SIGMA_FLOOR = 2.0e-5
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
        f"{hcache_path} missing — run build_v075_hcache.py first"
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
            f"plan_and_progress/archive/daily/2026-08-09.md, not a code edit")
    success_tasks = SUCCESS_VARIANCE_TASKS

    src_lru, shard_lru, h_lru = {}, {}, {}

    def srcU(universe, sid):
        key = (universe, sid)
        if key not in src_lru:
            if len(src_lru) > 6:
                src_lru.clear()
            src_lru[key] = torch.load(
                S.sources_for(universe) / f"{sid}.pt",
                weights_only=False)
        return src_lru[key]

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
        cont_reps{gid: [rep rows]} (v073/v074 only),
        recterm_success(float|None), next_key_suffix}."""
        if ak in payload_lru:
            return payload_lru[ak]
        if len(payload_lru) > 30:
            payload_lru.clear()
        a = anchors[ak]
        rows = {}
        if a["universe"] == UNIVERSE:
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
        else:
            raise RuntimeError(
                f"retired universe {a['universe']!r}: V7.5C "
                f"trains on the surviving v074 tranche alone")
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

    # registered fresh initialization (the V7.2 shared init is gone and
    # the architecture changed); seed 0 is set at entry, and the exact
    # initial weights are hashed into the run lineage below.
    model = V075State().to(device)
    pred = LatentPredictor().to(device)
    init_sha = hashlib.sha256(b"".join(
        v.detach().cpu().numpy().tobytes()
        for _, v in sorted(model.state_dict().items())
        if v.dtype.is_floating_point)).hexdigest()
    ema = make_ema(model)
    params = list(model.parameters()) + list(pred.parameters())
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    state_params = [p for n, p in model.named_parameters()
                    if n.split(".")[0] in STATE_MODULES]
    q_scale = Q_SCALE.to(device)

    def unroll(model_, ak, tvv, train, collect=False):
        """Recurrent state at the anchor. With collect=True also
        returns the per-decision state trace and the action that
        advanced each step, for the registered short rollout."""
        a = anchors[ak]
        sid, d = a["source_id"], a["decision"]
        z = None
        trace, acts = [], []
        rows_ = srcU(a["universe"], sid)["rows"]
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
            if collect:
                trace.append(z)
                if dd < d - 1:
                    nxt = rows_[dd]
                    acts.append((
                        nxt["chunk_norm"][None, :10].float().to(device),
                        (torch.arange(10, device=device)[None]
                         < nxt["executed_len"])))
        h, m = hget(f"{tvv}::src::{sid}::d{d}")
        if z is None:
            z = model_.initial_state(h, m)
        else:
            prev = rows_[d - 1]
            aa = prev["chunk_norm"][None, :10].float().to(device)
            am = (torch.arange(10, device=device)[None]
                  < prev["executed_len"])
            z = model_.step(z, aa, h, m, action_mask=am)
        if collect:
            if trace:
                acts.append((aa, am))
            trace.append(z)
            return z, trace, acts
        return z

    def roll_losses(trace, acts):
        """framework_design.md 3.2 -- short latent rollout.

        From a start state, predict ROLL_K steps ahead by applying only
        T(., E_a(a)) with NO intervening observation, against the
        observation-anchored states at those decisions.

        Both sides have the history-channel MEAN subtracted first. The
        raw states are near-degenerate, so a cosine distance between
        them is ~0 for any pair and carries no signal -- that is what
        let the inherited one-step `cl` term reach 1e-4. Subtracting
        the common mode makes the cosine act on the residuals, which
        are what actually distinguish states. (`apply_frozen` also
        divides by sigma; cosine is scale-invariant, so that factor is
        inert here and only the mean subtraction does work.)

        Starts must be states that still carry a graph: past the TBPTT
        boundary the trace is detached, and a rollout from a detached
        start trains only T and the predictor while applying no
        pressure whatsoever to the recurrent state. The window bound is
        therefore backed by an explicit requires_grad filter -- taking
        the window alone would still land on the detach boundary itself
        whenever the anchor's decision index is a multiple of TBPTT.
        """
        n = len(trace)
        last = n - 1 - ROLL_K
        if last < 0 or not acts:
            return []
        first = max(0, min(last, n - 1 - TBPTT))
        live = [j for j in range(first, last + 1)
                if trace[j].requires_grad]
        if not live:
            return []
        if len(live) <= ROLL_STARTS:
            starts = live
        else:
            step = (len(live) - 1) / (ROLL_STARTS - 1)
            starts = sorted({live[min(len(live) - 1, round(i * step))]
                             for i in range(ROLL_STARTS)})
        hc = model.hist_center
        out = []
        for j in starts:
            zr = trace[j]
            for i in range(ROLL_K):
                if j + i >= len(acts):
                    break
                aa, am = acts[j + i]
                zr = model.t(zr, model.e_a(aa, am))
                tgt = trace[j + i + 1].detach()
                out.append(latent_distance(
                    hc.apply_frozen(pred(zr)),
                    hc.apply_frozen(tgt), "cosine").mean())
        return out

    def visit_losses(ak, g, tvv, train=True, second_stream=True):
        a = anchors[ak]
        pl = payload(ak)
        z, trace, acts = unroll(model, ak, tvv, train, collect=True)
        with torch.no_grad():
            zbar = unroll(ema, ak, tvv, False)
        canon = S.canon_goal(a["task"])
        L = defaultdict(list)
        if train:
            L["roll"].extend(roll_losses(trace, acts))
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
        # eval() is REQUIRED, not cosmetic: hist_center.observe() fires
        # on every V075State.update(), so running the dev readouts in
        # training mode would fold dev-split states into the centering
        # statistics the training objective is normalized by. eval()
        # makes hist_center.training False and the readouts pure.
        # (No dropout/batchnorm in this model, so eval() changes
        # nothing else.)
        was_training = model.training
        model.eval()
        try:
            return _online_readouts(epoch)
        finally:
            model.train(was_training)

    def _online_readouts(epoch):
        disc = {"copy": [], "no_action": [], "stock": []}
        errs = {"true": [], "copy": [], "no_action": [],
                "stock": []}
        para_d, sep_d, rev_rows, sib_rows = [], [], [], []
        hd_tok, hd_pool, hd_tok_c, hd_rel = [], [], [], []
        roll_true, roll_ctrl = [], []
        ae_semwin_excl = 0
        for ak in dev_aks:
            a = anchors[ak]
            cg = S.canon_goal(a["task"])
            cp0 = f"{cg}_p0"
            pl = payload(ak)
            z, trace, acts = unroll(model, ak, cp0, False,
                                    collect=True)
            # ROLLOUT DISCRIMINATION (report-only). The registered
            # rollout term is only meaningful if the k-step
            # no-observation prediction can tell its TRUE target
            # from a distant one. At init it cannot (measured in
            # scripts/test_v075_contract.py), so without this
            # readout a vacuous term would stay invisible until
            # the Stage 3 gate. No requires_grad filter here --
            # this runs under no_grad and is measuring, not
            # training.
            n_tr = len(trace)
            last_j = n_tr - 1 - ROLL_K
            if last_j >= 0 and acts:
                first_j = max(0, min(last_j, n_tr - 1 - TBPTT))
                cand = list(range(first_j, last_j + 1))
                if len(cand) > ROLL_STARTS:
                    stp = (len(cand) - 1) / (ROLL_STARTS - 1)
                    cand = sorted({
                        cand[min(len(cand) - 1, round(i * stp))]
                        for i in range(ROLL_STARTS)})
                hcz = model.hist_center
                for j_ in cand:
                    zr = trace[j_]
                    for i_ in range(ROLL_K):
                        if j_ + i_ >= len(acts):
                            break
                        aa_, am_ = acts[j_ + i_]
                        zr = model.t(zr, model.e_a(aa_, am_))
                        pz = hcz.apply_frozen(pred(zr))
                        roll_true.append(float(latent_distance(
                            pz, hcz.apply_frozen(trace[j_ + i_ + 1]),
                            "cosine").mean()))
                        far = (j_ + i_ + 1 + n_tr // 2) % n_tr
                        roll_ctrl.append(float(latent_distance(
                            pz, hcz.apply_frozen(trace[far]),
                            "cosine").mean()))
            semwin = False      # retired universe; kept for the census
            # history_dependence: real recurrent state vs reset state at
            # the SAME anchor observation. This is the Stage 3 binding
            # quantity (plan_and_progress/archive/daily/2026-08-09.md V7.5 Stage 1b finding 1): the
            # `gain_other` contrast can pass on an open channel carrying
            # nothing, the RESET contrast cannot. Reported raw and in
            # the centered units the state is actually consumed in.
            sid, d = a["source_id"], a["decision"]
            hr, mr = hget(f"{cp0}::src::{sid}::d{d}")
            z_reset = model.initial_state(hr, mr)
            hd_tok.append(float((z - z_reset).norm()))
            hd_pool.append(float((z.mean(dim=1)
                                  - z_reset.mean(dim=1)).norm()))
            hc = model.hist_center
            hd_tok_c.append(float(
                ((z - z_reset) / hc.sigma).norm()))
            hd_rel.append(float((z - z_reset).norm()
                                / z.norm().clamp_min(1e-12)))
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
                   "pool_dist": quart(hd_pool),
                   "token_dist_centered": quart(hd_tok_c),
                   "token_dist_relative": quart(hd_rel),
                   "gate_quantity": "token_dist_relative",
                   "gate_threshold": HIST_GATE},
               "rollout_discrimination": {
                   "true": stat(roll_true),
                   "shuffled": stat(roll_ctrl),
                   "margin": (float(np.mean(roll_ctrl)
                                    - np.mean(roll_true))
                              if roll_true else "unsupported"),
                   "rule": "shuffled MINUS true; <= 0 means the "
                           "registered rollout term is vacuous"},
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
            "rollout_terms": len(roll_true),
            "dev_anchors": len(dev_aks)}

        def show(s):
            v = s.get("value", s.get("median"))
            return (f"{v:.3f}(n={s['n']})" if s["n"]
                    else "unsupported")

        roll_txt = (
            f"{float(np.mean(roll_ctrl) - np.mean(roll_true)):+.5f}"
            f"(n={len(roll_true)})" if roll_true else "unsupported")

        print(f"  [v075 ol ep{epoch}] "
              f"ae_copy={show(stat(disc['copy']))} "
              f"ae_noact={show(stat(disc['no_action']))} "
              f"ae_stock={show(stat(disc['stock']))} "
              f"para={show(para_m)} sep={show(sep_m)} "
              f"rev={show(stat(rev_rows))} "
              f"sib={show(stat(sib_rows))} "
              f"hist_tok={show(quart(hd_tok))} "
              f"hist_rel={show(quart(hd_rel))} "
              f"roll_margin={roll_txt} "
              f"sigma={float(model.hist_center.sigma):.3e}",
              flush=True)
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
        # 4 microbatches per task, 2 with the language second stream
        # active and 2 without. The inherited selector required a
        # DISTINCT anchor per microbatch, which silently capped a task
        # at its anchor count: on this universe t4 has ONE train anchor
        # and the set came to 17, not 20 (and `inv` would then have
        # missed its >=10 quota). Distinct anchors are still preferred;
        # a task that cannot supply them falls back to distinct QUERIES
        # of an anchor already used. The per-task anchor diversity is
        # recorded in gradient_balance.json rather than hidden.
        cal_diversity = {}
        for task in S.TASKS:
            vs = by_task_v[task]
            cp0 = S.canon_goal(task) + "_p0"
            pools = ([v for v in vs if v[2] != cp0],
                     [v for v in vs if v[2] == cp0])
            picks, used = [], set()

            def take(pool, want):
                got = 0
                for distinct_only in (True, False):
                    aks = {p[0] for p in picks}
                    for v in pool:
                        if got >= want:
                            return
                        if v in used:
                            continue
                        if distinct_only and v[0] in aks:
                            continue
                        picks.append(v)
                        used.add(v)
                        aks.add(v[0])
                        got += 1

            take(pools[0], 2)
            take(pools[1], 2)
            for pool in pools:            # top up to 4 if a pool was short
                for v in pool:
                    if len(picks) >= 4:
                        break
                    if v not in used:
                        picks.append(v)
                        used.add(v)
            picks = picks[:4]
            cal_diversity[task] = {
                "microbatches": len(picks),
                "distinct_anchors": len({p[0] for p in picks}),
                "inv_active": sum(1 for p in picks if p[2] != cp0)}
            cal.extend(picks)
        assert len(cal) == 20, \
            f"calibration set {len(cal)} != 20: {cal_diversity}"
        assert sum(d["inv_active"] for d in cal_diversity.values()) >= 10, \
            f"language second stream under-sampled: {cal_diversity}"
        norms = defaultdict(list)
        detail = []
        for ak, g, tvv in cal:
            row = {"anchor": ak, "goal": g, "text": tvv}
            for k in ("cl", "roll", "phys", "task", "inv"):
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
        for k in ("cl", "roll", "phys", "task", "inv"):
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
            "calibration_diversity": cal_diversity,
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
    for ak, _rep in order0:
        visits0[anchors[ak]["task"]] += 1
    assert len(set(visits0.values())) == 1, \
        f"epoch-0 schedule unbalanced: {dict(visits0)}"
    # per-ANCHOR mass is NOT balanced on this universe (t4 has one
    # train anchor against a longest queue of 6) and is reported, not
    # asserted away -- see plan_and_progress/archive/daily/2026-08-09.md V7.5.
    task_mass, anchor_mass = S.exposure_census(anchors, "train")
    # the registered point of the query rotation: a wrapped repeat must
    # be a DIFFERENT query, never the same sample twice
    for ak, r in order0:
        n_rep = sum(1 for a_, _ in order0 if a_ == ak)
        seen = {S.visit_query(anchors, tq, ak, 0, rr)
                for a_, rr in order0 if a_ == ak}
        assert len(seen) == min(n_rep, len(S.queries_for(anchors[ak]))), \
            f"wrapped repeats collapse to one query at {ak}"
    mp = OUT / "run_manifest.json"
    if not mp.exists():
        mp.write_text(json.dumps({
            "schema": "v075_lcwm_manifest_v1", "run_schema": "v075",
            "run_id": RID, "git_sha": git_sha,
            "init": {
                "rule": "FRESH at seed 0 (the V7.2 shared "
                        "initialization is deleted and the "
                        "architecture changed); registered in "
                        "plan_and_progress/archive/daily/2026-08-09.md V7.5",
                "state_sha256": init_sha},
            "architecture": {
                "module": "lcwm.v075_state.V075State",
                "repairs": ["r_norm desaturates the history bound",
                            "hist_center re-injects history centered"],
                "stage0_evidence":
                    "2026-08-09_v075_recurrence_diag_r1/"
                    "recurrence_gain.json",
                "stage1b_evidence":
                    "2026-08-09_v075_recurrence_diag_r1/"
                    "repair_check.json"},
            "short_rollout": {
                "spec": "framework_design.md 3.2",
                "k": ROLL_K, "starts_per_visit": ROLL_STARTS,
                "start_window": "final TBPTT window only (beyond it "
                                "the trace is detached, so a rollout "
                                "would apply no pressure to the "
                                "recurrent state)",
                "distance": "cosine on CENTERED states",
                "caveat": "framework 3.3 -- predictive pressure is "
                          "necessary but not sufficient for history "
                          "integration; the Stage 3 gate decides"},
            "data_root": str(V74DATA),
            "input_sha256": {
                "hcache_index.json": sha256_file(hcache_path),
                "v074b_anchor_manifest.json": sha256_file(
                    V74DATA / "anchor_manifest.json")},
            "universe": {
                "trains_on": UNIVERSE,
                "retired": ["v072B", "v072T", "v073",
                            "v073T_released"],
                "retired_reason":
                    "tensors deliberately deleted; v073T_released "
                    "additionally lacks the v073 sources the "
                    "recurrent unroll needs",
                "limitation_carried_into_every_result":
                    "physical-head support is materially thinner "
                    "than V7.3C's and the v071 semwin/replay "
                    "transition diversity is gone"},
            "universe_census": {u: dict(v) for u, v
                                in sorted(census.items())},
            "balanced_epoch0_task_visits": dict(visits0),
            "exposure": {
                "per_task_visits": task_mass,
                "per_anchor_visits_min": min(anchor_mass.values()),
                "per_anchor_visits_max": max(anchor_mass.values()),
                "per_anchor_visits": anchor_mass,
                "note": "per-TASK mass is balanced by construction; "
                        "per-ANCHOR mass is NOT (t4 has one train "
                        "anchor against a longest queue of 6). "
                        "Reported, never asserted away."},
            "history_gate": {
                "quantity": "dev-split relative reset contrast "
                            "||z_real - z_reset|| / ||z_real||",
                "threshold": HIST_GATE,
                "binding_at": "Stage 3, not here",
                "why_reset": "Stage 1b finding 1 -- a different-anchor "
                             "contrast can pass on an open channel "
                             "that carries nothing"},
            "sigma_floor": SIGMA_FLOOR,
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
        for i, (ak, rep_i) in enumerate(order):
            g, tvv = S.visit_query(anchors, tq, ak, epoch, rep_i)
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
        sigma = float(model.hist_center.sigma)
        assert sigma >= SIGMA_FLOOR, (
            f"history centering sigma collapsed to {sigma:.3e} "
            f"(floor {SIGMA_FLOOR:.3e}): the centered channel is "
            f"amplifying float32 noise, not content")
        means["sigma"] = sigma
        means["mu_norm"] = float(model.hist_center.mu.norm())
        logs.append({"epoch": epoch, **means})
        print(f"[v075 ep{epoch}] " + " ".join(
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
