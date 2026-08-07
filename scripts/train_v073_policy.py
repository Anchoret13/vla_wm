#!/usr/bin/env python
"""V7.3D — fine-tune pi0.5 through grounded + model-generated targets.

Five arms (four trained; `stock` is the untouched deployment
baseline evaluated in V7.3E):

  correction_bc   grounded targets, FROZEN GLOBAL-MEAN LC state
  lc_grounded     identical grounded targets, real recurrent state
  lc_full         grounded + model-teacher targets, real state
  lc_random       grounded + matched-random targets, real state

Trainable (byte-identical across arms): the bias-free zero-init
LCProj into the AdaRMS condition + the stock-initialized
`action_out_proj` weight/bias. PrefixVLM, the LCWM, and all other
action-expert blocks frozen.

  L = grounded_first10 + model_first10 + suffix_trust
      + retention_trust + demo_FM
each normalized by its own task->source->anchor (or demo) hierarchy
before the frozen unit coefficients. All weighted first-ten flow
matching goes through lcwm/v073_policy_loss.py (the reviewed module:
ONE batched forward per anchor, weights applied exactly once and
asserted pre-normalized, credit_bounded_actions suffix contract,
suffix trust on 10:50 under the STOCK head snapshot).

300 AdamW steps lr 1e-4 wd 1e-4 clip 1.0; checkpoints + A/B/C
readouts every 50; final-step rule. Videos for every scheduled
training-state environment evaluation.
Output: results/libero_loho_public_v1/2026-08-07_v073_policy_r1/
"""

from __future__ import annotations

import copy
import hashlib
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

from lcwm.v073_policy_loss import (suffix_trust,  # noqa: E402
                                   weighted_anchor_fm)
from lcwm.v06_model import V06State  # noqa: E402

RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
LCWM = RESULTS / "2026-08-06_v073_lcwm_r1"
TEACH = RESULTS / "2026-08-07_v073_teacher_r1"
V73 = RESULTS / "2026-08-04_v073_data_r1"
V72T = RESULTS / "2026-08-03_v072_teacher_r1"
DEMO_DIR = Path("/home/stargazer/Desktop/vla_wm/datasets"
                "/libero_loho_public_v1/demo_rehearsal_v067")
OUT = RESULTS / "2026-08-07_v073_policy_r1"
RID = "v073_policy_r1"
STEPS, LR, WD, CLIP, CKPT_EVERY = 300, 1e-4, 1e-4, 1.0, 50
T_BASE, R_BASE, D_BASE = 973_000, 973_500_000, 973_900_000
ARMS = {
    "correction_bc": {"state": "global_mean", "model_mass": False},
    "lc_grounded": {"state": "recurrent", "model_mass": False},
    "lc_full": {"state": "recurrent", "model_mass": "model"},
    "lc_random": {"state": "recurrent", "model_mass": "random"},
}


def sha_seed(p: str) -> int:
    return int.from_bytes(hashlib.sha256(
        p.encode()).digest()[:8], "big") & ((1 << 63) - 1)


class LCProj(nn.Module):
    """Bias-free zero-init state -> AdaRMS projection (v0.7)."""

    def __init__(self, d_z: int = 384, width: int = 1024):
        super().__init__()
        self.lin = nn.Linear(d_z, width, bias=False)
        nn.init.zeros_(self.lin.weight)

    def forward(self, pool):
        return self.lin(pool)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=None)
    args = ap.parse_args()
    from lcwm.chassis import Pi05Runner
    from lcwm.lc_flow import freeze_pi05_base
    from lcwm.sampler import prefix_forward
    from lcwm.seq_prefix_cache import normalize_actions
    from lcwm.v067_lineage import load_v067, sha256_file

    device = torch.device("cuda")
    torch.manual_seed(0)
    (OUT / "checkpoints").mkdir(parents=True, exist_ok=True)
    (OUT / "videos").mkdir(parents=True, exist_ok=True)
    ref = torch.load(Path("/home/stargazer/Desktop/vla_wm/datasets"
                          "/seq_prefix_cache_v1/task0_demo0.pt"),
                     weights_only=False)
    ref_mean, ref_std = ref["action_mean"], ref["action_std_eps"]

    runner = Pi05Runner(suite_name="libero_10")
    cfg = runner.policy.config
    freeze_pi05_base(runner.policy)
    for p in runner.policy.parameters():
        p.requires_grad_(False)
    aop = runner.policy.model.action_out_proj
    stock_aop = {k: v.detach().clone()
                 for k, v in aop.state_dict().items()}
    import copy as _copy
    stock_head = _copy.deepcopy(aop).to(device)
    stock_head.load_state_dict(stock_aop)
    for p_ in stock_head.parameters():
        p_.requires_grad_(False)
    stock_head.eval()

    wm = V06State().to(device)
    wm.load_state_dict(torch.load(LCWM / "checkpoints" / "final.pt",
                                  weights_only=False)["model"])
    wm.eval()
    for p in wm.parameters():
        p.requires_grad_(False)

    def norm_once(x, is_norm):
        if is_norm:
            cn = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        else:
            cn = normalize_actions(
                torch.from_numpy(np.asarray(x)).float(),
                ref_mean, ref_std)
        if cn.shape[0] < cfg.chunk_size:
            cn = torch.cat([cn, torch.zeros(
                cfg.chunk_size - cn.shape[0], cn.shape[1])], 0)
        return cn[:cfg.chunk_size].to(device)

    # ---------- grounded targets ------------------------------------
    grounded = [json.loads(x) for x in
                (TEACH / "grounded_ledger.jsonl").open()]
    v72_chunks = torch.load(V72T / "candidate_chunks.pt",
                            weights_only=False)
    anchors = {}          # akey -> {task, source_id, decision,
                          #          instr, obs_rows, targets{cid:chunk},
                          #          weights, origin}
    for g in grounded:
        ak = g["anchor"]
        if g["origin"] == "v073B":
            sid = ak.rsplit("_d", 1)[0]
            d = int(ak.rsplit("_d", 1)[1])
            sh = torch.load(V73 / "shards" / f"{ak}.pt",
                            weights_only=False)
            tr = next(t for t in sh["transitions"]
                      if t["branch_key"] == g["branch_key"])
            ch = norm_once(
                tr["chunk_norm"] if tr["chunk_norm"] is not None
                else tr["actions_env"],
                tr["chunk_norm"] is not None)
            src = torch.load(V73 / "sources" / f"{sid}.pt",
                             weights_only=False)
        else:
            sid = ak.rsplit("_d", 1)[0]
            d = int(ak.rsplit("_d", 1)[1])
            ch = norm_once(v72_chunks[f"{ak}_{g['branch_key']}"],
                           True)
            src = torch.load(V72T / "teacher_sources" / f"{sid}.pt",
                             weights_only=False)
        e = anchors.setdefault(f"{g['origin']}::{ak}", {
            "task": g["task"], "source_id": sid, "decision": d,
            "instr": src["language_canonical"],
            "rows": src["rows"], "targets": {}, "origin": "grounded"})
        e["targets"][g["branch_key"]] = ch
    # ---------- model / matched-random targets ----------------------
    model_led = {r["anchor"]: r for r in
                 [json.loads(x) for x in
                  (TEACH / "model_ledger_pre_outcome.jsonl").open()]}
    rand_led = {r["anchor"]: r for r in
                [json.loads(x) for x in
                 (TEACH / "matched_random_ledger.jsonl").open()]}
    teach_chunks = torch.load(TEACH / "candidate_chunks.pt",
                              weights_only=False)
    model_anchors = {}
    for ak, r in model_led.items():
        if r["mode"] != "model_teacher":
            continue
        sid, d = r["source_id"], r["decision"]
        src = torch.load(V73 / "sources" / f"{sid}.pt",
                         weights_only=False)
        model_anchors[ak] = {
            "task": r["task"], "source_id": sid, "decision": d,
            "instr": src["language_canonical"], "rows": src["rows"],
            "model_w": {c: w for c, w in r["weights"].items()
                        if w > 0},
            "random_w": {c: w for c, w
                         in rand_led[ak]["weights"].items()
                         if w > 0},
            "chunks": {c: norm_once(teach_chunks[f"{ak}_{c}"], True)
                       for c in r["weights"]}}

    # ---------- demo rehearsal (recurrent unroll from reset) --------
    dm = json.loads((DEMO_DIR / "manifest.json").read_text())
    demo_eps = {}
    for e in dm["episodes"]:
        ep = load_v067(DEMO_DIR / e["path"], "demo_rehearsal")
        if ep["split"] == "train":
            demo_eps[(e["task_id"], e["demo"])] = ep
    demo_keys = sorted(demo_eps)
    demo_rows = [(tid, di, ri) for (tid, di) in demo_keys
                 for ri in range(len(demo_eps[(tid, di)]["rows"]))]

    # ---------- frozen prefixes + states ----------------------------
    pfx_cache = {}

    @torch.no_grad()
    def prefix_of(key, obs, lang):
        if key not in pfx_cache:
            if len(pfx_cache) > 120:
                pfx_cache.clear()
            b = runner._obs_to_policy_batch(obs, lang)
            pfx_cache[key] = (prefix_forward(runner.policy, b), b)
        return pfx_cache[key]

    @torch.no_grad()
    def recurrent_pool(rows, d, instr, tag):
        z = None
        for dd in range(d + 1):
            pfx, _b = prefix_of(f"{tag}|{dd}", rows[dd]["obs"],
                                instr)
            h, m = pfx.hidden.float(), pfx.pad_masks.bool()
            if z is None:
                z = wm.initial_state(h, m)
            else:
                prev = rows[dd - 1]
                aa = prev["chunk_norm"][None, :10].float().to(device)
                # demo rehearsal rows carry no executed_len (all 50
                # expert actions executed) -> full 10-action mask
                am = (torch.arange(10, device=device)[None]
                      < int(prev.get("executed_len", 10)))
                z = wm.step(z, aa, h, m, action_mask=am)
        return z.mean(dim=1).detach()

    @torch.no_grad()
    def demo_pool(ep, ri, tag):
        """Demo state at row ri, unrolled at the WM's OWN 10-action
        decision stride over ep["obs_10"] (the dataset ships it for
        exactly this; ep["rows"] is a 50-step stride and feeding its
        first 10 actions would skip 40 of every 50 expert actions and
        put the transition far out of its training distribution —
        verification defect A)."""
        target_t = 50 * ri
        seq = [o for o in ep["obs_10"] if o["t"] <= target_t]
        z = None
        for j, o in enumerate(seq):
            pfx, _b = prefix_of(f"{tag}|t{o['t']}", o["obs"],
                                ep["language"])
            h, m = pfx.hidden.float(), pfx.pad_masks.bool()
            if z is None:
                z = wm.initial_state(h, m)
            else:
                ae = np.asarray(seq[j - 1]["chunk10_env"])
                aa = normalize_actions(
                    torch.from_numpy(ae).float(),
                    ref_mean, ref_std)[None].to(device)
                n_a = aa.shape[1]
                if n_a < 10:
                    aa = torch.cat([aa, torch.zeros(
                        1, 10 - n_a, 7, device=device)], dim=1)
                am = (torch.arange(10, device=device)[None] < n_a)
                z = wm.step(z, aa[:, :10], h, m, action_mask=am)
        return z.mean(dim=1).detach()

    STATE_SCHEMA = "v073_policy_states_v2"
    SCHED_SCHEMA = "v073_policy_sched_v2"
    state_path = OUT / "anchor_states.pt"
    _blob = (torch.load(state_path, weights_only=False)
             if state_path.exists() else None)
    if _blob is not None and _blob.get("schema") != STATE_SCHEMA:
        _blob = None            # stale pre-fix artifact: rebuild
    if _blob is not None:
        blob = _blob
        pools = {k: v.to(device) for k, v in blob["pools"].items()}
        demo_pools = {tuple(json.loads(k)): v.to(device)
                      for k, v in blob["demo_pools"].items()}
        gmean = blob["global_mean"].to(device)
    else:
        pools, demo_pools = {}, {}
        for ak, e in sorted(anchors.items()):
            pools[ak] = recurrent_pool(e["rows"], e["decision"],
                                       e["instr"], ak)
        for ak, e in sorted(model_anchors.items()):
            pools[f"model::{ak}"] = recurrent_pool(
                e["rows"], e["decision"], e["instr"],
                f"model::{ak}")
        for (tid, di) in demo_keys:
            ep = demo_eps[(tid, di)]
            for ri in range(len(ep["rows"])):
                demo_pools[(tid, di, ri)] = demo_pool(
                    ep, ri, f"demo{tid}_{di}")
        # probe-anchor variants for the A/B/C + reset/shuffled
        # attribution readouts (registered V7.3A item)
        probe = sorted(anchors)[0]
        pe = anchors[probe]
        with torch.no_grad():
            pf, _pb = prefix_of(f"{probe}|{pe['decision']}",
                                pe["rows"][pe["decision"]]["obs"],
                                pe["instr"])
            zr = wm.initial_state(pf.hidden.float(),
                                  pf.pad_masks.bool())
            pools["__reset__"] = zr.mean(dim=1).detach()
            # ORDER-only control: same length, same (obs, prev
            # action) pairing convention, same final anchor
            # observation as the real pool — only the order of the
            # PRE-anchor history is permuted (verification defect B:
            # the earlier variant also dropped the anchor obs and
            # shifted the action pairing).
            hist = list(range(pe["decision"]))
            gsh = torch.Generator().manual_seed(
                sha_seed(f"{RID}|shuffle|{probe}"))
            perm = ([hist[i] for i in torch.randperm(
                len(hist), generator=gsh).tolist()] if hist else [])
            seq_s = perm + [pe["decision"]]
            zs = None
            for j, dd in enumerate(seq_s):
                pfx_s, _ = prefix_of(f"{probe}|{dd}",
                                     pe["rows"][dd]["obs"],
                                     pe["instr"])
                h, m = pfx_s.hidden.float(), pfx_s.pad_masks.bool()
                if zs is None:
                    zs = wm.initial_state(h, m)
                else:
                    prev = pe["rows"][seq_s[j - 1]]
                    aa = prev["chunk_norm"][None, :10].float() \
                        .to(device)
                    am = (torch.arange(10, device=device)[None]
                          < int(prev.get("executed_len", 10)))
                    zs = wm.step(zs, aa, h, m, action_mask=am)
            pools["__shuffled__"] = zs.mean(dim=1).detach()
        # frozen task-balanced global mean over TRAIN histories
        by_task = defaultdict(list)
        for ak, e in anchors.items():
            by_task[e["task"]].append(pools[ak])
        gmean = torch.stack([torch.stack(v).mean(0)
                             for v in by_task.values()]).mean(0)
        torch.save({"schema": STATE_SCHEMA,
                    "pools": {k: v.cpu() for k, v in pools.items()},
                    "demo_pools": {json.dumps(list(k)): v.cpu()
                                   for k, v in demo_pools.items()},
                    "global_mean": gmean.cpu()}, state_path)

    # ---------- matched schedule ------------------------------------
    sched_path = OUT / "matched_training_schedule.pt"
    _s = (torch.load(sched_path, weights_only=False)
          if sched_path.exists() else None)
    if _s is not None and _s.get("schema") != SCHED_SCHEMA:
        sched_path.unlink()
    if not sched_path.exists():
        gkeys = sorted(anchors)
        mkeys = sorted(model_anchors)
        by_task_g = defaultdict(list)
        for ak in gkeys:
            by_task_g[anchors[ak]["task"]].append(ak)
        order = []
        i = 0
        while len(order) < STEPS:
            for task in sorted(by_task_g):
                lst = by_task_g[task]
                order.append(lst[i % len(lst)])
                if len(order) >= STEPS:
                    break
            i += 1
        m_order = [mkeys[k % len(mkeys)] if mkeys else None
                   for k in range(STEPS)]
        # (task_id, demo, ROW) triples; retention uses a disjoint
        # offset stream so the two components never share a row
        d_order = [demo_rows[k % len(demo_rows)]
                   for k in range(STEPS)]
        r_order = [demo_rows[(k + len(demo_rows) // 2)
                             % len(demo_rows)] for k in range(STEPS)]
        torch.save({"schema": SCHED_SCHEMA, "grounded": order,
                    "model": m_order, "demo": d_order,
                    "retention": r_order}, sched_path)
    sched = torch.load(sched_path, weights_only=False)

    git_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True,
        text=True, cwd=REPO_ROOT).stdout.strip()
    mp = OUT / "run_manifest.json"
    if not mp.exists():
        mp.write_text(json.dumps({
            "schema": "v073_policy_manifest_v1", "run_schema":
            "v073", "run_id": RID, "git_sha": git_sha,
            "arms": ARMS,
            "trainable": "LCProj(bias-free, zero-init) + "
                         "action_out_proj (stock init)",
            "frozen": "PrefixVLM, LCWM, all other expert blocks",
            "optimizer": {"steps": STEPS, "lr": LR, "wd": WD,
                          "clip": CLIP, "ckpt_every": CKPT_EVERY,
                          "selection": "final step ONLY"},
            "grounded_rows": len(grounded),
            "grounded_anchors": len(anchors),
            "model_anchors": len(model_anchors),
            "teacher_seal": json.loads(
                (TEACH / "ledger_seal.json").read_text()),
            "lcwm_final_sha": sha256_file(
                LCWM / "checkpoints" / "final.pt"),
        }, indent=2))

    def train_arm(arm):
        cfg_a = ARMS[arm]
        aop.load_state_dict(stock_aop)
        for p in aop.parameters():
            p.requires_grad_(True)
        lc = LCProj().to(device)
        params = list(lc.parameters()) + list(aop.parameters())
        opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
        groups = {"lc_proj": list(lc.parameters()),
                  "action_out_proj": list(aop.parameters())}
        logs, grad_report = [], {}

        probe = sorted(anchors)[0]
        pe = anchors[probe]

        @torch.no_grad()
        def readouts(lc_, bias_fn_, arm_, step_):
            """A/B/C LC attribution + reset/shuffled substitution +
            grounded-target reproduction, same-noise (registered)."""
            from lcwm.lc_flow import sample_chunks_lc
            from lcwm.sampler import sample_chunks
            from lcwm.v067_lineage import flow_noise
            pf, pb = prefix_of(f"{probe}|{pe['decision']}",
                               pe["rows"][pe["decision"]]["obs"],
                               pe["instr"])
            nz = flow_noise(sha_seed(f"{RID}|abc|{step_}"),
                            cfg.chunk_size,
                            cfg.max_action_dim).to(device)
            zb = torch.zeros(1, 1024, device=device)
            chA = sample_chunks_lc(runner.policy, pb,
                                   bias_fn_(pools[probe]), n=1,
                                   noise=nz, prefix=pf)[0]
            chB = sample_chunks_lc(runner.policy, pb, zb, n=1,
                                   noise=nz, prefix=pf)[0]
            live = runner.policy.model.action_out_proj
            try:
                runner.policy.model.action_out_proj = stock_head
                chC = sample_chunks(runner.policy, pb, n=1,
                                    noise=nz, prefix=pf)[0]
            finally:
                runner.policy.model.action_out_proj = live
            chR = sample_chunks_lc(runner.policy, pb,
                                   lc_(pools["__reset__"]), n=1,
                                   noise=nz, prefix=pf)[0]
            chS = sample_chunks_lc(runner.policy, pb,
                                   lc_(pools["__shuffled__"]), n=1,
                                   noise=nz, prefix=pf)[0]
            tgt = next(iter(pe["targets"].values()))[:10, :7]
            d10 = lambda a, b: float(
                (a[:10, :7] - b[:10, :7]).abs().mean())
            return {"A_minus_B_state_shift": d10(chA, chB),
                    "B_minus_C_head_drift": d10(chB, chC),
                    "A_minus_reset": d10(chA, chR),
                    "A_minus_shuffled": d10(chA, chS),
                    "target_repro_A": float(
                        (chA[:10, :7] - tgt).abs().mean()),
                    "target_repro_C": float(
                        (chC[:10, :7] - tgt).abs().mean())}

        @torch.no_grad()
        def env_eval(lc_, bias_fn_, arm_, step_):
            """Registered SMALL training-state environment
            evaluation: re-reach one probe anchor (rotating over the
            v073B anchors), execute 10 actions from THIS arm's
            policy, then 60 stock continuation steps. Video saved for
            every rollout regardless of outcome. Diagnostic only —
            cannot select an arm or checkpoint (final-step rule)."""
            from lcwm.lc_flow import sample_chunks_lc
            from lcwm.loho_public import make_public_env
            from lcwm.sampler import sample_chunks
            from lcwm.task_automaton import GoalAutomaton
            from lcwm.v067_lineage import flow_noise
            from lcwm.video_recorder import (VideoRecorder,
                                             write_index_row)
            eval_keys = [a for a in sorted(anchors)
                         if a.startswith("v073B::")]
            if not eval_keys:
                return {}
            ak_ = eval_keys[(step_ // CKPT_EVERY - 1)
                            % len(eval_keys)]
            e_ = anchors[ak_]
            sid_ = e_["source_id"]
            src_ = torch.load(V73 / "sources" / f"{sid_}.pt",
                              weights_only=False)
            task_ = e_["task"]
            gm = json.loads((RESULTS
                             / "goal_spec_manifest_v067.json")
                            .read_text())["tasks"][task_]
            sub = gm["goal_specs"][
                gm["canonical_goal_spec_id"]]["ordered_subgoals"]
            env = make_public_env(task_, 1400)
            try:
                runner.reset()
                env.reset(seed=src_["seed"])
                env._env.env.horizon = 1400
                au = GoalAutomaton(list(sub))
                au.start(env)
                au.evaluate(env, 0)
                st = 0
                for i_ in range(e_["decision"]):
                    for a_env in src_["rows"][i_]["actions_env"]:
                        env.step(a_env)
                        st += 1
                        au.evaluate(env, st)
                vr = VideoRecorder(
                    OUT / "videos"
                    / f"{arm_}_step{step_:03d}_{ak_[7:]}.mp4")
                obs_ = copy.deepcopy(env._format_raw_obs(
                    env._env.env._get_observations()))
                vr.add(obs_)
                b_ = runner._obs_to_policy_batch(obs_, e_["instr"])
                pf_ = prefix_forward(runner.policy, b_)
                nz = flow_noise(sha_seed(
                    f"{RID}|eval|{arm_}|{step_}"), cfg.chunk_size,
                    cfg.max_action_dim).to(device)
                ch_ = sample_chunks_lc(runner.policy, b_,
                                       bias_fn_(pools[ak_]), n=1,
                                       noise=nz, prefix=pf_)
                for a_env in runner.chunk_to_env(ch_[:, :10]):
                    _o, _r, tb, tr2, _i = env.step(a_env)
                    st += 1
                    au.evaluate(env, st)
                    obs_ = copy.deepcopy(env._format_raw_obs(
                        env._env.env._get_observations()))
                    vr.add(obs_)
                    if tb:
                        env._env.env.done = False
                    if tr2:
                        break
                q_after = au.q_valid()
                live = runner.policy.model.action_out_proj
                sc_, hit_term = 0, False
                try:
                    runner.policy.model.action_out_proj = stock_head
                    while sc_ < 60 and not hit_term:
                        b2 = runner._obs_to_policy_batch(
                            obs_, e_["instr"])
                        p2 = prefix_forward(runner.policy, b2)
                        nz2 = flow_noise(sha_seed(
                            f"{RID}|evalcont|{step_}|{sc_}"),
                            cfg.chunk_size, cfg.max_action_dim)
                        c2 = sample_chunks(runner.policy, b2, n=1,
                                           noise=nz2.to(device),
                                           prefix=p2)
                        for a_env in runner.chunk_to_env(
                                c2[:, :10]):
                            _o, _r, tm, tr3, _i = env.step(a_env)
                            sc_ += 1
                            au.evaluate(env, st + sc_)
                            obs_ = copy.deepcopy(
                                env._format_raw_obs(
                                    env._env.env
                                    ._get_observations()))
                            vr.add(obs_)
                            if tm or tr3:
                                hit_term = bool(tm)
                                break
                            if sc_ >= 60:
                                break
                finally:
                    runner.policy.model.action_out_proj = live
                vm = vr.close(completed=True)
                write_index_row(
                    OUT / "video_index.jsonl", vm, run_id=RID,
                    checkpoint_tag=f"{arm_}_step{step_}",
                    checkpoint_path=None, checkpoint_sha256=None,
                    manifest_sha256="policy_training_eval",
                    task=task_, seed=src_["seed"], arm=arm_,
                    split="train_state", steps=st + sc_,
                    success=bool(hit_term),
                    ordered_progress=au.ordered_prefix(),
                    damage=au.damage_unrecovered(),
                    termination=("terminal" if hit_term
                                 else "eval_end"), root=OUT)
                return {"eval_anchor": ak_,
                        "eval_terminal": bool(hit_term),
                        "eval_q_after10": float(q_after),
                        "eval_q_final": float(au.q_valid()),
                        "eval_prefix": int(au.ordered_prefix())}
            finally:
                env.close()

        def bias_for(pool):
            if cfg_a["state"] == "global_mean":
                return lc(gmean)
            return lc(pool)

        for k in range(STEPS):
            opt.zero_grad(set_to_none=True)
            g = torch.Generator().manual_seed(T_BASE + k)
            noise = torch.randn(1, cfg.chunk_size,
                                cfg.max_action_dim,
                                generator=g).to(device)
            time = torch.rand(1, generator=g).to(device)
            comps = {}
            # (1) grounded first-ten
            ak = sched["grounded"][k]
            e = anchors[ak]
            pfx, _b = prefix_of(f"{ak}|{e['decision']}",
                                e["rows"][e["decision"]]["obs"],
                                e["instr"])
            tgts = sorted(e["targets"])
            cands = torch.stack([e["targets"][c][:, :7]
                                 for c in tgts])
            w = torch.full((len(tgts),), 1.0 / len(tgts),
                           device=device)
            bias = bias_for(pools[ak])
            comps["grounded"], _ = weighted_anchor_fm(
                runner.policy, pfx, cands, w, bias, noise, time)
            # (2) model / matched-random first-ten
            mk = sched["model"][k]
            if mk is not None and cfg_a["model_mass"]:
                me = model_anchors[mk]
                wsrc = (me["model_w"] if cfg_a["model_mass"]
                        == "model" else me["random_w"])
                if wsrc:
                    mpfx, _mb = prefix_of(
                        f"model::{mk}|{me['decision']}",
                        me["rows"][me["decision"]]["obs"],
                        me["instr"])
                    cids = sorted(wsrc)
                    mc = torch.stack([me["chunks"][c][:, :7]
                                      for c in cids])
                    tot = sum(wsrc[c] for c in cids)
                    mw = torch.tensor([wsrc[c] / tot for c in cids],
                                      device=device)
                    mbias = bias_for(pools[f"model::{mk}"])
                    comps["model"], _ = weighted_anchor_fm(
                        runner.policy, mpfx, mc, mw, mbias,
                        noise, time)
            # (3) suffix trust at the correction anchor (10:50 only,
            #     against the STOCK head)
            u0 = e["rows"][e["decision"]]["chunk_norm"]
            comps["suffix_trust"] = suffix_trust(
                runner.policy, pfx,
                norm_once(u0, True)[:, :7], bias, noise, time,
                stock_head=stock_head)
            # (4) retention trust: full 0:50 on the retention stream
            rt = sched["retention"][k]
            rep = demo_eps[(rt[0], rt[1])]
            rpfx, _rb = prefix_of(f"demo{rt[0]}_{rt[1]}|{rt[2]}",
                                  rep["rows"][rt[2]]["obs"],
                                  rep["language"])
            rbias = bias_for(demo_pools[tuple(rt)])
            comps["retention_trust"] = suffix_trust(
                runner.policy, rpfx,
                rep["rows"][rt[2]]["chunk_norm"].to(device)[:, :7],
                rbias, noise, time, stock_head=stock_head,
                lo=0, hi=cfg.chunk_size)
            # (5) demo flow matching under the recurrent demo state
            dt = sched["demo"][k]
            dep = demo_eps[(dt[0], dt[1])]
            dpfx, _db = prefix_of(f"demo{dt[0]}_{dt[1]}|{dt[2]}",
                                  dep["rows"][dt[2]]["obs"],
                                  dep["language"])
            dbias = bias_for(demo_pools[tuple(dt)])
            g2 = torch.Generator().manual_seed(D_BASE + k)
            dn = torch.randn(1, cfg.chunk_size, cfg.max_action_dim,
                             generator=g2).to(device)
            dtm = torch.rand(1, generator=g2).to(device)
            dch = dep["rows"][dt[2]]["chunk_norm"].to(device)[None,
                                                              :, :7]
            comps["demo"], _ = weighted_anchor_fm(
                runner.policy, dpfx, dch,
                torch.ones(1, device=device), dbias, dn, dtm,
                credit_t=None)      # full-50 demonstration FM
            if k == 0:
                for name, term in comps.items():
                    opt.zero_grad(set_to_none=True)
                    term.backward(retain_graph=True)
                    grad_report[name] = {
                        gname: float(sum(
                            (p.grad ** 2).sum() for p in ps
                            if p.grad is not None).sqrt())
                        for gname, ps in groups.items()}
                assert grad_report["grounded"]["lc_proj"] >= 0
                assert grad_report["demo"]["action_out_proj"] > 0, \
                    "demo term constant on the action head"
                (OUT / f"grad_checks_{arm}.json").write_text(
                    json.dumps(grad_report, indent=2))
                opt.zero_grad(set_to_none=True)
            total = sum(comps.values())
            assert torch.isfinite(total)
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, CLIP)
            opt.step()
            if (k + 1) % CKPT_EVERY == 0:
                row = {"step": k + 1,
                       **readouts(lc, bias_for, arm, k + 1),
                       **env_eval(lc, bias_for, arm, k + 1),
                       **{c: float(v) for c, v in comps.items()}}
                with torch.no_grad():
                    row["lc_proj_norm"] = float(
                        lc.lin.weight.norm())
                    row["aop_delta"] = float(
                        (aop.weight - stock_aop["weight"]).norm())
                logs.append(row)
                torch.save({"schema": "v073_policy_ckpt_v1",
                            "arm": arm, "step": k + 1,
                            "lc_proj": lc.state_dict(),
                            "action_out_proj": aop.state_dict(),
                            "logs": logs},
                           OUT / "checkpoints"
                           / f"{arm}_step{k + 1:03d}.pt")
                print(f"[{arm} {k + 1}/{STEPS}] " + " ".join(
                    f"{c}={float(v):.4f}"
                    for c, v in comps.items())
                    + f" |lc|={row['lc_proj_norm']:.4f}"
                      f" |dAOP|={row['aop_delta']:.4f}",
                    flush=True)
        torch.save({"schema": "v073_policy_ckpt_v1", "arm": arm,
                    "step": STEPS, "lc_proj": lc.state_dict(),
                    "action_out_proj": aop.state_dict(),
                    "grad_checks": grad_report, "logs": logs},
                   OUT / "checkpoints" / f"{arm}_final.pt")
        aop.load_state_dict(stock_aop)
        return {"final": f"{arm}_final.pt",
                "grad_checks": grad_report, "logs": logs}

    results = {}
    for arm in ([args.arm] if args.arm else list(ARMS)):
        results[arm] = train_arm(arm)
        (OUT / "offline_metrics.json").write_text(
            json.dumps(results, indent=2))
    print(json.dumps({a: r["logs"][-1] for a, r in results.items()},
                     indent=1), flush=True)
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
