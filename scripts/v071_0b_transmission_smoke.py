#!/usr/bin/env python
"""V7.1.0B — one discriminating-target transmission smoke.

Anchor: the clean T2 anchor (t2_s2110_d16) whose executed pair is
sub0 winner (+1) vs sub1 nonwinner (−1). Two arms differing ONLY in
the distillation target (oracle=sub0 vs nonwinner=sub1): identical
prefix, recurrent LC state, flow noise/time, rehearsal, trust,
optimizer, execution CRNs, evaluation horizon.

Policy boundary N1 (narrow; every trainable parameter name bound):
  - pi0.5 action expert's `action_out_proj` (stock init, trainable);
  - new zero-init LC projection Pool(z_rec) -> AdaRMS condition
    (replaces the failed global-bias-only W_z route; the recurrent LC
    condition enters the trained boundary).
If N1 fails, ONE broadening to the final two action-expert blocks;
freeze the result either way. No rank/layer/seed search.

Pass requires BOTH:
  1. target-action reconstruction: the arm's generated chunk is closer
     to its target's first-ten actions than stock's generation is;
  2. fresh environment re-execution: the oracle arm's generated chunk,
     executed at the anchor under fresh CRN continuations (R=2),
     reproduces the win vs fresh u_0 (pref +1).
Offline flow loss alone cannot pass. Every simulator evaluation saves
a video under the run/video contract.

Output: results/libero_loho_public_v1/<DATE>_v071_0b_smoke_r1/
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

DATA = Path("/home/stargazer/Desktop/vla_wm/datasets/libero_loho_public_v1"
            "/v06_effect_crossed")
DEMO_DIR = Path("/home/stargazer/Desktop/vla_wm/datasets"
                "/libero_loho_public_v1/demo_rehearsal_v067")
RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
STAGE = "0b_smoke"
RUN_ID = "r1"
ANCHOR = ("loho_t2_basket3_correction_s2110", 16)
TASK = "loho_t2_basket3"
EPISODE_LEN = 900
N_STEPS = 200
LR, WD, GRAD_NORM = 1e-4, 1e-4, 1.0
T_BASE, R_BASE = 997_000, 997_500_000
HORIZONS = (10, 30, 60, 100)
R = 2
CONT_MAX = 100
ARMS = {"oracle_sub0": "sub0", "nonwinner_sub1": "sub1"}


def sha_seed(payload: str) -> int:
    return int.from_bytes(hashlib.sha256(
        payload.encode()).digest()[:8], "big") & ((1 << 63) - 1)


class LCProj(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(384, 1024)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, pool):
        return self.lin(pool)


def main() -> None:
    from lcwm.chassis import Pi05Runner
    from lcwm.goal_semantics import SuccessTracker, env_eval_fn
    from lcwm.lc_flow import (cached_branch_flow_loss,
                              denoise_step_with_lc_bias,
                              freeze_pi05_base,
                              raw_flow_losses_from_prefix,
                              sample_chunks_lc)
    from lcwm.loho_public import make_public_env
    from lcwm.probe_data import body_positions
    from lcwm.sampler import _expand_cache, prefix_forward, \
        sample_chunks
    from lcwm.snapshot import restore, snap
    from lcwm.task_automaton import (GoalAutomaton, fork_env_state,
                                     paired_preference,
                                     restore_env_state)
    from lcwm.v067_lineage import flow_noise, load_v067, sha256_file
    from lcwm.v06_model import V06State
    from lcwm.video_recorder import (VideoRecorder, video_name,
                                     write_index_row)
    from lerobot.utils.constants import ACTION
    from scripts.train_v069_grounded_wz import round_robin

    device = torch.device("cuda")
    torch.manual_seed(0)
    date = subprocess.run(["date", "+%F"], capture_output=True,
                          text=True,
                          env={"TZ": "America/Chicago"}).stdout.strip()
    root = None
    for p in sorted(RESULTS.glob(f"*_v071_{STAGE}_{RUN_ID}")):
        root = p
    root = root or RESULTS / f"{date}_v071_{STAGE}_{RUN_ID}"
    for sub in ("checkpoints", "eval/videos", "eval/traces", "train"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    sid, d = ANCHOR
    grp = torch.load(DATA / "corrections_v069" / f"{sid}_d{d}.pt",
                     weights_only=False)
    tolerances = grp["frozen_outcome_tolerances"]
    by_key = {str(b["key"]): b for b in grp["branch_summaries"]}
    targets = {arm: by_key[k]["chunk_norm"]
               for arm, k in ARMS.items()}
    source = torch.load(DATA / "corrections_sources_v069"
                        / f"{sid}.pt", weights_only=False)
    u0_chunk = source["rows"][d]["chunk_norm"]

    model = V06State().to(device)
    wm = torch.load(RESULTS / "v069_predictive"
                    / "checkpoint_selected.pt", weights_only=False)
    model.load_state_dict(wm["model"])
    model.eval()
    runner = Pi05Runner(suite_name="libero_10")
    cfg = runner.policy.config
    freeze_pi05_base(runner.policy)
    stock_out_proj = {
        k: v.detach().clone() for k, v in
        runner.policy.model.action_out_proj.state_dict().items()}

    # ---- recurrent z at the anchor (frozen) ----------------------------
    with torch.no_grad():
        z = None
        for i, row in enumerate(source["rows"]):
            if i > d:
                break
            batch = runner._obs_to_policy_batch(
                row["obs"], source["language_canonical"])
            prefix = prefix_forward(runner.policy, batch)
            h = prefix.hidden.float()
            m = prefix.pad_masks.bool()
            if z is None:
                z = model.initial_state(h, m)
            else:
                prev = source["rows"][i - 1]
                aa = prev["chunk_norm"][None, :10].float().to(device)
                am = (torch.arange(10, device=device)[None]
                      < prev["executed_len"])
                z = model.step(z, aa, h, m, action_mask=am)
            if i == d:
                anchor_prefix, anchor_batch = prefix, batch
                break
        pool = z.mean(dim=1).detach()

    # demo rehearsal rows (frozen V6.9 one-epoch schedule, cycled)
    demo_manifest = json.loads((DEMO_DIR / "manifest.json").read_text())
    demo_eps = {}
    for e in demo_manifest["episodes"]:
        ep = load_v067(DEMO_DIR / e["path"], "demo_rehearsal")
        if ep["split"] == "train":
            demo_eps[(e["task_id"], e["demo"])] = ep
    rehearsal_by_task = {}
    for (tid, di), ep in sorted(demo_eps.items()):
        for ri, _row in enumerate(ep["rows"]):
            rehearsal_by_task.setdefault(tid, []).append(
                (tid, di, ri))
    for tid in rehearsal_by_task:
        by_demo = {}
        for it in rehearsal_by_task[tid]:
            by_demo.setdefault(it[1], []).append(it)
        rehearsal_by_task[tid] = round_robin(by_demo)
    rehearsal_schedule = round_robin(rehearsal_by_task)

    manifest = {
        "schema": "v071_0b_manifest_v1", "run_schema": "v071",
        "run_id": RUN_ID, "stage": STAGE, "run_date": date,
        "anchor": f"{sid}_d{d}", "arms": ARMS,
        "boundary": "N1",
        "trainable_parameter_names": [
            "policy.model.action_out_proj.weight",
            "policy.model.action_out_proj.bias",
            "lc_proj.lin.weight (zero-init)",
            "lc_proj.lin.bias (zero-init)"],
        "n_steps": N_STEPS,
        "optimizer": {"lr": LR, "wd": WD, "grad_norm": GRAD_NORM},
        "flow_seed_bases": {"anchor": T_BASE, "demo": R_BASE},
        "eval_crn": ("v071_0b_r1|action / |cont| SHA signatures"),
        "video": {"layout": "side_by_side_external_wrist", "fps": 20},
        "pass_rule": ("target-action reconstruction better than stock "
                      "AND fresh re-execution pref vs fresh u_0 == +1 "
                      "for the oracle arm"),
        "predictive_sha256": sha256_file(
            RESULTS / "v069_predictive" / "checkpoint_selected.pt"),
    }
    (root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2))

    def velocity_trust(prefix, chunk, bias, noise, time):
        chunk = chunk[None].to(device).float()
        actions = runner.policy.prepare_action({ACTION: chunk})
        x_t = time[:, None, None] * noise \
            + (1 - time[:, None, None]) * actions
        cache = _expand_cache(prefix.past_key_values, 1)
        v_b = denoise_step_with_lc_bias(
            runner.policy.model, prefix.pad_masks, cache, x_t, time,
            bias)
        with torch.no_grad():
            v_0 = denoise_step_with_lc_bias(
                runner.policy.model, prefix.pad_masks, cache, x_t,
                time, torch.zeros_like(bias))
        return torch.nn.functional.mse_loss(v_b, v_0)

    prefix_lru = {}

    @torch.no_grad()
    def prefix_demo(key, obs, lang):
        if key not in prefix_lru:
            if len(prefix_lru) > 30:
                prefix_lru.clear()
            b = runner._obs_to_policy_batch(obs, lang)
            prefix_lru[key] = prefix_forward(runner.policy, b)
        return prefix_lru[key]

    results = {}
    for arm, target_key in ARMS.items():
        # reset boundary to exact stock/zero init
        runner.policy.model.action_out_proj.load_state_dict(
            stock_out_proj)
        lc_proj = LCProj().to(device)
        for p_ in runner.policy.model.action_out_proj.parameters():
            p_.requires_grad_(True)
        trainable = (list(
            runner.policy.model.action_out_proj.parameters())
            + list(lc_proj.parameters()))
        optimizer = torch.optim.AdamW(trainable, lr=LR,
                                      weight_decay=WD)
        target = targets[arm]
        for k in range(N_STEPS):
            optimizer.zero_grad(set_to_none=True)
            bias = lc_proj(pool)
            g = torch.Generator().manual_seed(T_BASE + k)
            noise = torch.randn(1, cfg.chunk_size,
                                cfg.max_action_dim,
                                generator=g).to(device)
            time = torch.rand(1, generator=g).to(device)
            loss_c, _ = cached_branch_flow_loss(
                runner.policy, anchor_prefix,
                target[None].to(device).float(), bias,
                torch.tensor([1.0], device=device), max_executed=10,
                noise=noise, time=time)
            trust = velocity_trust(anchor_prefix, u0_chunk, bias,
                                   noise, time)
            tid, di, ri = rehearsal_schedule[k % len(
                rehearsal_schedule)]
            ep = demo_eps[(tid, di)]
            drow = ep["rows"][ri]
            dprefix = prefix_demo(("demo", tid, di, ri),
                                  drow["obs"], ep["language"])
            g2 = torch.Generator().manual_seed(R_BASE + k)
            noise2 = torch.randn(1, cfg.chunk_size,
                                 cfg.max_action_dim,
                                 generator=g2).to(device)
            time2 = torch.rand(1, generator=g2).to(device)
            demo_loss = raw_flow_losses_from_prefix(
                runner.policy,
                drow["chunk_norm"][None].to(device).float(),
                torch.zeros(1, 1024, device=device), dprefix,
                noise=noise2, time=time2).mean()
            total = loss_c + trust + demo_loss
            total.backward()
            torch.nn.utils.clip_grad_norm_(trainable, GRAD_NORM)
            optimizer.step()
            if (k + 1) % 50 == 0:
                print(f"[{arm} {k + 1}/{N_STEPS}] corr="
                      f"{float(loss_c):.4f} trust={float(trust):.4f} "
                      f"demo={float(demo_loss):.4f}", flush=True)

        ckpt = {"schema": "v071_0b_checkpoint_v1",
                "run_schema": "v071", "arm": arm,
                "boundary": "N1",
                "action_out_proj": runner.policy.model
                .action_out_proj.state_dict(),
                "lc_proj": lc_proj.state_dict()}
        torch.save(ckpt, root / "checkpoints" / f"{arm}_final.pt")

        # ---- offline reconstruction ------------------------------------
        with torch.no_grad():
            bias = lc_proj(pool)
            rec_seed = sha_seed(f"v071_0b_r1|action|{sid}|{d}")
            nz = flow_noise(rec_seed, cfg.chunk_size,
                            cfg.max_action_dim)
            gen = sample_chunks_lc(runner.policy, anchor_batch, bias,
                                   n=1, noise=nz.to(device),
                                   prefix=anchor_prefix)
            runner.policy.model.action_out_proj.load_state_dict(
                stock_out_proj)
            gen_stock = sample_chunks(runner.policy, anchor_batch,
                                      n=1, noise=nz.to(device),
                                      prefix=anchor_prefix)
            runner.policy.model.action_out_proj.load_state_dict(
                ckpt["action_out_proj"])
            d_arm = float((gen[0, :10]
                           - target[None][0, :10].to(device))
                          .abs().mean())
            d_stock = float((gen_stock[0, :10]
                             - target[None][0, :10].to(device))
                            .abs().mean())
        results[arm] = {"recon_arm": d_arm, "recon_stock": d_stock,
                        "recon_pass": d_arm < d_stock,
                        "gen_chunk": gen[0].float().cpu()}
        print(f"[{arm}] recon {d_arm:.4f} vs stock {d_stock:.4f} "
              f"pass={d_arm < d_stock}", flush=True)

    # ---- fresh environment re-execution (videos for every rollout) -----
    vid_index = root / "eval" / "video_index.jsonl"
    env = make_public_env(TASK, EPISODE_LEN + 200)
    try:
        entry = json.loads((RESULTS
                            / "goal_spec_manifest_v067.json")
                           .read_text())["tasks"][TASK]
        canon_id = entry["canonical_goal_spec_id"]
        subgoals = entry["goal_specs"][canon_id]["ordered_subgoals"]
        term_preds = [tuple(p) for p in entry["goal_specs"][canon_id]
                      ["terminal_predicates"]]
        runner.reset()
        env.reset(seed=source["seed"])
        env._env.env.horizon = EPISODE_LEN + 300
        eval_fn = env_eval_fn(env)
        auto0 = GoalAutomaton(list(subgoals))
        auto0.start(env)
        auto0.evaluate(env, 0)
        t = 0
        for i, row in enumerate(source["rows"]):
            if i >= d:
                break
            for a_env in row["actions_env"]:
                env.step(a_env)
                t += 1
                auto0.evaluate(env, t)
        err = float(np.abs(body_positions(
            env, list(auto0.bodies.values()))
            - source["rows"][d]["obj_before"]).max())
        assert err < 2e-3
        snap_a = snap(env, t=t, suite_name="loho_public", task_id=0)
        anchor_state = fork_env_state(auto0)

        a_seed = sha_seed(f"v071_0b_r1|action|{sid}|{d}")
        u0_fresh = None
        with torch.no_grad():
            runner.policy.model.action_out_proj.load_state_dict(
                stock_out_proj)
            obs_a = env._format_raw_obs(
                env._env.env._get_observations())
            b = runner._obs_to_policy_batch(
                obs_a, source["language_canonical"])
            pfx = prefix_forward(runner.policy, b)
            u0_fresh = sample_chunks(
                runner.policy, b, n=1,
                noise=flow_noise(a_seed, cfg.chunk_size,
                                 cfg.max_action_dim).to(device),
                prefix=pfx)[0].float().cpu()

        exec_specs = [("fresh_u0", u0_fresh)] + [
            (arm, results[arm]["gen_chunk"]) for arm in ARMS]
        outcomes = {}
        for bname, chunk in exec_specs:
            reps = []
            for rep in range(R):
                restore(env, snap_a)
                env._env.env.done = False
                ba = GoalAutomaton(list(subgoals))
                ba.bodies, ba.start_pos = auto0.bodies, \
                    auto0.start_pos
                restore_env_state(ba, anchor_state)
                vr = VideoRecorder(
                    root / "eval" / "videos" / "final" / TASK /
                    video_name(RUN_ID, "final", TASK,
                               f"{source['seed']}r{rep}", bname))
                vr.add(env._format_raw_obs(
                    env._env.env._get_observations()))
                steps_b, term_b, trunc_b = 0, False, False
                for a_env in runner.chunk_to_env(
                        chunk[None, :10].to(device)):
                    _o, _r, tb, tr, _i = env.step(a_env)
                    steps_b += 1
                    ba.evaluate(env, steps_b)
                    vr.add(env._format_raw_obs(
                        env._env.env._get_observations()))
                    if tb:
                        term_b = True
                        env._env.env.done = False
                    if tr:
                        trunc_b = True
                        break
                ba.flips = []
                tracker = SuccessTracker(term_preds)
                obs_c = env._format_raw_obs(
                    env._env.env._get_observations())
                ba.evaluate(env, 0)
                tracker.update(env, 0, eval_fn)
                q_at, steps_c, cd = {}, 0, 0
                stop = term_b or trunc_b
                while steps_c < CONT_MAX and not stop:
                    c_seed = sha_seed(
                        f"v071_0b_r1|cont|{sid}|{d}|{canon_id}"
                        f"|{rep}|{cd}")
                    nz = flow_noise(c_seed, cfg.chunk_size,
                                    cfg.max_action_dim)
                    bc = runner._obs_to_policy_batch(
                        obs_c, source["language_canonical"])
                    pc = prefix_forward(runner.policy, bc)
                    ch = sample_chunks(runner.policy, bc, n=1,
                                       noise=nz.to(device),
                                       prefix=pc)
                    for a_env in runner.chunk_to_env(ch[:, :10]):
                        obs_c, _r, tm, tr2, _i = env.step(a_env)
                        steps_c += 1
                        ba.evaluate(env, steps_c)
                        gt = tracker.update(env, steps_c, eval_fn)
                        assert gt == bool(tm)
                        vr.add(obs_c)
                        if steps_c in HORIZONS:
                            q_at[steps_c] = ba.q_valid()
                        if tm or tr2:
                            stop = True
                            break
                        if steps_c >= CONT_MAX:
                            break
                    cd += 1
                for h in HORIZONS:
                    q_at.setdefault(h, ba.q_valid())
                meta = vr.close()
                write_index_row(
                    vid_index, meta, run_id=RUN_ID,
                    checkpoint_tag="final",
                    checkpoint_path=str(
                        root / "checkpoints" / f"{bname}_final.pt")
                    if bname in ARMS else None,
                    checkpoint_sha256=None,
                    manifest_sha256=manifest.get("manifest_sha256",
                                                 "n/a"),
                    task=TASK, seed=f"{source['seed']}r{rep}",
                    arm=bname, split="train",
                    steps=steps_b + steps_c,
                    success=bool(tracker.achieved),
                    ordered_progress=ba.ordered_prefix(),
                    damage=-(-ba.damage_unrecovered()),
                    termination=("terminal" if term_b or stop
                                 else "horizon"), root=root)
                reps.append({
                    "success_by_100": bool(tracker.achieved),
                    "neg_damage": -ba.damage_unrecovered(),
                    "p_valid_100": ba.p_valid(),
                    "q_valid_mean": sum(q_at[h] for h in HORIZONS)
                    / len(HORIZONS),
                    "neg_tau_next": -ba.tau_next(0),
                    "q_at_horizons": dict(q_at)})
            outcomes[bname] = reps
            print(f"[exec] {bname}: q_mean="
                  f"{[round(o['q_valid_mean'], 3) for o in reps]}",
                  flush=True)
    finally:
        env.close()

    verdict = {}
    for arm in ARMS:
        pref = paired_preference(outcomes[arm],
                                 outcomes["fresh_u0"], tolerances)
        verdict[arm] = {"recon_pass": results[arm]["recon_pass"],
                        "recon_arm": results[arm]["recon_arm"],
                        "recon_stock": results[arm]["recon_stock"],
                        "fresh_pref_vs_u0": pref}
    smoke_pass = (verdict["oracle_sub0"]["recon_pass"]
                  and verdict["oracle_sub0"]["fresh_pref_vs_u0"] == 1)
    out = {"verdict": verdict, "N1_pass": bool(smoke_pass),
           "outcomes": outcomes,
           "next": ("freeze N1 for all V7.1 policy experiments"
                    if smoke_pass else
                    "broaden ONCE to the final two action-expert "
                    "blocks and repeat; freeze that result")}
    (root / "eval" / "smoke_result.json").write_text(
        json.dumps(out, indent=2, default=str))
    print(json.dumps({"verdict": verdict, "N1_pass": smoke_pass},
                     indent=1, default=str), flush=True)
    print(f"-> {root}", flush=True)


if __name__ == "__main__":
    main()
