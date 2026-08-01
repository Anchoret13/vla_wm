#!/usr/bin/env python
"""V6.9.3 — grounded LC-state flow post-training (ONE W_z checkpoint).

Targets: ONLY actually executed clean improvements from the
failure-anchored bank (corrections_v069: robust winners over u_0 at
replay-clean anchors), including provenance-labeled competent
corrections (scripted servo / subgoal proposals). π0.5 and the
predictive LCWM are frozen; W_z is the only trainable tensor.

- z_t^ℓ unrolled from episode reset under the real composite prompt
  (fresh correction sources: prefixes recomputed from stored raw obs);
- correction targets credit only their first ten actions (scripted
  corrections: env→normalized affine map, zero-padded suffix carries
  no credit);
- genuine full-50 demonstration rehearsal + stock-flow trust;
- task/source/phase-stratified schedules; every unique target's
  exposure count recorded; budget = ONE complete rehearsal epoch;
- checkpoint selected on the frozen task-balanced held-out policy
  objective (dev demos of 10 tasks + dev LoHo states of 5 tasks),
  never public LoHo behavior;
- ONE checkpoint serves reset/current AND recurrent deployment
  (training averages both state modes per target; separately optimized
  state-mode checkpoints are forbidden);
- bias readouts saved (total RMS, across-state centered RMS,
  recurrent−reset RMS, state shuffle) — readouts, not launch gates.
- Uncovered tasks (no clean correction) receive demonstration
  rehearsal only and are reported explicitly; they do not cancel the
  run.

Output: results/libero_loho_public_v1/v069_policy/grounded_wz/
  target_manifest.pt  exposure_log.json  policy_metrics.json
  wz_selected.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

from lcwm.v06_model import V06State  # noqa: E402
from lcwm.v067_lineage import load_v067  # noqa: E402

DATA = Path("/home/stargazer/Desktop/vla_wm/datasets/libero_loho_public_v1"
            "/v06_effect_crossed")
DEMO_DIR = Path("/home/stargazer/Desktop/vla_wm/datasets"
                "/libero_loho_public_v1/demo_rehearsal_v067")
RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
WM_CKPT = RESULTS / "v069_predictive" / "checkpoint_selected.pt"
OUT = RESULTS / "v069_policy" / "grounded_wz"
TASKS = ["loho_t1_drawer", "loho_t2_basket3", "loho_t3_tray",
         "loho_t4_tray", "loho_t5_drawer_cabinet"]
LR, WD, GRAD_NORM = 1e-4, 1e-4, 1.0
TRUST_COEF = 1.0
T_BASE, R_BASE, D_BASE = 997_000, 997_500_000, 996_000
CKPT_EVERY = 50


def noise_time(policy, base, k, device):
    g = torch.Generator().manual_seed(base + k)
    cfg = policy.config
    noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim,
                        generator=g).to(device)
    time = torch.rand(1, generator=g).to(device)
    return noise, time


def round_robin(groups: dict):
    queues = {k: list(v) for k, v in sorted(groups.items()) if v}
    order = []
    while queues:
        for k in sorted(queues):
            order.append(queues[k].pop(0))
            if not queues[k]:
                del queues[k]
    return order


def cycle_schedule(groups: dict):
    base = round_robin(groups)

    def at(k):
        return base[k % len(base)] if base else None
    return at, len(base)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(0)

    from lcwm.chassis import Pi05Runner
    from lcwm.lc_flow import (cached_branch_flow_loss,
                              denoise_step_with_lc_bias,
                              freeze_pi05_base,
                              raw_flow_losses_from_prefix)
    from lcwm.sampler import _expand_cache, prefix_forward
    from lcwm.seq_prefix_cache import normalize_actions
    from lerobot.utils.constants import ACTION

    model = V06State().to(device)
    bundle = torch.load(WM_CKPT, weights_only=False)
    assert bundle.get("run_schema") == "v069", \
        "predictive checkpoint is not a v069 artifact"
    model.load_state_dict(bundle["model"])
    model.eval()
    wm_hash = hashlib.sha256(WM_CKPT.read_bytes()).hexdigest()
    frozen_ref = {n: p.detach().clone()
                  for n, p in model.named_parameters()
                  if not n.startswith("w_z")}
    for n, p in model.named_parameters():
        p.requires_grad_(n.startswith("w_z"))
    trainable = [p for n, p in model.named_parameters()
                 if n.startswith("w_z")]
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WD)

    runner = Pi05Runner(suite_name="libero_10")
    freeze_pi05_base(runner.policy)
    cfg = runner.policy.config

    ref_demo = torch.load(Path(
        "/home/stargazer/Desktop/vla_wm/datasets/seq_prefix_cache_v1"
        "/task0_demo0.pt"), weights_only=False)
    act_mean, act_std = ref_demo["action_mean"], \
        ref_demo["action_std_eps"]

    # ---- grounded targets from the failure-anchored bank ---------------
    # corrections are re-derived from the RECORDS under the amended
    # rule (outcome-level replay agreement + absolute gross bounds) so
    # files written before the amendment need no recollection
    from lcwm.task_automaton import paired_preference
    from scripts.collect_v069_corrections import GROSS_BOUNDS

    targets, sources = [], {}
    for cp in sorted((DATA / "corrections_v069").glob("*.pt")):
        grp = torch.load(cp, weights_only=False)
        assert grp["schema"] == "v069_corrections_v1"
        if grp["split"] != "train":
            continue
        tolerances = grp["frozen_outcome_tolerances"]

        def outs(k):
            return [r["outcome"] for r in sorted(
                (r for r in grp["records"] if r["branch_key"] == k),
                key=lambda r: r["repeat"])]
        outcome_unstable = paired_preference(
            outs("replay"), outs(0), tolerances) != 0
        rep = next(b for b in grp["branch_summaries"]
                   if b["kind"] == "replay")
        deltas = rep["replay_endpoint"]["deltas"]
        gross = any(deltas[k_] > b for k_, b in GROSS_BOUNDS.items())
        if outcome_unstable or gross:
            continue
        sid = grp["source_id"]
        if sid not in sources:
            sources[sid] = torch.load(
                DATA / "corrections_sources_v069" / f"{sid}.pt",
                weights_only=False)
        by_key = {str(b["key"]): b for b in grp["branch_summaries"]}
        correction_keys = [k_ for k_, v in
                           grp["judged_vs_u0"].items() if v == 1]
        for key in correction_keys:
            b = by_key[key]
            if b["chunk_norm"] is not None:
                chunk = b["chunk_norm"].clone()
            else:
                # scripted servo: env→normalized affine, zero-padded
                # suffix (first-ten credit only)
                acts = torch.from_numpy(b["actions_env"]).float()
                chunk = torch.zeros(cfg.chunk_size, 7)
                chunk[:acts.shape[0]] = normalize_actions(
                    acts, act_mean, act_std)
            targets.append({
                "source_id": sid, "task": grp["task"],
                "decision": grp["decision"],
                "key": key, "provenance": b["provenance"],
                "behavior_goal_id": b["behavior_goal_id"],
                "anchor_form": grp["anchor"]["form"],
                "chunk_norm": chunk,
                "judged": grp["judged_vs_u0"][key],
            })
    covered = {t: sum(1 for x in targets if x["task"] == t)
               for t in TASKS}
    uncovered = [t for t in TASKS if covered[t] == 0]
    assert targets, ("no clean grounded corrections exist — collect "
                     "before policy training (V6.9.2)")
    print(f"[targets] {len(targets)} grounded corrections; per task "
          f"{covered}; uncovered (rehearsal-only): {uncovered}",
          flush=True)

    # ---- z along fresh correction sources (recurrent + reset) ----------
    prefix_lru: dict[tuple, object] = {}

    @torch.no_grad()
    def prefix_corr(sid, d):
        key = ("corr", sid, d)
        if key not in prefix_lru:
            if len(prefix_lru) > 40:
                prefix_lru.clear()
            src = sources[sid]
            batch = runner._obs_to_policy_batch(
                src["rows"][d]["obs"], src["language_canonical"])
            prefix_lru[key] = prefix_forward(runner.policy, batch)
        return prefix_lru[key]

    corr_z: dict[tuple, tuple] = {}
    with torch.no_grad():
        for sid, src in sorted(sources.items()):
            need = sorted({x["decision"] for x in targets
                           if x["source_id"] == sid})
            if not need:
                continue
            z = None
            for i, row in enumerate(src["rows"]):
                if i > max(need):
                    break
                batch = runner._obs_to_policy_batch(
                    row["obs"], src["language_canonical"])
                prefix = prefix_forward(runner.policy, batch)
                h = prefix.hidden.float()
                m = prefix.pad_masks.bool()
                if z is None:
                    z = model.initial_state(h, m)
                else:
                    prev = src["rows"][i - 1]
                    a = prev["chunk_norm"][None, :10].float().to(
                        device)
                    am = (torch.arange(10, device=device)[None]
                          < prev["executed_len"])
                    z = model.step(z, a, h, m, action_mask=am)
                if i in need:
                    corr_z[(sid, i)] = (
                        z.detach(),
                        model.initial_state(h, m).detach())
            print(f"[z] {sid}: {len(need)} anchor states", flush=True)

    # ---- demo rehearsal bank + z along demos ---------------------------
    demo_manifest = json.loads((DEMO_DIR / "manifest.json").read_text())
    demo_eps = {}
    for e in demo_manifest["episodes"]:
        demo_eps[(e["task_id"], e["demo"])] = load_v067(
            DEMO_DIR / e["path"], "demo_rehearsal")
    demo_z = {}
    with torch.no_grad():
        for key, ep in sorted(demo_eps.items()):
            mean, std = ep["action_mean"], ep["action_std_eps"]
            z, zs = None, {}
            for j, o in enumerate(ep["obs_10"]):
                batch = runner._obs_to_policy_batch(
                    o["obs"], ep["language"])
                prefix = prefix_forward(runner.policy, batch)
                h = prefix.hidden.float()
                m = prefix.pad_masks.bool()
                if z is None:
                    z = model.initial_state(h, m)
                else:
                    prev = ep["obs_10"][j - 1]["chunk10_env"]
                    a = normalize_actions(prev, mean, std)[None].to(
                        device)
                    am = (torch.arange(10, device=device)[None]
                          < int(prev.shape[0]))
                    z = model.step(z, a, h, m, action_mask=am)
                if o["t"] % 50 == 0:
                    zs[o["t"]] = (z.detach(),
                                  model.initial_state(h, m).detach())
            demo_z[key] = zs
    print(f"[z] demo bank: {len(demo_z)} episodes", flush=True)

    # ---- frozen stratified schedules -----------------------------------
    rehearsal_by_task, dev_demo_by_task = {}, {}
    for (tid, di), ep in sorted(demo_eps.items()):
        for ri, _row in enumerate(ep["rows"]):
            item = (tid, di, ri)
            if ep["split"] == "train":
                rehearsal_by_task.setdefault(tid, []).append(item)
            else:
                dev_demo_by_task.setdefault(tid, []).append(item)
    for tid in rehearsal_by_task:
        by_demo = {}
        for it in rehearsal_by_task[tid]:
            by_demo.setdefault(it[1], []).append(it)
        rehearsal_by_task[tid] = round_robin(by_demo)
    rehearsal_schedule = round_robin(rehearsal_by_task)
    n_steps = len(rehearsal_schedule)

    tgt_by_stratum = {}
    for x in targets:
        tgt_by_stratum.setdefault(
            (x["task"], x["anchor_form"]), []).append(x)
    tgt_at, n_tgt = cycle_schedule(tgt_by_stratum)

    dev_loho = []
    for sp in sorted((DATA / "sources").glob("*.pt")):
        s = torch.load(sp, weights_only=False)
        if s["split"] == "dev":
            for row in s["rows"][::4]:
                dev_loho.append((s["source_id"], row["decision"]))
            sources.setdefault(s["source_id"], s)
    print(f"[schedule] steps={n_steps} (1 rehearsal epoch), "
          f"targets={n_tgt} in {len(tgt_by_stratum)} task/phase "
          f"strata", flush=True)

    z_dev_cache: dict[tuple, tuple] = {}

    @torch.no_grad()
    def z_dev_loho(sid, d):
        if (sid, d) not in z_dev_cache:
            src = sources[sid]
            z = None
            for i, row in enumerate(src["rows"]):
                if i > d:
                    break
                batch = runner._obs_to_policy_batch(
                    row["obs"], src["language_canonical"])
                prefix = prefix_forward(runner.policy, batch)
                h = prefix.hidden.float()
                m = prefix.pad_masks.bool()
                if z is None:
                    z = model.initial_state(h, m)
                else:
                    prev = src["rows"][i - 1]
                    a = prev["chunk_norm"][None, :10].float().to(
                        device)
                    am = (torch.arange(10, device=device)[None]
                          < prev["executed_len"])
                    z = model.step(z, a, h, m, action_mask=am)
                if i == d:
                    z_dev_cache[(sid, d)] = (
                        z.detach(),
                        model.initial_state(h, m).detach())
        return z_dev_cache[(sid, d)]

    b0 = model.policy_bias(corr_z[(targets[0]["source_id"],
                                   targets[0]["decision"])][0])
    assert float(b0.abs().max()) == 0.0, "W_z bias not 0 at init"

    @torch.no_grad()
    def prefix_demo(tid, di, ri):
        key = ("demo", tid, di, ri)
        if key not in prefix_lru:
            if len(prefix_lru) > 40:
                prefix_lru.clear()
            ep = demo_eps[(tid, di)]
            batch = runner._obs_to_policy_batch(
                ep["rows"][ri]["obs"], ep["language"])
            prefix_lru[key] = prefix_forward(runner.policy, batch)
        return prefix_lru[key]

    def flow_first10(prefix, chunk, bias, noise, time):
        loss, _ = cached_branch_flow_loss(
            runner.policy, prefix, chunk[None].to(device).float(),
            bias, torch.tensor([1.0], device=device), max_executed=10,
            noise=noise, time=time)
        return loss

    def rehearsal_and_trust(prefix, chunk, bias, noise, time):
        chunk = chunk[None].to(device).float()
        rl = raw_flow_losses_from_prefix(
            runner.policy, chunk, bias, prefix,
            noise=noise, time=time).mean()
        policy = runner.policy
        actions = policy.prepare_action({ACTION: chunk})
        x_t = time[:, None, None] * noise \
            + (1 - time[:, None, None]) * actions
        cache = _expand_cache(prefix.past_key_values, 1)
        v_a = denoise_step_with_lc_bias(
            policy.model, prefix.pad_masks, cache, x_t, time, bias)
        with torch.no_grad():
            v_s = denoise_step_with_lc_bias(
                policy.model, prefix.pad_masks, cache, x_t, time,
                torch.zeros_like(bias))
        return rl, torch.nn.functional.mse_loss(v_a, v_s)

    def demo_bias(tid, di, ri, mode="recurrent"):
        t = demo_eps[(tid, di)]["rows"][ri]["t"]
        z_rec, z_reset = demo_z[(tid, di)][t]
        return model.policy_bias(z_rec if mode == "recurrent"
                                 else z_reset)

    def dev_objective():
        with torch.no_grad():
            per_task, i = [], 0
            for tid in sorted(dev_demo_by_task):
                vals = []
                for (t_, di, ri) in dev_demo_by_task[tid][:4]:
                    nz, tm = noise_time(runner.policy, D_BASE, i,
                                        device)
                    i += 1
                    ep = demo_eps[(t_, di)]
                    rl, tr = rehearsal_and_trust(
                        prefix_demo(t_, di, ri),
                        ep["rows"][ri]["chunk_norm"],
                        demo_bias(t_, di, ri), nz, tm)
                    vals.append(float(rl + TRUST_COEF * tr))
                per_task.append(sum(vals) / len(vals))
            by_task_loho: dict[str, list] = {}
            for sid, d in dev_loho:
                by_task_loho.setdefault(sources[sid]["task"],
                                        []).append((sid, d))
            for task in sorted(by_task_loho):
                vals = []
                for sid, d in by_task_loho[task][:4]:
                    nz, tm = noise_time(runner.policy, D_BASE + 5000,
                                        i, device)
                    i += 1
                    src = sources[sid]
                    key = ("loho", sid, d)
                    if key not in prefix_lru:
                        if len(prefix_lru) > 40:
                            prefix_lru.clear()
                        batch = runner._obs_to_policy_batch(
                            src["rows"][d]["obs"],
                            src["language_canonical"],
                        )
                        prefix_lru[key] = prefix_forward(
                            runner.policy, batch)
                    rl, tr = rehearsal_and_trust(
                        prefix_lru[key],
                        src["rows"][d]["chunk_norm"],
                        model.policy_bias(z_dev_loho(sid, d)[0]),
                        nz, tm)
                    vals.append(float(rl + TRUST_COEF * tr))
                per_task.append(sum(vals) / len(vals))
        return sum(per_task) / len(per_task)

    exposure = {"targets": {}, "rehearsal": {}}
    logs = []
    OUT.mkdir(parents=True, exist_ok=True)
    for k in range(n_steps):
        optimizer.zero_grad(set_to_none=True)
        losses = {}
        x = tgt_at(k)
        tkey = f"{x['source_id']}_d{x['decision']}_{x['key']}"
        exposure["targets"][tkey] = exposure["targets"].get(tkey,
                                                           0) + 1
        noise, time = noise_time(runner.policy, T_BASE, k, device)
        z_rec, z_reset = corr_z[(x["source_id"], x["decision"])]
        corr_loss = 0.0
        for zz in (z_reset, z_rec):
            corr_loss = corr_loss + flow_first10(
                prefix_corr(x["source_id"], x["decision"]),
                x["chunk_norm"], model.policy_bias(zz), noise, time)
        losses["grounded"] = corr_loss / 2
        tid, di, ri = rehearsal_schedule[k]
        rkey = f"t{tid}_d{di}_r{ri}"
        exposure["rehearsal"][rkey] = \
            exposure["rehearsal"].get(rkey, 0) + 1
        noise2, time2 = noise_time(runner.policy, R_BASE, k, device)
        rl, trust = rehearsal_and_trust(
            prefix_demo(tid, di, ri),
            demo_eps[(tid, di)]["rows"][ri]["chunk_norm"],
            demo_bias(tid, di, ri), noise2, time2)
        losses["rehearsal"] = rl
        losses["trust"] = TRUST_COEF * trust
        total = sum(losses.values())
        total.backward()
        torch.nn.utils.clip_grad_norm_(trainable, GRAD_NORM)
        optimizer.step()
        if (k + 1) % CKPT_EVERY == 0 or (k + 1) == n_steps:
            dev_obj = dev_objective()
            torch.save({"w_z": model.w_z.state_dict(), "step": k + 1,
                        "dev_objective": dev_obj,
                        "run_schema": "v069",
                        "wm_checkpoint_sha256": wm_hash},
                       OUT / f"wz_step{k + 1:04d}.pt")
            logs.append({"step": k + 1, "dev_objective": dev_obj,
                         **{n: float(v) for n, v in losses.items()}})
            print(f"[grounded step {k + 1}/{n_steps}] " + " ".join(
                f"{n}={float(v):.4f}" for n, v in losses.items())
                + f" dev={dev_obj:.4f} |W_z|="
                f"{float(model.w_z.weight.norm()):.4f}", flush=True)

    for n, p in model.named_parameters():
        if n in frozen_ref:
            assert torch.equal(p.detach(), frozen_ref[n]), \
                f"frozen param {n} changed"

    with torch.no_grad():
        b_rec, b_reset = [], []
        for tid in sorted(dev_demo_by_task):
            for (t_, di, ri) in dev_demo_by_task[tid][:4]:
                b_rec.append(demo_bias(t_, di, ri, "recurrent"))
                b_reset.append(demo_bias(t_, di, ri, "reset"))
        for sid, d in dev_loho:
            zr, z0 = z_dev_loho(sid, d)
            b_rec.append(model.policy_bias(zr))
            b_reset.append(model.policy_bias(z0))
        B, Br = torch.cat(b_rec), torch.cat(b_reset)
        g = torch.Generator().manual_seed(1234)
        perm = torch.randperm(B.shape[0], generator=g)
        bias_stats = {
            "n_states": int(B.shape[0]),
            "total_rms": float(B.pow(2).mean().sqrt()),
            "across_state_centered_rms": float(
                (B - B.mean(0, keepdim=True)).pow(2).mean().sqrt()),
            "recurrent_minus_reset_rms": float(
                (B - Br).pow(2).mean().sqrt()),
            "state_shuffle_delta_rms": float(
                (B - B[perm]).pow(2).mean().sqrt()),
        }
    best = min(logs, key=lambda r: (r["dev_objective"], r["step"]))
    selected = torch.load(OUT / f"wz_step{best['step']:04d}.pt",
                          weights_only=False)
    torch.save({**selected, "selection": best,
                "bias_stats": bias_stats,
                "config": {"n_steps": n_steps, "lr": LR, "wd": WD,
                           "trust_coef": TRUST_COEF,
                           "budget": "1 complete rehearsal epoch",
                           "one_checkpoint_both_modes": True}},
               OUT / "wz_selected.pt")
    torch.save({"schema": "v069_target_manifest_v1",
                "run_schema": "v069",
                "targets": [{k_: v for k_, v in x.items()}
                            for x in targets],
                "covered_by_task": covered,
                "uncovered_tasks": uncovered,
                "wm_checkpoint_sha256": wm_hash},
               OUT / "target_manifest.pt")
    (OUT / "exposure_log.json").write_text(json.dumps(
        {k_: dict(sorted(v.items())) for k_, v in exposure.items()},
        indent=2))
    (OUT / "policy_metrics.json").write_text(json.dumps(
        {"logs": logs, "bias_stats": bias_stats,
         "covered_by_task": covered,
         "uncovered_tasks": uncovered}, indent=2))
    print(f"selected step {best['step']} "
          f"(dev {best['dev_objective']:.4f}); bias {bias_stats}",
          flush=True)


if __name__ == "__main__":
    main()
