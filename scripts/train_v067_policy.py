#!/usr/bin/env python
"""V6.7.5 — repaired π0.5 flow post-training (one of three matched
checkpoints: gt / random / wm). W_z remains the ONLY trainable tensor.

Repairs vs iteration 1 (registered in 2026-07-31.md):
- sorted `k % len` sampling → frozen task/source/phase-stratified
  round-robin schedules; budget = ONE complete rehearsal sampler epoch
  (every rehearsal row exactly once; every task and demo visited before
  any row repeats; N_STEPS = |rehearsal bank|);
- rehearsal = genuine full-50 DEMONSTRATION chunks
  (demo_rehearsal_v067; libero_10 expert demos), not a partial ordered
  slice of LoHo stock rollouts; the recurrent LC state is unrolled along
  each demo at the deployed 10-action decision stride;
- grounded/generated targets balanced by task round-robin; the exposure
  count of every target is recorded;
- checkpoint selection on a task-balanced held-out objective (dev demos
  of all 10 tasks + dev LoHo source states of all 5 tasks), never on
  public LoHo behavior;
- launch asserts: wm/random arms require the SAME nonempty emitted state
  mask and positive generated weight (empty → refuse, route V6.8); the
  gt arm requires the task-balance floor from the teacher calibration;
- bias statistics saved at selection: total and across-state centered
  b_t RMS, recurrent-minus-reset variation, frozen state-shuffle assay.

Unchanged: identical base batches + noise streams across arms; first-ten
credit for GT/generated; full-50 rehearsal flow loss; stock-flow trust
(coeff 1.0); AdamW lr 1e-4 / wd 1e-4 / clip 1.0; W_z zero-init parity.
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
from lcwm.v067_lineage import RUN_SCHEMA, load_v067  # noqa: E402

DATA = Path("/home/stargazer/Desktop/vla_wm/datasets/libero_loho_public_v1"
            "/v06_effect_crossed")
DEMO_DIR = Path("/home/stargazer/Desktop/vla_wm/datasets"
                "/libero_loho_public_v1/demo_rehearsal_v067")
RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
WM_CKPT = RESULTS / "v067_wm" / "checkpoint_final.pt"
TEACHERS = RESULTS / "v067_teachers"
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
    """Frozen interleave: cycle group keys in sorted order, popping one
    item per visit, until all queues empty."""
    queues = {k: list(v) for k, v in sorted(groups.items()) if v}
    order = []
    while queues:
        for k in sorted(queues):
            order.append(queues[k].pop(0))
            if not queues[k]:
                del queues[k]
    return order


def cycle_schedule(groups: dict):
    """Endless task-balanced cycle (for GT/generated banks)."""
    base = round_robin(groups)

    def at(k):
        return base[k % len(base)] if base else None
    return at, len(base)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True,
                        choices=("gt", "random", "wm"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(0)

    from lcwm.chassis import Pi05Runner
    from lcwm.lc_flow import (cached_branch_flow_loss,
                              denoise_step_with_lc_bias, freeze_pi05_base,
                              raw_flow_losses_from_prefix)
    from lcwm.sampler import _expand_cache, prefix_forward
    from lcwm.seq_prefix_cache import normalize_actions
    from lerobot.utils.constants import ACTION

    model = V06State().to(device)
    bundle = torch.load(WM_CKPT, weights_only=False)
    assert bundle.get("run_schema") == RUN_SCHEMA
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

    teachers = load_v067(TEACHERS / "teacher_manifest.pt", "teachers")
    gt = load_v067(TEACHERS / "gt_manifest.pt", "teachers")
    calib = json.loads((TEACHERS / "calibration.json").read_text())
    assert teachers["wm_checkpoint_sha256"] == wm_hash

    # ---- launch asserts (registered) -----------------------------------
    emit_rows = [r for r in teachers["rows"] if r["emit_model_teacher"]]
    if args.arm in ("wm", "random"):
        assert emit_rows, (
            "emitted teacher mask is EMPTY — do not spend a matrix on "
            "byte-identical arms; the generated channel routes to V6.8")
    if args.arm == "gt":
        assert calib["task_balance_floor_met"], (
            "GT task-balance floor not met (a public task has no clean "
            "reference-improving train group) — route to V6.8")
    assert gt["rows"], "GT bank empty"

    sources = {}
    for sp in sorted((DATA / "sources").glob("*.pt")):
        s = torch.load(sp, weights_only=False)
        sources[s["source_id"]] = s

    # ---- LoHo z caches (recurrent along source; reset at state) --------
    from scripts.build_v067_teachers import unroll_z
    feats_cache, z_cache, z_reset_cache = {}, {}, {}
    with torch.no_grad():
        for sid, s in sorted(sources.items()):
            feats = torch.load(
                DATA / "features" / f"{sid}__canonical.pt",
                weights_only=False)
            feats_cache[sid] = feats
            z_cache[sid] = [z.detach() for z in unroll_z(
                model, feats, s["rows"], device)]

    def z_loho(sid, d, mode):
        if mode == "recurrent":
            return z_cache[sid][d]
        key = (sid, d)
        if key not in z_reset_cache:
            feats = feats_cache[sid]
            with torch.no_grad():
                z_reset_cache[key] = model.initial_state(
                    feats["h"][d][None].float().to(device),
                    feats["mask"][d][None].to(device)).detach()
        return z_reset_cache[key]

    # ---- demo rehearsal bank + recurrent z along each demo -------------
    demo_manifest = json.loads((DEMO_DIR / "manifest.json").read_text())
    assert demo_manifest["run_schema"] == RUN_SCHEMA
    demo_eps = {}
    for e in demo_manifest["episodes"]:
        demo_eps[(e["task_id"], e["demo"])] = load_v067(
            DEMO_DIR / e["path"], "demo_rehearsal")
    demo_z = {}       # (task_id, demo) -> {t: (z_rec, z_reset)}
    with torch.no_grad():
        for key, ep in sorted(demo_eps.items()):
            mean, std = ep["action_mean"], ep["action_std_eps"]
            z = None
            zs = {}
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
                    n_exec = int(prev.shape[0])
                    am = (torch.arange(10, device=device)[None]
                          < n_exec)
                    z = model.step(z, a, h, m, action_mask=am)
                if o["t"] % 50 == 0:
                    zs[o["t"]] = (z.detach(),
                                  model.initial_state(h, m).detach())
            demo_z[key] = zs
            print(f"[z] demo task{key[0]} demo{key[1]}: "
                  f"{len(zs)} boundary states", flush=True)

    # ---- frozen stratified schedules -----------------------------------
    rehearsal_by_task = {}
    dev_demo_by_task = {}
    for (tid, di), ep in sorted(demo_eps.items()):
        for ri, row in enumerate(ep["rows"]):
            item = (tid, di, ri)
            if ep["split"] == "train":
                rehearsal_by_task.setdefault(tid, []).append(item)
            else:
                dev_demo_by_task.setdefault(tid, []).append(item)
    # within-task order: demo round-robin (source-stratified)
    for tid in rehearsal_by_task:
        by_demo = {}
        for it in rehearsal_by_task[tid]:
            by_demo.setdefault(it[1], []).append(it)
        rehearsal_by_task[tid] = round_robin(by_demo)
    rehearsal_schedule = round_robin(rehearsal_by_task)
    n_steps = len(rehearsal_schedule)

    gt_by_task = {}
    for r in sorted(gt["rows"], key=lambda r: (r["source_id"],
                                               r["decision"])):
        gt_by_task.setdefault(r["task"], []).append(r)
    gt_at, n_gt = cycle_schedule(gt_by_task)
    gen_by_task = {}
    for r in sorted(emit_rows, key=lambda r: (r["source_id"],
                                              r["decision"])):
        gen_by_task.setdefault(r["task"], []).append(r)
    gen_at, n_gen = cycle_schedule(gen_by_task)

    dev_loho = []
    for sid, s in sorted(sources.items()):
        if s["split"] == "dev":
            for row in s["rows"][::4]:
                dev_loho.append((sid, row["decision"]))
    print(f"[{args.arm}] steps={n_steps} (1 rehearsal epoch), "
          f"gt bank {n_gt}, generated bank {n_gen}, "
          f"dev: {sum(len(v) for v in dev_demo_by_task.values())} demo "
          f"+ {len(dev_loho)} loho", flush=True)

    b0 = model.policy_bias(z_loho(gt["rows"][0]["source_id"],
                                  gt["rows"][0]["decision"],
                                  "recurrent"))
    assert float(b0.abs().max()) == 0.0, "W_z bias not 0 at init"

    prefix_lru: dict[tuple, object] = {}

    @torch.no_grad()
    def prefix_loho(sid, d):
        key = ("loho", sid, d)
        if key not in prefix_lru:
            if len(prefix_lru) > 40:
                prefix_lru.clear()
            src = sources[sid]
            batch = runner._obs_to_policy_batch(
                src["rows"][d]["obs"], src["language_canonical"])
            prefix_lru[key] = prefix_forward(runner.policy, batch)
        return prefix_lru[key]

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
            runner.policy, prefix, chunk[None].to(device).float(), bias,
            torch.tensor([1.0], device=device), max_executed=10,
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
            per_task = []
            i = 0
            for tid in sorted(dev_demo_by_task):
                vals = []
                for (t_, di, ri) in dev_demo_by_task[tid][:4]:
                    nz, tm = noise_time(runner.policy, D_BASE, i, device)
                    i += 1
                    ep = demo_eps[(t_, di)]
                    rl, tr = rehearsal_and_trust(
                        prefix_demo(t_, di, ri), ep["rows"][ri][
                            "chunk_norm"], demo_bias(t_, di, ri), nz, tm)
                    vals.append(float(rl + TRUST_COEF * tr))
                per_task.append(sum(vals) / len(vals))
            by_task_loho: dict[str, list] = {}
            for sid, d in dev_loho:
                by_task_loho.setdefault(sources[sid]["task"],
                                        []).append((sid, d))
            for task in sorted(by_task_loho):
                vals = []
                for sid, d in by_task_loho[task][:4]:
                    nz, tm = noise_time(runner.policy, D_BASE + 5000, i,
                                        device)
                    i += 1
                    rl, tr = rehearsal_and_trust(
                        prefix_loho(sid, d),
                        sources[sid]["rows"][d]["chunk_norm"],
                        model.policy_bias(z_loho(sid, d, "recurrent")),
                        nz, tm)
                    vals.append(float(rl + TRUST_COEF * tr))
                per_task.append(sum(vals) / len(vals))
        return sum(per_task) / len(per_task)

    exposure = {"gt": {}, "generated": {}, "rehearsal": {}}
    logs = []
    out = RESULTS / "v067_policy" / f"v067_{args.arm}"
    out.mkdir(parents=True, exist_ok=True)
    for k in range(n_steps):
        optimizer.zero_grad(set_to_none=True)
        losses = {}
        g = gt_at(k)
        gkey = f"{g['source_id']}_d{g['decision']}"
        exposure["gt"][gkey] = exposure["gt"].get(gkey, 0) + 1
        noise, time = noise_time(runner.policy, T_BASE, k, device)
        gt_loss = 0.0
        for mode in ("reset", "recurrent"):
            bias = model.policy_bias(
                z_loho(g["source_id"], g["decision"], mode))
            gt_loss = gt_loss + flow_first10(
                prefix_loho(g["source_id"], g["decision"]),
                g["chunk_norm"], bias, noise, time)
        losses["gt"] = gt_loss / 2
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
        if args.arm in ("random", "wm"):
            e = gen_at(k)
            idx = e["i_star"] if args.arm == "wm" else e["i_random"]
            ekey = f"{e['source_id']}_d{e['decision']}"
            exposure["generated"][ekey] = \
                exposure["generated"].get(ekey, 0) + 1
            noise3, time3 = noise_time(
                runner.policy, T_BASE + 500_000, k, device)
            losses["generated"] = flow_first10(
                prefix_loho(e["source_id"], e["decision"]),
                e["candidates"][idx],
                model.policy_bias(
                    z_loho(e["source_id"], e["decision"], "recurrent")),
                noise3, time3)
        else:
            losses["generated"] = torch.zeros((), device=device)
        total = sum(losses.values())
        total.backward()
        torch.nn.utils.clip_grad_norm_(trainable, GRAD_NORM)
        optimizer.step()
        if (k + 1) % CKPT_EVERY == 0 or (k + 1) == n_steps:
            dev_obj = dev_objective()
            torch.save({"w_z": model.w_z.state_dict(), "step": k + 1,
                        "dev_objective": dev_obj,
                        "run_schema": RUN_SCHEMA,
                        "wm_checkpoint_sha256": wm_hash},
                       out / f"wz_step{k + 1:04d}.pt")
            logs.append({"step": k + 1, "dev_objective": dev_obj,
                         **{n: float(v) for n, v in losses.items()}})
            print(f"[{args.arm} step {k + 1}/{n_steps}] " + " ".join(
                f"{n}={float(v):.4f}" for n, v in losses.items())
                + f" dev={dev_obj:.4f} |W_z|="
                f"{float(model.w_z.weight.norm()):.4f}", flush=True)

    for n, p in model.named_parameters():
        if n in frozen_ref:
            assert torch.equal(p.detach(), frozen_ref[n]), \
                f"frozen param {n} changed"

    # ---- bias statistics on the frozen dev state set -------------------
    with torch.no_grad():
        b_rec, b_reset = [], []
        for tid in sorted(dev_demo_by_task):
            for (t_, di, ri) in dev_demo_by_task[tid][:4]:
                b_rec.append(demo_bias(t_, di, ri, "recurrent"))
                b_reset.append(demo_bias(t_, di, ri, "reset"))
        for sid, d in dev_loho:
            b_rec.append(model.policy_bias(z_loho(sid, d, "recurrent")))
            b_reset.append(model.policy_bias(z_loho(sid, d, "reset")))
        B = torch.cat(b_rec)
        Br = torch.cat(b_reset)
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
    selected = torch.load(out / f"wz_step{best['step']:04d}.pt",
                          weights_only=False)
    torch.save({**selected, "selection": best,
                "config": {"n_steps": n_steps, "lr": LR, "wd": WD,
                           "trust_coef": TRUST_COEF,
                           "budget": "1 complete rehearsal epoch"},
                "bias_stats": bias_stats},
               out / "wz_selected.pt")
    (out / "policy_logs.json").write_text(json.dumps(logs))
    (out / "exposure_counts.json").write_text(json.dumps(
        {k: dict(sorted(v.items())) for k, v in exposure.items()},
        indent=2))
    (out / "bias_stats.json").write_text(json.dumps(bias_stats,
                                                    indent=2))
    print(f"[{args.arm}] selected step {best['step']} "
          f"(dev {best['dev_objective']:.4f}); bias {bias_stats}",
          flush=True)


if __name__ == "__main__":
    main()
