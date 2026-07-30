#!/usr/bin/env python
"""H7.4 step 3 — one of the four matched π0.5 policy-interface finetunes.

Arms (registered table): v05_reset_wm / v05_reset_random /
v05_recurrent_wm / v05_recurrent_random.

Shared across ALL arms (H7.4 scoping, 2026-07-30):
- trainable set: W_c, W_w, W_g, α_w, α_g only; v0.5 WM + π0.5 frozen
  (bitwise-asserted at the end); total bias is exactly 0 at init
  (asserted) — stock parity at start;
- positive-state mask = frozen teacher table's selected != 0; identical
  item order (source_id, decision); identical per-item flow noise/time
  (teacher base 998000 + running index, rehearsal base 998500000 + index);
- teacher weight uniform 1.0, first-ten credit via credit-bounded flow
  loss; one optimizer step per item on (teacher + rehearsal)/2;
- rehearsal: stock train sources only, round-robin source-balanced,
  full-50 flow loss on the executed stock chunk;
- AdamW lr 1e-4 / wd 1e-4, grad-norm 1.0, 5 epochs.

Arm differences ONLY:
- state mode: reset arms bias from (c, w_reset, g_reset); recurrent arms
  from (c, w_rec, g_rec) — same modules/params, carry only;
- teacher index: wm arms use `selected`; random arms use `random_selected`
  at the SAME states.
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

from lcwm.v05_model import V05State  # noqa: E402

DATA = Path("/home/stargazer/Desktop/vla_wm/datasets/libero_loho_public_v1")
MANIFEST = DATA / "teacher_manifest"
WM_CKPT = (REPO_ROOT / "results" / "libero_loho_public_v1" / "v05_gated_wm"
           / "checkpoint_final.pt")
TEACHERS = (REPO_ROOT / "results" / "libero_loho_public_v1"
            / "v05_teachers" / "teachers.pt")
EPOCHS = 5
LR = 1e-4
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0
TEACHER_NOISE_BASE = 998_000
REHEARSAL_NOISE_BASE = 998_500_000
TRAINABLE = ("w_c", "w_w", "w_g", "alpha_w", "alpha_g")


def item_noise_time(policy, base: int, index: int, device):
    cfg = policy.config
    g = torch.Generator().manual_seed(base + index)
    noise = torch.randn(
        1, cfg.chunk_size, cfg.max_action_dim, generator=g).to(device)
    time = torch.rand(1, generator=g).to(device)
    return noise, time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=(
        "reset_wm", "reset_random", "recurrent_wm", "recurrent_random"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    state_mode = "reset" if args.arm.startswith("reset") else "recurrent"
    teacher_key = ("selected" if args.arm.endswith("_wm")
                   else "random_selected")

    from lcwm.chassis import Pi05Runner
    from lcwm.lc_flow import (cached_branch_flow_loss, freeze_pi05_base,
                              raw_flow_losses_from_prefix)
    from lcwm.sampler import prefix_forward

    model = V05State().to(device)
    bundle = torch.load(WM_CKPT, weights_only=False)
    model.load_state_dict(bundle["model"])
    model.eval()
    wm_hash = hashlib.sha256(WM_CKPT.read_bytes()).hexdigest()

    frozen_ref = {n: p.detach().clone()
                  for n, p in model.named_parameters()
                  if not any(n.startswith(t) for t in TRAINABLE)}
    trainable = []
    for n, p in model.named_parameters():
        keep = any(n.startswith(t) for t in TRAINABLE)
        p.requires_grad_(keep)
        if keep:
            trainable.append(p)
    optimizer = torch.optim.AdamW(
        trainable, lr=LR, weight_decay=WEIGHT_DECAY)

    runner = Pi05Runner(suite_name="libero_10")
    freeze_pi05_base(runner.policy)

    teachers = torch.load(TEACHERS, weights_only=False)
    assert teachers["wm_checkpoint_sha256"] == wm_hash
    sources = {}
    for path in sorted(MANIFEST.glob("loho_*.pt")):
        m = torch.load(path, weights_only=False)
        assert m["wm_checkpoint_sha256"] == wm_hash
        sources[m["source_id"]] = m

    def tokens_at(source_id: str, decision: int):
        row = sources[source_id]["rows"][decision]
        keys = (("c", "w_reset", "g_reset") if state_mode == "reset"
                else ("c", "w_rec", "g_rec"))
        return tuple(row[k][None].to(device) for k in keys)

    def bias_at(source_id: str, decision: int):
        c, w, g = tokens_at(source_id, decision)
        return model.policy_bias(c, w, g)

    @torch.no_grad()
    def prefix_at(source_id: str, decision: int):
        m = sources[source_id]
        obs = m["rows"][decision]["obs"]
        batch = runner._obs_to_policy_batch(obs, m["language_canonical"])
        return prefix_forward(runner.policy, batch), batch

    # positive-state mask, identical order across arms
    items = [r for r in sorted(teachers["rows"],
                               key=lambda r: (r["source_id"], r["decision"]))
             if r["selected"] != 0]
    stock_sources = sorted(s for s, m in sources.items()
                           if m["policy"] == "stock")
    rehearsal_pool = {
        s: sorted(sources[s]["rehearsal_decisions"])
        for s in stock_sources}
    print(f"[{args.arm}] {len(items)} positive teacher states, "
          f"rehearsal pool {sum(len(v) for v in rehearsal_pool.values())} "
          f"across {len(stock_sources)} stock sources", flush=True)

    # init parity: total bias must be exactly zero
    b0 = bias_at(items[0]["source_id"], items[0]["decision"])
    assert float(b0.abs().max()) == 0.0, "bias not exactly 0 at init"

    def teacher_loss(row, noise, time, suffix_perturb=None):
        m = sources[row["source_id"]]
        chunk = m["rows"][row["decision"]]["candidates"][
            row[teacher_key]][None].to(device).float()
        if suffix_perturb is not None:
            chunk = chunk.clone()
            chunk[:, 10:] += suffix_perturb
        prefix, _ = prefix_at(row["source_id"], row["decision"])
        bias = bias_at(row["source_id"], row["decision"])
        loss, _ = cached_branch_flow_loss(
            runner.policy, prefix, chunk, bias,
            torch.tensor([1.0], device=device),
            max_executed=10, noise=noise, time=time)
        return loss

    def rehearsal_loss(source_id, decision, noise, time,
                       suffix_perturb=None):
        chunk = sources[source_id]["rows"][decision]["chunk_norm"][
            None].to(device).float()
        if suffix_perturb is not None:
            chunk = chunk.clone()
            chunk[:, 10:] += suffix_perturb
        prefix, _ = prefix_at(source_id, decision)
        bias = bias_at(source_id, decision)
        return raw_flow_losses_from_prefix(
            runner.policy, chunk, bias, prefix,
            noise=noise, time=time).mean()

    # startup regression: first-ten credit only for teacher items
    noise0, time0 = item_noise_time(runner.policy, TEACHER_NOISE_BASE,
                                    10**6, device)
    perturb = 0.5 * torch.randn(1, 40, 7, device=device)

    def grad_of(fn):
        optimizer.zero_grad(set_to_none=True)
        fn().backward()
        return model.w_c.weight.grad.detach().clone()

    g_base = grad_of(lambda: teacher_loss(items[0], noise0, time0))
    g_pert = grad_of(lambda: teacher_loss(items[0], noise0, time0,
                                          suffix_perturb=perturb))
    assert torch.equal(g_base, g_pert), (
        "teacher first-ten gradient depends on unexecuted suffix")
    rs0 = stock_sources[0]
    rd0 = rehearsal_pool[rs0][0]
    r_base = grad_of(lambda: rehearsal_loss(rs0, rd0, noise0, time0))
    r_pert = grad_of(lambda: rehearsal_loss(rs0, rd0, noise0, time0,
                                            suffix_perturb=perturb))
    assert not torch.equal(r_base, r_pert), (
        "rehearsal must receive gradient from positions 10-49")
    print("suffix-credit regression PASSED", flush=True)
    optimizer.zero_grad(set_to_none=True)

    logs = []
    item_index = 0
    for epoch in range(EPOCHS):
        et, er = [], []
        for row in items:
            noise, time = item_noise_time(
                runner.policy, TEACHER_NOISE_BASE, item_index, device)
            rs = stock_sources[item_index % len(stock_sources)]
            pool = rehearsal_pool[rs]
            rd = pool[(item_index // len(stock_sources)) % len(pool)]
            noise2, time2 = item_noise_time(
                runner.policy, REHEARSAL_NOISE_BASE, item_index, device)
            optimizer.zero_grad(set_to_none=True)
            t_loss = teacher_loss(row, noise, time)
            r_loss = rehearsal_loss(rs, rd, noise2, time2)
            ((t_loss + r_loss) / 2).backward()
            torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM)
            optimizer.step()
            et.append(float(t_loss))
            er.append(float(r_loss))
            item_index += 1
        logs.append({
            "epoch": epoch,
            "teacher": sum(et) / len(et),
            "rehearsal": sum(er) / len(er),
            "tanh_alpha_w": float(torch.tanh(model.alpha_w)),
            "tanh_alpha_g": float(torch.tanh(model.alpha_g)),
            "norm_w_c": float(model.w_c.weight.norm()),
            "norm_w_w": float(model.w_w.weight.norm()),
            "norm_w_g": float(model.w_g.weight.norm()),
        })
        print(f"[{args.arm} epoch {epoch}] "
              f"teacher={logs[-1]['teacher']:.4f} "
              f"rehearsal={logs[-1]['rehearsal']:.4f} "
              f"tanh_aw={logs[-1]['tanh_alpha_w']:+.4f} "
              f"tanh_ag={logs[-1]['tanh_alpha_g']:+.4f} "
              f"|W_c|={logs[-1]['norm_w_c']:.4f}", flush=True)

    for n, p in model.named_parameters():
        if n in frozen_ref:
            assert torch.equal(p.detach(), frozen_ref[n]), (
                f"frozen param {n} changed during adapter stage")

    out = (REPO_ROOT / "results" / "libero_loho_public_v1" / "adapters"
           / f"v05_{args.arm}")
    out.mkdir(parents=True, exist_ok=True)
    torch.save({
        "arm": f"v05_{args.arm}",
        "state_mode": state_mode,
        "teacher_key": teacher_key,
        "interface": {n: p.detach().cpu()
                      for n, p in model.named_parameters()
                      if any(n.startswith(t) for t in TRAINABLE)},
        "wm_checkpoint_sha256": wm_hash,
        "teachers_sha256": hashlib.sha256(
            TEACHERS.read_bytes()).hexdigest(),
        "config": {"epochs": EPOCHS, "lr": LR, "wd": WEIGHT_DECAY,
                   "seed": args.seed,
                   "teacher_noise_base": TEACHER_NOISE_BASE,
                   "rehearsal_noise_base": REHEARSAL_NOISE_BASE},
    }, out / "adapter.pt")
    (out / "adapter_logs.json").write_text(json.dumps(logs))

    # fixed-state first-ten action shift vs stock (sanity, not a gate)
    from lcwm.lc_flow import sample_chunks_lc
    from lcwm.sampler import sample_chunks

    shifts = []
    by_source: dict[str, list] = {}
    for row in items:
        by_source.setdefault(row["source_id"], []).append(row)
    with torch.no_grad():
        for source_id, srows in sorted(by_source.items()):
            row = srows[len(srows) // 2]
            prefix, batch = prefix_at(source_id, row["decision"])
            g = torch.Generator().manual_seed(4242)
            noise = torch.randn(
                1, runner.policy.config.chunk_size,
                runner.policy.config.max_action_dim, generator=g,
            ).to(device)
            stock = sample_chunks(runner.policy, batch, n=1, noise=noise,
                                  prefix=prefix)
            bias = bias_at(source_id, row["decision"])
            adapted = sample_chunks_lc(runner.policy, batch, bias, n=1,
                                       noise=noise, prefix=prefix)
            delta = adapted[0, :10] - stock[0, :10]
            shifts.append({"source_id": source_id,
                           "decision": row["decision"],
                           "l2": float(delta.norm()),
                           "mean_abs": float(delta.abs().mean()),
                           "max_abs": float(delta.abs().max())})
    (out / "action_shift.json").write_text(json.dumps(shifts, indent=2))
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
