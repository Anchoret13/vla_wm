#!/usr/bin/env python
"""DAgger-style distillation of the cream-first behaviour into the full-prompt policy.

    python scripts/train_v097_dagger.py --seeds 30 --output results/v097_pi1

A single 10-action prefix cannot commit the policy (0/3 redirected), so the
correction is a TEACHER ROLLOUT: the atomic-cream prompt drives the arm until
the cream is picked, and every replan boundary along that rollout becomes a
distillation pair.

Crucially the pair is (FULL-prompt observation -> teacher chunk). The atomic
prompt is a training-time mechanism only (framework §6.4); at deployment the
policy sees the three-object instruction and must emit the cream-directed chunk
itself.

Two passes, so the teacher never drifts as the student trains:
  1. collect teacher ACTION SEQUENCES only (tiny storage, no policy updates);
  2. replay those actions open-loop - deterministic given the seed - and at each
     boundary compute the full-prompt FM loss on the teacher chunk and step.

Acquisition seeds are disjoint from the 3200-3399 behaviour block.
"""
from __future__ import annotations

import argparse, json, random, sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402
ensure_project_libero_config()

import numpy as np, torch  # noqa: E402
from lcwm.lc_flow import raw_flow_losses_from_prefix  # noqa: E402
from lcwm.sampler import prefix_forward, sample_chunks  # noqa: E402
from lcwm.task_automaton import GoalAutomaton  # noqa: E402
from lcwm.v080_bench import V080_TASKS, episode_length, make_v080_env  # noqa: E402

TASK, L = "chain2b_lr2", episode_length("chain2b_lr2")
ATOMIC = "pick up the cream cheese and place it in the basket"
CREAM_PICK, C, MAX_TEACHER = 2, 10, 160
ACQ = tuple(range(4900, 4990))
assert not (set(ACQ) & set(range(3200, 3400)))
SEED, LR, WD = 0, 1e-4, 1e-4
OUT = REPO / "results" / "v097_pi1"


def seed_all(s):
    torch.manual_seed(s); np.random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def collect(runner, env, seed, subgoals):
    """Pass 1: atomic teacher until the cream is picked. Records actions only."""
    seed_all(seed); runner.reset()
    obs, _ = env.reset(seed=seed)
    au = GoalAutomaton(subgoals); au.start(env); au.evaluate(env, 0)
    acts, chunks, t, done = [], [], 0, False
    while not done and t < MAX_TEACHER:
        po = runner._obs_to_policy_batch(obs, ATOMIC)
        with torch.no_grad():
            pf = prefix_forward(runner.policy, po)
            ch = sample_chunks(runner.policy, po, 1, seed=seed * 13 + t,
                               prefix=pf)[0, :C].detach().float().cpu()
        del pf
        env_ch = runner.chunk_to_env(ch)
        chunks.append(ch)
        for i in range(C):
            obs, _r, tm, tr, _inf = env.step(env_ch[i]); t += 1
            acts.append(env_ch[i]); done = bool(tm or tr)
            if done: break
        au.evaluate(env, t)
        if CREAM_PICK in au.events_achieved:
            break
    return {"seed": seed, "actions": np.stack(acts) if acts else None,
            "chunks": chunks, "steps": t,
            "picked": CREAM_PICK in au.events_achieved}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--output", type=Path, default=OUT)
    a = ap.parse_args()
    torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = a.output / stamp; out.mkdir(parents=True, exist_ok=True)

    from lcwm.chassis import DEFAULT_MODEL, Pi05Runner
    runner = Pi05Runner(model_id=DEFAULT_MODEL, suite_name="libero_10", n_action_steps=10)
    env = make_v080_env(TASK); subgoals = V080_TASKS[TASK]["ordered_subgoals"]
    dev = next(runner.policy.parameters()).device

    print("pass 1: teacher rollouts (no policy update)")
    demos, steps = [], 0
    for s in ACQ[:a.seeds]:
        d = collect(runner, env, s, subgoals)
        steps += d["steps"]
        if d["actions"] is not None and d["picked"]:
            demos.append(d)
        print(f"  s{s}: teacher {d['steps']} steps, cream picked={d['picked']} "
              f"({len(d['chunks'])} chunks)", flush=True)
    print(f"{len(demos)}/{a.seeds} teacher rollouts reached the cream pick; {steps} steps")
    if not demos:
        raise SystemExit("HALT: teacher never picks the cream; no correction exists")

    for p in runner.policy.parameters():
        p.requires_grad_(False)
    aop = runner.policy.model.action_out_proj
    stock = {k: v.detach().cpu().clone() for k, v in aop.state_dict().items()}
    for p in aop.parameters():
        p.requires_grad_(True)
    width = getattr(aop, "in_features", None)
    opt = torch.optim.AdamW(aop.parameters(), lr=LR, weight_decay=WD)
    print(f"trainable: action_out_proj {sum(p.numel() for p in aop.parameters()):,}")

    print("pass 2: replay the teacher actions, distil at FULL-prompt boundaries")
    curve = out / "metrics.jsonl"; curve.write_text("")
    for ep in range(a.epochs):
        order = list(range(len(demos))); random.Random(SEED + ep).shuffle(order)
        tot, n = 0.0, 0
        for di in order:
            d = demos[di]
            seed_all(d["seed"]); runner.reset()
            obs, _ = env.reset(seed=d["seed"])
            for ci, ch in enumerate(d["chunks"]):
                po = runner._obs_to_policy_batch(obs, env.task_description)  # FULL prompt
                with torch.no_grad():
                    pf = prefix_forward(runner.policy, po)
                opt.zero_grad(set_to_none=True)
                g = torch.Generator(device="cpu").manual_seed(SEED * 977 + ep * 31 + ci)
                noise = torch.randn(1, 50, runner.policy.config.max_action_dim,
                                    generator=g).to(dev)
                tt = torch.rand(1, generator=g).to(dev)
                pad = torch.zeros(1, 50, ch.shape[-1], device=dev)
                pad[:, :C] = ch.unsqueeze(0).to(dev)
                bias = torch.zeros(1, width, device=dev) if width else None
                raw = raw_flow_losses_from_prefix(runner.policy, pad, bias, pf,
                                                  noise=noise, time=tt)
                loss = raw[:, :C, :ch.shape[-1]].mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(aop.parameters(), 1.0)
                opt.step()
                tot += float(loss); n += 1
                del pf
                lo = ci * C
                for i in range(lo, min(lo + C, len(d["actions"]))):
                    obs, _r, tm, tr, _inf = env.step(d["actions"][i])
                    steps += 1
                    if bool(tm or tr): break
        row = {"epoch": ep, "fm": tot / max(n, 1), "pairs": n}
        with curve.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"epoch={ep} fm={row['fm']:.5f} over {n} pairs", flush=True)

    torch.save({"action_out_proj": {k: v.detach().cpu() for k, v in aop.state_dict().items()},
                "stock_action_out_proj": stock, "task": TASK,
                "demos": len(demos), "epochs": a.epochs}, out / "pi_1.pt")
    d_ = sum(float((aop.state_dict()[k].detach().cpu() - stock[k]).abs().mean())
             for k in stock) / len(stock)
    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "task": TASK, "teacher_rollouts": len(demos),
         "acq_seeds": [ACQ[0], ACQ[a.seeds - 1]], "epochs": a.epochs,
         "env_steps": steps, "mean_abs_head_delta": d_,
         "teacher_prompt": ATOMIC,
         "note": "atomic prompt is training-time only; deployment is full-prompt N=1"},
        indent=2))
    print(f"\nsaved pi_1 -> {out}/pi_1.pt  |head delta| {d_:.3e}  {steps} env steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
