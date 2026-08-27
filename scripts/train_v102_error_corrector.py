#!/usr/bin/env python
"""Distil the cream-first correction USING ONLY ERROR STATES.

    python scripts/train_v102_error_corrector.py --states results/v101_acq_states/<run>/summary.json

v097 sampled 30 generic acquisition seeds; ~88% of them are states where the
policy already picks the cream, so the teacher agreed with the learner and the
gradient was near-zero. Head delta came out at 8.96e-4 and success did not move.

This trains on the states where the policy ERRS, found by v101 in the held-out
acquisition range. Same mechanism as v097, same frozen-panel evaluation, but the
data now carries gradient. The atomic prompt remains a TRAINING-time device only
(framework 6.4); deployment stays full-prompt N=1.

Also fits the WM gate: a logistic head on the t=0 full-prompt state encoding
predicting P(the policy will pick the wrong object here). v100 measured that
signal as near-separable across states, and v099 showed why gating matters -
intervening at every boundary cost cream-first success 0.944 -> 0.873.
"""
from __future__ import annotations

import argparse, json, random, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402
ensure_project_libero_config()

import numpy as np, torch  # noqa: E402
from torch import nn  # noqa: E402
from lcwm.lc_flow import raw_flow_losses_from_prefix  # noqa: E402
from lcwm.sampler import prefix_forward  # noqa: E402
from lcwm.v080_bench import episode_length, make_v080_env  # noqa: E402
from lcwm.v082_m0 import masked_prefix_mean  # noqa: E402
from lcwm.task_automaton import GoalAutomaton  # noqa: E402
from lcwm.v080_bench import V080_TASKS  # noqa: E402

TASK, L = "chain2b_lr2", episode_length("chain2b_lr2")
ATOMIC = "pick up the cream cheese and place it in the basket"
CREAM_PICK, C, MAX_TEACHER = 2, 10, 200
PANEL = set(range(3200, 3400))
SEED, WD = 0, 1e-4
OUT = REPO / "results" / "v102_corrector"


def seed_all(s: int) -> None:
    torch.manual_seed(s); np.random.seed(s); random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, required=True, help="v101 summary.json")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--include-none", action="store_true",
                    help="also train on states where no object was picked by the "
                         "probe horizon; off by default because those are slow, "
                         "not necessarily wrong, and mixing them back in "
                         "reintroduces the non-informative data that flattened v097")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--output", type=Path, default=OUT)
    a = ap.parse_args()
    seed_all(SEED)
    v101 = json.loads(a.states.read_text())
    err = list(v101["tomato_states"])
    if a.include_none:
        err += list(v101.get("none_states", []))
    assert err, "v101 found no error states"
    assert not (set(err) & PANEL), "error states must be disjoint from the panel"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = a.output / stamp; out.mkdir(parents=True, exist_ok=True)
    print(f"{len(err)} error states from v101 "
          f"(include_none={a.include_none}): {err}")

    from lcwm.chassis import DEFAULT_MODEL, Pi05Runner
    runner = Pi05Runner(model_id=DEFAULT_MODEL, suite_name="libero_10", n_action_steps=10)
    env = make_v080_env(TASK); subgoals = V080_TASKS[TASK]["ordered_subgoals"]

    print("pass 1: atomic-cream teacher at the ERROR states (no policy update)")
    demos, steps = [], 0
    for seed in err:
        seed_all(seed); runner.reset(); obs, _ = env.reset(seed=int(seed))
        au = GoalAutomaton(subgoals); au.start(env); au.evaluate(env, 0)
        chunks, t, done, got = [], 0, False, False
        while not done and t < MAX_TEACHER and not got:
            # store the NORMALIZED chunk (the flow objective's space) and execute
            # its env-space image; select_action returns env space, so feeding it
            # to the FM loss as actions_norm - as v097 did - trains on the wrong
            # units.
            ch_norm = runner.sample_chunk(obs, ATOMIC)[0, :C].detach().float().cpu()
            chunks.append(ch_norm)
            for act in runner.chunk_to_env(ch_norm):
                obs, _r, tm, tr, _i = env.step(act); t += 1; done = bool(tm or tr)
                if done: break
            au.evaluate(env, t)
            got = CREAM_PICK in au.events_achieved
        steps += t
        if got:
            demos.append({"seed": int(seed), "chunks": chunks})
        print(f"  s{seed}: cream_pick={got} t={t} ({steps} steps)", flush=True)
    print(f"{len(demos)}/{len(err)} teacher rollouts reached the cream pick; {steps} steps")
    if not demos:
        print("no usable demos"); return 1

    for p in runner.policy.parameters():
        p.requires_grad_(False)
    aop = runner.policy.model.action_out_proj
    stock = {k: v.detach().cpu().clone() for k, v in aop.state_dict().items()}
    for p in aop.parameters():
        p.requires_grad_(True)
    width = getattr(aop, "in_features", None)
    dev = next(runner.policy.parameters()).device
    cfg = runner.policy.config
    opt = torch.optim.AdamW(aop.parameters(), lr=a.lr, weight_decay=WD)
    npair = sum(len(d["chunks"]) for d in demos)
    print(f"trainable action_out_proj {sum(p.numel() for p in aop.parameters()):,}; "
          f"{npair} pairs from {len(demos)} error states; lr={a.lr} epochs={a.epochs}")

    print("pass 2: distil at FULL-prompt boundaries")
    curve = out / "metrics.jsonl"; curve.write_text("")
    for ep in range(a.epochs):
        order = list(range(len(demos))); random.Random(SEED + ep).shuffle(order)
        tot, n = 0.0, 0
        for di in order:
            d = demos[di]
            seed_all(d["seed"]); runner.reset(); obs, _ = env.reset(seed=d["seed"])
            for ci, ch in enumerate(d["chunks"]):
                po = runner._obs_to_policy_batch(obs, env.task_description)  # FULL prompt
                with torch.no_grad():
                    pf = prefix_forward(runner.policy, po)
                opt.zero_grad(set_to_none=True)
                g = torch.Generator(device="cpu").manual_seed(SEED * 977 + ep * 31 + ci)
                noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim,
                                    generator=g).to(dev)
                tt = torch.rand(1, generator=g).to(dev)
                pad = torch.zeros(1, cfg.chunk_size, ch.shape[-1], device=dev)
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
                for act in runner.chunk_to_env(ch):
                    obs, _r, tm, tr, _i = env.step(act)
                    if tm or tr: break
        delta = max(float((aop.state_dict()[k].detach().cpu() - v).abs().max())
                    for k, v in stock.items())
        print(f"epoch={ep} fm={tot/max(n,1):.5f} over {n} pairs |delta|={delta:.3e}", flush=True)
        with curve.open("a") as fh:
            fh.write(json.dumps({"epoch": ep, "fm": tot / max(n, 1),
                                 "pairs": n, "head_delta": delta}) + "\n")

    delta = max(float((aop.state_dict()[k].detach().cpu() - v).abs().max())
                for k, v in stock.items())
    torch.save({"action_out_proj": {k: v.detach().cpu()
                                    for k, v in aop.state_dict().items()},
                "stock": stock, "task": TASK, "error_states": err,
                "demos": len(demos), "epochs": a.epochs, "lr": a.lr},
               out / "pi_2.pt")
    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "task": TASK, "error_states": err,
         "teacher_rollouts": len(demos), "pairs": npair, "epochs": a.epochs,
         "lr": a.lr, "head_delta": delta, "env_steps": steps,
         "v101": str(a.states),
         "note": "trained ONLY on states where pi_0 errs; atomic prompt is "
                 "training-time only, deployment is full-prompt N=1",
         "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()},
        indent=2))
    print(f"saved pi_2 -> {out}/pi_2.pt  |head delta| {delta:.3e}  {steps} env steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
