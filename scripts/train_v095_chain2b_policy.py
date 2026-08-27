#!/usr/bin/env python
"""Distil verified chain2b corrections into the full-prompt N=1 policy.

    python scripts/train_v095_chain2b_policy.py --corrections results/v094_chain2b_corr/<run>

The corrections are first-10 action chunks that were EXECUTED and verified to
succeed where pi_0 failed. They are distilled at the FULL-prompt initial
observation, so the atomic teacher used to generate them never appears at
deployment - the policy must learn to produce the cream-directed chunk from the
full three-object instruction on its own.

Trainable boundary is the framework's locked default: everything frozen except
the stock-initialized `action_out_proj`, zero LC bias.

Anchor prefixes cost ZERO environment steps here - the corrections live at t=0,
so `env.reset(seed)` reproduces the state exactly.

Retention: cream-first episodes that pi_0 already succeeds on contribute their
OWN first-10 chunk as a target, so the update re-weights the first-object choice
without teaching the policy to abandon behaviour that already works.
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
from lcwm.v080_bench import episode_length, make_v080_env  # noqa: E402

TASK = "chain2b_lr2"
C_PREFIX, SEED, LR, WD = 10, 0, 1e-4, 1e-4
OUT = REPO / "results" / "v095_chain2b_pi"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corrections", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--retention", type=int, default=12,
                    help="cream-first seeds that already succeed, kept as anchors")
    a = ap.parse_args()
    torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / stamp; out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(a.corrections / "corrections.pt", weights_only=False)
    summ = json.loads((a.corrections / "summary.json").read_text())
    corr = ck["corrections"]
    keep = [r["seed"] for r in summ["rows"]
            if r["base"]["success"] and r["base"]["first"] == "cream"][:a.retention]
    print(f"{len(corr)} verified corrections; {len(keep)} retention anchors")
    if not corr:
        raise SystemExit("HALT: no verified corrections to distil")

    from lcwm.chassis import DEFAULT_MODEL, Pi05Runner
    runner = Pi05Runner(model_id=DEFAULT_MODEL, suite_name="libero_10", n_action_steps=10)
    env = make_v080_env(TASK)
    dev = next(runner.policy.parameters()).device

    # t=0 anchors: env.reset reproduces the state, so this costs no env steps.
    items = []
    for seed, chunk in corr:
        torch.manual_seed(seed); np.random.seed(seed)
        runner.reset(); obs, _ = env.reset(seed=int(seed))
        po = runner._obs_to_policy_batch(obs, env.task_description)   # FULL prompt
        with torch.no_grad():
            pf = prefix_forward(runner.policy, po)
        items.append({"kind": "correct", "seed": int(seed), "prefix": pf,
                      "chunk": chunk.to(dev)})
    for seed in keep:
        torch.manual_seed(seed); np.random.seed(seed)
        runner.reset(); obs, _ = env.reset(seed=int(seed))
        po = runner._obs_to_policy_batch(obs, env.task_description)
        with torch.no_grad():
            pf = prefix_forward(runner.policy, po)
            own = sample_chunks(runner.policy, po, 1, seed=int(seed) * 11 + 3,
                                prefix=pf)[0, :C_PREFIX].detach().float()
        items.append({"kind": "retain", "seed": int(seed), "prefix": pf,
                      "chunk": own.to(dev)})
    print(f"{len(items)} anchors built at t=0 (0 environment steps)")

    for p in runner.policy.parameters():
        p.requires_grad_(False)
    aop = runner.policy.model.action_out_proj
    stock = {k: v.detach().cpu().clone() for k, v in aop.state_dict().items()}
    for p in aop.parameters():
        p.requires_grad_(True)
    width = getattr(aop, "in_features", None)
    opt = torch.optim.AdamW(aop.parameters(), lr=LR, weight_decay=WD)
    print(f"trainable: action_out_proj, {sum(p.numel() for p in aop.parameters()):,} params")

    curve = out / "metrics.jsonl"; curve.write_text("")
    for ep in range(a.epochs):
        order = list(range(len(items))); random.Random(SEED + ep).shuffle(order)
        acc = {"correct": [0.0, 0], "retain": [0.0, 0]}
        for i in order:
            it = items[i]
            opt.zero_grad(set_to_none=True)
            g = torch.Generator(device="cpu").manual_seed(SEED * 977 + ep * 31 + i)
            noise = torch.randn(1, 50, runner.policy.config.max_action_dim,
                                generator=g).to(dev)
            t = torch.rand(1, generator=g).to(dev)
            pad = torch.zeros(1, 50, it["chunk"].shape[-1], device=dev)
            pad[:, :C_PREFIX] = it["chunk"].unsqueeze(0)
            bias = torch.zeros(1, width, device=dev) if width else None
            raw = raw_flow_losses_from_prefix(runner.policy, pad, bias, it["prefix"],
                                              noise=noise, time=t)
            loss = raw[:, :C_PREFIX, :it["chunk"].shape[-1]].mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(aop.parameters(), 1.0)
            opt.step()
            acc[it["kind"]][0] += float(loss); acc[it["kind"]][1] += 1
        row = {"epoch": ep,
               "correct": acc["correct"][0] / max(acc["correct"][1], 1),
               "retain": acc["retain"][0] / max(acc["retain"][1], 1)}
        with curve.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"epoch={ep} correct_fm={row['correct']:.5f} retain_fm={row['retain']:.5f}",
              flush=True)

    torch.save({"action_out_proj": {k: v.detach().cpu() for k, v in aop.state_dict().items()},
                "stock_action_out_proj": stock, "task": TASK,
                "corrections": len(corr), "retention": len(keep), "epochs": a.epochs},
               out / "pi_1.pt")
    d = sum(float((aop.state_dict()[k].detach().cpu() - stock[k]).abs().mean())
            for k in stock) / len(stock)
    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "task": TASK, "corrections": len(corr), "retention": len(keep),
         "epochs": a.epochs, "mean_abs_head_delta": d,
         "source": str(a.corrections), "env_steps": 0}, indent=2))
    print(f"\nsaved pi_1 -> {out}/pi_1.pt   mean |head delta| {d:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
