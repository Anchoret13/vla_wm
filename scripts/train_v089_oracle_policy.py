#!/usr/bin/env python
"""ORACLE-correction VLA update — the upper bound of the 2M.6 loop.

    python scripts/train_v089_oracle_policy.py --output results/v089_oracle_pi

Not a WM result and not a promotion claim.  This isolates ONE link in the
project's causal chain:

    failure -> counterfactual interaction -> [WM selection] -> VLA distillation

by replacing WM selection with the oracle - the alternative whose OBSERVED mean
QPhi is best, taken from the 72 already-executed anchors at zero new
acquisition cost.  If a VLA fine-tuned on oracle corrections does not beat pi_0,
the distillation link is broken and no WM improvement can rescue the loop.  If
it does, the WM is the bottleneck and we know exactly where to work.

Trainable boundary is the framework's locked default: everything frozen except
the stock-initialized `action_out_proj`, with a zero LC bias (no world-model
state injection - the first closed loop uses executed corrections only).
"""
from __future__ import annotations

import argparse, glob, json, random, sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402
ensure_project_libero_config()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from lcwm import v086_bank as B  # noqa: E402
from lcwm.lc_flow import raw_flow_losses_from_prefix  # noqa: E402
from lcwm.sampler import prefix_forward  # noqa: E402
from lcwm.v080r_panel import make_env_at  # noqa: E402
from lcwm.v081p_exec import SegmentLedger  # noqa: E402

import collect_v086_bank as C  # noqa: E402

QI = B.CONTINUOUS_FIELDS.index("QPhi")
DI = B.BINARY_FIELDS.index("dmg")
SEED, EPOCHS, LR, WD, CREDIT_T = 0, 8, 1e-4, 1e-4, 10
W_RETAIN = 1.0          # keep stock behaviour where no correction was found
OUT_ROOT = REPO / "results" / "v089_oracle_pi"


def load_corrections():
    """Oracle pick per anchor from the already-executed banks. Zero new steps."""
    corr, retain = [], []
    for p in ["results/v086_bank/2026-08-24T084023Z/groups.pt",
              sorted(glob.glob("results/v087_testbank/2026-*/groups.pt"))[-1]]:
        d = torch.load(REPO / p, weights_only=False)
        for g in d["groups"]:
            ref = int(torch.nonzero(g["is_reference"].bool()).flatten()[0])
            q = g["continuous"][..., QI].mean(1)
            dm = g["binary"][..., DI].mean(1)
            ok = [i for i in range(q.shape[0])
                  if i != ref and float(q[i]) > float(q[ref])
                  and float(dm[i]) <= float(dm[ref])]
            seed = int(g["source_id"].split(":")[1])
            if ok:
                best = max(ok, key=lambda i: float(q[i]))
                corr.append({"seed": seed, "anchor": g["anchor_id"],
                             "chunk": g["actions_norm"][best].clone(),
                             "gain": float(q[best] - q[ref])})
            else:
                retain.append({"seed": seed, "anchor": g["anchor_id"],
                               "chunk": g["actions_norm"][ref].clone()})
    return corr, retain


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=OUT_ROOT)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    a = ap.parse_args()
    torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = a.output / stamp; out.mkdir(parents=True, exist_ok=True)

    corr, retain = load_corrections()
    print(f"oracle corrections {len(corr)} | retention anchors {len(retain)} "
          f"| mean gain {sum(c['gain'] for c in corr)/len(corr):+.4f}")

    from lcwm.chassis import DEFAULT_MODEL, Pi05Runner
    runner = Pi05Runner(model_id=DEFAULT_MODEL, suite_name="libero_10", n_action_steps=10)
    env = make_env_at(B.TASK, B.DEADLINE)
    scene = C.Scene(env)
    dev = next(runner.policy.parameters()).device

    # regenerate the anchor prefixes: the banks stored tapped hidden states but
    # no KV cache and no snapshot, so each anchor's source is replayed to tau.
    ledger = SegmentLedger(out / "segment_ledger.jsonl",
                           {"source": (len(corr) + len(retain)) * B.DEADLINE},
                           action_root=out)
    items = []
    for kind, rows in (("correct", corr), ("retain", retain)):
        for r in rows:
            s = C.run_source(runner, env, scene, r["seed"], ledger)
            if s["anchor"] is None:
                print(f"  skip {r['anchor']}: anchor not reproduced"); continue
            from lcwm.snapshot import restore
            restore(env, s["anchor"]["snapshot"]); runner.reset()
            obs = env._format_raw_obs(env._env.env._get_observations())
            po = runner._obs_to_policy_batch(obs, env.task_description)
            with torch.no_grad():
                pf = prefix_forward(runner.policy, po)
            items.append({"kind": kind, "anchor": r["anchor"], "prefix": pf,
                          "chunk": r["chunk"].to(dev)})
            print(f"  {kind} {r['anchor']} prefix ready ({ledger.total} steps)", flush=True)
    print(f"\n{len(items)} anchors usable; {ledger.total} env steps to regenerate")

    for p in runner.policy.parameters():
        p.requires_grad_(False)
    aop = runner.policy.model.action_out_proj
    stock_aop = {k: v.detach().clone() for k, v in aop.state_dict().items()}
    for p in aop.parameters():
        p.requires_grad_(True)
    n_train = sum(p.numel() for p in aop.parameters())
    print(f"trainable boundary: action_out_proj only, {n_train:,} params "
          f"(everything else frozen; zero LC bias)")

    opt = torch.optim.AdamW(aop.parameters(), lr=LR, weight_decay=WD)
    width = runner.policy.model.action_out_proj.in_features \
        if hasattr(runner.policy.model.action_out_proj, "in_features") else None
    curve = out / "metrics.jsonl"; curve.write_text("")
    for ep in range(a.epochs):
        order = list(range(len(items))); random.Random(SEED + ep).shuffle(order)
        tot = {"correct": 0.0, "retain": 0.0}; cnt = {"correct": 0, "retain": 0}
        for i in order:
            it = items[i]
            opt.zero_grad(set_to_none=True)
            ch = it["chunk"].unsqueeze(0)
            bias = torch.zeros(1, width, device=dev) if width else None
            g = torch.Generator(device="cpu").manual_seed(SEED * 1000 + ep * 97 + i)
            noise = torch.randn(1, 50, runner.policy.config.max_action_dim,
                                generator=g).to(dev)
            t = torch.rand(1, generator=g).to(dev)
            pad = torch.zeros(1, 50, ch.shape[-1], device=dev)
            pad[:, :CREDIT_T] = ch
            raw = raw_flow_losses_from_prefix(runner.policy, pad, bias, it["prefix"],
                                              noise=noise, time=t)
            loss = raw[:, :CREDIT_T, :ch.shape[-1]].mean()
            w = 1.0 if it["kind"] == "correct" else W_RETAIN
            (w * loss).backward()
            torch.nn.utils.clip_grad_norm_(aop.parameters(), 1.0)
            opt.step()
            tot[it["kind"]] += float(loss); cnt[it["kind"]] += 1
        row = {"epoch": ep,
               "correct": tot["correct"] / max(cnt["correct"], 1),
               "retain": tot["retain"] / max(cnt["retain"], 1)}
        with curve.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"epoch={ep} correct_fm={row['correct']:.5f} retain_fm={row['retain']:.5f}",
              flush=True)

    torch.save({"action_out_proj": {k: v.detach().cpu()
                                    for k, v in aop.state_dict().items()},
                "stock_action_out_proj": {k: v.cpu() for k, v in stock_aop.items()},
                "corrections": len(corr), "retention": len(retain),
                "epochs": a.epochs, "lr": LR, "credit_t": CREDIT_T},
               out / "pi_oracle.pt")
    delta = sum(float((aop.state_dict()[k].detach().cpu() - stock_aop[k].cpu()).abs().mean())
                for k in stock_aop) / len(stock_aop)
    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "corrections": len(corr), "retention": len(retain),
         "anchors_used": len(items), "env_steps": ledger.total,
         "trainable_params": n_train, "mean_abs_head_delta": delta,
         "note": "oracle upper bound of the 2M.6 loop; not a WM result"}, indent=2))
    print(f"\nsaved -> {out}/pi_oracle.pt   mean |head delta| {delta:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
