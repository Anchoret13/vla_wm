#!/usr/bin/env python
"""Label more acquisition states AND cache their features for the gate.

    python scripts/collect_v111_gate_data.py --start 5000 --seeds 200

v108's gate scored CV AUC 0.573 on 85 states with 10 positives against a 2048-d
pooled prefix. That is underpowered rather than informative: with ~0.12 positive
rate, 85 states buy ~10 positives, and a 2048-d feature overfits any of them.

This widens both. It labels `--seeds` fresh acquisition states by the object the
frozen policy actually picks (horizon 250, since 120 truncated ~19% of picks and
produced a spurious zero-tomato reading in v101's first pass), and caches the
t=0 prefix features so gate hyperparameters can be searched on CPU without
re-running the policy.

Seeds are checked disjoint from the 3200-3399 evaluation panel and default to a
range beyond the 4900-4989 block already used, so the two label sets pool.
"""
from __future__ import annotations

import argparse, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402
ensure_project_libero_config()

import numpy as np, torch  # noqa: E402
from lcwm.sampler import prefix_forward  # noqa: E402
from lcwm.task_automaton import GoalAutomaton  # noqa: E402
from lcwm.v080_bench import V080_TASKS, make_v080_env  # noqa: E402
from lcwm.v082_m0 import masked_prefix_mean  # noqa: E402

TASK = "chain2b_lr2"
PANEL = set(range(3200, 3400))
OUT = REPO / "results" / "v111_gate_data"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=5000)
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--horizon", type=int, default=250)
    ap.add_argument("--merge", type=Path, default=None,
                    help="an earlier v101 summary.json to pool labels with")
    a = ap.parse_args()
    seeds = list(range(a.start, a.start + a.seeds))
    assert not (set(seeds) & PANEL), "acquisition must not touch the evaluation panel"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / stamp; out.mkdir(parents=True, exist_ok=True)

    from lcwm.chassis import DEFAULT_MODEL, Pi05Runner
    runner = Pi05Runner(model_id=DEFAULT_MODEL, suite_name="libero_10", n_action_steps=10)
    env = make_v080_env(TASK); subgoals = V080_TASKS[TASK]["ordered_subgoals"]

    feats, labels, kept, steps = [], [], [], 0
    for i, seed in enumerate(seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        runner.reset(); obs, _ = env.reset(seed=int(seed))
        po = runner._obs_to_policy_batch(obs, env.task_description)
        with torch.no_grad():                       # feature BEFORE any stepping
            pf = prefix_forward(runner.policy, po)
            ft = masked_prefix_mean(pf.hidden[0].detach().float().cpu(),
                                    pf.pad_masks[0].detach().cpu())
        del pf
        au = GoalAutomaton(subgoals); au.start(env); au.evaluate(env, 0)
        t, done = 0, False
        while not done and t < a.horizon:
            obs, _r, tm, tr, _i = env.step(
                runner.select_action(obs, env.task_description))
            t += 1; done = bool(tm or tr)
            if t % 10 == 0 or done:
                au.evaluate(env, t)
                if au.events_achieved: break
        steps += t
        ev = {int(k): int(v) for k, v in au.events_achieved.items()}
        order = [i2 for i2, _ in sorted(ev.items(), key=lambda kv: kv[1])]
        first = ("cream" if order and order[0] == 2 else
                 "tomato" if order and order[0] == 0 else "none")
        if first != "none":                          # unlabelled, not a negative
            feats.append(ft); labels.append(1.0 if first == "tomato" else 0.0)
            kept.append({"seed": int(seed), "first": first, "steps": t})
        if i % 20 == 0:
            npos = int(sum(labels))
            print(f"  [{i+1}/{len(seeds)}] {len(labels)} labelled, {npos} tomato "
                  f"({steps} steps)", flush=True)

    X = torch.stack(feats); Y = torch.tensor(labels)
    src = [{"file": None, "rows": kept}]
    if a.merge and a.merge.exists():
        print(f"note: {a.merge} labels are pooled by the trainer, not re-featurised here")
    torch.save({"X": X, "y": Y, "rows": kept, "task": TASK,
                "horizon": a.horizon}, out / "gate_data.pt")
    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "task": TASK, "seed_range": [seeds[0], seeds[-1]],
         "horizon": a.horizon, "n_labelled": len(labels),
         "n_positive": int(Y.sum()), "n_unlabelled": len(seeds) - len(labels),
         "positive_rate": float(Y.mean()), "env_steps": steps, "rows": kept,
         "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()},
        indent=2))
    print(f"\n{len(labels)} labelled ({int(Y.sum())} tomato, "
          f"{float(Y.mean()):.3f}); {steps} steps -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
