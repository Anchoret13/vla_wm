#!/usr/bin/env python
"""When does the state become predictive? Features at t = 0, 10, 20, 30.

    python scripts/collect_v114_timed_features.py --start 5000 --seeds 200

Gate v2 reached CV AUC 0.748 from the t=0 prefix but only recall 0.037 at zero
false positives, far below the ~1.0 the McNemar arithmetic needs. Predicting
which object the policy will choose BEFORE the arm has moved is simply hard.

But nothing requires the prediction to happen at t=0. The v110 sweep showed a
20-step window already redirects 8/8, so a correction begun at t=10 or t=20 still
has room to act - and by then the arm's own motion is evidence. This captures the
prefix feature at several timepoints in a single rollout and labels each with the
object eventually picked, so the accuracy-vs-latency tradeoff can be read off
directly.

The reported quantity is recall at zero false positives, not AUC: the deployment
bar is set by v112's oracle bound (gained 6 / lost 0 -> p = 0.031, while gained 6
/ lost 1 -> p = 0.125), so specificity is what matters.
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
TAPS = (0, 10, 20, 30)
OUT = REPO / "results" / "v114_timed"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=5000)
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--horizon", type=int, default=250)
    a = ap.parse_args()
    seeds = list(range(a.start, a.start + a.seeds))
    assert not (set(seeds) & PANEL), "acquisition must not touch the evaluation panel"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / stamp; out.mkdir(parents=True, exist_ok=True)

    from lcwm.chassis import DEFAULT_MODEL, Pi05Runner
    runner = Pi05Runner(model_id=DEFAULT_MODEL, suite_name="libero_10", n_action_steps=10)
    env = make_v080_env(TASK); subgoals = V080_TASKS[TASK]["ordered_subgoals"]

    F = {t: [] for t in TAPS}
    labels, kept, steps = [], [], 0
    for i, seed in enumerate(seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        runner.reset(); obs, _ = env.reset(seed=int(seed))
        au = GoalAutomaton(subgoals); au.start(env); au.evaluate(env, 0)
        taps, t, done = {}, 0, False
        while not done and t < a.horizon:
            if t in TAPS:
                po = runner._obs_to_policy_batch(obs, env.task_description)
                with torch.no_grad():
                    pf = prefix_forward(runner.policy, po)
                    taps[t] = masked_prefix_mean(
                        pf.hidden[0].detach().float().cpu(),
                        pf.pad_masks[0].detach().cpu())
                del pf
            obs, _r, tm, tr, _i = env.step(
                runner.select_action(obs, env.task_description))
            t += 1; done = bool(tm or tr)
            if t % 10 == 0 or done:
                au.evaluate(env, t)
                if au.events_achieved: break
        steps += t
        ev = {int(k): int(v) for k, v in au.events_achieved.items()}
        order = [j for j, _ in sorted(ev.items(), key=lambda kv: kv[1])]
        first = ("cream" if order and order[0] == 2 else
                 "tomato" if order and order[0] == 0 else "none")
        if first != "none" and all(t in taps for t in TAPS):
            for t in TAPS:
                F[t].append(taps[t])
            labels.append(1.0 if first == "tomato" else 0.0)
            kept.append({"seed": int(seed), "first": first, "steps": t})
        if i % 20 == 0:
            print(f"  [{i+1}/{len(seeds)}] {len(labels)} labelled, "
                  f"{int(sum(labels))} tomato ({steps} steps)", flush=True)

    Y = torch.tensor(labels)
    torch.save({"X": {t: torch.stack(F[t]) for t in TAPS}, "y": Y,
                "taps": list(TAPS), "rows": kept, "task": TASK}, out / "timed.pt")
    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "task": TASK, "taps": list(TAPS),
         "seed_range": [seeds[0], seeds[-1]], "n_labelled": len(labels),
         "n_positive": int(Y.sum()), "env_steps": steps, "rows": kept,
         "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()},
        indent=2))
    print(f"\n{len(labels)} labelled ({int(Y.sum())} tomato); {steps} steps -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
