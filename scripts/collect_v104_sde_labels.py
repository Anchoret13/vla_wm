#!/usr/bin/env python
"""Collect (state, chunk) -> realized-outcome labels from SDE rollouts.

    python scripts/collect_v104_sde_labels.py --states results/v101_acq_states/<run>/summary.json \
        --sigma 0.3 --rollouts 8

The v098 scorer hit held-out 1.000 and was still useless online, because its
labels came from WHICH PROMPT produced a chunk - a shortcut absent at deployment,
where every candidate comes from the full prompt. This labels chunks by WHAT THEY
ACTUALLY CAUSED instead: run SDE rollouts under the FULL prompt, record the chunk
chosen at each of the first `--window` boundaries, and label all of them with the
first object the rollout went on to pick.

Every chunk here is therefore drawn from the deployment distribution, and the
label is an executed outcome rather than a proxy. Credit assignment is coarse -
all four boundary chunks inherit one rollout-level label - which is the honest
cost of not having per-chunk ground truth.

States come from the held-out acquisition range only; the 3200-3399 panel is
never touched.
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
from lcwm.sampler import prefix_forward, sample_chunks  # noqa: E402
from lcwm.task_automaton import GoalAutomaton  # noqa: E402
from lcwm.v080_bench import V080_TASKS, make_v080_env  # noqa: E402
from lcwm.v082_m0 import masked_prefix_mean  # noqa: E402

TASK, C, WINDOW = "chain2b_lr2", 10, 4
PANEL = set(range(3200, 3400))
OUT = REPO / "results" / "v104_sde_labels"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--rollouts", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=260)
    ap.add_argument("--max-cream", type=int, default=20,
                    help="cap easy states so the set is not swamped by them")
    a = ap.parse_args()
    v101 = json.loads(a.states.read_text())
    err = list(v101["tomato_states"])
    good = [r["seed"] for r in v101["rows"] if r["first"] == "cream"][:a.max_cream]
    states = [(s, "error") for s in err] + [(s, "cream") for s in good]
    assert states, "no states"
    assert not ({s for s, _ in states} & PANEL), "acquisition must not touch the panel"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / stamp; out.mkdir(parents=True, exist_ok=True)
    print(f"{len(err)} error states + {len(good)} cream states; sigma={a.sigma}")

    from lcwm.chassis import DEFAULT_MODEL, Pi05Runner
    runner = Pi05Runner(model_id=DEFAULT_MODEL, suite_name="libero_10", n_action_steps=10)
    env = make_v080_env(TASK); subgoals = V080_TASKS[TASK]["ordered_subgoals"]

    S, A, Y, G, meta = [], [], [], [], []
    steps = 0
    for si, (seed, grp) in enumerate(states):
        for r in range(a.rollouts):
            torch.manual_seed(seed * 1000 + r); np.random.seed(seed * 1000 + r)
            runner.reset(); obs, _ = env.reset(seed=int(seed))
            au = GoalAutomaton(subgoals); au.start(env); au.evaluate(env, 0)
            recs, t, done = [], 0, False
            for b in range(WINDOW):
                if done: break
                po = runner._obs_to_policy_batch(obs, env.task_description)
                with torch.no_grad():
                    pf = prefix_forward(runner.policy, po)
                    ch = sample_chunks(runner.policy, po, 1,
                                       seed=seed * 7919 + r * 131 + b,
                                       prefix=pf, sigma=a.sigma
                                       )[0, :C].detach().float().cpu()
                    st = masked_prefix_mean(pf.hidden[0].detach().float().cpu(),
                                            pf.pad_masks[0].detach().cpu())
                del pf
                recs.append((st, ch))
                for act in runner.chunk_to_env(ch):
                    obs, _r, tm, tr, _i = env.step(act); t += 1
                    done = bool(tm or tr)
                    if done: break
                au.evaluate(env, t)
            runner.reset()
            while not done and t < a.horizon and not au.events_achieved:
                obs, _r, tm, tr, _i = env.step(
                    runner.select_action(obs, env.task_description))
                t += 1; done = bool(tm or tr)
                if t % 10 == 0 or done: au.evaluate(env, t)
            steps += t
            ev = {int(k): int(v) for k, v in au.events_achieved.items()}
            order = [i for i, _ in sorted(ev.items(), key=lambda kv: kv[1])]
            first = ("cream" if order and order[0] == 2 else
                     "tomato" if order and order[0] == 0 else "none")
            if first == "none":
                continue                      # unlabelled, not a negative
            lab = 1.0 if first == "cream" else 0.0
            for st, ch in recs:
                S.append(st); A.append(ch); Y.append(lab); G.append(si)
            meta.append({"seed": int(seed), "group": grp, "rollout": r,
                         "first": first, "steps": t})
        n_c = sum(1 for m in meta if m["seed"] == seed and m["first"] == "cream")
        print(f"[{si+1}/{len(states)}] s{seed} ({grp}): cream {n_c}/{a.rollouts} "
              f"({steps} steps)", flush=True)

    if not Y:
        print("no labelled chunks"); return 1
    torch.save({"S": torch.stack(S), "A": torch.stack(A),
                "y": torch.tensor(Y), "g": torch.tensor(G),
                "sigma": a.sigma, "c": C, "window": WINDOW, "task": TASK},
               out / "labels.pt")
    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "task": TASK, "sigma": a.sigma, "rollouts": a.rollouts,
         "error_states": err, "cream_states": good,
         "n_chunks": len(Y), "n_positive": int(sum(Y)),
         "n_rollouts_labelled": len(meta), "env_steps": steps, "meta": meta,
         "note": "labels are executed outcomes, not prompt identity; all chunks "
                 "drawn under the FULL prompt from the deployment distribution",
         "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()},
        indent=2))
    print(f"\n{len(Y)} chunks ({int(sum(Y))} positive) from {len(meta)} rollouts; "
          f"{steps} steps -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
