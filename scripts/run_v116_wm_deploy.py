#!/usr/bin/env python
"""Deployment: run, predict the outcome with the WM, correct only if it fails.

    python scripts/run_v116_wm_deploy.py --head results/v102_corrector/<run>/pi_2.pt \
        --gate results/v115_gate_final/<run>/gate_final.pt --window 30 --panel 64

Policy weights frozen apart from the gated corrective; deployment prompt
unchanged; N=1 throughout. No oracle and no panel labels anywhere.

    1. run the frozen policy for `--detect-at` steps
    2. the WM scores the state: will this episode pick the wrong object?
    3. if it fires, the corrector drives the next `--window` steps
    4. the stock head resumes for the manipulation

Every piece is forced by a measurement rather than chosen:

  detect at 20  v114: AUC 0.745 at t=0 but 0.999 at t=20 - the choice is not
                predictable before the arm moves, and is nearly certain after
  correct 30    v110: every window redirects 8/8, and 30 minimises collateral
                damage (48/51 kept, vs 46/51 at 40)
  gate at all   v107: correcting unconditionally scores 40/64, below baseline
  hand back     v107: the corrector was fitted on the first ~70 steps and
                wrecks the grasp if left driving the whole episode

The target is v112's oracle bound of 57/64 (+6 -0, p = 0.031), which this must
approach WITHOUT reading panel labels.
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
from lcwm.seq_data import goal_atoms, predicate_bits  # noqa: E402
from lcwm.task_automaton import GoalAutomaton  # noqa: E402
from lcwm.v080_bench import V080_TASKS, episode_length, make_v080_env  # noqa: E402
from lcwm.v082_m0 import masked_prefix_mean  # noqa: E402
from lcwm.v08r_contract import clopper_pearson_lower, clopper_pearson_upper  # noqa: E402

TASK = "chain2b_lr2"
OUT = REPO / "results" / "v116_wm_deploy"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--gate", type=Path, required=True)
    ap.add_argument("--detect-at", type=int, default=20)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--panel", type=int, default=64)
    ap.add_argument("--panel-start", type=int, default=3200,
                    help="first panel seed; 3200 is the selection panel, so a\n                         confirmatory run must use a fresh range")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    L = episode_length(TASK)
    tag = a.tag or f"wm_d{a.detect_at}_w{a.window}"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / f"{tag}_{stamp}"; out.mkdir(parents=True, exist_ok=True)

    hk = torch.load(a.head, weights_only=False)
    trained, stock = hk["action_out_proj"], hk["stock"]
    g = torch.load(a.gate, weights_only=False)
    assert g["tap"] == a.detect_at, f"gate fitted at t={g['tap']}, deploying at {a.detect_at}"
    mu, sd, B, w, b, thr = g["mu"], g["sd"], g["B"], g["w"], g["b"], g["threshold"]
    print(f"gate t={g['tap']} oof recall@0%FP={g['oof_recall_at_0fp']:.3f} "
          f"thr={thr:.4f}; detect@{a.detect_at}, correct {a.window} steps")

    from lcwm.chassis import DEFAULT_MODEL, Pi05Runner
    runner = Pi05Runner(model_id=DEFAULT_MODEL, suite_name="libero_10", n_action_steps=10)
    env = make_v080_env(TASK); subgoals = V080_TASKS[TASK]["ordered_subgoals"]
    aop = runner.policy.model.action_out_proj
    dev = next(runner.policy.parameters()).device
    to_dev = lambda s_: {k: v.to(dev) for k, v in s_.items()}
    PANEL = tuple(range(a.panel_start, a.panel_start + a.panel))

    rows, steps, fired = [], 0, 0
    for seed in PANEL:
        torch.manual_seed(seed); np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        aop.load_state_dict(to_dev(stock))
        runner.reset(); obs, _ = env.reset(seed=seed)
        au = GoalAutomaton(subgoals); au.start(env); atoms = goal_atoms(env)
        au.evaluate(env, 0)
        t, done, succ = 0, False, None
        score, use, corr_until = None, False, -1

        while not done and t < L:
            if t == a.detect_at:                     # the WM's one decision
                po = runner._obs_to_policy_batch(obs, env.task_description)
                with torch.no_grad():
                    pf = prefix_forward(runner.policy, po)
                    ft = masked_prefix_mean(pf.hidden[0].detach().float().cpu(),
                                            pf.pad_masks[0].detach().cpu()
                                            ).numpy().astype(np.float64)
                del pf
                z = ((ft - mu) / sd) @ B
                score = float(1 / (1 + np.exp(-(z @ w + b))))
                use = score > thr
                fired += int(use)
                if use:
                    aop.load_state_dict(to_dev(trained)); runner.reset()
                    corr_until = t + a.window
            if corr_until >= 0 and t == corr_until:
                aop.load_state_dict(to_dev(stock)); runner.reset(); corr_until = -1
            obs, _r, tm, tr, inf = env.step(
                runner.select_action(obs, env.task_description))
            t += 1; done = bool(tm or tr)
            if succ is None and bool(inf.get("is_success", False)):
                succ = t
            if t % 10 == 0 or done:
                au.evaluate(env, t)
                if succ is None and predicate_bits(env, atoms).all():
                    succ = t
        steps += t
        ev = {int(k): int(v) for k, v in au.events_achieved.items()}
        order = [i for i, _ in sorted(ev.items(), key=lambda kv: kv[1])]
        first = ("cream" if order and order[0] == 2 else
                 "tomato" if order and order[0] == 0 else "none")
        rows.append({"seed": seed, "success": succ is not None, "success_step": succ,
                     "steps": t, "order": order, "first": first,
                     "wm_score": score, "corrected": use})
        print(f"{tag} s{seed}: succ={succ is not None!s:5s} first={first:6s} "
              f"wm={score if score is None else round(score,3)} corrected={use!s:5s} "
              f"({steps} steps)", flush=True)

    aop.load_state_dict(to_dev(stock))
    k = sum(r["success"] for r in rows)
    summary = {"task": TASK, "tag": tag, "utc": stamp, "detect_at": a.detect_at,
               "window": a.window, "head": str(a.head), "gate": str(a.gate),
               "gate_tap": g["tap"], "gate_threshold": float(thr),
               "gate_oof_recall_at_0fp": g["oof_recall_at_0fp"],
               "n_corrected": fired, "is_upper_bound": False,
               "panel": [PANEL[0], PANEL[-1], len(PANEL)],
               "successes": k, "n": len(rows), "rate": k / len(rows),
               "cp95": [clopper_pearson_lower(k, len(rows)),
                        clopper_pearson_upper(k, len(rows))],
               "first_object": {f: sum(1 for r in rows if r["first"] == f)
                                for f in ("cream", "tomato", "none")},
               "env_steps": steps, "episodes": rows,
               "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                     capture_output=True, text=True).stdout.strip()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{tag}: {k}/{len(rows)} = {k/len(rows):.3f} "
          f"CP95 [{summary['cp95'][0]:.3f},{summary['cp95'][1]:.3f}] "
          f"corrected={fired}/{len(rows)} first={summary['first_object']} "
          f"{steps} steps -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
