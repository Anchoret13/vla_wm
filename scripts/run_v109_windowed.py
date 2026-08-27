#!/usr/bin/env python
"""Apply the corrector ONLY during the commitment window, then hand back.

    python scripts/run_v109_windowed.py --head results/v102_corrector/<run>/pi_2.pt \
        --window 40 --panel 64

Ungated pi_2 redirected 8/8 tomato states - every panel episode went cream-first -
yet scored 40/64 against pi_0's 51/64, losing 18 previously-good states. Since
cream-first succeeds 0.933 of the time, object choice is fully fixed and the loss
is in EXECUTION: the corrected head was driving the grasp and the place, which it
was never trained for. It was fit on 72 pairs covering only the first ~70 steps.

So restrict it in time. The corrector drives the first `--window` steps - the
measured commitment window, where 10 steps redirect 0/4 and 40 redirect 4/4 -
then the stock head resumes for the manipulation.

`--gate` additionally consults the v108 outcome WM and applies the corrector only
where it predicts failure. That gate currently scores CV AUC 0.573 (near chance),
so it is an ablation, not the headline: --window alone is the primary condition
and the gated arm measures whether the WM adds anything on top.
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

sys.path.insert(0, str(REPO / "scripts"))
from train_v108_gate import OutcomeWM  # noqa: E402

TASK = "chain2b_lr2"
OUT = REPO / "results" / "v109_windowed"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--panel", type=int, default=64)
    ap.add_argument("--gate", type=Path, default=None)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    L = episode_length(TASK)
    tag = a.tag or (f"win{a.window}" + ("_gated" if a.gate else ""))
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / f"{tag}_{stamp}"; out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(a.head, weights_only=False)
    trained, stock = ck["action_out_proj"], ck["stock"]
    gate = None
    if a.gate:
        gk = torch.load(a.gate, weights_only=False)
        gate = OutcomeWM(gk["dim"]); gate.load_state_dict(gk["state_dict"]); gate.eval()
        print(f"gate CV AUC {gk['cv_auc']:.3f} recall {gk['cv_recall']:.3f} tau={a.tau}")
    print(f"window={a.window} steps, then the stock head resumes")

    from lcwm.chassis import DEFAULT_MODEL, Pi05Runner
    runner = Pi05Runner(model_id=DEFAULT_MODEL, suite_name="libero_10", n_action_steps=10)
    env = make_v080_env(TASK); subgoals = V080_TASKS[TASK]["ordered_subgoals"]
    aop = runner.policy.model.action_out_proj
    dev = next(runner.policy.parameters()).device
    to_dev = lambda sd: {k: v.to(dev) for k, v in sd.items()}
    PANEL = tuple(range(3200, 3200 + a.panel))

    rows, steps, fired = [], 0, 0
    for seed in PANEL:
        torch.manual_seed(seed); np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        aop.load_state_dict(to_dev(stock))
        runner.reset(); obs, _ = env.reset(seed=seed)
        au = GoalAutomaton(subgoals); au.start(env); atoms = goal_atoms(env)
        au.evaluate(env, 0)

        use = True
        if gate is not None:
            po = runner._obs_to_policy_batch(obs, env.task_description)
            with torch.no_grad():
                pf = prefix_forward(runner.policy, po)
                st = masked_prefix_mean(pf.hidden[0].detach().float().cpu(),
                                        pf.pad_masks[0].detach().cpu())
                p = float(torch.sigmoid(gate(st.unsqueeze(0)))[0])
            del pf
            use = p > a.tau
        fired += int(use)
        if use:
            aop.load_state_dict(to_dev(trained)); runner.reset()

        t, done, succ, swapped = 0, False, None, not use
        while not done and t < L:
            obs, _r, tm, tr, inf = env.step(
                runner.select_action(obs, env.task_description))
            t += 1; done = bool(tm or tr)
            if succ is None and bool(inf.get("is_success", False)):
                succ = t
            if not swapped and t >= a.window:
                aop.load_state_dict(to_dev(stock)); runner.reset(); swapped = True
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
                     "steps": t, "order": order, "first": first, "corrected": use})
        print(f"{tag} s{seed}: succ={succ is not None!s:5s} first={first:6s} "
              f"corrected={use!s:5s} ({steps} steps)", flush=True)

    aop.load_state_dict(to_dev(stock))
    k = sum(r["success"] for r in rows)
    summary = {"task": TASK, "tag": tag, "utc": stamp, "window": a.window,
               "head": str(a.head), "gate": str(a.gate) if a.gate else None,
               "tau": a.tau, "n_corrected": fired,
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
          f"first={summary['first_object']} corrected={fired}/{len(rows)} "
          f"{steps} steps -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
