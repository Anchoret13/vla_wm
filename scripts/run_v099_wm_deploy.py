#!/usr/bin/env python
"""Deploy-time WM-guided action selection on the frozen panel.

    python scripts/run_v099_wm_deploy.py --scorer results/v098_wm/<run>/scorer.pt --panel 64

The VLA's weights are UNCHANGED and the deployment prompt is UNCHANGED. At each
of the first `--window` replan boundaries the policy proposes `--n` candidate
chunks and the world model scores each for whether it commits to the success
mode; the argmax is executed. After the commitment window the episode is plain
full-prompt N=1.

`--n 1` reproduces stock behaviour exactly and is the control: same sampling,
same seeds, no selection.
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
from lcwm.seq_data import goal_atoms, predicate_bits  # noqa: E402
from lcwm.task_automaton import GoalAutomaton  # noqa: E402
from lcwm.v080_bench import V080_TASKS, episode_length, make_v080_env  # noqa: E402
from lcwm.v082_m0 import masked_prefix_mean  # noqa: E402
from lcwm.v08r_contract import clopper_pearson_lower, clopper_pearson_upper  # noqa: E402

sys.path.insert(0, str(REPO / "scripts"))
from train_v098_deploy_wm import ChunkScorer  # noqa: E402

TASK = "chain2b_lr2"
OUT = REPO / "results" / "v099_wm_deploy"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", type=Path, required=True)
    ap.add_argument("--panel", type=int, default=64)
    ap.add_argument("--n", type=int, default=8, help="candidates per boundary")
    ap.add_argument("--window", type=int, default=4, help="boundaries under WM control")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    L = episode_length(TASK)
    tag = a.tag or (f"wm_n{a.n}_w{a.window}" if a.n > 1 else "control_n1")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / f"{tag}_{stamp}"; out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(a.scorer, weights_only=False)
    scorer = ChunkScorer(ck["state_dim"], ck["act_dim"])
    scorer.load_state_dict(ck["state_dict"]); scorer.eval()
    C = ck["c"]
    print(f"scorer held-out acc {ck['val_acc']:.3f}; n={a.n} window={a.window} boundaries")

    from lcwm.chassis import DEFAULT_MODEL, Pi05Runner
    runner = Pi05Runner(model_id=DEFAULT_MODEL, suite_name="libero_10", n_action_steps=10)
    env = make_v080_env(TASK); subgoals = V080_TASKS[TASK]["ordered_subgoals"]
    PANEL = tuple(range(3200, 3200 + a.panel))

    rows, steps = [], 0
    for seed in PANEL:
        torch.manual_seed(seed); np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        runner.reset(); obs, _ = env.reset(seed=seed)
        au = GoalAutomaton(subgoals); au.start(env); atoms = goal_atoms(env)
        au.evaluate(env, 0)
        t, done, succ, picks = 0, False, None, []
        for b in range(a.window):
            if done or t >= L:
                break
            po = runner._obs_to_policy_batch(obs, env.task_description)
            with torch.no_grad():
                pf = prefix_forward(runner.policy, po)
                cand = sample_chunks(runner.policy, po, a.n, seed=seed * 17 + b,
                                     prefix=pf)[:, :C].detach().float().cpu()
                st = masked_prefix_mean(pf.hidden[0].detach().float().cpu(),
                                        pf.pad_masks[0].detach().cpu())
                sc = scorer(st.unsqueeze(0).expand(a.n, -1), cand)
            del pf
            k = int(torch.argmax(sc))
            picks.append({"boundary": b, "chosen": k,
                          "score": float(sc[k]), "min": float(sc.min())})
            env_ch = runner.chunk_to_env(cand[k])
            for i in range(C):
                obs, _r, tm, tr, inf = env.step(env_ch[i]); t += 1
                done = bool(tm or tr)
                if succ is None and bool(inf.get("is_success", False)):
                    succ = t
                if done: break
            au.evaluate(env, t)
        runner.reset()                      # clean boundary before plain N=1
        while not done and t < L:
            obs, _r, tm, tr, inf = env.step(runner.select_action(obs, env.task_description))
            t += 1; done = bool(tm or tr)
            if succ is None and bool(inf.get("is_success", False)):
                succ = t
            if t % 10 == 0 or done:
                au.evaluate(env, t)
                if succ is None and predicate_bits(env, atoms).all():
                    succ = t
        steps += t
        ev = {int(k_): int(v) for k_, v in au.events_achieved.items()}
        order = [i for i, _ in sorted(ev.items(), key=lambda kv: kv[1])]
        first = ("cream" if order and order[0] == 2 else
                 "tomato" if order and order[0] == 0 else "none")
        rows.append({"seed": seed, "success": succ is not None, "success_step": succ,
                     "steps": t, "order": order, "first": first, "picks": picks})
        print(f"{tag} s{seed}: succ={succ is not None!s:5s} first={first:6s} "
              f"({steps} steps)", flush=True)

    k = sum(r["success"] for r in rows)
    summary = {"task": TASK, "tag": tag, "utc": stamp, "n_candidates": a.n,
               "window": a.window, "scorer": str(a.scorer),
               "scorer_val_acc": ck["val_acc"], "panel": [PANEL[0], PANEL[-1], len(PANEL)],
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
          f"first={summary['first_object']}  {steps} steps -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
