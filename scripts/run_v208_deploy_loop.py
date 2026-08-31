#!/usr/bin/env python
"""The deployment loop: collect with the current actor, retrain the model on it,
improve again. Alternating, not one-shot.

    python scripts/run_v208_deploy_loop.py --iters 3

WHY THIS, AND WHY IT IS THE ORIGINAL BRIEF. Every world-model attempt in this repo
has been ONE-SHOT: train T_th on pi0.5's tapes, train a residual through it, deploy
once. v205/v207 measured what that produces - the residual pins itself to 94% of
the trust-region bound and deployment falls to 38/96 from a 45/96 base, while the
model-free AWR arm on the SAME data reaches 63/96.

The mechanism is not a weak model. The model predicts well on its own support
(1-step 0.230x identity, 2.94x worse under a shuffled action, head AUC 0.869 on
PREDICTED latents). The residual then moves off that support, and the frozen model
has never seen where it went. Ensemble disagreement cannot police this: it sits at
1.08 on base actions and 1.05 on the residual's, so pessimism has no signal to
price (measured in v207, and it is why lam changed almost nothing).

Both the literature and the brief say the same thing about that gap:

  ME-TRPO (ICLR 2018) alternates model fitting and policy optimisation; Dreamer
  collects with the CURRENT actor and refits the model on it. A model frozen on
  another policy's data cannot stay valid where a moving actor goes.

  the brief, from the start: "run vla 的时候收集新数据然后 train wm, 通过这种
  方法来做 policy improvement" - collect DURING deployment, then improve.

So each iteration: deploy the current actor on a fresh COLLECTION panel recording
the actions actually executed, add that tape to the buffer, refit the ensemble on
the buffer, retrain the actor, and evaluate on the held-out panel.

GOAL ANCHOR (five axes): task chain1b_lr2 at 0.469; object T_th(b, E_a(u)) rolled
under a candidate action; target future latent, no reconstruction; data collected
during deployment and now FED BACK, which is the axis every prior attempt only
half-satisfied; placement inside the improvement loop.

PRE-REGISTERED READING, fixed before iteration 0 runs:
  the eval rate rising across iterations and clearing 0.656 (AWR) is the first
  evidence a world model contributes here;
  a flat or falling eval rate, or one that rises but stays under AWR, says the
  loop does not rescue the placement and the model-free arm remains the better
  use of the same deployment data. Either way the number is reported.
"""
from __future__ import annotations

import argparse, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "v208_deploy_loop"
PY_ = "/home/stargazer/miniconda3/envs/vf0s/bin/python"


def run(cmd, log, env=None):
    import os
    e = dict(os.environ); e.update(env or {})
    with open(log, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=REPO, env=e)
    return r.returncode


def rate_of(d):
    s = json.loads((d / "summary.json").read_text())
    return s["successes"], s["n"], s["rate"]


def newest(pat):
    g = sorted(REPO.glob(pat))
    return g[-1] if g else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="chain1b_lr2")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--panel", type=int, default=96)
    ap.add_argument("--eval-start", type=int, default=7600)
    ap.add_argument("--collect-start", type=int, default=7700,
                    help="first collection panel; each iteration advances by --panel")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--ensemble", type=int, default=5)
    ap.add_argument("--actor", type=Path, required=True,
                    help="iteration-0 actor, trained on pi0.5's tapes alone")
    a = ap.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / f"{a.task}_{stamp}"; out.mkdir(parents=True, exist_ok=True)

    buffer = [str(p) for p in sorted(REPO.glob(
        f"results/v121_deploy_latents/{a.task}_*/tape.pt"))]
    print(f"buffer starts with {len(buffer)} pi0.5 tapes")
    actor = a.actor
    hist = []
    for it in range(a.iters):
        cs = a.collect_start + it * a.panel
        assert cs + a.panel <= 8000 and not (cs <= a.eval_start < cs + a.panel), \
            "collection panel must not touch the evaluation panel"
        # 1. deploy the CURRENT actor to collect its own distribution
        print(f"\n=== iteration {it}: collect on {cs}-{cs+a.panel-1} ===", flush=True)
        run([PY_, "-u", "scripts/run_v206_belief_residual_deploy.py",
             "--actor", str(actor), "--task", a.task, "--panel", str(a.panel),
             "--panel-start", str(cs), "--collect-tape",
             "--tag", f"loop{it}_collect"],
            out / f"it{it}_collect.log", {"MUJOCO_GL": "egl"})
        cd = newest(f"results/v206_belief_residual/{a.task}_loop{it}_collect_*")
        k, n, r = rate_of(cd)
        print(f"  collected: {k}/{n} = {r:.3f} on the collection panel", flush=True)
        buffer.append(str(cd / "tape.pt"))

        # 2. refit the ensemble on the buffer, which now covers where the actor goes
        print(f"  refit on {len(buffer)} tapes", flush=True)
        run([PY_, "-u", "scripts/train_v207_pessimistic_ensemble.py",
             "--tapes", *buffer, "--task", a.task, "--lam", str(a.lam),
             "--ensemble", str(a.ensemble)],
            out / f"it{it}_train.log", {"OMP_NUM_THREADS": "12"})
        td = newest(f"results/v207_pessimistic_ensemble/lam*_k{a.ensemble}_*")
        actor = td / "actor_0.pt"

        # 3. evaluate on the held-out panel
        print(f"  eval on {a.eval_start}-{a.eval_start+a.panel-1}", flush=True)
        run([PY_, "-u", "scripts/run_v206_belief_residual_deploy.py",
             "--actor", str(actor), "--task", a.task, "--panel", str(a.panel),
             "--panel-start", str(a.eval_start), "--tag", f"loop{it}_eval"],
            out / f"it{it}_eval.log", {"MUJOCO_GL": "egl"})
        ed = newest(f"results/v206_belief_residual/{a.task}_loop{it}_eval_*")
        ek, en, er = rate_of(ed)
        print(f"  ** iteration {it} eval: {ek}/{en} = {er:.3f} **", flush=True)
        hist.append({"iter": it, "collect_panel": cs, "collect_rate": r,
                     "buffer_tapes": len(buffer), "actor": str(actor),
                     "eval_dir": str(ed), "eval_successes": ek, "eval_n": en,
                     "eval_rate": er})
        (out / "history.json").write_text(json.dumps(
            {"utc": stamp, "task": a.task, "lam": a.lam, "ensemble": a.ensemble,
             "eval_panel": a.eval_start, "baseline_zero_residual": 45 / 96,
             "model_free_awr": 63 / 96, "history": hist,
             "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                   capture_output=True, text=True).stdout.strip()},
            indent=2))

    print("\niter  eval")
    for h in hist:
        print(f"  {h['iter']}   {h['eval_successes']}/{h['eval_n']} = {h['eval_rate']:.3f}")
    print(f"reference: Delta=0 45/96 = 0.469, model-free AWR 63/96 = 0.656")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
